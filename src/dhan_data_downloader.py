"""Resumable Dhan 1-minute history acquisition for discovered NIFTY futures."""

from __future__ import annotations

import json
import logging
import os
import time
import glob
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


API_URL = "https://api.dhan.co/v2/charts/intraday"
REQUIRED_MASTER_FIELDS = {"EXCH_ID", "SEGMENT", "SECURITY_ID", "INSTRUMENT",
                          "UNDERLYING_SYMBOL", "SM_EXPIRY_DATE"}


class DhanAPIError(RuntimeError):
    """REST error retaining Dhan's exact status and response body."""

    def __init__(self, message, status_code=None, response_body=None):
        self.status_code, self.response_body = status_code, response_body
        detail = f" status_code={status_code}" if status_code is not None else ""
        body = f" response_body={response_body}" if response_body is not None else ""
        super().__init__(f"{message}{detail}{body}")


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not 1 <= int(config["chunk_days"]) <= 90:
        raise ValueError("chunk_days must be between 1 and Dhan's 90-day maximum")
    if int(config["rollover_trading_days"]) < 0:
        raise ValueError("rollover_trading_days cannot be negative")
    return config


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "SEM_EXM_EXCH_ID": "EXCH_ID", "SEM_SEGMENT": "SEGMENT",
        "SEM_SMST_SECURITY_ID": "SECURITY_ID", "SEM_INSTRUMENT_NAME": "INSTRUMENT",
        "SEM_EXPIRY_DATE": "SM_EXPIRY_DATE", "SM_SYMBOL_NAME": "UNDERLYING_SYMBOL",
        "SEM_CUSTOM_SYMBOL": "DISPLAY_NAME", "SEM_LOT_UNITS": "LOT_SIZE",
        "SEM_TICK_SIZE": "TICK_SIZE",
    }
    return frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})


def discover_nifty_futures(paths: list[str], config: dict) -> pd.DataFrame:
    """Discover contracts from one or more current/archived Security Masters."""
    catalogs = []
    expanded_paths = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        expanded_paths.extend(matches or [pattern])
    for raw_path in dict.fromkeys(expanded_paths):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Security Master not found: {path}")
        frame = _canonical_columns(pd.read_csv(path, low_memory=False))
        missing = REQUIRED_MASTER_FIELDS.difference(frame.columns)
        if missing:
            raise ValueError(f"Security Master {path} missing columns: {sorted(missing)}")
        frame["security_master_source"] = str(path)
        catalogs.append(frame)
    master = pd.concat(catalogs, ignore_index=True)
    mask = (
        master["EXCH_ID"].astype(str).str.upper().eq(config["exchange_id"].upper())
        & master["SEGMENT"].astype(str).str.upper().eq(config["master_segment"].upper())
        & master["INSTRUMENT"].astype(str).str.upper().eq(config["instrument"].upper())
        & master["UNDERLYING_SYMBOL"].astype(str).str.strip().str.upper().eq(config["underlying_symbol"].upper())
    )
    contracts = master.loc[mask].copy()
    contracts["expiry_date"] = pd.to_datetime(contracts["SM_EXPIRY_DATE"], errors="coerce").dt.normalize()
    contracts["security_id"] = contracts["SECURITY_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    contracts = contracts.dropna(subset=["expiry_date"]).drop_duplicates(
        ["security_id", "expiry_date"], keep="last"
    ).sort_values(["expiry_date", "security_id"], kind="stable").reset_index(drop=True)
    if contracts.empty:
        raise ValueError("No NIFTY FUTIDX contracts found in configured Security Masters")
    keep = ["security_id", "expiry_date", "security_master_source"]
    for optional in ("DISPLAY_NAME", "SYMBOL_NAME", "LOT_SIZE", "TICK_SIZE"):
        if optional in contracts:
            keep.append(optional)
    return contracts[keep]


def discover_nifty_options(paths: list[str], config: dict) -> pd.DataFrame:
    """Discover CE and PE contracts, including expired archived-master rows."""
    catalogs = []
    expanded_paths = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        expanded_paths.extend(matches or [pattern])
    for raw_path in dict.fromkeys(expanded_paths):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Security Master not found: {path}")
        frame = _canonical_columns(pd.read_csv(path, low_memory=False))
        required = REQUIRED_MASTER_FIELDS | {"OPTION_TYPE", "STRIKE_PRICE"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Security Master {path} missing columns: {sorted(missing)}")
        frame["security_master_source"] = str(path)
        catalogs.append(frame)
    master = pd.concat(catalogs, ignore_index=True)
    mask = (
        master["EXCH_ID"].astype(str).str.upper().eq(config["exchange_id"].upper())
        & master["SEGMENT"].astype(str).str.upper().eq(config["master_segment"].upper())
        & master["INSTRUMENT"].astype(str).str.upper().eq("OPTIDX")
        & master["UNDERLYING_SYMBOL"].astype(str).str.strip().str.upper().eq(
            config["underlying_symbol"].upper())
        & master["OPTION_TYPE"].astype(str).str.upper().isin(["CE", "PE"])
    )
    contracts = master.loc[mask].copy()
    contracts["expiry_date"] = pd.to_datetime(
        contracts["SM_EXPIRY_DATE"], errors="coerce").dt.normalize()
    contracts["security_id"] = contracts["SECURITY_ID"].astype(str).str.replace(
        r"\.0$", "", regex=True)
    contracts["option_type"] = contracts["OPTION_TYPE"].astype(str).str.upper()
    contracts["strike"] = pd.to_numeric(contracts["STRIKE_PRICE"], errors="coerce")
    contracts = contracts.dropna(subset=["expiry_date", "strike"]).drop_duplicates(
        ["security_id", "expiry_date"], keep="last"
    ).sort_values(["expiry_date", "strike", "option_type"], kind="stable")
    if contracts.empty:
        raise ValueError("No NIFTY OPTIDX CE/PE contracts found in configured Security Masters")
    keep = [
        "security_id", "expiry_date", "option_type", "strike",
        "security_master_source",
    ]
    if "DISPLAY_NAME" in contracts:
        keep.append("DISPLAY_NAME")
    return contracts[keep].reset_index(drop=True)


def date_chunks(start: date, end_exclusive: date, chunk_days: int):
    cursor = start
    while cursor < end_exclusive:
        next_cursor = min(cursor + timedelta(days=chunk_days), end_exclusive)
        yield cursor, next_cursor
        cursor = next_cursor


def plan_contract_requests(
    contracts: pd.DataFrame, start: date, end_exclusive: date, config: dict
) -> list[dict]:
    """Return exact REST request payloads without making network calls."""
    plans = []
    for contract in contracts.itertuples(index=False):
        expiry = pd.Timestamp(contract.expiry_date).date()
        contract_start = max(start, expiry - timedelta(days=int(config["contract_history_days"])))
        contract_end = min(end_exclusive, expiry + timedelta(days=1))
        if contract_start >= contract_end:
            continue
        for chunk_start, chunk_end in date_chunks(
            contract_start, contract_end, int(config["chunk_days"])
        ):
            plans.append({
                "security_id": str(contract.security_id),
                "expiry_date": str(expiry),
                "chunk_start": str(chunk_start),
                "chunk_end_exclusive": str(chunk_end),
                "payload": {
                    "securityId": str(contract.security_id),
                    "exchangeSegment": config["exchange_segment"],
                    "instrument": config["instrument"],
                    "interval": str(config["interval_minutes"]),
                    "oi": bool(config["include_open_interest"]),
                    "fromDate": f"{chunk_start} 00:00:00",
                    "toDate": f"{chunk_end} 00:00:00",
                },
            })
    return plans


def response_to_frame(response: dict, timezone: str) -> pd.DataFrame:
    if response.get("status") == "failure" or response.get("errorCode"):
        raise DhanAPIError("Dhan API returned an error payload", response_body=json.dumps(response, ensure_ascii=False))
    payload = response.get("data", response)
    required = ("timestamp", "open", "high", "low", "close", "volume")
    if not all(key in payload for key in required):
        raise ValueError(f"Dhan response missing candle arrays: {sorted(set(required).difference(payload))}")
    length_keys = list(required) + (["open_interest"] if "open_interest" in payload else [])
    lengths = {len(payload[key]) for key in length_keys}
    if len(lengths) != 1:
        raise ValueError("Dhan response candle arrays have inconsistent lengths")
    timestamp = pd.to_datetime(payload["timestamp"], unit="s", utc=True).tz_convert(timezone)
    frame = pd.DataFrame({"timestamp": timestamp, **{key: payload[key] for key in required[1:]}})
    if "open_interest" in payload:
        frame["open_interest"] = payload["open_interest"]
    return frame


def validate_chunk(frame: pd.DataFrame, payload: dict, timezone: str, require_oi: bool) -> dict:
    """Fail closed on malformed, out-of-range, unordered, or incomplete candles."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Chunk missing normalized columns: {sorted(missing)}")
    if frame.empty:
        return {"valid": True, "empty": True, "rows": 0, "open_interest_present": "open_interest" in frame}
    if require_oi and "open_interest" not in frame:
        raise ValueError("Open Interest requested but open_interest is absent from non-empty Dhan response")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(timezone)
    if str(timestamp.dt.tz) != timezone:
        raise ValueError(f"Unexpected chunk timezone: {timestamp.dt.tz}")
    if timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("Chunk timestamps must be unique and chronologically sorted")
    start = pd.Timestamp(payload["fromDate"], tz=timezone)
    end = pd.Timestamp(payload["toDate"], tz=timezone)
    if timestamp.min() < start or timestamp.max() >= end:
        raise ValueError(f"Chunk timestamps outside requested half-open range [{start}, {end})")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    if require_oi:
        numeric_columns.append("open_interest")
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Chunk contains non-numeric or missing OHLCV/OI values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Chunk contains non-positive prices")
    if (numeric["volume"] < 0).any() or (require_oi and (numeric["open_interest"] < 0).any()):
        raise ValueError("Chunk contains negative volume or open interest")
    if numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Chunk contains invalid high prices")
    if numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Chunk contains invalid low prices")
    return {
        "valid": True, "empty": False, "rows": int(len(frame)),
        "first_timestamp": timestamp.iloc[0].isoformat(), "last_timestamp": timestamp.iloc[-1].isoformat(),
        "open_interest_present": "open_interest" in frame,
    }


class DhanHistoricalDownloader:
    def __init__(self, config: dict, requester: Callable | None = None, sleeper=time.sleep):
        self.config, self.sleeper = config, sleeper
        paths = config["paths"]
        self.chunk_dir = Path(paths["chunk_directory"])
        self.manifest_path = Path(paths["manifest"])
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        Path(paths["request_log"]).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"dhan_history.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.FileHandler(paths["request_log"], encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)
        self.client_id = os.getenv(config["credentials"]["client_id_env"])
        self.access_token = os.getenv(config["credentials"]["access_token_env"])
        self.requester = requester or self._post

    def close(self):
        """Flush and close request-log handles (important for clean restarts on Windows)."""
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _post(self, payload):
        if not self.access_token:
            raise EnvironmentError(
                f"Missing Dhan access token environment variable: {self.config['credentials']['access_token_env']}"
            )
        headers = {"Accept": "application/json", "Content-Type": "application/json",
                   "access-token": self.access_token}
        if self.client_id:
            headers["client-id"] = self.client_id
        response = requests.post(API_URL, headers=headers, json=payload,
                                 timeout=float(self.config["request_timeout_seconds"]))
        body = response.text
        if not response.ok:
            raise DhanAPIError("Dhan Historical REST request failed", response.status_code, body)
        try:
            return response.json()
        except ValueError as exc:
            raise DhanAPIError("Dhan Historical REST returned non-JSON", response.status_code, body) from exc

    def _manifest(self):
        if not self.manifest_path.exists():
            return {"version": 1, "chunks": {}}
        with self.manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _save_manifest(self, manifest):
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def download_contracts(self, contracts: pd.DataFrame, start: date, end_exclusive: date):
        manifest = self._manifest()
        output_files = []
        for contract in contracts.itertuples(index=False):
            expiry = pd.Timestamp(contract.expiry_date).date()
            contract_start = max(start, expiry - timedelta(days=int(self.config["contract_history_days"])))
            contract_end = min(end_exclusive, expiry + timedelta(days=1))
            if contract_start >= contract_end:
                continue
            for chunk_start, chunk_end in date_chunks(contract_start, contract_end, int(self.config["chunk_days"])):
                key = f"{contract.security_id}_{chunk_start}_{chunk_end}"
                path = self.chunk_dir / str(contract.security_id) / f"{chunk_start}_{chunk_end}.parquet"
                record = manifest["chunks"].get(key, {})
                if record.get("status") == "complete" and path.exists():
                    self.logger.info("resume-skip key=%s file=%s rows=%s", key, path, record.get("rows"))
                    output_files.append(path)
                    continue
                payload = {
                    "securityId": str(contract.security_id), "exchangeSegment": self.config["exchange_segment"],
                    "instrument": self.config["instrument"], "interval": str(self.config["interval_minutes"]),
                    "oi": bool(self.config["include_open_interest"]),
                    "fromDate": f"{chunk_start} 00:00:00", "toDate": f"{chunk_end} 00:00:00",
                }
                try:
                    frame, validation = self._download_with_retry(key, payload)
                except Exception as exc:
                    manifest["chunks"][key] = {
                        "status": "failed", "security_id": str(contract.security_id),
                        "from": str(chunk_start), "to": str(chunk_end), "exact_error": str(exc),
                        "failed_at": pd.Timestamp.now(tz=self.config["timezone"]).isoformat(),
                    }
                    self._save_manifest(manifest)
                    raise
                frame["security_id"] = str(contract.security_id)
                frame["contract_expiry"] = pd.Timestamp(contract.expiry_date)
                if hasattr(contract, "option_type"):
                    frame["option_type"] = contract.option_type
                if hasattr(contract, "strike"):
                    frame["strike"] = contract.strike
                if hasattr(contract, "DISPLAY_NAME"):
                    frame["contract_name"] = contract.DISPLAY_NAME
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.parquet")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
                manifest["chunks"][key] = {
                    "status": "complete", "file": str(path), "rows": len(frame),
                    "security_id": str(contract.security_id), "from": str(chunk_start), "to": str(chunk_end),
                    "validation": validation,
                }
                self._save_manifest(manifest)
                output_files.append(path)
        return output_files

    def _download_with_retry(self, key, payload):
        retries = int(self.config["max_retries"])
        for attempt in range(1, retries + 1):
            self.logger.info("request key=%s attempt=%s security_id=%s from=%s to=%s",
                             key, attempt, payload["securityId"], payload["fromDate"], payload["toDate"])
            try:
                response = self.requester(payload)
                frame = response_to_frame(response, self.config["timezone"])
                validation = validate_chunk(
                    frame, payload, self.config["timezone"], bool(self.config["include_open_interest"])
                )
                self.logger.info("success key=%s attempt=%s rows=%s validation=%s",
                                 key, attempt, len(frame), json.dumps(validation, ensure_ascii=False))
                return frame, validation
            except Exception as exc:
                self.logger.exception("failure key=%s attempt=%s error=%s", key, attempt, exc)
                if attempt == retries:
                    raise
                self.sleeper(float(self.config["retry_backoff_seconds"]) * (2 ** (attempt - 1)))
        raise RuntimeError("unreachable")


def catalog_coverage(contracts: pd.DataFrame, desired_start: date, desired_end: date) -> dict:
    expiries = pd.to_datetime(contracts["expiry_date"])
    gaps = expiries.sort_values().drop_duplicates().diff().dt.days.dropna()
    gap_count = int(gaps.gt(62).sum())
    complete = bool(
        expiries.min().date() <= desired_start + timedelta(days=120)
        and expiries.max().date() >= desired_end
        and gap_count == 0
    )
    return {
        "contracts_discovered": int(len(contracts)), "earliest_expiry": str(expiries.min().date()),
        "latest_expiry": str(expiries.max().date()), "desired_start": str(desired_start),
        "desired_end": str(desired_end),
        "catalog_gaps_over_62_days": gap_count,
        "maximum_expiry_gap_days": int(gaps.max()) if len(gaps) else None,
        "five_year_catalog_complete": complete,
    }
