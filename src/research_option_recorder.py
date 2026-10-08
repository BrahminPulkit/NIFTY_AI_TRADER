"""Research-only Dhan option-chain recorder.

This module deliberately has no order, prediction, strategy, or paper-trading
surface. Provider payloads are appended unchanged to an immutable JSONL archive;
normalization and quality reporting are separate derived steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable
from uuid import uuid4
import json
import math
import os

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from src.dhan_live_broker import BrokerCredentials, DhanReadOnlyClient, _safe_error


IST = ZoneInfo("Asia/Kolkata")
EXPIRY_LIST_URL = "https://api.dhan.co/v2/optionchain/expirylist"
OPTION_CHAIN_URL = "https://api.dhan.co/v2/optionchain"


class RecorderError(RuntimeError):
    pass


class AuthenticationExpired(RecorderError):
    pass


class ProviderRequestError(RecorderError):
    """Sanitized provider failure with enough detail for immutable auditing."""

    def __init__(self, detail: str, *, status_code: int | None = None,
                 response_body: Any = None, retry_after_seconds: float | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.response_body = response_body
        self.retry_after_seconds = retry_after_seconds

    @property
    def rate_limited(self) -> bool:
        return self.status_code == 429


@dataclass(frozen=True)
class RecorderConfig:
    schema_version: str
    recorder_version: str
    underlying: str
    underlying_segment: str
    sampling_interval_seconds: int
    strikes_each_side: int
    session_timezone: str
    session_open: str
    session_close: str
    synchronization_tolerance_seconds: float
    stale_quote_seconds: float
    minimum_snapshot_coverage_pct: float
    minimum_contracts_per_snapshot: int
    maximum_gap_seconds: float
    maximum_consecutive_failures: int
    maximum_auth_failures: int
    raw_root: str
    normalized_root: str
    audit_root: str
    rate_limit_backoff_seconds: float = 6.0
    network_backoff_seconds: float = 2.0
    maximum_backoff_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.sampling_interval_seconds < 3:
            raise ValueError("Dhan Option Chain interval cannot be below its documented 3-second limit")
        if self.rate_limit_backoff_seconds < 3:
            raise ValueError("Rate-limit backoff must be at least 3 seconds")

    @classmethod
    def load(cls, path: str | Path) -> "RecorderConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def market_times(self, session_date: date) -> tuple[datetime, datetime]:
        zone = ZoneInfo(self.session_timezone)
        start = datetime.combine(session_date, time.fromisoformat(self.session_open), zone)
        end = datetime.combine(session_date, time.fromisoformat(self.session_close), zone)
        return start, end


def credentials_from_environment() -> BrokerCredentials:
    client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
    token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not client_id or not token:
        raise RecorderError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are required in the environment")
    return BrokerCredentials(client_id, token)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=_json_safe, separators=(",", ":")) + "\n")
        handle.flush()


def _parse_provider_time(value: Any) -> pd.Timestamp | None:
    if value in (None, "", 0):
        return None
    try:
        if isinstance(value, (int, float)):
            stamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(IST)
        return stamp.tz_convert(IST)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


class DhanOptionChainSource:
    """Minimal authenticated source for expiry metadata and chain snapshots."""

    def __init__(self, credentials: BrokerCredentials, *, requester=requests,
                 timeout_seconds: int = 20):
        self.credentials = credentials
        self.requester = requester
        self.timeout_seconds = timeout_seconds
        self.profile_client = DhanReadOnlyClient(
            credentials, requester=requester, timeout_seconds=timeout_seconds)

    def authenticate(self) -> dict[str, Any]:
        health = self.profile_client.test_connection()
        if health.status != "CONNECTED":
            raise AuthenticationExpired(health.detail)
        return health.public_dict()

    def _post(self, url: str, payload: dict[str, Any]) -> tuple[int, Any]:
        try:
            response = self.requester.post(
                url, headers=self.credentials.headers(), json=payload,
                timeout=self.timeout_seconds)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise ConnectionError("Dhan option-chain endpoint could not be reached") from exc
        if response.status_code != 200:
            state, detail = _safe_error(response)
            if state in {"INVALID TOKEN", "TOKEN EXPIRED"}:
                raise AuthenticationExpired(detail)
            try:
                response_body = response.json()
            except Exception:
                response_body = {"unparsed_response": str(getattr(response, "text", ""))[:2000]}
            retry_after = None
            try:
                retry_after = float((getattr(response, "headers", {}) or {}).get("Retry-After"))
            except (TypeError, ValueError):
                pass
            raise ProviderRequestError(
                detail, status_code=response.status_code,
                response_body=response_body, retry_after_seconds=retry_after)
        return response.status_code, response.json()

    def active_expiry(self, *, underlying_security_id: int,
                      session_date: date) -> str:
        _, body = self._post(EXPIRY_LIST_URL, {
            "UnderlyingScrip": underlying_security_id,
            "UnderlyingSeg": "IDX_I",
        })
        data = body.get("data", body) if isinstance(body, dict) else body
        expiries = data if isinstance(data, list) else data.get("expiry", [])
        parsed = sorted(
            pd.Timestamp(item).date() for item in expiries
            if pd.Timestamp(item).date() >= session_date)
        if not parsed:
            raise RecorderError("Dhan returned no active NIFTY expiry")
        return parsed[0].isoformat()

    def snapshot(self, *, underlying_security_id: int, expiry: str) -> tuple[int, Any]:
        return self._post(OPTION_CHAIN_URL, {
            "UnderlyingScrip": underlying_security_id,
            "UnderlyingSeg": "IDX_I",
            "Expiry": expiry,
        })


def _strike_window(strikes: list[float], spot: float, each_side: int) -> tuple[float, list[float]]:
    ordered = sorted(set(strikes))
    if not ordered:
        raise RecorderError("Option chain contains no strikes")
    atm = min(ordered, key=lambda strike: (abs(strike - spot), strike))
    index = ordered.index(atm)
    desired = 2 * each_side + 1
    start = max(0, min(index - each_side, len(ordered) - desired))
    return atm, ordered[start:start + desired]


def normalize_snapshot(*, raw_body: dict[str, Any], snapshot_id: str,
                       received_at: datetime, expiry: str,
                       each_side: int, sync_tolerance_seconds: float,
                       stale_quote_seconds: float,
                       schema_version: str = "phase19-options-live-v1") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain = raw_body.get("oc", {}) if isinstance(raw_body, dict) else {}
    spot = _value(raw_body, "last_price", "underlying_ltp", "spot", "ltp")
    if spot is None or not chain:
        raise RecorderError("Dhan option-chain response is missing spot or contracts")
    spot = float(spot)
    available = [float(key) for key in chain]
    atm, selected = _strike_window(available, spot, each_side)
    received = pd.Timestamp(received_at).tz_convert(IST)
    expiry_date = pd.Timestamp(expiry).date()
    minutes_to_expiry = max(
        0.0, (pd.Timestamp(datetime.combine(expiry_date, time(15, 30), IST)) - received).total_seconds() / 60)
    calendar_dte = (expiry_date - received.date()).days
    dte_bucket = f"{calendar_dte}DTE" if calendar_dte <= 2 else "3DTE+"
    rows: list[dict[str, Any]] = []
    provider_times: list[pd.Timestamp] = []
    for strike in selected:
        legs = chain.get(str(strike), chain.get(f"{strike:.6f}", chain.get(str(int(strike)), {}))) or {}
        for side, key in (("CE", "ce"), ("PE", "pe")):
            quote = legs.get(key) or legs.get(side) or {}
            provider_time = _parse_provider_time(_value(
                quote, "last_trade_time", "lastTradeTime", "timestamp", "exchange_time"))
            if provider_time is not None:
                provider_times.append(provider_time)
            lag = ((received - provider_time).total_seconds()
                   if provider_time is not None else None)
            greeks = quote.get("greeks") or {}
            bid = _value(quote, "top_bid_price", "bid", "best_bid_price")
            ask = _value(quote, "top_ask_price", "ask", "best_ask_price")
            security_id = _value(quote, "security_id", "securityId")
            contract_id = f"NIFTY|{expiry}|{strike:g}|{side}"
            rows.append({
                "schema_version": schema_version,
                "snapshot_id": snapshot_id,
                "snapshot_timestamp": received.isoformat(),
                "provider_timestamp": provider_time.isoformat() if provider_time is not None else None,
                "local_receive_timestamp": received.isoformat(),
                "synchronization_lag_seconds": lag,
                "synchronization_ok": lag is not None and 0 <= lag <= sync_tolerance_seconds,
                "quote_stale": lag is None or lag > stale_quote_seconds,
                "underlying": "NIFTY", "spot_ltp": spot,
                "nifty_open": _value(raw_body, "open"),
                "nifty_high": _value(raw_body, "high"),
                "nifty_low": _value(raw_body, "low"),
                "nifty_close": _value(raw_body, "close"),
                "nifty_volume": None,
                "expiry": expiry, "calendar_dte": calendar_dte,
                "trading_session_dte": None,
                "minutes_to_expiry": minutes_to_expiry, "dte_bucket": dte_bucket,
                "strike": strike, "atm_strike": atm,
                "strike_offset": selected.index(strike) - selected.index(atm),
                "moneyness_points": strike - spot,
                "option_type": side, "security_id": security_id,
                "contract_id": contract_id,
                "ltp": _value(quote, "last_price", "ltp"),
                "bid": bid, "ask": ask,
                "spread": (float(ask) - float(bid)) if bid is not None and ask is not None else None,
                "volume": _value(quote, "volume"), "oi": _value(quote, "oi"),
                "iv": _value(quote, "implied_volatility", "iv"),
                "delta": _value(greeks, "delta"), "gamma": _value(greeks, "gamma"),
                "theta": _value(greeks, "theta"), "vega": _value(greeks, "vega"),
            })
    meta = {
        "spot": spot, "atm_strike": atm, "selected_strikes": selected,
        "contract_count": len(rows),
        "provider_timestamp_available": bool(provider_times),
    }
    return rows, meta


class ResearchOptionRecorder:
    def __init__(self, config: RecorderConfig, source: DhanOptionChainSource,
                 *, underlying_security_id: int = 13,
                 now: Callable[[], datetime] | None = None,
                 sleeper: Callable[[float], None] = sleep):
        self.config = config
        self.source = source
        self.underlying_security_id = underlying_security_id
        self.now = now or (lambda: datetime.now(IST))
        self.sleeper = sleeper

    def paths(self, session_date: date) -> dict[str, Path]:
        day = session_date.isoformat()
        return {
            "raw": Path(self.config.raw_root) / day / "provider_responses.jsonl",
            "raw_errors": Path(self.config.raw_root) / day / "provider_errors.jsonl",
            "events": Path(self.config.raw_root) / day / "recorder_events.jsonl",
            "normalized": Path(self.config.normalized_root) / day / "snapshots.parquet",
            "audit": Path(self.config.audit_root) / f"{day}.json",
        }

    def _event(self, path: Path, event: str, **detail: Any) -> None:
        _append_jsonl(path, {"timestamp": self.now().isoformat(), "event": event, **detail})

    def record_one(self, session_date: date, expiry: str | None = None) -> dict[str, Any]:
        paths = self.paths(session_date)
        expiry = expiry or self.source.active_expiry(
            underlying_security_id=self.underlying_security_id, session_date=session_date)
        snapshot_id = f"{session_date.isoformat()}-{uuid4()}"
        started = self.now()
        try:
            status, body = self.source.snapshot(
                underlying_security_id=self.underlying_security_id, expiry=expiry)
            received = self.now()
            raw_record = {
                "snapshot_id": snapshot_id, "requested_at": started.isoformat(),
                "received_at": received.isoformat(), "http_status": status,
                "endpoint": OPTION_CHAIN_URL,
                "request": {"UnderlyingScrip": self.underlying_security_id,
                            "UnderlyingSeg": "IDX_I", "Expiry": expiry},
                "response": body,
            }
            raw_record["response_sha256"] = sha256(
                json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
            _append_jsonl(paths["raw"], raw_record)
            provider_data = body.get("data", body) if isinstance(body, dict) else body
            rows, meta = normalize_snapshot(
                raw_body=provider_data, snapshot_id=snapshot_id, received_at=received,
                expiry=expiry, each_side=self.config.strikes_each_side,
                sync_tolerance_seconds=self.config.synchronization_tolerance_seconds,
                stale_quote_seconds=self.config.stale_quote_seconds,
                schema_version=self.config.schema_version)
            frame = pd.DataFrame(rows)
            paths["normalized"].parent.mkdir(parents=True, exist_ok=True)
            if paths["normalized"].exists():
                frame = pd.concat([pd.read_parquet(paths["normalized"]), frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["snapshot_id", "contract_id"], keep="first")
            frame.to_parquet(paths["normalized"], index=False)
            return {"snapshot_id": snapshot_id, "expiry": expiry, **meta}
        except ProviderRequestError as exc:
            _append_jsonl(paths["raw_errors"], {
                "snapshot_id": snapshot_id, "requested_at": started.isoformat(),
                "failed_at": self.now().isoformat(), "endpoint": OPTION_CHAIN_URL,
                "request": {"UnderlyingScrip": self.underlying_security_id,
                            "UnderlyingSeg": "IDX_I", "Expiry": expiry},
                "http_status": exc.status_code, "response": exc.response_body,
                "retry_after_seconds": exc.retry_after_seconds,
            })
            self._event(paths["events"], "RATE_LIMITED" if exc.rate_limited else "PROVIDER_ERROR",
                        http_status=exc.status_code,
                        retry_after_seconds=exc.retry_after_seconds)
            self._event(paths["events"], "SNAPSHOT_ERROR", error_type=type(exc).__name__, detail=str(exc))
            raise
        except Exception as exc:
            self._event(paths["events"], "SNAPSHOT_ERROR", error_type=type(exc).__name__, detail=str(exc))
            raise

    def _failure_delay(self, failures: int, exc: Exception) -> float:
        base = (self.config.rate_limit_backoff_seconds
                if isinstance(exc, ProviderRequestError) and exc.rate_limited
                else self.config.network_backoff_seconds)
        delay = base * (2 ** max(0, failures - 1))
        if isinstance(exc, ProviderRequestError) and exc.retry_after_seconds is not None:
            delay = max(delay, exc.retry_after_seconds)
        return min(delay, self.config.maximum_backoff_seconds)

    def run_session(self, session_date: date | None = None) -> dict[str, Any]:
        session_date = session_date or self.now().date()
        paths = self.paths(session_date)
        if session_date.weekday() >= 5:
            self._event(paths["events"], "MARKET_CLOSED", reason="WEEKEND")
            return {"status": "MARKET_CLOSED", "reason": "WEEKEND"}
        health = self.source.authenticate()
        self._event(paths["events"], "AUTHENTICATED", status=health["status"])
        start, end = self.config.market_times(session_date)
        while self.now() < start:
            self.sleeper(min(30, max(0.1, (start - self.now()).total_seconds())))
        if self.now() >= end:
            self._event(paths["events"], "MARKET_CLOSED", reason="AFTER_CLOSE")
            return {"status": "MARKET_CLOSED", "reason": "AFTER_CLOSE"}
        expiry = self.source.active_expiry(
            underlying_security_id=self.underlying_security_id, session_date=session_date)
        failures = reconnects = snapshots = 0
        while self.now() < end:
            delay = float(self.config.sampling_interval_seconds)
            try:
                self.record_one(session_date, expiry)
                snapshots += 1
                if failures:
                    reconnects += 1
                    self._event(paths["events"], "RECONNECTED", failures=failures)
                failures = 0
            except AuthenticationExpired:
                self._event(paths["events"], "AUTHENTICATION_EXPIRED")
                break
            except (ConnectionError, RecorderError) as exc:
                failures += 1
                delay = self._failure_delay(failures, exc)
                if failures >= self.config.maximum_consecutive_failures:
                    self._event(paths["events"], "OUTAGE", consecutive_failures=failures,
                                detail=str(exc))
            # Schedule from the completed attempt. Never replay missed deadlines.
            self.sleeper(delay)
        audit = audit_session(session_date, self.config)
        audit.update({"runtime_snapshots": snapshots, "runtime_reconnects": reconnects})
        paths["audit"].parent.mkdir(parents=True, exist_ok=True)
        paths["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return audit


def audit_session(session_date: date, config: RecorderConfig) -> dict[str, Any]:
    recorder_paths = ResearchOptionRecorder.__new__(ResearchOptionRecorder)
    recorder_paths.config = config
    paths = ResearchOptionRecorder.paths(recorder_paths, session_date)
    start, end = config.market_times(session_date)
    expected = math.ceil((end - start).total_seconds() / config.sampling_interval_seconds)
    if not paths["normalized"].exists():
        return {"date": session_date.isoformat(), "expected_snapshots": expected,
                "actual_snapshots": 0, "complete": False, "status": "NO_DATA"}
    frame = pd.read_parquet(paths["normalized"])
    snapshot_rows = frame.groupby("snapshot_id", sort=False).first().reset_index()
    stamps = pd.to_datetime(snapshot_rows["snapshot_timestamp"], utc=True).sort_values()
    gaps = stamps.diff().dt.total_seconds().dropna()
    counts = frame.groupby("snapshot_id")["contract_id"].nunique()
    actual = len(snapshot_rows)
    coverage = 100 * actual / expected if expected else 0
    duplicate_rows = int(frame.duplicated(["snapshot_id", "contract_id"]).sum())
    conflicts = int(frame.duplicated(["snapshot_timestamp", "contract_id"], keep=False).sum())
    fields = ["ltp", "bid", "ask", "volume", "oi", "iv", "delta", "gamma", "theta", "vega"]
    missing = {f"missing_{field}": int(frame[field].isna().sum()) for field in fields}
    max_gap = float(gaps.max()) if len(gaps) else None
    complete = bool(
        coverage >= config.minimum_snapshot_coverage_pct
        and counts.min() >= config.minimum_contracts_per_snapshot
        and duplicate_rows == 0
        and (max_gap is None or max_gap <= config.maximum_gap_seconds)
        and frame["ltp"].notna().all()
        and frame["bid"].notna().all()
        and frame["ask"].notna().all())
    raw_error_count = 0
    rate_limit_count = 0
    if paths["raw_errors"].exists():
        with paths["raw_errors"].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_error_count += 1
                try:
                    rate_limit_count += int(json.loads(line).get("http_status") == 429)
                except json.JSONDecodeError:
                    pass
    return {
        "date": session_date.isoformat(), "schema_version": config.schema_version,
        "expected_snapshots": expected, "actual_snapshots": actual,
        "missing_snapshots": max(0, expected - actual), "coverage_pct": coverage,
        "duplicate_normalized_rows": duplicate_rows,
        "conflicting_observations": conflicts,
        "median_gap_seconds": float(gaps.median()) if len(gaps) else None,
        "maximum_gap_seconds": max_gap,
        "contracts_min": int(counts.min()), "contracts_max": int(counts.max()),
        "strike_universe_changes": int(snapshot_rows["atm_strike"].nunique() - 1),
        "synchronization_available_pct": float(100 * frame["provider_timestamp"].notna().mean()),
        "synchronization_ok_pct": float(100 * frame["synchronization_ok"].mean()),
        "stale_quote_count": int(frame["quote_stale"].sum()),
        "raw_archive_exists": paths["raw"].exists(), **missing,
        "raw_error_count": raw_error_count, "rate_limit_count": rate_limit_count,
        "complete": complete, "status": "COMPLETE" if complete else "PARTIAL",
    }
