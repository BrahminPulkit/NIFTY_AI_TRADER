"""Research-only acquisition from explicit expired NIFTY option archives.

The source archive identifies a contract by expiry archive, strike/side filename,
and ticker.  Broker security IDs are intentionally left null.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import zipfile

import pandas as pd

from src.ce_pe_feature_preview import build_ce_pe_feature_preview, feature_columns
from src.option_contract_pipeline import normalize_option_contracts


FILE_PATTERN = re.compile(r"(?P<strike>\d+)(?P<side>CE|PE)_(?P<expiry>\d{8})\.csv$")
TICKER_PATTERN = re.compile(
    r"^NIFTY(?P<day>\d{2})(?P<month>[A-Z]{3})(?P<year>\d{2})(?P<side>CE|PE|C|P)(?P<strike>\d+)$"
)
SESSION_TIMESTAMPS = pd.date_range("09:15", "15:29", freq="1min").strftime("%H:%M").tolist()


@dataclass(frozen=True)
class SessionResult:
    date: str
    expiry: str
    nifty: pd.DataFrame
    ce: pd.DataFrame
    pe: pd.DataFrame
    audit: dict


def _archive_contracts(path: Path) -> dict[tuple[str, int], str]:
    contracts: dict[tuple[str, int], str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = FILE_PATTERN.search(name)
            if not match:
                continue
            if match.group("expiry") != path.stem:
                raise ValueError(f"Expiry mismatch in archive member: {name}")
            key = match.group("side"), int(match.group("strike"))
            if key in contracts:
                raise ValueError(f"Duplicate contract file for {key}")
            contracts[key] = name
    return contracts


def _read_contract(path: Path, member: str, *, date: str, expiry: str,
                   strike: int, side: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        raw = pd.read_csv(handle)
    raw = raw.loc[raw["Date"].astype(str).eq(date)].copy()
    if raw.empty:
        return raw
    expected_sides = {"C", "CE"} if side == "CE" else {"P", "PE"}
    for ticker in raw["Ticker"].astype(str).unique():
        parsed = TICKER_PATTERN.fullmatch(ticker)
        if (not parsed or parsed.group("side") not in expected_sides
                or int(parsed.group("strike")) != strike):
            raise ValueError(f"Ticker identity mismatch: {ticker}")
        ticker_expiry = pd.to_datetime(
            f"{parsed.group('day')}{parsed.group('month')}{parsed.group('year')}",
            format="%d%b%y",
        ).strftime("%Y-%m-%d")
        if ticker_expiry != expiry:
            raise ValueError(f"Ticker expiry mismatch: {ticker}")
    timestamp = pd.to_datetime(raw["Timestamp"], format="%d-%m-%Y %H:%M:%S")
    raw["timestamp"] = timestamp.dt.tz_localize("Asia/Kolkata")
    raw = raw.rename(columns={name: name.lower() for name in (
        "Open", "High", "Low", "Close", "Volume", "OI")})
    value_columns = ["open", "high", "low", "close", "volume", "oi", "Ticker"]
    conflicts = raw.groupby("timestamp")[value_columns].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting duplicate contract minutes: {int(conflicts.sum())}")
    duplicate_rows = int(raw.timestamp.duplicated().sum())
    raw = raw.drop_duplicates("timestamp", keep="first").copy()
    raw["underlying"] = "NIFTY"
    raw["expiry"] = expiry
    raw["strike"] = strike
    raw["option_type"] = side
    raw["security_id"] = pd.NA
    raw["source"] = "SHOONYA_TRADER_EXPIRED_OPTIONS"
    result = raw[["timestamp", "underlying", "expiry", "strike", "option_type",
                  "security_id", "open", "high", "low", "close", "volume", "oi", "source"]]
    result.attrs["source_duplicate_rows_dropped"] = duplicate_rows
    return result


def acquire_expiry_session(archive_path: str | Path, nifty_source: pd.DataFrame,
                           session_date: str | None = None) -> SessionResult:
    """Build one exact, rolling-ATM expiry-day research session without filling gaps."""
    path = Path(archive_path)
    expiry = pd.to_datetime(path.stem, format="%Y%m%d").strftime("%Y-%m-%d")
    date = pd.Timestamp(session_date or expiry).strftime("%Y-%m-%d")
    if date > expiry:
        raise ValueError("Session date cannot be after contract expiry")
    nifty = nifty_source.copy()
    nifty["timestamp"] = pd.to_datetime(nifty["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    nifty = nifty.loc[nifty.timestamp.dt.strftime("%Y-%m-%d").eq(date),
                      ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    nifty = nifty.sort_values("timestamp", kind="stable").reset_index(drop=True)
    contracts = _archive_contracts(path)
    common = sorted({strike for side, strike in contracts if side == "CE"}
                    & {strike for side, strike in contracts if side == "PE"})
    if not common:
        raise ValueError(f"No common CE/PE strikes for {expiry}")
    common_series = pd.Series(common, dtype="float64")
    selected = nifty.close.map(lambda spot: int(common_series.iloc[(common_series - spot).abs().argmin()]))
    needed = sorted(selected.unique())
    side_frames: dict[str, pd.DataFrame] = {}
    source_duplicates: dict[str, int] = {}
    for side in ("CE", "PE"):
        frames = [_read_contract(path, contracts[(side, strike)], date=date,
                                 expiry=expiry, strike=strike, side=side)
                  for strike in needed]
        source_duplicates[side] = sum(
            int(frame.attrs.get("source_duplicate_rows_dropped", 0)) for frame in frames)
        available = pd.concat(frames, ignore_index=True)
        selector = nifty[["timestamp"]].assign(strike=selected.to_numpy())
        chosen = selector.merge(available, on=["timestamp", "strike"], how="left",
                                validate="one_to_one")
        if chosen["close"].isna().any():
            raise ValueError(f"Missing {side} ATM quotes: {int(chosen.close.isna().sum())}")
        chosen["spot"] = nifty.close.to_numpy()
        chosen = normalize_option_contracts(chosen)
        chosen = chosen.sort_values("timestamp", kind="stable").reset_index(drop=True)
        transition = chosen.strike.ne(chosen.strike.shift()).cumsum()
        chosen["contract_segment_id"] = [
            hashlib.sha256(
                f"{date}|{expiry}|{side}|{int(strike)}|run-{int(run)}".encode()
            ).hexdigest()[:20]
            for strike, run in zip(chosen.strike, transition)
        ]
        side_frames[side] = normalize_option_contracts(chosen)

    ce, pe = side_frames["CE"], side_frames["PE"]
    expected = pd.to_datetime(
        [f"{date} {value}" for value in SESSION_TIMESTAMPS]
    ).tz_localize("Asia/Kolkata")
    expected_set = set(expected)
    nifty_set, ce_set, pe_set = map(set, (nifty.timestamp, ce.timestamp, pe.timestamp))
    duplicates = int(nifty.timestamp.duplicated().sum() + ce.timestamp.duplicated().sum()
                     + pe.timestamp.duplicated().sum())
    overlap = len(nifty_set & ce_set & pe_set)
    complete = (len(nifty) == len(ce) == len(pe) == 375 and duplicates == 0
                and nifty_set == ce_set == pe_set == expected_set)
    audit = {
        "date": date, "expiry": expiry, "expiry_policy": "WEEK_CURRENT_NEAR",
        "expiry_day_status": "EXPIRY_DAY" if date == expiry else "NON_EXPIRY_DAY",
        "strike_policy": "ATM_NEAREST_COMMON_CE_PE_STRIKE",
        "nifty_rows": len(nifty), "ce_rows": len(ce), "pe_rows": len(pe),
        "overlap_rows": overlap, "missing_nifty": len(expected_set - nifty_set),
        "missing_ce": len(expected_set - ce_set), "missing_pe": len(expected_set - pe_set),
        "duplicate_timestamps": duplicates, "strike_changes": int(selected.ne(selected.shift()).sum() - 1),
        "source_duplicate_rows_dropped_ce": source_duplicates["CE"],
        "source_duplicate_rows_dropped_pe": source_duplicates["PE"],
        "ce_segments": int(ce.contract_segment_id.nunique()),
        "pe_segments": int(pe.contract_segment_id.nunique()),
        "unique_strikes": [int(value) for value in sorted(selected.unique())],
        "security_id_available": False, "research_training_eligible": bool(complete),
        "live_trading_eligible": False, "status": "TRAINING_READY" if complete else "PARTIAL",
    }
    return SessionResult(date, expiry, nifty, ce, pe, audit)


def build_session_features(result: SessionResult) -> pd.DataFrame:
    features = build_ce_pe_feature_preview(result.nifty, result.ce, result.pe)
    if len(feature_columns()) != 49:
        raise AssertionError("Phase 6 feature contract is not 49 columns")
    features.insert(0, "session_date", result.date)
    return features


def load_selected_contract_panel(archive_path: str | Path, ce: pd.DataFrame,
                                 pe: pd.DataFrame, session_date: str | None = None) -> pd.DataFrame:
    """Load full-session candles for every contract selected at an entry timestamp."""
    path = Path(archive_path)
    expiry = pd.to_datetime(path.stem, format="%Y%m%d").strftime("%Y-%m-%d")
    date = pd.Timestamp(session_date or expiry).strftime("%Y-%m-%d")
    contracts = _archive_contracts(path)
    frames = []
    for side, selected in (("CE", ce), ("PE", pe)):
        for strike in sorted(pd.to_numeric(selected.strike).astype(int).unique()):
            member = contracts.get((side, strike))
            if member is None:
                raise ValueError(f"Full contract unavailable: {expiry} {strike} {side}")
            frame = _read_contract(path, member, date=date, expiry=expiry,
                                   strike=strike, side=side)
            if frame.empty:
                raise ValueError(f"Full contract has no session candles: {expiry} {strike} {side}")
            frames.append(frame)
    panel = normalize_option_contracts(pd.concat(frames, ignore_index=True))
    identity = ["expiry", "strike", "option_type"]
    if panel.groupby(identity).contract_segment_id.nunique().gt(1).any():
        raise AssertionError("A fixed contract was split into multiple panel segments")
    return panel.sort_values([*identity, "timestamp"], kind="stable").reset_index(drop=True)
