"""Build and validate an unadjusted continuous NIFTY futures series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def trading_day_roll_date(expiry, trading_days, rollover_days):
    expiry = pd.Timestamp(expiry).normalize()
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().unique().sort_values()
    # Completed contracts use observed sessions (and therefore actual holidays).
    # A still-active contract may expire after the latest downloaded session; in
    # that case extend with weekdays so its roll is not incorrectly moved earlier.
    if len(days) and days.max() < expiry:
        extension = pd.bdate_range(days.max() + pd.Timedelta(days=1), expiry)
        days = days.append(extension).unique().sort_values()
    eligible = days[days <= expiry]
    if not len(eligible):
        return expiry
    position = max(0, len(eligible) - 1 - int(rollover_days))
    return pd.Timestamp(eligible[position])


def build_continuous_futures(chunks: list[Path], contracts: pd.DataFrame, rollover_days=3):
    if not chunks:
        raise ValueError("No downloaded contract chunks available")
    raw = pd.concat((pd.read_parquet(path) for path in chunks), ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    raw["contract_expiry"] = pd.to_datetime(raw["contract_expiry"]).dt.normalize()
    source_duplicates = int(raw.duplicated(["timestamp", "security_id"]).sum())
    raw = raw.drop_duplicates(["timestamp", "security_id"], keep="last").sort_values(
        ["timestamp", "contract_expiry"], kind="stable"
    )
    trading_days = raw["timestamp"].dt.tz_localize(None).dt.normalize().unique()
    ordered = contracts.sort_values("expiry_date", kind="stable").reset_index(drop=True).copy()
    ordered["roll_date"] = [trading_day_roll_date(expiry, trading_days, rollover_days)
                            for expiry in ordered["expiry_date"]]
    pieces, rolls = [], []
    for index, contract in ordered.iterrows():
        start = ordered.at[index - 1, "roll_date"] if index > 0 else pd.Timestamp.min
        end = contract["roll_date"]
        contract_rows = raw[raw["security_id"].astype(str).eq(str(contract["security_id"]))]
        local_date = contract_rows["timestamp"].dt.tz_localize(None).dt.normalize()
        selected = contract_rows[(local_date >= start) & (local_date < end if index < len(ordered) - 1 else local_date <= pd.Timestamp.max)]
        pieces.append(selected)
        if index < len(ordered) - 1:
            rolls.append({"roll_date": end, "from_security_id": str(contract["security_id"]),
                          "to_security_id": str(ordered.at[index + 1, "security_id"]),
                          "from_expiry": contract["expiry_date"], "to_expiry": ordered.at[index + 1, "expiry_date"]})
    continuous = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    continuous = continuous.drop_duplicates("timestamp", keep="last").sort_values("timestamp", kind="stable").reset_index(drop=True)
    continuous.attrs["source_duplicate_contract_timestamps"] = source_duplicates
    roll_report = add_roll_prices(pd.DataFrame(rolls), raw)
    return continuous, roll_report


def add_roll_prices(rolls, raw):
    if rolls.empty:
        return rolls.assign(from_last_close=pd.Series(dtype=float), to_first_open=pd.Series(dtype=float),
                            transition_gap_points=pd.Series(dtype=float), transition_gap_percent=pd.Series(dtype=float))
    records = []
    local_date = raw["timestamp"].dt.tz_localize(None).dt.normalize()
    for roll in rolls.itertuples(index=False):
        old = raw[(raw.security_id.astype(str) == roll.from_security_id) & (local_date < roll.roll_date)]
        new = raw[(raw.security_id.astype(str) == roll.to_security_id) & (local_date >= roll.roll_date)]
        old_close = old.iloc[-1].close if len(old) else np.nan
        new_open = new.iloc[0].open if len(new) else np.nan
        gap = new_open - old_close if np.isfinite(old_close) and np.isfinite(new_open) else np.nan
        records.append({**roll._asdict(), "from_last_close": old_close, "to_first_open": new_open,
                        "transition_gap_points": gap,
                        "transition_gap_percent": gap / old_close if np.isfinite(gap) and old_close else np.nan,
                        "transition_valid": bool(len(old) and len(new))})
    return pd.DataFrame(records)


def validate_continuous(data, roll_report, config):
    frame = data.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(config["timezone"])
    duplicate_count = int(timestamps.duplicated().sum())
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    start_h, start_m = map(int, config["session_start"].split(":"))
    end_h, end_m = map(int, config["session_end_exclusive"].split(":"))
    outside = ~(minutes.ge(start_h * 60 + start_m) & minutes.lt(end_h * 60 + end_m))
    observed_days = timestamps.dt.normalize().unique()
    expected_parts = [pd.date_range(day + pd.Timedelta(config["session_start"] + ":00"),
                                    day + pd.Timedelta(config["session_end_exclusive"] + ":00") - pd.Timedelta(minutes=1),
                                    freq=f"{config['interval_minutes']}min")
                      for day in observed_days]
    expected = expected_parts[0].append(expected_parts[1:]) if expected_parts else pd.DatetimeIndex([])
    missing = expected.difference(pd.DatetimeIndex(timestamps))
    missing_frame = pd.DataFrame({"timestamp": missing, "trading_date": missing.date if len(missing) else []})
    previous_close = frame["close"].shift(1)
    same_contract = frame["security_id"].astype(str).eq(frame["security_id"].astype(str).shift())
    gap_pct = (frame["open"] - previous_close).abs() / previous_close.replace(0, np.nan)
    large_gaps = int((same_contract & gap_pct.gt(float(config["large_gap_percent"]))).sum())
    quality = {
        "total_rows": int(len(frame)), "duplicate_timestamps": duplicate_count,
        "source_duplicate_contract_timestamps": int(data.attrs.get("source_duplicate_contract_timestamps", 0)),
        "missing_candles": int(len(missing_frame)), "out_of_session_timestamps": int(outside.sum()),
        "large_same_contract_price_gaps": large_gaps,
        "trading_days": int(timestamps.dt.date.nunique()),
        "earliest_date": timestamps.min().isoformat() if len(frame) else None,
        "latest_date": timestamps.max().isoformat() if len(frame) else None,
        "contract_rolls": int(len(roll_report)),
        "invalid_contract_transitions": int((~roll_report["transition_valid"]).sum()) if len(roll_report) else 0,
        "timezone": str(timestamps.dt.tz),
    }
    return quality, missing_frame


def write_quality_report(path, quality, coverage, config):
    status = "COMPLETE" if coverage["five_year_catalog_complete"] else "PARTIAL_CONTRACT_CATALOG"
    report = f"""# Dhan NIFTY Futures Data Quality Report

## Status

`{status}`

The continuous series must not be promoted as the official ML source unless contract-catalog coverage is complete and all configured quality gates are reviewed.

Dhan's documented intraday endpoint covers active instruments. The latest Security Master contains active contracts only; therefore a five-year expired-futures series cannot be reconstructed from that file alone. Archived masters are accepted for discovery, but availability of expired security IDs must be confirmed with Dhan before promotion.

## Dataset

| Metric | Value |
|---|---:|
| Total rows | {quality['total_rows']} |
| Trading days | {quality['trading_days']} |
| Earliest timestamp | {quality['earliest_date']} |
| Latest timestamp | {quality['latest_date']} |
| Missing candles | {quality['missing_candles']} |
| Duplicate timestamps | {quality['duplicate_timestamps']} |
| Source duplicate contract timestamps removed | {quality['source_duplicate_contract_timestamps']} |
| Out-of-session timestamps | {quality['out_of_session_timestamps']} |
| Large same-contract gaps | {quality['large_same_contract_price_gaps']} |
| Contract rolls | {quality['contract_rolls']} |
| Invalid transitions | {quality['invalid_contract_transitions']} |
| Timezone | {quality['timezone']} |

## Contract catalog coverage

| Metric | Value |
|---|---:|
| Contracts discovered | {coverage['contracts_discovered']} |
| Earliest expiry | {coverage['earliest_expiry']} |
| Latest expiry | {coverage['latest_expiry']} |
| Desired history start | {coverage['desired_start']} |
| Five-year catalog complete | {coverage['five_year_catalog_complete']} |

Rollover occurs at session start {config['rollover_trading_days']} trading days before expiry. Completed contracts use observed sessions; future roll dates use weekdays until actual sessions become available. Prices are intentionally unadjusted, and transition gaps are reported rather than hidden.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(report, encoding="utf-8")
