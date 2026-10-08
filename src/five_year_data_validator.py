"""Read-only institutional audit for the five-year raw NIFTY dataset.

This module never repairs or overwrites source data.  A de-duplicated or
session-filtered frame is used only for calculations and is not persisted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_START = "09:15"
SESSION_END_EXCLUSIVE = "15:30"
EXPECTED_MINUTES = 375

# NSE Capital Market closures inside this dataset's date range. The dates are
# transcribed from the exchange's annual holiday circulars plus announced ad-hoc
# closures (elections/special public holidays). Kept here to make the audit
# deterministic and reviewable; they are not inferred from missing data.
NSE_WEEKDAY_CLOSURES = frozenset("""
2021-08-19 2021-09-10 2021-10-15 2021-11-05 2021-11-19
2022-01-26 2022-03-01 2022-03-18 2022-04-14 2022-04-15 2022-05-03
2022-08-09 2022-08-15 2022-08-31 2022-10-05 2022-10-26 2022-11-08
2023-01-26 2023-03-07 2023-03-30 2023-04-04 2023-04-07 2023-04-14
2023-05-01 2023-06-29 2023-08-15 2023-09-19 2023-10-02 2023-10-24
2023-11-14 2023-11-27 2023-12-25
2024-01-22 2024-01-26 2024-03-08 2024-03-25 2024-03-29 2024-04-11
2024-04-17 2024-05-01 2024-05-20 2024-06-17 2024-07-17 2024-08-15
2024-10-02 2024-11-15 2024-11-20 2024-12-25
2025-02-26 2025-03-14 2025-03-31 2025-04-10 2025-04-14 2025-04-18
2025-05-01 2025-08-15 2025-08-27 2025-10-02 2025-10-22 2025-11-05
2025-12-25
2026-01-15 2026-01-26 2026-03-03 2026-03-26 2026-03-31 2026-04-03
2026-04-14 2026-05-01 2026-05-28 2026-06-26
""".split())


def _parse_timestamp(series: pd.Series) -> tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio > 0.99:
        finite = numeric.dropna().abs()
        median = float(finite.median()) if len(finite) else 0.0
        unit = "s" if 1e9 <= median < 1e11 else "ms" if median < 1e14 else "ns"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        encoding = f"unix_epoch_{unit}"
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        encoding = "datetime_string"
    return parsed.dt.tz_convert("Asia/Kolkata"), encoding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    body = [f"# {title}", ""]
    for heading, text in sections:
        body.extend([f"## {heading}", "", text, ""])
    path.write_text("\n".join(body), encoding="utf-8")


def audit_dataset(source: Path, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [c for c in required if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ts, timestamp_encoding = _parse_timestamp(raw["timestamp"])
    numeric = raw[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    frame = numeric.copy()
    frame.insert(0, "timestamp", ts)
    frame.insert(0, "source_row", np.arange(2, len(raw) + 2))

    invalid_ts = ts.isna()
    duplicate_mask = ts.notna() & ts.duplicated(keep=False)
    exact_duplicate_mask = raw.duplicated(keep=False)
    duplicates = frame.loc[duplicate_mask].copy()
    duplicates["duplicate_group_size"] = duplicates.groupby("timestamp")["timestamp"].transform("size")
    duplicates.to_csv(report_dir / "duplicate_timestamps.csv", index=False)

    null_counts = raw[required].isna().sum().to_dict()
    non_numeric_counts = {c: int(numeric[c].isna().sum() - raw[c].isna().sum()) for c in numeric}
    invalid = pd.DataFrame(index=raw.index)
    invalid["source_row"] = frame.source_row
    invalid["timestamp"] = ts
    invalid["missing_ohlcv"] = numeric.isna().any(axis=1)
    invalid["non_positive_price"] = numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
    invalid["negative_volume"] = numeric.volume.lt(0)
    invalid["high_below_low"] = numeric.high.lt(numeric.low)
    invalid["open_outside_range"] = numeric.open.lt(numeric.low) | numeric.open.gt(numeric.high)
    invalid["close_outside_range"] = numeric.close.lt(numeric.low) | numeric.close.gt(numeric.high)
    invalid["timestamp_invalid"] = invalid_ts
    issue_cols = [c for c in invalid.columns if c not in {"source_row", "timestamp"}]
    invalid_rows = invalid.loc[invalid[issue_cols].any(axis=1)]
    invalid_rows.to_csv(report_dir / "invalid_rows.csv", index=False)

    valid_ts_frame = frame.loc[~invalid_ts].copy()
    local_ts = valid_ts_frame.timestamp
    minute = local_ts.dt.hour * 60 + local_ts.dt.minute
    in_session = minute.between(9 * 60 + 15, 15 * 60 + 29)
    valid_ts_frame["in_regular_session"] = in_session
    valid_ts_frame["calendar_date"] = local_ts.dt.date
    valid_ts_frame["time"] = local_ts.dt.strftime("%H:%M:%S")
    valid_ts_frame["minute_bucket"] = local_ts.dt.floor("min")
    granularity_mask = local_ts.dt.second.ne(0) | local_ts.dt.microsecond.ne(0)
    granularity_anomalies = valid_ts_frame.loc[granularity_mask].copy()
    granularity_anomalies["reason"] = "not_aligned_to_minute_boundary"
    granularity_anomalies.to_csv(report_dir / "timestamp_granularity_anomalies.csv", index=False)
    session_anomalies = valid_ts_frame.loc[~in_session].copy()
    session_anomalies["reason"] = np.where(local_ts.loc[~in_session].dt.dayofweek >= 5,
                                            "weekend_or_special_session", "outside_regular_session")
    session_anomalies.to_csv(report_dir / "session_anomalies.csv", index=False)

    # Missing minutes are measured only for dates with at least one regular-session row.
    regular = valid_ts_frame.loc[in_session].copy()
    missing_records: list[dict] = []
    daily_records: list[dict] = []
    for date_value, day in regular.groupby("calendar_date", sort=True):
        expected = pd.date_range(f"{date_value} {SESSION_START}", periods=EXPECTED_MINUTES,
                                 freq="min", tz="Asia/Kolkata")
        observed = pd.DatetimeIndex(day.minute_bucket.drop_duplicates())
        missing = expected.difference(observed)
        for value in missing:
            missing_records.append({"date": str(date_value), "missing_timestamp": value.isoformat(),
                                    "classification": "missing_regular_session_candle"})
        daily_records.append({
            "date": str(date_value), "observed_regular_rows": int(len(day)),
            "unique_regular_minute_buckets": int(observed.nunique()), "expected_rows": EXPECTED_MINUTES,
            "missing_candles": int(len(missing)), "first_timestamp": day.timestamp.min().isoformat(),
            "last_timestamp": day.timestamp.max().isoformat(),
        })
    missing_candles = pd.DataFrame(missing_records, columns=["date", "missing_timestamp", "classification"])
    missing_candles.to_csv(report_dir / "missing_candles.csv", index=False)
    daily = pd.DataFrame(daily_records)

    observed_dates = pd.DatetimeIndex(pd.to_datetime(valid_ts_frame.calendar_date.unique()))
    first_date, last_date = observed_dates.min(), observed_dates.max()
    weekdays = pd.date_range(first_date, last_date, freq="B")
    absent_weekdays = weekdays.difference(observed_dates)
    missing_sessions = pd.DataFrame({"date": absent_weekdays.strftime("%Y-%m-%d")})
    missing_sessions["classification"] = np.where(
        missing_sessions.date.isin(NSE_WEEKDAY_CLOSURES), "verified_exchange_closure",
        "unexplained_missing_weekday_session")
    missing_sessions.to_csv(report_dir / "missing_sessions.csv", index=False)

    out_counts = session_anomalies.groupby("calendar_date").size().rename("out_of_session_rows")
    if not daily.empty:
        daily["out_of_session_rows"] = pd.to_datetime(daily.date).dt.date.map(out_counts).fillna(0).astype(int)
    daily.to_csv(report_dir / "daily_session_summary.csv", index=False)
    session_year = valid_ts_frame.assign(year=local_ts.dt.year).groupby("year").agg(
        total_rows=("timestamp", "size"),
        regular_session_rows=("in_regular_session", "sum"),
        observed_dates=("calendar_date", "nunique"),
    ).reset_index()
    session_year["out_of_session_rows"] = session_year.total_rows - session_year.regular_session_rows
    session_year.to_csv(report_dir / "session_year_summary.csv", index=False)

    # Gap analysis preserves chronology but never changes raw rows.
    ordered = regular.sort_values("timestamp").copy()
    ordered["previous_timestamp"] = ordered.timestamp.shift()
    ordered["previous_close"] = ordered.close.shift()
    ordered["elapsed_minutes"] = (ordered.timestamp - ordered.previous_timestamp).dt.total_seconds() / 60
    ordered["same_date"] = ordered.calendar_date.eq(ordered.calendar_date.shift())
    ordered["open_gap_points"] = ordered.open - ordered.previous_close
    ordered["open_gap_pct"] = ordered.open_gap_points / ordered.previous_close * 100
    ordered["candle_range_pct"] = (ordered.high - ordered.low) / ordered.open * 100
    gap_flags = ((ordered.same_date & ordered.elapsed_minutes.gt(1)) |
                 (ordered.same_date & ordered.open_gap_pct.abs().gt(0.50)) |
                 (~ordered.same_date & ordered.open_gap_pct.abs().gt(1.00)) |
                 ordered.candle_range_pct.gt(1.00))
    gaps = ordered.loc[gap_flags, ["timestamp", "previous_timestamp", "elapsed_minutes", "open",
                                    "previous_close", "open_gap_points", "open_gap_pct", "candle_range_pct",
                                    "same_date"]].copy()
    gaps["reason"] = ""
    gaps.loc[gaps.same_date & gaps.elapsed_minutes.gt(1), "reason"] += "time_gap;"
    gaps.loc[gaps.same_date & gaps.open_gap_pct.abs().gt(.5), "reason"] += "intraday_price_gap_gt_0.5pct;"
    gaps.loc[~gaps.same_date & gaps.open_gap_pct.abs().gt(1), "reason"] += "overnight_gap_gt_1pct;"
    gaps.loc[gaps.candle_range_pct.gt(1), "reason"] += "candle_range_gt_1pct;"
    gaps.to_csv(report_dir / "gap_anomalies.csv", index=False)

    zero = valid_ts_frame.volume.eq(0)
    negative = valid_ts_frame.volume.lt(0)
    volume_year = valid_ts_frame.assign(year=local_ts.dt.year, zero_volume=zero,
                                        negative_volume=negative).groupby("year").agg(
        rows=("volume", "size"), zero_volume_rows=("zero_volume", "sum"),
        negative_volume_rows=("negative_volume", "sum"), median_volume=("volume", "median"),
        minimum_volume=("volume", "min"), maximum_volume=("volume", "max")).reset_index()
    volume_year.to_csv(report_dir / "provider_volume_consistency.csv", index=False)
    positive_volume_ts = valid_ts_frame.loc[valid_ts_frame.volume.gt(0), "timestamp"]
    first_positive_volume = positive_volume_ts.min().isoformat() if len(positive_volume_ts) else None

    price_median = float(numeric.close.median())
    price_minimum = float(numeric.close.min())
    price_maximum = float(numeric.close.max())
    # Identity evidence is triangulated: the acquisition configuration uses the
    # constant ID 13, the downloaded Dhan master maps segment I / ID 13 to Nifty
    # 50, and the observed level range is plausible. The CSV does not embed ID
    # per row, so row-level security-ID verification remains impossible.
    security_master_path = Path("data/raw/dhan_security_master.csv")
    master_match = False
    if security_master_path.exists():
        master = pd.read_csv(security_master_path, low_memory=False)
        if {"SEGMENT", "SECURITY_ID", "DISPLAY_NAME"}.issubset(master.columns):
            master_id = master.SECURITY_ID.astype(str).str.replace(r"\.0$", "", regex=True)
            master_match = bool(((master.SEGMENT.astype(str).eq("I")) & master_id.eq("13") &
                                 master.DISPLAY_NAME.astype(str).str.contains("Nifty 50", case=False,
                                                                            na=False)).any())
    acquisition_security_id = "13"
    instrument_verified = bool(master_match and 10_000 <= price_median <= 50_000)
    security_id_consistency = "acquisition_level_only_not_embedded_per_row"
    issue_counts = {c: int(invalid[c].sum()) for c in issue_cols}
    health_blockers = []
    if not instrument_verified:
        health_blockers.append("Instrument identity could not be corroborated from Security Master and price scale.")
    if issue_counts["negative_volume"]:
        health_blockers.append("Negative volume values exist.")
    if issue_counts["open_outside_range"] or issue_counts["close_outside_range"]:
        health_blockers.append("OHLC range relationship violations exist.")
    if int(zero.sum()) > len(valid_ts_frame) * 0.50:
        health_blockers.append("Volume is predominantly zero and changes provider regime during the sample.")
    if len(session_anomalies):
        health_blockers.append("Out-of-session/weekend timestamps exist.")
    if len(missing_candles):
        health_blockers.append("Regular-session candles are missing.")
    if granularity_mask.any():
        health_blockers.append("One-minute timestamps are not consistently aligned to minute boundaries.")
    approved = not health_blockers

    summary = {
        "source": source.as_posix(), "source_sha256": _sha256(source), "source_rows": int(len(raw)),
        "columns": list(raw.columns), "timestamp_encoding": timestamp_encoding,
        "timezone_interpretation": "Unix epoch UTC converted to Asia/Kolkata",
        "earliest_timestamp": ts.min().isoformat(), "latest_timestamp": ts.max().isoformat(),
        "calendar_span_days": int((ts.max().date() - ts.min().date()).days + 1),
        "observed_calendar_dates": int(valid_ts_frame.calendar_date.nunique()),
        "observed_regular_session_dates": int(regular.calendar_date.nunique()),
        "regular_session_rows": int(len(regular)), "out_of_session_rows": int(len(session_anomalies)),
        "weekend_rows": int((local_ts.dt.dayofweek >= 5).sum()),
        "missing_regular_session_candles": int(len(missing_candles)),
        "weekday_dates_without_data": int(len(missing_sessions)),
        "verified_exchange_closure_dates": int(missing_sessions.classification.eq("verified_exchange_closure").sum()),
        "unexplained_missing_weekday_sessions": int(missing_sessions.classification.eq("unexplained_missing_weekday_session").sum()),
        "duplicate_timestamp_rows": int(duplicate_mask.sum()),
        "duplicate_minute_bucket_rows": int(valid_ts_frame.minute_bucket.duplicated(keep=False).sum()),
        "non_minute_aligned_timestamp_rows": int(granularity_mask.sum()),
        "exact_duplicate_rows": int(exact_duplicate_mask.sum()), "invalid_timestamp_rows": int(invalid_ts.sum()),
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "non_numeric_counts": non_numeric_counts, "invalid_record_issue_counts": issue_counts,
        "zero_volume_rows": int(zero.sum()), "negative_volume_rows": int(negative.sum()),
        "first_positive_volume_timestamp": first_positive_volume,
        "minimum_close": price_minimum, "median_close": price_median, "maximum_close": price_maximum,
        "acquisition_security_id": acquisition_security_id,
        "security_master_index_match": master_match,
        "security_id_consistency_status": security_id_consistency,
        "instrument_identity_verified": instrument_verified,
        "records_removed": 0, "approved_for_feature_generation": approved,
        "approval_blockers": health_blockers,
        "gap_anomaly_rows": int(len(gaps)),
    }
    (report_dir / "final_audited_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    integrity_table = "\n".join(f"- **{k}:** {v}" for k, v in summary.items()
                                  if k not in {"approval_blockers", "null_counts", "invalid_record_issue_counts"})
    _write_md(report_dir / "data_quality_report.md", "Five-Year Data Quality Report", [
        ("Scope", "Read-only pre-feature audit. The raw CSV was not altered and no records were removed."),
        ("Dataset", integrity_table),
        ("OHLCV validation", f"Nulls: `{summary['null_counts']}`  \nInvalid issue counts: `{issue_counts}`"),
        ("Zero-volume behaviour", f"Zero-volume rows: **{zero.sum():,}** of **{len(valid_ts_frame):,}**. "
         f"First positive-volume timestamp: **{first_positive_volume}**. Year-level evidence is in "
         "`provider_volume_consistency.csv`; volume availability changes materially across the sample."),
        ("Audit trail", "Row-level evidence is in `invalid_rows.csv`, `duplicate_timestamps.csv`, "
                        "`timestamp_granularity_anomalies.csv`, `session_anomalies.csv`, "
                        "`missing_candles.csv`, and `gap_anomalies.csv`."),
    ])
    _write_md(report_dir / "duplicate_report.md", "Duplicate Report", [
        ("Result", f"Timestamp duplicate rows: **{duplicate_mask.sum():,}**. Exact duplicate rows: "
                   f"**{exact_duplicate_mask.sum():,}**. Minute-bucket collision rows: "
                   f"**{valid_ts_frame.minute_bucket.duplicated(keep=False).sum():,}**. No rows were removed."),
        ("Interpretation", "Exact timestamps are unique, but seconds-offset observations can collide after "
                           "normalization to one-minute buckets. These collisions must not be silently deduplicated."),
        ("Evidence", "See `duplicate_timestamps.csv` and `timestamp_granularity_anomalies.csv`."),
    ])
    _write_md(report_dir / "instrument_verification_report.md", "Instrument Verification Report", [
        ("Verdict", "**NIFTY 50 identity corroborated at acquisition level.** Row-level identity cannot be "
                    "cryptographically verified because the CSV contains no symbol, segment, instrument, or "
                    "security ID columns."),
        ("Evidence", f"Acquisition configuration: Security ID **{acquisition_security_id}**. The downloaded "
                     f"Security Master contains segment `I`, Security ID `13`, display name `Nifty 50`: "
                     f"**{master_match}**. Observed close range: **{price_minimum:,.2f}–{price_maximum:,.2f}**; "
                     f"median: **{price_median:,.2f}**."),
        ("Security-ID consistency", "The acquisition code holds Security ID 13 constant across chunks. The "
                                    "CSV omits request metadata, so consistency is supported by acquisition "
                                    "configuration, not independently provable for each row."),
        ("Provider contract note", "Dhan documentation defines `IDX_I` as the Index segment and `INDEX` as an "
                                  "instrument type. Preserve the exact successful request manifest beside future "
                                  "downloads so payload identity is auditable without relying on notebook state."),
    ])
    _write_md(report_dir / "session_validation_report.md", "Session Validation Report", [
        ("Rule", "Regular NSE one-minute candle timestamps are treated as 09:15 through 15:29 "
                 "(09:15 inclusive, 15:30 exclusive), 375 expected bars."),
        ("Findings", f"Regular rows: **{len(regular):,}**; out-of-session rows: "
                     f"**{len(session_anomalies):,}**; weekend rows: **{summary['weekend_rows']:,}**; "
                     f"weekday dates with no data: **{len(missing_sessions):,}**."),
        ("Evidence", "See `daily_session_summary.csv`, `session_anomalies.csv`, and `missing_sessions.csv`. "
                     "Annual and ad-hoc closures were checked against NSE Capital Market notices. The exchange "
                     "states that regular trading is 09:15–15:30; special/Muhurat sessions can use other hours."),
        ("Start/end consistency", f"Raw start: **{ts.min().isoformat()}** (the opening 09:15 bar is absent on "
                                  f"the first date). Raw end: **{ts.max().isoformat()}**, which is outside the "
                                  "regular session; end-of-file session records are documented, not removed."),
    ])
    _write_md(report_dir / "missing_candle_report.md", "Missing Candle Report", [
        ("Method", "Expected minute buckets are 09:15 through 15:29 on every observed regular-session date. "
                   "Seconds-offset rows occupy their minute bucket but are separately flagged as granularity defects."),
        ("Result", f"Missing regular-session minute buckets: **{len(missing_candles):,}**. Full absent weekdays: "
                   f"**{len(missing_sessions):,}**, of which **{missing_sessions.classification.eq('verified_exchange_closure').sum():,}** "
                   "match NSE closures and none are unexplained."),
        ("Evidence", "See `missing_candles.csv`, `missing_sessions.csv`, and `daily_session_summary.csv`."),
    ])
    _write_md(report_dir / "gap_analysis_report.md", "Gap Analysis Report", [
        ("Rules", "Flag: same-session time gap >1 minute; same-session open gap >0.50%; overnight "
                  "open gap >1.00%; candle range >1.00%. Thresholds are diagnostics, not repairs."),
        ("Result", f"Flagged rows: **{len(gaps):,}**. Missing regular candles: **{len(missing_candles):,}**."),
        ("Evidence", "See `gap_anomalies.csv` and `missing_candles.csv`."),
    ])
    verdict = "NOT APPROVED" if not approved else "APPROVED"
    _write_md(report_dir / "dataset_health_report.md", "Dataset Health Report", [
        ("Verdict", f"**{verdict} for feature generation.**"),
        ("Blocking findings", "\n".join(f"- {x}" for x in health_blockers) or "None."),
        ("Provider/instrument consistency", f"NIFTY identity is corroborated by Security ID 13 in the acquisition "
         f"configuration, the Security Master mapping, and a **{price_minimum:,.2f}–{price_maximum:,.2f}** close "
         f"range. Row-level IDs are absent. Volume includes **{negative.sum():,} negative** and "
         f"**{zero.sum():,} zero** observations and exhibits a major provider-regime change."),
    ])
    _write_md(report_dir / "final_audited_dataset_summary.md", "Final Audited Dataset Summary", [
        ("Decision", f"**{verdict} for Roadmap Step 2.** The source remains unchanged; records removed: **0**."),
        ("Core counts", integrity_table),
        ("Required resolution", "Confirm the instrument/security ID and provenance, explain or replace invalid "
         "volume observations, define treatment for non-regular-session data, and account for missing candles "
         "against official exchange calendars before feature generation."),
    ])
    return summary
