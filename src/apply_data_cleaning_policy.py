"""Apply the approved Step 14B policy to the official raw index dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning_policy import MUHURAT_DATES, SPECIAL_DATES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_policy(source: Path, cleaned_dir: Path, report_dir: Path) -> dict:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(source)
    raw = pd.read_csv(source)
    ts = pd.to_datetime(raw.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    frame = raw.copy()
    frame.insert(0, "source_row", np.arange(2, len(frame) + 2))
    frame["timestamp_ist"] = ts
    frame["date"] = ts.dt.strftime("%Y-%m-%d")
    frame["minute_bucket"] = ts.dt.floor("min")
    frame["second"] = ts.dt.second
    frame["day_of_week"] = ts.dt.dayofweek
    minute = ts.dt.hour * 60 + ts.dt.minute
    frame["regular_time"] = minute.between(555, 929)
    frame["quarantine_reasons"] = ""

    def quarantine(mask: pd.Series, reason: str) -> None:
        separator = np.where(frame.loc[mask, "quarantine_reasons"].eq(""), "", ";")
        frame.loc[mask, "quarantine_reasons"] += separator + reason

    muhurat = frame.date.isin(MUHURAT_DATES)
    special = frame.date.isin(SPECIAL_DATES)
    weekend_other = frame.day_of_week.ge(5) & ~muhurat & ~special
    ordinary_out = ~frame.regular_time & frame.day_of_week.lt(5) & ~muhurat
    quarantine(muhurat, "muhurat_session_segregated")
    quarantine(special, "known_special_session_segregated")
    quarantine(weekend_other, "weekend_provider_anomaly")
    quarantine(ordinary_out, "ordinary_out_of_session")

    eligible = frame.quarantine_reasons.eq("")
    collision = eligible & frame.minute_bucket.duplicated(keep=False)
    for _, group in frame[collision].groupby("minute_bucket"):
        exact = group[group.second.eq(0)]
        if len(exact) == 1:
            quarantine(frame.index.isin(group.index.difference(exact.index)),
                       "offset_duplicate_of_exact_minute")
        else:
            quarantine(frame.index.isin(group.index), "ambiguous_minute_bucket_collision")

    # Re-evaluate ordinary sessions after deterministic timestamp collision handling.
    eligible = frame.quarantine_reasons.eq("")
    incomplete_dates: list[str] = []
    missing_records: list[dict] = []
    for date, day in frame[eligible].groupby("date"):
        expected = pd.date_range(f"{date} 09:15", periods=375, freq="min", tz="Asia/Kolkata")
        missing = expected.difference(pd.DatetimeIndex(day.minute_bucket))
        if len(missing):
            incomplete_dates.append(date)
            missing_records.extend({"date": date, "missing_timestamp": value.isoformat(),
                                    "resolution": "unrecovered_source_gap",
                                    "policy_action": "quarantine_complete_session"} for value in missing)
    quarantine(frame.date.isin(incomplete_dates) & frame.quarantine_reasons.eq(""),
               "incomplete_session_unrecovered")

    # Approved precision normalization is applied only to rows still eligible.
    eligible = frame.quarantine_reasons.eq("")
    rounded = frame.loc[eligible, ["open", "high", "low", "close"]].round(2)
    modified_mask = rounded.ne(frame.loc[eligible, ["open", "high", "low", "close"]]).any(axis=1)
    modified_index = rounded.index[modified_mask]
    modifications = frame.loc[modified_index, ["source_row", "timestamp_ist", "open", "high", "low", "close"]].copy()
    modifications = modifications.rename(columns={c: f"raw_{c}" for c in ["open", "high", "low", "close"]})
    for column in ["open", "high", "low", "close"]:
        modifications[f"cleaned_{column}"] = rounded.loc[modified_index, column]
    modifications["action"] = "round_ohlc_to_two_decimals"
    modifications.to_csv(report_dir / "modified_rows_report.csv", index=False)
    frame.loc[eligible, ["open", "high", "low", "close"]] = rounded

    eligible = frame.quarantine_reasons.eq("")
    bad_ohlc = eligible & ((frame.high < frame.low) | (frame.open < frame.low) |
                           (frame.open > frame.high) | (frame.close < frame.low) |
                           (frame.close > frame.high))
    bad_dates = set(frame.loc[bad_ohlc, "date"])
    quarantine(frame.date.isin(bad_dates) & frame.quarantine_reasons.eq(""),
               "substantive_ohlc_session_unresolved")

    # Without an independent minute source, >1% range flags remain unconfirmed
    # and are quarantined exactly as the approved conservative policy requires.
    eligible = frame.quarantine_reasons.eq("")
    range_pct = (frame.high - frame.low) / frame.open * 100
    unconfirmed_gap = eligible & range_pct.gt(1.0)
    quarantine(unconfirmed_gap, "unconfirmed_abnormal_range_gap")

    quarantined = frame[frame.quarantine_reasons.ne("")].copy()
    quarantine_columns = ["source_row", "timestamp_ist", "open", "high", "low", "close", "volume",
                          "quarantine_reasons"]
    quarantined[quarantine_columns].to_csv(report_dir / "quarantined_rows_report.csv", index=False)
    pd.DataFrame(columns=quarantine_columns + ["removal_reason"]).to_csv(
        report_dir / "removed_rows_report.csv", index=False)

    accepted = frame[frame.quarantine_reasons.eq("")].copy().sort_values("timestamp_ist", kind="stable")
    cleaned = accepted[["source_row", "timestamp_ist", "open", "high", "low", "close", "volume"]].rename(
        columns={"timestamp_ist": "timestamp", "volume": "volume_raw"})
    cleaned["volume"] = pd.Series(pd.array([pd.NA] * len(cleaned), dtype="Float64"), index=cleaned.index)
    cleaned = cleaned.reset_index(drop=True)

    duplicate_timestamps = int(cleaned.timestamp.duplicated().sum())
    sorted_ok = bool(cleaned.timestamp.is_monotonic_increasing)
    local_minute = cleaned.timestamp.dt.hour * 60 + cleaned.timestamp.dt.minute
    session_invalid = int((~local_minute.between(555, 929)).sum())
    granularity_invalid = int(cleaned.timestamp.dt.second.ne(0).sum())
    ohlc_invalid = int(((cleaned.high < cleaned.low) | (cleaned.open < cleaned.low) |
                        (cleaned.open > cleaned.high) | (cleaned.close < cleaned.low) |
                        (cleaned.close > cleaned.high)).sum())
    if any([duplicate_timestamps, session_invalid, granularity_invalid, ohlc_invalid]) or not sorted_ok:
        raise RuntimeError("Cleaned dataset failed mandatory validation")

    csv_path = cleaned_dir / "nifty_5year_cleaned.csv"
    parquet_path = cleaned_dir / "nifty_5year_cleaned.parquet"
    cleaned.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    cleaned.to_parquet(parquet_path, index=False)
    source_hash_after = sha256(source)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Raw source changed during cleaning")

    # Missing buckets remaining in the accepted set are documented, including
    # gaps introduced by conservative quarantine of unconfirmed range events.
    remaining: list[dict] = []
    for date, day in cleaned.groupby(cleaned.timestamp.dt.strftime("%Y-%m-%d")):
        expected = pd.date_range(f"{date} 09:15", periods=375, freq="min", tz="Asia/Kolkata")
        for value in expected.difference(pd.DatetimeIndex(day.timestamp)):
            remaining.append({"date": date, "timestamp": value.isoformat(),
                              "anomaly": "missing_after_policy_quarantine"})
    pd.DataFrame(remaining, columns=["date", "timestamp", "anomaly"]).to_csv(
        report_dir / "remaining_anomalies.csv", index=False)

    hashes = {
        "raw_source": {"path": source.as_posix(), "sha256": source_hash_after},
        "cleaned_csv": {"path": csv_path.as_posix(), "sha256": sha256(csv_path)},
        "cleaned_parquet": {"path": parquet_path.as_posix(), "sha256": sha256(parquet_path)},
    }
    (report_dir / "dataset_hash_report.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    stats = {
        "policy_version": "1.0-approved", "raw_rows": int(len(raw)), "cleaned_rows": int(len(cleaned)),
        "quarantined_unique_rows": int(len(quarantined)), "removed_rows": 0,
        "modified_ohlc_rows": int(len(modifications)), "incomplete_sessions_quarantined": len(incomplete_dates),
        "unconfirmed_gap_rows_quarantined": int(unconfirmed_gap.sum()),
        "substantive_ohlc_sessions_quarantined": len(bad_dates),
        "duplicate_timestamps": duplicate_timestamps, "invalid_ohlc_rows": ohlc_invalid,
        "invalid_session_rows": session_invalid, "non_minute_aligned_rows": granularity_invalid,
        "sorted": sorted_ok, "timezone": str(cleaned.timestamp.dt.tz),
        "earliest": cleaned.timestamp.min().isoformat(), "latest": cleaned.timestamp.max().isoformat(),
        "remaining_missing_minute_buckets": len(remaining), "analytical_volume_null_rows": int(cleaned.volume.isna().sum()),
    }
    (report_dir / "final_dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    reason_counts = quarantined.quarantine_reasons.str.get_dummies(sep=";").sum().sort_values(ascending=False)
    reason_text = "\n".join(f"- {reason}: **{int(count):,}**" for reason, count in reason_counts.items())
    (report_dir / "cleaning_report.md").write_text(f"""# Step 14C Cleaning Report

Policy `1.0-approved` was applied reproducibly to the sole official raw dataset. The raw source remained unchanged. No raw record was physically deleted; excluded records are fully represented in `quarantined_rows_report.csv`.

## Results

- Raw rows: **{len(raw):,}**
- Cleaned rows: **{len(cleaned):,}**
- Unique quarantined raw rows: **{len(quarantined):,}**
- Permanently removed rows: **0**
- OHLC rows logged as modified: **{len(modifications):,}**
- Analytical volume: null by approved policy; original retained as `volume_raw`

## Quarantine reason counts

{reason_text}

## Validation

- Duplicate timestamps: **{duplicate_timestamps}**
- Invalid OHLC: **{ohlc_invalid}**
- Invalid session timestamps: **{session_invalid}**
- Seconds-offset timestamps: **{granularity_invalid}**
- Sorted: **{sorted_ok}**
- Timezone: **{cleaned.timestamp.dt.tz}**

No features, labels, training, or backtesting were performed.
""", encoding="utf-8")
    (report_dir / "remaining_anomalies_report.md").write_text(f"""# Remaining Anomalies Report

The cleaned dataset has no duplicate timestamps, invalid OHLC, invalid regular-session timestamps, or seconds-offset timestamps. It contains **{len(remaining):,}** absent minute buckets because approved quarantine intentionally does not synthesize candles. Analytical `volume` is null for all rows by policy; `volume_raw` is retained for provenance only.
""", encoding="utf-8")
    (report_dir / "dataset_hash_report.md").write_text(
        "# Dataset Hash Report\n\n" + "\n".join(f"- **{key}:** `{value['sha256']}` — `{value['path']}`"
                                                    for key, value in hashes.items()) + "\n", encoding="utf-8")
    return stats

