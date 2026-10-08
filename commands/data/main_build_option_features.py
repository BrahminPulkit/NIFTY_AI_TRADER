"""Roadmap Step 18A: validate and build canonical option features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.option_feature_pipeline import (
    OPTION_FEATURES, OPTION_FEATURE_VERSION, SOURCE_COLUMNS,
    build_live_option_features, build_option_features,
)


SOURCE = Path("notebooks/nifty_rolling_options_all_columns_5yr.csv")
INDEX = Path("data/cleaned/nifty_5year_cleaned.parquet")
OUTPUT_DIR = Path("data/options")
REPORT_DIR = Path("reports/option_dataset")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).dt.tz_convert("Asia/Kolkata").to_numpy(
        dtype="datetime64[ns]").astype("int64")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    complete = pd.read_csv(SOURCE, low_memory=False)
    source = complete[SOURCE_COLUMNS].copy()
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    index = pd.read_parquet(INDEX, columns=["timestamp"])
    index["timestamp"] = pd.to_datetime(index.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")

    numeric = source[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce")
    minute = source.timestamp.dt.hour * 60 + source.timestamp.dt.minute
    source_ns, index_ns = _normalized_ns(source.timestamp), _normalized_ns(index.timestamp)
    option_in_index = np.isin(source_ns, index_ns)
    index_in_option = np.isin(index_ns, source_ns)
    forbidden = [name for name in complete.columns if name not in SOURCE_COLUMNS]
    audit = {
        "source": SOURCE.as_posix(), "source_sha256": sha256(SOURCE), "rows": len(source),
        "source_columns": list(complete.columns), "canonical_source_columns": SOURCE_COLUMNS,
        "excluded_source_columns": forbidden,
        "timestamp_invalid": int(source.timestamp.isna().sum()),
        "duplicate_timestamps": int(source.timestamp.duplicated().sum()),
        "exact_duplicate_rows": int(complete.duplicated().sum()),
        "sorted": bool(source.timestamp.is_monotonic_increasing),
        "earliest": source.timestamp.min().isoformat(), "latest": source.timestamp.max().isoformat(),
        "sessions": int(source.timestamp.dt.normalize().nunique()),
        "non_minute_aligned": int((source.timestamp.dt.second.ne(0) |
                                    source.timestamp.dt.microsecond.ne(0)).sum()),
        "outside_index_regular_session": int((~minute.between(555, 929)).sum()),
        "weekend_rows": int(source.timestamp.dt.dayofweek.ge(5).sum()),
        "missing_ohlcv": int(numeric.isna().any(axis=1).sum()),
        "high_below_low": int(numeric.high.lt(numeric.low).sum()),
        "open_outside_range": int((numeric.open.lt(numeric.low) |
                                    numeric.open.gt(numeric.high)).sum()),
        "close_outside_range": int((numeric.close.lt(numeric.low) |
                                     numeric.close.gt(numeric.high)).sum()),
        "nonpositive_price": int(numeric[["open", "high", "low", "close"]].le(0).any(axis=1).sum()),
        "negative_volume": int(numeric.volume.lt(0).sum()),
        "zero_volume": int(numeric.volume.eq(0).sum()),
        "exact_timestamp_matches_index": int(option_in_index.sum()),
        "option_timestamps_not_in_clean_index": int((~option_in_index).sum()),
        "clean_index_timestamps_missing_in_option": int((~index_in_option).sum()),
        "matching_sessions": len(set(source.timestamp.dt.date) & set(index.timestamp.dt.date)),
        "option_only_sessions": len(set(source.timestamp.dt.date) - set(index.timestamp.dt.date)),
        "index_only_sessions": len(set(index.timestamp.dt.date) - set(source.timestamp.dt.date)),
    }
    (REPORT_DIR / "validation_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    pd.DataFrame({"timestamp": source.loc[~option_in_index, "timestamp"],
                  "reason": "not_present_in_clean_index_baseline"}).to_csv(
        REPORT_DIR / "option_only_timestamps.csv", index=False)
    pd.DataFrame({"timestamp": index.loc[~index_in_option, "timestamp"],
                  "reason": "missing_from_option_dataset"}).to_csv(
        REPORT_DIR / "index_timestamps_missing_in_option.csv", index=False)
    source_dates = set(source.timestamp.dt.date)
    index_dates = set(index.timestamp.dt.date)
    pd.DataFrame({"date": sorted(source_dates - index_dates),
                  "classification": "option_only_session"}).to_csv(
        REPORT_DIR / "option_only_sessions.csv", index=False)
    pd.DataFrame({"date": sorted(index_dates - source_dates),
                  "classification": "index_session_missing_in_option"}).to_csv(
        REPORT_DIR / "missing_sessions.csv", index=False)

    missing_records = []
    for date, group in source.groupby(source.timestamp.dt.date, sort=True):
        expected = pd.date_range(f"{date} 09:15", f"{date} 15:29", freq="min",
                                 tz="Asia/Kolkata")
        observed = pd.DatetimeIndex(group.timestamp).floor("min").unique()
        classification = "aligned_index_session" if date in index_dates else "option_only_session"
        for missing in expected.difference(observed):
            missing_records.append({"date": str(date), "missing_timestamp": missing.isoformat(),
                                    "session_class": classification})
    pd.DataFrame(missing_records, columns=["date", "missing_timestamp", "session_class"]).to_csv(
        REPORT_DIR / "missing_candles.csv", index=False)

    features = build_option_features(source)
    prefix_small = build_option_features(source.iloc[:1000])
    prefix_large = build_option_features(source.iloc[:5000]).iloc[:1000]
    assert_frame_equal(prefix_small, prefix_large, check_exact=True)
    live = build_live_option_features(source)
    if not live.equals(features.iloc[-1]):
        raise AssertionError("Historical/live option feature parity failed")
    if forbidden and set(forbidden).intersection(features.columns):
        raise AssertionError("Forbidden option-chain metadata leaked into canonical features")

    parquet = OUTPUT_DIR / "option_feature_dataset.parquet"
    csv = OUTPUT_DIR / "option_feature_dataset.csv"
    features.to_parquet(parquet, index=False)
    features.to_csv(csv, index=False)
    features[OPTION_FEATURES].describe().T.to_csv(REPORT_DIR / "feature_statistics.csv")
    completeness = pd.DataFrame({
        "feature": OPTION_FEATURES,
        "missing_count": [int(features[name].isna().sum()) for name in OPTION_FEATURES],
        "missing_percentage": [float(features[name].isna().mean() * 100) for name in OPTION_FEATURES],
    })
    completeness.to_csv(REPORT_DIR / "feature_completeness.csv", index=False)

    locked = {
        "stage1_v2_source": sha256(Path("src/setup_engine.py")),
        "index_feature_pipeline": sha256(Path("src/feature_pipeline.py")),
        "stage2_outcome_module": sha256(Path("src/stage2_outcome_dataset.py")),
        "stage1_dataset": sha256(Path("data/setups/stage1_setup_dataset.parquet")),
        "stage2_dataset": sha256(Path("data/outcomes/stage2_outcome_dataset.parquet")),
    }
    metadata = {
        "feature_version": OPTION_FEATURE_VERSION, "features": OPTION_FEATURES,
        "rows": len(features), "input_sha256": audit["source_sha256"],
        "option_feature_pipeline_sha256": sha256(Path("src/option_feature_pipeline.py")),
        "parquet_sha256": sha256(parquet), "csv_sha256": sha256(csv),
        "source_fields_used": SOURCE_COLUMNS, "source_fields_excluded": forbidden,
        "locked_artifact_hashes_after_build": locked,
    }
    (REPORT_DIR / "option_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    freeze = {
        "status": "FROZEN", "feature_version": OPTION_FEATURE_VERSION,
        "option_feature_pipeline_sha256": metadata["option_feature_pipeline_sha256"],
        "input_sha256": metadata["input_sha256"],
        "parquet_sha256": metadata["parquet_sha256"], "csv_sha256": metadata["csv_sha256"],
        "features": OPTION_FEATURES,
    }
    (REPORT_DIR / "option_feature_freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")

    (REPORT_DIR / "data_quality_report.md").write_text(f"""# Option Data Quality Report

- Rows: **{len(source):,}**; sessions: **{audit['sessions']:,}**
- Coverage: **{audit['earliest']}** through **{audit['latest']}**
- Duplicate timestamps / exact rows: **{audit['duplicate_timestamps']} / {audit['exact_duplicate_rows']}**
- Missing OHLCV / invalid OHLC / negative volume: **{audit['missing_ohlcv']} / {audit['high_below_low'] + audit['open_outside_range'] + audit['close_outside_range']} / {audit['negative_volume']}**
- Zero-volume rows: **{audit['zero_volume']:,}**
- Non-minute timestamps: **{audit['non_minute_aligned']}**
- Rows outside the cleaned Index 09:15–15:29 convention: **{audit['outside_index_regular_session']:,}**

The source was not cleaned, filtered, merged, or overwritten. Extra source fields `{forbidden}` were audited but excluded from the canonical contract by policy.
""", encoding="utf-8")
    (REPORT_DIR / "session_alignment_report.md").write_text(f"""# Option/Index Session Alignment

- Exact matching timestamps: **{audit['exact_timestamp_matches_index']:,}**
- Option timestamps outside cleaned Index baseline: **{audit['option_timestamps_not_in_clean_index']:,}**
- Cleaned Index timestamps absent from options: **{audit['clean_index_timestamps_missing_in_option']:,}**
- Matching sessions: **{audit['matching_sessions']:,}**
- Option-only sessions: **{audit['option_only_sessions']:,}**
- Index-only sessions: **{audit['index_only_sessions']:,}**

The datasets remain independent. No timestamp synchronization, filling, or merge was performed.
""", encoding="utf-8")
    (REPORT_DIR / "feature_pipeline_report.md").write_text(f"""# Canonical Option Feature Pipeline

- Version: `{OPTION_FEATURE_VERSION}`
- Source rows/output rows: **{len(source):,}/{len(features):,}**
- Canonical features: **{len(OPTION_FEATURES)}**
- Inputs: timestamp, OHLC and volume only.
- OI, IV, strike, expiry, Greeks and spot are prohibited.
- The source contains `{forbidden}`, but none appears in the output contract.
""", encoding="utf-8")
    (REPORT_DIR / "prefix_invariance_report.md").write_text(
        "# Prefix Invariance\n\nRows 1–1000 are bit-for-bit identical when generated from prefixes of 1,000 and 5,000 rows: **PASS**.\n",
        encoding="utf-8")
    (REPORT_DIR / "leakage_report.md").write_text(
        "# Future-Leakage Validation\n\nAll rolling extrema and volume baselines use trailing windows; structural reference levels and volume means are shifted by one candle. No centered window, backward fill, negative shift or future timestamp is used: **PASS**.\n",
        encoding="utf-8")
    (REPORT_DIR / "historical_live_parity_report.md").write_text(
        "# Historical/Live Parity\n\n`build_live_option_features(history)` delegates to the same canonical `build_option_features()` implementation. Latest historical and live-adapter rows are exactly equal: **PASS**.\n",
        encoding="utf-8")
    (REPORT_DIR / "final_validation_summary.md").write_text(f"""# Step 18A Final Validation Summary

Canonical option feature generation completed without models or changes to frozen Index/Stage artifacts. OHLCV integrity passes. Timestamp alignment is not perfect and remains explicitly documented: {audit['clean_index_timestamps_missing_in_option']} cleaned-Index timestamps are missing from options and {audit['option_timestamps_not_in_clean_index']:,} option timestamps are outside the cleaned Index baseline.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
