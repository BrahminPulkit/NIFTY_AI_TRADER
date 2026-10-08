"""Audit NIFTY VWAP provenance and evaluate the repaired 48-feature matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import feature_columns  # noqa: E402
from tools.run_phase12_feature_signal_audit import quality, rank_features  # noqa: E402


OUTPUT = ROOT / "reports/vwap_dependency_audit"
REPAIRED = ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet"


def volume_stats(values: pd.Series) -> dict:
    data = pd.to_numeric(values, errors="coerce")
    return {"rows": len(data), "non_null": int(data.notna().sum()),
            "positive": int(data.gt(0).sum()), "zero": int(data.eq(0).sum()),
            "negative": int(data.lt(0).sum()), "unique": int(data.nunique())}


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/nifty_50_5years_1min.csv", usecols=["timestamp", "volume"])
    cleaned = pd.read_parquet(ROOT / "data/cleaned/nifty_5year_cleaned.parquet",
                              columns=["timestamp", "volume_raw", "volume"])
    caches = []
    for path in sorted((ROOT / "data/cache/dhan_mtf_scalper").glob("nifty_index_*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        caches.append(frame)
    cache = pd.concat(caches, ignore_index=True)
    cache["timestamp"] = pd.to_datetime(cache.timestamp, errors="coerce")
    validated_dates = set()
    validated_dates.update(path.name for path in
                           (ROOT / "data/normalized/historical_options/validated20").iterdir()
                           if path.is_dir())
    for path in (ROOT / "data/normalized/historical_options/non_expiry15").iterdir():
        if path.is_dir():
            validated_dates.add(json.loads((path / "manifest.json").read_text())["date"])
    cache_dates = set(cache.timestamp.dt.strftime("%Y-%m-%d").dropna())
    source_audit = {
        "legacy_raw_index": {
            **volume_stats(raw.volume),
            "classification": "UNUSABLE_INCONSISTENT_PROVIDER_FIELD",
            "reason": "Mostly zero, includes negative values, and changes semantics across history.",
        },
        "approved_clean_index": {
            **volume_stats(cleaned.volume),
            "volume_raw": volume_stats(cleaned.volume_raw),
            "classification": "NO_APPROVED_ANALYTICAL_VOLUME",
        },
        "recent_dhan_index_cache": {
            **volume_stats(cache.volume), "rows_after_timestamp_dedup": int(cache.timestamp.nunique()),
            "date_count": int(cache.timestamp.dt.date.nunique()),
            "first_timestamp": str(cache.timestamp.min()), "last_timestamp": str(cache.timestamp.max()),
            "validated_session_overlap": sorted(validated_dates & cache_dates),
            "classification": "RECENT_ONLY_SEMANTICS_NOT_DOCUMENTED_FOR_INDEX_VOLUME",
        },
        "live_broker": {"status": "TOKEN_EXPIRED_DH_901", "historical_backfill_run": False},
        "official_api_contract": {
            "endpoint": "POST /v2/charts/intraday",
            "documents_generic_volume_array": True,
            "documents_IDX_I_volume_semantics": False,
            "url": "https://dhanhq.co/docs/v2/historical-data/",
        },
        "final_source_verdict": "NO_VALIDATED_NIFTY_VOLUME_FOR_THE_35_RESEARCH_SESSIONS",
        "price_only_replacement": {
            "drop_in_replacement_justified": False,
            "reason": "TWAP or an expanding typical-price mean measures time/price location, not traded-volume weighting.",
            "research_decision": "Evaluate the existing 48 non-VWAP features; do not rename a price-only feature as VWAP.",
        },
    }

    features49 = feature_columns()
    features48 = [name for name in features49 if name != "nifty_vwap_distance"]
    frame = pd.read_parquet(REPAIRED)
    labels = pd.concat([
        pd.read_parquet(ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"),
        pd.read_parquet(ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"),
    ], ignore_index=True)
    labels = labels.loc[labels.actual_exit_reason.isin(["WIN", "LOSS"])].sort_values(
        "timestamp", kind="stable").drop_duplicates(
            ["timestamp", "expiry", "strike", "option_type"], keep="first")
    labels["label"] = labels.actual_exit_reason.eq("WIN").astype("int8")
    meta = ["timestamp", "session_date", "expiry_day_status", "option_type", "trend_regime", "label"]
    data = labels[meta].merge(frame[["timestamp", *features48]], on="timestamp", how="left",
                              validate="many_to_one")
    quality_table = quality(frame, features48)
    rankings = [rank_features(data, features48, "ALL")]
    for column, values in {"option_type": ["CE", "PE"],
                           "expiry_day_status": ["EXPIRY_DAY", "NON_EXPIRY_DAY"],
                           "trend_regime": ["TREND_UP", "TREND_DOWN", "SIDEWAYS"]}.items():
        for value in values:
            rankings.append(rank_features(data.loc[data[column].eq(value)], features48,
                                          f"{column}={value}"))
    ranking = pd.concat(rankings, ignore_index=True)
    overall = ranking.loc[ranking.group.eq("ALL")].copy()
    complete = data[features48].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    stable = overall.loc[
        overall.walk_forward_folds.eq(5) & overall.mean_oos_auc.ge(.53)
        & overall.minimum_oos_auc.ge(.45) & overall.positive_oos_folds.ge(3)]
    predictive_verdict = "WEAK_BUT_USABLE" if len(stable) >= 3 else "NO_MEANINGFUL_SIGNAL"
    result = {
        "status": "VWAP_DEPENDENCY_AUDIT_COMPLETE",
        "volume_source_verdict": source_audit["final_source_verdict"],
        "vwap_removed_from_research_matrix": True,
        "vwap_replaced": False, "existing_48_features_changed": False,
        "feature_count": 48, "feature_rows": len(frame),
        "complete_48_feature_rows": int(frame[features48].notna().all(axis=1).sum()),
        "complete_labeled_observations": int(complete.sum()),
        "labeled_observations": len(data), "wins": int(data.label.sum()),
        "losses": int(data.label.eq(0).sum()),
        "minimum_feature_coverage_pct": float(quality_table.coverage_pct.min()),
        "predictive_verdict": predictive_verdict,
        "stable_features_at_declared_gate": int(len(stable)),
        "top_10": overall.head(10)[["feature", "mean_oos_auc", "minimum_oos_auc",
                                     "positive_oos_folds", "mean_train_mutual_information"]].to_dict("records"),
        "catboost_blocked": True, "production_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "source_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    quality_table.to_csv(OUTPUT / "feature_quality_48.csv", index=False)
    ranking.to_csv(OUTPUT / "feature_rankings_48.csv", index=False)
    print(json.dumps({"source": source_audit, "research_48": result}, indent=2))


if __name__ == "__main__":
    main()
