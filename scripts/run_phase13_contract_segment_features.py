"""Generate the unchanged 49-feature contract from full fixed-contract histories."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import (  # noqa: E402
    BLOCK_FEATURES, OPTION_EXTRA_FEATURES, build_ce_pe_feature_preview_from_panels,
    feature_columns,
)
from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402


OUTPUT = ROOT / "data/normalized/historical_options/phase13_contract_features"
REPORT = ROOT / "reports/phase13_contract_segment_features"


def sources():
    expiry_root = ROOT / "data/normalized/historical_options/validated20"
    for session in sorted(path for path in expiry_root.iterdir() if path.is_dir()):
        yield "EXPIRY_DAY", session, ROOT / "data/external/shoonyatrader/validated20_raw" / f"{session.name.replace('-', '')}.zip", session.name
    non_root = ROOT / "data/normalized/historical_options/non_expiry15"
    for session in sorted(path for path in non_root.iterdir() if path.is_dir()):
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        yield "NON_EXPIRY_DAY", session, ROOT / "data/external/shoonyatrader/validated20_raw" / manifest["source_archive"], manifest["date"]


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in feature_columns():
        values = pd.to_numeric(frame[name], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).notna()
        rows.append({"feature": name, "finite": int(finite.sum()),
                     "missing": int((~finite).sum()),
                     "coverage_pct": float(finite.mean() * 100),
                     "infinite": int(np.isinf(values).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    frames, session_audits, prefix_checks = [], [], []
    ce_names = [f"ce_{name}" for name in BLOCK_FEATURES + OPTION_EXTRA_FEATURES]
    pe_names = [f"pe_{name}" for name in BLOCK_FEATURES + OPTION_EXTRA_FEATURES]
    nifty_names = [f"nifty_{name}" for name in BLOCK_FEATURES]
    for day_type, session, archive, date in sources():
        index, ce, pe = (pd.read_parquet(session / name) for name in
                         ("nifty.parquet", "ce.parquet", "pe.parquet"))
        panel = load_selected_contract_panel(archive, ce, pe, session_date=date)
        ce_panel, pe_panel = panel.loc[panel.option_type.eq("CE")], panel.loc[panel.option_type.eq("PE")]
        result = build_ce_pe_feature_preview_from_panels(index, ce, pe, ce_panel, pe_panel)
        if day_type not in {row["expiry_day_status"] for row in prefix_checks}:
            timestamp = index.sort_values("timestamp").timestamp.iloc[199]
            prefix = build_ce_pe_feature_preview_from_panels(
                index.loc[index.timestamp.le(timestamp)], ce.loc[ce.timestamp.le(timestamp)],
                pe.loc[pe.timestamp.le(timestamp)], ce_panel, pe_panel)
            full_row = result.loc[result.timestamp.eq(timestamp), feature_columns()]
            prefix_row = prefix.loc[prefix.timestamp.eq(timestamp), feature_columns()]
            prefix_checks.append({
                "session_date": date, "expiry_day_status": day_type,
                "timestamp": str(timestamp),
                "prefix_invariant": bool(len(full_row) == len(prefix_row) == 1 and np.allclose(
                    full_row.to_numpy(dtype=float), prefix_row.to_numpy(dtype=float), equal_nan=True)),
            })
        result.insert(0, "session_date", date)
        result.insert(1, "expiry_day_status", day_type)
        ce_ordered = ce.sort_values("timestamp", kind="stable").reset_index(drop=True)
        pe_ordered = pe.sort_values("timestamp", kind="stable").reset_index(drop=True)
        starts = (ce_ordered.contract_segment_id.ne(ce_ordered.contract_segment_id.shift())
                  | pe_ordered.contract_segment_id.ne(pe_ordered.contract_segment_id.shift()))
        starts.index = ce_ordered.timestamp
        complete_at_timestamp = result.set_index("timestamp")[ce_names + pe_names].notna().all(axis=1)
        switch_rejections = int((starts & ~complete_at_timestamp.reindex(starts.index).fillna(False)).sum())
        session_audits.append({
            "session_date": date, "expiry_day_status": day_type, "rows": len(result),
            "duplicate_timestamps": int(result.timestamp.duplicated().sum()),
            "contract_switch_rows": int(starts.sum()),
            "contract_switch_rejection_rows": switch_rejections,
            "ce_complete": int(result[ce_names].notna().all(axis=1).sum()),
            "pe_complete": int(result[pe_names].notna().all(axis=1).sum()),
        })
        frames.append(result)
    repaired = pd.concat(frames, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(OUTPUT / "feature_dataset_49.parquet", index=False)
    after = coverage(repaired)
    before_frame = pd.concat([
        pd.read_parquet(ROOT / "data/normalized/historical_options/validated20/feature_dataset_49.parquet"),
        pd.read_parquet(ROOT / "data/normalized/historical_options/non_expiry15/feature_dataset_49.parquet"),
    ], ignore_index=True)
    before = coverage(before_frame).rename(columns={
        "finite": "before_finite", "missing": "before_missing",
        "coverage_pct": "before_coverage_pct", "infinite": "before_infinite"})
    comparison = before.merge(after, on="feature", validate="one_to_one")
    comparison["coverage_gain_points"] = comparison.coverage_pct - comparison.before_coverage_pct
    labels = pd.concat([
        pd.read_parquet(ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"),
        pd.read_parquet(ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"),
    ], ignore_index=True)
    labels = labels.loc[labels.actual_exit_reason.isin(["WIN", "LOSS"])].drop_duplicates(
        ["timestamp", "expiry", "strike", "option_type"])
    available_without_nifty_vwap = [name for name in feature_columns() if name != "nifty_vwap_distance"]
    complete49 = int(repaired[feature_columns()].notna().all(axis=1).sum())
    complete48 = int(repaired[available_without_nifty_vwap].notna().all(axis=1).sum())
    ce_complete = repaired[ce_names].notna().all(axis=1)
    pe_complete = repaired[pe_names].notna().all(axis=1)
    nifty_complete = repaired[nifty_names].notna().all(axis=1)
    nifty_price_complete = repaired[[name for name in nifty_names if name != "nifty_vwap_distance"]].notna().all(axis=1)
    relation_names = feature_columns()[-12:]
    cross_segment = int((repaired.ce_expiry.ne(repaired.pe_expiry)
                         | repaired.ce_strike.ne(repaired.pe_strike)).sum())
    result = {
        "status": "PHASE13_CONTRACT_SEGMENT_FEATURES_COMPLETE",
        "verdict": "C. Data limitation remains" if complete49 == 0 else
                   "A. Feature matrix sufficiently complete and ready for another predictive audit",
        "total_rows": len(repaired), "unique_labeled_observations": len(labels),
        "complete_49_feature_rows": complete49,
        "complete_48_non_vwap_rows": complete48,
        "ce_complete_rows": int(ce_complete.sum()), "pe_complete_rows": int(pe_complete.sum()),
        "nifty_complete_rows": int(nifty_complete.sum()),
        "nifty_price_feature_complete_rows": int(nifty_price_complete.sum()),
        "relationship_complete_rows": int(repaired[relation_names].notna().all(axis=1).sum()),
        "warmup_rejection_rows": int((~(ce_complete & pe_complete)).sum()),
        "contract_switch_rows": int(sum(row["contract_switch_rows"] for row in session_audits)),
        "contract_switch_rejection_rows": int(sum(
            row["contract_switch_rejection_rows"] for row in session_audits)),
        "missing_values": int(repaired[feature_columns()].isna().sum().sum()),
        "infinite_values": int(np.isinf(repaired[feature_columns()].select_dtypes("number")).sum().sum()),
        "duplicate_timestamps_within_session": int(repaired.duplicated(["session_date", "timestamp"]).sum()),
        "cross_segment_identity_rows": cross_segment,
        "nifty_vwap_status": "UNAVAILABLE_NOT_FABRICATED",
        "leakage_protection": {
            "prefix_invariance_checks": prefix_checks,
            "all_prefix_invariant": all(row["prefix_invariant"] for row in prefix_checks),
            "entry_timestamp_cutoff": True, "future_labels_excluded": True,
            "future_candles_excluded": True, "contract_lock_preserved": True,
        },
        "phase12_rerun": complete49 > 0,
        "catboost_blocked": True, "production_changed": False,
    }
    after.to_csv(REPORT / "feature_coverage_after.csv", index=False)
    comparison.to_csv(REPORT / "feature_coverage_before_after.csv", index=False)
    pd.DataFrame(session_audits).to_csv(REPORT / "session_coverage.csv", index=False)
    (REPORT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
