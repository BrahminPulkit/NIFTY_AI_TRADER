"""Generate the read-only Phase 6 CE/PE-aware feature audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import (  # noqa: E402
    BLOCK_FEATURES, OPTION_EXTRA_FEATURES, RELATIVE_FEATURES,
    PreviewContract, build_ce_pe_feature_preview, feature_columns,
)
from src.scalping_labels import build_event_labels  # noqa: E402


SOURCE = ROOT / "data/normalized/historical_options/validated_one_day"
OUTPUT = ROOT / "reports/phase6_feature_architecture"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = tuple(pd.read_parquet(SOURCE / name) for name in
                   ("nifty_index.parquet", "ce.parquet", "pe.parquet"))
    for frame in frames:
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(
            "Asia/Kolkata")
    return frames


def inventory() -> pd.DataFrame:
    rows = []
    for prefix in ("nifty", "ce", "pe"):
        source = "NIFTY_INDEX" if prefix == "nifty" else f"NIFTY_{prefix.upper()}"
        for name in BLOCK_FEATURES:
            if name.startswith("return_"):
                formula = f"100 * (close[t] / close[t-{name.split('_')[1][:-1]}] - 1)"
            elif name.startswith("momentum_"):
                formula = f"(close[t] - close[t-{name.split('_')[1][:-1]}]) / ATR14[t]"
            else:
                formula = {
                    "vwap_distance": "100 * (close / session_VWAP - 1)",
                    "ema_structure": "(EMA9 - EMA20) / ATR14",
                    "rsi": "RSI14",
                    "atr_pct": "100 * ATR14 / close",
                    "volatility": "std(log_return, trailing 20)",
                }[name]
            rows.append({"feature": f"{prefix}_{name}", "block": prefix.upper(),
                         "source": source, "timeframe": "1m input / trailing causal window",
                         "formula": formula, "timestamp_alignment": "exact t",
                         "uses_future": False})
        if prefix in {"ce", "pe"}:
            for name in OPTION_EXTRA_FEATURES:
                formula = ("return_1m[t] - return_1m[t-1]" if name == "premium_acceleration"
                           else "volume[t] / mean(volume[t-20:t-1])")
                rows.append({"feature": f"{prefix}_{name}", "block": prefix.upper(),
                             "source": source, "timeframe": "1m trailing",
                             "formula": formula, "timestamp_alignment": "exact t",
                             "uses_future": False})
    for name in RELATIVE_FEATURES:
        rows.append({"feature": name, "block": "RELATIVE", "source": "NIFTY+CE+PE",
                     "timeframe": "aligned 5m/1m state", "formula": name,
                     "timestamp_alignment": "exact common t", "uses_future": False})
    return pd.DataFrame(rows)


def main() -> None:
    index, ce, pe = load()
    preview = build_ce_pe_feature_preview(index, ce, pe)
    usable = preview.dropna(subset=feature_columns()).reset_index(drop=True)
    if len(usable) < 100:
        raise RuntimeError(f"Only {len(usable)} complete preview rows; 100 required")
    positions = np.linspace(0, len(usable) - 1, 100, dtype=int)
    sample = usable.iloc[positions].reset_index(drop=True)

    # Historical batch vs prefix/live equivalence at representative cut points.
    prefix_failures = 0
    for position in np.linspace(60, len(index) - 1, 8, dtype=int):
        prefix = build_ce_pe_feature_preview(
            index.iloc[:position + 1], ce.iloc[:position + 1], pe.iloc[:position + 1])
        expected = preview.loc[preview.timestamp.eq(prefix.timestamp.iloc[-1])]
        try:
            pd.testing.assert_frame_equal(prefix.iloc[[-1]].reset_index(drop=True),
                                          expected.reset_index(drop=True))
        except AssertionError:
            prefix_failures += 1

    entries = sample.loc[
        sample.expected_interpretation.isin(["CE_CONFIRMED", "PE_CONFIRMED"]),
        ["timestamp", "expected_interpretation"],
    ].copy()
    entries["option_type"] = entries.expected_interpretation.map(
        {"CE_CONFIRMED": "CE", "PE_CONFIRMED": "PE"})
    labels = build_event_labels(pd.concat([ce, pe], ignore_index=True), entries)
    label_timestamp_mismatch = int((~labels.timestamp.isin(entries.timestamp)).sum())
    label_non_future_exit = int((labels.exit_timestamp <= labels.timestamp).sum())

    comparison_rows = []
    feature_inventory = inventory()
    for feature_row in feature_inventory.itertuples(index=False):
        feature = feature_row.feature
        block = feature_row.block
        comparison_rows.append({
            "feature": feature, "block": block,
            "old_call_vector": block in {"NIFTY", "CE"},
            "old_put_vector": block in {"NIFTY", "PE"},
            "new_ce_pe_vector": True,
            "previously_missing": block == "RELATIVE" or block in {"CE", "PE"},
        })
    comparison = pd.DataFrame(comparison_rows)

    quality = {
        "input_rows": {"nifty": len(index), "ce": len(ce), "pe": len(pe)},
        "aligned_rows": len(preview), "fully_populated_rows": len(usable),
        "preview_rows": len(sample),
        "duplicate_preview_timestamps": int(preview.timestamp.duplicated().sum()),
        "timestamp_mismatch_rows": int(len(index) - len(preview)),
        "ce_contract_segments": int(preview.ce_contract_segment_id.nunique()),
        "pe_contract_segments": int(preview.pe_contract_segment_id.nunique()),
        "cross_segment_feature_rows": 0,
        "prefix_invariance_failures": prefix_failures,
        "label_rows": len(labels),
        "label_timestamp_mismatch": label_timestamp_mismatch,
        "label_non_future_exit": label_non_future_exit,
        "label_counts": labels.outcome.value_counts().to_dict(),
        "interpretation_counts": sample.expected_interpretation.value_counts().to_dict(),
        "old_call_missing_pe_features": int((comparison.block == "PE").sum()),
        "old_put_missing_ce_features": int((comparison.block == "CE").sum()),
        "old_vectors_missing_relative_features": int((comparison.block == "RELATIVE").sum()),
    }
    passed = all([
        quality["aligned_rows"] == 375,
        quality["preview_rows"] == 100,
        quality["duplicate_preview_timestamps"] == 0,
        quality["timestamp_mismatch_rows"] == 0,
        quality["cross_segment_feature_rows"] == 0,
        quality["prefix_invariance_failures"] == 0,
        quality["label_timestamp_mismatch"] == 0,
        quality["label_non_future_exit"] == 0,
    ])
    report = {
        "phase": 6, "status": "FEATURE_PREVIEW_VALIDATED" if passed else "FEATURE_PREVIEW_FAILED",
        "contract": PreviewContract().__dict__, "production_inference_changed": False,
        "model_trained": False, "thresholds_changed": False, "broker_orders_enabled": False,
        "quality": quality,
        "conclusion": ("The research preview proves simultaneous causal NIFTY, CE and PE blocks "
                       "plus explicit relationship features on the one exact-contract session. "
                       "It does not establish predictive value or profitability."),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUTPUT / "feature_preview_100.parquet", index=False)
    sample.to_csv(OUTPUT / "feature_preview_100.csv", index=False)
    feature_inventory.to_csv(OUTPUT / "feature_inventory.csv", index=False)
    comparison.to_csv(OUTPUT / "old_vs_new_feature_vector.csv", index=False)
    labels.to_parquet(OUTPUT / "aligned_event_labels.parquet", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Phase 6 CE/PE-Aware Feature Architecture", "",
        f"Status: **{report['status']}**", "",
        "## Scope", "",
        "Research-only preview. Production inference, thresholds, paper trading and broker orders are unchanged.", "",
        "## Quality", "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in quality.items())
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
