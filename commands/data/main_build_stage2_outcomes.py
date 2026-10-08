"""Build reproducible Step-18 Stage-2 and diagnostic artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.feature_pipeline import FEATURES
from src.stage2_outcome_dataset import (
    CONTRACT, build_diagnostic_decomposition, build_stage2_outcomes, summarize_outcomes,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = Path("data/setups/stage1_setup_dataset.parquet")
    output_dir = Path("data/outcomes")
    report_dir = Path("reports/stage2_outcome")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1 = pd.read_parquet(source)
    outcomes = build_stage2_outcomes(stage1)
    decomposition = build_diagnostic_decomposition(stage1, outcomes)
    parquet = output_dir / "stage2_outcome_dataset.parquet"
    csv = output_dir / "stage2_outcome_dataset.csv"
    diagnostic_parquet = output_dir / "stage2_diagnostic_decomposition.parquet"
    outcomes.to_parquet(parquet, index=False)
    outcomes.to_csv(csv, index=False)
    decomposition.to_parquet(diagnostic_parquet, index=False)

    distribution = outcomes.final_outcome.value_counts().rename_axis("final_outcome").reset_index(name="count")
    distribution["percentage"] = distribution["count"] / len(outcomes) * 100
    distribution.to_csv(report_dir / "outcome_distribution.csv", index=False)
    direction = summarize_outcomes(outcomes, ["direction"])
    yearly = summarize_outcomes(outcomes, ["year", "direction"])
    regime = summarize_outcomes(outcomes, ["trend_state", "direction"])
    session = summarize_outcomes(outcomes, ["session_segment", "direction"])
    direction.to_csv(report_dir / "direction_statistics.csv", index=False)
    yearly.to_csv(report_dir / "year_statistics.csv", index=False)
    regime.to_csv(report_dir / "regime_statistics.csv", index=False)
    session.to_csv(report_dir / "session_statistics.csv", index=False)
    decomposition.diagnostic_subtype.value_counts().rename_axis("diagnostic_subtype").reset_index(
        name="count").to_csv(report_dir / "diagnostic_decomposition.csv", index=False)

    feature_completeness = pd.DataFrame({
        "feature": FEATURES,
        "missing_count": [int(outcomes[name].isna().sum()) for name in FEATURES],
        "missing_percentage": [float(outcomes[name].isna().mean() * 100) for name in FEATURES],
    })
    feature_completeness.to_csv(report_dir / "feature_completeness.csv", index=False)
    metadata = {
        "contract": CONTRACT.as_dict(), "source": source.as_posix(), "source_sha256": sha256(source),
        "stage1_v2_sha256": sha256(Path("src/setup_engine.py")),
        "feature_pipeline_sha256": sha256(Path("src/feature_pipeline.py")),
        "rows": len(outcomes), "allowed_source_rows": int(stage1.trade_allowed.eq(1).sum()),
        "output_parquet_sha256": sha256(parquet), "output_csv_sha256": sha256(csv),
        "canonical_features": FEATURES,
    }
    (report_dir / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    counts = distribution.set_index("final_outcome")["count"].to_dict()
    report = f"""# Stage-2 Outcome Dataset Report

## Frozen contract

- Horizon: **5 candles**
- Entry: candle close
- Stop: 0.20%; target: 0.30% (1.5R)
- Conservative stop-first ambiguity; gap fill at first available open
- No cross-session outcomes

## Dataset

- Frozen Stage-1 allowed rows: **{int(stage1.trade_allowed.eq(1).sum()):,}**
- Stage-2 rows: **{len(outcomes):,}**
- WIN: **{counts.get('WIN', 0):,}**
- LOSS: **{counts.get('LOSS', 0):,}**
- TIMEOUT: **{counts.get('TIMEOUT', 0):,}**
- Expected R: **{outcomes.realized_r.mean():.6f}R**

The Stage-2 dataset contains no `NO_SETUP`, rejected setup, or `NO_TRADE` class. The full-row diagnostic decomposition is stored separately and must never be supplied to Stage-2 learning.
"""
    (report_dir / "outcome_dataset_report.md").write_text(report, encoding="utf-8")
    leakage = f"""# Leakage Verification

- Entry features are copied exactly from the frozen canonical feature snapshot at timestamp t.
- Outcome scanning begins at t+1 and is used only for exit/target columns.
- `final_outcome`, exit fields, MFE, MAE, and realized R are targets/audit fields, never canonical inputs.
- No outcome crosses `session_id`.
- Prefix invariance is tested for finalized rows at least five candles before a prefix boundary.
- Historical/live parity applies to entry features and frozen Stage-1 decisions; future outcomes do not exist in live inference.
"""
    (report_dir / "leakage_verification.md").write_text(leakage, encoding="utf-8")
    (report_dir / "final_summary.md").write_text(
        "# Final Stage-2 Summary\n\nDataset engineering completed without model training, calibration, threshold tuning, decision policy, or backtesting.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
