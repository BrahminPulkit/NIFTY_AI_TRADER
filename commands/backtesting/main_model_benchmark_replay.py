"""Execute the approved exact Step 20B replay and persist OOF probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.model_benchmark import prepare_benchmark_data, run_walk_forward
from src.model_benchmark_replay import (
    REPLAY_MODELS,
    attach_fold_ids,
    validate_replay_metrics,
)


PREDICTION = Path("data/prediction/prediction_dataset.parquet")
DECISIONS = Path("reports/strategy_decision/decision_matrix.csv")
FROZEN_SUMMARY = Path("reports/model_benchmark/benchmark_summary.csv")
OUTPUT = Path("data/prediction/oof")
REPORTS = Path("reports/model_benchmark_replay")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prediction = pd.read_parquet(PREDICTION)
    target = pd.read_csv(
        DECISIONS, usecols=["timestamp", "realized_expansion_hit"]
    ).rename(columns={"realized_expansion_hit": "research_target"})
    X, y, metadata, excluded = prepare_benchmark_data(prediction, target)
    outputs = {}
    for model_name in REPLAY_MODELS:
        predictions, _, _ = run_walk_forward(X, y, metadata, model_name)
        outputs[model_name] = attach_fold_ids(predictions, len(X))
    combined = pd.concat(outputs.values(), ignore_index=True)
    validation = validate_replay_metrics(
        combined, pd.read_csv(FROZEN_SUMMARY), tolerance=1e-12
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    files = {}
    for model_name, stem in (
        ("CATBOOST", "oof_predictions_catboost"),
        ("XGBOOST", "oof_predictions_xgboost"),
    ):
        frame = outputs[model_name]
        parquet = OUTPUT / f"{stem}.parquet"
        csv = OUTPUT / f"{stem}.csv"
        frame.to_parquet(parquet, index=False)
        frame.to_csv(csv, index=False)
        files[model_name] = {
            "parquet": str(parquet),
            "csv": str(csv),
            "rows": len(frame),
            "parquet_sha256": _hash(parquet),
            "csv_sha256": _hash(csv),
        }
    validation.to_csv(REPORTS / "metric_validation.csv", index=False)
    max_difference = float(validation.absolute_difference.max())
    report = f"""# Step 20B-R — Reproducible Replay Validation

- Purpose: persist row-level OOF predictions from the frozen Step 20B protocol.
- Models replayed: CatBoost and XGBoost only.
- OOF rows per model: **{len(outputs["CATBOOST"]):,}**
- Folds: **4 expanding chronological TimeSeriesSplit folds**
- Threshold: **0.50**
- Fully-null feature excluded exactly as Step 20B: {", ".join(excluded)}
- Metric checks: **{len(validation)} / {len(validation)} PASS**
- Maximum absolute metric difference: **{max_difference:.3e}**
- Tolerance: **1e-12**

No benchmark report, frozen dataset, model, feature, label, parameter,
preprocessor, fold, or threshold was changed. No model was promoted or saved.
"""
    (REPORTS / "replay_validation_report.md").write_text(report, encoding="utf-8")
    metadata_out = {
        "step": "20B-R",
        "purpose": "exact frozen benchmark replay for OOF persistence",
        "models": list(REPLAY_MODELS),
        "rows_per_model": len(outputs["CATBOOST"]),
        "folds": 4,
        "threshold": 0.5,
        "metric_tolerance": 1e-12,
        "metric_validation": "PASS",
        "maximum_absolute_metric_difference": max_difference,
        "outputs": files,
        "frozen_input_hashes": {
            "prediction_dataset": _hash(PREDICTION),
            "approved_target_source": _hash(DECISIONS),
            "benchmark_summary": _hash(FROZEN_SUMMARY),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

