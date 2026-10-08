"""Step 20B-R helpers for validating an exact benchmark replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_benchmark import PROTOCOL, chronological_splits, metrics


REPLAY_MODELS = ("CATBOOST", "XGBOOST")
METRIC_COLUMNS = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "brier_score",
    "log_loss",
    "tn",
    "fp",
    "fn",
    "tp",
)


def attach_fold_ids(predictions: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """Attach the exact frozen TimeSeriesSplit fold IDs to concatenated OOF rows."""
    expected_test_indices = []
    fold_ids = []
    for fold, (_, test) in enumerate(chronological_splits(total_rows, PROTOCOL), 1):
        expected_test_indices.extend(test.tolist())
        fold_ids.extend([fold] * len(test))
    if len(predictions) != len(expected_test_indices):
        raise AssertionError("OOF row count does not match frozen fold boundaries")
    output = pd.DataFrame({
        "timestamp": predictions.timestamp.to_numpy(),
        "fold_id": fold_ids,
        "model_name": predictions.model.to_numpy(),
        "predicted_probability": predictions.probability.to_numpy(float),
        "predicted_class": predictions.prediction.to_numpy("int8"),
        "true_label": predictions.target.to_numpy("int8"),
    })
    if output.timestamp.duplicated().any():
        raise AssertionError("Each model replay must contain one OOF row per timestamp")
    if not output.timestamp.is_monotonic_increasing:
        raise AssertionError("Replay predictions must be chronological")
    return output


def validate_replay_metrics(
    oof: pd.DataFrame,
    frozen_summary: pd.DataFrame,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    rows = []
    for model_name in REPLAY_MODELS:
        replay = oof.loc[oof.model_name.eq(model_name)]
        computed = metrics(
            replay.true_label,
            replay.predicted_probability.to_numpy(),
            PROTOCOL.threshold,
        )
        frozen = frozen_summary.set_index("model").loc[model_name]
        for metric_name in METRIC_COLUMNS:
            expected = float(frozen[metric_name])
            actual = float(computed[metric_name])
            difference = abs(actual - expected)
            rows.append({
                "model": model_name,
                "metric": metric_name,
                "frozen_value": expected,
                "replay_value": actual,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "match": bool(difference <= tolerance),
            })
    result = pd.DataFrame(rows)
    if not result["match"].all():
        failed = result.loc[~result["match"]]
        raise AssertionError(f"Replay metrics do not match frozen benchmark:\n{failed}")
    return result

