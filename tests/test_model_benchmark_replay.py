import pandas as pd
import numpy as np

from src.model_benchmark_replay import attach_fold_ids, validate_replay_metrics


def test_fold_ids_cover_exact_oof_rows():
    total = 100
    # Four TimeSeriesSplit test folds contain 80 rows for n=100.
    timestamp = pd.date_range(
        "2025-01-01", periods=80, freq="min", tz="Asia/Kolkata"
    )
    predictions = pd.DataFrame({
        "timestamp": timestamp, "model": ["CATBOOST"] * 80,
        "probability": [0.2, 0.8] * 40,
        "prediction": [0, 1] * 40, "target": [0, 1] * 40,
    })
    result = attach_fold_ids(predictions, total)
    assert len(result) == 80
    assert result.fold_id.value_counts().to_dict() == {1: 20, 2: 20, 3: 20, 4: 20}


def test_metric_replay_validation_matches_identical_summary():
    timestamp = pd.date_range(
        "2025-01-01", periods=8, freq="min", tz="Asia/Kolkata"
    )
    rows = []
    summary = []
    from src.model_benchmark import metrics
    for model in ("CATBOOST", "XGBOOST"):
        probability = np.array([0.1, 0.9] * 4)
        target = [0, 1] * 4
        rows.extend({
            "timestamp": timestamp[index], "model_name": model,
            "predicted_probability": probability[index],
            "predicted_class": target[index], "true_label": target[index],
        } for index in range(8))
        summary.append({"model": model, **metrics(pd.Series(target), probability)})
    result = validate_replay_metrics(pd.DataFrame(rows), pd.DataFrame(summary))
    assert result["match"].all()
