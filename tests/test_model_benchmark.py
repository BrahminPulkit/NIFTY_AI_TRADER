import numpy as np
import pandas as pd

from src.model_benchmark import (
    MODEL_NAMES,
    build_pipeline,
    chronological_splits,
    prepare_benchmark_data,
)


def test_time_series_splits_are_strictly_chronological():
    for train, test in chronological_splits(100):
        assert train[-1] < test[0]


def test_target_never_enters_feature_matrix():
    timestamp = pd.date_range("2025-01-01", periods=10, freq="min", tz="Asia/Kolkata")
    prediction = pd.DataFrame({
        "timestamp": timestamp, "feature": np.arange(10),
        "trend_regime": ["UP"] * 10, "volatility_regime": ["NORMAL"] * 10,
        "session_phase": ["OPEN"] * 10, "current_strategy": ["BREAKOUT"] * 10,
    })
    target = pd.DataFrame({"timestamp": timestamp, "research_target": [0, 1] * 5})
    X, y, _, _ = prepare_benchmark_data(prediction, target)
    assert "research_target" not in X
    assert y.sum() == 5


def test_all_requested_model_pipelines_construct():
    X = pd.DataFrame({"numeric": [1.0, 2.0], "category": ["a", "b"]})
    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(model_name, X)
        assert pipeline.named_steps["model"] is not None
