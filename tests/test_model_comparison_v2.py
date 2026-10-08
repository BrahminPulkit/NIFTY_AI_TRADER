import pandas as pd

from src.model_comparison_v2 import (
    choose_dataset, curve_tables, numeric_feature_stability,
)


def test_curve_tables_cover_each_model():
    data = pd.DataFrame({
        "model": ["A"] * 4 + ["B"] * 4,
        "target": [0, 1, 0, 1] * 2,
        "probability": [0.1, 0.9, 0.2, 0.8] * 2,
    })
    roc, pr = curve_tables(data)
    assert set(roc.model) == {"A", "B"}
    assert set(pr.model) == {"A", "B"}


def test_feature_stability_is_zero_for_identical_windows():
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2025-01-01", "2025-01-02"
        ], utc=True).tz_convert("Asia/Kolkata"),
        "feature": [1.0, 2.0, 1.0, 2.0],
    })
    result = numeric_feature_stability(
        data,
        pd.Timestamp("2024-12-31", tz="Asia/Kolkata"),
        pd.Timestamp("2025-01-01", tz="Asia/Kolkata"),
    )
    assert result.psi.iloc[0] == 0


def test_dataset_choice_is_exactly_one():
    data = pd.DataFrame({
        "dataset_version": ["V1", "V2"],
        "model": ["CATBOOST", "CATBOOST"],
        "roc_auc": [0.6, 0.61], "pr_auc": [0.5, 0.51],
        "median_feature_psi": [0.2, 0.05],
        "holdout_robust": [False, True],
    })
    recommendation, checks = choose_dataset(data)
    assert recommendation == "Production Dataset V2"
    assert len(checks) == 4

