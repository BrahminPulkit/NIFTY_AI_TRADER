import pandas as pd

from src.probability_threshold_research import (
    THRESHOLDS,
    evaluate_thresholds,
    pareto_frontier,
    validate_joined_dataset,
)


def _data(rows=20):
    timestamp = pd.date_range(
        "2025-01-01", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    return pd.DataFrame({
        "timestamp": timestamp, "fold_id": [1] * rows,
        "true_label": [0, 1] * (rows // 2),
        "catboost_predicted_probability": [0.2, 0.8] * (rows // 2),
        "xgboost_predicted_probability": [0.3, 0.7] * (rows // 2),
        "terminal_premium_return": [-0.1, 0.1] * (rows // 2),
        "premium_expansion": [0.01, 0.08] * (rows // 2),
        "premium_drawdown": [0.05, 0.02] * (rows // 2),
        "holding_time_minutes": [10.0] * rows,
        "strategy_family": ["BREAKOUT"] * rows,
        "market_regime": ["UP|NORMAL"] * rows,
        "session_phase": ["MID"] * rows,
    })


def test_threshold_grid_is_fixed_and_complete():
    assert THRESHOLDS == (
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 0.97, 0.99,
    )


def test_threshold_statistics_use_saved_probabilities():
    result = evaluate_thresholds(validate_joined_dataset(_data()))
    assert len(result) == 2 * len(THRESHOLDS)
    cat = result.loc[
        result.model.eq("CATBOOST") & result.threshold.eq(0.5)
    ].iloc[0]
    assert cat.trades == 10
    assert cat.win_rate == 1.0
    assert cat.expected_value_per_trade == 0.1


def test_pareto_rows_are_subset_of_statistics():
    statistics = evaluate_thresholds(validate_joined_dataset(_data()))
    frontier = pareto_frontier(statistics)
    assert set(frontier.model).issubset(set(statistics.model))
    assert set(frontier.threshold).issubset(set(statistics.threshold))
