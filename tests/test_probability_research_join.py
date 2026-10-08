import pandas as pd

from src.probability_research_join import (
    DECISION_COLUMNS,
    assert_join_prefix_parity,
    build_joined_probability_dataset,
)


def _frames(rows=5):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    def oof(model):
        return pd.DataFrame({
            "timestamp": timestamp, "fold_id": [1] * rows,
            "model_name": [model] * rows,
            "predicted_probability": [0.6] * rows,
            "predicted_class": [1] * rows, "true_label": [1] * rows,
        })
    values = {
        column: ([0.1] * rows if column != "timestamp" else timestamp)
        for column in DECISION_COLUMNS
    }
    for column in (
        "market_state", "trend_regime", "volatility_regime", "session_phase",
        "structure_context", "current_strategy", "activated_strategy",
        "estimate_source", "recommendation", "decision_reason",
    ):
        values[column] = ["STATE"] * rows
    return oof("CATBOOST"), oof("XGBOOST"), pd.DataFrame(values)


def test_exact_join_has_one_row_per_timestamp():
    result = build_joined_probability_dataset(*_frames())
    assert len(result) == 5
    assert result.timestamp.nunique() == 5
    assert result.catboost_predicted_probability.eq(0.6).all()


def test_prefix_parity():
    assert_join_prefix_parity(*_frames(), prefix_rows=3)


def test_mismatched_oof_timestamps_are_rejected():
    cat, xgb, decisions = _frames()
    xgb = xgb.iloc[:-1]
    try:
        build_joined_probability_dataset(cat, xgb, decisions)
    except AssertionError:
        return
    raise AssertionError("Mismatched OOF timestamps were accepted")
