import pandas as pd

from src.prediction_dataset import (
    DECISION_COLUMNS,
    FORBIDDEN_COLUMNS,
    INDEX_COLUMNS,
    OPTION_COLUMNS,
    SETUP_COLUMNS,
    assert_prefix_invariance,
    build_live_record,
    build_prediction_dataset,
)


def _frames(rows=5):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    aligned = pd.DataFrame({"timestamp": timestamp})
    for column in INDEX_COLUMNS + OPTION_COLUMNS:
        aligned[column] = 1.0
    setups = pd.DataFrame({
        "timestamp": timestamp, "trade_allowed": [1] * rows,
        **{column: [1] * rows for column in SETUP_COLUMNS},
    })
    decisions = pd.DataFrame({
        "timestamp": timestamp,
        **{column: [1] * rows for column in DECISION_COLUMNS},
    })
    return aligned, setups, decisions


def test_one_record_per_setup_and_no_forbidden_fields():
    result = build_prediction_dataset(*_frames())
    assert len(result) == 5
    assert not FORBIDDEN_COLUMNS.intersection(result.columns)
    assert not result.timestamp.duplicated().any()


def test_prefix_invariance():
    assert_prefix_invariance(*_frames(), prefix_rows=3)


def test_historical_live_builder_parity():
    aligned, setups, decisions = _frames()
    historical = build_prediction_dataset(aligned, setups, decisions).tail(1).reset_index(drop=True)
    live = build_live_record(
        aligned.tail(1), setups.tail(1), decisions.tail(1)
    )
    pd.testing.assert_frame_equal(historical, live)
