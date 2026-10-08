import numpy as np
import pandas as pd

from src.production_horizon_research import evaluate_allowed_setups


def fixture():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-02 09:30", periods=8, freq="min", tz="Asia/Kolkata"),
        "open": [100]*8, "high": [100, 100.1, 100.31, 100, 100, 100, 100, 100],
        "low": [100, 99.9, 99.9, 100, 100, 100, 100, 100], "close": [100]*8,
        "entry_signal": [1, -1, 0, 0, 0, 0, 0, 0], "trade_allowed": [1, 0, 0, 0, 0, 0, 0, 0],
        "trend_state": ["UPTREND"]*8, "minutes_from_open": np.arange(15, 23),
        "session_id": [20260102]*8,
    })


def test_only_allowed_direction_is_researched():
    result = evaluate_allowed_setups(fixture(), 5)
    assert len(result) == 1 and result.iloc[0].direction == "BUY_CALL"
    assert result.iloc[0].outcome == "WIN" and result.iloc[0].holding_candles == 2


def test_mfe_mae_and_expected_r_are_correct():
    row = evaluate_allowed_setups(fixture(), 5).iloc[0]
    assert np.isclose(row.mfe_r, 1.55)
    assert np.isclose(row.mae_r, 0.5)
    assert np.isclose(row.realized_r, 1.5)


def test_no_future_beyond_horizon_changes_result():
    data = fixture()
    before = evaluate_allowed_setups(data, 3).iloc[0]
    data.loc[4:, ["high", "low"]] = [1000, 1]
    after = evaluate_allowed_setups(data, 3).iloc[0]
    for field in ["outcome", "holding_candles", "mfe_r", "mae_r", "realized_r"]:
        assert before[field] == after[field]


def test_stop_first_and_no_cross_session():
    data = fixture(); data.loc[1, ["high", "low"]] = [100.4, 99.7]
    assert evaluate_allowed_setups(data, 5).iloc[0].outcome == "STOP"
    data = fixture(); data.loc[1:, "session_id"] = 20260105
    data.loc[1:, "timestamp"] = pd.date_range("2026-01-05 09:15", periods=7, freq="min", tz="Asia/Kolkata")
    assert evaluate_allowed_setups(data, 5).empty
