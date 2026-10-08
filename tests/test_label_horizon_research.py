import numpy as np
import pandas as pd

from src.label_horizon_research import evaluate_direction, summarize


def fixture(highs, lows, closes=None):
    size = len(highs)
    closes = closes or [100.0] * size
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-02 09:15", periods=size, freq="min", tz="Asia/Kolkata"),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "ema_20": [99.0] * size, "ema_50": [98.0] * size,
        "minutes_from_open": np.arange(size),
    })


def test_correct_target_stop_and_horizon_calculation():
    data = fixture([100, 100.1, 100.31, 100, 100, 100], [100, 99.9, 99.9, 100, 100, 100])
    result = evaluate_direction(data, 5, 1)
    assert result.iloc[0].outcome == "WIN"
    assert result.iloc[0].holding_candles == 2
    assert np.isclose(result.iloc[0].realized_r, 1.5)


def test_ambiguous_bar_is_conservative_stop_first():
    data = fixture([100, 100.4, 100], [100, 99.7, 100])
    result = evaluate_direction(data, 2, 1)
    assert result.iloc[0].outcome == "STOP"
    assert np.isclose(result.iloc[0].realized_r, -1.0)


def test_mfe_mae_calculations_for_long_and_short():
    data = fixture([100, 100.2, 100.1], [100, 99.9, 99.8])
    long = evaluate_direction(data, 2, 1).iloc[0]
    short = evaluate_direction(data, 2, -1).iloc[0]
    assert np.isclose(long.mfe_r, 1.0) and np.isclose(long.mae_r, 1.0)
    assert np.isclose(short.mfe_r, 1.0) and np.isclose(short.mae_r, 1.0)


def test_expected_r_is_mean_realized_r():
    data = fixture([100, 100.31, 100, 100], [100, 100, 99.7, 100])
    observations = evaluate_direction(data, 1, 1)
    stats = summarize(observations, ["horizon", "direction"])
    assert np.isclose(stats.expected_r.iloc[0], observations.realized_r.mean())
    expected_frequency = observations.outcome.ne("TIMEOUT").sum()
    assert np.isclose(stats.trade_frequency_per_day.iloc[0], expected_frequency)


def test_no_future_beyond_horizon_leakage():
    data = fixture([100] * 8, [100] * 8)
    original = evaluate_direction(data, 3, 1).iloc[0]
    mutated = data.copy()
    mutated.loc[4:, "high"] = 1000
    changed = evaluate_direction(mutated, 3, 1).iloc[0]
    for field in ["outcome", "holding_candles", "mfe_r", "mae_r", "realized_r"]:
        left, right = original[field], changed[field]
        assert (left == right) or (pd.isna(left) and pd.isna(right))


def test_labels_never_cross_session():
    data = fixture([100, 100, 100.5], [100, 100, 100])
    data.loc[2, "timestamp"] = pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    result = evaluate_direction(data, 5, 1)
    assert result.iloc[0].outcome == "TIMEOUT"
    assert result.iloc[0].holding_candles == 1
