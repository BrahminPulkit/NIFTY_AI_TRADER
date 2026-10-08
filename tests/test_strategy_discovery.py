import numpy as np
import pandas as pd

from src.strategy_discovery import (
    classify_entry_context,
    measure_premium_behaviour,
)


def _setups(periods=140):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=periods, freq="min", tz="Asia/Kolkata"
    )
    close = 100 + np.arange(periods) * 0.01
    data = pd.DataFrame({
        "timestamp": timestamp, "entry_signal": 0, "trade_allowed": 0,
        "open": close - 0.01, "high": close + 0.05, "low": close - 0.05,
        "close": close, "candle_range": 0.1, "atr_14": 0.08,
        "rolling_volatility": 0.001, "rolling_range": 1.0,
        "ema_5": close - 0.01, "ema_9": close - 0.02,
        "ema_20": close - 0.03, "ema_50": close - 0.04,
        "breakout_flag": 0, "breakdown_flag": 0,
    })
    data.loc[120, ["entry_signal", "trade_allowed", "breakout_flag"]] = [1, 1, 1]
    return data


def _aligned(setups):
    return pd.DataFrame({
        "timestamp": setups.timestamp, "option_close": [100.0] * len(setups),
        "option_high": [105.0] * len(setups), "option_low": [98.0] * len(setups),
        "option_volume": [10] * len(setups),
        "index_session_id": ["20250102"] * len(setups),
    })


def test_classification_does_not_change_frozen_decisions():
    setups = _setups()
    result = classify_entry_context(setups)
    assert len(result) == 1
    assert result.iloc[0].entry_signal == 1
    assert result.iloc[0].trade_allowed == 1


def test_premium_measurement_is_prefix_invariant():
    setups = _setups()
    entries = classify_entry_context(setups)
    aligned = _aligned(setups)
    short = measure_premium_behaviour(aligned.iloc[:136], entries, horizons=(15,))
    long = measure_premium_behaviour(aligned, entries, horizons=(15,))
    pd.testing.assert_frame_equal(short, long)


def test_future_change_after_horizon_does_not_change_result():
    setups = _setups()
    entries = classify_entry_context(setups)
    aligned = _aligned(setups)
    first = measure_premium_behaviour(aligned, entries, horizons=(15,))
    changed = aligned.copy()
    changed.loc[139, "option_high"] = 1000.0
    second = measure_premium_behaviour(changed, entries, horizons=(15,))
    pd.testing.assert_frame_equal(first, second)
