import numpy as np
import pandas as pd

from src.multi_timeframe_context_research import (
    attach_context,
    build_completed_15m_context,
)


def _minutes(periods=75):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=periods, freq="min", tz="Asia/Kolkata"
    )
    close = 100 + np.arange(periods) * 0.01
    return pd.DataFrame({
        "timestamp": timestamp, "open": close - 0.005, "high": close + 0.02,
        "low": close - 0.02, "close": close, "volume": np.arange(periods) + 1,
    })


def test_context_uses_only_completed_bars():
    context = build_completed_15m_context(_minutes())
    assert context.context_timestamp.min() == pd.Timestamp(
        "2025-01-02 09:30", tz="Asia/Kolkata"
    )


def test_context_prefix_invariance():
    short = build_completed_15m_context(_minutes(60))
    long = build_completed_15m_context(_minutes(75))
    pd.testing.assert_frame_equal(
        short.reset_index(drop=True),
        long.loc[long.context_timestamp.isin(short.context_timestamp)].reset_index(drop=True),
    )


def test_attached_context_is_not_from_future():
    raw = _minutes()
    raw["entry_signal"] = 0
    raw["trade_allowed"] = 0
    raw["breakout_flag"] = 0
    raw["breakdown_flag"] = 0
    raw["candle_body"] = raw.close - raw.open
    raw["upper_wick"] = raw.high - raw[["open", "close"]].max(axis=1)
    raw["lower_wick"] = raw[["open", "close"]].min(axis=1) - raw.low
    raw["candle_range"] = raw.high - raw.low
    raw.loc[30, ["entry_signal", "trade_allowed", "breakout_flag"]] = [1, 1, 1]
    outcomes = pd.DataFrame({
        "timestamp": [raw.loc[30, "timestamp"]], "trade_allowed": [1],
        "final_outcome": ["TIMEOUT"], "realized_r": [0.0],
        "direction": ["LONG"], "year": [2025], "session_segment": ["OPENING"],
    })
    result = attach_context(raw, outcomes, build_completed_15m_context(raw))
    assert (result.context_timestamp <= result.timestamp).all()
    assert result.loc[0, "confirm_1m_breakout"] == 1
