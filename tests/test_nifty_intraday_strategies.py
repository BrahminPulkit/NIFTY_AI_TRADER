import numpy as np
import pandas as pd

from src.nifty_intraday_strategies import (
    IntradayConfig,
    backtest_candidates,
    latest_live_candidates,
    prepare_intraday_data,
    strategy_summary,
)


def sample_bars(periods=240):
    timestamps = pd.date_range("2026-07-27 09:15", periods=periods, freq="min", tz="Asia/Kolkata")
    close = 25000 + np.linspace(0, 120, periods) + np.sin(np.arange(periods) / 4)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.arange(periods) + 100,
        }
    )


def test_completed_timeframes_do_not_leak_future_candles():
    features = prepare_intraday_data(sample_bars())
    before = features.loc[features.timestamp.dt.strftime("%H:%M") == "09:29"].iloc[0]
    at_close = features.loc[features.timestamp.dt.strftime("%H:%M") == "09:30"].iloc[0]
    assert pd.isna(before["close_15m"])
    assert at_close["close_15m"] == features.iloc[14]["close"]


def test_futures_volume_replaces_index_placeholder_volume():
    index = sample_bars()
    index["volume"] = 0
    futures = sample_bars()[["timestamp", "volume"]]
    features = prepare_intraday_data(index, futures)
    assert "volume" in features
    assert "volume_x" not in features
    assert "volume_y" not in features
    assert features["volume"].iloc[0] == futures["volume"].iloc[0]


def test_chop_features_are_causal_and_available():
    bars = sample_bars()
    features = prepare_intraday_data(bars)
    assert {"ema9_5m", "ema20_5m", "ema9_15m", "ema20_15m",
            "adx", "vwap_crosses_15m"}.issubset(features.columns)
    first = features.iloc[:15]["vwap_crosses_15m"]
    assert first.iloc[:-1].isna().all()
    assert pd.notna(first.iloc[-1])


def test_backtest_is_stop_first_when_both_levels_touch():
    bars = sample_bars(5)
    entry_time = bars.iloc[1].timestamp
    bars.loc[1, ["high", "low"]] = [110, 90]
    candidate = pd.DataFrame(
        [{
            "strategy": "ORB_RETEST", "signal_time": bars.iloc[0].timestamp,
            "entry_time": entry_time, "direction": 1, "side": "BUY CE",
            "entry": 100.0, "stop": 95.0, "target": 105.0,
            "risk_points": 5.0, "atr": 5.0, "adx": 25.0,
        }]
    )
    result = backtest_candidates(
        bars, candidate, IntradayConfig(cost_points_per_trade=0)
    )
    assert result.iloc[0].outcome == "LOSS"
    assert result.iloc[0].gross_points == -5


def test_scalp_partial_then_breakeven_books_half_r():
    bars = sample_bars(5)
    bars.loc[1, ["open", "high", "low", "close"]] = [100, 106, 99, 104]
    bars.loc[2, ["open", "high", "low", "close"]] = [104, 104, 99, 100]
    candidate = pd.DataFrame(
        [{
            "strategy": "GOLDEN_RULE", "signal_time": bars.iloc[0].timestamp,
            "entry_time": bars.iloc[1].timestamp, "direction": 1, "side": "BUY CE",
            "entry": 100.0, "stop": 95.0, "partial_target": 105.0,
            "target": 109.0, "risk_points": 5.0, "atr": 5.0, "adx": 25.0,
        }]
    )
    result = backtest_candidates(
        bars, candidate, IntradayConfig(cost_points_per_trade=0)
    )
    assert result.iloc[0].outcome == "BREAKEVEN"
    assert result.iloc[0].partial_1r_hit
    assert result.iloc[0].gross_points == 2.5
    assert result.iloc[0].result_r == 0.5


def test_summary_compares_strategies():
    trades = pd.DataFrame(
        {
            "strategy": ["GOLDEN_RULE", "GOLDEN_RULE", "ORB_RETEST"],
            "outcome": ["WIN", "LOSS", "WIN"],
            "net_points": [10, -5, 8],
            "result_r": [2, -1, 1.5],
        }
    )
    summary = strategy_summary(trades)
    assert set(summary.strategy) == {"GOLDEN_RULE", "ORB_RETEST"}
    assert summary.loc[summary.strategy == "GOLDEN_RULE", "trades"].iloc[0] == 2


def test_live_candidates_reject_stale_session():
    features = prepare_intraday_data(sample_bars())
    result = latest_live_candidates(
        features, 25100, now="2026-07-28 10:00+05:30"
    )
    assert result.empty
