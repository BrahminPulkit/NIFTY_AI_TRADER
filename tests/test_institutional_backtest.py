import pandas as pd

from src.institutional_backtest import (
    CONFIG,
    monte_carlo,
    simulate_scenario,
    trading_costs,
)


def _data():
    timestamp = pd.to_datetime([
        "2025-01-02 09:15", "2025-01-02 09:20", "2025-01-02 09:40"
    ], utc=True).tz_convert("Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": timestamp,
        "catboost_predicted_probability": [0.9, 0.9, 0.9],
        "true_label": [1, 1, 0],
        "terminal_premium_return": [0.10, 0.10, -0.05],
        "holding_time_minutes": [10.0, 10.0, 10.0],
        "strategy_family": ["A"] * 3,
        "market_regime": ["UP"] * 3,
        "session_phase": ["MID"] * 3,
    })


def test_trading_costs_are_positive_and_reduce_pnl():
    costs = trading_costs(50_000, 0.10)
    assert costs["total_costs"] > 0
    assert 50_000 * 0.10 - costs["total_costs"] < 5_000


def test_overlapping_signal_is_not_executed():
    trades, skipped = simulate_scenario(_data(), "CATBOOST", 0.5)
    assert len(trades) == 2
    assert len(skipped) == 1
    assert skipped.iloc[0].skip_reason == "OVERLAPPING_ACTIVE_TRADE"
    assert (trades.entry_timestamp.iloc[1:] >= trades.exit_timestamp.iloc[:-1].to_numpy()).all()


def test_monte_carlo_is_deterministic_and_complete():
    trades, _ = simulate_scenario(_data(), "CATBOOST", 0.5)
    first = monte_carlo(trades, simulations=100, seed=7)
    second = monte_carlo(trades, simulations=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100
