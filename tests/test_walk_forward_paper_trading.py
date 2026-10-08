import pandas as pd
import pytest

from src.institutional_backtest import PortfolioConfig, trading_costs
from src.walk_forward_paper_trading import (
    ReplayPolicy,
    replay_frozen_stream,
)


def _fixtures():
    timestamps = pd.date_range(
        "2024-01-02 09:15", periods=40, freq="min", tz="Asia/Kolkata"
    )
    stage = pd.DataFrame({
        "timestamp": timestamps,
        "entry_signal": 0,
        "trade_allowed": 0,
        "rejection_reason": "NO_SETUP",
        "session_id": 20240102,
    })
    for index in (1, 5, 20):
        stage.loc[index, ["entry_signal", "trade_allowed"]] = [1, 1]
    prediction = pd.DataFrame({
        "timestamp": timestamps[[1, 5, 20]],
        "recommendation": ["TRADE", "TRADE", "TRADE"],
        "decision_reason": ["ACTIVE_STRATEGY_MATCH"] * 3,
        "current_strategy": ["MOMENTUM_BREAKOUT"] * 3,
        "market_state": ["TREND_UP|NORMAL"] * 3,
    })
    joined = pd.DataFrame({
        "timestamp": timestamps[[1, 5, 20]],
        "catboost_predicted_probability": [0.95, 0.96, 0.95],
        "terminal_premium_return": [0.10, 0.20, -0.05],
        "true_label": [1, 1, 0],
        "fold_id": [1, 1, 1],
        "premium_expansion": [0.12, 0.22, 0.01],
        "premium_drawdown": [0.02, 0.01, 0.08],
        "strategy_family": ["MOMENTUM_BREAKOUT"] * 3,
        "market_regime": ["TREND_UP|NORMAL"] * 3,
        "session_phase": ["OPENING_DRIVE"] * 3,
    })
    return stage, prediction, joined


def test_replay_is_deterministic_and_chronological():
    inputs = _fixtures()
    one = replay_frozen_stream(*inputs)
    two = replay_frozen_stream(*inputs)
    for left, right in zip(one, two):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    trades, skipped, _ = one
    assert trades.entry_timestamp.is_monotonic_increasing
    assert trades.entry_timestamp.is_unique
    assert (trades.exit_timestamp >= trades.scheduled_exit_timestamp).all()
    assert "CAPITAL_UNAVAILABLE" in set(skipped.skip_category)


def test_future_response_is_not_settled_before_horizon():
    stage, prediction, joined = _fixtures()
    truncated = stage.iloc[:10]
    trades, skipped, events = replay_frozen_stream(
        truncated, prediction, joined
    )
    assert trades.empty
    assert events.event.tolist() == ["INITIAL_CAPITAL"]
    assert "DATASET_ENDED_BEFORE_EXIT" in set(skipped.skip_reason)


def test_capital_and_cost_accounting():
    stage, prediction, joined = _fixtures()
    trades, _, _ = replay_frozen_stream(stage, prediction, joined)
    assert len(trades) == 2
    assert trades.iloc[0].capital_after == pytest.approx(
        trades.iloc[0].capital_before + trades.iloc[0].net_pnl
    )
    assert trades.iloc[1].capital_before == pytest.approx(
        trades.iloc[0].capital_after
    )
    expected = trading_costs(
        trades.iloc[0].position_notional,
        trades.iloc[0].terminal_premium_return,
    )
    assert trades.iloc[0].total_costs == pytest.approx(expected["total_costs"])
    assert trades.total_costs.gt(0).all()


def test_probability_and_decision_filters_are_frozen():
    stage, prediction, joined = _fixtures()
    prediction.loc[prediction.index[2], "recommendation"] = "SKIP"
    joined.loc[joined.index[:2], "catboost_predicted_probability"] = 0.89
    trades, skipped, _ = replay_frozen_stream(
        stage, prediction, joined, ReplayPolicy(threshold=0.90)
    )
    assert trades.empty
    assert {
        "PROBABILITY_BELOW_THRESHOLD", "DECISION_ENGINE_REJECTED",
    }.issubset(set(skipped.skip_category))


def test_input_timestamp_duplicates_are_rejected():
    stage, prediction, joined = _fixtures()
    stage.loc[1, "timestamp"] = stage.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="unique"):
        replay_frozen_stream(stage, prediction, joined)
