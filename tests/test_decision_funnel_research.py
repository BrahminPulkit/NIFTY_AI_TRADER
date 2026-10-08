import pandas as pd

from src.decision_funnel_research import (
    build_decision_funnel,
    build_funnel_universe,
    counterfactual_scenarios,
)
from src.walk_forward_paper_trading import ReplayPolicy


POLICY = ReplayPolicy()


def _data():
    time = pd.date_range(
        "2024-01-02 10:00", periods=5, freq="20min", tz="Asia/Kolkata"
    )
    stage = pd.DataFrame({
        "timestamp": time,
        "entry_signal": [1] * 5,
        "trade_allowed": [0, 1, 1, 1, 1],
        "rejection_reason": ["WEAK", "OK", "OK", "OK", "OK"],
        "atr_14": [10.0] * 5,
    })
    prediction = pd.DataFrame({
        "timestamp": time[1:],
        "recommendation": ["SKIP", "TRADE", "TRADE", "TRADE"],
        "decision_reason": ["NO", "YES", "YES", "YES"],
        "current_strategy": ["BREAKOUT"] * 4,
        "market_state": ["TREND_UP"] * 4,
        "index_atr_14": [10.0] * 4,
        "history_count": [40] * 4,
        "activated_strategy": ["BREAKOUT"] * 4,
        "estimate_source": ["EXACT"] * 4,
        "terminal_return_95_lower": [0.01] * 4,
        "expansion_drawdown_ratio": [2.0] * 4,
    })
    joined = pd.DataFrame({
        "timestamp": time[1:],
        "catboost_predicted_probability": [0.95, 0.80, 0.95, 0.96],
        "terminal_premium_return": [0.1, 0.1, -0.1, 0.2],
        "premium_expansion": [0.2] * 4,
        "premium_drawdown": [0.05] * 4,
        "fold_id": [1] * 4,
        "true_label": [1, 1, 0, 1],
        "market_regime": ["TREND_UP"] * 4,
        "strategy_family": ["BREAKOUT"] * 4,
        "session_phase": ["MID_SESSION"] * 4,
    })
    return stage, prediction, joined


def test_funnel_reconciles_at_every_gate():
    universe = build_funnel_universe(*_data(), POLICY)
    funnel = build_decision_funnel(
        universe, {universe.timestamp.iloc[-1]}, POLICY
    )
    assert (
        funnel.total_rejected + funnel.total_passed
        == funnel.total_incoming
    ).all()
    assert funnel.set_index("gate").loc["STAGE1", "total_rejected"] == 1
    assert (
        funnel.set_index("gate")
        .loc["PROBABILITY_THRESHOLD", "total_rejected"] == 1
    )


def test_first_failure_attribution_is_exclusive():
    universe = build_funnel_universe(*_data(), POLICY)
    assert len(universe) == 5
    assert universe.first_failure_gate.value_counts().sum() == 5
    assert universe.first_failure_gate.iloc[0] == "STAGE1"
    assert universe.first_failure_gate.iloc[1] == "DECISION_ENGINE"
    assert universe.first_failure_gate.iloc[2] == "PROBABILITY_THRESHOLD"


def test_counterfactuals_are_deterministic_and_stage1_is_unevaluable():
    universe = build_funnel_universe(*_data(), POLICY)
    one = counterfactual_scenarios(universe, POLICY)
    two = counterfactual_scenarios(universe, POLICY)
    assert one["REMOVE_STAGE1"][0] is None
    for name in one:
        left, right = one[name][0], two[name][0]
        if left is not None:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
