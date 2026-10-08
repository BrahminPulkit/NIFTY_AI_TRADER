import pandas as pd

from src.strategy_decision_engine import build_walk_forward_decisions


def _observations(rows=80):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=rows, freq="30min", tz="Asia/Kolkata"
    )
    return pd.DataFrame({
        "timestamp": timestamp,
        "outcome_available_timestamp": timestamp + pd.Timedelta(minutes=15),
        "strategy_family": ["MOMENTUM_BREAKOUT"] * rows,
        "state_signature": ["TREND_UP|NORMAL|OPENING_DRIVE|BREAKOUT"] * rows,
        "market_regime": ["TREND_UP|NORMAL"] * rows,
        "trend_regime": ["TREND_UP"] * rows,
        "volatility_regime": ["NORMAL"] * rows,
        "session_phase": ["OPENING_DRIVE"] * rows,
        "structure_context": ["BREAKOUT"] * rows,
        "premium_expansion_hit": [True] * rows,
        "terminal_premium_return": [0.02] * rows,
        "premium_mfe": [0.08] * rows,
        "premium_mae": [0.03] * rows,
        "holding_minutes": [5] * rows,
    })


def test_no_decision_uses_unmatured_current_outcome():
    data = _observations()
    result = build_walk_forward_decisions(data)
    assert result.iloc[0].history_count == 0
    assert result.iloc[0].recommendation == "SKIP"


def test_future_outcomes_do_not_change_past_decisions():
    data = _observations()
    first = build_walk_forward_decisions(data)
    changed = data.copy()
    changed.loc[60:, "terminal_premium_return"] = -1.0
    second = build_walk_forward_decisions(changed)
    pd.testing.assert_frame_equal(first.iloc[:60], second.iloc[:60])


def test_positive_mature_history_can_activate_existing_family():
    result = build_walk_forward_decisions(_observations())
    assert result.iloc[-1].activated_strategy == "MOMENTUM_BREAKOUT"
    assert result.iloc[-1].recommendation == "TRADE"
