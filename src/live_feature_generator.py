"""Exact V1 feature assembly for unseen synchronized Index and ATM CALL candles."""

from __future__ import annotations

import pandas as pd

from src.feature_pipeline import FEATURES, build_features
from src.option_feature_pipeline import OPTION_FEATURES, build_option_features
from src.setup_engine import OUTPUT_COLUMNS, SetupContract, build_setups
from src.strategy_decision_engine import build_walk_forward_decisions
from src.strategy_discovery import classify_entry_context


class LiveFeatureError(ValueError):
    pass


def _timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="raise").tz_convert("Asia/Kolkata")


def current_decision_observation(setup_history: pd.DataFrame,
                                 option_premium: float) -> dict:
    """Build the current context row without any future response fields."""
    entries = classify_entry_context(setup_history)
    if entries.empty:
        raise LiveFeatureError("Stage-1 produced no allowed current setup")
    current = entries.iloc[-1]
    if _timestamp(current.timestamp) != _timestamp(setup_history.timestamp.iloc[-1]):
        raise LiveFeatureError("Latest candle is not an allowed Stage-1 setup")
    session_phase = (
        "OPENING_DRIVE" if current.minutes_from_open < 60 else
        "LATE_TREND" if current.minutes_from_open >= 240 else "MID_SESSION")
    structure = "PULLBACK" if current.pullback_state == "PULLBACK" else "BREAKOUT"
    signature = f"{current.market_regime}|{session_phase}|{structure}"
    return {
        "timestamp": current.timestamp, "entry_signal": current.entry_signal,
        "strategy_family": current.strategy_family,
        "market_regime": current.market_regime, "trend_regime": current.trend_regime,
        "volatility_regime": current.volatility_regime,
        "session_phase": session_phase, "structure_context": structure,
        "state_signature": signature, "entry_premium": float(option_premium),
        "outcome_available_timestamp": _timestamp(current.timestamp) + pd.Timedelta(days=36500),
        "premium_expansion_hit": False, "terminal_premium_return": 0.0,
        "premium_mfe": 0.0, "premium_mae": 0.0, "holding_minutes": 0,
    }


def frozen_decision_at(observations: pd.DataFrame, current: dict) -> pd.Series:
    """Invoke the frozen Decision Engine with only outcomes matured by now."""
    now = _timestamp(current["timestamp"])
    history = observations.copy()
    history["timestamp"] = pd.to_datetime(history.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    history["outcome_available_timestamp"] = pd.to_datetime(
        history.outcome_available_timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    history = history.loc[
        (history.timestamp < now) & (history.outcome_available_timestamp <= now)]
    combined = pd.concat([history, pd.DataFrame([current])], ignore_index=True)
    decision = build_walk_forward_decisions(combined).iloc[-1]
    if _timestamp(decision.timestamp) != now:
        raise LiveFeatureError("Frozen Decision Engine did not return the current timestamp")
    return decision


def build_live_contract_row(
    index_history: pd.DataFrame,
    option_history: pd.DataFrame,
    decision_observations: pd.DataFrame,
    expected_order: list[str],
) -> tuple[pd.DataFrame | None, dict]:
    """Return an exact ordered model row, or a Stage-1 SKIP audit."""
    if index_history.empty or option_history.empty:
        raise LiveFeatureError("Both Index and ATM CALL histories are required")
    index_features = build_features(index_history)
    option_features = build_option_features(option_history)
    index_time, option_time = (_timestamp(index_features.timestamp.iloc[-1]),
                               _timestamp(option_features.timestamp.iloc[-1]))
    if index_time != option_time:
        raise LiveFeatureError("Latest Index and option candles are not timestamp-aligned")
    setups = build_setups(index_features, SetupContract())
    setup = setups.iloc[-1]
    if int(setup.trade_allowed) != 1:
        return None, {
            "timestamp": index_time, "stage1": "REJECTED", "decision": "SKIP",
            "reason": str(setup.rejection_reason), "entry_signal": int(setup.entry_signal),
        }
    observation = current_decision_observation(setups, float(option_features.close.iloc[-1]))
    decision = frozen_decision_at(decision_observations, observation)
    values = {}
    for name in FEATURES:
        if name != "vwap":  # all-null V1 training field was explicitly excluded
            values[f"index_{name}"] = index_features.iloc[-1][name]
    for name in OPTION_FEATURES:
        values[f"option_{name}"] = option_features.iloc[-1][name]
    for name in OUTPUT_COLUMNS:
        if name not in {"trade_allowed", "rejection_reason"}:
            values[name] = setup[name]
    for name in (
        "market_state", "trend_regime", "volatility_regime", "session_phase",
        "structure_context", "current_strategy", "activated_strategy",
        "estimate_source", "history_count", "premium_expansion_probability",
        "expected_premium_expansion", "expected_drawdown",
        "expected_holding_minutes", "expected_terminal_return",
        "terminal_return_standard_error", "terminal_return_95_lower",
        "expansion_drawdown_ratio", "strategy_confidence_score",
        "recommendation", "decision_reason",
    ):
        values[name] = decision[name]
    missing = [name for name in expected_order if name not in values]
    unknown = [name for name in values if name not in expected_order]
    if missing or unknown:
        raise LiveFeatureError(f"Contract assembly mismatch missing={missing} unknown={unknown}")
    return pd.DataFrame([[values[name] for name in expected_order]],
                        columns=expected_order), {
        "timestamp": index_time, "stage1": "ALLOWED",
        "decision_engine": str(decision.recommendation),
        "decision_reason": str(decision.decision_reason),
        "strategy": str(decision.current_strategy),
        "regime": str(decision.market_state),
    }

