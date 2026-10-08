"""Canonical Stage-1 deterministic setup engine.

The engine consumes only the canonical Step-15 feature contract. It produces
candidate setups and filter decisions, never outcome labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SetupContract:
    version: str = "stage1_v2"
    institutional_refinement: bool = True
    bullish_rsi_min: float = 55.0
    bearish_rsi_max: float = 45.0
    minimum_atr_fraction: float = 0.00015
    maximum_atr_fraction: float = 0.00250
    first_allowed_minute: int = 15       # 09:30
    last_allowed_minute: int = 344       # 14:59
    bullish_rsi_max: float = 72.0
    bearish_rsi_min: float = 28.0
    minimum_breakout_atr: float = 0.25
    minimum_body_fraction: float = 0.50
    minimum_trend_separation_atr: float = 0.50

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def v1_baseline(cls) -> "SetupContract":
        return cls(version="stage1_v1", institutional_refinement=False)


AUDIT_COLUMNS = ["trend_state", "structure_state", "momentum_state",
                 "volatility_state", "session_state", "event_state",
                 "breakout_quality_state", "candle_quality_state",
                 "trend_strength_state", "rejection_reason"]
OUTPUT_COLUMNS = ["entry_signal", "trade_allowed", *AUDIT_COLUMNS]


def _validate(frame: pd.DataFrame) -> None:
    required = {"timestamp", "close", "ema_5", "ema_9", "ema_20", "ema_50", "rsi_14",
                "atr_14", "rolling_high", "rolling_low", "breakout_flag", "breakdown_flag",
                "candle_body", "candle_range", "minutes_from_open", "session_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical setup inputs: {missing}")
    timestamp = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("Setup input timestamps must be sorted and unique")


def build_setups(frame: pd.DataFrame, contract: SetupContract = SetupContract()) -> pd.DataFrame:
    """Add deterministic Stage-1 setup decisions and audit fields."""
    _validate(frame)
    output = frame.copy()

    uptrend = ((output.close > output.ema_5) & (output.ema_5 > output.ema_9) &
               (output.ema_9 > output.ema_20) & (output.ema_20 > output.ema_50))
    downtrend = ((output.close < output.ema_5) & (output.ema_5 < output.ema_9) &
                 (output.ema_9 < output.ema_20) & (output.ema_20 < output.ema_50))
    output["trend_state"] = np.select([uptrend, downtrend], ["UPTREND", "DOWNTREND"],
                                      default="NEUTRAL")

    bullish_breakout = output.breakout_flag.eq(1)
    bearish_breakdown = output.breakdown_flag.eq(1)
    output["structure_state"] = np.select(
        [bullish_breakout & bearish_breakdown, bullish_breakout, bearish_breakdown],
        ["DUAL_BREAK", "BULLISH_BREAKOUT", "BEARISH_BREAKDOWN"], default="RANGE")

    bullish_momentum = output.rsi_14.ge(contract.bullish_rsi_min)
    bearish_momentum = output.rsi_14.le(contract.bearish_rsi_max)
    if contract.institutional_refinement:
        bullish_momentum &= output.rsi_14.le(contract.bullish_rsi_max)
        bearish_momentum &= output.rsi_14.ge(contract.bearish_rsi_min)
    unavailable_momentum = output.rsi_14.isna()
    output["momentum_state"] = np.select(
        [unavailable_momentum, bullish_momentum, bearish_momentum],
        ["UNAVAILABLE", "BULLISH", "BEARISH"], default="NEUTRAL")

    atr_fraction = output.atr_14 / output.close
    output["volatility_state"] = np.select(
        [atr_fraction.isna(), atr_fraction < contract.minimum_atr_fraction,
         atr_fraction > contract.maximum_atr_fraction],
        ["UNAVAILABLE", "TOO_LOW", "TOO_HIGH"], default="SUITABLE")

    minute = output.minutes_from_open
    output["session_state"] = np.select(
        [minute < contract.first_allowed_minute, minute > contract.last_allowed_minute],
        ["OPENING_BLOCKED", "CLOSING_BLOCKED"], default="APPROVED")

    long_setup = uptrend & bullish_breakout & ~bearish_breakdown
    short_setup = downtrend & bearish_breakdown & ~bullish_breakout
    output["entry_signal"] = np.select([long_setup, short_setup], [1, -1], default=0).astype("int8")

    same_session = output.session_id.eq(output.session_id.shift(1))
    prior_bullish = output.breakout_flag.shift(1, fill_value=0).eq(1) & same_session
    prior_bearish = output.breakdown_flag.shift(1, fill_value=0).eq(1) & same_session
    new_event = ((long_setup & ~prior_bullish) | (short_setup & ~prior_bearish))
    output["event_state"] = np.select(
        [~output.entry_signal.ne(0), new_event], ["NO_SETUP", "NEW_EVENT"], default="REPEATED_EVENT")

    directional_distance = np.where(long_setup, output.close - output.rolling_high,
                                    output.rolling_low - output.close)
    breakout_strength = pd.Series(directional_distance, index=output.index) / output.atr_14
    close_confirmed = ((long_setup & (output.close > output.rolling_high)) |
                       (short_setup & (output.close < output.rolling_low)))
    breakout_quality = close_confirmed & breakout_strength.ge(contract.minimum_breakout_atr)
    output["breakout_quality_state"] = np.select(
        [~output.entry_signal.ne(0), breakout_quality, close_confirmed],
        ["NO_SETUP", "STRONG_CONFIRMED", "WEAK_CONFIRMED"], default="WICK_ONLY")

    body_fraction = output.candle_body.abs() / output.candle_range.replace(0, np.nan)
    directional_body = ((long_setup & output.candle_body.gt(0)) |
                        (short_setup & output.candle_body.lt(0)))
    candle_quality = directional_body & body_fraction.ge(contract.minimum_body_fraction)
    output["candle_quality_state"] = np.select(
        [~output.entry_signal.ne(0), candle_quality, directional_body],
        ["NO_SETUP", "STRONG_DIRECTIONAL_BODY", "WEAK_BODY"], default="OPPOSING_BODY")

    trend_separation = (output.ema_5 - output.ema_50).abs() / output.atr_14
    trend_quality = trend_separation.ge(contract.minimum_trend_separation_atr)
    output["trend_strength_state"] = np.select(
        [~output.entry_signal.ne(0), trend_quality], ["NO_SETUP", "STRONG"], default="WEAK")

    momentum_pass = ((long_setup & bullish_momentum) | (short_setup & bearish_momentum))
    volatility_pass = output.volatility_state.eq("SUITABLE")
    session_pass = output.session_state.eq("APPROVED")
    has_setup = output.entry_signal.ne(0)
    institutional_pass = pd.Series(True, index=output.index)
    if contract.institutional_refinement:
        institutional_pass = new_event & breakout_quality & candle_quality & trend_quality
    output["trade_allowed"] = (
        has_setup & momentum_pass & volatility_pass & session_pass & institutional_pass).astype("int8")

    reasons: list[str] = []
    for index in output.index:
        if not has_setup.at[index]:
            reasons.append("NO_SETUP")
            continue
        rejected: list[str] = []
        if not momentum_pass.at[index]:
            rejected.append("MOMENTUM_REJECTED")
        volatility = output.at[index, "volatility_state"]
        if volatility != "SUITABLE":
            rejected.append(f"VOLATILITY_{volatility}")
        session = output.at[index, "session_state"]
        if session != "APPROVED":
            rejected.append(session)
        if contract.institutional_refinement:
            if not new_event.at[index]:
                rejected.append("REPEATED_EVENT")
            if not breakout_quality.at[index]:
                rejected.append("BREAKOUT_NOT_STRONG_CONFIRMED")
            if not candle_quality.at[index]:
                rejected.append("CANDLE_QUALITY_REJECTED")
            if not trend_quality.at[index]:
                rejected.append("TREND_SEPARATION_WEAK")
        reasons.append("ALLOWED" if not rejected else "|".join(rejected))
    output["rejection_reason"] = reasons
    return output


def build_live_setup(feature_history: pd.DataFrame,
                     contract: SetupContract = SetupContract()) -> pd.Series:
    if feature_history.empty:
        raise ValueError("Feature history cannot be empty")
    return build_setups(feature_history, contract).iloc[-1]
