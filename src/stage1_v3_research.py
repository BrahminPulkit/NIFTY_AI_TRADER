"""Independent, causal Stage-1 V3 setup-family research.

This module does not import, wrap, or modify the frozen Stage-1 V2 engine.  It
consumes the canonical Step-15 feature contract and emits research candidates.
Outcome evaluation delegates to the existing label-contract research matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.label_contract_research import ContractSpec, OpportunityMatrix, summarize_contract


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    description: str


BENCHMARK_CONTRACTS = (
    ContractSpec("TRAIL_H15_ATR0.50", "ATR_TRAILING", 15, 0.50, None, "TRAILING"),
    ContractSpec("ATR_H15_SL0.75_RR1.25", "ATR", 15, 0.75, 1.25),
    ContractSpec("TIME_H15_ATR_RISK", "TIME_ATR", 15, 1.00, None, "TIME"),
)

REQUIRED = {
    "timestamp", "open", "high", "low", "close", "candle_body", "candle_range",
    "ema_5", "ema_9", "ema_20", "ema_50", "rsi_14", "atr_14",
    "rolling_volatility", "rolling_high", "rolling_low", "breakout_flag",
    "breakdown_flag", "previous_high", "previous_low", "minutes_from_open", "session_id",
}


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical V3 research inputs: {missing}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    if result.timestamp.duplicated().any() or not result.timestamp.is_monotonic_increasing:
        raise ValueError("Research input timestamps must be sorted and unique")
    return result


def _completed_timeframe_state(frame: pd.DataFrame, minutes: int) -> tuple[pd.Series, pd.Series]:
    """Trend from the previous fully completed intraday aggregation bucket."""
    bucket = (frame.minutes_from_open // minutes).astype(int)
    bars = frame.groupby(["session_id", bucket], sort=False).close.last().rename("bar_close").reset_index()
    bucket_name = bars.columns[1]
    fast = bars.groupby("session_id", sort=False).bar_close.transform(
        lambda value: value.ewm(span=2, adjust=False, min_periods=2).mean())
    slow = bars.groupby("session_id", sort=False).bar_close.transform(
        lambda value: value.ewm(span=5, adjust=False, min_periods=5).mean())
    state = np.sign(fast - slow)
    # The current bucket is incomplete at a minute-level decision. Only expose
    # the preceding completed bucket's state.
    bars["state"] = pd.Series(state, index=bars.index).groupby(bars.session_id, sort=False).shift(1)
    mapping = bars.set_index(["session_id", bucket_name]).state
    keys = pd.MultiIndex.from_arrays([frame.session_id, bucket])
    aligned = pd.Series(mapping.reindex(keys).to_numpy(), index=frame.index)
    return aligned.gt(0), aligned.lt(0)


def _recent_event(flag: pd.Series, session: pd.Series, lookback: int) -> pd.Series:
    result = pd.Series(False, index=flag.index)
    for offset in range(1, lookback + 1):
        result |= flag.shift(offset, fill_value=0).eq(1) & session.eq(session.shift(offset))
    return result


def _eventize(long_raw: pd.Series, short_raw: pd.Series, session: pd.Series,
              cooldown: int = 10) -> pd.Series:
    raw = np.select([long_raw, short_raw], [1, -1], default=0).astype("int8")
    accepted = np.zeros(len(raw), dtype="int8")
    session_values = session.to_numpy()
    last_session = None
    last_entry = -10_000
    previous_raw = 0
    for position, signal in enumerate(raw):
        current_session = session_values[position]
        if current_session != last_session:
            last_session, last_entry, previous_raw = current_session, -10_000, 0
        is_new_event = signal != 0 and signal != previous_raw
        if is_new_event and position - last_entry >= cooldown:
            accepted[position] = signal
            last_entry = position
        previous_raw = signal
    return pd.Series(accepted, index=long_raw.index, dtype="int8")


def build_candidate_setups(frame: pd.DataFrame) -> dict[str, tuple[Candidate, pd.DataFrame]]:
    """Return independent candidate decisions using information available at t."""
    data = _validate(frame)
    atr = data.atr_14.replace(0, np.nan)
    body_fraction = data.candle_body.abs() / data.candle_range.replace(0, np.nan)
    up = (data.ema_5 > data.ema_9) & (data.ema_9 > data.ema_20) & (data.ema_20 > data.ema_50)
    down = (data.ema_5 < data.ema_9) & (data.ema_9 < data.ema_20) & (data.ema_20 < data.ema_50)
    bullish_body, bearish_body = data.candle_body.gt(0), data.candle_body.lt(0)
    session_ok = data.minutes_from_open.between(15, 344)
    volatility_ok = (atr / data.close).between(0.00015, 0.0025)
    common = session_ok & volatility_ok
    breakout = data.breakout_flag.eq(1) & data.close.gt(data.rolling_high)
    breakdown = data.breakdown_flag.eq(1) & data.close.lt(data.rolling_low)
    trend_distance = (data.ema_5 - data.ema_50).abs() / atr
    ema9_distance = (data.close - data.ema_9).abs() / atr
    atr_baseline = data.atr_14.shift(1).rolling(20, min_periods=20).mean()
    atr_expansion = data.atr_14 / atr_baseline
    vol_baseline = data.rolling_volatility.shift(1).rolling(20, min_periods=20).mean()
    prior_contraction = data.rolling_volatility.shift(1).lt(
        data.rolling_volatility.shift(2).rolling(20, min_periods=20).mean())
    volatility_expansion = data.rolling_volatility.gt(vol_baseline * 1.10)
    hh = data.previous_high.gt(data.previous_high.shift(5))
    ll = data.previous_low.lt(data.previous_low.shift(5))
    same_previous = data.session_id.eq(data.session_id.shift(1))
    reclaim9_long = same_previous & data.close.shift(1).le(data.ema_9.shift(1)) & data.close.gt(data.ema_9)
    reclaim9_short = same_previous & data.close.shift(1).ge(data.ema_9.shift(1)) & data.close.lt(data.ema_9)
    reclaim20_long = same_previous & data.low.le(data.ema_20) & data.close.gt(data.ema_20)
    reclaim20_short = same_previous & data.high.ge(data.ema_20) & data.close.lt(data.ema_20)
    recent_break = _recent_event(data.breakout_flag, data.session_id, 3)
    recent_down = _recent_event(data.breakdown_flag, data.session_id, 3)
    retest_long = recent_break & data.low.le(data.rolling_high) & data.close.gt(data.rolling_high)
    retest_short = recent_down & data.high.ge(data.rolling_low) & data.close.lt(data.rolling_low)
    mtf5_up, mtf5_down = _completed_timeframe_state(data, 5)
    mtf15_up, mtf15_down = _completed_timeframe_state(data, 15)

    definitions: list[tuple[Candidate, pd.Series, pd.Series]] = [
        (Candidate("EMA_STRICT_RSI55", "EMA_ALIGNMENT", "Strict EMA stack with bounded RSI and directional body"),
         up & data.rsi_14.between(55, 70) & bullish_body,
         down & data.rsi_14.between(30, 45) & bearish_body),
        (Candidate("EMA_DISTANCE_075", "EMA_ALIGNMENT", "Strict EMA stack separated by at least 0.75 ATR"),
         up & trend_distance.ge(.75) & data.rsi_14.between(54, 70),
         down & trend_distance.ge(.75) & data.rsi_14.between(30, 46)),
        (Candidate("PULLBACK_EMA9", "PULLBACK", "EMA9 reclaim within established EMA trend"),
         up & reclaim9_long & bullish_body & data.rsi_14.between(50, 68),
         down & reclaim9_short & bearish_body & data.rsi_14.between(32, 50)),
        (Candidate("PULLBACK_EMA20", "PULLBACK", "EMA20 touch and directional close in established trend"),
         up & reclaim20_long & bullish_body & data.rsi_14.between(48, 65),
         down & reclaim20_short & bearish_body & data.rsi_14.between(35, 52)),
        (Candidate("BREAKOUT_RETEST_3", "BREAKOUT_RETEST", "Retest within three candles of a structure break"),
         up & retest_long & bullish_body, down & retest_short & bearish_body),
        (Candidate("TREND_CONTINUATION", "TREND_CONTINUATION", "EMA9 reclaim plus strong established trend"),
         up & reclaim9_long & trend_distance.ge(.75) & data.rsi_14.between(52, 68),
         down & reclaim9_short & trend_distance.ge(.75) & data.rsi_14.between(32, 48)),
        (Candidate("BREAKOUT_ATR_EXPANSION", "ATR_EXPANSION", "Confirmed break with ATR above its prior baseline"),
         up & breakout & atr_expansion.ge(1.10), down & breakdown & atr_expansion.ge(1.10)),
        (Candidate("BREAKOUT_BODY_070", "BODY_QUALITY", "Confirmed break with 70% directional candle body"),
         up & breakout & bullish_body & body_fraction.ge(.70),
         down & breakdown & bearish_body & body_fraction.ge(.70)),
        (Candidate("SWING_STRUCTURE", "SWING_STRUCTURE", "Confirmed break with rising/falling five-candle swing reference"),
         up & breakout & hh, down & breakdown & ll),
        (Candidate("CONTRACTION_EXPANSION", "VOLATILITY_CYCLE", "Prior contraction followed by volatility expansion and break"),
         up & breakout & prior_contraction & volatility_expansion,
         down & breakdown & prior_contraction & volatility_expansion),
        (Candidate("MTF5_BREAKOUT", "MULTI_TIMEFRAME", "Confirmed break aligned with previous completed 5-minute trend"),
         up & breakout & mtf5_up, down & breakdown & mtf5_down),
        (Candidate("MTF15_CONTINUATION", "MULTI_TIMEFRAME", "EMA9 continuation aligned with previous completed 15-minute trend"),
         up & reclaim9_long & mtf15_up, down & reclaim9_short & mtf15_down),
        (Candidate("RSI_MODERATE_BREAKOUT", "RSI_VARIANT", "Confirmed break with moderate 52/48 RSI threshold"),
         up & breakout & data.rsi_14.between(52, 68),
         down & breakdown & data.rsi_14.between(32, 48)),
        (Candidate("RSI_STRONG_BREAKOUT", "RSI_VARIANT", "Confirmed break with strong 58/42 RSI threshold"),
         up & breakout & data.rsi_14.between(58, 72),
         down & breakdown & data.rsi_14.between(28, 42)),
        (Candidate("EMA9_PROXIMITY_PULLBACK", "EMA_DISTANCE", "Trend pullback within 0.25 ATR of EMA9"),
         up & ema9_distance.le(.25) & bullish_body & data.rsi_14.between(50, 65),
         down & ema9_distance.le(.25) & bearish_body & data.rsi_14.between(35, 50)),
        (Candidate("HH_LL_BODY_BREAK", "HH_LL_CONFIRMATION", "Confirmed HH/LL break with 60% directional body"),
         up & breakout & hh & bullish_body & body_fraction.ge(.60),
         down & breakdown & ll & bearish_body & body_fraction.ge(.60)),
    ]

    regime = np.select(
        [up & data.rolling_volatility.gt(vol_baseline), down & data.rolling_volatility.gt(vol_baseline),
         up, down], ["UP_HIGH_VOL", "DOWN_HIGH_VOL", "UP_TREND", "DOWN_TREND"], default="RANGE")
    results: dict[str, tuple[Candidate, pd.DataFrame]] = {}
    evaluation_columns = ["timestamp", "open", "high", "low", "close", "atr_14",
                          "minutes_from_open", "session_id"]
    for candidate, long_condition, short_condition in definitions:
        signal = _eventize(common & long_condition.fillna(False),
                           common & short_condition.fillna(False), data.session_id)
        # Retain only fields required by OpportunityMatrix. Keeping sixteen
        # complete feature-table copies causes unnecessary multi-GB pressure.
        candidate_frame = data[evaluation_columns].copy()
        candidate_frame["entry_signal"] = signal
        candidate_frame["trade_allowed"] = signal.ne(0).astype("int8")
        candidate_frame["trend_state"] = regime
        results[candidate.candidate_id] = (candidate, candidate_frame)
    return results


def evaluate_candidates(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    candidates = build_candidate_setups(frame)
    all_outcomes, setup_rows = [], []
    sessions = frame.session_id.nunique()
    for candidate_id, (candidate, candidate_frame) in candidates.items():
        allowed = candidate_frame.trade_allowed.eq(1)
        long_count = int((allowed & candidate_frame.entry_signal.eq(1)).sum())
        short_count = int((allowed & candidate_frame.entry_signal.eq(-1)).sum())
        count = long_count + short_count
        setup_rows.append({"candidate_id": candidate_id, "family": candidate.family,
                           "description": candidate.description, "setups": count,
                           "sessions": sessions, "setups_per_session": count / sessions,
                           "long_setups": long_count, "short_setups": short_count,
                           "minority_direction_share": min(long_count, short_count) / count if count else 0.0})
        matrix = OpportunityMatrix(candidate_frame, maximum_horizon=15)
        for contract in BENCHMARK_CONTRACTS:
            outcomes = matrix.evaluate(contract)
            outcomes.insert(0, "candidate_id", candidate_id)
            outcomes.insert(1, "setup_family", candidate.family)
            all_outcomes.append(outcomes)
    outcomes = pd.concat(all_outcomes, ignore_index=True)
    setups = pd.DataFrame(setup_rows)
    overall = summarize_contract(outcomes, ["candidate_id", "setup_family", "contract_id"])
    yearly = summarize_contract(outcomes, ["candidate_id", "contract_id", "year"])
    regime = summarize_contract(outcomes, ["candidate_id", "contract_id", "regime"])
    session = summarize_contract(outcomes, ["candidate_id", "contract_id", "session_segment"])
    directional = summarize_contract(outcomes, ["candidate_id", "contract_id", "direction"])

    stability = overall.groupby("candidate_id").agg(
        median_expected_r=("expected_r", "median"), worst_contract_expected_r=("expected_r", "min"),
        mean_mfe_r=("average_mfe_r", "mean"), mean_mae_r=("average_mae_r", "mean"),
        positive_contracts=("expected_r", lambda x: int((x > 0).sum())),
    ).reset_index()
    year_gate = yearly.groupby("candidate_id").agg(
        yearly_cells=("expected_r", "size"), positive_year_fraction=("expected_r", lambda x: float((x > 0).mean())),
        worst_year_expected_r=("expected_r", "min"), yearly_expected_r_std=("expected_r", "std"),
    ).reset_index()
    direction_gate = directional.groupby("candidate_id").agg(
        worst_direction_expected_r=("expected_r", "min")).reset_index()
    regime_gate = regime.groupby("candidate_id").agg(
        worst_regime_expected_r=("expected_r", "min")).reset_index()
    session_gate = session.groupby("candidate_id").agg(
        worst_session_expected_r=("expected_r", "min")).reset_index()
    leaderboard = setups.merge(stability, on="candidate_id").merge(year_gate, on="candidate_id").merge(
        direction_gate, on="candidate_id").merge(regime_gate, on="candidate_id").merge(
        session_gate, on="candidate_id")
    leaderboard["robust_candidate"] = (
        leaderboard.setups.ge(500) & leaderboard.setups_per_session.between(.5, 8.0) &
        leaderboard.minority_direction_share.ge(.20) & leaderboard.worst_contract_expected_r.gt(0) &
        leaderboard.positive_year_fraction.ge(.80) & leaderboard.worst_year_expected_r.gt(0) &
        leaderboard.worst_direction_expected_r.gt(0) & leaderboard.worst_regime_expected_r.gt(0) &
        leaderboard.worst_session_expected_r.gt(0))
    leaderboard["rank_score"] = (
        leaderboard.median_expected_r + leaderboard.worst_contract_expected_r * .50 +
        leaderboard.positive_year_fraction * .10 - leaderboard.yearly_expected_r_std.fillna(0) * .25)
    leaderboard = leaderboard.sort_values(
        ["robust_candidate", "rank_score", "setups"], ascending=[False, False, False]).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    return {"leaderboard": leaderboard, "overall": overall, "yearly": yearly,
            "regime": regime, "session": session, "directional": directional,
            "outcomes": outcomes}
