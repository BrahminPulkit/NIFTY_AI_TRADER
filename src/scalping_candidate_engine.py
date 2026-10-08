"""Scored, causal NIFTY scalping candidates. Candidates are not trade approvals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STRATEGIES = (
    "TREND_PULLBACK", "VWAP_RECLAIM", "BREAKOUT",
    "MOMENTUM_CONTINUATION", "LIQUIDITY_SWEEP_REVERSAL",
)


@dataclass(frozen=True)
class CandidateConfig:
    minimum_candidate_score: int = 35
    first_minute: int = 15
    last_minute: int = 345
    cooldown_minutes: int = 5


def _completed_5m_direction(data: pd.DataFrame) -> pd.Series:
    bucket = (data.minutes_from_open // 5).astype(int)
    bars = data.groupby(["session_id", bucket], sort=False).close.last().rename("close").reset_index()
    key = bars.columns[1]
    fast = bars.groupby("session_id", sort=False).close.transform(
        lambda x: x.ewm(span=2, adjust=False, min_periods=2).mean())
    slow = bars.groupby("session_id", sort=False).close.transform(
        lambda x: x.ewm(span=5, adjust=False, min_periods=5).mean())
    bars["direction"] = pd.Series(np.sign(fast - slow), index=bars.index).groupby(
        bars.session_id, sort=False).shift(1)
    keys = pd.MultiIndex.from_arrays([data.session_id, bucket])
    return pd.Series(bars.set_index(["session_id", key]).direction.reindex(keys).to_numpy(), index=data.index)


def generate_scalping_candidates(features: pd.DataFrame,
                                  config: CandidateConfig = CandidateConfig()) -> pd.DataFrame:
    required = {
        "timestamp", "open", "high", "low", "close", "volume", "ema_9", "ema_20",
        "vwap", "rsi_14", "atr_14", "rolling_high", "rolling_low", "minutes_from_open",
        "session_id", "candle_body", "candle_range",
    }
    if missing := sorted(required.difference(features.columns)):
        raise ValueError(f"Candidate features missing: {missing}")
    data = features.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    m5 = _completed_5m_direction(data)
    previous_close = data.groupby("session_id", sort=False).close.shift(1)
    previous_vwap = data.groupby("session_id", sort=False).vwap.shift(1)
    previous_ema_9 = data.groupby("session_id", sort=False).ema_9.shift(1)
    prior_high = data.groupby("session_id", sort=False).rolling_high.shift(1)
    prior_low = data.groupby("session_id", sort=False).rolling_low.shift(1)
    volume_base = data.volume.groupby(data.session_id, sort=False).transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).median())
    volume_ok = data.volume.ge(volume_base * 1.10).fillna(False)
    bullish_momentum, bearish_momentum = data.rsi_14.ge(52).fillna(False), data.rsi_14.le(48).fillna(False)
    bullish_body, bearish_body = data.candle_body.gt(0), data.candle_body.lt(0)
    trend_up, trend_down = m5.eq(1), m5.eq(-1)
    above_vwap, below_vwap = data.close.gt(data.vwap).fillna(False), data.close.lt(data.vwap).fillna(False)
    reclaim_up = previous_close.le(previous_vwap) & above_vwap & bullish_body
    reclaim_down = previous_close.ge(previous_vwap) & below_vwap & bearish_body
    breakout_up = data.close.gt(prior_high) & bullish_body
    breakout_down = data.close.lt(prior_low) & bearish_body
    pullback_up = trend_up & data.low.le(data.ema_20) & data.close.gt(data.ema_20) & bullish_body
    pullback_down = trend_down & data.high.ge(data.ema_20) & data.close.lt(data.ema_20) & bearish_body
    continuation_up = trend_up & previous_close.le(previous_ema_9) & data.close.gt(data.ema_9) & bullish_body
    continuation_down = trend_down & previous_close.ge(previous_ema_9) & data.close.lt(data.ema_9) & bearish_body
    sweep_up = data.low.lt(prior_low) & data.close.gt(prior_low) & bullish_body
    sweep_down = data.high.gt(prior_high) & data.close.lt(prior_high) & bearish_body
    definitions = {
        "TREND_PULLBACK": (pullback_up, pullback_down),
        "VWAP_RECLAIM": (reclaim_up, reclaim_down),
        "BREAKOUT": (breakout_up, breakout_down),
        "MOMENTUM_CONTINUATION": (continuation_up, continuation_down),
        "LIQUIDITY_SWEEP_REVERSAL": (sweep_up, sweep_down),
    }
    session_ok = data.minutes_from_open.between(config.first_minute, config.last_minute)
    finite = data[["close", "ema_9", "ema_20", "rsi_14", "atr_14"]].notna().all(axis=1)
    records = []
    last: dict[tuple[int, str], pd.Timestamp] = {}
    for strategy, (long_trigger, short_trigger) in definitions.items():
        strategy_finite = finite & (data.vwap.notna() if strategy == "VWAP_RECLAIM" else True)
        for direction, trigger in ((1, long_trigger), (-1, short_trigger)):
            locations = data.index[(trigger.fillna(False) & session_ok & strategy_finite)]
            for location in locations:
                row = data.loc[location]
                key = (int(row.session_id), strategy, direction)
                if key in last and row.timestamp - last[key] < pd.Timedelta(minutes=config.cooldown_minutes):
                    continue
                last[key] = row.timestamp
                score_parts = {
                    "trend": 20 if (trend_up.at[location] if direction == 1 else trend_down.at[location]) else 0,
                    "vwap": 15 if (above_vwap.at[location] if direction == 1 else below_vwap.at[location]) else 0,
                    "momentum": 15 if (bullish_momentum.at[location] if direction == 1 else bearish_momentum.at[location]) else 0,
                    "volume": 10 if volume_ok.at[location] else 0,
                    "structure": 15,
                }
                score = sum(score_parts.values())
                if score < config.minimum_candidate_score:
                    continue
                records.append({
                    "timestamp": row.timestamp, "strategy": strategy, "direction": direction,
                    "option_type": "CE" if direction == 1 else "PE", "candidate_created": True,
                    "base_strategy_score": score, "score_components": score_parts,
                    "index_price": float(row.close), "atr": float(row.atr_14),
                })
    if not records:
        return pd.DataFrame(columns=["timestamp", "strategy", "direction", "option_type",
                                     "candidate_created", "base_strategy_score", "score_components",
                                     "index_price", "atr"])
    return pd.DataFrame(records).sort_values(["timestamp", "base_strategy_score", "strategy"],
                                             ascending=[True, False, True]).reset_index(drop=True)
