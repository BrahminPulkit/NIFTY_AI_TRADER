"""Step 19B: causal multi-timeframe context research over frozen setups."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score


CONTEXT_FEATURES = [
    "ctx15_trend_direction",
    "ctx15_hh",
    "ctx15_hl",
    "ctx15_lh",
    "ctx15_ll",
    "ctx15_ema20_slope",
    "ctx15_ema50_slope",
    "ctx15_distance_ema20",
    "ctx15_distance_ema50",
    "ctx15_distance_prev_swing_high",
    "ctx15_distance_prev_swing_low",
    "ctx15_opening_range_position",
    "ctx15_previous_day_position",
    "ctx15_trend_strength",
    "ctx15_atr",
    "ctx15_volatility_regime",
    "ctx15_volatility_expanding",
    "ctx15_pullback",
    "ctx15_near_previous_swing",
    "ctx15_orb_break",
]

CONFIRMATION_FEATURES = [
    "confirm_1m_breakout",
    "confirm_1m_rejection",
    "confirm_1m_engulfing",
    "confirm_1m_volume_expansion",
    "confirm_1m_momentum_candle",
]

CONDITION_FEATURES = [
    "condition_15m_trend_agrees",
    "condition_15m_ema20_slope_agrees",
    "condition_15m_ema50_slope_agrees",
    "condition_15m_volatility_expanding",
    "condition_15m_pullback_agrees",
    "condition_15m_near_previous_swing",
    "condition_15m_orb_agrees",
]


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata"
    )


def build_completed_15m_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features from fully closed 15-minute bars only.

    Bars are right-labelled: the 09:15--09:29 bar becomes observable at 09:30.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Canonical index data missing: {missing}")
    data = frame[list(required)].copy()
    data["timestamp"] = _timestamp(frame)
    data = data.sort_values("timestamp")
    if data.timestamp.duplicated().any():
        raise ValueError("Canonical timestamps must be unique")
    indexed = data.set_index("timestamp")
    bars = indexed.resample(
        "15min", origin="start_day", offset="0min", closed="left", label="right"
    ).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])
    # A valid regular-session 15m bar must begin at/after 09:15 and end by 15:30.
    end_minute = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(end_minute >= 570) & (end_minute <= 930)].copy()

    close = bars.close
    bars["ema20"] = close.ewm(span=20, adjust=False).mean()
    bars["ema50"] = close.ewm(span=50, adjust=False).mean()
    previous_close = close.shift(1)
    true_range = pd.concat([
        bars.high - bars.low,
        (bars.high - previous_close).abs(),
        (bars.low - previous_close).abs(),
    ], axis=1).max(axis=1)
    bars["atr"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    bars["prev_swing_high"] = bars.high.shift(1).rolling(8, min_periods=8).max()
    bars["prev_swing_low"] = bars.low.shift(1).rolling(8, min_periods=8).min()
    bars["ctx15_trend_direction"] = np.select(
        [(close > bars.ema20) & (bars.ema20 > bars.ema50),
         (close < bars.ema20) & (bars.ema20 < bars.ema50)],
        [1, -1], default=0,
    )
    bars["ctx15_hh"] = (bars.high > bars.prev_swing_high).astype("int8")
    bars["ctx15_hl"] = (bars.low > bars.prev_swing_low).astype("int8")
    bars["ctx15_lh"] = (bars.high < bars.prev_swing_high).astype("int8")
    bars["ctx15_ll"] = (bars.low < bars.prev_swing_low).astype("int8")
    bars["ctx15_ema20_slope"] = bars.ema20.diff()
    bars["ctx15_ema50_slope"] = bars.ema50.diff()
    bars["ctx15_distance_ema20"] = (close - bars.ema20) / close
    bars["ctx15_distance_ema50"] = (close - bars.ema50) / close
    bars["ctx15_distance_prev_swing_high"] = (bars.prev_swing_high - close) / close
    bars["ctx15_distance_prev_swing_low"] = (close - bars.prev_swing_low) / close

    day = bars.index.normalize()
    first_bar_high = bars.high.groupby(day).transform("first")
    first_bar_low = bars.low.groupby(day).transform("first")
    bars["ctx15_opening_range_position"] = np.select(
        [close > first_bar_high, close < first_bar_low], [1, -1], default=0
    )
    daily = indexed.groupby(indexed.index.normalize()).agg(
        day_high=("high", "max"), day_low=("low", "min")
    )
    daily[["prev_day_high", "prev_day_low"]] = daily[["day_high", "day_low"]].shift(1)
    bars = bars.join(daily[["prev_day_high", "prev_day_low"]], on=day)
    bars["ctx15_previous_day_position"] = np.select(
        [close > bars.prev_day_high, close < bars.prev_day_low], [1, -1], default=0
    )
    safe_atr = bars.atr.replace(0, np.nan)
    bars["ctx15_trend_strength"] = (bars.ema20 - bars.ema50).abs() / safe_atr
    bars["ctx15_atr"] = bars.atr
    atr_baseline = bars.atr.shift(1).rolling(40, min_periods=20).median()
    bars["ctx15_volatility_regime"] = np.select(
        [bars.atr > atr_baseline * 1.15, bars.atr < atr_baseline * 0.85],
        [1, -1], default=0,
    )
    bars["ctx15_volatility_expanding"] = (bars.atr > bars.atr.shift(1)).astype("int8")
    up_pullback = (bars.ctx15_trend_direction.eq(1) & (bars.low <= bars.ema20) & (close > bars.ema20))
    down_pullback = (bars.ctx15_trend_direction.eq(-1) & (bars.high >= bars.ema20) & (close < bars.ema20))
    bars["ctx15_pullback"] = np.select([up_pullback, down_pullback], [1, -1], default=0)
    nearest_swing = pd.concat([
        (close - bars.prev_swing_high).abs(),
        (close - bars.prev_swing_low).abs(),
    ], axis=1).min(axis=1)
    bars["ctx15_near_previous_swing"] = (nearest_swing <= safe_atr * 0.5).astype("int8")
    bars["ctx15_orb_break"] = bars.ctx15_opening_range_position.astype("int8")
    bars = bars.reset_index().rename(columns={"timestamp": "context_timestamp"})
    return bars[["context_timestamp", *CONTEXT_FEATURES]]


def attach_context(
    setups: pd.DataFrame, outcomes: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    required_setup = {
        "timestamp", "open", "high", "low", "close", "volume", "entry_signal",
        "trade_allowed", "breakout_flag", "breakdown_flag", "candle_body",
        "upper_wick", "lower_wick", "candle_range",
    }
    if missing := sorted(required_setup.difference(setups.columns)):
        raise ValueError(f"Frozen setup data missing: {missing}")
    selected = setups.copy()
    selected["timestamp"] = _timestamp(setups)
    labels = outcomes.copy()
    labels["timestamp"] = _timestamp(outcomes)
    if not labels.trade_allowed.eq(1).all():
        raise ValueError("Only frozen allowed outcomes may be researched")
    setup_rows = selected.merge(labels[[
        "timestamp", "final_outcome", "realized_r", "direction", "year",
        "session_segment",
    ]], on="timestamp", how="inner", validate="one_to_one")
    ctx = context.sort_values("context_timestamp")
    result = pd.merge_asof(
        setup_rows.sort_values("timestamp"), ctx,
        left_on="timestamp", right_on="context_timestamp",
        direction="backward", allow_exact_matches=True,
    )
    if (result.context_timestamp > result.timestamp).any():
        raise AssertionError("Future 15-minute context was attached")
    previous = selected[["timestamp", "open", "close", "volume"]].copy()
    previous[["prev_open", "prev_close", "prev_volume"]] = previous[
        ["open", "close", "volume"]
    ].shift(1)
    result = result.merge(
        previous[["timestamp", "prev_open", "prev_close", "prev_volume"]],
        on="timestamp", how="left", validate="one_to_one",
    )
    direction = result.entry_signal
    result["confirm_1m_breakout"] = np.where(
        direction.eq(1), result.breakout_flag.eq(1), result.breakdown_flag.eq(1)
    ).astype("int8")
    result["confirm_1m_rejection"] = np.where(
        direction.eq(1), result.lower_wick > result.candle_body.abs(),
        result.upper_wick > result.candle_body.abs(),
    ).astype("int8")
    bullish_engulf = (
        (result.close > result.open) & (result.prev_close < result.prev_open)
        & (result.open <= result.prev_close) & (result.close >= result.prev_open)
    )
    bearish_engulf = (
        (result.close < result.open) & (result.prev_close > result.prev_open)
        & (result.open >= result.prev_close) & (result.close <= result.prev_open)
    )
    result["confirm_1m_engulfing"] = np.where(
        direction.eq(1), bullish_engulf, bearish_engulf
    ).astype("int8")
    result["confirm_1m_volume_expansion"] = (
        (result.volume > result.prev_volume) & result.volume.gt(0)
    ).fillna(False).astype("int8")
    body_ratio = result.candle_body.abs() / result.candle_range.replace(0, np.nan)
    direction_body = np.where(
        direction.eq(1), result.close > result.open, result.close < result.open
    )
    result["confirm_1m_momentum_candle"] = (
        direction_body & body_ratio.ge(0.6)
    ).astype("int8")
    result["condition_15m_trend_agrees"] = (
        result.ctx15_trend_direction.eq(direction)
    ).astype("int8")
    result["condition_15m_ema20_slope_agrees"] = np.where(
        direction.eq(1), result.ctx15_ema20_slope.gt(0),
        result.ctx15_ema20_slope.lt(0),
    ).astype("int8")
    result["condition_15m_ema50_slope_agrees"] = np.where(
        direction.eq(1), result.ctx15_ema50_slope.gt(0),
        result.ctx15_ema50_slope.lt(0),
    ).astype("int8")
    result["condition_15m_volatility_expanding"] = (
        result.ctx15_volatility_expanding.eq(1)
    ).astype("int8")
    result["condition_15m_pullback_agrees"] = (
        result.ctx15_pullback.eq(direction)
    ).astype("int8")
    result["condition_15m_near_previous_swing"] = (
        result.ctx15_near_previous_swing.eq(1)
    ).astype("int8")
    result["condition_15m_orb_agrees"] = (
        result.ctx15_orb_break.eq(direction)
    ).astype("int8")
    return result


def conditional_statistics(
    data: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = data[feature]
        if values.nunique(dropna=True) > 10:
            # Descriptive quantile buckets only; these are not candidate gates
            # and are never persisted as production thresholds.
            grouping = pd.qcut(values, q=5, duplicates="drop").astype(str)
        else:
            grouping = values
        for value, group in data.groupby(grouping, dropna=False, sort=True):
            rows.append({
                "feature": feature,
                "value": value,
                "setups": len(group),
                "win_count": int(group.final_outcome.eq("WIN").sum()),
                "win_rate": float(group.final_outcome.eq("WIN").mean()),
                "loss_rate": float(group.final_outcome.eq("LOSS").mean()),
                "timeout_rate": float(group.final_outcome.eq("TIMEOUT").mean()),
                "expected_r": float(group.realized_r.mean()),
            })
    return pd.DataFrame(rows)


def rank_context(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = data.dropna(subset=CONTEXT_FEATURES).copy()
    y = complete.final_outcome.eq("WIN").astype(int)
    mi_values = mutual_info_classif(
        complete[CONTEXT_FEATURES].astype(float), y, discrete_features=False,
        random_state=42,
    )
    mi = pd.DataFrame({
        "feature": CONTEXT_FEATURES,
        "mutual_information": mi_values,
    })
    ig_rows = []
    for feature in CONTEXT_FEATURES:
        series = complete[feature]
        if series.nunique() > 10:
            bins = pd.qcut(series, q=min(10, series.nunique()), duplicates="drop")
            encoded = bins.cat.codes
        else:
            encoded = pd.factorize(series, sort=True)[0]
        ig_rows.append({
            "feature": feature,
            "information_gain": mutual_info_score(encoded, y),
        })
    information = pd.DataFrame(ig_rows)
    rankings = information.merge(mi, on="feature")
    rankings["shap_importance"] = np.nan
    rankings["shap_status"] = "NOT_COMPUTED_MODEL_RETRAINING_FORBIDDEN"
    rankings["combined_rank"] = (
        rankings.information_gain.rank(ascending=False, method="min")
        + rankings.mutual_information.rank(ascending=False, method="min")
    ) / 2
    rankings = rankings.sort_values(
        ["combined_rank", "information_gain", "mutual_information"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return information, mi, rankings


def interaction_statistics(data: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(top_features, 2):
        # Natural sign/state interactions only; no threshold search.
        left_state = np.sign(data[left].fillna(0)).astype(int)
        right_state = np.sign(data[right].fillna(0)).astype(int)
        for (lv, rv), group in data.groupby([left_state, right_state], sort=True):
            if len(group) < 30:
                continue
            rows.append({
                "feature_left": left, "feature_right": right,
                "left_state": lv, "right_state": rv, "setups": len(group),
                "win_rate": group.final_outcome.eq("WIN").mean(),
                "expected_r": group.realized_r.mean(),
            })
    return pd.DataFrame(rows)
