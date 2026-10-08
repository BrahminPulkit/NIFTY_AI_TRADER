"""Step 19E: causal, descriptive strategy and regime discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = (15, 30, 45, 60)


def classify_entry_context(setups: pd.DataFrame) -> pd.DataFrame:
    """Classify frozen setup rows without changing their decisions."""
    required = {
        "timestamp", "entry_signal", "trade_allowed", "open", "high", "low",
        "close", "candle_range", "atr_14", "rolling_volatility", "ema_5",
        "ema_9", "ema_20", "ema_50", "breakout_flag", "breakdown_flag",
    }
    if missing := sorted(required.difference(setups.columns)):
        raise ValueError(f"Frozen setup data missing: {missing}")
    data = setups.copy()
    data["timestamp"] = pd.to_datetime(
        data.timestamp, utc=True, errors="raise"
    ).dt.tz_convert("Asia/Kolkata")
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError("Setup timestamps must be chronological and unique")
    allowed = data.loc[
        data.trade_allowed.eq(1) & data.entry_signal.isin([-1, 1])
    ].copy()
    direction = allowed.entry_signal
    atr = allowed.atr_14.replace(0, np.nan)
    aligned_up = (
        (allowed.ema_5 > allowed.ema_9) & (allowed.ema_9 > allowed.ema_20)
        & (allowed.ema_20 > allowed.ema_50)
    )
    aligned_down = (
        (allowed.ema_5 < allowed.ema_9) & (allowed.ema_9 < allowed.ema_20)
        & (allowed.ema_20 < allowed.ema_50)
    )
    allowed["trend_regime"] = np.select(
        [aligned_up, aligned_down], ["TREND_UP", "TREND_DOWN"], default="SIDEWAYS"
    )
    baseline = data.rolling_volatility.shift(1).rolling(120, min_periods=60).median()
    allowed["volatility_ratio"] = (
        allowed.rolling_volatility / baseline.reindex(allowed.index).replace(0, np.nan)
    )
    allowed["volatility_regime"] = np.select(
        [allowed.volatility_ratio > 1.2, allowed.volatility_ratio < 0.8],
        ["EXPANSION", "CONTRACTION"], default="NORMAL",
    )
    pullback = np.where(
        direction.eq(1), allowed.low <= allowed.ema_9,
        allowed.high >= allowed.ema_9,
    )
    allowed["pullback_state"] = np.where(pullback, "PULLBACK", "NO_PULLBACK")
    allowed["breakout_state"] = np.where(
        np.where(direction.eq(1), allowed.breakout_flag.eq(1), allowed.breakdown_flag.eq(1)),
        "BREAKOUT", "NO_BREAKOUT",
    )
    range_atr = allowed.candle_range / atr
    distance_ema20_atr = (allowed.close - allowed.ema_20).abs() / atr
    allowed["range_atr_ratio"] = range_atr
    allowed["distance_ema20_atr"] = distance_ema20_atr
    allowed["strategy_family"] = np.select(
        [
            pullback,
            range_atr.ge(1.2),
            distance_ema20_atr.ge(1.0),
        ],
        [
            "PULLBACK_BREAKOUT",
            "MOMENTUM_BREAKOUT",
            "EXTENDED_BREAKOUT",
        ],
        default="CONTINUATION_BREAKOUT",
    )
    allowed["market_regime"] = (
        allowed.trend_regime + "|" + allowed.volatility_regime
    )
    return allowed


def measure_premium_behaviour(
    aligned: pd.DataFrame,
    entries: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    required = {
        "timestamp", "option_close", "option_high", "option_low", "option_volume",
        "index_session_id",
    }
    if missing := sorted(required.difference(aligned.columns)):
        raise ValueError(f"Aligned data missing: {missing}")
    market = aligned.copy()
    market["timestamp"] = pd.to_datetime(
        market.timestamp, utc=True, errors="raise"
    ).dt.tz_convert("Asia/Kolkata")
    market = market.sort_values("timestamp").reset_index(drop=True)
    entry_columns = [
        "timestamp", "entry_signal", "strategy_family", "market_regime",
        "trend_regime", "volatility_regime", "pullback_state", "breakout_state",
        "range_atr_ratio", "distance_ema20_atr", "ema_5", "ema_9", "ema_20",
        "ema_50", "atr_14", "rolling_volatility", "rolling_range",
    ]
    mapped = entries[entry_columns].merge(
        market[["timestamp", "option_close", "option_volume", "index_session_id"]],
        on="timestamp", how="inner", validate="one_to_one",
    ).rename(columns={
        "option_close": "entry_premium", "option_volume": "entry_option_volume"
    })
    lookup = pd.Series(market.index, index=market.timestamp)
    sessions = market.index_session_id.astype(str).to_numpy()
    rows = []
    for entry in mapped.itertuples(index=False):
        index = int(lookup.loc[entry.timestamp])
        premium = float(entry.entry_premium)
        if premium <= 0:
            continue
        for horizon in horizons:
            end = entry.timestamp + pd.Timedelta(minutes=horizon)
            stop = int(market.timestamp.searchsorted(end, side="right"))
            path = market.iloc[index + 1:stop]
            path = path.loc[path.index_session_id.astype(str).eq(sessions[index])]
            if path.empty:
                continue
            terminal = float(path.option_close.iloc[-1]) / premium - 1
            mfe = max(0.0, float(path.option_high.max()) / premium - 1)
            mae = max(0.0, 1 - float(path.option_low.min()) / premium)
            rows.append({
                **{column: getattr(entry, column) for column in entry_columns},
                "direction": "LONG" if entry.entry_signal == 1 else "SHORT",
                "horizon_minutes": horizon,
                "terminal_premium_return": terminal,
                "premium_mfe": mfe,
                "premium_mae": mae,
                "positive_close": terminal > 0,
                "entry_option_volume": float(entry.entry_option_volume),
                "future_option_volume_mean": float(path.option_volume.mean()),
                "full_horizon_observed": (
                    path.timestamp.iloc[-1] >= end
                ),
                "year": int(entry.timestamp.year),
            })
    return pd.DataFrame(rows)


def summarize_behaviour(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yearly = group.groupby("year").terminal_premium_return.mean()
        yearly_std = float(yearly.std()) if len(yearly) > 1 else np.nan
        expansion_drawdown = group.premium_mfe.mean() / max(group.premium_mae.mean(), 1e-12)
        rows.append({
            **dict(zip(groups, keys)),
            "setups": len(group),
            "positive_close_rate": float(group.positive_close.mean()),
            "average_terminal_return": float(group.terminal_premium_return.mean()),
            "median_terminal_return": float(group.terminal_premium_return.median()),
            "average_premium_expansion": float(group.premium_mfe.mean()),
            "maximum_premium_excursion": float(group.premium_mfe.max()),
            "average_drawdown": float(group.premium_mae.mean()),
            "expansion_drawdown_ratio": float(expansion_drawdown),
            "yearly_mean_return_std": yearly_std,
            "stability_score": float(1 / (1 + abs(yearly_std))) if np.isfinite(yearly_std) else np.nan,
            "full_horizon_fraction": float(group.full_horizon_observed.mean()),
        })
    return pd.DataFrame(rows)


def strategy_regime_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["descriptive_grade"] = np.select(
        [
            (result.average_terminal_return >= 0.005)
            & (result.positive_close_rate >= 0.52)
            & (result.expansion_drawdown_ratio >= 1.15),
            (result.average_terminal_return > 0)
            & (result.expansion_drawdown_ratio >= 1.0),
            (result.average_terminal_return <= 0)
            & (result.expansion_drawdown_ratio < 1.0),
        ],
        ["EXCELLENT", "GOOD", "POOR"], default="MIXED",
    )
    return result


def confidence_ranking(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    result = strategy_summary.copy()
    max_sample = max(float(result.setups.max()), 1.0)
    result["sample_confidence"] = np.sqrt(result.setups / max_sample)
    result["confidence_score"] = (
        result.sample_confidence
        * result.stability_score.fillna(0)
        * np.minimum(result.expansion_drawdown_ratio, 2.0) / 2.0
    )
    return result.sort_values(
        ["confidence_score", "average_premium_expansion"], ascending=False
    ).reset_index(drop=True)


def minimal_profitable_set(data: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    subset = data.loc[data.horizon_minutes.eq(horizon)].copy()
    subset["positive_return_contribution"] = subset.terminal_premium_return.clip(lower=0)
    grouped = subset.groupby("strategy_family").agg(
        setups=("timestamp", "size"),
        positive_return_contribution=("positive_return_contribution", "sum"),
        average_terminal_return=("terminal_premium_return", "mean"),
    ).reset_index().sort_values("positive_return_contribution", ascending=False)
    total = grouped.positive_return_contribution.sum()
    grouped["contribution_share"] = (
        grouped.positive_return_contribution / total if total > 0 else 0.0
    )
    grouped["cumulative_share"] = grouped.contribution_share.cumsum()
    grouped["in_smallest_80pct_set"] = (
        grouped.cumulative_share.shift(fill_value=0) < 0.80
    )
    return grouped


def relationship_statistics(data: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "atr_14", "rolling_volatility", "rolling_range", "range_atr_ratio",
        "distance_ema20_atr", "entry_option_volume", "future_option_volume_mean",
    ]
    outcomes = ["terminal_premium_return", "premium_mfe", "premium_mae"]
    rows = []
    for variable in variables:
        for outcome in outcomes:
            pair = data[[variable, outcome]].dropna()
            rows.append({
                "input_variable": variable,
                "premium_measure": outcome,
                "observations": len(pair),
                "spearman_correlation": float(pair.corr(method="spearman").iloc[0, 1]),
            })
    return pd.DataFrame(rows)
