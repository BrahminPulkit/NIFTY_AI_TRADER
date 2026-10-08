"""Research-only evaluation of the fixed production label contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchContract:
    stop_pct: float = 0.002
    target_pct: float = 0.003
    ambiguous_bar: str = "stop"

    @property
    def target_r(self) -> float:
        return self.target_pct / self.stop_pct


def classify_regime(frame: pd.DataFrame) -> pd.Series:
    up = (frame.close > frame.ema_20) & (frame.ema_20 > frame.ema_50)
    down = (frame.close < frame.ema_20) & (frame.ema_20 < frame.ema_50)
    return pd.Series(np.select([up, down], ["UPTREND", "DOWNTREND"], default="SIDEWAYS"),
                     index=frame.index, name="regime")


def classify_session(frame: pd.DataFrame) -> pd.Series:
    minute = frame.minutes_from_open
    return pd.Series(np.select([minute < 60, minute >= 315], ["OPENING", "CLOSING"],
                               default="MID_SESSION"), index=frame.index, name="session_segment")


def evaluate_direction(frame: pd.DataFrame, horizon: int, direction: int,
                       contract: ResearchContract = ResearchContract()) -> pd.DataFrame:
    """Evaluate overlapping label opportunities; this is not a trade backtest."""
    if horizon < 1 or direction not in (-1, 1):
        raise ValueError("Invalid horizon or direction")
    required = {"timestamp", "open", "high", "low", "close", "ema_20", "ema_50",
                "minutes_from_open"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame.reset_index(drop=True)
    n = len(data)
    close = data.close.to_numpy(float)
    opens = data.open.to_numpy(float)
    highs = data.high.to_numpy(float)
    lows = data.low.to_numpy(float)
    session_id = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime(
        "%Y%m%d").to_numpy()
    positions = np.arange(n)
    reverse_count = pd.Series(session_id).groupby(session_id, sort=False).cumcount(ascending=False).to_numpy()
    available = np.minimum(reverse_count, horizon)
    valid = available >= 1

    risk = close * contract.stop_pct
    stop = close - risk if direction == 1 else close + risk
    target = close + risk * contract.target_r if direction == 1 else close - risk * contract.target_r
    exit_offset = available.copy()
    exit_index = np.minimum(positions + exit_offset, n - 1)
    exit_price = close[exit_index].copy()
    outcome = np.full(n, "TIMEOUT", dtype=object)
    unresolved = valid.copy()
    mfe_points = np.full(n, np.nan)
    mae_points = np.full(n, np.nan)
    mfe_points[valid] = 0.0
    mae_points[valid] = 0.0

    for offset in range(1, horizon + 1):
        idx = positions + offset
        exists = valid & (available >= offset) & (idx < n)
        safe_idx = np.minimum(idx, n - 1)
        if direction == 1:
            favorable = highs[safe_idx] - close
            adverse = close - lows[safe_idx]
            hit_stop = lows[safe_idx] <= stop
            hit_target = highs[safe_idx] >= target
        else:
            favorable = close - lows[safe_idx]
            adverse = highs[safe_idx] - close
            hit_stop = highs[safe_idx] >= stop
            hit_target = lows[safe_idx] <= target
        mfe_points[exists] = np.maximum(mfe_points[exists], favorable[exists])
        mae_points[exists] = np.maximum(mae_points[exists], adverse[exists])
        candidates = unresolved & exists
        # Conservative ambiguity rule: stop is evaluated first.
        stopped = candidates & hit_stop
        if stopped.any():
            fill = np.minimum(opens[safe_idx], stop) if direction == 1 else np.maximum(opens[safe_idx], stop)
            exit_price[stopped] = fill[stopped]
            exit_offset[stopped] = offset
            outcome[stopped] = "STOP"
            unresolved[stopped] = False
        won = unresolved & exists & hit_target
        if won.any():
            fill = np.maximum(opens[safe_idx], target) if direction == 1 else np.minimum(opens[safe_idx], target)
            exit_price[won] = fill[won]
            exit_offset[won] = offset
            outcome[won] = "WIN"
            unresolved[won] = False

    timestamp = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    result = pd.DataFrame({
        "timestamp": timestamp, "year": timestamp.dt.year, "direction": "BUY_CALL" if direction == 1 else "BUY_PUT",
        "horizon": horizon, "outcome": outcome, "holding_candles": exit_offset,
        "mfe_r": mfe_points / risk, "mae_r": mae_points / risk,
        "realized_r": direction * (exit_price - close) / risk,
        "regime": classify_regime(data), "session_segment": classify_session(data),
        "session_id": session_id,
    })
    return result.loc[valid].reset_index(drop=True)


def summarize(observations: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = observations.groupby(group_columns, dropna=False, sort=True)
    rows = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        days = group.session_id.nunique()
        rows.append({
            **dict(zip(group_columns, keys)), "observations": len(group),
            "win_rate": group.outcome.eq("WIN").mean(),
            "stop_loss_hit_rate": group.outcome.eq("STOP").mean(),
            "timeout_rate": group.outcome.eq("TIMEOUT").mean(),
            "average_holding_time": group.holding_candles.mean(),
            "average_mfe_r": group.mfe_r.mean(), "median_mfe_r": group.mfe_r.median(),
            "average_mae_r": group.mae_r.mean(), "median_mae_r": group.mae_r.median(),
            "expected_r": group.realized_r.mean(),
            "trade_frequency_per_day": group.outcome.ne("TIMEOUT").sum() / days,
            "opportunity_frequency_per_day": len(group) / days,
            "trading_days": days,
        })
    return pd.DataFrame(rows)
