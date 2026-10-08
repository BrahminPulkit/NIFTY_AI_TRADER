"""Step 19A: audit-only analysis of the frozen Stage-2 outcome sparsity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.stage2_outcome_dataset import CONTRACT


MOVE_LEVELS = (0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030)
TARGET_LEVELS = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030)


def _validate(setups: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_required = {"timestamp", "open", "high", "low", "close", "session_id"}
    outcome_required = {
        "timestamp", "direction", "entry_signal", "trade_allowed", "entry_price",
        "final_outcome", "exit_reason", "label_horizon",
    }
    if missing := sorted(setup_required.difference(setups.columns)):
        raise ValueError(f"Setup data missing columns: {missing}")
    if missing := sorted(outcome_required.difference(outcomes.columns)):
        raise ValueError(f"Outcome data missing columns: {missing}")
    s = setups.copy()
    o = outcomes.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    if s.timestamp.duplicated().any() or not s.timestamp.is_monotonic_increasing:
        raise ValueError("Setup timestamps must be unique and chronological")
    if o.timestamp.duplicated().any() or not o.timestamp.is_monotonic_increasing:
        raise ValueError("Outcome timestamps must be unique and chronological")
    if not o.trade_allowed.eq(1).all():
        raise ValueError("Step 19A accepts frozen allowed outcomes only")
    if not o.label_horizon.eq(CONTRACT.horizon).all():
        raise ValueError("Frozen five-candle horizon was not preserved")
    return s.reset_index(drop=True), o.reset_index(drop=True)


def _simulate(
    candles: list[tuple[int, float, float, float, float]],
    entry: float,
    direction: int,
    target_pct: float,
) -> tuple[str, int, float]:
    """Replay the frozen stop-first, gap-aware contract at one diagnostic target."""
    stop = entry * (1 - CONTRACT.stop_pct * direction)
    target = entry * (1 + target_pct * direction)
    exit_price = entry
    for offset, candle_open, high, low, close in candles:
        exit_price = close
        if direction == 1:
            hit_stop, hit_target = low <= stop, high >= target
        else:
            hit_stop, hit_target = high >= stop, low <= target
        if hit_stop:
            fill = min(candle_open, stop) if direction == 1 else max(candle_open, stop)
            return "LOSS", offset, direction * (fill - entry) / (entry * CONTRACT.stop_pct)
        if hit_target:
            fill = max(candle_open, target) if direction == 1 else min(candle_open, target)
            return "WIN", offset, direction * (fill - entry) / (entry * CONTRACT.stop_pct)
    return "TIMEOUT", candles[-1][0], direction * (exit_price - entry) / (
        entry * CONTRACT.stop_pct
    )


def analyze_outcome_paths(
    setups: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-setup excursion diagnostics and target simulations."""
    data, labels = _validate(setups, outcomes)
    lookup = pd.Series(data.index, index=data.timestamp)
    sessions = data.session_id.astype(str).to_numpy()
    path_rows: list[dict] = []
    simulation_rows: list[dict] = []

    for row in labels.itertuples(index=False):
        index = int(lookup.loc[row.timestamp])
        entry = float(row.entry_price)
        direction = int(row.entry_signal)
        candles: list[tuple[int, float, float, float, float]] = []
        favourable: list[float] = []
        adverse: list[float] = []
        for offset in range(1, CONTRACT.horizon + 1):
            future = index + offset
            if future >= len(data) or sessions[future] != sessions[index]:
                break
            candle = data.iloc[future]
            candles.append((
                offset, float(candle.open), float(candle.high),
                float(candle.low), float(candle.close),
            ))
            favourable.append(
                (float(candle.high) - entry) / entry if direction == 1
                else (entry - float(candle.low)) / entry
            )
            adverse.append(
                (entry - float(candle.low)) / entry if direction == 1
                else (float(candle.high) - entry) / entry
            )
        if not candles:
            raise AssertionError(f"No observable path for frozen outcome {row.timestamp}")
        mfe = max(0.0, max(favourable))
        mae = max(0.0, max(adverse))
        time_mfe = favourable.index(max(favourable)) + 1 if max(favourable) > 0 else 0
        time_mae = adverse.index(max(adverse)) + 1 if max(adverse) > 0 else 0
        path = {
            "timestamp": row.timestamp,
            "direction": row.direction,
            "production_outcome": row.final_outcome,
            "production_exit_reason": row.exit_reason,
            "entry_price": entry,
            "observable_candles": len(candles),
            "mfe_pct": mfe,
            "mae_pct": mae,
            "mfe_points": mfe * entry,
            "mae_points": mae * entry,
            "time_to_mfe": time_mfe,
            "time_to_mae": time_mae,
            "almost_reached_030": 0.0027 <= mfe < 0.0030,
        }
        for level in MOVE_LEVELS:
            path[f"reached_{level * 100:.2f}pct"] = mfe >= level
        path_rows.append(path)

        for target in TARGET_LEVELS:
            outcome, holding, realized_r = _simulate(candles, entry, direction, target)
            simulation_rows.append({
                "timestamp": row.timestamp,
                "direction": row.direction,
                "diagnostic_target_pct": target,
                "outcome": outcome,
                "holding_candles": holding,
                "realized_r": realized_r,
                "raw_target_reached": mfe >= target,
                "production_outcome": row.final_outcome,
            })
    return pd.DataFrame(path_rows), pd.DataFrame(simulation_rows)


def summarize_mfe(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scope_values = [("ALL", paths), *list(paths.groupby("direction", sort=True))]
    for direction, group in scope_values:
        for outcome, subset in [("ALL", group), *list(group.groupby("production_outcome"))]:
            values = subset.mfe_pct
            rows.append({
                "direction": direction,
                "production_outcome": outcome,
                "setups": len(subset),
                "mfe_mean_pct": values.mean(),
                "mfe_median_pct": values.median(),
                "mfe_p25_pct": values.quantile(0.25),
                "mfe_p75_pct": values.quantile(0.75),
                "mfe_p90_pct": values.quantile(0.90),
                "mae_mean_pct": subset.mae_pct.mean(),
                "median_time_to_mfe": subset.time_to_mfe.median(),
                "median_time_to_mae": subset.time_to_mae.median(),
            })
    return pd.DataFrame(rows)


def summarize_target_reach(paths: pd.DataFrame, simulations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target, target_group in simulations.groupby("diagnostic_target_pct", sort=True):
        for direction, group in [("ALL", target_group), *list(target_group.groupby("direction"))]:
            total = len(group)
            rows.append({
                "diagnostic_target_pct": target,
                "direction": direction,
                "setups": total,
                "raw_reach_count": int(group.raw_target_reached.sum()),
                "raw_reach_rate": float(group.raw_target_reached.mean()),
                "executable_win_count": int(group.outcome.eq("WIN").sum()),
                "executable_win_rate": float(group.outcome.eq("WIN").mean()),
                "loss_count": int(group.outcome.eq("LOSS").sum()),
                "timeout_count": int(group.outcome.eq("TIMEOUT").sum()),
                "expected_r": float(group.realized_r.mean()),
            })
    return pd.DataFrame(rows)


def verify_frozen_replay(outcomes: pd.DataFrame, simulations: pd.DataFrame) -> None:
    replay = simulations.loc[
        simulations.diagnostic_target_pct.eq(CONTRACT.target_pct),
        ["timestamp", "outcome"],
    ].set_index("timestamp").outcome
    expected = outcomes.set_index("timestamp").final_outcome
    if not replay.reindex(expected.index).eq(expected).all():
        mismatch = int((replay.reindex(expected.index) != expected).sum())
        raise AssertionError(f"Frozen 0.30% contract replay mismatch: {mismatch}")

