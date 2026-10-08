"""Roadmap Step 18: deterministic Stage-2 outcome dataset engineering.

Stage-1 decides whether a setup exists and whether it is allowed. This module
never changes those decisions; it labels only frozen Stage-1 V2 opportunities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.feature_pipeline import FEATURES


@dataclass(frozen=True)
class ProductionDiagnosticContract:
    version: str = "production_diagnostic_h5_v1"
    horizon: int = 5
    stop_pct: float = 0.002
    target_pct: float = 0.003
    entry_rule: str = "CANDLE_CLOSE"
    ambiguity_rule: str = "STOP_FIRST"
    gap_rule: str = "FIRST_AVAILABLE_OPEN"
    cross_session: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


CONTRACT = ProductionDiagnosticContract()
SETUP_METADATA = [
    "trend_state", "structure_state", "momentum_state", "volatility_state",
    "session_state", "event_state", "breakout_quality_state",
    "candle_quality_state", "trend_strength_state", "rejection_reason",
]


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "open", "high", "low", "close", "atr_14", "rsi_14",
        "entry_signal", "trade_allowed", "session_id", *FEATURES, *SETUP_METADATA,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing frozen Stage-1/feature inputs: {missing}")
    data = frame.reset_index(drop=True).copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError("Stage-2 input timestamps must be sorted and unique")
    if not data.trade_allowed.isin([0, 1]).all() or not data.entry_signal.isin([-1, 0, 1]).all():
        raise ValueError("Invalid frozen Stage-1 decision values")
    invalid_allowed = data.trade_allowed.eq(1) & ~data.entry_signal.isin([-1, 1])
    if invalid_allowed.any():
        raise ValueError("Every allowed setup must have LONG or SHORT direction")
    return data


def build_stage2_outcomes(
    frame: pd.DataFrame, contract: ProductionDiagnosticContract = CONTRACT
) -> pd.DataFrame:
    """Build diagnostic outcomes for allowed setups; no NO_TRADE class exists."""
    if contract != CONTRACT:
        raise ValueError("The Step-18 diagnostic production contract is frozen")
    data = _validate(frame)
    timestamps = data.timestamp
    sessions = timestamps.dt.strftime("%Y%m%d").to_numpy()
    eligible = np.flatnonzero(
        data.trade_allowed.eq(1).to_numpy() & data.entry_signal.isin([-1, 1]).to_numpy()
    )
    records: list[dict] = []
    for index in eligible:
        direction_value = int(data.at[index, "entry_signal"])
        entry = float(data.at[index, "close"])
        risk_points = entry * contract.stop_pct
        stop = entry - risk_points if direction_value == 1 else entry + risk_points
        target = entry + entry * contract.target_pct if direction_value == 1 else entry - entry * contract.target_pct
        outcome = "TIMEOUT"
        reason = "HORIZON_TIMEOUT"
        exit_index = index
        exit_price = entry
        mfe_points = 0.0
        mae_points = 0.0
        for offset in range(1, contract.horizon + 1):
            future = index + offset
            if future >= len(data) or sessions[future] != sessions[index]:
                reason = "SESSION_END_TIMEOUT"
                break
            exit_index = future
            candle_open = float(data.at[future, "open"])
            high = float(data.at[future, "high"])
            low = float(data.at[future, "low"])
            exit_price = float(data.at[future, "close"])
            if direction_value == 1:
                mfe_points = max(mfe_points, high - entry)
                mae_points = max(mae_points, entry - low)
                hit_stop, hit_target = low <= stop, high >= target
            else:
                mfe_points = max(mfe_points, entry - low)
                mae_points = max(mae_points, high - entry)
                hit_stop, hit_target = high >= stop, low <= target
            # Frozen conservative rule: stop has priority on ambiguous candles.
            if hit_stop:
                exit_price = min(candle_open, stop) if direction_value == 1 else max(candle_open, stop)
                outcome, reason = "LOSS", "STOP_LOSS"
                break
            if hit_target:
                exit_price = max(candle_open, target) if direction_value == 1 else min(candle_open, target)
                outcome, reason = "WIN", "TARGET"
                break
        if exit_index == index:
            # Stage-1 currently blocks late entries, but retaining this guard
            # prevents an unobservable zero-horizon label.
            continue
        realized_r = direction_value * (exit_price - entry) / risk_points
        base = data.loc[index, FEATURES + SETUP_METADATA].to_dict()
        records.append({
            "timestamp": timestamps.iat[index],
            "direction": "LONG" if direction_value == 1 else "SHORT",
            "entry_signal": direction_value,
            "trade_allowed": 1,
            "entry_price": entry,
            "entry_atr": float(data.at[index, "atr_14"]),
            "entry_rsi": float(data.at[index, "rsi_14"]),
            **base,
            "exit_timestamp": timestamps.iat[exit_index],
            "exit_price": exit_price,
            "exit_candle": int(exit_index - index),
            "exit_reason": reason,
            "stop_price": stop,
            "target_price": target,
            "risk_points": risk_points,
            "mfe_points": mfe_points,
            "mae_points": mae_points,
            "mfe_r": mfe_points / risk_points,
            "mae_r": mae_points / risk_points,
            "realized_r": realized_r,
            # Per-row contribution whose arithmetic mean is aggregate Expected R.
            "expected_r_contribution": realized_r,
            "final_outcome": outcome,
            "label_horizon": contract.horizon,
            "contract_version": contract.version,
            "year": int(timestamps.iat[index].year),
            "session_segment": (
                "OPENING" if data.at[index, "minutes_from_open"] < 60 else
                "CLOSING" if data.at[index, "minutes_from_open"] >= 315 else "MID_SESSION"
            ),
        })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("Frozen Stage-1 produced no observable allowed setups")
    if not set(result.final_outcome.unique()).issubset({"WIN", "LOSS", "TIMEOUT"}):
        raise AssertionError("Stage-2 outcome contract was violated")
    return result


def build_diagnostic_decomposition(frame: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Audit-only decomposition; never an input to Stage-2 learning."""
    data = _validate(frame)
    outcome_lookup = outcomes.set_index("timestamp")[["final_outcome", "realized_r"]]
    aligned = data[["timestamp", "entry_signal", "trade_allowed", "trend_state",
                    "session_state", "rejection_reason", "session_id"]].copy()
    aligned = aligned.join(outcome_lookup, on="timestamp")
    subtype = np.full(len(aligned), "NO_SETUP", dtype=object)
    rejected = aligned.entry_signal.ne(0) & aligned.trade_allowed.eq(0)
    subtype[rejected] = "FILTER_REJECTED"
    allowed = aligned.trade_allowed.eq(1)
    long = aligned.entry_signal.eq(1)
    win = aligned.final_outcome.eq("WIN")
    loss = aligned.final_outcome.eq("LOSS")
    timeout = aligned.final_outcome.eq("TIMEOUT")
    subtype[allowed & long & win] = "BUY_CALL_WIN"
    subtype[allowed & ~long & win] = "BUY_PUT_WIN"
    subtype[allowed & long & loss] = "LONG_SETUP_LOSS"
    subtype[allowed & ~long & loss] = "SHORT_SETUP_LOSS"
    subtype[allowed & long & timeout] = "LONG_TIMEOUT"
    subtype[allowed & ~long & timeout] = "SHORT_TIMEOUT"
    aligned["diagnostic_subtype"] = subtype
    aligned["year"] = data.timestamp.dt.year
    aligned["session_segment"] = np.select(
        [data.minutes_from_open.lt(60), data.minutes_from_open.ge(315)],
        ["OPENING", "CLOSING"], default="MID_SESSION")
    return aligned


def summarize_outcomes(outcomes: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in outcomes.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(groups, keys)), "observations": len(group),
            "win_rate": float(group.final_outcome.eq("WIN").mean()),
            "loss_rate": float(group.final_outcome.eq("LOSS").mean()),
            "timeout_rate": float(group.final_outcome.eq("TIMEOUT").mean()),
            "expected_r": float(group.realized_r.mean()),
            "average_mfe_r": float(group.mfe_r.mean()),
            "average_mae_r": float(group.mae_r.mean()),
            "average_holding_candles": float(group.exit_candle.mean()),
        })
    return pd.DataFrame(rows)
