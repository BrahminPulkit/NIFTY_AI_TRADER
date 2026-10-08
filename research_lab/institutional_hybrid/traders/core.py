"""Deterministic research primitives shared only for identical measurement.

Signal rules live in each trader package. This module contains no optimiser and
never selects or tunes parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]


def guard_output(path: Path) -> Path:
    resolved = path.resolve()
    if LAB_ROOT not in resolved.parents and resolved != LAB_ROOT:
        raise ValueError(f"Research output outside lab refused: {resolved}")
    return resolved


def frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


@dataclass(frozen=True)
class FixedExecution:
    stop_pct: float = 0.04
    target_pct: float = 0.08
    max_hold_bars: int = 15
    round_trip_cost_pct: float = 0.002
    slippage_pct: float = 0.001


def backtest(signals: pd.DataFrame, execution: FixedExecution) -> pd.DataFrame:
    """One-position ATM-premium simulation using realized forward premiums."""
    required = {"signal", "premium", "atr_expansion", "strategy_score", "regime"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"Missing realized backtest fields: {sorted(missing)}")
    trades, i = [], 0
    while i < len(signals):
        row = signals.iloc[i]
        if row.signal == 0 or not np.isfinite(row.premium) or row.premium <= 0:
            i += 1
            continue
        entry = float(row.premium) * (1 + execution.slippage_pct)
        end, reason = min(i + execution.max_hold_bars, len(signals) - 1), "time"
        exit_i = end
        for j in range(i + 1, end + 1):
            mark = float(signals.iloc[j].premium) * (1 - execution.slippage_pct)
            ret = mark / entry - 1
            if ret >= execution.target_pct:
                exit_i, reason = j, "target"; break
            if ret <= -execution.stop_pct:
                exit_i, reason = j, "stop"; break
        exit_price = float(signals.iloc[exit_i].premium) * (1 - execution.slippage_pct)
        net = exit_price / entry - 1 - execution.round_trip_cost_pct
        window = signals.iloc[i:exit_i + 1].premium
        trades.append({
            "entry_time": signals.index[i], "exit_time": signals.index[exit_i],
            "entry_premium": entry, "exit_premium": exit_price, "net_return": net,
            "r_multiple": net / execution.stop_pct, "premium_expansion": exit_price / entry - 1,
            "maximum_premium_expansion": float(window.max() / entry - 1),
            "hold_bars": exit_i - i, "exit_reason": reason,
            "strategy_score": float(row.strategy_score), "atr_expansion": float(row.atr_expansion),
            "regime": row.regime, "volume": float(row.get("volume", np.nan)),
            "open_interest": float(row.get("open_interest", np.nan)),
        })
        i = exit_i + 1
    return pd.DataFrame(trades)


def streaks(wins: pd.Series) -> tuple[int, int]:
    max_w = max_l = run_w = run_l = 0
    for won in wins.astype(bool):
        run_w, run_l = (run_w + 1, 0) if won else (0, run_l + 1)
        max_w, max_l = max(max_w, run_w), max(max_l, run_l)
    return max_w, max_l


def metrics(trades: pd.DataFrame, rejected_pct: float = 0.0) -> dict[str, float]:
    names = (
        "total_trades win_rate loss_rate profit_factor expectancy average_r_multiple "
        "average_premium_expansion maximum_drawdown sharpe sortino calmar recovery_factor "
        "average_hold_time maximum_winning_streak maximum_losing_streak strategy_stability "
        "consistency_score rejected_setup_pct"
    ).split()
    if trades.empty:
        return {name: 0.0 for name in names}
    r = trades.net_return.astype(float)
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    downside = r[r < 0].std(ddof=1)
    std = r.std(ddof=1)
    span_days = max((pd.to_datetime(trades.exit_time).max() -
                     pd.to_datetime(trades.entry_time).min()).days, 1)
    annual_return = equity.iloc[-1] ** (365 / span_days) - 1
    max_dd = float(dd.min())
    max_w, max_l = streaks(r > 0)
    monthly = trades.assign(month=pd.to_datetime(trades.entry_time).dt.to_period("M")).groupby("month").net_return.sum()
    consistency = float((monthly > 0).mean()) if len(monthly) else 0.0
    return {
        "total_trades": float(len(trades)), "win_rate": float((r > 0).mean()),
        "loss_rate": float((r <= 0).mean()),
        "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf"),
        "expectancy": float(r.mean()), "average_r_multiple": float(trades.r_multiple.mean()),
        "average_premium_expansion": float(trades.premium_expansion.mean()),
        "maximum_drawdown": max_dd,
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std and np.isfinite(std) else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(252)) if downside and np.isfinite(downside) else 0.0,
        "calmar": float(annual_return / abs(max_dd)) if max_dd else 0.0,
        "recovery_factor": float((equity.iloc[-1] - 1) / abs(max_dd)) if max_dd else 0.0,
        "average_hold_time": float(trades.hold_bars.mean()),
        "maximum_winning_streak": float(max_w), "maximum_losing_streak": float(max_l),
        "strategy_stability": float(1 / (1 + monthly.std(ddof=1))) if len(monthly) > 1 else 0.0,
        "consistency_score": consistency, "rejected_setup_pct": float(rejected_pct),
    }


def write_json(path: Path, payload: dict) -> None:
    guard_output(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
