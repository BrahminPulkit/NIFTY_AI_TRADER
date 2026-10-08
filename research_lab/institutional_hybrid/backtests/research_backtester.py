"""Independent, single-position research backtester for ATM option scalps."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    max_hold_bars: int = 15
    target_pct: float = 0.08
    stop_pct: float = 0.04
    round_trip_cost_pct: float = 0.002
    cooldown_bars: int = 2


class ResearchBacktester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, signals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        required = {"signal", "premium", "atr_expansion", "institutional_score"}
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"Missing backtest columns: {sorted(missing)}")
        rows: list[dict] = []
        i = 0
        while i < len(signals):
            row = signals.iloc[i]
            if row["signal"] == 0 or not np.isfinite(row["premium"]) or row["premium"] <= 0:
                i += 1
                continue
            entry = float(row["premium"])
            end = min(i + self.config.max_hold_bars, len(signals) - 1)
            exit_i, reason = end, "time"
            for j in range(i + 1, end + 1):
                ret = float(signals.iloc[j]["premium"]) / entry - 1
                if ret >= self.config.target_pct:
                    exit_i, reason = j, "target"
                    break
                if ret <= -self.config.stop_pct:
                    exit_i, reason = j, "stop"
                    break
            exit_premium = float(signals.iloc[exit_i]["premium"])
            gross = exit_premium / entry - 1
            net = gross - self.config.round_trip_cost_pct
            window = signals.iloc[i:exit_i + 1]["premium"]
            rows.append({
                "entry_time": signals.index[i], "exit_time": signals.index[exit_i],
                "direction_context": int(row["signal"]), "entry_premium": entry,
                "exit_premium": exit_premium, "premium_expansion": gross,
                "net_return": net, "hold_bars": exit_i - i, "exit_reason": reason,
                "institutional_score": float(row["institutional_score"]),
                "atr_expansion": float(row["atr_expansion"]),
                "max_favorable_expansion": float(window.max() / entry - 1),
                "initial_risk": self.config.stop_pct,
                "initial_reward": self.config.target_pct,
                "trade_quality": float(row.get("entry_quality", row["institutional_score"])),
            })
            i = exit_i + self.config.cooldown_bars + 1
        trades = pd.DataFrame(rows)
        return trades, self.metrics(trades)

    @staticmethod
    def metrics(trades: pd.DataFrame) -> dict[str, float]:
        if trades.empty:
            return {k: 0.0 for k in (
                "trades", "win_rate", "profit_factor", "expectancy", "maximum_drawdown",
                "average_hold_time", "average_premium_expansion", "average_atr_expansion",
                "trade_frequency", "average_reward", "average_risk", "average_trade_quality",
            )}
        returns = trades["net_return"]
        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        span_days = max((pd.to_datetime(trades["exit_time"]).max() -
                         pd.to_datetime(trades["entry_time"]).min()).total_seconds() / 86400, 1)
        return {
            "trades": float(len(trades)),
            "win_rate": float((returns > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "expectancy": float(returns.mean()),
            "maximum_drawdown": float(drawdown.min()),
            "average_hold_time": float(trades["hold_bars"].mean()),
            "average_premium_expansion": float(trades["premium_expansion"].mean()),
            "average_atr_expansion": float(trades["atr_expansion"].mean()),
            "trade_frequency": float(len(trades) / span_days),
            "average_reward": float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0,
            "average_risk": float(-returns[returns < 0].mean()) if (returns < 0).any() else 0.0,
            "average_trade_quality": float(trades["trade_quality"].mean()),
        }
