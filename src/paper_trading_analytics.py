"""Institutional summaries calculated only from completed paper trades."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "trade_id", "entry_timestamp", "exit_timestamp", "strategy", "regime",
    "probability", "entry_premium", "exit_premium", "quantity", "stop_loss",
    "target", "gross_pnl", "brokerage", "stt", "exchange_charges",
    "sebi_charges", "stamp_duty", "gst", "slippage", "total_costs",
    "net_pnl", "return_pct", "holding_time_minutes",
    "maximum_favourable_excursion", "maximum_adverse_excursion", "exit_reason",
]


def empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def performance(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "average_win": 0.0,
            "average_loss": 0.0, "profit_factor": 0.0, "sharpe_ratio": 0.0,
            "maximum_drawdown": 0.0, "expectancy": 0.0, "paper_cagr": 0.0,
            "consecutive_wins": 0, "consecutive_losses": 0,
            "average_holding_minutes": 0.0, "net_pnl": 0.0,
            "ending_equity": initial_capital,
        }
    data = trades.copy()
    pnl = pd.to_numeric(data.net_pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = initial_capital + pnl.cumsum()
    drawdown = equity / equity.cummax() - 1
    dates = pd.to_datetime(data.exit_timestamp)
    daily = data.assign(date=dates.dt.date).groupby("date").net_pnl.sum()
    daily_returns = daily / initial_capital
    sharpe = (
        np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1)
        if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else 0.0)
    years = max((dates.max() - dates.min()).total_seconds() / (365.25 * 86400), 1 / 365.25)
    ending = float(equity.iloc[-1])
    cagr = (ending / initial_capital) ** (1 / years) - 1 if ending > 0 else -1.0
    max_wins = max_losses = current_wins = current_losses = 0
    for value in pnl:
        if value > 0:
            current_wins, current_losses = current_wins + 1, 0
        else:
            current_losses, current_wins = current_losses + 1, 0
        max_wins, max_losses = max(max_wins, current_wins), max(max_losses, current_losses)
    return {
        "total_trades": len(data), "win_rate": float(pnl.gt(0).mean()),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "sharpe_ratio": float(sharpe), "maximum_drawdown": float(drawdown.min()),
        "expectancy": float(pnl.mean()), "paper_cagr": float(cagr),
        "consecutive_wins": max_wins, "consecutive_losses": max_losses,
        "average_holding_minutes": float(data.holding_time_minutes.mean()),
        "net_pnl": float(pnl.sum()), "ending_equity": ending,
    }


def time_summary(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "trades", "wins", "net_pnl",
                                     "gross_pnl", "costs", "win_rate"])
    data = trades.copy()
    timestamp = pd.to_datetime(data.exit_timestamp)
    data["period"] = timestamp.dt.to_period(frequency).astype(str)
    return data.groupby("period").agg(
        trades=("trade_id", "size"), wins=("net_pnl", lambda x: int((x > 0).sum())),
        net_pnl=("net_pnl", "sum"), gross_pnl=("gross_pnl", "sum"),
        costs=("total_costs", "sum"), win_rate=("net_pnl", lambda x: float((x > 0).mean())),
    ).reset_index()


def grouped_summary(trades: pd.DataFrame, field: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[field, "trades", "win_rate", "net_pnl",
                                     "average_trade", "profit_factor"])
    rows = []
    for value, data in trades.groupby(field, dropna=False):
        wins = data.loc[data.net_pnl > 0, "net_pnl"].sum()
        losses = -data.loc[data.net_pnl < 0, "net_pnl"].sum()
        rows.append({
            field: value, "trades": len(data),
            "win_rate": float(data.net_pnl.gt(0).mean()),
            "net_pnl": float(data.net_pnl.sum()),
            "average_trade": float(data.net_pnl.mean()),
            "profit_factor": float(wins / losses) if losses else float("inf"),
        })
    return pd.DataFrame(rows)


def equity_and_drawdown(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "trade_id", "net_pnl", "equity", "drawdown"])
    equity = initial_capital + trades.net_pnl.cumsum()
    return pd.DataFrame({
        "timestamp": trades.exit_timestamp, "trade_id": trades.trade_id,
        "net_pnl": trades.net_pnl, "equity": equity,
        "drawdown": equity / equity.cummax() - 1,
    })

