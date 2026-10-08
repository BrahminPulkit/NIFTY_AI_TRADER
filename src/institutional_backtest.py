"""Step 20D: research-only portfolio simulation over frozen OOF signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    position_sizing_mode: str = "fixed_risk"
    risk_per_trade_pct: float = 0.01
    assumed_premium_risk_pct: float = 0.20
    maximum_position_pct: float = 0.10
    fixed_notional: float = 50_000.0
    brokerage_per_order: float = 20.0
    stt_sell_rate: float = 0.0015
    exchange_rate_each_side: float = 0.0003553
    sebi_rate_each_side: float = 0.000001
    stamp_duty_buy_rate: float = 0.00003
    gst_rate: float = 0.18
    slippage_bps_each_side: float = 2.0
    monte_carlo_simulations: int = 1000
    random_seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = PortfolioConfig()


def position_notional(capital: float, config: PortfolioConfig = CONFIG) -> float:
    if config.position_sizing_mode == "fixed_risk":
        risk_budget = capital * config.risk_per_trade_pct
        desired = risk_budget / config.assumed_premium_risk_pct
    elif config.position_sizing_mode == "equity_fraction":
        desired = capital * config.maximum_position_pct
    elif config.position_sizing_mode == "fixed_notional":
        desired = config.fixed_notional
    else:
        raise ValueError(f"Unsupported position sizing mode: {config.position_sizing_mode}")
    return float(max(0.0, min(desired, capital * config.maximum_position_pct, capital)))


def trading_costs(
    entry_turnover: float,
    gross_return: float,
    config: PortfolioConfig = CONFIG,
) -> dict:
    exit_turnover_before_slippage = max(0.0, entry_turnover * (1 + gross_return))
    slippage = (
        entry_turnover + exit_turnover_before_slippage
    ) * config.slippage_bps_each_side / 10_000
    brokerage = 2 * config.brokerage_per_order
    exchange = (
        entry_turnover + exit_turnover_before_slippage
    ) * config.exchange_rate_each_side
    sebi = (
        entry_turnover + exit_turnover_before_slippage
    ) * config.sebi_rate_each_side
    stt = exit_turnover_before_slippage * config.stt_sell_rate
    stamp = entry_turnover * config.stamp_duty_buy_rate
    gst = config.gst_rate * (brokerage + exchange + sebi)
    total = brokerage + exchange + sebi + stt + stamp + gst + slippage
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_charges": exchange,
        "sebi_charges": sebi,
        "stamp_duty": stamp,
        "gst": gst,
        "slippage": slippage,
        "total_costs": total,
    }


def simulate_scenario(
    data: pd.DataFrame,
    model: str,
    threshold: float,
    config: PortfolioConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_column = f"{model.lower()}_predicted_probability"
    if probability_column not in data:
        raise ValueError(f"Missing frozen probability: {probability_column}")
    selected = data.loc[data[probability_column].ge(threshold)].sort_values("timestamp")
    capital = config.initial_capital
    last_exit = None
    trades = []
    skipped = []
    for row in selected.itertuples(index=False):
        entry_time = row.timestamp
        holding_minutes = max(float(row.holding_time_minutes), 0.0)
        exit_time = entry_time + pd.Timedelta(minutes=holding_minutes)
        if last_exit is not None and entry_time < last_exit:
            skipped.append({
                "timestamp": entry_time, "model": model, "threshold": threshold,
                "skip_reason": "OVERLAPPING_ACTIVE_TRADE",
            })
            continue
        gross_return = float(row.terminal_premium_return)
        if not np.isfinite(gross_return) or gross_return < -1:
            skipped.append({
                "timestamp": entry_time, "model": model, "threshold": threshold,
                "skip_reason": "INVALID_OR_IMPOSSIBLE_RETURN",
            })
            continue
        notional = position_notional(capital, config)
        if notional <= 0:
            skipped.append({
                "timestamp": entry_time, "model": model, "threshold": threshold,
                "skip_reason": "INSUFFICIENT_CAPITAL",
            })
            continue
        costs = trading_costs(notional, gross_return, config)
        gross_pnl = notional * gross_return
        net_pnl = gross_pnl - costs["total_costs"]
        capital_before = capital
        capital = max(0.0, capital + net_pnl)
        portfolio_return = net_pnl / capital_before if capital_before else 0.0
        trades.append({
            "model": model,
            "threshold": threshold,
            "entry_timestamp": entry_time,
            "exit_timestamp": exit_time,
            "probability": float(getattr(row, probability_column)),
            "true_label": int(row.true_label),
            "strategy_family": row.strategy_family,
            "market_regime": row.market_regime,
            "session_phase": row.session_phase,
            "holding_time_minutes": holding_minutes,
            "capital_before": capital_before,
            "position_notional": notional,
            "gross_premium_return": gross_return,
            "gross_pnl": gross_pnl,
            **costs,
            "net_pnl": net_pnl,
            "portfolio_return": portfolio_return,
            "capital_after": capital,
        })
        last_exit = exit_time
    return pd.DataFrame(trades), pd.DataFrame(skipped)


def _maximum_consecutive_losses(values: pd.Series) -> int:
    maximum = current = 0
    for loss in values.lt(0):
        current = current + 1 if loss else 0
        maximum = max(maximum, current)
    return maximum


def performance_metrics(
    trades: pd.DataFrame,
    selected_signals: int,
    config: PortfolioConfig = CONFIG,
) -> dict:
    if trades.empty:
        return {
            "executed_trades": 0, "selected_signals": selected_signals,
            "overlap_or_invalid_skips": selected_signals,
            "net_return": np.nan, "cagr": np.nan, "maximum_drawdown": np.nan,
            "profit_factor": np.nan, "win_rate": np.nan, "average_trade": np.nan,
            "expectancy": np.nan, "sharpe_ratio": np.nan, "sortino_ratio": np.nan,
            "calmar_ratio": np.nan, "maximum_consecutive_losses": 0,
        }
    equity = trades.set_index("exit_timestamp").capital_after
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    elapsed_days = max(
        (trades.exit_timestamp.max() - trades.entry_timestamp.min()).total_seconds()
        / 86_400,
        1,
    )
    years = elapsed_days / 365.25
    ending = float(equity.iloc[-1])
    net_return = ending / config.initial_capital - 1
    cagr = (
        (ending / config.initial_capital) ** (1 / years) - 1
        if ending > 0 else -1.0
    )
    gains = trades.loc[trades.net_pnl.gt(0), "net_pnl"].sum()
    losses = -trades.loc[trades.net_pnl.lt(0), "net_pnl"].sum()
    daily = equity.groupby(equity.index.normalize()).last().pct_change().dropna()
    sharpe = (
        np.sqrt(252) * daily.mean() / daily.std(ddof=1)
        if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    )
    downside = daily.loc[daily.lt(0)]
    sortino = (
        np.sqrt(252) * daily.mean() / downside.std(ddof=1)
        if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    )
    maximum_drawdown = float(drawdown.min())
    return {
        "executed_trades": len(trades),
        "selected_signals": selected_signals,
        "overlap_or_invalid_skips": selected_signals - len(trades),
        "net_return": net_return,
        "cagr": cagr,
        "maximum_drawdown": maximum_drawdown,
        "profit_factor": gains / losses if losses > 0 else np.inf,
        "win_rate": float(trades.net_pnl.gt(0).mean()),
        "average_trade": float(trades.net_pnl.mean()),
        "expectancy": float(trades.portfolio_return.mean()),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.inf,
        "maximum_consecutive_losses": _maximum_consecutive_losses(trades.net_pnl),
        "total_costs": float(trades.total_costs.sum()),
        "gross_profit": float(trades.gross_pnl.sum()),
        "net_profit": float(trades.net_pnl.sum()),
        "ending_capital": ending,
    }


def monte_carlo(
    trades: pd.DataFrame,
    simulations: int = 1000,
    initial_capital: float = 1_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    returns = trades.portfolio_return.to_numpy(float)
    random = np.random.default_rng(seed)
    rows = []
    for simulation in range(simulations):
        sample = random.choice(returns, size=len(returns), replace=True)
        path = initial_capital * np.cumprod(1 + sample)
        running = np.maximum.accumulate(path)
        drawdown = path / running - 1
        rows.append({
            "simulation": simulation + 1,
            "ending_equity": float(path[-1]),
            "net_return": float(path[-1] / initial_capital - 1),
            "maximum_drawdown": float(drawdown.min()),
        })
    return pd.DataFrame(rows)


def monte_carlo_summary(distribution: pd.DataFrame) -> dict:
    if distribution.empty:
        return {}
    return {
        "worst_case": float(distribution.net_return.min()),
        "percentile_5": float(distribution.net_return.quantile(0.05)),
        "percentile_25": float(distribution.net_return.quantile(0.25)),
        "median_case": float(distribution.net_return.quantile(0.50)),
        "percentile_50": float(distribution.net_return.quantile(0.50)),
        "percentile_75": float(distribution.net_return.quantile(0.75)),
        "percentile_95": float(distribution.net_return.quantile(0.95)),
        "best_case": float(distribution.net_return.max()),
        "median_maximum_drawdown": float(distribution.maximum_drawdown.median()),
        "percentile_5_maximum_drawdown": float(
            distribution.maximum_drawdown.quantile(0.05)
        ),
    }

