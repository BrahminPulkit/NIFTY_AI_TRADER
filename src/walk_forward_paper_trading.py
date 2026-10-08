"""Step 20E: deterministic replay over frozen, leakage-free artifacts.

The replay never calls a model or recalculates a probability.  A frozen
terminal response is settled only when its frozen 15-minute research horizon
has elapsed in the chronological candle stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.institutional_backtest import (
    CONFIG as COST_CONFIG,
    PortfolioConfig,
    position_notional,
    trading_costs,
)


RESEARCH_EXIT_MINUTES = 15


@dataclass(frozen=True)
class ReplayPolicy:
    model: str = "CATBOOST"
    threshold: float = 0.90
    exit_minutes: int = RESEARCH_EXIT_MINUTES

    @property
    def probability_column(self) -> str:
        return f"{self.model.lower()}_predicted_probability"


POLICY = ReplayPolicy()


def _timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata"
    )


def _validate_inputs(
    stage1: pd.DataFrame,
    predictions: pd.DataFrame,
    joined: pd.DataFrame,
    policy: ReplayPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_required = {
        "timestamp", "entry_signal", "trade_allowed", "rejection_reason",
        "session_id",
    }
    prediction_required = {
        "timestamp", "recommendation", "decision_reason",
        "current_strategy", "market_state",
    }
    joined_required = {
        "timestamp", policy.probability_column, "terminal_premium_return",
        "true_label", "fold_id", "premium_expansion", "premium_drawdown",
    }
    for name, frame, required in (
        ("Stage-1", stage1, stage_required),
        ("Prediction", predictions, prediction_required),
        ("OOF join", joined, joined_required),
    ):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{name} missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} is empty")

    stage = stage1.copy()
    prediction = predictions.copy()
    oof = joined.copy()
    for frame in (stage, prediction, oof):
        frame["timestamp"] = _timestamps(frame["timestamp"])
        frame.sort_values("timestamp", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame.timestamp.duplicated().any():
            raise ValueError("Frozen input timestamps must be unique")
        if not frame.timestamp.is_monotonic_increasing:
            raise ValueError("Frozen input timestamps must be chronological")
    if not oof[policy.probability_column].between(0, 1).all():
        raise ValueError("Frozen OOF probability is outside [0, 1]")
    return stage, prediction, oof


def replay_frozen_stream(
    stage1: pd.DataFrame,
    predictions: pd.DataFrame,
    joined: pd.DataFrame,
    policy: ReplayPolicy = POLICY,
    costs: PortfolioConfig = COST_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay every Stage-1 candle and settle at the declared research horizon.

    Returns completed trades, every rejected setup opportunity, and a
    chronological realized-equity event stream.
    """
    stage, prediction, oof = _validate_inputs(
        stage1, predictions, joined, policy
    )
    decision_lookup = prediction.set_index("timestamp")
    oof_lookup = oof.set_index("timestamp")
    capital = float(costs.initial_capital)
    active: dict | None = None
    trades: list[dict] = []
    skipped: list[dict] = []
    equity_events = [{
        "timestamp": stage.timestamp.iloc[0],
        "equity": capital,
        "event": "INITIAL_CAPITAL",
    }]

    def reject(timestamp, entry_signal, category, detail, **extra) -> None:
        skipped.append({
            "timestamp": timestamp,
            "entry_signal": int(entry_signal),
            "skip_category": category,
            "skip_reason": str(detail),
            **extra,
        })

    for candle in stage.itertuples(index=False):
        now = candle.timestamp

        # Settlement occurs only after the frozen response horizon is observable.
        if active is not None and now >= active["scheduled_exit_timestamp"]:
            gross_return = active["terminal_premium_return"]
            charge = trading_costs(active["position_notional"], gross_return, costs)
            gross_pnl = active["position_notional"] * gross_return
            net_pnl = gross_pnl - charge["total_costs"]
            before = capital
            capital = max(0.0, capital + net_pnl)
            trades.append({
                **active,
                "exit_timestamp": now,
                "capital_before": before,
                "gross_pnl": gross_pnl,
                **charge,
                "net_pnl": net_pnl,
                "portfolio_return": net_pnl / before if before else 0.0,
                "capital_after": capital,
                "win": bool(net_pnl > 0),
            })
            equity_events.append({
                "timestamp": now, "equity": capital, "event": "TRADE_EXIT",
            })
            active = None

        # A zero signal is a normal candle, not a rejected trade opportunity.
        if int(candle.entry_signal) == 0:
            continue
        if int(candle.trade_allowed) != 1:
            reject(
                now, candle.entry_signal, "STAGE1_REJECTED",
                candle.rejection_reason,
            )
            continue
        if now not in decision_lookup.index:
            reject(
                now, candle.entry_signal, "DECISION_ENGINE_REJECTED",
                "FROZEN_DECISION_UNAVAILABLE",
            )
            continue

        decision = decision_lookup.loc[now]
        if decision.recommendation != "TRADE":
            reject(
                now, candle.entry_signal, "DECISION_ENGINE_REJECTED",
                decision.decision_reason,
                strategy_family=decision.current_strategy,
                market_state=decision.market_state,
            )
            continue
        if now not in oof_lookup.index:
            reject(
                now, candle.entry_signal, "RISK_CONTROL",
                "FROZEN_OOF_PROBABILITY_UNAVAILABLE",
                strategy_family=decision.current_strategy,
                market_state=decision.market_state,
            )
            continue

        response = oof_lookup.loc[now]
        probability = float(response[policy.probability_column])
        if probability < policy.threshold:
            reject(
                now, candle.entry_signal, "PROBABILITY_BELOW_THRESHOLD",
                f"{probability:.12g} < {policy.threshold:.12g}",
                probability=probability,
                strategy_family=decision.current_strategy,
                market_state=decision.market_state,
            )
            continue
        if active is not None:
            reject(
                now, candle.entry_signal, "CAPITAL_UNAVAILABLE",
                "ONE_ACTIVE_TRADE_LIMIT",
                probability=probability,
                strategy_family=decision.current_strategy,
                market_state=decision.market_state,
            )
            continue

        terminal_return = float(response.terminal_premium_return)
        notional = position_notional(capital, costs)
        scheduled_exit = now + pd.Timedelta(minutes=policy.exit_minutes)
        same_session_exit = (
            scheduled_exit.normalize() == now.normalize()
            and scheduled_exit.time() <= pd.Timestamp("15:30").time()
        )
        if (
            not np.isfinite(terminal_return)
            or terminal_return < -1
            or notional <= 0
            or not same_session_exit
        ):
            reason = (
                "INVALID_FROZEN_RETURN"
                if not np.isfinite(terminal_return) or terminal_return < -1
                else "INSUFFICIENT_CAPITAL"
                if notional <= 0
                else "EXIT_OUTSIDE_SESSION"
            )
            reject(
                now, candle.entry_signal, "RISK_CONTROL", reason,
                probability=probability,
                strategy_family=decision.current_strategy,
                market_state=decision.market_state,
            )
            continue

        active = {
            "entry_timestamp": now,
            "scheduled_exit_timestamp": scheduled_exit,
            "model": policy.model,
            "threshold": policy.threshold,
            "fold_id": int(response.fold_id),
            "probability": probability,
            "entry_signal": int(candle.entry_signal),
            "strategy_family": response.strategy_family,
            "market_regime": response.market_regime,
            "session_phase": response.session_phase,
            "true_label": int(response.true_label),
            "position_notional": notional,
            "terminal_premium_return": terminal_return,
            "premium_expansion": float(response.premium_expansion),
            "premium_drawdown": float(response.premium_drawdown),
        }

    if active is not None:
        reject(
            active["entry_timestamp"], active["entry_signal"], "RISK_CONTROL",
            "DATASET_ENDED_BEFORE_EXIT", probability=active["probability"],
            strategy_family=active["strategy_family"],
            market_state=active["market_regime"],
        )

    return (
        pd.DataFrame(trades),
        pd.DataFrame(skipped),
        pd.DataFrame(equity_events),
    )


def _maximum_streak(values: pd.Series, desired: bool) -> int:
    maximum = current = 0
    for value in values.astype(bool):
        current = current + 1 if value is desired else 0
        maximum = max(maximum, current)
    return maximum


def replay_metrics(
    trades: pd.DataFrame,
    initial_capital: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    daily_returns: pd.Series | None = None,
) -> dict:
    if trades.empty:
        return {
            "executed_trades": 0, "net_return": 0.0, "cagr": 0.0,
            "win_rate": np.nan, "average_winner": np.nan,
            "average_loser": np.nan, "expectancy": np.nan,
            "profit_factor": np.nan, "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan, "calmar_ratio": np.nan,
            "maximum_drawdown": 0.0, "recovery_factor": np.nan,
            "maximum_consecutive_wins": 0, "maximum_consecutive_losses": 0,
        }
    ending = float(trades.capital_after.iloc[-1])
    net_return = ending / initial_capital - 1
    years = max((end - start).total_seconds() / (365.25 * 86_400), 1 / 365.25)
    cagr = (ending / initial_capital) ** (1 / years) - 1 if ending > 0 else -1.0
    equity = trades.set_index("exit_timestamp").capital_after
    drawdown = equity / equity.cummax().clip(lower=initial_capital) - 1
    daily = (
        daily_returns.dropna()
        if daily_returns is not None
        else equity.groupby(equity.index.normalize()).last().pct_change().dropna()
    )
    standard_deviation = daily.std(ddof=1)
    downside = daily.loc[daily.lt(0)].std(ddof=1)
    sharpe = (
        np.sqrt(252) * daily.mean() / standard_deviation
        if len(daily) > 1 and standard_deviation > 0 else np.nan
    )
    sortino = (
        np.sqrt(252) * daily.mean() / downside
        if len(daily.loc[daily.lt(0)]) > 1 and downside > 0 else np.nan
    )
    gains = trades.loc[trades.net_pnl.gt(0), "net_pnl"]
    losses = trades.loc[trades.net_pnl.lt(0), "net_pnl"]
    maximum_drawdown = float(drawdown.min())
    return {
        "executed_trades": len(trades),
        "net_return": net_return,
        "cagr": cagr,
        "win_rate": float(trades.win.mean()),
        "average_winner": float(gains.mean()) if len(gains) else np.nan,
        "average_loser": float(losses.mean()) if len(losses) else np.nan,
        "expectancy": float(trades.net_pnl.mean()),
        "profit_factor": (
            float(gains.sum() / -losses.sum()) if len(losses) else np.inf
        ),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": (
            cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.inf
        ),
        "maximum_drawdown": maximum_drawdown,
        "recovery_factor": (
            net_return / abs(maximum_drawdown)
            if maximum_drawdown < 0 else np.inf
        ),
        "maximum_consecutive_wins": _maximum_streak(trades.win, True),
        "maximum_consecutive_losses": _maximum_streak(trades.win, False),
        "ending_capital": ending,
        "net_profit": float(trades.net_pnl.sum()),
        "total_costs": float(trades.total_costs.sum()),
    }
