"""Research-only decomposition of the frozen Step 20E decision funnel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.institutional_backtest import (
    CONFIG,
    PortfolioConfig,
    monte_carlo,
    monte_carlo_summary,
    position_notional,
    trading_costs,
)
from src.walk_forward_paper_trading import ReplayPolicy, replay_metrics


GATES = (
    "STAGE1",
    "DECISION_ENGINE",
    "PROBABILITY_THRESHOLD",
    "RISK_FILTER",
    "EXECUTION",
)


def build_funnel_universe(
    stage1: pd.DataFrame,
    predictions: pd.DataFrame,
    joined: pd.DataFrame,
    policy: ReplayPolicy,
) -> pd.DataFrame:
    """Attach frozen evidence by timestamp without recomputing any field."""
    setup = stage1.loc[stage1.entry_signal.ne(0)].copy()
    if setup.timestamp.duplicated().any():
        raise ValueError("Stage-1 setup timestamps must be unique")
    decision_columns = [
        "timestamp", "recommendation", "decision_reason", "current_strategy",
        "market_state", "index_atr_14", "history_count",
        "activated_strategy", "estimate_source",
        "terminal_return_95_lower", "expansion_drawdown_ratio",
    ]
    outcome_columns = [
        "timestamp", policy.probability_column, "terminal_premium_return",
        "premium_expansion", "premium_drawdown", "fold_id", "true_label",
        "market_regime", "strategy_family", "session_phase",
    ]
    data = setup.merge(
        predictions[decision_columns],
        on="timestamp", how="left", validate="one_to_one",
    ).merge(
        joined[outcome_columns],
        on="timestamp", how="left", validate="one_to_one",
        suffixes=("", "_oof"),
    )
    data["atr"] = data.index_atr_14.fillna(data.atr_14)
    data["regime"] = data.market_regime.fillna(data.market_state)
    data["strategy"] = data.strategy_family.fillna(data.current_strategy)

    data["first_failure_gate"] = "EXECUTED_CANDIDATE"
    data["first_failure_reason"] = "PASSED_ALL_SIGNAL_GATES"
    stage_fail = data.trade_allowed.ne(1)
    decision_missing = data.recommendation.isna()
    decision_fail = data.recommendation.eq("SKIP")
    probability_missing = data[policy.probability_column].isna()
    probability_fail = data[policy.probability_column].lt(policy.threshold)

    data.loc[stage_fail, ["first_failure_gate", "first_failure_reason"]] = [
        "STAGE1", "STAGE1_FILTER_REJECTED",
    ]
    mask = ~stage_fail & decision_missing
    data.loc[mask, ["first_failure_gate", "first_failure_reason"]] = [
        "DECISION_ENGINE", "FROZEN_DECISION_UNAVAILABLE",
    ]
    mask = ~stage_fail & decision_fail
    data.loc[mask, "first_failure_gate"] = "DECISION_ENGINE"
    data.loc[mask, "first_failure_reason"] = data.loc[mask, "decision_reason"]
    mask = ~stage_fail & data.recommendation.eq("TRADE") & probability_missing
    data.loc[mask, ["first_failure_gate", "first_failure_reason"]] = [
        "PROBABILITY_THRESHOLD", "FROZEN_OOF_PROBABILITY_UNAVAILABLE",
    ]
    mask = (
        ~stage_fail & data.recommendation.eq("TRADE")
        & ~probability_missing & probability_fail
    )
    data.loc[mask, ["first_failure_gate", "first_failure_reason"]] = [
        "PROBABILITY_THRESHOLD", "PROBABILITY_BELOW_THRESHOLD",
    ]
    data["gross_profitable"] = data.terminal_premium_return.gt(0)
    return data.sort_values("timestamp").reset_index(drop=True)


def _dominant(values: pd.Series) -> str:
    available = values.dropna().astype(str)
    return available.mode().iloc[0] if len(available) else "UNAVAILABLE"


def _gate_row(
    gate: str,
    incoming: pd.DataFrame,
    rejected: pd.DataFrame,
    passed: pd.DataFrame,
    probability_column: str,
) -> dict:
    def mean(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].mean()) if frame[column].notna().any() else np.nan

    return {
        "gate": gate,
        "total_incoming": len(incoming),
        "total_rejected": len(rejected),
        "total_passed": len(passed),
        "rejection_pct": len(rejected) / len(incoming) if len(incoming) else np.nan,
        "observed_probability_rows": int(incoming[probability_column].notna().sum()),
        "observed_outcome_rows": int(incoming.terminal_premium_return.notna().sum()),
        "average_probability": mean(incoming, probability_column),
        "average_premium_expansion": mean(incoming, "premium_expansion"),
        "average_premium_return": mean(incoming, "terminal_premium_return"),
        "average_atr": mean(incoming, "atr"),
        "dominant_regime": _dominant(incoming.regime),
        "dominant_strategy_family": _dominant(incoming.strategy),
    }


def build_decision_funnel(
    universe: pd.DataFrame,
    executed_entries: set[pd.Timestamp],
    policy: ReplayPolicy,
) -> pd.DataFrame:
    probability = policy.probability_column
    stage_in = universe
    stage_pass = stage_in.loc[stage_in.trade_allowed.eq(1)]
    stage_reject = stage_in.loc[stage_in.trade_allowed.ne(1)]

    decision_in = stage_pass
    decision_pass = decision_in.loc[decision_in.recommendation.eq("TRADE")]
    decision_reject = decision_in.loc[~decision_in.recommendation.eq("TRADE")]

    probability_in = decision_pass
    probability_pass = probability_in.loc[
        probability_in[probability].notna()
        & probability_in[probability].ge(policy.threshold)
    ]
    probability_reject = probability_in.drop(probability_pass.index)

    valid = (
        probability_pass.terminal_premium_return.notna()
        & probability_pass.terminal_premium_return.ge(-1)
        & (
            probability_pass.timestamp + pd.Timedelta(minutes=policy.exit_minutes)
        ).dt.time.le(pd.Timestamp("15:30").time())
    )
    risk_in = probability_pass
    risk_pass = risk_in.loc[valid]
    risk_reject = risk_in.loc[~valid]

    execution_in = risk_pass
    execution_pass = execution_in.loc[
        execution_in.timestamp.isin(executed_entries)
    ]
    execution_reject = execution_in.drop(execution_pass.index)
    rows = [
        _gate_row("STAGE1", stage_in, stage_reject, stage_pass, probability),
        _gate_row(
            "DECISION_ENGINE", decision_in, decision_reject, decision_pass,
            probability,
        ),
        _gate_row(
            "PROBABILITY_THRESHOLD", probability_in, probability_reject,
            probability_pass, probability,
        ),
        _gate_row("RISK_FILTER", risk_in, risk_reject, risk_pass, probability),
        _gate_row(
            "EXECUTION", execution_in, execution_reject, execution_pass,
            probability,
        ),
    ]
    return pd.DataFrame(rows)


def add_net_profitability(
    universe: pd.DataFrame,
    config: PortfolioConfig = CONFIG,
) -> pd.DataFrame:
    data = universe.copy()
    notional = position_notional(config.initial_capital, config)

    def net_return(value) -> float:
        if pd.isna(value) or value < -1:
            return np.nan
        charge = trading_costs(notional, float(value), config)
        return float(value) - charge["total_costs"] / notional

    data["counterfactual_net_return"] = data.terminal_premium_return.map(net_return)
    data["net_profitable"] = data.counterfactual_net_return.gt(0)
    return data


def simulate_selection(
    selected: pd.DataFrame,
    policy: ReplayPolicy,
    config: PortfolioConfig = CONFIG,
    prevent_overlap: bool = True,
) -> pd.DataFrame:
    """Simulate a fixed frozen selection; outcomes are settled after 15 minutes."""
    capital = config.initial_capital
    last_exit = None
    trades = []
    for row in selected.sort_values("timestamp").itertuples(index=False):
        entry = row.timestamp
        exit_time = entry + pd.Timedelta(minutes=policy.exit_minutes)
        if prevent_overlap and last_exit is not None and entry < last_exit:
            continue
        gross_return = float(row.terminal_premium_return)
        if not np.isfinite(gross_return) or gross_return < -1:
            continue
        notional = position_notional(capital, config)
        if notional <= 0 or exit_time.time() > pd.Timestamp("15:30").time():
            continue
        charge = trading_costs(notional, gross_return, config)
        gross_pnl = notional * gross_return
        net_pnl = gross_pnl - charge["total_costs"]
        before = capital
        capital = max(0.0, capital + net_pnl)
        trades.append({
            "entry_timestamp": entry,
            "exit_timestamp": exit_time,
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "total_costs": charge["total_costs"],
            "capital_before": before,
            "capital_after": capital,
            "portfolio_return": net_pnl / before if before else 0.0,
            "win": net_pnl > 0,
        })
        last_exit = exit_time
    return pd.DataFrame(trades)


def counterfactual_scenarios(
    universe: pd.DataFrame,
    policy: ReplayPolicy,
) -> dict[str, tuple[pd.DataFrame | None, str]]:
    probability = policy.probability_column
    observed = universe.loc[
        universe[probability].notna()
        & universe.terminal_premium_return.notna()
    ]
    decision = observed.recommendation.eq("TRADE")
    threshold = observed[probability].ge(policy.threshold)
    return {
        "BASELINE": (
            simulate_selection(observed.loc[decision & threshold], policy),
            "all frozen filters",
        ),
        "REMOVE_STAGE1": (
            None,
            "unevaluable: rejected Stage-1 rows have no frozen OOF probability/outcome",
        ),
        "REMOVE_DECISION_ENGINE": (
            simulate_selection(observed.loc[threshold], policy),
            "Stage-1 allowed + probability threshold; Decision Engine bypassed",
        ),
        "REMOVE_PROBABILITY_THRESHOLD": (
            simulate_selection(observed.loc[decision], policy),
            "Stage-1 allowed + Decision Engine TRADE; threshold bypassed",
        ),
        "REMOVE_RISK_FILTER": (
            simulate_selection(observed.loc[decision & threshold], policy),
            "all observed baseline rows are valid; risk filter bypass has no effect",
        ),
        "REMOVE_OVERLAP_FILTER": (
            simulate_selection(
                observed.loc[decision & threshold], policy,
                prevent_overlap=False,
            ),
            "diagnostic only: overlapping capital use is economically impossible",
        ),
    }


def summarize_counterfactuals(
    scenarios: dict[str, tuple[pd.DataFrame | None, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    simulations: int = 1000,
) -> pd.DataFrame:
    rows = []
    for index, (name, (trades, note)) in enumerate(scenarios.items()):
        if trades is None:
            rows.append({
                "scenario": name, "evaluable": False, "note": note,
                "trades": np.nan,
            })
            continue
        metrics = replay_metrics(
            trades, CONFIG.initial_capital, start, end
        )
        distribution = monte_carlo(
            trades, simulations=simulations,
            initial_capital=CONFIG.initial_capital,
            seed=CONFIG.random_seed + index,
        )
        mc = monte_carlo_summary(distribution)
        rows.append({
            "scenario": name,
            "evaluable": True,
            "note": note,
            "trades": len(trades),
            **metrics,
            "monte_carlo_simulations": simulations if len(trades) else 0,
            "mc_worst_case": mc.get("worst_case", np.nan),
            "mc_percentile_5": mc.get("percentile_5", np.nan),
            "mc_median": mc.get("median_case", np.nan),
            "mc_percentile_95": mc.get("percentile_95", np.nan),
            "mc_best_case": mc.get("best_case", np.nan),
        })
    return pd.DataFrame(rows)
