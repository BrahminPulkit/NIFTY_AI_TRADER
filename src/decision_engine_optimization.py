"""Step 20G research over frozen Decision Engine fields and OOF outcomes."""

from __future__ import annotations

from itertools import chain, combinations

import numpy as np
import pandas as pd

from src.decision_funnel_research import simulate_selection
from src.institutional_backtest import CONFIG, monte_carlo, monte_carlo_summary
from src.walk_forward_paper_trading import ReplayPolicy, replay_metrics


RULES = (
    "STRATEGY_MATCH",
    "POSITIVE_CONFIDENCE_LOWER_BOUND",
    "EXPANSION_GT_DRAWDOWN",
)
SOURCE_MODES = {
    "ALL_FALLBACKS": {"EXACT", "REGIME", "STRATEGY"},
    "EXACT_ONLY": {"EXACT"},
    "REGIME_ONLY": {"REGIME"},
    "STRATEGY_ONLY": {"STRATEGY"},
    "EXACT_OR_REGIME": {"EXACT", "REGIME"},
    "REGIME_OR_STRATEGY": {"REGIME", "STRATEGY"},
}


def all_rule_subsets() -> list[tuple[str, ...]]:
    return [
        tuple(values)
        for values in chain.from_iterable(
            combinations(RULES, size) for size in range(len(RULES) + 1)
        )
    ]


def rule_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ESTIMATE_AVAILABLE": (
            data.history_count.ge(30)
            & data.activated_strategy.ne("NONE")
            & data.estimate_source.isin({"EXACT", "REGIME", "STRATEGY"})
        ),
        "STRATEGY_MATCH": data.current_strategy.eq(data.activated_strategy),
        "POSITIVE_CONFIDENCE_LOWER_BOUND": data.terminal_return_95_lower.gt(0),
        "EXPANSION_GT_DRAWDOWN": data.expansion_drawdown_ratio.gt(1),
    }


def candidate_mask(
    data: pd.DataFrame,
    enabled_rules: tuple[str, ...],
    source_mode: str,
) -> pd.Series:
    masks = rule_masks(data)
    result = masks["ESTIMATE_AVAILABLE"] & data.estimate_source.isin(
        SOURCE_MODES[source_mode]
    )
    for rule in enabled_rules:
        result &= masks[rule]
    return result


def candidate_definitions() -> list[dict]:
    definitions = []
    for source_mode in SOURCE_MODES:
        for enabled in all_rule_subsets():
            name = (
                source_mode + "__"
                + ("+".join(enabled) if enabled else "ESTIMATE_ONLY")
            )
            definitions.append({
                "candidate_id": name,
                "source_mode": source_mode,
                "enabled_rules": enabled,
                "evaluable": True,
                "note": "evaluated from frozen entry-time fields",
            })
    definitions.extend([
        {
            "candidate_id": "MIN_HISTORY_20",
            "source_mode": "ALL_FALLBACKS",
            "enabled_rules": RULES,
            "evaluable": False,
            "note": "frozen rows below 30 do not retain candidate estimates",
        },
        {
            "candidate_id": "MIN_HISTORY_15",
            "source_mode": "ALL_FALLBACKS",
            "enabled_rules": RULES,
            "evaluable": False,
            "note": "frozen rows below 30 do not retain candidate estimates",
        },
    ])
    return definitions


def evaluate_candidates(
    data: pd.DataFrame,
    policy: ReplayPolicy,
    start: pd.Timestamp,
    end: pd.Timestamp,
    simulations: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_gate = data[policy.probability_column].ge(policy.threshold)
    candidate_rows = []
    mc_rows = []
    for number, definition in enumerate(candidate_definitions()):
        if not definition["evaluable"]:
            candidate_rows.append({
                **definition,
                "enabled_rules": "+".join(definition["enabled_rules"]),
                "trades": np.nan,
            })
            continue
        decision = candidate_mask(
            data, definition["enabled_rules"], definition["source_mode"]
        )
        selected = data.loc[decision & probability_gate]
        trades = simulate_selection(selected, policy)
        metrics = replay_metrics(trades, CONFIG.initial_capital, start, end)
        yearly = []
        if len(trades):
            trade_year = trades.entry_timestamp.dt.year
            yearly = [
                float(group.net_pnl.mean())
                for _, group in trades.groupby(trade_year)
            ]
        stability = (
            float(np.mean(np.asarray(yearly) > 0)) if yearly else np.nan
        )
        expectancy_stability = (
            float(np.std(yearly, ddof=1)) if len(yearly) > 1 else np.nan
        )
        distribution = monte_carlo(
            trades, simulations=simulations,
            initial_capital=CONFIG.initial_capital,
            seed=CONFIG.random_seed + number,
        )
        mc = monte_carlo_summary(distribution)
        row = {
            **definition,
            "enabled_rules": "+".join(definition["enabled_rules"]),
            "trades": len(trades),
            **metrics,
            "yearly_positive_fraction": stability,
            "yearly_expectancy_std": expectancy_stability,
            "mc_percentile_5": mc.get("percentile_5", np.nan),
            "mc_median": mc.get("median_case", np.nan),
            "mc_percentile_95": mc.get("percentile_95", np.nan),
        }
        candidate_rows.append(row)
        mc_rows.append({
            "candidate_id": definition["candidate_id"],
            "simulations": simulations if len(trades) else 0,
            **mc,
        })
    candidates = pd.DataFrame(candidate_rows)
    return candidates, pd.DataFrame(mc_rows)


def pareto_frontier(candidates: pd.DataFrame) -> pd.DataFrame:
    pool = candidates.loc[
        candidates.evaluable.eq(True)
        & candidates.trades.ge(20)
        & candidates.cagr.notna()
        & candidates.profit_factor.notna()
        & candidates.maximum_drawdown.notna()
    ].copy()
    rows = []
    for index, row in pool.iterrows():
        dominated = (
            (pool.cagr.ge(row.cagr))
            & (pool.maximum_drawdown.ge(row.maximum_drawdown))
            & (pool.profit_factor.ge(row.profit_factor))
            & (pool.yearly_positive_fraction.ge(row.yearly_positive_fraction))
            & (
                (pool.cagr.gt(row.cagr))
                | (pool.maximum_drawdown.gt(row.maximum_drawdown))
                | (pool.profit_factor.gt(row.profit_factor))
                | (
                    pool.yearly_positive_fraction.gt(
                        row.yearly_positive_fraction
                    )
                )
            )
        )
        if not dominated.drop(index=index).any():
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cagr", "maximum_drawdown", "profit_factor",
         "yearly_positive_fraction"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def independent_rule_importance(
    data: pd.DataFrame,
    candidates: pd.DataFrame,
    policy: ReplayPolicy,
) -> pd.DataFrame:
    masks = rule_masks(data)
    original = (
        masks["ESTIMATE_AVAILABLE"]
        & masks["STRATEGY_MATCH"]
        & masks["POSITIVE_CONFIDENCE_LOWER_BOUND"]
        & masks["EXPANSION_GT_DRAWDOWN"]
    )
    rows = []
    baseline = candidates.loc[
        candidates.candidate_id.eq(
            "ALL_FALLBACKS__"
            "STRATEGY_MATCH+POSITIVE_CONFIDENCE_LOWER_BOUND+"
            "EXPANSION_GT_DRAWDOWN"
        )
    ].iloc[0]
    for rule in ("ESTIMATE_AVAILABLE",) + RULES:
        failed = data.loc[~masks[rule]]
        if rule == "ESTIMATE_AVAILABLE":
            removal = None
        else:
            enabled = tuple(value for value in RULES if value != rule)
            candidate_id = "ALL_FALLBACKS__" + (
                "+".join(enabled) if enabled else "ESTIMATE_ONLY"
            )
            removal = candidates.loc[
                candidates.candidate_id.eq(candidate_id)
            ].iloc[0]
        observed = failed.terminal_premium_return.notna()
        rows.append({
            "rule": rule,
            "trades_rejected_independently": len(failed),
            "rejected_with_observed_outcome": int(observed.sum()),
            "profitable_trades_rejected": int(
                failed.counterfactual_net_return.gt(0).sum()
            ),
            "losing_trades_prevented": int(
                failed.counterfactual_net_return.le(0).sum()
            ),
            "rejected_net_expectancy": failed.counterfactual_net_return.mean(),
            "baseline_maximum_drawdown": baseline.maximum_drawdown,
            "drawdown_without_rule": (
                removal.maximum_drawdown if removal is not None else np.nan
            ),
            "drawdown_contribution": (
                removal.maximum_drawdown - baseline.maximum_drawdown
                if removal is not None else np.nan
            ),
            "removal_evaluable": removal is not None,
            "note": (
                "lower-history estimates absent from frozen data"
                if rule == "ESTIMATE_AVAILABLE"
                else "remove-one-rule counterfactual"
            ),
        })
    return pd.DataFrame(rows)
