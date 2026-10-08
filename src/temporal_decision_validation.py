"""Step 20H-R helpers for strict temporal Decision Engine validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision_engine_optimization import candidate_mask
from src.decision_funnel_research import simulate_selection
from src.institutional_backtest import CONFIG, monte_carlo, monte_carlo_summary
from src.walk_forward_paper_trading import ReplayPolicy, replay_metrics


RESEARCH_END = pd.Timestamp("2024-12-31 23:59:59", tz="Asia/Kolkata")
VALIDATION_START = pd.Timestamp("2025-01-01 00:00:00", tz="Asia/Kolkata")

# Declared before validation inspection.
TOLERANCES = {
    "minimum_validation_trades": 30,
    "minimum_validation_profit_factor": 1.25,
    "minimum_validation_mc_p05": 0.0,
    "maximum_relative_cagr_degradation": 0.50,
    "maximum_drawdown_increase": 0.05,
    "maximum_probability_psi": 0.25,
    "maximum_median_feature_psi": 0.10,
    "maximum_relative_coverage_shift": 0.50,
}


def select_profiles(candidates: pd.DataFrame) -> dict[str, pd.Series]:
    """Apply the exact fixed Step 20G profile-selection policy."""
    eligible = candidates.loc[
        candidates.evaluable.eq(True)
        & candidates.trades.ge(50)
        & candidates.mc_percentile_5.gt(0)
    ].copy()
    if eligible.empty:
        raise ValueError("No research-window candidate passes robustness screen")
    conservative = eligible.sort_values(
        ["maximum_drawdown", "profit_factor", "cagr"],
        ascending=[False, False, False],
    ).iloc[0]
    aggressive = eligible.sort_values(
        ["cagr", "profit_factor"], ascending=[False, False]
    ).iloc[0]
    eligible["rank_score"] = (
        eligible.cagr.rank(pct=True)
        + eligible.maximum_drawdown.rank(pct=True)
        + eligible.profit_factor.rank(pct=True)
        + eligible.yearly_positive_fraction.rank(pct=True)
    )
    balanced = eligible.sort_values(
        ["rank_score", "cagr"], ascending=[False, False]
    ).iloc[0]
    return {
        "CONSERVATIVE": conservative,
        "BALANCED": balanced,
        "AGGRESSIVE": aggressive,
    }


def evaluate_frozen_engine(
    data: pd.DataFrame,
    candidate: pd.Series,
    policy: ReplayPolicy,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int,
    simulations: int = 1000,
) -> tuple[pd.DataFrame, dict]:
    enabled = tuple(
        value for value in str(candidate.enabled_rules).split("+") if value
    )
    decision = candidate_mask(
        data, enabled, str(candidate.source_mode)
    )
    selected = data.loc[
        decision & data[policy.probability_column].ge(policy.threshold)
    ]
    trades = simulate_selection(selected, policy)
    metrics = replay_metrics(trades, CONFIG.initial_capital, start, end)
    distribution = monte_carlo(
        trades, simulations=simulations,
        initial_capital=CONFIG.initial_capital, seed=seed,
    )
    mc = monte_carlo_summary(distribution)
    metrics.update({
        "eligible_oof_rows": len(data),
        "decision_pass_rows": int(decision.sum()),
        "probability_pass_rows": int(
            data[policy.probability_column].ge(policy.threshold).sum()
        ),
        "coverage": len(trades) / len(data) if len(data) else np.nan,
        "average_holding_time": (
            15.0 if len(trades) else np.nan
        ),
        "average_premium_expansion": (
            float(selected.loc[
                selected.timestamp.isin(trades.entry_timestamp),
                "premium_expansion",
            ].mean()) if len(trades) else np.nan
        ),
        "average_terminal_premium_return": (
            float(selected.loc[
                selected.timestamp.isin(trades.entry_timestamp),
                "terminal_premium_return",
            ].mean()) if len(trades) else np.nan
        ),
        "mc_simulations": simulations if len(trades) else 0,
        "mc_percentile_5": mc.get("percentile_5", np.nan),
        "mc_median": mc.get("median_case", np.nan),
        "mc_percentile_95": mc.get("percentile_95", np.nan),
    })
    return trades, metrics


def population_stability_index(
    research: pd.Series, validation: pd.Series, bins: int = 10
) -> float:
    left = pd.to_numeric(research, errors="coerce").dropna().to_numpy(float)
    right = pd.to_numeric(validation, errors="coerce").dropna().to_numpy(float)
    if len(left) == 0 or len(right) == 0:
        return np.nan
    edges = np.unique(np.quantile(left, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(left[0], right) else np.inf
    edges[0], edges[-1] = -np.inf, np.inf
    left_counts = np.histogram(left, bins=edges)[0].astype(float)
    right_counts = np.histogram(right, bins=edges)[0].astype(float)
    left_share = np.clip(left_counts / left_counts.sum(), 1e-6, None)
    right_share = np.clip(right_counts / right_counts.sum(), 1e-6, None)
    return float(np.sum((right_share - left_share) * np.log(right_share / left_share)))


def feature_drift_table(
    research: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        left, right = research[column], validation[column]
        left_numeric = pd.to_numeric(left, errors="coerce").astype(float)
        right_numeric = pd.to_numeric(right, errors="coerce").astype(float)
        left_mean = left_numeric.mean()
        right_mean = right_numeric.mean()
        left_std = left_numeric.std()
        denominator = (
            max(float(left_std), 1e-12) if pd.notna(left_std) else np.nan
        )
        rows.append({
            "feature": column,
            "research_non_null": int(left.notna().sum()),
            "validation_non_null": int(right.notna().sum()),
            "research_mean": left_mean,
            "validation_mean": right_mean,
            "standardized_mean_shift": (
                (right_mean - left_mean) / denominator
            ),
            "psi": population_stability_index(left, right),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def stability_verdict(
    research: dict,
    validation: dict,
    probability_psi: float,
    median_feature_psi: float,
) -> dict:
    relative_cagr_degradation = (
        (research["cagr"] - validation["cagr"]) / abs(research["cagr"])
        if research["cagr"] != 0 else np.inf
    )
    drawdown_increase = max(
        0.0,
        abs(validation["maximum_drawdown"])
        - abs(research["maximum_drawdown"]),
    )
    relative_coverage_shift = (
        abs(validation["coverage"] - research["coverage"])
        / research["coverage"]
        if research["coverage"] > 0 else np.inf
    )
    checks = {
        "validation_trade_count": (
            validation["executed_trades"]
            >= TOLERANCES["minimum_validation_trades"]
        ),
        "validation_profit_factor": (
            validation["profit_factor"]
            >= TOLERANCES["minimum_validation_profit_factor"]
        ),
        "validation_mc_p05": (
            validation["mc_percentile_5"]
            >= TOLERANCES["minimum_validation_mc_p05"]
        ),
        "cagr_degradation": (
            relative_cagr_degradation
            <= TOLERANCES["maximum_relative_cagr_degradation"]
        ),
        "drawdown_increase": (
            drawdown_increase <= TOLERANCES["maximum_drawdown_increase"]
        ),
        "probability_psi": (
            probability_psi <= TOLERANCES["maximum_probability_psi"]
        ),
        "median_feature_psi": (
            median_feature_psi
            <= TOLERANCES["maximum_median_feature_psi"]
        ),
        "coverage_shift": (
            relative_coverage_shift
            <= TOLERANCES["maximum_relative_coverage_shift"]
        ),
    }
    return {
        "relative_cagr_degradation": relative_cagr_degradation,
        "drawdown_increase": drawdown_increase,
        "relative_coverage_shift": relative_coverage_shift,
        **{f"check_{key}": value for key, value in checks.items()},
        "passes_all_tolerances": all(checks.values()),
    }
