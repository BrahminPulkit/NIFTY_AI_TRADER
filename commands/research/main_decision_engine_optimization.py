"""Execute Step 20G over frozen probability and decision fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision_engine_optimization import (
    evaluate_candidates,
    independent_rule_importance,
    pareto_frontier,
)
from src.decision_funnel_research import (
    add_net_profitability,
    build_funnel_universe,
)
from src.walk_forward_paper_trading import POLICY


STAGE1 = Path("data/setups/stage1_setup_dataset.parquet")
PREDICTIONS = Path("data/prediction/prediction_dataset.parquet")
JOINED = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
PROFITABLE_REJECTS = Path(
    "reports/decision_funnel_research/rejected_profitable_trades.csv"
)
REPORTS = Path("reports/decision_engine_optimization")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(pool: pd.DataFrame, kind: str) -> pd.Series:
    eligible = pool.loc[
        pool.evaluable.eq(True)
        & pool.trades.ge(50)
        & pool.mc_percentile_5.gt(0)
    ].copy()
    if eligible.empty:
        return pd.Series(dtype=object)
    if kind == "conservative":
        return eligible.sort_values(
            ["maximum_drawdown", "profit_factor", "cagr"],
            ascending=[False, False, False],
        ).iloc[0]
    if kind == "aggressive":
        return eligible.sort_values(
            ["cagr", "profit_factor"], ascending=[False, False]
        ).iloc[0]
    # Balanced: equal-rank blend, fixed before observing the result.
    eligible["rank_score"] = (
        eligible.cagr.rank(pct=True)
        + eligible.maximum_drawdown.rank(pct=True)
        + eligible.profit_factor.rank(pct=True)
        + eligible.yearly_positive_fraction.rank(pct=True)
    )
    return eligible.sort_values(
        ["rank_score", "cagr"], ascending=[False, False]
    ).iloc[0]


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError(f"{REPORTS} exists; outputs will not be overwritten")
    frozen = (STAGE1, PREDICTIONS, JOINED, PROFITABLE_REJECTS)
    before = {str(path): sha256(path) for path in frozen}
    stage1 = pd.read_parquet(STAGE1)
    predictions = pd.read_parquet(PREDICTIONS)
    joined = pd.read_parquet(JOINED)
    universe = add_net_profitability(
        build_funnel_universe(stage1, predictions, joined, POLICY)
    )
    observed = universe.loc[
        universe[POLICY.probability_column].notna()
        & universe.terminal_premium_return.notna()
    ].copy()
    start = pd.to_datetime(stage1.timestamp.iloc[0])
    end = pd.to_datetime(stage1.timestamp.iloc[-1])
    candidates, mc = evaluate_candidates(
        observed, POLICY, start, end, simulations=1000
    )
    candidates["ranking_position"] = np.nan
    ranked_index = (
        candidates.loc[candidates.evaluable.eq(True)]
        .sort_values(
            ["cagr", "maximum_drawdown", "profit_factor",
             "yearly_positive_fraction"],
            ascending=[False, False, False, False],
        ).index
    )
    candidates.loc[ranked_index, "ranking_position"] = range(
        1, len(ranked_index) + 1
    )
    frontier = pareto_frontier(candidates)
    importance = independent_rule_importance(observed, candidates, POLICY)

    conservative = _profile(candidates, "conservative")
    balanced = _profile(candidates, "balanced")
    aggressive = _profile(candidates, "aggressive")
    REPORTS.mkdir(parents=True)
    candidates.to_csv(REPORTS / "decision_engine_candidates.csv", index=False)
    importance.to_csv(REPORTS / "rule_importance.csv", index=False)
    frontier.to_csv(REPORTS / "pareto_decision_engines.csv", index=False)
    mc.to_csv(REPORTS / "monte_carlo_summary.csv", index=False)

    def describe(name: str, row: pd.Series) -> str:
        if row.empty:
            return f"- {name}: no candidate met the fixed robustness screen"
        return (
            f"- {name}: `{row.candidate_id}` — {int(row.trades)} trades, "
            f"CAGR {row.cagr:.2%}, PF {row.profit_factor:.3f}, "
            f"max DD {row.maximum_drawdown:.2%}, "
            f"MC 5% {row.mc_percentile_5:.2%}"
        )

    baseline_id = (
        "ALL_FALLBACKS__STRATEGY_MATCH+"
        "POSITIVE_CONFIDENCE_LOWER_BOUND+EXPANSION_GT_DRAWDOWN"
    )
    baseline = candidates.loc[candidates.candidate_id.eq(baseline_id)].iloc[0]
    report = f"""# Step 20G Decision Engine Optimisation Research

## Scope

Evaluated {int(candidates.evaluable.sum())} reproducible candidates from every
combination of the three observable frozen rules across six fallback-source
policies. Two requested minimum-history candidates are retained as unevaluable:
the frozen Decision Engine does not store candidate estimates below 30 samples.

## Frozen baseline

- Candidate: `{baseline_id}`
- Trades after CatBoost {POLICY.threshold:.2f}: {int(baseline.trades)}
- CAGR: {baseline.cagr:.2%}
- Profit factor: {baseline.profit_factor:.3f}
- Maximum drawdown: {baseline.maximum_drawdown:.2%}

## Research profiles

{describe("Conservative", conservative)}
{describe("Balanced", balanced)}
{describe("Aggressive", aggressive)}

These profiles are descriptive research results and are not promoted.
"""
    (REPORTS / "decision_engine_report.md").write_text(report, encoding="utf-8")
    recommendation = f"""# Step 20G Recommendation

{describe("Conservative research engine", conservative)}

{describe("Balanced research engine", balanced)}

{describe("Aggressive research engine", aggressive)}

No candidate should replace the frozen Decision Engine. The same frozen OOF
sample was used to compare many rule combinations, creating selection risk.
The selected profiles require a new untouched chronological validation period
before any production discussion.

Keep Stage-1 V2, probabilities, CatBoost threshold, risk controls, non-overlap
execution and cost assumptions frozen.
"""
    (REPORTS / "recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed")
    metadata = {
        "step": "20G",
        "research_only": True,
        "model_retraining": False,
        "probability_regeneration": False,
        "model": POLICY.model,
        "threshold": POLICY.threshold,
        "candidate_count": len(candidates),
        "evaluable_candidates": int(candidates.evaluable.sum()),
        "monte_carlo_simulations_per_evaluable_candidate": 1000,
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "oof_rows_used": len(observed),
            "step20f_profitable_reject_rows": len(
                pd.read_csv(PROFITABLE_REJECTS)
            ),
            "pareto_candidates": len(frontier),
            "minimum_history_20_evaluable": False,
            "minimum_history_15_evaluable": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

