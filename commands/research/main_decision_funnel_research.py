"""Run the Step 20E decision-funnel diagnostic without frozen mutations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.decision_funnel_research import (
    add_net_profitability,
    build_decision_funnel,
    build_funnel_universe,
    counterfactual_scenarios,
    summarize_counterfactuals,
)
from src.walk_forward_paper_trading import POLICY


STAGE1 = Path("data/setups/stage1_setup_dataset.parquet")
PREDICTIONS = Path("data/prediction/prediction_dataset.parquet")
JOINED = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
PAPER_TRADES = Path("reports/walk_forward_paper_trading/trade_log.csv")
STEP20D = Path("reports/institutional_backtest/metadata.json")
REPORTS = Path("reports/decision_funnel_research")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError(f"{REPORTS} exists; research output will not be overwritten")
    frozen = (STAGE1, PREDICTIONS, JOINED, PAPER_TRADES, STEP20D)
    before = {str(path): sha256(path) for path in frozen}
    stage1 = pd.read_parquet(STAGE1)
    predictions = pd.read_parquet(PREDICTIONS)
    joined = pd.read_parquet(JOINED)
    paper = pd.read_csv(PAPER_TRADES)
    paper["entry_timestamp"] = pd.to_datetime(
        paper.entry_timestamp, utc=True
    ).dt.tz_convert("Asia/Kolkata")

    universe = add_net_profitability(
        build_funnel_universe(stage1, predictions, joined, POLICY)
    )
    funnel = build_decision_funnel(
        universe, set(paper.entry_timestamp), POLICY
    )
    rejected = universe.loc[
        universe.first_failure_gate.ne("EXECUTED_CANDIDATE")
        & universe.net_profitable
    ].copy()
    accepted_bad = universe.loc[
        universe.timestamp.isin(paper.entry_timestamp)
        & ~universe.net_profitable
    ].copy()
    filter_rows = []
    for gate, group in universe.loc[
        universe.first_failure_gate.ne("EXECUTED_CANDIDATE")
    ].groupby("first_failure_gate", sort=False):
        observed = group.terminal_premium_return.notna()
        filter_rows.append({
            "filter": gate,
            "total_rejected": len(group),
            "rejected_with_observed_outcome": int(observed.sum()),
            "profitable_rejected": int(group.net_profitable.sum()),
            "profitable_fraction_of_observed": (
                group.loc[observed, "net_profitable"].mean()
                if observed.any() else float("nan")
            ),
            "average_probability": group[POLICY.probability_column].mean(),
            "average_premium_return": group.terminal_premium_return.mean(),
            "outcome_coverage": float(observed.mean()),
            "interpretation": (
                "measurable only inside frozen OOF universe"
                if observed.any()
                else "not measurable: no frozen outcome"
            ),
        })
    filter_importance = pd.DataFrame(filter_rows).sort_values(
        ["profitable_rejected", "total_rejected"], ascending=False
    )
    scenarios = counterfactual_scenarios(universe, POLICY)
    counterfactual = summarize_counterfactuals(
        scenarios,
        pd.to_datetime(stage1.timestamp.iloc[0]),
        pd.to_datetime(stage1.timestamp.iloc[-1]),
    )

    REPORTS.mkdir(parents=True)
    funnel.to_csv(REPORTS / "decision_funnel.csv", index=False)
    filter_importance.to_csv(REPORTS / "filter_importance.csv", index=False)
    counterfactual.to_csv(REPORTS / "counterfactual_results.csv", index=False)
    evidence_columns = [
        "timestamp", "entry_signal", "first_failure_gate",
        "first_failure_reason", POLICY.probability_column,
        "terminal_premium_return", "counterfactual_net_return",
        "premium_expansion", "premium_drawdown", "atr", "regime", "strategy",
    ]
    rejected[evidence_columns].to_csv(
        REPORTS / "rejected_profitable_trades.csv", index=False
    )
    accepted_bad[evidence_columns].to_csv(
        REPORTS / "accepted_bad_trades.csv", index=False
    )
    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed during funnel research")

    decision_row = filter_importance.loc[
        filter_importance["filter"].eq("DECISION_ENGINE")
    ].iloc[0]
    probability_row = filter_importance.loc[
        filter_importance["filter"].eq("PROBABILITY_THRESHOLD")
    ].iloc[0]
    baseline = counterfactual.loc[
        counterfactual.scenario.eq("BASELINE")
    ].iloc[0]
    no_decision = counterfactual.loc[
        counterfactual.scenario.eq("REMOVE_DECISION_ENGINE")
    ].iloc[0]
    no_probability = counterfactual.loc[
        counterfactual.scenario.eq("REMOVE_PROBABILITY_THRESHOLD")
    ].iloc[0]
    report = f"""# Decision Funnel Diagnostic

## Funnel

The replay starts with {len(universe):,} non-zero Stage-1 setup candidates.
Stage-1 allows {int(funnel.iloc[0].total_passed):,}; the frozen Decision Engine
allows {int(funnel.iloc[1].total_passed):,}; the CatBoost {POLICY.threshold:.2f}
threshold allows {int(funnel.iloc[2].total_passed):,}; and non-overlap execution
completes {int(funnel.iloc[-1].total_passed):,} trades.

## Profitable rejection attribution

- Decision Engine: {int(decision_row.profitable_rejected):,} observed profitable
  rejections from {int(decision_row.rejected_with_observed_outcome):,}
  outcome-observable rejected rows.
- Probability threshold: {int(probability_row.profitable_rejected):,} observed
  profitable rejections from
  {int(probability_row.rejected_with_observed_outcome):,} observable rows.
- Stage-1 profitability cannot be measured because frozen OOF probabilities and
  premium outcomes exist only for Stage-1-allowed setups.
- Accepted trades that were net losers: {len(accepted_bad):,}.

## One-filter counterfactuals

- Baseline: {int(baseline.trades)} trades, {baseline.win_rate:.2%} win rate,
  PF {baseline.profit_factor:.3f}, CAGR {baseline.cagr:.2%}.
- Without Decision Engine: {int(no_decision.trades)} trades,
  {no_decision.win_rate:.2%} win rate, PF {no_decision.profit_factor:.3f},
  CAGR {no_decision.cagr:.2%}.
- Without probability threshold: {int(no_probability.trades)} trades,
  {no_probability.win_rate:.2%} win rate,
  PF {no_probability.profit_factor:.3f},
  CAGR {no_probability.cagr:.2%}.

All counterfactuals use frozen outcomes, Step 20D costs, a 15-minute response
horizon, chronological capital, and 1,000 bootstrap Monte Carlo paths.
"""
    (REPORTS / "decision_funnel_report.md").write_text(report, encoding="utf-8")

    recommendation = f"""# Research Recommendation

## Relaxation candidate

The Decision Engine is the first filter to research for relaxation. It removes
{int(decision_row.profitable_rejected):,} net-profitable, outcome-observable
setups, compared with {int(probability_row.profitable_rejected):,} at the
probability gate. This does **not** authorize a production change.

## Keep frozen

- Stage-1 V2: keep frozen; rejected rows have no comparable OOF outcome evidence.
- CatBoost probabilities: keep frozen; they must never be regenerated here.
- Risk and non-overlap execution: keep frozen because they enforce feasible
  capital use rather than predictive selection.
- Step 20D costs: keep frozen.

The four-trade baseline remains statistically insufficient. Any relaxation must
be validated later on newly generated out-of-sample data, not promoted from
these counterfactuals.
"""
    (REPORTS / "recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    metadata = {
        "step": "20E-funnel-diagnostic",
        "research_only": True,
        "model": POLICY.model,
        "threshold": POLICY.threshold,
        "probabilities_regenerated": False,
        "models_retrained": False,
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "universe_rows": len(universe),
            "funnel_reconciles": bool(
                (funnel.total_rejected + funnel.total_passed)
                .eq(funnel.total_incoming).all()
            ),
            "accepted_bad_trades": len(accepted_bad),
            "rejected_profitable_trades": len(rejected),
            "counterfactual_monte_carlo_paths": 1000,
            "stage1_counterfactual_evaluable": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

