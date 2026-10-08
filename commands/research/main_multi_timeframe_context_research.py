"""Run Step 19B research without modifying frozen production components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multi_timeframe_context_research import (
    CONDITION_FEATURES,
    CONFIRMATION_FEATURES,
    CONTEXT_FEATURES,
    attach_context,
    build_completed_15m_context,
    conditional_statistics,
    interaction_statistics,
    rank_context,
)


FEATURES = ROOT / "data/features/feature_dataset.parquet"
SETUPS = ROOT / "data/setups/stage1_setup_dataset.parquet"
OUTCOMES = ROOT / "data/outcomes/stage2_outcome_dataset.parquet"
REPORTS = ROOT / "reports/multi_timeframe_context"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    feature_data = pd.read_parquet(FEATURES)
    setups = pd.read_parquet(SETUPS)
    outcomes = pd.read_parquet(OUTCOMES)
    context = build_completed_15m_context(feature_data)
    research = attach_context(setups, outcomes, context)
    info, mutual, rankings = rank_context(research)
    stats15 = conditional_statistics(research, CONTEXT_FEATURES + CONDITION_FEATURES)
    stats1 = conditional_statistics(research, CONFIRMATION_FEATURES)
    interactions = interaction_statistics(research, rankings.feature.head(10).tolist())
    REPORTS.mkdir(parents=True, exist_ok=True)
    research[CONTEXT_FEATURES + CONDITION_FEATURES + CONFIRMATION_FEATURES].describe(
        include="all"
    ).T.to_csv(
        REPORTS / "context_feature_statistics.csv"
    )
    info.to_csv(REPORTS / "context_information_gain.csv", index=False)
    mutual.to_csv(REPORTS / "context_mutual_information.csv", index=False)
    rankings.to_csv(REPORTS / "context_rankings.csv", index=False)
    stats15.to_csv(REPORTS / "15m_vs_stage1_statistics.csv", index=False)
    stats1.to_csv(REPORTS / "1m_confirmation_statistics.csv", index=False)
    interactions.to_csv(REPORTS / "context_interactions.csv", index=False)

    top = rankings.head(10)[["feature", "information_gain", "mutual_information"]]
    interaction_report = f"""# Context Interaction Report

- Fully closed 15-minute context was attached to {len(research):,} frozen setups.
- Context timestamps never exceed entry timestamps.
- Interactions use natural sign/state categories; no thresholds were optimized.
- Sparse interaction cells below 30 observations were excluded.
- SHAP was not computed: new-context SHAP requires fitting a model, while Step
  19B explicitly forbids model retraining. No surrogate SHAP values were invented.

## Top 10 univariate context variables

{top.to_csv(index=False)}
"""
    (REPORTS / "context_interaction_report.md").write_text(
        interaction_report, encoding="utf-8"
    )
    trend = stats15.loc[stats15.feature.eq("condition_15m_trend_agrees")].set_index("value")
    momentum = stats1.loc[
        stats1.feature.eq("confirm_1m_momentum_candle")
    ].set_index("value")
    top_names = ", ".join(rankings.feature.head(10))
    recommendation = f"""# Research Findings and Direction

This report is observational and does not recommend a production change.

## Questions answered

1. **Does 15-minute trend agreement improve Stage-1 quality? No.** Agreement
   produced a {trend.loc[1, "win_rate"]:.4%} WIN rate and
   {trend.loc[1, "expected_r"]:.6f} Expected R, versus
   {trend.loc[0, "win_rate"]:.4%} and {trend.loc[0, "expected_r"]:.6f} without
   agreement.
2. **Does one-minute confirmation reduce false entries? Not demonstrated.**
   Breakout is invariant and rejection/volume-expansion are absent under the
   canonical data/definitions. Momentum candles improve Expected R from
   {momentum.loc[0, "expected_r"]:.6f} to {momentum.loc[1, "expected_r"]:.6f},
   but WIN and timeout rates are nearly unchanged.
3. **Strongest exploratory variables:** {top_names}. Their absolute information
   scores are small and only 31 WIN observations exist.
4. **Can a 15m -> frozen setup -> 1m research architecture be evaluated without
   changing Stage-1 V2? Yes technically, through the external causal attachment
   demonstrated here. Current evidence does not justify production adoption.**

## Stage-1 V4 research direction

For a separately authorized Stage-1 V4 research phase, evaluate the top-ranked
15-minute ATR and swing-distance variables as pre-specified context strata—not
optimized gates—and use chronological, year-wise validation with frozen V2 as
the comparator. Retain momentum-candle confirmation only as a hypothesis. A real
delayed-entry study requires separately authorized counterfactual entries and
outcomes; it was not performed because Step 19B forbids new labels and exits.
"""
    (REPORTS / "recommendation.md").write_text(recommendation, encoding="utf-8")
    metadata = {
        "step": "19B",
        "research_only": True,
        "rows": len(research),
        "wins": int(research.final_outcome.eq("WIN").sum()),
        "context_bars": len(context),
        "context_rule": "fully_closed_15m_right_labelled_asof_backward",
        "one_minute_analysis": "same-entry conditional proxy; no delayed relabel",
        "shap_status": "not computed because model retraining is forbidden",
        "frozen_hashes": {
            "feature_dataset": _hash(FEATURES),
            "stage1_setup_dataset": _hash(SETUPS),
            "stage2_outcome_dataset": _hash(OUTCOMES),
        },
        "context_features": CONTEXT_FEATURES,
        "confirmation_features": CONFIRMATION_FEATURES,
        "conditional_agreement_features": CONDITION_FEATURES,
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
