"""Execute Roadmap Step 19A without changing frozen production artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.outcome_sparsity_research import (
    MOVE_LEVELS,
    analyze_outcome_paths,
    summarize_mfe,
    summarize_target_reach,
    verify_frozen_replay,
)


SETUPS = Path("data/setups/stage1_setup_dataset.parquet")
OUTCOMES = Path("data/outcomes/stage2_outcome_dataset.parquet")
REPORTS = Path("reports/outcome_sparsity")


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.reset_index()
    headers = [str(column) for column in values.columns]
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    line = "| " + " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([line, separator, *body])


def main() -> None:
    setups = pd.read_parquet(SETUPS)
    outcomes = pd.read_parquet(OUTCOMES)
    paths, simulations = analyze_outcome_paths(setups, outcomes)
    verify_frozen_replay(outcomes, simulations)
    mfe = summarize_mfe(paths)
    target = summarize_target_reach(paths, simulations)
    REPORTS.mkdir(parents=True, exist_ok=True)
    paths.to_csv(REPORTS / "mfe_distribution.csv", index=False)
    target.to_csv(REPORTS / "target_reach_statistics.csv", index=False)

    non_wins = paths.loc[paths.production_outcome.ne("WIN")]
    original = target.loc[
        target.diagnostic_target_pct.eq(0.003) & target.direction.eq("ALL")
    ].iloc[0]
    lowest = target.loc[
        target.diagnostic_target_pct.eq(0.001) & target.direction.eq("ALL")
    ].iloc[0]
    reach_lines = []
    for level in MOVE_LEVELS:
        column = f"reached_{level * 100:.2f}pct"
        reach_lines.append(
            f"- {level * 100:.2f}%: {int(non_wins[column].sum()):,} "
            f"({non_wins[column].mean():.2%}) of LOSS/TIMEOUT setups"
        )
    direction_table = _markdown_table(
        pd.crosstab(paths.direction, paths.production_outcome)
    )
    target_table = _markdown_table(
        target.loc[target.direction.eq("ALL")].reset_index(drop=True)
    )
    almost = int(non_wins.almost_reached_030.sum())

    report = f"""# Step 19A — Outcome Sparsity Root Cause Analysis

## Frozen scope

- Stage-1 opportunities: **{len(paths):,}**
- LOSS/TIMEOUT investigated: **{len(non_wins):,}**
- Production WINs: **{paths.production_outcome.eq("WIN").sum():,}**
- Contract replay at 0.30%: exact match with frozen Stage-2 labels.
- No production artifact, feature, setup, label, model or threshold was changed.

## Outcome distribution

{direction_table}

## Five-candle favourable movement among non-winners

{chr(10).join(reach_lines)}

- Almost reached 0.30% (at least 0.27%, below 0.30%): **{almost:,}**

## Diagnostic target replay

Raw reach ignores event ordering. Executable WIN applies the frozen 0.20% stop,
gap-aware fills and conservative stop-first ambiguity.

{target_table}

## Central result

At the production 0.30% target, only **{int(original.executable_win_count):,}**
setups win. At the diagnostic 0.10% target, **{int(lowest.executable_win_count):,}**
would win, but that is a counterfactual research result—not a label change.
"""
    (REPORTS / "outcome_sparsity_report.md").write_text(report, encoding="utf-8")

    timeout_rate = paths.production_outcome.eq("TIMEOUT").mean()
    low_mfe = non_wins.mfe_pct.lt(0.001).mean()
    bottleneck = f"""# Bottleneck Analysis

1. **Five-candle target attainability is the primary measured bottleneck.**
   The production TIMEOUT rate is {timeout_rate:.2%}; {low_mfe:.2%} of
   non-winning setups fail to produce even 0.10% favourable excursion.
2. **The fixed 0.30% target is large relative to observed five-candle MFE.**
   Diagnostic lower targets materially alter WIN counts; see
   `target_reach_statistics.csv`.
3. **The fixed 0.20% stop is not the main source of class sparsity.**
   Frozen labels contain only {paths.production_outcome.eq("LOSS").sum():,}
   stop losses versus {paths.production_outcome.eq("TIMEOUT").sum():,} timeouts.
4. **The five-candle timeout interacts directly with the target.**
   This study can establish that the target is rarely reached inside five
   candles; it cannot infer whether a longer horizon is better without changing
   the frozen research question.
5. **Stage-1 entry quality remains a contributor.**
   The low MFE distribution shows that many allowed entries receive little
   immediate follow-through, but target/horizon effects cannot be separated
   causally from entry quality using labels alone.
6. **Market volatility explains mechanical attainability, not label validity.**
   MFE/MAE distributions quantify the available movement. No volatility
   threshold was optimized.
7. **No label-construction defect was found.**
   Independent replay reproduces every frozen 0.30% WIN/LOSS/TIMEOUT label.
"""
    (REPORTS / "bottleneck_analysis.md").write_text(bottleneck, encoding="utf-8")

    recommendation = """# Recommendation

Do not calibrate or retrain the model. The nine CALL WINs are a consequence of
the frozen target/horizon contract and weak immediate follow-through, not a
replay or implementation defect.

Use the diagnostic target table to choose the next explicitly authorized
research step. Any production target or horizon change would be a contract
redesign and must remain separate from these frozen labels. Stage-1 quality
should also be evaluated against continuous MFE, rather than inferred from the
current binary WIN class alone.
"""
    (REPORTS / "recommendation.md").write_text(recommendation, encoding="utf-8")


if __name__ == "__main__":
    main()
