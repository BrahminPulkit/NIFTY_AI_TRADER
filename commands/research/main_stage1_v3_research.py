"""Run independent Stage-1 V3 setup-family research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.stage1_v3_research import evaluate_candidates


def main() -> None:
    source = Path("data/features/feature_dataset.parquet")
    report_dir = Path("reports/stage1_v3_research")
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    results = evaluate_candidates(frame)
    for name, data in results.items():
        if name != "outcomes":
            data.to_csv(report_dir / f"{name}.csv", index=False)
    # Full event outcomes support review without creating production labels.
    results["outcomes"].to_parquet(report_dir / "research_outcomes.parquet", index=False)

    board = results["leaderboard"]
    robust = board.loc[board.robust_candidate]
    top = board.iloc[0]
    recommendation = (
        f"{len(robust)} candidate(s) pass all predeclared robustness gates. "
        f"The leading robust candidate is `{robust.iloc[0].candidate_id}`."
        if len(robust) else
        "No candidate passes all predeclared robustness gates. Do not promote a V3 setup definition."
    )
    report = f"""# Stage-1 V3 Setup Research Report

Stage-1 V2 remained frozen. This research consumed only the canonical feature dataset and produced no production labels or models.

## Scope

- Candidate families tested: **{board.family.nunique()}**
- Candidate definitions tested: **{len(board)}**
- Sessions: **{int(top.sessions)}**
- Benchmark exit contracts: `TRAIL_H15_ATR0.50`, `ATR_H15_SL0.75_RR1.25`, and `TIME_H15_ATR_RISK`
- Event cooldown: 10 candles within the same session
- Maximum research horizon: 15 candles, with no cross-session outcomes

## Ranking policy

Candidates require at least 500 setups, 0.5–8 setups/session, at least 20% minority direction, positive Expected R under every benchmark exit, at least 80% positive year-contract cells, and positive worst-year, direction, regime, and session Expected R. Ranking cannot promote a candidate that fails these gates.

## Result

{recommendation}

Top numerical candidate (whether robust or not): `{top.candidate_id}`; setups/session={top.setups_per_session:.3f}, median Expected R={top.median_expected_r:.4f}, worst-contract Expected R={top.worst_contract_expected_r:.4f}, positive-year fraction={top.positive_year_fraction:.3f}.

These are gross research outcomes, not cost-adjusted backtests and not production approval.
"""
    (report_dir / "stage1_v3_research_report.md").write_text(report, encoding="utf-8")
    (report_dir / "recommendation.md").write_text(
        "# Recommendation\n\n" + recommendation +
        "\n\nStage-1 V2 remains unchanged and V3 remains research-only.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
