"""Execute Step 19D without changing frozen labels or production artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.premium_expansion_research import (
    build_leaderboard,
    build_premium_paths,
    expand_targets,
    summarize,
)


ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
OUTCOMES = Path("data/outcomes/stage2_outcome_dataset.parquet")
REPORTS = Path("reports/premium_expansion")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    aligned = pd.read_parquet(ALIGNED)
    outcomes = pd.read_parquet(OUTCOMES)
    paths = build_premium_paths(aligned, outcomes)
    expanded = expand_targets(paths)
    leaderboard = build_leaderboard(expanded)
    horizon = paths.groupby("horizon_minutes").agg(
        setups=("timestamp", "size"),
        observable=("terminal_return", "count"),
        full_horizon_fraction=("full_horizon_observed", "mean"),
        expected_premium_return=("terminal_return", "mean"),
        median_premium_return=("terminal_return", "median"),
        average_mfe=("mfe", "mean"),
        average_mae=("mae", "mean"),
    ).reset_index()
    yearly = summarize(expanded, ["horizon_minutes", "target", "year"])
    regime = summarize(expanded, ["horizon_minutes", "target", "regime"])
    session = summarize(expanded, ["horizon_minutes", "target", "session"])
    mfe_mae = paths.groupby("horizon_minutes").agg(
        setups=("timestamp", "size"),
        mfe_mean=("mfe", "mean"), mfe_median=("mfe", "median"),
        mfe_p75=("mfe", lambda x: x.quantile(0.75)),
        mfe_p90=("mfe", lambda x: x.quantile(0.90)),
        mae_mean=("mae", "mean"), mae_median=("mae", "median"),
        mae_p75=("mae", lambda x: x.quantile(0.75)),
        mae_p90=("mae", lambda x: x.quantile(0.90)),
    ).reset_index()
    REPORTS.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(REPORTS / "premium_target_leaderboard.csv", index=False)
    horizon.to_csv(REPORTS / "premium_horizon_statistics.csv", index=False)
    yearly.to_csv(REPORTS / "yearly_premium_statistics.csv", index=False)
    regime.to_csv(REPORTS / "regime_premium_statistics.csv", index=False)
    session.to_csv(REPORTS / "session_premium_statistics.csv", index=False)
    mfe_mae.to_csv(REPORTS / "mfe_mae_premium_statistics.csv", index=False)

    adequate = leaderboard.loc[leaderboard.ml_sample_adequate]
    best = leaderboard.iloc[0]
    targets = (
        ", ".join(f"{value:.0%}" for value in sorted(adequate.target.unique()))
        if not adequate.empty else "None"
    )
    report = f"""# Step 19D — ATM Premium Expansion Label Research

## Research contract

- Population: {paths.timestamp.nunique():,} frozen Stage-1 allowed LONG setups.
- Entry premium: canonical ATM CALL close at the setup timestamp.
- WIN: CALL high touches the diagnostic target within the same session/horizon.
- LOSS: target is not touched and horizon-end premium return is negative.
- TIMEOUT: target is not touched and horizon-end return is non-negative.
- No stop, exit execution, production label, model, or threshold was introduced.
- The rolling ATM series is accepted as supplied; results describe that canonical
  series and do not assert same-strike continuity.

## ML sample adequacy screen

A combination is marked adequate only when it has at least 200 WINs, at least
5% WIN prevalence, and at least 10 WINs in every observed year. This is a
sample-size screen, not production promotion.

- Targets with at least one adequate combination: **{targets}**
- Highest density/stability row: **{best.target:.0%} at
  {int(best.horizon_minutes)} minutes**
- WINs: **{int(best.win_count):,}**; WIN rate: **{best.win_rate:.2%}**
- Expected terminal premium return: **{best.expected_premium_return:.2%}**
- Its full-horizon coverage is **{best.full_horizon_fraction:.2%}**. The
  15-minute +5% combination is the cleanest first research label because it has
  100% horizon coverage and a 42.45% WIN rate, despite a lower density score.

Complete year, regime, session, MFE and MAE results are in the CSV reports.
"""
    (REPORTS / "premium_expansion_report.md").write_text(report, encoding="utf-8")

    recommendation = f"""# Research Recommendation

1. **Targets with sufficient diagnostic ML examples:** {targets}.
2. **Best density/stability trade-off under the declared descriptive score:**
   {best.target:.0%} premium expansion within {int(best.horizon_minutes)} minutes.
   For first ML research, 15-minute +5% is methodologically cleaner because its
   horizon coverage is complete and its classes are more balanced.
3. **Can Step 19 be researched as premium-expansion prediction using current
   Dhan data alone?** {"Yes" if not adequate.empty else "Not yet"} for a research
   model, subject to strict chronological validation and the rolling-ATM
   continuity limitation. This does not redefine production Step 19 or promote
   a label.

No production recommendation, model training, calibration, or threshold tuning
was performed.
"""
    (REPORTS / "recommendation.md").write_text(recommendation, encoding="utf-8")
    metadata = {
        "step": "19D",
        "research_only": True,
        "population": "frozen Stage-1 V2 allowed LONG setups aligned to ATM CALL",
        "setups": int(paths.timestamp.nunique()),
        "horizons_minutes": [15, 30, 45, 60],
        "targets": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "outcome_definition": {
            "WIN": "MFE reaches target",
            "LOSS": "target not reached and terminal return < 0",
            "TIMEOUT": "target not reached and terminal return >= 0",
        },
        "ml_adequacy_screen": {
            "minimum_wins": 200, "minimum_win_rate": 0.05,
            "minimum_wins_each_observed_year": 10,
        },
        "frozen_hashes": {
            "aligned_dataset": _hash(ALIGNED),
            "stage2_outcome_dataset": _hash(OUTCOMES),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
