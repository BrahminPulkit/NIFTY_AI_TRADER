"""Execute Roadmap Step 19C using frozen Stage-2 CALL research inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictive_information_capacity import (  # noqa: E402
    class_separability,
    feature_redundancy,
    information_capacity,
)
from src.stage2_call_model import (  # noqa: E402
    MODEL_B_FEATURES,
    prepare_call_dataset,
)


ALIGNED = ROOT / "data/aligned/canonical_index_option_aligned.parquet"
OUTCOMES = ROOT / "data/outcomes/stage2_outcome_dataset.parquet"
STEP19_COMPARISON = ROOT / "reports/stage2_call_model/model_comparison.csv"
REPORTS = ROOT / "reports/predictive_information_capacity"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    aligned = pd.read_parquet(ALIGNED)
    outcomes = pd.read_parquet(OUTCOMES)
    _, model_b, target, metadata = prepare_call_dataset(aligned, outcomes)
    data = model_b.copy()
    data["target_win"] = target.to_numpy()
    data["year"] = metadata.year.to_numpy()
    information, entropy = information_capacity(data, MODEL_B_FEATURES, "target_win")
    redundancy = feature_redundancy(data, MODEL_B_FEATURES, information)
    separability = class_separability(data, MODEL_B_FEATURES, "target_win")
    REPORTS.mkdir(parents=True, exist_ok=True)
    information.to_csv(REPORTS / "information_gain.csv", index=False)
    entropy.to_csv(REPORTS / "entropy_analysis.csv", index=False)
    redundancy.to_csv(REPORTS / "feature_redundancy.csv", index=False)
    separability.to_csv(REPORTS / "class_separability.csv", index=False)

    observed = pd.read_csv(STEP19_COMPARISON)
    roc_low, roc_high = observed.roc_auc.min(), observed.roc_auc.max()
    pr_low, pr_high = observed.pr_auc.min(), observed.pr_auc.max()
    best_uni = separability.iloc[0]
    limit_report = f"""# Theoretical Performance Limit

## What is identifiable

- Research population: {len(data):,} CALL setups; {int(data.target_win.sum())} WINs
  ({data.target_win.mean():.4%} prevalence).
- Existing frozen Step-19 walk-forward ROC-AUC evidence spans
  **{roc_low:.4f}–{roc_high:.4f}**.
- Existing frozen Step-19 PR-AUC evidence spans **{pr_low:.4f}–{pr_high:.4f}**.
- Strongest orientation-free univariate AUC is **{best_uni.orientation_free_univariate_auc:.4f}**
  for `{best_uni.feature}`, with a positive-class bootstrap interval of
  **{best_uni.auc_bootstrap_95_low:.4f}–{best_uni.auc_bootstrap_95_high:.4f}**.

## What is not identifiable

The theoretical Bayes-optimal limit cannot be estimated reliably from only nine
positive observations. Expected precision and recall also require a decision
rule or threshold, which Step 19C forbids. Therefore:

- Maximum achievable ROC-AUC: **not statistically identifiable**
- Maximum achievable PR-AUC: **not statistically identifiable**
- Expected precision range: **not identifiable without a threshold**
- Expected recall range: **not identifiable without a threshold**

The observed ranges above are empirical benchmarks, not claimed upper bounds.
Reporting tighter ranges would fabricate certainty unsupported by the sample.
"""
    (REPORTS / "theoretical_performance_limit.md").write_text(
        limit_report, encoding="utf-8"
    )

    missing = pd.DataFrame([
        (1, "Historical full option-chain snapshots", "VERY_HIGH",
         "Adds PUT/CALL cross-section, strikes, OI and IV; directly observes positioning and skew."),
        (2, "NIFTY Futures price, volume, OI and basis", "HIGH",
         "Adds leveraged directional positioning and index-futures dislocation."),
        (3, "ATM PUT premium OHLCV", "HIGH",
         "Restores missing downside-premium confirmation and CALL/PUT asymmetry."),
        (4, "Strike-wise Open Interest and change in OI", "HIGH",
         "Measures positioning, writing and potential support/resistance."),
        (5, "Implied volatility surface and skew", "MEDIUM_HIGH",
         "Separates premium movement caused by volatility from underlying direction."),
        (6, "Put-call ratios", "MEDIUM",
         "Compact positioning summary, but less informative than the underlying chain."),
        (7, "Option Greeks", "MEDIUM",
         "Useful exposure normalization; mostly derived from price, IV, strike and expiry."),
    ], columns=["rank", "missing_data_source", "expected_usefulness", "research_rationale"])
    missing.to_csv(REPORTS / "missing_information_rankings.csv", index=False)

    high_redundancy = int(redundancy.highly_redundant_095.sum())
    zero_info = int(redundancy.near_zero_information.sum())
    top_unique = ", ".join(redundancy.feature.head(5))
    conclusion = f"""# Final Research Conclusion

## Central question

**The current Index + ATM CALL dataset shows weak but non-zero predictive
association. It is not sufficient to establish an institutional-grade
predictive model with the frozen target. Additional market data is required
before such capability can be claimed.**

1. **Can Index + ATM CALL alone realistically support an institutional model?**
   Not demonstrated. Existing ROC-AUC is above chance, but PR-AUC remains only
   {pr_low:.4f}–{pr_high:.4f}, precision/recall are zero at the frozen Step-19
   decision point, and there are only nine WINs.
2. **Data-limited or model-limited?** Primarily data/target-information limited.
   The positive sample is too small to estimate capacity robustly; no evidence
   shows that model complexity is the binding constraint.
3. **Will more research on current data likely provide meaningful improvement?**
   It can improve understanding and robustness estimates, but large predictive
   improvement is not statistically supported by the current information scores.
4. **Single missing source with largest expected improvement:** historical full
   NIFTY option-chain snapshots containing CALL/PUT strikes, OI and IV.

## Supporting diagnostics

- Highly redundant features at |Spearman| >= 0.95: **{high_redundancy}**
- Features with near-zero discretized information: **{zero_info}**
- Highest independent-signal scores: {top_unique}

These are research findings only. They are not a production recommendation.
"""
    (REPORTS / "final_recommendation.md").write_text(conclusion, encoding="utf-8")

    metadata = {
        "step": "19C",
        "research_only": True,
        "population": "frozen allowed LONG setups aligned to ATM CALL",
        "rows": len(data),
        "wins": int(data.target_win.sum()),
        "features": MODEL_B_FEATURES,
        "no_model_fitted": True,
        "no_threshold_selected": True,
        "frozen_hashes": {
            "aligned_dataset": _hash(ALIGNED),
            "stage2_outcomes": _hash(OUTCOMES),
            "step19_model_comparison": _hash(STEP19_COMPARISON),
        },
        "limitations": [
            "Only nine positive CALL outcomes",
            "Theoretical Bayes limit is not identifiable",
            "Missing PUT, option chain, OI, IV, strike and expiry context",
        ],
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
