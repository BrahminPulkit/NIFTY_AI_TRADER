"""Execute Step 20I without changing any frozen feature or dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.feature_drift_mitigation import (
    attach_root_cause,
    original_feature_drift,
    representation_candidates,
)
from src.temporal_decision_validation import RESEARCH_END, VALIDATION_START


PREDICTIONS = Path("data/prediction/prediction_dataset.parquet")
JOINED = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
REPORTS = Path("reports/feature_drift_mitigation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError(f"{REPORTS} exists; outputs will not be overwritten")
    frozen = (PREDICTIONS, JOINED, ALIGNED)
    before = {str(path): sha256(path) for path in frozen}
    predictions = pd.read_parquet(PREDICTIONS)
    joined = pd.read_parquet(JOINED, columns=["timestamp"])
    aligned = pd.read_parquet(
        ALIGNED, columns=["timestamp", "index_close", "option_close"]
    )
    data = (
        joined.merge(predictions, on="timestamp", how="left", validate="one_to_one")
        .merge(aligned, on="timestamp", how="left", validate="one_to_one")
        .sort_values("timestamp").reset_index(drop=True)
    )
    if data[["index_close", "option_close"]].isna().any().any():
        raise ValueError("Price anchors missing after frozen timestamp join")
    research_mask = data.timestamp.le(RESEARCH_END)
    validation_mask = data.timestamp.ge(VALIDATION_START)
    drift = original_feature_drift(data, research_mask, validation_mask)
    candidates = representation_candidates(data, research_mask, validation_mask)
    root_cause = attach_root_cause(drift, candidates)
    reductions = candidates.loc[
        candidates.expected_absolute_psi_reduction.gt(0)
    ].sort_values(
        ["expected_absolute_psi_reduction", "candidate_psi"],
        ascending=[False, True],
    )

    REPORTS.mkdir(parents=True)
    root_cause.to_csv(REPORTS / "feature_drift_root_cause.csv", index=False)
    candidates.to_csv(REPORTS / "normalisation_candidates.csv", index=False)
    reductions.to_csv(REPORTS / "expected_psi_reduction.csv", index=False)

    price_features = root_cause.loc[root_cause.price_level_increase_driven]
    replacement = root_cause.loc[
        root_cause.original_psi.ge(0.10)
        & root_cause.best_expected_relative_psi_reduction.gt(0)
    ]
    plan_lines = "\n".join(
        f"- `{row.feature}` → `{row.best_candidate_representation}` "
        f"(PSI {row.original_psi:.3f} → {row.best_candidate_psi:.3f})"
        for row in replacement.head(30).itertuples(index=False)
    )
    plan = f"""# Feature Replacement Research Plan

No feature is replaced in this step.

## Root cause

- Features audited: {len(root_cause)}
- Features above PSI 0.10: {int(root_cause.original_psi.ge(0.10).sum())}
- Absolute-price features identified as price-level-driven:
  {len(price_features)}

Absolute NIFTY EMAs, rolling highs/lows, previous highs/lows and similar point
features drift because the validation-period index price level is substantially
higher. This is distribution drift, not proof that market behaviour changed.

## Candidate replacements

{plan_lines or "- No candidate achieved positive estimated PSI reduction."}

Rolling transformations here use only the prior/current eligible-setup stream.
Before any future feature-contract change, the chosen formulas would require a
canonical candle-stream implementation, prefix-invariance testing, and model
retraining in a separately approved roadmap step.
"""
    (REPORTS / "feature_replacement_plan.md").write_text(plan, encoding="utf-8")
    median_original = float(
        root_cause.original_psi.replace([float("inf")], pd.NA).median()
    )
    median_best = float(
        root_cause.best_candidate_psi.replace([float("inf")], pd.NA).median()
    )
    recommendation = f"""# Step 20I Recommendation

The feature-drift rejection is driven materially by absolute price-level
features: {len(price_features)} were classified as NIFTY-level-driven.

Research median original PSI: {median_original:.4f}.
Median best available candidate PSI: {median_best:.4f}.

Prefer relative-to-price or ATR-normalised distance representations for EMAs,
VWAP and rolling/previous levels. Prefer rolling z-score or percentile
representations for point magnitudes. Preserve log returns and existing
relative features that already show low PSI.

Do not change the frozen feature contract or model from this evidence alone.
"""
    (REPORTS / "recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed")
    metadata = {
        "step": "20I",
        "research_only": True,
        "model_retraining": False,
        "probability_regeneration": False,
        "feature_contract_modified": False,
        "research_rows": int(research_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "features_audited": len(root_cause),
        "normalisation_candidates": len(candidates),
        "price_level_driven_features": len(price_features),
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "future_values_used": False,
            "rolling_windows_centered": False,
            "datasets_written": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

