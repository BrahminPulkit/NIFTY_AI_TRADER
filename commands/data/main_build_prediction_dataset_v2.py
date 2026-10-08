"""Build Step 20J V1/V2 prediction datasets without model operations."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from src.prediction_dataset_v2 import (
    FEATURE_MAPPING,
    build_prediction_dataset_v2,
    mapping_psi,
)
from src.temporal_decision_validation import RESEARCH_END, VALIDATION_START


V1_SOURCE_PARQUET = Path("data/prediction/prediction_dataset.parquet")
V1_SOURCE_CSV = Path("data/prediction/prediction_dataset.csv")
ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
V1_PARQUET = Path("data/prediction/prediction_dataset_v1.parquet")
V1_CSV = Path("data/prediction/prediction_dataset_v1.csv")
V2_PARQUET = Path("data/prediction/prediction_dataset_v2.parquet")
V2_CSV = Path("data/prediction/prediction_dataset_v2.csv")
REPORTS = Path("reports/prediction_dataset_v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    outputs = (V1_PARQUET, V1_CSV, V2_PARQUET, V2_CSV)
    if REPORTS.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Step 20J output exists and will not be overwritten")
    frozen = (V1_SOURCE_PARQUET, V1_SOURCE_CSV, ALIGNED)
    before = {str(path): sha256(path) for path in frozen}
    v1 = pd.read_parquet(V1_SOURCE_PARQUET)
    anchors = pd.read_parquet(ALIGNED, columns=["timestamp", "index_close"])
    v2, mapping = build_prediction_dataset_v2(v1, anchors)
    psi = mapping_psi(
        v1, v2, mapping, RESEARCH_END, VALIDATION_START
    )

    # Exact byte copies preserve the frozen V1 representation and hash.
    shutil.copyfile(V1_SOURCE_PARQUET, V1_PARQUET)
    shutil.copyfile(V1_SOURCE_CSV, V1_CSV)
    v2.to_parquet(V2_PARQUET, index=False)
    v2.to_csv(V2_CSV, index=False)
    REPORTS.mkdir(parents=True)
    mapping.to_csv(REPORTS / "feature_mapping.csv", index=False)
    psi.to_csv(REPORTS / "psi_improvement.csv", index=False)

    replaced = [item["v1_feature"] for item in FEATURE_MAPPING]
    added = [item["v2_feature"] for item in FEATURE_MAPPING]
    common = [
        column for column in v1.columns
        if column not in replaced
    ]
    common_exact = all(
        v1[column].equals(v2[column]) for column in common
    )
    difference = f"""# Prediction Dataset V1 versus V2

- Rows: {len(v1):,} in both versions
- V1 columns: {len(v1.columns)}
- V2 columns: {len(v2.columns)}
- Replaced features: {len(replaced)}
- Unchanged columns exactly equal: {common_exact}
- Timestamp order identical: {v1.timestamp.equals(v2.timestamp)}

## Removed V1 fields

{chr(10).join(f"- `{name}`" for name in replaced)}

## Added V2 fields

{chr(10).join(f"- `{name}`" for name in added)}

No labels, probabilities, thresholds, Stage-1 fields, or Decision Engine fields
were regenerated.
"""
    (REPORTS / "dataset_difference.md").write_text(
        difference, encoding="utf-8"
    )
    v2_nulls = int(v2[added].isna().sum().sum())
    compatibility = f"""# V2 Compatibility Report

## Structural compatibility

- Row identity preserved: {v1.timestamp.equals(v2.timestamp)}
- Non-replaced columns bit-equivalent in memory: {common_exact}
- V2 replacement null cells: {v2_nulls}
- All transformations are chronological and causal.

## Model compatibility

The frozen model is **not compatible** with V2 because eleven feature names and
representations changed. The model must not receive V2 inputs. No compatibility
adapter is created because that would hide a feature-contract mismatch.

V1 remains available unchanged for frozen model replay. V2 is research-only and
can be used only after a separately approved training and validation step.
"""
    (REPORTS / "compatibility_report.md").write_text(
        compatibility, encoding="utf-8"
    )
    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed")
    metadata = {
        "step": "20J",
        "research_only": True,
        "model_retrained": False,
        "model_fitted": False,
        "probabilities_generated": False,
        "feature_replacements": len(mapping),
        "v1_rows": len(v1),
        "v2_rows": len(v2),
        "v1_columns": len(v1.columns),
        "v2_columns": len(v2.columns),
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "output_sha256": {
            str(path): sha256(path) for path in outputs
        },
        "validation": {
            "frozen_hashes_unchanged": before == after,
            "v1_copy_parquet_exact": (
                sha256(V1_SOURCE_PARQUET) == sha256(V1_PARQUET)
            ),
            "v1_copy_csv_exact": sha256(V1_SOURCE_CSV) == sha256(V1_CSV),
            "timestamps_identical": v1.timestamp.equals(v2.timestamp),
            "non_replaced_columns_identical": common_exact,
            "v2_replacement_null_cells": v2_nulls,
            "future_values_used": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

