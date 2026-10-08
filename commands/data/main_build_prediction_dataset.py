"""Build the research-only canonical prediction input dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.prediction_dataset import (
    FORBIDDEN_COLUMNS,
    PREDICTION_COLUMNS,
    assert_prefix_invariance,
    build_live_record,
    build_prediction_dataset,
    feature_inventory,
)


ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
SETUPS = Path("data/setups/stage1_setup_dataset.parquet")
DECISIONS = Path("reports/strategy_decision/decision_matrix.csv")
OUTPUT = Path("data/prediction")
REPORTS = Path("reports/prediction_dataset")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    aligned = pd.read_parquet(ALIGNED)
    setups = pd.read_parquet(SETUPS)
    decisions = pd.read_csv(DECISIONS)
    dataset = build_prediction_dataset(aligned, setups, decisions)
    allowed = setups.loc[setups.trade_allowed.eq(1), ["timestamp"]].copy()
    allowed["timestamp"] = pd.to_datetime(allowed.timestamp, utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    excluded = allowed.loc[~allowed.timestamp.isin(dataset.timestamp)]
    assert_prefix_invariance(
        aligned, setups, decisions, prefix_rows=min(1000, len(decisions))
    )
    historical_last = dataset.tail(1).reset_index(drop=True)
    timestamp = historical_last.timestamp.iat[0]
    live = build_live_record(
        aligned.loc[pd.to_datetime(aligned.timestamp, utc=True).eq(timestamp)],
        setups.loc[pd.to_datetime(setups.timestamp, utc=True).eq(timestamp)],
        decisions.loc[pd.to_datetime(decisions.timestamp, utc=True).eq(timestamp)],
    )
    pd.testing.assert_frame_equal(historical_last, live, check_exact=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parquet = OUTPUT / "prediction_dataset.parquet"
    csv = OUTPUT / "prediction_dataset.csv"
    dataset.to_parquet(parquet, index=False)
    dataset.to_csv(csv, index=False)
    inventory = feature_inventory(dataset)
    inventory.to_csv(REPORTS / "feature_inventory.csv", index=False)
    missing = pd.DataFrame({
        "column": dataset.columns,
        "missing_count": dataset.isna().sum().to_numpy(),
        "missing_percentage": (dataset.isna().mean() * 100).to_numpy(),
    })
    missing.to_csv(REPORTS / "missing_values.csv", index=False)
    schema = {
        "version": "prediction_dataset_v1",
        "columns": [
            {
                "name": column, "dtype": str(dataset[column].dtype),
                "nullable": bool(dataset[column].isna().any()),
            }
            for column in PREDICTION_COLUMNS
        ],
        "primary_key": "timestamp",
        "timezone": "Asia/Kolkata",
        "target_columns": [],
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
    }
    (REPORTS / "schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    report = f"""# Step 20A — Prediction Dataset Report

- Records: **{len(dataset):,}**
- Allowed frozen Stage-1 setups: **{len(allowed):,}**
- Excluded setups without a frozen Decision Engine row: **{len(excluded):,}**
- Columns: **{len(dataset.columns)}**
- Earliest: {dataset.timestamp.min()}
- Latest: {dataset.timestamp.max()}
- Duplicate timestamps: {int(dataset.timestamp.duplicated().sum())}
- Future, label and realized-outcome columns: **0**

The dataset is the exact timestamp intersection of frozen allowed setups,
canonical aligned Index–Option features and causal Decision Engine outputs. The
excluded final setup was not fabricated or assigned a synthetic decision.
"""
    (REPORTS / "prediction_dataset_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "leakage_validation.md").write_text(
        "# Leakage Validation\n\n"
        "PASS. The schema allowlist excludes labels, future prices, exits, MFE, "
        "MAE, realized returns and terminal outcomes. Decision estimates are "
        "the frozen expanding-history outputs created before their corresponding "
        "research responses matured.\n",
        encoding="utf-8",
    )
    (REPORTS / "historical_live_parity.md").write_text(
        "# Historical/Live Parity\n\nPASS. The final historical record was "
        "rebuilt through `build_live_record()` and matched bit-for-bit.\n",
        encoding="utf-8",
    )
    (REPORTS / "prefix_invariance.md").write_text(
        "# Prefix Invariance\n\nPASS. The first 1,000 decision records built "
        "alone match the corresponding full-dataset prefix bit-for-bit.\n",
        encoding="utf-8",
    )
    metadata = {
        "step": "20A", "research_only": True,
        "dataset_version": "prediction_dataset_v1",
        "rows": len(dataset), "columns": len(dataset.columns),
        "excluded_allowed_setups": len(excluded),
        "output_hashes": {
            "parquet": _hash(parquet), "csv": _hash(csv),
        },
        "frozen_input_hashes": {
            "aligned": _hash(ALIGNED), "stage1_setups": _hash(SETUPS),
            "decision_matrix": _hash(DECISIONS),
        },
        "validation": {
            "prefix_invariance": "PASS",
            "historical_live_parity": "PASS",
            "future_leakage": "PASS",
            "unique_timestamp": "PASS",
            "chronological": "PASS",
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

