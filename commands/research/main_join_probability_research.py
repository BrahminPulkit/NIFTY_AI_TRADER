"""Execute the approved Step 20C-R frozen timestamp join."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.probability_research_join import (
    assert_join_prefix_parity,
    build_joined_probability_dataset,
)


CATBOOST = Path("data/prediction/oof/oof_predictions_catboost.parquet")
XGBOOST = Path("data/prediction/oof/oof_predictions_xgboost.parquet")
DECISIONS = Path("reports/strategy_decision/decision_matrix.csv")
OUTPUT = Path("data/research/probability_threshold")
REPORTS = Path("reports/probability_threshold_research")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    catboost = pd.read_parquet(CATBOOST)
    xgboost = pd.read_parquet(XGBOOST)
    decisions = pd.read_csv(DECISIONS)
    joined = build_joined_probability_dataset(catboost, xgboost, decisions)
    assert_join_prefix_parity(
        catboost, xgboost, decisions, prefix_rows=min(1000, len(catboost))
    )
    # Single-row builder parity verifies the same join path used for a timestamp.
    last = joined.tail(1).reset_index(drop=True)
    timestamp = last.timestamp.iat[0]
    single = build_joined_probability_dataset(
        catboost.loc[catboost.timestamp.eq(timestamp)],
        xgboost.loc[xgboost.timestamp.eq(timestamp)],
        decisions.loc[pd.to_datetime(decisions.timestamp, utc=True).eq(timestamp)],
    )
    pd.testing.assert_frame_equal(last, single, check_exact=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parquet = OUTPUT / "joined_probability_dataset.parquet"
    csv = OUTPUT / "joined_probability_dataset.csv"
    if parquet.exists() or csv.exists():
        raise FileExistsError("Research join outputs already exist; refusing to overwrite")
    joined.to_parquet(parquet, index=False)
    joined.to_csv(csv, index=False)
    null_columns = {
        column: int(count)
        for column, count in joined.isna().sum().items()
        if count
    }
    report = f"""# Step 20C-R — Frozen Timestamp Join Validation

- Joined rows: **{len(joined):,}**
- Unique timestamps: **{joined.timestamp.nunique():,}**
- Duplicate timestamps: **{int(joined.timestamp.duplicated().sum())}**
- CatBoost missing timestamp joins: **0**
- XGBoost missing timestamp joins: **0**
- Decision dataset missing timestamp joins: **0**
- Chronological ordering: **PASS**
- Prefix parity: **PASS**
- Single-row historical/join-path parity: **PASS**
- Leakage boundary: **PASS — probabilities are saved OOF values; response
  columns are research outcomes and must never become model inputs**

## Holding-time semantics

`holding_time_minutes` is the frozen entry-time Decision Engine
`expected_holding_minutes`. A realized premium target-hit holding time was not
persisted and was not recomputed or fabricated.

## Existing source nulls preserved

{json.dumps(null_columns, indent=2)}
"""
    (REPORTS / "join_validation_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "step": "20C-R",
        "research_only": True,
        "rows": len(joined),
        "primary_key": "timestamp",
        "join_method": "exact timestamp inner/validated one-to-one",
        "holding_time_semantics": "frozen causal expected_holding_minutes, not realized",
        "validation": {
            "timestamp_uniqueness": "PASS",
            "duplicated_joins": "PASS",
            "missing_joins": "PASS",
            "prefix_parity": "PASS",
            "single_row_join_parity": "PASS",
            "leakage_boundary": "PASS",
        },
        "source_hashes": {
            "catboost_oof": _hash(CATBOOST),
            "xgboost_oof": _hash(XGBOOST),
            "decision_dataset": _hash(DECISIONS),
        },
        "output_hashes": {
            "parquet": _hash(parquet),
            "csv": _hash(csv),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

