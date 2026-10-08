"""Roadmap Step 1 entry point: audit raw five-year data, and nothing else."""

from pathlib import Path

from src.config import FIVE_YEAR_INDEX_RAW_DATASET
from src.five_year_data_validator import audit_dataset


if __name__ == "__main__":
    result = audit_dataset(Path(FIVE_YEAR_INDEX_RAW_DATASET), Path("reports/five_year_data_quality"))
    print("Five-year data audit complete")
    print(f"Rows: {result['source_rows']:,}")
    print(f"Range: {result['earliest_timestamp']} -> {result['latest_timestamp']}")
    print(f"Missing candles: {result['missing_regular_session_candles']:,}")
    print(f"Out-of-session rows: {result['out_of_session_rows']:,}")
    print(f"Approved for feature generation: {result['approved_for_feature_generation']}")

