from pathlib import Path

from src.apply_data_cleaning_policy import apply_policy
from src.config import FIVE_YEAR_INDEX_RAW_DATASET


if __name__ == "__main__":
    result = apply_policy(Path(FIVE_YEAR_INDEX_RAW_DATASET), Path("data/cleaned"),
                          Path("reports/five_year_data_quality/step14c"))
    print("Step 14C complete")
    print(f"Cleaned rows: {result['cleaned_rows']:,}")
    print(f"Quarantined rows: {result['quarantined_unique_rows']:,}")

