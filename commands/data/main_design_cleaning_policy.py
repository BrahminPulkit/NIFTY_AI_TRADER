from pathlib import Path

from src.config import FIVE_YEAR_INDEX_RAW_DATASET
from src.data_cleaning_policy import design_policy


if __name__ == "__main__":
    result = design_policy(Path(FIVE_YEAR_INDEX_RAW_DATASET), Path("reports/five_year_data_quality/step14b"))
    print("Step 14B policy evidence generated; no cleaning applied.")
    print(f"Raw rows: {result['raw_rows']:,}")
    print(f"Estimated final range: {result['conservative_rows_if_substantive_ohlc_sessions_also_unresolved']:,} - "
          f"{result['target_rows_if_all_missing_bars_recovered']:,}")
