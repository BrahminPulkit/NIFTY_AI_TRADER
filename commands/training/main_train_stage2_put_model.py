"""Train and validate the five-year Stage-2 PUT candidate."""

from pathlib import Path
import json

from src.put_model_training import train_and_validate


def main() -> None:
    result = train_and_validate(
        Path("data/prediction/prediction_dataset_v1.parquet"),
        Path("data/processed/nifty_atm_put_rolling_1min.parquet"),
        Path("models/stage2_put_candidate_v1"),
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
