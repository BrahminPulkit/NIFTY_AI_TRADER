"""CLI entry point for resumable Dhan NIFTY futures data acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.continuous_futures_builder import build_continuous_futures, validate_continuous, write_quality_report
from src.dhan_data_downloader import (
    DhanHistoricalDownloader, catalog_coverage, discover_nifty_futures,
    load_config, plan_contract_requests,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dhan_historical_data.json")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--start", help="Inclusive date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Exclusive date in YYYY-MM-DD format.")
    parser.add_argument("--plan-only", action="store_true", help="Print exact payloads without API calls.")
    args = parser.parse_args()
    config = load_config(args.config)
    contracts = discover_nifty_futures(config["security_master_paths"], config)
    today = pd.Timestamp.now(tz=config["timezone"]).date()
    desired_start = pd.Timestamp(args.start).date() if args.start else (
        pd.Timestamp(today) - pd.DateOffset(years=int(config["history_years"])
    )).date()
    desired_end = pd.Timestamp(args.end).date() if args.end else today + pd.Timedelta(days=1)
    if desired_start >= desired_end:
        raise ValueError("--start must be earlier than the exclusive --end")
    coverage = catalog_coverage(contracts, desired_start, desired_end - pd.Timedelta(days=1))
    plans = plan_contract_requests(contracts, desired_start, desired_end, config)
    if args.plan_only:
        print(json.dumps({"endpoint": "https://api.dhan.co/v2/charts/intraday", "requests": plans}, indent=2))
        return
    if args.catalog_only:
        write_catalog_only_report(config, coverage)
        print(json.dumps(coverage, indent=2))
        return

    with DhanHistoricalDownloader(config) as downloader:
        chunks = downloader.download_contracts(contracts, desired_start, desired_end)
    continuous, rolls = build_continuous_futures(chunks, contracts, config["rollover_trading_days"])
    quality, missing = validate_continuous(continuous, rolls, config)
    paths = config["paths"]
    Path(paths["continuous_csv"]).parent.mkdir(parents=True, exist_ok=True)
    continuous.to_csv(paths["continuous_csv"], index=False)
    continuous.to_parquet(paths["continuous_parquet"], index=False)
    Path(paths["rollover_report"]).parent.mkdir(parents=True, exist_ok=True)
    rolls.to_csv(paths["rollover_report"], index=False)
    missing.to_csv(paths["missing_candles"], index=False)
    write_quality_report(paths["quality_report"], quality, coverage, config)
    quality_passed = bool(
        quality["total_rows"] > 0 and quality["duplicate_timestamps"] == 0
        and quality["missing_candles"] == 0 and quality["out_of_session_timestamps"] == 0
        and quality["invalid_contract_transitions"] == 0
    )
    metadata = {
        "status": "official" if coverage["five_year_catalog_complete"] and quality_passed else "validation_required_not_official",
        "quality_gates_passed": quality_passed,
        "source": "Dhan historical intraday API",
        "generated_at": pd.Timestamp.now(tz=config["timezone"]).isoformat(),
        "continuous_csv": paths["continuous_csv"], "continuous_parquet": paths["continuous_parquet"],
        "quality": quality, "catalog_coverage": coverage,
    }
    Path(paths["official_source_metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def write_catalog_only_report(config, coverage):
    paths = config["paths"]
    quality = {"total_rows": 0, "trading_days": 0, "earliest_date": None, "latest_date": None,
               "missing_candles": 0, "duplicate_timestamps": 0, "source_duplicate_contract_timestamps": 0,
               "out_of_session_timestamps": 0,
               "large_same_contract_price_gaps": 0, "contract_rolls": 0,
               "invalid_contract_transitions": 0, "timezone": config["timezone"]}
    write_quality_report(paths["quality_report"], quality, coverage, config)
    Path(paths["rollover_report"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["roll_date", "from_security_id", "to_security_id", "from_expiry", "to_expiry",
                          "from_last_close", "to_first_open", "transition_gap_points",
                          "transition_gap_percent", "transition_valid"]).to_csv(paths["rollover_report"], index=False)
    pd.DataFrame(columns=["timestamp", "trading_date"]).to_csv(paths["missing_candles"], index=False)


if __name__ == "__main__":
    main()
