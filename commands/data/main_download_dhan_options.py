"""Download contract-aware NIFTY CE/PE history from archived Dhan masters."""

from __future__ import annotations

import argparse
from datetime import date

from src.dhan_data_downloader import (
    DhanHistoricalDownloader, discover_nifty_options, load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dhan_option_history.json")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Exclusive end date (YYYY-MM-DD)")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("--start must be earlier than --end")
    config = load_config(args.config)
    contracts = discover_nifty_options(config["security_master_paths"], config)
    with DhanHistoricalDownloader(config) as downloader:
        files = downloader.download_contracts(contracts, args.start, args.end)
    print(f"Downloaded or resumed {len(files)} validated option chunks")


if __name__ == "__main__":
    main()
