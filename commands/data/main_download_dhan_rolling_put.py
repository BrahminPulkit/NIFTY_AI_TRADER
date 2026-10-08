"""Download five years of NIFTY ATM weekly rolling PUT minute data."""

from __future__ import annotations

import argparse
from datetime import date
import json

import pandas as pd

from src.dhan_rolling_option_downloader import RollingOptionDownloader, request_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("--start must be earlier than exclusive --end")
    plan = request_plan(args.start, args.end)
    if args.plan_only:
        print(json.dumps({"requests": len(plan), "first": plan[0], "last": plan[-1]}, indent=2))
        return
    files = RollingOptionDownloader().download(args.start, args.end)
    rows = sum(len(pd.read_parquet(path, columns=["timestamp"])) for path in files)
    print(json.dumps({"status": "complete", "chunks": len(files), "rows": rows,
                      "start": str(args.start), "end": str(args.end)}, indent=2))


if __name__ == "__main__":
    main()
