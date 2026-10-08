"""Acquire raw NIFTY ATM CE and PE rolling candles with WEEK/expiryCode=0."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dhan_rolling_option_downloader import RollingOptionDownloader, request_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("--start must be earlier than exclusive --end")
    plans = {
        side: request_plan(args.start, args.end, option_type=side,
                           expiry_flag="WEEK", expiry_code=0)
        for side in ("CALL", "PUT")
    }
    if args.plan_only:
        print(json.dumps({
            "policy": "WEEK_EXPIRY_CODE_0_ATM", "requests": sum(map(len, plans.values())),
            "call_requests": len(plans["CALL"]), "put_requests": len(plans["PUT"]),
            "first_call": plans["CALL"][0], "last_put": plans["PUT"][-1],
        }, indent=2))
        return
    downloader = RollingOptionDownloader(
        "data/raw/dhan_rolling_options_weekly_current_v1")
    files = []
    for side in ("CALL", "PUT"):
        files.extend(downloader.download(
            args.start, args.end, option_type=side, expiry_flag="WEEK", expiry_code=0))
    print(json.dumps({"status": "complete", "policy": "WEEK_EXPIRY_CODE_0_ATM",
                      "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
