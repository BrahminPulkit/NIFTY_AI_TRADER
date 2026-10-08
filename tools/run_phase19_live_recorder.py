"""Run or audit the Phase 19 research-only Dhan recorder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research_option_recorder import (  # noqa: E402
    DhanOptionChainSource, RecorderConfig, ResearchOptionRecorder,
    audit_session, credentials_from_environment,
)


CONFIG = ROOT / "config" / "phase19_live_recorder_v2.json"
REPORT_ROOT_V1 = ROOT / "reports" / "phase19_live_recorder"


def _rows_to_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def aggregate_audits(config: RecorderConfig) -> dict:
    report_root = (ROOT / "reports" / "phase19_live_recorder_v2"
                   if config.schema_version.endswith("v2") else REPORT_ROOT_V1)
    normalized_root = ROOT / config.normalized_root
    dates = []
    if normalized_root.exists():
        for child in normalized_root.iterdir():
            try:
                dates.append(date.fromisoformat(child.name))
            except ValueError:
                continue
    sessions = [audit_session(item, config) for item in sorted(dates)]
    complete = sum(bool(row["complete"]) for row in sessions)
    summary = {
        "phase": 19, "source": "Dhan v2 Option Chain",
        "endpoint": "POST /v2/optionchain", "sampling_interval_seconds": config.sampling_interval_seconds,
        "target_complete_sessions": 5, "sessions_recorded": len(sessions),
        "complete_sessions": complete, "partial_sessions": len(sessions) - complete,
        "authenticated_schema_test": (
            "AVAILABLE" if os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN")
            else "NOT_RUN_CREDENTIALS_NOT_IN_PROCESS_ENV"),
        "provider_fields": {
            "nifty_volume": "UNAVAILABLE_NULL",
            "trading_session_dte": "UNAVAILABLE_WITHOUT_VERIFIED_EXCHANGE_CALENDAR",
            "provider_timestamp": "RECORDED_WHEN_SUPPLIED",
            "iv_greeks": "RECORDED_WHEN_SUPPLIED",
        },
        "production_behavior_changed": False,
        "verdict": "5_SESSION_ACQUISITION_PASS" if complete >= 5 else "LIVE_DATA_SOURCE_INSUFFICIENT",
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "acquisition_audit.json").write_text(
        json.dumps({"summary": summary, "sessions": sessions}, indent=2), encoding="utf-8")
    (report_root / "report.md").write_text(
        "# Phase 19 Prospective Recorder\n\n"
        f"- Verdict: `{summary['verdict']}`\n"
        f"- Complete sessions: {complete}/5\n"
        f"- Cadence: {config.sampling_interval_seconds} seconds (provider rate compliant)\n"
        "- Universe: causal NIFTY ATM +/-5 strikes, CE and PE\n"
        "- Source: authenticated Dhan v2 Option Chain snapshots\n"
        "- Raw archive: append-only provider JSONL\n"
        "- Normalized archive: contract-level Parquet\n"
        "- NIFTY volume: NULL (not supplied; no VWAP fabricated)\n"
        "- Provider timestamps/IV/Greeks: retained when supplied, otherwise NULL\n\n"
        "## Commands\n\n"
        "```powershell\n"
        "$env:DHAN_CLIENT_ID='<client-id>'\n"
        "$env:DHAN_ACCESS_TOKEN='<access-token>'\n"
        "python tools/run_phase19_live_recorder.py record\n"
        "python tools/run_phase19_live_recorder.py audit\n"
        "```\n\n"
        "Start `record` before 09:15 IST. It waits for open and stops at 15:30 IST. "
        "Credentials remain process-memory only and are never included in archives.\n",
        encoding="utf-8")
    _rows_to_csv(report_root / "session_completeness.csv", sessions)
    field_rows = []
    gap_rows = []
    sync_rows = []
    contract_rows = []
    dte_rows = []
    strike_rows = []
    reliability_rows = []
    integrity_rows = []
    for row in sessions:
        day = row["date"]
        field_rows.append({"date": day, **{key: value for key, value in row.items() if key.startswith("missing_")}})
        gap_rows.append({"date": day, "median_gap_seconds": row.get("median_gap_seconds"),
                         "maximum_gap_seconds": row.get("maximum_gap_seconds")})
        sync_rows.append({"date": day, "available_pct": row.get("synchronization_available_pct"),
                          "within_tolerance_pct": row.get("synchronization_ok_pct"),
                          "stale_quote_count": row.get("stale_quote_count")})
        contract_rows.append({"date": day, "contracts_min": row.get("contracts_min"),
                              "contracts_max": row.get("contracts_max"),
                              "universe_changes": row.get("strike_universe_changes")})
        normalized = ROOT / config.normalized_root / day / "snapshots.parquet"
        if normalized.exists():
            frame = pd.read_parquet(normalized)
            for bucket, count in frame.groupby("dte_bucket").size().items():
                dte_rows.append({"date": day, "dte_bucket": bucket, "rows": int(count)})
            for offset, count in frame.groupby(["option_type", "strike_offset"]).size().items():
                strike_rows.append({"date": day, "option_type": offset[0],
                                    "strike_offset": offset[1], "rows": int(count)})
        reliability_rows.append({"date": day, "status": row.get("status"),
                                 "actual_snapshots": row.get("actual_snapshots")})
        integrity_rows.append({"date": day, "raw_archive_exists": row.get("raw_archive_exists"),
                               "duplicate_rows": row.get("duplicate_normalized_rows"),
                               "conflicting_observations": row.get("conflicting_observations")})
    for name, rows in {
        "field_coverage.csv": field_rows, "timestamp_gaps.csv": gap_rows,
        "synchronization.csv": sync_rows, "contract_universe.csv": contract_rows,
        "dte_distribution.csv": dte_rows, "strike_moneyness_coverage.csv": strike_rows,
        "api_reliability.csv": reliability_rows, "raw_integrity.csv": integrity_rows,
    }.items():
        _rows_to_csv(report_root / name, rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "once", "audit"))
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = RecorderConfig.load(args.config)
    # Paths in configuration are repository-relative regardless of caller cwd.
    for name in ("raw_root", "normalized_root", "audit_root"):
        object.__setattr__(config, name, str(ROOT / getattr(config, name)))
    if args.command == "audit":
        print(json.dumps(aggregate_audits(config), indent=2))
        return 0
    source = DhanOptionChainSource(credentials_from_environment())
    recorder = ResearchOptionRecorder(config, source)
    if args.command == "once":
        health = source.authenticate()
        result = recorder.record_one(args.date or pd.Timestamp.now(tz="Asia/Kolkata").date())
        print(json.dumps({"connection": health["status"], **result}, indent=2))
    else:
        result = recorder.run_session(args.date)
        print(json.dumps(result, indent=2))
    aggregate_audits(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
