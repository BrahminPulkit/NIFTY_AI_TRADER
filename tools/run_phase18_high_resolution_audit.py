"""Read-only Phase 18 acquisition gate for high-resolution option research."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OUTPUT = ROOT / "reports/phase18_high_resolution_option_research"
REQUIRED = ["timestamp", "spot", "expiry", "strike", "option_type", "contract_id",
            "ltp", "bid", "ask", "volume", "oi"]
OPTIONAL = ["iv", "delta", "gamma", "theta", "vega"]


def validate_research_frame(frame: pd.DataFrame) -> dict:
    missing = [column for column in REQUIRED if column not in frame.columns]
    result = {"rows": len(frame), "missing_required_columns": missing,
              "schema_complete": not missing}
    if missing or frame.empty:
        result.update({"high_resolution": False, "quote_executable": False})
        return result
    timestamps = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    intervals = timestamps.sort_values().drop_duplicates().diff().dt.total_seconds().dropna()
    modal = float(intervals.mode().iloc[0]) if len(intervals) else None
    result.update({
        "timestamp_parse_failures": int(timestamps.isna().sum()),
        "modal_interval_seconds": modal,
        "high_resolution": bool(modal is not None and modal <= 15),
        "bid_coverage_pct": float(frame.bid.notna().mean() * 100),
        "ask_coverage_pct": float(frame.ask.notna().mean() * 100),
        "quote_executable": bool(modal is not None and modal <= 15
                                 and frame.bid.notna().all() and frame.ask.notna().all()),
    })
    return result


def audit_local_collections() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "data/collected/dhan_options").glob("*/nifty_atm_options.parquet")):
        frame = pd.read_parquet(path)
        timestamps = pd.to_datetime(frame.timestamp, utc=True)
        collected = pd.to_datetime(frame.collected_at, utc=True)
        intervals = timestamps.sort_values().drop_duplicates().diff().dt.total_seconds().dropna()
        rows.append({
            "path": str(path.relative_to(ROOT)), "rows": len(frame),
            "timestamps": int(timestamps.nunique()), "sessions": int(timestamps.dt.date.nunique()),
            "modal_interval_seconds": float(intervals.mode().iloc[0]) if len(intervals) else None,
            "option_types": sorted(frame.option_type.dropna().unique().tolist()),
            "unique_strikes": int(frame.strike.nunique()), "unique_expiries": int(frame.expiry.nunique()),
            "bid_coverage_pct": float(frame.bid.notna().mean() * 100),
            "ask_coverage_pct": float(frame.ask.notna().mean() * 100),
            "iv_coverage_pct": float(frame.iv.notna().mean() * 100) if "iv" in frame else 0.0,
            "greeks_available": bool(set(OPTIONAL[1:]) <= set(frame.columns)),
            "maximum_quote_candle_alignment_error_seconds": float((collected - timestamps).dt.total_seconds().abs().max()),
            "historically_executable": False,
            "reason": "Minute candle timestamps and later live quote collection timestamps are not synchronized.",
        })
    return rows


def main() -> None:
    local = audit_local_collections()
    sources = [
        {
            "source": "LOCAL_VALIDATED_35", "availability": "LOCAL", "resolution": "1 minute",
            "historical_sessions": 35, "bid_ask": False, "multi_strike": False,
            "dte": "Mostly 0DTE/1DTE; limited 2DTE", "oi": True, "volume": True,
            "iv_greeks": False, "training_suitable": False,
            "reason": "ATM-focused OHLCV cannot support quote-executable high-resolution research.",
        },
        {
            "source": "LOCAL_DHAN_QUOTE_CACHE", "availability": "LOCAL", "resolution": "1 minute candles plus asynchronous live quote snapshot",
            "historical_sessions": len(local), "bid_ask": True, "multi_strike": False,
            "dte": "Single active expiry", "oi": False, "volume": True, "iv_greeks": False,
            "training_suitable": False,
            "reason": "Bid/ask collection time is not aligned to historical candle time.",
        },
        {
            "source": "DHAN_INTRADAY_HISTORICAL_V2", "availability": "API", "resolution": "1/5/15/25/60 minute",
            "historical_sessions": None, "bid_ask": False, "multi_strike": "Only individually resolved active instruments",
            "dte": "Contract dependent", "oi": True, "volume": True, "iv_greeks": False,
            "training_suitable": False,
            "reason": "No tick/5-second historical quotes; documentation limits output to candles.",
            "url": "https://dhanhq.co/docs/v2/historical-data/",
        },
        {
            "source": "DHAN_OPTION_CHAIN_V2", "availability": "REAL_TIME_ONLY", "resolution": "Snapshot; one request per 3 seconds",
            "historical_sessions": 0, "bid_ask": True, "multi_strike": True, "dte": "Active expiries",
            "oi": True, "volume": True, "iv_greeks": True, "training_suitable": False,
            "reason": "Can prospectively collect suitable snapshots, but cannot backfill 60-100 historical sessions.",
            "url": "https://dhanhq.co/docs/v2/option-chain/",
        },
        {
            "source": "NSE_HISTORICAL_ORDER_TRADE", "availability": "PAID_SUBSCRIPTION_NOT_CONFIGURED",
            "resolution": "All F&O order/trade ticks plus index ticks", "historical_sessions": None,
            "bid_ask": "Requires deterministic order-book reconstruction", "multi_strike": True,
            "dte": "All listed contracts", "oi": "Not established in tick schema", "volume": True,
            "iv_greeks": False, "training_suitable": "Potentially, after license and validation",
            "reason": "Only verified historical high-resolution path; no subscription/files exist locally.",
            "licensing": "NSE paid historical-data subscription and undertaking required; redistribution terms must be observed.",
            "url": "https://www.nseindia.com/static/market-data/eod-historical-data-subscription",
            "specification": "https://nsearchives.nseindia.com/web/mediaattachment/2026-01/NSE_Hist_Order_Trade_Data_117_20260105112141.pdf",
        },
    ]
    result = {
        "status": "HIGH_RESOLUTION_OPTION_DATA_UNAVAILABLE",
        "verdict": "HIGH_RESOLUTION_DATA_INSUFFICIENT",
        "legitimate_high_resolution_data_obtained": False,
        "usable_resolution": None, "usable_sessions": 0, "usable_strikes_per_session": 0,
        "ce_pe_coverage": "No synchronized high-resolution historical CE/PE quotes",
        "dte_coverage": "No high-resolution historical DTE coverage",
        "bid_ask_coverage": 0.0, "iv_greeks_coverage": 0.0,
        "response_asymmetry_tested": False, "executable_edge_tested": False,
        "reason_not_tested": "Ask-at-entry and bid-at-exit cannot be reconstructed from available minute/asynchronous data.",
        "local_collections": local, "source_audit": sources,
        "exact_missing_fields": ["synchronized <=15-second NIFTY spot", "synchronized option LTP/bid/ask",
                                 "timestamp-specific ATM +/-5 contract panel", "0/1/2/3+ DTE coverage",
                                 "immutable contract ID", "quote-level volume/OI or validated snapshots"],
        "resolution_limitation": "Best legitimate historical local/API resolution is 1 minute.",
        "minimum_required": {
            "sessions": 60, "preferred_sessions": 100, "strikes_each_side_of_atm": 5,
            "sides": ["CE", "PE"], "dte_groups": ["0DTE", "1DTE", "2DTE", "3DTE+"],
            "resolution_seconds_max": 15, "major_subgroup_observations": 100,
            "confirmation_policy": "Untouched sessions plus five-fold expanding walk-forward and one-session embargo",
        },
        "decision": "Pause high-resolution edge testing until licensed historical data is acquired; do not fall back to 1-minute profitability research.",
        "recommended_path": "Obtain licensed NSE F&O historical order/trade plus index tick data, or prospectively archive Dhan option-chain snapshots for 60-100 sessions.",
        "candidate_registry": [], "production_changed": False, "strategy_selected": False,
        "catboost_trained": False, "decision_engine_changed": False,
        "paper_trading_changed": False, "broker_orders_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(sources).to_csv(OUTPUT / "source_audit.csv", index=False)
    pd.DataFrame(local).to_csv(OUTPUT / "local_data_audit.csv", index=False)
    (OUTPUT / "blocker_report.md").write_text(
        "# HIGH_RESOLUTION_OPTION_DATA_UNAVAILABLE\n\n"
        "No synchronized historical <=15-second multi-strike bid/ask dataset is available locally or through the configured historical API.\n\n"
        "Do not substitute minute candles or asynchronous quote snapshots.\n", encoding="utf-8")
    (OUTPUT / "required_schema.json").write_text(json.dumps({
        "required": REQUIRED, "optional_not_to_be_fabricated": OPTIONAL,
        "resolution_seconds_max": 15, "atm_offsets": list(range(-5, 6)),
        "dte_groups": ["0DTE", "1DTE", "2DTE", "3DTE+"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
