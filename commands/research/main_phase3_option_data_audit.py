"""Phase-3 truth audit and acquisition plan; never creates a training dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _date_coverage(option_audit: pd.DataFrame, collected: pd.DataFrame) -> pd.DataFrame:
    data = option_audit.copy()
    data["date"] = data.timestamp.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")
    grouped = data.groupby(["date", "option_type"]).agg(
        source_rows=("timestamp", "size"),
        resolved_rows=("resolution_status", lambda value: int(value.eq("RESOLVED").sum())),
        strikes=("strike", "nunique"),
    ).reset_index()
    totals = grouped.pivot(index="date", columns="option_type", values="source_rows").fillna(0)
    resolved = grouped.pivot(index="date", columns="option_type", values="resolved_rows").fillna(0)
    coverage = pd.DataFrame(index=sorted(set(totals.index) | set(resolved.index)))
    for side in ("CE", "PE"):
        coverage[f"{side.lower()}_source_rows"] = totals.get(side, 0)
        coverage[f"{side.lower()}_resolved_rows"] = resolved.get(side, 0)
        coverage[f"{side.lower()}_identity_complete"] = (
            coverage[f"{side.lower()}_source_rows"].gt(0)
            & coverage[f"{side.lower()}_resolved_rows"].eq(coverage[f"{side.lower()}_source_rows"])
        )
    coverage["legacy_ce_pe_policy_consistent"] = False
    coverage["training_ready"] = False
    if len(collected):
        live = collected.copy()
        live["date"] = pd.to_datetime(live.timestamp, utc=True).dt.tz_convert(
            "Asia/Kolkata").dt.strftime("%Y-%m-%d")
        live_counts = live.groupby(["date", "option_type"]).size().unstack(fill_value=0)
        coverage = coverage.join(live_counts.rename(columns={
            "CE": "forward_exact_ce_rows", "PE": "forward_exact_pe_rows"}), how="outer")
    for name in ("forward_exact_ce_rows", "forward_exact_pe_rows"):
        if name not in coverage:
            coverage[name] = 0
        coverage[name] = coverage[name].fillna(0).astype(int)
    coverage["forward_session_complete"] = (
        coverage.forward_exact_ce_rows.ge(375) & coverage.forward_exact_pe_rows.ge(375))
    return coverage.reset_index(names="date").sort_values("date")


def main() -> None:
    phase2_path = ROOT / "data/normalized/historical_option_contract_audit.parquet"
    collected_paths = sorted((ROOT / "data/collected/dhan_options").glob(
        "*/nifty_atm_options.parquet"))
    option_audit = pd.read_parquet(phase2_path)
    collected = pd.concat([pd.read_parquet(path) for path in collected_paths], ignore_index=True) \
        if collected_paths else pd.DataFrame()
    coverage = _date_coverage(option_audit, collected)
    index = pd.read_parquet(ROOT / "data/features/feature_dataset.parquet", columns=["timestamp"])
    index_dates = set(pd.to_datetime(index.timestamp, utc=True).dt.tz_convert(
        "Asia/Kolkata").dt.strftime("%Y-%m-%d"))
    coverage["index_session_present"] = coverage.date.isin(index_dates)
    coverage["ce_contract_correct_date"] = coverage.ce_identity_complete & coverage.index_session_present
    coverage["pe_contract_correct_date"] = coverage.pe_identity_complete & coverage.index_session_present
    destination = ROOT / "reports/phase3_option_data"
    destination.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(destination / "date_coverage.csv", index=False)

    resolved_dates = coverage.loc[
        coverage.ce_contract_correct_date | coverage.pe_contract_correct_date,
        ["date", "ce_contract_correct_date", "pe_contract_correct_date"]]
    both_identity = coverage.loc[
        coverage.ce_contract_correct_date & coverage.pe_contract_correct_date, "date"].tolist()
    collected_identity = {
        "rows": len(collected),
        "ce_rows": int(collected.option_type.eq("CE").sum()) if len(collected) else 0,
        "pe_rows": int(collected.option_type.eq("PE").sum()) if len(collected) else 0,
        "dates": sorted(pd.to_datetime(collected.timestamp, utc=True).dt.tz_convert(
            "Asia/Kolkata").dt.strftime("%Y-%m-%d").unique().tolist()) if len(collected) else [],
        "identity_complete_rows": int(collected[["timestamp", "underlying", "expiry", "strike",
                                                   "option_type", "security_id"]].notna().all(axis=1).sum())
        if len(collected) else 0,
        "complete_375_candle_sessions": int(coverage.forward_session_complete.sum()),
    }
    phase2 = json.loads((ROOT / "reports/phase2_option_contract_identity/audit.json").read_text(
        encoding="utf-8"))
    audit = {
        "phase": 3,
        "status": "ACQUISITION_REQUIRED_MODEL_VALIDATION_BLOCKED",
        "training_ready_created": False,
        "production_inference_changed": False,
        "model_trained": False,
        "broker_orders_enabled": False,
        "available_sources": {
            "legacy_call": {
                "path": "notebooks/nifty_rolling_options_all_columns_5yr.csv",
                "api": "POST https://api.dhan.co/v2/charts/rollingoption",
                "coverage": ["2021-07-26", "2026-07-24"], "rows": 463488,
                "policy": "MONTH / expiryCode 1 / ATM / CALL",
                "correct_interpretation": "next monthly rolling ATM CE",
                "strike": "complete", "expiry": "absent", "security_id": "absent",
                "training_use": "PROHIBITED_POLICY_AND_IDENTITY_MISMATCH",
            },
            "legacy_put": {
                "path": "data/processed/nifty_atm_put_rolling_1min.parquet",
                "api": "POST https://api.dhan.co/v2/charts/rollingoption",
                "coverage": ["2021-08-09", "2026-08-07"], "rows": 462951,
                "policy": "WEEK / expiryCode 1 / ATM / PUT",
                "correct_interpretation": "next weekly rolling ATM PE",
                "strike": "complete", "expiry": "absent", "security_id": "absent",
                "training_use": "PROHIBITED_POLICY_AND_IDENTITY_MISMATCH",
            },
            "timestamped_security_masters": {
                "count": len(phase2["source"]["timestamped_masters"]),
                "paths": phase2["source"]["timestamped_masters"],
                "coverage": "six sparse snapshots; not continuous 2021-2026",
            },
            "exact_contract_chunks": {
                "path": "data/raw/dhan_option_chunks", "files": 0, "rows": 0,
                "status": "DOWNLOADER_EXISTS_BUT_HAS_NOT_DOWNLOADED_HISTORY",
            },
            "forward_collector": {
                "path": "data/collected/dhan_options/YYYY-MM-DD/nifty_atm_options.parquet",
                **collected_identity,
                "status": "IDENTITY_COMPLETE_BUT_SESSION_HISTORY_INCOMPLETE",
            },
        },
        "official_api_capability": {
            "rollingoption": {
                "history": "up to five years", "maximum_request_days": 30,
                "returns": ["timestamp", "OHLC", "volume", "iv", "oi", "strike", "spot"],
                "does_not_return": ["expiry", "security_id"],
                "conclusion": "cannot independently create the required contract identity",
            },
            "intraday": {
                "history": "up to five years for active instruments", "maximum_request_days": 90,
                "requires": ["security_id", "instrument", "exchange_segment"],
                "conclusion": "cannot discover missing expired security IDs and is not documented as an expired-contract catalog",
            },
            "instrument_master": {
                "returns": ["security_id", "expiry", "strike", "option_type", "expiry_flag"],
                "limitation": "current file does not contain the complete expired five-year catalog",
            },
        },
        "phase2_identity": phase2["validation"],
        "date_coverage": {
            "source_dates": int(len(coverage)),
            "dates_with_complete_ce_identity": int(coverage.ce_contract_correct_date.sum()),
            "dates_with_complete_pe_identity": int(coverage.pe_contract_correct_date.sum()),
            "dates_with_complete_identity_both_sides": both_identity,
            "identity_complete_but_not_index_session": coverage.loc[
                (coverage.ce_identity_complete | coverage.pe_identity_complete)
                & ~coverage.index_session_present, "date"].tolist(),
            "resolved_date_details": resolved_dates.to_dict(orient="records"),
            "date_report": "reports/phase3_option_data/date_coverage.csv",
        },
        "expiry_policy_decision": {
            "policy_version": "nifty_weekly_current_atm_v1",
            "expiry_flag": "WEEK", "expiry_code": 0, "strike": "ATM",
            "sides": ["CE", "PE"],
            "reason": "same current/near weekly contract family for both directions and parity with live earliest-expiry selection",
            "legacy_data_compatible": False,
        },
        "contract_switch_handling": {
            "identity": ["underlying", "expiry", "strike", "option_type", "security_id"],
            "new_segment_when": ["strike changes", "expiry changes", "option_type changes", "security_id changes"],
            "cross_segment_returns_allowed": False, "forward_fill_allowed": False,
        },
        "acquisition_requirement": {
            "preferred": (
                "Acquire contract-specific one-minute NIFTY current-week ATM CE and PE candles from "
                "a historical derivatives vendor/exchange dataset that includes exact expiry and security ID."
            ),
            "dhan_combination": (
                "Re-download both sides with rollingoption WEEK/expiryCode=0 and obtain authoritative "
                "timestamped Dhan/NSE option masters covering every rollover window; map only exact "
                "timestamp/strike/side/expiry records and retain unresolved rows."
            ),
            "minimum_fields": ["timestamp", "underlying", "expiry", "strike", "option_type",
                               "security_id", "open", "high", "low", "close", "volume"],
            "required_range": ["2021-07-26", "2026-08-07"],
            "required_sides": ["CE", "PE"],
            "acceptance": [
                "100% non-null identity on included rows", "same WEEK/code-0 policy for CE and PE",
                "no duplicate timestamp within contract", "no mixed contract segment",
                "authoritative expiry/security ID provenance", "complete session coverage reported",
            ],
        },
        "blocking_reasons": [
            "896367 legacy rows still lack expiry and security ID",
            "legacy CE and PE use different expiry flags",
            "both legacy rolling requests use expiryCode 1 (next), not selected code 0 (current/near)",
            "only six sparse archived masters exist",
            "no exact historical Dhan option chunks exist locally",
            "forward exact collection contains no complete 375-candle CE/PE session",
        ],
    }
    (destination / "audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (destination / "report.md").write_text(
        "# Phase 3 Option Data Report\n\n"
        "Status: **acquisition required; model validation remains blocked**.\n\n"
        "The legacy CALL history is next-month rolling ATM CE. The legacy PUT history is next-week "
        "rolling ATM PE. Dhan rolling responses provide strike but not expiry or contract security ID. "
        "Six local archived masters resolve only 30,072 of 926,439 rows.\n\n"
        "The selected future policy is current/near weekly ATM for both sides: `WEEK`, expiry code `0`, "
        "CE and PE. Existing histories are not compatible with that policy. No training-ready artifact "
        "was created. See `date_coverage.csv` and `audit.json` for exact coverage.\n",
        encoding="utf-8")
    print(json.dumps({"status": audit["status"], "training_ready_created": False,
                      "date_coverage_rows": len(coverage), **collected_identity}, indent=2))


if __name__ == "__main__":
    main()
