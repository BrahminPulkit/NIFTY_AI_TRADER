"""Classify truthful WEEK/code-0 option data without training or synthetic identity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.option_contract_pipeline import normalize_option_contracts


EXPECTED_MINUTES = 375
CANONICAL = [
    "timestamp", "underlying", "expiry", "strike", "option_type", "security_id",
    "open", "high", "low", "close", "volume", "oi", "iv", "spot",
    "contract_segment_id", "classification", "classification_reason", "source",
]


def _master_contracts(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    aliases = {
        "SEM_SMST_SECURITY_ID": "security_id", "SECURITY_ID": "security_id",
        "SEM_INSTRUMENT_NAME": "instrument", "INSTRUMENT": "instrument",
        "SEM_EXPIRY_DATE": "expiry", "SM_EXPIRY_DATE": "expiry",
        "SEM_STRIKE_PRICE": "strike", "STRIKE_PRICE": "strike",
        "SEM_OPTION_TYPE": "option_type", "OPTION_TYPE": "option_type",
        "SEM_EXPIRY_FLAG": "expiry_flag", "EXPIRY_FLAG": "expiry_flag",
        "SM_SYMBOL_NAME": "underlying", "UNDERLYING_SYMBOL": "underlying",
    }
    data = data.rename(columns={name: aliases[name] for name in data if name in aliases})
    required = {"security_id", "instrument", "expiry", "strike", "option_type",
                "expiry_flag", "underlying"}
    if missing := sorted(required.difference(data.columns)):
        raise ValueError(f"Current master missing required contract fields: {missing}")
    mask = (
        data.instrument.astype(str).str.upper().eq("OPTIDX")
        & data.underlying.astype(str).str.strip().str.upper().eq("NIFTY")
        & data.option_type.astype(str).str.upper().isin(["CE", "PE"])
    )
    output = data.loc[mask, ["security_id", "expiry", "strike", "option_type",
                             "expiry_flag", "underlying"]].copy()
    output["security_id"] = output.security_id.astype(str).str.replace(r"\.0$", "", regex=True)
    output["expiry"] = pd.to_datetime(output.expiry).dt.strftime("%Y-%m-%d")
    output["strike"] = pd.to_numeric(output.strike, errors="coerce")
    output["option_type"] = output.option_type.astype(str).str.upper()
    output["expiry_flag"] = output.expiry_flag.astype(str).str.upper().str[:1]
    return output.drop_duplicates(["security_id", "expiry", "strike", "option_type"])


def _validate_forward(rows: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    data["security_id"] = data.security_id.astype(str)
    data["expiry"] = pd.to_datetime(data.expiry).dt.strftime("%Y-%m-%d")
    data["strike"] = pd.to_numeric(data.strike)
    data["option_type"] = data.option_type.astype(str).str.upper()
    lookup = master.rename(columns={
        "expiry_flag": "master_expiry_flag", "underlying": "master_underlying"})
    data = data.merge(lookup, on=["security_id", "expiry", "strike", "option_type"],
                      how="left", validate="many_to_one")
    data["contract_master_match"] = data.master_expiry_flag.eq("W")
    data["near_expiry_match"] = False
    for day, indices in data.groupby(data.timestamp.dt.date).groups.items():
        eligible = master.loc[pd.to_datetime(master.expiry).dt.date >= day]
        nearest = pd.to_datetime(eligible.loc[eligible.expiry_flag.eq("W"), "expiry"]).min()
        if pd.notna(nearest):
            data.loc[indices, "near_expiry_match"] = data.loc[indices, "expiry"].eq(
                nearest.strftime("%Y-%m-%d"))
    valid = data.contract_master_match & data.near_expiry_match
    if not valid.all():
        failed = int((~valid).sum())
        raise ValueError(f"Forward collection contains {failed} rows outside WEEK/current policy")
    data["oi"] = data.get("open_interest")
    if "iv" not in data:
        data["iv"] = pd.NA
    data["source"] = data.get("source", "DHAN_SHARED_CACHE")
    normalized = normalize_option_contracts(data)
    normalized["classification"] = "PARTIAL"
    normalized["classification_reason"] = "CONTRACT_VALID_SESSION_INCOMPLETE"
    return normalized


def _session_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["date", "option_type", "rows", "missing_candles",
                                     "complete_session", "unique_contracts"])
    data = rows.copy()
    data["date"] = data.timestamp.dt.strftime("%Y-%m-%d")
    result = data.groupby(["date", "option_type"]).agg(
        rows=("timestamp", "nunique"), unique_contracts=("contract_segment_id", "nunique"),
        unique_expiries=("expiry", "nunique"), unique_strikes=("strike", "nunique"),
        first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"),
    ).reset_index()
    result["missing_candles"] = (EXPECTED_MINUTES - result.rows).clip(lower=0)
    result["complete_session"] = result.rows.eq(EXPECTED_MINUTES)
    return result


def main() -> None:
    target = ROOT / "data/normalized/historical_options"
    reports = ROOT / "reports/phase4_option_data"
    target.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    collected_paths = sorted((ROOT / "data/collected/dhan_options").glob(
        "*/nifty_atm_options.parquet"))
    collected = pd.concat([pd.read_parquet(path) for path in collected_paths], ignore_index=True) \
        if collected_paths else pd.DataFrame()
    master = _master_contracts(ROOT / "data/raw/dhan_security_master_latest.csv")
    partial = _validate_forward(collected, master) if len(collected) else pd.DataFrame(columns=CANONICAL)
    for name in CANONICAL:
        if name not in partial:
            partial[name] = pd.NA
    partial[CANONICAL].to_parquet(target / "partial.parquet", index=False)

    legacy = pd.read_parquet(ROOT / "data/normalized/historical_option_contract_audit.parquet")
    legacy["oi"] = legacy.get("oi")
    legacy["source"] = legacy.source_name
    legacy["classification"] = "UNRESOLVED"
    legacy["classification_reason"] = "LEGACY_EXPIRY_POLICY_MISMATCH_CODE_1"
    for name in CANONICAL:
        if name not in legacy:
            legacy[name] = pd.NA
    legacy[CANONICAL].to_parquet(target / "unresolved.parquet", index=False)

    daily = _session_coverage(partial)
    daily.to_csv(reports / "daily_coverage.csv", index=False)
    contracts = partial.groupby(
        ["contract_segment_id", "security_id", "expiry", "strike", "option_type"],
        dropna=False).agg(
            rows=("timestamp", "size"), trading_dates=("timestamp", lambda value: value.dt.date.nunique()),
            first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"),
            duplicate_timestamps=("timestamp", lambda value: int(value.duplicated().sum())),
        ).reset_index() if len(partial) else pd.DataFrame()
    contracts.to_csv(reports / "contract_coverage.csv", index=False)

    per_side = {}
    for side in ("CE", "PE"):
        side_rows = partial.loc[partial.option_type.eq(side)]
        side_daily = daily.loc[daily.option_type.eq(side)]
        per_side[side] = {
            "total_rows": len(side_rows),
            "trading_dates": int(side_rows.timestamp.dt.date.nunique()) if len(side_rows) else 0,
            "unique_contracts": int(side_rows.contract_segment_id.nunique()),
            "unique_expiries": int(side_rows.expiry.nunique()),
            "strike_changes": int(side_rows.strike.ne(side_rows.strike.shift()).sum() - bool(len(side_rows))),
            "average_candles_per_session": float(side_daily.rows.mean()) if len(side_daily) else 0.0,
            "complete_sessions": int(side_daily.complete_session.sum()),
            "missing_candles": int(side_daily.missing_candles.sum()),
            "contract_identity_coverage": float(side_rows[["timestamp", "underlying", "expiry", "strike",
                                                            "option_type", "security_id"]].notna().all(axis=1).mean())
            if len(side_rows) else 0.0,
        }
    complete = daily.loc[daily.complete_session].pivot(
        index="date", columns="option_type", values="complete_session") if len(daily) else pd.DataFrame()
    complete_both = complete.index[
        complete.get("CE", False).fillna(False) & complete.get("PE", False).fillna(False)
    ].tolist() if len(complete) and {"CE", "PE"}.issubset(complete.columns) else []
    contamination = int(partial.groupby("contract_segment_id").agg(
        security_ids=("security_id", "nunique"), expiries=("expiry", "nunique"),
        strikes=("strike", "nunique"), sides=("option_type", "nunique")
    ).gt(1).any(axis=1).sum()) if len(partial) else 0
    audit = {
        "phase": 4, "status": "NOT_TRAINING_READY", "policy": {
            "underlying": "NIFTY", "interval": "1m", "strike": "ATM",
            "expiry_flag": "WEEK", "expiry_code": 0, "sides": ["CE", "PE"]},
        "acquisition": {
            "credentials_available_during_run": False,
            "new_historical_download_rows": 0,
            "command": "python -m commands.data.main_download_dhan_weekly_current_options --start YYYY-MM-DD --end YYYY-MM-DD",
            "raw_destination": "data/raw/dhan_rolling_options_weekly_current_v1",
            "rolling_identity_limitation": "Dhan response omits expiry and security_id",
        },
        "classification": {"training_ready_rows": 0, "partial_rows": len(partial),
                           "unresolved_rows": len(legacy)},
        "sides": per_side,
        "complete_sessions_with_both_ce_pe": len(complete_both),
        "complete_session_dates": complete_both,
        "validation": {
            "duplicate_timestamp_within_contract": int(contracts.duplicate_timestamps.sum()) if len(contracts) else 0,
            "cross_contract_contamination": contamination,
            "ce_pe_contamination": contamination,
            "missing_partial_oi": int(partial.oi.isna().sum()),
            "missing_partial_iv": int(partial.iv.isna().sum()),
            "label_boundary_policy": "group by contract_segment_id; cross-segment horizons prohibited",
            "label_boundary_crossings_allowed": False,
        },
        "maximum_trustworthy_period": {
            "start": partial.timestamp.min().isoformat() if len(partial) else None,
            "end": partial.timestamp.max().isoformat() if len(partial) else None,
            "complete_sessions": len(complete_both),
        },
        "minimum_data_required": (
            "A sufficiently broad research sample of complete NIFTY WEEK/code-0 ATM sessions with "
            "both CE and PE, exact expiry/security ID, and outcome horizons contained inside each contract segment."
        ),
        "remaining_gap": {
            "legacy_rows_wrong_policy_or_identity": len(legacy),
            "partial_rows_missing_for_complete_sessions": int(daily.missing_candles.sum()),
            "historical_contract_masters": "complete authoritative rollover coverage unavailable",
        },
        "training_ready_parquet_created": False,
        "model_retrained": False, "real_orders_enabled": False,
    }
    (reports / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (reports / "report.md").write_text(
        "# Phase 4 Contract-Consistent Option Data\n\n"
        "Decision: **NOT TRAINING READY**.\n\n"
        f"- Target policy: NIFTY ATM, WEEK, expiry code 0, CE + PE, 1 minute.\n"
        f"- Partial exact rows: {len(partial):,}.\n"
        f"- Unresolved legacy rows: {len(legacy):,}.\n"
        f"- Complete sessions with both CE and PE: {len(complete_both)}.\n"
        "- `training_ready.parquet` was not created.\n\n"
        "The authenticated historical acquisition could not run because Dhan credentials were not "
        "available. Even after rolling download, authoritative expiry/security-ID mapping remains "
        "mandatory; rolling candles alone cannot satisfy the contract schema.\n",
        encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
