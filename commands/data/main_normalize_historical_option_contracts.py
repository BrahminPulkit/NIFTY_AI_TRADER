"""Build an evidence-bound historical option identity audit without training models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.historical_option_contract_resolver import (
    build_snapshot_catalog, normalize_rolling_source, resolve_contract_identity,
)


def main() -> None:
    call_path = ROOT / "notebooks/nifty_rolling_options_all_columns_5yr.csv"
    put_path = ROOT / "data/processed/nifty_atm_put_rolling_1min.parquet"
    master_paths = sorted((ROOT / "backup/reconstruction/data/security_master_archive").glob(
        "dhan_master_*.csv"))
    call_raw = pd.read_csv(call_path, low_memory=False)
    put_raw = pd.read_parquet(put_path)
    call = normalize_rolling_source(
        call_raw, option_type="CE", expiry_flag="M",
        source_name=str(call_path.relative_to(ROOT)))
    put = normalize_rolling_source(
        put_raw, option_type="PE", expiry_flag="W",
        source_name=str(put_path.relative_to(ROOT)))
    catalog = build_snapshot_catalog(master_paths)
    normalized = resolve_contract_identity(pd.concat([call, put], ignore_index=True), catalog)

    output = ROOT / "data/normalized"
    reports = ROOT / "reports/phase2_option_contract_identity"
    output.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output / "historical_option_contract_audit.parquet", index=False)
    resolved = normalized.loc[normalized.resolution_status.eq("RESOLVED")].copy()
    unresolved = normalized.loc[normalized.resolution_status.ne("RESOLVED")].copy()
    canonical = ["timestamp", "underlying", "expiry", "strike", "option_type", "security_id",
                 "open", "high", "low", "close", "volume", "contract_segment_id"]
    resolved[canonical].to_parquet(output / "historical_options_resolved.parquet", index=False)
    unresolved.to_parquet(output / "historical_options_unresolved.parquet", index=False)

    duplicate = int(resolved.duplicated(["timestamp", "security_id"]).sum())
    contamination = int(resolved.groupby("contract_segment_id").agg(
        strikes=("strike", "nunique"), expiries=("expiry", "nunique"),
        sides=("option_type", "nunique"), ids=("security_id", "nunique")
    ).gt(1).any(axis=1).sum()) if len(resolved) else 0
    contracts = resolved.groupby(
        ["expiry", "strike", "option_type", "security_id"], dropna=False
    ).agg(rows=("timestamp", "size"), start=("timestamp", "min"), end=("timestamp", "max"))
    contracts.reset_index().to_csv(reports / "resolved_contracts.csv", index=False)
    per_expiry = contracts.reset_index().groupby("expiry").agg(
        contracts=("security_id", "nunique"), strikes=("strike", "nunique"),
        rows=("rows", "sum"), start=("start", "min"), end=("end", "max")
    ).reset_index()
    per_expiry.to_csv(reports / "contracts_per_expiry.csv", index=False)
    report = {
        "status": "PARTIALLY_RESOLVED_NOT_TRAINING_ELIGIBLE",
        "source": {
            "call": {"path": str(call_path.relative_to(ROOT)), "provider": "Dhan rollingoption",
                     "request": {"securityId": 13, "instrument": "OPTIDX", "strike": "ATM",
                                 "expiryFlag": "MONTH", "expiryCode": 1, "drvOptionType": "CALL"},
                     "rows": len(call), "columns": call_raw.columns.tolist(),
                     "timestamp_storage": "ISO-8601 text with +05:30 offset",
                     "date_range": [call.timestamp.min().isoformat(), call.timestamp.max().isoformat()],
                     "identity_in_source": {"option_type": False, "strike": True, "expiry": False,
                                            "security_id": False, "underlying": False},
                     "series_type": "ROLLING_ATM; strike may change candle by candle",
                     "intraday_strike_changes": int((call.strike.ne(call.strike.shift()) &
                                                     call.timestamp.dt.date.eq(call.timestamp.shift().dt.date)).sum())},
            "put": {"path": str(put_path.relative_to(ROOT)), "provider": "Dhan rollingoption",
                    "request": {"securityId": 13, "instrument": "OPTIDX", "strike": "ATM",
                                "expiryFlag": "WEEK", "expiryCode": 1, "drvOptionType": "PUT"},
                    "rows": len(put), "columns": put_raw.columns.tolist(),
                    "timestamp_storage": str(put_raw.timestamp.dtype),
                    "date_range": [put.timestamp.min().isoformat(), put.timestamp.max().isoformat()],
                    "identity_in_source": {"option_type": True, "strike": True, "expiry": False,
                                           "security_id": False, "underlying": False},
                    "series_type": "ROLLING_ATM; strike may change candle by candle",
                    "intraday_strike_changes": int((put.strike.ne(put.strike.shift()) &
                                                    put.timestamp.dt.date.eq(put.timestamp.shift().dt.date)).sum())},
            "timestamped_masters": [str(path.relative_to(ROOT)) for path in master_paths],
        },
        "validation": {
            "total_option_rows": len(normalized), "resolved_contract_rows": len(resolved),
            "unresolved_rows": len(unresolved),
            "call_rows": int(normalized.option_type.eq("CE").sum()),
            "put_rows": int(normalized.option_type.eq("PE").sum()),
            "missing_strike": int(normalized.strike.isna().sum()),
            "missing_expiry": int(normalized.expiry.isna().sum()),
            "missing_security_id": int(normalized.security_id.isna().sum()),
            "duplicate_timestamps_within_contract": duplicate,
            "cross_contract_contamination": contamination,
            "unique_expiries": int(resolved.expiry.nunique()),
            "unique_strikes": int(resolved.strike.nunique()),
            "unique_contracts": int(resolved.contract_segment_id.nunique()),
            "resolved_call_rows": int(resolved.option_type.eq("CE").sum()),
            "resolved_put_rows": int(resolved.option_type.eq("PE").sum()),
            "unresolved_reasons": unresolved.resolution_reason.value_counts().to_dict(),
            "contracts_per_expiry": json.loads(per_expiry.to_json(orient="records", date_format="iso")),
        },
        "training_allowed": False,
        "model_validation_blocked": True,
        "acquisition_requirement": (
            "Acquire timestamped Dhan/NSE option contract masters covering every historical trading "
            "date, or exact per-candle contract history containing expiry and security_id for the "
            "recorded strike/side. CALL requires MONTH expiryCode=1 identity; PUT requires WEEK "
            "expiryCode=1 identity."
        ),
    }
    (reports / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (reports / "data_acquisition_requirement.md").write_text(
        "# Historical Contract Data Requirement\n\n"
        "The partial normalized artifact is not training eligible. Obtain either:\n\n"
        "1. Exact historical option candles keyed by `timestamp, underlying, expiry, strike, "
        "option_type, security_id`, including OHLCV; or\n"
        "2. Complete timestamped Dhan/NSE security-master snapshots covering every trading date "
        "in 2021-07-26 through 2026-08-07.\n\n"
        "CALL provenance requires the nearest monthly (`MONTH`, expiry code 1) ATM CE contract. "
        "PUT provenance requires the nearest weekly (`WEEK`, expiry code 1) ATM PE contract. "
        "A current master, a calculated expiry calendar without authoritative holiday adjustments, "
        "or forward-filled identities are not acceptable substitutes.\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
