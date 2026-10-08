"""Canonical option-contract observations shared by training, replay and live paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OPTION_CONTRACT_SCHEMA_VERSION = "nifty_option_contract_v1"
REQUIRED = (
    "timestamp", "strike", "expiry", "option_type", "underlying",
    "open", "high", "low", "close", "volume",
)
IDENTITY = ("underlying", "option_type", "expiry", "strike", "security_id")


def select_option_contract(frame: pd.DataFrame, *, timestamp, option_type: str,
                           underlying_price: float) -> pd.Series:
    """Select the nearest-expiry ATM contract using the canonical rule for every runtime."""
    data = normalize_option_contracts(frame)
    moment = pd.Timestamp(timestamp)
    moment = (moment.tz_localize("Asia/Kolkata") if moment.tzinfo is None
              else moment.tz_convert("Asia/Kolkata"))
    side = option_type.upper().replace("CALL", "CE").replace("PUT", "PE")
    eligible = data.loc[
        data.timestamp.eq(moment)
        & data.option_type.eq(side)
        & pd.to_datetime(data.expiry).dt.date.__ge__(moment.date())
    ].copy()
    if eligible.empty:
        raise LookupError(f"No eligible {side} contract at {moment.isoformat()}")
    eligible["strike_distance"] = (eligible.strike - float(underlying_price)).abs()
    eligible["expiry_value"] = pd.to_datetime(eligible.expiry)
    # Contract identity is the final deterministic tie-breaker when duplicate quotes exist.
    eligible["security_sort"] = eligible.security_id.fillna("").astype(str)
    return eligible.sort_values(
        ["expiry_value", "strike_distance", "volume", "security_sort"],
        ascending=[True, True, False, True], kind="stable",
    ).iloc[0]


def normalize_option_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate observations and explicitly segment every contract transition."""
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Option contract data missing required fields: {missing}")
    data = frame.copy()
    supplied_segments = "contract_segment_id" in data.columns
    if "security_id" not in data:
        data["security_id"] = pd.NA
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    data["expiry"] = pd.to_datetime(data.expiry, errors="raise").dt.strftime("%Y-%m-%d")
    data["option_type"] = data.option_type.astype(str).str.upper().replace({"CALL": "CE", "PUT": "PE"})
    if not data.option_type.isin(["CE", "PE"]).all():
        raise ValueError("option_type must be CE or PE")
    data["underlying"] = data.underlying.astype(str).str.upper()
    for name in ("strike", "open", "high", "low", "close", "volume"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    invalid = (
        data[["strike", "open", "high", "low", "close", "volume"]].isna().any(axis=1)
        | data[["strike", "open", "high", "low", "close"]].le(0).any(axis=1)
        | data.volume.lt(0) | data.high.lt(data.low)
        | data.open.lt(data.low) | data.open.gt(data.high)
        | data.close.lt(data.low) | data.close.gt(data.high)
    )
    if invalid.any():
        raise ValueError(f"Invalid option contract observations: {int(invalid.sum())}")
    data = data.sort_values([*IDENTITY, "timestamp"], kind="stable")
    if data.duplicated([*IDENTITY, "timestamp"]).any():
        raise ValueError("Duplicate timestamp within the same option contract")
    identity = data[list(IDENTITY)].fillna("").astype(str).agg("|".join, axis=1)
    if supplied_segments:
        if data.contract_segment_id.isna().any():
            raise ValueError("Explicit contract_segment_id cannot be null")
        segment_identity_counts = pd.DataFrame({
            "segment": data.contract_segment_id.astype(str), "identity": identity,
        }).groupby("segment").identity.nunique()
        if segment_identity_counts.gt(1).any():
            raise ValueError("A contract segment contains multiple contract identities")
        data["contract_segment_id"] = data.contract_segment_id.astype(str)
    else:
        data["contract_segment_id"] = identity.map(
            lambda value: hashlib.sha256(value.encode()).hexdigest()[:20])
    data["contract_schema_version"] = OPTION_CONTRACT_SCHEMA_VERSION
    return data.reset_index(drop=True)


def contract_fingerprint(frame: pd.DataFrame) -> str:
    data = normalize_option_contracts(frame)
    payload = data[["timestamp", *IDENTITY, "contract_segment_id"]].to_json(
        orient="records", date_format="iso")
    return hashlib.sha256(payload.encode()).hexdigest()


def artifact_manifest(feature_order: list[str], feature_version: str,
                      preprocessing: dict, configuration: dict) -> dict:
    return {
        "manifest_version": 1,
        "feature_order": feature_order,
        "feature_version": feature_version,
        "option_contract_schema_version": OPTION_CONTRACT_SCHEMA_VERSION,
        "preprocessing": preprocessing,
        "configuration": configuration,
    }


def validate_artifact_manifest(path: str | Path, *, feature_order: list[str],
                               feature_version: str) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = {
        "feature_order": manifest.get("feature_order") == feature_order,
        "feature_version": manifest.get("feature_version") == feature_version,
        "contract_schema": manifest.get("option_contract_schema_version") == OPTION_CONTRACT_SCHEMA_VERSION,
        "preprocessing": isinstance(manifest.get("preprocessing"), dict),
        "configuration": isinstance(manifest.get("configuration"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"MODEL_VALIDATION_BLOCKED: incompatible artifact {failed}")
    return manifest
