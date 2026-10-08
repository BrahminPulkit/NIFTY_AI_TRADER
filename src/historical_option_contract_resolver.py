"""Evidence-bound reconstruction of rolling Dhan option contract identity."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


MASTER_ALIASES = {
    "SEM_SMST_SECURITY_ID": "security_id",
    "SECURITY_ID": "security_id",
    "SEM_INSTRUMENT_NAME": "instrument",
    "INSTRUMENT": "instrument",
    "SEM_EXPIRY_DATE": "expiry",
    "SM_EXPIRY_DATE": "expiry",
    "SEM_STRIKE_PRICE": "strike",
    "STRIKE_PRICE": "strike",
    "SEM_OPTION_TYPE": "option_type",
    "OPTION_TYPE": "option_type",
    "SEM_EXPIRY_FLAG": "expiry_flag",
    "EXPIRY_FLAG": "expiry_flag",
    "SEM_CUSTOM_SYMBOL": "display_name",
    "DISPLAY_NAME": "display_name",
    "SM_SYMBOL_NAME": "underlying",
    "UNDERLYING_SYMBOL": "underlying",
}


def snapshot_timestamp(path: str | Path) -> pd.Timestamp:
    match = re.search(r"dhan_master_(\d{14})\.csv$", Path(path).name)
    if not match:
        raise ValueError(f"Security master has no timestamped snapshot identity: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S").tz_localize("Asia/Kolkata")


def load_nifty_option_master(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    data = pd.read_csv(source, low_memory=False)
    data.columns = data.columns.str.strip()
    data = data.rename(columns={name: MASTER_ALIASES[name] for name in data if name in MASTER_ALIASES})
    required = {"security_id", "instrument", "expiry", "strike", "option_type",
                "expiry_flag", "display_name"}
    if missing := sorted(required.difference(data.columns)):
        raise ValueError(f"Archived master {source} missing fields: {missing}")
    underlying = data.get("underlying", pd.Series("", index=data.index)).astype(str).str.strip().str.upper()
    display = data.display_name.astype(str).str.strip().str.upper()
    mask = (
        data.instrument.astype(str).str.upper().eq("OPTIDX")
        & (underlying.eq("NIFTY") | display.str.startswith("NIFTY "))
        & data.option_type.astype(str).str.upper().isin(["CE", "PE"])
    )
    result = data.loc[mask, ["security_id", "expiry", "strike", "option_type",
                             "expiry_flag", "display_name"]].copy()
    result["security_id"] = result.security_id.astype(str).str.replace(r"\.0$", "", regex=True)
    result["expiry"] = pd.to_datetime(result.expiry, errors="coerce").dt.normalize()
    result["strike"] = pd.to_numeric(result.strike, errors="coerce")
    result["option_type"] = result.option_type.astype(str).str.upper()
    result["expiry_flag"] = result.expiry_flag.astype(str).str.upper().str[:1]
    result = result.dropna(subset=["expiry", "strike"])
    result["snapshot_timestamp"] = snapshot_timestamp(source)
    result["master_source"] = str(source)
    return result.drop_duplicates(
        ["security_id", "expiry", "strike", "option_type", "expiry_flag"]
    ).reset_index(drop=True)


def build_snapshot_catalog(paths: list[str | Path]) -> pd.DataFrame:
    frames = [load_nifty_option_master(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["snapshot_timestamp", "expiry", "strike", "option_type"], kind="stable"
    ).reset_index(drop=True)


def normalize_rolling_source(frame: pd.DataFrame, *, option_type: str,
                             expiry_flag: str, source_name: str) -> pd.DataFrame:
    required = {"timestamp", "strike", "open", "high", "low", "close", "volume"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Rolling source missing fields: {missing}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    data["underlying"] = "NIFTY"
    data["option_type"] = option_type.upper().replace("CALL", "CE").replace("PUT", "PE")
    data["requested_expiry_flag"] = expiry_flag.upper()[0]
    data["source_name"] = source_name
    data["source_row"] = range(len(data))
    return data


def resolve_contract_identity(rows: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Resolve only inside a snapshot's evidence window; never extrapolate after nearest expiry."""
    output = rows.copy()
    for name in ("expiry", "security_id", "master_source", "contract_segment_id"):
        output[name] = pd.NA
    output["resolution_status"] = "UNRESOLVED"
    output["resolution_reason"] = "NO_ELIGIBLE_TIMESTAMPED_MASTER"
    if catalog.empty:
        return output
    for captured, master in catalog.groupby("snapshot_timestamp", sort=True):
        later = catalog.loc[catalog.snapshot_timestamp.gt(captured), "snapshot_timestamp"]
        next_snapshot = later.min() if len(later) else pd.NaT
        for flag in output.requested_expiry_flag.unique():
            scoped = master.loc[master.expiry_flag.eq(flag)]
            future = scoped.loc[scoped.expiry.dt.date >= captured.date(), "expiry"]
            if future.empty:
                continue
            evidence_end = future.min().date()
            row_mask = (
                output.timestamp.ge(captured)
                & (output.timestamp.dt.date <= evidence_end)
                & output.requested_expiry_flag.eq(flag)
            )
            if pd.notna(next_snapshot):
                row_mask &= output.timestamp.lt(next_snapshot)
            indices = output.index[row_mask & output.resolution_status.eq("UNRESOLVED")]
            if not len(indices):
                continue
            keys = output.loc[indices, ["strike", "option_type"]].drop_duplicates()
            for key in keys.itertuples(index=False):
                candidates = scoped.loc[
                    scoped.strike.eq(float(key.strike))
                    & scoped.option_type.eq(key.option_type)
                    & (scoped.expiry.dt.date >= captured.date())
                ].sort_values("expiry")
                if candidates.empty:
                    selection = indices[
                        output.loc[indices, "strike"].eq(float(key.strike))
                        & output.loc[indices, "option_type"].eq(key.option_type)]
                    output.loc[selection, "resolution_reason"] = "CONTRACT_NOT_IN_ELIGIBLE_MASTER"
                    continue
                nearest = candidates.loc[candidates.expiry.eq(candidates.expiry.min())]
                if nearest.security_id.nunique() != 1:
                    continue
                contract = nearest.iloc[0]
                selection = indices[
                    output.loc[indices, "strike"].eq(float(key.strike))
                    & output.loc[indices, "option_type"].eq(key.option_type)
                    & (output.loc[indices, "timestamp"].dt.date <= contract.expiry.date())]
                output.loc[selection, "expiry"] = contract.expiry.strftime("%Y-%m-%d")
                output.loc[selection, "security_id"] = contract.security_id
                output.loc[selection, "master_source"] = contract.master_source
                output.loc[selection, "resolution_status"] = "RESOLVED"
                output.loc[selection, "resolution_reason"] = "TIMESTAMPED_MASTER_NEAREST_EXPIRY"
    resolved = output.resolution_status.eq("RESOLVED")
    identity = (
        output.underlying.astype(str) + "|" + output.option_type.astype(str) + "|"
        + output.expiry.astype(str) + "|" + output.strike.astype(str) + "|"
        + output.security_id.astype(str)
    )
    output.loc[resolved, "contract_segment_id"] = identity.loc[resolved]
    return output
