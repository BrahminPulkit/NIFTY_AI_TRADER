"""Deterministic timestamp-only alignment of frozen Index and Option datasets."""

from __future__ import annotations

import pandas as pd


ALIGNMENT_VERSION = "index_option_timestamp_inner_v1"


def _validate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise ValueError(f"{name} dataset has no timestamp")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    if output.timestamp.duplicated().any() or not output.timestamp.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be sorted and unique")
    return output


def align_index_option(index: pd.DataFrame, option: pd.DataFrame) -> pd.DataFrame:
    """Inner join on exact timestamps; never fill, interpolate or synthesize."""
    left = _validate(index, "Index")
    right = _validate(option, "Option")
    left = left.rename(columns={name: f"index_{name}" for name in left.columns if name != "timestamp"})
    right = right.rename(columns={name: f"option_{name}" for name in right.columns if name != "timestamp"})
    output = left.merge(right, on="timestamp", how="inner", validate="one_to_one", sort=True)
    if output.timestamp.duplicated().any() or not output.timestamp.is_monotonic_increasing:
        raise AssertionError("Aligned timestamps are not unique and causal")
    return output.reset_index(drop=True)


def build_live_alignment(index_history: pd.DataFrame, option_history: pd.DataFrame) -> pd.Series:
    """Latest exact common timestamp from the identical alignment implementation."""
    aligned = align_index_option(index_history, option_history)
    if aligned.empty:
        raise ValueError("No exact common timestamp exists")
    return aligned.iloc[-1]
