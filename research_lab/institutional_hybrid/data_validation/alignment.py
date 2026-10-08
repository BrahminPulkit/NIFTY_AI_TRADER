"""Leakage-safe timestamp alignment."""
from __future__ import annotations

import pandas as pd


def align_option_to_index(
    index: pd.DataFrame, option: pd.DataFrame, tolerance_seconds: int = 0
) -> pd.DataFrame:
    """Exact match, or latest prior option observation within tolerance.

    `direction='backward'` is invariant: a future option candle is never used.
    """
    if tolerance_seconds < 0:
        raise ValueError("Backward tolerance cannot be negative")
    left = index.sort_index().reset_index(names="index_timestamp")
    right = option.sort_index().reset_index(names="option_timestamp")
    if tolerance_seconds == 0:
        joined = left.merge(
            right, left_on="index_timestamp", right_on="option_timestamp",
            how="left", suffixes=("_index", "_option"),
        )
    else:
        joined = pd.merge_asof(
            left.sort_values("index_timestamp"), right.sort_values("option_timestamp"),
            left_on="index_timestamp", right_on="option_timestamp",
            direction="backward", tolerance=pd.Timedelta(seconds=tolerance_seconds),
            suffixes=("_index", "_option"),
        )
    matched = joined["option_timestamp"].notna()
    if (joined.loc[matched, "option_timestamp"] > joined.loc[matched, "index_timestamp"]).any():
        raise AssertionError("Future option candle detected")
    joined["alignment_lag_seconds"] = (
        joined["index_timestamp"] - joined["option_timestamp"]
    ).dt.total_seconds()
    joined["exact_timestamp_match"] = joined["alignment_lag_seconds"].eq(0)
    return joined.set_index("index_timestamp")

