"""Thin live adapters around the two frozen canonical feature pipelines."""

from __future__ import annotations

import pandas as pd

from src.feature_pipeline import FEATURES, build_features
from src.option_feature_pipeline import OPTION_FEATURES, build_option_features


def build_live_index_history(history: pd.DataFrame) -> pd.DataFrame:
    """Use the frozen batch implementation; no live-only formula exists."""
    return build_features(history)


def build_live_option_history(history: pd.DataFrame) -> pd.DataFrame:
    return build_option_features(history)


def latest_aligned_features(index_history: pd.DataFrame, option_history: pd.DataFrame) -> pd.Series:
    """Return the latest exact timestamp intersection without any filling."""
    index = build_live_index_history(index_history)[["timestamp", *FEATURES]].rename(
        columns={name: f"index_{name}" for name in FEATURES})
    option = build_live_option_history(option_history)[["timestamp", *OPTION_FEATURES]].rename(
        columns={name: f"option_{name}" for name in OPTION_FEATURES})
    aligned = index.merge(option, on="timestamp", how="inner", validate="one_to_one")
    if aligned.empty:
        raise ValueError("Index and option histories have no common timestamp")
    return aligned.iloc[-1]

