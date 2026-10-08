"""Option-chain quality and short-horizon premium expansion research."""
from __future__ import annotations

import numpy as np
import pandas as pd


OPTION_COLUMNS = {"bid", "ask", "option_volume", "open_interest", "delta", "gamma", "theta", "iv", "premium"}


def option_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = OPTION_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing option columns: {sorted(missing)}")
    out = df.copy()
    mid = (out["bid"] + out["ask"]) / 2
    out["spread_pct"] = (out["ask"] - out["bid"]) / mid.replace(0, np.nan)
    out["oi_rank"] = out["open_interest"].rolling(50, min_periods=10).rank(pct=True)
    out["option_volume_rank"] = out["option_volume"].rolling(50, min_periods=10).rank(pct=True)
    out["greek_quality"] = (
        out["delta"].abs().between(0.35, 0.65).astype(float) * 0.4
        + out["gamma"].clip(lower=0).rank(pct=True) * 0.3
        + (1 - out["theta"].abs().rank(pct=True)) * 0.15
        + (1 - out["iv"].rank(pct=True)) * 0.15
    )
    out["premium_quality"] = (
        (1 - out["spread_pct"].clip(0, 0.05) / 0.05) * 0.35
        + out["oi_rank"].fillna(0) * 0.20
        + out["option_volume_rank"].fillna(0) * 0.25
        + out["greek_quality"].fillna(0) * 0.20
    ).clip(0, 1)
    return out


def premium_expansion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate expansion capacity; not a directional price forecast."""
    out = option_liquidity_features(df) if "premium_quality" not in df else df.copy()
    required = {"atr_expansion", "range_expansion", "relative_volume"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing joined futures features: {sorted(missing)}")
    raw = (
        out["premium_quality"] * 0.35
        + out["atr_expansion"].clip(0.5, 2.0).sub(0.5).div(1.5) * 0.25
        + out["range_expansion"].clip(0.5, 2.5).sub(0.5).div(2.0) * 0.20
        + out["relative_volume"].clip(0.5, 3.0).sub(0.5).div(2.5) * 0.20
    )
    out["premium_expansion_score"] = (raw * 100).clip(0, 100)
    out["expansion_5_10_likely"] = out["premium_expansion_score"] >= 65
    return out
