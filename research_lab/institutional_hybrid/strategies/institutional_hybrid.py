"""Institutional score composition and setup decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.engines import (
    breakout_features, ema_pullback_features, futures_volume_features,
    liquidity_features, momentum_features, regime_features,
)
from .option_engines import premium_expansion_features


DEFAULT_WEIGHTS = {
    "regime": 20, "trend": 15, "liquidity": 15, "momentum": 15,
    "breakout": 10, "volume": 10, "option_liquidity": 10, "premium_expansion": 5,
}


class InstitutionalHybridStrategy:
    def __init__(self, threshold: float = 65.0, weights: dict[str, float] | None = None):
        self.threshold = threshold
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        if abs(sum(self.weights.values()) - 100) > 1e-9:
            raise ValueError("Institutional weights must sum to 100")

    def build_features(self, futures: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
        x = regime_features(futures)
        x = ema_pullback_features(x)
        x = liquidity_features(x)
        x = momentum_features(x)
        x = breakout_features(x)
        x = futures_volume_features(x)
        option = options.rename(columns={"volume": "option_volume"}).reindex(x.index)
        joined = x.join(option, rsuffix="_option")
        return premium_expansion_features(joined)

    def score(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        direction = np.sign(
            out["trend_direction"] + out["breakout_direction"]
            + out["liquidity_direction"] + out["market_structure_shift"]
        )
        direction = pd.Series(direction, index=out.index).replace(0, np.nan).ffill(limit=2).fillna(0)
        components = pd.DataFrame(index=out.index)
        components["regime"] = out["regime"].isin(["bull_trend", "bear_trend", "expansion", "gap_day"]).astype(float)
        components["trend"] = ((out["trend_direction"] != 0) & (out["ema_pullback"] | out["continuation"])).astype(float)
        components["liquidity"] = (out["false_break"] | (out["market_structure_shift"] != 0)).astype(float)
        components["momentum"] = (
            (out["atr_expansion"] >= 1.1) & (out["range_expansion"] >= 1.2) & out["trend_bar"]
        ).astype(float)
        components["breakout"] = (out["breakout_direction"] != 0).astype(float)
        components["volume"] = ((out["relative_volume"] >= 1.2) | (out["volume_acceleration"] > 0.25)).astype(float)
        components["option_liquidity"] = out["premium_quality"].fillna(0)
        components["premium_expansion"] = out["premium_expansion_score"].fillna(0) / 100
        for name in components:
            out[f"score_{name}"] = components[name] * self.weights[name]
        out["institutional_score"] = sum(out[f"score_{k}"] for k in self.weights)
        out["signal"] = np.where(out["institutional_score"] >= self.threshold, direction, 0).astype(int)
        out["rejection_reason"] = np.where(
            out["signal"] == 0,
            np.where(out["premium_quality"] < 0.5, "option_liquidity", "score_below_threshold"),
            "",
        )
        return out
