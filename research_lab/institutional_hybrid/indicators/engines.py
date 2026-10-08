"""Vectorised, causal features for NIFTY futures bars.

Every rolling reference is shifted where necessary so a row only uses
information available at that bar's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _required(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    return pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    _required(df, {"open", "high", "low", "close"})
    out = df.copy()
    for span in (20, 50, 200):
        out[f"ema{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    out["atr"] = atr(out)
    out["atr_ratio"] = out["atr"] / out["atr"].rolling(50).median().shift()
    out["ema_spread_atr"] = (out["ema20"] - out["ema50"]).abs() / out["atr"]
    out["gap_atr"] = (out["open"] - out["close"].shift()).abs() / out["atr"].shift()
    slope = out["ema20"].diff(5)
    bull = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"]) & (slope > 0)
    bear = (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"]) & (slope < 0)
    conditions = [
        out["gap_atr"] >= 0.75,
        out["atr_ratio"] >= 1.35,
        (out["atr_ratio"] <= 0.72) & (out["ema_spread_atr"] < 0.5),
        bull,
        bear,
    ]
    out["regime"] = np.select(
        conditions, ["gap_day", "expansion", "compression", "bull_trend", "bear_trend"],
        default="sideways",
    )
    return out


def ema_pullback_features(df: pd.DataFrame) -> pd.DataFrame:
    out = regime_features(df) if "ema20" not in df else df.copy()
    out["trend_direction"] = np.select(
        [(out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"]),
         (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"])],
        [1, -1], default=0,
    )
    tolerance = out["atr"] * 0.20
    touches20 = (out["low"] <= out["ema20"] + tolerance) & (out["high"] >= out["ema20"] - tolerance)
    holds_bull = out["close"] > out["ema20"]
    holds_bear = out["close"] < out["ema20"]
    out["ema_pullback"] = touches20 & (((out["trend_direction"] == 1) & holds_bull) |
                                        ((out["trend_direction"] == -1) & holds_bear))
    out["continuation"] = ((out["trend_direction"] == 1) & (out["close"] > out["high"].shift())) | (
        (out["trend_direction"] == -1) & (out["close"] < out["low"].shift())
    )
    return out


def liquidity_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    _required(df, {"open", "high", "low", "close"})
    out = df.copy()
    out["prior_high"] = out["high"].rolling(lookback).max().shift()
    out["prior_low"] = out["low"].rolling(lookback).min().shift()
    out["high_sweep"] = (out["high"] > out["prior_high"]) & (out["close"] < out["prior_high"])
    out["low_sweep"] = (out["low"] < out["prior_low"]) & (out["close"] > out["prior_low"])
    out["false_break"] = out["high_sweep"] | out["low_sweep"]
    out["liquidity_direction"] = out["low_sweep"].astype(int) - out["high_sweep"].astype(int)
    # Restricted ICT vocabulary: a sweep followed by displacement through the
    # prior bar is the only Market Structure Shift used by this lab.
    out["bullish_mss"] = out["low_sweep"].shift().fillna(False).astype(bool) & (out["close"] > out["high"].shift())
    out["bearish_mss"] = out["high_sweep"].shift().fillna(False).astype(bool) & (out["close"] < out["low"].shift())
    out["market_structure_shift"] = out["bullish_mss"].astype(int) - out["bearish_mss"].astype(int)
    return out


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    _required(df, {"open", "high", "low", "close"})
    out = df.copy()
    out["atr"] = out.get("atr", atr(out))
    out["bar_range"] = out["high"] - out["low"]
    out["range_expansion"] = out["bar_range"] / out["bar_range"].rolling(20).median().shift()
    out["body_atr"] = (out["close"] - out["open"]).abs() / out["atr"]
    out["trend_bar"] = (out["body_atr"] >= 0.6) & (
        ((out["close"] > out["open"]) & ((out["high"] - out["close"]) <= out["bar_range"] * .2))
        | ((out["close"] < out["open"]) & ((out["close"] - out["low"]) <= out["bar_range"] * .2))
    )
    out["pullback_bar"] = ((out["close"] - out["open"]) * out["close"].diff().rolling(3).mean() < 0)
    out["momentum_3_atr"] = out["close"].diff(3).abs() / out["atr"]
    out["atr_expansion"] = out["atr"] / out["atr"].rolling(50).median().shift()
    out["volatility_expansion"] = out["close"].pct_change().rolling(10).std() / (
        out["close"].pct_change().rolling(50).std().shift()
    )
    return out


def breakout_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = liquidity_features(df, lookback) if "prior_high" not in df else df.copy()
    out["range_breakout_up"] = out["close"] > out["prior_high"]
    out["range_breakout_down"] = out["close"] < out["prior_low"]
    out["swing_high"] = (out["high"].shift(2) < out["high"].shift()) & (out["high"] < out["high"].shift())
    out["swing_low"] = (out["low"].shift(2) > out["low"].shift()) & (out["low"] > out["low"].shift())
    compressed = (out.get("atr_ratio", pd.Series(index=out.index, dtype=float)).shift() < 0.8)
    out["compression_breakout"] = compressed & (out["range_breakout_up"] | out["range_breakout_down"])
    if "volume" in out:
        rv = out["volume"] / out["volume"].rolling(20).median().shift()
        out["volume_breakout"] = (rv >= 1.5) & (out["range_breakout_up"] | out["range_breakout_down"])
    else:
        out["volume_breakout"] = False
    out["breakout_direction"] = out["range_breakout_up"].astype(int) - out["range_breakout_down"].astype(int)
    return out


def futures_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    _required(df, {"volume"})
    out = df.copy()
    baseline = out["volume"].rolling(20).median().shift()
    out["relative_volume"] = out["volume"] / baseline
    out["volume_acceleration"] = out["relative_volume"].diff(3)
    out["volume_spike"] = out["relative_volume"] >= 1.5
    out["participation"] = (out["relative_volume"].clip(0, 3) / 3).fillna(0)
    return out
