"""Canonical, causal feature pipeline for historical and live use.

All feature consumers must call :func:`build_features`.  The live helper is a
thin adapter and contains no independent feature calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "canonical_v1"
STRUCTURE_WINDOW = 20
VOLATILITY_WINDOW = 20

FEATURES = [
    "candle_body", "upper_wick", "lower_wick", "candle_range",
    "log_return", "percent_return",
    "ema_5", "ema_9", "ema_20", "ema_50", "vwap",
    "rsi_14", "atr_14", "rolling_volatility", "rolling_range",
    "previous_high", "previous_low", "rolling_high", "rolling_low",
    "breakout_flag", "breakdown_flag",
    "minutes_from_open", "minutes_to_close", "weekday", "month", "session_id",
]


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical input columns: {missing}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    if result.timestamp.duplicated().any():
        raise ValueError("Duplicate timestamps are forbidden")
    if not result.timestamp.is_monotonic_increasing:
        raise ValueError("Input timestamps must be sorted")
    minute = result.timestamp.dt.hour * 60 + result.timestamp.dt.minute
    if not minute.between(555, 929).all():
        raise ValueError("Input contains timestamps outside 09:15-15:29")
    if result.timestamp.dt.second.ne(0).any():
        raise ValueError("Input timestamps must be minute aligned")
    numeric = ["open", "high", "low", "close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC values cannot be null")
    invalid = ((result.high < result.low) | (result.open < result.low) |
               (result.open > result.high) | (result.close < result.low) |
               (result.close > result.high))
    if invalid.any():
        raise ValueError("Invalid OHLC relationships")
    return result


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _session_vwap(frame: pd.DataFrame, session_key: pd.Series) -> pd.Series:
    # Only approved analytical volume is used. volume_raw is intentionally not
    # accepted as a substitute.
    volume = frame.volume.where(frame.volume.ge(0))
    typical = (frame.high + frame.low + frame.close) / 3.0
    numerator = (typical * volume).groupby(session_key, sort=False).cumsum()
    denominator = volume.groupby(session_key, sort=False).cumsum()
    return numerator / denominator.replace(0, np.nan)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return input columns plus the canonical causal feature contract."""
    output = _validate(frame)
    timestamp = output.timestamp
    session_key = timestamp.dt.strftime("%Y-%m-%d")

    output["candle_body"] = output.close - output.open
    output["upper_wick"] = output.high - output[["open", "close"]].max(axis=1)
    output["lower_wick"] = output[["open", "close"]].min(axis=1) - output.low
    output["candle_range"] = output.high - output.low
    output["log_return"] = np.log(output.close / output.close.shift(1))
    output["percent_return"] = output.close.pct_change(fill_method=None) * 100.0

    for period in (5, 9, 20, 50):
        output[f"ema_{period}"] = output.close.ewm(span=period, adjust=False, min_periods=period).mean()
    output["vwap"] = _session_vwap(output, session_key)
    output["rsi_14"] = _rsi(output.close, 14)
    output["atr_14"] = _atr(output.high, output.low, output.close, 14)
    output["rolling_volatility"] = output.log_return.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std(ddof=0)
    output["rolling_range"] = (
        output.high.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).max()
        - output.low.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).min()
    )

    output["previous_high"] = output.high.shift(1)
    output["previous_low"] = output.low.shift(1)
    output["rolling_high"] = output.high.shift(1).rolling(
        STRUCTURE_WINDOW, min_periods=STRUCTURE_WINDOW).max()
    output["rolling_low"] = output.low.shift(1).rolling(
        STRUCTURE_WINDOW, min_periods=STRUCTURE_WINDOW).min()
    output["breakout_flag"] = (output.high > output.rolling_high).astype("int8")
    output["breakdown_flag"] = (output.low < output.rolling_low).astype("int8")

    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    output["minutes_from_open"] = (minute - 555).astype("int16")
    output["minutes_to_close"] = (929 - minute).astype("int16")
    output["weekday"] = timestamp.dt.weekday.astype("int8")
    output["month"] = timestamp.dt.month.astype("int8")
    output["session_id"] = timestamp.dt.strftime("%Y%m%d").astype("int32")
    return output


def build_live_features(history: pd.DataFrame) -> pd.Series:
    """Build the latest live row by invoking the canonical batch pipeline."""
    if history.empty:
        raise ValueError("Live history cannot be empty")
    return build_features(history).iloc[-1]

