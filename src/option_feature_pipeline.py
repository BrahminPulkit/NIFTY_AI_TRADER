"""Canonical causal feature pipeline for rolling NIFTY option OHLCV data.

Only timestamp, OHLC and volume are accepted as source inputs.  Option-chain
metadata (OI, IV, strike, expiry, Greeks and spot) is deliberately excluded.
The same function is used for historical generation and live history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


OPTION_FEATURE_VERSION = "option_contract_boundary_v2"
STRUCTURE_WINDOW = 20
ROLLING_WINDOW = 20

OPTION_FEATURES = [
    "candle_body", "upper_wick", "lower_wick", "candle_range",
    "log_return", "percent_return",
    "ema_5", "ema_9", "ema_20", "ema_50", "vwap",
    "rsi_14", "atr_14", "rolling_volatility", "rolling_range",
    "rolling_volume_20", "volume_ratio_20",
    "previous_high", "previous_low", "rolling_high", "rolling_low",
    "breakout_flag", "breakdown_flag",
    "minutes_from_open", "minutes_to_close", "weekday", "month", "session_id",
]

SOURCE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical option inputs: {missing}")
    grouping_columns = [
        name for name in ("strike", "option_type", "expiry_code") if name in frame.columns
    ]
    result = frame[SOURCE_COLUMNS + grouping_columns].copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")
    if result.timestamp.duplicated().any() or not result.timestamp.is_monotonic_increasing:
        raise ValueError("Option timestamps must be sorted and unique")
    if result.timestamp.dt.second.ne(0).any() or result.timestamp.dt.microsecond.ne(0).any():
        raise ValueError("Option timestamps must be minute-aligned")
    numeric = ["open", "high", "low", "close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("Canonical option OHLCV cannot contain null/non-numeric values")
    invalid = (
        result[["open", "high", "low", "close"]].le(0).any(axis=1)
        | result.volume.lt(0)
        | result.high.lt(result.low)
        | result.open.lt(result.low) | result.open.gt(result.high)
        | result.close.lt(result.low) | result.close.gt(result.high)
    )
    if invalid.any():
        raise ValueError("Canonical option source contains invalid OHLCV")
    return result


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss
    result = 100 - 100 / (1 + rs)
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return result.mask((average_loss == 0) & (average_gain == 0), 50.0)


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame.close.shift(1)
    true_range = pd.concat([
        frame.high - frame.low,
        (frame.high - previous_close).abs(),
        (frame.low - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_option_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the complete causal option feature contract."""
    output = _validate(frame)
    timestamp = output.timestamp
    session = timestamp.dt.strftime("%Y-%m-%d")
    group_columns = [session.rename("session")]
    for name in ("strike", "option_type", "expiry_code"):
        if name in output.columns:
            group_columns.append(output[name].rename(name))
    group_key = pd.MultiIndex.from_arrays(group_columns)

    def grouped(series: pd.Series):
        return series.groupby(group_key, sort=False, dropna=False)

    output["candle_body"] = output.close - output.open
    output["upper_wick"] = output.high - output[["open", "close"]].max(axis=1)
    output["lower_wick"] = output[["open", "close"]].min(axis=1) - output.low
    output["candle_range"] = output.high - output.low
    previous_close = grouped(output.close).shift(1)
    output["log_return"] = np.log(output.close / previous_close)
    output["percent_return"] = (output.close / previous_close - 1.0) * 100.0
    for period in (5, 9, 20, 50):
        output[f"ema_{period}"] = grouped(output.close).transform(
            lambda values: values.ewm(span=period, adjust=False, min_periods=period).mean())

    typical = (output.high + output.low + output.close) / 3.0
    cumulative_value = (typical * output.volume).groupby(session, sort=False).cumsum()
    cumulative_volume = output.volume.groupby(session, sort=False).cumsum()
    output["vwap"] = cumulative_value / cumulative_volume.replace(0, np.nan)
    output["rsi_14"] = grouped(output.close).transform(_rsi)
    true_range = pd.concat([
        output.high - output.low,
        (output.high - previous_close).abs(),
        (output.low - previous_close).abs(),
    ], axis=1).max(axis=1)
    output["atr_14"] = grouped(true_range).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean())
    output["rolling_volatility"] = grouped(output.log_return).transform(
        lambda values: values.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std(ddof=0))
    rolling_high_current = grouped(output.high).transform(
        lambda values: values.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).max())
    rolling_low_current = grouped(output.low).transform(
        lambda values: values.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).min())
    output["rolling_range"] = rolling_high_current - rolling_low_current
    prior_volume = grouped(output.volume).shift(1)
    output["rolling_volume_20"] = grouped(prior_volume).transform(
        lambda values: values.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean())
    output["volume_ratio_20"] = output.volume / output.rolling_volume_20.replace(0, np.nan)
    output["previous_high"] = grouped(output.high).shift(1)
    output["previous_low"] = grouped(output.low).shift(1)
    output["rolling_high"] = grouped(output.previous_high).transform(
        lambda values: values.rolling(STRUCTURE_WINDOW, min_periods=STRUCTURE_WINDOW).max())
    output["rolling_low"] = grouped(output.previous_low).transform(
        lambda values: values.rolling(STRUCTURE_WINDOW, min_periods=STRUCTURE_WINDOW).min())
    output["breakout_flag"] = (output.high > output.rolling_high).astype("int8")
    output["breakdown_flag"] = (output.low < output.rolling_low).astype("int8")
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    output["minutes_from_open"] = minute - (9 * 60 + 15)
    output["minutes_to_close"] = (15 * 60 + 29) - minute
    output["weekday"] = timestamp.dt.weekday.astype("int8")
    output["month"] = timestamp.dt.month.astype("int8")
    output["session_id"] = timestamp.dt.strftime("%Y%m%d").astype("int32")
    return output[SOURCE_COLUMNS + OPTION_FEATURES]


def build_live_option_features(history: pd.DataFrame) -> pd.Series:
    """Return the latest row from the identical canonical history pipeline."""
    if history.empty:
        raise ValueError("Option feature history cannot be empty")
    return build_option_features(history).iloc[-1]
