import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_pipeline import FEATURES, build_features, build_live_features


def sample(rows=100):
    ts = pd.date_range("2026-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata")
    close = pd.Series(np.linspace(100, 120, rows))
    return pd.DataFrame({"timestamp": ts, "open": close - .2, "high": close + .5,
                         "low": close - .5, "close": close, "volume": np.arange(rows) + 1.0})


def test_ema_matches_causal_pandas_definition():
    source = sample()
    result = build_features(source)
    expected = source.close.ewm(span=5, adjust=False, min_periods=5).mean()
    pd.testing.assert_series_equal(result.ema_5, expected, check_names=False)


def test_rsi_for_monotonic_rise_reaches_100():
    result = build_features(sample())
    assert result.rsi_14.dropna().eq(100.0).all()


def test_vwap_is_session_cumulative_and_resets():
    first = sample(3)
    second = sample(2)
    second.timestamp = pd.date_range("2026-01-05 09:15", periods=2, freq="min", tz="Asia/Kolkata")
    source = pd.concat([first, second], ignore_index=True)
    result = build_features(source)
    typical = (source.high + source.low + source.close) / 3
    expected_first = (typical.iloc[:3] * source.volume.iloc[:3]).cumsum() / source.volume.iloc[:3].cumsum()
    np.testing.assert_allclose(result.vwap.iloc[:3], expected_first)
    assert result.vwap.iloc[3] == typical.iloc[3]


def test_atr_matches_true_range_wilder_ewm():
    source = sample()
    result = build_features(source)
    previous = source.close.shift()
    tr = pd.concat([source.high-source.low, (source.high-previous).abs(),
                    (source.low-previous).abs()], axis=1).max(axis=1)
    expected = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pd.testing.assert_series_equal(result.atr_14, expected, check_names=False)


def test_rolling_high_low_use_only_prior_rows():
    source = sample()
    result = build_features(source)
    high = source.high.shift(1).rolling(20, min_periods=20).max()
    low = source.low.shift(1).rolling(20, min_periods=20).min()
    pd.testing.assert_series_equal(result.rolling_high, high, check_names=False)
    pd.testing.assert_series_equal(result.rolling_low, low, check_names=False)


def test_prefix_invariance_and_no_future_leakage():
    source = sample(100)
    short = build_features(source.iloc[:40].copy())
    long_prefix = build_features(source).iloc[:40]
    assert_frame_equal(short, long_prefix, check_exact=True)
    mutated = source.copy()
    mutated.loc[40:, ["open", "high", "low", "close"]] *= 1.1
    assert_frame_equal(short, build_features(mutated).iloc[:40], check_exact=True)


def test_historical_and_live_use_identical_pipeline():
    source = sample()
    historical = build_features(source)
    assert historical.iloc[-1].equals(build_live_features(source))
    assert set(FEATURES).issubset(historical.columns)
