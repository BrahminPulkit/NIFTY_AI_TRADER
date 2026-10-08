import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.option_feature_pipeline import (
    OPTION_FEATURES, build_live_option_features, build_option_features,
)


def fixture(rows=120):
    timestamp = pd.date_range("2026-07-01 09:15", periods=rows, freq="min",
                              tz="Asia/Kolkata")
    close = pd.Series(np.linspace(100, 112, rows))
    return pd.DataFrame({"timestamp": timestamp, "open": close-.2, "high": close+.5,
                         "low": close-.5, "close": close, "volume": np.arange(rows)+100})


def test_contract_excludes_option_chain_metadata():
    data = fixture(); data["oi"] = 1; data["iv"] = 10; data["strike"] = 25000
    result = build_option_features(data)
    assert set(OPTION_FEATURES).issubset(result.columns)
    assert not {"oi", "iv", "strike", "expiry", "greeks", "spot"}.intersection(result.columns)


def test_prefix_invariance_and_future_mutation():
    data = fixture()
    short = build_option_features(data.iloc[:60])
    long = build_option_features(data)
    assert_frame_equal(short, long.iloc[:60], check_exact=True)
    changed = data.copy(); changed.loc[60:, ["open", "high", "low", "close"]] += 50
    assert_frame_equal(short, build_option_features(changed).iloc[:60], check_exact=True)


def test_historical_live_parity():
    data = fixture(); historical = build_option_features(data)
    assert_series_equal(historical.iloc[-1], build_live_option_features(data), check_exact=True)


def test_causal_structure_and_volume_reference():
    result = build_option_features(fixture())
    assert result.loc[20, "rolling_high"] == fixture().high.iloc[:20].max()
    assert result.loc[20, "rolling_low"] == fixture().low.iloc[:20].min()
    assert result.loc[20, "rolling_volume_20"] == fixture().volume.iloc[:20].mean()


def test_vwap_uses_only_current_and_prior_session_rows():
    data = fixture(10)
    result = build_option_features(data)
    typical = (data.high + data.low + data.close) / 3
    expected = (typical * data.volume).cumsum() / data.volume.cumsum()
    assert_series_equal(result.vwap, expected, check_names=False)


def test_features_reset_at_session_and_contract_boundaries():
    first = fixture(30)
    second = fixture(30)
    second["timestamp"] = pd.date_range(
        "2026-07-02 09:15", periods=30, freq="min", tz="Asia/Kolkata")
    first["strike"] = 25000
    second["strike"] = 25050
    combined = pd.concat([first, second], ignore_index=True)

    result = build_option_features(combined)
    boundary = len(first)
    assert pd.isna(result.loc[boundary, "log_return"])
    assert pd.isna(result.loc[boundary, "previous_high"])
    assert pd.isna(result.loc[boundary + 19, "rolling_high"])
    assert result.loc[boundary + 20, "rolling_high"] == second.high.iloc[:20].max()
