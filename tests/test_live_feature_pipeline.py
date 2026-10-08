import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_pipeline import build_features
from src.live_feature_pipeline import build_live_index_history


def _history(rows=80):
    close = pd.Series(range(rows), dtype=float) / 10 + 100
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01 09:15", periods=rows, freq="min", tz="Asia/Kolkata"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close + .2, "volume": 100,
    })


def test_live_pipeline_is_exact_canonical_pipeline():
    frame = _history()
    assert_frame_equal(build_live_index_history(frame), build_features(frame), check_exact=True)


def test_live_feature_prefix_invariance():
    frame = _history()
    assert_frame_equal(
        build_live_index_history(frame.iloc[:60]),
        build_live_index_history(frame).iloc[:60],
        check_exact=True,
    )

