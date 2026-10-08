import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.index_option_alignment import align_index_option, build_live_alignment


def frames():
    time = pd.date_range("2026-01-01 09:15", periods=6, freq="min", tz="Asia/Kolkata")
    index = pd.DataFrame({"timestamp": time, "close": range(6)})
    option = pd.DataFrame({"timestamp": time.delete(2), "close": range(5)})
    return index, option


def test_exact_inner_join_without_fill():
    index, option = frames(); result = align_index_option(index, option)
    assert len(result) == 5 and index.timestamp.iloc[2] not in set(result.timestamp)
    assert not result.isna().any().any()


def test_alignment_prefix_invariance_and_parity():
    index, option = frames(); full = align_index_option(index, option)
    prefix = align_index_option(index.iloc[:4], option.iloc[:3])
    assert_frame_equal(prefix, full.iloc[:3], check_exact=True)
    assert_series_equal(full.iloc[-1], build_live_alignment(index, option), check_exact=True)
