import pandas as pd
import pytest

from src.premium_expansion_research import (
    build_premium_paths,
    expand_targets,
)


def _frames(periods=70):
    timestamp = pd.date_range(
        "2025-01-02 09:15", periods=periods, freq="min", tz="Asia/Kolkata"
    )
    high = [100.0] * periods
    low = [100.0] * periods
    close = [100.0] * periods
    high[5] = 106.0
    aligned = pd.DataFrame({
        "timestamp": timestamp, "option_open": close, "option_high": high,
        "option_low": low, "option_close": close, "option_volume": [10] * periods,
        "index_session_id": ["20250102"] * periods,
    })
    outcomes = pd.DataFrame({
        "timestamp": [timestamp[0]], "direction": ["LONG"], "trade_allowed": [1],
        "trend_state": ["UPTREND"], "session_segment": ["OPENING"], "year": [2025],
    })
    return aligned, outcomes


def test_target_touch_creates_research_win():
    paths = build_premium_paths(*_frames(), horizons=(15,))
    result = expand_targets(paths, targets=(0.05,))
    assert result.loc[0, "outcome"] == "WIN"
    assert result.loc[0, "mfe"] == pytest.approx(0.06)


def test_prefix_invariance_for_completed_horizon():
    aligned, outcomes = _frames()
    short = build_premium_paths(aligned.iloc[:20], outcomes, horizons=(15,))
    long = build_premium_paths(aligned, outcomes, horizons=(15,))
    pd.testing.assert_frame_equal(short, long)


def test_only_long_allowed_setups_are_included():
    aligned, outcomes = _frames()
    extra = outcomes.copy()
    extra["timestamp"] = aligned.timestamp.iloc[1]
    extra["direction"] = "SHORT"
    combined = pd.concat([outcomes, extra], ignore_index=True)
    paths = build_premium_paths(aligned, combined, horizons=(15,))
    assert paths.timestamp.nunique() == 1
