import pandas as pd
import pytest

from src.outcome_sparsity_research import (
    analyze_outcome_paths,
    verify_frozen_replay,
)


def _frames():
    timestamps = pd.date_range("2025-01-02 09:15", periods=8, freq="min", tz="Asia/Kolkata")
    setups = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 8,
        "high": [100.0, 100.11, 100.16, 100.31, 100.0, 100.0, 100.0, 100.0],
        "low": [100.0, 99.99, 99.99, 99.99, 100.0, 100.0, 100.0, 100.0],
        "close": [100.0] * 8,
        "session_id": ["20250102"] * 8,
    })
    outcomes = pd.DataFrame({
        "timestamp": [timestamps[0]],
        "direction": ["LONG"],
        "entry_signal": [1],
        "trade_allowed": [1],
        "entry_price": [100.0],
        "final_outcome": ["WIN"],
        "exit_reason": ["TARGET"],
        "label_horizon": [5],
    })
    return setups, outcomes


def test_excursion_and_times_are_causal_and_correct():
    paths, _ = analyze_outcome_paths(*_frames())
    assert paths.loc[0, "mfe_pct"] == pytest.approx(0.0031)
    assert paths.loc[0, "time_to_mfe"] == 3
    assert bool(paths.loc[0, "reached_0.30pct"])


def test_frozen_target_replay_matches_labels():
    setups, outcomes = _frames()
    _, simulations = analyze_outcome_paths(setups, outcomes)
    verify_frozen_replay(outcomes, simulations)


def test_prefix_invariance():
    setups, outcomes = _frames()
    first, _ = analyze_outcome_paths(setups.iloc[:6], outcomes)
    longer, _ = analyze_outcome_paths(setups, outcomes)
    pd.testing.assert_frame_equal(first, longer)
