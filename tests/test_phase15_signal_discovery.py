import pandas as pd

from tools.run_phase15_signal_discovery import FAMILIES, FEATURES, add_interactions, causal_percentile


def test_all_family_columns_belong_to_frozen_features():
    assert all(column in FEATURES for columns in FAMILIES.values() for column in columns)


def test_causal_percentile_is_prefix_invariant_and_session_reset():
    values = pd.Series([3.0, 1.0, 2.0, 9.0, 8.0])
    sessions = pd.Series(["a", "a", "a", "b", "b"])
    full = causal_percentile(values, sessions)
    prefix = causal_percentile(values.iloc[:3], sessions.iloc[:3])
    pd.testing.assert_series_equal(full.iloc[:3], prefix)
    assert full.iloc[3] == 1.0


def test_interaction_count_is_bounded():
    frame = pd.DataFrame({name: [1.0, 2.0] for name in FEATURES})
    assert len(add_interactions(frame)) == 12
