import pandas as pd

from tools.run_phase12_feature_signal_audit import folds, quality


def test_walk_forward_has_session_embargo():
    dates = pd.date_range("2026-01-01", periods=35, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"session_date": dates})
    for train, test in folds(frame):
        assert max(train) < min(test)
        assert dates[dates.get_loc(min(test)) - 1] not in train


def test_quality_marks_all_missing_feature_as_zero_coverage():
    result = quality(pd.DataFrame({"x": [None, None]}), ["x"]).iloc[0]
    assert result.coverage_pct == 0
    assert result.missing == 2
