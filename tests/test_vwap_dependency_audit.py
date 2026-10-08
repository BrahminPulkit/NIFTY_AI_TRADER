import pandas as pd

from tools.run_vwap_dependency_audit import volume_stats


def test_volume_audit_does_not_treat_zero_or_negative_as_valid_volume():
    result = volume_stats(pd.Series([10, 0, -1, None]))
    assert result == {"rows": 4, "non_null": 3, "positive": 1,
                      "zero": 1, "negative": 1, "unique": 3}
