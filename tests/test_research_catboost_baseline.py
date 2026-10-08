import numpy as np

from tools.run_research_catboost_baseline import metrics, probability_distribution


def test_metrics_use_fixed_half_threshold_and_expected_confusion_matrix():
    result = metrics([0, 0, 1, 1], [.1, .6, .4, .9])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["accuracy"] == .5


def test_probability_distribution_accounts_for_all_rows():
    import pandas as pd
    result = probability_distribution(pd.Series(np.linspace(0, .99, 20)))
    assert sum(row["count"] for row in result) == 20
