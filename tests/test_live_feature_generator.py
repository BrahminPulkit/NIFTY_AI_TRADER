import pandas as pd
from pandas.testing import assert_series_equal

from src.live_feature_generator import frozen_decision_at


def test_future_decision_observation_cannot_change_current_decision():
    history = pd.read_parquet("data/runtime/decision_observations.parquet")
    current = history.iloc[-1].to_dict()
    current["timestamp"] = pd.Timestamp("2030-01-02 10:00", tz="Asia/Kolkata")
    current["outcome_available_timestamp"] = pd.Timestamp("2130-01-01", tz="Asia/Kolkata")
    baseline = frozen_decision_at(history, current)
    malicious = history.iloc[[-1]].copy()
    malicious["timestamp"] = pd.Timestamp("2030-01-02 09:59", tz="Asia/Kolkata")
    malicious["outcome_available_timestamp"] = pd.Timestamp("2030-01-02 10:01", tz="Asia/Kolkata")
    with_future = frozen_decision_at(pd.concat([history, malicious]), current)
    columns = ["recommendation", "decision_reason", "history_count",
               "strategy_confidence_score"]
    assert_series_equal(baseline[columns], with_future[columns])
