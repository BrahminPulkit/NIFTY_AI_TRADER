import pandas as pd

from commands.data.main_build_phase4_option_dataset import _session_coverage


def test_complete_session_requires_375_unique_minutes_per_side():
    timestamp = pd.date_range("2026-08-07 09:15", periods=374, freq="min", tz="Asia/Kolkata")
    rows = pd.DataFrame({"timestamp": timestamp, "option_type": "CE",
                         "contract_segment_id": "one", "expiry": "2026-08-11", "strike": 24500})
    result = _session_coverage(rows).iloc[0]
    assert result.missing_candles == 1
    assert not result.complete_session


def test_contract_switches_are_counted_as_separate_contracts():
    rows = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-07 10:00", "2026-08-07 10:01"]).tz_localize("Asia/Kolkata"),
        "option_type": ["PE", "PE"], "contract_segment_id": ["a", "b"],
        "expiry": ["2026-08-11", "2026-08-11"], "strike": [24500, 24550],
    })
    result = _session_coverage(rows).iloc[0]
    assert result.unique_contracts == 2
