import pandas as pd

from tools.run_phase18_high_resolution_audit import validate_research_frame


def test_minute_quotes_are_not_high_resolution():
    timestamps = pd.date_range("2026-01-01 09:15", periods=3, freq="min", tz="Asia/Kolkata")
    frame = pd.DataFrame({
        "timestamp": timestamps, "spot": 25000, "expiry": "2026-01-01", "strike": 25000,
        "option_type": "CE", "contract_id": "x", "ltp": 100, "bid": 99.9,
        "ask": 100.1, "volume": 10, "oi": 20,
    })
    result = validate_research_frame(frame)
    assert result["modal_interval_seconds"] == 60
    assert not result["high_resolution"]
    assert not result["quote_executable"]


def test_five_second_complete_quotes_pass_schema_gate():
    timestamps = pd.date_range("2026-01-01 09:15", periods=3, freq="5s", tz="Asia/Kolkata")
    frame = pd.DataFrame({
        "timestamp": timestamps, "spot": 25000, "expiry": "2026-01-01", "strike": 25000,
        "option_type": "CE", "contract_id": "x", "ltp": 100, "bid": 99.9,
        "ask": 100.1, "volume": 10, "oi": 20,
    })
    result = validate_research_frame(frame)
    assert result["high_resolution"]
    assert result["quote_executable"]
