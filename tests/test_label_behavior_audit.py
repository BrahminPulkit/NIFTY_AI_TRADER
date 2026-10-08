import pandas as pd
import pytest

from tools.run_label_behavior_audit import path_metrics


def test_path_metrics_start_after_entry_and_preserve_ohlc_ambiguity():
    timestamps = pd.date_range("2026-01-01 10:00", periods=3, freq="min", tz="Asia/Kolkata")
    contract = pd.DataFrame({
        "timestamp": timestamps, "open": [100, 100, 102], "high": [200, 106, 104],
        "low": [1, 96, 99], "close": [100, 102, 103],
    })
    entry = pd.Series({"timestamp": timestamps[0], "entry_premium": 100,
                       "target": 105, "stop": 97})
    result = path_metrics(entry, contract)
    assert result["first_candle_target_touched"]
    assert result["first_candle_stop_touched"]
    assert result["first_candle_both_touched"]
    assert result["mfe_1m_pct"] == pytest.approx(6)
    assert result["mae_1m_pct"] == pytest.approx(-4)
