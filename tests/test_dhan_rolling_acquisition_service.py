from datetime import date

import pandas as pd

from src.dhan_rolling_acquisition_service import DhanRollingAcquisitionService


class Client:
    def rolling_options(self, payload):
        key = "ce" if payload["drvOptionType"] == "CALL" else "pe"
        return {"data": {key: {
            "timestamp": [1756698300], "open": [100], "high": [102], "low": [99],
            "close": [101], "volume": [10], "oi": [20], "iv": [12.5],
            "strike": [24500], "spot": [24510],
        }}}


def test_acquisition_uses_same_verified_policy_and_never_saves_credentials(tmp_path):
    result = DhanRollingAcquisitionService(Client(), tmp_path).acquire(
        date(2025, 9, 1), date(2025, 9, 2))
    assert result["parameters"]["expiryFlag"] == "WEEK"
    assert result["parameters"]["expiryCode"] == 0
    assert result["quality"]["ce_rows"] == result["quality"]["pe_rows"] == 1
    assert "token" not in (tmp_path / "manifest.json").read_text().lower()


def test_segmentation_resets_when_strike_changes():
    timestamp = pd.date_range("2026-08-07 10:00", periods=3, freq="min", tz="Asia/Kolkata")
    frame = pd.DataFrame({
        "timestamp": timestamp, "option_type": ["CE"] * 3, "strike": [24500, 24500, 24550],
        "oi": [1] * 3, "iv": [1] * 3, "spot": [1] * 3,
    })
    result = DhanRollingAcquisitionService.segment(frame)
    assert result.segment_id.nunique() == 2
