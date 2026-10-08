import pandas as pd
import pytest

from src.live_data_provider import CSVReplayProvider, validate_candles


def _candles():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01 09:15", periods=3, freq="min", tz="Asia/Kolkata"),
        "open": [100, 101, 102], "high": [102, 103, 104],
        "low": [99, 100, 101], "close": [101, 102, 103], "volume": [1, 2, 3],
    })


def test_csv_replay_is_deterministic(tmp_path):
    path = tmp_path / "candles.csv"
    _candles().to_csv(path, index=False)
    first = list(CSVReplayProvider(path).candles())
    second = list(CSVReplayProvider(path).candles())
    assert first == second


def test_provider_rejects_duplicate_timestamp():
    frame = pd.concat([_candles(), _candles().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Invalid candle"):
        validate_candles(frame)

