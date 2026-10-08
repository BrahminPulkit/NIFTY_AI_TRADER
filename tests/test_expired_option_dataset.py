from pathlib import Path

from src.expired_option_dataset import _archive_contracts, acquire_expiry_session
from src.option_contract_pipeline import normalize_option_contracts
import pandas as pd


def test_expiry_archive_exposes_explicit_ce_and_pe_identity():
    path = Path("data/external/shoonyatrader/proof5_raw/20260407.zip")
    if not path.exists():
        return
    contracts = _archive_contracts(path)
    assert ("CE", 23450) in contracts
    assert ("PE", 23450) in contracts


def test_explicit_contiguous_contract_segments_are_preserved():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:15", periods=3, freq="min", tz="Asia/Kolkata"),
        "underlying": ["NIFTY"] * 3, "expiry": ["2026-01-01"] * 3,
        "strike": [24000, 24050, 24000], "option_type": ["CE"] * 3,
        "security_id": [pd.NA] * 3, "open": [10, 10, 10], "high": [11, 11, 11],
        "low": [9, 9, 9], "close": [10, 10, 10], "volume": [1, 1, 1],
        "contract_segment_id": ["first-24000", "middle-24050", "second-24000"],
    })
    result = normalize_option_contracts(frame)
    assert result.contract_segment_id.nunique() == 3


def test_session_date_cannot_be_after_expiry():
    path = Path("data/external/shoonyatrader/proof5_raw/20260407.zip")
    if not path.exists():
        return
    import pytest
    with pytest.raises(ValueError, match="after contract expiry"):
        acquire_expiry_session(path, pd.DataFrame(), session_date="2026-04-08")
