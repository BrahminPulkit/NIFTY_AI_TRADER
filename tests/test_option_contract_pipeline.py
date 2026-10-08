import pandas as pd
import pytest

from src.option_contract_pipeline import normalize_option_contracts, select_option_contract


def rows():
    ts = pd.date_range("2026-08-07 10:00", periods=4, freq="min", tz="Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": ts, "strike": [24500, 24500, 24550, 24550],
        "expiry": ["2026-08-11"] * 4, "option_type": ["CE"] * 4,
        "security_id": ["1", "1", "2", "2"], "underlying": ["NIFTY"] * 4,
        "open": [100] * 4, "high": [102] * 4, "low": [98] * 4,
        "close": [101] * 4, "volume": [10] * 4,
    })


def test_atm_transition_creates_new_contract_segment():
    result = normalize_option_contracts(rows())
    assert result.contract_segment_id.nunique() == 2
    assert result.contract_segment_id.iloc[1] != result.contract_segment_id.iloc[2]


def test_contract_identity_is_mandatory():
    with pytest.raises(ValueError, match="strike"):
        normalize_option_contracts(rows().drop(columns="strike"))


def test_shared_selector_uses_nearest_expiry_then_atm():
    moment = rows().timestamp.iloc[0]
    chain = pd.DataFrame({
        "timestamp": [moment] * 3, "strike": [24450, 24500, 24500],
        "expiry": ["2026-08-11", "2026-08-11", "2026-08-18"],
        "option_type": ["CE"] * 3, "security_id": ["10", "11", "12"],
        "underlying": ["NIFTY"] * 3, "open": [100] * 3, "high": [102] * 3,
        "low": [98] * 3, "close": [101] * 3, "volume": [10] * 3,
    })
    selected = select_option_contract(
        chain, timestamp=moment, option_type="CALL", underlying_price=24490)
    assert selected.security_id == "11"


def test_shared_selector_never_uses_expired_contract():
    chain = rows().iloc[:1].copy()
    chain["expiry"] = "2026-08-06"
    with pytest.raises(LookupError):
        select_option_contract(
            chain, timestamp=chain.timestamp.iloc[0], option_type="CE",
            underlying_price=24500)
