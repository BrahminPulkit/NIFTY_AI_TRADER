import pandas as pd
import pytest

from src.mtf_ema_pullback_scalper import ScalperConfig, option_chain_confirmation


def test_rr_cannot_be_below_two():
    with pytest.raises(ValueError):
        ScalperConfig(rr=1.99)


def test_option_chain_confirmation_is_strict():
    assert option_chain_confirmation(None, 1)[0] is False
    chain = pd.DataFrame([{"ce_oi": 100, "pe_oi": 120, "ce_oi_change": 5, "pe_oi_change": 20}])
    approved, detail = option_chain_confirmation(chain, 1)
    assert approved is True
    assert detail["status"] == "CONFIRMED"


def test_bearish_chain_confirmation():
    chain = pd.DataFrame([{"ce_oi": 120, "pe_oi": 100, "ce_oi_change": 20, "pe_oi_change": 5}])
    assert option_chain_confirmation(chain, -1)[0] is True
