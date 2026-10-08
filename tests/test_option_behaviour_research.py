import pandas as pd
import pytest

from src.option_behaviour_research import evaluate_option_behaviour


def fixture():
    timestamp = pd.date_range("2026-01-02 09:15", periods=8, freq="min", tz="Asia/Kolkata")
    aligned = pd.DataFrame({"timestamp":timestamp, "option_open":100., "option_high":101.,
                            "option_low":99., "option_close":[100,101,102,103,104,105,106,107],
                            "option_volume":[100,110,120,130,140,150,160,170]})
    setups = pd.DataFrame({"timestamp":timestamp, "entry_signal":[1,0,0,0,0,0,0,0],
                           "trade_allowed":[1,0,0,0,0,0,0,0], "trend_state":"UPTREND",
                           "minutes_from_open":range(8), "session_id":20260102})
    return aligned, setups


def test_premium_path_and_holding_are_exact():
    aligned, setups = fixture(); result = evaluate_option_behaviour(aligned, setups)
    row=result.iloc[0]
    assert row.holding_candles == 5 and row.complete_horizon
    assert row.final_premium_return == pytest.approx(.05) and row.option_type == "CALL"


def test_only_allowed_and_no_cross_gap_jump():
    aligned, setups = fixture(); aligned=aligned.drop(index=2).reset_index(drop=True)
    result=evaluate_option_behaviour(aligned,setups)
    assert len(result)==1 and result.iloc[0].holding_candles==1
    assert not result.iloc[0].complete_horizon
