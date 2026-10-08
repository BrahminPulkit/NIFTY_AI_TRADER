import numpy as np
import pandas as pd

from src.label_contract_research import ContractSpec, OpportunityMatrix


def fixture():
    return pd.DataFrame({
        "timestamp":pd.date_range("2026-01-02 09:30",periods=7,freq="min",tz="Asia/Kolkata"),
        "open":[100]*7,"high":[100,100.1,100.31,100,100,100,100],
        "low":[100,99.9,99.9,100,100,100,100],"close":[100]*7,
        "atr_14":[.2]*7,"entry_signal":[1,0,0,0,0,0,0],"trade_allowed":[1,0,0,0,0,0,0],
        "trend_state":["UPTREND"]*7,"minutes_from_open":range(15,22),"session_id":[20260102]*7})


def test_fixed_barrier_target_and_excursions():
    matrix=OpportunityMatrix(fixture(),5)
    spec=ContractSpec("x","FIXED",5,.002,1.5)
    row=matrix.evaluate(spec).iloc[0]
    assert row.outcome=="WIN" and row.holding_candles==2
    assert np.isclose(row.realized_r,1.5) and np.isclose(row.mfe_r,1.55)


def test_stop_first_ambiguity():
    data=fixture();data.loc[1,["high","low"]]=[100.4,99.7]
    row=OpportunityMatrix(data,5).evaluate(ContractSpec("x","FIXED",5,.002,1.5)).iloc[0]
    assert row.outcome=="STOP" and np.isclose(row.realized_r,-1)


def test_atr_risk_and_time_exit():
    matrix=OpportunityMatrix(fixture(),5)
    row=matrix.evaluate(ContractSpec("x","TIME_ATR",2,1.0,None,"TIME")).iloc[0]
    assert row.outcome=="TIMEOUT" and row.holding_candles==2 and np.isclose(row.realized_r,0)


def test_no_future_beyond_horizon_leakage():
    data=fixture();spec=ContractSpec("x","FIXED",2,.002,1.5)
    before=OpportunityMatrix(data,5).evaluate(spec).iloc[0]
    data.loc[3:,["high","low"]]=[1000,1]
    after=OpportunityMatrix(data,5).evaluate(spec).iloc[0]
    assert before[["outcome","holding_candles","mfe_r","mae_r","realized_r"]].equals(
        after[["outcome","holding_candles","mfe_r","mae_r","realized_r"]])
