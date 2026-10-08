import numpy as np
import pandas as pd
from src.contract_robustness import apply_scenario,metrics


def observations():
    return pd.DataFrame({"realized_r":[1.0,-1.0],"risk_points":[2.0,2.0],"direction":["BUY_CALL","BUY_PUT"],
                         "year":[2025,2025],"regime":["UPTREND","DOWNTREND"]})


def test_execution_cost_is_adverse_and_exact():
    scenario=pd.Series({"scenario_id":"x","slippage_ticks_per_side":1,"spread_ticks_roundtrip":2,
                        "tracking_points_per_side":.25,"ambiguity_penalty_ticks":1})
    result=apply_scenario(observations(),scenario)
    # total points = .10 slippage + .10 spread + .50 tracking + .05 ambiguity
    np.testing.assert_allclose(result.net_r, observations().realized_r-.75/2)


def test_profit_factor_and_expected_r():
    data=observations().rename(columns={"realized_r":"net_r"})
    data["realized_r"]=data.net_r;data["execution_cost_points"]=0
    result=metrics(data,["year"]).iloc[0]
    assert result.expected_r==0 and result.profit_factor==1


def test_more_cost_never_improves_expected_r():
    low=pd.Series({"scenario_id":"a","slippage_ticks_per_side":0,"spread_ticks_roundtrip":0,
                   "tracking_points_per_side":0.,"ambiguity_penalty_ticks":0})
    high=pd.Series({"scenario_id":"b","slippage_ticks_per_side":3,"spread_ticks_roundtrip":4,
                    "tracking_points_per_side":1.,"ambiguity_penalty_ticks":1})
    assert apply_scenario(observations(),high).net_r.mean()<apply_scenario(observations(),low).net_r.mean()
