from pathlib import Path
import sys
import numpy as np

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB.parent))
from institutional_hybrid.index_only.dashboard.monte_carlo import SimulationConfig,simulate
from institutional_hybrid.index_only.dashboard.analytics import enrich,replay_snapshot
import pandas as pd


def test_monte_carlo_is_deterministic_and_accepts_ten_rupees():
    returns=np.array([.01,-.005,.002,-.001])
    cfg=SimulationConfig(starting_capital=10,simulations=100,trades_per_path=20,seed=7)
    a=simulate(returns,cfg);b=simulate(returns,cfg)
    np.testing.assert_array_equal(a["equity"],b["equity"])
    assert a["equity"].shape==(100,21)
    assert 0<=a["risk_of_ruin"]<=1


def test_fixed_cost_makes_small_capital_more_fragile():
    returns=np.array([.01])
    cheap=simulate(returns,SimulationConfig(10,100,20,fixed_cost_rupees=1,seed=1))
    large=simulate(returns,SimulationConfig(10000,100,20,fixed_cost_rupees=1,seed=1))
    assert cheap["median_final"]<large["median_final"]
    assert cheap["risk_of_ruin"]>=large["risk_of_ruin"]


def test_replay_uses_only_closed_trades_for_capital():
    t=enrich(pd.DataFrame({
        "entry_time":["2025-01-01 09:16:00+05:30","2025-01-01 09:20:00+05:30"],
        "exit_time":["2025-01-01 09:18:00+05:30","2025-01-01 09:30:00+05:30"],
        "net_return":[.10,-.20],"hold_bars":[2,10],"regime":["bull_trend","expansion"],
    }))
    snap=replay_snapshot(t,pd.Timestamp("2025-01-01 09:25:00+05:30"),100)
    assert snap["closed"]==1 and snap["active"]==1
    assert abs(snap["capital"]-110)<1e-9
