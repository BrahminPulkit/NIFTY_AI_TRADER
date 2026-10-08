"""Execution-cost robustness for a frozen research label contract."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


TICK_SIZE = 0.05


def execution_scenarios() -> pd.DataFrame:
    rows = []
    for slip, spread, tracking, ambiguity in product(
            (0, 1, 2, 3), (0, 1, 2, 4), (0.0, 0.25, 0.50, 1.0), (0, 1)):
        scenario_id = f"SLIP{slip}_SPREAD{spread}_TRACK{tracking:.2f}_AMB{ambiguity}"
        rows.append({"scenario_id": scenario_id, "slippage_ticks_per_side": slip,
                     "spread_ticks_roundtrip": spread, "tracking_points_per_side": tracking,
                     "ambiguity_penalty_ticks": ambiguity,
                     "is_realistic_reference": slip == 1 and spread == 2 and tracking == .25 and ambiguity == 1})
    return pd.DataFrame(rows)


def apply_scenario(observations: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    result = observations.copy()
    base_points = (2*scenario.slippage_ticks_per_side + scenario.spread_ticks_roundtrip) * TICK_SIZE
    tracking_points = 2*scenario.tracking_points_per_side
    # Every trailing exit is conservatively treated as OHLC sequence-sensitive;
    # the ambiguity stress is an additional adverse tick at exit.
    ambiguity_points = scenario.ambiguity_penalty_ticks * TICK_SIZE
    total_points = base_points + tracking_points + ambiguity_points
    result["execution_cost_points"] = total_points
    result["net_r"] = result.realized_r - total_points/result.risk_points
    result["scenario_id"] = scenario.scenario_id
    return result


def metrics(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows=[]
    for keys, group in data.groupby(groups, sort=True, dropna=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        wins=group.net_r[group.net_r>0]; losses=group.net_r[group.net_r<0]
        rows.append({**dict(zip(groups,keys)),"observations":len(group),
                     "expected_r":group.net_r.mean(),"win_rate":group.net_r.gt(0).mean(),
                     "loss_rate":group.net_r.lt(0).mean(),
                     "profit_factor":wins.sum()/abs(losses.sum()) if len(losses) and losses.sum()!=0 else np.inf,
                     "minimum_trade_r":group.net_r.min(),"maximum_trade_r":group.net_r.max(),
                     "average_execution_cost_r":(group.realized_r-group.net_r).mean()})
    return pd.DataFrame(rows)

