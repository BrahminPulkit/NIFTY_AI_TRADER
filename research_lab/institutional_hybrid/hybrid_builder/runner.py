"""Execute four declared hybrids on the same immutable feature frame."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

LAB=Path(__file__).resolve().parents[1]
if str(LAB.parent) not in sys.path: sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.experiments.phase1_engine import load_data
from institutional_hybrid.strategies import InstitutionalHybridStrategy
from institutional_hybrid.traders.core import FixedExecution, backtest, frame_hash, metrics, write_json
from institutional_hybrid.traders.charting import all_charts
from institutional_hybrid.hybrid_builder.definitions import HYBRIDS
from institutional_hybrid.utils.research_gate import require_approved_data

OUTPUT=Path(__file__).resolve().parent/"outputs"


def execute(data=None):
    require_approved_data()
    futures,options=data or load_data({})
    builder=InstitutionalHybridStrategy()
    features=builder.score(builder.build_features(futures,options))
    digest=frame_hash(features); rows=[]
    for hybrid in HYBRIDS:
        entry=importlib.import_module(f"institutional_hybrid.traders.{hybrid.entry_package}.rules").entry
        filter_fn=importlib.import_module(f"institutional_hybrid.traders.{hybrid.filter_package}.filters").apply
        raw=entry(features).fillna(False); accepted=filter_fn(features,raw).fillna(False)
        signals=features.copy()
        direction=np.where(signals["market_structure_shift"]!=0,signals["market_structure_shift"],
                           np.where(signals["breakout_direction"]!=0,signals["breakout_direction"],signals["trend_direction"]))
        signals["signal"]=np.where(accepted,direction,0).astype(int)
        signals["strategy_score"]=signals[["institutional_score","premium_expansion_score"]].mean(axis=1)
        signals["rejection_reason"]=np.where(raw & ~accepted,f"{hybrid.filter_package}_filter","")
        trades=backtest(signals,FixedExecution())
        rejected=signals.loc[signals.rejection_reason.ne("")]
        candidate_count=int((raw).sum())
        result=metrics(trades,len(rejected)/candidate_count if candidate_count else 0)
        result["strategy"]=hybrid.name
        folder=OUTPUT/hybrid.name; folder.mkdir(parents=True,exist_ok=True)
        trades.to_csv(folder/"trade_journal.csv",index=False)
        rejected.to_csv(folder/"rejected_setups.csv")
        all_charts(trades,rejected,folder/"charts",hybrid.name.upper())
        write_json(folder/"metrics.json",result)
        write_json(folder/"manifest.json",{
            "hybrid":hybrid.name,"entry":hybrid.entry_package,"filter":hybrid.filter_package,
            "exit":hybrid.exit_package,"dataset_hash":digest,"optimised":False,
        })
        rows.append(result)
    board=pd.DataFrame(rows).sort_values("expectancy",ascending=False)
    board.to_csv(OUTPUT/"hybrid_leaderboard.csv",index=False)
    return board


if __name__=="__main__": execute()
