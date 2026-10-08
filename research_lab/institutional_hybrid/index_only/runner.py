"""Execute deterministic index-only methodology evaluation."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

LAB=Path(__file__).resolve().parents[1]
if str(LAB.parent) not in sys.path:sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.index_only.data import build_features,config,load_and_validate
from institutional_hybrid.index_only.backtest import metrics,run
from institutional_hybrid.traders import TRADERS

OUT=Path(__file__).resolve().parent/"outputs"


def chart_set(trades:pd.DataFrame,folder:Path,title:str):
    folder.mkdir(parents=True,exist_ok=True)
    r=trades.get("net_return",pd.Series(dtype=float))
    eq=(1+r).cumprod(); dd=eq/eq.cummax()-1 if len(eq) else eq
    series=[
        ("equity_curve",eq,"line"),("drawdown_curve",dd,"line"),
        ("return_distribution",r,"hist"),
        ("holding_time",trades.get("hold_bars",pd.Series(dtype=float)),"hist"),
        ("r_multiple",trades.get("r_multiple",pd.Series(dtype=float)),"hist"),
    ]
    for name,s,kind in series:
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot(pd.Series(s).reset_index(drop=True)) if kind=="line" else ax.hist(pd.Series(s).dropna(),bins=30)
        ax.set_title(f"{title} — {name.replace('_',' ').title()}")
        fig.tight_layout();fig.savefig(folder/f"{name}.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    if not trades.empty:
        t=trades.copy();t["entry_time"]=pd.to_datetime(t.entry_time)
        t.groupby(t.entry_time.dt.to_period("M")).net_return.sum().plot.bar(ax=ax)
    ax.set_title(f"{title} — Monthly Returns");fig.tight_layout()
    fig.savefig(folder/"monthly_returns.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    if not trades.empty:trades.groupby("regime").net_return.mean().plot.bar(ax=ax)
    ax.set_title(f"{title} — Regime Performance");fig.tight_layout()
    fig.savefig(folder/"regime_performance.png",dpi=160);plt.close(fig)


def execute():
    cfg=config();frame,audit=load_and_validate()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"index_data_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
    frame.loc[frame.quarantined].to_csv(OUT/"quarantined_index_rows.csv")
    if not audit["approved"]:
        raise RuntimeError(f"Index validation failed: {audit}")
    features=build_features(frame);records=[]
    for trader in TRADERS:
        strategy=importlib.import_module(f"institutional_hybrid.traders.{trader}.strategy")
        signals=strategy.signal(features.copy())
        journal=run(signals,cfg);result=metrics(journal);result["strategy"]=trader
        result["average_strategy_score"]=float(journal.strategy_score.mean()) if not journal.empty else 0.
        result["rejected_setups"]=int(signals.rejection_reason.ne("").sum())
        folder=OUT/trader;folder.mkdir(parents=True,exist_ok=True)
        journal.to_csv(folder/"trade_journal.csv",index=False)
        signals.loc[signals.rejection_reason.ne(""),["rejection_reason"]].to_csv(folder/"rejected_setups.csv")
        pd.DataFrame([result]).to_csv(folder/"metrics.csv",index=False)
        chart_set(journal,folder/"charts",trader)
        records.append(result)
    board=pd.DataFrame(records)
    eligible=board.total_trades>=cfg["ranking"]["minimum_trades_for_candidate"]
    board["eligible"]=eligible
    board["rank"]=np.nan
    board.loc[eligible,"rank"]=board.loc[eligible,"expectancy"].rank(ascending=False,method="min")
    board=board.sort_values(["eligible","rank","expectancy"],ascending=[False,True,False])
    board.to_csv(OUT/"leaderboard.csv",index=False)
    board.to_parquet(OUT/"leaderboard.parquet",index=False)
    (OUT/"leaderboard.html").write_text(board.to_html(index=False),encoding="utf-8")
    winner=board.loc[board.eligible].iloc[0].strategy if eligible.any() else None
    report="# Index-only Strategy Evaluation\n\n"
    report+=("This is theoretical NIFTY spot signal evaluation, not a directly executable "
             "cash-index trading result. Signals enter at next-bar open with fixed costs and slippage.\n\n")
    report+=f"Source hash: `{audit['sha256']}`\n\n"
    display=board.copy()
    headers=list(display.columns)
    report+="| "+" | ".join(headers)+" |\n"
    report+="|"+"|".join(["---"]*len(headers))+"|\n"
    for values in display.itertuples(index=False,name=None):
        report+="| "+" | ".join("" if pd.isna(v) else str(v) for v in values)+" |\n"
    report+="\n"
    report+=f"Highest eligible expectancy: **{winner or 'none'}**.\n"
    (OUT/"Index_Only_Strategy_Report.md").write_text(report,encoding="utf-8")
    return board


if __name__=="__main__":execute()
