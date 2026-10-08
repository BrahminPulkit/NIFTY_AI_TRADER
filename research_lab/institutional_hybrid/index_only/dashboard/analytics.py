"""Daily and conditional edge diagnostics from real journals."""
from __future__ import annotations
import numpy as np
import pandas as pd


def drawdown(returns:pd.Series)->float:
    if returns.empty:return 0.
    eq=(1+returns).cumprod();return float((eq/eq.cummax()-1).min())


def daily_performance(trades:pd.DataFrame)->pd.DataFrame:
    t=trades.copy();t["date"]=pd.to_datetime(t.entry_time).dt.date
    return t.groupby("date").net_return.apply(lambda x:(1+x).prod()-1).rename("return").reset_index()


def breakdown(trades:pd.DataFrame,column:str)->pd.DataFrame:
    def stats(g):
        r=g.net_return
        loss=-r[r<0].sum()
        return pd.Series({
            "trades":len(g),"win_rate":(r>0).mean(),"expectancy":r.mean(),
            "profit_factor":r[r>0].sum()/loss if loss>0 else np.inf,
            "total_compounded_return":(1+r).prod()-1,
            "maximum_drawdown":drawdown(r.reset_index(drop=True)),
            "average_hold":g.hold_bars.mean(),
        })
    return trades.groupby(column,observed=True).apply(stats,include_groups=False).reset_index()


def enrich(trades:pd.DataFrame)->pd.DataFrame:
    t=trades.copy()
    t["entry_time"]=pd.to_datetime(t.entry_time)
    t["exit_time"]=pd.to_datetime(t.exit_time)
    minutes=t.entry_time.dt.hour*60+t.entry_time.dt.minute
    t["time_slot"]=pd.cut(
        minutes,bins=[554,629,719,809,899,930],
        labels=["09:15–10:29","10:30–11:59","12:00–13:29","13:30–14:59","15:00–15:30"],
    )
    t["outcome"]=np.where(t.net_return>0,"Win","Loss")
    t["volatility_context"]=t.regime.map({
        "compression":"Low / Compression","expansion":"High / Expansion",
        "gap_day":"Gap / Event",
    }).fillna("Normal / Trend-Sideways")
    return t


def replay_snapshot(trades:pd.DataFrame,replay_time,starting_capital:float)->dict:
    """Capital and state using only events known by the replay timestamp."""
    replay_time=pd.Timestamp(replay_time)
    entered=trades[trades.entry_time<=replay_time].copy()
    closed=entered[entered.exit_time<=replay_time].copy()
    active=entered[entered.exit_time>replay_time].copy()
    capital=float(starting_capital*((1+closed.net_return).prod() if len(closed) else 1))
    pnl=capital-starting_capital
    return {
        "capital":capital,"pnl":pnl,"entered":len(entered),"closed":len(closed),
        "active":len(active),"wins":int((closed.net_return>0).sum()),
        "losses":int((closed.net_return<=0).sum()),"closed_trades":closed,"active_trades":active,
    }
