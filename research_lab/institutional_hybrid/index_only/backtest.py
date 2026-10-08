"""Causal next-open index signal backtester."""
from __future__ import annotations
import numpy as np
import pandas as pd


def run(signals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    e=cfg["execution"]; trades=[]; i=0
    index=signals.index
    opens=signals.open.to_numpy(float); highs=signals.high.to_numpy(float)
    lows=signals.low.to_numpy(float); closes=signals.close.to_numpy(float)
    signal=signals.signal.to_numpy(int); quarantined=signals.quarantined.to_numpy(bool)
    scores=signals.strategy_score.to_numpy(float); regimes=signals.regime.to_numpy()
    while i<len(signals)-1:
        if signal[i]==0 or quarantined[i] or quarantined[i+1]:
            i+=1; continue
        if index[i+1]-index[i]!=pd.Timedelta(minutes=1):
            i+=1; continue
        if index[i+1].date()!=index[i].date():
            i+=1; continue
        direction=int(signal[i]); entry_i=i+1
        entry=opens[entry_i]*(1+direction*e["slippage_each_side_pct"])
        stop=entry*(1-direction*e["stop_pct"]); target=entry*(1+direction*e["target_pct"])
        exit_i=entry_i; exit_price=closes[entry_i]; reason="time"
        max_i=min(entry_i+e["max_hold_bars"]-1,len(signals)-1)
        for j in range(entry_i,max_i+1):
            if quarantined[j]:
                exit_i=max(j-1,entry_i); exit_price=closes[exit_i]; reason="quarantine"; break
            if j>entry_i:
                gap=index[j]-index[j-1]
                if index[j].date()!=index[j-1].date():
                    exit_i=j-1; exit_price=closes[j-1]; reason="session_close"; break
                if gap!=pd.Timedelta(minutes=1):
                    exit_i=j; exit_price=opens[j]; reason="data_gap"; break
            hit_stop=(lows[j]<=stop) if direction>0 else (highs[j]>=stop)
            hit_target=(highs[j]>=target) if direction>0 else (lows[j]<=target)
            if hit_stop:
                exit_i=j; exit_price=stop; reason="stop"; break
            if hit_target:
                exit_i=j; exit_price=target; reason="target"; break
            exit_i=j; exit_price=closes[j]
        exit_price*=1-direction*e["slippage_each_side_pct"]
        gross=direction*(exit_price/entry-1); net=gross-e["round_trip_cost_pct"]
        trades.append({
            "signal_time":index[i],"entry_time":index[entry_i],
            "exit_time":index[exit_i],"direction":direction,
            "entry":entry,"exit":exit_price,"gross_return":gross,"net_return":net,
            "r_multiple":net/e["stop_pct"],"hold_bars":exit_i-entry_i+1,
            "exit_reason":reason,"strategy_score":scores[i],
            "regime":regimes[i],
        })
        i=exit_i+1
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    names="total_trades win_rate loss_rate profit_factor expectancy average_r_multiple average_hold_time maximum_drawdown sharpe sortino calmar recovery_factor maximum_winning_streak maximum_losing_streak".split()
    if trades.empty:return {x:0.0 for x in names}
    r=trades.net_return; eq=(1+r).cumprod(); dd=eq/eq.cummax()-1
    std=r.std(ddof=1); down=r[r<0].std(ddof=1); maxdd=float(dd.min())
    wins=(r>0).astype(int); groups=wins.ne(wins.shift()).cumsum()
    win_streak=int(wins[wins.eq(1)].groupby(groups).size().max()) if wins.any() else 0
    loss_streak=int(wins[wins.eq(0)].groupby(groups).size().max()) if (~wins.astype(bool)).any() else 0
    days=max((pd.to_datetime(trades.exit_time).max()-pd.to_datetime(trades.entry_time).min()).days,1)
    annual=eq.iloc[-1]**(365/days)-1
    return {
        "total_trades":float(len(trades)),"win_rate":float((r>0).mean()),"loss_rate":float((r<=0).mean()),
        "profit_factor":float(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else float("inf"),
        "expectancy":float(r.mean()),"average_r_multiple":float(trades.r_multiple.mean()),
        "average_hold_time":float(trades.hold_bars.mean()),"maximum_drawdown":maxdd,
        "sharpe":float(r.mean()/std*np.sqrt(252)) if std and np.isfinite(std) else 0.,
        "sortino":float(r.mean()/down*np.sqrt(252)) if down and np.isfinite(down) else 0.,
        "calmar":float(annual/abs(maxdd)) if maxdd else 0.,
        "recovery_factor":float((eq.iloc[-1]-1)/abs(maxdd)) if maxdd else 0.,
        "maximum_winning_streak":float(win_streak),"maximum_losing_streak":float(loss_streak),
    }
