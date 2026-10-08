"""Local Streamlit dashboard; completely separate from production UI."""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]
if str(LAB.parent) not in sys.path:sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.index_only.dashboard.analytics import breakdown,daily_performance,enrich,replay_snapshot
from institutional_hybrid.index_only.dashboard.monte_carlo import SimulationConfig,simulate
from institutional_hybrid.index_only.data import config as index_config

OUTPUT=LAB/"index_only"/"outputs"
STRATEGIES=sorted(p.name for p in OUTPUT.iterdir() if p.is_dir() and (p/"trade_journal.csv").exists())

st.set_page_config(page_title="Institutional Index Strategy Lab",layout="wide")
st.title("Institutional Index Strategy Evaluation")
st.caption("Research-only theoretical NIFTY spot signals. No options, broker execution, or production integration.")

with st.sidebar:
    strategy=st.selectbox("Strategy",STRATEGIES,index=STRATEGIES.index("oliver_velez") if "oliver_velez" in STRATEGIES else 0)
    compare_strategies=st.multiselect(
        "Strategies in daily replay",STRATEGIES,
        default=["oliver_velez","linda_raschke"] if {"oliver_velez","linda_raschke"}.issubset(STRATEGIES) else STRATEGIES[:2],
    )
    st.subheader("Capital stress test")
    capital=st.number_input("Starting capital (₹)",min_value=10.0,value=10000.0,step=10.0)
    exposure=st.slider("Capital exposed per trade",.05,1.0,1.0,.05)
    fixed_cost=st.number_input("Additional fixed cost/trade (₹)",min_value=0.0,value=0.0,step=.5)
    simulations=st.slider("Monte Carlo paths",500,10000,2000,500)
    horizon=st.slider("Trades per path",25,1000,250,25)
    block=st.slider("Bootstrap block size",1,25,5)
    ruin=st.slider("Ruin threshold (% initial capital)",5,90,25)/100
    seed=st.number_input("Random seed",min_value=0,value=42,step=1)

@st.cache_data
def load(name):
    return enrich(pd.read_csv(OUTPUT/name/"trade_journal.csv"))

@st.cache_data
def load_index():
    cfg=index_config()
    frame=pd.read_csv(cfg["source"],usecols=["timestamp","open","high","low","close","volume"])
    frame["timestamp"]=pd.to_datetime(frame.timestamp,unit="s",utc=True).dt.tz_convert(cfg["timezone"])
    return frame.set_index("timestamp").between_time(cfg["session"]["start"],cfg["session"]["end"]).sort_index()

t=load(strategy)
index_data=load_index()
r=t.net_return
equity=(1+r).cumprod()
dd=equity/equity.cummax()-1
days=daily_performance(t)

cols=st.columns(6)
cols[0].metric("Trades",f"{len(t):,}")
cols[1].metric("Win rate",f"{(r>0).mean():.2%}")
cols[2].metric("Expectancy",f"{r.mean():.4%}")
cols[3].metric("Profit factor",f"{r[r>0].sum()/-r[r<0].sum():.3f}")
cols[4].metric("Positive days",f"{(days['return']>0).sum():,}")
cols[5].metric("Negative days",f"{(days['return']<0).sum():,}")

if r.mean()<=0:
    st.error("Historical expectancy is negative after the configured percentage costs. Monte Carlo projections are stress tests, not forecasts.")

tab0,tab1,tab2,tab3,tab4=st.tabs(["Daily Multi-Strategy Replay","Performance","When it works / fails","Monte Carlo","Trade journal"])
with tab0:
    st.subheader("Historical 1-minute replay")
    available_dates=pd.Index(index_data.index.date).unique().sort_values()
    selected_date=st.date_input(
        "Trading date",value=available_dates[-1],
        min_value=available_dates[0],max_value=available_dates[-1],
    )
    day=index_data[index_data.index.date==selected_date]
    if day.empty:
        st.warning("No index candles exist for this date.")
    elif not compare_strategies:
        st.warning("Select at least one strategy in the sidebar.")
    else:
        replay_pos=st.slider("Replay minute",0,len(day)-1,len(day)-1)
        replay_time=day.index[replay_pos]
        visible=day.iloc[:replay_pos+1]
        st.caption(f"Replay state: {replay_time}")
        colors=["#00d4ff","#ffb000","#ff4b4b","#7bdc65","#b58cff","#ff79c6"]
        fig=go.Figure(go.Candlestick(
            x=visible.index,open=visible.open,high=visible.high,low=visible.low,close=visible.close,
            name="NIFTY 50",increasing_line_color="#26a69a",decreasing_line_color="#ef5350",
        ))
        comparison=[];logs=[]
        for color,name in zip(colors,compare_strategies):
            journal=load(name)
            daily=journal[journal.entry_time.dt.date==selected_date].copy()
            snap=replay_snapshot(daily,replay_time,capital)
            closed=snap["closed_trades"]; active=snap["active_trades"]
            known_entries=daily[daily.entry_time<=replay_time]
            fig.add_trace(go.Scatter(
                x=known_entries.entry_time,y=known_entries.entry,mode="markers",
                marker=dict(symbol="triangle-up",size=10,color=color),
                name=f"{name} entries",
                text=known_entries.direction.map({1:"LONG",-1:"SHORT"}),
            ))
            if not closed.empty:
                fig.add_trace(go.Scatter(
                    x=closed.exit_time,y=closed.exit,mode="markers",
                    marker=dict(symbol="x",size=9,color=color),name=f"{name} exits",
                ))
            comparison.append({
                "strategy":name,"active_capital":snap["capital"],"running_pnl":snap["pnl"],
                "pnl_pct":snap["pnl"]/capital if capital else 0,"trades_entered":snap["entered"],
                "trades_closed":snap["closed"],"active_positions":snap["active"],
                "wins":snap["wins"],"losses":snap["losses"],
            })
            if not known_entries.empty:
                entry_logs=known_entries.assign(strategy=name,event="ENTRY",event_time=known_entries.entry_time,
                    details=known_entries.apply(lambda x:f"{'LONG' if x.direction==1 else 'SHORT'} @ {x.entry:.2f}",axis=1))
                logs.append(entry_logs[["event_time","strategy","event","details"]])
            if not closed.empty:
                exit_logs=closed.assign(strategy=name,event="EXIT",event_time=closed.exit_time,
                    details=closed.apply(lambda x:f"{x.exit_reason} @ {x.exit:.2f}; P&L {x.net_return:.3%}",axis=1))
                logs.append(exit_logs[["event_time","strategy","event","details"]])
        fig.update_layout(
            title=f"NIFTY 50 — {selected_date}",height=650,xaxis_rangeslider_visible=False,
            template="plotly_dark",legend_orientation="h",
        )
        st.plotly_chart(fig,use_container_width=True)
        comp=pd.DataFrame(comparison).sort_values("running_pnl",ascending=False)
        st.markdown("#### Strategy comparison at replay time")
        st.dataframe(
            comp.style.format({"active_capital":"₹{:,.2f}","running_pnl":"₹{:,.2f}","pnl_pct":"{:.2%}"}),
            use_container_width=True,hide_index=True,
        )
        if len(comp):
            leader=comp.iloc[0]
            st.info(f"Current leader: **{leader.strategy}** with running P&L ₹{leader.running_pnl:,.2f}. This is historical replay, not live trading.")
        st.markdown("#### How each selected strategy works")
        descriptions={
            "linda_raschke":"EMA pullback plus trend continuation in aligned trend/expansion regimes.",
            "al_brooks":"Trend bar after a pullback bar, excluding sideways/compression and failed-break contexts.",
            "oliver_velez":"Trend continuation with relative-volume participation and range expansion.",
            "adam_grimes":"EMA pullback with three-bar momentum and trend-regime context.",
            "mark_minervini":"Intraday contraction breakout with relative volume and trend alignment.",
            "ict_research":"Restricted liquidity sweep followed by market-structure shift and EMA alignment.",
        }
        for name in compare_strategies: st.markdown(f"- **{name}:** {descriptions[name]}")
        st.markdown("#### Execution log")
        if logs:
            log=pd.concat(logs).sort_values(["event_time","strategy"])
            st.dataframe(log,use_container_width=True,hide_index=True)
        else: st.caption("No entries or exits occurred before this replay minute.")
        st.markdown("#### Trade details")
        detail=[]
        for name in compare_strategies:
            journal=load(name)
            d=journal[(journal.entry_time.dt.date==selected_date)&(journal.entry_time<=replay_time)].copy()
            if len(d): d.insert(0,"strategy",name);detail.append(d)
        if detail: st.dataframe(pd.concat(detail).sort_values("entry_time"),use_container_width=True,hide_index=True)
        else: st.caption("No strategy trades on this date up to the replay minute.")
with tab1:
    c1,c2=st.columns(2)
    c1.plotly_chart(px.line(x=equity.index,y=equity,title="Historical Equity Multiple"),use_container_width=True)
    c2.plotly_chart(px.area(x=dd.index,y=dd,title="Historical Drawdown"),use_container_width=True)
    c1.plotly_chart(px.histogram(days,x="return",color=days["return"].gt(0).map({True:"Positive",False:"Negative"}),
                                      title="Daily Return Distribution"),use_container_width=True)
    daily_counts=pd.DataFrame({"Outcome":["Positive","Negative","Flat"],
       "Days":[(days["return"]>0).sum(),(days["return"]<0).sum(),(days["return"]==0).sum()]})
    c2.plotly_chart(px.bar(daily_counts,x="Outcome",y="Days",color="Outcome",title="Positive vs Negative Days"),use_container_width=True)
with tab2:
    dimension=st.radio("Breakdown",["regime","volatility_context","time_slot"],horizontal=True)
    table=breakdown(t,dimension).sort_values("expectancy",ascending=False)
    st.dataframe(table,use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    c1.plotly_chart(px.bar(table,x=dimension,y="expectancy",color="expectancy",
                           color_continuous_scale="RdYlGn",title="Expectancy by Condition"),use_container_width=True)
    c2.plotly_chart(px.bar(table,x=dimension,y="maximum_drawdown",color="maximum_drawdown",
                           color_continuous_scale="RdYlGn",title="Drawdown by Condition"),use_container_width=True)
    st.markdown("**Works best:** highest measured expectancy. **Breaks down:** negative expectancy and deepest conditional drawdown. Small groups should not be treated as reliable edges.")
with tab3:
    cfg=SimulationConfig(capital,simulations,horizon,block,exposure,fixed_cost,ruin,seed)
    mc=simulate(r,cfg)
    c=st.columns(6)
    c[0].metric("Risk of ruin",f"{mc['risk_of_ruin']:.1%}")
    c[1].metric("P(final > start)",f"{mc['probability_of_success']:.1%}")
    c[2].metric("P(profit ≥10%)",f"{mc['probability_of_profit_10pct']:.1%}")
    c[3].metric("Median final",f"₹{mc['median_final']:,.2f}")
    c[4].metric("5th percentile",f"₹{mc['p05_final']:,.2f}")
    c[5].metric("Median max DD",f"{mc['median_max_drawdown']:.1%}")
    eqs=mc["equity"];x=list(range(eqs.shape[1]))
    fig=go.Figure()
    for q,name in ((.05,"5th"),(.5,"Median"),(.95,"95th")):
        fig.add_trace(go.Scatter(x=x,y=pd.DataFrame(eqs).quantile(q),name=name))
    fig.update_layout(title="Monte Carlo Equity Fan",xaxis_title="Trade",yaxis_title="Capital (₹)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Block bootstrap resamples historical trade-return sequences. Fixed rupee cost is deducted after every simulated trade. Results are conditional on this historical sample.")
with tab4:
    st.dataframe(t.sort_values("entry_time",ascending=False),use_container_width=True,hide_index=True)
