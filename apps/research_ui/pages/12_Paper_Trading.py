"""Commercial paper-trading workflow over existing paper artifacts."""

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.charts import bars, line, style
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import app_footer, configure_page, icon, page_header


ROOT = Path("reports/paper_trading")


def csv(name: str) -> pd.DataFrame:
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


configure_page("Paper Trading")
page_header(
    "Paper Trading", "VIRTUAL EXECUTION",
    "Follow every simulated position from prediction to journal.")

trades = csv("paper_trade_log.csv")
risk = csv("risk_statistics.csv")
equity = csv("equity_curve.csv")
daily, monthly = csv("daily_summary.csv"), csv("monthly_summary.csv")
strategy, regime = csv("strategy_statistics.csv"), csv("regime_statistics.csv")
position_path = ROOT / "open_position.json"
position = (
    json.loads(position_path.read_text(encoding="utf-8"))
    if position_path.exists() else None)
stats = risk.iloc[0].to_dict() if len(risk) else {}

st.markdown(
    f"""<div class="workspace-status online">
      <div><i></i><strong>PAPER MODE</strong>
      <span>No broker orders or live capital</span></div>
      <div><span>Position</span><b>{'OPEN' if position else 'FLAT'}</b></div>
      <div><span>Journal</span><b>{len(trades)} TRADES</b></div>
    </div>""", unsafe_allow_html=True)

workflow = [
    ("Prediction", True), ("Open Position", bool(position)),
    ("Live P&L", bool(position)), ("Exit", bool(len(trades))),
    ("Journal", bool(len(trades))), ("Performance", bool(len(equity))),
]
st.markdown(
    '<section class="paper-flow">' + "".join(
        f"""<div class="paper-flow-step {'ready' if ready else 'waiting'}">
        {icon('shield-check' if ready else 'activity', 16)}
        <span>{label}</span><b>{'READY' if ready else 'WAITING'}</b></div>
        {f'<i>→</i>' if index < len(workflow)-1 else ''}"""
        for index, (label, ready) in enumerate(workflow)) + "</section>",
    unsafe_allow_html=True)

if position:
    position_values = [
        ("Strategy", str(position.get("strategy", "Unavailable")), "Virtual position"),
        ("Entry", f"₹{float(position.get('entry_premium', 0)):,.2f}", str(position.get("entry_timestamp", "—"))),
        ("Current", f"₹{float(position.get('current_premium', 0)):,.2f}", "Latest paper mark"),
        ("Live P&L", f"₹{float(position.get('current_pnl', 0)):,.2f}", "Unrealised"),
        ("Stop", f"₹{float(position.get('stop_loss', 0)):,.2f}", "Frozen at entry"),
        ("Target", f"₹{float(position.get('target', 0)):,.2f}", "Frozen at entry"),
    ]
    st.markdown(
        '<div class="desk-section-head"><div><span>CURRENT POSITION</span>'
        '<h3>Active virtual trade</h3></div><b>POSITION OPEN</b></div>',
        unsafe_allow_html=True)
else:
    position_values = []
    st.markdown(
        f"""<section class="paper-flat">{icon('wallet-cards', 28)}
        <div><span>CURRENT POSITION</span><h3>No open paper trade</h3>
        <p>The desk is ready for the next approved, contract-resolved prediction.</p></div>
        <a href="/" target="_self">Return to Trading Desk</a></section>""",
        unsafe_allow_html=True)

if position_values:
    cols = st.columns(6)
    for col, (label, value, detail) in zip(cols, position_values):
        with col:
            st.markdown(
                f"""<div class="desk-metric"><span>{label}</span><strong>{value}</strong>
                <small>{detail}</small></div>""", unsafe_allow_html=True)

summary = [
    ("Paper capital", f"₹{float(stats.get('ending_capital', 100000)):,.0f}", "Virtual balance"),
    ("Net P&L", f"₹{float(stats.get('net_pnl', 0)):,.0f}", "After configured costs"),
    ("Win rate", f"{float(stats.get('win_rate', 0)):.1%}", "Completed trades"),
    ("Profit factor", f"{float(stats.get('profit_factor', 0)):.2f}", "Gross wins / losses"),
    ("Total trades", f"{int(stats.get('total_trades', 0)):,}", "Paper journal"),
]
st.markdown(
    '<div class="desk-section-head"><div><span>ACCOUNT SUMMARY</span>'
    '<h3>Paper performance at a glance</h3></div></div>',
    unsafe_allow_html=True)
cols = st.columns(5)
for col, (label, value, detail) in zip(cols, summary):
    with col:
        st.markdown(
            f"""<div class="desk-metric"><span>{label}</span><strong>{value}</strong>
            <small>{detail}</small></div>""", unsafe_allow_html=True)

tabs = st.tabs([
    "Equity & Drawdown", "Daily / Monthly", "Distribution",
    "Strategy / Regime", "Trade Journal",
])
with tabs[0]:
    if len(equity):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                line(equity, "timestamp", "equity", None, "Paper equity"),
                use_container_width=True)
        with c2:
            fig = go.Figure(go.Scatter(
                x=equity.timestamp, y=equity.drawdown, fill="tozeroy",
                line=dict(color="#FF5A5F"), name="Drawdown"))
            fig.update_layout(title="Drawdown", yaxis_tickformat=".1%")
            st.plotly_chart(style(fig), use_container_width=True)
    else:
        st.info("Equity history will appear after completed paper trades.")
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        if len(daily):
            st.plotly_chart(
                bars(daily, "period", "net_pnl", None, "Daily P&L"),
                use_container_width=True)
        else:
            st.info("No daily summary is available.")
    with c2:
        if len(monthly):
            st.plotly_chart(
                bars(monthly, "period", "net_pnl", None, "Monthly P&L"),
                use_container_width=True)
        else:
            st.info("No monthly summary is available.")
with tabs[2]:
    if len(trades):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                trades, x="net_pnl", color=trades.net_pnl.gt(0),
                title="Win / loss distribution")
            st.plotly_chart(style(fig), use_container_width=True)
        with c2:
            fig = px.histogram(
                trades, x="holding_time_minutes",
                title="Holding-time distribution")
            st.plotly_chart(style(fig), use_container_width=True)
    else:
        st.info("Trade distributions require completed positions.")
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        if len(strategy):
            st.plotly_chart(
                bars(strategy, "strategy", "net_pnl", None, "Strategy performance"),
                use_container_width=True)
        else:
            st.info("Strategy results are unavailable.")
    with c2:
        if len(regime):
            st.plotly_chart(
                bars(regime, "regime", "net_pnl", None, "Regime performance"),
                use_container_width=True)
        else:
            st.info("Regime results are unavailable.")
with tabs[4]:
    enterprise_table(trades, key="paper_candidate_trades", page_size=25)

app_footer()
