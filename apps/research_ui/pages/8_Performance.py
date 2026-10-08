import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.charts import line, style
from apps.research_ui.components.data import load_frame
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import app_footer, configure_page, metric_card, page_header, section_header, unavailable

configure_page("Performance")
page_header("Performance Analytics", "PORTFOLIO INTELLIGENCE",
            "Institutional risk, return and consistency analysis from frozen paper research.")
trades, equity = load_frame("paper_trades"), load_frame("paper_equity")
if len(trades):
    returns = pd.to_numeric(trades.net_pnl, errors="coerce")
    gross_win, gross_loss = returns[returns > 0].sum(), -returns[returns < 0].sum()
    metrics = [("Net P&L", f"₹{returns.sum():,.0f}", "After configured costs"),
               ("Win Rate", f"{returns.gt(0).mean():.1%}", "Executed paper trades"),
               ("Profit Factor", f"{gross_win/gross_loss:.2f}" if gross_loss else "∞", "Gross win / loss"),
               ("Average Trade", f"₹{returns.mean():,.0f}", "Net expectancy"),
               ("Trades", f"{len(trades):,}", "Frozen replay")]
else:
    metrics = [(name, None, "Unavailable") for name in ("Net P&L", "Win Rate", "Profit Factor", "Average Trade", "Trades")]
cols = st.columns(5)
for col, item in zip(cols, metrics):
    with col: metric_card(*item, "neutral", "◒")

section_header("Capital & Drawdown", "Frozen paper-equity progression")
left, right = st.columns([1.3, .7])
with left:
    if len(equity): st.plotly_chart(line(equity, "date", "equity", None, "Equity curve"), use_container_width=True)
    else: unavailable("Paper equity is unavailable.")
with right:
    if len(equity):
        series = pd.to_numeric(equity.equity, errors="coerce")
        drawdown = series / series.cummax() - 1
        fig = go.Figure(go.Scatter(x=equity.date, y=drawdown, fill="tozeroy",
                                   line=dict(color="#EF4444"), name="Drawdown"))
        fig.update_layout(title="Underwater curve", yaxis_tickformat=".1%")
        st.plotly_chart(style(fig), use_container_width=True)
    else: unavailable("Drawdown cannot be calculated.")

section_header("Monthly Return Matrix", "Calendar consistency heatmap")
if len(trades) and "exit_timestamp" in trades:
    temp = trades.copy()
    temp["date"] = pd.to_datetime(temp.exit_timestamp)
    temp["year"], temp["month"] = temp.date.dt.year, temp.date.dt.strftime("%b")
    pivot = temp.pivot_table(index="year", columns="month", values="net_pnl", aggfunc="sum")
    order = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] if m in pivot]
    pivot = pivot[order]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                               colorscale=[[0,"#EF4444"],[.5,"#EEF3FA"],[1,"#00C853"]],
                               text=np.round(pivot.values, 0), texttemplate="₹%{text:,.0f}"))
    fig.update_layout(title="Monthly net P&L")
    st.plotly_chart(style(fig, 390), use_container_width=True)
else: unavailable("Monthly return inputs are unavailable.")
section_header("Institutional Trade Ledger", "Searchable frozen execution research")
enterprise_table(trades, key="performance_v2", page_size=20)
app_footer()
