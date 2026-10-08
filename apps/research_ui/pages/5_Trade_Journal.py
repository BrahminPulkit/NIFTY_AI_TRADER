import pandas as pd
import streamlit as st

from apps.research_ui.components.charts import line
from apps.research_ui.components.data import load_frame
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import (
    app_footer, configure_page, empty_state, metric_card, page_header,
    panel, section_header,
)

configure_page("Trade Journal")
page_header("Trade Journal", "AUDIT & REPLAY",
            "Institutional paper-trade ledger, skip attribution and deterministic replay.")
trades, skips, equity = load_frame("paper_trades"), load_frame("paper_skips"), load_frame("paper_equity")
cols = st.columns(5)
values = [
    ("Executed", f"{len(trades):,}", "Paper trades"), ("Skipped", f"{len(skips):,}", "Rejected opportunities"),
    ("Net P&L", f"₹{trades.net_pnl.sum():,.0f}" if len(trades) else None, "After costs"),
    ("Win Rate", f"{trades.net_pnl.gt(0).mean():.1%}" if len(trades) else None, "Executed"),
    ("Audit State", "COMPLETE", "Deterministic ledger"),
]
for col, item in zip(cols, values):
    with col: metric_card(*item, "neutral", "▤")

tabs = st.tabs(["Executed Trades", "Skipped Signals", "Equity", "Trade Replay"])
with tabs[0]:
    section_header("Execution Ledger", "Search, sort and inspect frozen paper trades")
    enterprise_table(trades, key="journal_v2_executed", page_size=20)
with tabs[1]:
    section_header("Skip Attribution", "Every opportunity rejected by the frozen funnel")
    enterprise_table(skips, key="journal_v2_skips", page_size=25)
with tabs[2]:
    section_header("Paper Equity", "Chronological account progression")
    if len(equity): st.plotly_chart(line(equity, "date", "equity", None, "Paper equity"), use_container_width=True)
    else: empty_state("Equity unavailable", "Frozen paper-equity artifact is missing.")
with tabs[3]:
    section_header("Deterministic Trade Replay", "Inspect one completed paper trade")
    if trades.empty:
        empty_state("No trade to replay", "The frozen ledger contains no executed trades.")
    else:
        index = st.selectbox("Select trade", range(len(trades)),
            format_func=lambda i: f"{trades.iloc[i].entry_timestamp} · {trades.iloc[i].strategy_family}")
        row = trades.iloc[index]
        rc = st.columns(4)
        with rc[0]: metric_card("Entry", row.entry_timestamp, "Virtual fill", "neutral", "→")
        with rc[1]: metric_card("Exit", row.exit_timestamp, "Frozen exit", "neutral", "←")
        with rc[2]: metric_card("Probability", f"{float(row.probability):.1%}", "OOF probability", "good", "◎")
        with rc[3]: metric_card("Net P&L", f"₹{float(row.net_pnl):,.0f}", f"Costs ₹{float(row.total_costs):,.0f}", "good" if row.net_pnl > 0 else "bad", "₹")
        panel("Replay Details", "<br>".join(f"<b>{key}</b>: {value}" for key, value in row.to_dict().items()), "FROZEN")
app_footer()
