import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.analytics import strategy_performance
from apps.research_ui.components.charts import style
from apps.research_ui.components.data import latest_snapshot, load_frame
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import app_footer, configure_page, metric_card, page_header, section_header, unavailable

configure_page("Strategy Center")
page_header("Strategy Center", "SYSTEMATIC PLAYBOOK",
            "Institutional leaderboard and regime intelligence for frozen strategy families.")
data = strategy_performance(load_frame("decision_matrix"))
snapshot = latest_snapshot()
section_header("Active Playbook", "Strategy selected for the latest market conditions")
cols = st.columns(5)
active = str(snapshot.get("current_strategy", "UNAVAILABLE")).replace("_", " ")
values = [("Active Strategy", active, "Current frozen family"), ("Decision", snapshot.get("recommendation", "SKIP"), snapshot.get("decision_reason", "")),
          ("Expected Expansion", f"{float(snapshot.get('expected_premium_expansion',0)):.2%}", "Premium MFE"),
          ("Expected Drawdown", f"{float(snapshot.get('expected_drawdown',0)):.2%}", "Premium MAE"),
          ("Holding", f"{float(snapshot.get('expected_holding_minutes',0)):.1f} MIN", "Expected duration")]
for col, item in zip(cols, values):
    with col: metric_card(*item, "neutral", "⌁")

section_header("Institutional Leaderboard", "Ranked frozen research performance", "FILTER · SORT · SEARCH")
if data.empty:
    unavailable("Strategy performance data is unavailable.")
else:
    board = data.copy().reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    board["strategy"] = board.strategy.str.replace("_", " ")
    board["status"] = board.profit_factor.map(lambda value: "ROBUST" if value >= 1.5 else "WATCH" if value >= 1 else "WEAK")
    board = board.rename(columns={"observations": "trades", "win_rate": "accuracy",
                                  "profit_factor": "pf"})
    enterprise_table(board, key="strategy_leaderboard", page_size=10, height=345)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=board.strategy, y=board.pf, name="Profit Factor", marker_color="#00C853"))
    fig.add_trace(go.Scatter(x=board.strategy, y=board.accuracy, name="Win Rate",
                             mode="lines+markers", yaxis="y2", line=dict(color="#2563EB")))
    fig.update_layout(title="Strategy quality comparison", yaxis2=dict(overlaying="y", side="right",
                      tickformat=".0%"), legend=dict(orientation="h"))
    st.plotly_chart(style(fig, 400), use_container_width=True)

section_header("Regime Matrix", "Strategy behaviour by market state")
matrix = load_frame("decision_matrix")
if len(matrix) and {"current_strategy", "market_regime", "realized_terminal_return"}.issubset(matrix.columns):
    pivot = matrix.pivot_table(index="current_strategy", columns="market_regime",
                               values="realized_terminal_return", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                               colorscale=[[0, "#EF4444"], [.5, "#EEF3FA"], [1, "#00C853"]]))
    fig.update_layout(title="Average terminal return by strategy and regime")
    st.plotly_chart(style(fig, 470), use_container_width=True)
else: unavailable("Regime matrix columns are unavailable.")
app_footer()
