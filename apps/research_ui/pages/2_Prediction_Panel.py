import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.charts import bars, gauge, line, style
from apps.research_ui.components.data import latest_snapshot, load_frame
from apps.research_ui.components.operational import decision_explanation
from apps.research_ui.components.theme import (
    ai_thinking, app_footer, configure_page, empty_state, metric_card,
    page_header, panel, section_header, timeline,
)

configure_page("AI Predictions")
page_header("AI Prediction Centre", "MODEL INTELLIGENCE",
            "Transparent probability, risk and feature evidence from the frozen research model.")
row = latest_snapshot()
if not row:
    empty_state("Prediction unavailable", "No frozen probability observation is available.")
    st.stop()
probability = float(row.get("catboost_predicted_probability", 0))
decision = str(row.get("recommendation", "SKIP"))

section_header("Current Prediction", f"Frozen record · {pd.to_datetime(row['timestamp'])}",
               "NO NEW INFERENCE")
left, right = st.columns([.82, 1.18])
with left:
    st.plotly_chart(gauge(probability, f"{decision} · CatBoost probability"), use_container_width=True)
    signal_cols = st.columns(2)
    with signal_cols[0]: metric_card("Confidence", f"{float(row.get('strategy_confidence_score', 0)):.1%}", "Decision confidence", "good", "◎")
    with signal_cols[1]: metric_card("Risk", f"{float(row.get('expected_drawdown', 0)):.2%}", "Expected drawdown", "bad", "↘")
    signal_cols = st.columns(2)
    with signal_cols[0]: metric_card("Reward", f"{float(row.get('expected_premium_expansion', 0)):.2%}", "Expected expansion", "good", "↗")
    with signal_cols[1]: metric_card("Holding", f"{float(row.get('expected_holding_minutes', 0)):.1f} MIN", "Estimated duration", "neutral", "◷")
with right:
    ai_thinking(decision)
    panel("Decision Explanation",
          f"<b>{decision}</b><br>{str(row.get('decision_reason','Unavailable')).replace('_',' ')}<br><br>"
          f"Strategy: {str(row.get('current_strategy','N/A')).replace('_',' ')}<br>"
          f"Regime: {str(row.get('market_state','N/A')).replace('_',' ')}", "CAUSAL")
    with st.expander("Explain every rule and contribution", expanded=True):
        decision_explanation(row, load_frame("feature_importance"))

section_header("Feature Intelligence", "Model importance and SHAP attribution", "FROZEN RESEARCH")
importance, shap = load_frame("feature_importance"), load_frame("shap")
c1, c2 = st.columns(2)
with c1:
    if len(importance):
        subset = importance.loc[importance.model.eq("CATBOOST")].nlargest(12, "importance")
        st.plotly_chart(bars(subset, "transformed_feature", "importance", None,
                             "Top CatBoost features"), use_container_width=True)
    else: empty_state("Feature importance unavailable", "Frozen importance file is missing.")
with c2:
    if len(shap):
        subset = shap.loc[shap.model.eq("CATBOOST")].nlargest(12, "mean_absolute_shap")
        st.plotly_chart(bars(subset, "transformed_feature", "mean_absolute_shap", None,
                             "Mean absolute SHAP"), use_container_width=True)
    else: empty_state("SHAP unavailable", "Frozen SHAP summary is missing.")

section_header("Probability History", "Leakage-free OOF probability timeline")
joined = load_frame("joined_probability")
if len(joined):
    history = joined.sort_values("timestamp").tail(500)
    st.plotly_chart(line(history, "timestamp", "catboost_predicted_probability", None,
                         "Recent frozen probability history"), use_container_width=True)
    items = []
    for item in history.tail(5).iloc[::-1].itertuples():
        items.append({"time": pd.to_datetime(item.timestamp).strftime("%d %b · %H:%M"),
                      "signal": getattr(item, "recommendation", "SKIP"),
                      "strategy": str(getattr(item, "current_strategy", "N/A")).replace("_", " "),
                      "probability": f"{float(item.catboost_predicted_probability):.1%}",
                      "reason": str(getattr(item, "decision_reason", "")).replace("_", " ")})
    timeline(items)
else: empty_state("History unavailable", "Joined probability research is unavailable.")
app_footer()
