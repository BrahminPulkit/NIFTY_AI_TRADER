import plotly.express as px
import streamlit as st

from apps.research_ui.components.charts import bars, line, style
from apps.research_ui.components.data import load_frame
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import app_footer, configure_page, empty_state, metric_card, page_header, section_header

configure_page("Research Analytics")
page_header("Research Analytics", "VALIDATION LAB",
            "Professional model discrimination, drift, robustness and readiness evidence.")
tabs = st.tabs(["ROC & PR", "Feature PSI", "Monte Carlo", "Walk Forward", "Decision Funnel", "Readiness"])
with tabs[0]:
    roc, pr = load_frame("v2_roc"), load_frame("v2_pr")
    c1, c2 = st.columns(2)
    with c1:
        if len(roc): st.plotly_chart(line(roc, "false_positive_rate", "true_positive_rate", "model", "Receiver Operating Characteristic"), use_container_width=True)
        else: empty_state("ROC unavailable", "Frozen ROC curve is missing.")
    with c2:
        if len(pr): st.plotly_chart(line(pr, "recall", "precision", "model", "Precision–Recall Curve"), use_container_width=True)
        else: empty_state("PR unavailable", "Frozen PR curve is missing.")
    metric_card("Interpretation", "DISCRIMINATION", "ROC measures ranking; PR focuses on positive-class quality", "neutral", "?")
with tabs[1]:
    data = load_frame("feature_drift")
    if len(data):
        shown = data.nlargest(30, "original_psi")
        st.plotly_chart(bars(shown, "feature", "original_psi", "classification", "Population Stability Index"), use_container_width=True)
        enterprise_table(shown, key="research_psi")
    else: empty_state("PSI unavailable", "Feature drift research is missing.")
with tabs[2]:
    data = load_frame("monte_carlo")
    if len(data):
        subset = data.loc[(data.model == "CATBOOST") & (data.threshold == .9)]
        fig = px.histogram(subset, x="net_return", nbins=55, title="CatBoost 0.90 Monte Carlo distribution")
        st.plotly_chart(style(fig), use_container_width=True)
    else: empty_state("Monte Carlo unavailable", "Simulation distribution is missing.")
with tabs[3]:
    data = load_frame("walk_forward")
    if len(data):
        st.plotly_chart(bars(data, "fold", "roc_auc", "model", "Walk-forward stability"), use_container_width=True)
        enterprise_table(data, key="research_wf")
    else: empty_state("Walk forward unavailable", "Fold results are missing.")
with tabs[4]:
    data = load_frame("decision_funnel")
    if len(data):
        st.plotly_chart(bars(data, "gate", "total_passed", None, "Decision funnel throughput"), use_container_width=True)
        enterprise_table(data, key="research_funnel")
    else: empty_state("Decision funnel unavailable", "Funnel research is missing.")
with tabs[5]:
    data = load_frame("readiness")
    if len(data):
        st.plotly_chart(bars(data, "dataset_version", "production_readiness_score", "classification", "Production readiness score"), use_container_width=True)
        enterprise_table(data, key="research_readiness")
    else: empty_state("Readiness unavailable", "Final readiness research is missing.")
app_footer()

