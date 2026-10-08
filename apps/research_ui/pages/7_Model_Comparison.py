import streamlit as st

from apps.research_ui.components.charts import bars, confusion_matrix
from apps.research_ui.components.data import load_frame, load_text
from apps.research_ui.components.theme import configure_page, metric_card, page_header

configure_page("Model Comparison")
page_header("Model Comparison", "V1 · V2 FINAL VALIDATION", "Direct predictive, stability, risk, and readiness comparison from Step 20L.")

comparison = load_frame("model_comparison")
readiness = load_frame("readiness")
recommendation = load_text("deployment_recommendation")

if len(comparison):
    c1, c2 = st.columns(2)
    for column, version in zip((c1, c2), ("V1", "V2")):
        row = comparison.loc[comparison.dataset_version.eq(version)].iloc[0]
        with column:
            metric_card(f"{version} ROC-AUC", f"{row.roc_auc:.4f}", f"PR-AUC {row.pr_auc:.4f}")
            metric_card(f"{version} Feature PSI", f"{row.median_feature_psi:.4f}", f"Probability PSI {row.probability_psi:.4f}", "good" if row.median_feature_psi < .1 else "warn")
    metric_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy"]
    melted = comparison.melt(id_vars=["dataset_version"], value_vars=metric_columns, var_name="metric", value_name="value")
    st.plotly_chart(bars(melted, "metric", "value", "dataset_version", "V1 versus V2 predictive metrics"), use_container_width=True)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

if len(readiness):
    st.plotly_chart(bars(readiness, "dataset_version", "production_readiness_score", "classification", "Readiness score"), use_container_width=True)
st.subheader("Final recommendation")
st.markdown(recommendation or "**Production Dataset V1** · Research Ready · Live deployment not approved.")

