import plotly.express as px
import streamlit as st

from apps.research_ui.components.charts import bars, style
from apps.research_ui.components.data import load_frame
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import app_footer, configure_page, empty_state, page_header, section_header

configure_page("Feature Explorer")
page_header("Feature Explorer", "MODEL TRANSPARENCY",
            "Search, compare and inspect model importance, SHAP attribution, drift and correlation.")
importance, shap, drift = load_frame("feature_importance"), load_frame("shap"), load_frame("feature_drift")
tabs = st.tabs(["Feature Catalogue", "Importance", "SHAP", "Drift", "Correlation"])
with tabs[0]:
    frames = []
    if len(importance): frames.append(importance.rename(columns={"transformed_feature":"feature"}))
    catalogue = frames[0] if frames else importance
    enterprise_table(catalogue, key="feature_catalogue", page_size=30)
with tabs[1]:
    if len(importance):
        model = st.selectbox("Model", sorted(importance.model.unique()), key="v2_imp_model")
        shown = importance.loc[importance.model.eq(model)].nlargest(30, "importance")
        st.plotly_chart(bars(shown.sort_values("importance"), "transformed_feature", "importance", None, f"{model} importance"), use_container_width=True)
        enterprise_table(shown, key="feature_imp_v2")
    else: empty_state("Importance unavailable", "Frozen feature importance is missing.")
with tabs[2]:
    if len(shap):
        model = st.selectbox("Model", sorted(shap.model.unique()), key="v2_shap_model")
        shown = shap.loc[shap.model.eq(model)].nlargest(30, "mean_absolute_shap")
        st.plotly_chart(bars(shown.sort_values("mean_absolute_shap"), "transformed_feature", "mean_absolute_shap", None, f"{model} SHAP impact"), use_container_width=True)
    else: empty_state("SHAP unavailable", "Frozen SHAP summary is missing.")
with tabs[3]:
    if len(drift):
        st.plotly_chart(bars(drift.nlargest(35, "original_psi"), "feature", "original_psi", "classification", "Feature drift profile"), use_container_width=True)
        enterprise_table(drift, key="feature_drift_v2", page_size=30)
    else: empty_state("Drift unavailable", "Feature PSI research is missing.")
with tabs[4]:
    correlation = load_frame("correlation")
    if len(correlation):
        key = correlation.columns[0]
        matrix = correlation.set_index(key)
        top = matrix.abs().mean().nlargest(35).index
        fig = px.imshow(matrix.loc[top, top], color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Top feature correlation matrix")
        st.plotly_chart(style(fig, 750), use_container_width=True)
    else: empty_state("Correlation unavailable", "Frozen correlation matrix is missing.")
app_footer()
