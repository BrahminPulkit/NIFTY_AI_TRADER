import streamlit as st
from apps.research_ui.components.theme import app_footer, configure_page, metric_card, page_header, section_header, panel

configure_page("About")
page_header("About QuantEdge AI", "PRODUCT IDENTITY",
            "Institutional AI Trading Research Platform for rigorous, leakage-safe market research.")
cols = st.columns(3)
with cols[0]: metric_card("Platform", "QuantEdge AI", "Commercial research interface", "good", "Q")
with cols[1]: metric_card("Version", "1.0 RESEARCH", "Frontend release", "neutral", "v")
with cols[2]: metric_card("Safety", "READ ONLY", "Execution permanently absent", "good", "✓")
section_header("Platform principles")
panel("Research integrity",
      "Frozen artifacts · chronological validation · causal features · deterministic replay · explicit unavailable states",
      "INSTITUTIONAL")
panel("Risk disclosure",
      "This platform visualises historical research. It does not provide investment advice, place orders, or guarantee future performance.",
      "IMPORTANT")
app_footer()
