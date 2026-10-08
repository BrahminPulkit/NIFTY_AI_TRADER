"""Frontend-only appearance preferences."""

import streamlit as st

from apps.research_ui.components.data import artifact_health
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import (
    DEFAULT_PREFERENCES, THEMES, app_footer, configure_page, metric_card,
    page_header, preferences, section_header, status_banner,
)


configure_page("Settings")
page_header(
    "Settings", "PERSONAL WORKSPACE",
    "Tailor the interface to your screen and trading style.")
status_banner(
    "Appearance preferences are session-only and never alter models, signals, "
    "data, thresholds, or trading rules.", "info", "SAFE PERSONALISATION")

current = preferences()
cols = st.columns(4)
with cols[0]:
    metric_card("Theme", current["theme"], "Applied instantly", "positive", "settings")
with cols[1]:
    metric_card("Text Size", current["font_size"], "Interface scale", "neutral", "scan-search")
with cols[2]:
    metric_card("Layout", "COMPACT" if current["compact_mode"] else "COMFORTABLE",
                "Card density", "neutral", "layout-dashboard")
with cols[3]:
    metric_card("Motion", "ON" if current["animations"] else "OFF",
                "Subtle transitions", "neutral", "activity")

section_header("Appearance", "Changes apply immediately across every page")
c1, c2 = st.columns(2)


def sync_theme():
    st.session_state.ui_theme = st.session_state.settings_theme


with c1:
    st.selectbox(
        "Colour theme", list(THEMES),
        index=list(THEMES).index(current["theme"]),
        key="settings_theme", on_change=sync_theme,
        help="Controls buttons, cards, charts, badges and navigation.")
    st.selectbox(
        "Text size", ["Compact", "Comfortable", "Large"],
        index=["Compact", "Comfortable", "Large"].index(current["font_size"]),
        key="ui_font_size")
    st.selectbox(
        "Sidebar width", ["Narrow", "Standard", "Wide"],
        index=["Narrow", "Standard", "Wide"].index(current["sidebar_width"]),
        key="ui_sidebar_width")
with c2:
    st.selectbox(
        "Chart density", ["Compact", "Balanced", "Detailed"],
        index=["Compact", "Balanced", "Detailed"].index(current["chart_density"]),
        key="ui_chart_density")
    st.selectbox(
        "Card spacing", ["Tight", "Comfortable", "Airy"],
        index=["Tight", "Comfortable", "Airy"].index(current["card_spacing"]),
        key="ui_card_spacing")
    st.toggle("Compact cards", value=current["compact_mode"], key="ui_compact_mode")
    st.toggle("Subtle animations", value=current["animations"], key="ui_animations")

if st.button("Restore default appearance", use_container_width=True):
    for key, value in DEFAULT_PREFERENCES.items():
        st.session_state[f"ui_{key}"] = value
    st.session_state.settings_theme = DEFAULT_PREFERENCES["theme"]
    st.rerun()

section_header("Workspace Health", "Read-only availability of supporting information")
enterprise_table(artifact_health(), key="settings_health", page_size=20)
app_footer()
