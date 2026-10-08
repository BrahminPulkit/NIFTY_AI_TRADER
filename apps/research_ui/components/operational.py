"""Operational frontend widgets backed exclusively by frozen artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from apps.research_ui.components.data import artifact_health
from apps.research_ui.components.theme import empty_state, metric_card


def market_overview(snapshot: dict | None = None) -> None:
    """Live-ready market overview without fabricated values."""
    snapshot = snapshot or {}
    cols = st.columns(4)
    for col, market, icon in zip(
        cols, ("NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"), ("N", "B", "S", "V")):
        with col:
            metric_card(market, "AWAITING FEED", "Approved live backend required",
                        "neutral", icon)
    cols = st.columns(6)
    context = [
        ("Market Breadth", "—", "Breadth feed required", "▦"),
        ("Advance / Decline", "—", "Constituent feed required", "↕"),
        ("Volatility", snapshot.get("volatility_regime", "—"), "Frozen context", "≋"),
        ("Trend", str(snapshot.get("trend_regime", "—")).replace("_", " "), "Frozen context", "⌁"),
        ("Session", str(snapshot.get("session_phase", "—")).replace("_", " "), "Market phase", "◷"),
        ("AI Bias", snapshot.get("recommendation", "—"), "Frozen decision", "✦"),
    ]
    for col, item in zip(cols, context):
        with col: metric_card(*item, "neutral")


def alert_items(snapshot: dict | None = None) -> list[dict]:
    snapshot = snapshot or {}
    health = artifact_health()
    alerts = []
    missing = health.loc[health.status.ne("AVAILABLE"), "artifact"].tolist()
    if missing:
        alerts.append({"severity": "CRITICAL", "title": "Artifact unavailable",
                       "message": ", ".join(missing[:4])})
    else:
        alerts.append({"severity": "OK", "title": "Artifact registry healthy",
                       "message": f"All {len(health)} registered frontend sources are available."})
    timestamp = snapshot.get("timestamp")
    if timestamp:
        age = datetime.now(timezone.utc) - pd.to_datetime(timestamp, utc=True).to_pydatetime()
        alerts.append({"severity": "INFO", "title": "Historical research snapshot",
                       "message": f"Latest frozen record is {age.days:,} days old; it is not live data."})
    if snapshot.get("recommendation") == "SKIP":
        alerts.append({"severity": "INFO", "title": "Prediction skipped",
                       "message": str(snapshot.get("decision_reason", "Decision Engine rejection")).replace("_", " ")})
    alerts.append({"severity": "WARNING", "title": "Unseen live inference unavailable",
                   "message": "Decision Engine live state has not been approved for production."})
    return alerts


def alert_centre(snapshot: dict | None = None) -> None:
    alerts = alert_items(snapshot)
    rows = "".join(
        f"""<div class="alert-row {item['severity'].lower()}"><i></i><div>
        <b>{escape(item['title'])}</b><span>{escape(item['message'])}</span></div>
        <em>{escape(item['severity'])}</em></div>""" for item in alerts)
    st.markdown(f'<div class="alert-centre">{rows}</div>', unsafe_allow_html=True)


def decision_explanation(row: dict, feature_importance: pd.DataFrame | None = None) -> None:
    decision = str(row.get("recommendation", row.get("decision", "SKIP")))
    reason = str(row.get("decision_reason", row.get("reason", "Unavailable"))).replace("_", " ")
    passed = []
    failed = []
    for label, field, expected in (
        ("Strong setup found", "entry_signal", lambda value: float(value) != 0),
        ("Market conditions confirmed", "recommendation", lambda value: str(value) == "TRADE"),
        ("Confidence requirement met", "catboost_predicted_probability", lambda value: float(value) >= .9),
    ):
        try:
            (passed if expected(row.get(field, 0)) else failed).append(label)
        except (TypeError, ValueError):
            failed.append(f"{label} (unavailable)")
    st.markdown(
        f"""<div class="explain-card"><header><span>{escape(decision)}</span>
        <b>{escape(reason)}</b></header><div class="rule-columns">
        <section><label>WHAT LOOKS STRONG</label>{''.join(f'<p>Confirmed · {escape(x)}</p>' for x in passed) or '<p>— None recorded</p>'}</section>
        <section><label>WHAT NEEDS CONFIRMATION</label>{''.join(f'<p>Waiting · {escape(x)}</p>' for x in failed) or '<p>— None</p>'}</section>
        </div><footer>Confidence reflects the frozen model probability and is not a guarantee of outcome.</footer></div>""",
        unsafe_allow_html=True)
    if feature_importance is not None and len(feature_importance):
        shown = feature_importance.loc[
            feature_importance.model.eq("CATBOOST")].nlargest(8, "importance")
        st.caption("Global CatBoost feature contribution (frozen importance; row-level SHAP is unavailable)")
        st.dataframe(shown[["transformed_feature", "importance"]],
                     use_container_width=True, hide_index=True)


DEFAULT_WIDGETS = [
    "Market Overview", "AI Decision", "Market Intelligence", "Strategy Leaderboard",
    "Confidence Timeline", "Alert Centre", "Recent Trades",
]


def personalisation_controls() -> list[str]:
    if "dashboard_widgets" not in st.session_state:
        st.session_state.dashboard_widgets = DEFAULT_WIDGETS.copy()
    with st.expander("Customise workspace", icon="⚙️"):
        selected = st.multiselect(
            "Visible widgets — selection order controls dashboard order",
            DEFAULT_WIDGETS, default=st.session_state.dashboard_widgets,
            key="widget_selection")
        c1, c2 = st.columns(2)
        if c1.button("Save layout", use_container_width=True):
            st.session_state.dashboard_widgets = selected
            st.toast("Workspace layout saved for this session")
        if c2.button("Restore default", use_container_width=True):
            st.session_state.dashboard_widgets = DEFAULT_WIDGETS.copy()
            st.rerun()
    return st.session_state.dashboard_widgets
