"""QuantEdge AI professional design system.

Presentation only: this module never imports or invokes trading, feature,
prediction, model, or execution code.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import json

import streamlit as st

from apps.research_ui.components.data import PROJECT_ROOT, artifact_health
from src.dhan_connection_manager import get_dhan_manager


THEMES = {
    "Institutional Blue": {
        "accent": "#4F8CFF", "accent_hover": "#73A4FF",
        "accent_soft": "rgba(79,140,255,.14)", "accent_rgb": "79,140,255",
    },
    "Emerald": {
        "accent": "#00C853", "accent_hover": "#35D875",
        "accent_soft": "rgba(0,200,83,.14)", "accent_rgb": "0,200,83",
    },
    "Royal Purple": {
        "accent": "#8B5CF6", "accent_hover": "#A78BFA",
        "accent_soft": "rgba(139,92,246,.14)", "accent_rgb": "139,92,246",
    },
    "Graphite": {
        "accent": "#94A3B8", "accent_hover": "#CBD5E1",
        "accent_soft": "rgba(148,163,184,.14)", "accent_rgb": "148,163,184",
    },
    "Crimson": {
        "accent": "#EF4444", "accent_hover": "#F87171",
        "accent_soft": "rgba(239,68,68,.14)", "accent_rgb": "239,68,68",
    },
}

DEFAULT_PREFERENCES = {
    "theme": "Institutional Blue",
    "font_size": "Comfortable",
    "compact_mode": False,
    "animations": True,
    "chart_density": "Balanced",
    "sidebar_width": "Standard",
    "card_spacing": "Comfortable",
}

NAVIGATION = (
    ("Trade", (
        ("layout-dashboard", "Trading Desk", "app.py"),
        ("activity", "Live Market", "pages/3_Live_Market.py"),
        ("chart-candlestick", "Option Chain", "pages/4_Options_Analytics.py"),
        ("wallet-cards", "Paper Trading", "pages/12_Paper_Trading.py"),
        ("chart-no-axes-combined", "Performance", "pages/8_Performance.py"),
    )),
    ("Intelligence", (
        ("history", "Market Replay", "pages/11_AI_Market_Replay.py"),
        ("brain-circuit", "Prediction Centre", "pages/2_Prediction_Panel.py"),
        ("layers", "Strategy Centre", "pages/1_Strategy_Monitor.py"),
    )),
    ("Advanced", (
        ("flask-conical", "Research Analytics", "pages/4_Research_Analytics.py"),
        ("notebook-tabs", "Trade Journal", "pages/5_Trade_Journal.py"),
        ("scan-search", "Feature Explorer", "pages/6_Feature_Explorer.py"),
        ("git-compare-arrows", "Model Comparison", "pages/7_Model_Comparison.py"),
        ("heart-pulse", "Monitoring", "pages/13_System_Monitoring.py"),
    )),
    ("System", (
        ("shield-check", "Broker", "pages/14_Broker_Connection.py"),
        ("settings", "Settings", "pages/9_Settings.py"),
        ("circle-help", "About", "pages/10_About.py"),
    )),
)


_ICON_PATHS = {
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "bell": '<path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/>',
    "brain-circuit": '<path d="M9.5 4.5A3 3 0 0 0 4 6v1a3 3 0 0 0-1 5.8V14a3 3 0 0 0 3 3h1"/><path d="M14.5 4.5A3 3 0 0 1 20 6v1a3 3 0 0 1 1 5.8V14a3 3 0 0 1-3 3h-1"/><path d="M12 4v16M8 9h4m0 6h4"/>',
    "chart-candlestick": '<path d="M9 5v4m0 4v6M5 9h8v4H5zM17 3v3m0 4v8m-4-12h8v4h-8z"/>',
    "chart-no-axes-combined": '<path d="m3 17 6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "circle-help": '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 1 1 5.8 1c0 2-3 2-3 4m.1 4h.01"/>',
    "flask-conical": '<path d="M10 2v7L4 19a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3L14 9V2M8.5 2h7M7 16h10"/>',
    "git-compare-arrows": '<path d="m5 3-3 3 3 3M2 6h13a4 4 0 0 1 4 4v1m0 4 3 3-3 3m3-3H9a4 4 0 0 1-4-4v-1"/>',
    "heart-pulse": '<path d="M19 14c1.5-1.5 3-3.2 3-5.5A5.5 5.5 0 0 0 12 5a5.5 5.5 0 0 0-10 3.5C2 12 5 15 12 21l2-2"/><path d="M3 12h5l1-3 3 7 2-4h7"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5m4-2v6l4 2"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
    "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "notebook-tabs": '<path d="M2 6h4M2 10h4M2 14h4M2 18h4"/><rect width="16" height="20" x="4" y="2" rx="2"/>',
    "scan-search": '<path d="M3 7V5a2 2 0 0 1 2-2h2m10 0h2a2 2 0 0 1 2 2v2M3 17v2a2 2 0 0 0 2 2h2"/><circle cx="12" cy="12" r="3"/><path d="m14 14 4 4"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "settings": '<path d="M12.2 2h-.4a2 2 0 0 0-2 2v.2a2 2 0 0 1-1 1.7l-.4.2a2 2 0 0 1-2 0l-.2-.1a2 2 0 0 0-2.7.7l-.2.4a2 2 0 0 0 .7 2.7l.2.1a2 2 0 0 1 1 1.7v.5a2 2 0 0 1-1 1.7l-.2.1a2 2 0 0 0-.7 2.7l.2.4a2 2 0 0 0 2.7.7l.2-.1a2 2 0 0 1 2 0l.4.2a2 2 0 0 1 1 1.7v.2a2 2 0 0 0 2 2h.4a2 2 0 0 0 2-2v-.2a2 2 0 0 1 1-1.7l.4-.2a2 2 0 0 1 2 0l.2.1a2 2 0 0 0 2.7-.7l.2-.4a2 2 0 0 0-.7-2.7l-.2-.1a2 2 0 0 1-1-1.7v-.5a2 2 0 0 1 1-1.7l.2-.1a2 2 0 0 0 .7-2.7l-.2-.4a2 2 0 0 0-2.7-.7l-.2.1a2 2 0 0 1-2 0l-.4-.2a2 2 0 0 1-1-1.7V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-8 9-4.5-1.5-8-4-8-9V5l8-3 8 3v8z"/><path d="m9 12 2 2 4-4"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "wallet-cards": '<rect width="18" height="14" x="3" y="5" rx="2"/><path d="M16 13h2M3 10h18"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.8-1.6l9-11A.5.5 0 0 1 13 2v7h7a1 1 0 0 1 .8 1.6l-9 11A.5.5 0 0 1 11 21v-7z"/>',
}


def icon(name: str, size: int = 18, css_class: str = "") -> str:
    paths = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    return (
        f'<svg class="lucide {escape(css_class)}" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


def preferences() -> dict:
    for key, value in DEFAULT_PREFERENCES.items():
        st.session_state.setdefault(f"ui_{key}", value)
    return {key: st.session_state[f"ui_{key}"] for key in DEFAULT_PREFERENCES}


def active_theme() -> dict:
    selected = preferences()["theme"]
    return THEMES.get(selected, THEMES["Institutional Blue"])


def chart_palette() -> dict:
    theme = active_theme()
    return {
        "accent": theme["accent"], "accent_hover": theme["accent_hover"],
        "bull": "#2ECC71", "bear": "#FF5A5F", "neutral": "#F5B041",
        "info": "#4F8CFF", "probability": "#9B7BFF",
        "ema20": "#F5B041", "ema50": "#4F8CFF", "vwap": "#9B7BFF",
        "muted": "#6B7280", "panel": "#1C2940",
    }


def _token_css() -> str:
    pref, theme = preferences(), active_theme()
    font = {"Compact": ".92", "Comfortable": "1", "Large": "1.08"}[pref["font_size"]]
    sidebar = {"Narrow": "232px", "Standard": "268px", "Wide": "304px"}[pref["sidebar_width"]]
    spacing = {"Tight": ".75rem", "Comfortable": "1rem", "Airy": "1.3rem"}[pref["card_spacing"]]
    motion = "180ms" if pref["animations"] else "0ms"
    density = "116px" if pref["compact_mode"] else "132px"
    return f"""
    :root {{
      --accent:{theme['accent']};--accent-hover:{theme['accent_hover']};
      --accent-soft:{theme['accent_soft']};--accent-rgb:{theme['accent_rgb']};
      --font-scale:{font};--sidebar-width:{sidebar};--card-gap:{spacing};
      --motion:{motion};--metric-min-height:{density};
    }}
    """


def apply_theme() -> None:
    css = Path(__file__).resolve().parents[1] / "assets" / "institutional.css"
    stylesheet = css.read_text(encoding="utf-8-sig")
    st.markdown(
        f"<style>{stylesheet}{_token_css()}</style>",
        unsafe_allow_html=True)


def _page_url(path: str) -> str:
    if path == "app.py":
        return "/"
    return "/" + Path(path).stem.split("_", 1)[-1]


def _page_nav_link(
    path: str, label: str, icon_name: str | None = None, key_suffix: str = "",
) -> None:
    """Use a Streamlit rerun, preserving the process-wide broker manager."""
    del icon_name
    key = (
        "_nav_" + path.replace("/", "_").replace(".", "_")
        + "_" + label.lower().replace(" ", "_") + key_suffix)
    if st.button(label, key=key, use_container_width=True):
        st.switch_page(path)


def _sidebar(current_page: str) -> None:
    with st.sidebar:
        st.markdown(
            f"""<div class="brand-v2"><div class="brand-mark">Q</div>
            <div><strong>QuantEdge <em>AI</em></strong>
            <span>TRADING INTELLIGENCE</span></div></div>""",
            unsafe_allow_html=True)
        selected = st.selectbox(
            "Theme", list(THEMES), key="ui_theme", label_visibility="collapsed")
        st.markdown(
            f'<div class="sidebar-theme">{icon("zap",14)}<span>{escape(selected)}</span>'
            '<b>ACTIVE</b></div>', unsafe_allow_html=True)
        st.session_state.setdefault("ui_experience", "Simple")
        st.segmented_control(
            "Experience", ["Simple", "Advanced"], key="ui_experience",
            help="Simple shows the essential trading workflow. Advanced keeps research tools visible.")
        query = st.text_input(
            "Search navigation", key="_nav_search", placeholder="Search workspace…",
            label_visibility="collapsed")
        query = query.strip().lower()
        simple_pages = {
            "Trading Desk", "Live Market", "Paper Trading", "Option Chain",
            "Performance", "Broker", "Settings", "About",
        }
        for section, items in NAVIGATION:
            if st.session_state.ui_experience == "Simple" and section == "Advanced":
                continue
            visible = [item for item in items if not query or query in item[1].lower()]
            if st.session_state.ui_experience == "Simple":
                visible = [item for item in visible if item[1] in simple_pages]
            if not visible:
                continue
            st.markdown(f'<div class="nav-group">{escape(section)}</div>',
                        unsafe_allow_html=True)
            for icon_name, label, path in visible:
                _page_nav_link(path, label)
        st.markdown('<div class="nav-group">Quick actions</div>', unsafe_allow_html=True)
        quick_left, quick_right = st.columns(2)
        with quick_left:
            _page_nav_link(
                "pages/14_Broker_Connection.py", label="Connect broker",
                icon_name=":material/shield:", key_suffix="_quick")
        with quick_right:
            _page_nav_link(
                "app.py", label="Trading desk",
                icon_name=":material/monitoring:", key_suffix="_quick")
        health = artifact_health()
        available = int(health.status.eq("AVAILABLE").sum())
        st.markdown(
            f"""<div class="profile-dock">
              <div class="profile-avatar">{icon("user",17)}</div>
              <div><strong>QuantEdge Trader</strong><span>Research & paper mode</span></div>
              <b>{available}/{len(health)}</b>
            </div>""", unsafe_allow_html=True)


def configure_page(title: str, icon_name: str = "Q") -> None:
    st.set_page_config(
        page_title=f"{title} | QuantEdge AI", page_icon="Q",
        layout="wide", initial_sidebar_state="auto")
    preferences()
    apply_theme()
    _sidebar(title)


def _header_state() -> tuple[str, str]:
    connected = get_dhan_manager().snapshot()["connected"]
    metadata_path = PROJECT_ROOT / "production_model/production_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return ("Connected" if connected else "Offline",
            metadata.get("model_version", "Model unavailable"))


def page_header(title: str, eyebrow: str, description: str) -> None:
    connection, model = _header_state()
    clock = datetime.now().astimezone().strftime("%H:%M IST")
    status_class = "connected" if connection == "Connected" else "offline"
    st.markdown(
        f"""<header class="terminal-header">
          <div class="page-identity"><span>{escape(eyebrow)}</span><h1>{escape(title)}</h1>
          <p>{escape(description)}</p></div>
          <div class="header-tools">
            <div class="context-chip"><span>Market</span><b>NIFTY</b></div>
            <div class="context-chip {status_class}">{icon("activity",14)}
              <span>Connection</span><b>{escape(connection)}</b></div>
            <div class="context-chip">{icon("brain-circuit",14)}
              <span>AI status</span><b>Ready</b></div>
            <div class="model-chip"><span>Model</span><b>{escape(str(model))}</b></div>
            <div class="theme-chip">{escape(preferences()['theme'])}</div>
            <div class="clock">{escape(clock)}</div>
            <button class="icon-button" aria-label="Notifications">{icon("bell",17)}</button>
            <div class="user-v2">{icon("user",17)}</div>
          </div>
        </header>""", unsafe_allow_html=True)


def hero(signal: str, probability: float, regime: str, confidence: float) -> None:
    tone = "positive" if signal == "TRADE" else "cautious"
    readable = "Trade opportunity" if signal == "TRADE" else "Wait for confirmation"
    st.markdown(
        f"""<section class="hero-v2">
          <div class="hero-copy"><div class="hero-kicker">LIVE MARKET INTELLIGENCE</div>
          <h2>Clear signals.<br><span>Confident decisions.</span></h2>
          <p>AI-assisted market research built for disciplined traders.</p>
          <div class="hero-tags"><span>CAUSAL DATA</span><span>VERIFIED MODEL</span><span>PAPER MODE</span></div></div>
          <div class="signal-console {tone}"><div class="console-label">{icon("activity",15)}
          CURRENT GUIDANCE</div><strong>{escape(readable)}</strong>
          <div class="signal-score">{probability:.1%}<small>MODEL CONFIDENCE</small></div>
          <div class="console-bars"><span style="width:{min(max(probability,0),1)*100:.1f}%"></span></div>
          <div class="console-meta"><span>SETUP QUALITY <b>{confidence:.1%}</b></span>
          <span>MARKET STATE <b>{escape(regime)}</b></span></div></div>
        </section>""", unsafe_allow_html=True)


def metric_card(label: str, value, detail: str = "", tone: str = "neutral",
                icon_name: str = "activity", delta: str = "") -> None:
    shown = "—" if value is None or str(value) in {"nan", "<NA>", "N/A"} else str(value)
    aliases = {"good": "positive", "bad": "negative", "warn": "warning"}
    tone = aliases.get(tone, tone)
    st.markdown(
        f"""<article class="data-tile {escape(tone)}"><div class="tile-top">
          <span class="tile-icon">{icon(icon_name if icon_name in _ICON_PATHS else "activity",16)}</span>
          <label>{escape(label)}</label></div><strong>{escape(shown)}</strong>
          <footer><span>{escape(str(detail))}</span><b>{escape(str(delta))}</b></footer></article>""",
        unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", action: str = "") -> None:
    st.markdown(
        f"""<div class="section-v2"><div><h2>{escape(title)}</h2>
        <p>{escape(subtitle)}</p></div><b>{escape(action)}</b></div>""",
        unsafe_allow_html=True)


def status_banner(message: str, tone: str = "info", title: str = "System notice") -> None:
    st.markdown(
        f"""<div class="notice-v2 {escape(tone)}">{icon("shield-check",18)}
        <div><b>{escape(title)}</b><p>{escape(message)}</p></div></div>""",
        unsafe_allow_html=True)


def ai_thinking(decision: str) -> None:
    checks = (
        "Reading market structure", "Checking momentum", "Checking option activity",
        "Measuring volatility", "Understanding market state", "Assessing confidence")
    rows = "".join(
        f'<div class="thought">{icon("shield-check",14)}<span>{escape(item)}</span>'
        '<b>CONFIRMED</b></div>' for item in checks)
    readable = "Trade opportunity" if decision == "TRADE" else "Wait for confirmation"
    st.markdown(
        f"""<div class="thinking-panel"><div class="thinking-head">
        <div class="ai-orb">{icon("brain-circuit",22)}</div>
        <div><span>QUANTEDGE ANALYSIS</span><h3>How the AI reached this view</h3></div>
        <em>COMPLETE</em></div><div class="thoughts">{rows}</div>
        <div class="thinking-result"><span>CURRENT GUIDANCE</span>
        <strong>{escape(readable)}</strong></div></div>""", unsafe_allow_html=True)


def timeline(items: list[dict]) -> None:
    if not items:
        empty_state("No predictions recorded", "Predictions will appear here as they are recorded.")
        return
    content = []
    for item in items:
        content.append(
            f"""<div class="timeline-item"><div class="timeline-dot"></div>
            <time>{escape(str(item.get('time','—')))}</time>
            <div><strong>{escape(str(item.get('signal','WAIT')))}</strong>
            <span>{escape(str(item.get('strategy','Unknown')))}</span></div>
            <b>{escape(str(item.get('probability','—')))}</b>
            <em>{escape(str(item.get('reason','Decision recorded')))}</em></div>""")
    st.markdown(f'<div class="timeline-v2">{"".join(content)}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str, icon_name: str = "activity") -> None:
    st.markdown(
        f"""<div class="empty-v2">{icon(icon_name if icon_name in _ICON_PATHS else "activity",28)}
        <strong>{escape(title)}</strong><p>{escape(message)}</p>
        <span>WE WILL UPDATE THIS VIEW AUTOMATICALLY</span></div>""", unsafe_allow_html=True)


def panel(title: str, content: str, badge: str = "") -> None:
    st.markdown(
        f"""<section class="glass-panel"><header><span>{escape(title)}</span>
        <b>{escape(badge)}</b></header><div>{content}</div></section>""",
        unsafe_allow_html=True)


def unavailable(message: str) -> None:
    empty_state("Information unavailable", message)


def app_footer() -> None:
    st.markdown(
        f"""<footer class="footer-v2"><div><div class="footer-mark">Q</div>
        <span><b>QuantEdge AI</b>Institutional AI Trading Platform</span></div>
        <p>Research and paper trading only. Markets involve risk.</p>
        <aside><b>v1.0.0</b><span>{icon("shield-check",13)} SECURE RESEARCH MODE</span></aside>
        </footer>""", unsafe_allow_html=True)
