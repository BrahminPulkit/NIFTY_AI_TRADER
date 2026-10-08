"""QuantEdge AI commercial Trading Desk.

Presentation only. This page reads existing artifacts and session state; it
never invokes models, places orders, or mutates trading configuration.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from apps.research_ui.components.data import latest_snapshot, load_frame
from apps.research_ui.components.theme import app_footer, configure_page, icon, page_header
from apps.research_ui.components.trader_ui import (
    data_freshness, safe_action, strategy_guide, trade_direction,
)
from src.dhan_connection_manager import get_dhan_manager


def safe_csv(path: str) -> pd.DataFrame:
    source = Path(path)
    return pd.read_csv(source) if source.exists() else pd.DataFrame()


def readable(value: object, fallback: str = "Unavailable") -> str:
    if value is None or str(value) in {"", "nan", "<NA>", "None"}:
        return fallback
    return str(value).replace("_", " ").replace("|", " · ").title()


def desk_metric(label: str, value: str, detail: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""<div class="desk-metric {escape(tone)}">
        <span>{escape(label)}</span><strong>{escape(value)}</strong>
        <small>{escape(detail)}</small></div>""", unsafe_allow_html=True)


def load_frontend_snapshot() -> tuple[dict, bool, str]:
    snapshot = latest_snapshot()
    journal = Path("logs/live_inference/predictions.jsonl")
    if not journal.exists() or not journal.stat().st_size:
        return snapshot, False, "Frozen research record"
    try:
        record = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    except (json.JSONDecodeError, OSError, IndexError):
        return snapshot, False, "Frozen research record"
    return {
        **snapshot,
        "timestamp": record.get("timestamp"),
        "catboost_predicted_probability": record.get("probability") or 0,
        "strategy_confidence_score": record.get("confidence") or 0,
        "recommendation": record.get("decision", "SKIP"),
        "decision_reason": record.get("reason", "UNAVAILABLE"),
        "current_strategy": record.get("strategy", "UNAVAILABLE"),
        "market_state": record.get("regime", "UNAVAILABLE"),
        "live_model_version": record.get("model_version"),
        "live_latency_ms": record.get("latency_ms"),
        "risk_status": record.get("risk_status"),
    }, True, "Native inference journal"


configure_page("Trading Desk")
page_header(
    "Trading Desk", "NIFTY AI OPERATIONS",
    "One decision surface for market context, AI guidance, risk and next action.")

snapshot, is_live_record, source_label = load_frontend_snapshot()
if not snapshot:
    st.markdown(
        f"""<section class="desk-empty">{icon("activity",26)}
        <div><span>TRADING DESK WAITING</span><h2>No verified prediction is available</h2>
        <p>The platform will remain inactive until a valid record is present.</p></div>
        </section>""", unsafe_allow_html=True)
    app_footer()
    st.stop()

probability = float(snapshot.get("catboost_predicted_probability", 0) or 0)
confidence = float(snapshot.get("strategy_confidence_score", probability) or 0)
action, action_tone, action_reason = safe_action(snapshot)
direction = trade_direction(snapshot)
guide = strategy_guide(snapshot.get("current_strategy"))
freshness = data_freshness(snapshot.get("timestamp"), is_live_record)
decision_reason = readable(snapshot.get("decision_reason"), "Decision unavailable")
regime = readable(
    snapshot.get("market_state", snapshot.get("market_regime")),
    "Market state unavailable")
holding = float(snapshot.get("expected_holding_minutes", 0) or 0)
drawdown = abs(float(snapshot.get("expected_drawdown", 0) or 0))
expansion = float(snapshot.get("expected_premium_expansion", 0) or 0)

market_cache = get_dhan_manager().snapshot()
connected = market_cache["connected"]
quotes = market_cache["quotes"]
options = market_cache["options"]
contract_key = (
    "NIFTY_ATM_PE" if direction == "BEARISH"
    else "NIFTY_ATM_CE" if direction == "BULLISH" else "")
contract = options.get(contract_key) if contract_key else None
contract_name = (
    f"NIFTY {float(contract['strike']):,.0f} {contract['option_type']}"
    if contract and contract.get("strike") is not None else "Awaiting verified contract")
expiry = str(contract["expiry"]) if contract and contract.get("expiry") else "Unavailable"

mode_tone = "positive" if freshness["label"] == "LIVE" else "warning"
st.markdown(
    f"""<div class="desk-status {mode_tone}">
      <div><i></i><strong>{escape(str(freshness['label']))}</strong>
      <span>{escape(source_label)} · {escape(str(freshness['age']))}</span></div>
      <div><span>Broker</span><b>{'CONNECTED' if connected else 'OFFLINE'}</b></div>
      <div><span>Last record</span><b>{pd.to_datetime(snapshot['timestamp']).strftime('%d %b · %H:%M')}</b></div>
    </div>""", unsafe_allow_html=True)

# Primary decision surface.
st.markdown(
    f"""<section class="trade-command {escape(action_tone)}">
      <div class="command-main">
        <div class="command-kicker"><i></i>AI RECOMMENDATION</div>
        <h2>{escape(action)}</h2>
        <p>{escape(action_reason)}</p>
        <div class="command-context">
          <span>Strategy<b>{escape(guide['name'])}</b></span>
          <span>Market regime<b>{escape(regime)}</b></span>
          <span>Expected hold<b>{holding:.0f} minutes</b></span>
        </div>
      </div>
      <div class="confidence-orbit" style="--confidence:{probability * 360:.1f}deg">
        <div><strong>{probability:.0%}</strong><span>Confidence</span></div>
      </div>
      <div class="command-contract">
        <div class="contract-head"><span>SUGGESTED CONTRACT</span>
        <b>{'VERIFIED' if contract else 'LOCKED'}</b></div>
        <strong>{escape(contract_name)}</strong>
        <div><span>Expiry<b>{escape(expiry)}</b></span>
        <span>Strike<b>{f"{float(contract['strike']):,.0f}" if contract and contract.get('strike') is not None else '—'}</b></span></div>
        <div><span>Entry<b>—</b></span><span>Stop<b>—</b></span>
        <span>Target<b>—</b></span><span>R:R<b>—</b></span></div>
        <a class="paper-action {'disabled' if action == 'NO TRADE' or not contract else ''}"
           href="/Paper_Trading" target="_self">{icon("wallet-cards",17)}
           {'Open paper desk' if action != 'NO TRADE' and contract else 'Paper trade unavailable'}</a>
      </div>
    </section>""", unsafe_allow_html=True)

# Decision explanation: facts only.
checklist = [
    ("Trend direction", direction != "UNKNOWN", readable(snapshot.get("trend_regime"), "Unavailable")),
    ("Strategy selected", str(snapshot.get("current_strategy", "")).upper() in {
        "PULLBACK_BREAKOUT", "MOMENTUM_BREAKOUT", "EXTENDED_BREAKOUT",
        "CONTINUATION_BREAKOUT",
    }, guide["name"]),
    ("Decision gate", str(snapshot.get("recommendation", "SKIP")).upper() == "TRADE", decision_reason),
    ("Probability threshold", probability >= .90, f"{probability:.1%} / 90.0%"),
    ("Risk context", snapshot.get("expected_drawdown") is not None, f"Expected {drawdown:.2%}"),
    ("Option contract", contract is not None, contract_name),
    ("Fresh market data", freshness["label"] == "LIVE" and not freshness["stale"], str(freshness["label"])),
]
check_rows = "".join(
    f"""<div class="decision-check {'pass' if passed else 'wait'}">
    {icon('shield-check' if passed else 'activity',16)}
    <span>{escape(label)}<small>{escape(detail)}</small></span>
    <b>{'READY' if passed else 'WAITING'}</b></div>"""
    for label, passed, detail in checklist)
st.markdown(
    f"""<section class="decision-panel">
      <header><div><span>{'TRADE CONTEXT' if action != 'NO TRADE' else 'WHY NO TRADE'}</span>
      <h3>{escape(guide['summary'])}</h3></div>
      <b>{sum(item[1] for item in checklist)}/{len(checklist)} checks ready</b></header>
      <div class="decision-checks">{check_rows}</div>
      <footer><strong>What happens next</strong><span>{escape(guide['watch'])}</span>
      <em>No entry, stop or target is estimated without verified contract pricing.</em></footer>
    </section>""", unsafe_allow_html=True)

# Market summary.
market_specs = [
    ("NIFTY 50", "NIFTY"), ("BANKNIFTY", "BANKNIFTY"),
    ("SENSEX", "SENSEX"), ("INDIA VIX", "INDIA_VIX"),
]
market_cards = []
for label, key in market_specs:
    price = quotes.get(key)
    market_cards.append(
        f"""<div class="market-tile"><span>{escape(label)}</span>
        <strong>{f'{float(price):,.2f}' if price is not None else '—'}</strong>
        <small>{'LIVE QUOTE' if price is not None else 'FEED OFFLINE'}</small></div>""")
st.markdown(
    f"""<div class="desk-section-head"><div><span>MARKET SUMMARY</span>
    <h3>Context at a glance</h3></div><b>{'CONNECTED' if connected else 'WAITING FOR DHAN'}</b></div>
    <section class="market-strip">{"".join(market_cards)}
      <div class="market-tile"><span>SESSION</span><strong>{escape(readable(snapshot.get('session_phase'), '—'))}</strong><small>MARKET PHASE</small></div>
      <div class="market-tile"><span>AI BIAS</span><strong>{escape(readable(direction, '—'))}</strong><small>MODEL CONTEXT</small></div>
    </section>""", unsafe_allow_html=True)

# Visual pipeline. Statuses are presentation of existing state, not new logic.
pipeline = [
    ("Market", connected, "Live feed" if connected else "Offline"),
    ("Stage-1", snapshot.get("current_strategy") is not None, guide["name"]),
    ("Decision", snapshot.get("recommendation") is not None, decision_reason),
    ("Model", snapshot.get("catboost_predicted_probability") is not None, f"{probability:.1%}"),
    ("Risk", snapshot.get("expected_drawdown") is not None, f"{drawdown:.2%} drawdown"),
    ("Paper", action != "NO TRADE" and contract is not None, "Available" if action != "NO TRADE" and contract else "Waiting"),
    ("Position", False, "No open position"),
    ("Journal", True, "Ready"),
]
pipeline_html = "".join(
    f"""<div class="flow-step {'ready' if ready else 'waiting'}"><i></i>
    <span>{escape(label)}</span><b>{escape(detail)}</b></div>
    {f'<div class="flow-arrow">→</div>' if index < len(pipeline) - 1 else ''}"""
    for index, (label, ready, detail) in enumerate(pipeline))
st.markdown(
    f"""<div class="desk-section-head"><div><span>TRADING WORKFLOW</span>
    <h3>From market data to journal</h3></div><b>READ-ONLY STATUS</b></div>
    <section class="workflow-rail">{pipeline_html}</section>""", unsafe_allow_html=True)

# Secondary intelligence remains compact and below the core decision.
paper_risk = safe_csv("reports/paper_trading/risk_statistics.csv")
paper_stats = paper_risk.iloc[0].to_dict() if len(paper_risk) else {}
joined = load_frame("joined_probability")
health = safe_csv("reports/system_monitoring/component_health.csv")
left, centre, right = st.columns([1, 1.25, 1])
with left:
    desk_metric("Paper capital", f"₹{float(paper_stats.get('ending_capital', 100000)):,.0f}",
                "Virtual account only", "positive")
    desk_metric("Paper win rate", f"{float(paper_stats.get('win_rate', 0)):.1%}",
                f"{int(paper_stats.get('total_trades', 0))} completed trades")
with centre:
    st.markdown('<div class="mini-panel"><header><span>RECENT DECISIONS</span><a href="/AI_Predictions">View all</a></header>', unsafe_allow_html=True)
    recent_rows = []
    for item in joined.sort_values("timestamp").tail(4).iloc[::-1].itertuples() if len(joined) else []:
        item_action, item_tone, _ = safe_action(item._asdict())
        recent_rows.append(
            f"""<div class="recent-decision"><time>{pd.to_datetime(item.timestamp).strftime('%d %b · %H:%M')}</time>
            <span>{escape(strategy_guide(getattr(item, 'current_strategy', ''))['name'])}</span>
            <b class="{escape(item_tone)}">{escape(item_action)}</b>
            <em>{float(getattr(item, 'catboost_predicted_probability', 0)):.0%}</em></div>""")
    st.markdown(
        f"""<div class="recent-list">{''.join(recent_rows) if recent_rows else '<p class="no-data">No decisions available</p>'}</div></div>""",
        unsafe_allow_html=True)
with right:
    ready = int(health.status.eq("READY").sum()) if len(health) else 0
    desk_metric("Pipeline health", f"{ready}/{len(health)} ready" if len(health) else "Unavailable",
                "Latest monitoring snapshot", "positive" if ready and ready == len(health) else "warning")
    desk_metric("Expected expansion", f"{expansion:.2%}", "Historical estimate")

st.markdown(
    '<div class="commercial-disclaimer">Research and paper trading only. AI output is not financial advice.</div>',
    unsafe_allow_html=True)
app_footer()
