"""Live workspace consuming only the process-wide Dhan cache."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.research_ui.components.charts import style
from apps.research_ui.components.theme import app_footer, configure_page, icon, page_header
from src.dhan_connection_manager import get_dhan_manager
from src.live_inference_engine import SafeBlock, TrueLiveInferenceEngine
from src.live_pipeline_monitor import LivePipelineMonitor


def resample_candles(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty or minutes == 1:
        return frame.copy()
    data = frame.copy().set_index("timestamp")
    result = data.resample(
        f"{minutes}min", origin="start_day", offset="15min",
        label="left", closed="left",
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return result.reset_index()


def market_chart(frame: pd.DataFrame, latest: dict | None) -> go.Figure:
    data = frame.copy()
    data["ema_5"] = data.close.ewm(span=5, adjust=False).mean()
    data["ema_20"] = data.close.ewm(span=20, adjust=False).mean()
    volume = pd.to_numeric(data.volume, errors="coerce").fillna(0)
    typical = (data.high + data.low + data.close) / 3
    denominator = volume.groupby(data.timestamp.dt.date).cumsum()
    numerator = (typical * volume).groupby(data.timestamp.dt.date).cumsum()
    data["vwap"] = numerator.div(denominator.where(denominator.gt(0)))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=.03,
        row_heights=[.78, .22])
    fig.add_trace(go.Candlestick(
        x=data.timestamp, open=data.open, high=data.high,
        low=data.low, close=data.close, name="NIFTY",
        increasing_line_color="#2ECC71", decreasing_line_color="#FF5A5F",
        increasing_fillcolor="#2ECC71", decreasing_fillcolor="#FF5A5F"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.timestamp, y=data.ema_5, name="EMA 5",
        line=dict(color="#F5B041", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.timestamp, y=data.ema_20, name="EMA 20",
        line=dict(color="#4F8CFF", width=1.5)), row=1, col=1)
    if data.vwap.notna().any():
        fig.add_trace(go.Scatter(
            x=data.timestamp, y=data.vwap, name="VWAP",
            line=dict(color="#9B7BFF", width=1.2, dash="dot")), row=1, col=1)
    colors = [
        "#2ECC71" if close >= open_ else "#FF5A5F"
        for open_, close in zip(data.open, data.close)]
    fig.add_trace(go.Bar(
        x=data.timestamp, y=volume, name="Volume",
        marker_color=colors, opacity=.45), row=2, col=1)
    if latest and latest.get("timestamp"):
        signal_time = pd.to_datetime(latest["timestamp"], errors="coerce")
        if pd.notna(signal_time):
            nearest = data.iloc[(data.timestamp - signal_time).abs().argmin()]
            fig.add_trace(go.Scatter(
                x=[nearest.timestamp], y=[nearest.close], mode="markers+text",
                text=[latest.get("decision", "AI")], textposition="top center",
                marker=dict(size=11, color="#9B7BFF", symbol="diamond"),
                name="AI signal"), row=1, col=1)
    fig.update_layout(
        title="NIFTY completed candles", xaxis_rangeslider_visible=False,
        yaxis_title="Index", yaxis2_title="Volume")
    return style(fig, 590)


configure_page("Live Market")
page_header(
    "Live Market", "MARKET WORKSPACE",
    "Shared completed-candle cache, live quotes and controlled inference.")

manager = get_dhan_manager()
cache = manager.snapshot()
connected = cache["connected"]
latest = st.session_state.get("_quantedge_latest_inference")

st.markdown(
    f"""<div class="workspace-status {'online' if connected else 'offline'}">
    <div><i></i><strong>{'LIVE CACHE CONNECTED' if connected else 'LIVE FEED DISCONNECTED'}</strong>
    <span>{cache['manager_status']} · {cache['market_status']} ·
    {'worker running' if cache['worker_alive'] else 'worker stopped'}</span></div>
    <div><span>Updated</span><b>{cache['last_data_update'] or 'WAITING'}</b></div>
    <a href="/Broker_Connection" target="_self">{'Session status' if connected else 'Connect broker'}</a>
    </div>""", unsafe_allow_html=True)

market_specs = [
    ("NIFTY 50", "NIFTY"), ("BANKNIFTY", "BANKNIFTY"),
    ("SENSEX", "SENSEX"), ("INDIA VIX", "INDIA_VIX"),
]
cards = []
for label, key in market_specs:
    value = cache["quotes"].get(key)
    cards.append(
        f"""<div class="quote-card"><span>{label}</span>
        <strong>{f'{float(value):,.2f}' if value is not None else '—'}</strong>
        <small>{'SHARED CACHE' if value is not None else 'AWAITING FEED'}</small></div>""")
st.markdown(f'<section class="quote-strip">{"".join(cards)}</section>', unsafe_allow_html=True)

toolbar, action = st.columns([4, 1])
with toolbar:
    timeframe = st.segmented_control(
        "Timeframe", ["1m", "3m", "5m", "15m"], default="1m",
        label_visibility="collapsed")
with action:
    infer_clicked = st.button(
        "Run latest inference", icon=":material/neurology:",
        use_container_width=True,
        disabled=not connected or "NIFTY" not in cache["candles"])

if infer_clicked:
    try:
        index_history, option_history = manager.cached_inference_histories()
        engine = st.session_state.get("_quantedge_true_live_engine")
        if not isinstance(engine, TrueLiveInferenceEngine):
            engine = TrueLiveInferenceEngine(
                "production_model", "logs/live_inference/predictions.jsonl",
                LivePipelineMonitor("logs/live_inference/pipeline_health.json"))
            st.session_state["_quantedge_true_live_engine"] = engine
        observations = pd.read_parquet(
            Path("data/runtime/decision_observations.parquet"))
        result = engine.infer_histories(
            index_history, option_history, observations)
        st.session_state["_quantedge_latest_inference"] = asdict(result)
        st.toast(
            f"{result.decision} · {result.probability:.1%}"
            if result.probability is not None
            else f"{result.decision} · No strong setup")
        st.rerun()
    except (SafeBlock, Exception) as exc:
        st.session_state["_quantedge_latest_inference"] = None
        st.error(f"SAFE BLOCK: {str(exc)}")

source = cache["candles"].get("NIFTY", pd.DataFrame())
minutes = int(str(timeframe or "1m").removesuffix("m"))
chart_data = resample_candles(source, minutes).tail(500)
if len(chart_data):
    st.plotly_chart(
        market_chart(chart_data, latest),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False})
else:
    st.markdown(
        f"""<section class="chart-workspace"><div class="chart-offline">
        {icon('chart-candlestick',34)}
        <strong>{'Candle cache is warming up' if connected else 'Live feed disconnected'}</strong>
        <p>{'The background worker is requesting genuine completed candles.'
             if connected else 'Connect Dhan once. Every workspace will then share the same desktop session.'}</p>
        <a href="/Broker_Connection" target="_self">{'View cache status' if connected else 'Connect Dhan'}</a>
        </div></section>""", unsafe_allow_html=True)

options = cache["options"]
nifty_ce = options.get("NIFTY_ATM_CE")
facts = [
    ("Manager", cache["manager_status"], "Singleton lifecycle"),
    ("NIFTY ATM", f"{float(nifty_ce['strike']):,.0f}" if nifty_ce else "—",
     str(nifty_ce["expiry"]) if nifty_ce else "Contract unresolved"),
    ("Candle rows", f"{len(source):,}", "Completed 1-minute cache"),
    ("Failures", str(cache["consecutive_failures"]), cache["last_error"] or "Worker healthy"),
]
st.markdown(
    '<div class="desk-section-head"><div><span>MARKET OPERATIONS</span>'
    '<h3>Shared cache state</h3></div></div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, (label, value, detail) in zip(cols, facts):
    with col:
        st.markdown(
            f"""<div class="desk-metric"><span>{label}</span><strong>{value}</strong>
            <small>{detail}</small></div>""", unsafe_allow_html=True)

app_footer()
