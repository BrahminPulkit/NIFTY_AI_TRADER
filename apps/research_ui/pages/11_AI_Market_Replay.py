"""Candle-by-candle replay UI over frozen historical artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.charts import style
from apps.research_ui.components.data import PROJECT_ROOT, load_frame
from apps.research_ui.components.operational import decision_explanation
from apps.research_ui.components.theme import (
    ai_thinking, app_footer, configure_page, metric_card, page_header,
    section_header, status_banner,
)


SETUPS = PROJECT_ROOT / "data/setups/stage1_setup_dataset.parquet"
ALIGNED = PROJECT_ROOT / "data/aligned/canonical_index_option_aligned.parquet"


@st.cache_data(show_spinner=False)
def available_dates(modified_ns: int) -> list:
    del modified_ns
    timestamps = pd.read_parquet(SETUPS, columns=["timestamp"]).timestamp
    return sorted(pd.to_datetime(timestamps).dt.date.unique().tolist())


@st.cache_data(show_spinner=False)
def replay_day(session_id: int, setup_ns: int, aligned_ns: int) -> pd.DataFrame:
    del setup_ns, aligned_ns
    setups = pd.read_parquet(SETUPS, filters=[("session_id", "==", session_id)])
    aligned = pd.read_parquet(ALIGNED, filters=[("index_session_id", "==", session_id)])
    keep = ["timestamp", "option_close", "option_volume", "option_rsi_14"]
    return setups.merge(aligned[keep], on="timestamp", how="left", validate="one_to_one")


configure_page("AI Market Replay")
page_header("AI Market Replay", "HISTORICAL DECISION THEATRE",
            "Replay frozen market history candle-by-candle without regenerating models or probabilities.")
status_banner("Saved OOF probabilities exist only for historically eligible model rows. Other candles are explicitly marked NOT EVALUATED.",
              "info", "LEAKAGE-SAFE REPLAY")

dates = available_dates(SETUPS.stat().st_mtime_ns)
selected = st.date_input("Trading date", value=dates[-1], min_value=dates[0], max_value=dates[-1])
if selected not in dates:
    st.warning("The selected date is not a recorded trading session.")
    st.stop()
session_id = int(pd.Timestamp(selected).strftime("%Y%m%d"))
day = replay_day(session_id, SETUPS.stat().st_mtime_ns, ALIGNED.stat().st_mtime_ns)
day["timestamp"] = pd.to_datetime(day.timestamp)

if st.session_state.get("replay_date") != str(selected):
    st.session_state.replay_date = str(selected)
    st.session_state.replay_index = 0
    st.session_state.replay_playing = False
st.session_state.setdefault("replay_index", 0)
st.session_state.setdefault("replay_playing", False)

controls = st.columns([1, 1, 1, 1, 2])
if controls[0].button("◀ Previous", use_container_width=True):
    st.session_state.replay_playing = False
    st.session_state.replay_index = max(0, st.session_state.replay_index - 1)
if controls[1].button("▶ Play", use_container_width=True):
    st.session_state.replay_playing = True
if controls[2].button("Ⅱ Pause", use_container_width=True):
    st.session_state.replay_playing = False
if controls[3].button("Next ▶", use_container_width=True):
    st.session_state.replay_playing = False
    st.session_state.replay_index = min(len(day) - 1, st.session_state.replay_index + 1)
speed = controls[4].select_slider("Playback speed", options=["0.5×", "1×", "2×", "4×"], value="2×")
index = st.slider("Replay position", 0, len(day) - 1, st.session_state.replay_index,
                  format="%d candle")
st.session_state.replay_index = index
current = day.iloc[index]
timestamp = current.timestamp

oof = load_frame("catboost_oof")
prediction = load_frame("prediction_v1")
for frame in (oof, prediction):
    if len(frame): frame["timestamp"] = pd.to_datetime(frame.timestamp)
oof_row = oof.loc[oof.timestamp.eq(timestamp)] if len(oof) else oof
decision_row = prediction.loc[prediction.timestamp.eq(timestamp)] if len(prediction) else prediction
probability = float(oof_row.iloc[0].predicted_probability) if len(oof_row) else None
decision = decision_row.iloc[0].to_dict() if len(decision_row) else {}
final = (
    "TRADE" if current.trade_allowed == 1
    and decision.get("recommendation") == "TRADE"
    and probability is not None and probability >= .9 else "SKIP"
)

section_header("Replay Clock", f"Candle {index + 1} of {len(day)} · {timestamp}", "FROZEN HISTORY")
cols = st.columns(6)
items = [
    ("Index", f"{current.close:,.2f}", "Candle close", "N"),
    ("Setup Quality", "STRONG" if current.trade_allowed else "WAIT", current.rejection_reason, "1"),
    ("Setup", {1:"LONG",-1:"SHORT",0:"NONE"}.get(int(current.entry_signal),"NONE"), current.structure_state, "⌁"),
    ("Probability", f"{probability:.1%}" if probability is not None else "NOT EVALUATED", "Saved OOF only", "◎"),
    ("Final Guidance", decision.get("recommendation", "NOT EVALUATED"), decision.get("decision_reason", "No eligible row"), "◇"),
    ("Final Decision", final, "Threshold 0.90", "✦"),
]
for col, item in zip(cols, items):
    with col: metric_card(item[0], item[1], str(item[2]).replace("_"," "), "neutral", item[3])

visible = day.iloc[:index + 1]
fig = go.Figure(go.Candlestick(
    x=visible.timestamp, open=visible.open, high=visible.high,
    low=visible.low, close=visible.close, name="NIFTY"))
fig.add_trace(go.Scatter(x=visible.timestamp, y=visible.ema_20, name="EMA 20",
                         line=dict(color="#2563EB", width=1)))
fig.add_trace(go.Scatter(x=visible.timestamp, y=visible.ema_50, name="EMA 50",
                         line=dict(color="#F59E0B", width=1)))
fig.update_layout(title=f"NIFTY replay · {selected}", xaxis_rangeslider_visible=False)
st.plotly_chart(style(fig, 500), use_container_width=True)

left, right = st.columns([.8, 1.2])
with left:
    ai_thinking(final)
with right:
    section_header("Candle Feature State", "Only values available at this candle")
    features = {
        "RSI 14": current.rsi_14, "ATR 14": current.atr_14,
        "EMA 5": current.ema_5, "EMA 9": current.ema_9,
        "EMA 20": current.ema_20, "EMA 50": current.ema_50,
        "Rolling volatility": current.rolling_volatility,
        "Trend": current.trend_state, "Momentum": current.momentum_state,
        "Structure": current.structure_state, "Session": current.session_state,
        "Option premium": current.option_close, "Option volume": current.option_volume,
    }
    st.dataframe(pd.DataFrame({
        "Feature": list(features.keys()),
        "Value": [str(value) if not isinstance(value, float) else f"{value:.6g}"
                  for value in features.values()],
    }),
                 use_container_width=True, hide_index=True, height=345)
    with st.expander("Explain this decision", expanded=True):
        explanation = {**current.to_dict(), **decision,
                       "catboost_predicted_probability": probability or 0,
                       "recommendation": final}
        decision_explanation(explanation, load_frame("feature_importance"))

if st.session_state.replay_playing and index < len(day) - 1:
    st.session_state.replay_index += 1
    delay = {"0.5×": 2.0, "1×": 1.0, "2×": .5, "4×": .25}[speed]
    time.sleep(delay)
    st.rerun()
elif index >= len(day) - 1:
    st.session_state.replay_playing = False
app_footer()
