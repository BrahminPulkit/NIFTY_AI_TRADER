"""Read-only institutional monitoring console for Step 26."""

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.research_ui.components.charts import style
from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import (
    app_footer, configure_page, empty_state, metric_card, page_header,
    section_header, status_banner,
)


ROOT = Path("reports/system_monitoring")
PAPER = Path("reports/paper_trading")
CONFIG = Path("config/system_monitoring.json")


def frame(name: str, *, paper: bool = False) -> pd.DataFrame:
    path = (PAPER if paper else ROOT) / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def value(row: dict, key: str, fallback="—"):
    result = row.get(key, fallback)
    return fallback if pd.isna(result) else result


def tone(status: str) -> str:
    return {
        "READY": "positive", "WAITING": "neutral", "STALE": "warning",
        "BLOCKED": "negative", "FAILED": "negative",
    }.get(str(status).upper(), "neutral")


configure_page("System Monitoring")
page_header(
    "System Monitoring", "INSTITUTIONAL OPERATIONS CENTRE",
    "Read-only observability across data, features, inference and paper execution.")

config = payload(CONFIG)
refresh_seconds = int(config.get("auto_refresh_seconds", 30))
status_banner(
    f"Monitoring summaries refresh every {refresh_seconds} seconds. "
    "No inference, trading, or model mutation occurs on this page.",
    "info", "READ-ONLY TELEMETRY")


@st.fragment(run_every=f"{refresh_seconds}s")
def monitoring_console() -> None:
    health = frame("component_health.csv")
    pipeline = frame("pipeline_status.csv")
    data = frame("data_health.csv")
    features = frame("feature_health.csv")
    predictions = frame("prediction_health.csv")
    latency = frame("latency_statistics.csv")
    drift = frame("drift_monitor.csv")
    alerts = frame("alerts.csv")
    daily = frame("daily_report.csv")
    risk = frame("risk_statistics.csv", paper=True)
    equity = frame("equity_curve.csv", paper=True)
    metadata = payload(ROOT / "metadata.json")

    if health.empty:
        empty_state(
            "Monitoring snapshot unavailable",
            "Run main_system_monitoring.py to generate the first deterministic snapshot.")
        return

    health_counts = health.status.value_counts()
    prediction_row = predictions.iloc[0].to_dict() if len(predictions) else {}
    latency_row = latency.iloc[0].to_dict() if len(latency) else {}
    feature_row = features.iloc[0].to_dict() if len(features) else {}
    risk_row = risk.iloc[0].to_dict() if len(risk) else {}

    section_header("Executive Health", "Current status of every frozen pipeline stage",
                   str(metadata.get("generated_utc", "SNAPSHOT UNAVAILABLE")))
    cards = st.columns(6)
    card_values = [
        ("Ready", int(health_counts.get("READY", 0)), "Verified components", "READY"),
        ("Stale", int(health_counts.get("STALE", 0)), "Heartbeat overdue", "STALE"),
        ("Blocked", int(health_counts.get("BLOCKED", 0)), "Unavailable components", "BLOCKED"),
        ("Features",
         f"{int(value(feature_row, 'valid_features', 0))}/{int(value(feature_row, 'required_features', 0))}",
         "Feature contract", value(feature_row, "status", "WAITING")),
        ("Latency",
         f"{float(value(latency_row, 'current_latency_ms', 0)):.1f} ms"
         if pd.notna(latency_row.get("current_latency_ms")) else "—",
         "Latest true inference", value(latency_row, "status", "WAITING")),
        ("Active Alerts", len(alerts), "Operational notices",
         "BLOCKED" if len(alerts) and alerts.severity.eq("CRITICAL").any() else "WAITING"),
    ]
    for col, item in zip(cards, card_values):
        with col:
            metric_card(item[0], item[1], item[2], tone(item[3]), "◆")

    c1, c2 = st.columns([1.15, .85])
    with c1:
        section_header("Pipeline Flow", "Freshness and last successful action by stage")
        fig = go.Figure()
        palette = {
            "READY": "#00C853", "WAITING": "#64748B", "STALE": "#F59E0B",
            "BLOCKED": "#EF4444", "FAILED": "#DC2626",
        }
        fig.add_trace(go.Bar(
            x=pipeline.stage, y=[1] * len(pipeline),
            marker_color=[palette.get(str(x), "#64748B") for x in pipeline.status],
            text=pipeline.status, textposition="inside",
            customdata=pipeline[["last_success", "detail"]].fillna("Unavailable"),
            hovertemplate="<b>%{x}</b><br>Status: %{text}<br>Last: %{customdata[0]}"
                          "<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(yaxis_visible=False, showlegend=False, title="Operational pipeline")
        st.plotly_chart(style(fig, 330), use_container_width=True)
    with c2:
        section_header("Model & Contract", "Frozen package verification")
        model_rows = health.loc[health.component.isin(["Model", "Feature Contract", "Threshold"])]
        enterprise_table(model_rows, key="monitor_model", page_size=5, searchable=False, height=260)

    tabs = st.tabs([
        "Data Health", "Prediction Health", "Paper Performance",
        "Drift & Latency", "Alert Centre", "Recent Events",
    ])
    with tabs[0]:
        enterprise_table(data, key="monitor_data", page_size=12, height=430)
    with tabs[1]:
        cols = st.columns(5)
        pvalues = [
            ("Predictions Today", int(value(prediction_row, "predictions", 0)), "Recorded signals"),
            ("Trades Today", int(value(prediction_row, "trades", 0)), "TRADE decisions"),
            ("Skipped Today", int(value(prediction_row, "skipped", 0)), "SKIP decisions"),
            ("Average Probability",
             f"{float(value(prediction_row, 'average_probability', 0)):.1%}"
             if pd.notna(prediction_row.get("average_probability")) else "—", "True inference only"),
            ("Trade Rate", f"{float(value(prediction_row, 'trade_pct', 0)):.1%}", "Prediction coverage"),
        ]
        for col, item in zip(cols, pvalues):
            with col:
                metric_card(*item, "neutral", "◇")
        if len(predictions):
            enterprise_table(predictions, key="monitor_prediction", page_size=5, searchable=False)
    with tabs[2]:
        cols = st.columns(5)
        paper_values = [
            ("Net P&L", f"₹{float(value(risk_row, 'net_pnl', 0)):,.2f}", "Paper only"),
            ("Win Rate", f"{float(value(risk_row, 'win_rate', 0)):.1%}", "Completed trades"),
            ("Profit Factor", f"{float(value(risk_row, 'profit_factor', 0)):.2f}", "Gross win / loss"),
            ("Expectancy", f"₹{float(value(risk_row, 'expectancy', 0)):,.2f}", "Per completed trade"),
            ("Max Drawdown", f"₹{float(value(risk_row, 'maximum_drawdown', 0)):,.2f}", "Paper equity"),
        ]
        for col, item in zip(cols, paper_values):
            with col:
                metric_card(*item, "neutral", "▣")
        if len(equity):
            fig = px.line(equity, x="timestamp", y="equity", title="Paper equity")
            st.plotly_chart(style(fig), use_container_width=True)
        else:
            empty_state("No paper equity", "No completed virtual trade exists yet.")
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            if len(latency):
                enterprise_table(latency, key="monitor_latency", page_size=5, searchable=False)
            else:
                empty_state("Latency unavailable", "No true inference latency has been recorded.")
        with c2:
            if len(drift):
                enterprise_table(drift, key="monitor_drift", page_size=5, searchable=False)
            else:
                empty_state("Drift unavailable", "A minimum live sample is required.")
    with tabs[4]:
        if len(alerts):
            enterprise_table(alerts, key="monitor_alerts", page_size=20, height=460)
        else:
            empty_state("No active alerts", "The current snapshot contains no operational alert.")
    with tabs[5]:
        events = pd.concat([
            pipeline.rename(columns={"stage": "source", "last_success": "timestamp"})[
                ["timestamp", "source", "status", "detail"]],
            alerts.rename(columns={"severity": "status", "message": "detail"})[
                ["timestamp", "status", "detail"]].assign(source="Alert"),
        ], ignore_index=True)
        enterprise_table(events, key="monitor_events", page_size=20, height=460)

    section_header("Feature Contract Status", "Exact contract failures; never silently tolerated")
    enterprise_table(features, key="monitor_features", page_size=5, searchable=False, height=240)
    if len(daily):
        section_header("Reporting Snapshot", "Daily operational summary")
        enterprise_table(daily, key="monitor_daily", page_size=5, searchable=False, height=240)


monitoring_console()
app_footer()
