"""Controls for the process-wide single-user Dhan connection manager."""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from apps.research_ui.components.tables import enterprise_table
from apps.research_ui.components.theme import (
    app_footer, configure_page, empty_state, metric_card, page_header,
    section_header, status_banner,
)
from src.dhan_connection_manager import get_dhan_manager
from src.live_broker_reporting import write_broker_reports


CONFIG = json.loads(Path("config/live_broker.json").read_text(encoding="utf-8"))
manager = get_dhan_manager()


def persist_public_state(snapshot: dict) -> None:
    instruments = list(snapshot["indices"].values()) + list(
        snapshot["options"].values())
    metadata_path = Path("reports/live_broker/metadata.json")
    baseline = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {})
    baseline.update({
        "step": "27C",
        "mode": "SINGLE_USER_PROCESS_MANAGER",
        "credential_storage": "PROCESS_MEMORY_ONLY",
        "credentials_exported": False,
        "order_api_available": False,
        "completed_candles_only": True,
        "worker_name": snapshot.get("worker_name"),
    })
    write_broker_reports(
        "reports/live_broker", snapshot["health"], instruments, baseline)


configure_page("Broker Connection")
page_header(
    "Broker Connection", "DESKTOP SESSION CONTROL",
    "One process-wide Dhan login shared by every workspace.")

snapshot = manager.snapshot()
health = snapshot["health"]
status_banner(
    "Credentials remain in process memory only. Page switching and new tabs "
    "share the same local desktop connection. Closing the application or "
    "disconnecting removes the session.",
    "info", "SINGLE-USER SESSION")

section_header("Session Status", "Authentication, worker and shared-cache health")
cols = st.columns(6)
items = [
    ("Manager", snapshot["manager_status"], "Process-wide lifecycle"),
    ("Authentication", health["status"], health["detail"]),
    ("Worker", "RUNNING" if snapshot["worker_alive"] else "STOPPED",
     snapshot.get("worker_name") or "No worker"),
    ("Market", snapshot["market_status"], "India market session"),
    ("Last Update", snapshot["last_data_update"] or "—", "Shared cache"),
    ("Latency", f"{health['latency_ms']:.0f} ms"
     if health["latency_ms"] is not None else "—", "Dhan profile"),
]
for col, item in zip(cols, items):
    with col:
        tone = (
            "positive" if str(item[1]) in {"CONNECTED", "RUNNING", "OPEN"}
            else "warning" if str(item[1]) in {"RECONNECTING", "PRE_OPEN"}
            else "neutral")
        metric_card(*item, tone, "shield-check")

if not snapshot["connected"]:
    section_header("Secure Login", "Authenticate once for this application lifetime")
    with st.form("dhan_connection_form", clear_on_submit=False):
        client_id = st.text_input(
            "Dhan Client ID", autocomplete="off",
            placeholder="Enter your Dhan Client ID")
        access_token = st.text_input(
            "Dhan Access Token", type="password", autocomplete="off",
            placeholder="Stored in process memory only")
        connect = st.form_submit_button(
            "Connect Dhan", use_container_width=True)
    if connect:
        try:
            result = manager.connect(client_id, access_token)
            persist_public_state(result)
            if result["connected"]:
                st.toast("Dhan connected for the application lifetime")
                st.rerun()
            else:
                st.error(
                    f"{result['manager_status']}: "
                    f"{result['health']['detail']}")
        except Exception as exc:
            st.error(f"Connection failed: {str(exc)}")
else:
    section_header("Session Controls", "Controls never create a second Dhan client")
    left, right = st.columns(2)
    refresh = left.button(
        "Refresh shared cache", icon=":material/refresh:",
        use_container_width=True)
    disconnect = right.button(
        "Disconnect", icon=":material/power_settings_new:",
        use_container_width=True, type="secondary")
    if refresh:
        manager.request_refresh()
        st.toast("Background refresh requested")
    if disconnect:
        result = manager.disconnect()
        persist_public_state(result)
        st.rerun()


@st.fragment(run_every="3s")
def cache_monitor() -> None:
    state = manager.snapshot()
    if not state["connected"]:
        if state["last_error"]:
            st.warning(state["last_error"])
        return
    section_header(
        "Shared Market Cache",
        "Every workspace reads this snapshot; no page calls Dhan",
        state["manager_status"])
    cache_stats = st.columns(5)
    values = [
        ("Index Quotes", len(state["quotes"]), "Latest values"),
        ("Option Quotes", len(state["option_quotes"]), "Resolved contracts"),
        ("Resolved Options", len(state["options"]), "ATM instruments"),
        ("Candle Series", len(state["candles"]), "Completed only"),
        ("Failures", state["consecutive_failures"], "Consecutive worker cycles"),
    ]
    for col, item in zip(cache_stats, values):
        with col:
            metric_card(*item, "neutral", "activity")
    instruments = list(state["indices"].values()) + list(
        state["options"].values())
    mapping = pd.DataFrame(instruments)
    if len(mapping):
        enterprise_table(
            mapping, key="broker_mapping", page_size=20, height=360)
    else:
        empty_state(
            "Instrument cache waiting",
            "The background worker is resolving genuine Dhan instruments.")


cache_monitor()
app_footer()
