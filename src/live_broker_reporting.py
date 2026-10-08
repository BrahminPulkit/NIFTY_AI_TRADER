"""Credential-free Step-27 operational reporting."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PUBLIC_HEALTH_FIELDS = [
    "status", "last_connection_time", "last_heartbeat", "latency_ms",
    "token_valid", "detail",
]
PUBLIC_INSTRUMENT_FIELDS = [
    "market", "security_id", "exchange_segment", "instrument", "display_name",
    "expiry", "strike", "option_type",
]


def write_broker_reports(report_dir: str | Path, health: dict,
                         instruments: list[dict], metadata: dict) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    health_row = {name: health.get(name) for name in PUBLIC_HEALTH_FIELDS}
    pd.DataFrame([health_row]).to_csv(report_dir / "broker_health.csv", index=False)
    mapping = pd.DataFrame([
        {name: item.get(name) for name in PUBLIC_INSTRUMENT_FIELDS}
        for item in instruments
    ], columns=PUBLIC_INSTRUMENT_FIELDS)
    mapping.to_csv(report_dir / "instrument_mapping.csv", index=False)
    pd.DataFrame([{
        "timestamp": health.get("last_heartbeat"),
        "latency_ms": health.get("latency_ms"),
        "status": health.get("status"),
    }]).to_csv(report_dir / "latency.csv", index=False)
    safe_metadata = dict(metadata)
    forbidden = {"access_token", "access-token", "client_id", "client-id", "credentials"}
    if forbidden.intersection(key.lower() for key in safe_metadata):
        raise ValueError("Credential-like metadata field rejected")
    (report_dir / "metadata.json").write_text(
        json.dumps(safe_metadata, indent=2, default=str), encoding="utf-8")
    (report_dir / "connection_report.md").write_text(
        f"""# Dhan Live Broker Connection Report

- Status: {health_row['status']}
- Token valid: {health_row['token_valid']}
- Last connection: {health_row['last_connection_time']}
- Last heartbeat: {health_row['last_heartbeat']}
- Latency: {health_row['latency_ms']}
- Dynamically resolved instruments: {len(mapping)}
- Credential storage: Streamlit Session State only
- Completed candles only: Yes
- Automatic reconnect: Enabled
- Trading/order API surface: Absent

## Safety verdict

The integration is read-only and fail-closed. Credentials are deliberately
excluded from this report and every exported artifact. A disconnected snapshot
does not imply a Dhan failure; it means no user credentials were supplied to the
offline validation command.
""", encoding="utf-8")
