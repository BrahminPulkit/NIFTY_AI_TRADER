"""Generate Step-27 offline validation artifacts without credentials or API calls."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from commands.operations.main_system_monitoring import frozen_paths
from src.dhan_live_broker import DhanInstrumentResolver
from src.institutional_monitoring import sha256
from src.live_broker_reporting import write_broker_reports


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/live_broker"
MASTER = ROOT / "data/raw/dhan_security_master_latest.csv"


def main() -> None:
    protected = frozen_paths()
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    indices = DhanInstrumentResolver(MASTER).resolve_indices()
    health = {
        "status": "DISCONNECTED", "last_connection_time": None,
        "last_heartbeat": None, "latency_ms": None, "token_valid": False,
        "detail": "Offline validation; connect through Streamlit session",
    }
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    if before != after:
        raise AssertionError("Frozen artifact changed during Step-27 validation")
    metadata = {
        "step": "27",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "OFFLINE_VALIDATION",
        "credential_storage": "STREAMLIT_SESSION_STATE_ONLY",
        "encrypted_local_storage_enabled": False,
        "credentials_present": False,
        "network_request_performed": False,
        "order_api_available": False,
        "completed_candles_only": True,
        "dynamic_security_ids": True,
        "automatic_reconnect_attempts": 3,
        "frozen_hashes_before": before,
        "frozen_hashes_after": after,
        "frozen_hashes_identical": before == after,
    }
    write_broker_reports(
        REPORTS, health,
        [instrument.public_dict() for instrument in indices.values()],
        metadata)


if __name__ == "__main__":
    main()
