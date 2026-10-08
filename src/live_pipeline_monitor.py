"""Thread-safe in-process health and latency telemetry for live inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import threading


@dataclass
class PipelineHealth:
    status: str = "INITIALISING"
    data_status: str = "UNKNOWN"
    feature_status: str = "UNKNOWN"
    contract_status: str = "UNKNOWN"
    model_status: str = "UNKNOWN"
    decision_status: str = "UNKNOWN"
    journal_status: str = "UNKNOWN"
    last_timestamp: str | None = None
    last_latency_ms: float | None = None
    last_error: str | None = None
    updated_utc: str | None = None


class LivePipelineMonitor:
    def __init__(self, status_path: str | Path):
        self.status_path = Path(status_path)
        self.health = PipelineHealth()
        self._lock = threading.Lock()

    def update(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                if not hasattr(self.health, key):
                    raise KeyError(f"Unknown health field: {key}")
                setattr(self.health, key, value)
            self.health.updated_utc = datetime.now(timezone.utc).isoformat()
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(self.health), indent=2), encoding="utf-8")
            temporary.replace(self.status_path)

    def safe_block(self, error: Exception | str) -> None:
        self.update(status="SAFE_BLOCK", last_error=str(error))

    def snapshot(self) -> dict:
        return asdict(self.health)

