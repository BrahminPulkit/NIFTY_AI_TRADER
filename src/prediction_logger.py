"""Append-only structured journal for paper predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class PredictionRecord:
    timestamp: str
    market: str
    probability: float
    prediction: int
    decision: str
    reason: str
    strategy: str
    regime: str
    confidence: float
    source: str
    features: dict


class PredictionLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: PredictionRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["timestamp"] = pd.Timestamp(payload["timestamp"]).isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=PredictionRecord.__dataclass_fields__)
        return pd.DataFrame(json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line)

