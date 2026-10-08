"""Build the causal historical observation state required by the frozen engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.strategy_decision_engine import build_decision_observations


SETUPS = Path("data/setups/stage1_setup_dataset.parquet")
ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
OUTPUT = Path("data/runtime/decision_observations.parquet")
METADATA = Path("data/runtime/decision_observations_metadata.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    setup_columns = [
        "timestamp", "entry_signal", "trade_allowed", "open", "high", "low",
        "close", "candle_range", "atr_14", "rolling_volatility", "rolling_range",
        "ema_5", "ema_9", "ema_20", "ema_50", "breakout_flag",
        "breakdown_flag", "minutes_from_open",
    ]
    market_columns = [
        "timestamp", "option_close", "option_high", "option_low",
        "index_session_id",
    ]
    before = {"setups": sha256(SETUPS), "aligned": sha256(ALIGNED)}
    setups = pd.read_parquet(SETUPS, columns=setup_columns)
    aligned = pd.read_parquet(ALIGNED, columns=market_columns)
    observations = build_decision_observations(setups, aligned)
    if observations.timestamp.duplicated().any():
        raise AssertionError("Decision observation timestamps must be unique")
    if not observations.timestamp.is_monotonic_increasing:
        raise AssertionError("Decision observations must be chronological")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(OUTPUT, index=False)
    after = {"setups": sha256(SETUPS), "aligned": sha256(ALIGNED)}
    if before != after:
        raise AssertionError("Frozen source changed while building runtime state")
    METADATA.write_text(json.dumps({
        "purpose": "Step 24 frozen Decision Engine runtime observation state",
        "rows": len(observations),
        "start": str(observations.timestamp.min()),
        "end": str(observations.timestamp.max()),
        "source_sha256": before,
        "output_sha256": sha256(OUTPUT),
        "causal_rule": "Only outcome_available_timestamp <= inference timestamp may enter state",
        "research_artifacts_modified": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

