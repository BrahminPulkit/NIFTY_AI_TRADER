"""Step-22 paper-prediction runner. Defaults to deterministic OOF replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.live_prediction_engine import (
    FrozenDecisionReplay, FrozenModelPredictor, FrozenOOFReplay,
    LiveInferenceUnavailable, LivePredictionEngine,
)
from src.prediction_logger import PredictionLogger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/live_prediction.json")
    parser.add_argument("--mode", choices=["replay", "live"])
    parser.add_argument("--max-events", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mode = args.mode or config["mode"]
    decisions = FrozenDecisionReplay(config["prediction_dataset"])
    if mode == "replay":
        predictor = FrozenOOFReplay(config["oof_predictions"])
        timestamps = predictor.frame.index[:args.max_events]
    else:
        predictor = FrozenModelPredictor(config["model_path"], config["model_features_path"])
        raise LiveInferenceUnavailable(
            "Live polling is disabled until the frozen CatBoost model and causal "
            "Decision Engine state are persisted and hash-verified")
    engine = LivePredictionEngine(
        predictor, decisions, PredictionLogger(config["journal_path"]), config["threshold"])
    for timestamp in timestamps:
        result = engine.process_replay_timestamp(timestamp)
        print(json.dumps({**result.__dict__, "timestamp": result.timestamp.isoformat()}, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LiveInferenceUnavailable as exc:
        print(f"SAFE_BLOCK: {exc}")
        raise SystemExit(2)

