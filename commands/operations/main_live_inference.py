"""Step 24 CLI: true CatBoost inference; never reads OOF probabilities."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

from src.live_data_provider import DhanIntradayProvider
from src.live_inference_engine import SafeBlock, TrueLiveInferenceEngine
from src.live_pipeline_monitor import LivePipelineMonitor


def _config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay(engine: TrueLiveInferenceEngine, path: str, max_events: int) -> None:
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    if "timestamp" not in frame:
        raise SafeBlock("Replay source requires timestamp plus the exact feature contract")
    timestamps = frame.pop("timestamp")
    expected = engine.predictor.contract["feature_order"]
    extras = [name for name in frame.columns if name not in expected]
    approved_excluded = engine.predictor.metadata.get("excluded_all_null_features", [])
    if extras != approved_excluded:
        raise SafeBlock(f"Replay source has unapproved fields: {extras}")
    frame = frame[expected]
    for index in range(min(len(frame), max_events)):
        result = engine.infer_contract_row(
            frame.iloc[[index]], timestamp=timestamps.iloc[index])
        print(json.dumps(asdict(result), sort_keys=True))


def _provider_frame(provider: DhanIntradayProvider) -> pd.DataFrame:
    rows = [item.as_dict() for item in provider.candles()]
    return pd.DataFrame(rows).drop(columns=["market", "source"])


def live_once(engine: TrueLiveInferenceEngine, config: dict) -> None:
    live = config["dhan"]
    index, option = live.get("index"), live.get("option")
    if not index or not option:
        raise SafeBlock("Dhan Index and ATM CALL contracts must both be configured")
    now = datetime.now().astimezone()
    common = {
        "from_date": live["from_date"],
        "to_date": live.get("to_date") or now.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id_env": live["client_id_env"], "access_token_env": live["access_token_env"],
    }
    index_history = _provider_frame(DhanIntradayProvider(market="NIFTY", **index, **common))
    option_history = _provider_frame(DhanIntradayProvider(market="ATM_CALL", **option, **common))
    observation_path = Path(config["decision_observations"])
    if not observation_path.exists():
        raise SafeBlock("Frozen causal Decision Engine observations are unavailable")
    observations = pd.read_parquet(observation_path)
    result = engine.infer_histories(index_history, option_history, observations)
    print(json.dumps(asdict(result), sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/live_inference.json")
    parser.add_argument("--mode", choices=["replay", "live"])
    parser.add_argument("--input")
    parser.add_argument("--max-events", type=int, default=10)
    args = parser.parse_args()
    config = _config(args.config)
    mode = args.mode or config["mode"]
    monitor = LivePipelineMonitor(config["health_path"])
    try:
        engine = TrueLiveInferenceEngine(
            config["production_model"], config["journal_path"], monitor)
        if mode == "replay":
            replay(engine, args.input or config["replay_source"], args.max_events)
        else:
            live_once(engine, config)
        return 0
    except Exception as exc:
        monitor.safe_block(exc)
        print(f"SAFE_BLOCK: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
