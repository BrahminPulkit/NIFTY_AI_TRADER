import json

import pandas as pd
import pytest

from src.live_prediction_engine import (
    FrozenDecisionReplay, FrozenModelPredictor, FrozenOOFReplay,
    LiveInferenceUnavailable, LivePredictionEngine,
)
from src.prediction_logger import PredictionLogger


def test_oof_replay_returns_saved_probability(tmp_path):
    timestamp = pd.Timestamp("2026-07-01 10:00", tz="Asia/Kolkata")
    path = tmp_path / "oof.parquet"
    pd.DataFrame([{"timestamp": timestamp, "fold_id": 1, "model_name": "CATBOOST",
                  "predicted_probability": .81, "predicted_class": 1,
                  "true_label": 1}]).to_parquet(path)
    replay = FrozenOOFReplay(path)
    assert replay.predict(timestamp) == (.81, 1)
    with pytest.raises(LiveInferenceUnavailable):
        replay.predict(timestamp + pd.Timedelta(minutes=1))


def test_missing_frozen_model_fails_closed(tmp_path):
    with pytest.raises(LiveInferenceUnavailable):
        FrozenModelPredictor(tmp_path / "missing.pkl", tmp_path / "features.json")


def test_replay_decision_and_journal(tmp_path):
    timestamp = pd.Timestamp("2026-07-01 10:00", tz="Asia/Kolkata")
    oof, prediction, journal = (tmp_path / "oof.parquet",
                                tmp_path / "prediction.parquet",
                                tmp_path / "journal.jsonl")
    pd.DataFrame([{"timestamp": timestamp, "fold_id": 1, "model_name": "CATBOOST",
                  "predicted_probability": .95, "predicted_class": 1,
                  "true_label": 1}]).to_parquet(oof)
    pd.DataFrame([{"timestamp": timestamp, "entry_signal": 1,
                  "recommendation": "TRADE",
                  "decision_reason": "ACTIVE_STRATEGY_MATCH",
                  "current_strategy": "BREAKOUT", "market_state": "TREND_UP",
                  "strategy_confidence_score": .8}]).to_parquet(prediction)
    engine = LivePredictionEngine(
        FrozenOOFReplay(oof), FrozenDecisionReplay(prediction),
        PredictionLogger(journal), .9)
    assert engine.process_replay_timestamp(timestamp).decision == "TRADE"
    payload = json.loads(journal.read_text().strip())
    assert payload["probability"] == .95
    assert "realised_outcome" not in payload

