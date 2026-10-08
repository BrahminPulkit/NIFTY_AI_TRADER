"""Fail-closed Step-22 paper prediction orchestration.

Replay may consume frozen OOF probabilities. Unseen inference requires an
explicitly persisted frozen model and Decision Engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from src.prediction_logger import PredictionLogger, PredictionRecord


class LiveInferenceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    timestamp: pd.Timestamp
    market: str
    probability: float
    prediction: int
    decision: str
    reason: str
    strategy: str
    regime: str
    confidence: float
    source: str


class FrozenOOFReplay:
    """Timestamp lookup only; never recomputes or regenerates probability."""

    def __init__(self, path: str | Path, model_name: str = "catboost"):
        frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        if frame.timestamp.duplicated().any():
            raise ValueError("OOF timestamps must be unique")
        self.frame = frame.set_index("timestamp")
        self.model_name = model_name

    def predict(self, timestamp: pd.Timestamp, _: pd.Series | None = None) -> tuple[float, int]:
        timestamp = pd.to_datetime(timestamp, utc=True).tz_convert("Asia/Kolkata")
        if timestamp not in self.frame.index:
            raise LiveInferenceUnavailable("Timestamp is outside the frozen OOF replay set")
        row = self.frame.loc[timestamp]
        probability_column = "predicted_probability"
        class_column = "predicted_class"
        return float(row[probability_column]), int(row[class_column])


class FrozenModelPredictor:
    def __init__(self, model_path: str | Path, feature_path: str | Path):
        model_path, feature_path = Path(model_path), Path(feature_path)
        if not model_path.exists() or not feature_path.exists():
            raise LiveInferenceUnavailable(
                "Frozen CatBoost model and exact feature-contract artifact are required")
        with model_path.open("rb") as handle:
            self.model = pickle.load(handle)
        self.features = json.loads(feature_path.read_text(encoding="utf-8"))
        if isinstance(self.features, dict):
            self.features = self.features.get("features", self.features.get("columns"))
        if not self.features:
            raise LiveInferenceUnavailable("Frozen model feature contract is empty")

    def predict(self, _: pd.Timestamp, row: pd.Series) -> tuple[float, int]:
        missing = sorted(set(self.features).difference(row.index))
        if missing:
            raise LiveInferenceUnavailable(f"Live feature contract mismatch: {missing}")
        matrix = pd.DataFrame([{name: row[name] for name in self.features}])
        probability = float(self.model.predict_proba(matrix)[0, 1])
        return probability, int(probability >= 0.5)


class FrozenDecisionReplay:
    def __init__(self, prediction_dataset: str | Path):
        frame = pd.read_parquet(prediction_dataset)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        if frame.timestamp.duplicated().any():
            raise ValueError("Decision replay timestamps must be unique")
        self.frame = frame.set_index("timestamp")

    def get(self, timestamp: pd.Timestamp) -> pd.Series:
        timestamp = pd.to_datetime(timestamp, utc=True).tz_convert("Asia/Kolkata")
        if timestamp not in self.frame.index:
            raise LiveInferenceUnavailable(
                "No frozen Decision Engine state exists for this unseen timestamp")
        return self.frame.loc[timestamp]


class LivePredictionEngine:
    def __init__(self, predictor, decisions: FrozenDecisionReplay,
                 logger: PredictionLogger, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")
        self.predictor, self.decisions, self.logger, self.threshold = (
            predictor, decisions, logger, float(threshold))

    def process_replay_timestamp(self, timestamp: pd.Timestamp, market: str = "NIFTY") -> PredictionResult:
        row = self.decisions.get(timestamp)
        probability, prediction = self.predictor.predict(timestamp, row)
        stage1_allowed = int(row.get("entry_signal", 0)) != 0
        engine_trade = str(row.get("recommendation", "SKIP")) == "TRADE"
        decision = "TRADE" if stage1_allowed and engine_trade and probability >= self.threshold else "SKIP"
        if not stage1_allowed:
            reason = "STAGE1_REJECTED"
        elif not engine_trade:
            reason = f"DECISION_ENGINE_{row.get('decision_reason', 'REJECTED')}"
        elif probability < self.threshold:
            reason = "PROBABILITY_BELOW_FROZEN_THRESHOLD"
        else:
            reason = "PAPER_SIGNAL_ONLY"
        result = PredictionResult(
            pd.to_datetime(timestamp, utc=True).tz_convert("Asia/Kolkata"), market,
            probability, prediction, decision, reason,
            str(row.get("current_strategy", "UNKNOWN")),
            str(row.get("market_state", "UNKNOWN")),
            float(row.get("strategy_confidence_score", np.nan)), "FROZEN_OOF_REPLAY",
        )
        self.logger.append(PredictionRecord(
            result.timestamp.isoformat(), result.market, result.probability,
            result.prediction, result.decision, result.reason, result.strategy,
            result.regime, result.confidence, result.source,
            {key: value for key, value in row.items() if not str(key).startswith("realized_")},
        ))
        return result

