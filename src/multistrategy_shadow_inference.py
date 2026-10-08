"""Paper-only inference for validated multi-strategy CALL/PUT candidates."""

from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import pandas as pd

from src.feature_pipeline import build_features
from src.option_feature_pipeline import OPTION_FEATURES, OPTION_FEATURE_VERSION, build_option_features
from src.stage1_v3_research import build_candidate_setups


class _Package:
    def __init__(self, root: Path):
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation["status"] != "FORWARD_PAPER_CANDIDATE":
            raise ValueError("Candidate did not pass final chronological test")
        if validation.get("feature_data_contract") != OPTION_FEATURE_VERSION:
            raise ValueError("Candidate uses an invalid or obsolete option feature contract")
        self.strategy = validation["strategy"]
        self.direction = validation["direction"]
        self.threshold = float(validation["selected_threshold"])
        self.feature_order = validation["feature_order"]
        self.preprocessor = joblib.load(root / "preprocessor.joblib")
        self.model = CatBoostClassifier()
        self.model.load_model(root / "model.cbm")

    def probability(self, row: pd.DataFrame) -> float:
        return float(self.model.predict_proba(
            self.preprocessor.transform(row[self.feature_order]))[0, 1])


class MultiStrategyShadowEngine:
    def __init__(self, root: str | Path = "models/multistrategy_paper_v1"):
        base = Path(root)
        self.packages = []
        for validation_path in sorted(base.glob("*/validation.json")):
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if (validation.get("status") == "FORWARD_PAPER_CANDIDATE"
                    and validation.get("feature_data_contract") == OPTION_FEATURE_VERSION):
                self.packages.append(_Package(validation_path.parent))

    def infer(self, index: pd.DataFrame, call: pd.DataFrame, put: pd.DataFrame) -> dict:
        features = build_features(index)
        timestamp = pd.to_datetime(features.timestamp.iloc[-1])
        if not self.packages:
            return {
                "timestamp": timestamp.isoformat(), "best_signal": None, "checks": [],
                "status": "MODEL_VALIDATION_BLOCKED", "paper_only": True,
                "reason": "NO_MODEL_PASSED_CURRENT_OPTION_DATA_CONTRACT",
            }
        setups = build_candidate_setups(features)
        option_rows = {
            "CALL": build_option_features(call).iloc[-1],
            "PUT": build_option_features(put).iloc[-1],
        }
        checks = []
        for package in self.packages:
            setup = setups[package.strategy][1].iloc[-1]
            required_direction = 1 if package.direction == "CALL" else -1
            if int(setup.entry_signal) != required_direction:
                continue
            feature = features.iloc[-1].to_dict()
            for name in OPTION_FEATURES:
                feature[f"option_{name}"] = option_rows[package.direction][name]
            feature["strategy"] = package.strategy
            probability = package.probability(pd.DataFrame([feature]))
            approved = probability >= package.threshold
            checks.append({
                "timestamp": timestamp.isoformat(), "side": package.direction,
                "action": f"BUY {'CE' if package.direction == 'CALL' else 'PE'}" if approved else "NO TRADE",
                "decision": "PAPER_SIGNAL" if approved else "REJECTED",
                "strategy": package.strategy, "probability": probability,
                "threshold": package.threshold, "paper_only": True,
                "reason": "MULTISTRATEGY_THRESHOLD_PASSED" if approved else "PROBABILITY_BELOW_STRATEGY_THRESHOLD",
            })
        approved = [check for check in checks if check["decision"] == "PAPER_SIGNAL"]
        best = max(approved, key=lambda item: item["probability"], default=None)
        return {"timestamp": timestamp.isoformat(), "best_signal": best, "checks": checks,
                "status": "PAPER_SIGNAL" if best else "NO_APPROVED_SETUP", "paper_only": True}

    def replay(self, index: pd.DataFrame, call: pd.DataFrame, put: pd.DataFrame,
               session_date: str) -> list[dict]:
        """Batch-score every causal strategy event in one completed session."""
        features = build_features(index)
        features["timestamp"] = pd.to_datetime(features.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        setups = build_candidate_setups(features)
        feature_rows = features.set_index("timestamp")
        options = {
            "CALL": build_option_features(call).set_index("timestamp"),
            "PUT": build_option_features(put).set_index("timestamp"),
        }
        events = []
        for package in self.packages:
            direction = 1 if package.direction == "CALL" else -1
            candidate = setups[package.strategy][1]
            timestamps = pd.DatetimeIndex(candidate.loc[
                candidate.entry_signal.eq(direction), "timestamp"])
            timestamps = timestamps[timestamps.strftime("%Y-%m-%d") == session_date]
            timestamps = timestamps.intersection(feature_rows.index).intersection(
                options[package.direction].index)
            if timestamps.empty:
                continue
            rows = []
            for timestamp in timestamps:
                row = feature_rows.loc[timestamp].to_dict()
                option_row = options[package.direction].loc[timestamp]
                for name in OPTION_FEATURES:
                    row[f"option_{name}"] = option_row[name]
                row["strategy"] = package.strategy
                rows.append(row)
            matrix = pd.DataFrame(rows)[package.feature_order]
            probabilities = package.model.predict_proba(
                package.preprocessor.transform(matrix))[:, 1]
            for timestamp, probability in zip(timestamps, probabilities):
                approved = float(probability) >= package.threshold
                option_row = options[package.direction].loc[timestamp]
                events.append({
                    "timestamp": timestamp.isoformat(), "side": package.direction,
                    "action": f"BUY {'CE' if package.direction == 'CALL' else 'PE'}" if approved else "NO TRADE",
                    "decision": "PAPER_SIGNAL" if approved else "REJECTED",
                    "strategy": package.strategy, "probability": float(probability),
                    "confidence": float(probability), "threshold": package.threshold,
                    "reason": "MULTISTRATEGY_THRESHOLD_PASSED" if approved else "PROBABILITY_BELOW_STRATEGY_THRESHOLD",
                    "source": f"RETRAINED_MULTISTRATEGY_{package.direction}_PAPER", "paper_only": True,
                    "entry_premium": float(option_row.close),
                    "index_price": float(feature_rows.loc[timestamp, "close"]),
                })
        return sorted(events, key=lambda event: event["timestamp"])
