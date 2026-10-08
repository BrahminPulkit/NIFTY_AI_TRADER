"""Paper-only PUT candidate inference; never places broker orders."""

from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import pandas as pd

from src.live_feature_generator import build_live_contract_row


class PutShadowInferenceEngine:
    def __init__(self, package_dir: str | Path):
        root = Path(package_dir)
        self.contract = json.loads((root / "feature_contract.json").read_text(encoding="utf-8"))
        self.validation = json.loads((root / "threshold_validation.json").read_text(encoding="utf-8"))
        if self.validation.get("status") != "FORWARD_PAPER_CANDIDATE":
            raise ValueError("PUT candidate has not passed the temporal holdout gate")
        self.threshold = float(self.validation["selected_threshold"])
        self.preprocessor = joblib.load(root / "preprocessor.joblib")
        self.model = CatBoostClassifier()
        self.model.load_model(root / "model.cbm")
        production_contract = Path("production_model/feature_contract.json")
        self.assembly_order = json.loads(
            production_contract.read_text(encoding="utf-8"))["feature_order"]

    def infer_histories(self, index: pd.DataFrame, option: pd.DataFrame,
                        observations: pd.DataFrame) -> dict:
        row, audit = build_live_contract_row(
            index, option, observations, self.assembly_order)
        timestamp = str(audit["timestamp"])
        if row is None:
            return self._skip(timestamp, str(audit["reason"]))
        return self.infer_contract_row(row, timestamp, audit)

    def infer_contract_row(self, row: pd.DataFrame, timestamp, audit: dict) -> dict:
        """Score an already assembled causal row during chronological replay."""
        if int(row.iloc[0]["entry_signal"]) != -1:
            return self._skip(str(timestamp), "BULLISH_SETUP_NOT_PUT")
        put_row = row[self.contract["feature_order"]]
        probability = float(self.model.predict_proba(self.preprocessor.transform(put_row))[0, 1])
        ready = probability >= self.threshold
        return {
            "timestamp": str(timestamp), "action": "BUY PE" if ready else "NO TRADE",
            "decision": "PAPER_SIGNAL" if ready else "SKIP", "probability": probability,
            "threshold": self.threshold, "reason": (
                "PUT_SHADOW_THRESHOLD_PASSED" if ready else "PROBABILITY_BELOW_PUT_THRESHOLD"),
            "strategy": str(audit.get("strategy", "BEARISH_STAGE1")),
            "model_version": self.contract["model_version"],
            "instrument_scope": "ATM_PUT_PAPER_ONLY", "paper_only": True,
        }

    def _skip(self, timestamp: str, reason: str) -> dict:
        return {
            "timestamp": timestamp, "action": "NO TRADE", "decision": "SKIP",
            "probability": None, "threshold": self.threshold, "reason": reason,
            "strategy": "NONE", "model_version": self.contract["model_version"],
            "instrument_scope": "ATM_PUT_PAPER_ONLY", "paper_only": True,
        }
