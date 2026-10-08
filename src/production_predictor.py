"""Strict inference interface for the frozen Step-23 CatBoost candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd


class ContractViolation(ValueError):
    """Raised when inference input violates the immutable feature contract."""


class ProductionPredictor:
    def __init__(self, package_dir: str | Path | None = None):
        self.package_dir = Path(package_dir or Path(__file__).resolve().parent)
        self.contract = json.loads((self.package_dir / "feature_contract.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((self.package_dir / "production_metadata.json").read_text(encoding="utf-8"))
        self.preprocessor = joblib.load(self.package_dir / "preprocessor.joblib")
        self.model = CatBoostClassifier()
        self.model.load_model(self.package_dir / "model.cbm", format="cbm")
        self._verify_artifacts()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _verify_artifacts(self) -> None:
        hashes_path = self.package_dir / "model_hashes.json"
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        for name, expected in hashes.get("outputs", {}).items():
            path = self.package_dir / name
            if not path.exists() or self._sha256(path) != expected:
                raise RuntimeError(f"Production artifact hash mismatch: {name}")
        if self.metadata["model_version"] != self.contract["model_version"]:
            raise RuntimeError("Model and feature-contract versions differ")

    def _frame(self, features) -> pd.DataFrame:
        if isinstance(features, pd.Series):
            frame = features.to_frame().T
        elif isinstance(features, dict):
            frame = pd.DataFrame([features], columns=list(features.keys()))
        elif isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            raise ContractViolation("Features must be a dict, Series, or DataFrame")
        expected, received = self.contract["feature_order"], frame.columns.tolist()
        missing = [name for name in expected if name not in received]
        unknown = [name for name in received if name not in expected]
        if missing:
            raise ContractViolation(f"Missing required features: {missing}")
        if unknown:
            raise ContractViolation(f"Unknown features: {unknown}")
        if received != expected:
            raise ContractViolation("Feature columns are reordered")
        if len(frame) != 1:
            raise ContractViolation("predict(features) accepts exactly one observation")
        for field in self.contract["fields"]:
            name = field["name"]
            if field["kind"] == "numeric":
                converted = pd.to_numeric(frame[name], errors="coerce")
                invalid = frame[name].notna() & converted.isna()
                if invalid.any() or np.isinf(converted.dropna()).any():
                    raise ContractViolation(f"Incompatible numeric feature: {name}")
                frame[name] = converted
            else:
                frame[name] = frame[name].astype("string")
        return frame

    def predict(self, features) -> dict:
        frame = self._frame(features)
        transformed = self.preprocessor.transform(frame)
        if not np.isfinite(transformed).all():
            raise ContractViolation("Preprocessing produced non-finite model inputs")
        probability = float(self.model.predict_proba(transformed)[0, 1])
        threshold = float(self.metadata["approved_probability_threshold"])
        decision = "TRADE" if probability >= threshold else "SKIP"
        return {
            "probability": probability,
            "decision": decision,
            "confidence": probability,
            "reason": (
                "PROBABILITY_AT_OR_ABOVE_APPROVED_THRESHOLD"
                if decision == "TRADE" else "PROBABILITY_BELOW_APPROVED_THRESHOLD"
            ),
            "model_version": self.metadata["model_version"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


_DEFAULT: ProductionPredictor | None = None


def predict(features) -> dict:
    """Simple package-level API requested by Step 23."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ProductionPredictor(Path(__file__).resolve().parent)
    return _DEFAULT.predict(features)

