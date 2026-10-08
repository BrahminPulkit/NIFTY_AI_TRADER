"""Walk-forward training and complete artifact packaging for corrected CE/PE models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.model_benchmark import BenchmarkProtocol, build_pipeline
from src.option_contract_pipeline import artifact_manifest
from src.scalping_signal_engine import SCALPING_FEATURE_VERSION


@dataclass(frozen=True)
class TrainingConfig:
    minimum_model_probability: float = 0.50
    minimum_final_signals: int = 20
    minimum_final_expectancy_r: float = 0.0
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20


def train_side_model(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame,
                     option_type: str, output: str | Path,
                     config: TrainingConfig = TrainingConfig()) -> dict:
    """Train on complete sessions and leave the latest sessions untouched."""
    side = option_type.upper()
    if side not in {"CE", "PE"}:
        raise ValueError("option_type must be CE or PE")
    required = {"timestamp", "realized_r", "outcome", "contract_segment_id"}
    if missing := sorted(required.difference(metadata.columns)):
        raise ValueError(f"Training metadata missing: {missing}")
    if len(X) != len(y) or len(X) != len(metadata) or X.columns.duplicated().any():
        raise ValueError("Feature, label and metadata rows must be aligned and unique")
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True).copy()
    meta = metadata.reset_index(drop=True).copy()
    eligible = meta.outcome.ne("AMBIGUOUS")
    X, y, meta = X.loc[eligible].reset_index(drop=True), y.loc[eligible].reset_index(drop=True), meta.loc[eligible].reset_index(drop=True)
    if X.empty:
        raise ValueError("No training-eligible rows after excluding AMBIGUOUS labels")
    sessions = pd.to_datetime(meta.timestamp).dt.strftime("%Y-%m-%d")
    unique = pd.Index(sessions.unique())
    train_cut = max(1, int(len(unique) * config.train_fraction))
    calibration_cut = max(train_cut + 1, int(len(unique) * (config.train_fraction + config.calibration_fraction)))
    train_mask = sessions.isin(unique[:train_cut]); calibration_mask = sessions.isin(unique[train_cut:calibration_cut])
    test_mask = sessions.isin(unique[calibration_cut:])
    if min(int(train_mask.sum()), int(calibration_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("Insufficient complete sessions for train/calibration/test")
    protocol = BenchmarkProtocol(n_splits=4, threshold=config.minimum_model_probability, random_state=42)
    pipeline = build_pipeline("CATBOOST", X.loc[train_mask], protocol)
    pipeline.fit(X.loc[train_mask], y.loc[train_mask])
    test_probability = pipeline.predict_proba(X.loc[test_mask])[:, 1]
    selected = test_probability >= config.minimum_model_probability
    test_y = y.loc[test_mask].to_numpy()
    test_r = pd.to_numeric(meta.loc[test_mask, "realized_r"], errors="coerce").to_numpy()
    signals = int(selected.sum())
    expectancy = float(np.nanmean(test_r[selected])) if signals else float("nan")
    result = {
        "status": ("FORWARD_PAPER_CANDIDATE" if signals >= config.minimum_final_signals
                   and np.isfinite(expectancy) and expectancy > config.minimum_final_expectancy_r
                   else "RESEARCH_ONLY_VALIDATION_FAILED"),
        "paper_only": True, "broker_orders_enabled": False, "option_type": side,
        "threshold": config.minimum_model_probability, "final_signals": signals,
        "final_precision": float(precision_score(test_y, selected, zero_division=0)),
        "final_recall": float(recall_score(test_y, selected, zero_division=0)),
        "final_expectancy_r": expectancy, "feature_version": SCALPING_FEATURE_VERSION,
        "train_rows": int(train_mask.sum()), "calibration_rows": int(calibration_mask.sum()),
        "final_test_rows": int(test_mask.sum()), "configuration": asdict(config),
    }
    destination = Path(output); destination.mkdir(parents=True, exist_ok=True)
    final_pipeline = build_pipeline("CATBOOST", X.loc[train_mask | calibration_mask], protocol)
    final_pipeline.fit(X.loc[train_mask | calibration_mask], y.loc[train_mask | calibration_mask])
    final_pipeline.named_steps["model"].save_model(destination / "model.cbm", format="cbm")
    joblib.dump(final_pipeline.named_steps["preprocessor"], destination / "preprocessor.joblib", compress=3)
    (destination / "feature_contract.json").write_text(json.dumps({
        "model_version": "scalping_contract_v1", "option_type": side,
        "feature_order": X.columns.tolist(), "threshold": config.minimum_model_probability,
    }, indent=2), encoding="utf-8")
    manifest = artifact_manifest(X.columns.tolist(), SCALPING_FEATURE_VERSION,
                                 {"pipeline": "model_benchmark_column_transformer"}, asdict(config))
    (destination / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
