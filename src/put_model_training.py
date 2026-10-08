"""Leakage-controlled Stage-2 PUT candidate dataset and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_benchmark import BenchmarkProtocol, build_pipeline, metrics, run_walk_forward
from src.option_feature_pipeline import OPTION_FEATURES, build_option_features


TARGET_PCT = 0.05
HORIZON_MINUTES = 15
THRESHOLD = 0.90
SETUP_FIELDS = [
    "entry_signal", "trend_state", "structure_state", "momentum_state",
    "volatility_state", "session_state", "event_state",
    "breakout_quality_state", "candle_quality_state", "trend_strength_state",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def put_targets(option: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """Label same-strike +5% premium touches strictly after entry."""
    source = option.copy()
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    source = source.sort_values("timestamp", kind="stable").reset_index(drop=True)
    locations = pd.Series(source.index, index=source.timestamp)
    rows = []
    normalized = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert("Asia/Kolkata")
    for raw in normalized:
        if raw not in locations:
            continue
        index = int(locations.loc[raw])
        entry = source.iloc[index]
        future = source.iloc[index + 1:index + 1 + HORIZON_MINUTES]
        future = future.loc[
            future.timestamp.dt.date.eq(raw.date())
            & future.timestamp.le(raw + pd.Timedelta(minutes=HORIZON_MINUTES))
            & future.strike.eq(entry.strike)
        ]
        if future.empty:
            continue
        rows.append({
            "timestamp": raw, "entry_premium": float(entry.close),
            "entry_strike": float(entry.strike),
            "target_win": int(float(future.high.max()) >= float(entry.close) * (1 + TARGET_PCT)),
            "forward_rows": len(future),
        })
    return pd.DataFrame(rows)


def build_put_dataset(prediction: pd.DataFrame, option: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    prediction = prediction.copy()
    prediction["timestamp"] = pd.to_datetime(prediction.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    bearish = prediction.loc[prediction.entry_signal.eq(-1)].copy()
    option_features = build_option_features(option).rename(columns={
        name: f"option_{name}" for name in OPTION_FEATURES
    })
    keep = ["timestamp", *[f"option_{name}" for name in OPTION_FEATURES]]
    base_fields = [name for name in bearish if name.startswith("index_") and name != "index_vwap"]
    base_fields += SETUP_FIELDS
    features = bearish[["timestamp", *base_fields]].merge(
        option_features[keep], on="timestamp", how="inner", validate="one_to_one", sort=True)
    labels = put_targets(option, features.timestamp)
    merged = features.merge(labels, on="timestamp", how="inner", validate="one_to_one", sort=True)
    metadata = merged[["timestamp", "entry_premium", "entry_strike", "forward_rows"]].copy()
    y = merged.pop("target_win").astype("int8")
    X = merged.drop(columns=["timestamp", "entry_premium", "entry_strike", "forward_rows"])
    if X.columns.duplicated().any() or X.isin([np.inf, -np.inf]).any().any():
        raise ValueError("PUT feature matrix violates the finite unique-column contract")
    return X, y, metadata


def train_and_validate(prediction_path: Path, option_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_parquet(prediction_path)
    option = pd.read_parquet(option_path)
    X, y, metadata = build_put_dataset(prediction, option)
    protocol = BenchmarkProtocol(n_splits=4, threshold=THRESHOLD, random_state=42)
    oof, folds, _ = run_walk_forward(X, y, metadata.assign(
        trend_regime="BEARISH", volatility_regime="UNKNOWN",
        session_phase="UNKNOWN", current_strategy="PUT_SETUP"), "CATBOOST", protocol)
    summary = metrics(oof.target, oof.probability.to_numpy(), THRESHOLD)
    selected = oof.loc[oof.probability.ge(THRESHOLD)].copy()
    selected["year"] = pd.to_datetime(selected.timestamp).dt.year
    signal_years = int(selected.year.nunique()) if len(selected) else 0
    fold_signal_counts = (folds["tp"] + folds["fp"]).astype(int)
    gates = {
        "minimum_oof_predictions_30": int(len(selected)) >= 30,
        "minimum_precision_0_70": float(summary["precision"]) >= .70,
        "minimum_signal_years_3": signal_years >= 3,
        "all_folds_have_both_classes": bool(folds.positives.gt(0).all() and (folds.observations-folds.positives).gt(0).all()),
        "minimum_5_signals_each_fold": bool(fold_signal_counts.ge(5).all()),
        "minimum_0_60_precision_each_fold": bool(folds.precision.ge(.60).all()),
    }
    promoted = all(gates.values())
    pipeline = build_pipeline("CATBOOST", X, protocol)
    pipeline.fit(X, y)
    pipeline.named_steps["model"].save_model(output / "model.cbm", format="cbm")
    joblib.dump(pipeline.named_steps["preprocessor"], output / "preprocessor.joblib", compress=3)
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    contract = {
        "model_version": "stage2_put_candidate_v1.0.0", "direction": "BUY_PUT",
        "target": "SAME_STRIKE_ATM_PUT_5PCT_WITHIN_15_MINUTES",
        "threshold": THRESHOLD, "feature_order": X.columns.tolist(),
        "feature_count": len(X.columns), "missing_fields": "REJECT",
        "unknown_fields": "REJECT", "training_rows": len(X),
    }
    (output / "feature_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    result = {
        "status": "PROMOTED_PAPER_CANDIDATE" if promoted else "RESEARCH_ONLY_VALIDATION_FAILED",
        "live_enabled": False, "paper_only": True, "gates": gates,
        "training_rows": len(X), "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
        "oof_rows": len(oof), "threshold_signals": len(selected), "signal_years": signal_years,
        "fold_signal_counts": fold_signal_counts.tolist(),
        "metrics": summary, "folds": folds.to_dict("records"),
        "dataset_sha256": sha256(option_path), "prediction_source_sha256": sha256(prediction_path),
    }
    (output / "validation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
