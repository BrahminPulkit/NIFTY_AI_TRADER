"""Stage-2 V1 BUY-CALL outcome modelling with strict walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.feature_pipeline import FEATURES as INDEX_FEATURES
from src.option_feature_pipeline import OPTION_FEATURES


MODEL_VERSION = "stage2_call_v1"
MODEL_A_FEATURES = [f"index_{name}" for name in INDEX_FEATURES]
MODEL_B_FEATURES = MODEL_A_FEATURES + [f"option_{name}" for name in OPTION_FEATURES]
TARGET = "target_win"
FORBIDDEN_INPUTS = {
    "final_outcome", "exit_timestamp", "exit_price", "exit_candle", "exit_reason",
    "mfe_points", "mae_points", "mfe_r", "mae_r", "realized_r",
    "expected_r_contribution", "stop_price", "target_price", "risk_points",
}


@dataclass(frozen=True)
class TrainingProtocol:
    n_splits: int = 4
    threshold: float = 0.5
    random_state: int = 42


PROTOCOL = TrainingProtocol()


def prepare_call_dataset(aligned: pd.DataFrame, outcomes: pd.DataFrame):
    """Return identical CALL rows for Index-only and Index+ATM-CALL models."""
    required_outcomes = {"timestamp", "direction", "trade_allowed", "final_outcome",
                         "year", "trend_state", "session_segment"}
    missing = required_outcomes.difference(outcomes.columns)
    if missing:
        raise ValueError(f"Missing Stage-2 fields: {sorted(missing)}")
    missing_features = set(MODEL_B_FEATURES).difference(aligned.columns)
    if missing_features:
        raise ValueError(f"Missing aligned features: {sorted(missing_features)}")
    call = outcomes.loc[outcomes.trade_allowed.eq(1) & outcomes.direction.eq("LONG")].copy()
    call["timestamp"] = pd.to_datetime(call.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    base = aligned[["timestamp", *MODEL_B_FEATURES]].copy()
    base["timestamp"] = pd.to_datetime(base.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    data = call.merge(base, on="timestamp", how="inner", validate="one_to_one", sort=True)
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError("Stage-2 CALL rows must be unique and chronological")
    data[TARGET] = data.final_outcome.eq("WIN").astype("int8")
    if FORBIDDEN_INPUTS.intersection(MODEL_B_FEATURES):
        raise AssertionError("Target-side information entered the model contract")
    metadata = data[["timestamp", "year", "trend_state", "session_segment", "final_outcome"]].copy()
    return data[MODEL_A_FEATURES].astype(float), data[MODEL_B_FEATURES].astype(float), data[TARGET], metadata


def walk_forward_splits(rows: int, protocol: TrainingProtocol = PROTOCOL):
    if rows <= protocol.n_splits:
        raise ValueError("Insufficient rows for walk-forward validation")
    return list(TimeSeriesSplit(n_splits=protocol.n_splits).split(np.arange(rows)))


def build_model(y_train: pd.Series, protocol: TrainingProtocol = PROTOCOL) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    if positives == 0:
        raise ValueError("Training fold has no BUY-CALL wins")
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=3, learning_rate=0.03, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
        scale_pos_weight=negatives / positives, random_state=protocol.random_state,
        n_jobs=4, tree_method="hist",
    )


def classification_metrics(y_true: pd.Series, probability: np.ndarray,
                           threshold: float = 0.5) -> dict:
    prediction = (probability >= threshold).astype(int)
    both = pd.Series(y_true).nunique() == 2
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "observations": len(y_true), "positives": int(np.sum(y_true)),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if both else np.nan,
        "pr_auc": average_precision_score(y_true, probability) if both else np.nan,
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }


def run_walk_forward(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame,
                     model_name: str, protocol: TrainingProtocol = PROTOCOL):
    predictions, folds = [], []
    for fold, (train, test) in enumerate(walk_forward_splits(len(X), protocol), 1):
        if train[-1] >= test[0]:
            raise AssertionError("Walk-forward train/test overlap")
        model = build_model(y.iloc[train], protocol)
        model.fit(X.iloc[train], y.iloc[train])
        probability = model.predict_proba(X.iloc[test])[:, 1]
        metrics = classification_metrics(y.iloc[test], probability, protocol.threshold)
        folds.append({"model": model_name, "fold": fold,
                      "train_start": metadata.timestamp.iloc[train[0]],
                      "train_end": metadata.timestamp.iloc[train[-1]],
                      "test_start": metadata.timestamp.iloc[test[0]],
                      "test_end": metadata.timestamp.iloc[test[-1]],
                      "train_rows": len(train), "train_positives": int(y.iloc[train].sum()),
                      **metrics})
        piece = metadata.iloc[test].copy()
        piece["target"] = y.iloc[test].to_numpy()
        piece["probability"] = probability
        piece["prediction"] = (probability >= protocol.threshold).astype("int8")
        piece["fold"] = fold; piece["model"] = model_name
        predictions.append(piece)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def grouped_performance(predictions: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for value, data in predictions.groupby(group, sort=True, dropna=False):
        rows.append({"model": data.model.iat[0], group: value,
                     **classification_metrics(data.target, data.probability)})
    return pd.DataFrame(rows)


def raw_calibration(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    if predictions.target.nunique() < 2:
        return pd.DataFrame(columns=["model", "mean_predicted_probability", "observed_win_rate"])
    observed, predicted = calibration_curve(predictions.target, predictions.probability,
                                             n_bins=bins, strategy="quantile")
    return pd.DataFrame({"model": predictions.model.iat[0],
                         "mean_predicted_probability": predicted,
                         "observed_win_rate": observed})
