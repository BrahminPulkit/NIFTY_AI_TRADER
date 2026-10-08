"""Step 20B: fixed-protocol institutional model benchmark research."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


MODEL_NAMES = (
    "LOGISTIC_REGRESSION",
    "RANDOM_FOREST",
    "XGBOOST",
    "LIGHTGBM",
    "CATBOOST",
)


@dataclass(frozen=True)
class BenchmarkProtocol:
    n_splits: int = 4
    threshold: float = 0.5
    random_state: int = 42


PROTOCOL = BenchmarkProtocol()


def prepare_benchmark_data(
    prediction: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    if "timestamp" not in prediction or not {"timestamp", "research_target"}.issubset(target):
        raise ValueError("Timestamp and approved research target are required")
    data = prediction.copy()
    labels = target.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    labels["timestamp"] = pd.to_datetime(labels.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    merged = data.merge(labels, on="timestamp", how="inner", validate="one_to_one", sort=True)
    if merged.timestamp.duplicated().any() or not merged.timestamp.is_monotonic_increasing:
        raise ValueError("Benchmark rows must be chronological and unique")
    y = merged.pop("research_target").astype("int8")
    metadata = merged[[
        "timestamp", "trend_regime", "volatility_regime",
        "session_phase", "current_strategy",
    ]].copy()
    metadata["market_regime"] = (
        metadata.trend_regime.astype(str) + "|" + metadata.volatility_regime.astype(str)
    )
    X = merged.drop(columns="timestamp")
    all_null = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_null)
    if any(token in column.lower() for column in X.columns for token in (
        "realized", "realised", "mfe", "mae", "outcome", "exit", "label", "target"
    )):
        raise AssertionError("Forbidden target-side field entered benchmark features")
    return X, y, metadata, all_null


def chronological_splits(rows: int, protocol: BenchmarkProtocol = PROTOCOL):
    splits = list(TimeSeriesSplit(n_splits=protocol.n_splits).split(np.arange(rows)))
    if any(train[-1] >= test[0] for train, test in splits):
        raise AssertionError("Invalid chronological split")
    return splits


def _preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical = [column for column in X.columns if column not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def build_pipeline(
    model_name: str, X: pd.DataFrame, protocol: BenchmarkProtocol = PROTOCOL
) -> Pipeline:
    if model_name == "LOGISTIC_REGRESSION":
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=protocol.random_state
        )
        scale = True
    elif model_name == "RANDOM_FOREST":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=protocol.random_state, n_jobs=-1,
        )
        scale = False
    elif model_name == "XGBOOST":
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=5.0, eval_metric="logloss",
            random_state=protocol.random_state, n_jobs=4, tree_method="hist",
        )
        scale = False
    elif model_name == "LIGHTGBM":
        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=31,
            max_depth=-1, min_child_samples=20, reg_alpha=0.5, reg_lambda=5.0,
            random_state=protocol.random_state, n_jobs=4, verbosity=-1,
        )
        scale = False
    elif model_name == "CATBOOST":
        model = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.03, loss_function="Logloss",
            random_seed=protocol.random_state, verbose=False, thread_count=4,
            allow_writing_files=False,
        )
        scale = False
    else:
        raise ValueError(f"Unsupported benchmark model: {model_name}")
    return Pipeline([("preprocessor", _preprocessor(X, scale)), ("model", model)])


def metrics(y: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict:
    prediction = (probability >= threshold).astype("int8")
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "observations": len(y),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    model_name: str,
    protocol: BenchmarkProtocol = PROTOCOL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Pipeline]]:
    predictions, fold_rows, fitted = [], [], []
    for fold, (train, test) in enumerate(chronological_splits(len(X), protocol), 1):
        pipeline = build_pipeline(model_name, X, protocol)
        start = perf_counter()
        pipeline.fit(X.iloc[train], y.iloc[train])
        training_seconds = perf_counter() - start
        start = perf_counter()
        probability = pipeline.predict_proba(X.iloc[test])[:, 1]
        inference_seconds = perf_counter() - start
        fold_rows.append({
            "model": model_name, "fold": fold,
            "train_start": metadata.timestamp.iloc[train[0]],
            "train_end": metadata.timestamp.iloc[train[-1]],
            "test_start": metadata.timestamp.iloc[test[0]],
            "test_end": metadata.timestamp.iloc[test[-1]],
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            **metrics(y.iloc[test], probability, protocol.threshold),
        })
        piece = metadata.iloc[test].copy()
        piece["model"] = model_name
        piece["target"] = y.iloc[test].to_numpy()
        piece["probability"] = probability
        piece["prediction"] = (probability >= protocol.threshold).astype("int8")
        predictions.append(piece)
        fitted.append(pipeline)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows), fitted


def grouped_performance(predictions: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for (model, value), data in predictions.groupby(["model", group], dropna=False):
        if data.target.nunique() < 2:
            roc, pr = np.nan, np.nan
        else:
            roc = roc_auc_score(data.target, data.probability)
            pr = average_precision_score(data.target, data.probability)
        rows.append({
            "model": model, group: value,
            **metrics(data.target, data.probability, 0.5),
            "roc_auc": roc, "pr_auc": pr,
        })
    return pd.DataFrame(rows)


def transformed_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if model_name == "LOGISTIC_REGRESSION":
        values = np.abs(model.coef_[0])
    else:
        values = np.asarray(model.feature_importances_)
    return pd.DataFrame({"transformed_feature": names, "importance": values})

