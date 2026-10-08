"""Research-only CatBoost baseline on the frozen repaired 48-feature matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import feature_columns  # noqa: E402
from tools.run_phase12_feature_signal_audit import folds  # noqa: E402


FEATURES = [name for name in feature_columns() if name != "nifty_vwap_distance"]
DATASET = ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet"
OUTPUT = ROOT / "reports/research_catboost_baseline"
ARTIFACTS = ROOT / "models/research_phase14_catboost"
THRESHOLD = .5


def metrics(y_true, probability, threshold: float = THRESHOLD) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "rows": len(y), "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
                             "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
    }


def probability_distribution(values: pd.Series) -> list[dict]:
    bins = np.linspace(0, 1, 11)
    groups = pd.cut(values, bins=bins, include_lowest=True, right=False)
    counts = groups.value_counts(sort=False)
    return [{"bin": str(interval), "count": int(count)} for interval, count in counts.items()]


def load_data() -> pd.DataFrame:
    features = pd.read_parquet(DATASET)
    labels = pd.concat([
        pd.read_parquet(ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"),
        pd.read_parquet(ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"),
    ], ignore_index=True)
    if not labels.entry_contract_locked.all() or not labels.exit_contract_matches_entry.all():
        raise AssertionError("Contract lock is not valid")
    labels = labels.loc[labels.actual_exit_reason.isin(["WIN", "LOSS"])].sort_values(
        "timestamp", kind="stable").drop_duplicates(
            ["timestamp", "expiry", "strike", "option_type"], keep="first")
    labels["label"] = labels.actual_exit_reason.eq("WIN").astype("int8")
    meta = ["timestamp", "session_date", "expiry_day_status", "option_type", "trend_regime", "label"]
    data = labels[meta].merge(features[["timestamp", *FEATURES]], on="timestamp", how="left",
                              validate="many_to_one")
    if data[FEATURES].isna().any().any() or np.isinf(data[FEATURES].to_numpy()).any():
        raise AssertionError("Research labels require a complete finite 48-feature vector")
    phase13 = json.loads((ROOT / "reports/phase13_contract_segment_features/audit.json").read_text())
    if not phase13["leakage_protection"]["all_prefix_invariant"]:
        raise AssertionError("Prefix invariance is not valid")
    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


def main() -> None:
    data = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fold_rows, predictions, imbalance = [], [], []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train = data.loc[data.session_date.isin(train_dates)].copy()
        test = data.loc[data.session_date.isin(test_dates)].copy()
        negative, positive = int(train.label.eq(0).sum()), int(train.label.eq(1).sum())
        positive_weight = negative / positive
        imbalance.append({"fold": fold_id, "train_rows": len(train), "loss": negative,
                          "win": positive, "win_weight": positive_weight})
        models = {
            "MAJORITY": None,
            "LOGISTIC": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                   random_state=42)),
            "CATBOOST": CatBoostClassifier(
                iterations=300, depth=6, learning_rate=.05, loss_function="Logloss",
                eval_metric="AUC", class_weights=[1.0, positive_weight], random_seed=42,
                verbose=False, allow_writing_files=False, thread_count=1),
        }
        for name, model in models.items():
            if name == "MAJORITY":
                majority = int(positive > negative)
                probability = np.full(len(test), float(majority))
            else:
                model.fit(train[FEATURES], train.label)
                probability = model.predict_proba(test[FEATURES])[:, 1]
                if name == "CATBOOST":
                    model.save_model(ARTIFACTS / f"fold_{fold_id}.cbm")
            row = {"model": name, "fold": fold_id, "train_start": train_dates[0],
                   "train_end": train_dates[-1], "test_start": test_dates[0],
                   "test_end": test_dates[-1], **metrics(test.label, probability)}
            row.update({f"cm_{key}": value for key, value in row.pop("confusion_matrix").items()})
            fold_rows.append(row)
            part = test[["timestamp", "session_date", "option_type", "expiry_day_status",
                         "trend_regime", "label"]].copy()
            part["model"], part["fold"], part["probability"] = name, fold_id, probability
            predictions.append(part)
    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    overall, groups = [], []
    for name, frame in prediction_frame.groupby("model", sort=False):
        overall.append({"model": name, **metrics(frame.label, frame.probability),
                        "mean_fold_roc_auc": float(fold_frame.loc[
                            fold_frame.model.eq(name), "roc_auc"].mean()),
                        "minimum_fold_roc_auc": float(fold_frame.loc[
                            fold_frame.model.eq(name), "roc_auc"].min()),
                        "positive_auc_folds": int(fold_frame.loc[
                            fold_frame.model.eq(name), "roc_auc"].gt(.5).sum())})
        for dimension in ("option_type", "expiry_day_status", "trend_regime"):
            for value, group in frame.groupby(dimension, sort=False):
                if group.label.nunique() < 2:
                    continue
                groups.append({"model": name, "dimension": dimension, "group": value,
                               **metrics(group.label, group.probability)})
    overall_frame = pd.DataFrame(overall)
    group_frame = pd.DataFrame(groups)
    logistic = fold_frame.loc[fold_frame.model.eq("LOGISTIC")].set_index("fold")
    catboost = fold_frame.loc[fold_frame.model.eq("CATBOOST")].set_index("fold")
    cat_overall = overall_frame.loc[overall_frame.model.eq("CATBOOST")].iloc[0]
    log_overall = overall_frame.loc[overall_frame.model.eq("LOGISTIC")].iloc[0]
    cat_groups = group_frame.loc[group_frame.model.eq("CATBOOST")]
    stable_improvement = bool(
        cat_overall.roc_auc >= log_overall.roc_auc + .03
        and cat_overall.pr_auc >= log_overall.pr_auc + .03
        and int((catboost.roc_auc > logistic.roc_auc).sum()) >= 4
        and catboost.roc_auc.min() >= .52
        and cat_groups.roc_auc.ge(.52).all())
    weak = bool(cat_overall.roc_auc < .55 or not stable_improvement)
    manifest = {
        "artifact_type": "RESEARCH_ONLY", "feature_count": 48,
        "feature_order": FEATURES, "threshold": THRESHOLD,
        "fold_method": "expanding sessions with one-session embargo",
        "hyperparameters": {"iterations": 300, "depth": 6, "learning_rate": .05,
                            "class_weights": "fold-specific balanced", "random_seed": 42},
        "production_compatible": False,
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "status": ("CURRENT_DATA_LABEL_COMBINATION_HAS_WEAK_PREDICTIVE_SIGNAL" if weak
                   else "CATBOOST_STABLE_BASELINE_IMPROVEMENT"),
        "catboost_better_than_baseline": bool(cat_overall.roc_auc > log_overall.roc_auc),
        "stable_across_oos_folds": bool(int((catboost.roc_auc > logistic.roc_auc).sum()) >= 4),
        "stable_across_sides_and_regimes": bool(cat_groups.roc_auc.ge(.52).all()),
        "stable_improvement_gate": stable_improvement,
        "class_distribution": {"rows": len(data), "win": int(data.label.sum()),
                               "loss": int(data.label.eq(0).sum()),
                               "win_pct": float(data.label.mean() * 100)},
        "overall": overall, "probability_distributions": {
            name: probability_distribution(frame.probability)
            for name, frame in prediction_frame.groupby("model", sort=False)},
        "threshold_tuned": False, "strategy_selected": False,
        "production_changed": False, "paper_trading_changed": False,
    }
    fold_frame.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    overall_frame.to_json(OUTPUT / "overall_metrics.json", orient="records", indent=2)
    group_frame.to_csv(OUTPUT / "group_metrics.csv", index=False)
    prediction_frame.to_parquet(OUTPUT / "oos_predictions.parquet", index=False)
    pd.DataFrame(imbalance).to_csv(OUTPUT / "class_imbalance.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
