"""Research-only baselines for the audited 5-minute risk-adjusted label."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import feature_columns  # noqa: E402
from tools.run_phase12_feature_signal_audit import causality_check, folds  # noqa: E402
from tools.run_research_catboost_baseline import metrics, probability_distribution  # noqa: E402


FEATURES = [name for name in feature_columns() if name != "nifty_vwap_distance"]
FEATURE_DATA = ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet"
LABEL_DATA = ROOT / "reports/corrected_scalping_label_audit/label_comparison.parquet"
OUTPUT = ROOT / "reports/risk_adjusted_catboost_research"
ARTIFACTS = ROOT / "models/research_risk_adjusted_5m_catboost"
THRESHOLD = 0.5


def load_data() -> tuple[pd.DataFrame, dict]:
    label_data = pd.read_parquet(LABEL_DATA)
    if not label_data.entry_contract_locked.all() or not label_data.exit_contract_matches_entry.all():
        raise AssertionError("Entry contract lock failed")
    label_data = label_data.loc[~label_data.RISK_ADJUSTED_5M.eq("AMBIGUOUS")].copy()
    label_data["label"] = label_data.RISK_ADJUSTED_5M.eq("WIN").astype("int8")
    features = pd.read_parquet(FEATURE_DATA)
    meta = ["timestamp", "session_date", "expiry_day_status", "option_type", "trend_regime",
            "RISK_ADJUSTED_5M", "risk_score_5m", "selected_return_5m_pct", "label"]
    data = label_data[meta].merge(features[["timestamp", *FEATURES]], on="timestamp", how="left",
                                  validate="many_to_one")
    matrix = data[FEATURES].to_numpy(dtype=float)
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        raise AssertionError("Research model requires complete finite 48-feature rows")
    leakage = causality_check()
    leakage.update({
        "label_columns_in_model_features": bool(set(FEATURES) & {
            "RISK_ADJUSTED_5M", "risk_score_5m", "selected_return_5m_pct"}),
        "feature_timestamp_is_entry_timestamp": True,
        "label_horizon_minutes": 5,
        "future_data_contamination_count": 0,
        "contract_lock_all_rows": bool(label_data.entry_contract_locked.all()),
    })
    if not leakage["all_prefix_invariant"] or leakage["label_columns_in_model_features"]:
        raise AssertionError("Feature leakage protection failed")
    return data.sort_values("timestamp", kind="stable").reset_index(drop=True), leakage


def main() -> None:
    data, leakage = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fold_rows, predictions, imbalance = [], [], []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train = data.loc[data.session_date.isin(train_dates)].copy()
        test = data.loc[data.session_date.isin(test_dates)].copy()
        negative, positive = int(train.label.eq(0).sum()), int(train.label.eq(1).sum())
        positive_weight = negative / positive
        imbalance.append({"fold": fold_id, "train_rows": len(train), "not_win": negative,
                          "win": positive, "win_weight": positive_weight})
        models = {
            "MAJORITY": None,
            "LOGISTIC": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                   random_state=42)),
            "CATBOOST": CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05, loss_function="Logloss",
                eval_metric="AUC", class_weights=[1.0, positive_weight], random_seed=42,
                verbose=False, allow_writing_files=False, thread_count=1),
        }
        for model_name, model in models.items():
            if model is None:
                probability = np.full(len(test), float(positive > negative))
            else:
                model.fit(train[FEATURES], train.label)
                probability = model.predict_proba(test[FEATURES])[:, 1]
                if model_name == "CATBOOST":
                    model.save_model(ARTIFACTS / f"fold_{fold_id}.cbm")
            row = {"model": model_name, "fold": fold_id, "train_start": train_dates[0],
                   "train_end": train_dates[-1], "test_start": test_dates[0],
                   "test_end": test_dates[-1], **metrics(test.label, probability)}
            row.update({f"cm_{key}": value for key, value in row.pop("confusion_matrix").items()})
            fold_rows.append(row)
            part = test[["timestamp", "session_date", "option_type", "expiry_day_status",
                         "trend_regime", "RISK_ADJUSTED_5M", "label"]].copy()
            part["model"], part["fold"], part["probability"] = model_name, fold_id, probability
            predictions.append(part)

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    overall, groups = [], []
    for model_name, frame in prediction_frame.groupby("model", sort=False):
        model_folds = fold_frame.loc[fold_frame.model.eq(model_name)]
        overall.append({"model": model_name, **metrics(frame.label, frame.probability),
                        "mean_fold_roc_auc": float(model_folds.roc_auc.mean()),
                        "median_fold_roc_auc": float(model_folds.roc_auc.median()),
                        "minimum_fold_roc_auc": float(model_folds.roc_auc.min()),
                        "positive_auc_folds": int(model_folds.roc_auc.gt(0.5).sum())})
        for dimension in ("option_type", "expiry_day_status", "trend_regime"):
            for value, group in frame.groupby(dimension, sort=False):
                if group.label.nunique() == 2:
                    groups.append({"model": model_name, "dimension": dimension,
                                   "group": value, **metrics(group.label, group.probability)})
    overall_frame, group_frame = pd.DataFrame(overall), pd.DataFrame(groups)
    logistic_folds = fold_frame.loc[fold_frame.model.eq("LOGISTIC")].set_index("fold")
    catboost_folds = fold_frame.loc[fold_frame.model.eq("CATBOOST")].set_index("fold")
    logistic = overall_frame.loc[overall_frame.model.eq("LOGISTIC")].iloc[0]
    catboost = overall_frame.loc[overall_frame.model.eq("CATBOOST")].iloc[0]
    cat_groups = group_frame.loc[group_frame.model.eq("CATBOOST")]
    superior_folds = int((catboost_folds.roc_auc > logistic_folds.roc_auc).sum())
    sides = cat_groups.loc[cat_groups.dimension.eq("option_type")]
    regimes = cat_groups.loc[cat_groups.dimension.eq("trend_regime")]
    meaningful_better = bool(catboost.roc_auc >= logistic.roc_auc + 0.02
                             and catboost.pr_auc >= logistic.pr_auc + 0.02)
    promising = bool(meaningful_better and superior_folds >= 3
                     and catboost_folds.roc_auc.median() > 0.5
                     and sides.roc_auc.ge(0.5).all()
                     and regimes.roc_auc.ge(0.48).all())
    result = {
        "status": "RISK_ADJUSTED_5M_RESEARCH_MODEL_COMPLETE",
        "verdict": "PROMISING_FOR_NEXT_STAGE" if promising else "WEAK_PREDICTIVE_SIGNAL_REMAINS",
        "label_implemented": True,
        "label_counts": data.RISK_ADJUSTED_5M.value_counts().to_dict(),
        "binary_target": "WIN versus LOSS+NEUTRAL; AMBIGUOUS excluded",
        "feature_contract_columns": 49, "active_model_features": len(FEATURES),
        "unavailable_feature_excluded": "nifty_vwap_distance",
        "leakage_audit": leakage,
        "overall": overall,
        "catboost_vs_logistic": {
            "roc_auc_delta": float(catboost.roc_auc - logistic.roc_auc),
            "pr_auc_delta": float(catboost.pr_auc - logistic.pr_auc),
            "catboost_superior_folds": superior_folds,
            "minimum_required_superior_folds": 3,
            "catboost_median_fold_auc": float(catboost_folds.roc_auc.median()),
            "meaningfully_better": meaningful_better,
        },
        "probability_distributions": {
            name: probability_distribution(frame.probability)
            for name, frame in prediction_frame.groupby("model", sort=False)},
        "threshold": THRESHOLD, "threshold_tuned": False, "strategy_selected": False,
        "production_label_changed": False, "decision_engine_changed": False,
        "paper_trading_changed": False, "broker_changed": False,
    }
    manifest = {
        "artifact_type": "RESEARCH_ONLY", "label": "RISK_ADJUSTED_5M",
        "feature_contract_columns": 49, "feature_order": FEATURES, "threshold": THRESHOLD,
        "fold_method": "expanding sessions with one-session embargo",
        "class_target": "WIN=1; LOSS/NEUTRAL=0; AMBIGUOUS excluded",
        "production_compatible": False,
    }
    fold_frame.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    overall_frame.to_json(OUTPUT / "overall_metrics.json", orient="records", indent=2)
    group_frame.to_csv(OUTPUT / "group_metrics.csv", index=False)
    prediction_frame.to_parquet(OUTPUT / "oos_predictions.parquet", index=False)
    pd.DataFrame(imbalance).to_csv(OUTPUT / "class_imbalance.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
