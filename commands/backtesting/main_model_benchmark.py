"""Execute Step 20B fixed-protocol model benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from src.model_benchmark import (
    MODEL_NAMES,
    PROTOCOL,
    grouped_performance,
    metrics,
    prepare_benchmark_data,
    run_walk_forward,
    transformed_importance,
)


PREDICTION = Path("data/prediction/prediction_dataset.parquet")
DECISIONS = Path("reports/strategy_decision/decision_matrix.csv")
REPORTS = Path("reports/model_benchmark")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target() -> pd.DataFrame:
    decisions = pd.read_csv(DECISIONS, usecols=["timestamp", "realized_expansion_hit"])
    return decisions.rename(columns={"realized_expansion_hit": "research_target"})


def main() -> None:
    prediction = pd.read_parquet(PREDICTION)
    X, y, metadata, excluded = prepare_benchmark_data(prediction, _target())
    all_predictions, all_folds, importance_rows, permutation_rows, shap_rows = [], [], [], [], []
    final_fold_models = {}
    for model_name in MODEL_NAMES:
        predictions, folds, fitted = run_walk_forward(X, y, metadata, model_name)
        all_predictions.append(predictions)
        all_folds.append(folds)
        for fold, pipeline in enumerate(fitted, 1):
            item = transformed_importance(pipeline, model_name)
            item["model"], item["fold"] = model_name, fold
            importance_rows.append(item)
        final_fold_models[model_name] = fitted[-1]

    oof = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    summaries = []
    confusion = []
    for model_name, group in oof.groupby("model", sort=False):
        row = metrics(group.target, group.probability, PROTOCOL.threshold)
        timing = folds.loc[folds.model.eq(model_name)]
        row.update({
            "model": model_name,
            "training_seconds": timing.training_seconds.sum(),
            "inference_seconds": timing.inference_seconds.sum(),
        })
        summaries.append(row)
        confusion.append({
            "model": model_name, "tn": row["tn"], "fp": row["fp"],
            "fn": row["fn"], "tp": row["tp"],
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )

    # Final-fold, untouched-test permutation importance under the same fitted fold.
    last_test_start = len(X) - len(X) // (PROTOCOL.n_splits + 1)
    X_test = X.iloc[last_test_start:last_test_start + 500]
    y_test = y.iloc[last_test_start:last_test_start + 500]
    for model_name, pipeline in final_fold_models.items():
        result = permutation_importance(
            pipeline, X_test, y_test, scoring="average_precision",
            n_repeats=3, random_state=42, n_jobs=1,
        )
        permutation_rows.append(pd.DataFrame({
            "model": model_name, "feature": X.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }))
        if model_name in {"RANDOM_FOREST", "XGBOOST", "LIGHTGBM", "CATBOOST"}:
            transformed = pipeline.named_steps["preprocessor"].transform(X_test.iloc[:500])
            names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            explainer = shap.TreeExplainer(pipeline.named_steps["model"])
            values = explainer.shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            shap_rows.append(pd.DataFrame({
                "model": model_name, "transformed_feature": names,
                "mean_absolute_shap": np.abs(values).mean(axis=0),
            }))

    importance = pd.concat(importance_rows).groupby(
        ["model", "transformed_feature"]
    ).importance.mean().reset_index()
    permutation = pd.concat(permutation_rows, ignore_index=True)
    shap_summary = pd.concat(shap_rows, ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "benchmark_summary.csv", index=False)
    grouped_performance(oof.assign(year=oof.timestamp.dt.year), "year").to_csv(
        REPORTS / "yearly_performance.csv", index=False
    )
    grouped_performance(oof, "market_regime").to_csv(
        REPORTS / "regime_performance.csv", index=False
    )
    grouped_performance(oof, "session_phase").to_csv(
        REPORTS / "session_performance.csv", index=False
    )
    grouped_performance(oof, "current_strategy").to_csv(
        REPORTS / "strategy_performance.csv", index=False
    )
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)
    permutation.to_csv(REPORTS / "permutation_importance.csv", index=False)
    shap_summary.to_csv(REPORTS / "shap_summary.csv", index=False)
    pd.DataFrame(confusion).to_csv(REPORTS / "confusion_matrix.csv", index=False)
    folds.to_csv(REPORTS / "walk_forward_folds.csv", index=False)

    numeric = X.select_dtypes(include=np.number)
    correlation = numeric.corr(method="spearman")
    correlation.to_csv(REPORTS / "feature_correlation.csv")
    pairs = []
    for i, left in enumerate(correlation.columns):
        for right in correlation.columns[i + 1:]:
            value = correlation.at[left, right]
            if abs(value) >= 0.95:
                pairs.append({
                    "feature_left": left, "feature_right": right,
                    "absolute_spearman": abs(value),
                })
    pd.DataFrame(pairs).sort_values(
        "absolute_spearman", ascending=False
    ).to_csv(REPORTS / "redundant_features.csv", index=False)
    plot_columns = correlation.abs().mean().sort_values(ascending=False).head(30).index
    plt.figure(figsize=(16, 13))
    sns.heatmap(correlation.loc[plot_columns, plot_columns], cmap="coolwarm", center=0)
    plt.title("Top-30 Mean-Correlation Entry-Time Features")
    plt.tight_layout()
    plt.savefig(REPORTS / "correlation_heatmap.png", dpi=150)
    plt.close()

    best = summary.iloc[0]
    report = f"""# Step 20B — Institutional Model Benchmark

## Protocol

- Approved target: ATM CALL reaches +5% within 15 minutes.
- Four identical expanding TimeSeriesSplit folds; no shuffle or random split.
- Fixed 0.50 threshold; no threshold optimization or calibration.
- Preprocessing is fitted independently inside each training fold.
- Fully-null excluded feature: {", ".join(excluded)}

## Highest research benchmark

- Model: **{best.model}**
- ROC-AUC: **{best.roc_auc:.4f}**
- PR-AUC: **{best.pr_auc:.4f}**
- Precision / Recall / F1: **{best.precision:.4f} /
  {best.recall:.4f} / {best.f1:.4f}**
- Balanced accuracy: **{best.balanced_accuracy:.4f}**

This ranking is descriptive research only. No model was saved, promoted,
calibrated, tuned, or connected to execution.
"""
    (REPORTS / "model_benchmark_report.md").write_text(report, encoding="utf-8")
    metadata_out = {
        "step": "20B", "research_only": True,
        "target": "ATM_CALL_5PCT_WITHIN_15_MINUTES",
        "rows": len(X), "positive_rate": float(y.mean()),
        "models": list(MODEL_NAMES),
        "protocol": {
            "splitter": "TimeSeriesSplit", "n_splits": PROTOCOL.n_splits,
            "threshold": PROTOCOL.threshold, "shuffle": False,
            "hyperparameter_optimization": False, "calibration": False,
        },
        "excluded_all_null_features": excluded,
        "frozen_hashes": {
            "prediction_dataset": _hash(PREDICTION),
            "approved_target_source": _hash(DECISIONS),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
