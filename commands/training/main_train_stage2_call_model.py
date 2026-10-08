"""Train and compare Stage-2 CALL V1 under one fixed walk-forward protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import __version__ as xgboost_version

from src.stage2_call_model import (
    MODEL_A_FEATURES, MODEL_B_FEATURES, MODEL_VERSION, PROTOCOL,
    build_model, classification_metrics, grouped_performance, prepare_call_dataset,
    raw_calibration, run_walk_forward,
)


REPORTS = Path("reports/stage2_call_model")
MODELS = Path("data/models/stage2_call_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def explain(model, X: pd.DataFrame, model_name: str):
    importance = pd.DataFrame({"model": model_name, "feature": X.columns,
                               "importance": model.feature_importances_}).sort_values(
                                   "importance", ascending=False)
    sample = X.iloc[np.linspace(0, len(X)-1, min(1000, len(X)), dtype=int)]
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    summary = pd.DataFrame({"model": model_name, "feature": X.columns,
                            "mean_abs_shap": np.abs(values).mean(axis=0),
                            "mean_shap": np.asarray(values).mean(axis=0)}).sort_values(
                                "mean_abs_shap", ascending=False)
    return importance, summary


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    aligned = pd.read_parquet("data/aligned/canonical_index_option_aligned.parquet")
    outcomes = pd.read_parquet("data/outcomes/stage2_outcome_dataset.parquet")
    XA, XB, y, metadata = prepare_call_dataset(aligned, outcomes)
    results = []
    for name, X in (("MODEL_A_INDEX", XA), ("MODEL_B_INDEX_CALL", XB)):
        predictions, folds = run_walk_forward(X, y, metadata, name)
        aggregate = {"model": name, **classification_metrics(predictions.target,
                                                               predictions.probability)}
        results.append((name, X, predictions, folds, aggregate))

    comparison = pd.DataFrame([item[4] for item in results])
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    pd.concat([item[3] for item in results], ignore_index=True).to_csv(
        REPORTS / "walk_forward_results.csv", index=False)
    pd.concat([grouped_performance(item[2], "year") for item in results], ignore_index=True).to_csv(
        REPORTS / "yearly_performance.csv", index=False)
    pd.concat([grouped_performance(item[2], "trend_state") for item in results], ignore_index=True).to_csv(
        REPORTS / "regime_performance.csv", index=False)
    pd.concat([grouped_performance(item[2], "session_segment") for item in results], ignore_index=True).to_csv(
        REPORTS / "session_performance.csv", index=False)
    pd.concat([raw_calibration(item[2]) for item in results], ignore_index=True).to_csv(
        REPORTS / "raw_calibration.csv", index=False)
    comparison[["model", "tn", "fp", "fn", "tp"]].to_csv(
        REPORTS / "confusion_matrix.csv", index=False)

    importance_tables, shap_tables = [], []
    model_hashes = {}
    for name, X, _, _, _ in results:
        final = build_model(y); final.fit(X, y)
        path = MODELS / f"{name.lower()}.pkl"; joblib.dump(final, path)
        (MODELS / f"{name.lower()}_features.json").write_text(
            json.dumps(list(X.columns), indent=2), encoding="utf-8")
        model_hashes[name] = sha256(path)
        importance, summary = explain(final, X, name)
        importance_tables.append(importance); shap_tables.append(summary)
    pd.concat(importance_tables, ignore_index=True).to_csv(REPORTS / "feature_importance.csv", index=False)
    pd.concat(shap_tables, ignore_index=True).to_csv(REPORTS / "shap_summary.csv", index=False)

    a, b = comparison.set_index("model").loc[["MODEL_A_INDEX", "MODEL_B_INDEX_CALL"]].to_dict("index").values()
    improvement = bool(
        np.isfinite(b["roc_auc"]) and np.isfinite(a["roc_auc"]) and
        b["roc_auc"] > a["roc_auc"] and b["pr_auc"] > a["pr_auc"])
    metadata_payload = {
        "model_version": MODEL_VERSION, "protocol": PROTOCOL.__dict__,
        "rows": len(y), "wins": int(y.sum()), "non_wins": int((1-y).sum()),
        "model_a_features": MODEL_A_FEATURES, "model_b_features": MODEL_B_FEATURES,
        "xgboost_version": xgboost_version, "model_hashes": model_hashes,
        "atm_call_improves_roc_and_pr": improvement,
        "frozen_inputs": {
            "aligned": sha256(Path("data/aligned/canonical_index_option_aligned.parquet")),
            "outcomes": sha256(Path("data/outcomes/stage2_outcome_dataset.parquet")),
            "stage1": sha256(Path("src/setup_engine.py")),
            "index_pipeline": sha256(Path("src/feature_pipeline.py")),
            "option_pipeline": sha256(Path("src/option_feature_pipeline.py")),
        },
    }
    (REPORTS / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    verdict = (
        "ATM CALL features improve both aggregate walk-forward ROC-AUC and PR-AUC."
        if improvement else
        "ATM CALL features do not improve both aggregate walk-forward ROC-AUC and PR-AUC."
    )
    report = f"""# Stage-2 CALL Model V1

## Protocol

- Allowed Stage-1 LONG setups only; WIN=1, LOSS/TIMEOUT=0.
- Four expanding chronological folds; no shuffle or random split.
- Fixed XGBoost parameters and fixed 0.50 decision threshold; no tuning.
- Model A and B use identical rows, labels and fold boundaries.

## Data limitation

- Rows: **{len(y):,}**; wins: **{int(y.sum())}** ({y.mean():.3%}).
- This sample is too sparse for production confidence. Fold-level metrics with no test positives are reported as undefined.

## Comparison

- Model A ROC/PR: **{a['roc_auc']:.4f} / {a['pr_auc']:.4f}**
- Model B ROC/PR: **{b['roc_auc']:.4f} / {b['pr_auc']:.4f}**
- Model A precision/recall/F1: **{a['precision']:.4f} / {a['recall']:.4f} / {a['f1']:.4f}**
- Model B precision/recall/F1: **{b['precision']:.4f} / {b['recall']:.4f} / {b['f1']:.4f}**

{verdict} Raw calibration diagnostics are descriptive only; Step 20 calibration was not performed.
"""
    (REPORTS / "stage2_call_model_report.md").write_text(report, encoding="utf-8")
    ready = improvement and int(y.sum()) >= 30
    recommendation = (
        "Proceed to calibration research." if ready else
        "Do not begin probability calibration: incremental benefit and/or positive sample size is insufficient."
    )
    (REPORTS / "recommendation.md").write_text(
        f"# Recommendation\n\n{verdict}\n\n{recommendation}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
