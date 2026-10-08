"""Step 20K helpers for comparing the frozen V1 protocol with V2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.temporal_decision_validation import population_stability_index


def curve_tables(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows, pr_rows = [], []
    for model, group in oof.groupby("model", sort=True):
        false_positive, true_positive, roc_threshold = roc_curve(
            group.target, group.probability
        )
        precision, recall, pr_threshold = precision_recall_curve(
            group.target, group.probability
        )
        roc_rows.append(pd.DataFrame({
            "model": model, "false_positive_rate": false_positive,
            "true_positive_rate": true_positive, "threshold": roc_threshold,
        }))
        pr_rows.append(pd.DataFrame({
            "model": model, "precision": precision[:-1],
            "recall": recall[:-1], "threshold": pr_threshold,
        }))
    return pd.concat(roc_rows, ignore_index=True), pd.concat(pr_rows, ignore_index=True)


def numeric_feature_stability(
    data: pd.DataFrame,
    research_end: pd.Timestamp,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    research = data.timestamp.le(research_end)
    validation = data.timestamp.ge(validation_start)
    rows = []
    for column in data.select_dtypes(include=np.number).columns:
        if column.endswith("session_id"):
            continue
        rows.append({
            "feature": column,
            "psi": population_stability_index(
                data.loc[research, column], data.loc[validation, column]
            ),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def probability_stability(
    oof: pd.DataFrame,
    research_end: pd.Timestamp,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for model, group in oof.groupby("model", sort=True):
        left = group.loc[group.timestamp.le(research_end), "probability"]
        right = group.loc[group.timestamp.ge(validation_start), "probability"]
        rows.append({
            "model": model,
            "research_mean_probability": left.mean(),
            "validation_mean_probability": right.mean(),
            "probability_psi": population_stability_index(left, right),
        })
    return pd.DataFrame(rows)


def choose_dataset(final: pd.DataFrame) -> tuple[str, list[str]]:
    """Fixed, auditable V2 promotion screen; returns exactly one version."""
    lookup = final.set_index(["dataset_version", "model"])
    v1 = lookup.loc[("V1", "CATBOOST")]
    v2 = lookup.loc[("V2", "CATBOOST")]
    checks = {
        "CatBoost ROC-AUC non-decreasing": v2.roc_auc >= v1.roc_auc,
        "CatBoost PR-AUC non-decreasing": v2.pr_auc >= v1.pr_auc,
        "median feature PSI improved": (
            v2.median_feature_psi < v1.median_feature_psi
        ),
        "V2 holdout robust": bool(v2.holdout_robust),
    }
    recommended = "Production Dataset V2" if all(checks.values()) else "Production Dataset V1"
    return recommended, [
        f"{name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in checks.items()
    ]

