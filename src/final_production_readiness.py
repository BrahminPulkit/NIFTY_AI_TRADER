"""Statistical comparison and fixed production-readiness scoring for Step 20L."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import average_precision_score


def paired_delong(
    y: np.ndarray, probability_a: np.ndarray, probability_b: np.ndarray
) -> dict:
    y = np.asarray(y, dtype=int)
    positive = y == 1
    negative = ~positive
    if positive.sum() < 2 or negative.sum() < 2:
        raise ValueError("DeLong requires at least two observations per class")

    def components(scores):
        pos, neg = scores[positive], scores[negative]
        comparisons = (
            (pos[:, None] > neg[None, :]).astype(float)
            + 0.5 * (pos[:, None] == neg[None, :])
        )
        return comparisons.mean(), comparisons.mean(axis=1), comparisons.mean(axis=0)

    auc_a, pos_a, neg_a = components(np.asarray(probability_a, float))
    auc_b, pos_b, neg_b = components(np.asarray(probability_b, float))
    positive_covariance = np.cov(np.vstack([pos_a, pos_b]), ddof=1)
    negative_covariance = np.cov(np.vstack([neg_a, neg_b]), ddof=1)
    covariance = (
        positive_covariance / positive.sum()
        + negative_covariance / negative.sum()
    )
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    standard_error = np.sqrt(max(variance, 0.0))
    difference = auc_b - auc_a
    z = difference / standard_error if standard_error > 0 else np.nan
    p_value = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {
        "v1_roc_auc": auc_a, "v2_roc_auc": auc_b,
        "difference_v2_minus_v1": difference,
        "standard_error": standard_error,
        "z_statistic": z, "p_value": p_value,
    }


def paired_pr_bootstrap(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    simulations: int = 2000,
    seed: int = 42,
) -> dict:
    y = np.asarray(y, dtype=int)
    probability_a = np.asarray(probability_a, float)
    probability_b = np.asarray(probability_b, float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    random = np.random.default_rng(seed)
    differences = np.empty(simulations)
    for simulation in range(simulations):
        indices = np.concatenate([
            random.choice(positive, len(positive), replace=True),
            random.choice(negative, len(negative), replace=True),
        ])
        differences[simulation] = (
            average_precision_score(y[indices], probability_b[indices])
            - average_precision_score(y[indices], probability_a[indices])
        )
    observed = (
        average_precision_score(y, probability_b)
        - average_precision_score(y, probability_a)
    )
    p_value = min(
        1.0,
        2 * min(
            float((differences <= 0).mean()),
            float((differences >= 0).mean()),
        ),
    )
    return {
        "difference_v2_minus_v1": observed,
        "ci_2_5": float(np.quantile(differences, 0.025)),
        "ci_97_5": float(np.quantile(differences, 0.975)),
        "p_value": p_value,
        "bootstrap_simulations": simulations,
    }


def mcnemar_exact(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y = np.asarray(y, dtype=int)
    correct_a = (np.asarray(probability_a) >= threshold).astype(int) == y
    correct_b = (np.asarray(probability_b) >= threshold).astype(int) == y
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only
    p_value = (
        float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "v1_correct_v2_wrong": a_only,
        "v1_wrong_v2_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def readiness_score(row: pd.Series, v1_efficiency: pd.Series) -> dict:
    predictive = (
        6 * min(row.roc_auc / 0.90, 1)
        + 6 * min(row.pr_auc / 0.85, 1)
        + 5 * min(row.f1 / 0.80, 1)
        + 4 * min(row.balanced_accuracy / 0.85, 1)
        + 2 * max(0, 1 - row.brier_score / 0.25)
        + 2 * max(0, 1 - row.log_loss / 0.70)
    )
    probability_stability = 15 * max(0, 1 - row.probability_psi / 0.25)
    drift = 15 * max(0, 1 - row.median_feature_psi / 0.25)
    temporal = (
        5 * min(row.temporal_trades / 100, 1)
        + 5 * min(row.temporal_profit_factor / 2, 1)
        + 5 * float(row.temporal_cagr > 0)
        + 5 * float(row.temporal_mc_p05 > 0)
    )
    risk = (
        5 * float(abs(row.backtest_max_drawdown) <= 0.10)
        + 5 * float(row.backtest_profit_factor >= 1.5)
        + 5 * float(row.backtest_expectancy > 0)
    )
    efficiency = (
        5 * min(v1_efficiency.training_seconds / row.training_seconds, 1)
        + 5 * min(v1_efficiency.inference_seconds / row.inference_seconds, 1)
    )
    score = predictive + probability_stability + drift + temporal + risk + efficiency
    if score < 40:
        classification = "Not Ready"
    elif score < 60:
        classification = "Research Ready"
    elif score < 75:
        classification = "Paper Trading Ready"
    elif score < 90:
        classification = "Production Candidate"
    else:
        classification = "Institutional Grade"
    # The frozen Decision Engine has only four complete replay trades.
    if row.decision_engine_executed_trades < 30 and classification not in {
        "Not Ready", "Research Ready"
    }:
        classification = "Research Ready"
    return {
        "predictive_performance_score_25": predictive,
        "probability_stability_score_15": probability_stability,
        "drift_resistance_score_15": drift,
        "temporal_robustness_score_20": temporal,
        "risk_metrics_score_15": risk,
        "computational_efficiency_score_10": efficiency,
        "production_readiness_score": score,
        "classification": classification,
        "decision_engine_evidence_gate": (
            "PASS" if row.decision_engine_executed_trades >= 30
            else "FAIL_LT_30_TRADES"
        ),
    }
