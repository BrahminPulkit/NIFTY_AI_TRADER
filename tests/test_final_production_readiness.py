import numpy as np
import pandas as pd

from src.final_production_readiness import (
    mcnemar_exact, paired_delong, paired_pr_bootstrap, readiness_score,
)


def test_identical_probabilities_have_zero_auc_difference():
    y = np.array([0, 0, 1, 1, 0, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    result = paired_delong(y, probability, probability)
    assert result["difference_v2_minus_v1"] == 0


def test_pr_bootstrap_is_deterministic():
    y = np.array([0, 0, 1, 1, 0, 1])
    one = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    two = np.array([0.2, 0.1, 0.7, 0.8, 0.4, 0.9])
    assert paired_pr_bootstrap(y, one, two, 100, 7) == paired_pr_bootstrap(
        y, one, two, 100, 7
    )


def test_mcnemar_counts_discordance():
    y = np.array([0, 1, 0, 1])
    one = np.array([0.1, 0.9, 0.9, 0.1])
    two = np.array([0.1, 0.1, 0.2, 0.9])
    result = mcnemar_exact(y, one, two)
    assert result["discordant"] == 3


def test_sparse_decision_engine_caps_classification():
    row = pd.Series({
        "roc_auc": 0.9, "pr_auc": 0.85, "f1": 0.8,
        "balanced_accuracy": 0.85, "brier_score": 0.1, "log_loss": 0.3,
        "probability_psi": 0.01, "median_feature_psi": 0.01,
        "temporal_trades": 100, "temporal_profit_factor": 3,
        "temporal_cagr": 0.2, "temporal_mc_p05": 0.1,
        "backtest_max_drawdown": -0.05, "backtest_profit_factor": 3,
        "backtest_expectancy": 0.01, "training_seconds": 1,
        "inference_seconds": 1, "decision_engine_executed_trades": 4,
    })
    result = readiness_score(row, row)
    assert result["classification"] == "Research Ready"

