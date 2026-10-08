import pandas as pd

from src.temporal_decision_validation import (
    population_stability_index,
    select_profiles,
    stability_verdict,
)


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series(range(100))
    assert population_stability_index(values, values) == 0


def test_profile_selection_uses_research_metrics_only():
    data = pd.DataFrame({
        "candidate_id": ["SAFE", "FAST"],
        "evaluable": [True, True],
        "trades": [100, 100],
        "mc_percentile_5": [0.1, 0.1],
        "maximum_drawdown": [-0.02, -0.08],
        "profit_factor": [3.0, 2.0],
        "cagr": [0.1, 0.3],
        "yearly_positive_fraction": [1.0, 1.0],
    })
    profiles = select_profiles(data)
    assert profiles["CONSERVATIVE"].candidate_id == "SAFE"
    assert profiles["AGGRESSIVE"].candidate_id == "FAST"


def test_tolerance_failure_rejects_candidate():
    research = {
        "cagr": 0.20, "maximum_drawdown": -0.05, "coverage": 0.10,
    }
    validation = {
        "cagr": 0.01, "maximum_drawdown": -0.20, "coverage": 0.01,
        "executed_trades": 5, "profit_factor": 0.8, "mc_percentile_5": -0.1,
    }
    result = stability_verdict(research, validation, 0.5, 0.5)
    assert not result["passes_all_tolerances"]

