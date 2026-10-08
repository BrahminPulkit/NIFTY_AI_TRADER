import pandas as pd

from src.decision_engine_optimization import (
    RULES,
    all_rule_subsets,
    candidate_definitions,
    candidate_mask,
    pareto_frontier,
)


def _rows():
    return pd.DataFrame({
        "history_count": [40, 40, 40, 0],
        "activated_strategy": ["A", "B", "A", "NONE"],
        "current_strategy": ["A", "A", "A", "A"],
        "estimate_source": ["EXACT", "REGIME", "STRATEGY", "NONE"],
        "terminal_return_95_lower": [0.1, 0.1, -0.1, float("nan")],
        "expansion_drawdown_ratio": [2.0, 2.0, 2.0, float("nan")],
    })


def test_every_observable_rule_combination_exists():
    subsets = all_rule_subsets()
    assert len(subsets) == 2 ** len(RULES)
    definitions = candidate_definitions()
    assert sum(item["evaluable"] for item in definitions) == 48


def test_candidate_masks_use_only_declared_frozen_fields():
    data = _rows()
    all_rules = candidate_mask(data, RULES, "ALL_FALLBACKS")
    no_strategy = candidate_mask(
        data,
        ("POSITIVE_CONFIDENCE_LOWER_BOUND", "EXPANSION_GT_DRAWDOWN"),
        "ALL_FALLBACKS",
    )
    assert all_rules.tolist() == [True, False, False, False]
    assert no_strategy.tolist() == [True, True, False, False]


def test_pareto_frontier_removes_dominated_candidate():
    candidates = pd.DataFrame({
        "candidate_id": ["A", "B", "C"],
        "evaluable": [True] * 3,
        "trades": [100] * 3,
        "cagr": [0.2, 0.1, 0.25],
        "maximum_drawdown": [-0.1, -0.2, -0.2],
        "profit_factor": [2.0, 1.5, 1.8],
        "yearly_positive_fraction": [0.8, 0.7, 0.9],
    })
    frontier = pareto_frontier(candidates)
    assert "B" not in set(frontier.candidate_id)
    assert {"A", "C"} == set(frontier.candidate_id)

