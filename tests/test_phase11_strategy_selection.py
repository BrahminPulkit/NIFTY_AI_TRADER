from tools.run_phase11_strategy_selection import selection_gates
import pandas as pd


def test_strategy_gate_rejects_negative_expectancy_despite_large_sample():
    summary = {"eligible_trades": 1000, "ce_trades": 500, "pe_trades": 500,
               "sessions": 35, "after_cost_expectancy": -1, "profit_factor": .99,
               "positive_session_rate": .7, "total_trades": 1000}
    folds = pd.DataFrame({"after_cost_expectancy": [1, 1, 1, 1, 1]})
    gates = selection_gates(summary, summary, summary, folds)
    assert not gates["combined_expectancy_positive"]
    assert not gates["combined_profit_factor_above_one"]
