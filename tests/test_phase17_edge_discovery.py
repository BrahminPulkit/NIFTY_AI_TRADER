import pandas as pd

from tools.run_phase17_edge_discovery import bh_adjust, maximum_drawdown, profit_factor


def test_economic_helpers_are_deterministic():
    pnl = pd.Series([10.0, -5.0, 2.0, -1.0])
    assert profit_factor(pnl) == 2.0
    assert maximum_drawdown(pnl) == 5.0


def test_bh_adjustment_preserves_index_and_bounds():
    raw = pd.Series([0.01, 0.20, 0.03], index=["a", "b", "c"])
    adjusted = bh_adjust(raw)
    assert adjusted.index.equals(raw.index)
    assert adjusted.between(0, 1).all()
