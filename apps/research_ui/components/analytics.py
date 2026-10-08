"""Presentation-safe summaries calculated from frozen rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def strategy_performance(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    required = {
        "current_strategy", "realized_terminal_return",
        "realized_expansion_hit",
    }
    if not required.issubset(decisions.columns):
        return pd.DataFrame()
    rows = []
    for strategy, group in decisions.groupby("current_strategy", dropna=False):
        returns = pd.to_numeric(group.realized_terminal_return, errors="coerce").dropna()
        gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
        rows.append({
            "strategy": strategy,
            "observations": len(group),
            "win_rate": group.realized_expansion_hit.mean(),
            "positive_return_rate": returns.gt(0).mean(),
            "profit_factor": gains / losses if losses > 0 else np.inf,
            "average_return": returns.mean(),
        })
    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False)

