"""Execute Step 19F adaptive strategy-selection research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_decision_engine import (
    MINIMUM_HISTORY,
    RESEARCH_EXPANSION_TARGET,
    RESEARCH_HORIZON_MINUTES,
    build_decision_observations,
    build_walk_forward_decisions,
)


SETUPS = Path("data/setups/stage1_setup_dataset.parquet")
ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
REPORTS = Path("reports/strategy_decision")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    setups = pd.read_parquet(SETUPS)
    aligned = pd.read_parquet(ALIGNED)
    observations = build_decision_observations(setups, aligned)
    decisions = build_walk_forward_decisions(observations)
    REPORTS.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(REPORTS / "decision_matrix.csv", index=False)
    activation = decisions.groupby(
        ["market_state", "activated_strategy"], dropna=False
    ).agg(
        decisions=("timestamp", "size"),
        average_confidence=("strategy_confidence_score", "mean"),
        average_expansion_probability=("premium_expansion_probability", "mean"),
        average_expected_expansion=("expected_premium_expansion", "mean"),
        average_expected_drawdown=("expected_drawdown", "mean"),
        trade_rate=("recommendation", lambda x: x.eq("TRADE").mean()),
    ).reset_index()
    activation.to_csv(REPORTS / "strategy_activation_table.csv", index=False)
    trade_skip = decisions.groupby(
        ["recommendation", "decision_reason"], dropna=False
    ).agg(
        setups=("timestamp", "size"),
        realized_expansion_rate=("realized_expansion_hit", "mean"),
        realized_average_return=("realized_terminal_return", "mean"),
        realized_average_mfe=("realized_mfe", "mean"),
        realized_average_mae=("realized_mae", "mean"),
    ).reset_index()
    trade_skip.to_csv(REPORTS / "trade_skip_analysis.csv", index=False)
    bins = [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf]
    confidence = decisions.assign(
        confidence_band=pd.cut(decisions.strategy_confidence_score, bins=bins)
    ).groupby("confidence_band", observed=True).agg(
        setups=("timestamp", "size"),
        trade_rate=("recommendation", lambda x: x.eq("TRADE").mean()),
        realized_expansion_rate=("realized_expansion_hit", "mean"),
        realized_average_return=("realized_terminal_return", "mean"),
    ).reset_index()
    confidence.to_csv(REPORTS / "confidence_distribution.csv", index=False)
    decisions.groupby("activated_strategy", dropna=False).agg(
        decisions=("timestamp", "size"),
        expected_expansion=("expected_premium_expansion", "mean"),
        expansion_probability=("premium_expansion_probability", "mean"),
    ).reset_index().to_csv(REPORTS / "expected_expansion.csv", index=False)
    decisions.groupby("activated_strategy", dropna=False).agg(
        decisions=("timestamp", "size"),
        expected_drawdown=("expected_drawdown", "mean"),
        expected_holding_minutes=("expected_holding_minutes", "mean"),
    ).reset_index().to_csv(REPORTS / "expected_drawdown.csv", index=False)
    decisions.groupby([
        "trend_regime", "volatility_regime", "session_phase", "structure_context"
    ]).agg(
        setups=("timestamp", "size"),
        most_common_activation=(
            "activated_strategy",
            lambda x: x.mode().iat[0] if not x.mode().empty else "NONE",
        ),
        average_confidence=("strategy_confidence_score", "mean"),
        trade_rate=("recommendation", lambda x: x.eq("TRADE").mean()),
        realized_expansion_rate=("realized_expansion_hit", "mean"),
    ).reset_index().to_csv(REPORTS / "market_state_summary.csv", index=False)

    trades = decisions.recommendation.eq("TRADE")
    report = f"""# Step 19F Research Conclusion

- Decisions: **{len(decisions):,}**
- TRADE recommendations: **{int(trades.sum()):,}** ({trades.mean():.2%})
- SKIP recommendations: **{int((~trades).sum()):,}**
- Research expansion anchor: +{RESEARCH_EXPANSION_TARGET:.0%} within
  {RESEARCH_HORIZON_MINUTES} minutes
- Minimum matured history per estimate: {MINIMUM_HISTORY}

The selector uses expanding historical evidence only. Outcomes enter its state
after their observation window has completed. `TRADE` requires the current
morphology family to match the activated family and the historical terminal
return's 95% lower confidence bound to be positive.

This is a research decision layer, not a promoted model, calibrated probability,
threshold optimization, or backtest.
"""
    (REPORTS / "recommendation.md").write_text(report, encoding="utf-8")
    metadata = {
        "step": "19F", "research_only": True,
        "decision_method": "causal expanding-history conditional estimates",
        "research_horizon_minutes": RESEARCH_HORIZON_MINUTES,
        "research_expansion_target": RESEARCH_EXPANSION_TARGET,
        "minimum_history": MINIMUM_HISTORY,
        "trade_rule": (
            "current family matches activation; lower 95% historical mean "
            "terminal-return bound > 0; expansion/drawdown ratio > 1"
        ),
        "frozen_hashes": {
            "stage1_setup_dataset": _hash(SETUPS),
            "aligned_index_option_dataset": _hash(ALIGNED),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

