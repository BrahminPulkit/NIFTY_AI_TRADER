"""Execute Step 19E research without altering frozen components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.strategy_discovery import (
    classify_entry_context,
    confidence_ranking,
    measure_premium_behaviour,
    minimal_profitable_set,
    relationship_statistics,
    strategy_regime_matrix,
    summarize_behaviour,
)


SETUPS = Path("data/setups/stage1_setup_dataset.parquet")
ALIGNED = Path("data/aligned/canonical_index_option_aligned.parquet")
REPORTS = Path("reports/strategy_discovery")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    setups = pd.read_parquet(SETUPS)
    aligned = pd.read_parquet(ALIGNED)
    entries = classify_entry_context(setups)
    behaviour = measure_premium_behaviour(aligned, entries)
    individual = behaviour.sort_values(
        ["horizon_minutes", "premium_mfe", "terminal_premium_return"],
        ascending=[True, False, False],
    ).copy()
    individual["setup_rank_within_horizon"] = individual.groupby(
        "horizon_minutes"
    ).cumcount() + 1
    strategy = summarize_behaviour(
        behaviour, ["horizon_minutes", "strategy_family"]
    )
    regime = summarize_behaviour(
        behaviour, ["horizon_minutes", "market_regime"]
    )
    matrix = strategy_regime_matrix(summarize_behaviour(
        behaviour, ["horizon_minutes", "strategy_family", "market_regime"]
    ))
    confidence = confidence_ranking(strategy)
    minimal = minimal_profitable_set(behaviour, horizon=15)
    relationships = relationship_statistics(behaviour)
    REPORTS.mkdir(parents=True, exist_ok=True)
    individual.to_csv(REPORTS / "setup_rankings.csv", index=False)
    strategy.to_csv(REPORTS / "strategy_summary.csv", index=False)
    regime.to_csv(REPORTS / "regime_summary.csv", index=False)
    matrix.to_csv(REPORTS / "strategy_regime_matrix.csv", index=False)
    confidence.to_csv(REPORTS / "confidence_rankings.csv", index=False)
    minimal.to_csv(REPORTS / "minimal_strategy_set.csv", index=False)
    relationships.to_csv(REPORTS / "relationship_statistics.csv", index=False)

    best = confidence.iloc[0]
    selected = minimal.loc[minimal.in_smallest_80pct_set, "strategy_family"].tolist()
    report = f"""# Step 19E — Institutional Strategy Discovery

## Scope

- Frozen allowed Stage-1 setups: {len(entries):,}
- Premium behaviour observations: {len(behaviour):,}
- Horizons: 15, 30, 45 and 60 minutes
- No prediction model, optimization, threshold search, label, or production
  decision was created.

## Structural limitation

Frozen Stage-1 V2 contains one underlying setup archetype: confirmed directional
breakout. Pullback, momentum, extended and continuation families in this report
are causal entry-morphology subgroups, not independent setup engines.

## Highest confidence descriptive subgroup

- Family: **{best.strategy_family}**
- Horizon: **{int(best.horizon_minutes)} minutes**
- Positive-close rate: **{best.positive_close_rate:.2%}**
- Average premium expansion: **{best.average_premium_expansion:.2%}**
- Average terminal return: **{best.average_terminal_return:.2%}**
- Expansion/drawdown ratio: **{best.expansion_drawdown_ratio:.3f}**

## Smallest family set explaining at least 80% of positive 15-minute returns

{", ".join(selected)}

Grades in the Strategy × Regime matrix are descriptive summaries only and are
not production gates.
"""
    (REPORTS / "strategy_discovery_report.md").write_text(report, encoding="utf-8")
    conclusion = """# Research Conclusion

The current data supports descriptive study of how a rolling ATM CALL premium
responds to frozen Index breakout morphology. It does not contain multiple
independent Stage-1 strategies, so claims that one legacy strategy “beats”
another would be false.

Use the matrix to formulate pre-registered Stage-1 V4 hypotheses in a separately
authorized research step. Do not convert relative grades or confidence scores
into production filters without chronological validation.
"""
    (REPORTS / "recommendation.md").write_text(conclusion, encoding="utf-8")
    metadata = {
        "step": "19E", "research_only": True,
        "setup_count": len(entries), "behaviour_rows": len(behaviour),
        "win_definition": "positive horizon-end ATM CALL return",
        "strategy_families_are_morphology_subgroups": True,
        "regime_method": "deterministic causal rules; no learned clustering",
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

