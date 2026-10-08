"""Execute Step 20C using only the frozen joined OOF research dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.probability_threshold_research import (
    THRESHOLDS,
    candidate_thresholds,
    confidence_bands,
    equity_curves,
    evaluate_thresholds,
    grouped_threshold_statistics,
    pareto_frontier,
    validate_joined_dataset,
)


INPUT = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
REPORTS = Path("reports/probability_threshold_research/step20c")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = validate_joined_dataset(pd.read_parquet(INPUT))
    statistics = evaluate_thresholds(data)
    frontier = pareto_frontier(statistics)
    candidates = candidate_thresholds(statistics)
    equity = equity_curves(data)
    yearly = grouped_threshold_statistics(data, "year")
    regime = grouped_threshold_statistics(data, "market_regime")
    session = grouped_threshold_statistics(data, "session_phase")
    strategy = grouped_threshold_statistics(data, "strategy_family")
    fold = grouped_threshold_statistics(data, "fold_id")
    stability = fold.groupby(["model", "threshold"]).agg(
        folds=("fold_id", "nunique"),
        trades_mean=("trades", "mean"),
        trades_min=("trades", "min"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        expected_value_mean=("expected_value_per_trade", "mean"),
        expected_value_std=("expected_value_per_trade", "std"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
    ).reset_index()
    stability["stability_score"] = 1 / (
        1 + stability.precision_std.fillna(1)
        + stability.expected_value_std.fillna(1)
        + stability.coverage_std.fillna(1)
    )
    confidence = confidence_bands(data)

    REPORTS.mkdir(parents=True, exist_ok=False)
    statistics.to_csv(REPORTS / "threshold_statistics.csv", index=False)
    frontier.to_csv(REPORTS / "pareto_frontier.csv", index=False)
    equity.to_csv(REPORTS / "equity_curve.csv", index=False)
    stability.to_csv(REPORTS / "threshold_stability.csv", index=False)
    yearly.to_csv(REPORTS / "yearly_threshold_statistics.csv", index=False)
    regime.to_csv(REPORTS / "regime_threshold_statistics.csv", index=False)
    session.to_csv(REPORTS / "session_threshold_statistics.csv", index=False)
    strategy.to_csv(REPORTS / "strategy_threshold_statistics.csv", index=False)
    confidence.to_csv(REPORTS / "confidence_band_analysis.csv", index=False)
    candidates.to_csv(REPORTS / "candidate_threshold_profiles.csv", index=False)

    candidate_lines = "\n".join(
        f"- {row.model} {row.profile}: {row.threshold:.2f} "
        f"(trades={int(row.trades)}, precision={row.precision:.2%}, "
        f"EV={row.expected_value_per_trade:.2%})"
        for row in candidates.itertuples(index=False)
    )
    (REPORTS / "recommended_thresholds.md").write_text(
        "# Candidate Threshold Profiles\n\n"
        "These are retrospective OOF research profiles, not deployment thresholds.\n\n"
        + candidate_lines + "\n",
        encoding="utf-8",
    )
    best_stability = stability.sort_values(
        ["stability_score", "expected_value_mean"], ascending=False
    ).groupby("model").head(1)
    comparable_stability = stability.loc[stability.threshold.le(0.90)].groupby(
        "model"
    ).stability_score.mean()
    stability_winner = comparable_stability.idxmax()
    report = f"""# Step 20C — Institutional Probability Threshold Research

- Input rows: **{len(data):,} leakage-free joined OOF observations**
- Thresholds: {", ".join(f"{value:.2f}" for value in THRESHOLDS)}
- Models: CatBoost and XGBoost
- Retraining, probability regeneration, calibration and feature changes: **none**
- Transaction costs, overlap constraints and execution modelling: **not included**

## Candidate profiles

{candidate_lines}

## Stability

The complete fold/year/regime/session/strategy tables are provided. Stability
scores describe dispersion across the four OOF folds and do not promote a
threshold for live deployment.

Across the directly comparable 0.50–0.90 grid, **{stability_winner}** has the
higher mean fold-stability score
({comparable_stability[stability_winner]:.4f} versus
{comparable_stability.drop(stability_winner).iloc[0]:.4f}).

## Equity curve semantics

Curves assume one independent trade per selected signal using frozen terminal
premium return. They ignore overlap, capital constraints, costs and execution,
so they are descriptive research curves—not a backtest.
"""
    (REPORTS / "step20c_probability_threshold_report.md").write_text(
        report, encoding="utf-8"
    )
    metadata = {
        "step": "20C", "research_only": True,
        "input_rows": len(data), "thresholds": list(THRESHOLDS),
        "models": ["CATBOOST", "XGBOOST"],
        "probability_source": "saved leakage-free OOF only",
        "retraining": False, "calibration": False,
        "transaction_costs": False, "backtest": False,
        "threshold_stability_comparison_050_090": {
            model: float(value) for model, value in comparable_stability.items()
        },
        "more_stable_model_050_090": stability_winner,
        "input_hash": _hash(INPUT),
        "output_validation": {
            "threshold_rows": len(statistics),
            "expected_threshold_rows": 2 * len(THRESHOLDS),
            "pareto_rows": len(frontier),
            "equity_rows": len(equity),
            "all_trade_counts_valid": bool(
                statistics.trades.le(statistics.total_observations).all()
            ),
            "all_coverages_valid": bool(statistics.coverage.between(0, 1).all()),
            "equity_chronological": bool(
                equity.groupby(["model", "threshold"]).timestamp.apply(
                    lambda x: x.is_monotonic_increasing
                ).all()
            ),
        },
    }
    if metadata["output_validation"]["threshold_rows"] != 2 * len(THRESHOLDS):
        raise AssertionError("Threshold output is incomplete")
    if not all([
        metadata["output_validation"]["all_trade_counts_valid"],
        metadata["output_validation"]["all_coverages_valid"],
        metadata["output_validation"]["equity_chronological"],
    ]):
        raise AssertionError("One or more Step 20C output validations failed")
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
