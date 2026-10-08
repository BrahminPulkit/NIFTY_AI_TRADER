"""Execute Step 20H-R strict temporal Decision Engine validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.decision_engine_optimization import evaluate_candidates
from src.decision_funnel_research import add_net_profitability, build_funnel_universe
from src.temporal_decision_validation import (
    RESEARCH_END,
    TOLERANCES,
    VALIDATION_START,
    evaluate_frozen_engine,
    feature_drift_table,
    population_stability_index,
    select_profiles,
    stability_verdict,
)
from src.walk_forward_paper_trading import POLICY


STAGE1 = Path("data/setups/stage1_setup_dataset.parquet")
PREDICTIONS = Path("data/prediction/prediction_dataset.parquet")
JOINED = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
REPORTS = Path("reports/temporal_decision_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equity_curve(
    trades: pd.DataFrame, dates: pd.DatetimeIndex, profile: str
) -> pd.DataFrame:
    events = pd.Series(
        trades.capital_after.to_numpy(),
        index=pd.DatetimeIndex(trades.exit_timestamp),
    ) if len(trades) else pd.Series(dtype=float)
    initial = pd.Series(
        [1_000_000.0], index=pd.DatetimeIndex([dates.min()])
    )
    events = pd.concat([initial, events]).sort_index()
    equity = (
        events.reindex(events.index.union(dates)).sort_index()
        .ffill().reindex(dates)
    )
    result = pd.DataFrame({
        "profile": profile, "date": dates, "equity": equity.to_numpy(),
    })
    result["daily_return"] = result.equity.pct_change().fillna(0.0)
    result["drawdown"] = result.equity / result.equity.cummax() - 1
    return result


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError(f"{REPORTS} exists; outputs will not be overwritten")
    frozen = (STAGE1, PREDICTIONS, JOINED)
    before = {str(path): sha256(path) for path in frozen}
    stage1 = pd.read_parquet(STAGE1)
    predictions = pd.read_parquet(PREDICTIONS)
    joined = pd.read_parquet(JOINED)
    universe = add_net_profitability(
        build_funnel_universe(stage1, predictions, joined, POLICY)
    )
    observed = universe.loc[
        universe[POLICY.probability_column].notna()
        & universe.terminal_premium_return.notna()
    ].copy()
    research = observed.loc[observed.timestamp.le(RESEARCH_END)].copy()
    validation = observed.loc[observed.timestamp.ge(VALIDATION_START)].copy()
    if research.empty or validation.empty:
        raise ValueError("Both strict temporal windows must contain OOF rows")
    if research.timestamp.max() >= validation.timestamp.min():
        raise AssertionError("Research and validation windows overlap")

    research_candidates, _ = evaluate_candidates(
        research, POLICY, research.timestamp.min(), research.timestamp.max(),
        simulations=1000,
    )
    profiles = select_profiles(research_candidates)

    numeric_features = [
        column for column in predictions.columns
        if column != "timestamp"
        and pd.api.types.is_numeric_dtype(predictions[column])
        and not column.endswith("session_id")
    ]
    oof_times = joined[["timestamp"]].copy()
    feature_rows = oof_times.merge(
        predictions[["timestamp"] + numeric_features],
        on="timestamp", how="left", validate="one_to_one",
    )
    feature_research = feature_rows.loc[
        feature_rows.timestamp.le(RESEARCH_END)
    ]
    feature_validation = feature_rows.loc[
        feature_rows.timestamp.ge(VALIDATION_START)
    ]
    drift = feature_drift_table(
        feature_research, feature_validation, numeric_features
    )
    probability_psi = population_stability_index(
        research[POLICY.probability_column],
        validation[POLICY.probability_column],
    )
    median_feature_psi = float(drift.psi.replace([float("inf")], pd.NA).median())

    comparison_rows = []
    stability_rows = []
    validation_logs = []
    equity_curves = []
    validation_dates = pd.DatetimeIndex(
        validation.timestamp.dt.normalize().drop_duplicates()
    )
    for offset, (profile, candidate) in enumerate(profiles.items()):
        research_trades, research_metrics = evaluate_frozen_engine(
            research, candidate, POLICY,
            research.timestamp.min(), research.timestamp.max(),
            seed=100 + offset,
        )
        validation_trades, validation_metrics = evaluate_frozen_engine(
            validation, candidate, POLICY,
            validation.timestamp.min(), validation.timestamp.max(),
            seed=200 + offset,
        )
        for period, metrics in (
            ("RESEARCH", research_metrics),
            ("VALIDATION", validation_metrics),
        ):
            comparison_rows.append({
                "profile": profile,
                "candidate_id": candidate.candidate_id,
                "period": period,
                **metrics,
            })
        verdict = stability_verdict(
            research_metrics, validation_metrics,
            probability_psi, median_feature_psi,
        )
        stability_rows.append({
            "profile": profile,
            "candidate_id": candidate.candidate_id,
            "research_decision_coverage": research_metrics["coverage"],
            "validation_decision_coverage": validation_metrics["coverage"],
            "probability_psi": probability_psi,
            "median_feature_psi": median_feature_psi,
            **verdict,
        })
        validation_trades.insert(0, "profile", profile)
        validation_trades.insert(1, "candidate_id", candidate.candidate_id)
        validation_logs.append(validation_trades)
        equity_curves.append(
            _equity_curve(validation_trades, validation_dates, profile)
        )

    comparison = pd.DataFrame(comparison_rows)
    stability = pd.DataFrame(stability_rows)
    trade_log = pd.concat(validation_logs, ignore_index=True)
    equity = pd.concat(equity_curves, ignore_index=True)
    REPORTS.mkdir(parents=True)
    comparison.to_csv(REPORTS / "research_vs_validation.csv", index=False)
    trade_log.to_csv(REPORTS / "validation_trade_log.csv", index=False)
    equity.to_csv(REPORTS / "validation_equity_curve.csv", index=False)
    stability.to_csv(REPORTS / "engine_stability.csv", index=False)
    drift.to_csv(REPORTS / "feature_drift.csv", index=False)
    pd.DataFrame([{
        "model": POLICY.model,
        "research_mean_probability": research[POLICY.probability_column].mean(),
        "validation_mean_probability": validation[POLICY.probability_column].mean(),
        "research_std_probability": research[POLICY.probability_column].std(),
        "validation_std_probability": validation[POLICY.probability_column].std(),
        "probability_psi": probability_psi,
    }]).to_csv(REPORTS / "probability_drift.csv", index=False)

    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed")
    overall_pass = bool(stability.passes_all_tolerances.all())
    selected_lines = "\n".join(
        f"- {profile}: `{row.candidate_id}`"
        for profile, row in profiles.items()
    )
    result_lines = []
    for row in stability.itertuples(index=False):
        validation_row = comparison.loc[
            comparison.profile.eq(row.profile)
            & comparison.period.eq("VALIDATION")
        ].iloc[0]
        result_lines.append(
            f"- {row.profile}: {int(validation_row.executed_trades)} trades, "
            f"PF {validation_row.profit_factor:.3f}, "
            f"CAGR {validation_row.cagr:.2%}, "
            f"max DD {validation_row.maximum_drawdown:.2%}, "
            f"pass={row.passes_all_tolerances}"
        )
    report = f"""# Step 20H-R Strict Temporal Validation

## Isolation

- Research: {research.timestamp.min()} through {research.timestamp.max()}
- Validation: {validation.timestamp.min()} through {validation.timestamp.max()}
- Research OOF rows: {len(research):,}
- Untouched validation OOF rows: {len(validation):,}
- Overlap: none

Candidates were ranked using only the research window. The frozen selections:

{selected_lines}

## Validation

{chr(10).join(result_lines)}

- Probability PSI: {probability_psi:.4f}
- Median feature PSI: {median_feature_psi:.4f}
- Overall tolerance decision: {"PASS" if overall_pass else "REJECT"}

Every profile used the unchanged CatBoost OOF probability, threshold 0.90,
Step 20D costs, 15-minute response horizon, one active trade, and 1,000
Monte Carlo paths per window.
"""
    (REPORTS / "validation_report.md").write_text(report, encoding="utf-8")
    recommendation = (
        "# Step 20H-R Recommendation\n\n"
        + (
            "All predefined tolerances passed. The selected engine may be "
            "carried forward as **Production Candidate v1**, but this is not "
            "production promotion.\n"
            if overall_pass else
            "At least one predefined degradation or drift tolerance failed. "
            "**Reject production candidacy** and keep the current production "
            "state unchanged.\n"
        )
    )
    (REPORTS / "recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    metadata = {
        "step": "20H-R",
        "research_only": True,
        "research_end": str(RESEARCH_END),
        "validation_start": str(VALIDATION_START),
        "research_rows": len(research),
        "validation_rows": len(validation),
        "selected_profiles": {
            name: row.candidate_id for name, row in profiles.items()
        },
        "tolerances": TOLERANCES,
        "production_candidate_v1_recommended": overall_pass,
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "temporal_overlap": False,
            "selection_used_validation_rows": False,
            "monte_carlo_paths_per_profile_period": 1000,
            "probabilities_regenerated": False,
            "models_retrained": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

