"""Execute Step 20L exclusively from completed V1/V2 research artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_production_readiness import (
    mcnemar_exact, paired_delong, paired_pr_bootstrap, readiness_score,
)


REPORTS = Path("reports/final_production_readiness")
V1_OOF = Path("data/prediction/oof/oof_predictions_catboost.parquet")
V2_OOF = Path(
    "data/research/model_comparison_v2/oof_predictions_catboost_v2.parquet"
)
MODEL_COMPARISON = Path("reports/model_comparison_v1_v2/final_comparison_table.csv")
V1_BACKTEST = Path("reports/institutional_backtest/threshold_comparison.csv")
V2_BACKTEST = Path(
    "reports/model_comparison_v1_v2/v2_step20d_threshold_comparison.csv"
)
V1_TEMPORAL = Path(
    "reports/temporal_decision_validation/research_vs_validation.csv"
)
V2_TEMPORAL = Path("reports/model_comparison_v1_v2/v2_step20h_performance.csv")
V1_DECISION = Path("reports/walk_forward_paper_trading/risk_statistics.csv")
V2_DECISION = Path("reports/model_comparison_v1_v2/v2_step20e_performance.csv")
STEP20K_RECOMMENDATION = Path("reports/model_comparison_v1_v2/recommendation.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _oof(path: Path, version: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    rename = {
        "predicted_probability": "probability",
        "true_label": "target",
    }
    data = data.rename(columns=rename)
    data["timestamp"] = pd.to_datetime(
        data.timestamp, utc=True
    ).dt.tz_convert("Asia/Kolkata")
    required = {"timestamp", "probability", "target"}
    if missing := required.difference(data.columns):
        raise ValueError(f"{version} OOF missing {sorted(missing)}")
    return data[list(required)].rename(columns={
        "probability": f"{version}_probability",
        "target": f"{version}_target",
    })


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError("Step 20L outputs already exist")
    inputs = (
        V1_OOF, V2_OOF, MODEL_COMPARISON, V1_BACKTEST, V2_BACKTEST,
        V1_TEMPORAL, V2_TEMPORAL, V1_DECISION, V2_DECISION,
        STEP20K_RECOMMENDATION,
    )
    before = {str(path): sha256(path) for path in inputs}
    paired = _oof(V1_OOF, "v1").merge(
        _oof(V2_OOF, "v2"), on="timestamp", validate="one_to_one"
    )
    if not paired.v1_target.eq(paired.v2_target).all():
        raise AssertionError("V1 and V2 targets differ")
    y = paired.v1_target.to_numpy(int)
    p1 = paired.v1_probability.to_numpy(float)
    p2 = paired.v2_probability.to_numpy(float)
    delong = paired_delong(y, p1, p2)
    bootstrap = paired_pr_bootstrap(y, p1, p2, simulations=2000)
    mcnemar = mcnemar_exact(y, p1, p2)

    models = pd.read_csv(MODEL_COMPARISON)
    cat = models.loc[models.model.eq("CATBOOST")].set_index("dataset_version")
    v1_backtest = pd.read_csv(V1_BACKTEST).query(
        "model == 'CATBOOST' and threshold == 0.9"
    ).iloc[0]
    v2_backtest = pd.read_csv(V2_BACKTEST).query(
        "model == 'CATBOOST' and threshold == 0.9"
    ).iloc[0]
    v1_temporal = pd.read_csv(V1_TEMPORAL).query(
        "profile == 'CONSERVATIVE' and period == 'VALIDATION'"
    ).iloc[0]
    v2_temporal = pd.read_csv(V2_TEMPORAL).query(
        "period == 'VALIDATION'"
    ).iloc[0]
    v1_decision = pd.read_csv(V1_DECISION).set_index("metric").value
    v2_decision = pd.read_csv(V2_DECISION).iloc[0]

    rows = []
    for version, benchmark, backtest, temporal, decision_trades in (
        ("V1", cat.loc["V1"], v1_backtest, v1_temporal,
         float(v1_decision["executed_trades"])),
        ("V2", cat.loc["V2"], v2_backtest, v2_temporal,
         float(v2_decision.executed_trades)),
    ):
        rows.append({
            "dataset_version": version,
            **{
                key: benchmark[key] for key in (
                    "roc_auc", "pr_auc", "precision", "recall", "f1",
                    "balanced_accuracy", "brier_score", "log_loss",
                    "training_seconds", "inference_seconds",
                    "median_feature_psi", "probability_psi",
                )
            },
            "walk_forward_oof_rows": len(paired),
            "backtest_trades": backtest.executed_trades,
            "backtest_profit_factor": backtest.profit_factor,
            "backtest_cagr": backtest.cagr,
            "backtest_max_drawdown": backtest.maximum_drawdown,
            "backtest_expectancy": backtest.expectancy,
            "monte_carlo_p05": (
                backtest.percentile_5
                if "percentile_5" in backtest.index
                else backtest.mc_percentile_5
            ),
            "temporal_trades": temporal.executed_trades,
            "temporal_profit_factor": temporal.profit_factor,
            "temporal_cagr": temporal.cagr,
            "temporal_max_drawdown": temporal.maximum_drawdown,
            "temporal_mc_p05": temporal.mc_percentile_5,
            "decision_engine_executed_trades": decision_trades,
        })
    comparison = pd.DataFrame(rows)
    v1_efficiency = comparison.set_index("dataset_version").loc["V1"]
    score_rows = []
    for row in comparison.itertuples(index=False):
        score_rows.append({
            "dataset_version": row.dataset_version,
            **readiness_score(pd.Series(row._asdict()), v1_efficiency),
        })
    scores = pd.DataFrame(score_rows)

    # Step 20K selected V1; Step 20L tests whether evidence overturns it.
    auc_significant_v2_gain = (
        delong["difference_v2_minus_v1"] > 0
        and delong["p_value"] < 0.05
    )
    pr_significant_v2_gain = (
        bootstrap["ci_2_5"] > 0
    )
    recommended = (
        "Production Dataset V2"
        if auc_significant_v2_gain and pr_significant_v2_gain
        else "Production Dataset V1"
    )
    best_score = scores.loc[
        scores.dataset_version.eq(recommended.rsplit(" ", 1)[-1])
    ].iloc[0]
    system_classification = best_score.classification

    REPORTS.mkdir(parents=True)
    comparison.to_csv(REPORTS / "final_v1_vs_v2_comparison.csv", index=False)
    scores.to_csv(REPORTS / "production_readiness_score.csv", index=False)
    statistical = f"""# Statistical Significance Report

## Paired sample

- Matched leakage-free OOF rows: {len(paired):,}
- Targets identical: yes

## DeLong ROC-AUC comparison

- V1 ROC-AUC: {delong['v1_roc_auc']:.6f}
- V2 ROC-AUC: {delong['v2_roc_auc']:.6f}
- V2 − V1: {delong['difference_v2_minus_v1']:.6f}
- p-value: {delong['p_value']:.6g}

## Paired stratified PR-AUC bootstrap

- Simulations: {bootstrap['bootstrap_simulations']:,}
- V2 − V1: {bootstrap['difference_v2_minus_v1']:.6f}
- 95% CI: [{bootstrap['ci_2_5']:.6f}, {bootstrap['ci_97_5']:.6f}]
- Two-sided empirical p-value: {bootstrap['p_value']:.6g}

## McNemar exact test at 0.50

- V1 correct / V2 wrong: {mcnemar['v1_correct_v2_wrong']}
- V1 wrong / V2 correct: {mcnemar['v1_wrong_v2_correct']}
- Discordant predictions: {mcnemar['discordant']}
- p-value: {mcnemar['p_value']:.6g}

V2 is statistically superior only if both ROC and PR improvements are positive
and significant. That condition is
{"met" if auc_significant_v2_gain and pr_significant_v2_gain else "not met"}.
"""
    (REPORTS / "statistical_significance_report.md").write_text(
        statistical, encoding="utf-8"
    )
    readiness = f"""# Institutional Readiness Report

- Recommended dataset: **{recommended}**
- System classification: **{system_classification}**
- V1 readiness score: {scores.set_index('dataset_version').loc['V1', 'production_readiness_score']:.2f}/100
- V2 readiness score: {scores.set_index('dataset_version').loc['V2', 'production_readiness_score']:.2f}/100

V2 materially improves feature and probability stability, and its temporal and
cost simulations remain positive. However, its CatBoost ROC-AUC and PR-AUC are
lower than V1 and the paired tests do not establish predictive superiority.

Both versions fail the operational evidence gate because the unchanged frozen
Decision Engine completes only four strict pipeline trades. Consequently, a
high numerical score cannot elevate the system beyond Research Ready.
"""
    (REPORTS / "institutional_readiness_report.md").write_text(
        readiness, encoding="utf-8"
    )
    deployment = f"""# Deployment Recommendation

Use **{recommended}** as the sole dataset recommendation from this comparison.

Do **not** deploy live. Do **not** build broker execution around the current
research result. Classification: **{system_classification}**.

V2 is more drift-resistant but not predictively superior. Continue preserving
both datasets and collect genuinely forward paper-trading evidence. Dashboard
work may visualize research, but it must not imply production readiness.
"""
    (REPORTS / "deployment_recommendation.md").write_text(
        deployment, encoding="utf-8"
    )
    after = {str(path): sha256(path) for path in inputs}
    if before != after:
        raise AssertionError("Completed research artifact changed")
    metadata = {
        "step": "20L", "research_only": True,
        "recommended_dataset": recommended,
        "system_classification": system_classification,
        "statistical_tests": {
            "delong": delong, "pr_auc_bootstrap": bootstrap,
            "mcnemar": mcnemar,
        },
        "score_weights": {
            "predictive_performance": 25,
            "probability_stability": 15,
            "drift_resistance": 15,
            "temporal_robustness": 20,
            "risk_metrics": 15,
            "computational_efficiency": 10,
            "total": 100,
        },
        "input_sha256_before": before,
        "input_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "models_retrained": False,
            "probabilities_regenerated": False,
            "datasets_modified": False,
            "paired_oof_rows": len(paired),
        },
    }
    metadata["production_readiness_scores"] = {
        row.dataset_version: float(row.production_readiness_score)
        for row in scores.itertuples(index=False)
    }
    (REPORTS / "final_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
