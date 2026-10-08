"""Step 20K complete V1/V2 research comparison under the frozen protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.decision_engine_optimization import candidate_mask
from src.decision_funnel_research import simulate_selection
from src.institutional_backtest import (
    CONFIG, monte_carlo, monte_carlo_summary, performance_metrics,
    simulate_scenario,
)
from src.model_benchmark import (
    MODEL_NAMES, PROTOCOL, chronological_splits, metrics,
    prepare_benchmark_data, run_walk_forward, transformed_importance,
)
from src.model_comparison_v2 import (
    choose_dataset, curve_tables, numeric_feature_stability,
    probability_stability,
)
from src.probability_threshold_research import (
    THRESHOLDS, evaluate_thresholds, validate_joined_dataset,
)
from src.temporal_decision_validation import (
    RESEARCH_END, VALIDATION_START, evaluate_frozen_engine,
    feature_drift_table, population_stability_index, stability_verdict,
)
from src.walk_forward_paper_trading import ReplayPolicy, replay_frozen_stream, replay_metrics


V1 = Path("data/prediction/prediction_dataset_v1.parquet")
V2 = Path("data/prediction/prediction_dataset_v2.parquet")
DECISIONS = Path("reports/strategy_decision/decision_matrix.csv")
V1_SUMMARY = Path("reports/model_benchmark/benchmark_summary.csv")
V1_CAT = Path("data/prediction/oof/oof_predictions_catboost.parquet")
V1_XGB = Path("data/prediction/oof/oof_predictions_xgboost.parquet")
JOINED_V1 = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
STAGE1 = Path("data/setups/stage1_setup_dataset.parquet")
REPORTS = Path("reports/model_comparison_v1_v2")
DATA = Path("data/research/model_comparison_v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target() -> pd.DataFrame:
    data = pd.read_csv(
        DECISIONS, usecols=["timestamp", "realized_expansion_hit"]
    )
    return data.rename(columns={"realized_expansion_hit": "research_target"})


def benchmark_v2():
    prediction = pd.read_parquet(V2)
    X, y, metadata, excluded = prepare_benchmark_data(prediction, target())
    all_oof, folds, importance_rows, permutation_rows, shap_rows = [], [], [], [], []
    final_models = {}
    for model_name in MODEL_NAMES:
        oof, fold, fitted = run_walk_forward(X, y, metadata, model_name)
        all_oof.append(oof)
        folds.append(fold)
        for number, pipeline in enumerate(fitted, 1):
            item = transformed_importance(pipeline, model_name)
            item["model"], item["fold"] = model_name, number
            importance_rows.append(item)
        final_models[model_name] = fitted[-1]
    oof = pd.concat(all_oof, ignore_index=True)
    folds = pd.concat(folds, ignore_index=True)
    summaries = []
    for model_name, group in oof.groupby("model", sort=False):
        row = metrics(group.target, group.probability, PROTOCOL.threshold)
        timing = folds.loc[folds.model.eq(model_name)]
        row.update({
            "model": model_name,
            "training_seconds": timing.training_seconds.sum(),
            "inference_seconds": timing.inference_seconds.sum(),
        })
        summaries.append(row)
    summary = pd.DataFrame(summaries)

    last_test_start = len(X) - len(X) // (PROTOCOL.n_splits + 1)
    X_test, y_test = X.iloc[last_test_start:last_test_start + 500], y.iloc[
        last_test_start:last_test_start + 500
    ]
    for model_name, pipeline in final_models.items():
        result = permutation_importance(
            pipeline, X_test, y_test, scoring="average_precision",
            n_repeats=3, random_state=42, n_jobs=1,
        )
        permutation_rows.append(pd.DataFrame({
            "model": model_name, "feature": X.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }))
        if model_name != "LOGISTIC_REGRESSION":
            transformed = pipeline.named_steps["preprocessor"].transform(X_test)
            names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            values = shap.TreeExplainer(
                pipeline.named_steps["model"]
            ).shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            shap_rows.append(pd.DataFrame({
                "model": model_name, "transformed_feature": names,
                "mean_absolute_shap": np.abs(values).mean(axis=0),
            }))
    importance = pd.concat(importance_rows).groupby(
        ["model", "transformed_feature"]
    ).importance.mean().reset_index()
    return (
        prediction, X, y, metadata, excluded, oof, folds, summary, importance,
        pd.concat(permutation_rows, ignore_index=True),
        pd.concat(shap_rows, ignore_index=True),
    )


def joined_v2(oof: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(JOINED_V1).drop(columns=[
        "catboost_predicted_probability", "catboost_predicted_class",
        "xgboost_predicted_probability", "xgboost_predicted_class",
        "fold_id", "true_label",
    ])
    pieces = {}
    for model in ("CATBOOST", "XGBOOST"):
        item = oof.loc[oof.model.eq(model), [
            "timestamp", "target", "probability", "prediction",
        ]].copy()
        # Recover exact fold IDs from the identical TimeSeriesSplit test blocks.
        item["fold_id"] = np.concatenate([
            np.full(len(test), fold)
            for fold, (_, test) in enumerate(
                chronological_splits(7643), 1
            )
        ])
        item = item.rename(columns={
            "target": "true_label",
            "probability": f"{model.lower()}_predicted_probability",
            "prediction": f"{model.lower()}_predicted_class",
        })
        pieces[model] = item
    result = pieces["CATBOOST"].merge(
        pieces["XGBOOST"].drop(columns=["true_label", "fold_id"]),
        on="timestamp", validate="one_to_one",
    ).merge(base, on="timestamp", validate="one_to_one")
    return validate_joined_dataset(result.sort_values("timestamp").reset_index(drop=True))


def main() -> None:
    if REPORTS.exists() or DATA.exists():
        raise FileExistsError("Step 20K outputs already exist")
    frozen = (
        V1, V2, DECISIONS, V1_SUMMARY, V1_CAT, V1_XGB, JOINED_V1, STAGE1
    )
    before = {str(path): sha256(path) for path in frozen}
    started = perf_counter()
    (
        prediction_v2, X, y, metadata, excluded, oof, folds, v2_summary,
        importance, permutation, shap_summary,
    ) = benchmark_v2()
    DATA.mkdir(parents=True)
    REPORTS.mkdir(parents=True)
    oof.to_parquet(DATA / "oof_predictions_all_models_v2.parquet", index=False)
    oof.to_csv(DATA / "oof_predictions_all_models_v2.csv", index=False)
    for model in ("CATBOOST", "XGBOOST"):
        item = oof.loc[oof.model.eq(model)].copy()
        item.to_parquet(DATA / f"oof_predictions_{model.lower()}_v2.parquet", index=False)
    v2_summary.to_csv(REPORTS / "v2_benchmark_summary.csv", index=False)
    folds.to_csv(REPORTS / "v2_walk_forward_folds.csv", index=False)
    importance.to_csv(REPORTS / "v2_feature_importance.csv", index=False)
    permutation.to_csv(REPORTS / "v2_permutation_importance.csv", index=False)
    shap_summary.to_csv(REPORTS / "v2_shap_summary.csv", index=False)
    v2_summary[["model", "tn", "fp", "fn", "tp"]].to_csv(
        REPORTS / "v2_confusion_matrices.csv", index=False
    )
    roc, pr = curve_tables(oof)
    roc.to_csv(REPORTS / "v2_roc_curve.csv", index=False)
    pr.to_csv(REPORTS / "v2_pr_curve.csv", index=False)

    joined = joined_v2(oof)
    joined.to_parquet(DATA / "joined_probability_dataset_v2.parquet", index=False)
    joined.to_csv(DATA / "joined_probability_dataset_v2.csv", index=False)

    # Step 20C-V2
    threshold_stats = evaluate_thresholds(joined)
    threshold_stats.to_csv(REPORTS / "v2_step20c_thresholds.csv", index=False)

    # Step 20D-V2: exact frozen cost methodology.
    performance_rows, mc_rows = [], []
    all_trades = []
    for model in ("CATBOOST", "XGBOOST"):
        column = f"{model.lower()}_predicted_probability"
        for threshold in THRESHOLDS:
            selected = int(joined[column].ge(threshold).sum())
            trades, _ = simulate_scenario(joined, model, threshold, CONFIG)
            row = {
                "model": model, "threshold": threshold,
                **performance_metrics(trades, selected, CONFIG),
            }
            distribution = monte_carlo(
                trades, 1000, CONFIG.initial_capital,
                seed=CONFIG.random_seed + int(threshold * 100)
                + (0 if model == "CATBOOST" else 1000),
            )
            mc = monte_carlo_summary(distribution)
            row.update({
                "mc_percentile_5": mc.get("percentile_5", np.nan),
                "mc_median": mc.get("median_case", np.nan),
                "mc_percentile_95": mc.get("percentile_95", np.nan),
            })
            row["robust_after_costs"] = (
                row["net_return"] > 0 and row["profit_factor"] > 1
                and row["mc_percentile_5"] > 0
                and row["executed_trades"] >= 50
            )
            performance_rows.append(row)
            if len(trades):
                trades["model"], trades["threshold"] = model, threshold
                all_trades.append(trades)
            mc_rows.append({
                "model": model, "threshold": threshold,
                "simulations": len(distribution), **mc,
            })
    step20d = pd.DataFrame(performance_rows).sort_values(
        ["robust_after_costs", "calmar_ratio", "mc_percentile_5"],
        ascending=[False, False, False],
    )
    step20d.to_csv(REPORTS / "v2_step20d_threshold_comparison.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(REPORTS / "v2_step20d_monte_carlo.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(
        REPORTS / "v2_step20d_trade_log.csv", index=False
    )
    recommended = step20d.loc[step20d.robust_after_costs].iloc[0]
    policy = ReplayPolicy(
        model=recommended.model, threshold=float(recommended.threshold)
    )

    # Step 20E-V2, frozen original Decision Engine.
    stage1 = pd.read_parquet(STAGE1)
    prediction_v1 = pd.read_parquet(V1)
    paper, skipped, events = replay_frozen_stream(
        stage1, prediction_v1, joined, policy, CONFIG
    )
    paper.to_csv(REPORTS / "v2_step20e_trade_log.csv", index=False)
    paper_metrics = replay_metrics(
        paper, CONFIG.initial_capital,
        pd.to_datetime(stage1.timestamp.iloc[0]),
        pd.to_datetime(stage1.timestamp.iloc[-1]),
    )
    pd.DataFrame([paper_metrics]).to_csv(
        REPORTS / "v2_step20e_performance.csv", index=False
    )

    # Step 20H-R-V2 using the frozen Step 20G balanced engine definition.
    observed = joined.copy()
    selected_candidate = pd.Series({
        "source_mode": "REGIME_OR_STRATEGY",
        "enabled_rules": "EXPANSION_GT_DRAWDOWN",
    })
    research = observed.loc[observed.timestamp.le(RESEARCH_END)]
    validation = observed.loc[observed.timestamp.ge(VALIDATION_START)]
    research_trades, research_metrics = evaluate_frozen_engine(
        research, selected_candidate, policy,
        research.timestamp.min(), research.timestamp.max(), seed=501,
    )
    validation_trades, validation_metrics = evaluate_frozen_engine(
        validation, selected_candidate, policy,
        validation.timestamp.min(), validation.timestamp.max(), seed=502,
    )
    v2_numeric = numeric_feature_stability(
        prediction_v2, RESEARCH_END, VALIDATION_START
    )
    v1_prediction = pd.read_parquet(V1)
    v1_numeric = numeric_feature_stability(
        v1_prediction, RESEARCH_END, VALIDATION_START
    )
    probability_psi = population_stability_index(
        research[policy.probability_column],
        validation[policy.probability_column],
    )
    v2_median_psi = float(v2_numeric.psi.replace(np.inf, np.nan).median())
    verdict = stability_verdict(
        research_metrics, validation_metrics, probability_psi, v2_median_psi
    )
    pd.DataFrame([
        {"period": "RESEARCH", **research_metrics},
        {"period": "VALIDATION", **validation_metrics},
    ]).to_csv(REPORTS / "v2_step20h_performance.csv", index=False)
    validation_trades.to_csv(REPORTS / "v2_step20h_validation_trades.csv", index=False)
    v2_numeric.to_csv(REPORTS / "v2_feature_stability.csv", index=False)

    # Direct benchmark and stability comparison.
    v1_summary = pd.read_csv(V1_SUMMARY)
    v1_stability = probability_stability(
        pd.concat([
            pd.read_parquet(V1_CAT).assign(model="CATBOOST").rename(columns={
                "predicted_probability": "probability",
                "true_label": "target",
            }),
            pd.read_parquet(V1_XGB).assign(model="XGBOOST").rename(columns={
                "predicted_probability": "probability",
                "true_label": "target",
            }),
        ], ignore_index=True),
        RESEARCH_END, VALIDATION_START,
    )
    v2_stability = probability_stability(oof, RESEARCH_END, VALIDATION_START)
    probability_comparison = pd.concat([
        v1_stability.assign(dataset_version="V1"),
        v2_stability.assign(dataset_version="V2"),
    ])
    probability_comparison.to_csv(
        REPORTS / "probability_stability.csv", index=False
    )
    model_comparison = pd.concat([
        v1_summary.assign(dataset_version="V1"),
        v2_summary.assign(dataset_version="V2"),
    ], ignore_index=True)
    model_comparison.to_csv(REPORTS / "model_metrics_comparison.csv", index=False)

    final_rows = []
    for version, summary, median_psi in (
        ("V1", v1_summary, float(v1_numeric.psi.replace(np.inf, np.nan).median())),
        ("V2", v2_summary, v2_median_psi),
    ):
        for row in summary.itertuples(index=False):
            prob_row = probability_comparison.loc[
                probability_comparison.dataset_version.eq(version)
                & probability_comparison.model.eq(row.model)
            ]
            final_rows.append({
                "dataset_version": version,
                "model": row.model,
                "roc_auc": row.roc_auc, "pr_auc": row.pr_auc,
                "precision": row.precision, "recall": row.recall,
                "f1": row.f1, "balanced_accuracy": row.balanced_accuracy,
                "brier_score": row.brier_score, "log_loss": row.log_loss,
                "training_seconds": row.training_seconds,
                "inference_seconds": row.inference_seconds,
                "median_feature_psi": median_psi,
                "probability_psi": (
                    prob_row.probability_psi.iloc[0]
                    if len(prob_row) else np.nan
                ),
                "holdout_trades": (
                    validation_metrics["executed_trades"]
                    if version == "V2" and row.model == policy.model else np.nan
                ),
                "holdout_profit_factor": (
                    validation_metrics["profit_factor"]
                    if version == "V2" and row.model == policy.model else np.nan
                ),
                "holdout_cagr": (
                    validation_metrics["cagr"]
                    if version == "V2" and row.model == policy.model else np.nan
                ),
                "holdout_robust": (
                    verdict["passes_all_tolerances"]
                    if version == "V2" and row.model == policy.model else False
                ),
            })
    final = pd.DataFrame(final_rows)
    final.to_csv(REPORTS / "final_comparison_table.csv", index=False)
    recommendation, checks = choose_dataset(final)
    (REPORTS / "recommendation.md").write_text(
        "# Step 20K Final Dataset Recommendation\n\n"
        f"**{recommendation}**\n\n"
        + "\n".join(f"- {line}" for line in checks)
        + "\n\nResearch only; this is not production promotion.\n",
        encoding="utf-8",
    )

    after = {str(path): sha256(path) for path in frozen}
    if before != after:
        raise AssertionError("Frozen artifact changed")
    metadata_out = {
        "step": "20K", "research_only": True,
        "methodology": {
            "models": list(MODEL_NAMES), "folds": 4,
            "splitter": "expanding TimeSeriesSplit", "shuffle": False,
            "parameters": "src.model_benchmark frozen protocol",
            "target": "ATM_CALL_5PCT_WITHIN_15_MINUTES",
        },
        "recommended_dataset": recommendation,
        "recommendation_checks": checks,
        "v2_recommended_model": policy.model,
        "v2_recommended_threshold": policy.threshold,
        "runtime_seconds": perf_counter() - started,
        "frozen_sha256_before": before,
        "frozen_sha256_after": after,
        "validation": {
            "hashes_unchanged": before == after,
            "v2_oof_rows_per_model": int(
                oof.groupby("model").size().min()
            ),
            "v2_probabilities_regenerated_for_research": True,
            "v1_modified": False,
            "production_promoted": False,
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
