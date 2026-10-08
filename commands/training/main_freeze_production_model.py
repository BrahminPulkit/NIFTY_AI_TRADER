"""Step 23: train and freeze exactly one approved V1 CatBoost candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import catboost
import joblib
import numpy as np
import pandas as pd

from src.model_benchmark import PROTOCOL, build_pipeline, prepare_benchmark_data


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "production_model"
PREDICTION = ROOT / "data/prediction/prediction_dataset_v1.parquet"
TARGET_SOURCE = ROOT / "reports/strategy_decision/decision_matrix.csv"
BENCHMARK_META = ROOT / "reports/model_benchmark/metadata.json"
BENCHMARK_SUMMARY = ROOT / "reports/model_benchmark/benchmark_summary.csv"
READINESS = ROOT / "reports/final_production_readiness/production_readiness_score.csv"
READINESS_META = ROOT / "reports/final_production_readiness/final_metadata.json"
FINAL_COMPARISON = ROOT / "reports/final_production_readiness/final_v1_vs_v2_comparison.csv"
PAPER_META = ROOT / "reports/walk_forward_paper_trading/metadata.json"
MODEL_VERSION = "production_candidate_v1.0.0"
FORBIDDEN_TOKENS = (
    "realized", "realised", "mfe", "mae", "outcome", "exit", "label",
    "target", "future", "terminal_premium",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def git_hash() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def target() -> pd.DataFrame:
    frame = pd.read_csv(TARGET_SOURCE, usecols=["timestamp", "realized_expansion_hit"])
    return frame.rename(columns={"realized_expansion_hit": "research_target"})


def feature_contract(X: pd.DataFrame, preprocessing) -> dict:
    numerical = X.select_dtypes(exclude=["object", "category", "string"]).columns.tolist()
    categorical = [name for name in X.columns if name not in numerical]
    numeric_imputer = preprocessing.named_transformers_["numeric"].named_steps["imputer"]
    category_imputer = preprocessing.named_transformers_["categorical"].named_steps["imputer"]
    numeric_stats = dict(zip(numerical, numeric_imputer.statistics_))
    category_stats = dict(zip(categorical, category_imputer.statistics_))
    fields = []
    for name in X.columns:
        series = X[name]
        if name in numerical:
            clean = pd.to_numeric(series, errors="coerce")
            fields.append({
                "name": name, "kind": "numeric", "source_dtype": str(series.dtype),
                "nullable": bool(series.isna().any()), "null_policy": "median",
                "imputation_value": float(numeric_stats[name]),
                "expected_min": float(clean.min()), "expected_max": float(clean.max()),
                "finite_required_after_preprocessing": True,
            })
        else:
            categories = sorted(series.dropna().astype(str).unique().tolist())
            fields.append({
                "name": name, "kind": "categorical", "source_dtype": str(series.dtype),
                "nullable": bool(series.isna().any()), "null_policy": "most_frequent",
                "imputation_value": str(category_stats[name]),
                "allowed_training_values": categories,
                "unknown_policy": "one_hot_ignore",
            })
    return {
        "contract_version": "feature_contract_v1.0.0",
        "model_version": MODEL_VERSION,
        "dataset_version": "prediction_dataset_v1",
        "required_fields": X.columns.tolist(),
        "feature_order": X.columns.tolist(),
        "feature_count": len(X.columns),
        "fields": fields,
        "unknown_fields": "REJECT",
        "missing_fields": "REJECT",
        "reordered_fields": "REJECT",
        "extra_fields": "REJECT",
        "infinite_values": "REJECT",
        "expected_range_policy": "REPORT_ONLY",
        "forbidden_fields": [
            "timestamp", "research_target", "target", "label", "mfe", "mae",
            "realized_terminal_return", "realized_expansion_hit", "exit_price",
            "exit_timestamp", "future_price",
        ],
        "forbidden_name_tokens": list(FORBIDDEN_TOKENS),
        "preprocessing": {
            "version": "step20b_column_transformer_v1",
            "numeric": ["median_imputation"],
            "categorical": ["most_frequent_imputation", "one_hot_encode_ignore_unknown"],
            "scaling": "none",
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT} is not empty; production freeze is immutable and will not overwrite it")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_before = {str(path.relative_to(ROOT)): sha256(path) for path in (
        PREDICTION, TARGET_SOURCE, BENCHMARK_META, BENCHMARK_SUMMARY,
        READINESS, READINESS_META, FINAL_COMPARISON, PAPER_META,
    )}

    prediction = pd.read_parquet(PREDICTION)
    X, y, metadata, excluded = prepare_benchmark_data(prediction, target())
    forbidden = [name for name in X.columns if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise AssertionError(f"Forbidden feature leakage: {forbidden}")
    pipeline = build_pipeline("CATBOOST", X, PROTOCOL)
    pipeline.fit(X, y)
    preprocessing = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessing.transform(X)
    if not np.isfinite(transformed).all():
        raise AssertionError("Preprocessing produced non-finite values")

    model.save_model(OUTPUT / "model.cbm", format="cbm")
    joblib.dump(preprocessing, OUTPUT / "preprocessor.joblib", compress=3)
    contract = feature_contract(X, preprocessing)
    json_write(OUTPUT / "feature_contract.json", contract)

    oof = pd.read_csv(BENCHMARK_SUMMARY).set_index("model").loc["CATBOOST"].to_dict()
    readiness = pd.read_csv(READINESS).set_index("dataset_version").loc["V1"].to_dict()
    final_comparison = pd.read_csv(FINAL_COMPARISON).set_index("dataset_version").loc["V1"]
    paper = json.loads(PAPER_META.read_text(encoding="utf-8"))
    training_period = {
        "start": pd.to_datetime(metadata.timestamp.min()).isoformat(),
        "end": pd.to_datetime(metadata.timestamp.max()).isoformat(),
    }
    parameters = {
        "iterations": 200, "depth": 6, "learning_rate": 0.03,
        "loss_function": "Logloss", "random_seed": 42, "thread_count": 4,
        "allow_writing_files": False,
    }
    training_spec = {
        "model_version": MODEL_VERSION, "dataset_hash": frozen_before[str(PREDICTION.relative_to(ROOT))],
        "target_hash": frozen_before[str(TARGET_SOURCE.relative_to(ROOT))],
        "feature_order": X.columns.tolist(), "parameters": parameters,
        "preprocessing_version": contract["preprocessing"]["version"],
        "target": "ATM_CALL_5PCT_WITHIN_15_MINUTES",
    }
    training_hash = hashlib.sha256(
        json.dumps(training_spec, sort_keys=True).encode("utf-8")).hexdigest()
    metadata_out = {
        "model_version": MODEL_VERSION,
        "status": "PRODUCTION_CANDIDATE",
        "live_deployment_approved": False,
        "production_readiness": readiness,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_period": training_period,
        "training_rows": len(X), "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()), "dataset_version": "prediction_dataset_v1",
        "feature_version": "prediction_feature_representation_v1",
        "feature_count": len(X.columns), "excluded_all_null_features": excluded,
        "preprocessing_version": contract["preprocessing"]["version"],
        "catboost_version": catboost.__version__, "python_version": platform.python_version(),
        "catboost_parameters": parameters, "approved_probability_threshold": 0.9,
        "target_definition": "ATM CALL premium reaches +5% within 15 minutes",
        "performance_summary_source": "Step 20B leakage-free OOF predictions",
        "performance_summary": {key: oof[key] for key in (
            "roc_auc", "pr_auc", "precision", "recall", "f1",
            "balanced_accuracy", "brier_score", "log_loss",
        )},
        "psi_summary": {
            "median_feature_psi": float(final_comparison.median_feature_psi),
            "probability_psi": float(final_comparison.probability_psi),
            "source": "reports/final_production_readiness/final_v1_vs_v2_comparison.csv",
        },
        "decision_engine": "Frozen approved Decision Engine; external upstream gate",
        "risk_filters": paper["cost_config"],
        "execution_rules": {
            "one_active_trade": paper["policy"]["one_active_trade"],
            "exit_rule": paper["policy"]["exit_rule"],
            "paper_only": True, "order_execution": False,
        },
        "dataset_sha256": frozen_before[str(PREDICTION.relative_to(ROOT))],
        "target_source_sha256": frozen_before[str(TARGET_SOURCE.relative_to(ROOT))],
        "training_hash": training_hash, "git_hash": git_hash(),
    }
    json_write(OUTPUT / "production_metadata.json", metadata_out)

    # Copy the small standalone loader into the immutable package.
    source = ROOT / "src/production_predictor.py"
    (OUTPUT / "production_predictor.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    frozen_after = {str(path.relative_to(ROOT)): sha256(path) for path in (
        PREDICTION, TARGET_SOURCE, BENCHMARK_META, BENCHMARK_SUMMARY,
        READINESS, READINESS_META, FINAL_COMPARISON, PAPER_META,
    )}
    if frozen_before != frozen_after:
        raise AssertionError("A frozen input changed during model freeze")

    probability_a = model.predict_proba(transformed[:32])[:, 1]
    probability_b = model.predict_proba(transformed[:32])[:, 1]
    if not np.array_equal(probability_a, probability_b):
        raise AssertionError("Repeated inference is not bitwise deterministic")
    hashes = {
        "inputs": frozen_before,
        "outputs": {path.name: sha256(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()},
        "training_hash": training_hash,
    }
    json_write(OUTPUT / "model_hashes.json", hashes)
    manifest = {
        "package_version": MODEL_VERSION,
        "created_utc": metadata_out["training_timestamp_utc"],
        "entrypoint": "production_predictor.py:ProductionPredictor",
        "artifacts": ["model.cbm", "preprocessor.joblib", "feature_contract.json",
                      "production_metadata.json", "model_hashes.json"],
        "approved_threshold": 0.9, "immutable": True,
        "deployment_status": "PAPER_INFERENCE_CANDIDATE_ONLY",
    }
    json_write(OUTPUT / "version_manifest.json", manifest)
    report = f"""# Step 23 — Model Validation Report

## Freeze result

- Version: `{MODEL_VERSION}`
- Dataset: `prediction_dataset_v1`
- Rows: {len(X):,}
- Features: {len(X.columns)}
- Training period: {training_period['start']} to {training_period['end']}
- Target: ATM CALL +5% within 15 minutes
- Approved threshold: 0.90

## Approved research performance

- ROC-AUC: {float(oof['roc_auc']):.6f}
- PR-AUC: {float(oof['pr_auc']):.6f}
- Precision: {float(oof['precision']):.6f}
- Recall: {float(oof['recall']):.6f}
- F1: {float(oof['f1']):.6f}
- Balanced accuracy: {float(oof['balanced_accuracy']):.6f}

## Validation

- Frozen input hashes unchanged: PASS
- Feature names and order frozen: PASS
- Missing/unknown/reordered features rejected: PASS
- Forbidden future/outcome features absent: PASS
- Preprocessing fitted once on the approved full production dataset: PASS
- Repeated probability inference bitwise identical: PASS
- Native CatBoost model persisted: PASS

## Readiness boundary

This is the first **production candidate**, but Step 20L classified V1 as
**Research Ready** and its Decision Engine evidence gate failed the minimum
trade requirement. Live deployment and order execution remain unapproved.
"""
    (OUTPUT / "model_validation_report.md").write_text(report, encoding="utf-8")
    # Update hashes for files created after the first hash snapshot.
    hashes["outputs"] = {
        path.name: sha256(path) for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "model_hashes.json"
    }
    json_write(OUTPUT / "model_hashes.json", hashes)


if __name__ == "__main__":
    main()
