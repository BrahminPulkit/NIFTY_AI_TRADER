"""Central, read-only artifact registry and cached loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS = {
    "joined_probability": "data/research/probability_threshold/joined_probability_dataset.parquet",
    "prediction_v1": "data/prediction/prediction_dataset_v1.parquet",
    "prediction_v2": "data/prediction/prediction_dataset_v2.parquet",
    "decision_matrix": "reports/strategy_decision/decision_matrix.csv",
    "decision_funnel": "reports/decision_funnel_research/decision_funnel.csv",
    "filter_importance": "reports/decision_funnel_research/filter_importance.csv",
    "paper_trades": "reports/walk_forward_paper_trading/trade_log.csv",
    "paper_skips": "reports/walk_forward_paper_trading/skipped_trade_log.csv",
    "paper_equity": "reports/walk_forward_paper_trading/paper_equity_curve.csv",
    "benchmark_v1": "reports/model_benchmark/benchmark_summary.csv",
    "benchmark_v2": "reports/model_comparison_v1_v2/v2_benchmark_summary.csv",
    "model_comparison": "reports/final_production_readiness/final_v1_vs_v2_comparison.csv",
    "readiness": "reports/final_production_readiness/production_readiness_score.csv",
    "v2_roc": "reports/model_comparison_v1_v2/v2_roc_curve.csv",
    "v2_pr": "reports/model_comparison_v1_v2/v2_pr_curve.csv",
    "feature_importance": "reports/model_comparison_v1_v2/v2_feature_importance.csv",
    "shap": "reports/model_comparison_v1_v2/v2_shap_summary.csv",
    "feature_drift": "reports/feature_drift_mitigation/feature_drift_root_cause.csv",
    "correlation": "reports/model_benchmark/feature_correlation.csv",
    "monte_carlo": "reports/institutional_backtest/monte_carlo_distribution.csv",
    "walk_forward": "reports/model_comparison_v1_v2/v2_walk_forward_folds.csv",
    "readiness_metadata": "reports/final_production_readiness/final_metadata.json",
    "deployment_recommendation": "reports/final_production_readiness/deployment_recommendation.md",
    "aligned_market": "data/aligned/canonical_index_option_aligned.parquet",
    "stage1_setups": "data/setups/stage1_setup_dataset.parquet",
    "catboost_oof": "data/prediction/oof/oof_predictions_catboost.parquet",
    "live_pipeline_health": "logs/live_inference/pipeline_health.json",
}


def artifact_path(name: str) -> Path:
    if name not in ARTIFACTS:
        raise KeyError(f"Unknown dashboard artifact: {name}")
    return PROJECT_ROOT / ARTIFACTS[name]


@st.cache_data(show_spinner=False)
def _csv(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _parquet(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _json(path: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _text(path: str, modified_ns: int) -> str:
    del modified_ns
    return Path(path).read_text(encoding="utf-8")


def load_frame(name: str) -> pd.DataFrame:
    path = artifact_path(name)
    if not path.exists():
        return pd.DataFrame()
    loader = _parquet if path.suffix == ".parquet" else _csv
    return loader(str(path), path.stat().st_mtime_ns).copy()


def load_json(name: str) -> dict:
    path = artifact_path(name)
    return _json(str(path), path.stat().st_mtime_ns).copy() if path.exists() else {}


def load_text(name: str) -> str:
    path = artifact_path(name)
    return _text(str(path), path.stat().st_mtime_ns) if path.exists() else ""


def artifact_health() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "artifact": name,
            "status": "AVAILABLE" if artifact_path(name).exists() else "MISSING",
            "path": relative,
        }
        for name, relative in ARTIFACTS.items()
    ])


def latest_snapshot() -> dict:
    joined = load_frame("joined_probability")
    if joined.empty:
        return {}
    joined["timestamp"] = pd.to_datetime(joined.timestamp)
    row = joined.sort_values("timestamp").iloc[-1]
    return row.to_dict()
