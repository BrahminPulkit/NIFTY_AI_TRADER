import json

import pandas as pd
import pytest

from src.live_inference_engine import SafeBlock, TrueLiveInferenceEngine
from src.live_pipeline_monitor import LivePipelineMonitor


@pytest.fixture()
def engine(tmp_path):
    return TrueLiveInferenceEngine(
        "production_model", tmp_path / "journal.jsonl",
        LivePipelineMonitor(tmp_path / "health.json"))


def row(engine):
    data = pd.read_parquet("data/prediction/prediction_dataset_v1.parquet")
    return data.iloc[[-1]][engine.predictor.contract["feature_order"]]


def test_true_inference_is_deterministic_and_not_oof(engine):
    features = row(engine)
    first = engine.infer_contract_row(features, timestamp="2030-01-01 10:00+05:30")
    second = engine.infer_contract_row(features, timestamp="2030-01-01 10:00+05:30")
    assert first.probability == second.probability
    assert first.model_version == "production_candidate_v1.0.0"
    assert first.probability is not None


def test_contract_violation_safe_blocks(engine):
    features = row(engine).drop(columns=["index_rsi_14"])
    with pytest.raises(SafeBlock, match="Missing"):
        engine.infer_contract_row(features, timestamp="2030-01-01 10:00+05:30")


def test_stage1_skip_never_runs_model(engine):
    result = engine.infer_contract_row(
        pd.DataFrame(columns=engine.predictor.contract["feature_order"]),
        timestamp="2030-01-01 10:00+05:30", stage1_allowed=False,
        stage1_reason="NO_SETUP")
    assert result.decision == "SKIP"
    assert result.probability is None


def test_append_only_journal_and_health(engine):
    features = row(engine)
    engine.infer_contract_row(features, timestamp="2030-01-01 10:00+05:30")
    engine.infer_contract_row(features, timestamp="2030-01-01 10:01+05:30")
    lines = engine.journal.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] != json.loads(lines[1])["timestamp"]
    assert engine.monitor.snapshot()["status"] == "HEALTHY"


def test_runtime_never_reads_oof_files():
    source = open("src/live_inference_engine.py", encoding="utf-8").read()
    cli = open(
        "commands/operations/main_live_inference.py", encoding="utf-8").read()
    assert "oof_predictions" not in source.lower()
    assert "oof_predictions" not in cli.lower()
