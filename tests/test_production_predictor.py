import json
from pathlib import Path

import pandas as pd
import pytest

from src.production_predictor import ContractViolation, ProductionPredictor


PACKAGE = Path("production_model")


@pytest.fixture(scope="module")
def predictor():
    if not (PACKAGE / "model.cbm").exists():
        pytest.skip("Step-23 package has not been frozen")
    return ProductionPredictor(PACKAGE)


def valid_row(predictor):
    frame = pd.read_parquet("data/prediction/prediction_dataset_v1.parquet")
    return frame[predictor.contract["feature_order"]].iloc[[-1]]


def test_prediction_is_deterministic(predictor):
    row = valid_row(predictor)
    first, second = predictor.predict(row), predictor.predict(row)
    assert first["probability"] == second["probability"]
    assert first["decision"] == second["decision"]
    assert first["model_version"] == "production_candidate_v1.0.0"


def test_missing_unknown_and_reordered_fields_rejected(predictor):
    row = valid_row(predictor)
    with pytest.raises(ContractViolation, match="Missing"):
        predictor.predict(row.drop(columns=row.columns[0]))
    with pytest.raises(ContractViolation, match="Unknown"):
        predictor.predict(row.assign(unknown_feature=1))
    with pytest.raises(ContractViolation, match="reordered"):
        predictor.predict(row[row.columns[::-1]])


def test_contract_has_no_forbidden_feature(predictor):
    tokens = predictor.contract["forbidden_name_tokens"]
    assert not [name for name in predictor.contract["feature_order"]
                if any(token in name.lower() for token in tokens)]


def test_manifest_threshold_and_versions_match(predictor):
    manifest = json.loads((PACKAGE / "version_manifest.json").read_text())
    assert manifest["approved_threshold"] == .9
    assert manifest["package_version"] == predictor.metadata["model_version"]

