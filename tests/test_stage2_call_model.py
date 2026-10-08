import numpy as np
import pandas as pd

from src.stage2_call_model import (
    FORBIDDEN_INPUTS, MODEL_A_FEATURES, MODEL_B_FEATURES, TrainingProtocol,
    prepare_call_dataset, walk_forward_splits,
)


def fixture():
    timestamp = pd.date_range("2026-01-01", periods=20, freq="min", tz="Asia/Kolkata")
    aligned = pd.DataFrame({"timestamp": timestamp})
    for feature in MODEL_B_FEATURES:
        aligned[feature] = np.arange(20, dtype=float)
    outcomes = pd.DataFrame({"timestamp": timestamp, "direction": ["LONG"]*18+["SHORT"]*2,
                             "trade_allowed": 1, "final_outcome": ["WIN"]+["TIMEOUT"]*19,
                             "year": 2026, "trend_state":"UPTREND", "session_segment":"OPENING"})
    return aligned, outcomes


def test_call_dataset_is_allowed_long_and_same_rows():
    aligned, outcomes = fixture(); xa, xb, y, meta = prepare_call_dataset(aligned, outcomes)
    assert len(xa) == len(xb) == len(y) == len(meta) == 18
    assert list(xa.columns) == MODEL_A_FEATURES and list(xb.columns) == MODEL_B_FEATURES
    assert y.sum() == 1


def test_model_contract_excludes_target_side_fields():
    assert not FORBIDDEN_INPUTS.intersection(MODEL_A_FEATURES)
    assert not FORBIDDEN_INPUTS.intersection(MODEL_B_FEATURES)


def test_walk_forward_is_strictly_chronological_and_deterministic():
    protocol = TrainingProtocol(n_splits=4)
    first = walk_forward_splits(100, protocol); second = walk_forward_splits(100, protocol)
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)
        assert train_a[-1] < test_a[0]
