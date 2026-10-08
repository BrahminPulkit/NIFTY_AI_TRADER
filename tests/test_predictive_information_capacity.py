import numpy as np
import pandas as pd
import pytest

from src.predictive_information_capacity import (
    binary_entropy,
    class_separability,
    feature_redundancy,
    information_capacity,
)


def _frame():
    return pd.DataFrame({
        "signal": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
        "duplicate": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
        "constant": [1.0] * 8,
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
        "year": [2024] * 4 + [2025] * 4,
    })


def test_binary_entropy_balanced_target():
    assert binary_entropy(pd.Series([0, 0, 1, 1])) == pytest.approx(1.0)


def test_information_and_redundancy_are_deterministic():
    frame = _frame()
    first, _ = information_capacity(frame, ["signal", "duplicate", "constant"], "target")
    second, _ = information_capacity(frame, ["signal", "duplicate", "constant"], "target")
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    redundancy = feature_redundancy(
        frame, ["signal", "duplicate", "constant"], first
    )
    signal = redundancy.set_index("feature").loc["signal"]
    assert signal.max_absolute_spearman == pytest.approx(1.0)
    assert bool(signal.highly_redundant_095)


def test_separability_detects_known_signal():
    result = class_separability(_frame(), ["signal"], "target").iloc[0]
    assert result.orientation_free_univariate_auc == pytest.approx(1.0)
    assert result.ks_statistic == pytest.approx(1.0)
