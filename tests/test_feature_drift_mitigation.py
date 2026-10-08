import pandas as pd

from src.feature_drift_mitigation import (
    categorical_psi,
    classify_feature,
    representation_candidates,
)


def test_feature_classification_is_deterministic():
    assert classify_feature("index_ema_20") == "Absolute price feature"
    assert classify_feature("index_log_return") == "Relative feature"
    assert classify_feature("index_minutes_from_open") == "Time feature"
    assert classify_feature("option_ema_20") == "Option feature"


def test_categorical_psi_is_zero_for_identical_data():
    values = pd.Series(["A", "B", "A"])
    assert categorical_psi(values, values) == 0


def test_relative_price_representation_reduces_level_drift():
    rows = 300
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "index_close": [100.0] * 150 + [200.0] * 150,
        "option_close": [10.0] * rows,
        "index_ema_20": [99.0] * 150 + [198.0] * 150,
        "index_atr_14": [2.0] * 150 + [4.0] * 150,
    })
    research = data.index < 150
    validation = ~research
    candidates = representation_candidates(data, research, validation)
    relative = candidates.loc[
        candidates.source_feature.eq("index_ema_20")
        & candidates.representation.eq("RELATIVE_HIGH_LOW")
    ].iloc[0]
    assert relative.candidate_psi < relative.original_psi

