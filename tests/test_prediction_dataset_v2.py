import pandas as pd

from src.prediction_dataset_v2 import (
    FEATURE_MAPPING,
    build_prediction_dataset_v2,
)


def _frame(rows=250):
    timestamps = pd.date_range(
        "2024-01-01 09:15", periods=rows, freq="min", tz="Asia/Kolkata"
    )
    data = {"timestamp": timestamps, "untouched": range(rows)}
    for mapping in FEATURE_MAPPING:
        data[mapping["v1_feature"]] = [
            100.0 + index * 0.1 for index in range(rows)
        ]
    data["index_atr_14"] = [2.0 + index * 0.001 for index in range(rows)]
    frame = pd.DataFrame(data)
    anchors = pd.DataFrame({
        "timestamp": timestamps,
        "index_close": [100.0 + index * 0.1 for index in range(rows)],
    })
    return frame, anchors


def test_v2_preserves_rows_and_unmodified_fields():
    v1, anchors = _frame()
    original = v1.copy(deep=True)
    v2, mapping = build_prediction_dataset_v2(v1, anchors)
    pd.testing.assert_frame_equal(v1, original, check_exact=True)
    assert v2.timestamp.equals(v1.timestamp)
    assert v2.untouched.equals(v1.untouched)
    assert len(mapping) == 11


def test_v2_replaces_every_approved_feature():
    v1, anchors = _frame()
    v2, _ = build_prediction_dataset_v2(v1, anchors)
    for mapping in FEATURE_MAPPING:
        assert mapping["v1_feature"] not in v2
        assert mapping["v2_feature"] in v2
    assert not v2[
        [mapping["v2_feature"] for mapping in FEATURE_MAPPING]
    ].isna().any().any()


def test_v2_prefix_invariance():
    v1, anchors = _frame()
    full, _ = build_prediction_dataset_v2(v1, anchors)
    prefix, _ = build_prediction_dataset_v2(v1.iloc[:100], anchors)
    pd.testing.assert_frame_equal(
        prefix, full.iloc[:100].reset_index(drop=True), check_exact=True
    )

