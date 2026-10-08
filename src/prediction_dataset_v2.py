"""Step 20J research-only normalized prediction representation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.temporal_decision_validation import population_stability_index


FEATURE_MAPPING = (
    {
        "v1_feature": "index_rolling_high",
        "v2_feature": "index_rolling_high_atr_distance",
        "representation": "ATR_NORMALISED_DISTANCE",
    },
    {
        "v1_feature": "index_ema_5",
        "v2_feature": "index_ema_5_log_change",
        "representation": "LOG_RETURN_REPRESENTATION",
    },
    {
        "v1_feature": "index_ema_9",
        "v2_feature": "index_ema_9_log_change",
        "representation": "LOG_RETURN_REPRESENTATION",
    },
    {
        "v1_feature": "index_previous_low",
        "v2_feature": "index_previous_low_atr_distance",
        "representation": "ATR_NORMALISED_DISTANCE",
    },
    {
        "v1_feature": "index_ema_50",
        "v2_feature": "index_ema_50_atr_distance",
        "representation": "ATR_NORMALISED_DISTANCE",
    },
    {
        "v1_feature": "index_ema_20",
        "v2_feature": "index_ema_20_log_change",
        "representation": "LOG_RETURN_REPRESENTATION",
    },
    {
        "v1_feature": "index_previous_high",
        "v2_feature": "index_previous_high_atr_distance",
        "representation": "ATR_NORMALISED_DISTANCE",
    },
    {
        "v1_feature": "index_rolling_low",
        "v2_feature": "index_rolling_low_atr_distance",
        "representation": "ATR_NORMALISED_DISTANCE",
    },
    {
        "v1_feature": "index_atr_14",
        "v2_feature": "index_atr_14_rolling_percentile_200",
        "representation": "ROLLING_PERCENTILE",
    },
    {
        "v1_feature": "index_rolling_range",
        "v2_feature": "index_rolling_range_atr_ratio",
        "representation": "ATR_NORMALISED_MAGNITUDE",
    },
    {
        "v1_feature": "index_candle_range",
        "v2_feature": "index_candle_range_log_change",
        "representation": "LOG_RETURN_REPRESENTATION",
    },
)


def _causal_percentile(values: pd.Series, window: int = 200) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    # Expanding warm-up avoids missing V2 records; at 200 observations this
    # becomes the exact fixed setup-history window.
    return numeric.rolling(window, min_periods=1).rank(pct=True)


def _log_change(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    positive = numeric.where(numeric.gt(0))
    result = np.log(positive / positive.shift(1))
    # The first observable setup has no prior setup. A declared neutral value
    # preserves row identity without consulting any future row.
    return result.fillna(0.0)


def build_prediction_dataset_v2(
    prediction_v1: pd.DataFrame,
    price_anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "timestamp", "index_atr_14",
        *(item["v1_feature"] for item in FEATURE_MAPPING),
    }
    if missing := sorted(required.difference(prediction_v1.columns)):
        raise ValueError(f"V1 prediction dataset missing: {missing}")
    if missing := sorted(
        {"timestamp", "index_close"}.difference(price_anchors.columns)
    ):
        raise ValueError(f"Price anchors missing: {missing}")
    v1 = prediction_v1.copy()
    anchors = price_anchors[["timestamp", "index_close"]].copy()
    for frame in (v1, anchors):
        frame["timestamp"] = pd.to_datetime(
            frame.timestamp, utc=True, errors="raise"
        ).dt.tz_convert("Asia/Kolkata")
    if v1.timestamp.duplicated().any() or not v1.timestamp.is_monotonic_increasing:
        raise ValueError("V1 timestamps must be unique and chronological")
    if anchors.timestamp.duplicated().any():
        raise ValueError("Anchor timestamps must be unique")
    data = v1.merge(
        anchors, on="timestamp", how="left", validate="one_to_one"
    )
    if data.index_close.isna().any():
        raise ValueError("Every V1 row requires a same-timestamp index close")

    atr = pd.to_numeric(data.index_atr_14, errors="coerce").astype(float)
    close = pd.to_numeric(data.index_close, errors="coerce").astype(float)
    replacements: dict[str, pd.Series] = {}
    audit_rows = []
    for mapping in FEATURE_MAPPING:
        source = mapping["v1_feature"]
        target = mapping["v2_feature"]
        representation = mapping["representation"]
        values = pd.to_numeric(data[source], errors="coerce").astype(float)
        if representation == "ATR_NORMALISED_DISTANCE":
            transformed = (values - close) / atr.replace(0, np.nan)
            formula = f"({source} - index_close) / index_atr_14"
        elif representation == "ATR_NORMALISED_MAGNITUDE":
            transformed = values / atr.replace(0, np.nan)
            formula = f"{source} / index_atr_14"
        elif representation == "ROLLING_PERCENTILE":
            transformed = _causal_percentile(values)
            formula = f"causal rolling_rank({source}, window=200, min_periods=1)"
        elif representation == "LOG_RETURN_REPRESENTATION":
            transformed = _log_change(values)
            formula = f"log({source}[t] / {source}[previous_setup]); first=0"
        else:
            raise ValueError(f"Unsupported representation: {representation}")
        replacements[target] = transformed
        audit_rows.append({
            **mapping,
            "formula": formula,
            "causal": True,
            "warmup_policy": (
                "expanding percentile until 200 setup rows"
                if representation == "ROLLING_PERCENTILE"
                else "neutral zero for first log-change row"
                if representation == "LOG_RETURN_REPRESENTATION"
                else "none"
            ),
        })

    v2 = v1.drop(
        columns=[item["v1_feature"] for item in FEATURE_MAPPING]
    ).copy()
    for target, values in replacements.items():
        v2[target] = values.to_numpy()
    if v2.timestamp.duplicated().any() or len(v2) != len(v1):
        raise AssertionError("V2 must preserve one row per V1 timestamp")
    return v2, pd.DataFrame(audit_rows)


def mapping_psi(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    mapping: pd.DataFrame,
    research_end: pd.Timestamp,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    research_v1 = v1.timestamp.le(research_end)
    validation_v1 = v1.timestamp.ge(validation_start)
    research_v2 = v2.timestamp.le(research_end)
    validation_v2 = v2.timestamp.ge(validation_start)
    rows = []
    for item in mapping.itertuples(index=False):
        original = population_stability_index(
            v1.loc[research_v1, item.v1_feature],
            v1.loc[validation_v1, item.v1_feature],
        )
        replacement = population_stability_index(
            v2.loc[research_v2, item.v2_feature],
            v2.loc[validation_v2, item.v2_feature],
        )
        rows.append({
            "v1_feature": item.v1_feature,
            "v2_feature": item.v2_feature,
            "representation": item.representation,
            "v1_psi": original,
            "v2_psi": replacement,
            "absolute_psi_improvement": original - replacement,
            "relative_psi_improvement": (
                (original - replacement) / original
                if original > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "absolute_psi_improvement", ascending=False
    )

