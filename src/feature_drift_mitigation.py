"""Step 20I research-only feature drift diagnostics and representations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.temporal_decision_validation import population_stability_index


ABSOLUTE_TOKENS = (
    "ema_", "vwap", "previous_high", "previous_low", "rolling_high",
    "rolling_low", "candle_body", "upper_wick", "lower_wick",
    "candle_range", "atr_", "rolling_range",
)
TIME_TOKENS = (
    "minutes_from_open", "minutes_to_close", "weekday", "month",
    "session_id", "history_count",
)
VOLATILITY_NORMALISED_TOKENS = (
    "rolling_volatility", "expansion_drawdown_ratio",
    "terminal_return_standard_error",
)
VOLUME_TOKENS = ("volume",)


def classify_feature(feature: str) -> str:
    if any(token in feature for token in TIME_TOKENS):
        return "Time feature"
    if feature.startswith("option_"):
        return "Option feature"
    if any(token in feature for token in VOLUME_TOKENS):
        return "Volume feature"
    if any(token in feature for token in VOLATILITY_NORMALISED_TOKENS):
        return "Volatility-normalised feature"
    if any(token in feature for token in ABSOLUTE_TOKENS):
        return "Absolute price feature"
    return "Relative feature"


def categorical_psi(research: pd.Series, validation: pd.Series) -> float:
    left = research.fillna("<NA>").astype(str)
    right = validation.fillna("<NA>").astype(str)
    levels = sorted(set(left).union(right))
    left_share = left.value_counts(normalize=True).reindex(levels, fill_value=0)
    right_share = right.value_counts(normalize=True).reindex(levels, fill_value=0)
    left_share = left_share.clip(lower=1e-6)
    right_share = right_share.clip(lower=1e-6)
    return float(
        ((right_share - left_share) * np.log(right_share / left_share)).sum()
    )


def original_feature_drift(
    data: pd.DataFrame,
    research_mask: pd.Series,
    validation_mask: pd.Series,
) -> pd.DataFrame:
    rows = []
    for feature in data.columns:
        if feature in {"timestamp", "index_close", "option_close"}:
            continue
        left, right = data.loc[research_mask, feature], data.loc[
            validation_mask, feature
        ]
        numeric = pd.api.types.is_numeric_dtype(data[feature])
        psi = (
            population_stability_index(left, right)
            if numeric else categorical_psi(left, right)
        )
        rows.append({
            "feature": feature,
            "classification": classify_feature(feature),
            "dtype": str(data[feature].dtype),
            "original_psi": psi,
            "research_mean": (
                pd.to_numeric(left, errors="coerce").mean()
                if numeric else np.nan
            ),
            "validation_mean": (
                pd.to_numeric(right, errors="coerce").mean()
                if numeric else np.nan
            ),
            "research_non_null": int(left.notna().sum()),
            "validation_non_null": int(right.notna().sum()),
        })
    return pd.DataFrame(rows)


def _rolling_zscore(values: pd.Series, window: int = 200) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    mean = numeric.rolling(window, min_periods=50).mean()
    std = numeric.rolling(window, min_periods=50).std()
    return (numeric - mean) / std.replace(0, np.nan)


def _rolling_percentile(values: pd.Series, window: int = 200) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.rolling(window, min_periods=50).rank(pct=True)


def representation_candidates(
    data: pd.DataFrame,
    research_mask: pd.Series,
    validation_mask: pd.Series,
) -> pd.DataFrame:
    """Estimate drift for causal setup-history representations."""
    rows = []
    for feature in data.columns:
        if feature in {"timestamp", "index_close", "option_close"}:
            continue
        if not pd.api.types.is_numeric_dtype(data[feature]):
            continue
        values = pd.to_numeric(data[feature], errors="coerce").astype(float)
        category = classify_feature(feature)
        if (
            category == "Time feature"
            or "flag" in feature
            or feature == "entry_signal"
        ):
            continue
        prefix = "option" if feature.startswith("option_") else "index"
        close_column = f"{prefix}_close"
        atr_column = f"{prefix}_atr_14"
        close = pd.to_numeric(data.get(close_column), errors="coerce")
        atr = pd.to_numeric(data.get(atr_column), errors="coerce")
        variants: dict[str, pd.Series] = {
            "ROLLING_Z_SCORE": _rolling_zscore(values),
            "ROLLING_PERCENTILE": _rolling_percentile(values),
        }
        if category in {"Absolute price feature", "Option feature"}:
            if close is not None:
                if any(token in feature for token in (
                    "ema_", "vwap", "previous_high", "previous_low",
                    "rolling_high", "rolling_low",
                )):
                    variants["RELATIVE_HIGH_LOW"] = values / close - 1
                    if "ema_" in feature:
                        variants["DISTANCE_FROM_EMA_PCT"] = (
                            (close - values) / close * 100
                        )
                else:
                    variants["PERCENT_OF_PRICE"] = values / close
            if atr is not None and feature != atr_column:
                if any(token in feature for token in (
                    "ema_", "vwap", "previous_high", "previous_low",
                    "rolling_high", "rolling_low",
                )):
                    variants["ATR_NORMALISED_DISTANCE"] = (
                        values - close
                    ) / atr.replace(0, np.nan)
                else:
                    variants["ATR_NORMALISED_MAGNITUDE"] = (
                        values / atr.replace(0, np.nan)
                    )
            elif feature == atr_column and close is not None:
                variants["ATR_PERCENT_OF_PRICE"] = values / close * 100
            positive = values.where(values.gt(0))
            variants["LOG_RETURN_REPRESENTATION"] = np.log(
                positive / positive.shift(1)
            )
        if "volume" in feature:
            rolling = values.rolling(200, min_periods=50).mean()
            variants["RELATIVE_TO_ROLLING_VOLUME"] = (
                values / rolling.replace(0, np.nan)
            )
            variants["LOG1P_VOLUME"] = np.log1p(values.clip(lower=0))

        original_psi = population_stability_index(
            values.loc[research_mask], values.loc[validation_mask]
        )
        for representation, transformed in variants.items():
            psi = population_stability_index(
                transformed.loc[research_mask],
                transformed.loc[validation_mask],
            )
            reduction = original_psi - psi if np.isfinite(psi) else np.nan
            rows.append({
                "source_feature": feature,
                "classification": category,
                "representation": representation,
                "original_psi": original_psi,
                "candidate_psi": psi,
                "expected_absolute_psi_reduction": reduction,
                "expected_relative_psi_reduction": (
                    reduction / original_psi if original_psi > 0 else np.nan
                ),
                "causal_window": (
                    "200 prior/current eligible setup rows"
                    if representation.startswith("ROLLING_")
                    or representation == "RELATIVE_TO_ROLLING_VOLUME"
                    else "current and previous entry-time values only"
                ),
            })
    return pd.DataFrame(rows)


def attach_root_cause(
    drift: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    best = (
        candidates.sort_values(
            ["source_feature", "candidate_psi", "representation"]
        )
        .groupby("source_feature", as_index=False).first()
        .rename(columns={
            "representation": "best_candidate_representation",
            "candidate_psi": "best_candidate_psi",
            "expected_absolute_psi_reduction": "best_expected_psi_reduction",
            "expected_relative_psi_reduction":
                "best_expected_relative_psi_reduction",
        })
    )
    result = drift.merge(
        best[[
            "source_feature", "best_candidate_representation",
            "best_candidate_psi", "best_expected_psi_reduction",
            "best_expected_relative_psi_reduction",
        ]],
        left_on="feature", right_on="source_feature", how="left",
    ).drop(columns="source_feature")
    result["price_level_increase_driven"] = (
        result.classification.eq("Absolute price feature")
        & result.original_psi.ge(0.10)
        & result.best_expected_relative_psi_reduction.ge(0.50)
        & pd.to_numeric(result.validation_mean, errors="coerce").gt(
            pd.to_numeric(result.research_mean, errors="coerce")
        )
    )
    result["root_cause"] = np.select(
        [
            result.price_level_increase_driven,
            result.classification.eq("Time feature"),
            result.classification.eq("Option feature"),
            result.original_psi.lt(0.10),
        ],
        [
            "NIFTY_PRICE_LEVEL_INCREASE",
            "TEMPORAL_OR_SAMPLE_AGE_SHIFT",
            "OPTION_PREMIUM_OR_CONTRACT_DISTRIBUTION_SHIFT",
            "STABLE_OR_MINOR_DRIFT",
        ],
        default="DISTRIBUTION_OR_REGIME_SHIFT",
    )
    return result.sort_values("original_psi", ascending=False)
