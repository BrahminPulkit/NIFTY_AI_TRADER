"""Step 20C-R: exact timestamp join for probability-threshold research."""

from __future__ import annotations

import pandas as pd


OOF_REQUIRED = {
    "timestamp", "fold_id", "model_name", "predicted_probability",
    "predicted_class", "true_label",
}
DECISION_COLUMNS = [
    "timestamp",
    "market_state",
    "trend_regime",
    "volatility_regime",
    "session_phase",
    "structure_context",
    "current_strategy",
    "activated_strategy",
    "estimate_source",
    "history_count",
    "premium_expansion_probability",
    "expected_premium_expansion",
    "expected_drawdown",
    "expected_holding_minutes",
    "expected_terminal_return",
    "terminal_return_standard_error",
    "terminal_return_95_lower",
    "expansion_drawdown_ratio",
    "strategy_confidence_score",
    "recommendation",
    "decision_reason",
    "realized_terminal_return",
    "realized_mfe",
    "realized_mae",
]


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata"
    )


def _validate_oof(frame: pd.DataFrame, expected_model: str) -> pd.DataFrame:
    if missing := sorted(OOF_REQUIRED.difference(frame.columns)):
        raise ValueError(f"{expected_model} OOF missing: {missing}")
    data = frame[list(OOF_REQUIRED)].copy()
    data["timestamp"] = _timestamp(data)
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError(f"{expected_model} OOF timestamps must be unique and chronological")
    if set(data.model_name.unique()) != {expected_model}:
        raise ValueError(f"Unexpected model identity in {expected_model} OOF")
    if not data.predicted_probability.between(0, 1).all():
        raise ValueError("OOF probabilities must be between zero and one")
    return data


def build_joined_probability_dataset(
    catboost: pd.DataFrame,
    xgboost: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per OOF timestamp; no target or response is recomputed."""
    cat = _validate_oof(catboost, "CATBOOST").drop(columns="model_name").rename(columns={
        "predicted_probability": "catboost_predicted_probability",
        "predicted_class": "catboost_predicted_class",
        "true_label": "catboost_true_label",
        "fold_id": "catboost_fold_id",
    })
    xgb = _validate_oof(xgboost, "XGBOOST").drop(columns="model_name").rename(columns={
        "predicted_probability": "xgboost_predicted_probability",
        "predicted_class": "xgboost_predicted_class",
        "true_label": "xgboost_true_label",
        "fold_id": "xgboost_fold_id",
    })
    joined = cat.merge(xgb, on="timestamp", how="inner", validate="one_to_one", sort=True)
    if len(joined) != len(cat) or len(joined) != len(xgb):
        raise AssertionError("CatBoost and XGBoost OOF timestamps do not match exactly")
    if not joined.catboost_fold_id.eq(joined.xgboost_fold_id).all():
        raise AssertionError("OOF fold IDs differ between models")
    if not joined.catboost_true_label.eq(joined.xgboost_true_label).all():
        raise AssertionError("Frozen true labels differ between models")
    joined["fold_id"] = joined.pop("catboost_fold_id")
    joined = joined.drop(columns="xgboost_fold_id")
    joined["true_label"] = joined.pop("catboost_true_label")
    joined = joined.drop(columns="xgboost_true_label")

    if missing := sorted(set(DECISION_COLUMNS).difference(decisions.columns)):
        raise ValueError(f"Frozen decision dataset missing: {missing}")
    decision = decisions[DECISION_COLUMNS].copy()
    decision["timestamp"] = _timestamp(decision)
    if decision.timestamp.duplicated().any():
        raise ValueError("Frozen decision timestamps must be unique")
    result = joined.merge(
        decision, on="timestamp", how="left", validate="one_to_one", indicator=True
    )
    if not result["_merge"].eq("both").all():
        raise AssertionError("One or more OOF timestamps lack a frozen decision row")
    result = result.drop(columns="_merge")
    result["terminal_premium_return"] = result.pop("realized_terminal_return")
    result["premium_expansion"] = result.pop("realized_mfe")
    result["premium_drawdown"] = result.pop("realized_mae")
    # The only frozen holding estimate available at row level is the causal
    # Decision Engine expectation. It is not a realized future holding label.
    result["holding_time_minutes"] = result["expected_holding_minutes"]
    result["strategy_family"] = result["current_strategy"]
    result["market_regime"] = (
        result["trend_regime"].astype(str) + "|" + result["volatility_regime"].astype(str)
    )
    result = result.sort_values("timestamp").reset_index(drop=True)
    if result.timestamp.duplicated().any() or not result.timestamp.is_monotonic_increasing:
        raise AssertionError("Joined research dataset must have one chronological row per timestamp")
    return result


def assert_join_prefix_parity(
    catboost: pd.DataFrame,
    xgboost: pd.DataFrame,
    decisions: pd.DataFrame,
    prefix_rows: int,
) -> None:
    full = build_joined_probability_dataset(catboost, xgboost, decisions)
    prefix = build_joined_probability_dataset(
        catboost.iloc[:prefix_rows], xgboost.iloc[:prefix_rows], decisions
    )
    pd.testing.assert_frame_equal(
        prefix, full.iloc[:len(prefix)].reset_index(drop=True), check_exact=True
    )

