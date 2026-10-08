"""Step 20A: canonical entry-time prediction dataset construction."""

from __future__ import annotations

import pandas as pd

from src.feature_pipeline import FEATURES as INDEX_FEATURES
from src.option_feature_pipeline import OPTION_FEATURES


INDEX_COLUMNS = [f"index_{feature}" for feature in INDEX_FEATURES]
OPTION_COLUMNS = [f"option_{feature}" for feature in OPTION_FEATURES]
SETUP_COLUMNS = [
    "entry_signal",
    "trend_state",
    "structure_state",
    "momentum_state",
    "volatility_state",
    "session_state",
    "event_state",
    "breakout_quality_state",
    "candle_quality_state",
    "trend_strength_state",
]
DECISION_COLUMNS = [
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
]
PREDICTION_COLUMNS = [
    "timestamp", *INDEX_COLUMNS, *OPTION_COLUMNS, *SETUP_COLUMNS, *DECISION_COLUMNS
]
FORBIDDEN_COLUMNS = {
    "final_outcome",
    "target",
    "target_win",
    "label",
    "exit_price",
    "exit_timestamp",
    "exit_reason",
    "mfe",
    "mae",
    "realized_r",
    "realised_return",
    "realized_terminal_return",
    "realized_expansion_hit",
    "realized_mfe",
    "realized_mae",
    "terminal_outcome",
}


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata"
    )


def build_prediction_dataset(
    aligned: pd.DataFrame,
    setups: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one entry-safe record per matched frozen decision."""
    aligned_required = {"timestamp", *INDEX_COLUMNS, *OPTION_COLUMNS}
    setup_required = {"timestamp", "trade_allowed", *SETUP_COLUMNS}
    decision_required = {"timestamp", *DECISION_COLUMNS}
    for name, frame, required in (
        ("aligned", aligned, aligned_required),
        ("setups", setups, setup_required),
        ("decisions", decisions, decision_required),
    ):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{name} missing required columns: {missing}")
    market = aligned[["timestamp", *INDEX_COLUMNS, *OPTION_COLUMNS]].copy()
    setup = setups[["timestamp", "trade_allowed", *SETUP_COLUMNS]].copy()
    decision = decisions[["timestamp", *DECISION_COLUMNS]].copy()
    for frame in (market, setup, decision):
        frame["timestamp"] = _timestamp(frame)
        if frame.timestamp.duplicated().any():
            raise ValueError("Every source timestamp must be unique")
    setup = setup.loc[setup.trade_allowed.eq(1)].drop(columns="trade_allowed")
    result = decision.merge(
        setup, on="timestamp", how="inner", validate="one_to_one", sort=True
    ).merge(
        market, on="timestamp", how="inner", validate="one_to_one", sort=True
    )
    result = result[PREDICTION_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    if result.empty:
        raise ValueError("No eligible prediction records")
    if result.timestamp.duplicated().any() or not result.timestamp.is_monotonic_increasing:
        raise AssertionError("Prediction records must be unique and chronological")
    forbidden_present = FORBIDDEN_COLUMNS.intersection(result.columns)
    if forbidden_present:
        raise AssertionError(f"Future/target columns entered prediction data: {forbidden_present}")
    return result


def feature_inventory(dataset: pd.DataFrame) -> pd.DataFrame:
    roles = {}
    for column in dataset.columns:
        if column == "timestamp":
            role = "IDENTIFIER"
        elif column in INDEX_COLUMNS:
            role = "INDEX_FEATURE"
        elif column in OPTION_COLUMNS:
            role = "OPTION_FEATURE"
        elif column in SETUP_COLUMNS:
            role = "FROZEN_STAGE1_METADATA"
        elif column in DECISION_COLUMNS:
            role = "CAUSAL_DECISION_OUTPUT"
        else:
            role = "UNCLASSIFIED"
        roles[column] = role
    return pd.DataFrame([
        {
            "column": column,
            "role": roles[column],
            "dtype": str(dataset[column].dtype),
            "non_null_count": int(dataset[column].notna().sum()),
            "null_count": int(dataset[column].isna().sum()),
            "unique_values": int(dataset[column].nunique(dropna=True)),
            "entry_time_available": True,
        }
        for column in dataset.columns
    ])


def assert_prefix_invariance(
    aligned: pd.DataFrame,
    setups: pd.DataFrame,
    decisions: pd.DataFrame,
    prefix_rows: int,
) -> None:
    full = build_prediction_dataset(aligned, setups, decisions)
    prefix_decisions = decisions.iloc[:prefix_rows].copy()
    prefix = build_prediction_dataset(aligned, setups, prefix_decisions)
    expected = full.iloc[:len(prefix)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, expected, check_exact=True)


def build_live_record(
    aligned_row: pd.DataFrame,
    setup_row: pd.DataFrame,
    decision_row: pd.DataFrame,
) -> pd.DataFrame:
    """The live path deliberately calls the exact historical builder."""
    return build_prediction_dataset(aligned_row, setup_row, decision_row)

