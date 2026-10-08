"""Causal three-block NIFTY/CE/PE feature preview for Phase 6 research.

This module is deliberately not wired into production inference. It proves the
new feature contract on completed candles before any model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feature_pipeline import build_features
from src.option_contract_pipeline import normalize_option_contracts
from src.option_feature_pipeline import build_option_features


BLOCK_FEATURES = (
    "return_1m", "return_3m", "return_5m",
    "momentum_1m", "momentum_3m", "momentum_5m",
    "vwap_distance", "ema_structure", "rsi", "atr_pct", "volatility",
)
OPTION_EXTRA_FEATURES = ("premium_acceleration", "volume_ratio")
RELATIVE_FEATURES = (
    "ce_return_5m_minus_nifty_return_5m",
    "pe_return_5m_minus_nifty_return_5m",
    "pe_return_5m_plus_nifty_return_5m",
    "ce_return_5m_minus_pe_return_5m",
    "ce_momentum_5m_minus_pe_momentum_5m",
    "ce_volume_ratio_minus_pe_volume_ratio",
    "ce_strength_vs_nifty",
    "pe_strength_vs_nifty",
    "ce_strength_vs_pe",
    "pe_strength_vs_ce",
    "bullish_option_confirmation",
    "bearish_option_confirmation",
)


@dataclass(frozen=True)
class PreviewContract:
    version: str = "nifty_ce_pe_aware_preview_v1"
    timezone: str = "Asia/Kolkata"
    completed_candles_only: bool = True


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.timestamp, utc=True, errors="raise").dt.tz_convert(
        "Asia/Kolkata")


def _base_block(features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = features.copy()
    close = pd.to_numeric(data.close, errors="raise")
    atr = pd.to_numeric(data.atr_14, errors="coerce").replace(0, np.nan)
    output = pd.DataFrame({"timestamp": _timestamp(data)})
    for horizon in (1, 3, 5):
        displacement = close - close.shift(horizon)
        output[f"{prefix}_return_{horizon}m"] = close.pct_change(
            horizon, fill_method=None) * 100.0
        output[f"{prefix}_momentum_{horizon}m"] = displacement / atr
    output[f"{prefix}_vwap_distance"] = (close / data.vwap - 1.0) * 100.0
    output[f"{prefix}_ema_structure"] = (data.ema_9 - data.ema_20) / atr
    output[f"{prefix}_rsi"] = data.rsi_14
    output[f"{prefix}_atr_pct"] = data.atr_14 / close * 100.0
    output[f"{prefix}_volatility"] = data.rolling_volatility
    return output


def _contract_option_features(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    contracts = normalize_option_contracts(frame)
    contracts = contracts.loc[contracts.option_type.eq(side)].copy()
    if contracts.empty:
        raise ValueError(f"No {side} option observations")
    rows = []
    for segment_id, segment in contracts.groupby("contract_segment_id", sort=False):
        segment = segment.sort_values("timestamp", kind="stable")
        features = build_option_features(segment)
        block = _base_block(features, side.lower())
        block[f"{side.lower()}_premium_acceleration"] = (
            block[f"{side.lower()}_return_1m"].diff())
        block[f"{side.lower()}_volume_ratio"] = features.volume_ratio_20.to_numpy()
        block[f"{side.lower()}_contract_segment_id"] = segment_id
        block[f"{side.lower()}_expiry"] = segment.expiry.to_numpy()
        block[f"{side.lower()}_strike"] = segment.strike.to_numpy()
        rows.append(block)
    result = pd.concat(rows, ignore_index=True).sort_values("timestamp", kind="stable")
    if result.timestamp.duplicated().any():
        raise ValueError(f"Multiple {side} contracts exist at one preview timestamp")
    return result.reset_index(drop=True)


def _panel_option_features(selected: pd.DataFrame, panel: pd.DataFrame,
                           side: str) -> pd.DataFrame:
    """Build on full fixed contracts, then select the contract active at each timestamp."""
    selected_contracts = normalize_option_contracts(selected)
    selected_contracts = selected_contracts.loc[selected_contracts.option_type.eq(side)].copy()
    panel_contracts = normalize_option_contracts(panel)
    panel_contracts = panel_contracts.loc[panel_contracts.option_type.eq(side)].copy()
    if selected_contracts.empty or panel_contracts.empty:
        raise ValueError(f"No {side} selected/full contract observations")
    rows = []
    prefix = side.lower()
    for segment_id, segment in panel_contracts.groupby("contract_segment_id", sort=False):
        segment = segment.sort_values("timestamp", kind="stable")
        features = build_option_features(segment)
        block = _base_block(features, prefix)
        block[f"{prefix}_premium_acceleration"] = block[f"{prefix}_return_1m"].diff()
        block[f"{prefix}_volume_ratio"] = features.volume_ratio_20.to_numpy()
        block["expiry"] = segment.expiry.to_numpy()
        block["strike"] = segment.strike.to_numpy()
        block["panel_contract_segment_id"] = segment_id
        rows.append(block)
    available = pd.concat(rows, ignore_index=True)
    keys = selected_contracts[["timestamp", "expiry", "strike", "contract_segment_id"]].copy()
    keys = keys.rename(columns={"contract_segment_id": f"{prefix}_contract_segment_id"})
    result = keys.merge(available, on=["timestamp", "expiry", "strike"], how="left",
                        validate="one_to_one")
    if result.timestamp.duplicated().any():
        raise ValueError(f"Multiple selected {side} contracts exist at one preview timestamp")
    result = result.rename(columns={"expiry": f"{prefix}_expiry", "strike": f"{prefix}_strike"})
    return result.drop(columns="panel_contract_segment_id").sort_values(
        "timestamp", kind="stable").reset_index(drop=True)


def _add_relationship_features(output: pd.DataFrame) -> pd.DataFrame:
    output = output.copy()
    output["ce_return_5m_minus_nifty_return_5m"] = (
        output.ce_return_5m - output.nifty_return_5m)
    output["pe_return_5m_minus_nifty_return_5m"] = (
        output.pe_return_5m - output.nifty_return_5m)
    output["pe_return_5m_plus_nifty_return_5m"] = (
        output.pe_return_5m + output.nifty_return_5m)
    output["ce_return_5m_minus_pe_return_5m"] = output.ce_return_5m - output.pe_return_5m
    output["ce_momentum_5m_minus_pe_momentum_5m"] = (
        output.ce_momentum_5m - output.pe_momentum_5m)
    output["ce_volume_ratio_minus_pe_volume_ratio"] = (
        output.ce_volume_ratio - output.pe_volume_ratio)
    output["ce_strength_vs_nifty"] = output.ce_return_5m - output.nifty_return_5m
    output["pe_strength_vs_nifty"] = output.pe_return_5m + output.nifty_return_5m
    output["ce_strength_vs_pe"] = output.ce_return_5m - output.pe_return_5m
    output["pe_strength_vs_ce"] = output.pe_return_5m - output.ce_return_5m
    output["bullish_option_confirmation"] = (
        output.nifty_return_5m.gt(0) & output.ce_return_5m.gt(0)
        & output.pe_return_5m.lt(0)).astype("int8")
    output["bearish_option_confirmation"] = (
        output.nifty_return_5m.lt(0) & output.pe_return_5m.gt(0)
        & output.ce_return_5m.lt(0)).astype("int8")
    output["expected_interpretation"] = np.select(
        [output.bullish_option_confirmation.eq(1),
         output.bearish_option_confirmation.eq(1),
         output.ce_return_5m.gt(0) & output.pe_return_5m.gt(0)],
        ["CE_CONFIRMED", "PE_CONFIRMED", "CONFLICT_BOTH_STRONG"],
        default="NO_OPTION_CONFIRMATION",
    )
    return output


def build_ce_pe_feature_preview(index: pd.DataFrame, ce: pd.DataFrame,
                                pe: pd.DataFrame) -> pd.DataFrame:
    """Return one causal row containing NIFTY, CE, PE and relationship blocks."""
    nifty = _base_block(build_features(index), "nifty")
    ce_block = _contract_option_features(ce, "CE")
    pe_block = _contract_option_features(pe, "PE")
    output = nifty.merge(ce_block, on="timestamp", how="inner", validate="one_to_one")
    output = output.merge(pe_block, on="timestamp", how="inner", validate="one_to_one")
    if output.empty:
        raise ValueError("No exactly aligned NIFTY/CE/PE timestamps")

    return _add_relationship_features(output).sort_values(
        "timestamp", kind="stable").reset_index(drop=True)


def build_ce_pe_feature_preview_from_panels(
        index: pd.DataFrame, ce: pd.DataFrame, pe: pd.DataFrame,
        ce_panel: pd.DataFrame, pe_panel: pd.DataFrame) -> pd.DataFrame:
    """Return the same 49 features using complete fixed-contract histories."""
    nifty = _base_block(build_features(index), "nifty")
    ce_block = _panel_option_features(ce, ce_panel, "CE")
    pe_block = _panel_option_features(pe, pe_panel, "PE")
    output = nifty.merge(ce_block, on="timestamp", how="inner", validate="one_to_one")
    output = output.merge(pe_block, on="timestamp", how="inner", validate="one_to_one")
    if output.empty:
        raise ValueError("No exactly aligned NIFTY/CE/PE panel timestamps")
    return _add_relationship_features(output).sort_values(
        "timestamp", kind="stable").reset_index(drop=True)


def feature_columns() -> list[str]:
    columns = []
    for prefix in ("nifty", "ce", "pe"):
        columns.extend(f"{prefix}_{name}" for name in BLOCK_FEATURES)
        if prefix in {"ce", "pe"}:
            columns.extend(f"{prefix}_{name}" for name in OPTION_EXTRA_FEATURES)
    return columns + list(RELATIVE_FEATURES)
