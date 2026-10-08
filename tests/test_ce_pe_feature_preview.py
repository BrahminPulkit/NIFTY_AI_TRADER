import pandas as pd
import pytest

from src.ce_pe_feature_preview import (
    build_ce_pe_feature_preview, build_ce_pe_feature_preview_from_panels, feature_columns,
)


def market(side=None, strike=24500, expiry="2026-08-11", periods=60):
    timestamp = pd.date_range("2026-08-07 09:15", periods=periods, freq="min",
                              tz="Asia/Kolkata")
    base = pd.Series(range(periods), dtype=float)
    frame = pd.DataFrame({
        "timestamp": timestamp, "open": 100 + base,
        "high": 101 + base, "low": 99 + base, "close": 100.5 + base,
        "volume": 1000 + base * 10,
    })
    if side:
        frame = frame.assign(underlying="NIFTY", expiry=expiry, strike=strike,
                             option_type=side, security_id=f"{side}-{strike}")
    return frame


def test_preview_contains_both_sides_and_relationships():
    result = build_ce_pe_feature_preview(market(), market("CE"), market("PE"))
    assert set(feature_columns()).issubset(result.columns)
    assert result[["ce_contract_segment_id", "pe_contract_segment_id"]].notna().all().all()
    assert result.timestamp.is_unique


def test_prefix_invariance_proves_causality():
    index, ce, pe = market(), market("CE"), market("PE")
    full = build_ce_pe_feature_preview(index, ce, pe)
    prefix = build_ce_pe_feature_preview(index.iloc[:45], ce.iloc[:45], pe.iloc[:45])
    pd.testing.assert_frame_equal(
        full.iloc[:45].reset_index(drop=True), prefix.reset_index(drop=True))


def test_timestamp_mismatch_does_not_forward_fill():
    pe = market("PE").iloc[1:].reset_index(drop=True)
    result = build_ce_pe_feature_preview(market(), market("CE"), pe)
    assert result.timestamp.min() == pe.timestamp.min()
    assert len(result) == len(pe)


def test_multiple_contracts_at_same_timestamp_are_rejected():
    ce = pd.concat([market("CE"), market("CE", strike=24550)], ignore_index=True)
    with pytest.raises(ValueError, match="Multiple CE contracts"):
        build_ce_pe_feature_preview(market(), ce, market("PE"))


def test_contract_change_resets_option_rolling_features():
    first = market("CE", periods=30)
    second = market("CE", strike=24550, periods=30).copy()
    second["timestamp"] += pd.Timedelta(minutes=30)
    ce = pd.concat([first, second], ignore_index=True)
    index = market(periods=60)
    pe = market("PE", periods=60)
    result = build_ce_pe_feature_preview(index, ce, pe)
    boundary = second.timestamp.iloc[0]
    row = result.loc[result.timestamp.eq(boundary)].iloc[0]
    assert pd.isna(row.ce_return_1m)
    assert pd.isna(row.ce_momentum_5m)
    assert pd.isna(row.ce_volume_ratio)


def test_panel_features_use_fixed_contract_history_without_cross_strike_state():
    index, pe = market(), market("PE")
    ce_a, ce_b = market("CE"), market("CE", strike=24550)
    selected = ce_a.copy()
    selected.loc[20:39, selected.columns] = ce_b.loc[20:39].to_numpy()
    selected["contract_segment_id"] = [
        "a-first" if i < 20 else "b" if i < 40 else "a-revisited" for i in range(60)]
    result = build_ce_pe_feature_preview_from_panels(
        index, selected, pe, pd.concat([ce_a, ce_b], ignore_index=True), pe)
    switch = result.loc[result.timestamp.eq(index.timestamp.iloc[20])].iloc[0]
    revisit = result.loc[result.timestamp.eq(index.timestamp.iloc[40])].iloc[0]
    assert switch.ce_contract_segment_id == "b"
    assert revisit.ce_contract_segment_id == "a-revisited"
    assert pd.notna(switch.ce_return_1m)
    assert pd.notna(revisit.ce_return_1m)


def test_ce_and_pe_panel_selection_boundaries_are_independent():
    index = market()
    ce_a, ce_b = market("CE"), market("CE", strike=24550)
    pe_a, pe_b = market("PE"), market("PE", strike=24450)
    ce = ce_a.copy()
    ce.loc[30:, ce.columns] = ce_b.loc[30:].to_numpy()
    pe = pe_a.copy()
    pe.loc[40:, pe.columns] = pe_b.loc[40:].to_numpy()
    result = build_ce_pe_feature_preview_from_panels(
        index, ce, pe, pd.concat([ce_a, ce_b]), pd.concat([pe_a, pe_b]))
    assert result.loc[30, "ce_strike"] == 24550
    assert result.loc[30, "pe_strike"] == 24500
    assert result.loc[40, "pe_strike"] == 24450
