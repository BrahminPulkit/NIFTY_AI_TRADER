import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_pipeline import FEATURES
from src.setup_engine import build_setups
from src.stage2_outcome_dataset import CONTRACT, build_diagnostic_decomposition, build_stage2_outcomes


def fixture(rows=100):
    timestamp = pd.date_range("2026-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata")
    close = pd.Series(np.full(rows, 100.0))
    data = pd.DataFrame({
        "timestamp": timestamp, "open": close, "high": close+.05, "low": close-.05,
        "close": close, "volume": 1., "candle_body": 0., "upper_wick": .05,
        "lower_wick": .05, "candle_range": .1, "log_return": 0., "percent_return": 0.,
        "ema_5": 99.8, "ema_9": 99.6, "ema_20": 99.4, "ema_50": 99., "vwap": 100.,
        "rsi_14": 60., "atr_14": .1, "rolling_volatility": .001, "rolling_range": 1.,
        "previous_high": 99.9, "previous_low": 99., "rolling_high": 99.8,
        "rolling_low": 98., "breakout_flag": 0, "breakdown_flag": 0,
        "minutes_from_open": np.arange(rows), "minutes_to_close": 374-np.arange(rows),
        "weekday": 4, "month": 1, "session_id": 20260102,
    })
    setup = build_setups(data)
    setup.loc[:, "entry_signal"] = 0; setup.loc[:, "trade_allowed"] = 0
    setup.loc[20, "entry_signal"] = 1; setup.loc[20, "trade_allowed"] = 1
    setup.loc[20, "rejection_reason"] = "ALLOWED"
    return setup


def test_frozen_contract_and_stop_first_gap_fill():
    data = fixture()
    data.loc[21, ["open", "high", "low", "close"]] = [99.7, 100.4, 99.6, 100.0]
    result = build_stage2_outcomes(data)
    assert CONTRACT.horizon == 5
    assert result.iloc[0].final_outcome == "LOSS"
    assert result.iloc[0].exit_price == 99.7


def test_only_allowed_rows_and_no_no_trade_class():
    data = fixture(); result = build_stage2_outcomes(data)
    assert len(result) == 1 and result.trade_allowed.eq(1).all()
    assert set(result.final_outcome).issubset({"WIN", "LOSS", "TIMEOUT"})
    assert set(FEATURES).issubset(result.columns)


def test_determinism_and_finalized_prefix_invariance():
    data = fixture()
    first = build_stage2_outcomes(data.iloc[:50].copy())
    repeat = build_stage2_outcomes(data.iloc[:50].copy())
    full = build_stage2_outcomes(data)
    assert_frame_equal(first, repeat, check_exact=True)
    assert_frame_equal(first, full.loc[full.timestamp.isin(first.timestamp)].reset_index(drop=True), check_exact=True)


def test_future_after_horizon_cannot_change_label_or_features():
    data = fixture(); baseline = build_stage2_outcomes(data)
    changed = data.copy(); changed.loc[26:, ["open", "high", "low", "close"]] += 50
    altered = build_stage2_outcomes(changed)
    assert_frame_equal(baseline, altered, check_exact=True)


def test_diagnostic_decomposition_is_separate():
    data = fixture(); outcome = build_stage2_outcomes(data)
    diagnostic = build_diagnostic_decomposition(data, outcome)
    assert diagnostic.loc[20, "diagnostic_subtype"] == "LONG_TIMEOUT"
    assert "NO_SETUP" in set(diagnostic.diagnostic_subtype)
