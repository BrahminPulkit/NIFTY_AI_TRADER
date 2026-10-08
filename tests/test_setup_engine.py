import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.setup_engine import OUTPUT_COLUMNS, build_live_setup, build_setups


def fixture(rows=100):
    timestamp = pd.date_range("2026-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata")
    close = pd.Series(np.linspace(100, 120, rows))
    return pd.DataFrame({
        "timestamp": timestamp, "close": close, "ema_5": close-1, "ema_9": close-2,
        "ema_20": close-3, "ema_50": close-4, "rsi_14": 60.0, "atr_14": close*.001,
        "rolling_high": close-.5, "rolling_low": close-5, "breakout_flag": 1,
        "breakdown_flag": 0, "candle_body": 0.6, "candle_range": 1.0,
        "minutes_from_open": np.arange(rows), "session_id": 20260102,
    })


def test_long_setup_and_filter_stages():
    data = fixture(); data.loc[19, "breakout_flag"] = 0
    result = build_setups(data)
    assert result.iloc[20].entry_signal == 1 and result.iloc[20].trade_allowed == 1
    assert result.iloc[5].entry_signal == 1 and result.iloc[5].trade_allowed == 0
    assert "OPENING_BLOCKED" in result.iloc[5].rejection_reason
    assert "REPEATED_EVENT" in result.iloc[5].rejection_reason


def test_short_setup_and_momentum_rejection():
    data = fixture()
    data[["ema_5", "ema_9", "ema_20", "ema_50"]] = pd.DataFrame({
        "ema_5": data.close+1, "ema_9": data.close+2, "ema_20": data.close+3,
        "ema_50": data.close+4})
    data["breakout_flag"] = 0; data["breakdown_flag"] = 1; data["rsi_14"] = 50
    result = build_setups(data)
    assert result.iloc[20].entry_signal == -1
    assert result.iloc[20].trade_allowed == 0
    assert "MOMENTUM_REJECTED" in result.iloc[20].rejection_reason


def test_no_setup_is_auditable():
    data = fixture(); data["breakout_flag"] = 0
    result = build_setups(data)
    assert result.entry_signal.eq(0).all() and result.rejection_reason.eq("NO_SETUP").all()


def test_determinism_prefix_invariance_and_no_future_leakage():
    data = fixture()
    first = build_setups(data.iloc[:40].copy())
    assert_frame_equal(first, build_setups(data.iloc[:40].copy()), check_exact=True)
    assert_frame_equal(first, build_setups(data).iloc[:40], check_exact=True)
    changed = data.copy(); changed.loc[40:, "rsi_14"] = 0
    assert_frame_equal(first, build_setups(changed).iloc[:40], check_exact=True)


def test_historical_live_parity():
    data = fixture()
    historical = build_setups(data)
    assert historical.iloc[-1].equals(build_live_setup(data))
    assert set(OUTPUT_COLUMNS).issubset(historical.columns)
