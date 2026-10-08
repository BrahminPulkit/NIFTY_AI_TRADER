import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.stage1_v3_research import build_candidate_setups


def fixture(rows=180):
    timestamp = pd.date_range("2026-01-02 09:15", periods=rows, freq="min", tz="Asia/Kolkata")
    close = pd.Series(np.linspace(100, 120, rows))
    high, low = close + .3, close - .3
    return pd.DataFrame({
        "timestamp": timestamp, "open": close-.1, "high": high, "low": low, "close": close,
        "candle_body": .1, "candle_range": .6, "ema_5": close-.1, "ema_9": close-.2,
        "ema_20": close-.3, "ema_50": close-.5, "rsi_14": 60., "atr_14": .2,
        "rolling_volatility": .001, "rolling_high": close-.2, "rolling_low": close-2,
        "breakout_flag": 1, "breakdown_flag": 0, "previous_high": high.shift(1),
        "previous_low": low.shift(1), "minutes_from_open": np.arange(rows),
        "session_id": 20260102,
    })


def decisions(frame):
    return {key: value[1][["entry_signal", "trade_allowed"]]
            for key, value in build_candidate_setups(frame).items()}


def test_v3_is_deterministic_and_prefix_invariant():
    data = fixture()
    short = decisions(data.iloc[:100].copy())
    assert short.keys() == decisions(data.iloc[:100].copy()).keys()
    long = decisions(data)
    for key in short:
        assert_frame_equal(short[key], long[key].iloc[:100], check_exact=True)


def test_future_mutation_does_not_change_prior_decisions():
    data = fixture()
    baseline = decisions(data)
    changed = data.copy()
    changed.loc[120:, ["close", "high", "low"]] += 50
    altered = decisions(changed)
    for key in baseline:
        assert_frame_equal(baseline[key].iloc[:120], altered[key].iloc[:120], check_exact=True)


def test_candidate_outputs_are_directional_and_binary():
    for _, candidate in build_candidate_setups(fixture()).values():
        assert set(candidate.entry_signal.unique()).issubset({-1, 0, 1})
        assert set(candidate.trade_allowed.unique()).issubset({0, 1})
        assert candidate.trade_allowed.eq(candidate.entry_signal.ne(0).astype("int8")).all()
