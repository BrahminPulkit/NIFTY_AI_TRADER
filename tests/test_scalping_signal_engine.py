import numpy as np
import pandas as pd

import src.scalping_signal_engine as module
from src.scalping_signal_engine import ScalpingSignalEngine, SignalConfig


class Model:
    def __init__(self, probability=.8): self.value = probability
    def probability(self, _values): return self.value


def index():
    ts = pd.date_range("2026-08-07 10:00", periods=80, freq="min", tz="Asia/Kolkata")
    close = pd.Series(np.linspace(24500, 24600, len(ts)))
    return pd.DataFrame({"timestamp": ts, "open": close-.5, "high": close+2,
                         "low": close-2, "close": close, "volume": np.arange(len(ts))+100})


def options(timestamp):
    return pd.DataFrame({
        "timestamp": [timestamp, timestamp], "strike": [24600, 24600],
        "expiry": ["2026-08-11"] * 2, "option_type": ["CE", "PE"],
        "security_id": ["11", "12"], "underlying": ["NIFTY"] * 2,
        "open": [100, 100], "high": [102, 102], "low": [98, 98],
        "close": [100, 100], "volume": [1000, 1000],
    })


def candidate(timestamp, direction):
    return pd.DataFrame([{"timestamp": timestamp, "strategy": "BREAKOUT",
        "direction": direction, "option_type": "CE" if direction == 1 else "PE",
        "candidate_created": True, "base_strategy_score": 60,
        "score_components": {}, "index_price": 24600, "atr": 10}])


def run(monkeypatch, direction):
    idx = index(); timestamp = idx.timestamp.iloc[-1]
    monkeypatch.setattr(module, "generate_scalping_candidates", lambda *_args, **_kwargs: candidate(timestamp, direction))
    # Avoid warm-up requirements in this routing unit test.
    monkeypatch.setattr(module, "build_option_features", lambda frame: pd.DataFrame([
        {"close": 100, **{name: 1 for name in module.OPTION_FEATURES}}]))
    engine = ScalpingSignalEngine({"CE": Model(), "PE": Model()},
        config=SignalConfig(minimum_strategy_score=50, minimum_model_probability=.5))
    return engine.evaluate(idx, options(timestamp), now=timestamp)[0]


def test_bullish_routes_only_to_ce(monkeypatch):
    result = run(monkeypatch, 1)
    assert result["final_decision"] == "BUY CE"
    assert result["selected_option"]["option_type"] == "CE"


def test_bearish_routes_only_to_pe(monkeypatch):
    result = run(monkeypatch, -1)
    assert result["final_decision"] == "BUY PE"
    assert result["selected_option"]["option_type"] == "PE"


def test_neutral_produces_no_trade(monkeypatch):
    idx = index()
    monkeypatch.setattr(module, "generate_scalping_candidates", lambda *_args, **_kwargs: pd.DataFrame())
    result = ScalpingSignalEngine({}).evaluate(idx, options(idx.timestamp.iloc[-1]), now=idx.timestamp.iloc[-1])[0]
    assert result["final_decision"] == "NO TRADE"
    assert result["direction"] == "NEUTRAL"


def test_stale_data_is_hard_gate(monkeypatch):
    idx = index(); timestamp = idx.timestamp.iloc[-1]
    monkeypatch.setattr(module, "generate_scalping_candidates", lambda *_args, **_kwargs: candidate(timestamp, 1))
    result = ScalpingSignalEngine({"CE": Model()}).evaluate(
        idx, options(timestamp), now=timestamp + pd.Timedelta(minutes=3))[0]
    assert result["primary_rejection_reason"] == "STALE_DATA"
    assert result["model_probability"] is None
