import time
import tempfile
from pathlib import Path

import pandas as pd

from src.dhan_connection_manager import DhanConnectionManager, get_dhan_manager
from src.dhan_live_broker import ConnectionHealth, Instrument
from src.dhan_option_data_collector import OptionMarketDataCollector
from src.live_inference_engine import LiveInferenceResult


class FakeClient:
    def __init__(self):
        self.health = ConnectionHealth(
            "CONNECTED", "now", "now", 10.0, True, "ok")
        self.fail = False
        self.fail_quotes = False

    def test_connection(self):
        if self.fail:
            self.health = ConnectionHealth(
                "NO INTERNET", "now", "later", 12.0, True, "temporary")
        else:
            self.health = ConnectionHealth(
                "CONNECTED", "now", "later", 11.0, True, "ok")
        return self.health

    def quote(self, instruments):
        if self.fail_quotes:
            raise RuntimeError("temporary quote failure")
        payload = {}
        for item in instruments:
            payload.setdefault(item.exchange_segment, {})[
                item.security_id] = {"last_price": 100.0}
        return payload

    def completed_candles(self, instrument, _from, _to, *, now=None):
        del instrument, now
        timestamps = pd.date_range(
            "2026-07-28 09:15", periods=3, freq="min",
            tz="Asia/Kolkata")
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": [100, 101, 102], "high": [101, 102, 103],
            "low": [99, 100, 101], "close": [100, 101, 102],
            "volume": [10, 11, 12],
        })


class FakeSession:
    creations = 0

    def __init__(self, credentials, security_master):
        del credentials, security_master
        type(self).creations += 1
        self.client = FakeClient()
        self.indices = {
            "NIFTY": Instrument("NIFTY", "13", "IDX_I", "INDEX", "NIFTY")}
        self.options = {
            "NIFTY_ATM_CE": Instrument(
                "NIFTY_ATM_CE", "14", "NSE_FNO", "OPTIDX",
                "NIFTY CE", "2026-07-30", 25000, "CE"),
            "NIFTY_ATM_PE": Instrument(
                "NIFTY_ATM_PE", "15", "NSE_FNO", "OPTIDX",
                "NIFTY PE", "2026-07-30", 25000, "PE"),
        }
        self.last_quotes = {"NIFTY": 25000.0}
        self.resolver = self

    def connect(self):
        return self.client.health.public_dict()

    def refresh_instruments(self):
        return self.options

    def resolve_option_chain(self, _underlying, _spot, *, strikes_each_side):
        del strikes_each_side
        return list(self.options.values())

    @staticmethod
    def _ltp(payload, instrument):
        return float(
            payload[instrument.exchange_segment][instrument.security_id][
                "last_price"])


def wait_until(predicate, timeout=2):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition not reached")


def manager():
    FakeSession.creations = 0
    return DhanConnectionManager(
        "unused.csv", heartbeat_seconds=5, session_factory=FakeSession,
        option_collector=OptionMarketDataCollector(
            Path(tempfile.gettempdir()) / "nifty-ai-trader-tests"))


def test_login_once_creates_one_session_and_one_worker():
    service = manager()
    try:
        first = service.connect("1001", "token")
        second = service.connect("1001", "token")
        wait_until(lambda: len(service.snapshot()["candles"]) == 3)
        assert first["connected"] and second["connected"]
        assert FakeSession.creations == 1
        assert service.snapshot()["worker_alive"]
        assert service.snapshot()["worker_name"] == "dhan-connection-worker"
    finally:
        service.disconnect()


def test_cache_snapshot_is_defensive_and_contains_no_client_or_credentials():
    service = manager()
    try:
        service.connect("1001", "secret-token")
        wait_until(lambda: bool(service.snapshot()["candles"]))
        snapshot = service.snapshot()
        snapshot["quotes"]["NIFTY"] = -1
        snapshot["candles"]["NIFTY"].loc[:, "close"] = -1
        fresh = service.snapshot()
        assert fresh["quotes"]["NIFTY"] == 25000.0
        assert fresh["candles"]["NIFTY"].close.min() > 0
        assert "client" not in fresh
        assert "credentials" not in fresh
        assert "secret-token" not in str(fresh)
    finally:
        service.disconnect()


def test_transient_failure_retains_last_known_good_cache_and_session():
    service = manager()
    try:
        service.connect("1001", "token")
        wait_until(lambda: bool(service.snapshot()["candles"]))
        original = service.snapshot()["quotes"]
        service._session.client.fail = True
        service.request_refresh()
        wait_until(
            lambda: service.snapshot()["manager_status"] == "RECONNECTING")
        state = service.snapshot()
        assert state["connected"]
        assert state["quotes"] == original
        assert state["consecutive_failures"] >= 1
        assert FakeSession.creations == 1
    finally:
        service.disconnect()


def test_disconnect_stops_worker_and_clears_cache():
    service = manager()
    service.connect("1001", "token")
    state = service.disconnect()
    assert not state["connected"]
    assert not state["worker_alive"]
    assert state["quotes"] == {}
    assert state["candles"] == {}


def test_pages_have_no_direct_dhan_session_or_client_access():
    from pathlib import Path

    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "apps/research_ui/app.py",
            "apps/research_ui/components/theme.py",
            "apps/research_ui/pages/3_Live_Market.py",
            "apps/research_ui/pages/4_Options_Analytics.py",
            "apps/research_ui/pages/14_Broker_Connection.py",
        ))
    for forbidden in (
        "DhanLiveSession", "_quantedge_dhan_session",
        ".client.test_connection", ".client.quote",
        ".client.completed_candles", ".infer_latest(",
    ):
        assert forbidden not in source


def test_process_accessor_returns_exactly_one_manager():
    assert get_dhan_manager() is get_dhan_manager()


def test_optional_quote_failure_does_not_block_candle_cache():
    service = manager()
    try:
        service.connect("1001", "token")
        service._session.client.fail_quotes = True
        service.request_refresh()
        wait_until(lambda: service.snapshot()["manager_status"] == "DEGRADED")
        state = service.snapshot()
        assert state["connected"]
        assert state["candles"]
        assert "quote" in state["last_error"]
    finally:
        service.disconnect()


def test_shared_worker_publishes_explicit_call_signal_once_per_candle():
    service = manager()
    timestamps = pd.date_range(
        "2026-07-28 09:15", periods=3, freq="min", tz="Asia/Kolkata")
    candles = pd.DataFrame({
        "timestamp": timestamps, "open": [1, 1, 1], "high": [1, 1, 1],
        "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1],
    })

    class FakeEngine:
        calls = 0

        def infer_histories(self, index, option, observations):
            del index, option, observations
            self.calls += 1
            return LiveInferenceResult(
                str(timestamps[-1]), "NIFTY", "ATM_CALL", .94, 1, .94,
                "TRADE", "TRADE", "APPROVED", "ALL_FROZEN_GATES_PASSED",
                "MOMENTUM_BREAKOUT", "TREND_UP", 12.0, "v1", "now")

    engine = FakeEngine()
    service._cache["candles"] = {
        "NIFTY": candles, "NIFTY_ATM_CE": candles.copy()}
    service._inference_dependencies = lambda: (engine, pd.DataFrame())
    service._update_live_signal()
    service._update_live_signal()
    state = service.snapshot()
    assert state["signal"]["action"] == "BUY CE"
    assert state["signal"]["strategy"] == "MOMENTUM_BREAKOUT"
    assert state["signal"]["instrument_scope"] == "ATM_CALL_ONLY"
    assert engine.calls == 1
