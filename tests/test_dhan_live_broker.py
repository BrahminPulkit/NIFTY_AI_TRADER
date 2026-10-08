import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from streamlit.testing.v1 import AppTest

from apps.api.routers import broker as broker_router
from src.dhan_live_broker import (
    BrokerCredentials, DhanInstrumentResolver, DhanReadOnlyClient, Instrument,
)
from src.live_broker_reporting import write_broker_reports


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "apps/research_ui"


class Response:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class Requester:
    def __init__(self, get_response=None, post_response=None):
        self.get_response = get_response or Response()
        self.post_response = post_response or Response()
        self.last_headers = None
        self.last_json = None

    def get(self, _url, **kwargs):
        self.last_headers = kwargs["headers"]
        return self.get_response

    def post(self, _url, **kwargs):
        self.last_headers = kwargs["headers"]
        self.last_json = kwargs["json"]
        return self.post_response


def test_credentials_are_masked_and_profile_connection_is_classified():
    credentials = BrokerCredentials("1001", "very-secret-token-ABCD")
    requester = Requester(get_response=Response(body={
        "dhanClientId": "1001", "tokenValidity": "27/07/2026 23:59"}))
    client = DhanReadOnlyClient(credentials, requester=requester, sleeper=lambda _: None)
    health = client.test_connection()
    assert health.status == "CONNECTED"
    assert credentials.masked_token == "********ABCD"
    assert "very-secret" not in json.dumps(health.public_dict())


def test_expired_and_invalid_tokens_are_distinguished():
    credentials = BrokerCredentials("1001", "token")
    expired = Requester(get_response=Response(401, {
        "errorCode": "807", "errorMessage": "Access token is expired"}))
    invalid = Requester(get_response=Response(401, {
        "errorCode": "808", "errorMessage": "Authentication failed"}))
    assert DhanReadOnlyClient(credentials, requester=expired).test_connection().status == "TOKEN EXPIRED"
    assert DhanReadOnlyClient(credentials, requester=invalid).test_connection().status == "INVALID TOKEN"


def test_rolling_option_transport_preserves_zero_expiry_code():
    credentials = BrokerCredentials("1001", "token")
    requester = Requester(post_response=Response(body={"data": {"ce": None}}))
    client = DhanReadOnlyClient(credentials, requester=requester)
    client.health = client.health.__class__(
        "CONNECTED", "now", "now", 1.0, True, "ok")
    payload = {
        "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": 13,
        "instrument": "OPTIDX", "expiryFlag": "WEEK", "expiryCode": 0,
        "strike": "ATM", "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close"],
        "fromDate": "2026-08-07", "toDate": "2026-08-08",
    }

    client.rolling_options(payload)

    assert "expiryCode" in requester.last_json
    assert requester.last_json["expiryCode"] == 0
    assert type(requester.last_json["expiryCode"]) is int


def test_api_connect_rejects_failed_authentication_without_echoing_credentials(monkeypatch):
    secret = "never-echo-this-token"

    class Manager:
        def connect(self, client_id, access_token):
            assert client_id == "1001"
            assert access_token == secret
            return {
                "connected": False,
                "manager_status": "INVALID TOKEN",
                "health": {
                    "status": "INVALID TOKEN",
                    "last_connection_time": None,
                    "last_heartbeat": "now",
                    "latency_ms": 10.0,
                    "token_valid": False,
                    "detail": "HTTP 401; code=808; Details not found",
                },
                "market_status": "CLOSED",
                "last_data_update": None,
                "last_error": "HTTP 401; code=808; Details not found",
                "worker_alive": False,
            }

    monkeypatch.setattr(broker_router, "get_dhan_manager", lambda: Manager())
    with pytest.raises(HTTPException) as caught:
        broker_router.connect(broker_router.BrokerLogin(
            client_id="1001", access_token=secret))
    assert caught.value.status_code == 401
    assert caught.value.detail == "HTTP 401; code=808; Details not found"
    assert secret not in caught.value.detail


def test_master_resolves_indices_and_dynamic_atm_contracts():
    resolver = DhanInstrumentResolver(ROOT / "data/raw/dhan_security_master_latest.csv")
    indices = resolver.resolve_indices()
    assert set(indices) == {"NIFTY", "BANKNIFTY", "SENSEX", "INDIA_VIX"}
    assert all(item.security_id for item in indices.values())
    ce, pe = resolver.resolve_atm_options(
        "NIFTY", 25_000, as_of="2026-07-27 10:00+05:30")
    assert ce.expiry == pe.expiry
    assert ce.strike == pe.strike
    assert (ce.option_type, pe.option_type) == ("CE", "PE")
    chain = resolver.resolve_option_chain(
        "NIFTY", 25_000, strikes_each_side=10,
        as_of="2026-07-27 10:00+05:30")
    assert chain
    assert len({item.strike for item in chain}) <= 21
    assert {item.option_type for item in chain} == {"CE", "PE"}


def test_partial_candle_is_excluded():
    timestamps = pd.date_range(
        "2026-07-27 09:58", periods=3, freq="min", tz="Asia/Kolkata")
    epoch = [int(item.timestamp()) for item in timestamps]
    response = Response(body={
        "timestamp": epoch, "open": [100, 101, 102], "high": [101, 102, 103],
        "low": [99, 100, 101], "close": [100, 101, 102], "volume": [10, 11, 12],
    })
    requester = Requester(
        get_response=Response(body={"dhanClientId": "1001"}),
        post_response=response)
    client = DhanReadOnlyClient(
        BrokerCredentials("1001", "token"), requester=requester)
    client.test_connection()
    instrument = Instrument("NIFTY", "13", "IDX_I", "INDEX", "Nifty 50")
    result = client.completed_candles(
        instrument, "2026-07-27 09:58:00", "2026-07-27 10:01:00",
        now="2026-07-27 10:00+05:30")
    assert result.timestamp.max().strftime("%H:%M") == "09:59"
    assert len(result) == 2


def test_post_close_synthetic_candles_are_excluded():
    timestamps = pd.DatetimeIndex([
        pd.Timestamp("2026-07-27 15:29", tz="Asia/Kolkata"),
        pd.Timestamp("2026-07-27 15:30", tz="Asia/Kolkata"),
        pd.Timestamp("2026-07-27 15:31", tz="Asia/Kolkata"),
    ])
    epoch = [int(item.timestamp()) for item in timestamps]
    response = Response(body={
        "timestamp": epoch, "open": [100, 100, 100],
        "high": [101, 100, 100], "low": [99, 100, 100],
        "close": [100, 100, 100], "volume": [10, 999, 999],
    })
    client = DhanReadOnlyClient(
        BrokerCredentials("1001", "token"),
        requester=Requester(
            get_response=Response(body={"dhanClientId": "1001"}),
            post_response=response),
        sleeper=lambda _: None)
    client.test_connection()
    result = client.completed_candles(
        Instrument("NIFTY", "13", "IDX_I", "INDEX", "Nifty 50"),
        "2026-07-27 15:29:00", "2026-07-27 15:32:00",
        now="2026-07-27 15:33+05:30")

    assert result.timestamp.dt.strftime("%H:%M").tolist() == ["15:29"]
    assert result.volume.tolist() == [10]


def test_quote_rate_limit_retries_without_invalidating_authenticated_session():
    class SequencedRequester(Requester):
        def __init__(self):
            super().__init__(get_response=Response(body={"dhanClientId": "1001"}))
            self.responses = [
                Response(429, {"errorCode": "DH-904", "errorMessage": "Too many requests"}),
                Response(body={"data": {"IDX_I": {"13": {"last_price": 25000}}}}),
            ]

        def post(self, _url, **kwargs):
            self.last_headers = kwargs["headers"]
            self.last_json = kwargs["json"]
            return self.responses.pop(0)

    waits = []
    client = DhanReadOnlyClient(
        BrokerCredentials("1001", "token"), requester=SequencedRequester(),
        sleeper=waits.append)
    client.test_connection()
    instrument = Instrument("NIFTY", "13", "IDX_I", "INDEX", "Nifty 50")
    result = client.quote([instrument])

    assert result["IDX_I"]["13"]["last_price"] == 25000
    assert client.health.status == "CONNECTED"
    assert client.health.token_valid
    assert any(wait >= 1 for wait in waits)


def test_exhausted_quote_429_preserves_token_health():
    requester = Requester(
        get_response=Response(body={"dhanClientId": "1001"}),
        post_response=Response(429, {
            "errorCode": "DH-904", "errorMessage": "Too many requests"}))
    client = DhanReadOnlyClient(
        BrokerCredentials("1001", "token"), requester=requester,
        retries=2, sleeper=lambda _: None)
    client.test_connection()
    instrument = Instrument("NIFTY", "13", "IDX_I", "INDEX", "Nifty 50")

    try:
        client.quote([instrument])
        raise AssertionError("rate-limited quote should fail")
    except RuntimeError as error:
        assert "HTTP 429" in str(error)
    assert client.health.status == "CONNECTED"
    assert client.health.token_valid


def test_report_writer_rejects_credentials_and_exports_no_token(tmp_path):
    health = {
        "status": "DISCONNECTED", "last_connection_time": None,
        "last_heartbeat": None, "latency_ms": None, "token_valid": False,
        "detail": "offline",
    }
    write_broker_reports(tmp_path, health, [], {"step": "27"})
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    assert "access-token" not in combined.lower()
    assert "secret" not in combined.lower()


def test_broker_frontend_loads_disconnected_without_network():
    app = AppTest.from_file(str(UI_ROOT / "pages/14_Broker_Connection.py"))
    app.run(timeout=30)
    assert not app.exception
    live = AppTest.from_file(str(UI_ROOT / "pages/3_Live_Market.py"))
    live.run(timeout=30)
    assert not live.exception


def test_step27_modules_have_no_order_surface_or_hardcoded_token():
    source = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "src/dhan_live_broker.py", "src/dhan_live_integration.py",
            "apps/research_ui/pages/14_Broker_Connection.py",
        ))
    for forbidden in ("place_order", "modify_order", "cancel_order", "eyJ0eXAi"):
        assert forbidden not in source
