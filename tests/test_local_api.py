from fastapi.testclient import TestClient

from apps.api.main import app, market_contract, public_broker
from src.dhan_connection_manager import DhanConnectionManager


def test_health_contract_is_available():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "local-api"}


def test_versioned_api_and_request_id_are_available():
    response = TestClient(app).get("/api/v1/system/health")
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_versioned_broker_session_is_sanitized():
    response = TestClient(app).get("/api/v1/broker/session")
    assert response.status_code == 200
    assert "credentials" not in response.text.lower()
    assert "access_token" not in response.text.lower()


def test_intraday_replay_is_chronological_and_frozen():
    client = TestClient(app)
    sessions = client.get("/api/v1/replay/sessions")
    assert sessions.status_code == 200
    session_date = sessions.json()["latest"]
    response = client.get("/api/v1/replay/session", params={"date": session_date})
    assert response.status_code == 200
    payload = response.json()
    candle_times = [row["timestamp"] for row in payload["candles"]]
    event_times = [row["timestamp"] for row in payload["events"]]
    assert candle_times == sorted(candle_times)
    assert event_times == sorted(event_times)
    assert payload["source"] == "FROZEN_OOF_REPLAY"
    assert payload["threshold"] == .9
    assert all(event["source"] == "FROZEN_OOF_REPLAY" for event in payload["events"])
    assert "true_label" not in response.text
    assert "realized" not in response.text


def test_api_contracts_never_publish_session_or_credentials(tmp_path):
    snapshot = DhanConnectionManager(tmp_path / "master.csv").snapshot()
    broker = public_broker(snapshot)
    market = market_contract(snapshot)
    assert set(broker) == {
        "connected", "status", "health", "market_status",
        "last_data_update", "last_error", "worker_alive",
    }
    assert "session" not in broker
    assert "credentials" not in str({**broker, **market}).lower()
