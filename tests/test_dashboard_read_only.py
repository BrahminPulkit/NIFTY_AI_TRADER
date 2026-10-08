from apps.research_ui.components.data import ARTIFACTS, artifact_health, latest_snapshot, load_frame


def test_dashboard_registry_has_frozen_sources():
    assert "joined_probability" in ARTIFACTS
    assert "readiness" in ARTIFACTS
    health = artifact_health()
    assert set(health.status).issubset({"AVAILABLE", "MISSING"})


def test_cached_loader_returns_defensive_copy():
    one = load_frame("readiness")
    two = load_frame("readiness")
    assert one is not two
    assert one.equals(two)


def test_latest_snapshot_is_historical_and_read_only():
    snapshot = latest_snapshot()
    assert "timestamp" in snapshot
    assert "catboost_predicted_probability" in snapshot

