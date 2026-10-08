import json

from src.live_pipeline_monitor import LivePipelineMonitor


def test_health_write_is_atomic_and_reproducible(tmp_path):
    path = tmp_path / "health.json"
    monitor = LivePipelineMonitor(path)
    monitor.update(status="READY", model_status="VERIFIED")
    payload = json.loads(path.read_text())
    assert payload["status"] == "READY"
    assert payload["model_status"] == "VERIFIED"
    assert not path.with_suffix(".tmp").exists()

