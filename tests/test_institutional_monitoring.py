import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.institutional_monitoring import (
    IncrementalJSONL, InstitutionalMonitor, MonitoringConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def test_probability_psi_is_deterministic_and_detects_shift():
    same = pd.Series([.05, .15, .25, .35, .45, .55, .65, .75, .85, .95])
    shifted = pd.Series([.91, .92, .93, .94, .95, .96, .97, .98, .99, .90])
    assert InstitutionalMonitor.probability_psi(same, same) == 0
    first = InstitutionalMonitor.probability_psi(same, shifted)
    assert first > 0
    assert first == InstitutionalMonitor.probability_psi(same, shifted)


def test_incremental_journal_reads_only_new_rows(tmp_path):
    source = tmp_path / "journal.jsonl"
    cache = tmp_path / "cache.parquet"
    cursor = tmp_path / "cursor.json"
    source.write_text(json.dumps({"timestamp": "2026-01-01", "value": 1}) + "\n")
    reader = IncrementalJSONL(source, cache, cursor)
    assert len(reader.update()) == 1
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": "2026-01-02", "value": 2}) + "\n")
    result = reader.update()
    assert len(result) == 2
    assert result.value.tolist() == [1, 2]


def test_health_statuses_and_frozen_model_hashes():
    monitor = InstitutionalMonitor(ROOT, MonitoringConfig())
    predictions = monitor.predictions()
    health = monitor.system_health(predictions)
    valid = {"READY", "WAITING", "STALE", "BLOCKED", "FAILED"}
    assert set(health.status).issubset(valid)
    model = health.loc[health.component.eq("Model")].iloc[0]
    assert model.status == "READY"


def test_monitoring_metadata_proves_no_mutation_or_inference():
    metadata = json.loads(
        (ROOT / "reports/system_monitoring/metadata.json").read_text(encoding="utf-8"))
    assert metadata["frozen_hashes_identical"] is True
    assert metadata["frozen_hashes_before"] == metadata["frozen_hashes_after"]
    assert metadata["model_inference_invoked"] is False
    assert metadata["probabilities_regenerated"] is False
    assert metadata["datasets_modified"] is False


def test_monitoring_dashboard_loads_without_exception():
    app = AppTest.from_file(str(
        ROOT / "apps/research_ui/pages/13_System_Monitoring.py"))
    app.run(timeout=30)
    assert not app.exception
