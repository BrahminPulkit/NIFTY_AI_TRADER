import json

import pandas as pd

from apps.api.services import signal_audit


def test_signal_audit_reports_coverage_and_retrospective_move(tmp_path, monkeypatch):
    journal = tmp_path / "predictions.jsonl"
    journal.write_text(json.dumps({
        "timestamp": "2026-08-07 10:00:00+05:30", "decision": "SKIP",
        "probability": None, "reason": "NO_SETUP", "strategy": "NONE",
    }) + "\n", encoding="utf-8")
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-07 10:00", periods=16, freq="min", tz="Asia/Kolkata"),
        "open": [100.0] * 16, "high": [100.0] + [100.2] * 15,
        "low": [100.0] * 16, "close": [100.0] * 16, "volume": [1.0] * 16,
    })
    monkeypatch.setattr(signal_audit, "JOURNAL", journal)
    monkeypatch.setattr(signal_audit, "_session_candles", lambda _: candles)

    result = signal_audit.signal_audit_payload("2026-08-07")

    assert result["summary"]["market_candles"] == 16
    assert result["summary"]["audited_candles"] == 1
    assert result["summary"]["model_evaluations"] == 0
    assert result["summary"]["review_candidates"] == 1
    assert result["events"][0]["model"] == "NOT_RUN"
    assert "not live signals" in result["methodology"]["note"]
