from datetime import date, datetime, timedelta
from pathlib import Path
import json

import pandas as pd

from src.research_option_recorder import (
    ProviderRequestError, RecorderConfig, ResearchOptionRecorder,
    audit_session, normalize_snapshot,
)


def config(tmp_path: Path, **updates) -> RecorderConfig:
    values = {
        "schema_version": "phase19-options-live-v1", "recorder_version": "test",
        "underlying": "NIFTY", "underlying_segment": "IDX_I",
        "sampling_interval_seconds": 4, "strikes_each_side": 5,
        "session_timezone": "Asia/Kolkata", "session_open": "09:15:00",
        "session_close": "09:15:08", "synchronization_tolerance_seconds": 5,
        "stale_quote_seconds": 15, "minimum_snapshot_coverage_pct": 95,
        "minimum_contracts_per_snapshot": 22, "maximum_gap_seconds": 16,
        "maximum_consecutive_failures": 5, "maximum_auth_failures": 1,
        "raw_root": str(tmp_path / "raw"), "normalized_root": str(tmp_path / "normalized"),
        "audit_root": str(tmp_path / "audit"),
    }
    values.update(updates)
    return RecorderConfig(**values)


def chain_body(timestamp="2026-08-13T09:15:00+05:30"):
    oc = {}
    for index, strike in enumerate(range(24750, 25300, 50)):
        def leg(side):
            return {
                "security_id": f"{side}{strike}", "last_price": 100 + index,
                "top_bid_price": 99 + index, "top_ask_price": 101 + index,
                "volume": 1000 + index, "oi": 2000 + index,
                "implied_volatility": 12.5,
                "last_trade_time": timestamp,
                "greeks": {"delta": .5, "gamma": .02, "theta": -3, "vega": 4},
            }
        oc[str(strike)] = {"ce": leg("CE"), "pe": leg("PE")}
    return {"last_price": 25010, "oc": oc}


def test_normalization_selects_causal_atm_plus_minus_five_and_both_sides():
    rows, meta = normalize_snapshot(
        raw_body=chain_body(), snapshot_id="one",
        received_at=datetime.fromisoformat("2026-08-13T09:15:01+05:30"),
        expiry="2026-08-13", each_side=5, sync_tolerance_seconds=5,
        stale_quote_seconds=15)
    assert meta["atm_strike"] == 25000
    assert len(rows) == 22
    assert {row["option_type"] for row in rows} == {"CE", "PE"}
    assert {row["strike_offset"] for row in rows} == set(range(-5, 6))
    assert all(row["synchronization_ok"] for row in rows)
    assert all(row["nifty_volume"] is None for row in rows)
    assert all("vwap" not in row for row in rows)


def test_missing_provider_timestamp_is_explicit_not_fabricated():
    body = chain_body()
    for legs in body["oc"].values():
        for leg in legs.values():
            leg.pop("last_trade_time")
    rows, _ = normalize_snapshot(
        raw_body=body, snapshot_id="one", received_at=datetime.now().astimezone(),
        expiry="2026-08-20", each_side=5, sync_tolerance_seconds=5,
        stale_quote_seconds=15)
    assert all(row["provider_timestamp"] is None for row in rows)
    assert all(row["quote_stale"] for row in rows)


class FakeSource:
    def __init__(self, body):
        self.body = body
    def active_expiry(self, **kwargs):
        return "2026-08-13"
    def snapshot(self, **kwargs):
        return 200, self.body
    def authenticate(self):
        return {"status": "CONNECTED"}


def test_raw_archive_preserves_duplicate_provider_responses(tmp_path):
    cfg = config(tmp_path)
    current = datetime.fromisoformat("2026-08-13T09:15:01+05:30")
    recorder = ResearchOptionRecorder(cfg, FakeSource(chain_body()), now=lambda: current)
    recorder.record_one(date(2026, 8, 13))
    recorder.record_one(date(2026, 8, 13))
    paths = recorder.paths(date(2026, 8, 13))
    raw = paths["raw"].read_text(encoding="utf-8").splitlines()
    normalized = pd.read_parquet(paths["normalized"])
    assert len(raw) == 2
    assert len(normalized) == 44
    assert "access-token" not in paths["raw"].read_text(encoding="utf-8")


def test_completeness_requires_coverage_contracts_and_quotes(tmp_path):
    cfg = config(tmp_path)
    times = iter([
        datetime.fromisoformat("2026-08-13T09:15:00+05:30"),
        datetime.fromisoformat("2026-08-13T09:15:00+05:30"),
        datetime.fromisoformat("2026-08-13T09:15:04+05:30"),
        datetime.fromisoformat("2026-08-13T09:15:04+05:30"),
    ])
    recorder = ResearchOptionRecorder(cfg, FakeSource(chain_body()), now=lambda: next(times))
    recorder.record_one(date(2026, 8, 13))
    recorder.record_one(date(2026, 8, 13))
    result = audit_session(date(2026, 8, 13), cfg)
    assert result["actual_snapshots"] == 2
    assert result["complete"]
    assert result["missing_bid"] == 0
    assert result["missing_ask"] == 0


def test_contract_identity_uses_expiry_strike_and_side():
    rows, _ = normalize_snapshot(
        raw_body=chain_body(), snapshot_id="one",
        received_at=datetime.fromisoformat("2026-08-13T09:15:01+05:30"),
        expiry="2026-08-13", each_side=5, sync_tolerance_seconds=5,
        stale_quote_seconds=15)
    assert len({row["contract_id"] for row in rows}) == 22
    assert all(row["expiry"] in row["contract_id"] for row in rows)


class AdvancingClock:
    def __init__(self, value):
        self.value = value
        self.sleeps = []
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class RateLimitedThenHealthy(FakeSource):
    def __init__(self, body, clock):
        super().__init__(body)
        self.clock = clock
        self.calls = []
    def snapshot(self, **kwargs):
        self.calls.append(self.clock.now())
        if len(self.calls) == 1:
            raise ProviderRequestError(
                "HTTP 429; rate limited", status_code=429,
                response_body={"errorCode": "DH-429"}, retry_after_seconds=4)
        return 200, self.body


def test_429_uses_bounded_backoff_without_catchup_or_fabrication(tmp_path):
    cfg = config(
        tmp_path, schema_version="phase19-options-live-v2",
        recorder_version="2.0.0", sampling_interval_seconds=5,
        session_close="09:15:16", rate_limit_backoff_seconds=6,
        network_backoff_seconds=2, maximum_backoff_seconds=60)
    clock = AdvancingClock(datetime.fromisoformat("2026-08-13T09:15:00+05:30"))
    source = RateLimitedThenHealthy(chain_body(), clock)
    recorder = ResearchOptionRecorder(cfg, source, now=clock.now, sleeper=clock.sleep)
    result = recorder.run_session(date(2026, 8, 13))
    assert clock.sleeps[:3] == [6, 5.0, 5.0]
    assert all((right - left).total_seconds() >= 5 for left, right in zip(source.calls, source.calls[1:]))
    assert result["runtime_snapshots"] == 2
    frame = pd.read_parquet(recorder.paths(date(2026, 8, 13))["normalized"])
    assert frame["snapshot_id"].nunique() == 2
    assert len(frame) == 44


def test_429_raw_error_and_events_are_preserved_without_credentials(tmp_path):
    cfg = config(tmp_path, sampling_interval_seconds=5,
                 rate_limit_backoff_seconds=6)
    clock = AdvancingClock(datetime.fromisoformat("2026-08-13T09:15:00+05:30"))
    source = RateLimitedThenHealthy(chain_body(), clock)
    recorder = ResearchOptionRecorder(cfg, source, now=clock.now, sleeper=clock.sleep)
    try:
        recorder.record_one(date(2026, 8, 13), "2026-08-13")
    except ProviderRequestError:
        pass
    paths = recorder.paths(date(2026, 8, 13))
    error = json.loads(paths["raw_errors"].read_text(encoding="utf-8").splitlines()[0])
    events = paths["events"].read_text(encoding="utf-8")
    assert error["http_status"] == 429
    assert error["response"] == {"errorCode": "DH-429"}
    assert "RATE_LIMITED" in events
    assert "access-token" not in paths["raw_errors"].read_text(encoding="utf-8")
    assert not paths["raw"].exists()


def test_unsustainable_option_chain_interval_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="documented 3-second limit"):
        config(tmp_path, sampling_interval_seconds=2)
