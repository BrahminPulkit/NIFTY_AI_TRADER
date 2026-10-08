from types import SimpleNamespace

import pandas as pd

from src.exact_contract_batch_acquisition import ExactContractBatchAcquisition


class Resolver:
    def __init__(self, resolved=True):
        timestamps = pd.date_range(
            "2026-08-07 09:15", periods=375, freq="min", tz="Asia/Kolkata")
        self.index = pd.DataFrame({"timestamp": timestamps, "index_close": 24550.0})
        self.resolved = resolved

    def resolve_contract(self, *, date, option_type):
        if not self.resolved:
            return SimpleNamespace(status="UNRESOLVED", reason="NO_MASTER", expiry=None,
                                   strike=None, security_id=None, option_type=option_type)
        return SimpleNamespace(
            status="RESOLVED", reason="ok", expiry="2026-08-11", strike=24550.0,
            security_id="101" if option_type == "CE" else "102",
            option_type=option_type,
            contract_segment_id=f"segment-{option_type}")


class Client:
    def __init__(self, rows=375):
        self.rows = rows

    def completed_candles(self, instrument, _start, _end):
        timestamp = pd.date_range(
            "2026-08-07 09:15", periods=self.rows, freq="min", tz="Asia/Kolkata")
        return pd.DataFrame({
            "timestamp": timestamp, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 10,
        })


def service(tmp_path, *, rows=375, resolved=True):
    return ExactContractBatchAcquisition(
        Client(rows), resolver=Resolver(resolved),
        output_root=tmp_path / "data", report_root=tmp_path / "reports")


def test_complete_exact_pair_is_ready_but_does_not_emit_small_training_set(tmp_path):
    result = service(tmp_path).acquire(["2026-08-07"])

    assert result["training_ready_sessions"] == 1
    assert result["training_ready_file"] is None
    assert not (tmp_path / "data/training_ready.parquet").exists()
    frame = pd.read_parquet(tmp_path / "data/2026-08-07/ce_pe.parquet")
    assert len(frame) == 750
    assert frame.spot.notna().all()


def test_missing_candles_are_partial_without_forward_fill(tmp_path):
    result = service(tmp_path, rows=374).acquire(["2026-08-07"])

    assert result["partial_sessions"] == 1
    coverage = pd.read_csv(tmp_path / "reports/exact_contract_data_coverage.csv")
    assert coverage.loc[0, "missing_ce"] == coverage.loc[0, "missing_pe"] == 1


def test_missing_contract_identity_is_unresolved_and_makes_no_data(tmp_path):
    result = service(tmp_path, resolved=False).acquire(["2026-08-07"])

    assert result["unresolved_sessions"] == 1
    assert not (tmp_path / "data/2026-08-07/ce_pe.parquet").exists()
