import pandas as pd

from src.historical_contract_resolution_service import HistoricalContractResolver


def _master(path):
    pd.DataFrame([
        {"SECURITY_ID": 101, "INSTRUMENT": "OPTIDX", "UNDERLYING_SYMBOL": "NIFTY",
         "SM_EXPIRY_DATE": "2026-08-11", "STRIKE_PRICE": 24550,
         "OPTION_TYPE": "CE", "EXPIRY_FLAG": "W", "DISPLAY_NAME": "NIFTY CE"},
        {"SECURITY_ID": 102, "INSTRUMENT": "OPTIDX", "UNDERLYING_SYMBOL": "NIFTY",
         "SM_EXPIRY_DATE": "2026-08-11", "STRIKE_PRICE": 24550,
         "OPTION_TYPE": "PE", "EXPIRY_FLAG": "W", "DISPLAY_NAME": "NIFTY PE"},
    ]).to_csv(path, index=False)


def test_resolves_exact_pair_only_inside_snapshot_evidence_window(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _master(archive / "dhan_master_20260807080000.csv")
    index = tmp_path / "index.parquet"
    pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-08-07 09:15", periods=375, freq="min", tz="Asia/Kolkata"),
        "index_close": [24549.0] * 375,
    }).to_parquet(index, index=False)
    resolver = HistoricalContractResolver(
        archive=archive, index_path=index,
        validated_manifest=tmp_path / "missing.json")

    ce = resolver.resolve_contract(date="2026-08-07", option_type="CE")
    pe = resolver.resolve_contract(date="2026-08-07", option_type="PE")

    assert (ce.status, ce.security_id, ce.strike) == ("RESOLVED", "101", 24550.0)
    assert (pe.status, pe.security_id, pe.expiry) == ("RESOLVED", "102", "2026-08-11")
    assert ce.contract_segment_id != pe.contract_segment_id


def test_returns_explicit_unresolved_instead_of_extrapolating(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _master(archive / "dhan_master_20260807080000.csv")
    index = tmp_path / "index.parquet"
    pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-08-06 15:29", tz="Asia/Kolkata")],
        "index_close": [24549.0],
    }).to_parquet(index, index=False)
    resolver = HistoricalContractResolver(
        archive=archive, index_path=index,
        validated_manifest=tmp_path / "missing.json")

    result = resolver.resolve_contract(date="2026-08-06", option_type="CE")

    assert result.status == "UNRESOLVED"
    assert result.reason == "NO_ELIGIBLE_TIMESTAMPED_MASTER"


def test_rejects_unsupported_policy_without_guessing(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    resolver = HistoricalContractResolver(
        archive=archive, index_path=tmp_path / "missing.parquet",
        validated_manifest=tmp_path / "missing.json")

    result = resolver.resolve_contract(
        date="2026-08-07", option_type="CE", expiry_policy="MONTH/NEXT")

    assert result.status == "UNRESOLVED"
    assert result.reason == "UNSUPPORTED_EXPIRY_POLICY"
