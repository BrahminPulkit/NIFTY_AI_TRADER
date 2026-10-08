from pathlib import Path

import pandas as pd

from src.historical_option_contract_resolver import resolve_contract_identity


def test_resolver_stops_at_snapshot_nearest_expiry():
    captured = pd.Timestamp("2026-07-22 08:00", tz="Asia/Kolkata")
    catalog = pd.DataFrame({
        "security_id": ["10", "20"], "expiry": pd.to_datetime(["2026-07-28", "2026-08-04"]),
        "strike": [24500, 24500], "option_type": ["PE", "PE"], "expiry_flag": ["W", "W"],
        "display_name": ["NIFTY", "NIFTY"], "snapshot_timestamp": [captured, captured],
        "master_source": ["master.csv", "master.csv"],
    })
    rows = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-23 10:00", "2026-07-29 10:00"]).tz_localize("Asia/Kolkata"),
        "strike": [24500, 24500], "option_type": ["PE", "PE"],
        "requested_expiry_flag": ["W", "W"], "underlying": ["NIFTY", "NIFTY"],
    })
    result = resolve_contract_identity(rows, catalog)
    assert result.resolution_status.tolist() == ["RESOLVED", "UNRESOLVED"]
    assert result.security_id.iloc[0] == "10"


def test_resolver_never_fills_missing_strike_contract():
    captured = pd.Timestamp("2026-07-22 08:00", tz="Asia/Kolkata")
    catalog = pd.DataFrame({
        "security_id": ["10"], "expiry": pd.to_datetime(["2026-07-28"]),
        "strike": [24500], "option_type": ["PE"], "expiry_flag": ["W"],
        "display_name": ["NIFTY"], "snapshot_timestamp": [captured],
        "master_source": ["master.csv"],
    })
    rows = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-07-23 10:00", tz="Asia/Kolkata")],
        "strike": [24550], "option_type": ["PE"], "requested_expiry_flag": ["W"],
        "underlying": ["NIFTY"],
    })
    result = resolve_contract_identity(rows, catalog)
    assert result.resolution_status.iloc[0] == "UNRESOLVED"
    assert pd.isna(result.security_id.iloc[0])
