import pandas as pd

from commands.research.main_phase3_option_data_audit import _date_coverage


def test_date_coverage_never_marks_mixed_legacy_policy_training_ready():
    source = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:00"]).tz_localize("Asia/Kolkata"),
        "option_type": ["CE", "PE"], "strike": [24000, 24000],
        "resolution_status": ["RESOLVED", "RESOLVED"],
    })
    result = _date_coverage(source, pd.DataFrame()).iloc[0]
    assert result.ce_identity_complete
    assert result.pe_identity_complete
    assert not result.legacy_ce_pe_policy_consistent
    assert not result.training_ready


def test_forward_collection_requires_full_session_both_sides():
    source = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 10:00", tz="Asia/Kolkata")],
        "option_type": ["CE"], "strike": [24000], "resolution_status": ["UNRESOLVED"],
    })
    collected = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:00"], utc=True),
        "option_type": ["CE", "PE"],
    })
    result = _date_coverage(source, collected).iloc[0]
    assert not result.forward_session_complete
