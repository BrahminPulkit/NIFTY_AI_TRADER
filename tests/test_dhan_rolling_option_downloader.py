from datetime import date

from src.dhan_rolling_option_downloader import date_chunks, request_plan, response_frame


def test_five_year_plan_obeys_thirty_day_limit():
    plan = request_plan(date(2021, 8, 8), date(2026, 8, 8))
    assert len(plan) == 61
    assert all((date.fromisoformat(row["toDate"]) - date.fromisoformat(row["fromDate"])).days <= 30 for row in plan)
    assert all(row["drvOptionType"] == "PUT" for row in plan)


def test_put_response_is_normalized():
    body = {"data": {"pe": {"timestamp": [1756698300], "open": [100],
            "high": [102], "low": [99], "close": [101], "volume": [10],
            "oi": [20], "iv": [12.5], "strike": [24500], "spot": [24510]}, "ce": None}}
    frame = response_frame(body, "PUT")
    assert frame.option_type.iat[0] == "PUT"
    assert frame.close.iat[0] == 101


def test_invalid_chunk_size_is_rejected():
    try:
        list(date_chunks(date(2026, 1, 1), date(2026, 4, 1), 90))
    except ValueError as exc:
        assert "30 days" in str(exc)
    else:
        raise AssertionError("90-day rolling option chunk was accepted")


def test_contract_consistent_plan_uses_weekly_current_both_sides():
    call = request_plan(
        date(2026, 1, 1), date(2026, 1, 2), option_type="CALL",
        expiry_flag="WEEK", expiry_code=0)
    put = request_plan(
        date(2026, 1, 1), date(2026, 1, 2), option_type="PUT",
        expiry_flag="WEEK", expiry_code=0)
    assert call[0]["drvOptionType"] == "CALL"
    assert put[0]["drvOptionType"] == "PUT"
    assert call[0]["expiryFlag"] == put[0]["expiryFlag"] == "WEEK"
    assert call[0]["expiryCode"] == put[0]["expiryCode"] == 0
