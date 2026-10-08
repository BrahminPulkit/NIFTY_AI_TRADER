import pandas as pd

from src.five_year_data_validator import _parse_timestamp


def test_epoch_seconds_are_parsed_as_utc_then_kolkata():
    result, encoding = _parse_timestamp(pd.Series([1626925560.0]))
    assert encoding == "unix_epoch_s"
    assert result.iloc[0].isoformat() == "2021-07-22T09:16:00+05:30"


def test_datetime_strings_remain_timezone_aware():
    result, encoding = _parse_timestamp(pd.Series(["2024-01-02T09:15:00+05:30"]))
    assert encoding == "datetime_string"
    assert str(result.dt.tz) == "Asia/Kolkata"
