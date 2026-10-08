import pandas as pd

from src.put_model_training import put_targets


def test_put_target_uses_future_same_strike_only():
    timestamp = pd.date_range("2026-08-07 10:00", periods=4, freq="min", tz="Asia/Kolkata")
    option = pd.DataFrame({"timestamp":timestamp,"open":[100]*4,"high":[100,110,106,100],
        "low":[99]*4,"close":[100]*4,"volume":[1]*4,"strike":[24500,24600,24500,24500]})
    result = put_targets(option, pd.Series([timestamp[0]]))
    assert result.target_win.iat[0] == 1
    assert result.forward_rows.iat[0] == 2


def test_put_target_never_uses_next_session():
    timestamp = pd.to_datetime(["2026-08-07 15:29+05:30","2026-08-10 09:15+05:30"])
    option = pd.DataFrame({"timestamp":timestamp,"open":[100,100],"high":[100,120],
        "low":[99,99],"close":[100,100],"volume":[1,1],"strike":[24500,24500]})
    assert put_targets(option, pd.Series([timestamp[0]])).empty
