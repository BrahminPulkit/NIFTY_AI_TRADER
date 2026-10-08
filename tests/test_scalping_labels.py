import pandas as pd

from src.scalping_labels import ScalpingLabelConfig, build_event_labels


def options(highs, lows):
    count = len(highs)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-08-07 10:00", periods=count, freq="min", tz="Asia/Kolkata"),
        "strike": [24500] * count, "expiry": ["2026-08-11"] * count,
        "option_type": ["CE"] * count, "security_id": ["1"] * count,
        "underlying": ["NIFTY"] * count, "open": [100] * count,
        "high": highs, "low": lows, "close": [100] * count, "volume": [10] * count,
    })


def entry():
    return pd.DataFrame({"timestamp": [pd.Timestamp("2026-08-07 10:00", tz="Asia/Kolkata")],
                         "option_type": ["CE"]})


def test_target_first_is_win_and_stop_first_is_loss():
    cfg = ScalpingLabelConfig(target_pct=.05, stop_pct=.03, brokerage_per_order=0)
    assert build_event_labels(options([100, 106, 100], [100, 99, 96]), entry(), cfg).outcome.iat[0] == "WIN"
    assert build_event_labels(options([100, 101, 106], [100, 96, 99]), entry(), cfg).outcome.iat[0] == "LOSS"


def test_same_candle_target_and_stop_is_ambiguous_and_excluded():
    result = build_event_labels(
        options([100, 106], [100, 96]), entry(),
        ScalpingLabelConfig(target_pct=.05, stop_pct=.03, brokerage_per_order=0))
    assert result.outcome.iat[0] == "AMBIGUOUS"
    assert not bool(result.training_eligible.iat[0])


def test_contract_transition_is_not_used_as_future_path():
    data = options([100, 100, 110], [100, 99, 99])
    data.loc[2, ["strike", "security_id"]] = [24550, "2"]
    result = build_event_labels(data, entry(), ScalpingLabelConfig(brokerage_per_order=0))
    assert result.outcome.iat[0] == "TIMEOUT"
