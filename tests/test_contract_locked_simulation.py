import pandas as pd

from tools.run_validated20_strategy_validation import simulate_contract_locked


def option_rows(strike, closes):
    timestamps = pd.date_range("2026-01-01 10:00", periods=len(closes), freq="min", tz="Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": timestamps, "underlying": "NIFTY", "expiry": "2026-01-01",
        "strike": float(strike), "option_type": "CE", "security_id": pd.NA,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": 1, "contract_segment_id": f"contract-{strike}",
    })


def test_exit_stays_on_entry_contract_after_atm_switch():
    first = option_rows(24500, [100, 102, 106])
    second = option_rows(24550, [90, 80, 70])
    selected = pd.concat([first.iloc[[0]], second.iloc[[1]], second.iloc[[2]]], ignore_index=True)
    panel = pd.concat([first, second], ignore_index=True)
    candidates = pd.DataFrame({
        "timestamp": [first.timestamp.iloc[0]], "strategy": ["BREAKOUT"],
        "option_type": ["CE"],
    })
    result = simulate_contract_locked(candidates, selected, panel).iloc[0]
    assert result.strike == 24500
    assert result.outcome == "WIN"
    assert result.exit_timestamp == first.timestamp.iloc[2]
    assert result.exit_contract_matches_entry
