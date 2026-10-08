from apps.api import contracts


def test_desk_reports_strongest_call_and_put_writer_oi_proxy(monkeypatch):
    monkeypatch.setattr(contracts, "latest_prediction", lambda: ({
        "timestamp": "2026-08-07T10:00:00+05:30",
        "catboost_predicted_probability": 0.0,
        "recommendation": "SKIP", "decision_reason": "NO_SETUP",
        "current_strategy": "NONE", "market_state": "NONE",
        "expected_drawdown": 0.05,
    }, True, "test"))
    cache = {
        "options": {},
        "option_chain": [
            {"option_type": "CE", "strike": 24600, "oi": 200, "ltp": 100},
            {"option_type": "CE", "strike": 24700, "oi": 500, "ltp": 60},
            {"option_type": "PE", "strike": 24500, "oi": 700, "ltp": 90},
            {"option_type": "PE", "strike": 24400, "oi": 100, "ltp": 50},
        ],
    }
    writer = contracts.desk_contract(cache)["writer_activity"]

    assert writer["strongest_call"]["strike"] == 24700
    assert writer["strongest_put"]["strike"] == 24500
    assert writer["dominant"] == "PUT WRITERS"
    assert writer["call_total_oi"] == 700
    assert writer["put_total_oi"] == 800
