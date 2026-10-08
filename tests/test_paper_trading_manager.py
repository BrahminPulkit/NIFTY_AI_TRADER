import pandas as pd

from apps.api.services.replay import session_payload
from src.paper_trading_engine import PaperTradingConfig
from src.paper_trading_manager import PaperTradingManager


def test_replay_paper_lifecycle_uses_frozen_signal_and_chronological_candles(tmp_path):
    replay = session_payload("2024-03-27")
    signal = next(event for event in replay["events"] if event["decision"] == "TRADE")
    row = next(candle for candle in replay["candles"] if candle["timestamp"] == signal["timestamp"])
    config = PaperTradingConfig(
        initial_capital=100_000, quantity=100, brokerage_per_order=0,
        stt_sell_rate=0, exchange_rate_each_side=0, sebi_rate_each_side=0,
        stamp_duty_buy_rate=0, gst_rate=0, slippage_bps_each_side=0,
    )
    manager = PaperTradingManager(tmp_path, config)
    candle = manager.candle(row)
    review = manager.review(signal, candle)
    assert review["eligible"]
    assert review["broker_orders_enabled"] is False
    opened = manager.open("replay", signal, candle, "2024-03-27")
    assert opened["result"]["status"] == "OPENED"

    later = [
        item for item in replay["candles"]
        if pd.Timestamp(item["timestamp"]) > pd.Timestamp(signal["timestamp"])
    ]
    result = None
    for item in later:
        result = manager.update_replay("2024-03-27", manager.candle(item))
        if result["status"] == "CLOSED":
            break
    assert result["status"] == "CLOSED"
    account = manager.account("replay", "2024-03-27")
    assert account["position"] is None
    assert len(account["trades"]) == 1
    assert manager.account("live")["trades"] == []


def test_paper_account_restores_open_position_without_broker_interface(tmp_path):
    config = PaperTradingConfig(quantity=2)
    manager = PaperTradingManager(tmp_path, config)
    timestamp = "2030-01-02T10:00:00+05:30"
    prediction = {
        "timestamp": timestamp, "decision": "TRADE", "probability": .95,
        "reason": "ALL_FROZEN_GATES_PASSED", "strategy": "MOMENTUM_BREAKOUT",
        "regime": "TREND_UP",
    }
    candle = manager.candle({
        "timestamp": timestamp, "open": 100, "high": 101,
        "low": 99, "close": 100,
    })
    manager.open("live", prediction, candle)
    restored = PaperTradingManager(tmp_path, config)
    assert restored.account("live")["position"]["entry_premium"] == 100
    assert restored.account("live")["rules"]["broker_orders_enabled"] is False


def test_manual_pe_position_tracks_selected_live_contract(tmp_path):
    config = PaperTradingConfig(
        initial_capital=100_000, quantity=1, brokerage_per_order=0,
        stt_sell_rate=0, exchange_rate_each_side=0, sebi_rate_each_side=0,
        stamp_duty_buy_rate=0, gst_rate=0, slippage_bps_each_side=0)
    manager = PaperTradingManager(tmp_path, config)
    contract = {
        "security_id": "41014", "option_type": "PE", "strike": 24550,
        "expiry": "2026-08-11", "market": "NIFTY_ATM_PE",
        "ltp": 110, "bid": 109.5, "ask": 110.5,
    }
    review = manager.review_manual(
        contract, "2026-08-07T10:00:00+05:30", quantity=2)
    opened = manager.open_manual(
        contract, "2026-08-07T10:00:00+05:30", quantity=2)

    assert review["eligible"]
    assert review["entry_premium"] == 110.5
    assert opened["account"]["position"]["option_type"] == "PE"
    assert opened["account"]["position"]["security_id"] == "41014"
    marked = manager.update_live(
        pd.DataFrame(), option_chain=[{**contract, "bid": 112}],
        timestamp="2026-08-07T10:01:00+05:30")
    assert marked["status"] == "OPEN"
    assert manager.account("live")["position"]["current_pnl"] == 3.0


def test_corrected_ai_pe_uses_evaluated_threshold_and_selected_contract(tmp_path):
    config = PaperTradingConfig(
        initial_capital=100_000, quantity=1, brokerage_per_order=0,
        stt_sell_rate=0, exchange_rate_each_side=0, sebi_rate_each_side=0,
        stamp_duty_buy_rate=0, gst_rate=0, slippage_bps_each_side=0)
    manager = PaperTradingManager(tmp_path, config)
    timestamp = "2026-08-07T10:00:00+05:30"
    prediction = {
        "timestamp": timestamp, "decision": "TRADE", "probability": .74,
        "required_probability": .72, "strategy": "VWAP_RECLAIM",
        "regime": "BEARISH", "selected_option": {
            "security_id": "41014", "option_type": "PE", "strike": 24550,
            "expiry": "2026-08-11",
        },
    }
    candle = manager.candle({
        "timestamp": timestamp, "open": 100, "high": 101, "low": 99, "close": 100})

    assert manager.review(prediction, candle)["eligible"]
    opened = manager.open("live", prediction, candle)
    position = opened["account"]["position"]
    assert position["option_type"] == "PE"
    assert position["security_id"] == "41014"
