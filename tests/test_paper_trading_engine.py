import json

import pandas as pd
import pytest

from src.paper_trading_analytics import performance
from src.paper_trading_engine import (
    OptionCandle, PaperTradingConfig, PaperTradingEngine, trading_costs,
)


def config(**changes):
    base = dict(
        initial_capital=100_000, quantity=100, stop_loss_pct=.20,
        target_pct=.05, maximum_holding_minutes=15,
        brokerage_per_order=0, stt_sell_rate=0, exchange_rate_each_side=0,
        sebi_rate_each_side=0, stamp_duty_buy_rate=0, gst_rate=0,
        slippage_bps_each_side=0,
    )
    base.update(changes)
    return PaperTradingConfig(**base)


def prediction(timestamp="2030-01-02 10:00+05:30", decision="TRADE"):
    return {
        "timestamp": timestamp, "decision": decision, "probability": .95,
        "reason": "ALL_FROZEN_GATES_PASSED", "strategy": "MOMENTUM_BREAKOUT",
        "regime": "TREND_UP|NORMAL", "inference_timestamp": timestamp,
    }


def candle(timestamp="2030-01-02 10:00+05:30", open_=100, high=101, low=99, close=100):
    return OptionCandle(timestamp, open_, high, low, close)


def test_entry_overlap_and_conservative_stop_first(tmp_path):
    engine = PaperTradingEngine(tmp_path, config())
    assert engine.consume_prediction(
        prediction(), candle(), {"status": "HEALTHY"})["status"] == "OPENED"
    assert engine.consume_prediction(
        prediction("2030-01-02 10:01+05:30"),
        candle("2030-01-02 10:01+05:30"), {"status": "HEALTHY"})["reason"] == "OVERLAPPING_POSITION"
    # Candle reaches both 80 stop and 105 target; stop is deliberately first.
    result = engine.on_candle(candle(
        "2030-01-02 10:02+05:30", open_=100, high=106, low=79, close=102))
    assert result["exit_reason"] == "STOP_LOSS_HIT"
    assert result["exit_premium"] == 80
    assert result["gross_pnl"] == -2000
    assert engine.position is None


def test_target_time_manual_and_gap_exits(tmp_path):
    engine = PaperTradingEngine(tmp_path, config())
    engine.consume_prediction(prediction(), candle(), {"status": "READY"})
    assert engine.on_candle(candle(
        "2030-01-02 10:01+05:30", open_=106, high=108, low=105, close=107))["exit_reason"] == "GAP_THROUGH_TARGET"
    engine.consume_prediction(prediction("2030-01-03 10:00+05:30"),
                              candle("2030-01-03 10:00+05:30"), {"status": "READY"})
    assert engine.on_candle(candle(
        "2030-01-03 10:15+05:30", open_=100, high=101, low=99, close=100))["exit_reason"] == "TIME_EXIT"
    engine.consume_prediction(prediction("2030-01-04 10:00+05:30"),
                              candle("2030-01-04 10:00+05:30"), {"status": "READY"})
    assert engine.manual_close(candle(
        "2030-01-04 10:01+05:30", close=101))["exit_reason"] == "MANUAL_CLOSE"
    engine.consume_prediction(prediction("2030-01-05 15:28+05:30"),
                              candle("2030-01-05 15:28+05:30"), {"status": "READY"})
    assert engine.on_candle(candle(
        "2030-01-05 15:29+05:30", open_=100, high=101, low=99, close=100))["exit_reason"] == "END_OF_SESSION"


def test_missing_fill_and_pipeline_fail_closed(tmp_path):
    engine = PaperTradingEngine(tmp_path, config())
    assert engine.consume_prediction(
        prediction(), None, {"status": "HEALTHY"})["status"] == "SAFE_BLOCK"
    assert engine.consume_prediction(
        prediction(), candle(), {"status": "SAFE_BLOCK"})["reason"] == "PIPELINE_NOT_READY"
    assert engine.position is None


def test_position_restart_and_append_only_journal(tmp_path):
    engine = PaperTradingEngine(tmp_path, config())
    opened = engine.consume_prediction(prediction(), candle(), {"status": "HEALTHY"})
    restarted = PaperTradingEngine(tmp_path, config())
    assert restarted.position.trade_id == opened["trade_id"]
    restarted.on_candle(candle(
        "2030-01-02 10:01+05:30", open_=106, high=107, low=105, close=106))
    rows = restarted.trades()
    assert len(rows) == 1
    assert len(restarted.events_path.read_text().splitlines()) >= 2
    assert restarted.parquet_path.exists()


def test_cost_calculation_and_analytics():
    cfg = PaperTradingConfig(quantity=50)
    cost = trading_costs(100, 110, 50, cfg)
    assert cost["total_costs"] == pytest.approx(sum(
        cost[key] for key in ["brokerage", "stt", "exchange_charges",
                              "sebi_charges", "stamp_duty", "gst", "slippage"]))
    trades = pd.DataFrame([{
        "net_pnl": 100.0, "exit_timestamp": "2030-01-02",
        "holding_time_minutes": 5,
    }, {
        "net_pnl": -50.0, "exit_timestamp": "2030-01-03",
        "holding_time_minutes": 10,
    }])
    stats = performance(trades, 100_000)
    assert stats["win_rate"] == .5
    assert stats["profit_factor"] == 2
    assert stats["expectancy"] == 25
