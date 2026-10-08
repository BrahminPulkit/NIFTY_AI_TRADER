import json

import pandas as pd

from src.dhan_option_data_collector import DATA_COLUMNS, OptionMarketDataCollector


def _snapshot():
    timestamps = pd.date_range(
        "2026-07-30 09:15", periods=2, freq="min", tz="Asia/Kolkata")
    candles = pd.DataFrame({
        "timestamp": timestamps, "open": [100, 101], "high": [102, 103],
        "low": [99, 100], "close": [101, 102], "volume": [10, 20],
    })

    def metadata(market, security_id, option_type):
        return {
            "market": market, "security_id": security_id,
            "exchange_segment": "NSE_FNO", "instrument": "OPTIDX",
            "display_name": f"NIFTY 25000 {option_type}",
            "expiry": "2026-07-30", "strike": 25000.0,
            "option_type": option_type,
        }

    return {
        "quotes": {"NIFTY": 25002.5},
        "options": {
            "NIFTY_ATM_CE": metadata("NIFTY_ATM_CE", "101", "CE"),
            "NIFTY_ATM_PE": metadata("NIFTY_ATM_PE", "102", "PE"),
        },
        "candles": {
            "NIFTY_ATM_CE": candles,
            "NIFTY_ATM_PE": candles.assign(close=[110, 109]),
        },
        "option_chain": [
            {"security_id": "101", "ltp": 102, "oi": 500, "bid": 101,
             "ask": 102, "spread": 1},
            {"security_id": "102", "ltp": 109, "oi": 600, "bid": 108,
             "ask": 109, "spread": 1},
        ],
    }


def test_collector_writes_contract_aware_ce_and_pe_and_deduplicates(tmp_path):
    collector = OptionMarketDataCollector(tmp_path)
    now = "2026-07-30 09:17:20+05:30"
    first = collector.collect_snapshot(_snapshot(), collected_at=now)
    second = collector.collect_snapshot(_snapshot(), collected_at=now)

    frame = pd.read_parquet(first["file"])
    assert set(frame.columns) == set(DATA_COLUMNS)
    assert set(frame.option_type) == {"CE", "PE"}
    assert set(frame.security_id) == {"101", "102"}
    assert len(frame) == 2
    assert second["total_rows"] == 2
    manifest = json.loads(
        (tmp_path / "2026-07-30" / "manifest.json").read_text())
    assert manifest["ce_rows"] == manifest["pe_rows"] == 1


def test_collector_waits_when_candles_are_unavailable(tmp_path):
    result = OptionMarketDataCollector(tmp_path).collect_snapshot({
        "options": {}, "candles": {}, "option_chain": []})
    assert result["status"] == "WAITING"
    assert not list(tmp_path.rglob("*.parquet"))
