"""Translate existing process state into public presentation contracts."""

from __future__ import annotations

from apps.api.contracts import candle_contract, desk_contract, paper_summary
from src.dhan_connection_manager import get_dhan_manager


def market_contract(cache: dict) -> dict:
    return {
        "connected": cache["connected"],
        "status": cache["market_status"],
        "manager_status": cache["manager_status"],
        "quotes": cache["quotes"],
        "updated_at": cache["last_data_update"],
        "last_error": cache["last_error"],
        "signal": cache.get("signal"),
        "signal_status": cache.get("signal_status", "UNAVAILABLE"),
        "signal_error": cache.get("signal_error"),
        "put_signal": cache.get("put_signal"),
        "put_signal_status": cache.get("put_signal_status", "UNAVAILABLE"),
        "put_signal_error": cache.get("put_signal_error"),
        "multistrategy_signal": cache.get("multistrategy_signal"),
        "multistrategy_status": cache.get("multistrategy_status", "UNAVAILABLE"),
        "multistrategy_error": cache.get("multistrategy_error"),
        "scalping_evaluations": cache.get("scalping_evaluations", []),
        "scalping_status": cache.get("scalping_status", "UNAVAILABLE"),
        "scalping_error": cache.get("scalping_error"),
    }


def public_broker(cache: dict) -> dict:
    return {
        "connected": cache["connected"],
        "status": cache["manager_status"],
        "health": cache["health"],
        "market_status": cache["market_status"],
        "last_data_update": cache["last_data_update"],
        "last_error": cache["last_error"],
        "worker_alive": cache["worker_alive"],
    }


def desk_payload() -> dict:
    cache = get_dhan_manager().snapshot()
    return {"desk": desk_contract(cache), "market": market_contract(cache)}


def market_payload() -> dict:
    cache = get_dhan_manager().snapshot()
    return {
        "market": market_contract(cache),
        "candles": candle_contract(cache),
        "option_candles": {
            "CE": candle_contract(cache, "NIFTY_ATM_CE"),
            "PE": candle_contract(cache, "NIFTY_ATM_PE"),
        },
        "instruments": {
            "CE": cache.get("options", {}).get("NIFTY_ATM_CE"),
            "PE": cache.get("options", {}).get("NIFTY_ATM_PE"),
        },
        "watched_candles": {
            key.removeprefix("OPTION_"): candle_contract(cache, key)
            for key in cache.get("candles", {}) if key.startswith("OPTION_")
        },
    }


def options_payload() -> dict:
    cache = get_dhan_manager().snapshot()
    return {
        "connected": cache["connected"], "status": cache["manager_status"],
        "updated_at": cache["last_data_update"], "rows": cache["option_chain"],
    }


def paper_payload() -> dict:
    return paper_summary()
