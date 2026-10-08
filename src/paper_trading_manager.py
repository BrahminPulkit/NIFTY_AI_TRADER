"""Process-wide orchestration around the existing broker-free paper engine."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import json
import math
import threading

import pandas as pd

from src.paper_trading_analytics import performance
from src.paper_trading_engine import OptionCandle, PaperTradingConfig, PaperTradingEngine


class PaperRequestError(RuntimeError):
    pass


def _load_config(path: str | Path = "config/paper_trading.json") -> PaperTradingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PaperTradingConfig)}
    return PaperTradingConfig(**{key: value for key, value in payload.items() if key in allowed})


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso")) if len(frame) else []


def _json_safe(values: dict) -> dict:
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in values.items()
    }


class PaperTradingManager:
    def __init__(self, root: str | Path = "logs/paper_trading", config: PaperTradingConfig | None = None):
        self.root = Path(root)
        self.config = config or _load_config()
        self._lock = threading.RLock()
        self._live = PaperTradingEngine(self.root / "live", self.config)
        self._replay: dict[str, PaperTradingEngine] = {}
        self._replay_last_candle: dict[str, str] = {}
        self._live_last_candle: str | None = (
            self._live.position.entry_timestamp if self._live.position else None)

    def _replay_engine(self, session_date: str) -> PaperTradingEngine:
        with self._lock:
            if session_date not in self._replay:
                self._replay[session_date] = PaperTradingEngine(
                    self.root / "replay" / session_date, self.config)
                if self._replay[session_date].position:
                    self._replay_last_candle[session_date] = (
                        self._replay[session_date].position.entry_timestamp)
            return self._replay[session_date]

    @staticmethod
    def candle(row: pd.Series | dict) -> OptionCandle:
        value = row if isinstance(row, dict) else row.to_dict()
        return OptionCandle(
            str(value["timestamp"]), float(value.get("option_open", value.get("open"))),
            float(value.get("option_high", value.get("high"))),
            float(value.get("option_low", value.get("low"))),
            float(value.get("option_close", value.get("close"))),
        )

    def account(self, mode: str = "live", session_date: str | None = None) -> dict:
        engine = self._engine(mode, session_date)
        trades = engine.trades()
        return {
            "mode": mode, "session_date": session_date,
            "capital": engine.capital(), "initial_capital": self.config.initial_capital,
            "position": asdict(engine.position) if engine.position else None,
            "statistics": _json_safe(performance(
                trades, self.config.initial_capital)),
            "trades": _records(trades.tail(100).iloc[::-1]),
            "rules": {
                "quantity": self.config.quantity,
                "stop_loss_pct": self.config.stop_loss_pct,
                "target_pct": self.config.target_pct,
                "maximum_holding_minutes": self.config.maximum_holding_minutes,
                "maximum_daily_trades": self.config.maximum_daily_trades,
                "daily_loss_limit": self.config.daily_loss_limit,
                "daily_profit_target": self.config.daily_profit_target,
                "broker_orders_enabled": False,
            },
        }

    def review(self, prediction: dict, candle: OptionCandle) -> dict:
        candle.validate()
        premium = candle.close
        eligible = (
            prediction.get("decision") == "TRADE"
            and float(prediction.get("probability") or 0)
            >= float(prediction.get("required_probability", .90))
        )
        return {
            "eligible": eligible,
            "reason": "READY_FOR_PAPER_ENTRY" if eligible else str(prediction.get("reason", "SIGNAL_NOT_APPROVED")),
            "timestamp": str(prediction.get("timestamp")),
            "strategy": str(prediction.get("strategy", "UNKNOWN")),
            "probability": float(prediction.get("probability") or 0),
            "entry_premium": premium, "quantity": self.config.quantity,
            "stop_loss": premium * (1 - self.config.stop_loss_pct),
            "target": premium * (1 + self.config.target_pct),
            "capital_required": premium * self.config.quantity,
            "maximum_loss": premium * self.config.stop_loss_pct * self.config.quantity,
            "broker_orders_enabled": False,
        }

    def review_manual(self, contract: dict, timestamp,
                      quantity: int | None = None) -> dict:
        candle = self.quote_candle(contract, timestamp, side="entry")
        selected_quantity = self.config.quantity if quantity is None else int(quantity)
        if selected_quantity <= 0 or selected_quantity > 10_000:
            raise PaperRequestError("Virtual quantity must be between 1 and 10000")
        premium = candle.close
        return {
            "eligible": self._live.position is None,
            "reason": "READY_FOR_MANUAL_PAPER_ENTRY" if self._live.position is None
            else "OVERLAPPING_POSITION",
            "timestamp": candle.timestamp, "strategy": "MANUAL_OPTION",
            "probability": 0.0, "entry_premium": premium,
            "quantity": selected_quantity,
            "stop_loss": premium * (1 - self.config.stop_loss_pct),
            "target": premium * (1 + self.config.target_pct),
            "capital_required": premium * selected_quantity,
            "maximum_loss": premium * self.config.stop_loss_pct * selected_quantity,
            "broker_orders_enabled": False,
            "security_id": str(contract["security_id"]),
            "option_type": str(contract["option_type"]),
            "strike": float(contract["strike"]),
            "expiry": str(contract.get("expiry") or ""),
        }

    def open_manual(self, contract: dict, timestamp,
                    quantity: int | None = None) -> dict:
        candle = self.quote_candle(contract, timestamp, side="entry")
        result = self._live.open_manual(contract, candle, quantity)
        if result.get("status") == "OPENED":
            self._live_last_candle = candle.timestamp
        return {"result": result, "account": self.account("live")}

    @staticmethod
    def quote_candle(contract: dict, timestamp, *, side: str) -> OptionCandle:
        preferred = "ask" if side == "entry" else "bid"
        raw_price = contract.get(preferred)
        if raw_price is None:
            raw_price = contract.get("ltp")
        try:
            price = float(raw_price)
        except (TypeError, ValueError) as exc:
            raise PaperRequestError(f"Live {preferred} or LTP is unavailable") from exc
        if price <= 0:
            raise PaperRequestError(f"Live {preferred} or LTP must be positive")
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return OptionCandle(stamp.isoformat(), price, price, price, price)

    def open(self, mode: str, prediction: dict, candle: OptionCandle,
             session_date: str | None = None) -> dict:
        engine = self._engine(mode, session_date)
        result = engine.consume_prediction(prediction, candle, {"status": "HEALTHY"})
        if result.get("status") not in {"OPENED", "SKIP", "SAFE_BLOCK"}:
            raise PaperRequestError("Unexpected paper engine response")
        if mode == "live" and result.get("status") == "OPENED":
            self._live_last_candle = str(candle.timestamp)
        if mode == "replay" and session_date and result.get("status") == "OPENED":
            self._replay_last_candle[session_date] = str(candle.timestamp)
        return {"result": result, "account": self.account(mode, session_date)}

    def update_live(self, frame: pd.DataFrame, *, option_chain: list[dict] | None = None,
                    timestamp=None) -> dict:
        if self._live.position is None:
            return {"status": "NO_POSITION"}
        if self._live.position.security_id:
            contract = next((row for row in (option_chain or [])
                             if str(row.get("security_id")) == self._live.position.security_id), None)
            if contract is None or timestamp is None:
                return {"status": "WAITING_FOR_SELECTED_CONTRACT"}
            candle = self.quote_candle(contract, timestamp, side="exit")
        else:
            if frame.empty:
                return {"status": "WAITING_FOR_OPTION_CANDLE"}
            candle = self.candle(frame.iloc[-1])
        if str(candle.timestamp) == self._live_last_candle:
            return {"status": "UNCHANGED"}
        result = self._live.on_candle(candle)
        self._live_last_candle = str(candle.timestamp)
        return result

    def update_replay(self, session_date: str, candle: OptionCandle) -> dict:
        engine = self._replay_engine(session_date)
        if engine.position is None:
            return {"status": "NO_POSITION"}
        previous = self._replay_last_candle.get(
            session_date, engine.position.entry_timestamp)
        if pd.Timestamp(candle.timestamp) <= pd.Timestamp(previous):
            return {"status": "UNCHANGED"}
        result = engine.on_candle(candle)
        self._replay_last_candle[session_date] = str(candle.timestamp)
        return result

    def manual_close(self, mode: str, candle: OptionCandle,
                     session_date: str | None = None) -> dict:
        engine = self._engine(mode, session_date)
        result = engine.manual_close(candle)
        return {"result": result, "account": self.account(mode, session_date)}

    def _engine(self, mode: str, session_date: str | None) -> PaperTradingEngine:
        if mode == "live":
            return self._live
        if mode == "replay" and session_date:
            return self._replay_engine(session_date)
        raise PaperRequestError("Mode must be live, or replay with session_date")


_PAPER_MANAGER: PaperTradingManager | None = None
_PAPER_LOCK = threading.Lock()


def get_paper_manager() -> PaperTradingManager:
    global _PAPER_MANAGER
    if _PAPER_MANAGER is None:
        with _PAPER_LOCK:
            if _PAPER_MANAGER is None:
                _PAPER_MANAGER = PaperTradingManager()
    return _PAPER_MANAGER
