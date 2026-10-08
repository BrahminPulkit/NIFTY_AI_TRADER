"""Deterministic, broker-free paper position lifecycle for Step 25."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import threading

import pandas as pd


IST = "Asia/Kolkata"


class PaperSafetyBlock(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperTradingConfig:
    initial_capital: float = 1_000_000.0
    quantity: int = 1
    stop_loss_pct: float = 0.20
    target_pct: float = 0.05
    maximum_holding_minutes: int = 15
    maximum_daily_trades: int = 5
    maximum_consecutive_losses: int = 3
    daily_loss_limit: float = 10_000.0
    daily_profit_target: float = 20_000.0
    one_active_position_per_strategy: bool = True
    no_overlapping_trades: bool = True
    end_of_session_hour: int = 15
    end_of_session_minute: int = 29
    decision_engine_exit: bool = True
    brokerage_per_order: float = 20.0
    stt_sell_rate: float = 0.0015
    exchange_rate_each_side: float = 0.0003553
    sebi_rate_each_side: float = 0.000001
    stamp_duty_buy_rate: float = 0.00003
    gst_rate: float = 0.18
    slippage_bps_each_side: float = 2.0

    def validate(self) -> None:
        if self.initial_capital <= 0 or self.quantity <= 0:
            raise ValueError("Capital and virtual quantity must be positive")
        if not 0 < self.target_pct < 1 or not 0 < self.stop_loss_pct < 1:
            raise ValueError("Stop and target must be fractions in (0, 1)")
        if self.maximum_holding_minutes <= 0:
            raise ValueError("Maximum holding time must be positive")


@dataclass
class PaperPosition:
    trade_id: str
    entry_timestamp: str
    strategy: str
    regime: str
    probability: float
    entry_premium: float
    quantity: int
    stop_loss: float
    target: float
    current_premium: float
    current_pnl: float = 0.0
    maximum_favourable_excursion: float = 0.0
    maximum_adverse_excursion: float = 0.0
    entry_mode: str = "AI_SIGNAL"
    market: str = "NIFTY_ATM_CE"
    security_id: str | None = None
    option_type: str = "CE"
    strike: float | None = None
    expiry: str | None = None


@dataclass(frozen=True)
class OptionCandle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float

    def validate(self) -> None:
        values = [self.open, self.high, self.low, self.close]
        if any(value <= 0 for value in values):
            raise PaperSafetyBlock("Option candle prices must be positive")
        if self.high < self.low or not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise PaperSafetyBlock("Invalid option OHLC relationship")


def trading_costs(entry: float, exit_: float, quantity: int,
                  config: PaperTradingConfig) -> dict:
    buy, sell = entry * quantity, exit_ * quantity
    brokerage = 2 * config.brokerage_per_order
    exchange = (buy + sell) * config.exchange_rate_each_side
    sebi = (buy + sell) * config.sebi_rate_each_side
    stamp = buy * config.stamp_duty_buy_rate
    stt = sell * config.stt_sell_rate
    gst = (brokerage + exchange + sebi) * config.gst_rate
    slippage = (buy + sell) * config.slippage_bps_each_side / 10_000
    total = brokerage + exchange + sebi + stamp + stt + gst + slippage
    return {
        "brokerage": brokerage, "stt": stt, "exchange_charges": exchange,
        "sebi_charges": sebi, "stamp_duty": stamp, "gst": gst,
        "slippage": slippage, "total_costs": total,
    }


class PaperTradingEngine:
    """One-position paper engine. It contains no broker or order API."""

    def __init__(self, root: str | Path, config: PaperTradingConfig):
        config.validate()
        self.root, self.config = Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "paper_events.jsonl"
        self.csv_path = self.root / "paper_trade_log.csv"
        self.parquet_path = self.root / "paper_trade_log.parquet"
        self.position_path = self.root / "open_position.json"
        self.equity_path = self.root / "equity_curve.csv"
        self._lock = threading.RLock()
        self.position = self._load_position()

    def _load_position(self) -> PaperPosition | None:
        if not self.position_path.exists():
            return None
        return PaperPosition(**json.loads(self.position_path.read_text(encoding="utf-8")))

    def _save_position(self) -> None:
        if self.position is None:
            self.position_path.unlink(missing_ok=True)
            return
        temporary = self.position_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self.position), indent=2), encoding="utf-8")
        temporary.replace(self.position_path)

    def _event(self, event: str, timestamp, reason: str, payload: dict | None = None) -> None:
        if self.events_path.exists():
            with self.events_path.open(encoding="utf-8") as existing:
                sequence = sum(1 for _ in existing)
        else:
            sequence = 0
        identity = f"{sequence}|{event}|{pd.Timestamp(timestamp).isoformat()}|{reason}"
        record = {
            "event_id": hashlib.sha256(identity.encode()).hexdigest()[:24], "event": event,
            "timestamp": pd.Timestamp(timestamp).isoformat(), "reason": reason,
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def trades(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()

    def capital(self) -> float:
        trades = self.trades()
        return self.config.initial_capital + (
            float(trades.net_pnl.sum()) if len(trades) else 0.0)

    def _risk_status(self, timestamp) -> tuple[bool, str]:
        trades = self.trades()
        if trades.empty:
            return True, "PASS"
        dates = pd.to_datetime(trades.exit_timestamp).dt.date
        today = pd.Timestamp(timestamp).date()
        daily = trades.loc[dates.eq(today)]
        if len(daily) >= self.config.maximum_daily_trades:
            return False, "MAXIMUM_DAILY_TRADES"
        pnl = float(daily.net_pnl.sum()) if len(daily) else 0.0
        if pnl <= -self.config.daily_loss_limit:
            return False, "DAILY_LOSS_LIMIT"
        if pnl >= self.config.daily_profit_target:
            return False, "DAILY_PROFIT_TARGET"
        losses = 0
        for value in reversed(trades.net_pnl.tolist()):
            if value < 0: losses += 1
            else: break
        if losses >= self.config.maximum_consecutive_losses:
            return False, "MAXIMUM_CONSECUTIVE_LOSSES"
        if self.capital() <= 0:
            return False, "CAPITAL_UNAVAILABLE"
        return True, "PASS"

    def consume_prediction(self, prediction: dict, candle: OptionCandle | None,
                           pipeline_health: dict) -> dict:
        """Open only from a Step-24 TRADE and an exact timestamp candle."""
        timestamp = pd.Timestamp(prediction["timestamp"])
        if prediction.get("decision") != "TRADE":
            reason = str(prediction.get("reason", "PREDICTION_SKIP"))
            self._event("SKIP", timestamp, reason, prediction)
            if self.position and self.config.decision_engine_exit:
                if candle is None:
                    self._event("EXIT_BLOCKED", timestamp, "DECISION_EXIT_REQUIRES_CANDLE")
                else:
                    return self.close(candle.close, timestamp, "DECISION_ENGINE_EXIT")
            return {"status": "SKIP", "reason": reason}
        required_probability = float(prediction.get("required_probability", .90))
        if float(prediction.get("probability", 0)) < required_probability:
            reason = f"PROBABILITY_BELOW_{required_probability:.2f}"
            self._event("SKIP", timestamp, reason, prediction)
            return {"status": "SKIP", "reason": reason}
        if pipeline_health.get("status") not in {"READY", "HEALTHY"}:
            self._event("SKIP", timestamp, "PIPELINE_NOT_READY", pipeline_health)
            return {"status": "SKIP", "reason": "PIPELINE_NOT_READY"}
        if candle is None:
            self._event("SKIP", timestamp, "MISSING_OPTION_CANDLE")
            return {"status": "SAFE_BLOCK", "reason": "MISSING_OPTION_CANDLE"}
        candle.validate()
        if pd.Timestamp(candle.timestamp) != timestamp:
            self._event("SKIP", timestamp, "CANDLE_TIMESTAMP_MISMATCH")
            return {"status": "SAFE_BLOCK", "reason": "CANDLE_TIMESTAMP_MISMATCH"}
        with self._lock:
            if self.position is not None:
                self._event("SKIP", timestamp, "OVERLAPPING_POSITION")
                return {"status": "SKIP", "reason": "OVERLAPPING_POSITION"}
            passed, reason = self._risk_status(timestamp)
            if not passed:
                self._event("SKIP", timestamp, reason)
                return {"status": "SKIP", "reason": reason}
            premium = float(candle.close)
            strategy = str(prediction.get("strategy", "UNKNOWN"))
            identity = (
                f"{timestamp.isoformat()}|{strategy}|{prediction['probability']}|"
                f"{premium}|{self.config.quantity}")
            contract = prediction.get("selected_option") or {}
            option_type = str(contract.get("option_type") or "CE").upper()
            if option_type not in {"CE", "PE"}:
                self._event("SKIP", timestamp, "INVALID_OPTION_TYPE", contract)
                return {"status": "SAFE_BLOCK", "reason": "INVALID_OPTION_TYPE"}
            self.position = PaperPosition(
                trade_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                entry_timestamp=timestamp.isoformat(),
                strategy=strategy, regime=str(prediction.get("regime", "UNKNOWN")),
                probability=float(prediction["probability"]), entry_premium=premium,
                quantity=self.config.quantity,
                stop_loss=premium * (1 - self.config.stop_loss_pct),
                target=premium * (1 + self.config.target_pct),
                current_premium=premium,
                market=f"NIFTY_ATM_{option_type}",
                security_id=(str(contract["security_id"])
                             if contract.get("security_id") is not None else None),
                option_type=option_type,
                strike=(float(contract["strike"]) if contract.get("strike") is not None else None),
                expiry=(str(contract["expiry"]) if contract.get("expiry") is not None else None),
            )
            if self.capital() < premium * self.config.quantity:
                self.position = None
                self._event("SKIP", timestamp, "INSUFFICIENT_CAPITAL")
                return {"status": "SKIP", "reason": "INSUFFICIENT_CAPITAL"}
            self._save_position()
            self._event("ENTRY", timestamp, "ALL_GATES_PASSED", asdict(self.position))
            return {"status": "OPENED", "trade_id": self.position.trade_id}

    def open_manual(self, contract: dict, candle: OptionCandle,
                    quantity: int | None = None) -> dict:
        """Open a virtual long option from an explicitly selected live contract."""
        candle.validate()
        timestamp = pd.Timestamp(candle.timestamp)
        option_type = str(contract.get("option_type", "")).upper()
        security_id = str(contract.get("security_id", "")).strip()
        if option_type not in {"CE", "PE"} or not security_id:
            raise PaperSafetyBlock("A valid live CE or PE contract is required")
        selected_quantity = self.config.quantity if quantity is None else int(quantity)
        if selected_quantity <= 0 or selected_quantity > 10_000:
            raise PaperSafetyBlock("Virtual quantity must be between 1 and 10000")
        with self._lock:
            if self.position is not None:
                return {"status": "SKIP", "reason": "OVERLAPPING_POSITION"}
            passed, reason = self._risk_status(timestamp)
            if not passed:
                return {"status": "SKIP", "reason": reason}
            premium = float(candle.close)
            if self.capital() < premium * selected_quantity:
                return {"status": "SKIP", "reason": "INSUFFICIENT_CAPITAL"}
            identity = (
                f"MANUAL|{timestamp.isoformat()}|{security_id}|"
                f"{premium}|{selected_quantity}"
            )
            self.position = PaperPosition(
                trade_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                entry_timestamp=timestamp.isoformat(), strategy="MANUAL_OPTION",
                regime="USER_DIRECTED", probability=0.0,
                entry_premium=premium, quantity=selected_quantity,
                stop_loss=premium * (1 - self.config.stop_loss_pct),
                target=premium * (1 + self.config.target_pct),
                current_premium=premium, entry_mode="MANUAL_OPTION",
                market=str(contract.get("market") or f"NIFTY_{option_type}"),
                security_id=security_id, option_type=option_type,
                strike=float(contract["strike"]), expiry=str(contract.get("expiry") or ""),
            )
            self._save_position()
            self._event("ENTRY", timestamp, "USER_MANUAL_ENTRY", asdict(self.position))
            return {"status": "OPENED", "trade_id": self.position.trade_id}

    def on_candle(self, candle: OptionCandle) -> dict:
        candle.validate()
        if self.position is None:
            return {"status": "NO_POSITION"}
        timestamp = pd.Timestamp(candle.timestamp)
        entry_time = pd.Timestamp(self.position.entry_timestamp)
        if timestamp <= entry_time:
            raise PaperSafetyBlock("Position monitoring requires a later candle")
        entry = self.position.entry_premium
        self.position.current_premium = candle.close
        self.position.current_pnl = (candle.close - entry) * self.position.quantity
        self.position.maximum_favourable_excursion = max(
            self.position.maximum_favourable_excursion, candle.high / entry - 1)
        self.position.maximum_adverse_excursion = max(
            self.position.maximum_adverse_excursion, 1 - candle.low / entry)
        # Gap-aware and conservative stop-first ambiguity.
        if candle.open <= self.position.stop_loss:
            return self.close(candle.open, timestamp, "GAP_THROUGH_STOP")
        if candle.open >= self.position.target:
            return self.close(candle.open, timestamp, "GAP_THROUGH_TARGET")
        if candle.low <= self.position.stop_loss:
            return self.close(self.position.stop_loss, timestamp, "STOP_LOSS_HIT")
        if candle.high >= self.position.target:
            return self.close(self.position.target, timestamp, "TARGET_HIT")
        if (timestamp.hour, timestamp.minute) >= (
            self.config.end_of_session_hour, self.config.end_of_session_minute):
            return self.close(candle.close, timestamp, "END_OF_SESSION")
        if timestamp - entry_time >= pd.Timedelta(minutes=self.config.maximum_holding_minutes):
            return self.close(candle.close, timestamp, "TIME_EXIT")
        self._save_position()
        return {"status": "OPEN", "current_pnl": self.position.current_pnl}

    def manual_close(self, candle: OptionCandle) -> dict:
        candle.validate()
        if self.position is None:
            return {"status": "NO_POSITION"}
        return self.close(candle.close, candle.timestamp, "MANUAL_CLOSE")

    def close(self, exit_premium: float, timestamp, reason: str) -> dict:
        with self._lock:
            if self.position is None:
                raise PaperSafetyBlock("No paper position is open")
            position = self.position
            exit_premium = float(exit_premium)
            if exit_premium <= 0:
                raise PaperSafetyBlock("Exit premium must be positive")
            costs = trading_costs(
                position.entry_premium, exit_premium, position.quantity, self.config)
            gross = (exit_premium - position.entry_premium) * position.quantity
            net = gross - costs["total_costs"]
            exit_time = pd.Timestamp(timestamp)
            entry_time = pd.Timestamp(position.entry_timestamp)
            row = {
                "trade_id": position.trade_id, "entry_timestamp": position.entry_timestamp,
                "exit_timestamp": exit_time.isoformat(), "strategy": position.strategy,
                "regime": position.regime, "probability": position.probability,
                "entry_premium": position.entry_premium, "exit_premium": exit_premium,
                "quantity": position.quantity, "stop_loss": position.stop_loss,
                "target": position.target, "gross_pnl": gross,
                **costs, "net_pnl": net,
                "return_pct": exit_premium / position.entry_premium - 1,
                "holding_time_minutes": (exit_time - entry_time).total_seconds() / 60,
                "maximum_favourable_excursion": position.maximum_favourable_excursion,
                "maximum_adverse_excursion": position.maximum_adverse_excursion,
                "entry_mode": position.entry_mode, "market": position.market,
                "security_id": position.security_id,
                "option_type": position.option_type, "strike": position.strike,
                "expiry": position.expiry,
                "exit_reason": reason,
            }
            frame = pd.DataFrame([row])
            existing = self.trades()
            snapshot = pd.concat([existing, frame], ignore_index=True, sort=False)
            csv_temporary = self.csv_path.with_suffix(".tmp.csv")
            snapshot.to_csv(csv_temporary, index=False)
            csv_temporary.replace(self.csv_path)
            temporary = self.parquet_path.with_suffix(".tmp.parquet")
            snapshot.to_parquet(temporary, index=False)
            temporary.replace(self.parquet_path)
            capital = self.config.initial_capital + float(snapshot.net_pnl.sum())
            pd.DataFrame([{
                "timestamp": exit_time.isoformat(), "trade_id": position.trade_id,
                "net_pnl": net, "equity": capital,
            }]).to_csv(self.equity_path, mode="a",
                       header=not self.equity_path.exists(), index=False)
            self.position = None
            self._save_position()
            self._event("EXIT", exit_time, reason, row)
            if capital < 0:
                raise AssertionError("Paper capital became negative")
            return {"status": "CLOSED", **row, "equity": capital}
