"""Process-wide Dhan connection lifecycle for the single-user desktop app.

This is the only runtime service that may own a DhanLiveSession or perform
broker I/O. Streamlit pages consume defensive cache snapshots exclusively.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Callable

import pandas as pd

from src.dhan_live_broker import BrokerCredentials, ConnectionHealth, Instrument
from src.dhan_live_integration import DhanLiveSession
from src.dhan_option_data_collector import OptionMarketDataCollector
from src.live_data_provider import IST
from src.live_inference_engine import SafeBlock, TrueLiveInferenceEngine
from src.live_pipeline_monitor import LivePipelineMonitor
from src.paper_trading_manager import get_paper_manager
from src.put_shadow_inference import PutShadowInferenceEngine
from src.multistrategy_shadow_inference import MultiStrategyShadowEngine
from src.scalping_signal_engine import ScalpingSignalEngine
from src.dhan_rolling_acquisition_service import DhanRollingAcquisitionService
from src.exact_contract_batch_acquisition import ExactContractBatchAcquisition


EMPTY_HEALTH = {
    "status": "DISCONNECTED",
    "last_connection_time": None,
    "last_heartbeat": None,
    "latency_ms": None,
    "token_valid": False,
    "detail": "No active desktop broker session",
}


def _market_status(now: pd.Timestamp | None = None) -> str:
    current = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    current = (
        current.tz_localize(IST) if current.tzinfo is None
        else current.tz_convert(IST))
    if current.weekday() >= 5:
        return "CLOSED"
    clock = current.time()
    if clock < pd.Timestamp("09:15").time():
        return "PRE_OPEN"
    if clock >= pd.Timestamp("15:30").time():
        return "CLOSED"
    return "OPEN"


class DhanConnectionManager:
    """Own exactly one session, one worker and one shared in-memory cache."""

    def __init__(
        self,
        security_master: str | Path,
        *,
        heartbeat_seconds: int = 30,
        candle_lookback_days: int = 5,
        session_factory: Callable[..., DhanLiveSession] = DhanLiveSession,
        production_model: str | Path = "production_model",
        decision_observations: str | Path = "data/runtime/decision_observations.parquet",
        inference_journal: str | Path = "logs/live_inference/predictions.jsonl",
        inference_health: str | Path = "logs/live_inference/pipeline_health.json",
        option_collector: OptionMarketDataCollector | None = None,
    ):
        self.security_master = Path(security_master)
        self.heartbeat_seconds = max(5, int(heartbeat_seconds))
        self.candle_lookback_days = max(1, int(candle_lookback_days))
        self.session_factory = session_factory
        self.production_model = Path(production_model)
        self.decision_observations_path = Path(decision_observations)
        self.inference_journal = Path(inference_journal)
        self.inference_health = Path(inference_health)
        self.option_collector = option_collector or OptionMarketDataCollector()
        self._lock = threading.RLock()
        self._refresh_event = threading.Event()
        self._stop_event = threading.Event()
        self._session: DhanLiveSession | None = None
        self._worker: threading.Thread | None = None
        self._inference_engine: TrueLiveInferenceEngine | None = None
        self._put_inference_engine: PutShadowInferenceEngine | None = None
        self._multistrategy_engine: MultiStrategyShadowEngine | None = None
        self._scalping_engine: ScalpingSignalEngine | None = None
        self._decision_observations: pd.DataFrame | None = None
        self._watched_options: dict[str, Instrument] = {}
        self._cache = self._empty_cache()

    @staticmethod
    def _empty_cache() -> dict:
        return {
            "connected": False,
            "manager_status": "DISCONNECTED",
            "health": dict(EMPTY_HEALTH),
            "market_status": _market_status(),
            "quotes": {},
            "option_quotes": {},
            "indices": {},
            "options": {},
            "option_chain": [],
            "candles": {},
            "last_data_update": None,
            "last_candle_update": None,
            "last_error": None,
            "signal": None,
            "signal_status": "WAITING_FOR_MARKET_DATA",
            "signal_error": None,
            "put_signal": None,
            "put_signal_status": "WAITING_FOR_MARKET_DATA",
            "put_signal_error": None,
            "last_put_inference_timestamp": None,
            "multistrategy_signal": None,
            "multistrategy_status": "WAITING_FOR_MARKET_DATA",
            "multistrategy_error": None,
            "scalping_evaluations": [],
            "scalping_status": "WAITING_FOR_MARKET_DATA",
            "scalping_error": None,
            "last_multistrategy_timestamp": None,
            "last_inference_timestamp": None,
            "collector_status": "WAITING",
            "collector_last_write": None,
            "collector_error": None,
            "consecutive_failures": 0,
            "worker_alive": False,
            "worker_name": None,
        }

    def snapshot(self) -> dict:
        """Return a defensive page-safe copy with no credentials or client."""
        with self._lock:
            result = {
                key: value.copy() if isinstance(value, dict) else value
                for key, value in self._cache.items()
            }
            result["health"] = dict(self._cache["health"])
            result["indices"] = deepcopy(self._cache["indices"])
            result["options"] = deepcopy(self._cache["options"])
            result["quotes"] = dict(self._cache["quotes"])
            result["option_quotes"] = dict(self._cache["option_quotes"])
            result["option_chain"] = deepcopy(self._cache["option_chain"])
            result["candles"] = {
                key: value.copy(deep=True)
                for key, value in self._cache["candles"].items()
            }
            result["worker_alive"] = bool(
                self._worker is not None and self._worker.is_alive())
            return result

    def connect(self, client_id: str, access_token: str) -> dict:
        """Authenticate once and retain the session for the process lifetime."""
        with self._lock:
            if self._session is not None:
                return self.snapshot()
        credentials = BrokerCredentials(client_id.strip(), access_token.strip())
        candidate = self.session_factory(credentials, self.security_master)
        health = candidate.connect()
        if health["status"] != "CONNECTED":
            with self._lock:
                self._cache["health"] = dict(health)
                self._cache["manager_status"] = health["status"]
                self._cache["last_error"] = health["detail"]
            return self.snapshot()
        with self._lock:
            # A concurrent connect cannot replace the first successful session.
            if self._session is None:
                self._session = candidate
                self._stop_event.clear()
                self._refresh_event.clear()
                self._publish_session(candidate, manager_status="CONNECTED")
                self._start_worker_locked()
        return self.snapshot()

    def disconnect(self) -> dict:
        """Stop the sole worker, remove credentials and clear market cache."""
        with self._lock:
            self._stop_event.set()
            self._refresh_event.set()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)
        with self._lock:
            self._session = None
            self._worker = None
            self._watched_options.clear()
            self._inference_engine = None
            self._put_inference_engine = None
            self._multistrategy_engine = None
            self._scalping_engine = None
            self._decision_observations = None
            self._cache = self._empty_cache()
        return self.snapshot()

    def request_refresh(self) -> None:
        """Wake the worker; never perform Dhan I/O in the caller/page thread."""
        with self._lock:
            if self._session is not None:
                self._refresh_event.set()

    def acquire_rolling_options(self, start, end, *, expiry_code: int = 0) -> dict:
        """Use the existing memory-only authenticated session for read-only history."""
        with self._lock:
            session = self._session
        if session is None or not session.client.health.token_valid:
            raise RuntimeError("Dhan broker session is not authenticated")
        return DhanRollingAcquisitionService(session.client).acquire(
            start, end, expiry_code=expiry_code)

    def acquire_exact_contract_batch(self, dates: list[str]) -> dict:
        """Download exact historical CE/PE contracts; never used by live inference."""
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("Broker is not connected")
        return ExactContractBatchAcquisition(session.client).acquire(dates)

    def watch_option(self, security_id: str) -> dict:
        """Queue a cached chain contract for worker-owned candle updates."""
        with self._lock:
            row = next((item for item in self._cache["option_chain"]
                        if str(item.get("security_id")) == str(security_id)), None)
            if row is None:
                raise ValueError("Selected option is not available in the live chain")
            instrument = Instrument(
                market=str(row.get("market", "NIFTY")),
                security_id=str(row["security_id"]),
                exchange_segment=str(row["exchange_segment"]),
                instrument=str(row["instrument"]),
                display_name=str(row.get("display_name") or row.get("instrument_name") or security_id),
                expiry=row.get("expiry"), strike=row.get("strike"),
                option_type=row.get("option_type"),
            )
            self._watched_options[str(security_id)] = instrument
            # Bound the desktop watchlist so refresh cost cannot grow forever.
            while len(self._watched_options) > 6:
                self._watched_options.pop(next(iter(self._watched_options)))
            self._refresh_event.set()
            return instrument.public_dict()

    def cached_inference_histories(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return cached NIFTY and ATM-CE histories for inference."""
        snapshot = self.snapshot()
        index = snapshot["candles"].get("NIFTY", pd.DataFrame())
        option = snapshot["candles"].get("NIFTY_ATM_CE", pd.DataFrame())
        if index.empty or option.empty:
            raise RuntimeError("Shared candle cache is not ready")
        return index, option

    def _start_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="dhan-connection-worker",
            daemon=True,
        )
        self._worker.start()
        self._cache["worker_name"] = self._worker.name
        self._cache["worker_alive"] = True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._update_once()
            except Exception as exc:
                self._publish_failure(exc)
            self._refresh_event.wait(self.heartbeat_seconds)
            self._refresh_event.clear()

    def _update_once(self) -> None:
        with self._lock:
            session = self._session
        if session is None:
            return
        health = session.client.test_connection()
        if health.status != "CONNECTED":
            raise ConnectionError(health.detail)
        session.refresh_instruments()
        with self._lock:
            if session is not self._session:
                return
            self._publish_session(session, manager_status="CONNECTED")
        errors = []
        for name, loader in (
            ("option quotes", self._load_option_quotes),
            ("option chain", self._load_option_chain),
            ("candles", self._load_candles),
        ):
            try:
                value = loader(session)
                with self._lock:
                    if session is not self._session:
                        return
                    if name == "option quotes" and value:
                        self._cache["option_quotes"] = value
                    elif name == "option chain" and value:
                        self._cache["option_chain"] = value
                    elif name == "candles" and value:
                        self._cache["candles"] = value
                        self._cache["last_candle_update"] = datetime.now(
                            timezone.utc).isoformat()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        with self._lock:
            if session is not self._session:
                return
            self._cache["last_data_update"] = datetime.now(
                timezone.utc).isoformat()
            if errors:
                self._cache["manager_status"] = "DEGRADED"
                self._cache["last_error"] = " | ".join(errors)
                self._cache["consecutive_failures"] = int(
                    self._cache["consecutive_failures"]) + 1
            else:
                self._cache["manager_status"] = "CONNECTED"
                self._cache["last_error"] = None
                self._cache["consecutive_failures"] = 0
        self._update_live_signal()
        self._update_put_shadow_signal()
        self._update_multistrategy_signal()
        self._update_corrected_scalping_signal()
        self._collect_option_data()
        try:
            with self._lock:
                option_candles = self._cache["candles"].get(
                    "NIFTY_ATM_CE", pd.DataFrame()).copy(deep=True)
                option_chain = deepcopy(self._cache["option_chain"])
                quote_timestamp = self._cache["last_data_update"]
            get_paper_manager().update_live(
                option_candles, option_chain=option_chain,
                timestamp=quote_timestamp)
        except Exception as exc:
            with self._lock:
                self._cache["last_error"] = f"paper position update: {exc}"

    def _collect_option_data(self) -> None:
        """Persist from the cache only; collection never owns a Dhan client."""
        try:
            result = self.option_collector.collect_snapshot(self.snapshot())
            with self._lock:
                self._cache["collector_status"] = result["status"]
                self._cache["collector_last_write"] = result.get("last_write")
                self._cache["collector_error"] = None
        except Exception as exc:
            # Storage failure must not tear down an authenticated market feed.
            with self._lock:
                self._cache["collector_status"] = "ERROR"
                self._cache["collector_error"] = str(exc)

    def _update_live_signal(self) -> None:
        """Run native inference once per completed shared candle timestamp."""
        with self._lock:
            index = self._cache["candles"].get("NIFTY", pd.DataFrame()).copy(deep=True)
            option = self._cache["candles"].get("NIFTY_ATM_CE", pd.DataFrame()).copy(deep=True)
            previous = self._cache["last_inference_timestamp"]
        if index.empty or option.empty:
            with self._lock:
                self._cache["signal_status"] = "WAITING_FOR_MARKET_DATA"
            return
        timestamp = str(pd.to_datetime(index["timestamp"].iloc[-1]))
        if timestamp == previous:
            return
        try:
            engine, observations = self._inference_dependencies()
            result = engine.infer_histories(index, option, observations)
            payload = asdict(result)
            payload["action"] = "BUY CE" if result.decision == "TRADE" else "NO TRADE"
            payload["instrument_scope"] = "ATM_CALL_ONLY"
            with self._lock:
                self._cache["signal"] = payload
                self._cache["signal_status"] = "READY"
                self._cache["signal_error"] = None
                self._cache["last_inference_timestamp"] = timestamp
        except (SafeBlock, Exception) as exc:
            with self._lock:
                self._cache["signal_status"] = "SAFE_BLOCKED"
                self._cache["signal_error"] = str(exc)
                self._cache["last_inference_timestamp"] = timestamp

    def _inference_dependencies(
        self,
    ) -> tuple[TrueLiveInferenceEngine, pd.DataFrame]:
        with self._lock:
            engine = self._inference_engine
            observations = self._decision_observations
        if engine is None:
            engine = TrueLiveInferenceEngine(
                self.production_model,
                self.inference_journal,
                LivePipelineMonitor(self.inference_health),
            )
        if observations is None:
            if not self.decision_observations_path.exists():
                raise SafeBlock("Frozen causal Decision Engine observations are unavailable")
            observations = pd.read_parquet(self.decision_observations_path)
        with self._lock:
            if self._inference_engine is None:
                self._inference_engine = engine
            if self._decision_observations is None:
                self._decision_observations = observations
            return self._inference_engine, self._decision_observations

    def _update_put_shadow_signal(self) -> None:
        """Evaluate the validated PUT candidate for paper observation only."""
        with self._lock:
            index = self._cache["candles"].get("NIFTY", pd.DataFrame()).copy(deep=True)
            option = self._cache["candles"].get("NIFTY_ATM_PE", pd.DataFrame()).copy(deep=True)
            previous = self._cache["last_put_inference_timestamp"]
        if index.empty or option.empty:
            with self._lock:
                self._cache["put_signal_status"] = "WAITING_FOR_MARKET_DATA"
            return
        timestamp = str(pd.to_datetime(index["timestamp"].iloc[-1]))
        if timestamp == previous:
            return
        try:
            _, observations = self._inference_dependencies()
            with self._lock:
                engine = self._put_inference_engine
            if engine is None:
                engine = PutShadowInferenceEngine("models/stage2_put_candidate_v1")
                with self._lock:
                    if self._put_inference_engine is None:
                        self._put_inference_engine = engine
                    engine = self._put_inference_engine
            payload = engine.infer_histories(index, option, observations)
            with self._lock:
                self._cache["put_signal"] = payload
                self._cache["put_signal_status"] = "PAPER_READY"
                self._cache["put_signal_error"] = None
                self._cache["last_put_inference_timestamp"] = timestamp
        except Exception as exc:
            with self._lock:
                self._cache["put_signal_status"] = "SAFE_BLOCKED"
                self._cache["put_signal_error"] = str(exc)
                self._cache["last_put_inference_timestamp"] = timestamp

    def _update_multistrategy_signal(self) -> None:
        """Evaluate retrained strategy candidates in paper-only shadow mode."""
        with self._lock:
            index = self._cache["candles"].get("NIFTY", pd.DataFrame()).copy(deep=True)
            call = self._cache["candles"].get("NIFTY_ATM_CE", pd.DataFrame()).copy(deep=True)
            put = self._cache["candles"].get("NIFTY_ATM_PE", pd.DataFrame()).copy(deep=True)
            previous = self._cache["last_multistrategy_timestamp"]
        if index.empty or call.empty or put.empty:
            return
        timestamp = str(pd.to_datetime(index.timestamp.iloc[-1]))
        if timestamp == previous:
            return
        try:
            engine = self._multistrategy_engine or MultiStrategyShadowEngine()
            payload = engine.infer(index, call, put)
            with self._lock:
                self._multistrategy_engine = engine
                self._cache["multistrategy_signal"] = payload
                self._cache["multistrategy_status"] = payload["status"]
                self._cache["multistrategy_error"] = None
                self._cache["last_multistrategy_timestamp"] = timestamp
        except Exception as exc:
            with self._lock:
                self._cache["multistrategy_status"] = "SAFE_BLOCKED"
                self._cache["multistrategy_error"] = str(exc)
                self._cache["last_multistrategy_timestamp"] = timestamp

    def _update_corrected_scalping_signal(self) -> None:
        """Evaluate the contract-corrected CE/PE paper path; never sends orders."""
        with self._lock:
            index = self._cache["candles"].get("NIFTY", pd.DataFrame()).copy(deep=True)
            sides = {
                "CE": self._cache["candles"].get("NIFTY_ATM_CE", pd.DataFrame()).copy(deep=True),
                "PE": self._cache["candles"].get("NIFTY_ATM_PE", pd.DataFrame()).copy(deep=True),
            }
            instruments = deepcopy(self._cache["options"])
        if index.empty or any(frame.empty for frame in sides.values()):
            return
        option_rows = []
        for side, frame in sides.items():
            contract = instruments.get(f"NIFTY_ATM_{side}") or {}
            enriched = frame.copy()
            enriched["strike"] = contract.get("strike")
            enriched["expiry"] = contract.get("expiry")
            enriched["option_type"] = side
            enriched["security_id"] = contract.get("security_id")
            enriched["underlying"] = "NIFTY"
            option_rows.append(enriched)
        try:
            engine = self._scalping_engine or ScalpingSignalEngine(
                {}, journal_path="logs/scalping_funnel/evaluations.jsonl")
            evaluations = engine.evaluate(index, pd.concat(option_rows, ignore_index=True))
            reason = evaluations[0].get("primary_rejection_reason") if evaluations else "NO_EVALUATION"
            with self._lock:
                self._scalping_engine = engine
                self._cache["scalping_evaluations"] = evaluations
                self._cache["scalping_status"] = (
                    "PAPER_SIGNAL" if any(item["final_decision"].startswith("BUY") for item in evaluations)
                    else reason or "NO_TRADE")
                self._cache["scalping_error"] = None
        except Exception as exc:
            with self._lock:
                self._cache["scalping_status"] = "SAFE_BLOCKED"
                self._cache["scalping_error"] = str(exc)
    @staticmethod
    def _load_option_quotes(session: DhanLiveSession) -> dict[str, float]:
        if not session.options:
            return {}
        payload = session.client.quote(list(session.options.values()))
        output = {}
        for name, instrument in session.options.items():
            output[name] = session._ltp(payload, instrument)
        return output

    @staticmethod
    def _load_option_chain(session: DhanLiveSession) -> list[dict]:
        spot = session.last_quotes.get("NIFTY")
        if spot is None:
            return []
        instruments = session.resolver.resolve_option_chain(
            "NIFTY", spot, strikes_each_side=10)
        if not instruments:
            return []
        payload = session.client.quote(instruments)
        rows = []
        for instrument in instruments:
            segment = payload.get(instrument.exchange_segment, {})
            quote = segment.get(instrument.security_id)
            if quote is None:
                quote = segment.get(int(instrument.security_id), {})
            quote = quote or {}
            depth = quote.get("depth", {}) or {}
            buy = depth.get("buy", []) or []
            sell = depth.get("sell", []) or []
            bid = buy[0].get("price") if buy else None
            ask = sell[0].get("price") if sell else None
            row = instrument.public_dict()
            row.update({
                "ltp": quote.get("last_price"),
                "oi": quote.get("oi"),
                "volume": quote.get("volume"),
                "bid": bid,
                "ask": ask,
                "spread": (
                    float(ask) - float(bid)
                    if bid is not None and ask is not None else None),
            })
            rows.append(row)
        return rows

    def _load_candles(self, session: DhanLiveSession) -> dict[str, pd.DataFrame]:
        if "NIFTY" not in session.indices:
            return {}
        now = pd.Timestamp.now(tz=IST)
        start = now - pd.Timedelta(days=self.candle_lookback_days)
        from_date = start.strftime("%Y-%m-%d 09:15:00")
        to_date = now.strftime("%Y-%m-%d %H:%M:%S")
        instruments = {"NIFTY": session.indices["NIFTY"]}
        for name in ("NIFTY_ATM_CE", "NIFTY_ATM_PE"):
            if name in session.options:
                instruments[name] = session.options[name]
        with self._lock:
            for security_id, instrument in self._watched_options.items():
                instruments[f"OPTION_{security_id}"] = instrument
        output = {}
        for name, instrument in instruments.items():
            frame = session.client.completed_candles(
                instrument, from_date, to_date, now=now)
            if not frame.empty:
                output[name] = frame
        return output

    def _publish_session(
        self, session: DhanLiveSession, *, manager_status: str,
    ) -> None:
        health = session.client.health.public_dict()
        self._cache.update({
            "connected": health["status"] == "CONNECTED",
            "manager_status": manager_status,
            "health": health,
            "market_status": _market_status(),
            "quotes": dict(session.last_quotes),
            "indices": {
                key: value.public_dict()
                for key, value in session.indices.items()
            },
            "options": {
                key: value.public_dict()
                for key, value in session.options.items()
            },
        })

    def _publish_failure(self, error: Exception) -> None:
        with self._lock:
            failures = int(self._cache["consecutive_failures"]) + 1
            self._cache["consecutive_failures"] = failures
            self._cache["manager_status"] = "RECONNECTING"
            self._cache["market_status"] = _market_status()
            self._cache["last_error"] = str(error)
            # Retain the last known-good quotes/candles and authenticated
            # session. A transient feed failure must not force another login.
            if self._session is not None:
                health = self._session.client.health.public_dict()
                self._cache["health"] = health
                self._cache["connected"] = health.get("token_valid", False)


_MANAGER: DhanConnectionManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_dhan_manager() -> DhanConnectionManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = DhanConnectionManager(
                    "data/raw/dhan_security_master_latest.csv",
                    heartbeat_seconds=30,
                )
    return _MANAGER
