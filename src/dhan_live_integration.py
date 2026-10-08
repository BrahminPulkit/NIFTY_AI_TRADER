"""Session-scoped orchestration from Dhan candles into frozen live inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.dhan_live_broker import (
    BrokerCredentials, ConnectionHealth, DhanInstrumentResolver,
    DhanReadOnlyClient, Instrument,
)
from src.live_inference_engine import SafeBlock, TrueLiveInferenceEngine


@dataclass
class DhanLiveSession:
    """Memory-only broker session. It owns no persistence mechanism."""

    credentials: BrokerCredentials
    security_master: Path
    requester: object | None = None

    def __post_init__(self):
        kwargs = {"requester": self.requester} if self.requester is not None else {}
        self.client = DhanReadOnlyClient(self.credentials, **kwargs)
        self.resolver = DhanInstrumentResolver(self.security_master)
        self.indices: dict[str, Instrument] = {}
        self.options: dict[str, Instrument] = {}
        self.last_quotes: dict[str, float] = {}

    def connect(self) -> dict:
        health = self.client.test_connection()
        if health.status != "CONNECTED":
            return health.public_dict()
        try:
            self.indices = self.resolver.resolve_indices()
            self.refresh_instruments()
        except Exception:
            self.client.health = ConnectionHealth(
                "API ERROR", health.last_connection_time,
                pd.Timestamp.now(tz="UTC").isoformat(), health.latency_ms,
                health.token_valid,
                "Authenticated, but live quote or instrument discovery failed")
        return self.client.health.public_dict()

    @staticmethod
    def _ltp(quotes: dict, instrument: Instrument) -> float:
        segment = quotes.get(instrument.exchange_segment, {})
        item = segment.get(instrument.security_id)
        if item is None:
            item = segment.get(int(instrument.security_id))
        if not item or "last_price" not in item:
            raise SafeBlock(f"No genuine Dhan quote for {instrument.market}")
        price = float(item["last_price"])
        if price <= 0:
            raise SafeBlock(f"Invalid genuine Dhan quote for {instrument.market}")
        return price

    def refresh_instruments(self) -> dict[str, Instrument]:
        if self.client.health.status != "CONNECTED":
            health = self.client.test_connection()
            if health.status != "CONNECTED":
                raise SafeBlock(f"Dhan reconnect failed: {health.status}")
        quotes = self.client.quote(list(self.indices.values()))
        prices = {name: self._ltp(quotes, instrument)
                  for name, instrument in self.indices.items()}
        options: dict[str, Instrument] = {}
        for name in ("NIFTY", "BANKNIFTY", "SENSEX"):
            ce, pe = self.resolver.resolve_atm_options(name, prices[name])
            options[ce.market] = ce
            options[pe.market] = pe
        self.options, self.last_quotes = options, prices
        return options

    def public_state(self) -> dict:
        return {
            "health": self.client.health.public_dict(),
            "masked_token": self.credentials.masked_token,
            "indices": {key: value.public_dict() for key, value in self.indices.items()},
            "options": {key: value.public_dict() for key, value in self.options.items()},
            "last_quotes": self.last_quotes,
        }

    def infer_latest(self, engine: TrueLiveInferenceEngine,
                     decision_observations: pd.DataFrame,
                     *, lookback_days: int = 5, now=None):
        """Fetch completed NIFTY/ATM CE candles and invoke frozen inference."""
        if "NIFTY" not in self.indices or "NIFTY_ATM_CE" not in self.options:
            raise SafeBlock("NIFTY instruments are not resolved")
        current = pd.Timestamp.now(tz="Asia/Kolkata") if now is None else pd.Timestamp(now)
        current = (current.tz_localize("Asia/Kolkata") if current.tzinfo is None
                   else current.tz_convert("Asia/Kolkata"))
        start = current - pd.Timedelta(days=lookback_days)
        from_date = start.strftime("%Y-%m-%d 09:15:00")
        to_date = current.strftime("%Y-%m-%d %H:%M:%S")
        index_history = self.client.completed_candles(
            self.indices["NIFTY"], from_date, to_date, now=current)
        option_history = self.client.completed_candles(
            self.options["NIFTY_ATM_CE"], from_date, to_date, now=current)
        if index_history.empty or option_history.empty:
            raise SafeBlock("Dhan returned no completed aligned candles")
        common = index_history.timestamp[index_history.timestamp.isin(option_history.timestamp)]
        if len(common) == 0:
            raise SafeBlock("Dhan Index and ATM CALL histories have no common timestamps")
        last = common.max()
        index_history = index_history.loc[index_history.timestamp <= last]
        option_history = option_history.loc[option_history.timestamp <= last]
        return engine.infer_histories(index_history, option_history, decision_observations)
