"""Provider-independent, read-only candle sources for paper prediction.

No class in this module can place, modify, or cancel an order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
import os

import pandas as pd
import requests


IST = "Asia/Kolkata"
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketCandle:
    market: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(OHLCV).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")
    out = frame[OHLCV].copy()
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="raise").dt.tz_convert(IST)
    out[OHLCV[1:]] = out[OHLCV[1:]].apply(pd.to_numeric, errors="raise")
    invalid = (
        out.timestamp.duplicated() | out[OHLCV[1:]].isna().any(axis=1)
        | out[["open", "high", "low", "close"]].le(0).any(axis=1)
        | out.volume.lt(0) | out.high.lt(out.low)
        | out.open.lt(out.low) | out.open.gt(out.high)
        | out.close.lt(out.low) | out.close.gt(out.high)
    )
    if invalid.any():
        raise ValueError(f"Invalid candle rows: {int(invalid.sum())}")
    if not out.timestamp.is_monotonic_increasing:
        raise ValueError("Candles must be chronologically sorted")
    return out


class MarketDataProvider(ABC):
    """Minimal interface implemented by every read-only data provider."""

    @abstractmethod
    def candles(self) -> Iterator[MarketCandle]:
        raise NotImplementedError


class CSVReplayProvider(MarketDataProvider):
    def __init__(self, path: str | Path, market: str = "NIFTY"):
        self.path = Path(path)
        self.market = market.upper()

    def candles(self) -> Iterator[MarketCandle]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        frame = pd.read_parquet(self.path) if self.path.suffix.lower() == ".parquet" else pd.read_csv(self.path)
        for row in validate_candles(frame).itertuples(index=False):
            yield MarketCandle(
                self.market, row.timestamp, float(row.open), float(row.high),
                float(row.low), float(row.close), float(row.volume), "CSV_REPLAY",
            )


class DhanIntradayProvider(MarketDataProvider):
    """Polling adapter for Dhan's read-only intraday chart endpoint.

    Credentials and security identifiers are configuration inputs. They are
    never logged and there are deliberately no trading methods.
    """

    endpoint = "https://api.dhan.co/v2/charts/intraday"

    def __init__(
        self,
        *,
        market: str,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        from_date: str,
        to_date: str,
        client_id_env: str = "DHAN_CLIENT_ID",
        access_token_env: str = "DHAN_ACCESS_TOKEN",
        timeout_seconds: int = 30,
    ):
        self.market = market.upper()
        self.security_id = str(security_id)
        self.exchange_segment = exchange_segment
        self.instrument = instrument
        self.from_date = from_date
        self.to_date = to_date
        self.client_id_env = client_id_env
        self.access_token_env = access_token_env
        self.timeout_seconds = timeout_seconds

    def _request(self) -> pd.DataFrame:
        client_id, token = os.getenv(self.client_id_env), os.getenv(self.access_token_env)
        if not client_id or not token:
            raise RuntimeError("Dhan credentials are unavailable in configured environment variables")
        payload = {
            "securityId": self.security_id,
            "exchangeSegment": self.exchange_segment,
            "instrument": self.instrument,
            "interval": "1",
            "oi": False,
            "fromDate": self.from_date,
            "toDate": self.to_date,
        }
        response = requests.post(
            self.endpoint,
            headers={"Accept": "application/json", "Content-Type": "application/json",
                     "client-id": client_id, "access-token": token},
            json=payload, timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Dhan market-data request failed HTTP {response.status_code}: {response.text}")
        body = response.json()
        data = body.get("data", body)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(data):
            raise RuntimeError(f"Dhan response lacks candle arrays; keys={sorted(data)}")
        timestamp = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST)
        return validate_candles(pd.DataFrame({
            "timestamp": timestamp, "open": data["open"], "high": data["high"],
            "low": data["low"], "close": data["close"], "volume": data["volume"],
        }))

    def candles(self) -> Iterator[MarketCandle]:
        for row in self._request().itertuples(index=False):
            yield MarketCandle(
                self.market, row.timestamp, float(row.open), float(row.high),
                float(row.low), float(row.close), float(row.volume), "DHAN",
            )

