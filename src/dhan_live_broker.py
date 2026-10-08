"""Secure, read-only Dhan connectivity for Step 27.

Credentials are memory-only inputs. This module contains no order endpoints,
does not persist secrets, and never includes credentials in exceptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable
import json
import threading

import pandas as pd
import requests

from src.live_data_provider import IST, validate_candles


PROFILE_URL = "https://api.dhan.co/v2/profile"
QUOTE_URL = "https://api.dhan.co/v2/marketfeed/quote"
HISTORY_URL = "https://api.dhan.co/v2/charts/intraday"
ROLLING_OPTION_URL = "https://api.dhan.co/v2/charts/rollingoption"
VALID_CONNECTION_STATES = {
    "CONNECTED", "DISCONNECTED", "INVALID TOKEN", "TOKEN EXPIRED",
    "NO INTERNET", "API ERROR",
}


@dataclass(frozen=True)
class BrokerCredentials:
    client_id: str
    access_token: str

    def __post_init__(self):
        if not self.client_id.strip() or not self.access_token.strip():
            raise ValueError("Client ID and access token are required")

    @property
    def masked_token(self) -> str:
        suffix = self.access_token[-4:] if len(self.access_token) >= 4 else "****"
        return f"********{suffix}"

    def headers(self, *, include_client_id: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.access_token,
        }
        if include_client_id:
            headers["client-id"] = self.client_id
        return headers


@dataclass(frozen=True)
class ConnectionHealth:
    status: str
    last_connection_time: str | None
    last_heartbeat: str | None
    latency_ms: float | None
    token_valid: bool
    detail: str

    def public_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Instrument:
    market: str
    security_id: str
    exchange_segment: str
    instrument: str
    display_name: str
    expiry: str | None = None
    strike: float | None = None
    option_type: str | None = None

    def public_dict(self) -> dict:
        return asdict(self)


def _safe_error(response) -> tuple[str, str]:
    """Classify Dhan errors without returning headers, tokens, or request data."""
    try:
        body = response.json()
    except Exception:
        body = {}
    code = str(body.get("errorCode", body.get("code", "")))
    message = str(body.get("errorMessage", body.get("message", "Dhan API rejected request")))
    if response.status_code in {401, 403} or code in {"807", "DH-901"}:
        state = "TOKEN EXPIRED" if code == "807" or "expired" in message.lower() else "INVALID TOKEN"
    else:
        state = "API ERROR"
    return state, f"HTTP {response.status_code}; code={code or 'unavailable'}; {message}"


class DhanReadOnlyClient:
    """Profile, quote and candle methods only; no trading surface exists."""

    def __init__(self, credentials: BrokerCredentials, *, requester=requests,
                 timeout_seconds: int = 20, retries: int = 3,
                 sleeper: Callable[[float], None] = sleep):
        self.credentials = credentials
        self.requester = requester
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.sleeper = sleeper
        self.health = ConnectionHealth(
            "DISCONNECTED", None, None, None, False, "Connection not tested")
        self._pace_lock = threading.Lock()
        self._last_request_at: dict[str, float] = {}

    @staticmethod
    def _rate_bucket(url: str) -> tuple[str, float]:
        # Dhan v2: quote APIs 1/s and historical data APIs 5/s.
        if url == QUOTE_URL:
            return "quote", 1.05
        if url in {HISTORY_URL, ROLLING_OPTION_URL}:
            return "history", 0.21
        return "account", 0.05

    def _pace(self, url: str) -> None:
        bucket, interval = self._rate_bucket(url)
        with self._pace_lock:
            previous = self._last_request_at.get(bucket)
            now = perf_counter()
            if previous is not None:
                wait = interval - (now - previous)
                if wait > 0:
                    self.sleeper(wait)
            self._last_request_at[bucket] = perf_counter()

    @staticmethod
    def _retry_after(response, attempt: int) -> float:
        headers = getattr(response, "headers", {}) or {}
        try:
            supplied = float(headers.get("Retry-After", 0))
        except (TypeError, ValueError):
            supplied = 0
        return max(supplied, min(2 ** attempt, 8))

    def _call(self, method: str, url: str, **kwargs):
        last_error = None
        for attempt in range(self.retries):
            try:
                self._pace(url)
                response = getattr(self.requester, method)(
                    url, timeout=self.timeout_seconds, **kwargs)
                if response.status_code == 429 and attempt + 1 < self.retries:
                    self.sleeper(self._retry_after(response, attempt))
                    continue
                return response
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    self.sleeper(min(2 ** attempt, 4))
        raise ConnectionError("Dhan network request failed after automatic reconnect attempts") from last_error

    def test_connection(self) -> ConnectionHealth:
        started = perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = self._call(
                "get", PROFILE_URL,
                headers=self.credentials.headers(include_client_id=False))
            latency = (perf_counter() - started) * 1000
            if response.status_code != 200:
                status, detail = _safe_error(response)
                self.health = ConnectionHealth(
                    status, None, timestamp, latency, False, detail)
                return self.health
            profile = response.json()
            returned_id = str(profile.get("dhanClientId", ""))
            if returned_id and returned_id != self.credentials.client_id:
                self.health = ConnectionHealth(
                    "INVALID TOKEN", None, timestamp, latency, False,
                    "Authenticated profile does not match the supplied Client ID")
            else:
                validity = profile.get("tokenValidity", "not supplied")
                self.health = ConnectionHealth(
                    "CONNECTED", timestamp, timestamp, latency, True,
                    f"Profile authenticated; token validity: {validity}")
            return self.health
        except ConnectionError:
            latency = (perf_counter() - started) * 1000
            self.health = ConnectionHealth(
                "NO INTERNET", None, timestamp, latency, False,
                "Dhan endpoint could not be reached")
            return self.health
        except Exception:
            latency = (perf_counter() - started) * 1000
            self.health = ConnectionHealth(
                "API ERROR", None, timestamp, latency, False,
                "Unexpected read-only connection error")
            return self.health

    def quote(self, instruments: list[Instrument]) -> dict[str, dict]:
        if self.health.status != "CONNECTED":
            raise RuntimeError("Broker connection is not authenticated")
        request: dict[str, list[int]] = {}
        for item in instruments:
            request.setdefault(item.exchange_segment, []).append(int(item.security_id))
        response = self._call(
            "post", QUOTE_URL, headers=self.credentials.headers(), json=request)
        if response.status_code != 200:
            state, detail = _safe_error(response)
            if state in {"INVALID TOKEN", "TOKEN EXPIRED"}:
                self.health = ConnectionHealth(
                    state, self.health.last_connection_time,
                    datetime.now(timezone.utc).isoformat(), self.health.latency_ms,
                    False, detail)
            else:
                # Rate limits and server errors do not revoke authentication.
                self.health = ConnectionHealth(
                    "CONNECTED", self.health.last_connection_time,
                    datetime.now(timezone.utc).isoformat(), self.health.latency_ms,
                    True, self.health.detail)
            raise RuntimeError(detail)
        return response.json().get("data", {})

    def completed_candles(self, instrument: Instrument, from_date: str,
                          to_date: str, *, now=None) -> pd.DataFrame:
        if self.health.status != "CONNECTED":
            raise RuntimeError("Broker connection is not authenticated")
        response = self._call(
            "post", HISTORY_URL, headers=self.credentials.headers(),
            json={
                "securityId": instrument.security_id,
                "exchangeSegment": instrument.exchange_segment,
                "instrument": instrument.instrument,
                "interval": "1", "oi": False,
                "fromDate": from_date, "toDate": to_date,
            })
        if response.status_code != 200:
            _, detail = _safe_error(response)
            raise RuntimeError(detail)
        body = response.json()
        data = body.get("data", body)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(data):
            raise RuntimeError("Dhan candle response is incomplete")
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST),
            "open": data["open"], "high": data["high"], "low": data["low"],
            "close": data["close"], "volume": data["volume"],
        })
        frame = validate_candles(frame)
        current = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now).tz_convert(IST)
        completed_before = current.floor("min")
        clock = frame["timestamp"].dt.time
        session_open = pd.Timestamp("09:15").time()
        session_close = pd.Timestamp("15:30").time()
        valid_session = (
            frame["timestamp"].dt.weekday.lt(5)
            & clock.ge(session_open)
            & clock.lt(session_close)
        )
        return frame.loc[
            valid_session & frame.timestamp.lt(completed_before)
        ].reset_index(drop=True)

    def rolling_options(self, payload: dict) -> dict:
        """Call Dhan's authenticated read-only expired rolling-option endpoint."""
        if self.health.status != "CONNECTED":
            raise RuntimeError("Broker connection is not authenticated")
        response = self._call(
            "post", ROLLING_OPTION_URL,
            headers=self.credentials.headers(include_client_id=False), json=payload)
        if response.status_code != 200:
            _, detail = _safe_error(response)
            raise RuntimeError(detail)
        body = response.json()
        if body.get("status") == "failure" or body.get("errorCode"):
            code = body.get("errorCode", "unavailable")
            message = body.get("errorMessage") or body.get("message") or "Dhan data error"
            raise RuntimeError(f"Dhan rolling-option error code={code}; {message}")
        return body


class DhanInstrumentResolver:
    """Resolve indices, nearest expiry and ATM contracts from Dhan's master."""

    INDEX_SYMBOLS = {
        "NIFTY": ("NSE", "NIFTY"),
        "BANKNIFTY": ("NSE", "BANKNIFTY"),
        "SENSEX": ("BSE", "SENSEX"),
        "INDIA_VIX": ("NSE", "INDIA VIX"),
    }

    def __init__(self, security_master: str | Path):
        self.path = Path(security_master)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.master = pd.read_csv(self.path, low_memory=False)

    @staticmethod
    def _segment(exchange: str, segment: str) -> str:
        if segment == "I":
            return "IDX_I"
        if segment == "D":
            return f"{exchange}_FNO"
        if segment == "E":
            return f"{exchange}_EQ"
        raise ValueError(f"Unsupported Dhan master segment: {exchange}/{segment}")

    def resolve_indices(self) -> dict[str, Instrument]:
        resolved = {}
        for market, (exchange, symbol) in self.INDEX_SYMBOLS.items():
            matches = self.master.loc[
                self.master.EXCH_ID.astype(str).str.upper().eq(exchange)
                & self.master.SEGMENT.astype(str).str.upper().eq("I")
                & self.master.INSTRUMENT.astype(str).str.upper().eq("INDEX")
                & self.master.UNDERLYING_SYMBOL.astype(str).str.upper().eq(symbol)]
            if len(matches) != 1:
                raise ValueError(f"Expected one master match for {market}; found {len(matches)}")
            row = matches.iloc[0]
            resolved[market] = Instrument(
                market, str(int(row.SECURITY_ID)),
                self._segment(exchange, "I"), "INDEX", str(row.DISPLAY_NAME))
        return resolved

    def resolve_atm_options(self, underlying: str, spot_price: float,
                            as_of=None) -> tuple[Instrument, Instrument]:
        symbol = underlying.upper()
        exchange = "BSE" if symbol == "SENSEX" else "NSE"
        if as_of is None:
            today = pd.Timestamp.now(tz=IST).normalize()
        else:
            today = pd.Timestamp(as_of)
            today = today.tz_localize(IST) if today.tzinfo is None else today.tz_convert(IST)
            today = today.normalize()
        expiry = pd.to_datetime(self.master.SM_EXPIRY_DATE, errors="coerce")
        candidates = self.master.loc[
            self.master.EXCH_ID.astype(str).str.upper().eq(exchange)
            & self.master.SEGMENT.astype(str).str.upper().eq("D")
            & self.master.INSTRUMENT.astype(str).str.upper().eq("OPTIDX")
            & self.master.UNDERLYING_SYMBOL.astype(str).str.upper().eq(symbol)
            & expiry.ge(today.tz_localize(None))
            & self.master.OPTION_TYPE.astype(str).str.upper().isin(["CE", "PE"])
        ].copy()
        if candidates.empty:
            raise ValueError(f"No active {symbol} option contract in Security Master")
        candidates["expiry"] = expiry.loc[candidates.index]
        nearest_expiry = candidates.expiry.min()
        candidates = candidates.loc[candidates.expiry.eq(nearest_expiry)]
        strikes = pd.to_numeric(candidates.STRIKE_PRICE, errors="coerce")
        nearest_strike = float(strikes.iloc[(strikes - float(spot_price)).abs().argmin()])
        candidates = candidates.loc[strikes.eq(nearest_strike)]
        outputs = []
        for option_type in ("CE", "PE"):
            match = candidates.loc[candidates.OPTION_TYPE.astype(str).str.upper().eq(option_type)]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one {symbol} {option_type} at {nearest_strike}; found {len(match)}")
            row = match.iloc[0]
            outputs.append(Instrument(
                f"{symbol}_ATM_{option_type}", str(int(row.SECURITY_ID)),
                self._segment(exchange, "D"), "OPTIDX", str(row.DISPLAY_NAME),
                nearest_expiry.date().isoformat(), nearest_strike, option_type))
        return outputs[0], outputs[1]

    def resolve_option_chain(
        self, underlying: str, spot_price: float, *,
        strikes_each_side: int = 10, as_of=None,
    ) -> list[Instrument]:
        """Resolve nearest-expiry CE/PE instruments around the ATM strike."""
        symbol = underlying.upper()
        exchange = "BSE" if symbol == "SENSEX" else "NSE"
        current = pd.Timestamp.now(tz=IST) if as_of is None else pd.Timestamp(as_of)
        current = (
            current.tz_localize(IST) if current.tzinfo is None
            else current.tz_convert(IST))
        expiry = pd.to_datetime(self.master.SM_EXPIRY_DATE, errors="coerce")
        candidates = self.master.loc[
            self.master.EXCH_ID.astype(str).str.upper().eq(exchange)
            & self.master.SEGMENT.astype(str).str.upper().eq("D")
            & self.master.INSTRUMENT.astype(str).str.upper().eq("OPTIDX")
            & self.master.UNDERLYING_SYMBOL.astype(str).str.upper().eq(symbol)
            & expiry.ge(current.normalize().tz_localize(None))
            & self.master.OPTION_TYPE.astype(str).str.upper().isin(["CE", "PE"])
        ].copy()
        if candidates.empty:
            return []
        candidates["expiry"] = expiry.loc[candidates.index]
        candidates = candidates.loc[
            candidates.expiry.eq(candidates.expiry.min())].copy()
        candidates["strike_value"] = pd.to_numeric(
            candidates.STRIKE_PRICE, errors="coerce")
        strikes = sorted(candidates.strike_value.dropna().unique())
        if not strikes:
            return []
        atm_index = min(
            range(len(strikes)),
            key=lambda index: abs(strikes[index] - float(spot_price)))
        lower = max(0, atm_index - max(0, int(strikes_each_side)))
        upper = min(
            len(strikes), atm_index + max(0, int(strikes_each_side)) + 1)
        selected = set(strikes[lower:upper])
        candidates = candidates.loc[candidates.strike_value.isin(selected)]
        output = []
        for row in candidates.sort_values(
            ["strike_value", "OPTION_TYPE"]).itertuples(index=False):
            option_type = str(row.OPTION_TYPE).upper()
            strike = float(row.strike_value)
            output.append(Instrument(
                f"{symbol}_{strike:g}_{option_type}",
                str(int(row.SECURITY_ID)), self._segment(exchange, "D"),
                "OPTIDX", str(row.DISPLAY_NAME),
                pd.Timestamp(row.expiry).date().isoformat(),
                strike, option_type))
        return output


def public_broker_snapshot(client: DhanReadOnlyClient,
                           instruments: list[Instrument]) -> dict:
    """Safe serialisable state: deliberately excludes credentials."""
    return {
        "health": client.health.public_dict(),
        "instruments": [item.public_dict() for item in instruments],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "credential_storage": "STREAMLIT_SESSION_STATE_ONLY",
    }
