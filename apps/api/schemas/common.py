from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HealthResponse(ApiModel):
    status: str
    service: str


class BrokerLogin(ApiModel):
    client_id: str = Field(min_length=1, max_length=100)
    access_token: str = Field(min_length=1, max_length=4096)


class BrokerHealth(ApiModel):
    status: str
    last_connection_time: str | None = None
    last_heartbeat: str | None = None
    latency_ms: float | None = None
    token_valid: bool = False
    detail: str


class BrokerState(ApiModel):
    connected: bool
    status: str
    health: BrokerHealth
    market_status: str
    last_data_update: str | None = None
    last_error: str | None = None
    worker_alive: bool


class SignalState(ApiModel):
    timestamp: str
    probability: float | None = None
    confidence: float | None = None
    decision: str
    action: str
    reason: str
    strategy: str
    regime: str
    risk_status: str
    latency_ms: float
    model_version: str
    instrument_scope: str


class MarketState(ApiModel):
    connected: bool
    status: str
    manager_status: str
    quotes: dict[str, float]
    updated_at: str | None = None
    last_error: str | None = None
    signal: SignalState | None = None
    signal_status: str
    signal_error: str | None = None
    put_signal: dict[str, Any] | None = None
    put_signal_status: str = "UNAVAILABLE"
    put_signal_error: str | None = None
    multistrategy_signal: dict[str, Any] | None = None
    multistrategy_status: str = "UNAVAILABLE"
    multistrategy_error: str | None = None
    scalping_evaluations: list[dict[str, Any]] = Field(default_factory=list)
    scalping_status: str = "UNAVAILABLE"
    scalping_error: str | None = None


class Candle(ApiModel):
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class MarketResponse(ApiModel):
    market: MarketState
    candles: list[Candle]
    option_candles: dict[str, list[Candle]] = Field(default_factory=dict)
    instruments: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    watched_candles: dict[str, list[Candle]] = Field(default_factory=dict)


class OptionRow(ApiModel):
    security_id: str | int | None = None
    instrument_name: str | None = None
    option_type: str | None = None
    strike: float | None = None
    expiry: str | None = None
    ltp: float | None = None
    oi: float | None = None
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None


class OptionChainResponse(ApiModel):
    connected: bool
    status: str
    updated_at: str | None = None
    rows: list[OptionRow]


class ErrorBody(ApiModel):
    code: str
    message: str
    request_id: str
