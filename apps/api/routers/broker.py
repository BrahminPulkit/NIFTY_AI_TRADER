from datetime import date
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from apps.api.schemas.common import BrokerLogin, BrokerState
from apps.api.services.runtime import public_broker
from src.dhan_connection_manager import get_dhan_manager

router = APIRouter(prefix="/broker", tags=["broker"])
logger = logging.getLogger("nifty_ai.api.broker")


class HistoricalOptionRequest(BaseModel):
    start: date
    end: date
    expiry_code: int = 0


class ExactContractBatchRequest(BaseModel):
    dates: list[date]


@router.get("/session", response_model=BrokerState)
def session() -> dict:
    return public_broker(get_dhan_manager().snapshot())


@router.post("/connect", response_model=BrokerState)
def connect(payload: BrokerLogin) -> dict:
    broker = public_broker(
        get_dhan_manager().connect(payload.client_id, payload.access_token))
    if broker["connected"]:
        logger.info("dhan_connect_succeeded status=%s", broker["status"])
        return broker

    broker_status = broker["health"]["status"]
    detail = broker["health"]["detail"]
    logger.warning(
        "dhan_connect_rejected status=%s detail=%s", broker_status, detail)
    http_status = {
        "INVALID TOKEN": status.HTTP_401_UNAUTHORIZED,
        "TOKEN EXPIRED": status.HTTP_401_UNAUTHORIZED,
        "NO INTERNET": status.HTTP_503_SERVICE_UNAVAILABLE,
    }.get(broker_status, status.HTTP_502_BAD_GATEWAY)
    raise HTTPException(status_code=http_status, detail=detail)


@router.post("/disconnect", response_model=BrokerState)
def disconnect() -> dict:
    return public_broker(get_dhan_manager().disconnect())


@router.post("/refresh", response_model=BrokerState)
def refresh() -> dict:
    manager = get_dhan_manager()
    if not manager.snapshot()["connected"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Broker is not connected")
    manager.request_refresh()
    return public_broker(manager.snapshot())


@router.post("/historical-options/acquire")
def acquire_historical_options(payload: HistoricalOptionRequest) -> dict:
    if payload.start >= payload.end:
        raise HTTPException(status_code=422, detail="start must be earlier than exclusive end")
    try:
        if payload.expiry_code not in {0, 1, 2}:
            raise HTTPException(status_code=422, detail="expiry_code must be 0, 1, or 2")
        return get_dhan_manager().acquire_rolling_options(
            payload.start, payload.end, expiry_code=payload.expiry_code)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/historical-options/exact-contract-batch")
def acquire_exact_contract_batch(payload: ExactContractBatchRequest) -> dict:
    if not payload.dates:
        raise HTTPException(status_code=422, detail="At least one historical date is required")
    try:
        return get_dhan_manager().acquire_exact_contract_batch(
            [str(value) for value in payload.dates])
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
