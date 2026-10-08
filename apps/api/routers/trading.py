from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.dhan_connection_manager import get_dhan_manager

from apps.api.schemas.common import MarketResponse, OptionChainResponse
from apps.api.services.runtime import desk_payload, market_payload, options_payload, paper_payload

router = APIRouter(tags=["trading"])

class WatchOption(BaseModel):
    security_id: str


@router.get("/desk")
def desk() -> dict:
    return desk_payload()


@router.get("/market", response_model=MarketResponse)
def market() -> dict:
    return market_payload()


@router.get("/options", response_model=OptionChainResponse)
def options() -> dict:
    return options_payload()

@router.post("/market/watch-option")
def watch_option(payload: WatchOption) -> dict:
    try:
        return {"status": "QUEUED", "instrument": get_dhan_manager().watch_option(payload.security_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/paper")
def paper() -> dict:
    return paper_payload()
