from fastapi import APIRouter, HTTPException, Query

from apps.api.services.signal_audit import signal_audit_payload

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/signals")
def signals(date: str | None = Query(default=None)) -> dict:
    try:
        return signal_audit_payload(date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
