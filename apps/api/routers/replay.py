from fastapi import APIRouter, HTTPException, Query

from apps.api.services.replay import (available_sessions, broker_session_payload, broker_sessions,
                                      rolling_session_payload, rolling_sessions, session_payload)

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/sessions")
def sessions() -> dict:
    values = available_sessions()
    return {"sessions": values, "count": len(values), "latest": values[0] if values else None}


@router.get("/session")
def session(date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict:
    try:
        return session_payload(date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/broker/sessions")
def dhan_sessions() -> dict:
    values = broker_sessions()
    return {"sessions": values, "count": len(values), "latest": values[0] if values else None}

@router.get("/broker/session")
def dhan_session(date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict:
    try:
        return broker_session_payload(date)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/rolling/sessions")
def atm_sessions() -> dict:
    values = rolling_sessions()
    return {"sessions": values, "count": len(values), "latest": values[0] if values else None}


@router.get("/rolling/session")
def atm_session(date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict:
    try:
        return rolling_session_payload(date)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
