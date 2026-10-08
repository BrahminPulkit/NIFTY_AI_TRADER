from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from apps.api.services.replay import session_payload
from src.dhan_connection_manager import get_dhan_manager
from src.paper_trading_engine import OptionCandle, PaperSafetyBlock
from src.paper_trading_manager import PaperRequestError, get_paper_manager

router = APIRouter(prefix="/paper", tags=["paper"])


class PaperContext(BaseModel):
    mode: Literal["live", "replay"] = "live"
    session_date: str | None = None
    timestamp: str | None = None


class ManualPaperOrder(BaseModel):
    security_id: str
    quantity: int = 1


def _manual_contract(payload: ManualPaperOrder) -> tuple[dict, str]:
    cache = get_dhan_manager().snapshot()
    if not cache.get("connected") or cache.get("market_status") != "OPEN":
        raise HTTPException(status_code=409, detail="Live market connection is not open")
    updated = cache.get("last_data_update")
    if not updated:
        raise HTTPException(status_code=409, detail="Live option data is unavailable")
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(updated)
    if age > pd.Timedelta(seconds=90):
        raise HTTPException(status_code=409, detail="Live option data is stale")
    contract = next((row for row in cache.get("option_chain", [])
                     if str(row.get("security_id")) == payload.security_id), None)
    if contract is None:
        raise HTTPException(status_code=404, detail="Selected live option contract was not found")
    if str(contract.get("option_type", "")).upper() not in {"CE", "PE"}:
        raise HTTPException(status_code=422, detail="Selected contract must be CE or PE")
    return contract, updated


def _context(payload: PaperContext) -> tuple[dict, OptionCandle]:
    if payload.mode == "live":
        cache = get_dhan_manager().snapshot()
        evaluation = next((item for item in cache.get("scalping_evaluations", [])
                           if str(item.get("final_decision", "")).startswith("BUY ")), None)
        if not evaluation:
            raise HTTPException(status_code=409, detail="No corrected live scalping signal is approved")
        side = str(evaluation["final_decision"]).split()[-1]
        frame = cache.get("candles", {}).get(f"NIFTY_ATM_{side}", pd.DataFrame())
        if frame.empty:
            raise HTTPException(status_code=409, detail=f"No live ATM {side} candle is available")
        timestamp = pd.Timestamp(evaluation["timestamp"])
        synchronized = frame.loc[pd.to_datetime(frame.timestamp).eq(timestamp)]
        if synchronized.empty:
            raise HTTPException(status_code=409, detail="Signal and option candle are not aligned")
        candle = get_paper_manager().candle(synchronized.iloc[-1])
        signal = {
            "timestamp": evaluation["timestamp"], "decision": "TRADE",
            "action": evaluation["final_decision"],
            "probability": evaluation["model_probability"],
            "required_probability": evaluation["required_probability"],
            "strategy": evaluation["strategy"], "regime": evaluation["direction"],
            "reason": "CORRECTED_SCALPING_APPROVED",
            "selected_option": evaluation["selected_option"],
        }
        return signal, candle
    if not payload.session_date or not payload.timestamp:
        raise HTTPException(status_code=422, detail="Replay requires session_date and timestamp")
    replay = session_payload(payload.session_date)
    event = next((item for item in replay["events"] if item["timestamp"] == payload.timestamp), None)
    row = next((item for item in replay["candles"] if item["timestamp"] == payload.timestamp), None)
    if event is None or row is None:
        raise HTTPException(status_code=404, detail="Replay signal or synchronized candle was not found")
    return event, get_paper_manager().candle(row)


@router.get("/account")
def account(mode: Literal["live", "replay"] = "live", session_date: str | None = None) -> dict:
    try:
        return get_paper_manager().account(mode, session_date)
    except PaperRequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/review")
def review(payload: PaperContext) -> dict:
    prediction, candle = _context(payload)
    return get_paper_manager().review(prediction, candle)


@router.post("/orders")
def open_order(payload: PaperContext) -> dict:
    prediction, candle = _context(payload)
    try:
        return get_paper_manager().open(
            payload.mode, prediction, candle, payload.session_date)
    except (PaperRequestError, PaperSafetyBlock) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/manual/review")
def manual_review(payload: ManualPaperOrder) -> dict:
    contract, timestamp = _manual_contract(payload)
    try:
        return get_paper_manager().review_manual(
            contract, timestamp, payload.quantity)
    except (PaperRequestError, PaperSafetyBlock) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/manual/orders")
def manual_open(payload: ManualPaperOrder) -> dict:
    contract, timestamp = _manual_contract(payload)
    try:
        return get_paper_manager().open_manual(
            contract, timestamp, payload.quantity)
    except (PaperRequestError, PaperSafetyBlock) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/mark")
def replay_mark(payload: PaperContext) -> dict:
    if payload.mode != "replay" or not payload.session_date or not payload.timestamp:
        raise HTTPException(status_code=422, detail="Replay mark requires session_date and timestamp")
    replay = session_payload(payload.session_date)
    target = next((item for item in replay["candles"] if item["timestamp"] == payload.timestamp), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Replay candle was not found")
    try:
        account = get_paper_manager().account("replay", payload.session_date)
        position = account["position"]
        result = {"status": "NO_POSITION"}
        if position:
            for row in replay["candles"]:
                timestamp = pd.Timestamp(row["timestamp"])
                if pd.Timestamp(position["entry_timestamp"]) < timestamp <= pd.Timestamp(payload.timestamp):
                    result = get_paper_manager().update_replay(
                        payload.session_date, get_paper_manager().candle(row))
                    if result.get("status") == "CLOSED":
                        break
        return {"result": result, "account": get_paper_manager().account("replay", payload.session_date)}
    except PaperSafetyBlock as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/exit")
def exit_position(payload: PaperContext) -> dict:
    if payload.mode == "live":
        cache = get_dhan_manager().snapshot()
        account = get_paper_manager().account("live")
        position = account.get("position")
        if position and position.get("security_id"):
            contract = next((row for row in cache.get("option_chain", [])
                             if str(row.get("security_id")) == position.get("security_id")), None)
            if contract is None or not cache.get("last_data_update"):
                raise HTTPException(status_code=409, detail="Selected live option quote is unavailable")
            candle = get_paper_manager().quote_candle(
                contract, cache["last_data_update"], side="exit")
        else:
            side = str((position or {}).get("option_type") or "CE")
            frame = cache.get("candles", {}).get(f"NIFTY_ATM_{side}", pd.DataFrame())
            if frame.empty:
                raise HTTPException(status_code=409, detail="No live option candle is available")
            candle = get_paper_manager().candle(frame.iloc[-1])
    else:
        if not payload.session_date or not payload.timestamp:
            raise HTTPException(status_code=422, detail="Replay exit requires session_date and timestamp")
        replay = session_payload(payload.session_date)
        row = next((item for item in replay["candles"] if item["timestamp"] == payload.timestamp), None)
        if row is None:
            raise HTTPException(status_code=404, detail="Replay candle was not found")
        candle = get_paper_manager().candle(row)
    try:
        return get_paper_manager().manual_close(payload.mode, candle, payload.session_date)
    except (PaperRequestError, PaperSafetyBlock) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
