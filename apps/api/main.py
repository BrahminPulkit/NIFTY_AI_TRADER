"""Local presentation API; all domain behavior remains in existing services."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.router import api_router
from apps.api.routers.broker import connect, disconnect, refresh, session
from apps.api.routers.system import health
from apps.api.routers.trading import desk, market, options, paper
from apps.api.schemas.common import BrokerLogin
from apps.api.services.runtime import market_contract, public_broker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nifty_ai.api")

app = FastAPI(title="NIFTY AI Trader Local API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)
app.include_router(api_router, prefix="/api/v1")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "Request failed", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code,
        (perf_counter() - started) * 1000,
    )
    return response


# Temporary compatibility routes for the first React client and local scripts.
app.add_api_route("/api/health", health, methods=["GET"])
app.add_api_route("/api/desk", desk, methods=["GET"])
app.add_api_route("/api/market", market, methods=["GET"])
app.add_api_route("/api/options", options, methods=["GET"])
app.add_api_route("/api/paper", paper, methods=["GET"])
app.add_api_route("/api/broker", session, methods=["GET"])
app.add_api_route("/api/broker/connect", connect, methods=["POST"])
app.add_api_route("/api/broker/disconnect", disconnect, methods=["POST"])
app.add_api_route("/api/broker/refresh", refresh, methods=["POST"])

__all__ = ["app", "BrokerLogin", "market_contract", "public_broker"]
