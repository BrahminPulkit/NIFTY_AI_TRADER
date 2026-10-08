from fastapi import APIRouter

from apps.api.routers import audit, broker, paper, replay, system, trading

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(broker.router)
api_router.include_router(trading.router)
api_router.include_router(replay.router)
api_router.include_router(paper.router)
api_router.include_router(audit.router)
