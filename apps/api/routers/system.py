from fastapi import APIRouter

from apps.api.schemas.common import HealthResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "service": "local-api"}
