"""
Health check endpoint.
"""
from fastapi import APIRouter

from app.models.response_models import HealthResponse
from app.utils.helpers import get_current_timestamp

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the health status of the service.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns healthy status for Cloud Run and load balancer probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=get_current_timestamp(),
        version="1.0.0",
    )
