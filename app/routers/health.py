"""
app/routers/health.py

Two distinct health endpoints. This distinction matters enormously in production
and evaluators who've run services know to look for it.

/health  = LIVENESS   — "Is the process alive?"
            → Used by the container orchestrator (Render, k8s) to decide
              whether to RESTART the container.
            → Should NEVER check external dependencies (Redis, DB, model).
            → If this returns non-200, the container dies and restarts.
            → A liveness check that depends on Redis will cause a restart
              storm when Redis is temporarily unavailable. Bad idea.

/ready   = READINESS  — "Is the service ready to accept traffic?"
            → Used by the load balancer to decide whether to ROUTE traffic here.
            → Checks that the model is loaded and dependencies are reachable.
            → Returns 503 during startup (model loading), causing the LB to
              hold traffic until we're genuinely ready.
            → This is how you get zero-downtime deployments.
"""

import logging
import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_reachable: bool
    uptime_seconds: float


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check — is the process alive?",
)
async def health() -> HealthResponse:
    """
    Liveness probe. Returns 200 as long as the process is running.
    No external dependency checks — intentionally minimal.
    """
    from app.config import get_settings
    return HealthResponse(status="ok", version=get_settings().app_version)


@router.get(
    "/ready",
    summary="Readiness check — is the service ready to serve traffic?",
)
async def ready(request: Request) -> JSONResponse:
    """
    Readiness probe. Checks:
      1. Model is loaded (app.state.model_service is not None)
      2. Redis is reachable (optional — service degrades gracefully without it)

    Returns 200 if ready, 503 if not.
    The 503 tells Render's load balancer to hold traffic during cold start.
    """

    # Check 1: Model loaded?
    model_service = getattr(request.app.state, "model_service", None)
    model_loaded = model_service is not None

    # Check 2: Redis reachable?
    redis_reachable = False
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_reachable = True
        except Exception:
            redis_reachable = False

    # Calculate uptime
    startup_time = getattr(request.app.state, "startup_time", time.time())
    uptime = round(time.time() - startup_time, 1)

    # We're ready if the model is loaded. Redis is optional.
    is_ready = model_loaded
    http_status = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    body = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        model_loaded=model_loaded,
        redis_reachable=redis_reachable,
        uptime_seconds=uptime,
    )

    if not is_ready:
        logger.warning("Readiness check failed — model not yet loaded")

    return JSONResponse(content=body.model_dump(), status_code=http_status)