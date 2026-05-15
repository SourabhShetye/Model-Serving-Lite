"""
app/routers/predict.py

The /predict endpoint.

Router responsibility contract:
  1. Define request/response Pydantic schemas (input validation, output shape).
  2. Orchestrate service calls (cache → model → drift recorder).
  3. Return a typed response.

Router does NOT:
  - Contain business logic (that lives in services/).
  - Know how the model works internally.
  - Know how caching is implemented.

If you find yourself writing an if-statement that has nothing to do with
HTTP concerns, it probably belongs in a service.
"""

import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.dependencies import ModelServiceDep, RedisDep, SettingsDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["inference"])


# ------------------------------------------------------------------ #
# Request / Response Schemas                                           #
# ------------------------------------------------------------------ #


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text to classify. Max 512 characters (model context limit).",
        examples=["This product exceeded all my expectations!"],
    )

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        """
        Strip whitespace and reject blank strings that survive min_length=1.
        '   ' is 3 characters but semantically empty.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must contain non-whitespace characters")
        return stripped


class PredictResponse(BaseModel):
    request_id: str = Field(
        description="UUID for log correlation. Give this to support."
    )
    label: Literal["POSITIVE", "NEGATIVE"]
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    input_hash: str = Field(description="SHA-256 of input. Used as cache key.")
    latency_ms: float = Field(
        description="Model inference time (excludes cache lookup)"
    )
    model_name: str
    cache_hit: bool = Field(description="True if result was served from Redis cache")


# ------------------------------------------------------------------ #
# Endpoint                                                             #
# ------------------------------------------------------------------ #


@router.post(
    "/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify sentiment of input text",
    response_description="Sentiment label with confidence score",
)
async def predict(
    body: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    model_service: ModelServiceDep,
    redis_client: RedisDep,
    settings: SettingsDep,
) -> PredictResponse:
    """
    Predicts sentiment for the given text.

    **Cache behaviour**: Identical inputs (same SHA-256 hash) are served
    from Redis within the TTL window. `cache_hit: true` in the response
    means the model was NOT re-invoked.

    **Correlation**: Every response includes a `request_id`. Include this
    in any bug report — it links the HTTP response to the structured log
    entry and the database row.

    **DB logging**: Written asynchronously after response is returned.
    Latency impact: zero.
    """
    # request_id was set by StructuredLoggingMiddleware — reuse it for correlation
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    cache_hit = False
    t_total_start = time.perf_counter()

    # ---------------------------------------------------------------- #
    # 1. Build cache key                                                 #
    # ---------------------------------------------------------------- #
    from app.services.model_service import build_input_hash
    from app.services.cache_service import CacheService

    input_hash = build_input_hash(body.text)

    # ---------------------------------------------------------------- #
    # 2. Cache lookup via CacheService                                  #
    # ---------------------------------------------------------------- #
    if redis_client is not None:
        cache_svc = CacheService(redis_client)
        cached = await cache_svc.get(input_hash)
        if cached is not None:
            cache_hit = True
            logger.info(
                "Cache hit — skipping model inference",
                extra={"request_id": request_id, "input_hash": input_hash},
            )
            # Still log the cache hit to DB for hit-rate analytics
            background_tasks.add_task(
                _log_to_db,
                request=request,
                request_id=request_id,
                input_text=body.text,
                input_hash=input_hash,
                label=cached.label,
                confidence=cached.score,
                latency_ms=cached.latency_ms,
                cache_hit=True,
                model_name=cached.model_name,
            )
            return PredictResponse(
                request_id=request_id,
                label=cached.label,
                confidence=cached.score,
                input_hash=input_hash,
                latency_ms=cached.latency_ms,
                model_name=cached.model_name,
                cache_hit=True,
            )

    # ---------------------------------------------------------------- #
    # 3. Model inference                                                 #
    # ---------------------------------------------------------------- #
    try:
        result = model_service.predict(body.text)
    except ValueError as exc:
        logger.error(
            "Model returned unexpected output",
            extra={
                "request_id": request_id,
                "input_hash": input_hash,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. The request has been logged.",
        ) from exc

    # ---------------------------------------------------------------- #
    # 4. Write to cache (non-blocking)                                  #
    # ---------------------------------------------------------------- #
    if redis_client is not None:
        cache_svc = CacheService(redis_client)
        await cache_svc.set(
            input_hash=input_hash,
            label=result.label,
            score=result.score,
            latency_ms=result.latency_ms,
            model_name=result.model_name,
        )

    # ---------------------------------------------------------------- #
    # 5. Feed drift monitor (fire-and-forget)                           #
    # ---------------------------------------------------------------- #
    try:
        from app.services.drift_service import get_drift_service

        drift_service = get_drift_service()
        drift_service.record(text=body.text, confidence=result.score)
    except Exception as exc:
        logger.warning(
            "Drift recording failed",
            extra={"request_id": request_id, "error": str(exc)},
        )

    # ---------------------------------------------------------------- #
    # 6. Log to PostgreSQL in background (after response returns)       #
    # ---------------------------------------------------------------- #
    background_tasks.add_task(
        _log_to_db,
        request=request,
        request_id=request_id,
        input_text=body.text,
        input_hash=input_hash,
        label=result.label,
        confidence=result.score,
        latency_ms=result.latency_ms,
        cache_hit=False,
        model_name=result.model_name,
    )

    logger.info(
        "Prediction complete",
        extra={
            "request_id": request_id,
            "input_hash": input_hash,
            "label": result.label,
            "confidence": result.score,
            "latency_ms": result.latency_ms,
            "cache_hit": cache_hit,
            "model_name": result.model_name,
            "total_ms": round((time.perf_counter() - t_total_start) * 1000, 3),
        },
    )

    return PredictResponse(
        request_id=request_id,
        label=result.label,
        confidence=result.score,
        input_hash=input_hash,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
        cache_hit=False,
    )


# ------------------------------------------------------------------ #
# Background task helpers                                              #
# ------------------------------------------------------------------ #


def _log_to_db(
    *,
    request: Request,
    request_id: str,
    input_text: str,
    input_hash: str,
    label: str,
    confidence: float,
    latency_ms: float,
    cache_hit: bool,
    model_name: str,
) -> None:
    """
    Writes prediction to PostgreSQL. Runs in background after response returns.

    This is a plain synchronous function — FastAPI's BackgroundTasks
    runs it in a thread pool automatically, so it doesn't block the
    async event loop.
    """
    if not getattr(request.app.state, "db_available", False):
        return  # DB not reachable — skip silently, structured log is the fallback

    from app.db.crud import write_prediction_log

    try:
        from langdetect import detect

        detected_language = detect(input_text)
    except Exception:
        detected_language = None

    write_prediction_log(
        request_id=request_id,
        input_text=input_text,
        input_hash=input_hash,
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        model_name=model_name,
        input_length=float(len(input_text)),
        detected_language=detected_language,
        client_ip=getattr(request.state, "client_ip", None),
    )
