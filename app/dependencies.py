"""
app/dependencies.py

FastAPI dependency injection providers.

Why does this file exist separately?
  Without it, every route would do:  request.app.state.model_service
  That's fine for 2 routes. For 10+ routes it's repetitive, untestable,
  and couples your routes to the request object structure.

  With this file:
    - Routes declare what they need:  model: ModelService = Depends(get_model_service)
    - Tests override it:  app.dependency_overrides[get_model_service] = lambda: MockModelService()
    - Zero changes to route code when the underlying source changes.

  This is the FastAPI-idiomatic dependency injection pattern.
  It's the difference between "I've read the docs" and "I've built with this."
"""

import logging
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Settings                                                             #
# ------------------------------------------------------------------ #

def get_settings_dep() -> Settings:
    """
    Wraps get_settings() for use as a FastAPI dependency.
    Allows override in tests without touching the lru_cache singleton.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# ------------------------------------------------------------------ #
# Model Service                                                        #
# ------------------------------------------------------------------ #

def get_model_service(request: Request) -> ModelService:
    """
    Retrieves the ModelService instance from app.state.

    app.state.model_service is set during the lifespan startup event
    in main.py. If it's not there, we fail loudly — a missing model
    service is a hard startup error, not something to handle gracefully
    per-request.
    """
    model_service: ModelService = request.app.state.model_service
    if model_service is None:
        # This should never happen if lifespan is wired correctly.
        # If it does, it means the app started without the model loading —
        # a critical bug we want to surface immediately.
        raise RuntimeError(
            "ModelService not initialised. "
            "Check that lifespan startup completed without errors."
        )
    return model_service


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


# ------------------------------------------------------------------ #
# Redis Client                                                         #
# ------------------------------------------------------------------ #

async def get_redis(request: Request) -> aioredis.Redis | None:
    """
    Returns the async Redis client from app.state, or None if Redis
    is unavailable or cache is disabled.

    Why return None instead of raising?
      Cache is an optimisation, not a hard dependency. If Redis goes
      down, we want predictions to keep working — just slower.
      Returning None lets callers do: `if redis_client: ...`
      without try/except blocks in every route.

    This is the "fail open" vs "fail closed" decision. For a cache,
    fail open is almost always right. For auth, fail closed.
    """
    settings = get_settings()
    if not settings.cache_enabled:
        return None

    redis_client: aioredis.Redis | None = getattr(request.app.state, "redis_client", None)
    if redis_client is None:
        logger.warning("Redis client not available — cache disabled for this request")
    return redis_client


RedisDep = Annotated[aioredis.Redis | None, Depends(get_redis)]
