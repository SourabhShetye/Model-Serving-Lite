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
from app.services.in_memory_cache import InMemoryRedis
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


async def get_redis(request: Request) -> aioredis.Redis | InMemoryRedis:
    """
    Returns the async Redis-like client from app.state (real Redis or
    an in-memory fallback). If it is missing, create one once and store
    it on app.state so it persists across requests in this process.
    """
    redis_client = getattr(request.app.state, "redis_client", None)

    if redis_client is None:
        logger.warning(
            "Redis client not available in app.state — creating in-memory fallback"
        )
        redis_client = InMemoryRedis()
        request.app.state.redis_client = redis_client

    return redis_client


RedisDep = Annotated[aioredis.Redis | InMemoryRedis | None, Depends(get_redis)]
