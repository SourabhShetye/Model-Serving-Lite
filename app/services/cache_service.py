"""
app/services/cache_service.py

Redis-backed prediction cache — your Feature Store stub.

Why extract this from the router into its own service?
  In predict.py we had inline Redis calls. That works for one endpoint.
  The moment you add a /batch endpoint or a background retraining job
  that needs cache invalidation, you need this logic in ONE place.

  More importantly: the router test shouldn't need a real Redis instance.
  With this service extracted, you can inject a MockCacheService in tests
  and test routing logic in complete isolation.

Design:
  - Async-first: all methods are coroutines (await-able).
  - Fail-open: every method catches Redis exceptions and returns None/False.
    The caller never needs try/except — it just checks the return value.
  - Typed: returns typed dataclasses, not raw dicts. The router gets a
    CachedPrediction object, not json.loads() output it has to handle itself.

This is the "Feature Store stub" the brief asks for. In a real system,
this would connect to a proper feature store (Feast, Tecton) with:
  - Feature versioning
  - TTL policies per feature group
  - Monitoring on hit/miss rates

For this service, Redis with SHA-256 keys is the right level of complexity.
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class CachedPrediction:
    """
    Typed representation of a cache hit.
    Mirrors PredictionResult but comes from Redis, not the model.
    """
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float
    latency_ms: float
    model_name: str


class CacheService:
    """
    Async Redis cache wrapper.

    Constructor takes the redis client so the service is independently
    instantiable and testable — no app.state coupling.

    Usage:
        cache = CacheService(redis_client)
        hit = await cache.get(input_hash)
        if hit:
            return hit.label  # served from cache
        # ... run model ...
        await cache.set(input_hash, result)
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client
        self._ttl = settings.cache_ttl_seconds

    def _make_key(self, input_hash: str) -> str:
        """
        Namespace all cache keys under 'prediction:' to avoid collisions
        if this Redis instance is shared with other services.

        Key format: prediction:{sha256_of_input}
        Example:    prediction:a3f5c2d1...

        Why namespace?
          If you ever add a second service (e.g., a summarisation model)
          to the same Redis instance, their SHA-256 hashes could collide.
          Namespacing costs nothing and prevents silent data corruption.
        """
        return f"prediction:{input_hash}"

    async def get(self, input_hash: str) -> CachedPrediction | None:
        """
        Attempts a cache lookup. Returns CachedPrediction on hit, None on miss or error.

        Never raises — Redis errors are logged and treated as cache misses.
        The model will run as fallback.
        """
        key = self._make_key(input_hash)
        try:
            raw = await self._client.get(key)
            if raw is None:
                logger.debug("Cache miss", extra={"cache_key": key})
                return None

            data = json.loads(raw)
            logger.debug("Cache hit", extra={"cache_key": key})
            return CachedPrediction(
                label=data["label"],
                score=data["score"],
                latency_ms=data["latency_ms"],
                model_name=data["model_name"],
            )

        except aioredis.RedisError as exc:
            logger.warning(
                "Redis GET failed — treating as cache miss",
                extra={"cache_key": key, "error": str(exc), "error_type": type(exc).__name__},
            )
            return None

        except (json.JSONDecodeError, KeyError) as exc:
            # Corrupt or schema-mismatched cache entry.
            # Delete it so the next request doesn't hit the same bad data.
            logger.error(
                "Corrupt cache entry — deleting",
                extra={"cache_key": key, "error": str(exc)},
            )
            await self.delete(input_hash)
            return None

    async def set(
        self,
        input_hash: str,
        label: str,
        score: float,
        latency_ms: float,
        model_name: str,
    ) -> bool:
        """
        Writes a prediction to cache with the configured TTL.
        Returns True on success, False on failure.

        Why store latency_ms in the cache?
          When we serve a cache hit, the response still includes latency_ms.
          We store the ORIGINAL model latency so the caller can see
          "this result was originally computed in 42ms" — useful for
          detecting if the cached version came from a slower model version.
        """
        key = self._make_key(input_hash)
        payload = json.dumps({
            "label": label,
            "score": score,
            "latency_ms": latency_ms,
            "model_name": model_name,
        })

        try:
            await self._client.setex(key, self._ttl, payload)
            logger.debug(
                "Cache write",
                extra={"cache_key": key, "ttl_seconds": self._ttl},
            )
            return True

        except aioredis.RedisError as exc:
            logger.warning(
                "Redis SET failed — prediction will not be cached",
                extra={"cache_key": key, "error": str(exc)},
            )
            return False

    async def delete(self, input_hash: str) -> bool:
        """
        Explicitly evicts a cache entry.
        Used for: corrupt entry cleanup, post-retrain cache invalidation.

        Post-retrain invalidation is important:
          After retraining, the new model may produce different predictions
          for the same inputs. Old cached results from the previous model
          would be silently served, masking the retrain's effect.
          The CI retrain workflow should call this (or flush the entire
          prediction: namespace) after promoting a new model.
        """
        key = self._make_key(input_hash)
        try:
            deleted = await self._client.delete(key)
            logger.info(
                "Cache entry deleted",
                extra={"cache_key": key, "existed": deleted > 0},
            )
            return deleted > 0
        except aioredis.RedisError as exc:
            logger.warning(
                "Redis DELETE failed",
                extra={"cache_key": key, "error": str(exc)},
            )
            return False

    async def flush_all_predictions(self) -> int:
        """
        Evicts ALL prediction cache entries (prediction:* pattern).
        Called by the CI pipeline after a model promotion to force
        fresh inference with the new model.

        Returns the number of keys deleted.

        Why SCAN instead of KEYS?
          KEYS blocks Redis while it scans the entire keyspace.
          On a large Redis instance, KEYS "prediction:*" can freeze
          the server for seconds. SCAN is non-blocking and cursor-based.
          This is a common Redis footgun — NEVER use KEYS in production.
        """
        deleted_count = 0
        try:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match="prediction:*",
                    count=100,   # Process 100 keys per iteration
                )
                if keys:
                    deleted_count += await self._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(
                "Cache flushed after model promotion",
                extra={"deleted_keys": deleted_count},
            )
            return deleted_count

        except aioredis.RedisError as exc:
            logger.error(
                "Cache flush failed",
                extra={"error": str(exc)},
            )
            return 0


def build_cache_service(redis_client: aioredis.Redis) -> CacheService:
    """
    Factory function. Called from dependencies.py or directly in tests.
    """
    return CacheService(redis_client)
