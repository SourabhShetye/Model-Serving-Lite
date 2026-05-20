"""
app/services/in_memory_cache.py

Simple in-memory async-compatible Redis-like client used as a fallback
when a real Redis server is unavailable. Mirrors the small subset of the
`redis.asyncio.Redis` API used by `CacheService` and the rest of the app:
  - ping()
  - get(key)
  - setex(key, ttl, value)
  - delete(*keys)
  - scan(cursor, match, count)
  - aclose()

This implementation is intentionally lightweight and intended for demos
and local testing only. TTLs are honoured in-memory for the duration of
the process. It is NOT persisted across restarts.
"""

from __future__ import annotations

import fnmatch
import time
from typing import Dict, List, Optional, Tuple


class InMemoryRedis:
    """A tiny async-compatible in-memory Redis-like client.

    Values are stored as strings (JSON payloads in our app). TTL is
    supported by tracking expiry timestamps per key.
    """

    def __init__(self) -> None:
        # store: key -> (value: str, expiry_ts: Optional[float])
        self._store: Dict[str, Tuple[str, Optional[float]]] = {}

    async def ping(self) -> bool:  # pragma: no cover - trivial
        return True

    async def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and time.time() >= expiry:
            # expired
            self._store.pop(key, None)
            return None
        return value

    async def setex(self, key: str, ttl: int | None, value: str) -> None:
        expiry = None if ttl is None or ttl <= 0 else (time.time() + float(ttl))
        self._store[key] = (value, expiry)

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for k in keys:
            if k in self._store:
                self._store.pop(k, None)
                deleted += 1
        return deleted

    async def scan(
        self,
        cursor: int = 0,
        match: str = "*",
        count: int = 100,
    ) -> tuple[int, list[str]]:
        # Simplified: return all matching keys in a single pass and signal done
        now = time.time()
        keys: List[str] = []
        for k, (v, expiry) in list(self._store.items()):
            if expiry is not None and now >= expiry:
                # lazy-expire
                self._store.pop(k, None)
                continue
            if fnmatch.fnmatch(k, match):
                keys.append(k)
        return (0, keys)

    async def aclose(self) -> None:  # pragma: no cover - trivial
        # nothing to clean up for in-memory store
        return None

    # Helper used by tests/demos
    def clear(self) -> None:
        self._store.clear()
