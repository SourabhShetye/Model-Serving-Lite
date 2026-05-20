import pytest

from app.services.in_memory_cache import InMemoryRedis


class TestInMemoryRedis:
    @pytest.mark.asyncio
    async def test_get_after_setex_returns_value(self):
        cache = InMemoryRedis()

        await cache.setex("prediction:1", 10, "value")

        assert await cache.get("prediction:1") == "value"

    @pytest.mark.asyncio
    async def test_expired_key_is_removed_on_get(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(
            "app.services.in_memory_cache.time.time", lambda: current_time
        )

        cache = InMemoryRedis()
        await cache.setex("prediction:expired", 1, "value")
        assert await cache.get("prediction:expired") == "value"

        monkeypatch.setattr(
            "app.services.in_memory_cache.time.time", lambda: current_time + 2.0
        )
        assert await cache.get("prediction:expired") is None

    @pytest.mark.asyncio
    async def test_scan_returns_matching_keys_and_skips_expired(self, monkeypatch):
        current_time = 2000.0
        monkeypatch.setattr(
            "app.services.in_memory_cache.time.time", lambda: current_time
        )

        cache = InMemoryRedis()
        await cache.setex("prediction:one", 10, "one")
        await cache.setex("prediction:expired", 1, "expired")
        await cache.setex("other:two", 10, "two")

        monkeypatch.setattr(
            "app.services.in_memory_cache.time.time", lambda: current_time + 2.0
        )
        cursor, keys = await cache.scan(cursor=0, match="prediction:*", count=100)

        assert cursor == 0
        assert keys == ["prediction:one"]

    @pytest.mark.asyncio
    async def test_delete_returns_number_of_deleted_keys(self):
        cache = InMemoryRedis()
        await cache.setex("prediction:one", 10, "one")
        await cache.setex("prediction:two", 10, "two")

        deleted = await cache.delete("prediction:one", "missing")

        assert deleted == 1
        assert await cache.get("prediction:one") is None

    @pytest.mark.asyncio
    async def test_setex_with_non_positive_ttl_stores_value_permanently(self):
        cache = InMemoryRedis()
        await cache.setex("prediction:forever", 0, "forever")

        assert await cache.get("prediction:forever") == "forever"

    @pytest.mark.asyncio
    async def test_setex_with_none_ttl_stores_value_permanently(self):
        cache = InMemoryRedis()
        await cache.setex("prediction:forever-none", None, "forever")

        assert await cache.get("prediction:forever-none") == "forever"
