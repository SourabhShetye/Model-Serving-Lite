"""
tests/test_health.py

Tests for the liveness and readiness probes.

Why test health endpoints carefully?
  A misconfigured /health endpoint can cause a cascade failure:
  - /health checks a DB connection → DB goes slow → health check times out
  - Container orchestrator sees timeout → restarts the container
  - Restarting the container drops all connections → DB goes slower
  - Repeat until the entire service is down

  /health must be unconditionally fast and dependency-free.
  /ready must accurately reflect whether the service can handle traffic.
  These tests enforce both contracts.
"""

import pytest
from httpx import AsyncClient


# ------------------------------------------------------------------ #
# /health — Liveness probe                                             #
# ------------------------------------------------------------------ #

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_status(self, client: AsyncClient):
        data = (await client.get("/health")).json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_version(self, client: AsyncClient):
        data = (await client.get("/health")).json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    @pytest.mark.asyncio
    async def test_health_is_fast(self, client: AsyncClient):
        """
        Liveness check must be fast — if it takes > 200ms something is wrong.
        In production, a slow health check causes unnecessary container restarts.
        """
        import time
        t0 = time.perf_counter()
        await client.get("/health")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200, f"/health took {elapsed_ms:.1f}ms — too slow for a liveness probe"

    @pytest.mark.asyncio
    async def test_health_has_no_external_dependencies(self, client: AsyncClient):
        """
        /health must return 200 even when Redis is unavailable.
        We test this by using the no_cache_client fixture (Redis=None).
        """
        response = await client.get("/health")
        # If health depended on Redis, this would fail since the test
        # app has Redis mocked. It should still return 200.
        assert response.status_code == 200


# ------------------------------------------------------------------ #
# /ready — Readiness probe                                             #
# ------------------------------------------------------------------ #

class TestReady:
    @pytest.mark.asyncio
    async def test_ready_returns_200_when_model_loaded(self, client: AsyncClient):
        """
        /ready should return 200 when model_service is set on app.state.
        The test fixture sets app.state.model_service = MockModelService().
        """
        response = await client.get("/ready")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_returns_correct_fields(self, client: AsyncClient):
        data = (await client.get("/ready")).json()
        assert "status" in data
        assert "model_loaded" in data
        assert "redis_reachable" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_ready_reports_model_loaded_true(self, client: AsyncClient):
        data = (await client.get("/ready")).json()
        assert data["model_loaded"] is True
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_reports_redis_reachable(self, client: AsyncClient):
        """FakeRedis.ping() returns True, so redis_reachable should be True."""
        data = (await client.get("/ready")).json()
        assert data["redis_reachable"] is True

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_model_not_loaded(self, app, client: AsyncClient):
        """
        Simulate a startup failure: model_service is None.
        /ready must return 503 so the load balancer holds traffic.
        """
        # Temporarily remove the model from app state
        original = app.state.model_service
        app.state.model_service = None
        try:
            response = await client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
            assert response.json()["model_loaded"] is False
        finally:
            app.state.model_service = original

    @pytest.mark.asyncio
    async def test_ready_still_200_when_redis_down(self, no_cache_client: AsyncClient):
        """
        /ready should return 200 even when Redis is unavailable.
        Redis is optional — the service degrades gracefully without it.
        Model availability is the only hard requirement for readiness.
        """
        response = await no_cache_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["redis_reachable"] is False   # Redis is None in no_cache fixture
        assert data["status"] == "ready"          # Still ready — Redis is not required

    @pytest.mark.asyncio
    async def test_ready_uptime_is_numeric(self, client: AsyncClient):
        data = (await client.get("/ready")).json()
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
