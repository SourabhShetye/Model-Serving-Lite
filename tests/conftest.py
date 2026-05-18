"""
tests/conftest.py

Shared pytest fixtures for the entire test suite.

Architecture of the test setup:
  The key challenge is that app.main.create_app() triggers a lifespan
  that loads a 260MB HuggingFace model. We cannot do that in CI without
  torch installed and without waiting 3+ minutes per test run.

  Solution: dependency_overrides.
  FastAPI's dependency_overrides lets us replace any Depends() provider
  with a mock at test time. We override:
    - get_model_service  → MockModelService (deterministic, instant)
    - get_redis          → FakeRedis (in-memory, no server needed)
    - get_settings_dep   → test-specific settings

  The result: tests run in milliseconds, test 100% of route logic,
  and require zero external infrastructure.

  Integration tests that DO need Redis use the real Redis service
  spun up by the CI workflow (see ci.yml services block).
"""

import asyncio


from typing import AsyncGenerator


import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.dependencies import get_model_service, get_redis, get_settings_dep
from app.main import create_app
from app.services.drift_service import reset_drift_service
from app.services.model_service import ModelService, PredictionResult


# ------------------------------------------------------------------ #
# Settings override                                                    #
# ------------------------------------------------------------------ #

TEST_SETTINGS = Settings(
    app_name="sentiment-service-test",
    environment="development",
    log_level="WARNING",  # Suppress logs during tests
    redis_url="redis://localhost:6379/0",
    cache_enabled=True,
    database_url="sqlite:///./test_predictions.db",
    drift_window_size=10,  # Small window so drift tests are fast
    drift_ks_threshold=0.05,
    drift_confidence_drop_threshold=0.10,
    drift_language_threshold=0.30,
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    model_cache_dir="/tmp/hf_cache_test",
)


# ------------------------------------------------------------------ #
# Mock model service                                                   #
# ------------------------------------------------------------------ #


class MockModelService(ModelService):
    """
    Deterministic mock that never loads HuggingFace weights.

    Returns predictable results based on simple keyword matching:
      - Texts containing 'great', 'good', 'love', 'excellent' → POSITIVE
      - Everything else → NEGATIVE

    Why keyword-based and not always-POSITIVE?
      Tests that assert on the label value need predictable behaviour.
      A fixed POSITIVE response would make test_negative_sentiment
      trivially pass or fail regardless of route logic.

    Latency is fixed at 42.0ms — a memorable value that makes it
    easy to verify in assertions that the mock was used (not a real model).
    """

    def __init__(self) -> None:
        # Deliberately do NOT call super().__init__() — we don't want
        # to load a real pipeline in tests.
        self._model_name = "mock-model-test"

    def predict(self, text: str) -> PredictionResult:
        from app.services.model_service import build_input_hash

        positive_keywords = {
            "great",
            "good",
            "love",
            "excellent",
            "wonderful",
            "amazing",
            "fantastic",
            "best",
            "happy",
            "perfect",
        }
        words = set(text.lower().split())
        is_positive = bool(words & positive_keywords)

        return PredictionResult(
            label="POSITIVE" if is_positive else "NEGATIVE",
            score=0.9998 if is_positive else 0.9991,
            input_hash=build_input_hash(text),
            latency_ms=42.0,  # Fixed — easy to assert on
            model_name=self._model_name,
        )


# ------------------------------------------------------------------ #
# Fake Redis (in-memory, no server)                                    #
# ------------------------------------------------------------------ #


class FakeRedis:
    """
    In-memory Redis mock for unit tests.

    Implements the subset of the redis.asyncio.Redis API used by
    CacheService: get, setex, delete, scan, ping.

    Why not use fakeredis library?
      fakeredis is a heavy dependency and requires the redis package
      to be installed. This implementation covers only what we use
      and has zero dependencies — it's faster and more explicit.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value  # TTL not enforced in tests

    async def delete(self, *keys: str) -> int:
        deleted = sum(1 for k in keys if k in self._store)
        for k in keys:
            self._store.pop(k, None)
        return deleted

    async def scan(self, cursor: int, match: str = "*", count: int = 100):
        # Simplified scan — returns all matching keys in one pass
        import fnmatch

        matching = [k for k in self._store if fnmatch.fnmatch(k, match)]
        return (0, matching)  # cursor=0 signals "done"

    async def aclose(self) -> None:
        pass

    def clear(self) -> None:
        """Test helper: reset all stored keys between tests."""
        self._store.clear()


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(autouse=True)
def reset_drift():
    """
    Resets the DriftService singleton before every test.
    Prevents state leaking between tests that trigger drift recording.
    autouse=True means it runs automatically without explicit use.
    """
    reset_drift_service()
    yield
    reset_drift_service()


@pytest.fixture
def fake_redis() -> FakeRedis:
    """A fresh FakeRedis instance per test."""
    return FakeRedis()


@pytest.fixture
def mock_model() -> MockModelService:
    """A MockModelService instance."""
    return MockModelService()


@pytest.fixture
def app(mock_model: MockModelService, fake_redis: FakeRedis) -> FastAPI:
    """
    FastAPI app with all external dependencies replaced by mocks.

    This fixture:
      1. Creates the app via the factory (same code path as production)
      2. Manually sets app.state (bypasses lifespan — no real model load)
      3. Overrides all Depends() providers with mocks

    Tests using this fixture exercise 100% of route and middleware logic
    with zero external dependencies.
    """
    application = create_app()

    # Set state directly — bypasses the lifespan startup
    application.state.model_service = mock_model
    application.state.redis_client = fake_redis
    application.state.db_available = False  # Disable DB writes in unit tests
    application.state.startup_time = 0.0

    # Override dependency providers
    application.dependency_overrides[get_model_service] = lambda: mock_model
    application.dependency_overrides[get_redis] = lambda: fake_redis
    application.dependency_overrides[get_settings_dep] = lambda: TEST_SETTINGS

    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client.

    Uses ASGITransport so requests go directly to the ASGI app
    without starting a real HTTP server.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def no_cache_app(mock_model: MockModelService) -> FastAPI:
    """
    App variant with Redis explicitly disabled.
    Used to test the cache-miss code path in isolation.
    """
    application = create_app()
    application.state.model_service = mock_model
    application.state.redis_client = None  # Simulates Redis being unavailable
    application.state.db_available = False
    application.state.startup_time = 0.0

    application.dependency_overrides[get_model_service] = lambda: mock_model
    application.dependency_overrides[get_redis] = lambda: None
    application.dependency_overrides[get_settings_dep] = lambda: TEST_SETTINGS

    return application


@pytest_asyncio.fixture
async def no_cache_client(no_cache_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with Redis disabled."""
    async with AsyncClient(
        transport=ASGITransport(app=no_cache_app),
        base_url="http://test",
    ) as ac:
        yield ac
