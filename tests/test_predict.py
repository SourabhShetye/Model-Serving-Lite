"""
tests/test_predict.py

Tests for the POST /predict/ endpoint.

Test groups:
  TestPredictHappyPath   — valid inputs, correct response shape
  TestPredictCaching     — cache hit/miss behaviour
  TestPredictValidation  — input validation (empty, too long, whitespace)
  TestPredictNoCacheMode — correct fallback when Redis is unavailable
  TestPredictHeaders     — X-Request-ID and X-Latency-Ms header injection
  TestPredictDriftIntegration — drift service receives observations

Coverage targets:
  - Every branch in predict() route handler
  - Cache hit and miss code paths independently
  - Redis failure fallthrough (fail-open behaviour)
  - Input validation for all documented edge cases
  - Background task (DB write) is triggered — not awaited in tests
"""

import pytest
from conftest import FakeRedis  # noqa: F401 — used in type hints
from httpx import AsyncClient


PREDICT_URL = "/predict/"
POSITIVE_TEXT = "This product is absolutely great and I love it!"
NEGATIVE_TEXT = "Complete disaster, terrible experience overall."
NEUTRAL_TEXT = "The item arrived in a box on Tuesday."


# ------------------------------------------------------------------ #
# Happy Path                                                           #
# ------------------------------------------------------------------ #


class TestPredictHappyPath:
    @pytest.mark.asyncio
    async def test_returns_200(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        required = {
            "request_id",
            "label",
            "confidence",
            "input_hash",
            "latency_ms",
            "model_name",
            "cache_hit",
        }
        assert required.issubset(
            data.keys()
        ), f"Missing fields: {required - data.keys()}"

    @pytest.mark.asyncio
    async def test_positive_text_returns_positive_label(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert data["label"] == "POSITIVE"

    @pytest.mark.asyncio
    async def test_negative_text_returns_negative_label(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": NEGATIVE_TEXT})).json()
        assert data["label"] == "NEGATIVE"

    @pytest.mark.asyncio
    async def test_confidence_is_between_0_and_1(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_request_id_is_uuid_format(self, client: AsyncClient):
        import re

        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(
            uuid_pattern, data["request_id"]
        ), f"request_id is not UUID format: {data['request_id']}"

    @pytest.mark.asyncio
    async def test_input_hash_is_sha256(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        # SHA-256 hex digest is always 64 characters
        assert len(data["input_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in data["input_hash"])

    @pytest.mark.asyncio
    async def test_latency_ms_is_positive(self, client: AsyncClient):
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert data["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_mock_model_latency_is_42ms(self, client: AsyncClient):
        """
        Verifies the MockModelService is being used (not a real model).
        MockModelService.predict() always returns latency_ms=42.0.
        If a real model were used, latency would vary and likely be > 42ms.
        """
        data = (await client.post(PREDICT_URL, json={"text": NEGATIVE_TEXT})).json()
        assert (
            data["latency_ms"] == 42.0
        ), "Expected mock model latency of 42.0ms — real model may have loaded"

    @pytest.mark.asyncio
    async def test_same_input_produces_same_hash(self, client: AsyncClient):
        """Input hash must be deterministic for the cache to work correctly."""
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        # Clear the cache so second request is also a miss
        r2 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert r1["input_hash"] == r2["input_hash"]

    @pytest.mark.asyncio
    async def test_different_inputs_produce_different_hashes(self, client: AsyncClient):
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        r2 = (await client.post(PREDICT_URL, json={"text": NEGATIVE_TEXT})).json()
        assert r1["input_hash"] != r2["input_hash"]

    @pytest.mark.asyncio
    async def test_each_request_has_unique_request_id(self, client: AsyncClient):
        """Even identical inputs must produce unique request_ids."""
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        r2 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert r1["request_id"] != r2["request_id"]


# ------------------------------------------------------------------ #
# Caching                                                              #
# ------------------------------------------------------------------ #


class TestPredictCaching:
    @pytest.mark.asyncio
    async def test_first_request_is_cache_miss(self, client: AsyncClient, fake_redis):
        fake_redis.clear()
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert data["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_second_identical_request_is_cache_hit(
        self, client: AsyncClient, fake_redis
    ):
        fake_redis.clear()
        await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        data = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert data["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_cache_hit_returns_same_label(self, client: AsyncClient, fake_redis):
        fake_redis.clear()
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        r2 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert r1["label"] == r2["label"]

    @pytest.mark.asyncio
    async def test_cache_hit_returns_same_confidence(
        self, client: AsyncClient, fake_redis
    ):
        fake_redis.clear()
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        r2 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert r1["confidence"] == r2["confidence"]

    @pytest.mark.asyncio
    async def test_cache_hit_has_unique_request_id(
        self, client: AsyncClient, fake_redis
    ):
        """Cache hits must still get a fresh request_id for log correlation."""
        fake_redis.clear()
        r1 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        r2 = (await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})).json()
        assert r1["request_id"] != r2["request_id"]

    @pytest.mark.asyncio
    async def test_different_texts_do_not_share_cache(
        self, client: AsyncClient, fake_redis
    ):
        fake_redis.clear()
        await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        data = (await client.post(PREDICT_URL, json={"text": NEGATIVE_TEXT})).json()
        assert data["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_text_whitespace_stripped_before_hashing(
        self, client: AsyncClient, fake_redis
    ):
        """
        Leading/trailing whitespace is stripped by the validator.
        '  great product  ' and 'great product' should share the same cache entry.
        """
        fake_redis.clear()
        text = "great product"
        await client.post(PREDICT_URL, json={"text": text})
        data = (await client.post(PREDICT_URL, json={"text": f"  {text}  "})).json()
        assert data["cache_hit"] is True


# ------------------------------------------------------------------ #
# No-Cache Mode (Redis unavailable)                                    #
# ------------------------------------------------------------------ #


class TestPredictNoCacheMode:
    @pytest.mark.asyncio
    async def test_predict_works_without_redis(self, no_cache_client: AsyncClient):
        """
        Verifies fail-open behaviour: predictions work when Redis is None.
        This is the core contract — cache is an optimisation, not a dependency.
        """
        response = await no_cache_client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_no_cache_response_has_correct_fields(
        self, no_cache_client: AsyncClient
    ):
        data = (
            await no_cache_client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        ).json()
        assert "label" in data
        assert "confidence" in data
        assert data["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_no_cache_always_returns_cache_miss(
        self, no_cache_client: AsyncClient
    ):
        """Without Redis, every request is a cache miss — model is always invoked."""
        r1 = (
            await no_cache_client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        ).json()
        r2 = (
            await no_cache_client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        ).json()
        assert r1["cache_hit"] is False
        assert r2["cache_hit"] is False


# ------------------------------------------------------------------ #
# Input Validation                                                     #
# ------------------------------------------------------------------ #


class TestPredictValidation:
    @pytest.mark.asyncio
    async def test_empty_string_returns_422(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_422(self, client: AsyncClient):
        """Whitespace-only strings pass min_length=1 but fail the strip validator."""
        response = await client.post(PREDICT_URL, json={"text": "   "})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_text_too_long_returns_422(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": "x" * 513})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_text_at_max_length_returns_200(self, client: AsyncClient):
        """512 characters is the documented maximum — must be accepted."""
        response = await client.post(PREDICT_URL, json={"text": "good " * 102})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_text_field_returns_422(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"wrong_field": "hello"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_text_returns_422(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": None})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_single_character_text_returns_200(self, client: AsyncClient):
        """Single non-whitespace character is a valid input."""
        response = await client.post(PREDICT_URL, json={"text": "A"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_non_json_body_returns_422(self, client: AsyncClient):
        response = await client.post(
            PREDICT_URL,
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_422_response_has_detail_field(self, client: AsyncClient):
        """Validation errors must include a detail field for the client to read."""
        data = (await client.post(PREDICT_URL, json={"text": ""})).json()
        assert "detail" in data


# ------------------------------------------------------------------ #
# Response Headers                                                     #
# ------------------------------------------------------------------ #


class TestPredictHeaders:
    @pytest.mark.asyncio
    async def test_x_request_id_header_present(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        assert "x-request-id" in response.headers

    @pytest.mark.asyncio
    async def test_x_request_id_matches_body(self, client: AsyncClient):
        """
        The X-Request-ID header and the request_id in the body must match.
        This is how a customer links their network trace to a log line.
        """
        response = await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        body_id = response.json()["request_id"]
        header_id = response.headers["x-request-id"]
        assert body_id == header_id

    @pytest.mark.asyncio
    async def test_x_latency_ms_header_present(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        assert "x-latency-ms" in response.headers

    @pytest.mark.asyncio
    async def test_x_latency_ms_is_numeric(self, client: AsyncClient):
        response = await client.post(PREDICT_URL, json={"text": POSITIVE_TEXT})
        latency = float(response.headers["x-latency-ms"])
        assert latency > 0


# ------------------------------------------------------------------ #
# Drift Integration                                                    #
# ------------------------------------------------------------------ #


class TestPredictDriftIntegration:
    @pytest.mark.asyncio
    async def test_prediction_increments_drift_observations(self, client: AsyncClient):
        """
        Each non-cached prediction should be recorded by the DriftService.
        We verify via the /drift/status endpoint.
        """
        before = (await client.get("/drift/status")).json()
        obs_before = before["total_observations"]

        # Send a prediction with a unique text (to avoid cache hit)
        import uuid

        await client.post(
            PREDICT_URL, json={"text": f"Unique test text {uuid.uuid4()}"}
        )

        after = (await client.get("/drift/status")).json()
        obs_after = after["total_observations"]

        assert (
            obs_after == obs_before + 1
        ), f"Expected total_observations to increment by 1: {obs_before} → {obs_after}"

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_increment_drift_observations(
        self, client: AsyncClient, fake_redis: "FakeRedis"
    ):
        """
        Cache hits don't represent new incoming data distributions.
        The drift monitor should only record model invocations, not cache replays.
        """
        fake_redis.clear()
        text = "Great excellent wonderful product!"

        # First request: cache miss → drift records it
        await client.post(PREDICT_URL, json={"text": text})
        obs_after_first = (await client.get("/drift/status")).json()[
            "total_observations"
        ]

        # Second request: cache hit → drift should NOT record it
        await client.post(PREDICT_URL, json={"text": text})
        obs_after_second = (await client.get("/drift/status")).json()[
            "total_observations"
        ]

        assert (
            obs_after_first == obs_after_second
        ), "Cache hit should not increment drift total_observations"
