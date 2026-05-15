"""
app/services/model_service.py

HuggingFace pipeline wrapper with a clean interface.

Design decisions:
  1. This module knows NOTHING about FastAPI — it's pure Python.
     That means it's independently unit-testable without spinning up
     a server. A critical distinction most juniors miss.

  2. The pipeline is NOT a global variable. It gets instantiated once
     during the app lifespan and stored on app.state. This is the
     FastAPI-idiomatic pattern since v0.95 (lifespan replaces on_event).

  3. We return a typed dataclass from predict(), not a raw dict.
     The router is responsible for serialising to JSON — not this service.
"""

import hashlib
import logging
import time
import os

from dataclasses import dataclass

from transformers import pipeline, Pipeline

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
os.environ["TRANSFORMERS_CACHE"] = settings.model_cache_dir


@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable result object. frozen=True means no accidental mutation
    downstream. The router turns this into a Pydantic response model.
    """

    label: str  # "POSITIVE" or "NEGATIVE"
    score: float  # Confidence, 0.0–1.0
    input_hash: str  # SHA-256 of input text — used as cache key and log correlation ID
    latency_ms: float  # Time taken inside the model (excludes cache overhead)
    model_name: (
        str  # Which model produced this — critical for debugging after a retrain
    )


def build_input_hash(text: str) -> str:
    """
    Deterministic SHA-256 hash of the input text.

    Why SHA-256 and not just the text as a key?
      - Inputs can be arbitrarily long; Redis keys should be short and fixed-length.
      - The hash IS the cache key AND the log correlation ID. Given an input,
        you can always reconstruct the key — no lookup table needed.
      - SHA-256 is collision-resistant enough that we will never have a false
        cache hit in practice.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pipeline() -> Pipeline:
    """
    Loads the HuggingFace pipeline. Called once during app startup.

    Why device=-1 explicitly?
      We're on CPU-only free-tier compute (Render). Setting device=-1
      avoids an unnecessary CUDA detection scan on startup and makes the
      log output clean.

    Why return_all_scores=False?
      We only need the top prediction. Returning all scores doubles the
      payload size for no operational benefit on this model.
    """
    logger.info(
        "Loading model pipeline",
        extra={
            "model_name": settings.model_name,
            "cache_dir": settings.model_cache_dir,
        },
    )

    t0 = time.perf_counter()

    pipe = pipeline(
        task="sentiment-analysis",
        model=settings.model_name,
        device=-1,
        top_k=1,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "Model pipeline ready",
        extra={"model_name": settings.model_name, "load_time_ms": round(elapsed, 2)},
    )
    return pipe


class ModelService:
    """
    Thin wrapper around the HuggingFace pipeline.

    Why a class and not just a function?
      We need to carry state: the pipeline instance and the model name.
      A class makes that explicit and testable — you can instantiate
      ModelService(mock_pipeline) in tests without monkey-patching globals.
    """

    def __init__(self, pipe: Pipeline) -> None:
        self._pipe = pipe
        self._model_name = settings.model_name

    def predict(self, text: str) -> PredictionResult:
        """
        Run inference. Returns a PredictionResult dataclass.

        Timing wraps only the pipeline call — not Redis or logging.
        That means latency_ms in your logs is model-only latency,
        which is what you want for performance regression detection
        after a retrain.
        """
        input_hash = build_input_hash(text)

        t0 = time.perf_counter()
        # pipeline() returns [[{"label": "POSITIVE", "score": 0.9998}]]
        # We asked for top_k=1, so it's always a single-element list of lists.
        raw = self._pipe(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Defensive unwrapping — log clearly if the shape is unexpected
        try:
            top = raw[0][0]
            label: str = top["label"]
            score: float = round(top["score"], 6)
        except (IndexError, KeyError, TypeError) as exc:
            logger.error(
                "Unexpected pipeline output shape",
                extra={"raw_output": str(raw), "error": str(exc)},
            )
            raise ValueError(f"Model returned unexpected output shape: {raw}") from exc

        return PredictionResult(
            label=label,
            score=score,
            input_hash=input_hash,
            latency_ms=round(latency_ms, 3),
            model_name=self._model_name,
        )
