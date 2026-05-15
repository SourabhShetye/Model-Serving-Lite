"""
app/main.py

FastAPI application factory.

The lifespan pattern (contextlib.asynccontextmanager) replaced the deprecated
@app.on_event("startup") / @app.on_event("shutdown") in FastAPI v0.95.

Why lifespan over on_event?
  1. It's a single block — startup AND shutdown logic are co-located.
     You can't forget to clean up because the cleanup is right below the setup.
  2. It uses standard Python context manager semantics (yield).
  3. It's tested and recommended by the FastAPI maintainers going forward.
  4. It makes the startup/shutdown sequence explicit and readable.

Startup sequence (order matters):
  1. Settings validated (happens at import time via get_settings())
  2. Logging configured
  3. Redis connected (optional — app continues if this fails)
  4. Model loaded (blocking — app doesn't accept traffic until this finishes)
  5. Readiness probe starts returning 200

Shutdown sequence (yield resumes):
  1. Redis connection closed cleanly
  2. (Model cleanup handled by GC — HuggingFace pipelines don't need explicit teardown)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.logging_middleware import StructuredLoggingMiddleware
from app.db.crud import create_tables, check_connection
from app.routers import health, predict, drift
from app.services.model_service import ModelService, load_pipeline

settings = get_settings()


# ------------------------------------------------------------------ #
# Logging Setup                                                        #
# ------------------------------------------------------------------ #
# Configure BEFORE anything else so all startup log messages are captured.
# We use python-json-logger here so every log line is valid JSON from
# the very first message. This is what structured log aggregators expect.


def configure_logging() -> None:
    """
    Sets up JSON structured logging for the entire application.

    Why configure here and not in each module?
      Each module does `logger = logging.getLogger(__name__)`.
      The root logger configuration here propagates to all of them.
      One place to change the format for the entire service.
    """
    from pythonjsonlogger import jsonlogger

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove any default handlers (important in some container environments)
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


configure_logging()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Lifespan                                                             #
# ------------------------------------------------------------------ #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the full lifecycle of the application.

    Code BEFORE yield = startup.
    Code AFTER yield = shutdown.
    """
    logger.info(
        "Starting up",
        extra={"environment": settings.environment, "version": settings.app_version},
    )
    app.state.startup_time = time.time()

    # ---------------------------------------------------------------- #
    # 1. Connect to Redis                                               #
    # ---------------------------------------------------------------- #
    redis_client = None
    if settings.cache_enabled:
        try:
            redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=10,  # Don't hang for 30 seconds if Redis is down
                socket_timeout=5,  # Individual command timeout
            )
            await redis_client.ping()
            app.state.redis_client = redis_client
            logger.info("Redis connected", extra={"url": settings.redis_url})
        except Exception as exc:
            # Non-fatal: log and continue without cache
            logger.warning(
                "Redis connection failed — cache disabled",
                extra={"url": settings.redis_url, "error": str(exc)},
            )
            app.state.redis_client = None
    else:
        app.state.redis_client = None
        logger.info("Cache disabled by config (CACHE_ENABLED=false)")

    # ---------------------------------------------------------------- #
    # 2. Initialise database tables                                     #
    # ---------------------------------------------------------------- #
    try:
        create_tables()
        db_ok = check_connection()
        app.state.db_available = db_ok
        if not db_ok:
            logger.warning("Database unreachable — prediction logging to DB disabled")
    except Exception as exc:
        logger.warning(
            "Database init failed — prediction logging to DB disabled",
            extra={"error": str(exc)},
        )
        app.state.db_available = False

    # ---------------------------------------------------------------- #
    # 3. Load model (blocking — intentional)                            #
    # ---------------------------------------------------------------- #
    # The /ready endpoint returns 503 until this completes.
    # Render's health check will hold traffic until we return 200.
    # This is correct behaviour — we don't want to serve requests before
    # the model is in memory.
    try:
        pipe = load_pipeline()
        app.state.model_service = ModelService(pipe)
        logger.info("Application ready to serve traffic")
    except Exception as exc:
        logger.critical(
            "Failed to load model — cannot start",
            extra={"model_name": settings.model_name, "error": str(exc)},
        )
        # Re-raise: a failed model load is unrecoverable. Crash loudly.
        raise

    # ---------------------------------------------------------------- #
    # Serve traffic (yield hands control to FastAPI)                    #
    # ---------------------------------------------------------------- #
    yield

    # ---------------------------------------------------------------- #
    # Shutdown (after yield)                                            #
    # ---------------------------------------------------------------- #
    logger.info("Shutting down")
    if redis_client is not None:
        await redis_client.aclose()
        logger.info("Redis connection closed")


# ------------------------------------------------------------------ #
# App Factory                                                          #
# ------------------------------------------------------------------ #


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Why a factory function instead of a module-level `app = FastAPI()`?
      - Testable: tests call create_app() and get a fresh instance each time.
      - Configurable: you can pass different settings/lifespan in tests.
      - Explicit: all app configuration is in one place.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production sentiment analysis service. "
            "Includes request logging, Redis caching, and drift monitoring."
        ),
        docs_url="/docs",  # Swagger UI — useful during the walkthrough demo
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------- #
    # Middleware (added in reverse order — last added = first to run)   #
    # ---------------------------------------------------------------- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(StructuredLoggingMiddleware)

    # ---------------------------------------------------------------- #
    # Routers                                                            #
    # ---------------------------------------------------------------- #
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(drift.router)

    return app


# The module-level `app` is what uvicorn imports.
# `uvicorn app.main:app` — this is the entry point.
app = create_app()
