"""
app/middleware/logging_middleware.py

Structured JSON request/response logging middleware.

Why middleware instead of logging inside each route handler?
  Route handlers shouldn't care about observability plumbing.
  More practically: middleware runs for EVERY route — including ones
  you add later and forget to instrument. It's a forcing function.

Why ASGI BaseHTTPMiddleware instead of a route decorator?
  BaseHTTPMiddleware wraps at the ASGI transport layer, which means:
    - It captures the true wall-clock latency including serialization.
    - It catches unhandled exceptions BEFORE they become 500 responses.
    - It runs even if a dependency injection step fails.

What does every log line contain?
  {
    "timestamp": "2024-01-15T10:23:41",
    "level": "INFO",
    "logger": "app.middleware.logging_middleware",
    "message": "request_complete",
    "request_id": "a3f5-...",      ← UUID, returned in response header too
    "method": "POST",
    "path": "/predict/",
    "status_code": 200,
    "latency_ms": 312.4,           ← wall-clock, includes serialisation
    "client_ip": "203.0.113.5",
    "user_agent": "curl/8.1.2",
    "content_length_bytes": 42,    ← request body size
    "error": null                  ← populated on 4xx/5xx
  }

With this in Render logs (or any log aggregator), you can:
  grep '"request_id": "a3f5"'    → find every log line for one request
  grep '"status_code": 500'      → find all failures
  grep '"latency_ms"' | jq ...   → compute p99 from raw logs

That's the 'debug a bad prediction a week later' story from the brief.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths we deliberately skip logging (noise reduction)
# Health checks fire every 10 seconds from the load balancer — log them
# at DEBUG only, not INFO, so they don't drown out real traffic.
_NOISY_PATHS = {"/health", "/metrics"}


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured JSON log line per request.

    The X-Request-ID header is injected into the response so the
    caller can include it in bug reports — giving you a direct key
    to search logs with.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ------------------------------------------------------------ #
        # 1. Assign a request ID before anything else                   #
        # ------------------------------------------------------------ #
        # Check if the upstream proxy/load balancer already set one.
        # Render and most cloud LBs send X-Request-ID. Reuse it for
        # end-to-end tracing (LB log → app log → DB row → response header).
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Store on request.state so route handlers can access it
        # without re-parsing the header.
        request.state.request_id = request_id

        # ------------------------------------------------------------ #
        # 2. Capture request metadata                                   #
        # ------------------------------------------------------------ #
        client_ip = self._get_client_ip(request)
        t_start = time.perf_counter()

        # ------------------------------------------------------------ #
        # 3. Call the actual route handler                              #
        # ------------------------------------------------------------ #
        status_code = 500   # Default: assume failure until we hear otherwise
        error_detail: str | None = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code

            # Capture error detail from 4xx/5xx responses for the log.
            # We don't read the body on success — that would consume the
            # stream and break the response.
            if status_code >= 400:
                error_detail = f"HTTP {status_code}"

        except Exception as exc:
            # Unhandled exception — log it and re-raise so FastAPI's
            # exception handlers can return a proper 500 response.
            status_code = 500
            error_detail = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Unhandled exception in request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise

        finally:
            # -------------------------------------------------------- #
            # 4. Emit the structured log line                           #
            # This runs whether the request succeeded or failed.        #
            # -------------------------------------------------------- #
            latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

            log_payload = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_string": str(request.url.query) or None,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent"),
                "content_length_bytes": request.headers.get("content-length"),
                "error": error_detail,
            }

            # Reduce noise: health checks log at DEBUG
            is_noisy = request.url.path in _NOISY_PATHS
            log_level = logging.DEBUG if is_noisy else logging.INFO
            log_msg = "request_complete"

            logger.log(log_level, log_msg, extra=log_payload)

        # ------------------------------------------------------------ #
        # 5. Inject request ID into response headers                    #
        # ------------------------------------------------------------ #
        # This is what makes the request_id useful in practice.
        # When a customer reports a problem, you ask:
        # "What does X-Request-ID say in your browser's Network tab?"
        # Then: grep '"request_id": "<that-value>"' in your logs.
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency_ms)

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extracts the real client IP, accounting for reverse proxies.

        Render (and most cloud platforms) set X-Forwarded-For.
        Without this, all requests appear to come from the proxy's IP.

        Security note: X-Forwarded-For can be spoofed by clients.
        For rate limiting, use the rightmost IP (trusted proxy chain).
        For logging purposes, the leftmost (claimed origin) is fine.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # X-Forwarded-For: client, proxy1, proxy2
            # Take the leftmost (original client)
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
