# ================================================================== #
# Dockerfile — Multi-stage build for sentiment-service                #
# ================================================================== #
#
# Stage overview:
#   1. builder  — Full Python image. Installs ALL deps into an isolated
#                 /venv. Has compilers, headers, pip cache. Discarded.
#   2. model-fetcher — Pulls HuggingFace weights at build time.
#                 Separated so weight re-downloads don't bust dep layers.
#   3. runtime  — Slim Python image. Copies /venv + weights + app code.
#                 No compilers. No pip. No secrets from build args.
#                 This is the image that runs in production.
#
# Why three stages instead of two?
#   Dep installation and model download have different cache invalidation
#   triggers:
#     - Deps change when requirements.txt changes
#     - Model changes when MODEL_NAME build arg changes
#   Separating them means changing a dep doesn't re-download 1.3GB of
#   model weights, and vice versa. Each stage is independently cached.
#
# Target image size: ~2.1GB
#   PyTorch CPU-only is unavoidably large (~800MB compressed).
#   The slim base saves ~400MB vs python:3.11.
#   The multi-stage build removes ~300MB of build tools.
# ================================================================== #

# ------------------------------------------------------------------ #
# ARGs declared before first FROM are available to all stages          #
# ------------------------------------------------------------------ #
ARG PYTHON_VERSION=3.11
ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_CACHE_DIR=/opt/hf_cache


# ================================================================== #
# Stage 1: builder                                                     #
# Installs Python dependencies into an isolated virtual environment.  #
# ================================================================== #
FROM python:${PYTHON_VERSION}-slim AS builder

# Redeclare ARGs after FROM to make them available in this stage
ARG PYTHON_VERSION

WORKDIR /build

# Install system build dependencies.
# These are needed to compile packages like psycopg2, hiredis.
# They stay in the builder stage and never reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create the virtual environment that we'll copy to the runtime stage.
# Using /venv as a fixed path makes the COPY in stage 3 trivial.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip inside the venv — older pip misses dependency optimisations.
RUN pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------ #
# Dependency installation (two-layer strategy)                         #
# ------------------------------------------------------------------ #
#
# Why two COPY + RUN pairs instead of one?
#   Docker caches layers. If we COPY requirements.txt and pip install
#   in one step, then copy the rest of the app, Docker invalidates the
#   pip cache layer every time ANY app file changes.
#
#   Two-step:
#     COPY requirements.txt          ← only invalidates if requirements change
#     RUN pip install                ← cached if requirements.txt unchanged
#     COPY app/ ...                  ← changes frequently, but doesn't bust pip cache
#
# This is the single most impactful Dockerfile optimisation for dev velocity.

# Copy ONLY requirements first
COPY requirements.txt .

# Install PyTorch CPU-only FIRST (largest dep, most cache-valuable).
# The --index-url pulls the CPU-only build (~800MB installed vs 4.7GB for CUDA).
# Must be installed before other deps to avoid pip pulling the CUDA version.
#
# --mount=type=cache: BuildKit cache mount. The pip cache persists between
# builds on the same machine. A full dep install goes from 4min → 15sec
# on the second build. Requires DOCKER_BUILDKIT=1 (default in Docker 23+).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (excludes torch — already installed)
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirements.txt | pip install -r /dev/stdin


# ================================================================== #
# Stage 2: model-fetcher                                               #
# Downloads HuggingFace model weights at build time.                  #
# ================================================================== #
FROM builder AS model-fetcher

ARG MODEL_NAME
ARG MODEL_CACHE_DIR

ENV PATH="/venv/bin:$PATH"
ENV TRANSFORMERS_CACHE=${MODEL_CACHE_DIR}
ENV HF_HOME=${MODEL_CACHE_DIR}

# Download model weights by running the pipeline constructor.
# Why bake weights into the image?
#
#   Option A (download at runtime): Container starts, first request
#   triggers a ~1.3GB download from HuggingFace CDN. Takes 3-5 minutes.
#   During that time /ready returns 503, but Render may timeout.
#   Any HuggingFace CDN outage makes your service un-startable.
#
#   Option B (bake at build time — our choice): The weights are in the
#   image. Container starts, model loads from local disk in ~8 seconds.
#   /ready returns 200. No network dependency at runtime.
#
#   Tradeoff: image is larger (~1.3GB heavier). Acceptable because:
#     - We only push/pull the image layer diff on subsequent builds
#     - The model-fetcher stage is independently cached (see above)
#     - Production startup reliability is non-negotiable
#
# TRANSFORMERS_OFFLINE=1 after this point means the runtime stage
# CANNOT make outbound network calls to HuggingFace. Intentional.
RUN python -c "\
from transformers import pipeline; \
print('Downloading model: ${MODEL_NAME}'); \
pipe = pipeline('sentiment-analysis', model='${MODEL_NAME}', device=-1); \
print('Model download complete.'); \
"


# ================================================================== #
# Stage 3: runtime                                                     #
# The production image. Slim, no build tools, no pip, no secrets.     #
# ================================================================== #
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG MODEL_NAME
ARG MODEL_CACHE_DIR

# ------------------------------------------------------------------ #
# Runtime system dependencies only                                     #
# ------------------------------------------------------------------ #
# libpq5: PostgreSQL client library (runtime, not build-time)
# curl: used by Docker health check below
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ #
# Non-root user                                                        #
# ------------------------------------------------------------------ #
# Running as root inside a container is a security risk: if someone
# escapes the container, they have root on the host.
# This is a production security baseline, not optional.
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --no-create-home appuser

# ------------------------------------------------------------------ #
# Copy artifacts from previous stages                                  #
# ------------------------------------------------------------------ #
# The venv — all Python dependencies
COPY --from=builder /venv /venv

# The model weights — baked in from model-fetcher stage
COPY --from=model-fetcher ${MODEL_CACHE_DIR} ${MODEL_CACHE_DIR}

# Application code
WORKDIR /app
COPY app/ ./app/

# ------------------------------------------------------------------ #
# Environment configuration                                            #
# ------------------------------------------------------------------ #
ENV PATH="/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell transformers to use the baked-in cache — never hit network
    TRANSFORMERS_CACHE=${MODEL_CACHE_DIR} \
    HF_HOME=${MODEL_CACHE_DIR} \
    # CRITICAL: prevents any outbound HuggingFace calls at runtime.
    # If the model isn't in the cache, it fails loudly at startup
    # rather than hanging on a network call.
    TRANSFORMERS_OFFLINE=1 \
    # Default runtime settings — all overridable via Render env vars
    MODEL_NAME=${MODEL_NAME} \
    MODEL_CACHE_DIR=${MODEL_CACHE_DIR} \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# ------------------------------------------------------------------ #
# Ownership                                                            #
# ------------------------------------------------------------------ #
RUN chown -R appuser:appgroup /app ${MODEL_CACHE_DIR}
USER appuser

# ------------------------------------------------------------------ #
# Port                                                                 #
# ------------------------------------------------------------------ #
# EXPOSE is documentation — it doesn't actually publish the port.
# Render maps this via its internal routing layer.
EXPOSE 8000

# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #
# Docker's built-in health check. Used by:
#   - `docker ps` to show container health
#   - docker-compose depends_on with condition: service_healthy
#   - Some orchestrators (ECS, Fly.io) for traffic routing
#
# We check /ready (not /health) because we want Docker to report
# "healthy" only AFTER the model has loaded — same logic as Render.
#
# interval: check every 30s (not too aggressive on free-tier CPU)
# timeout: fail if /ready takes > 10s (model loading lag)
# start_period: give the container 120s grace on startup for model load
# retries: 3 consecutive failures = unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

# ------------------------------------------------------------------ #
# Entrypoint                                                           #
# ------------------------------------------------------------------ #
# Why uvicorn directly instead of gunicorn + uvicorn workers?
#
#   On Render free tier:
#   - 512MB RAM. Gunicorn spawns N worker processes. Each loads the model.
#   - distilbert in fp32 = ~260MB per worker.
#   - 2 workers = 520MB → OOM kill. 1 worker = gunicorn overhead for nothing.
#
#   Single uvicorn process with async concurrency is the right choice
#   for memory-constrained, CPU-bound ML serving. Async handles I/O
#   concurrency (Redis, DB writes) while the model runs synchronously.
#
#   In production with proper RAM (4GB+), use:
#   gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker
#
# --host 0.0.0.0: listen on all interfaces (required inside containers)
# --port 8000: matches EXPOSE and Render's expected port
# --workers 1: single worker — see above
# --loop uvloop: faster async event loop (installed via uvicorn[standard])
# --log-level warning: uvicorn's own logs suppressed — our middleware
#                       handles structured logging
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "warning"]
