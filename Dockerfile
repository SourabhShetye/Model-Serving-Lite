# ================================================================== #
# Dockerfile — HuggingFace Spaces deployment                         #
# ================================================================== #
#
# HuggingFace Spaces requirements:
#   - Non-root user with UID 1000
#   - HOME=/home/user, PATH includes /home/user/.local/bin
#   - WORKDIR set to $HOME/app
#   - EXPOSE 7860
#   - Listens on 0.0.0.0:7860
#
# ================================================================== #

ARG PYTHON_VERSION=3.11

# ================================================================== #
# Stage 1: builder                                                     #
# ================================================================== #
FROM python:${PYTHON_VERSION}-slim AS builder

ARG PYTHON_VERSION

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch" requirements.txt | pip install -r /dev/stdin


# ================================================================== #
# Stage 2: model-fetcher                                               #
# ================================================================== #
FROM builder AS model-fetcher

ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_CACHE_DIR=/opt/hf_cache

ENV PATH="/venv/bin:$PATH"
ENV TRANSFORMERS_CACHE=${MODEL_CACHE_DIR}
ENV HF_HOME=${MODEL_CACHE_DIR}

# Download model weights at build time
RUN python -c "\
from transformers import pipeline; \
print('Downloading model: ${MODEL_NAME}'); \
pipe = pipeline('sentiment-analysis', model='${MODEL_NAME}', device=-1); \
print('Model download complete.'); \
"


# ================================================================== #
# Stage 3: runtime (HuggingFace Spaces compatible)                    #
# ================================================================== #
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_CACHE_DIR=/opt/hf_cache

# ================================================================== #
# Runtime system dependencies                                         #
# ================================================================== #
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ================================================================== #
# HuggingFace Spaces: non-root user with UID 1000                     #
# ================================================================== #
RUN useradd -m -u 1000 user

# ================================================================== #
# Copy artifacts from builder stages                                  #
# ================================================================== #
COPY --from=builder /venv /venv
COPY --from=model-fetcher ${MODEL_CACHE_DIR} ${MODEL_CACHE_DIR}

# ================================================================== #
# HuggingFace Spaces: HOME and PATH setup                             #
# ================================================================== #
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH="${HOME}/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=${MODEL_CACHE_DIR} \
    HF_HOME=${MODEL_CACHE_DIR} \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_CACHE_DIR=${MODEL_CACHE_DIR} \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# ================================================================== #
# HuggingFace Spaces: working directory and app code                  #
# ================================================================== #
WORKDIR ${HOME}/app
COPY --chown=user . ${HOME}/app

# ================================================================== #
# Ownership of model cache                                             #
# ================================================================== #
RUN chown -R user:user ${MODEL_CACHE_DIR}

# ================================================================== #
# Switch to non-root user                                              #
# ================================================================== #
USER user

# ================================================================== #
# HuggingFace Spaces: port 7860                                        #
# ================================================================== #
EXPOSE 7860

# ================================================================== #
# Health check                                                         #
# ================================================================== #
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ready || exit 1

# ================================================================== #
# Entrypoint: Listen on 0.0.0.0:7860 for HuggingFace Spaces          #
# ================================================================== #
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
