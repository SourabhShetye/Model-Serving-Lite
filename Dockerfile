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

# 1. Create the non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 2. Copy application files and grant ownership to the user
COPY --chown=user . $HOME/app

# 3. Switch to the non-root user BEFORE running pip
USER user

# 4. Install requirements directly into the user's local path
RUN pip install --no-cache-dir --user -r requirements.txt

# 5. Expose port and start the app
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
