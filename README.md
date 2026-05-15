# sentiment-service

> Production sentiment analysis API — from notebook to monitored, retrainable service.

A pretrained DistilBERT model wrapped in a FastAPI service with structured logging, Redis caching, PostgreSQL audit trail, three-signal drift monitoring, and a CI/CD pipeline that auto-rejects regressed models.

Built for Case 9 of the MLOps Engineering take-home assessment.

---

## Quick Start

```bash
# Clone and start the full stack (app + Redis + Postgres)
git clone https://github.com/your-username/sentiment-service
cd sentiment-service
cp .env.example .env
docker compose up --build
```

The first build downloads model weights (~1.3GB). Subsequent starts use the cached image and are ready in ~10 seconds.

```bash
# Make a prediction
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "This product exceeded all my expectations!"}' | jq .
```

```json
{
  "request_id": "a3f5c2d1-7e84-4b2a-9f12-3d8e1c7b4a90",
  "label": "POSITIVE",
  "confidence": 0.9998,
  "input_hash": "e3b0c44298fc1c149afb....",
  "latency_ms": 84.3,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "cache_hit": false
}
```

---

## Live API

| Environment | URL |
|---|---|
| Production (Render) | `https://sentiment-service.onrender.com` |
| Local (Docker Compose) | `http://localhost:8000` |
| Interactive docs | `https://sentiment-service.onrender.com/docs` |

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              sentiment-service               │
                        │                                              │
  POST /predict/  ──►   │  StructuredLoggingMiddleware                 │
                        │    assigns request_id, starts wall-clock     │
                        │    timer, emits one JSON log line per req    │
                        │               │                              │
                        │               ▼                              │
                        │  predict() route handler                     │
                        │    │                                         │
                        │    ├─► CacheService.get(sha256(text))        │◄──► Redis
                        │    │      hit  ──► return immediately        │     (cache)
                        │    │      miss ──► ModelService.predict()    │
                        │    │                 HuggingFace pipeline    │
                        │    │                 loaded once at startup  │
                        │    │                                         │
                        │    ├─► CacheService.set(...)  [async]        │◄──► Redis
                        │    │                                         │
                        │    ├─► DriftService.record(...)  [sync]      │
                        │    │      fills baseline → live window       │
                        │    │      KS-test + language + confidence    │
                        │    │                                         │
                        │    └─► BackgroundTask: write_prediction_log  │◄──► PostgreSQL
                        │              written AFTER response returns  │     (audit log)
                        │              zero latency impact             │
                        └─────────────────────────────────────────────┘
```

**Design principles:**

- The model loads **once** at startup via FastAPI's `lifespan` context manager, stored on `app.state`, injected via dependency injection. Zero per-request model loading.
- Redis is **fail-open**: if Redis is unavailable, predictions continue — just without caching. A monitoring bug or cache failure never fails a customer request.
- The PostgreSQL write is **async** (FastAPI `BackgroundTasks`): the response returns to the client first, the DB write happens after. Client latency = model latency only.
- Drift recording is **fire-and-forget** inside a `try/except`: monitoring failures are logged and swallowed. They never surface to the caller.

---

## Endpoints

### `POST /predict/`

Classifies the sentiment of input text.

**Request**
```json
{ "text": "Your text here (1–512 characters)" }
```

**Response**
```json
{
  "request_id": "uuid-v4",
  "label": "POSITIVE | NEGATIVE",
  "confidence": 0.9998,
  "input_hash": "sha256-of-input",
  "latency_ms": 84.3,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "cache_hit": false
}
```

**curl examples**

```bash
# Basic prediction
curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely loved it, would buy again."}'

# Negative sentiment
curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Complete waste of money. Broke after one day."}'

# Cache hit — send the same text twice, observe cache_hit: true
curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Great service!"}'

curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Great service!"}' | jq '.cache_hit'
# → true

# The X-Request-ID response header is your log correlation handle
curl -I -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Test"}' | grep -i x-request-id
```

**Validation errors**

```bash
# Empty text → 422
curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": ""}' | jq .

# Text too long (> 512 chars) → 422
curl -X POST https://sentiment-service.onrender.com/predict/ \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$(python3 -c 'print("x" * 513)')\"}" | jq .
```

---

### `GET /health`

Liveness probe. Returns `200` as long as the process is running. No external dependency checks — intentionally minimal. Used by the container orchestrator to decide whether to **restart** the container.

```bash
curl https://sentiment-service.onrender.com/health
# {"status": "ok", "version": "0.1.0"}
```

### `GET /ready`

Readiness probe. Returns `200` only after the model has loaded and the service is ready to accept traffic. Returns `503` during the startup model-load period. Used by the load balancer to decide whether to **route** traffic here.

```bash
curl https://sentiment-service.onrender.com/ready | jq .
```
```json
{
  "status": "ready",
  "model_loaded": true,
  "redis_reachable": true,
  "uptime_seconds": 142.7
}
```

### `GET /drift/status`

Current state of the three-signal drift monitor. Shows whether the baseline window has been established and how full the live window is.

```bash
curl https://sentiment-service.onrender.com/drift/status | jq .
```
```json
{
  "baseline_established": true,
  "baseline_size": 100,
  "live_window_size": 47,
  "window_capacity": 100,
  "next_analysis_in": 53,
  "total_observations": 347,
  "baseline_mean_confidence": 0.9821,
  "live_mean_confidence": 0.9794,
  "message": "Baseline established. Next analysis in 53 predictions."
}
```

### `POST /drift/simulate`

Injects synthetic samples to trigger a drift alert on demand. **For demo and testing only** — not present in a real production service.

```bash
# Simulate confidence collapse
curl -X POST https://sentiment-service.onrender.com/drift/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "confidence_collapse", "num_samples": 100}' | jq .

# Simulate input length shift
curl -X POST https://sentiment-service.onrender.com/drift/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "length_shift", "num_samples": 100}' | jq .

# Simulate non-English traffic
curl -X POST https://sentiment-service.onrender.com/drift/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario": "language_shift", "num_samples": 100}' | jq .
```

After each simulation, watch the server logs for `DRIFT_ALERT` log lines:

```bash
# Local
docker compose logs -f app | grep DRIFT_ALERT

# Render
# Render dashboard → Service → Logs → filter: DRIFT_ALERT
```

---

## Observability

### Structured Logs

Every request produces exactly one JSON log line. Every field is documented:

```json
{
  "timestamp": "2024-01-15T10:23:41",
  "level": "INFO",
  "message": "request_complete",
  "request_id": "a3f5c2d1-...",
  "method": "POST",
  "path": "/predict/",
  "status_code": 200,
  "latency_ms": 84.3,
  "cache_hit": false,
  "label": "POSITIVE",
  "confidence": 0.9998,
  "input_hash": "e3b0c442...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "client_ip": "203.0.113.5"
}
```

**The `request_id` is your primary debugging handle.** It is returned in the `X-Request-ID` response header, written to the log, and stored in the PostgreSQL `prediction_logs` table. Given a customer complaint, you can reconstruct any prediction in under two minutes:

```bash
# Find the log line
grep '"request_id": "a3f5c2d1"' app.log | jq .

# Query the database for full context including input text
psql $DATABASE_URL -c \
  "SELECT * FROM prediction_logs WHERE request_id = 'a3f5c2d1-...';"

# Check if drift was active at that time
psql $DATABASE_URL -c \
  "SELECT created_at, label, confidence, cache_hit, model_version
   FROM prediction_logs
   WHERE created_at BETWEEN '2024-01-15 10:00:00' AND '2024-01-15 11:00:00'
   ORDER BY created_at DESC LIMIT 20;"
```

### Drift Monitoring — Three Signals

The drift monitor runs three independent statistical tests every 100 non-cached predictions:

| Signal | Method | Threshold | What It Catches |
|---|---|---|---|
| Input Length | Two-sample KS-test on `len(text)` | p-value < 0.05 | Bot traffic, API misuse, source change |
| Language Distribution | `langdetect` fraction non-English | > 15% | Geographic shift, wrong-language inputs |
| Confidence Collapse | Rolling mean vs baseline mean | > 10% relative drop | OOD inputs, silent model degradation |

A `DRIFT_ALERT` log line fires when any signal breaches its threshold. A `CRITICAL` alert fires when confidence collapses by more than 25%.

**Why confidence is the most important signal:** Softmax confidence drops *before* accuracy does. When the model encounters out-of-distribution inputs, it becomes uncertain (confidence → 0.5) before it starts misclassifying. This gives a warning window of hundreds of requests before customers see wrong answers — without needing ground-truth labels.

### Redis Cache (Feature Store Stub)

Prediction results are cached by `SHA-256(input_text)` with a 1-hour TTL. Cache hits return the original model latency so you can distinguish "this result was cached from a 42ms inference" from "this is a fresh 84ms inference".

```bash
# Inspect cache contents (with redis-commander GUI)
docker compose --profile debug up
# → http://localhost:8081

# Or directly with redis-cli
docker exec sentiment-redis redis-cli keys "prediction:*"
docker exec sentiment-redis redis-cli get "prediction:<sha256>"

# Check hit rate from the DB
psql $DATABASE_URL -c \
  "SELECT cache_hit, COUNT(*), ROUND(AVG(latency_ms)::numeric, 1) as avg_latency_ms
   FROM prediction_logs GROUP BY cache_hit;"
```

---

## Project Structure

```
sentiment-service/
│
├── .github/workflows/
│   ├── ci.yml              # Lint + typecheck + test on every PR
│   └── retrain.yml         # Retrain → evaluate → promote-gate → deploy
│
├── app/
│   ├── main.py             # App factory, lifespan startup/shutdown
│   ├── config.py           # Pydantic Settings — all env vars in one place
│   ├── dependencies.py     # FastAPI DI providers (model, redis, settings)
│   │
│   ├── routers/
│   │   ├── predict.py      # POST /predict/
│   │   ├── health.py       # GET /health, GET /ready
│   │   └── drift.py        # GET /drift/status, POST /drift/simulate
│   │
│   ├── services/
│   │   ├── model_service.py   # HuggingFace pipeline wrapper
│   │   ├── cache_service.py   # Redis get/set/flush with typed returns
│   │   └── drift_service.py   # Three-signal drift detector (singleton)
│   │
│   ├── middleware/
│   │   └── logging_middleware.py  # Structured JSON request/response logging
│   │
│   └── db/
│       ├── models.py       # SQLAlchemy: prediction_logs table
│       └── crud.py         # write_prediction_log(), get_recent_confidence_stats()
│
├── training/
│   ├── train.py            # Fine-tune classification head, save artifact
│   ├── evaluate.py         # Held-out test set evaluation → metrics.json
│   └── baseline_metrics.json  # Gate baseline — updated on each promotion
│
├── tests/
│   └── test_drift.py       # Unit tests for all three drift signals
│
├── Dockerfile              # Three-stage build: builder → model-fetcher → runtime
├── docker-compose.yml      # App + Redis + Postgres + optional Redis Commander
└── .env.example            # All required environment variables with defaults
```

---

## CI/CD Pipeline

### `ci.yml` — Every PR

```
push/PR to main
    │
    ├─► lint      ruff check + ruff format --check
    ├─► typecheck  mypy --strict-optional
    └─► test       pytest --cov=app --cov-fail-under=80
```

Coverage gate: **80% minimum**. The workflow fails if coverage drops below this threshold.

### `retrain.yml` — Training Data PRs

Triggered when any file under `training/data/**` changes:

```
training/data/ PR opened
    │
    ▼
Job 1: retrain       train.py → model artifact (GitHub artifact, 30-day retention)
    │
    ▼
Job 2: evaluate      evaluate.py → metrics.json
    │
    ▼
Job 3: promote-gate  ← THE CRITICAL JOB
    │   Gate 1: F1 ≥ 0.80 (absolute floor)
    │   Gate 2: F1 regression ≤ 2% vs baseline_metrics.json
    │
    │   REJECTED → workflow fails, PR comment posted with exact delta
    │   APPROVED ↓
    ▼
Job 4: push-image    docker build + push to ghcr.io
                     commit new baseline_metrics.json
                     trigger Render deploy hook
    │
    ▼
Job 5: cache-bust    DELETE prediction:* keys from Redis
                     immediate consistency with new model
```

**Simulating a rejected model (for the walkthrough demo):**

```bash
# Temporarily inflate the baseline so the new model can't match it
# Edit training/baseline_metrics.json:
{
  "f1_score": 0.999,   ← impossible to beat
  ...
}

# Push a training/data/ file to open a PR
echo "text,label" > training/data/test_trigger.csv
git add training/data/test_trigger.csv
git commit -m "test: trigger retrain pipeline"
git push origin feature/retrain-demo

# Open a PR → watch the promote-gate job fail with:
# "REJECTED: F1 regressed by 0.0727 (max allowed: 0.02)"
# → PR comment is posted automatically with the table
```

---

## Local Development

### Running with Docker Compose (recommended)

```bash
# Full stack — app + Redis + Postgres
docker compose up --build

# Dependencies only — run uvicorn locally for faster iteration
docker compose up redis postgres

# In a second terminal
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# With Redis Commander GUI for cache inspection
docker compose --profile debug up
# → http://localhost:8081
```

### Running Tests

```bash
pip install pytest pytest-asyncio httpx pytest-cov

# All tests with coverage report
pytest tests/ --cov=app --cov-report=term-missing -v

# Drift tests only (no infrastructure needed — pure unit tests)
pytest tests/test_drift.py -v

# Single test class
pytest tests/test_drift.py::TestConfidenceDrift -v
```

### Environment Variables

Copy `.env.example` to `.env` and adjust for local development. All variables have safe defaults.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model identifier |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/sentiment_logs` | PostgreSQL connection string |
| `CACHE_ENABLED` | `true` | Set to `false` to bypass Redis entirely |
| `CACHE_TTL_SECONDS` | `3600` | Prediction cache TTL (1 hour) |
| `DRIFT_WINDOW_SIZE` | `100` | Requests before drift analysis fires |
| `DRIFT_KS_THRESHOLD` | `0.05` | KS-test p-value below which length drift alerts |
| `DRIFT_CONFIDENCE_DROP_THRESHOLD` | `0.10` | Relative confidence drop before alert |
| `DRIFT_LANGUAGE_THRESHOLD` | `0.15` | Non-English fraction before alert |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose output locally |
| `ENVIRONMENT` | `development` | Controls CORS and SQL echo |

---

## Deployment (Render)

### One-time setup

**1. Provision services on Render**

- New **Web Service** → connect GitHub repo → Runtime: Docker
- New **Redis** instance (Render add-on)
- New **PostgreSQL** instance (Render add-on)

**2. Set environment variables** (Render dashboard → Service → Environment)

```
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
REDIS_URL=<from Render Redis dashboard>
DATABASE_URL=<from Render Postgres dashboard>
ENVIRONMENT=production
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

**3. Set GitHub Actions secrets** (repo → Settings → Secrets → Actions)

```
RENDER_DEPLOY_HOOK_URL   # Render → Service → Settings → Deploy Hook
RENDER_SERVICE_URL       # e.g. https://sentiment-service.onrender.com
ADMIN_API_KEY            # Any secret string — used for cache flush endpoint auth
```

**4. Render health check configuration**

```
Health Check Path: /ready
Health Check Timeout: 120s   ← model load grace period
```

### Deploy behaviour

- **Automatic deploys**: disabled on Render — triggered only by the `retrain.yml` promote-gate via the deploy hook, ensuring only validated models reach production.
- **Manual deploy**: Render dashboard → Service → Manual Deploy → Deploy latest commit.
- **Rollback**: Render dashboard → Service → Deploys → select previous deploy → Rollback.

---

## Docker

### Build and run locally

```bash
# Build (uses BuildKit cache — fast after first run)
DOCKER_BUILDKIT=1 docker build -t sentiment-service:local .

# Run with environment variables
docker run -p 8000:8000 \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/sentiment_logs \
  -e CACHE_ENABLED=false \
  sentiment-service:local

# Check image size
docker images sentiment-service:local --format "{{.Size}}"
# → ~2.1GB (PyTorch CPU-only is unavoidably large)
```

### Why three build stages?

| Stage | Purpose | Discarded? |
|---|---|---|
| `builder` | Installs all deps including gcc, compilers | Yes — compilers never reach runtime |
| `model-fetcher` | Downloads HuggingFace weights at build time | No — weights copied to runtime |
| `runtime` | Slim Python + app code only | No — this is the final image |

Model weights are baked into the image at build time so the container starts in ~8 seconds with no runtime network dependency on HuggingFace CDN. `TRANSFORMERS_OFFLINE=1` enforces this — the container cannot make outbound model download calls.

---

## Observability Architecture — "How Would I Know Before Customers Do?"

See [`INCIDENT_WRITEUP.docx`](./INCIDENT_WRITEUP.docx) for the full strategy document.

**Summary of detection windows:**

| Failure Mode | Signal | Detection Window |
|---|---|---|
| Process crash / 5xx spike | HTTP error rate > 1% | < 60 seconds |
| Latency degradation | p99 > 500ms | < 5 minutes |
| Confidence collapse | Rolling mean drops > 10% | ~100 requests |
| Input distribution shift | KS-test p-value < 0.05 | ~100 requests |
| Non-English traffic surge | Language fraction > 15% | ~100 requests |
| Post-retrain regression | CI promote-gate F1 delta | Before deployment |

---

## Known Limitations and Production Evolution

This service is scoped to free-tier infrastructure. In a production environment, the following would be added:

- **Evidently AI** drift reports — full HTML reports from `prediction_logs`, replacing the in-process KS-test stub
- **Prometheus + Grafana** — `/metrics` already exports all signals; adding Grafana converts them to time-series alerts
- **Persistent drift baseline** — serialised to Redis on establishment, rehydrated on restart, eliminating the 100-request recalibration window
- **Human-in-the-loop labelling** — 1–2% of production predictions sampled for ground-truth accuracy measurement
- **Shadow deployment** — candidate model runs in parallel; promotes only when live-traffic agreement rate exceeds threshold
- **Alembic migrations** — replacing `Base.metadata.create_all()` for versioned, reversible schema changes
- **Gunicorn + multiple uvicorn workers** — feasible at 4GB+ RAM (currently single worker to fit 512MB free tier)

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| API framework | FastAPI 0.111 | Native async, automatic OpenAPI docs, lifespan context manager |
| Model serving | HuggingFace Transformers | Standard interface for pretrained models |
| ML model | DistilBERT SST-2 | 97% of BERT accuracy at 40% the size — CPU-viable |
| Cache / Feature Store | Redis 7 (async via `redis-py`) | Sub-millisecond lookup, TTL eviction, SCAN-safe flush |
| Audit log | PostgreSQL 16 + SQLAlchemy | Queryable history — grep can't do `GROUP BY confidence` |
| Structured logging | `python-json-logger` | Every log line is valid JSON from process start |
| Drift detection | `scipy` KS-test + `langdetect` | Zero infrastructure overhead, in-process |
| Container runtime | Docker (multi-stage, BuildKit) | CPU-only torch build, model baked at build time |
| CI/CD | GitHub Actions | Path-filtered retrain trigger, promote-gate, Render webhook |
| Hosting | Render | Free-tier HTTPS, deploy hooks, managed Redis and Postgres |

---

## License

MIT