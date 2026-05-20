---
title: Model Serving Lite
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app/main.py
pinned: false
---

**Live demo:** https://sourabhshetye04-model-serving-lite.hf.space/docs#/inference/predict

**Repo:** https://github.com/SourabhShetye/Model-Serving-Lite

> A production-ready, monitored FastAPI service for real-time sentiment analysis using HuggingFace.

## How to run locally
1. `git clone <repo>`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn main:app --reload`

## Stack
* **FastAPI:** Chosen for native async support, speed, and automatic Swagger docs.
* **HuggingFace Transformers:** Chosen for easy access to state-of-the-art open-source pre-trained NLP models.
* **Redis:** Chosen for fast, in-memory caching of inference results to reduce compute load.
* **Docker:** Chosen for consistent, reproducible multi-stage builds.
* **GitHub Actions:** Chosen for integrated CI/CD and automated retraining checks.

## What's NOT done
* **Shadow Deployment:** De-scoped to prioritize a robust caching layer and ensure timely delivery within the 48-hour constraint.
* **Persistent Database:** We are relying on Redis for ephemeral caching rather than storing a permanent log of all requests in PostgreSQL.

## In production, I would also add
* Prometheus/Grafana for real-time visualization of latency and drift metrics.
* A more robust, distributed feature store instead of a local/simple Redis stub.
* Automated alerts (PagerDuty/Slack) triggered by our drift monitoring service.

## License
MIT License
