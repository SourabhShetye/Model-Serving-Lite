# Project Report: Case 9 - Model Serving Lite

## 1. Executive Summary & Business Impact

**Context:** The transition from experimental data science to production engineering often requires crossing a significant operational chasm. A raw model residing in a Jupyter notebook cannot support real-world applications; it must be transformed into a resilient, monitored, and safely retrainable service capable of handling concurrent requests.

**The Solution:** For this project, we successfully deployed a production-grade REST API wrapping the `distilbert-base-uncased-finetuned-sst-2-english` model. This solution elevates the model from a static artifact to an active, reliable microservice capable of real-time sentiment analysis, supported by comprehensive CI/CD pipelines, caching mechanisms, and observability features.

**Infrastructure Pivot:** A critical engineering challenge encountered during development was infrastructure limitations. Initially, deployment was targeted for Render; however, its strict 512MB RAM tier proved insufficient for loading the DistilBERT model into memory, resulting in frequent Out-Of-Memory (OOM) crashes. The solution required a strategic pivot to Hugging Face Spaces. We engineered a custom multi-stage Dockerfile that baked the model weights directly into the container image at build time. This architectural optimization significantly reduced initialization overhead, dropping container cold starts to just 8 seconds and ensuring stable operation within resource constraints.

## 2. Architecture & Technology Stack

The system architecture is designed for high performance, reliability, and graceful degradation under stress.

*   **Core Stack:**
    *   **FastAPI:** Provides a high-performance, fully asynchronous REST API framework, crucial for handling concurrent inference requests efficiently.
    *   **HuggingFace Transformers:** Serves as the core NLP library to load and execute the DistilBERT model.
    *   **Redis 7:** Implemented as an aggressive, low-latency caching layer to bypass redundant model inference for previously seen inputs.
    *   **PostgreSQL 16:** Maintains a persistent audit trail of all predictions, essential for subsequent drift analysis, compliance, and model retraining.
    *   **GitHub Actions:** Automates the CI/CD pipeline, ensuring rigorous testing and controlled deployments of new model versions.

*   **Resilient Engineering (Fail-Open Architecture):** The system was explicitly designed to prioritize availability over secondary features. We implemented a robust "Fail-Open" architecture. In the event that either the Redis cache or the PostgreSQL database drops connection or becomes unavailable, the API gracefully degrades. It disables caching and/or logging but continues serving real-time model predictions without crashing or returning HTTP 500 errors to the client. Furthermore, database writes are handled as asynchronous background tasks, ensuring that logging telemetry does not add latency to the critical path of the prediction response.

## 3. Application Walkthrough (Frontend & API)

To validate the backend and provide accessibility to stakeholders, the system includes both an interactive frontend and comprehensive developer documentation.

*   **Page 0 - Streamlit App (Interactive Frontend):** We developed a user-facing web interface built with Streamlit to consume the FastAPI backend seamlessly. This dashboard allows users to enter arbitrary text and instantly receive the predicted sentiment (POSITIVE or NEGATIVE) along with the model's confidence score. Crucially, the interface actively demonstrates the performance benefits of our caching layer; users can observe sub-millisecond response times when submitting identical text, triggering a visible `cache_hit` indicator on the frontend.

*   **Swagger UI (/docs):** For developer integration and system administration, FastAPI automatically generates interactive Swagger UI documentation. This portal allows engineers to directly test and interact with the API endpoints:
    *   `POST /predict/`: The primary endpoint for submitting text for sentiment analysis.
    *   `GET /health`: Deep health check verifying connectivity to the model, Redis, and PostgreSQL.
    *   `GET /ready`: Readiness probe indicating the service is fully booted and accepting traffic.
    *   `GET /drift/status`: Exposes current drift metrics calculated by the internal monitor.
    *   `POST /drift/simulate`: A developer endpoint to inject synthetic data and validate drift alerts.

## 4. Observability, Drift, & CI/CD

Maintaining a machine learning model in production requires continuous monitoring of its operational and statistical behavior.

*   **Prediction Caching:** To optimize throughput and reduce compute costs, we integrated a Redis caching layer. By generating SHA-256 hashes of the incoming input text, we can instantly retrieve prior predictions. This optimization drastically drops inference latency for repeated queries from approximately ~100ms down to ~2ms.

*   **Three-Signal Drift Monitor:** We implemented a lightweight, in-process drift monitor designed to catch data quality issues before they silently degrade business value. It tracks three specific signals:
    1.  **Input Length (KS-test):** Uses the Kolmogorov-Smirnov test to detect significant shifts in input length distributions, serving as an early warning for anomalous bot traffic or changes in user behavior.
    2.  **Language Distribution:** Monitors for out-of-domain inputs, alerting if more than 15% of the traffic evaluates as non-English text.
    3.  **Confidence Collapse:** Tracks the model's internal confidence scores, triggering an alert if the rolling average drops by more than 10% compared to the established baseline.

*   **Automated CI/CD:** The deployment lifecycle is governed by a rigorous `retrain.yml` GitHub Actions workflow. A critical component of this pipeline is the automated "promote-gate." This safeguard algorithmically blocks any candidate model version from being deployed to production if its F1 score regresses by more than 2% or falls below an absolute threshold of 0.80 when evaluated against the historic `baseline_metrics.json`.

## 5. Future Production Roadmap

While the current architecture provides a robust foundation, we have identified key areas for strategic enhancement.

*   **Immediate/Short-Term:**
    *   **Shadow Deployments:** Implement a mechanism to mirror live production traffic to newly trained candidate models. This allows us to validate performance and catch edge cases in a real-world context before promoting them to handle active user requests.
    *   **Advanced Monitoring:** Swap the current lightweight, in-process drift monitor for scheduled, comprehensive HTML reports generated by Evidently AI, providing deeper statistical insights to the data science team.

*   **Long-Term:**
    *   **Operational Dashboards:** Deploy a dedicated Prometheus/Grafana stack to visualize real-time system metrics, with automated alerting for p99 latency spikes or infrastructure saturation.
    *   **Persistent Baselines:** Transition drift monitoring from in-memory to persistent Redis-backed baselines, allowing for long-term historical comparison and more resilient monitoring across pod restarts.
    *   **Active Learning:** Implement human-in-the-loop labeling queues for low-confidence predictions to continuously curate a high-quality fine-tuning dataset.
    *   **Global Expansion:** Enhance the pipeline to route non-English inputs to a dedicated multilingual model (e.g., XLM-RoBERTa), broadening the application's addressable market.
