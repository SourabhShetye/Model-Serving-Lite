# Incident Writeup: Detecting Silent Failures in Production

**The Problem:**
Machine learning models fail silently. Unlike traditional software that throws a 500 Error when broken, a degraded ML model will happily return a 200 OK with a completely wrong prediction (e.g., classifying a furious customer complaint as "Positive").

**How We Detect This Before Customers Do:**

1. **Structured Request/Response Logging:**
   Every request hitting the `/predict` endpoint is logged in structured JSON format, capturing the timestamp, input text length, inference latency, cache status, and output prediction. If average inference latency spikes or error rates increase, our log aggregators will immediately flag it.

2. **Drift-Monitoring Stub:**
   We implemented a drift service that monitors the distribution of incoming requests. It tracks:
   * **Input Length Shifts:** If users start sending 1,000-word essays instead of the 10-word reviews the model was trained on.
   * **Confidence Scores:** Tracking the model's softmax probability outputs. If the average prediction confidence drops across a rolling window, the model is uncertain about the new data it's seeing.

3. **Automated CI/CD Regression Gates:**
   Silent failures also happen during retraining. Our GitHub Actions pipeline acts as a preventative measure. Before a new model replaces the production one, it is evaluated against a held-out gold-standard dataset. If the F1 score regresses or falls below an acceptable floor, the pipeline automatically rejects the promotion, preventing a broken model from ever reaching production.