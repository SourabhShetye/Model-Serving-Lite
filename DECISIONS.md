# Decisions Log — Case 9

## Assumptions I made
1. **Traffic Patterns:** I assumed the service will receive repeated queries for identical or similar texts, making a Redis cache highly valuable.
2. **Drift Definition:** I assumed that significant changes in input length or vocabulary confidence represent a valid proxy for conceptual data drift.

## Trade-offs
| Choice | Alternative | Why I picked this |
|---|---|---|
| FastAPI | Flask | Built-in async support handles concurrent API requests better, and automatic OpenAPI docs save time. |
| Redis Cache | No Cache | Adding a cache increases architecture complexity but drastically reduces expensive ML inference times for duplicate requests. |
| TextBlob Lexicon | TinyBERT Neural Network | We removed all PyTorch/HuggingFace dependencies and pivoted to TextBlob's pre-trained lexicon model to guarantee < 50MB memory footprint. Infrastructure stability within a strict 512MB Render free-tier limit is more critical than neural network-level accuracy for this microservice. |

## What I de-scoped and why
* **Shadow Deployment:** De-scoped. While valuable, setting up traffic mirroring within the 48-hour window would have compromised the quality of the core API, logging, and Redis caching implementations.

## What I'd do differently with another day
* I would implement a full ELK stack (Elasticsearch, Logstash, Kibana) to parse the structured JSON logs and build live dashboards for the drift metrics, rather than just logging them to stdout.