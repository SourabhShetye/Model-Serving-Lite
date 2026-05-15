"""
app/db/models.py

SQLAlchemy ORM model for the prediction audit log.

Why PostgreSQL in addition to structured logs?
  Logs are append-only streams — great for tailing, bad for querying.
  With PostgreSQL you can ask:

    -- Find every prediction the model was uncertain about last week
    SELECT * FROM prediction_logs
    WHERE confidence < 0.7
      AND created_at > NOW() - INTERVAL '7 days'
    ORDER BY confidence ASC;

    -- Find all requests that were cache hits vs model runs
    SELECT cache_hit, COUNT(*), AVG(latency_ms)
    FROM prediction_logs
    GROUP BY cache_hit;

    -- Find the specific prediction a customer complained about
    SELECT * FROM prediction_logs
    WHERE input_hash = 'a3f5c2d1...'
    ORDER BY created_at DESC
    LIMIT 1;

  None of these are practical with grep on log files.

Schema design decisions:
  - input_text is stored. This feels obvious but is a real tradeoff:
    storing PII in a database requires data retention policies.
    In a real system you'd store the hash only and provide input_text
    as an optional field gated on a data classification review.
    For this assessment, storing it is correct — it enables the
    "reconstruct the exact input" debugging story.

  - model_version is separate from model_name. After a retrain, the
    name stays the same but the version changes. This lets you compare
    predictions across model versions for the same input.
"""

import uuid


from sqlalchemy import Boolean, Column, DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """
    One row per prediction request.

    Every column has a comment explaining its observability purpose —
    the DBA reading this schema should understand the monitoring story.
    """
    __tablename__ = "prediction_logs"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Internal row ID. Not exposed in API responses.",
    )

    # Correlation
    request_id = Column(
        String(36),
        nullable=False,
        index=True,
        comment="UUID returned in X-Request-ID header. Primary debug handle.",
    )

    # Input
    input_text = Column(
        Text,
        nullable=False,
        comment="Raw input text. Used to reconstruct failed predictions.",
    )
    input_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 of input_text. Also serves as the Redis cache key.",
    )

    # Output
    label = Column(
        String(16),
        nullable=False,
        comment="POSITIVE or NEGATIVE.",
    )
    confidence = Column(
        Float,
        nullable=False,
        comment="Model confidence 0.0-1.0. Watch rolling mean for silent degradation.",
    )

    # Performance
    latency_ms = Column(
        Float,
        nullable=False,
        comment="Model inference time in ms (excludes cache lookup, network).",
    )
    cache_hit = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True = served from Redis. False = model was invoked.",
    )

    # Provenance
    model_name = Column(
        String(128),
        nullable=False,
        comment="HuggingFace model identifier.",
    )
    model_version = Column(
        String(64),
        nullable=True,
        comment="Git SHA of the model artifact. Set by CI after retrain.",
    )

    # Drift signals (denormalized here for fast SQL aggregation)
    input_length = Column(
        Float,
        nullable=True,
        comment="len(input_text). Drift monitor baseline signal.",
    )
    detected_language = Column(
        String(8),
        nullable=True,
        comment="langdetect result. Drift monitor language signal.",
    )

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Wall-clock time of the request (server time, UTC).",
    )
    client_ip = Column(
        String(45),   # IPv6 max length
        nullable=True,
        comment="Client IP from X-Forwarded-For or direct connection.",
    )

    # ---------------------------------------------------------------- #
    # Indexes for common query patterns                                 #
    # ---------------------------------------------------------------- #
    __table_args__ = (
        # Most common query: "show me all predictions in the last N hours"
        Index("ix_prediction_logs_created_at_label", "created_at", "label"),
        # Drift analysis: "show me confidence trends over time"
        Index("ix_prediction_logs_created_at_confidence", "created_at", "confidence"),
        # Cache analysis: "what % of requests are cache hits?"
        Index("ix_prediction_logs_cache_hit", "cache_hit"),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionLog id={self.id} "
            f"label={self.label} "
            f"confidence={self.confidence:.3f} "
            f"cache_hit={self.cache_hit}>"
        )