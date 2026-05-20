"""
app/config.py

Single source of truth for all configuration.
Pydantic Settings reads from environment variables first,
then falls back to the defaults defined here.

Why Pydantic Settings instead of os.getenv()?
  - Type validation at startup, not at runtime when it's too late.
  - A missing required env var crashes the app immediately with a clear
    error message, not 10 minutes later with a cryptic KeyError.
  - One place to look when someone asks "what does this service need to run?"
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    app_name: str = "sentiment-service"
    app_version: str = "0.1.0"
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    # Why store this in config and not hardcode it?
    # So the CI retrain pipeline can point to a newly trained model
    # by changing one env var — no code changes needed.
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    model_cache_dir: str = "/tmp/hf_cache"

    # ------------------------------------------------------------------ #
    # Redis (Feature Store / Cache)                                        #
    # ------------------------------------------------------------------ #
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600  # 1 hour — predictions don't expire often
    cache_enabled: bool = True  # Kill switch: set to false to bypass cache

    # ------------------------------------------------------------------ #
    # PostgreSQL (Prediction Log sink)                                     #
    # ------------------------------------------------------------------ #
    database_url: str = "postgresql://postgres:postgres@localhost:5432/sentiment_logs"

    # ------------------------------------------------------------------ #
    # Drift Monitoring                                                     #
    # ------------------------------------------------------------------ #
    # How many requests to accumulate before running the KS-test
    drift_window_size: int = 100
    # p-value threshold below which we declare drift
    drift_ks_threshold: float = 0.05
    # If rolling mean confidence drops more than this fraction below baseline, alert
    drift_confidence_drop_threshold: float = 0.10
    # If non-English requests exceed this fraction of the window, alert
    drift_language_threshold: float = 0.15

    # ------------------------------------------------------------------ #
    # Pydantic Settings Config                                             #
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env if present (local dev)
        env_file_encoding="utf-8",
        case_sensitive=False,  # MODEL_NAME and model_name both work
        extra="ignore",  # Don't crash on unknown env vars
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Why @lru_cache?
    Settings() reads files and validates types. We don't want that
    happening on every request. lru_cache(maxsize=1) makes it a
    singleton without a global variable — and it's trivially overridable
    in tests via app.dependency_overrides.
    """
    return Settings()
