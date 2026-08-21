"""Service configuration, read from the environment."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from inventory_engine.config import PROJECT_ROOT


class ServiceSettings(BaseSettings):
    """Tier 2 runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://inventory:inventory@localhost:5432/inventory"
    redis_url: str = "redis://localhost:6379"

    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    max_upload_bytes: int = 50 * 1024 * 1024

    sentry_dsn: str | None = None
    posthog_api_key: str | None = None
    posthog_host: str = "https://us.i.posthog.com"
    analytics_salt: str | None = None
    environment: str = "development"

    dashboard_dir: Path | None = None
    festival_region: str | None = "IN"
    job_timeout_seconds: int = 900
    upload_retention_days: int = 30


def get_settings() -> ServiceSettings:
    """Return the settings for this process."""
    return ServiceSettings()
