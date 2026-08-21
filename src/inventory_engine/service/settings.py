"""Service configuration, read from the environment.

Nothing here has a secret as its default. ``DATABASE_URL`` points at the Docker Compose
Postgres for local work and at a hosted Postgres in deployment -- the schema and the code
are identical either way, which is the whole reason for choosing plain Postgres over a
provider-specific client.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from inventory_engine.config import PROJECT_ROOT


class ServiceSettings(BaseSettings):
    """Tier 2 runtime configuration.

    Attributes:
        database_url: SQLAlchemy URL. The Compose service and a hosted Postgres both speak
            this; only the value changes.
        redis_url: Where arq keeps the job queue.
        upload_dir: Where uploaded CSVs are written. Uploads are streamed to disk rather
            than held in memory, so a large file costs disk rather than the API's RSS.
        max_upload_bytes: Hard ceiling on an upload, enforced while streaming.
        sentry_dsn: Error tracking, on both API and worker. Unset disables it.
        posthog_api_key: Product analytics. Unset disables it.
        posthog_host: PostHog ingestion endpoint, for EU or self-hosted instances.
        analytics_salt: Seed for the day-scoped visitor hash. Unset means each process
            invents its own, which over-counts visitors after a restart.
        environment: Tag attached to Sentry events and PostHog properties.
        dashboard_dir: Directory holding the built ``index.html``, served at ``/`` so the
            page and the API share an origin. ``None`` serves no page, which is right for a
            worker or a bare API. Stated explicitly rather than derived from
            ``PROJECT_ROOT``: that is meaningless once the package is pip-installed, and it
            is exactly how the festival demand table went missing in the first deployment.
        festival_region: Which festival calendar the forecast reads, or ``None`` to run
            with no calendar at all -- no proximity features, no banners and no order
            adjustment. One switch for the whole feature, so the model's view of the
            calendar and the order's cannot disagree.
        job_timeout_seconds: Ceiling on one forecast job in the worker.
        upload_retention_days: Age past which an uploaded CSV and everything derived from
            it is destroyed. Users upload their own sales history; keeping it forever is
            not a neutral default, so there is a finite one. Must be positive --
            :func:`~inventory_engine.service.retention.purge_expired` refuses zero rather
            than reading it as "delete everything".

    """

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
    """Return the settings for this process.

    Not cached: tests override the environment between cases, and constructing this is
    cheap relative to anything that reads it.
    """
    return ServiceSettings()
