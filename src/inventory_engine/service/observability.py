"""Logging, error tracking and product analytics — integrated, not built."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlparse

from inventory_engine.service.settings import ServiceSettings, get_settings

log = logging.getLogger("inventory_engine.service")

#: The id that joins an API log line to the worker log line it caused.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Which process a log line or Sentry event came from.
component_var: ContextVar[str] = ContextVar("component", default="api")


def new_request_id() -> str:
    """Return a fresh request id, for a caller that did not supply one."""
    return uuid.uuid4().hex


#: An inbound X-Request-ID is echoed into logs and into every analytics event, so it is
#: treated as untrusted input. Letters, digits and dashes only, bounded -- which keeps
#: ordinary trace ids ("trace-42", a uuid) and rejects free text, addresses and payloads.
_ID_SHAPE: Final = re.compile(r"\A[A-Za-z0-9-]{1,64}\Z")


def _safe_request_id(value: str | None) -> str | None:
    """Return the id only if it looks like one."""
    return value if value and _ID_SHAPE.match(value) else None


def bind_request_id(request_id: str | None) -> str:
    """Bind ``request_id`` for this context, generating one if absent. Returns it."""
    resolved = _safe_request_id(request_id) or new_request_id()
    request_id_var.set(resolved)
    return resolved


# --------------------------------------------------------------------------- logging

#: Attributes ``logging`` puts on every record. Anything outside this set was passed by the
#: caller as ``extra`` and belongs in the JSON payload.
_STANDARD_ATTRS: Final = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request id attached automatically."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "component": component_var.get(),
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(component: str, *, level: int = logging.INFO) -> None:
    """Send structured JSON to stdout, and make it the *only* thing on stdout."""
    component_var.set(component)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    for existing in logging.root.manager.loggerDict.values():
        if isinstance(existing, logging.Logger):
            existing.handlers = []
            existing.propagate = True

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


# --------------------------------------------------------------------------- sentry

#: ``extra`` keys that must never reach Sentry. These are the ones that carry a user's own
#: identifiers verbatim.
DENIED_EXTRA_KEYS: Final = frozenset({"sku", "filename", "original_filename", "storage_uri"})


def _scrub(event: dict, hint: dict) -> dict | None:
    """Strip the channels that predictably carry user data."""
    event.pop("request", None)
    event.pop("breadcrumbs", None)
    extra = event.get("extra")
    if isinstance(extra, dict):
        for key in list(extra):
            if key in DENIED_EXTRA_KEYS:
                extra[key] = "[redacted]"
    for exception in (event.get("exception") or {}).get("values") or []:
        for frame in (exception.get("stacktrace") or {}).get("frames") or []:
            # Local variables at the point of failure routinely hold a DataFrame slice.
            frame.pop("vars", None)
    return event


def init_sentry(component: str, settings: ServiceSettings | None = None) -> bool:
    """Initialise error tracking for this process."""
    settings = settings or get_settings()
    if not settings.sentry_dsn:
        return False

    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        # No user identity is collected anywhere in tier 2 -- there are no accounts yet --
        # and this makes that structural rather than incidental.
        send_default_pii=False,
        max_request_body_size="never",
        before_send=_scrub,
        traces_sample_rate=0.0,
    )
    sentry_sdk.set_tag("component", component)
    return True


# --------------------------------------------------------------------------- analytics

EVENT_UPLOAD_RECEIVED: Final = "upload_received"
EVENT_UPLOAD_REJECTED: Final = "upload_rejected"
EVENT_DATASET_CREATED: Final = "dataset_created"
EVENT_FORECAST_ENQUEUED: Final = "forecast_enqueued"
EVENT_FORECAST_COMPLETED: Final = "forecast_completed"
EVENT_FORECAST_FAILED: Final = "forecast_failed"
#: A dashboard page load.
EVENT_PAGE_VIEW: Final = "page_view"
#: A deletion carried out. The one feature outside the funnel that already talks to the
#: server, so counting it costs no new request and breaks no promise made on the page.
EVENT_DATA_DELETED: Final = "data_deleted"

FUNNEL: Final[tuple[str, ...]] = (
    EVENT_UPLOAD_RECEIVED,
    EVENT_DATASET_CREATED,
    EVENT_FORECAST_ENQUEUED,
    EVENT_FORECAST_COMPLETED,
)

#: The only property keys that may be sent.
ALLOWED_PROPERTIES: Final = frozenset(
    {
        "component",
        "environment",
        "request_id",
        "dataset_id",
        "job_id",
        "rows_read",
        "sku_count",
        "sku_count_admitted",
        "sku_count_rejected",
        "byte_size",
        "horizon",
        "critical_ratio",
        "elapsed_seconds",
        "n_folds",
        "method_used",
        "rejection_reason",
        "status_code",
        "fatal",
        # Page views. `path` is the dashboard route only ("/"), never a query string,
        # and `jobs_cancelled` is a count.
        "path",
        "jobs_cancelled",
        # Host only, never the path or query.
        "referrer_host",
    }
)


class Analytics:
    """PostHog wrapper that can only send allowlisted, non-identifying properties."""

    def __init__(self, settings: ServiceSettings | None = None) -> None:
        """Configure the client, or disable it when no key is set."""
        self.settings = settings or get_settings()
        self._client = None
        if not self.settings.posthog_api_key:
            return
        try:
            from posthog import Posthog

            self._client = Posthog(
                project_api_key=self.settings.posthog_api_key,
                host=self.settings.posthog_host,
            )
        except Exception:  # noqa: BLE001 - analytics must never break startup
            log.warning("posthog unavailable; product analytics disabled")

    @property
    def enabled(self) -> bool:
        """Whether events will actually be sent."""
        return self._client is not None

    def capture(self, event: str, distinct_id: str | None = None, **properties: Any) -> dict:
        """Send one event, after filtering its properties through the allowlist."""
        safe, dropped = self.filter_properties(properties)
        if dropped:
            # Loud, because a dropped key means someone tried to send business data and
            # the allowlist caught it. That is worth fixing at the call site.
            log.warning(
                "analytics properties dropped by the allowlist",
                extra={"event": event, "dropped_keys": sorted(dropped)},
            )
        safe.setdefault("component", component_var.get())
        safe.setdefault("environment", self.settings.environment)
        safe.setdefault("request_id", request_id_var.get())

        if self._client is not None:
            try:
                self._client.capture(
                    distinct_id=distinct_id or "anonymous", event=event, properties=safe
                )
            except Exception:  # noqa: BLE001 - never break a request for telemetry
                log.warning("analytics capture failed", extra={"event": event})
        return safe

    @staticmethod
    def filter_properties(properties: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        """Split properties into the allowed ones and the names of the dropped ones."""
        allowed = {k: v for k, v in properties.items() if k in ALLOWED_PROPERTIES}
        return allowed, set(properties) - set(allowed)


_analytics: Analytics | None = None


def analytics() -> Analytics:
    """Return the process-wide analytics client."""
    global _analytics
    if _analytics is None:
        _analytics = Analytics()
    return _analytics


def reset_analytics() -> None:
    """Drop the cached client. For tests that change the environment."""
    global _analytics
    _analytics = None


# --------------------------------------------------------------------------- visitors

#: Regenerated whenever the UTC date changes.
_salt: tuple[str, bytes] | None = None


def _daily_salt(settings: ServiceSettings | None = None) -> bytes:
    """Return today's salt, rotating it when the date changes."""
    global _salt
    settings = settings or get_settings()
    today = datetime.now(UTC).date().isoformat()
    # Keyed on the seed as well as the date: keyed on the date alone, a caller passing
    # a different seed silently got the cached one.
    key = f"{today}|{settings.analytics_salt or ''}"
    if _salt is None or _salt[0] != key:
        # A configured seed keeps every process agreeing on the same identifier for the same
        # visitor.
        seed = (settings.analytics_salt or os.urandom(32).hex()).encode()
        _salt = (key, hashlib.sha256(seed + today.encode()).digest())
    return _salt[1]


def visitor_id(
    ip: str | None,
    user_agent: str | None,
    settings: ServiceSettings | None = None,
) -> str:
    """Return an opaque, day-scoped identifier for one visitor."""
    material = _daily_salt(settings) + (ip or "-").encode() + b"|" + (user_agent or "-").encode()
    return hashlib.sha256(material).hexdigest()[:32]


#: Our own hostname, so a visitor moving between our pages is not counted as a referral.
SELF_HOST: Final = "fly.dev"


def referrer_host(referer: str | None) -> str | None:
    """Return just the host of a referring URL, or None when there is nothing usable."""
    if not referer:
        return None
    try:
        host = urlparse(referer).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower()
    # Boundary-aware: "myfly.dev" is somebody else's site, not this one.
    return None if host == SELF_HOST or host.endswith("." + SELF_HOST) else host


def reset_salt() -> None:
    """Drop the cached salt. For tests that pin a date or a seed."""
    global _salt
    _salt = None
