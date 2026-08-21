"""Logging, error tracking and product analytics — integrated, not built.

Three separate concerns, deliberately kept separate:

``configure_logging``
    Structured JSON to stdout, every line carrying the request id. Ours, because a log
    format is not a service.

``init_sentry``
    Error tracking on the API and the worker. Third-party, because writing an exception
    aggregator is not this project's job.

``analytics``
    PostHog, for the upload -> forecast funnel. Third-party for the same reason.

The privacy constraint that shapes all of this
----------------------------------------------
Users upload **their own sales history**: product names, daily volumes, prices. That is
commercially sensitive, and none of it has any business leaving the operator's own
infrastructure.

So the rule is not "be careful what we send" — it is that the analytics layer **cannot**
send it. :func:`Analytics.capture` filters every property through
:data:`ALLOWED_PROPERTIES`, an explicit allowlist of non-identifying keys, and drops
anything else. A future contributor adding ``sku=...`` to a capture call does not leak a
product name; they get a dropped key and a warning, and
``tests/test_service_observability`` fails.

Sentry is the harder case and is described honestly rather than optimistically. Stack
traces can contain fragments of whatever was being processed, and no ``before_send`` hook
can reliably scrub an arbitrary exception message. What :func:`_scrub` does do is remove
the channels that would *predictably* carry user data: request bodies, headers, cookies,
local variables, and our own denied ``extra`` keys. The residual risk is stated in the
README rather than papered over -- an operator who cannot accept it should leave
``SENTRY_DSN`` unset, which disables the integration entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final

from inventory_engine.service.settings import ServiceSettings, get_settings

log = logging.getLogger("inventory_engine.service")

#: The id that joins an API log line to the worker log line it caused. Set by the API's
#: middleware and re-bound by the worker from the job row, so one upload can be followed
#: across two processes with a single grep.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Which process a log line or Sentry event came from.
component_var: ContextVar[str] = ContextVar("component", default="api")


def new_request_id() -> str:
    """Return a fresh request id, for a caller that did not supply one."""
    return uuid.uuid4().hex


def bind_request_id(request_id: str | None) -> str:
    """Bind ``request_id`` for this context, generating one if absent. Returns it."""
    resolved = request_id or new_request_id()
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
    """One JSON object per line, with the request id attached automatically.

    Automatically rather than by convention: a call site that forgets to pass the id still
    produces a joinable line, which is the whole point of putting it in a contextvar.
    """

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
    """Send structured JSON to stdout, and make it the *only* thing on stdout.

    Every existing logger is stripped of its own handlers and set to propagate, so all
    records converge on the single JSON handler installed on the root.

    Naming the known-noisy loggers instead was not enough. Clearing ``arq.worker`` still
    left every arq line printed twice -- once raw, once as JSON -- because the handler
    responsible sits on the **parent** ``arq`` logger, which ``arq.worker`` propagates
    through on its way to the root. A stream that is JSON except for occasional plain-text
    lines is not parseable, which defeats the point of structured logging, so this sweeps
    the whole hierarchy rather than playing whack-a-mole with library names.

    **Known exception, stated:** arq prints two banner lines ("Starting worker for 1
    functions", and the Redis version) before it calls ``on_startup``, the earliest hook a
    worker gets. Nothing here can catch those. Moving this call to module-import time would
    beat the banner, but would also wipe pytest's log capture the moment a test imports
    ``worker`` -- so the trade is two plain-text lines per worker start, and a log shipper
    should tolerate them.
    """
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
    """Strip the channels that predictably carry user data.

    Not a guarantee that no user data reaches Sentry -- an exception message can contain
    anything -- but it removes request bodies, headers, cookies and our own denied keys,
    which is where the data would otherwise arrive in bulk.
    """
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
    """Initialise error tracking for this process.

    Args:
        component: ``"api"`` or ``"worker"``, attached as a tag so the two are separable.
        settings: Overrides the environment, for tests.

    Returns:
        Whether Sentry was enabled. An unset DSN disables it silently -- the local stack
        must run with no accounts and no keys.

    """
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
#: A dashboard page load. Counted server-side, because every other event on this list
#: fires only once somebody uploads a file -- so the funnel could show four people and
#: the site could have had four thousand readers, with no way to tell the two apart.
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

#: The only property keys that may be sent. Counts, durations, enums and opaque ids --
#: nothing a product name, a price or a sales figure could travel in.
#:
#: Adding a key here is the moment to ask whether it can identify a business or its
#: products. ``rejection_reason`` is on the list because the gate's reasons are a fixed
#: vocabulary about thresholds ("90 days of history needed"); the *rendered* message that
#: names SKUs is not sent.
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
    }
)


class Analytics:
    """PostHog wrapper that can only send allowlisted, non-identifying properties.

    Never raises. An analytics outage must not turn into a failed upload for a user who
    did nothing wrong.
    """

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
        """Send one event, after filtering its properties through the allowlist.

        Returns the properties that were actually sent, so a test can assert on them
        without a PostHog account.
        """
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

#: Regenerated whenever the UTC date changes. Yesterday's identifiers cannot be recomputed
#: once it rolls, which is the property that makes this a counting mechanism rather than a
#: tracking one: there is no way, even for us, to link a visitor across two days.
_salt: tuple[str, bytes] | None = None


def _daily_salt(settings: ServiceSettings | None = None) -> bytes:
    """Return today's salt, rotating it when the date changes."""
    global _salt
    today = datetime.now(UTC).date().isoformat()
    if _salt is None or _salt[0] != today:
        settings = settings or get_settings()
        # A configured seed keeps every process agreeing on the same identifier for the
        # same visitor. Without one each process invents its own, which over-counts after a
        # restart -- honest but blunter, and preferable to inventing a default seed that
        # would be identical across every deployment of this code.
        seed = (settings.analytics_salt or os.urandom(32).hex()).encode()
        _salt = (today, hashlib.sha256(seed + today.encode()).digest())
    return _salt[1]


def visitor_id(
    ip: str | None,
    user_agent: str | None,
    settings: ServiceSettings | None = None,
) -> str:
    """Return an opaque, day-scoped identifier for one visitor.

    Neither the address nor the user agent is stored or sent anywhere -- they are salted
    and hashed here, and only the digest leaves this function. The digest changes at
    midnight UTC, so "unique visitors" means unique *today* and nothing longer. That is the
    honest limit of counting people who have not been asked to identify themselves, and it
    is the same approach Plausible and similar tools take.
    """
    material = _daily_salt(settings) + (ip or "-").encode() + b"|" + (user_agent or "-").encode()
    return hashlib.sha256(material).hexdigest()[:32]


def reset_salt() -> None:
    """Drop the cached salt. For tests that pin a date or a seed."""
    global _salt
    _salt = None
