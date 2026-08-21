"""The tier 2 FastAPI application."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inventory_engine.service.observability import (
    EVENT_PAGE_VIEW,
    analytics,
    bind_request_id,
    configure_logging,
    init_sentry,
    referrer_host,
    visitor_id,
)
from inventory_engine.service.routers import full_forecast
from inventory_engine.service.settings import get_settings

log = logging.getLogger(__name__)

#: Header carrying the id that joins an API log line to the worker log line it caused.
REQUEST_ID_HEADER = "X-Request-ID"

#: Anonymous per-browser id, so PostHog can count users and active users without accounts.
CLIENT_ID_HEADER = "X-Client-ID"


def create_app(*, enable_cors: bool = True, configure_observability: bool = True) -> FastAPI:
    """Build the tier 2 app."""
    if configure_observability:
        configure_logging("api")
        init_sentry("api")

    app = FastAPI(
        title="Inventory Optimization Engine — Full Forecast",
        description=(
            "Upload your own sales history, get a backtested per-SKU quantile forecast and "
            "an order recommendation. Refuses data that cannot support one, and says why."
        ),
        version=full_forecast.VERSION,
    )

    if enable_cors:
        # allow_origins=["*"] is correct *because* there is no auth and no cookie: there is no
        # ambient authority for a cross-site request to abuse.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            # DELETE is here because /datasets/{id} exists.
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER, CLIENT_ID_HEADER],
        )

    app.middleware("http")(_request_id_middleware)
    app.include_router(full_forecast.router)
    _mount_dashboard(app)
    return app


def _mount_dashboard(app: FastAPI) -> None:
    """Serve the built dashboard at ``/``, if one is configured and actually present."""
    configured = get_settings().dashboard_dir
    if configured is None or not configured.is_dir():
        return
    app.mount("/", StaticFiles(directory=configured, html=True), name="dashboard")


async def _request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Attach a request id to the request, the log context and the response."""
    request_id = bind_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    request.state.client_id = request.headers.get(CLIENT_ID_HEADER) or "anonymous"
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    _count_page_view(request, response)
    return response


#: Paths that mean "somebody opened the dashboard". The mount serves the page at both,
#: depending on whether the browser followed the directory index.
_PAGE_PATHS: Final = frozenset({"/", "/index.html"})


def _client_ip(request: Request) -> str | None:
    """Return the visitor's address, from the proxy header the platform controls.

    ``request.client.host`` behind Fly is the proxy, identical for everybody, which
    collapses the visitor hash to one value per user agent. ``Fly-Client-IP`` is set by the
    proxy and overwritten on every request, so unlike ``X-Forwarded-For`` a client cannot
    spoof it to fragment or forge its own identity.
    """
    header = request.headers.get("fly-client-ip")
    if header:
        return header.strip()
    client = request.client
    return client.host if client else None


def _count_page_view(request: Request, response: Response) -> None:
    """Record one dashboard page load, if that is what this request was."""
    if request.method != "GET" or request.url.path not in _PAGE_PATHS:
        return
    if response.status_code >= 400:
        return
    try:
        analytics().capture(
            EVENT_PAGE_VIEW,
            visitor_id(_client_ip(request), request.headers.get("user-agent")),
            path=request.url.path,
            referrer_host=referrer_host(request.headers.get("referer")),
        )
    except Exception:  # noqa: BLE001 - analytics must never break a page load
        log.debug("page view not counted", exc_info=True)


app = create_app()
