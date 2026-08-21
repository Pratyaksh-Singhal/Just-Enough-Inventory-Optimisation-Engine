"""Logging, Sentry and analytics — with the privacy guarantees actually enforced.

Users upload their own sales history. The claim that none of it reaches a third party has
to be a test, not a paragraph, or it decays the first time someone adds a helpful property
to a capture call.
"""

from __future__ import annotations

import json
import logging
import re

import pytest

from inventory_engine.service.observability import (
    ALLOWED_PROPERTIES,
    DENIED_EXTRA_KEYS,
    FUNNEL,
    Analytics,
    JsonFormatter,
    _scrub,
    bind_request_id,
    component_var,
    init_sentry,
    referrer_host,
    request_id_var,
    reset_salt,
    visitor_id,
)
from inventory_engine.service.settings import ServiceSettings


@pytest.fixture(autouse=True)
def clean_context():
    """Reset the contextvars between tests."""
    request_id_var.set(None)
    component_var.set("api")
    yield
    request_id_var.set(None)
    component_var.set("api")


def record(msg="hello", **extra) -> logging.LogRecord:
    """A log record carrying ``extra`` fields."""
    r = logging.LogRecord("test", logging.INFO, __file__, 1, msg, None, None)
    r.__dict__.update(extra)
    return r


# --------------------------------------------------------------------------- logging


def test_a_log_line_is_one_json_object():
    line = JsonFormatter().format(record("dataset created"))
    payload = json.loads(line)
    assert payload["message"] == "dataset created"
    assert payload["level"] == "INFO"
    assert "\n" not in line


def test_the_request_id_rides_along_without_the_call_site_passing_it():
    """A contextvar, so a handler that forgets is still joinable."""
    bind_request_id("trace-42")
    assert json.loads(JsonFormatter().format(record()))["request_id"] == "trace-42"


def test_extra_fields_become_json_keys():
    payload = json.loads(JsonFormatter().format(record(job_id="j1", skus=7)))
    assert payload["job_id"] == "j1"
    assert payload["skus"] == 7


def test_the_component_distinguishes_api_from_worker():
    component_var.set("worker")
    assert json.loads(JsonFormatter().format(record()))["component"] == "worker"


def test_a_non_serialisable_value_does_not_break_the_line():
    """Logging must never be the thing that raises."""
    payload = json.loads(JsonFormatter().format(record(obj=object())))
    assert isinstance(payload["obj"], str)


def test_an_exception_is_captured_as_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        r = record("failed")
        r.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(r))
    assert "ValueError: boom" in payload["exception"]


def test_bind_request_id_generates_one_when_absent():
    assert bind_request_id(None)
    assert request_id_var.get()


def test_configure_logging_leaves_exactly_one_handler_in_the_whole_hierarchy():
    """Regression: arq's lines were printed twice, once raw and once as JSON.

    Clearing the named ``arq.worker`` logger was not enough — the handler responsible sits
    on the parent ``arq`` logger, which the child propagates through. A stream that is JSON
    except for occasional plain-text lines cannot be parsed, which is the whole point.
    """
    from inventory_engine.service.observability import configure_logging

    saved_root = logging.getLogger().handlers[:]
    noisy = logging.getLogger("arq")
    noisy_child = logging.getLogger("arq.worker")
    noisy.handlers = [logging.StreamHandler()]
    noisy_child.handlers = [logging.StreamHandler()]
    try:
        configure_logging("worker")
        assert noisy.handlers == []
        assert noisy_child.handlers == []
        assert noisy.propagate and noisy_child.propagate
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        logging.getLogger().handlers = saved_root
        noisy.handlers = []
        noisy_child.handlers = []


# --------------------------------------------------------------------------- analytics


def analytics_off() -> Analytics:
    """An analytics client with no key, so nothing is actually sent."""
    return Analytics(ServiceSettings(posthog_api_key=None))


def test_a_sku_name_cannot_be_sent_even_if_a_call_site_tries():
    """The guarantee. Product names are the user's commercial data."""
    sent = analytics_off().capture("forecast_completed", "client-1", sku="FOODS_3_086", job_id="j")
    assert "sku" not in sent
    assert sent["job_id"] == "j"


@pytest.mark.parametrize(
    "leaky",
    [
        {"sku": "MILK_1L"},
        {"filename": "acme-sales-2025.csv"},
        {"skus": ["A", "B"]},
        {"unit_price": 4.99},
        {"order_qty": 412},
        {"storage_uri": "file:///data/uploads/abc.csv"},
        {"company": "Acme Foods"},
        {"sample_rows": [{"sku": "A", "units": 3}]},
    ],
)
def test_business_data_is_dropped_whatever_shape_it_arrives_in(leaky):
    sent = analytics_off().capture("upload_received", "c", **leaky)
    for key in leaky:
        assert key not in sent


def test_the_allowlist_contains_no_free_text_or_identifier_keys():
    """A standing check on the allowlist itself, not on one call."""
    for key in ALLOWED_PROPERTIES:
        assert "sku" not in key or key.startswith("sku_count"), key
        assert "name" not in key and "price" not in key and "file" not in key, key


def test_allowed_properties_survive():
    sent = analytics_off().capture(
        "forecast_completed", "c", horizon=28, sku_count=7, elapsed_seconds=33.7
    )
    assert sent["horizon"] == 28
    assert sent["sku_count"] == 7
    assert sent["elapsed_seconds"] == 33.7


def test_every_capture_carries_component_environment_and_request_id():
    bind_request_id("r-1")
    sent = analytics_off().capture("upload_received", "c")
    assert sent["component"] == "api"
    assert sent["environment"]
    assert sent["request_id"] == "r-1"


def test_analytics_is_disabled_without_a_key():
    assert analytics_off().enabled is False


def test_capture_never_raises_when_the_client_explodes():
    """An analytics outage must not become a failed upload."""

    class Exploding:
        def capture(self, **_):
            raise RuntimeError("posthog down")

    a = analytics_off()
    a._client = Exploding()
    assert a.capture("upload_received", "c", horizon=7)["horizon"] == 7


def test_the_funnel_is_ordered_upload_to_forecast():
    assert FUNNEL[0] == "upload_received"
    assert FUNNEL[-1] == "forecast_completed"
    assert len(set(FUNNEL)) == len(FUNNEL)


def test_dropped_keys_are_logged_loudly(caplog):
    """A dropped key means someone tried to send business data; that is worth seeing."""
    with caplog.at_level(logging.WARNING):
        analytics_off().capture("upload_received", "c", sku="MILK")
    assert any("allowlist" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- sentry


def test_sentry_is_disabled_without_a_dsn():
    """The local stack must run with no accounts and no keys."""
    assert init_sentry("api", ServiceSettings(sentry_dsn=None)) is False


def test_the_scrubber_drops_the_request_entirely():
    event = _scrub({"request": {"data": "sku,date,units\nMILK,2025-01-01,7"}}, {})
    assert "request" not in event


def test_the_scrubber_redacts_denied_extra_keys():
    event = _scrub({"extra": {"sku": "MILK_1L", "job_id": "j1"}}, {})
    assert event["extra"]["sku"] == "[redacted]"
    assert event["extra"]["job_id"] == "j1"


def test_the_scrubber_removes_stack_frame_locals():
    """Locals at the point of failure routinely hold a slice of the user's data."""
    event = _scrub(
        {
            "exception": {
                "values": [
                    {"stacktrace": {"frames": [{"function": "fit", "vars": {"frame": "..."}}]}}
                ]
            }
        },
        {},
    )
    assert "vars" not in event["exception"]["values"][0]["stacktrace"]["frames"][0]


def test_the_denied_keys_cover_the_identifiers_the_code_actually_logs():
    assert {"sku", "filename", "storage_uri"} <= DENIED_EXTRA_KEYS


# --------------------------------------------------------------------------- visitors


def test_the_same_visitor_hashes_the_same_way_within_a_day():
    reset_salt()
    a = visitor_id("203.0.113.9", "Mozilla/5.0")
    b = visitor_id("203.0.113.9", "Mozilla/5.0")
    assert a == b


@pytest.mark.parametrize(
    "ip,ua",
    [("203.0.113.10", "Mozilla/5.0"), ("203.0.113.9", "Firefox/1.0")],
)
def test_a_different_visitor_hashes_differently(ip, ua):
    reset_salt()
    assert visitor_id(ip, ua) != visitor_id("203.0.113.9", "Mozilla/5.0")


def test_the_address_and_agent_never_appear_in_the_digest():
    """The digest is the only thing that leaves ``visitor_id``; it must carry neither input."""
    reset_salt()
    digest = visitor_id("203.0.113.9", "Mozilla/5.0 (Macintosh)")
    assert "203.0.113.9" not in digest
    assert "Mozilla" not in digest
    assert re.fullmatch(r"[0-9a-f]{32}", digest)


def test_the_identifier_cannot_be_recomputed_after_the_salt_rotates():
    """Rotation is what makes this counting rather than tracking.

    If yesterday's identifier could be recomputed, the digests would link a visitor across
    days and the mechanism would be a tracker with extra steps.
    """
    reset_salt()
    today = visitor_id("203.0.113.9", "Mozilla/5.0")
    reset_salt()  # stands in for the date rolling over: a fresh random seed
    assert visitor_id("203.0.113.9", "Mozilla/5.0") != today


def test_a_missing_address_still_produces_an_identifier():
    """A proxy that strips the client address must not raise inside a page load."""
    reset_salt()
    assert re.fullmatch(r"[0-9a-f]{32}", visitor_id(None, None))


def test_a_configured_seed_makes_two_processes_agree():
    """Without it each process invents its own and the same visitor is counted twice."""
    settings = ServiceSettings(analytics_salt="a-fixed-seed")
    reset_salt()
    first = visitor_id("203.0.113.9", "Mozilla/5.0", settings)
    reset_salt()
    assert visitor_id("203.0.113.9", "Mozilla/5.0", settings) == first


# --------------------------------------------------------------------------- referrers


@pytest.mark.parametrize(
    "referer,expected",
    [
        ("https://news.ycombinator.com/item?id=123", "news.ycombinator.com"),
        ("https://LinkedIn.com/feed", "linkedin.com"),
        ("http://example.org:8080/a/b", "example.org"),
    ],
)
def test_only_the_host_survives_a_referrer(referer, expected):
    assert referrer_host(referer) == expected


def test_a_search_referrer_keeps_none_of_the_query():
    """The query string on a search engine is what somebody typed to find you."""
    got = referrer_host("https://www.google.com/search?q=inventory+forecasting+for+my+shop")
    assert got == "www.google.com"
    assert "inventory" not in got
    assert "?" not in got


def test_our_own_pages_are_not_a_traffic_source():
    """A visitor moving between our own tabs is not a referral."""
    assert referrer_host("https://inventory-optimization-engine.fly.dev/") is None


@pytest.mark.parametrize("bad", [None, "", "not a url", "://", "  "])
def test_an_unusable_referrer_is_dropped_rather_than_guessed_at(bad):
    assert referrer_host(bad) is None


def test_the_referrer_host_is_on_the_allowlist():
    """A property the middleware sends but the allowlist rejects is silently dropped."""
    assert "referrer_host" in ALLOWED_PROPERTIES
