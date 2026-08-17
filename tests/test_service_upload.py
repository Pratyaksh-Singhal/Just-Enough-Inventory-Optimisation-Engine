"""``POST /upload`` and ``GET /health`` against a real database.

SQLite here, Postgres in production. The models declare their types through SQLAlchemy's
dialect-neutral ``Uuid``/``JSON`` with a Postgres variant attached (see
``service/db/models.py``), so these run in-process with no container while production still
gets native ``uuid`` and ``jsonb``. ``tests/test_service_postgres.py`` covers what only a
real Postgres can show.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inventory_engine.service.app import create_app
from inventory_engine.service.db.models import Base, Dataset, DatasetSku
from inventory_engine.service.db.session import get_session
from inventory_engine.service.routers.full_forecast import get_storage
from inventory_engine.service.settings import ServiceSettings, get_settings
from inventory_engine.service.storage import LocalDiskStorage


def csv_bytes(sku="WIDGET", days=120, start="2025-01-01", units=5, price=2.5) -> bytes:
    """A clean daily CSV for one SKU."""
    dates = pd.date_range(start, periods=days, freq="D")
    frame = pd.DataFrame(
        {
            "sku": sku,
            "date": dates.strftime("%Y-%m-%d"),
            "units_sold": units,
            "unit_price": price,
        }
    )
    return frame.to_csv(index=False).encode()


@pytest.fixture
def client(tmp_path):
    """A test client wired to a fresh SQLite file and a temp upload directory."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    storage = LocalDiskStorage(tmp_path / "uploads")
    settings = ServiceSettings(upload_dir=tmp_path / "uploads", max_upload_bytes=1024 * 1024)

    def session_override():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Observability off: configure_logging replaces the root handlers, which would
    # take pytest's log capture with them, and no test may reach a real Sentry DSN.
    app = create_app(configure_observability=False)
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as c:
        c.factory = factory
        c.storage = storage
        yield c


def post(client, data: bytes, name="sales.csv"):
    """POST a CSV to /upload."""
    return client.post("/upload", files={"file": (name, io.BytesIO(data), "text/csv")})


# --------------------------------------------------------------------------- happy path


def test_a_good_upload_is_stored_and_returns_a_dataset_id(client):
    response = post(client, csv_bytes())
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["rows_read"] == 120
    assert [v["sku"] for v in body["admitted"]] == ["WIDGET"]
    assert body["excluded"] == []
    assert body["column_mapping"]["units_sold"] == "units_sold"

    with client.factory() as session:
        dataset = session.get(Dataset, __import__("uuid").UUID(body["dataset_id"]))
        assert dataset.sku_count_admitted == 1
        assert dataset.byte_size > 0
        assert len(dataset.sha256) == 64


def test_the_stored_file_survives_the_request(client):
    body = post(client, csv_bytes()).json()
    with client.factory() as session:
        dataset = session.get(Dataset, __import__("uuid").UUID(body["dataset_id"]))
    with client.storage.open(dataset.storage_uri) as handle:
        assert len(pd.read_csv(handle)) == 120


def test_every_sku_is_persisted_admitted_or_not(client):
    data = (csv_bytes("GOOD", days=120).decode() + "\n").encode() + b"".join(
        line + b"\n" for line in csv_bytes("THIN", days=30).split(b"\n")[1:] if line
    )
    body = post(client, data).json()
    assert response_skus(body, "admitted") == ["GOOD"]
    assert response_skus(body, "excluded") == ["THIN"]

    with client.factory() as session:
        rows = session.query(DatasetSku).all()
        assert {r.sku: r.admitted for r in rows} == {"GOOD": True, "THIN": False}
        thin = next(r for r in rows if r.sku == "THIN")
        assert "90 days of history needed, 30 found" in thin.reasons[0]


def response_skus(body, key):
    """SKU names from one half of an upload response."""
    return [v["sku"] for v in body[key]]


# --------------------------------------------------------------------------- refusals


def test_a_too_small_upload_is_refused_with_the_shortfall(client):
    response = post(client, csv_bytes(days=21))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "90 days of history needed, 21 found" in detail["detail"]
    assert detail["use_quick_calculator"] is True
    assert detail["excluded"][0]["n_days"] == 21


def test_a_refused_upload_leaves_nothing_stored(client):
    post(client, csv_bytes(days=21))
    with client.factory() as session:
        assert session.query(Dataset).count() == 0
    assert list(client.storage.root.glob("*.csv")) == []


def test_a_missing_column_is_refused(client):
    frame = pd.read_csv(io.BytesIO(csv_bytes())).drop(columns=["units_sold"])
    response = post(client, frame.to_csv(index=False).encode())
    assert response.status_code == 422
    assert "'units_sold'" in response.json()["detail"]["detail"]


def test_an_oversized_upload_is_refused_before_it_is_parsed(client):
    client.app.dependency_overrides[get_settings] = lambda: ServiceSettings(
        upload_dir=client.storage.root, max_upload_bytes=512
    )
    response = post(client, csv_bytes(days=120))
    assert response.status_code == 413
    assert list(client.storage.root.glob("*.csv")) == []


def test_binary_content_is_refused_as_unreadable(client):
    response = post(client, b"PK\x03\x04" + bytes(range(256)) * 4, name="sales.xlsx")
    assert response.status_code == 422
    assert "UTF-8" in response.json()["detail"] or "Could not read" in response.json()["detail"]


# --------------------------------------------------------------------------- readback


def test_a_dataset_can_be_read_back_without_re_uploading(client):
    dataset_id = post(client, csv_bytes()).json()["dataset_id"]
    response = client.get(f"/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.json()["admitted"][0]["sku"] == "WIDGET"


def test_an_unknown_dataset_is_a_404(client):
    response = client.get("/datasets/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


# --------------------------------------------------------------------------- health


def test_health_reports_each_dependency_separately(client):
    """The invariant, not the environment.

    An earlier version of this asserted ``queue is False`` because no Redis was running at
    the time. That passed for the wrong reason and then failed the moment ``docker compose
    up`` made it true -- a test pinned to an accident of the machine rather than to
    behaviour. What is actually promised is that each dependency is probed independently
    and ``status`` is the conjunction.
    """
    body = client.get("/health").json()
    assert body["database"] is True
    assert body["storage_writable"] is True
    assert isinstance(body["queue"], bool)
    expected = all((body["database"], body["queue"], body["storage_writable"]))
    assert body["status"] == ("ok" if expected else "degraded")


def test_a_dead_queue_degrades_health_without_failing_the_endpoint(client, monkeypatch):
    """The separation is the point: a queue outage must not take the probe down with it."""
    from inventory_engine.service.routers import full_forecast

    def unreachable(_url):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(full_forecast, "_ping_redis", unreachable)
    body = client.get("/health").json()

    assert body["queue"] is False
    assert body["database"] is True
    assert body["status"] == "degraded"


def test_cors_allows_every_method_the_api_actually_exposes(client):
    """Regression: DELETE /datasets/{id} shipped while the preflight said "GET, POST".

    The API offered a way to delete your own data and the browser was forbidden from
    calling it. Derived from the route table rather than hard-coded, so the next endpoint
    with a new verb fails here instead of in someone's console.
    """
    exposed = {
        method
        for route in client.app.routes
        for method in getattr(route, "methods", set())
        if method not in {"HEAD", "OPTIONS"}
    }
    response = client.options(
        "/upload",
        headers={"Origin": "http://localhost:8080", "Access-Control-Request-Method": "POST"},
    )
    allowed = {m.strip() for m in response.headers["access-control-allow-methods"].split(",")}
    assert exposed <= allowed, f"CORS forbids {sorted(exposed - allowed)}, which the API exposes"


def test_the_request_id_is_echoed_back(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"


def test_a_request_id_is_generated_when_absent(client):
    assert client.get("/health").headers["X-Request-ID"]


# ------------------------------------------------------------------ the dashboard mount


def test_the_dashboard_is_served_at_the_root_when_configured(tmp_path, monkeypatch):
    """The deployed service serves the page as well as the API, from one origin.

    Not a convenience. The dashboard's Full forecast tab fetches this API, and the published
    Artifact runtime blocks requests to any external host -- so a page hosted anywhere other
    than beside its own API cannot reach it. Same-origin removes the problem rather than
    working around it.
    """
    from fastapi.testclient import TestClient

    from inventory_engine.service.app import create_app

    page = tmp_path / "dash"
    page.mkdir()
    (page / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DIR", str(page))

    with TestClient(create_app(configure_observability=False)) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "dashboard" in root.text


def test_the_api_routes_win_over_the_static_mount(tmp_path, monkeypatch):
    """A mount at "/" must not shadow the endpoints.

    An index.html that happened to sit at dashboard/health would otherwise be served in
    place of the health check, and the failure would look like the API had vanished.
    """
    from fastapi.testclient import TestClient

    from inventory_engine.service.app import create_app

    page = tmp_path / "dash"
    page.mkdir()
    (page / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
    (page / "health").write_text("NOT THE API", encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DIR", str(page))

    with TestClient(create_app(configure_observability=False)) as client:
        body = client.get("/health").json()
        assert set(body) >= {"status", "database", "queue", "storage_writable", "version"}


def test_no_dashboard_configured_leaves_the_api_alone(monkeypatch):
    """Unset is the normal case for a worker or a bare API. It must not raise on startup."""
    from fastapi.testclient import TestClient

    from inventory_engine.service.app import create_app

    monkeypatch.delenv("DASHBOARD_DIR", raising=False)
    with TestClient(create_app(configure_observability=False)) as client:
        assert client.get("/").status_code == 404
        assert client.get("/health").status_code == 200


def test_a_configured_but_missing_dashboard_dir_does_not_break_the_api(tmp_path, monkeypatch):
    """A wrong path must degrade to "no page", never to a dead service.

    Mounting StaticFiles on a directory that is not there raises at construction time, which
    would take the whole API down over a static file -- the same class of fault as resolving
    the festival table from a repository root that does not exist once installed.
    """
    from fastapi.testclient import TestClient

    from inventory_engine.service.app import create_app

    monkeypatch.setenv("DASHBOARD_DIR", str(tmp_path / "nope"))
    with TestClient(create_app(configure_observability=False)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 404
