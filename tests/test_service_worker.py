"""The enqueue path and the worker's contract with it.

No Redis and no LightGBM here. What is worth testing at this seam is that the producer and
the consumer agree on a task name, that the endpoint writes a QUEUED row and nothing more,
and that a queue outage is reported rather than swallowed -- a job accepted into a queue
that is down sits in QUEUED forever looking merely slow.
"""

from __future__ import annotations

import io
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inventory_engine.service.app import create_app
from inventory_engine.service.db.models import Base, ForecastJob, JobStatus
from inventory_engine.service.db.session import get_session
from inventory_engine.service.jobs import FORECAST_TASK
from inventory_engine.service.routers import full_forecast
from inventory_engine.service.routers.full_forecast import get_storage
from inventory_engine.service.settings import ServiceSettings, get_settings
from inventory_engine.service.storage import LocalDiskStorage


def csv_bytes(sku="WIDGET", days=200, start="2025-01-01") -> bytes:
    """A CSV that clears the gate."""
    dates = pd.date_range(start, periods=days, freq="D")
    return (
        pd.DataFrame(
            {"sku": sku, "date": dates.strftime("%Y-%m-%d"), "units_sold": 5, "unit_price": 2.5}
        )
        .to_csv(index=False)
        .encode()
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client with a fake queue that records what was enqueued."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'w.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    storage = LocalDiskStorage(tmp_path / "uploads")
    settings = ServiceSettings(upload_dir=tmp_path / "uploads")

    enqueued: list[tuple] = []

    def fake_enqueue(redis_url, job_id, request_id):
        enqueued.append((redis_url, job_id, request_id))
        return f"arq:{job_id}"

    monkeypatch.setattr(full_forecast, "enqueue_forecast", fake_enqueue)

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
        c.enqueued = enqueued
        yield c


def upload(client, data=None):
    """Upload a CSV and return its dataset id."""
    response = client.post(
        "/upload", files={"file": ("s.csv", io.BytesIO(data or csv_bytes()), "text/csv")}
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


# --------------------------------------------------------------------------- contract


def test_the_producer_and_the_worker_agree_on_the_task_name():
    """A rename on either side orphans every job the API enqueues; catch it here."""
    from inventory_engine.service.worker import WorkerSettings

    registered = {f.name for f in WorkerSettings.functions}
    assert FORECAST_TASK in registered


def test_the_enqueue_module_does_not_import_the_pipeline():
    """Importing the worker to enqueue it would drag LightGBM into the API process."""
    import ast
    from pathlib import Path

    import inventory_engine

    source = (Path(inventory_engine.__file__).parent / "service" / "jobs.py").read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("pipeline" in m or "worker" in m for m in imported)


# --------------------------------------------------------------------------- run endpoint


def test_run_enqueues_and_returns_202_immediately(client):
    dataset_id = upload(client)
    response = client.post("/forecast/run", json={"dataset_id": dataset_id, "horizon": 28})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["skus_queued"] == 1
    assert body["poll_url"] == f"/forecast/{body['job_id']}"
    # CR from the default cost model: 30% margin, 60% spoilage. Not 0.95.
    assert body["critical_ratio"] == pytest.approx(0.4087, abs=1e-3)

    assert len(client.enqueued) == 1
    _, enqueued_job_id, _ = client.enqueued[0]
    assert enqueued_job_id == body["job_id"]


def test_the_job_row_is_written_as_queued_and_not_advanced_by_the_api(client):
    dataset_id = upload(client)
    job_id = client.post("/forecast/run", json={"dataset_id": dataset_id}).json()["job_id"]

    with client.factory() as session:
        job = session.get(ForecastJob, uuid.UUID(job_id))
        assert job.status is JobStatus.QUEUED
        assert job.started_at is None and job.finished_at is None
        assert job.horizon == 28
        assert job.margin_rate == pytest.approx(0.30)


def test_only_the_job_id_travels_on_the_queue(client):
    """Parameters live on the row; duplicating them into the payload invites disagreement."""
    dataset_id = upload(client)
    client.post("/forecast/run", json={"dataset_id": dataset_id, "horizon": 14})
    _, job_id, request_id = client.enqueued[0]
    assert uuid.UUID(job_id)
    assert isinstance(request_id, str) and request_id


def test_the_request_id_flows_from_the_header_onto_the_job(client):
    dataset_id = upload(client)
    job_id = client.post(
        "/forecast/run", json={"dataset_id": dataset_id}, headers={"X-Request-ID": "trace-99"}
    ).json()["job_id"]

    with client.factory() as session:
        assert session.get(ForecastJob, uuid.UUID(job_id)).request_id == "trace-99"
    assert client.enqueued[0][2] == "trace-99"


def test_custom_cost_rates_are_recorded_and_change_the_critical_ratio(client):
    dataset_id = upload(client)
    response = client.post(
        "/forecast/run",
        json={"dataset_id": dataset_id, "margin_rate": 0.5, "spoilage_rate": 0.1},
    )
    body = response.json()
    assert body["critical_ratio"] > 0.8

    with client.factory() as session:
        job = session.get(ForecastJob, uuid.UUID(body["job_id"]))
        assert job.margin_rate == 0.5 and job.spoilage_rate == 0.1


# --------------------------------------------------------------------------- refusals


def test_an_unknown_dataset_is_a_404(client):
    response = client.post(
        "/forecast/run", json={"dataset_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert response.status_code == 404
    assert client.enqueued == []


def test_a_horizon_outside_the_supported_range_is_rejected(client):
    dataset_id = upload(client)
    assert (
        client.post("/forecast/run", json={"dataset_id": dataset_id, "horizon": 0}).status_code
        == 422
    )
    assert (
        client.post("/forecast/run", json={"dataset_id": dataset_id, "horizon": 99}).status_code
        == 422
    )
    assert client.enqueued == []


def test_an_impossible_margin_rate_is_rejected_before_it_reaches_the_cost_model(client):
    dataset_id = upload(client)
    response = client.post("/forecast/run", json={"dataset_id": dataset_id, "margin_rate": 1.5})
    assert response.status_code == 422


def test_a_queue_outage_is_a_503_and_leaves_no_orphan_job(client, monkeypatch):
    """A row in QUEUED with nothing to pick it up is worse than an honest failure."""

    def boom(*_args, **_kwargs):
        raise ConnectionError("redis is down")

    dataset_id = upload(client)
    monkeypatch.setattr(full_forecast, "enqueue_forecast", boom)

    response = client.post("/forecast/run", json={"dataset_id": dataset_id})
    assert response.status_code == 503
    assert "queue" in response.json()["detail"]

    with client.factory() as session:
        assert session.query(ForecastJob).count() == 0


# --------------------------------------------------------------------------- worker mapping


def test_only_one_model_is_fitted_at_a_time():
    """Regression: concurrent LightGBM fits in one process deadlock in OpenMP.

    Two threads passed the demo upload twice at 12s and wedged on the third -- flat CPU,
    job stuck in ``running``, nothing after "forecast job starting" in the log. LightGBM
    releasing the GIL is true and irrelevant; its training loop is OMP-parallel and is not
    safe to enter twice at once. Parallelism lives in LightGBM's own ``num_threads`` and in
    arq's ``max_jobs``, not here.
    """
    from inventory_engine.service.worker import FIT_THREADS

    assert FIT_THREADS == 1


def test_nan_metrics_become_null_rather_than_nan():
    """NaN in a Float column is neither queryable nor JSON-serialisable."""
    from inventory_engine.service.worker import _finite

    assert _finite(float("nan")) is None
    assert _finite(None) is None
    assert _finite(1.5) == 1.5


def test_worker_rows_carry_the_method_and_both_methods_scores():
    from inventory_engine.service.pipeline import MethodScore, SkuForecast
    from inventory_engine.service.worker import _to_rows

    forecast = SkuForecast(
        sku="A",
        method_used="seasonal_naive",
        method_reason="baseline won",
        n_folds=3,
        model=MethodScore("quantile_gbm", (1.4, 1.5), (90.0, 92.0)),
        baseline=MethodScore("seasonal_naive", (1.1, 1.2), (50.0, 52.0)),
        critical_ratio=0.4087,
        order_qty=123.4,
        expected_cost=9.9,
        unit_price=2.5,
        price_is_fallback=False,
        series={"history": [], "forecast": []},
    )
    (row,) = _to_rows(uuid.uuid4(), [forecast])

    assert row.method_used == "seasonal_naive"
    assert row.mase_model == pytest.approx(1.45)
    assert row.mase_baseline == pytest.approx(1.15)
    assert row.pinball_model == pytest.approx(91.0)
    assert row.pinball_baseline == pytest.approx(51.0)
    assert row.order_qty == 123.4
