"""``GET /forecast/{job_id}`` across all four states.

The interesting assertions are about honesty rather than plumbing: both methods' scores
travel with every result, the gate's exclusions are still reachable from the results view,
and a partial success is not dressed up as a clean one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from inventory_engine.service.app import create_app
from inventory_engine.service.db.models import (
    Base,
    Dataset,
    DatasetSku,
    ForecastJob,
    ForecastResult,
    JobStatus,
)
from inventory_engine.service.db.session import get_session

SERIES = {
    "history": [{"d": "2025-01-01", "v": 5.0}, {"d": "2025-01-02", "v": 7.0}],
    "forecast": [{"d": "2025-01-03", "point": 6.0, "lo": 2.0, "hi": 11.0}],
    "band": {"low_level": 0.1, "high_level": 0.9, "label": "Where sales landed 80% of the time"},
    "order": {"total": 168.0, "daily_rate": 6.0, "label": "Order covers 28 days"},
}


@pytest.fixture
def client(tmp_path):
    """Client plus a factory, on a throwaway SQLite file."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'r.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    def session_override():
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    # Observability off: configure_logging replaces the root handlers, which would
    # take pytest's log capture with them, and no test may reach a real Sentry DSN.
    app = create_app(configure_observability=False)
    app.dependency_overrides[get_session] = session_override
    with TestClient(app) as c:
        c.factory = factory
        yield c


def seed(factory, *, status=JobStatus.DONE, results=1, excluded=0, error=None, horizon=28):
    """Insert a dataset and a job in the requested state."""
    with factory() as session:
        dataset = Dataset(
            id=uuid.uuid4(),
            storage_uri="file:///tmp/x.csv",
            sha256="0" * 64,
            byte_size=1,
            rows_read=200,
            sku_count_admitted=results,
            sku_count_rejected=excluded,
            column_mapping={"sku": "sku"},
            warnings=[],
        )
        job = ForecastJob(
            id=uuid.uuid4(),
            dataset=dataset,
            status=status,
            horizon=horizon,
            margin_rate=0.30,
            spoilage_rate=0.60,
            holding_rate=0.02,
            request_id="req-1",
            error=error,
        )
        if status in (JobStatus.DONE, JobStatus.FAILED):
            job.started_at = datetime.now(UTC) - timedelta(seconds=12)
            job.finished_at = datetime.now(UTC)
        session.add_all([dataset, job])

        for i in range(results):
            session.add(
                ForecastResult(
                    job=job,
                    sku=f"SKU{i}",
                    method_used="seasonal_naive" if i % 2 else "quantile_gbm",
                    method_reason="because the backtest said so",
                    n_folds=3,
                    mase_model=1.10 + i,
                    mase_model_spread=0.20,
                    mase_baseline=1.30 + i,
                    mase_baseline_spread=0.25,
                    pinball_model=40.0,
                    pinball_baseline=55.0,
                    critical_ratio=0.4087,
                    order_qty=100.0 + 10 * i,
                    expected_cost=9.5,
                    unit_price=2.5,
                    price_is_fallback=False,
                    series=SERIES,
                )
            )
        for i in range(excluded):
            session.add(
                DatasetSku(
                    dataset=dataset,
                    sku=f"THIN{i}",
                    admitted=False,
                    n_days=21,
                    n_obs=21,
                    n_nonnull=21,
                    max_gap_days=0,
                    reasons=["90 days of history needed, 21 found"],
                )
            )
        session.commit()
        return str(job.id)


# --------------------------------------------------------------------------- states


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_an_in_flight_job_reports_progress_and_no_results(client, status):
    job_id = seed(client.factory, status=status, results=0)
    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == status.value
    assert body["results"] == []
    assert body["message"]
    assert body["finished_at"] is None


def test_a_done_job_returns_results(client):
    job_id = seed(client.factory, results=3)
    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == "done"
    assert len(body["results"]) == 3
    assert body["message"] == "Done. 3 product(s) forecast."
    assert body["elapsed_seconds"] > 0


def test_a_failed_job_carries_its_error(client):
    job_id = seed(client.factory, status=JobStatus.FAILED, results=0, error="storage unreachable")
    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == "failed"
    assert body["error"] == "storage unreachable"
    assert "failed" in body["message"].lower()


def test_a_partial_success_is_not_dressed_up_as_a_clean_one(client):
    """Some SKUs failed, the rest produced results. Both facts have to survive."""
    job_id = seed(client.factory, results=2, error="SKU9: ValueError: bad series")
    body = client.get(f"/forecast/{job_id}").json()
    assert body["status"] == "done"
    assert len(body["results"]) == 2
    assert "with problems" in body["message"]
    assert "SKU9" in body["error"]


def test_an_unknown_job_is_a_404(client):
    assert client.get(f"/forecast/{uuid.uuid4()}").status_code == 404


# --------------------------------------------------------------------------- honesty


def test_both_methods_scores_travel_with_every_result(client):
    """Reporting only the winner's score would make the comparison unfalsifiable."""
    job_id = seed(client.factory, results=2)
    for result in client.get(f"/forecast/{job_id}").json()["results"]:
        a = result["accuracy"]
        assert a["mase_model"] is not None
        assert a["mase_baseline"] is not None
        assert a["pinball_model"] is not None
        assert a["pinball_baseline"] is not None
        assert result["method_reason"]


def test_the_excluded_skus_are_reachable_from_the_results_view(client):
    """'Why is this product missing from my order list' is asked here, not at upload."""
    job_id = seed(client.factory, results=1, excluded=2)
    body = client.get(f"/forecast/{job_id}").json()
    assert len(body["excluded"]) == 2
    assert body["excluded"][0]["reasons"] == ["90 days of history needed, 21 found"]
    assert body["excluded"][0]["admitted"] is False


def test_the_critical_ratio_is_derived_from_the_jobs_own_cost_rates(client):
    job_id = seed(client.factory, results=1)
    assert client.get(f"/forecast/{job_id}").json()["critical_ratio"] == pytest.approx(
        0.4087, abs=1e-3
    )


# --------------------------------------------------------------------------- chart payload


def test_the_order_is_also_given_as_a_daily_rate(client):
    """A 28-day total plotted on a per-day axis would sit far above every point."""
    job_id = seed(client.factory, results=1, horizon=28)
    result = client.get(f"/forecast/{job_id}").json()["results"][0]
    assert result["order_qty"] == 100.0
    assert result["order_daily_rate"] == pytest.approx(100.0 / 28)


def test_the_chart_payload_is_complete_and_the_band_is_plain_language(client):
    job_id = seed(client.factory, results=1)
    series = client.get(f"/forecast/{job_id}").json()["results"][0]["series"]
    assert series["history"] and series["forecast"]
    assert "q10" not in series["band"]["label"] and "q0.1" not in series["band"]["label"]
    assert series["band"]["label"] == "Where sales landed 80% of the time"
    for point in series["forecast"]:
        assert point["lo"] <= point["point"] <= point["hi"]


def test_results_are_ordered_by_size_so_the_biggest_orders_are_first(client):
    job_id = seed(client.factory, results=3)
    quantities = [r["order_qty"] for r in client.get(f"/forecast/{job_id}").json()["results"]]
    assert quantities == sorted(quantities, reverse=True)
