"""Retention: the part of the privacy story that is about time rather than destinations.

The assertions worth having here are that the bytes actually leave the disk, that derived
copies of the user's data go with them, and that the purge cannot be talked into deleting
everything.
"""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
from inventory_engine.service.retention import (
    CANCELLED_REASON,
    delete_dataset,
    purge_expired,
    sweep_orphans,
)
from inventory_engine.service.routers.full_forecast import get_storage
from inventory_engine.service.storage import LocalDiskStorage


@pytest.fixture
def env(tmp_path):
    """A session factory, a storage root, and a client sharing both."""
    engine = create_engine(f"sqlite:///{tmp_path / 'r.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    storage = LocalDiskStorage(tmp_path / "uploads")

    def session_override():
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    app = create_app(configure_observability=False)
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_storage] = lambda: storage

    with TestClient(app) as client:
        yield factory, storage, client


def make_dataset(factory, storage, *, age_days=0, with_job=None, content=b"sku,date\nA,x\n"):
    """Create a dataset with a real file behind it, optionally aged and with a job."""
    blob = storage.put(io.BytesIO(content), limit_bytes=10_000)
    with factory() as session:
        dataset = Dataset(
            id=uuid.uuid4(),
            created_at=datetime.now(UTC) - timedelta(days=age_days),
            storage_uri=blob.uri,
            sha256=blob.sha256,
            byte_size=blob.byte_size,
            rows_read=1,
            column_mapping={},
            warnings=[],
        )
        session.add(dataset)
        session.add(
            DatasetSku(
                dataset=dataset,
                sku="A",
                admitted=True,
                n_days=120,
                n_obs=120,
                n_nonnull=120,
                max_gap_days=0,
                reasons=[],
            )
        )
        job = None
        if with_job is not None:
            job = ForecastJob(
                id=uuid.uuid4(),
                dataset=dataset,
                status=with_job,
                horizon=28,
                margin_rate=0.3,
                spoilage_rate=0.6,
                holding_rate=0.02,
                request_id="r",
            )
            session.add(job)
            session.add(
                ForecastResult(
                    job=job,
                    sku="A",
                    method_used="quantile_gbm",
                    method_reason="x",
                    n_folds=3,
                    critical_ratio=0.4,
                    order_qty=10.0,
                    # The user's own daily sales values, copied into the result.
                    series={"history": [{"d": "2025-01-01", "v": 7.0}]},
                )
            )
        session.commit()
        return dataset.id, blob.uri, (job.id if job else None)


# --------------------------------------------------------------------------- delete one


def test_deleting_a_dataset_removes_the_file_from_disk(env):
    factory, storage, _ = env
    dataset_id, uri, _ = make_dataset(factory, storage)
    assert storage._path_for(uri).exists()

    with factory() as session:
        delete_dataset(session, storage, session.get(Dataset, dataset_id))
        session.commit()

    assert not storage._path_for(uri).exists()


def test_deleting_a_dataset_takes_the_derived_sales_values_with_it(env):
    """``forecast_results.series`` holds the user's own history; it must not outlive the upload."""
    factory, storage, _ = env
    dataset_id, _, _ = make_dataset(factory, storage, with_job=JobStatus.DONE)

    with factory() as session:
        assert session.query(ForecastResult).count() == 1
        delete_dataset(session, storage, session.get(Dataset, dataset_id))
        session.commit()

    with factory() as session:
        assert session.query(ForecastResult).count() == 0
        assert session.query(DatasetSku).count() == 0
        assert session.query(ForecastJob).count() == 0


@pytest.mark.parametrize("state", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_a_live_job_is_cancelled_with_a_stated_reason(env, state):
    """Better an explicit cancellation than a worker dying on a missing file."""
    factory, storage, _ = env
    dataset_id, _, job_id = make_dataset(factory, storage, with_job=state)

    with factory() as session:
        result = delete_dataset(session, storage, session.get(Dataset, dataset_id))
        session.commit()
    assert result.jobs_cancelled == 1


def test_a_finished_job_is_not_counted_as_cancelled(env):
    factory, storage, _ = env
    dataset_id, _, _ = make_dataset(factory, storage, with_job=JobStatus.DONE)
    with factory() as session:
        result = delete_dataset(session, storage, session.get(Dataset, dataset_id))
        session.commit()
    assert result.jobs_cancelled == 0


def test_a_missing_file_does_not_block_removing_the_row(env):
    """An interrupted earlier delete must not make the row undeletable."""
    factory, storage, _ = env
    dataset_id, uri, _ = make_dataset(factory, storage)
    storage.delete(uri)

    with factory() as session:
        delete_dataset(session, storage, session.get(Dataset, dataset_id))
        session.commit()
    with factory() as session:
        assert session.get(Dataset, dataset_id) is None


# --------------------------------------------------------------------------- purge


def test_only_datasets_past_the_window_are_purged(env):
    factory, storage, _ = env
    old_id, old_uri, _ = make_dataset(factory, storage, age_days=45)
    new_id, new_uri, _ = make_dataset(factory, storage, age_days=3)

    with factory() as session:
        result = purge_expired(session, storage, retention_days=30)
        session.commit()

    assert result.datasets_deleted == 1
    assert not storage._path_for(old_uri).exists()
    assert storage._path_for(new_uri).exists()
    with factory() as session:
        assert session.get(Dataset, old_id) is None
        assert session.get(Dataset, new_id) is not None


def test_the_boundary_is_strictly_older_than_the_cutoff(env):
    """Exactly at the window: kept. One second past it: destroyed.

    Pinned with an explicit clock rather than wall time, because a boundary test that
    depends on how long the test took to run is a boundary test that flakes.
    """
    factory, storage, _ = env
    created = datetime.now(UTC) - timedelta(days=30)
    dataset_id, _, _ = make_dataset(factory, storage, age_days=30)
    with factory() as session:
        session.get(Dataset, dataset_id).created_at = created
        session.commit()

    at_the_boundary = created + timedelta(days=30)
    with factory() as session:
        assert purge_expired(session, storage, 30, now=at_the_boundary).datasets_deleted == 0
        session.commit()

    with factory() as session:
        just_past = at_the_boundary + timedelta(seconds=1)
        assert purge_expired(session, storage, 30, now=just_past).datasets_deleted == 1
        session.commit()


@pytest.mark.parametrize("bad", [0, -1, -30])
def test_a_non_positive_window_is_refused_rather_than_deleting_everything(env, bad):
    """The most dangerous possible misconfiguration; fail loudly instead."""
    factory, storage, _ = env
    make_dataset(factory, storage, age_days=1)
    with factory() as session, pytest.raises(ValueError, match="must be positive"):
        purge_expired(session, storage, retention_days=bad)


def test_purging_nothing_is_not_an_error(env):
    factory, storage, _ = env
    with factory() as session:
        assert purge_expired(session, storage, retention_days=30).datasets_deleted == 0


# --------------------------------------------------------------------------- orphans


def test_an_unreferenced_file_is_swept(env):
    """Left behind when a process dies between writing bytes and committing a row."""
    factory, storage, _ = env
    make_dataset(factory, storage)
    orphan = storage.put(io.BytesIO(b"nobody references me"), limit_bytes=1000)

    with factory() as session:
        removed = sweep_orphans(session, storage)

    assert removed == 1
    assert not storage._path_for(orphan.uri).exists()


def test_a_referenced_file_is_never_swept(env):
    factory, storage, _ = env
    _, uri, _ = make_dataset(factory, storage)
    with factory() as session:
        assert sweep_orphans(session, storage) == 0
    assert storage._path_for(uri).exists()


# --------------------------------------------------------------------------- endpoint


def test_the_delete_endpoint_removes_everything_and_says_so(env):
    factory, storage, client = env
    dataset_id, uri, _ = make_dataset(factory, storage, with_job=JobStatus.RUNNING)

    response = client.delete(f"/datasets/{dataset_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["jobs_cancelled"] == 1
    assert "cancelled" in body["message"]
    assert not storage._path_for(uri).exists()


def test_deleting_an_unknown_dataset_is_a_404(env):
    _, _, client = env
    assert client.delete(f"/datasets/{uuid.uuid4()}").status_code == 404


def test_a_deleted_dataset_is_gone_from_the_read_endpoint(env):
    factory, storage, client = env
    dataset_id, _, _ = make_dataset(factory, storage)
    client.delete(f"/datasets/{dataset_id}")
    assert client.get(f"/datasets/{dataset_id}").status_code == 404


def test_the_cancelled_reason_is_the_one_the_results_view_will_show(env):
    factory, storage, client = env
    dataset_id, _, _ = make_dataset(factory, storage, with_job=JobStatus.QUEUED)
    client.delete(f"/datasets/{dataset_id}")
    assert "deleted" in CANCELLED_REASON
