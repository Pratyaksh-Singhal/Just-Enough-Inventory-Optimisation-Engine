"""What only a real Postgres can show.

``tests/test_service_upload.py`` runs the endpoints on SQLite, which proves the handler
logic but says nothing about the two things the schema actually depends on Postgres for:
native ``jsonb`` columns, and the ``job_status`` enum type. Those are exercised here.

Skipped unless a Postgres is reachable, so ``pytest`` stays a no-setup command. Bring one
up with ``docker compose up -d db`` and these run; without it they skip loudly rather than
passing vacuously.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from inventory_engine.service.db.models import Base, Dataset, DatasetSku, ForecastJob, JobStatus

#: Where a Postgres is expected to be listening. Only ever connected to in order to reach
#: the server -- never written to, never dropped. See :data:`TEST_DB`.
ADMIN_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://inventory:inventory@localhost:5432/inventory",
)

#: Databases these tests own outright and may destroy. Deliberately *not* the database the
#: application uses.
#:
#: The first version of this file pointed the fixture straight at ``ADMIN_URL`` and called
#: ``Base.metadata.drop_all`` in teardown. That is a test suite that deletes the developer's
#: schema as a side effect of running, and it did exactly that: an end-to-end run
#: immediately afterwards failed with ``relation "datasets" does not exist``, because
#: ``pytest`` had dropped the tables the migration had just created. A test may destroy only
#: what it created.
TEST_DB = "inventory_pytest"
MIGRATION_CHECK_DB = "inventory_migration_check"


def _url_for(database: str) -> str:
    """Swap the database name in :data:`ADMIN_URL`, keeping host and credentials."""
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{database}"


def _reachable(url: str) -> bool:
    """Whether a Postgres is listening at ``url``."""
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 2})
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception:  # noqa: BLE001 - driver errors vary by failure mode
        return False
    return True


def _recreate(database: str) -> None:
    """Drop and recreate ``database``, evicting anything still connected to it."""
    admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :d AND pid <> pg_backend_pid()"
            ),
            {"d": database},
        )
        connection.execute(text(f'DROP DATABASE IF EXISTS "{database}"'))
        connection.execute(text(f'CREATE DATABASE "{database}"'))
    admin.dispose()


def _drop(database: str) -> None:
    """Remove a database these tests created."""
    admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :d AND pid <> pg_backend_pid()"
            ),
            {"d": database},
        )
        connection.execute(text(f'DROP DATABASE IF EXISTS "{database}"'))
    admin.dispose()


pytestmark = pytest.mark.skipif(
    not _reachable(ADMIN_URL),
    reason=f"no Postgres at {ADMIN_URL}; run `docker compose up -d db`",
)


@pytest.fixture
def session():
    """A session against a throwaway database, created and dropped by this fixture."""
    _recreate(TEST_DB)
    engine = create_engine(_url_for(TEST_DB), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    try:
        with factory() as s:
            yield s
    finally:
        engine.dispose()
        _drop(TEST_DB)


def _dataset(**overrides) -> Dataset:
    """A minimal valid dataset row."""
    defaults = dict(
        id=uuid.uuid4(),
        storage_uri="file:///tmp/x.csv",
        sha256="0" * 64,
        byte_size=10,
        rows_read=120,
        column_mapping={"sku": "sku"},
        warnings=[],
    )
    return Dataset(**{**defaults, **overrides})


def test_the_migration_and_the_models_agree(monkeypatch):
    """Apply the migration to a scratch database and diff it against ``Base.metadata``.

    This is the test that matters most in this file, and the one that was missing. Every
    other test here builds its schema with ``create_all`` from the models, so it can only
    ever confirm the models are self-consistent -- it cannot see the migration at all.

    That gap hid a real drift: both unique constraints were written in the migration and
    omitted from the models. Production got them (migrations run there), tests did not, and
    the next ``alembic revision --autogenerate`` would have read them as deleted and
    proposed dropping them.
    """
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    import inventory_engine

    root = Path(inventory_engine.__file__).parents[2]
    _recreate(MIGRATION_CHECK_DB)
    scratch_url = _url_for(MIGRATION_CHECK_DB)
    try:
        # env.py reads the URL from ServiceSettings, which is deliberately uncached.
        monkeypatch.setenv("DATABASE_URL", scratch_url)
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "migrations"))
        command.upgrade(config, "head")

        engine = create_engine(scratch_url)
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True, "target_metadata": Base.metadata}
            )
            diff = compare_metadata(context, Base.metadata)
        engine.dispose()
    finally:
        _drop(MIGRATION_CHECK_DB)

    assert diff == [], (
        "the migration and the ORM models have drifted:\n  "
        + "\n  ".join(str(d) for d in diff)
    )


def test_json_columns_are_jsonb_not_json(session):
    """``jsonb`` is indexable and ``json`` is not; the variant must survive to Postgres."""
    columns = {c["name"]: c["type"] for c in inspect(session.bind).get_columns("datasets")}
    assert str(columns["column_mapping"]).upper() == "JSONB"
    assert str(columns["warnings"]).upper() == "JSONB"


def test_jsonb_round_trips_a_nested_structure(session):
    dataset = _dataset(column_mapping={"sku": "Product", "units_sold": "QTY"}, warnings=["a", "b"])
    session.add(dataset)
    session.commit()
    session.expunge_all()

    loaded = session.get(Dataset, dataset.id)
    assert loaded.column_mapping == {"sku": "Product", "units_sold": "QTY"}
    assert loaded.warnings == ["a", "b"]


def test_the_job_status_enum_type_exists_and_rejects_a_bad_value(session):
    dataset = _dataset()
    job = ForecastJob(
        id=uuid.uuid4(),
        dataset=dataset,
        horizon=28,
        margin_rate=0.3,
        spoilage_rate=0.6,
        holding_rate=0.02,
        request_id="req-1",
    )
    session.add_all([dataset, job])
    session.commit()

    assert session.get(ForecastJob, job.id).status is JobStatus.QUEUED
    # Postgres rejects the literal at the type level, which is the point of using a real
    # enum rather than a VARCHAR with a comment about what belongs in it.
    with pytest.raises(DBAPIError):
        session.execute(
            text("UPDATE forecast_jobs SET status = 'nonsense' WHERE id = :i"), {"i": job.id}
        )
        session.commit()


def test_deleting_a_dataset_cascades_to_its_skus(session):
    """The gate's per-SKU verdicts must not outlive the dataset they describe."""
    dataset = _dataset()
    session.add(dataset)
    session.add(
        DatasetSku(
            dataset=dataset, sku="A", admitted=True, n_days=120, n_obs=120, n_nonnull=120,
            max_gap_days=0, reasons=[],
        )
    )
    session.commit()

    session.delete(dataset)
    session.commit()
    assert session.query(DatasetSku).count() == 0


def test_the_same_sku_cannot_be_recorded_twice_for_one_dataset(session):
    """A duplicate would mean the gate emitted two verdicts for one product."""
    from sqlalchemy.exc import IntegrityError

    dataset = _dataset()
    session.add(dataset)
    for _ in range(2):
        session.add(
            DatasetSku(
                dataset=dataset, sku="A", admitted=True, n_days=120, n_obs=120,
                n_nonnull=120, max_gap_days=0, reasons=[],
            )
        )
    with pytest.raises(IntegrityError):
        session.commit()
