"""The tier 2 schema.

Postgres, not DuckDB
--------------------
Tier 1 reads DuckDB through a fresh read-only connection per request because DuckDB gives a
writer *exclusive* access to the file -- see ``api/deps.py``, which documents the constraint,
and ``api/precompute.py``, which works around it with a shadow copy and an atomic swap. That
design is sound for a single nightly writer and many readers of a static warehouse.

It cannot survive this workload. Here the writers are concurrent and unscheduled: every
upload writes, every job transition writes, every worker completion writes. A shadow-copy
swap has no meaning when there is no quiet moment to swap in.

Portable types on purpose
-------------------------
``Uuid`` and ``JSON`` are declared through SQLAlchemy's dialect-neutral types with a
Postgres variant attached, so the same models create native ``uuid``/``jsonb`` columns
against Postgres and ordinary ones against SQLite. That is not hedging about the database
choice -- production is Postgres -- it is what lets the endpoint tests run in-process
without a container, while the Postgres-specific behaviour is still exercised by the
integration tests that do use one.

Why the gate's verdict is stored
--------------------------------
``dataset_skus`` persists the per-SKU decision made at upload time. The worker then reads
which SKUs to forecast instead of re-running the gate, so the API's answer to "which
products were excluded and why" and the worker's idea of what to fit cannot drift apart.
Re-deriving it in two places is exactly the class of bug ``test_metric_ownership`` exists to
prevent in tier 1.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: JSONB on Postgres, plain JSON elsewhere. See the module docstring.
JsonBlob = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Declarative base for every tier 2 table."""


class JobStatus(enum.StrEnum):
    """Lifecycle of a forecast job. The four states the brief names, and no others."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Dataset(Base):
    """One uploaded CSV that passed the gate well enough to be worth storing.

    The file itself is not in this table. ``storage_uri`` points at it -- on disk today,
    S3 tomorrow, behind the same interface -- because holding a user's sales history as a
    bytes column would put it in every database backup and every query plan.
    """

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    original_filename: Mapped[str | None] = mapped_column(String(512))
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    #: Content hash of the stored bytes. Lets a re-upload of the same file be recognised,
    #: and lets a support question about "the file I sent" be answered exactly.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)

    rows_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sku_count_admitted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sku_count_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_date: Mapped[date | None] = mapped_column(Date)
    last_date: Mapped[date | None] = mapped_column(Date)

    #: Canonical name -> the column it was read from. Stored so a later "why did it
    #: forecast that column" question is answerable from the record alone.
    column_mapping: Mapped[dict] = mapped_column(JsonBlob, nullable=False, default=dict)
    #: Every transformation the gate applied, verbatim from ``GateReport.warnings()``.
    warnings: Mapped[list] = mapped_column(JsonBlob, nullable=False, default=list)

    skus: Mapped[list[DatasetSku]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[ForecastJob]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetSku(Base):
    """The gate's verdict on one SKU, with the numbers that produced it.

    One row per SKU in the upload, admitted or not. The rejected rows are kept rather than
    discarded: "which products did you drop, and why" is the question the gate exists to
    answer, and it has to survive past the upload response to be answerable later.
    """

    __tablename__ = "dataset_skus"
    #: The gate produces exactly one verdict per product, so a second row for the same
    #: ``(dataset, sku)`` means a bug upstream rather than legitimate data. Declared here
    #: as well as in the migration: the two must agree, or ``alembic --autogenerate`` reads
    #: the constraint as absent from the model and proposes dropping it.
    __table_args__ = (UniqueConstraint("dataset_id", "sku", name="uq_dataset_skus_dataset_sku"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sku: Mapped[str] = mapped_column(String(256), nullable=False)
    admitted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    n_days: Mapped[int] = mapped_column(Integer, nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    n_nonnull: Mapped[int] = mapped_column(Integer, nullable=False)
    max_gap_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_date: Mapped[date | None] = mapped_column(Date)
    last_date: Mapped[date | None] = mapped_column(Date)
    #: Empty for an admitted SKU. Each entry names a threshold and both values, e.g.
    #: "90 days of history needed, 21 found".
    reasons: Mapped[list] = mapped_column(JsonBlob, nullable=False, default=list)

    dataset: Mapped[Dataset] = relationship(back_populates="skus")


class ForecastJob(Base):
    """One enqueued forecast run over one dataset.

    Created by the API in ``QUEUED`` and never advanced by it. Every later transition is
    the worker's, which is what keeps "the API enqueues, the worker trains" true in the
    data as well as in the import graph.
    """

    __tablename__ = "forecast_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )

    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The cost assumptions this run used, mirroring ``optimize.costs.CostModel``. Stored
    #: per job because they are the caller's assumptions, not the service's, and the order
    #: quantities are meaningless without them.
    margin_rate: Mapped[float] = mapped_column(Float, nullable=False)
    spoilage_rate: Mapped[float] = mapped_column(Float, nullable=False)
    holding_rate: Mapped[float] = mapped_column(Float, nullable=False)

    #: Flows API -> queue -> worker so one request's log lines can be joined across both
    #: processes. Set by the API from the ``X-Request-ID`` header, or generated.
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    enqueued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Populated only in ``FAILED``. A message, not a traceback -- the traceback goes to
    #: Sentry, which is built to hold one.
    error: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped[Dataset] = relationship(back_populates="jobs")
    results: Mapped[list[ForecastResult]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ForecastResult(Base):
    """The forecast and order recommendation for one SKU in one job.

    Scalars are columns because the results table sorts and filters on them. The chart
    series is one JSONB blob because the chart reads a whole SKU at once: normalising it
    would be roughly 28 dates x 7 quantile levels of rows per SKU, fetched together every
    single time, joined back into the same shape on the way out.
    """

    __tablename__ = "forecast_results"
    #: One result per SKU per job. A duplicate would mean the worker wrote the same SKU
    #: twice, which the results view would render as two contradictory order quantities.
    __table_args__ = (UniqueConstraint("job_id", "sku", name="uq_forecast_results_job_sku"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("forecast_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sku: Mapped[str] = mapped_column(String(256), nullable=False)

    #: Which method actually produced the numbers returned -- ``"quantile_gbm"`` or
    #: ``"seasonal_naive"``. When the baseline wins on the user's own data, this says so
    #: rather than the service quietly serving the loser.
    method_used: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Why that method was chosen, in plain language, including the losing score.
    method_reason: Mapped[str] = mapped_column(Text, nullable=False)

    n_folds: Mapped[int] = mapped_column(Integer, nullable=False)
    mase_model: Mapped[float | None] = mapped_column(Float)
    mase_model_spread: Mapped[float | None] = mapped_column(Float)
    mase_baseline: Mapped[float | None] = mapped_column(Float)
    mase_baseline_spread: Mapped[float | None] = mapped_column(Float)
    #: Pinball loss at the critical ratio. MASE ranks point accuracy; the order policy
    #: consumes a quantile, so this is the score that actually governs the decision.
    pinball_model: Mapped[float | None] = mapped_column(Float)
    pinball_baseline: Mapped[float | None] = mapped_column(Float)

    critical_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    order_qty: Mapped[float] = mapped_column(Float, nullable=False)
    #: What the newsvendor asked for before any festival adjustment. Nullable because rows
    #: written before the festival feature existed genuinely do not have it, and a
    #: back-filled copy of ``order_qty`` would assert that nothing was adjusted -- which is
    #: unknowable for those rows and would be a lie for any that were.
    order_qty_before_festival: Mapped[float | None] = mapped_column(Float)
    #: The festival decision: state, factor, and every match with the keyword and category
    #: that produced it. A blob rather than columns because it is read whole with the row
    #: and never filtered on -- and because the shape belongs to
    #: ``service.adjust.FestivalPlan``, which is where it should be free to change.
    festival: Mapped[dict] = mapped_column(JsonBlob, nullable=False, default=dict)
    expected_cost: Mapped[float | None] = mapped_column(Float)
    unit_price: Mapped[float | None] = mapped_column(Float)
    price_is_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: ``{"history": [{"d", "v"}], "forecast": [{"d", "point", "lo", "hi"}],
    #:   "quantile_levels": {"lo": 0.1, "hi": 0.9}}`` -- everything the chart draws.
    series: Mapped[dict] = mapped_column(JsonBlob, nullable=False, default=dict)

    job: Mapped[ForecastJob] = relationship(back_populates="results")
