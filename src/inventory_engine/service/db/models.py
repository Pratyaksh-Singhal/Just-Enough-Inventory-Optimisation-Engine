"""The tier 2 schema."""

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
    """One uploaded CSV that passed the gate well enough to be worth storing."""

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
    """The gate's verdict on one SKU, with the numbers that produced it."""

    __tablename__ = "dataset_skus"
    #: The gate produces exactly one verdict per product, so a second row for the same ``(dataset,
    #: sku)`` means a bug upstream rather than legitimate data.
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
    """One enqueued forecast run over one dataset."""

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
    #: The cost assumptions this run used, mirroring ``optimize.costs.CostModel``.
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
    """The forecast and order recommendation for one SKU in one job."""

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
    #: ``"seasonal_naive"``.
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
    #: What the newsvendor asked for before any festival adjustment.
    order_qty_before_festival: Mapped[float | None] = mapped_column(Float)
    #: The festival decision: state, factor, and every match with the keyword and category that
    #: produced it.
    festival: Mapped[dict] = mapped_column(JsonBlob, nullable=False, default=dict)
    expected_cost: Mapped[float | None] = mapped_column(Float)
    unit_price: Mapped[float | None] = mapped_column(Float)
    price_is_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: ``{"history": [{"d", "v"}], "forecast": [{"d", "point", "lo", "hi"}],
    #:   "quantile_levels": {"lo": 0.1, "hi": 0.9}}`` -- everything the chart draws.
    series: Mapped[dict] = mapped_column(JsonBlob, nullable=False, default=dict)

    job: Mapped[ForecastJob] = relationship(back_populates="results")
