"""Request and response models for the tier 2 endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from inventory_engine.service.gate import GateReport, SkuVerdict


class SkuVerdictOut(BaseModel):
    """Why one SKU was admitted or excluded, with the numbers behind the decision."""

    sku: str
    admitted: bool
    n_days: int = Field(description="Calendar span of this product's history, in days")
    n_obs: int = Field(description="Rows for this product after cleaning")
    n_nonnull: int = Field(description="Rows with a recorded units figure. Zero counts.")
    max_gap_days: int = Field(description="Longest stretch inside the span with no rows")
    first_date: date | None
    last_date: date | None
    reasons: list[str] = Field(
        default_factory=list,
        description="Empty when admitted. Each entry names the threshold and both values.",
    )

    @classmethod
    def of(cls, verdict: SkuVerdict) -> SkuVerdictOut:
        """Build from the gate's own dataclass."""
        return cls(
            sku=verdict.sku,
            admitted=verdict.admitted,
            n_days=verdict.n_days,
            n_obs=verdict.n_obs,
            n_nonnull=verdict.n_nonnull,
            max_gap_days=verdict.max_gap_days,
            first_date=verdict.first_date,
            last_date=verdict.last_date,
            reasons=list(verdict.reasons),
        )


class UploadResponse(BaseModel):
    """``POST /upload`` when at least one SKU can be forecast.

    A partial pass is still a success: the dataset exists, ``excluded`` says what was left
    out and why, and the caller decides whether that is acceptable before running anything.
    """

    dataset_id: uuid.UUID
    rows_read: int
    admitted: list[SkuVerdictOut]
    excluded: list[SkuVerdictOut]
    column_mapping: dict[str, str] = Field(
        description="Which column in your file was read as which field"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Every transformation applied while reading -- merges, clips, drops",
    )
    message: str


class UploadRefusal(BaseModel):
    """``POST /upload`` when the file cannot support a forecast at all.

    Returned with 422. ``excluded`` carries the per-SKU shortfall so the caller can see how
    far short they are, not merely that they are short.
    """

    detail: str
    excluded: list[SkuVerdictOut] = Field(default_factory=list)
    column_mapping: dict[str, str] = Field(default_factory=dict)
    rows_read: int = 0
    use_quick_calculator: bool = Field(
        default=True,
        description="The browser-only calculator handles this data; this service will not.",
    )

    @classmethod
    def of(cls, report: GateReport, detail: str) -> UploadRefusal:
        """Build from a failed gate report."""
        return cls(
            detail=detail,
            excluded=[SkuVerdictOut.of(v) for v in report.rejected],
            column_mapping=report.column_mapping,
            rows_read=report.rows_read,
        )


class ForecastRunRequest(BaseModel):
    """Body for ``POST /forecast/run``.

    The cost rates arrive per request rather than being service constants, because they are
    the caller's business, not ours. Their defaults match
    :mod:`inventory_engine.optimize.costs` exactly, so tier 1's calculator and tier 2 start
    from the same assumptions -- and both label them as assumptions.
    """

    dataset_id: uuid.UUID
    horizon: int = Field(default=28, ge=1, le=28, description="Days the order must cover, 1-28")
    margin_rate: float = Field(
        default=0.30, gt=0.0, lt=1.0, description="Gross margin as a fraction of shelf price"
    )
    spoilage_rate: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Fraction of unsold units written off"
    )
    holding_rate: float = Field(
        default=0.02, ge=0.0, description="Daily carrying cost as a fraction of unit cost"
    )


class ForecastRunResponse(BaseModel):
    """``POST /forecast/run`` — accepted, not completed.

    Returns immediately with 202. Fitting happens on the worker; poll
    ``GET /forecast/{job_id}``.
    """

    job_id: uuid.UUID
    status: Literal["queued"]
    dataset_id: uuid.UUID
    horizon: int
    critical_ratio: float = Field(
        description="Service level the order targets, derived from the cost rates -- "
        "not chosen. CR = Cu / (Cu + Co)."
    )
    skus_queued: int
    poll_url: str


class SkuAccuracy(BaseModel):
    """How both methods scored on this SKU's own history.

    Both are always present, whichever won. A service that reported only the winner's score
    would be unfalsifiable: there would be no way to see that the "model" it is serving is
    the seasonal naive it was supposed to beat.
    """

    n_folds: int = Field(description="Rolling-origin folds this history supported")
    mase_model: float | None = None
    mase_model_spread: float | None = Field(
        default=None, description="Half the fold range. Null when fewer than two folds scored."
    )
    mase_baseline: float | None = None
    mase_baseline_spread: float | None = None
    pinball_model: float | None = Field(
        default=None, description="Loss at the critical ratio -- the metric that chose"
    )
    pinball_baseline: float | None = None


class SkuResult(BaseModel):
    """The forecast, the order and the evidence, for one SKU."""

    sku: str
    method_used: str = Field(description="'quantile_gbm' or 'seasonal_naive'")
    method_reason: str = Field(description="Why, in plain language, including the loser's score")

    order_qty: float = Field(description="Units to cover the whole horizon")
    order_daily_rate: float = Field(description="order_qty / horizon, for the chart's axis")
    expected_cost: float | None = None
    unit_price: float | None = None
    price_is_fallback: bool = False
    critical_ratio: float

    accuracy: SkuAccuracy
    series: dict = Field(
        description="history, forecast, band and order -- everything the chart draws"
    )


class ForecastStatusResponse(BaseModel):
    """``GET /forecast/{job_id}``.

    One endpoint for all four states. ``results`` is populated only when ``status`` is
    ``done``; the rest of the envelope is present throughout so a poller can render
    progress without special-casing.
    """

    job_id: uuid.UUID
    dataset_id: uuid.UUID
    status: Literal["queued", "running", "done", "failed"]
    horizon: int
    critical_ratio: float
    enqueued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None

    results: list[SkuResult] = Field(default_factory=list)
    excluded: list[SkuVerdictOut] = Field(
        default_factory=list, description="SKUs the gate refused, carried through with reasons"
    )
    error: str | None = Field(
        default=None,
        description="Set when status is failed, or when some SKUs failed inside a done job",
    )
    message: str = Field(description="One line suitable for showing while polling")


class DeleteResponse(BaseModel):
    """``DELETE /datasets/{dataset_id}`` — the upload and everything derived from it, gone."""

    dataset_id: uuid.UUID
    deleted: bool
    jobs_cancelled: int = Field(
        default=0, description="Queued or running forecasts stopped because their data went"
    )
    message: str


class ServiceHealth(BaseModel):
    """``GET /health`` for tier 2.

    Reports each dependency separately rather than collapsing to one boolean: a queue that
    is down and a database that is down need different responses from whoever is paged.
    """

    status: Literal["ok", "degraded"]
    database: bool
    queue: bool
    storage_writable: bool
    version: str
