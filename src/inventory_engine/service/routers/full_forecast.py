"""Tier 2 endpoints: upload, health.

``/forecast/run`` and ``/forecast/{job_id}`` land in steps 2 and 3.

This module reads and writes rows and pushes work onto a queue. It does not fit anything,
and it must not import anything that can -- the AST scan in ``tests/test_service_layering``
fails the build if it does.
"""

from __future__ import annotations

import logging
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from inventory_engine.optimize.costs import CostModel
from inventory_engine.service.db.models import (
    Dataset,
    DatasetSku,
    ForecastJob,
    ForecastResult,
    JobStatus,
)
from inventory_engine.service.db.session import get_session
from inventory_engine.service.gate import GateReport, evaluate, refusal_message
from inventory_engine.service.jobs import enqueue_forecast
from inventory_engine.service.observability import (
    EVENT_DATASET_CREATED,
    EVENT_FORECAST_ENQUEUED,
    EVENT_UPLOAD_RECEIVED,
    EVENT_UPLOAD_REJECTED,
    analytics,
)
from inventory_engine.service.retention import delete_dataset
from inventory_engine.service.schemas import (
    DeleteResponse,
    FestivalNote,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastStatusResponse,
    ServiceHealth,
    SkuAccuracy,
    SkuResult,
    SkuVerdictOut,
    UploadRefusal,
    UploadResponse,
)
from inventory_engine.service.settings import ServiceSettings, get_settings
from inventory_engine.service.storage import LocalDiskStorage, UploadTooLarge

VERSION = "2.0.0"

log = logging.getLogger("inventory_engine.service.api")

router = APIRouter(tags=["full-forecast"])


def get_storage(settings: ServiceSettings = Depends(get_settings)) -> LocalDiskStorage:
    """FastAPI dependency: the configured blob store."""
    return LocalDiskStorage(settings.upload_dir)


@router.get("/health", response_model=ServiceHealth)
def health(
    session: Session = Depends(get_session),
    storage: LocalDiskStorage = Depends(get_storage),
    settings: ServiceSettings = Depends(get_settings),
) -> ServiceHealth:
    """Liveness and readiness for the service tier.

    Each dependency is probed and reported separately. A cheap ``SELECT 1`` and a Redis
    ``PING`` are enough to distinguish "cannot reach the database" from "cannot reach the
    queue", which is the distinction that matters at 3am.
    """
    database = _probe(lambda: session.execute(text("SELECT 1")))
    queue = _probe(lambda: _ping_redis(settings.redis_url))
    writable = _probe(lambda: storage.root.mkdir(parents=True, exist_ok=True))
    return ServiceHealth(
        status="ok" if (database and queue and writable) else "degraded",
        database=database,
        queue=queue,
        storage_writable=writable,
        version=VERSION,
    )


def _probe(fn) -> bool:
    """Run a health check, converting any failure into ``False``.

    A health endpoint that raises is a health endpoint that cannot report which dependency
    is down, which is the only thing it is for.
    """
    try:
        fn()
    except Exception:  # noqa: BLE001 - the whole point is to report rather than propagate
        return False
    return True


def _ping_redis(url: str) -> None:
    """Ping Redis. Imported locally so the module has no import-time queue dependency."""
    from redis import Redis

    client = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
    finally:
        client.close()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    responses={422: {"model": UploadRefusal}, 413: {"model": UploadRefusal}},
)
def upload(
    request: Request,
    file: UploadFile = File(..., description="CSV with sku, date, units_sold[, unit_price]"),
    session: Session = Depends(get_session),
    storage: LocalDiskStorage = Depends(get_storage),
    settings: ServiceSettings = Depends(get_settings),
) -> UploadResponse:
    """Validate a CSV against the data gate and store it.

    Order of operations is deliberate: the bytes are streamed to storage **first**, then
    parsed from disk. Parsing the ``UploadFile`` in memory to decide whether to keep it
    would make peak memory a function of upload size for files we are about to reject,
    which is exactly backwards.

    Raises:
        HTTPException: 413 if the file exceeds the configured ceiling; 422 if the gate
            refuses it -- either fatally (missing columns, unreadable dates) or because
            no SKU cleared the thresholds. The 422 body carries the per-SKU shortfall.

    """
    try:
        blob = storage.put(file.file, limit_bytes=settings.max_upload_bytes)
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        with storage.open(blob.uri) as handle:
            frame = pd.read_csv(handle)
    except UnicodeDecodeError as exc:
        storage.delete(blob.uri)
        raise HTTPException(
            status_code=422,
            detail="The file is not readable as UTF-8 text. Export it as CSV, not XLSX.",
        ) from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        storage.delete(blob.uri)
        raise HTTPException(status_code=422, detail=f"Could not read the CSV: {exc}") from exc

    client_id = getattr(request.state, "client_id", "anonymous")
    # Counts only. Never the filename, never a SKU -- the allowlist in observability.py
    # enforces that rather than trusting this call site to remember it.
    analytics().capture(
        EVENT_UPLOAD_RECEIVED,
        client_id,
        rows_read=len(frame),
        byte_size=blob.byte_size,
    )

    report = evaluate(frame)
    if not report.usable:
        # A file nothing can be forecast from is not kept: it would be a row pointing at
        # a user's sales history that this service has already said it will not use.
        storage.delete(blob.uri)
        analytics().capture(
            EVENT_UPLOAD_REJECTED,
            client_id,
            rows_read=report.rows_read,
            sku_count_rejected=len(report.rejected),
            fatal=report.fatal is not None,
            rejection_reason="missing_columns" if report.fatal else "insufficient_history",
        )
        log.info(
            "upload refused by the data gate",
            extra={"rows_read": report.rows_read, "rejected": len(report.rejected)},
        )
        raise HTTPException(
            status_code=422,
            detail=UploadRefusal.of(report, refusal_message(report)).model_dump(mode="json"),
        )

    dataset = _persist(session, report, blob, file.filename)
    session.flush()

    analytics().capture(
        EVENT_DATASET_CREATED,
        client_id,
        dataset_id=str(dataset.id),
        rows_read=report.rows_read,
        sku_count_admitted=len(report.admitted),
        sku_count_rejected=len(report.rejected),
    )
    log.info(
        "dataset created",
        extra={
            "dataset_id": str(dataset.id),
            "admitted": len(report.admitted),
            "rejected": len(report.rejected),
        },
    )

    return UploadResponse(
        dataset_id=dataset.id,
        rows_read=report.rows_read,
        admitted=[SkuVerdictOut.of(v) for v in report.admitted],
        excluded=[SkuVerdictOut.of(v) for v in report.rejected],
        column_mapping=report.column_mapping,
        warnings=list(report.warnings()),
        message=report.summary(),
    )


def _persist(session: Session, report: GateReport, blob, filename: str | None) -> Dataset:
    """Write the dataset and one row per SKU, admitted or not.

    Rejected SKUs are stored too. The upload response carries them once; the row is what
    makes "which products did it skip, and why" answerable a week later.
    """
    everything = report.admitted + report.rejected
    dates = [v.first_date for v in everything if v.first_date] or [None]
    last = [v.last_date for v in everything if v.last_date] or [None]

    dataset = Dataset(
        id=uuid.uuid4(),
        original_filename=filename,
        storage_uri=blob.uri,
        sha256=blob.sha256,
        byte_size=blob.byte_size,
        rows_read=report.rows_read,
        sku_count_admitted=len(report.admitted),
        sku_count_rejected=len(report.rejected),
        first_date=min(d for d in dates if d) if any(dates) else None,
        last_date=max(d for d in last if d) if any(last) else None,
        column_mapping=report.column_mapping,
        warnings=list(report.warnings()),
    )
    session.add(dataset)
    session.add_all(
        DatasetSku(
            dataset=dataset,
            sku=v.sku,
            admitted=v.admitted,
            n_days=v.n_days,
            n_obs=v.n_obs,
            n_nonnull=v.n_nonnull,
            max_gap_days=v.max_gap_days,
            first_date=v.first_date,
            last_date=v.last_date,
            reasons=list(v.reasons),
        )
        for v in everything
    )
    return dataset


@router.post("/forecast/run", response_model=ForecastRunResponse, status_code=202)
def run_forecast(
    body: ForecastRunRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: ServiceSettings = Depends(get_settings),
) -> ForecastRunResponse:
    """Enqueue a forecast run and return immediately.

    This handler writes one row and pushes one id. It does not read the CSV, does not build
    a feature matrix, and does not import anything that could fit a model -- the whole
    reason the job exists is that fitting takes minutes and a request must not.

    The queued payload is the job id alone. Every parameter is already on the row, so the
    worker reads its instructions from one place rather than reconciling a queue payload
    against a database row that could have been written by a different version.

    Raises:
        HTTPException: 404 if the dataset is unknown; 409 if it has no admitted SKUs to
            forecast; 503 if the queue cannot be reached, since a job accepted into a
            queue that is down would sit in ``QUEUED`` forever looking merely slow.

    """
    dataset = session.get(Dataset, body.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset {body.dataset_id}")
    if dataset.sku_count_admitted == 0:
        raise HTTPException(
            status_code=409,
            detail="This dataset has no products that cleared the data gate; "
            "there is nothing to forecast.",
        )

    costs = CostModel(
        margin_rate=body.margin_rate,
        spoilage_rate=body.spoilage_rate,
        holding_rate=body.holding_rate,
    )
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    job = ForecastJob(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        status=JobStatus.QUEUED,
        horizon=body.horizon,
        margin_rate=body.margin_rate,
        spoilage_rate=body.spoilage_rate,
        holding_rate=body.holding_rate,
        request_id=request_id,
    )
    session.add(job)
    # Flushed before enqueueing so the row exists by the time a worker can pick the id up.
    # A worker that got there first would look up an id that had not been committed yet and
    # fail a job that was in fact fine.
    session.flush()

    try:
        enqueue_forecast(settings.redis_url, str(job.id), request_id)
    except Exception as exc:  # noqa: BLE001 - reported as unavailable, not as a 500
        session.rollback()
        raise HTTPException(
            status_code=503,
            detail=f"could not reach the job queue, so nothing was started: {exc}",
        ) from exc

    analytics().capture(
        EVENT_FORECAST_ENQUEUED,
        getattr(request.state, "client_id", "anonymous"),
        dataset_id=str(dataset.id),
        job_id=str(job.id),
        horizon=body.horizon,
        critical_ratio=round(costs.critical_ratio(), 4),
        sku_count=dataset.sku_count_admitted,
    )
    log.info(
        "forecast enqueued",
        extra={"job_id": str(job.id), "horizon": body.horizon, "skus": dataset.sku_count_admitted},
    )

    return ForecastRunResponse(
        job_id=job.id,
        status="queued",
        dataset_id=dataset.id,
        horizon=body.horizon,
        critical_ratio=costs.critical_ratio(),
        skus_queued=dataset.sku_count_admitted,
        poll_url=f"/forecast/{job.id}",
    )


#: What to show while a job is in flight. Held here rather than in the client so the API
#: and any other consumer say the same thing about the same state.
STATUS_MESSAGES = {
    JobStatus.QUEUED: "Queued. Waiting for a worker to pick this up.",
    JobStatus.RUNNING: "Fitting and backtesting each product against its own history.",
    JobStatus.FAILED: "The run failed. Nothing was charged to your data.",
}


@router.get("/forecast/{job_id}", response_model=ForecastStatusResponse)
def forecast_status(
    job_id: uuid.UUID, session: Session = Depends(get_session)
) -> ForecastStatusResponse:
    """Job status, and the results once there are any.

    One endpoint for all four states rather than a separate results resource. A client
    polling this never has to guess when to switch URLs, and ``status`` is the single thing
    it branches on.

    The excluded SKUs travel with the results, not only with the upload response. By the
    time someone is looking at order quantities they have usually forgotten which products
    were dropped at the gate, and "why is this product missing from my order list" is
    exactly the question the results view has to be able to answer.

    Raises:
        HTTPException: 404 if no such job.

    """
    job = session.get(ForecastJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no forecast job {job_id}")

    costs = CostModel(
        margin_rate=job.margin_rate,
        spoilage_rate=job.spoilage_rate,
        holding_rate=job.holding_rate,
    )
    rows = session.scalars(
        select(ForecastResult)
        .where(ForecastResult.job_id == job_id)
        # Biggest order first: the products with the most money moving through them are the
        # ones a buyer checks, and scrolling to find them is a tax on every visit.
        .order_by(ForecastResult.order_qty.desc())
    ).all()
    excluded = session.scalars(
        select(DatasetSku)
        .where(DatasetSku.dataset_id == job.dataset_id, DatasetSku.admitted.is_(False))
        .order_by(DatasetSku.sku)
    ).all()

    elapsed = (
        (job.finished_at - job.started_at).total_seconds()
        if job.started_at and job.finished_at
        else None
    )
    return ForecastStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        status=job.status.value,
        horizon=job.horizon,
        critical_ratio=costs.critical_ratio(),
        enqueued_at=job.enqueued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        elapsed_seconds=elapsed,
        results=[_result_of(r, job.horizon) for r in rows],
        excluded=[
            SkuVerdictOut(
                sku=v.sku,
                admitted=False,
                n_days=v.n_days,
                n_obs=v.n_obs,
                n_nonnull=v.n_nonnull,
                max_gap_days=v.max_gap_days,
                first_date=v.first_date,
                last_date=v.last_date,
                reasons=list(v.reasons),
            )
            for v in excluded
        ],
        error=job.error,
        message=_status_message(job, len(rows)),
    )


def _status_message(job: ForecastJob, n_results: int) -> str:
    """One line describing the job's current state.

    A ``done`` job that also carries an error is a partial success -- some SKUs failed while
    the rest produced results -- and says so, rather than showing a bare success next to an
    unexplained error field.
    """
    if job.status is JobStatus.DONE:
        if job.error:
            return (
                f"Done, with problems: {n_results} product(s) forecast, and some could not "
                "be. See 'error' for which."
            )
        return f"Done. {n_results} product(s) forecast."
    return STATUS_MESSAGES[job.status]


def _result_of(row: ForecastResult, horizon: int) -> SkuResult:
    """Map a stored result row onto the response model.

    ``festival`` is left null rather than defaulted to a "none" state when the stored blob
    is empty: rows written before the feature existed have nothing to say about festivals,
    and "we did not look" is not the same claim as "we looked and there was nothing".
    """
    return SkuResult(
        sku=row.sku,
        method_used=row.method_used,
        method_reason=row.method_reason,
        order_qty=row.order_qty,
        order_qty_before_festival=row.order_qty_before_festival,
        festival=FestivalNote(**row.festival) if row.festival else None,
        order_daily_rate=row.order_qty / horizon if horizon else row.order_qty,
        expected_cost=row.expected_cost,
        unit_price=row.unit_price,
        price_is_fallback=row.price_is_fallback,
        critical_ratio=row.critical_ratio,
        accuracy=SkuAccuracy(
            n_folds=row.n_folds,
            mase_model=row.mase_model,
            mase_model_spread=row.mase_model_spread,
            mase_baseline=row.mase_baseline,
            mase_baseline_spread=row.mase_baseline_spread,
            pinball_model=row.pinball_model,
            pinball_baseline=row.pinball_baseline,
        ),
        series=row.series,
    )


@router.delete("/datasets/{dataset_id}", response_model=DeleteResponse)
def delete_dataset_endpoint(
    dataset_id: uuid.UUID,
    session: Session = Depends(get_session),
    storage: LocalDiskStorage = Depends(get_storage),
) -> DeleteResponse:
    """Delete an upload and everything derived from it, now.

    Hard delete, not a flag. A soft delete would leave the CSV on disk and the user's own
    daily sales values in ``forecast_results.series``, which is not what "delete my data"
    means to the person asking.

    Idempotent: deleting an already-deleted dataset is a 404, but the second call has
    nothing left to do either way.

    Raises:
        HTTPException: 404 if no such dataset.

    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id}")

    result = delete_dataset(session, storage, dataset)
    log.info(
        "dataset deleted on request",
        extra={"dataset_id": str(dataset_id), "jobs_cancelled": result.jobs_cancelled},
    )
    return DeleteResponse(
        dataset_id=dataset_id,
        deleted=True,
        jobs_cancelled=result.jobs_cancelled,
        message=(
            "Deleted: the uploaded file, its forecasts and its stored history are gone."
            + (
                f" {result.jobs_cancelled} running or queued forecast(s) were cancelled."
                if result.jobs_cancelled
                else ""
            )
        ),
    )


@router.get("/datasets/{dataset_id}", response_model=UploadResponse)
def get_dataset(dataset_id: uuid.UUID, session: Session = Depends(get_session)) -> UploadResponse:
    """Re-read a stored dataset's gate verdict, without re-uploading the file."""
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset {dataset_id}")

    rows = session.scalars(
        select(DatasetSku).where(DatasetSku.dataset_id == dataset_id).order_by(DatasetSku.sku)
    ).all()
    to_out = [
        SkuVerdictOut(
            sku=r.sku,
            admitted=r.admitted,
            n_days=r.n_days,
            n_obs=r.n_obs,
            n_nonnull=r.n_nonnull,
            max_gap_days=r.max_gap_days,
            first_date=r.first_date,
            last_date=r.last_date,
            reasons=list(r.reasons),
        )
        for r in rows
    ]
    return UploadResponse(
        dataset_id=dataset.id,
        rows_read=dataset.rows_read,
        admitted=[v for v in to_out if v.admitted],
        excluded=[v for v in to_out if not v.admitted],
        column_mapping=dataset.column_mapping,
        warnings=list(dataset.warnings),
        message=(
            f"{dataset.sku_count_admitted} SKU(s) admitted, "
            f"{dataset.sku_count_rejected} excluded from {dataset.rows_read} rows"
        ),
    )
