"""The worker. The only process in tier 2 that fits a model.

Run it with ``arq inventory_engine.service.worker.WorkerSettings``.

Concurrency
-----------
arq is asyncio and model fitting is CPU-bound, so fitting is pushed into a thread executor
rather than run on the event loop -- otherwise one long fit would stall the worker's
heartbeat and arq would consider the job lost while it was in fact progressing.

**One fit at a time.** :data:`FIT_THREADS` is 1, and that is a correctness constraint, not
a tuning choice. The first version used two threads on the reasoning that LightGBM releases
the GIL during training, so threads would give real parallelism for free. The GIL part is
true and the conclusion was still wrong: LightGBM's training loop is OpenMP-parallel, and
concurrent ``train()`` calls from several Python threads in one process can deadlock inside
the OMP runtime.

It did. Two runs of the demo upload finished in 12s; the third wedged with the process
sitting at a flat 91 seconds of CPU and no further progress -- the job stuck in ``running``
forever with nothing in the log after "forecast job starting". A race, so it passed twice
before it failed.

Parallelism is not lost, only moved: LightGBM's own ``num_threads`` still uses several
cores inside each fit, and arq's ``max_jobs`` still runs separate jobs concurrently. What
is forbidden is two ``train()`` calls in flight in one interpreter.

Failure handling
----------------
A SKU that fails does not fail the job. Its error is recorded against that SKU and the
remaining SKUs still produce results, because a single pathological series should not cost
the user their whole run. A failure of the job *itself* -- storage gone, database
unreachable -- moves the row to ``FAILED`` with a message, and the traceback goes to
Sentry rather than into a database column.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pandas as pd
from arq import cron, func
from arq.connections import RedisSettings
from sqlalchemy import select

from inventory_engine.optimize.costs import CostModel
from inventory_engine.service.db.models import (
    Dataset,
    DatasetSku,
    ForecastJob,
    ForecastResult,
    JobStatus,
)
from inventory_engine.service.db.session import get_sessionmaker
from inventory_engine.service.gate import evaluate
from inventory_engine.service.jobs import FORECAST_TASK
from inventory_engine.service.observability import (
    EVENT_FORECAST_COMPLETED,
    EVENT_FORECAST_FAILED,
    analytics,
    bind_request_id,
    configure_logging,
    init_sentry,
)
from inventory_engine.service.pipeline import SkuForecast, forecast_sku
from inventory_engine.service.retention import purge_expired, sweep_orphans
from inventory_engine.service.settings import get_settings
from inventory_engine.service.storage import LocalDiskStorage

log = logging.getLogger("inventory_engine.worker")

#: Fitting threads. **Must stay 1** -- see the module docstring. Raising it reintroduces
#: concurrent LightGBM ``train()`` calls in one process, which deadlock in OpenMP.
#: ``tests/test_service_worker`` asserts the value so the reason survives the next person
#: who reads "1" as a performance oversight.
FIT_THREADS = 1


async def run_forecast_job(ctx: dict, job_id: str, request_id: str) -> dict:
    """Fit every admitted SKU in a dataset and record the results.

    Args:
        ctx: arq context.
        job_id: ``forecast_jobs.id`` to process.
        request_id: The API request that caused this, for log correlation.

    Returns:
        A small summary dict, which arq stores as the task result.

    """
    settings = get_settings()
    factory = get_sessionmaker()
    started = datetime.now(UTC)
    # Re-bind the API's request id in this process, so every line the worker writes for
    # this job joins the API's lines for the request that caused it. This is the whole
    # point of carrying the id through the queue rather than generating a new one here.
    bind_request_id(request_id)
    log.info("forecast job starting", extra={"job_id": job_id})

    with factory() as session:
        job = session.get(ForecastJob, uuid.UUID(job_id))
        if job is None:
            log.error("no such job", extra={"job_id": job_id, "request_id": request_id})
            return {"job_id": job_id, "status": "missing"}
        job.status = JobStatus.RUNNING
        job.started_at = started
        session.commit()

        dataset = session.get(Dataset, job.dataset_id)
        admitted = list(
            session.scalars(
                select(DatasetSku.sku)
                .where(DatasetSku.dataset_id == job.dataset_id, DatasetSku.admitted.is_(True))
                .order_by(DatasetSku.sku)
            )
        )
        horizon = job.horizon
        costs = CostModel(
            margin_rate=job.margin_rate,
            spoilage_rate=job.spoilage_rate,
            holding_rate=job.holding_rate,
        )
        storage_uri = dataset.storage_uri

    try:
        frames = _load_admitted(settings.upload_dir, storage_uri, admitted)
        results, failures = await _fit_all(frames, horizon, costs)
    except Exception as exc:  # noqa: BLE001 - recorded on the job, re-raised to Sentry
        with factory() as session:
            failed = session.get(ForecastJob, uuid.UUID(job_id))
            failed.status = JobStatus.FAILED
            failed.finished_at = datetime.now(UTC)
            failed.error = f"{type(exc).__name__}: {exc}"
            session.commit()
        analytics().capture(
            EVENT_FORECAST_FAILED,
            job_id=job_id,
            horizon=horizon,
            elapsed_seconds=round((datetime.now(UTC) - started).total_seconds(), 1),
        )
        # Re-raised so Sentry sees the traceback. The database column gets the message
        # only; a stack trace is not a thing to store in a row and read back through an API.
        log.exception("forecast job failed", extra={"job_id": job_id})
        raise

    with factory() as session:
        session.add_all(_to_rows(uuid.UUID(job_id), results))
        done = session.get(ForecastJob, uuid.UUID(job_id))
        done.status = JobStatus.DONE
        done.finished_at = datetime.now(UTC)
        if failures:
            # Partial success is still success, but it is not silent: the job carries the
            # per-SKU failures so the results view can say which products are missing.
            done.error = "; ".join(f"{sku}: {msg}" for sku, msg in failures)
        session.commit()

    elapsed = round((datetime.now(UTC) - started).total_seconds(), 1)
    analytics().capture(
        EVENT_FORECAST_COMPLETED,
        job_id=job_id,
        horizon=horizon,
        sku_count=len(results),
        elapsed_seconds=elapsed,
        # Which method won, aggregated. "baseline beat the model on 3 of 7 products" is a
        # product question worth answering; which product it was is not analytics' business.
        method_used=_dominant_method(results),
    )
    log.info(
        "forecast job done",
        extra={"job_id": job_id, "skus": len(results), "failed": len(failures), "seconds": elapsed},
    )
    return {"job_id": job_id, "skus": len(results), "failed": len(failures)}


def _dominant_method(results: list[SkuForecast]) -> str:
    """Which method won most often in this job, as one enum value."""
    if not results:
        return "none"
    baseline = sum(1 for r in results if r.method_used == "seasonal_naive")
    if baseline == len(results):
        return "all_baseline"
    return "all_model" if baseline == 0 else "mixed"


def _load_admitted(upload_dir, storage_uri: str, admitted: list[str]) -> dict[str, pd.DataFrame]:
    """Read the stored CSV and split it into the admitted SKUs' frames.

    The gate is re-run rather than trusted from the row, but only for its *cleaning*: the
    duplicate merge, negative clip and date parse have to be applied identically to what
    the verdict was computed on, or the model would see rows the gate never scored. Which
    SKUs to keep still comes from the database, so the admission decision itself is made
    exactly once.
    """
    storage = LocalDiskStorage(upload_dir)
    with storage.open(storage_uri) as handle:
        raw = pd.read_csv(handle)

    report = evaluate(raw)
    mapping = report.column_mapping
    frame = pd.DataFrame(
        {
            "sku": raw[mapping["sku"]].astype("string").str.strip(),
            "date": pd.to_datetime(raw[mapping["date"]], errors="coerce", format="mixed"),
            "units_sold": pd.to_numeric(raw[mapping["units_sold"]], errors="coerce").clip(lower=0),
        }
    ).dropna(subset=["date"])
    if "unit_price" in mapping:
        frame["unit_price"] = pd.to_numeric(raw[mapping["unit_price"]], errors="coerce")

    wanted = set(admitted)
    return {
        str(sku): group.sort_values("date").reset_index(drop=True)
        for sku, group in frame.groupby("sku")
        if str(sku) in wanted
    }


async def _fit_all(
    frames: dict[str, pd.DataFrame], horizon: int, costs: CostModel
) -> tuple[list[SkuForecast], list[tuple[str, str]]]:
    """Fit every SKU off the event loop, collecting failures rather than raising them."""
    loop = asyncio.get_running_loop()
    results: list[SkuForecast] = []
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=FIT_THREADS) as pool:
        tasks = {
            sku: loop.run_in_executor(pool, forecast_sku, sku, frame, horizon, costs)
            for sku, frame in frames.items()
        }
        for sku, task in tasks.items():
            try:
                results.append(await task)
            except Exception as exc:  # noqa: BLE001 - one bad series must not fail the run
                log.warning("sku failed", extra={"sku": sku, "error": str(exc)})
                failures.append((sku, f"{type(exc).__name__}: {exc}"))
    return results, failures


def _to_rows(job_id: uuid.UUID, results: list[SkuForecast]) -> list[ForecastResult]:
    """Map pipeline output onto ``forecast_results`` rows."""
    return [
        ForecastResult(
            job_id=job_id,
            sku=r.sku,
            method_used=r.method_used,
            method_reason=r.method_reason,
            n_folds=r.n_folds,
            mase_model=_finite(r.model.mase_mean),
            mase_model_spread=_finite(r.model.mase_spread),
            mase_baseline=_finite(r.baseline.mase_mean),
            mase_baseline_spread=_finite(r.baseline.mase_spread),
            pinball_model=_finite(r.model.pinball_mean),
            pinball_baseline=_finite(r.baseline.pinball_mean),
            critical_ratio=r.critical_ratio,
            order_qty=r.order_qty,
            expected_cost=_finite(r.expected_cost),
            unit_price=r.unit_price,
            price_is_fallback=r.price_is_fallback,
            series=r.series,
        )
        for r in results
    ]


def _finite(value: float | None) -> float | None:
    """Map NaN to NULL. A NaN in a Float column is neither queryable nor JSON-serialisable."""
    if value is None:
        return None
    return float(value) if pd.notna(value) else None


async def purge_uploads(ctx: dict) -> dict:
    """Destroy uploads past the retention window, plus any orphaned files.

    Runs on the worker rather than as a separate cron container: it needs the same
    database, the same uploads volume and the same settings, and a second deployable to
    delete some rows is a second thing to forget to deploy.
    """
    settings = get_settings()
    bind_request_id(f"purge-{datetime.now(UTC):%Y%m%d}")
    storage = LocalDiskStorage(settings.upload_dir)

    with get_sessionmaker()() as session:
        result = purge_expired(session, storage, settings.upload_retention_days)
        orphans = sweep_orphans(session, storage)
        session.commit()

    log.info(
        "retention run complete",
        extra={
            "retention_days": settings.upload_retention_days,
            "datasets_deleted": result.datasets_deleted,
            "orphans_deleted": orphans,
            "jobs_cancelled": result.jobs_cancelled,
        },
    )
    return {"datasets_deleted": result.datasets_deleted, "orphans_deleted": orphans}


class WorkerSettings:
    """arq entry point. ``arq inventory_engine.service.worker.WorkerSettings``."""

    #: Registered under an explicit name rather than letting arq infer ``__name__``, so
    #: that renaming the coroutine cannot silently orphan every job the API enqueues under
    #: the old name. ``tests/test_service_worker`` asserts producer and consumer agree.
    functions = [func(run_forecast_job, name=FORECAST_TASK)]
    #: Retention runs at 03:15 daily. Not on a timer from process start: a worker that
    #: restarts often would either never reach the interval or run the purge on every boot.
    cron_jobs = [cron(purge_uploads, hour=3, minute=15, run_at_startup=False)]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2
    job_timeout = get_settings().job_timeout_seconds
    keep_result = 3600

    @staticmethod
    async def on_startup(ctx: dict) -> None:
        """Install JSON logging and Sentry for the worker process.

        Separate from the API's initialisation and tagged separately: "the worker is
        erroring" and "the API is erroring" are different pages and different fixes.
        """
        configure_logging("worker")
        init_sentry("worker")
        log.info("worker started")
