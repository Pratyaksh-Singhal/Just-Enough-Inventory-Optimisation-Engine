"""Putting work on the queue. Importable by request handlers; knows nothing about models.

The task is referred to by **name**, not by importing the function. That is the mechanism
that makes "the API enqueues, the worker trains" enforceable rather than aspirational: if
this module imported ``worker.run_forecast_job`` to enqueue it, it would transitively
import the pipeline and LightGBM into the API process, and the layering test would fail --
correctly, because at that point one bad refactor is all that stands between a request and
a 13-minute model fit.
"""

from __future__ import annotations

import asyncio
from typing import Final

from arq import create_pool
from arq.connections import RedisSettings

#: The worker registers a coroutine under this name. Kept in one place so the producer and
#: the consumer cannot disagree about it, and asserted by ``tests/test_service_worker``.
FORECAST_TASK: Final = "run_forecast_job"


async def enqueue_forecast_async(redis_url: str, job_id: str, request_id: str) -> str | None:
    """Push a forecast job onto the queue.

    Args:
        redis_url: Queue location.
        job_id: The ``forecast_jobs`` row this task should process. Only the id travels:
            the job's parameters already live in Postgres, and duplicating them into the
            queue payload would create a second source of truth that could disagree with
            the first.
        request_id: Carried through so the worker's log lines join the API's.

    Returns:
        arq's own job id, or ``None`` if the task was already queued.

    """
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    try:
        job = await pool.enqueue_job(
            FORECAST_TASK, job_id, request_id, _job_id=f"forecast:{job_id}"
        )
    finally:
        await pool.close()
    return job.job_id if job else None


def enqueue_forecast(redis_url: str, job_id: str, request_id: str) -> str | None:
    """Enqueue from synchronous code, for calling from a sync FastAPI handler.

    Opens a connection, enqueues, and closes it. That is a connection per enqueue rather
    than a shared pool, which would matter if this ran per page view -- it runs once per
    forecast run, against work that then takes minutes, so the round trip is not worth an
    application-lifetime pool and the shutdown hook to close it.

    FastAPI runs sync handlers in a threadpool, where no event loop is running, so
    ``asyncio.run`` is safe here.
    """
    return asyncio.run(enqueue_forecast_async(redis_url, job_id, request_id))
