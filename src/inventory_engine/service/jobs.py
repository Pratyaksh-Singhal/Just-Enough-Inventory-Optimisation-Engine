"""Putting work on the queue."""

from __future__ import annotations

import asyncio
from typing import Final

from arq import create_pool
from arq.connections import RedisSettings

#: The worker registers a coroutine under this name. Kept in one place so the producer and
#: the consumer cannot disagree about it, and asserted by ``tests/test_service_worker``.
FORECAST_TASK: Final = "run_forecast_job"


async def enqueue_forecast_async(redis_url: str, job_id: str, request_id: str) -> str | None:
    """Push a forecast job onto the queue."""
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    try:
        job = await pool.enqueue_job(
            FORECAST_TASK, job_id, request_id, _job_id=f"forecast:{job_id}"
        )
    finally:
        await pool.close()
    return job.job_id if job else None


def enqueue_forecast(redis_url: str, job_id: str, request_id: str) -> str | None:
    """Enqueue from synchronous code, for calling from a sync FastAPI handler."""
    return asyncio.run(enqueue_forecast_async(redis_url, job_id, request_id))
