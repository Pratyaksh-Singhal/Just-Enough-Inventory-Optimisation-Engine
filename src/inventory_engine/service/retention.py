"""Deleting users' uploaded data — on request, and on a clock."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from inventory_engine.service.db.models import Dataset, ForecastJob, JobStatus
from inventory_engine.service.storage import BlobStorage

log = logging.getLogger("inventory_engine.service.retention")


@dataclass(frozen=True)
class PurgeResult:
    """What a retention run destroyed."""

    datasets_deleted: int = 0
    blobs_deleted: int = 0
    jobs_cancelled: int = 0
    orphans_deleted: int = 0
    cutoff: datetime | None = None

    def render(self) -> str:
        """One-line summary for a log."""
        return (
            f"purged {self.datasets_deleted} dataset(s), {self.blobs_deleted} file(s), "
            f"{self.orphans_deleted} orphan(s), cancelled {self.jobs_cancelled} job(s)"
        )


#: Jobs in these states still expect their dataset to exist.
LIVE_STATES = (JobStatus.QUEUED, JobStatus.RUNNING)

CANCELLED_REASON = "cancelled: the uploaded data was deleted"


def delete_dataset(session: Session, storage: BlobStorage, dataset: Dataset) -> PurgeResult:
    """Delete one dataset, its stored file, and everything derived from it."""
    cancelled = 0
    for job in session.scalars(
        select(ForecastJob).where(
            ForecastJob.dataset_id == dataset.id, ForecastJob.status.in_(LIVE_STATES)
        )
    ).all():
        job.status = JobStatus.FAILED
        job.finished_at = datetime.now(UTC)
        job.error = CANCELLED_REASON
        cancelled += 1

    blobs = 0
    try:
        storage.delete(dataset.storage_uri)
        blobs = 1
    except Exception:  # noqa: BLE001 - a missing file must not block removing the row
        log.warning("could not delete blob; removing the row anyway", exc_info=True)

    session.delete(dataset)
    return PurgeResult(datasets_deleted=1, blobs_deleted=blobs, jobs_cancelled=cancelled)


def purge_expired(
    session: Session, storage: BlobStorage, retention_days: int, *, now: datetime | None = None
) -> PurgeResult:
    """Delete every dataset older than ``retention_days``."""
    if retention_days <= 0:
        raise ValueError(
            f"retention_days must be positive, got {retention_days}; "
            "a zero or negative window would delete every upload on the next run"
        )

    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    expired = session.scalars(select(Dataset).where(Dataset.created_at < cutoff)).all()

    total = PurgeResult(cutoff=cutoff)
    for dataset in expired:
        one = delete_dataset(session, storage, dataset)
        total = PurgeResult(
            datasets_deleted=total.datasets_deleted + one.datasets_deleted,
            blobs_deleted=total.blobs_deleted + one.blobs_deleted,
            jobs_cancelled=total.jobs_cancelled + one.jobs_cancelled,
            cutoff=cutoff,
        )
    return total


def sweep_orphans(session: Session, storage: BlobStorage) -> int:
    """Delete stored files that no dataset row references."""
    root = getattr(storage, "root", None)
    if root is None:
        return 0

    referenced = {
        uri
        for (uri,) in session.execute(select(Dataset.storage_uri)).all()  # noqa: C416
    }
    removed = 0
    for path in root.glob("*.csv"):
        if path.as_uri() not in referenced:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
