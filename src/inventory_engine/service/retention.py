"""Deleting users' uploaded data — on request, and on a clock.

Why this exists
---------------
Everything else in tier 2 is careful about a user's sales history: the analytics allowlist
cannot send it, Sentry's hook strips the channels that would carry it, the upload directory
is gitignored. All of that is undermined by keeping every CSV ever uploaded, forever, with
no way to remove one. Retention is the part of the privacy story that is about *time*
rather than about *destinations*.

Two paths, and they share one implementation:

* :func:`delete_dataset` — the user asks. Immediate.
* :func:`purge_expired` — the clock asks. Anything past
  :attr:`~inventory_engine.service.settings.ServiceSettings.upload_retention_days`.

Order of operations
-------------------
**The bytes go first, then the row.** The two orderings fail differently and only one of
them fails safely:

* row first, then a crash -> the file survives with nothing pointing at it. Sensitive data
  lingers, and nothing knows it is there. This is the bad one.
* bytes first, then a crash -> a row referencing a file that is gone. The data is
  destroyed, which was the point, and :func:`sweep_orphans` and a plain 404 clean up after.

So the sequence is deliberate, not incidental.

Cascades do the rest: ``forecast_jobs`` and ``forecast_results`` are ``ON DELETE CASCADE``
from ``datasets``, and ``forecast_results.series`` holds the user's own daily sales values,
so the derived copy dies with the original rather than outliving it.
"""

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
    """What a retention run destroyed.

    Attributes:
        datasets_deleted: Rows removed, each with its stored file.
        blobs_deleted: Files removed. Should equal ``datasets_deleted`` unless a previous
            run was interrupted between the two steps.
        jobs_cancelled: Queued or running jobs marked failed because their dataset went.
        orphans_deleted: Files on disk with no row pointing at them.
        cutoff: Datasets created before this moment were in scope.

    """

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


#: Jobs in these states still expect their dataset to exist. Deleting underneath them
#: would leave the worker failing on a missing file with a confusing error, so they are
#: failed explicitly with a reason the results view can show.
LIVE_STATES = (JobStatus.QUEUED, JobStatus.RUNNING)

CANCELLED_REASON = "cancelled: the uploaded data was deleted"


def delete_dataset(session: Session, storage: BlobStorage, dataset: Dataset) -> PurgeResult:
    """Delete one dataset, its stored file, and everything derived from it.

    A queued or running job against this dataset is failed with a stated reason rather than
    left to die on a missing file. Honouring "delete my data" beats protecting a job that
    is, by then, computing on data the user has withdrawn.
    """
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
    """Delete every dataset older than ``retention_days``.

    Args:
        session: Open session. The caller commits.
        storage: Where the blobs live.
        retention_days: Age in days past which an upload is destroyed. Zero or negative is
            rejected rather than silently interpreted as "delete everything".
        now: Overrides the clock, for tests.

    Returns:
        A :class:`PurgeResult`.

    Raises:
        ValueError: If ``retention_days`` is not positive.

    """
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
    """Delete stored files that no dataset row references.

    These come from the gap between writing bytes and committing a row: an upload refused
    by the gate deletes its own blob, but a process killed between the two leaves a file
    behind. Without this they are invisible and permanent -- precisely the failure mode the
    "bytes first" ordering accepts in exchange for never orphaning data the other way.

    Only implemented for local disk, which is the only backend that can enumerate itself
    cheaply. An S3 backend would do this with a lifecycle rule instead.
    """
    root = getattr(storage, "root", None)
    if root is None:
        return 0

    referenced = {
        uri for (uri,) in session.execute(select(Dataset.storage_uri)).all()  # noqa: C416
    }
    removed = 0
    for path in root.glob("*.csv"):
        if path.as_uri() not in referenced:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
