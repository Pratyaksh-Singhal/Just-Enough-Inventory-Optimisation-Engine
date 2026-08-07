"""E5-S1 — rolling-origin fold definitions, shared by every model in the project.

Defined once, here, and imported by baselines, the GBM and the scorer alike. If each model
derived its own folds, "model A beats model B" would silently become "model A was
evaluated on easier weeks", which is the kind of bug that never surfaces as an error.

The scheme
----------
Five folds, 28-day horizon, origins stepping backwards from the end of the panel::

    ... training ................ | origin | <- 28-day test window ->
                        fold 0:   2016-01-03   2016-01-04 .. 2016-01-31
                        fold 1:   2016-01-31   2016-02-01 .. 2016-02-28
                        fold 2:   2016-02-28   2016-02-29 .. 2016-03-27
                        fold 3:   2016-03-27   2016-03-28 .. 2016-04-24
                        fold 4:   2016-04-24   2016-04-25 .. 2016-05-22

Test windows are contiguous and non-overlapping, and together they cover exactly the last
:data:`~inventory_engine.config.BACKTEST_DAYS` days -- the same region Phase 1's item
sampler was forbidden from seeing. Training is expanding, not sliding: fold 4 trains on
everything before its origin, including the weeks fold 0 tested on. That is the honest
production analogue, where you would refit on all history available at the time.

Never a random split. Never a single holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from inventory_engine.config import BACKTEST_DAYS, HORIZON, N_FOLDS
from inventory_engine.data.schema import FACT_SALES


@dataclass(frozen=True)
class Fold:
    """One rolling-origin evaluation window.

    Attributes:
        index: Fold number, 0-based. Fold 0 is the earliest test window.
        origin_date: Last date of training data. A model for this fold may use
            observations up to and including this date, and nothing after it.
        test_start: First date being forecast, always ``origin_date + 1``.
        test_end: Last date being forecast, ``horizon`` days after the origin.
        horizon: Length of the test window in days.

    """

    index: int
    origin_date: date
    test_start: date
    test_end: date
    horizon: int

    def target_dates(self) -> list[date]:
        """Every date in this fold's test window, in order."""
        return [self.test_start + timedelta(days=i) for i in range(self.horizon)]

    def horizon_of(self, target: date) -> int:
        """Days ahead ``target`` sits from this fold's origin (1-based).

        Raises:
            ValueError: If ``target`` is outside the test window.

        """
        h = (target - self.origin_date).days
        if not 1 <= h <= self.horizon:
            raise ValueError(
                f"{target} is not in fold {self.index} ({self.test_start} .. {self.test_end})"
            )
        return h


def panel_bounds(con: duckdb.DuckDBPyConnection, table: str = FACT_SALES) -> tuple[date, date]:
    """Return ``(min_date, max_date)`` of the panel."""
    lo, hi = con.execute(f"SELECT min(date), max(date) FROM {table}").fetchone()
    if lo is None:
        raise ValueError(f"{table} is empty; build the warehouse first.")
    return lo, hi


def make_folds(
    last_date: date,
    n_folds: int = N_FOLDS,
    horizon: int = HORIZON,
    *,
    first_date: date | None = None,
) -> tuple[Fold, ...]:
    """Build the rolling-origin folds ending at ``last_date``.

    Args:
        last_date: Final date of the panel. The last fold's test window ends here, so no
            evaluation day is wasted.
        n_folds: Number of folds.
        horizon: Forecast horizon in days; also the length of each test window.
        first_date: Optional panel start, used only to check there is training data left
            in front of fold 0.

    Returns:
        Folds ordered earliest test window first.

    Raises:
        ValueError: If ``n_folds`` or ``horizon`` is below 1, or if the panel is too short
            to leave any training data before the first origin.

    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    folds = []
    for i in range(n_folds):
        # Fold i's test window ends (n_folds - 1 - i) horizons before the panel end.
        test_end = last_date - timedelta(days=(n_folds - 1 - i) * horizon)
        test_start = test_end - timedelta(days=horizon - 1)
        folds.append(
            Fold(
                index=i,
                origin_date=test_start - timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
            )
        )

    if first_date is not None and folds[0].origin_date <= first_date:
        raise ValueError(
            f"panel starts {first_date} but fold 0 trains only up to"
            f" {folds[0].origin_date}: {n_folds} folds x {horizon} days leaves no"
            " training data. Shorten the horizon, drop a fold, or load more history."
        )
    return tuple(folds)


def describe_folds(folds: tuple[Fold, ...]) -> str:
    """Render the fold layout as a human-readable table."""
    lines = [f"  {'fold':>4}  {'train through':<14} {'test window':<25} days"]
    for f in folds:
        lines.append(
            f"  {f.index:>4}  {f.origin_date!s:<14} {f.test_start} .. {f.test_end}  {f.horizon}"
        )
    total = sum(f.horizon for f in folds)
    lines.append(f"  {'':>4}  {'':14} {'total evaluated':<25} {total}")
    return "\n".join(lines)


def assert_no_training_leak(folds: tuple[Fold, ...]) -> None:
    """Fail if any fold's training window reaches into its own test window.

    Trivially true by construction here, but asserted because the cost of it silently
    becoming false -- after someone "simplifies" the arithmetic -- is every metric in the
    project quietly becoming optimistic.

    Raises:
        ValueError: If a fold's origin is not strictly before its test window.

    """
    for f in folds:
        if f.origin_date >= f.test_start:
            raise ValueError(
                f"fold {f.index} trains through {f.origin_date} but tests from"
                f" {f.test_start}: training overlaps evaluation."
            )
    if BACKTEST_DAYS != N_FOLDS * HORIZON:
        raise ValueError(
            f"BACKTEST_DAYS ({BACKTEST_DAYS}) must equal N_FOLDS x HORIZON"
            f" ({N_FOLDS} x {HORIZON}); Phase 1's sampler reserved the wrong region."
        )
