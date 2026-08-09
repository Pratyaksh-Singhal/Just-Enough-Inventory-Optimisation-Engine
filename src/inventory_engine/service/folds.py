"""How many rolling-origin folds a user's own history can actually support.

Tier 1 has a fixed answer -- five folds, 28-day horizon, over an 1,941-day M5 panel (see
:mod:`inventory_engine.backtest.folds`). Tier 2 cannot: the gate admits a SKU at 90 days,
and five 28-day folds need 140 test days plus training data in front of them. Asking
``make_folds`` for five folds there would raise, or worse, quietly produce folds whose
training windows are a fortnight long.

So the fold count is derived from what the user actually has, and the derivation is
reported alongside the result. Two folds is a defensible backtest with a visible caveat;
five folds fabricated from 90 days is not a backtest at all.
"""

from __future__ import annotations

from typing import Final

#: Shortest training window that can produce the features and the metric denominator the
#: rest of the pipeline needs: a 28-day lag, and a naive scale from
#: :func:`inventory_engine.backtest.metrics.naive_scale` measured over more than a handful
#: of differences.
MIN_TRAIN_DAYS: Final = 28

#: Upper bound, matching tier 1's :data:`inventory_engine.config.N_FOLDS`. More folds on a
#: short series means shorter training windows, not better evidence.
MAX_FOLDS: Final = 5


def fold_count_for(n_days: int, horizon: int) -> int:
    """Return how many non-overlapping rolling-origin folds ``n_days`` supports.

    Args:
        n_days: Calendar span of the series, as counted by the gate.
        horizon: Forecast horizon in days; also each fold's test-window length.

    Returns:
        Fold count in ``0..MAX_FOLDS``. Zero means the series cannot be backtested at this
        horizon at all -- the caller should refuse or shorten the horizon rather than
        report an unvalidated forecast.

    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    testable = n_days - MIN_TRAIN_DAYS
    if testable < horizon:
        return 0
    return min(MAX_FOLDS, testable // horizon)


def spread_caveat(n_folds: int) -> str | None:
    """Plain-language warning about how much the fold spread can be trusted, if any.

    Returns ``None`` when there are enough folds for the spread to mean something.
    """
    if n_folds >= 3:
        return None
    if n_folds == 2:
        return (
            "Only 2 backtest folds fit in this history, so the range shown is the gap "
            "between two numbers, not a distribution."
        )
    return (
        "Only 1 backtest fold fits in this history, so there is no spread to report -- "
        "the accuracy figure is a single measurement."
    )
