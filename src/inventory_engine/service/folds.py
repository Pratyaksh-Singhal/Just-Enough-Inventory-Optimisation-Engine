"""How many rolling-origin folds a user's own history can actually support."""

from __future__ import annotations

from typing import Final

#: Shortest training window that can produce the features and the metric denominator the rest of
#: the pipeline needs: a 28-day lag, and a naive scale from
MIN_TRAIN_DAYS: Final = 28

#: Upper bound, matching tier 1's :data:`inventory_engine.config.N_FOLDS`. More folds on a
#: short series means shorter training windows, not better evidence.
MAX_FOLDS: Final = 5


def fold_count_for(n_days: int, horizon: int) -> int:
    """Return how many non-overlapping rolling-origin folds ``n_days`` supports."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    testable = n_days - MIN_TRAIN_DAYS
    if testable < horizon:
        return 0
    return min(MAX_FOLDS, testable // horizon)


def spread_caveat(n_folds: int) -> str | None:
    """Plain-language warning about how much the fold spread can be trusted, if any."""
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
