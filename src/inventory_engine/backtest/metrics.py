"""E5-S2 — forecast accuracy metrics.

Why these four
--------------
**MASE** and **RMSSE** are scale-free: they divide the error by how well a naive forecast
would have done *on that series' own training history*. That matters here because the
panel spans SKUs selling 40 units a day and SKUs selling 40 units a year, and an unscaled
error would simply rank series by volume.

**Bias / ME** is signed, and the sign is the entire point of this project. An over-forecast
becomes dead stock and, for fresh food, spoilage; an under-forecast becomes a stockout and
lost margin. They cost different amounts, so an unsigned error metric throws away the
information the newsvendor layer in E7 is built to act on.

**Pinball loss** scores a quantile rather than a point, which is what E7 actually consumes.
A model can have excellent MASE and a useless 95th percentile.

Why not MAPE
------------
MAPE divides by the actual. 61.6% of this panel is zero-demand days, so the denominator is
zero on most rows and the metric is undefined -- and on the near-zero rows that remain it
explodes to arbitrarily large values, so any average is dominated by the quietest days.
It is not a defensible metric on intermittent demand and is deliberately absent.

The scale denominator
---------------------
Both scaled metrics divide by the mean 1-step naive error over the **training** window,
measured from each series' first non-zero sale (the M5 convention). A series that never
sells during training has no defined scale; those series are excluded and the exclusion is
counted and reported rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def naive_scale(train: np.ndarray, seasonality: int = 1) -> float:
    """Mean absolute ``seasonality``-step naive error over the training window.

    Measured from the first non-zero observation, per the M5 convention: the leading zeros
    of a not-yet-introduced SKU are not demand history, and including them deflates the
    denominator, which inflates the apparent skill of every model scored against it.

    Args:
        train: Training actuals for one series, in date order.
        seasonality: Step size for the naive comparison. 1 is standard for M5.

    Returns:
        The scale, or ``nan`` if the series has no usable variation in training.

    """
    train = np.asarray(train, dtype=float)
    nz = np.nonzero(train)[0]
    if nz.size == 0:
        return float("nan")  # never sold in training: no defined scale
    active = train[nz[0] :]
    if active.size <= seasonality:
        return float("nan")
    diffs = np.abs(active[seasonality:] - active[:-seasonality])
    scale = float(diffs.mean())
    return scale if scale > EPS else float("nan")


def mase(actual: np.ndarray, predicted: np.ndarray, scale: float) -> float:
    """Mean absolute scaled error. Lower is better; 1.0 means "as good as naive"."""
    if not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float))) / scale)


def rmsse(actual: np.ndarray, predicted: np.ndarray, scale_sq: float) -> float:
    """Root mean squared scaled error -- the per-series component of M5's WRMSSE.

    Args:
        actual: Realised units over the test window.
        predicted: Forecast units over the same window.
        scale_sq: Mean squared 1-step naive error over training, from
            :func:`naive_scale_squared`.

    """
    if not np.isfinite(scale_sq):
        return float("nan")
    err = np.asarray(actual, float) - np.asarray(predicted, float)
    return float(np.sqrt(np.mean(err**2) / scale_sq))


def naive_scale_squared(train: np.ndarray) -> float:
    """Mean squared 1-step naive error over training, from the first non-zero sale."""
    train = np.asarray(train, dtype=float)
    nz = np.nonzero(train)[0]
    if nz.size == 0:
        return float("nan")
    active = train[nz[0] :]
    if active.size <= 1:
        return float("nan")
    scale = float(np.mean((active[1:] - active[:-1]) ** 2))
    return scale if scale > EPS else float("nan")


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean error, signed. Positive = over-forecast (dead stock); negative = stockout."""
    return float(np.mean(np.asarray(predicted, float) - np.asarray(actual, float)))


def pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss at level ``quantile``.

    Penalises under-forecasting by ``q`` and over-forecasting by ``1 - q``, so a high
    quantile is punished mostly for being too low -- which is the asymmetry a service-level
    target is meant to encode.

    Raises:
        ValueError: If ``quantile`` is not strictly between 0 and 1.

    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    diff = np.asarray(actual, float) - np.asarray(predicted, float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


@dataclass(frozen=True)
class SeriesScore:
    """Metrics for one series over one fold."""

    mase: float
    rmsse: float
    bias: float

    def is_scorable(self) -> bool:
        """Whether the scaled metrics are defined for this series."""
        return np.isfinite(self.mase) and np.isfinite(self.rmsse)


def score_series(
    train: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonality: int = 1,
) -> SeriesScore:
    """Score one series over one fold.

    Args:
        train: Training actuals up to the fold origin, in date order.
        actual: Realised units over the test window.
        predicted: Forecast units over the same window.
        seasonality: Naive step for MASE's denominator.

    """
    return SeriesScore(
        mase=mase(actual, predicted, naive_scale(train, seasonality)),
        rmsse=rmsse(actual, predicted, naive_scale_squared(train)),
        bias=bias(actual, predicted),
    )


def aggregate(values: np.ndarray) -> tuple[float, int, int]:
    """Mean over scorable series.

    Returns:
        ``(mean, n_scored, n_unscorable)``. The unscorable count is returned rather than
        hidden so that a model scored on fewer series than another is visible rather than
        flattering.

    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mean = float(values[finite].mean()) if finite.any() else float("nan")
    return mean, int(finite.sum()), int((~finite).sum())
