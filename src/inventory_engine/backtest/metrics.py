"""E5-S2 — forecast accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def naive_scale(train: np.ndarray, seasonality: int = 1) -> float:
    """Mean absolute ``seasonality``-step naive error over the training window."""
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
    """Root mean squared scaled error -- the per-series component of M5's WRMSSE."""
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
    """Pinball (quantile) loss at level ``quantile``."""
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
    """Score one series over one fold."""
    return SeriesScore(
        mase=mase(actual, predicted, naive_scale(train, seasonality)),
        rmsse=rmsse(actual, predicted, naive_scale_squared(train)),
        bias=bias(actual, predicted),
    )


def aggregate(values: np.ndarray) -> tuple[float, int, int]:
    """Mean over scorable series."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mean = float(values[finite].mean()) if finite.any() else float("nan")
    return mean, int(finite.sum()), int((~finite).sum())
