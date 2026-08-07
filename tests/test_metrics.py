"""E5-S2 — metric arithmetic, and the conventions that make the numbers comparable."""

from __future__ import annotations

import numpy as np
import pytest

from inventory_engine.backtest.metrics import (
    aggregate,
    bias,
    mase,
    naive_scale,
    naive_scale_squared,
    pinball,
    score_series,
)

# ---------------------------------------------------------------------------
# The scale denominator
# ---------------------------------------------------------------------------


def test_scale_ignores_leading_zeros():
    """M5 convention: history starts at the first sale, not at the start of the panel.

    Leading zeros are a not-yet-introduced SKU, not quiet demand. Counting them shrinks the
    denominator, which inflates the apparent skill of every model scored against it.
    """
    with_lead_in = naive_scale(np.array([0, 0, 0, 3, 5, 8]))
    without = naive_scale(np.array([3, 5, 8]))
    assert with_lead_in == pytest.approx(without)
    assert with_lead_in == pytest.approx(2.5)  # |5-3|=2, |8-5|=3


def test_scale_squared_matches_hand_computation():
    assert naive_scale_squared(np.array([0, 3, 5, 8])) == pytest.approx((4 + 9) / 2)


def test_scale_is_undefined_for_a_series_that_never_sells():
    assert np.isnan(naive_scale(np.zeros(50)))
    assert np.isnan(naive_scale_squared(np.zeros(50)))


def test_scale_is_undefined_for_a_flat_series():
    """A constant series has zero naive error; dividing by it would be infinite skill."""
    assert np.isnan(naive_scale(np.full(20, 4.0)))


def test_scale_is_undefined_when_history_is_too_short():
    assert np.isnan(naive_scale(np.array([5.0])))


def test_seasonal_scale_uses_the_requested_step():
    y = np.array([1.0, 9, 1, 9, 1, 9, 1, 9])
    assert naive_scale(y, seasonality=1) == pytest.approx(8.0)
    assert naive_scale(y, seasonality=2) == pytest.approx(0.0, abs=1e-9) or np.isnan(
        naive_scale(y, seasonality=2)
    )


# ---------------------------------------------------------------------------
# Scaled metrics
# ---------------------------------------------------------------------------


def test_mase_matches_hand_computation():
    assert mase(np.array([3.0, 3.0]), np.array([4.0, 5.0]), 2.5) == pytest.approx(1.5 / 2.5)


def test_mase_of_one_means_as_good_as_naive():
    actual = np.array([10.0, 12.0, 8.0])
    predicted = actual + 2.5  # error exactly equal to the scale
    assert mase(actual, predicted, 2.5) == pytest.approx(1.0)


def test_perfect_forecast_scores_zero():
    y = np.array([4.0, 0.0, 7.0])
    assert mase(y, y, 3.0) == pytest.approx(0.0)


def test_scaled_metrics_propagate_an_undefined_scale():
    """An unscorable series must be NaN, never silently 0 or infinity."""
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), float("nan")))


# ---------------------------------------------------------------------------
# Bias: the sign is the point
# ---------------------------------------------------------------------------


def test_over_forecast_is_positive_bias():
    """Positive = ordered too much = dead stock. E7 acts on this sign."""
    assert bias(np.array([3.0, 3.0]), np.array([5.0, 4.0])) == pytest.approx(1.5)


def test_under_forecast_is_negative_bias():
    """Negative = stockout."""
    assert bias(np.array([5.0, 4.0]), np.array([3.0, 3.0])) == pytest.approx(-1.5)


def test_bias_cancels_and_that_is_intentional():
    """Bias measures systematic direction, not magnitude -- MASE covers magnitude."""
    assert bias(np.array([5.0, 5.0]), np.array([7.0, 3.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pinball
# ---------------------------------------------------------------------------


def test_high_quantile_punishes_under_forecasting_hardest():
    """A 0.9 forecast that is too low should hurt roughly 9x more than one too high."""
    too_low = pinball(np.array([10.0]), np.array([8.0]), 0.9)
    too_high = pinball(np.array([8.0]), np.array([10.0]), 0.9)
    assert too_low == pytest.approx(1.8)
    assert too_high == pytest.approx(0.2)
    assert too_low > too_high


def test_median_quantile_is_symmetric():
    lo = pinball(np.array([10.0]), np.array([8.0]), 0.5)
    hi = pinball(np.array([8.0]), np.array([10.0]), 0.5)
    assert lo == pytest.approx(hi)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_pinball_rejects_degenerate_quantiles(q):
    with pytest.raises(ValueError, match="quantile must be in"):
        pinball(np.array([1.0]), np.array([1.0]), q)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_excludes_unscorable_series_and_counts_them():
    """A model scored on fewer series than another must be visible, not flattering."""
    mean, scored, excluded = aggregate(np.array([1.0, 2.0, np.nan, np.inf]))
    assert mean == pytest.approx(1.5)
    assert (scored, excluded) == (2, 2)


def test_aggregate_of_nothing_is_nan_not_zero():
    mean, scored, excluded = aggregate(np.array([np.nan, np.nan]))
    assert np.isnan(mean)
    assert (scored, excluded) == (0, 2)


def test_score_series_ties_the_pieces_together():
    train = np.array([0.0, 0.0, 3.0, 5.0, 8.0])
    actual = np.array([3.0, 3.0])
    predicted = np.array([4.0, 5.0])
    score = score_series(train, actual, predicted)
    assert score.mase == pytest.approx(1.5 / 2.5)
    assert score.rmsse == pytest.approx(np.sqrt(2.5 / 6.5))
    assert score.bias == pytest.approx(1.5)
    assert score.is_scorable()


def test_score_series_is_unscorable_when_the_series_never_sold():
    score = score_series(np.zeros(10), np.array([1.0]), np.array([0.0]))
    assert not score.is_scorable()
    assert score.bias == pytest.approx(-1.0), "bias stays defined without a scale"
