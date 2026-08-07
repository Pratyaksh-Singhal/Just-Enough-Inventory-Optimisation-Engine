"""E3-S1 — seasonal naive is the number every later result is quoted against.

Its whole value is that it has no parameters to tune, so it cannot be quietly improved
until it flatters the comparison. That makes its definition worth pinning exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from inventory_engine.models.baselines import BASELINE_MODELS, seasonal_naive

WEEK = np.array([1.0, 2, 3, 4, 5, 6, 7])


def test_first_horizon_day_uses_the_same_weekday_last_week():
    """Target t+1 must read t-6, which is the same weekday one week earlier."""
    assert seasonal_naive(WEEK, horizon=1)[0] == 1.0


def test_seventh_day_uses_the_final_training_day():
    """Target t+7 is the same weekday as t."""
    assert seasonal_naive(WEEK, horizon=7)[6] == 7.0


def test_pattern_repeats_beyond_one_week():
    """A 28-day horizon reuses the last training week four times over."""
    out = seasonal_naive(WEEK, horizon=28)
    assert out.tolist() == WEEK.tolist() * 4


def test_only_the_last_week_is_used():
    """Earlier history must not leak in via averaging -- this baseline does not smooth."""
    long_history = np.concatenate([np.full(200, 99.0), WEEK])
    assert seasonal_naive(long_history, horizon=7).tolist() == WEEK.tolist()


def test_zeros_are_forecast_as_zeros():
    """On an intermittent series the naive forecast is genuinely zero, not imputed."""
    sparse = np.array([0.0, 0, 4, 0, 0, 0, 0])
    assert seasonal_naive(sparse, horizon=7).tolist() == sparse.tolist()


def test_partial_horizon_truncates_cleanly():
    out = seasonal_naive(WEEK, horizon=10)
    assert out.tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]


def test_short_history_is_rejected_rather_than_padded():
    with pytest.raises(ValueError, match="at least 7 training observations"):
        seasonal_naive(np.array([1.0, 2.0, 3.0]), horizon=7)


def test_custom_season_length():
    out = seasonal_naive(np.array([1.0, 2.0, 3.0]), horizon=6, season_length=3)
    assert out.tolist() == [1, 2, 3, 1, 2, 3]


def test_output_length_always_matches_horizon():
    for h in (1, 5, 28, 60):
        assert len(seasonal_naive(WEEK, horizon=h)) == h


def test_registry_lists_every_implemented_baseline():
    """The brief asks for a naive bar, an intermittent-demand method, and a classical one."""
    assert set(BASELINE_MODELS) == {"seasonal_naive", "croston", "tsb", "ets"}
