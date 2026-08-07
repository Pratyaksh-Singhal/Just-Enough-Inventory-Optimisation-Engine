"""E5-S1 — the fold layout is the one thing every model is compared through.

If these are wrong, every accuracy number in the project is wrong in a way that looks
entirely plausible, so the properties are asserted rather than eyeballed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from inventory_engine.backtest.folds import (
    Fold,
    assert_no_training_leak,
    describe_folds,
    make_folds,
)

LAST = date(2016, 5, 22)


@pytest.fixture
def folds():
    return make_folds(LAST, n_folds=5, horizon=28)


def test_last_fold_ends_on_the_panel_end(folds):
    """No evaluation day is wasted at the end of the panel."""
    assert folds[-1].test_end == LAST


def test_windows_are_contiguous_and_non_overlapping(folds):
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_end < later.test_start
        assert later.test_start == earlier.test_end + timedelta(days=1)


def test_windows_cover_exactly_the_reserved_region(folds):
    """Coverage must match the region Phase 1's sampler was forbidden from seeing."""
    covered = sum(f.horizon for f in folds)
    assert covered == 5 * 28
    assert folds[0].test_start == LAST - timedelta(days=covered - 1)


def test_origin_is_strictly_before_its_own_test_window(folds):
    for f in folds:
        assert f.origin_date < f.test_start
        assert f.test_start == f.origin_date + timedelta(days=1)
    assert_no_training_leak(folds)


def test_training_is_expanding_not_sliding(folds):
    """Each fold trains on everything before its origin, including earlier test windows."""
    origins = [f.origin_date for f in folds]
    assert origins == sorted(origins)
    assert folds[-1].origin_date > folds[0].test_end


def test_horizon_of_counts_from_the_origin(folds):
    f = folds[0]
    assert f.horizon_of(f.test_start) == 1
    assert f.horizon_of(f.test_end) == 28
    assert f.horizon_of(f.test_start + timedelta(days=6)) == 7


def test_horizon_of_rejects_dates_outside_the_window(folds):
    f = folds[0]
    with pytest.raises(ValueError, match="not in fold"):
        f.horizon_of(f.origin_date)
    with pytest.raises(ValueError, match="not in fold"):
        f.horizon_of(f.test_end + timedelta(days=1))


def test_target_dates_match_the_window(folds):
    f = folds[2]
    targets = f.target_dates()
    assert len(targets) == f.horizon
    assert targets[0] == f.test_start
    assert targets[-1] == f.test_end
    assert targets == sorted(targets)


def test_known_layout_is_stable():
    """Pin the exact dates the README and docs quote."""
    folds = make_folds(LAST, 5, 28)
    assert [(f.origin_date.isoformat(), f.test_start.isoformat()) for f in folds] == [
        ("2016-01-03", "2016-01-04"),
        ("2016-01-31", "2016-02-01"),
        ("2016-02-28", "2016-02-29"),
        ("2016-03-27", "2016-03-28"),
        ("2016-04-24", "2016-04-25"),
    ]


@pytest.mark.parametrize(("n_folds", "horizon"), [(0, 28), (5, 0), (-1, 28)])
def test_invalid_arguments_rejected(n_folds, horizon):
    with pytest.raises(ValueError, match="must be >= 1"):
        make_folds(LAST, n_folds, horizon)


def test_panel_too_short_is_rejected_with_a_useful_message():
    """A panel that leaves no training data must fail loudly, not train on nothing."""
    with pytest.raises(ValueError, match="no training data|leaves no"):
        make_folds(LAST, n_folds=5, horizon=28, first_date=LAST - timedelta(days=100))


def test_training_leak_is_detected():
    bad = (
        Fold(
            index=0,
            origin_date=date(2016, 2, 1),
            test_start=date(2016, 2, 1),
            test_end=date(2016, 2, 28),
            horizon=28,
        ),
    )
    with pytest.raises(ValueError, match="training overlaps evaluation"):
        assert_no_training_leak(bad)


def test_describe_renders_every_fold(folds):
    text = describe_folds(folds)
    assert text.count("2016-") >= len(folds) * 2
    assert "total evaluated" in text
