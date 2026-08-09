"""Features for a user-supplied series: leak-free, and defined on short history.

The leakage test here is the important one. Tier 1 has ``test_no_leakage.py`` guarding the
M5 panel; this is its tier 2 counterpart, and it works the same way -- mutate the future and
assert the features do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_engine.service.features import (
    FEATURE_COLUMNS,
    LAGS,
    WARMUP_DAYS,
    WINDOWS,
    supervised_frame,
    to_daily,
    total_frame,
)


def daily(days=200, start="2025-01-01", seed=0) -> pd.Series:
    """A reproducible daily series with a weekday cycle."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D")
    weekday = 5 + 4 * (index.dayofweek >= 5)
    return pd.Series(np.maximum(0, weekday + rng.normal(0, 1.5, days)).round(), index=index)


# --------------------------------------------------------------------------- warmup


def test_warmup_is_the_longest_lag_not_the_longest_window():
    """Regression: warmup at ``max(WINDOWS)`` silently disabled the model on short uploads.

    At 91 the warmup exceeded every fold's training window on a 120-day upload, so
    ``supervised_frame`` returned nothing and the baseline was served with the reason "the
    model could not be fitted" on data that fits one perfectly well. Caught by running the
    pipeline on real M5 series; pinned here so it cannot come back.
    """
    assert WARMUP_DAYS == max(LAGS) == 28
    assert max(WINDOWS) > WARMUP_DAYS


def test_a_ninety_day_series_produces_training_rows():
    frame = supervised_frame(daily(days=90), horizon=28)
    assert not frame.empty
    assert frame["origin"].nunique() == 90 - WARMUP_DAYS


def test_rolling_windows_are_defined_before_their_full_window_has_elapsed():
    """``min_periods`` is what makes a 91-day mean usable on a 90-day file."""
    frame = supervised_frame(daily(days=90), horizon=1)
    assert frame["roll_mean_91"].notna().any()


# --------------------------------------------------------------------------- leakage


def test_no_feature_can_see_past_its_origin():
    series = daily(days=200)
    origin = series.index[120]

    before = supervised_frame(series, horizon=28, origins=pd.DatetimeIndex([origin]))
    tampered = series.copy()
    tampered.iloc[121:] *= 100  # rewrite the entire future
    after = supervised_frame(tampered, horizon=28, origins=pd.DatetimeIndex([origin]))

    pd.testing.assert_frame_equal(
        before[list(FEATURE_COLUMNS)], after[list(FEATURE_COLUMNS)], check_dtype=False
    )


def test_the_target_does_move_when_the_future_changes():
    """Control for the test above: if nothing moved, that test would prove nothing."""
    series = daily(days=200)
    origin = series.index[120]
    before = supervised_frame(series, horizon=28, origins=pd.DatetimeIndex([origin]))
    tampered = series.copy()
    tampered.iloc[121:] *= 100
    after = supervised_frame(tampered, horizon=28, origins=pd.DatetimeIndex([origin]))
    assert not np.allclose(before["y"], after["y"])


def test_lag_1_is_the_origins_own_value():
    """The origin's sales are known when the order is placed, so lag_1 is today, not yesterday."""
    series = daily(days=100)
    origin = series.index[60]
    row = supervised_frame(series, horizon=1, origins=pd.DatetimeIndex([origin])).iloc[0]
    assert row["lag_1"] == pytest.approx(series.loc[origin])


# --------------------------------------------------------------------------- shape


def test_one_row_per_origin_per_horizon_step():
    frame = supervised_frame(daily(days=100), horizon=14)
    assert len(frame) == (100 - WARMUP_DAYS) * 14
    assert set(frame["horizon"]) == set(range(1, 15))


def test_targets_are_the_right_calendar_days():
    series = daily(days=100)
    origin = series.index[50]
    frame = supervised_frame(series, horizon=7, origins=pd.DatetimeIndex([origin]))
    expected = [origin + pd.Timedelta(days=h) for h in range(1, 8)]
    assert list(frame["target"]) == expected
    assert list(frame["target_dow"]) == [d.dayofweek for d in expected]


def test_every_declared_feature_column_is_actually_produced():
    frame = supervised_frame(daily(days=100), horizon=3)
    assert set(FEATURE_COLUMNS) <= set(frame.columns)


# --------------------------------------------------------------------------- total grain


def test_the_total_target_is_the_forward_window_sum():
    series = daily(days=120)
    frame = total_frame(series, horizon=28).set_index("origin")
    origin = series.index[50]
    expected = series.iloc[51:79].sum()
    assert frame.loc[origin, "y"] == pytest.approx(expected)


def test_the_last_origins_have_no_complete_forward_window():
    """They must be NaN, not a short sum -- a 12-day total is not a 28-day total."""
    series = daily(days=120)
    frame = total_frame(series, horizon=28).set_index("origin")
    assert pd.isna(frame.loc[series.index[-1], "y"])
    assert pd.isna(frame.loc[series.index[-5], "y"])


def test_total_frame_has_one_row_per_origin():
    frame = total_frame(daily(days=120), horizon=28)
    assert len(frame) == 120 - WARMUP_DAYS
    assert frame["origin"].is_unique


# --------------------------------------------------------------------------- to_daily


def test_to_daily_fills_gaps_with_nan_not_zero():
    """A missing row is an absent record; a zero is a recorded day with no sales."""
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-05"],
            "units_sold": [3.0, 4.0, 5.0],
        }
    )
    series = to_daily(frame)
    assert len(series) == 5
    assert series.isna().sum() == 2
    assert series.iloc[0] == 3.0


def test_to_daily_sums_repeated_dates():
    frame = pd.DataFrame(
        {"date": ["2025-01-01", "2025-01-01", "2025-01-02"], "units_sold": [3.0, 4.0, 5.0]}
    )
    assert to_daily(frame).iloc[0] == 7.0
