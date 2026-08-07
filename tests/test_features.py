"""E2-S2/S3/S4/S5 — feature values are correct, not merely leakage-free.

`test_no_leakage.py` proves no feature reads the future. That is necessary but not
sufficient: a builder returning all zeros would sail through it. These tests pin the
actual arithmetic against hand-computed expectations on a single deterministic series.

Horizon is 3 here rather than 28 so the expected slices stay short enough to reason about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic import deterministic_panel, features, ts

H = 3


@pytest.fixture(scope="module")
def built():
    cal, fact, units = deterministic_panel()
    return features(fact, cal, H).set_index("date"), units


def at(df: pd.DataFrame, i: int, col: str):
    """Feature ``col`` on the row targeting panel day ``i``."""
    return df.loc[ts(i), col]


# ---------------------------------------------------------------------------
# Lags — numbered from the forecast origin, not the target date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [7, 14, 28])
@pytest.mark.parametrize("i", [60, 85, 110])
def test_lag_reads_lag_days_before_the_origin(built, lag, i):
    df, units = built
    assert at(df, i, f"units_lag_{lag}") == units[i - H - lag]


def test_lag_is_null_before_the_series_is_long_enough(built):
    df, _ = built
    # Targeting day 5 with horizon 3 would need units at day -5.
    assert pd.isna(at(df, 5, "units_lag_7"))


# ---------------------------------------------------------------------------
# Rolling statistics — window ends at the origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [7, 28, 90])
def test_rolling_mean_matches_the_shifted_slice(built, window):
    df, units = built
    i = 110
    expected = units[i - H - window + 1 : i - H + 1]
    assert len(expected) == window
    assert at(df, i, f"units_roll_mean_{window}") == pytest.approx(expected.mean())


def test_rolling_max_and_std_match_the_shifted_slice(built):
    df, units = built
    i = 100
    window = 28
    expected = units[i - H - window + 1 : i - H + 1]
    assert at(df, i, f"units_roll_max_{window}") == expected.max()
    # Sample stddev, matching DuckDB's stddev_samp.
    assert at(df, i, f"units_roll_std_{window}") == pytest.approx(expected.std(ddof=1))


def test_rolling_window_excludes_the_origin_day_itself_and_beyond(built):
    """The newest value a window may touch is units[i - horizon]."""
    df, units = built
    i = 90
    included = units[i - H - 6 : i - H + 1]
    assert at(df, i, "units_roll_mean_7") == pytest.approx(included.mean())
    # A window that wrongly ended at the target would have this mean instead.
    wrong = units[i - 6 : i + 1]
    assert at(df, i, "units_roll_mean_7") != pytest.approx(wrong.mean())


def test_partial_windows_use_available_data(built):
    """A window reaching before the series start aggregates what exists.

    This is `min_periods=1` semantics, not pandas' default of NULL-until-full. It is a
    deliberate choice -- a partial mean is more useful to a GBM than a NULL -- but it does
    mean an early `units_roll_mean_90` is a mean of fewer than 90 days. The first-listing
    filter and the 365-day lag warm-up make this a small share of rows; it is pinned here
    so the behaviour is known rather than discovered.
    """
    df, units = built
    i = 10
    expected = units[: i - H + 1]  # frame start is before day 0
    assert len(expected) < 90
    assert at(df, i, "units_roll_mean_90") == pytest.approx(expected.mean())


def test_rolling_is_null_only_when_the_whole_frame_is_out_of_range(built):
    df, _ = built
    assert pd.isna(at(df, 2, "units_roll_mean_7")), "frame ends at day -1, nothing to read"
    assert not pd.isna(at(df, 3, "units_roll_mean_7")), "frame reaches day 0"


# ---------------------------------------------------------------------------
# Intermittency
# ---------------------------------------------------------------------------


def test_days_since_last_sale_is_measured_from_the_origin(built):
    df, units = built
    for i in (40, 77, 100):
        origin = i - H
        prior = np.nonzero(units[: origin + 1])[0]
        assert at(df, i, "days_since_last_sale") == origin - prior[-1]


def test_days_since_last_sale_is_zero_when_the_origin_day_sold(built):
    df, units = built
    i = 50
    assert units[i - H] > 0, "fixture: origin day should have a sale"
    assert at(df, i, "days_since_last_sale") == 0


def test_zero_share_90_matches_the_shifted_slice(built):
    df, units = built
    i = 110
    window = units[i - H - 89 : i - H + 1]
    assert len(window) == 90
    assert at(df, i, "zero_share_90") == pytest.approx((window == 0).mean())


# ---------------------------------------------------------------------------
# Calendar — known in advance, therefore unshifted
# ---------------------------------------------------------------------------


def test_calendar_features_describe_the_target_date_not_the_origin(built):
    """Calendar features are the one thing that should describe t, not t - horizon."""
    df, _ = built
    row = df.loc[ts(60)]
    target = ts(60)
    assert row["month"] == target.month
    assert row["week_of_year"] == target.isocalendar().week


def test_days_to_and_since_event(built):
    """Events fall every 25 days in this fixture."""
    df, _ = built
    assert at(df, 25, "days_since_event") == 0
    assert at(df, 25, "days_to_event") == 0
    assert at(df, 30, "days_since_event") == 5
    assert at(df, 30, "days_to_event") == 20
    assert at(df, 49, "days_since_event") == 24
    assert at(df, 49, "days_to_event") == 1


def test_is_weekend_follows_m5_wday(built):
    """M5 numbers wday from Saturday=1, so the weekend is wday 1 and 2."""
    df, _ = built
    weekend = df[df["is_weekend"]]
    assert set(weekend["wday"].unique()) == {1, 2}


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


def test_price_change_flag_fires_once_on_the_step(built):
    """Fixture price steps from 2.00 to 3.00 at day 60."""
    df, _ = built
    assert at(df, 60, "price") == 3.00
    assert at(df, 59, "price") == 2.00
    assert bool(at(df, 60, "price_changed"))
    assert not bool(at(df, 61, "price_changed"))
    assert not bool(at(df, 59, "price_changed"))


def test_price_rel_28_is_price_over_its_trailing_mean(built):
    df, _ = built
    i = 65
    # 28-day trailing mean including today: days 38..65 -> 22 days at 2.00, 6 at 3.00.
    expected_mean = (22 * 2.00 + 6 * 3.00) / 28
    assert at(df, i, "price_rel_28") == pytest.approx(3.00 / expected_mean)


def test_price_rel_28_is_one_when_price_is_flat(built):
    df, _ = built
    assert at(df, 40, "price_rel_28") == pytest.approx(1.0)


def test_is_listed_is_true_when_priced(built):
    df, _ = built
    assert bool(df["is_listed"].all())


# ---------------------------------------------------------------------------
# Panel shape
# ---------------------------------------------------------------------------


def test_target_is_the_actual_at_the_target_date(built):
    df, units = built
    for i in (0, 33, 119):
        assert at(df, i, "units") == units[i]


def test_horizon_column_records_the_build(built):
    df, _ = built
    assert (df["horizon"] == H).all()
