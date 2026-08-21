"""Measuring festival uplift from a SKU's own history.

Synthetic series with a known planted effect, so the tests assert on a number we chose
rather than on whatever the code happens to produce.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from inventory_engine.service.festivals import Festival
from inventory_engine.service.uplift import (
    Source,
    Uplift,
    measure,
    measure_all,
    measure_sku,
    relevant_for,
)


def flat_series(start="2023-01-01", days=900, level=10.0) -> pd.Series:
    """Perfectly steady demand — any ratio the code reports is an artefact."""
    index = pd.date_range(start, periods=days, freq="D")
    return pd.Series(np.full(days, level), index=index)


def plant(series: pd.Series, day: date, lead: int, factor: float) -> pd.Series:
    """Multiply the run-up before ``day`` by ``factor``."""
    out = series.copy()
    window = pd.date_range(day - timedelta(days=lead), day - timedelta(days=1), freq="D")
    out.loc[out.index.isin(window)] *= factor
    return out


def fest(day: date, lead: int = 14, tail: int = 2) -> Festival:
    """A festival object for an arbitrary date."""
    return Festival("test", "Test Festival", "IN", day, lead, tail)


def plant_every(series: pd.Series, key: str, factor: float, region: str = "IN") -> pd.Series:
    """Plant the same effect on every instance of ``key`` the series actually spans.

    Needed because a multi-year synthetic series covers more Diwalis than one might plant by
    hand, and ``measure_sku`` correctly finds all of them -- the unplanted ones come back at
    1.0 and drag the median. That is the code being right and the fixture being naive.
    """
    from inventory_engine.service.festivals import windows_in

    out = series.copy()
    first, last = out.index.min().date(), out.index.max().date()
    for f in windows_in(first, last, region):
        if f.key == key:
            out = plant(out, f.day, f.lead_days, factor)
    return out


DIWALI_2024 = date(2024, 11, 1)


# --------------------------------------------------------------------------- measurement


def test_a_flat_series_measures_no_uplift():
    """The control. If this is not 1.0, every other number here is suspect."""
    occ = measure(flat_series(), fest(DIWALI_2024))
    assert occ is not None
    assert occ.lead_ratio == pytest.approx(1.0, abs=1e-9)


def test_a_planted_uplift_is_recovered():
    series = plant(flat_series(), DIWALI_2024, lead=14, factor=1.8)
    occ = measure(series, fest(DIWALI_2024))
    assert occ.lead_ratio == pytest.approx(1.8, rel=1e-6)


def test_a_planted_drop_is_recovered_too():
    """Navratri suppresses onion and non-veg demand; ratios below 1 must survive."""
    series = plant(flat_series(), DIWALI_2024, lead=14, factor=0.6)
    assert measure(series, fest(DIWALI_2024)).lead_ratio == pytest.approx(0.6, rel=1e-6)


def test_the_festival_day_is_excluded_from_the_ordering_signal():
    """Regression: a closure on the day dragged the whole measurement below 1.

    Real M5 numbers behind this -- 24 Dec 1,168 units, 25 Dec **0**, 26 Dec 1,135. Averaged
    into one window that reported Christmas at 0.84x, i.e. demand falling at Christmas.
    """
    series = plant(flat_series(), DIWALI_2024, lead=14, factor=1.8)
    shut = series.copy()
    shut.loc[shut.index.date == DIWALI_2024] = 0.0

    occ = measure(shut, fest(DIWALI_2024))
    assert occ.lead_ratio == pytest.approx(1.8, rel=1e-6)  # untouched by the closure
    assert occ.day_ratio < 1.0  # and the closure drags the day itself down
    assert occ.closed_days == 1


def test_a_zero_baseline_yields_no_ratio_rather_than_infinity():
    """A SKU with no normal demand has no meaningful multiple of it."""
    index = pd.date_range("2024-01-01", periods=600, freq="D")
    series = pd.Series(np.zeros(600), index=index)
    series.loc[series.index.date == DIWALI_2024] = 5.0
    assert measure(series, fest(DIWALI_2024)) is None


def test_too_few_observations_yields_nothing():
    index = pd.date_range("2024-10-25", periods=10, freq="D")
    assert measure(pd.Series(np.full(10, 5.0), index=index), fest(DIWALI_2024)) is None


def test_the_baseline_matches_on_weekday():
    """Otherwise a window containing two weekends reports the weekend as festival uplift."""
    index = pd.date_range("2023-01-01", periods=900, freq="D")
    weekend_heavy = pd.Series(np.where(index.dayofweek >= 5, 30.0, 10.0), index=index, dtype=float)
    occ = measure(weekend_heavy, fest(DIWALI_2024))
    # No planted festival effect at all, so despite a huge weekday swing the ratio is ~1.
    assert occ.lead_ratio == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------- per SKU


def test_a_short_history_refuses_rather_than_guessing():
    """The 90-day user gets the truth, not a number."""
    short = flat_series(start="2025-01-01", days=120)
    u = measure_sku("WIDGET", short, "diwali", region="IN")
    assert not u.available
    assert u.source is Source.UNAVAILABLE
    assert "13 months" in u.reason


def test_a_long_history_measures_and_labels_its_source():
    series = plant_every(flat_series(start="2023-06-01", days=900), "diwali", 1.6)
    u = measure_sku("PANEER", series, "diwali", region="IN")
    assert u.available and u.source is Source.MEASURED
    assert u.multiplier == pytest.approx(1.6, rel=0.02)
    assert u.n_years >= 1


def test_the_median_across_years_resists_one_odd_year():
    """Two normal Diwalis and one freak year should not move the order much."""
    series = flat_series(start="2022-01-01", days=1200)  # spans Diwali 2022, 2023, 2024
    series = plant(series, date(2022, 10, 24), 14, 1.5)
    series = plant(series, date(2023, 11, 12), 14, 1.5)
    series = plant(series, date(2024, 11, 1), 14, 6.0)  # the freak
    u = measure_sku("X", series, "diwali", region="IN")
    assert u.n_years == 3
    assert u.multiplier == pytest.approx(1.5, rel=0.05)
    assert u.spread[1] > 5  # but the outlier is still visible in the range


def test_measure_all_ranks_by_how_far_from_normal():
    series = flat_series(start="2023-01-01", days=1000)
    series = plant(series, date(2024, 11, 1), 14, 2.0)  # diwali, strong
    series = plant(series, date(2024, 3, 25), 7, 1.1)  # holi, weak
    ranked = measure_all("X", series, region="IN")
    assert ranked[0].festival_key == "diwali"
    assert all(u.source is Source.MEASURED for u in ranked)


def test_an_empty_series_returns_nothing_rather_than_raising():
    assert measure_all("X", pd.Series(dtype=float), region="IN") == []


# ------------------------------------------------------- prior-only and partial calendars


def test_a_prior_only_festival_is_never_measured_however_much_history_there_is():
    """Independence Day is a decision, not a data shortage.

    The point of the flag is that it cannot be routed around: this series plants a large,
    clean, repeated effect on exactly the right dates and the answer is still "not
    measured", because measuring it was never on offer.
    """
    series = plant_every(flat_series(start="2019-01-01", days=2200), "independence_day", 1.9)
    u = measure_sku("TIRANGA FLAG", series, "independence_day", region="IN")
    assert u.source is Source.UNAVAILABLE
    assert "by design" in u.reason
    assert "13 months" not in u.reason  # not a history problem, and must not read as one


def test_a_measured_festival_with_a_gappy_calendar_says_so():
    """Maha Shivratri is dated in four years of nine, and the number carries that."""
    series = plant_every(flat_series(start="2018-06-01", days=3000), "maha_shivaratri", 1.5)
    u = measure_sku("SABUDANA 500G", series, "maha_shivaratri", region="IN")
    assert u.source is Source.MEASURED
    assert u.partial_calendar
    assert u.multiplier == pytest.approx(1.5, rel=0.02)
    assert "missing 2020" in u.describe()


def test_a_year_we_cannot_date_is_not_reported_as_a_missing_year_of_data():
    """Two different problems: "upload more history" is wrong advice for a calendar hole.

    2023 and 2024 are both inside this series and neither has a Maha Shivratri date, so the
    festival is unmeasurable for a reason no amount of extra uploading would fix.
    """
    series = flat_series(start="2023-04-01", days=420)  # spans no dated Maha Shivratri
    u = measure_sku("X", series, "maha_shivaratri", region="IN")
    assert u.source is Source.UNAVAILABLE
    assert u.partial_calendar
    assert "our calendar has no date" in u.reason
    assert "13 months" not in u.reason


# --------------------------------------------------------------------------- reporting


def test_an_unavailable_uplift_describes_why():
    u = measure_sku("X", flat_series(days=100), "diwali", region="IN")
    assert "not measurable" in u.describe()


def test_a_single_year_is_flagged_as_thin_evidence():
    series = plant(flat_series(start="2024-04-01", days=420), DIWALI_2024, 14, 1.7)
    u = measure_sku("X", series, "diwali", region="IN")
    assert u.n_years == 1
    assert "One year of evidence" in u.describe()


def test_a_prior_says_so_and_never_claims_to_be_measured():
    """The whole point of Source: a reference figure must not read like the user's own data."""
    u = Uplift(
        "X",
        "diwali",
        "Diwali",
        source=Source.PRIOR,
        prior_multiplier=2.0,
        prior_reason="Biggest retail event of the year.",
    )
    text = u.describe()
    assert u.multiplier == 2.0
    assert "suggested" in text and "not measured from your sales" in text
    assert "your sales ran" not in text


def test_a_closure_is_mentioned_rather_than_absorbed():
    series = plant_every(flat_series(start="2023-06-01", days=900), "diwali", 1.5)
    for day in (date(2023, 11, 12), DIWALI_2024, date(2025, 10, 20)):
        series.loc[series.index.date == day] = 0.0
    u = measure_sku("X", series, "diwali", region="IN")
    assert u.closed_on_the_day
    assert "no sales on the day itself" in u.describe()


# --------------------------------------------------------------------------- relevance


def test_only_festivals_inside_the_horizon_are_relevant():
    """A Diwali uplift measured in March is a fact about the past, not a reason to order."""
    keys = {f.key for f in relevant_for(date(2025, 10, 1), horizon=28, region="IN")}
    assert "diwali" in keys
    assert "holi" not in keys


def test_a_quiet_horizon_has_no_relevant_festivals():
    """Derived, not hardcoded.

    This used to assert on June 2025, which was quiet in the hand-typed calendar and is not
    in the library-sourced one -- Eid al-Adha falls on 7 June. Pinning a date meant the test
    encoded a fact about the old table rather than the behaviour, so it found a real gap in
    the calendar and reported it as a bug in the code.
    """
    from inventory_engine.service.festivals import festivals

    windows = [(f.window_start, f.window_end) for f in festivals("IN")]
    quiet = next(
        d
        for d in (date(2025, 1, 1) + timedelta(days=i) for i in range(365))
        if not any(s <= d + timedelta(days=14) and e >= d + timedelta(days=1) for s, e in windows)
    )
    assert relevant_for(quiet, horizon=14, region="IN") == []


def test_a_partial_festival_missing_from_the_coverage_map_degrades_rather_than_raises(
    monkeypatch,
):
    """``SOURCES[key].partial`` and ``PARTIAL`` are separate dicts and can disagree.

    ``measure_sku`` used to index ``partial_coverage()[key]`` directly, so marking a
    ``Source`` partial without adding a matching ``PARTIAL`` row raised ``KeyError`` inside
    the worker and failed the whole forecast run. ``Uplift._partial_note`` already used
    ``.get()`` with a default for exactly this reason.
    """
    from inventory_engine.service import festivals, uplift

    monkeypatch.setattr(festivals, "PARTIAL", {}, raising=True)
    monkeypatch.setattr(uplift, "partial_coverage", lambda: {}, raising=False)
    monkeypatch.setattr(uplift, "is_partial", lambda key: True, raising=False)

    # April to May carries no Maha Shivratri, so `instances` is empty and the branch
    # that reads the coverage map is the one that runs.
    index = pd.date_range("2025-04-01", periods=60, freq="D")
    series = pd.Series(np.full(60, 8.0), index=index)

    result = uplift.measure_sku("Paneer 200g", series, "maha_shivaratri")

    assert not result.available
    assert "not measurable" in result.reason
