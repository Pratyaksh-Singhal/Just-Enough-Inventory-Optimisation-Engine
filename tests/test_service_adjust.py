"""The three states, and the guarantee that an unmatched product is never touched.

The load-bearing test in this file is
``test_an_unmatched_products_order_is_identical_with_the_feature_on_and_off``. Everything
else here describes behaviour; that one describes a promise. The whole festival feature
rests on a keyword match against a free-text product name, which is a guess, and the only
thing that makes a guess safe is that failing to make it costs nothing at all.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from inventory_engine.optimize.costs import CostModel
from inventory_engine.service.adjust import (
    LOOKAHEAD_DAYS,
    FestivalPlan,
    State,
    plan_for,
)
from inventory_engine.service.festivals import festivals
from inventory_engine.service.pipeline import forecast_sku
from inventory_engine.service.uplift import Source

#: Products the shipped demand table can recognise, and products it cannot. The opaque
#: codes are the realistic case, not a contrived one -- a great many real catalogues are
#: entirely made of them.
MATCHED = "Amul Paneer 200g"
MATCHED_NON_FOOD = "Diwali Diya Terracotta Pack of 12"
UNMATCHED = "SKU-88213"
UNMATCHED_TOO = "AT-1L-BLU"


def flat(start: str, days: int, level: float = 10.0) -> pd.Series:
    """A perfectly steady daily series, so any movement comes from the code under test."""
    index = pd.date_range(start, periods=days, freq="D")
    return pd.Series(np.full(days, level), index=index)


def frame_of(series: pd.Series) -> pd.DataFrame:
    """The upload-shaped frame ``forecast_sku`` takes."""
    return pd.DataFrame({"date": series.index, "units_sold": series.to_numpy()})


def ending_before(key: str, year: int, days_before: int, length: int = 200) -> pd.Series:
    """A series ending ``days_before`` days ahead of a named festival's own day."""
    day = next(f.day for f in festivals("IN") if f.key == key and f.day.year == year)
    last = day - timedelta(days=days_before)
    return flat((last - timedelta(days=length - 1)).isoformat(), length)


def quiet_series(length: int = 200) -> pd.Series:
    """A series whose order window and look-ahead contain no festival at all.

    Derived from the calendar rather than pinned to a date somebody once checked: the
    shipped set changes, and a hardcoded "June is quiet" is a fact about an old table.
    """
    windows = [(f.window_start, f.window_end) for f in festivals("IN")]
    horizon, reach = 28, 28 + LOOKAHEAD_DAYS
    last = next(
        d
        for d in (date(2025, 1, 1) + timedelta(days=i) for i in range(700))
        if not any(
            s <= d + timedelta(days=reach) and e >= d + timedelta(days=1) for s, e in windows
        )
    )
    assert horizon <= reach
    return flat((last - timedelta(days=length - 1)).isoformat(), length)


# --------------------------------------------------------------------- state 1: adjusted


def test_a_matched_product_inside_a_festival_window_is_adjusted_and_says_by_what():
    """Diwali, a recognisable dairy product, and a number the user can trace back."""
    plan = plan_for(MATCHED, ending_before("diwali", 2025, 15), horizon=28)

    assert plan.state is State.ADJUSTED
    assert plan.adjusts and plan.factor > 1.0
    assert plan.apply(100) > 100

    (match,) = plan.matches
    assert match.festival_key == "diwali"
    assert match.keyword == "paneer"
    assert match.category
    assert match.source is Source.PRIOR  # 200 days of history cannot reach a past Diwali
    assert "paneer" in plan.message and "Diwali" in plan.message


def test_a_non_food_product_is_matched_too():
    """Regression against a grocery-only table: a diya is the thing bought *for* Diwali."""
    plan = plan_for(MATCHED_NON_FOOD, ending_before("diwali", 2025, 15), horizon=28)
    (match,) = plan.matches
    assert match.keyword == "diya"
    assert "lighting" in match.category
    assert plan.factor > 1.0


def test_the_ratio_is_spread_over_the_run_up_rather_than_the_whole_order():
    """A 2.0x fortnight inside a 28-day order is not a 2.0x order.

    Diwali's run-up is 14 days. Half the order window sits inside it, so the factor is the
    mean of fourteen days at 2.0x and fourteen at 1.0x. Multiplying the whole order by 2.0x
    would buy a month of Diwali, which is the mistake this arithmetic exists to avoid.
    """
    # Ends the day before the run-up opens, so all 14 run-up days fall inside the horizon.
    diwali = next(f for f in festivals("IN") if f.key == "diwali" and f.day.year == 2025)
    series = flat((diwali.window_start - timedelta(days=200)).isoformat(), 200)
    plan = plan_for(MATCHED, series, horizon=28)

    (match,) = plan.matches
    assert match.days_in_window == 14
    assert plan.factor == pytest.approx(1 + (match.multiplier - 1) * 14 / 28)
    assert plan.factor == pytest.approx(1.5)


def test_a_suppressed_category_lowers_the_order():
    """Maha Shivratri suppresses alliums. A factor below 1.0 must survive the whole path."""
    plan = plan_for("Red Onion 1kg", ending_before("maha_shivaratri", 2025, 4), horizon=28)
    assert plan.state is State.ADJUSTED
    assert plan.matches[0].multiplier < 1.0
    assert plan.factor < 1.0
    assert plan.apply(100) < 100


def test_a_measured_ratio_outranks_the_reference_and_is_labelled_measured():
    """The shop's own Diwalis beat the table, and the label says which one won."""
    series = flat("2022-06-01", 1300)
    for f in festivals("IN"):
        if f.key == "diwali" and f.window_start >= series.index.min().date():
            run_up = pd.date_range(f.window_start, f.day - timedelta(days=1), freq="D")
            series.loc[series.index.isin(run_up)] *= 1.2

    plan = plan_for(MATCHED, series.loc[:"2025-10-05"], horizon=28)
    (match,) = plan.matches
    assert match.source is Source.MEASURED
    assert match.multiplier == pytest.approx(1.2, rel=0.05)
    assert match.multiplier != 2.0  # the reference figure did not leak in
    assert match.keyword == "paneer"  # still shown, so the user can still disagree


def test_a_prior_only_festival_arrives_as_a_suggestion_however_long_the_history():
    """Independence Day is never measured, so a flag gets the reference figure or nothing.

    Six years of history covering six 15 Augusts, and the source is still ``prior``. That
    is the flag doing its job: the refusal is a decision about the festival, not a report
    on how much data the shop has.
    """
    series = flat("2019-01-01", 2400)  # ends mid-2025, spans six Independence Days
    last = series.index.max().date()
    ahead = (date(last.year, 8, 15) - last).days - 3
    plan = plan_for("National Flag Small", series, horizon=max(ahead, 5))

    flag = next(m for m in plan.matches if m.festival_key == "independence_day")
    assert flag.source is Source.PRIOR
    assert flag.multiplier > 1.0
    assert flag.keyword == "national flag"  # the longer keyword wins over bare "flag"


# --------------------------------------------------------------------- state 2: advisory


def test_an_unmatched_product_near_a_festival_is_told_and_left_alone():
    """The state that makes the feature safe: a banner, and not one unit of movement."""
    plan = plan_for(UNMATCHED, ending_before("diwali", 2025, 15), horizon=28)

    assert plan.state is State.ADVISORY
    assert plan.factor == 1.0
    assert not plan.adjusts
    assert plan.apply(137.42) == 137.42
    assert plan.matches == ()
    assert "Diwali" in plan.message
    assert "no festival pattern was found for this item" in plan.message
    assert "unchanged" in plan.message


def test_the_advisory_says_which_festival_and_why_it_could_not_help():
    plan = plan_for(UNMATCHED, ending_before("diwali", 2025, 15), horizon=28)
    assert "Diwali" in plan.nearby
    assert any("Diwali" in u for u in plan.unresolved)


def test_a_festival_just_past_the_order_window_still_earns_a_mention():
    """Awareness reaches further than the adjustment does, and only ever as awareness.

    A recognisable product, a festival the table has a large figure for, and the run-up
    starting four days after this order window closes. The buyer hears about it; the
    quantity is for the days it covers and stays exactly as forecast.
    """
    plan = plan_for(MATCHED, ending_before("diwali", 2025, 25), horizon=7)
    assert plan.state is State.ADVISORY
    assert plan.factor == 1.0
    assert plan.matches == ()
    assert "Diwali" in plan.nearby


# ------------------------------------------------------------------------- state 3: none


def test_a_quiet_order_window_produces_no_banner_and_no_adjustment():
    """An inventory tool that mentions Diwali in June is one nobody reads in October."""
    for sku in (MATCHED, UNMATCHED):
        plan = plan_for(sku, quiet_series(), horizon=28)
        assert plan.state is State.NONE
        assert plan.factor == 1.0
        assert plan.message == ""
        assert plan.nearby == ()


def test_no_calendar_means_no_plan_at_all():
    plan = plan_for(MATCHED, ending_before("diwali", 2025, 15), horizon=28, region=None)
    assert plan.state is State.NONE
    assert plan == FestivalPlan.unchanged()


def test_an_unknown_region_degrades_to_doing_nothing_rather_than_failing_a_forecast():
    """A misconfigured calendar must not cost the user their order quantity."""
    plan = plan_for(MATCHED, ending_before("diwali", 2025, 15), horizon=28, region="XX")
    assert plan.state is State.NONE
    assert plan.factor == 1.0


def test_an_empty_series_is_survivable():
    assert plan_for(MATCHED, pd.Series(dtype=float), horizon=28).state is State.NONE


# ------------------------------------------------------------------ the promise, end to end


def test_an_unmatched_products_order_is_identical_with_the_feature_on_and_off():
    """**The** test. Matching may never cost anything when it fails.

    The history is short enough that both runs are served by the seasonal-naive baseline,
    which reads no features at all -- so the calendar's only possible influence on the
    number is the adjustment itself, and the comparison is exact rather than approximate.
    The assertion is equality to the last bit, not ``approx``: "unchanged" is a promise
    about the quantity, and a quantity that moved by 1e-9 did move.
    """
    series = ending_before("diwali", 2025, 15, length=140)
    frame = frame_of(series)

    off = forecast_sku(UNMATCHED, frame, horizon=28, costs=CostModel(), region=None)
    on = forecast_sku(UNMATCHED, frame, horizon=28, costs=CostModel(), region="IN")

    assert on.method_used == off.method_used == "seasonal_naive"  # the premise, stated
    assert on.order_qty == off.order_qty
    assert on.expected_cost == off.expected_cost
    assert on.series["order"] == off.series["order"]

    # And it is not that the feature did nothing at all: it looked, and it said so.
    assert on.festival["state"] == State.ADVISORY.value
    assert off.festival["state"] == State.NONE.value


def test_the_same_series_moves_only_because_of_the_product_name():
    """The other half of the promise: with the calendar identical, matching is the variable.

    Same frame, same region, same fit -- one product the table recognises and one it does
    not. Any difference in the order quantity is attributable to the match and to nothing
    else, which is what "never silently guess" has to mean in practice.
    """
    frame = frame_of(ending_before("diwali", 2025, 15, length=140))

    matched = forecast_sku(MATCHED, frame, horizon=28, costs=CostModel(), region="IN")
    unmatched = forecast_sku(UNMATCHED, frame, horizon=28, costs=CostModel(), region="IN")

    assert unmatched.order_qty == unmatched.order_qty_before_festival
    assert matched.order_qty > matched.order_qty_before_festival
    assert matched.order_qty_before_festival == unmatched.order_qty_before_festival


def test_both_quantities_are_always_reported_so_the_adjustment_can_be_undone():
    """A buyer who disagrees with the match needs the number it started from."""
    frame = frame_of(ending_before("diwali", 2025, 15, length=140))
    result = forecast_sku(MATCHED, frame, horizon=28, costs=CostModel(), region="IN")

    assert result.order_qty_before_festival > 0
    assert result.order_qty == pytest.approx(
        result.order_qty_before_festival * result.festival["factor"]
    )


def test_the_match_reaches_the_response_with_the_keyword_that_caused_it():
    """Visible confirmation, not an invisible background adjustment."""
    frame = frame_of(ending_before("diwali", 2025, 15, length=140))
    note = forecast_sku(MATCHED, frame, horizon=28, costs=CostModel(), region="IN").festival

    assert note["state"] == State.ADJUSTED.value
    (match,) = note["matches"]
    assert match["keyword"] == "paneer"
    assert match["festival_name"] == "Diwali"
    assert match["source"] in {"measured", "prior"}
    assert match["category"]


def test_the_chart_only_shades_windows_that_actually_moved_the_order():
    matched = forecast_sku(
        MATCHED,
        frame_of(ending_before("diwali", 2025, 15, length=140)),
        28,
        CostModel(),
        region="IN",
    )
    unmatched = forecast_sku(
        UNMATCHED,
        frame_of(ending_before("diwali", 2025, 15, length=140)),
        28,
        CostModel(),
        region="IN",
    )
    assert matched.series["festival_windows"]
    assert unmatched.series["festival_windows"] == []


# --------------------------------------------------------------------------- serialisation


def test_the_plan_serialises_to_something_a_column_can_hold():
    plan = plan_for(MATCHED, ending_before("diwali", 2025, 15), horizon=28)
    blob = plan.to_dict()

    import json

    assert json.loads(json.dumps(blob)) == blob
    assert blob["state"] == "adjusted"
    assert blob["matches"][0]["source"] == "prior"


@pytest.mark.parametrize("sku", [UNMATCHED, UNMATCHED_TOO, "", "   ", "12345"])
def test_nothing_a_catalogue_can_contain_produces_an_accidental_match(sku):
    """Opaque codes, blanks and bare numbers all resolve to "we cannot tell"."""
    plan = plan_for(sku, ending_before("diwali", 2025, 15), horizon=28)
    assert plan.matches == ()
    assert plan.apply(500.0) == 500.0
