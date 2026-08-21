"""What the festival calendar does to one product's order quantity, and what it says about it.

Three states, and the user sees all three
-----------------------------------------
Everything in :mod:`~inventory_engine.service.festivals`,
:mod:`~inventory_engine.service.uplift` and :mod:`~inventory_engine.service.priors` produces
*numbers about festivals*. None of it changes an order. This module is where it becomes a
decision, so it is also where the decision has to be legible:

**Adjusted.** The product matched a festival category and there is a ratio to apply — the
shop's own measured run-up, or the reference figure where nothing could be measured. The
order quantity moves, and the response names the festival, the category, the keyword that
caused the match and which of the two sources drove it.

**Advisory.** A festival falls in or near the order window and the product matched nothing.
The order quantity is **not** touched. The user is told the festival is coming and told, in
those words, that no festival pattern was found for this item. This is the state that makes
the feature safe: the default answer to "is this product affected" is *we don't know*, and
"we don't know" must never spend money.

**None.** No festival anywhere near the order window. No banner, no adjustment, nothing —
an inventory tool that mentions Diwali in June is one a buyer learns to ignore in October.

Why matching is never silent
----------------------------
The whole feature rests on a keyword match against a product name, which is a guess. "Milk
Bikis" is a biscuit that matches ``milk``; "AT-1L-BLU" is milk that matches nothing. A wrong
match here is not a wrong number on a screen, it is twice as much paneer as anyone can sell.

So a match is only ever allowed to *change* an order while it is simultaneously *shown*:
:attr:`FestivalPlan.matches` carries the keyword and the category for every product it
touched, and ``tests/test_service_adjust.py`` pins the other half — that an unmatched product
gets a byte-identical order quantity whether this module runs or not.

How a ratio over a fortnight becomes a factor on a month's order
----------------------------------------------------------------
The uplift is a **run-up ratio**: how much more this product sold over the days before the
festival than over a comparable normal stretch. It is not a statement about the whole
horizon, and applying it as one is the obvious mistake. A 28-day order covering Diwali's
14-day run-up is 14 ordinary days and 14 festival days; multiplying the entire order by 2.0x
would buy a month of Diwali.

So the factor is the **mean of the per-day multipliers over the horizon**: each day inside a
matched festival's run-up contributes that festival's ratio, every other day contributes
1.0. For one festival this is exactly ``1 + (ratio - 1) * covered_days / horizon``. Where two
run-ups overlap, the nearer festival owns the day, using the same tie-break the calendar
applies in :func:`~inventory_engine.service.festivals.active` — the shopper buying on that
day is buying for one of them, not both, and the two ratios must never multiply together.

The festival day itself and its tail are deliberately excluded, because the ratio being
applied was measured over the run-up excluding them. Stock has to be on the shelf *before*
the festival; that is when the order is placed and that is the window the number describes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import StrEnum
from typing import Final

import pandas as pd

from inventory_engine.service.festivals import (
    DEFAULT_REGION,
    Festival,
    coverage,
    covers,
    tie_break,
    windows_in,
)
from inventory_engine.service.priors import match, resolve
from inventory_engine.service.uplift import Source, Uplift

#: Days past the end of the order window still worth mentioning. A festival three days
#: after the horizon does not belong in this order, and a buyer placing it would still
#: rather hear about it now than be surprised at the next cycle. Awareness only: nothing
#: outside the horizon can move a quantity.
LOOKAHEAD_DAYS: Final = 14

#: Below this, a factor is not worth showing as a change. 1.004x on a 28-day order is a
#: rounding difference dressed up as advice.
MIN_MATERIAL_FACTOR: Final = 0.01


class State(StrEnum):
    """Which of the three outcomes a product landed in. Always present in the response."""

    ADJUSTED = "adjusted"
    ADVISORY = "advisory"
    NONE = "none"


@dataclass(frozen=True)
class Match:
    """One festival that moved this product, with everything needed to overrule it.

    Attributes:
        festival_key: Calendar key.
        festival_name: Display name.
        source: ``measured`` or ``prior`` — never inferred, always carried.
        multiplier: The run-up ratio itself, before it is spread over the horizon.
        category: The demand-table category this product was read as.
        keyword: The word in the product name that caused the match, or ``None`` when the
            ratio was measured from the product's own sales and no keyword was involved.
        days_in_window: Days of this festival's run-up that fall inside the order window.
        partial_calendar: The calendar can only date this festival in some years.
        detail: The full sentence from the uplift, including its evidence.

    """

    festival_key: str
    festival_name: str
    source: Source
    multiplier: float
    category: str
    keyword: str | None
    days_in_window: int
    partial_calendar: bool
    detail: str

    def describe(self) -> str:
        """One line naming the festival, the category and why this product was picked."""
        why = (
            f"matched '{self.keyword}' as {self.category}"
            if self.keyword
            else f"measured from this product's own sales at {self.festival_name}"
        )
        return (
            f"{self.festival_name}: {why} — {self.multiplier:.2f}x over the "
            f"{self.days_in_window} day(s) of the run-up inside this order "
            f"({self.source.value})."
        )

    def to_dict(self) -> dict:
        """JSON-safe form, for the results column and the API response."""
        return {
            "festival_key": self.festival_key,
            "festival_name": self.festival_name,
            "source": self.source.value,
            "multiplier": round(self.multiplier, 4),
            "category": self.category,
            "keyword": self.keyword,
            "days_in_window": self.days_in_window,
            "partial_calendar": self.partial_calendar,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FestivalPlan:
    """What, if anything, the calendar does to one product's order.

    ``factor`` is 1.0 in every state but :attr:`State.ADJUSTED`, and :meth:`apply` is the
    only thing that touches a quantity. Both are true by construction rather than by
    convention — see :meth:`unchanged`.
    """

    state: State = State.NONE
    factor: float = 1.0
    matches: tuple[Match, ...] = ()
    nearby: tuple[str, ...] = ()
    message: str = ""
    #: Festivals in the window that could not be resolved for this product, with the reason.
    #: Shown in the advisory state so "no pattern found" is answerable rather than blank.
    unresolved: tuple[str, ...] = ()

    @classmethod
    def unchanged(cls) -> FestivalPlan:
        """Return the do-nothing plan, which is what a caller with no calendar gets."""
        return cls()

    @property
    def adjusts(self) -> bool:
        """Whether this plan moves the order quantity at all."""
        return self.state is State.ADJUSTED and abs(self.factor - 1.0) >= MIN_MATERIAL_FACTOR

    def apply(self, order_qty: float) -> float:
        """Return the order quantity this plan recommends.

        Returns ``order_qty`` unchanged in every state except a material adjustment, which
        is the property the on/off test pins.
        """
        return float(order_qty * self.factor) if self.adjusts else float(order_qty)

    def to_dict(self) -> dict:
        """JSON-safe form, stored on the result row and returned by the API."""
        return {
            "state": self.state.value,
            "factor": round(self.factor, 4),
            "matches": [m.to_dict() for m in self.matches],
            "nearby": list(self.nearby),
            "unresolved": list(self.unresolved),
            "message": self.message,
        }


def _lead_days(festival: Festival, first: date, last: date) -> list[date]:
    """Days of ``festival``'s run-up that fall inside ``[first, last]``.

    The run-up is ``[window_start, day)`` — the festival day and its tail are excluded, for
    the reason in the module docstring: the ratio being applied was measured without them.
    """
    start = max(festival.window_start, first)
    end = min(festival.day - timedelta(days=1), last)
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _keyword_for(sku: str, festival_key: str, path: str | None) -> tuple[str | None, str]:
    """Return the keyword and category the demand table read this product as, if any.

    Looked up even when the uplift was *measured*, because "your own Diwali sales" is the
    better number and "we think this is a dairy product" is still the fact the user needs in
    order to disagree.
    """
    found = match(sku, festival_key, path=path)
    return (found.keyword, found.row.category) if found else (None, "")


def plan_for(
    sku: str,
    series: pd.Series,
    *,
    horizon: int,
    region: str | None = DEFAULT_REGION,
    path: str | None = None,
) -> FestivalPlan:
    """Decide which of the three states this product is in, and by how much.

    Args:
        sku: Product identifier, used for keyword matching as well as reporting.
        series: Daily units, date-indexed. The last observation dates the order window.
        horizon: Days the order must cover, starting the day after the last observation.
        region: Festival calendar, or ``None`` for no calendar at all — which returns
            :meth:`FestivalPlan.unchanged` and is what makes the feature switchable off.
        path: Override the reference table, for tests.

    Returns:
        A :class:`FestivalPlan`. Never raises for a series the pipeline accepted; a
        calendar that cannot speak about these dates is an unchanged plan, not an error.

    """
    observed = series.dropna()
    if region is None or observed.empty or horizon < 1:
        return FestivalPlan.unchanged()

    last = observed.index.max().date()
    first_day = last + timedelta(days=1)
    last_day = last + timedelta(days=horizon)

    try:
        in_horizon = windows_in(first_day, last_day, region)
        nearby = windows_in(first_day, last_day + timedelta(days=LOOKAHEAD_DAYS), region)
    except KeyError:
        # An unknown region is a configuration problem, not this product's problem. It has
        # already been reported by anything that asked the calendar a direct question.
        return FestivalPlan.unchanged()

    if not nearby:
        # Two different facts, and only one of them is "nothing is coming". Past the last
        # year the calendar holds, every product would report `none` for ever -- the exact
        # silent failure this module's docstring says the design forbids, and the kind that
        # gets quieter as it gets more wrong.
        if not covers(last_day, region):
            first_known, last_known = coverage(region)
            return replace(
                FestivalPlan.unchanged(),
                unresolved=(
                    f"our festival calendar runs {first_known.isoformat()} to "
                    f"{last_known.isoformat()} and cannot speak about "
                    f"{last_day.isoformat()}; no festival adjustment was considered",
                ),
                message=(
                    "No festival adjustment was considered for this order window, because "
                    f"our calendar ends on {last_known.isoformat()}. That is a gap in what "
                    "we know, not a statement that nothing is coming."
                ),
            )
        return FestivalPlan.unchanged()

    nearby_names = tuple(dict.fromkeys(f.name for f in nearby))
    horizon_days = [first_day + timedelta(days=i) for i in range(horizon)]

    matches: dict[str, Match] = {}
    unresolved: list[str] = []

    for key in dict.fromkeys(f.key for f in in_horizon):
        days = sorted(
            {d for f in in_horizon if f.key == key for d in _lead_days(f, first_day, last_day)}
        )
        if not days:
            # In the window only by its tail: nothing left to order against.
            continue

        uplift = resolve(sku, observed, key, region=region, path=path)
        if not uplift.available:
            unresolved.append(f"{uplift.festival_name}: {uplift.reason}")
            continue

        keyword, category = _keyword_for(sku, key, path)
        matches[key] = Match(
            festival_key=key,
            festival_name=uplift.festival_name,
            source=uplift.source,
            multiplier=uplift.multiplier,
            category=category or _category_of(uplift),
            keyword=keyword,
            days_in_window=len(days),
            partial_calendar=uplift.partial_calendar,
            detail=uplift.describe(),
        )

    if not matches:
        return FestivalPlan(
            state=State.ADVISORY,
            factor=1.0,
            nearby=nearby_names,
            unresolved=tuple(unresolved),
            message=_advisory_message(nearby_names),
        )

    plan = FestivalPlan(
        state=State.ADJUSTED,
        factor=_factor(horizon_days, in_horizon, matches),
        matches=tuple(matches.values()),
        nearby=nearby_names,
        unresolved=tuple(unresolved),
    )
    # The message depends on the finished factor, so it is filled in on a copy rather than
    # recomputed from loose parts that could drift from the object being described.
    return replace(plan, message=_adjusted_message(plan))


def _category_of(uplift: Uplift) -> str:
    """Fallback category label for a measured uplift with no keyword behind it."""
    return "this product's own history" if uplift.source is Source.MEASURED else ""


def _factor(
    horizon_days: list[date], in_horizon: list[Festival], matches: dict[str, Match]
) -> float:
    """Mean per-day multiplier over the order window.

    One day is owned by at most one festival. Where two run-ups overlap the nearer wins, by
    the calendar's own tie-break, because a shopper on that day is buying for one of them —
    multiplying 2.0x by 1.6x would order for a festival that does not exist.
    """
    total = 0.0
    for day in horizon_days:
        owners = [f for f in in_horizon if f.key in matches and f.window_start <= day < f.day]
        if not owners:
            total += 1.0
            continue
        total += matches[min(owners, key=lambda f: tie_break(f, day)).key].multiplier
    return total / len(horizon_days)


def _advisory_message(nearby: tuple[str, ...]) -> str:
    """Build the awareness banner, which says the quantity did not move in those words."""
    which = _join(nearby)
    return (
        f"{which} {'falls' if len(nearby) == 1 else 'fall'} in or near this order window, but "
        "no festival pattern was found for this item — its name matches none of the "
        "categories we have figures for, and there is nothing measurable in its own sales. "
        "The order quantity below is unchanged."
    )


def _adjusted_message(plan: FestivalPlan) -> str:
    """Build the confirmation banner, naming what matched and what it did to the number."""
    direction = "raised" if plan.factor > 1 else "lowered"
    change = abs(plan.factor - 1.0) * 100
    lines = " ".join(m.describe() for m in plan.matches)
    if not plan.adjusts:
        return (
            f"{lines} Spread over this order window that comes to no material change, so "
            "the order quantity below is as forecast."
        )
    return (
        f"{lines} Spread over the order window, that {direction} the order by "
        f"{change:.0f}%. Check the match: if this product is not what we read it as, the "
        "adjustment is wrong and the unadjusted quantity is the one to use."
    )


def _join(names: tuple[str, ...]) -> str:
    """Join names for a sentence rather than a log line: "A", "A and B", "A, B and C"."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"
