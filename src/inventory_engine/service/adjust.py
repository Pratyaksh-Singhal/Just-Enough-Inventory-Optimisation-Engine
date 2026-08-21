"""What the festival calendar does to one product's order quantity, and what it says about it."""

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

#: Days past the end of the order window still worth mentioning.
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
    """One festival that moved this product, with everything needed to overrule it."""

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
    """What, if anything, the calendar does to one product's order."""

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
        """Return the order quantity this plan recommends."""
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
    """Days of ``festival``'s run-up that fall inside ``[first, last]``."""
    start = max(festival.window_start, first)
    end = min(festival.day - timedelta(days=1), last)
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _keyword_for(sku: str, festival_key: str, path: str | None) -> tuple[str | None, str]:
    """Return the keyword and category the demand table read this product as, if any."""
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
    """Decide which of the three states this product is in, and by how much."""
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

    # Checked before `nearby` is consulted. Inside `if not nearby` a single festival at the
    # very edge suppressed the warning -- and that straddling window is the case that most
    # needs it, because _factor counts every uncoverable day as 1.0 and quietly dilutes.
    gap_note = ()
    gap_message = ""
    if not covers(last_day, region):
        first_known, last_known = coverage(region)
        gap_note = (
            f"our festival calendar runs {first_known.isoformat()} to "
            f"{last_known.isoformat()} and cannot speak about {last_day.isoformat()}",
        )
        gap_message = (
            f"Part of this order window falls past {last_known.isoformat()}, where our "
            "festival calendar ends. That is a gap in what we know, not a statement that "
            "nothing is coming."
        )

    if not nearby:
        if gap_note:
            return replace(FestivalPlan.unchanged(), unresolved=gap_note, message=gap_message)
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
            unresolved=tuple(unresolved) + gap_note,
            message=_advisory_message(nearby_names),
        )

    plan = FestivalPlan(
        state=State.ADJUSTED,
        factor=_factor(horizon_days, in_horizon, matches),
        matches=tuple(matches.values()),
        nearby=nearby_names,
        unresolved=tuple(unresolved) + gap_note,
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
    """Mean per-day multiplier over the order window."""
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
