"""The festival calendar — dates sourced from the ``holidays`` library, not typed by hand.

Why not a hand-maintained table
-------------------------------
The first version of this module was one. It worked, and it was the wrong shape: measuring a
festival's effect needs five or six years of *historical* dates per festival, and most Indian
festivals follow lunar calendars, so every one of those dates is an independent opportunity
to type something wrong. A wrong Diwali date raises no error. It measures the wrong fortnight,
divides by the wrong baseline, and hands back a confident ratio for a week that was never a
festival at all.

So dates come from ``holidays.India``, which is maintained, versioned, and wrong in public
rather than wrong in this file.

Eight festivals, chosen rather than collected
---------------------------------------------
:data:`SOURCES` is deliberately small. The library offers a few dozen dates and an earlier
version of this file took most of them, on the reasoning that more calendar is more signal.
It is not: every extra festival is another set of windows a product can be matched against,
another chance to attach a suggestion to a product that has nothing to do with it, and one
more row of a reference table nobody has checked. The eight here are the ones with a demand
effect large enough for a grocer to order against.

Six of them carry a full treatment -- measured against the shop's own history where the
history reaches, a labelled reference figure where it does not. Two, Independence Day and
Republic Day, are :attr:`Source.prior_only`: their effect is small, narrow, and confined to
snacks, cold drinks and flags, and there is nothing to be gained from measuring a 1.1x on a
new user's three months of data. :data:`OUT_OF_SCOPE` records what was dropped and that it
is available to re-add, because "we removed it" and "the library does not have it" are
different facts.

Regional, not national
----------------------
All eight are observed nationwide. :data:`SOURCES` still records which subdivision each is
read from and whether the *observance* is regional, because that is a property of the
festival rather than of the current shortlist -- Raksha Bandhan falls on one lunar date
across India and the library merely happens to file it under HR.

What is deliberately absent, and what is only partly there
----------------------------------------------------------
:data:`UNAVAILABLE` names festivals wanted by the demand table that this library does not
provide. Neither those nor anything else is substituted with a guessed date. A festival
missing from the calendar produces "we have no dates for this" -- which is true and fixable
-- rather than a plausible-looking ratio measured over an invented window.

``Ganesh Chaturthi`` is the instructive one. It was in the hand-typed table, it is in the
demand reference, it is a major Maharashtra event, and the library does not have it under any
subdivision. Checking rather than assuming is the only reason it is not still in here with
five dates from memory.

:data:`PARTIAL` is the middle case, and Maha Shivratri is the only entry. The library dates
it in four of the nine years this calendar spans, and it is wanted in the shipped set
anyway. Partial coverage was previously treated as disqualifying, for a good reason: a
festival measurable in some years and invisible in others produces a per-SKU history whose
coverage silently depends on which years the user uploaded. The word doing the work there is
*silently*. So it is admitted and marked: :attr:`Source.partial` is carried through to the
uplift, the missing years are listed in :data:`PARTIAL`, and both the coverage test and the
user-facing text say the calendar is incomplete for it. What is forbidden is the quiet
version, not the festival.

Windows, not days
-----------------
A festival is not a spike on one day: stocking up starts before it and demand resumes after.
Each entry carries ``lead_days`` and ``tail_days``, and those are **assumptions about
shopping behaviour**, the same status as the cost rates in ``optimize/costs.py``. What gets
measured is the uplift itself, per SKU, in :mod:`inventory_engine.service.uplift`.

Why the target date's festival proximity is not leakage
-------------------------------------------------------
Every other feature in ``features.py`` is measured at the *origin*, because sales after it
are unknown. Calendar features are the documented exception, for the same reason tier 1's
``dim_calendar`` extends past the end of its sales data: event calendars are genuinely known
ahead of time. Knowing the target date is three days before Diwali needs nothing the buyer
would not have when placing the order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Final


@dataclass(frozen=True)
class Festival:
    """One dated occurrence, with the window over which it moves demand."""

    key: str
    name: str
    region: str
    day: date
    lead_days: int
    tail_days: int

    @property
    def window_start(self) -> date:
        """First date in this festival's demand window."""
        return self.day - timedelta(days=self.lead_days)

    @property
    def window_end(self) -> date:
        """Last date in this festival's demand window."""
        return self.day + timedelta(days=self.tail_days)

    def covers(self, when: date) -> bool:
        """Whether ``when`` falls inside the window."""
        return self.window_start <= when <= self.window_end


@dataclass(frozen=True)
class Source:
    """How one festival's dates are obtained from the ``holidays`` library.

    Attributes:
        name: Display name used in responses.
        library_name: Exact string the library reports, which is matched as a substring
            because the library appends "(estimated)" to Islamic dates it has projected
            rather than confirmed.
        subdiv: Subdivision to read from, or ``None`` for the national calendar.
        regional: Whether the *observance* is regional. ``False`` with a subdivision set
            means the date is nationwide and the subdivision is only where the library
            files it.
        lead_days: Assumed days of build-up before the date.
        tail_days: Assumed days of after-effect.
        partial: The library dates this festival in only some of :data:`YEARS`. Admitted
            anyway, but never quietly -- see :data:`PARTIAL` and the module docstring.
        prior_only: Never measured from a shop's own sales, only ever suggested from the
            reference table. For the two civic holidays, whose effect is small and narrow
            enough that a measurement off a new user's history would be noise with a
            decimal point.

    """

    name: str
    library_name: str
    subdiv: str | None
    regional: bool
    lead_days: int
    tail_days: int
    partial: bool = False
    prior_only: bool = False


#: The eight festivals this service models, read from ``holidays.India``. Verified present
#: with complete year coverage over :data:`YEARS` -- ``tests/test_service_festivals.py``
#: re-checks that on every run, so a library upgrade that drops one fails the build instead
#: of silently shrinking the calendar. The one exception is marked ``partial`` and is
#: checked against :data:`PARTIAL` instead.
SOURCES: Final[dict[str, Source]] = {
    "diwali": Source("Diwali", "Diwali (Deepavali)", None, False, 14, 2),
    "holi": Source("Holi", "Holi", None, False, 7, 1),
    # Dated by the library in 2019, 2022, 2025 and 2027 only. Admitted with the gap
    # recorded and reported rather than dropped -- see PARTIAL and the module docstring.
    "maha_shivaratri": Source("Maha Shivratri", "Maha Shivaratri", None, False, 5, 1, True),
    "raksha_bandhan": Source("Raksha Bandhan", "Raksha Bandhan", "HR", False, 5, 1),
    "eid_al_fitr": Source("Eid al-Fitr", "Eid al-Fitr", None, False, 7, 1),
    "christmas": Source("Christmas", "Christmas", None, False, 10, 1),
    # The two civic holidays. Fixed Gregorian dates, so no lunar drift to get wrong, and
    # prior-only: a short run-up on a narrow set of categories, never measured.
    "independence_day": Source(
        "Independence Day", "Independence Day", None, False, 3, 0, prior_only=True
    ),
    "republic_day": Source("Republic Day", "Republic Day", None, False, 2, 0, prior_only=True),
}

#: In :data:`SOURCES` but dated by the library in only some of :data:`YEARS`. The value is
#: the years it *does* have, pinned so that a library release which fills the gap -- or
#: widens it -- fails the test rather than silently changing what gets measured.
PARTIAL: Final[dict[str, tuple[int, ...]]] = {
    "maha_shivaratri": (2019, 2022, 2025, 2027),
}

#: Present in ``holidays.India`` with full coverage and deliberately not modelled. Removing
#: a festival and being unable to date one are different problems with different fixes, so
#: they are recorded separately: everything here is a one-line addition to :data:`SOURCES`
#: plus a row in the demand table, whenever there is a reason to want it.
OUT_OF_SCOPE: Final[dict[str, str]] = {
    "dussehra": "dated nationally; dropped to keep the modelled set to eight.",
    "navratri_sharad": (
        "dated via Mahanavami (JK). The nine-day fast is a real and large retail effect; "
        "dropped for scope, and it is the first candidate to add back."
    ),
    "navratri_chaitra": "dated via '1st Navratra' (JK); the smaller spring fast.",
    "janmashtami": "dated nationally; dairy-centric.",
    "eid_al_adha": "dated nationally; meat-dominant.",
    "guru_nanak": "dated nationally; langar-driven bulk demand.",
    "onam": "Kerala only (KL).",
    "pongal": "Tamil Nadu only (TN).",
    "chhath": "Bihar only (BR).",
    "ugadi": "Andhra Pradesh only (AP).",
    "gudi_padwa": "Maharashtra only (MH).",
}

#: Wanted by ``data/india_festival_demand.csv``, absent from ``holidays.India`` under every
#: subdivision. Searched for by name substring across all 36 subdivisions -- see the test.
#: Left out rather than guessed; the value here is the reason, which is what tells a
#: maintainer what to go and find.
UNAVAILABLE: Final[dict[str, str]] = {
    "ganesh_chaturthi": (
        "not in holidays.India under any subdivision, despite being a major Maharashtra "
        "festival. Needs a Hindu-calendar source (e.g. drikpanchang) to add."
    ),
    "durga_puja": (
        "not listed. The library has Dussehra nationally, which is the same culminating day "
        "(Vijayadashami), but not the preceding Durga Puja days that carry the Bengal "
        "sweets and fish demand."
    ),
    "karwa_chauth": "not listed; it is not a public holiday in any state.",
    "new_year_eve": (
        "not listed as a holiday. The library has Parsi and Tamil new years, which are "
        "different events on different dates."
    ),
}

#: Years the calendar is built for. Six historical years is what makes a measured uplift
#: rest on more than one or two observations.
YEARS: Final = tuple(range(2019, 2028))

#: A festival must have a date in every one of :data:`YEARS` to be admitted, unless it is
#: marked :attr:`Source.partial` and its years are pinned in :data:`PARTIAL`.
MIN_YEARS: Final = len(YEARS)

DEFAULT_REGION: Final = "IN"

#: Beyond this many days, "the next festival" stops being a feature and starts being a proxy
#: for the time of year. Distances are clipped here.
HORIZON_DAYS: Final = 45

#: US dates, retained because the mechanism is validated against the M5 panel this project
#: ships and Indian festivals show nothing in 2011-2016 US grocery data. Hand-typed, but
#: cross-checked against tier 1's ``dim_calendar`` by the test suite -- verified against real
#: data rather than memory, which is the standard the Indian table could not meet.
US_DATES: Final[dict[str, tuple[str, ...]]] = {
    "thanksgiving": (
        "2011-11-24",
        "2012-11-22",
        "2013-11-28",
        "2014-11-27",
        "2015-11-26",
        "2016-11-24",
    ),
    "christmas": (
        "2011-12-25",
        "2012-12-25",
        "2013-12-25",
        "2014-12-25",
        "2015-12-25",
        "2016-12-25",
    ),
    "super_bowl": (
        "2011-02-06",
        "2012-02-05",
        "2013-02-03",
        "2014-02-02",
        "2015-02-01",
        "2016-02-07",
    ),
}

US_PROFILES: Final[dict[str, tuple[str, int, int]]] = {
    "thanksgiving": ("Thanksgiving", 7, 1),
    "christmas": ("Christmas", 10, 1),
    "super_bowl": ("Super Bowl", 3, 0),
}


@lru_cache(maxsize=8)
def _subdivision_dates(subdiv: str | None) -> dict[str, tuple[date, ...]]:
    """Every holiday name in one subdivision, mapped to its dates over :data:`YEARS`.

    Cached because building a subdivision costs real time and the calendar is read once per
    SKU per forecast.
    """
    import holidays

    table = holidays.India(subdiv=subdiv, years=YEARS) if subdiv else holidays.India(years=YEARS)
    out: dict[str, list[date]] = {}
    for day, label in table.items():
        for name in label.split("; "):
            out.setdefault(name, []).append(day)
    return {name: tuple(sorted(days)) for name, days in out.items()}


def _dates_for(source: Source) -> tuple[date, ...]:
    """Dates for one source, matching the library name as a prefix.

    Prefix rather than equality because the library suffixes projected Islamic dates with
    "(estimated)" -- ``Eid al-Fitr`` and ``Eid al-Fitr (estimated)`` are the same festival
    and both are wanted.
    """
    table = _subdivision_dates(source.subdiv)
    days: set[date] = set()
    for name, found in table.items():
        if name == source.library_name or name.startswith(source.library_name + " ("):
            days.update(found)
    return tuple(sorted(days))


@lru_cache(maxsize=4)
def festivals(region: str = DEFAULT_REGION) -> tuple[Festival, ...]:
    """Every known occurrence for ``region``, ascending by date.

    Raises:
        KeyError: For a region with no calendar, rather than returning an empty tuple that
            would read as "a year with no festivals".

    """
    if region == "US":
        return tuple(
            sorted(
                (
                    Festival(
                        key,
                        US_PROFILES[key][0],
                        "US",
                        date.fromisoformat(iso),
                        US_PROFILES[key][1],
                        US_PROFILES[key][2],
                    )
                    for key, isos in US_DATES.items()
                    for iso in isos
                ),
                key=lambda f: f.day,
            )
        )
    if region != "IN":
        raise KeyError(f"no festival calendar for region {region!r}; have 'IN', 'US'")

    out = [
        Festival(key, source.name, "IN", day, source.lead_days, source.tail_days)
        for key, source in SOURCES.items()
        for day in _dates_for(source)
    ]
    return tuple(sorted(out, key=lambda f: f.day))


def coverage(region: str = DEFAULT_REGION) -> tuple[date, date]:
    """First and last date the calendar knows about, so a caller can say when it runs out."""
    known = festivals(region)
    return known[0].day, known[-1].day


def covers(when: date, region: str = DEFAULT_REGION) -> bool:
    """Whether the calendar can speak about ``when`` at all."""
    first, last = coverage(region)
    return first <= when <= last


def gaps() -> dict[str, str]:
    """Everything the calendar cannot fully speak about, and why.

    Two kinds, reported together because a caller asking "what are the holes" wants both:
    festivals with no dates at all (:data:`UNAVAILABLE`), and festivals dated in only some
    years (:data:`PARTIAL`). Surfaced deliberately -- "we have no dates for Ganesh
    Chaturthi" is an actionable fact, and it is the only honest alternative to a guessed
    date.
    """
    return {**UNAVAILABLE, **partial_coverage()}


def partial_coverage() -> dict[str, str]:
    """Festivals in the calendar whose year coverage has holes, and which years are missing.

    Derived from :data:`PARTIAL` rather than restated, so the sentence a user reads cannot
    drift from the tuple the test pins.
    """
    return {
        key: (
            f"holidays.India dates it in {len(years)} of the {len(YEARS)} years "
            f"{YEARS[0]}-{YEARS[-1]} (missing "
            f"{', '.join(str(y) for y in sorted(set(YEARS) - set(years)))}). It is still "
            "used, and anything measured from it rests only on the years above."
        )
        for key, years in PARTIAL.items()
    }


def is_partial(key: str) -> bool:
    """Whether this festival's dates have known year gaps."""
    return key in SOURCES and SOURCES[key].partial


def is_prior_only(key: str) -> bool:
    """Whether this festival is never measured from a shop's own sales.

    Read by :mod:`inventory_engine.service.uplift` before it measures anything, so the
    decision lives with the calendar entry rather than being re-stated at each call site.
    """
    return key in SOURCES and SOURCES[key].prior_only


def next_festival(after: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the first festival on or after ``after``, or ``None`` past the table's end."""
    return next((f for f in festivals(region) if f.day >= after), None)


def previous_festival(before: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the last festival on or before ``before``, or ``None`` before it starts."""
    return next((f for f in reversed(festivals(region)) if f.day <= before), None)


def active(when: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the festival whose window contains ``when``, if any.

    Where windows overlap -- Maha Shivratri and Holi fall a fortnight apart, and Diwali's
    fourteen-day run-up is long enough to touch a neighbour -- the nearer festival wins,
    because that is the one the shopper is buying for.

    Two festivals can land on the *same* day, at which point "nearer" decides nothing:
    Raksha Bandhan fell on 15 August in 2019. The tie goes to the festival that is not
    :attr:`Source.prior_only`, so the sweets event outranks the civic one, and to the key
    alphabetically after that. Deterministic rather than dependent on dict order, which is
    the actual requirement -- a banner that changed between two runs on the same date would
    be worse than either answer.
    """
    inside = [f for f in festivals(region) if f.covers(when)]
    return min(inside, key=lambda f: tie_break(f, when)) if inside else None


def tie_break(festival: Festival, when: date) -> tuple[int, int, str]:
    """Sort key for choosing between overlapping festivals: nearest, substantive, stable.

    Public because :mod:`inventory_engine.service.adjust` decides day by day which festival
    owns a day of the order window, and it must reach the same answer this module does.
    """
    return (
        abs((festival.day - when).days),
        int(is_prior_only(festival.key)),
        festival.key,
    )


def upcoming(on: date, within_days: int = 30, region: str = DEFAULT_REGION) -> list[Festival]:
    """Festivals whose window overlaps the next ``within_days``.

    Deliberately about the window rather than the day: a fortnight of Diwali build-up matters
    from the start of the fortnight.
    """
    horizon = on + timedelta(days=within_days)
    return [f for f in festivals(region) if on <= f.window_end and f.window_start <= horizon]


def distance(when: date, region: str = DEFAULT_REGION) -> tuple[int, int, bool]:
    """Days to the next festival, days since the last, and whether ``when`` is in a window.

    Both distances are clipped to :data:`HORIZON_DAYS`, which is also what an off-calendar
    date returns: "far from any festival" and "we have no calendar here" look the same to the
    model, and both are honestly represented by the maximum distance.
    """
    nxt = next_festival(when, region)
    prev = previous_festival(when, region)
    to_next = min((nxt.day - when).days, HORIZON_DAYS) if nxt else HORIZON_DAYS
    since_prev = min((when - prev.day).days, HORIZON_DAYS) if prev else HORIZON_DAYS
    return to_next, since_prev, active(when, region) is not None


def windows_in(first: date, last: date, region: str = DEFAULT_REGION) -> list[Festival]:
    """Every festival whose window overlaps ``[first, last]``.

    Used by the uplift measurement to find which festivals a user's history actually lived
    through -- an uplift can only be measured for a festival the data covers.
    """
    return [f for f in festivals(region) if f.window_end >= first and f.window_start <= last]
