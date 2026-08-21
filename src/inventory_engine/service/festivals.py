"""The festival calendar — dates sourced from the ``holidays`` library, not typed by hand."""

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
    """How one festival's dates are obtained from the ``holidays`` library."""

    name: str
    library_name: str
    subdiv: str | None
    regional: bool
    lead_days: int
    tail_days: int
    partial: bool = False
    prior_only: bool = False


#: The eight festivals this service models, read from ``holidays.India``.
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

#: In :data:`SOURCES` but dated by the library in only some of :data:`YEARS`.
PARTIAL: Final[dict[str, tuple[int, ...]]] = {
    "maha_shivaratri": (2019, 2022, 2025, 2027),
}

#: Present in ``holidays.India`` with full coverage and deliberately not modelled.
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

#: Wanted by ``inventory_engine/data/india_festival_demand.csv``, absent from ``holidays.India``
#: under every subdivision.
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

#: US dates, retained because the mechanism is validated against the M5 panel this project ships
#: and Indian festivals show nothing in 2011-2016 US grocery data.
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


#: The language the library is asked for its holiday *names* in, stated rather than inherited.
LIBRARY_LANGUAGE: Final = "en_US"


@lru_cache(maxsize=8)
def _subdivision_dates(subdiv: str | None) -> dict[str, tuple[date, ...]]:
    """Every holiday name in one subdivision, mapped to its dates over :data:`YEARS`."""
    import holidays

    table = holidays.India(subdiv=subdiv, years=YEARS, language=LIBRARY_LANGUAGE)
    out: dict[str, list[date]] = {}
    for day, label in table.items():
        for name in label.split("; "):
            out.setdefault(name, []).append(day)
    return {name: tuple(sorted(days)) for name, days in out.items()}


def library_names(subdiv: str | None = None) -> set[str]:
    """Every holiday name the library reports, under :data:`LIBRARY_LANGUAGE`."""
    return set(_subdivision_dates(subdiv))


def _dates_for(source: Source) -> tuple[date, ...]:
    """Dates for one source, matching the library name as a prefix."""
    table = _subdivision_dates(source.subdiv)
    days: set[date] = set()
    for name, found in table.items():
        if name == source.library_name or name.startswith(source.library_name + " ("):
            days.update(found)
    return tuple(sorted(days))


@lru_cache(maxsize=4)
def festivals(region: str = DEFAULT_REGION) -> tuple[Festival, ...]:
    """Every known occurrence for ``region``, ascending by date."""
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
    """Everything the calendar cannot fully speak about, and why."""
    return {**UNAVAILABLE, **partial_coverage()}


def partial_coverage() -> dict[str, str]:
    """Festivals in the calendar whose year coverage has holes, and which years are missing."""
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
    """Whether this festival is never measured from a shop's own sales."""
    return key in SOURCES and SOURCES[key].prior_only


def next_festival(after: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the first festival on or after ``after``, or ``None`` past the table's end."""
    return next((f for f in festivals(region) if f.day >= after), None)


def previous_festival(before: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the last festival on or before ``before``, or ``None`` before it starts."""
    return next((f for f in reversed(festivals(region)) if f.day <= before), None)


def active(when: date, region: str = DEFAULT_REGION) -> Festival | None:
    """Return the festival whose window contains ``when``, if any."""
    inside = [f for f in festivals(region) if f.covers(when)]
    return min(inside, key=lambda f: tie_break(f, when)) if inside else None


def tie_break(festival: Festival, when: date) -> tuple[int, int, str]:
    """Sort key for choosing between overlapping festivals: nearest, substantive, stable."""
    return (
        abs((festival.day - when).days),
        int(is_prior_only(festival.key)),
        festival.key,
    )


def upcoming(on: date, within_days: int = 30, region: str = DEFAULT_REGION) -> list[Festival]:
    """Festivals whose window overlaps the next ``within_days``."""
    horizon = on + timedelta(days=within_days)
    return [f for f in festivals(region) if on <= f.window_end and f.window_start <= horizon]


def distance(when: date, region: str = DEFAULT_REGION) -> tuple[int, int, bool]:
    """Days to the next festival, days since the last, and whether ``when`` is in a window."""
    nxt = next_festival(when, region)
    prev = previous_festival(when, region)
    to_next = min((nxt.day - when).days, HORIZON_DAYS) if nxt else HORIZON_DAYS
    since_prev = min((when - prev.day).days, HORIZON_DAYS) if prev else HORIZON_DAYS
    return to_next, since_prev, active(when, region) is not None


def windows_in(first: date, last: date, region: str = DEFAULT_REGION) -> list[Festival]:
    """Every festival whose window overlaps ``[first, last]``."""
    return [f for f in festivals(region) if f.window_end >= first and f.window_start <= last]
