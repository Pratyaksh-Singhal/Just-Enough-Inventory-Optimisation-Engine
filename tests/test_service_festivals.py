"""The festival calendar, now sourced from ``holidays.India``.

Four tests here do real work rather than restating the code:

* ``test_every_source_has_complete_year_coverage`` re-derives coverage from the library on
  every run, so an upgrade that drops or renames a festival fails the build instead of
  silently shrinking the calendar.
* ``test_the_partial_festival_has_exactly_the_years_we_say_it_has`` pins the one known hole,
  so a library release that fills it or widens it is noticed.
* ``test_named_gaps_are_genuinely_absent`` checks the "we don't have this" notes are still
  true, so they cannot rot into misinformation.
* ``test_the_us_dates_match_the_m5_calendar`` checks the one remaining hand-typed table
  against real event dates in tier 1's warehouse.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from inventory_engine.service.festivals import (
    HORIZON_DAYS,
    OUT_OF_SCOPE,
    PARTIAL,
    SOURCES,
    UNAVAILABLE,
    US_DATES,
    YEARS,
    active,
    coverage,
    covers,
    distance,
    festivals,
    gaps,
    is_prior_only,
    next_festival,
    previous_festival,
    upcoming,
    windows_in,
)

#: The shipped scope, decided rather than discovered. Written out here so that adding a
#: festival to SOURCES without deciding to is a test failure and not a surprise.
SHIPPED = {
    "diwali",
    "holi",
    "maha_shivaratri",
    "raksha_bandhan",
    "eid_al_fitr",
    "christmas",
    "independence_day",
    "republic_day",
}


def instance(key: str, year: int):
    """The dated occurrence of ``key`` in ``year``."""
    return next(f for f in festivals("IN") if f.key == key and f.day.year == year)


def full_coverage_keys() -> list[str]:
    """Festivals expected to have a date in every year."""
    return [key for key in SOURCES if key not in PARTIAL]


def library_names(subdiv: str | None = None) -> set[str]:
    """Every holiday name the library emits, for one subdivision or nationally."""
    import holidays

    table = holidays.India(subdiv=subdiv, years=YEARS) if subdiv else holidays.India(years=YEARS)
    return {n for label in table.values() for n in label.split("; ")}


def diagnosis(key: str | None = None) -> str:
    """What the installed library actually offers, for an assertion message.

    Written after a CI failure that took two wrong guesses and a pasted log to identify.
    ``SOURCES`` matches the library's *display strings*, so the whole class of failure here
    is "the name changed" -- and the one thing the failure never said was which names exist
    now, or which version was answering. Both go in the message, so the next occurrence is
    diagnosable from the log rather than from a reproduction.
    """
    import holidays

    lines = [f"installed holidays=={holidays.__version__}"]
    if key and key in SOURCES:
        source = SOURCES[key]
        lines.append(f"{key} looks for {source.library_name!r} in subdiv={source.subdiv!r}")
        found = sorted(library_names(source.subdiv))
        head = source.library_name.split()[0].lower()
        near = [n for n in found if head in n.lower()] or found[:25]
        lines.append(f"names available there: {near}")
    return "\n  ".join(lines)


# --------------------------------------------------------------------------- the scope


def test_the_calendar_is_exactly_the_eight_festivals_we_decided_on():
    """Scope is a decision, so it is asserted rather than left to whoever edits SOURCES."""
    assert set(SOURCES) == SHIPPED
    assert len(SOURCES) == 8


def test_the_two_civic_holidays_are_prior_only_and_nothing_else_is():
    """Independence and Republic Day are never measured from a shop's own sales."""
    assert {k for k in SOURCES if SOURCES[k].prior_only} == {"independence_day", "republic_day"}
    assert is_prior_only("republic_day")
    assert not is_prior_only("diwali")


def test_dropped_festivals_are_recorded_as_dropped_rather_than_forgotten():
    """ "We removed it" and "the library hasn't got it" are different facts, kept apart."""
    assert OUT_OF_SCOPE
    assert not set(OUT_OF_SCOPE) & set(SOURCES)
    assert not set(OUT_OF_SCOPE) & set(UNAVAILABLE)


def test_a_dropped_festival_is_still_available_to_re_add():
    """The out-of-scope note claims these are datable. Re-derived, or it is just a claim."""
    import holidays

    every_name: set[str] = set()
    for subdiv in [None, *sorted(holidays.India.subdivisions)]:
        table = (
            holidays.India(subdiv=subdiv, years=YEARS) if subdiv else holidays.India(years=YEARS)
        )
        for label in table.values():
            every_name.update(n.lower() for n in label.split("; "))

    for key, needle in {
        "dussehra": "dussehra",
        "janmashtami": "janmashtami",
        "onam": "onam",
        "eid_al_adha": "eid al-adha",
    }.items():
        assert key in OUT_OF_SCOPE
        assert any(needle in n for n in every_name), (
            f"{key} is no longer datable under any subdivision, so its OUT_OF_SCOPE note "
            f"is now wrong. Most likely the library renamed it.\n  {diagnosis()}"
        )


# --------------------------------------------------------------------------- the source


def test_every_source_has_complete_year_coverage():
    """A festival in the calendar must have a date in every year we claim to cover.

    Partial coverage measures some years and skips others, and which ones depends on what
    the user happened to upload -- tolerable only when it is declared, which is what PARTIAL
    is for. ``holidays`` is a moving dependency, so this re-checks rather than trusting the
    day it was written.
    """
    known = festivals("IN")
    for key in full_coverage_keys():
        years = {f.day.year for f in known if f.key == key}
        missing = sorted(set(YEARS) - years)
        assert not missing, (
            f"{key} is missing dates for {missing}. Either the library renamed it -- fix "
            f"SOURCES and the pin -- or it genuinely has gaps and belongs in PARTIAL."
            f"\n  {diagnosis(key)}"
        )


def test_the_partial_festival_has_exactly_the_years_we_say_it_has():
    """Maha Shivratri is dated in four years of nine. Pinned, so a change is visible.

    If the library starts dating it in every year, this fails and PARTIAL should be emptied
    -- the festival gets better, not the test.
    """
    known = festivals("IN")
    for key, declared in PARTIAL.items():
        assert SOURCES[key].partial, f"{key} is in PARTIAL but not flagged partial"
        assert {f.day.year for f in known if f.key == key} == set(declared)


def test_the_partial_gap_is_reported_with_the_years_it_is_missing():
    reported = gaps()["maha_shivaratri"]
    for year in (2020, 2021, 2023, 2024, 2026):
        assert str(year) in reported


def test_the_calendar_is_populated_and_sorted():
    known = festivals("IN")
    expected = len(full_coverage_keys()) * len(YEARS) + sum(len(y) for y in PARTIAL.values())
    assert len(known) >= expected, (
        f"the calendar holds {len(known)} occurrences, short of the {expected} the shipped "
        f"set implies -- a festival resolved to no dates.\n  {diagnosis()}"
    )
    assert [f.day for f in known] == sorted(f.day for f in known)


def test_named_gaps_are_genuinely_absent_from_the_library():
    """A stale "we don't have this" note is misinformation; re-derive it."""
    import holidays

    every_name: set[str] = set()
    for subdiv in [None, *sorted(holidays.India.subdivisions)]:
        table = (
            holidays.India(subdiv=subdiv, years=YEARS) if subdiv else holidays.India(years=YEARS)
        )
        for label in table.values():
            every_name.update(n.lower() for n in label.split("; "))

    for key, needle in {
        "ganesh_chaturthi": "ganesh",
        "durga_puja": "durga",
        "karwa_chauth": "karwa",
    }.items():
        assert key in UNAVAILABLE
        hits = [n for n in every_name if needle in n]
        assert not hits, f"{key} is documented as unavailable but the library now has {hits}"


def test_gaps_are_reported_with_an_actionable_reason():
    reported = gaps()
    assert set(UNAVAILABLE) | set(PARTIAL) == set(reported)
    for key, why in reported.items():
        assert len(why) > 30, f"{key} needs a reason a maintainer can act on"


# --------------------------------------------------------------------------- known dates


@pytest.mark.parametrize(
    ("key", "year", "expected"),
    [
        ("diwali", 2024, date(2024, 10, 31)),
        ("diwali", 2025, date(2025, 10, 20)),
        ("holi", 2025, date(2025, 3, 14)),
        ("maha_shivaratri", 2025, date(2025, 2, 26)),
        ("raksha_bandhan", 2025, date(2025, 8, 9)),
        ("christmas", 2025, date(2025, 12, 25)),
        ("independence_day", 2025, date(2025, 8, 15)),
        ("republic_day", 2025, date(2025, 1, 26)),
    ],
)
def test_spot_dates_are_what_the_library_reports(key, year, expected):
    """Anchors, so a wholesale library change is obvious rather than subtle."""
    assert instance(key, year).day == expected


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_the_civic_holidays_never_move(year):
    """15 August and 26 January, every year. The one thing here with no lunar drift."""
    assert instance("independence_day", year).day == date(year, 8, 15)
    assert instance("republic_day", year).day == date(year, 1, 26)


def test_every_shipped_festival_is_nationwide():
    """The shortlist is deliberately all-India: a nationwide banner is correct for each.

    The ``regional`` flag stays on Source because it is a property of a festival rather
    than of this shortlist, and re-adding Onam should not need a new field.
    """
    for key in SOURCES:
        assert not SOURCES[key].regional, f"{key} is regional; the shipped set is nationwide"


def test_estimated_islamic_dates_are_still_collected():
    """The library suffixes projected dates with "(estimated)"; both spellings are wanted."""
    years = {f.day.year for f in festivals("IN") if f.key == "eid_al_fitr"}
    assert years == set(YEARS), (
        f"Eid al-Fitr resolved to {sorted(years)}. An empty set means the library is not "
        f"naming it 'Eid al-Fitr' at all.\n  {diagnosis('eid_al_fitr')}"
    )


# --------------------------------------------------------------------------- the US oracle


@pytest.mark.parametrize("key", ["thanksgiving", "christmas", "super_bowl"])
def test_the_us_dates_match_the_m5_calendar(key):
    """The one hand-typed table left, checked against real dates in tier 1's warehouse."""
    duckdb = pytest.importorskip("duckdb")
    from inventory_engine.config import WAREHOUSE_PATH

    if not WAREHOUSE_PATH.is_file():
        pytest.skip(f"no warehouse at {WAREHOUSE_PATH}; run build-warehouse")

    m5_name = {
        "thanksgiving": "Thanksgiving",
        "christmas": "Christmas",
        "super_bowl": "SuperBowl",
    }[key]
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        real = {
            row[0]
            for row in con.execute(
                "SELECT date FROM dim_calendar WHERE event_name_1 = ?", [m5_name]
            ).fetchall()
        }
    finally:
        con.close()

    ours = {date.fromisoformat(i) for i in US_DATES[key]}
    overlap = {d.year for d in real} & {d.year for d in ours}
    assert overlap, f"no overlapping years to check for {key}"
    assert {d for d in ours if d.year in overlap} == {d for d in real if d.year in overlap}


# --------------------------------------------------------------------------- lookups


def test_next_and_previous_bracket_a_date():
    when = date(2024, 6, 1)
    assert previous_festival(when, "IN").day <= when <= next_festival(when, "IN").day


def test_the_calendar_runs_out_rather_than_inventing_a_festival():
    _, last = coverage("IN")
    assert next_festival(last + timedelta(days=1), "IN") is None
    assert covers(last, "IN")
    assert not covers(last + timedelta(days=1), "IN")


def test_a_date_inside_the_window_is_active_even_though_it_is_not_the_day():
    diwali = instance("diwali", 2025)
    assert active(diwali.day - timedelta(days=10), "IN").key == "diwali"


def test_overlapping_windows_resolve_to_the_nearer_festival():
    """On a festival's own day, the answer is a festival happening that day.

    Not necessarily *this* one: two can share a date. What must never happen is a neighbour
    whose own day is a week away winning over one that is today.
    """
    for f in festivals("IN"):
        assert active(f.day, "IN").day == f.day, f"{f.key} {f.day} lost its own day"


def test_two_festivals_on_one_day_resolve_the_same_way_every_time():
    """Raksha Bandhan fell on 15 August 2019, which is also Independence Day.

    "Nearer" decides nothing at zero distance. The tie goes to the substantive festival
    over the civic one, and it must be the same answer on every run -- a banner that
    alternated between two truths would read as a bug in the data.
    """
    collision = date(2019, 8, 15)
    assert {f.key for f in festivals("IN") if f.day == collision} == {
        "raksha_bandhan",
        "independence_day",
    }
    assert {active(collision, "IN").key for _ in range(5)} == {"raksha_bandhan"}


def test_upcoming_is_about_the_window_not_the_day():
    diwali = instance("diwali", 2025)
    on = diwali.window_start + timedelta(days=1)
    assert "diwali" in {f.key for f in upcoming(on, within_days=3, region="IN")}


def test_an_unknown_region_raises():
    with pytest.raises(KeyError, match="no festival calendar"):
        festivals("XX")


# --------------------------------------------------------------------------- distances


def test_distance_reports_both_directions_and_the_window_flag():
    to_next, since_prev, inside = distance(instance("diwali", 2025).day, "IN")
    assert to_next == 0 and since_prev == 0 and inside is True


def test_distances_are_clipped_so_they_do_not_become_a_calendar_proxy():
    to_next, since_prev, _ = distance(date(2025, 6, 15), "IN")
    assert to_next <= HORIZON_DAYS and since_prev <= HORIZON_DAYS


def test_an_off_calendar_date_reads_as_far_away():
    _, last = coverage("IN")
    to_next, _, inside = distance(last + timedelta(days=400), "IN")
    assert to_next == HORIZON_DAYS and inside is False


def test_windows_in_finds_only_festivals_the_history_lived_through():
    keys = {f.key for f in windows_in(date(2025, 10, 1), date(2025, 10, 31), "IN")}
    assert "diwali" in keys
    assert "holi" not in keys and "christmas" not in keys
