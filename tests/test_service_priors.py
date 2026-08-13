"""The reference demand table, and the rule that measurement always outranks it."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from inventory_engine.service.festivals import SOURCES
from inventory_engine.service.priors import (
    MIN_KEYWORD_CHARS,
    coverage_report,
    load,
    match,
    prior_uplift,
    resolve,
)
from inventory_engine.service.uplift import Source

TABLE = """festival_key,festival,direction,category,items,suggested_multiplier,notes
diwali,Diwali,up,"sweets, dairy","milk, khoya, paneer, dry fruits, ghee",2.0,Biggest event.
diwali,Diwali,up,lighting,"diya, diyas, string lights",2.5,Non-food.
maha_shivaratri,Maha Shivratri,up,"fasting foods","sabudana, kuttu, sendha namak",1.5,Fast.
maha_shivaratri,Maha Shivratri,down,"non-veg, alliums","onion, garlic, chicken",0.8,Avoidance.
holi,Holi,up,"dairy","milk, khoya, paneer",1.6,Dairy spike.
"""

#: Keywords that would mean the row is about groceries. The two civic holidays must not
#: carry any of them -- see ``test_the_civic_holidays_only_claim_their_own_categories``.
FOOD_STAPLES = frozenset(
    {
        "milk",
        "khoya",
        "paneer",
        "ghee",
        "butter",
        "curd",
        "cream",
        "atta",
        "flour",
        "maida",
        "rice",
        "wheat",
        "sugar",
        "oil",
        "dal",
        "besan",
        "sabudana",
        "meat",
        "mutton",
        "chicken",
        "fish",
        "eggs",
        "onion",
        "garlic",
        "vegetables",
        "dry fruits",
        "sweets",
        "mithai",
    }
)


@pytest.fixture
def table(tmp_path):
    """A small reference table on disk, so tests do not depend on the shipped one."""
    path = tmp_path / "demand.csv"
    path.write_text(TABLE, encoding="utf-8")
    load.cache_clear()
    yield str(path)
    load.cache_clear()


def series_over(start: str, days: int, level: float = 10.0) -> pd.Series:
    """A flat daily series."""
    index = pd.date_range(start, periods=days, freq="D")
    return pd.Series(np.full(days, level), index=index)


# --------------------------------------------------------------------------- loading


def test_the_shipped_table_loads_and_every_row_maps_to_a_dated_festival():
    """The real file, not a fixture -- a key with no dates could never be shown."""
    load.cache_clear()
    rows = load()
    assert rows
    assert {r.direction for r in rows} <= {"up", "down"}
    assert any(r.suppresses for r in rows), "the table should express at least one suppression"


# ----------------------------------------------------------- the shipped table, checked
#
# The cross-check. ``load()`` already refuses a row whose festival has no dates; these go
# the other way and ask whether the eight festivals we ship dates for are actually
# described. A calendar entry with no row is a festival the service can see coming and has
# nothing to say about, which is the same silence as not having it at all.


def test_every_festival_in_the_calendar_has_at_least_one_row():
    load.cache_clear()
    described = {r.festival_key for r in load()}
    missing = sorted(set(SOURCES) - described)
    assert not missing, f"in the calendar with nothing to suggest for: {missing}"


def test_the_table_describes_nothing_the_calendar_cannot_date():
    """The other direction, stated as a set rather than trusted to load()'s per-row check."""
    load.cache_clear()
    assert {r.festival_key for r in load()} <= set(SOURCES)


def test_the_two_big_festivals_carry_non_food_categories():
    """Diwali is diyas and Holi is gulal.

    A festival table that only knows about groceries can compute a number for milk and has
    nothing to say about the products people actually buy *because* of the festival. This
    is the assertion that the table is about Diwali rather than about dairy in October.
    """
    load.cache_clear()
    rows = load()
    for key, expected in {
        "diwali": {"diya", "string lights", "gift box"},
        "holi": {"gulal", "water gun", "white kurta"},
    }.items():
        keywords = {k for r in rows if r.festival_key == key for k in r.keywords}
        assert expected <= keywords, f"{key} is missing {sorted(expected - keywords)}"


def test_the_civic_holidays_only_claim_their_own_categories():
    """Independence Day and Republic Day get snacks, soft drinks and flags. Not atta.

    The failure this prevents is quiet and wide: one food-staple keyword on a civic holiday
    would attach a multiplier to most of a grocery catalogue every 15 August, on the
    strength of a long weekend.
    """
    load.cache_clear()
    for key in ("independence_day", "republic_day"):
        rows = [r for r in load() if r.festival_key == key]
        assert rows, f"{key} has no rows"
        keywords = {k for r in rows if r for k in r.keywords}
        assert not keywords & FOOD_STAPLES, f"{key} claims staples: {keywords & FOOD_STAPLES}"
        assert {"flag", "soft drink"} <= keywords


def test_the_civic_holidays_suggest_less_than_the_religious_ones():
    """ "Smaller multipliers" is the decision; this is where it is written down."""
    load.cache_clear()
    rows = load()
    civic = max(
        r.multiplier for r in rows if r.festival_key in {"independence_day", "republic_day"}
    )
    big = max(r.multiplier for r in rows if r.festival_key in {"diwali", "holi"})
    assert civic < big


def test_no_row_suggests_a_multiplier_that_is_not_a_multiplier():
    load.cache_clear()
    for row in load():
        assert 0.0 < row.multiplier < 5.0, f"{row.festival_key}/{row.category}: {row.multiplier}"
        assert (row.multiplier < 1.0) == row.suppresses, (
            f"{row.festival_key}/{row.category} is marked {row.direction} at {row.multiplier}"
        )


def test_every_row_carries_a_note_a_buyer_can_weigh():
    """The multiplier alone does not say whether being wrong costs waste or shelf space."""
    load.cache_clear()
    for row in load():
        assert len(row.notes) > 20, f"{row.festival_key}/{row.category} has no usable note"


def test_a_festival_key_with_no_calendar_dates_is_rejected_loudly(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "festival_key,festival,direction,category,items,suggested_multiplier,notes\n"
        "ganesh_chaturthi,Ganesh Chaturthi,up,dairy,milk,1.5,no dates for this\n",
        encoding="utf-8",
    )
    load.cache_clear()
    with pytest.raises(ValueError, match="has no dates in the calendar"):
        load(str(bad))
    load.cache_clear()


def test_an_unknown_direction_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "festival_key,festival,direction,category,items,suggested_multiplier,notes\n"
        "diwali,Diwali,sideways,dairy,milk,1.5,nonsense\n",
        encoding="utf-8",
    )
    load.cache_clear()
    with pytest.raises(ValueError, match="direction must be"):
        load(str(bad))
    load.cache_clear()


def test_a_missing_table_fails_rather_than_degrading():
    load.cache_clear()
    with pytest.raises(FileNotFoundError):
        load("no/such/table.csv")
    load.cache_clear()


def test_very_short_keywords_are_discarded():
    """Two-letter fragments would match half a catalogue."""
    load.cache_clear()
    for row in load():
        for keyword in row.keywords:
            assert len(keyword) >= MIN_KEYWORD_CHARS


# --------------------------------------------------------------------------- matching


def test_a_recognisable_product_matches(table):
    found = match("Amul Paneer 200g", "diwali", path=table)
    assert found is not None and found.keyword == "paneer"
    assert found.multiplier == 2.0


def test_an_opaque_product_code_matches_nothing(table):
    """The honest outcome for "SKU-88213" is no suggestion, not a guessed one."""
    assert match("SKU-88213", "diwali", path=table) is None
    assert match("AT-1L-BLU", "diwali", path=table) is None


def test_a_multi_word_keyword_needs_the_words_adjacent(table):
    """ "dry fruits" should match the product, not any sentence containing both words."""
    assert match("Haldiram Dry Fruits Mix 500g", "diwali", path=table) is not None
    assert match("Dry roasted peanuts with fruits", "diwali", path=table) is None


def test_matching_is_case_and_punctuation_insensitive(table):
    for name in ("PANEER", "paneer.", "Fresh-Paneer_200G"):
        assert match(name, "diwali", path=table) is not None


def test_a_keyword_must_be_a_whole_word(table):
    """Substring matching would classify anything containing the letters."""
    assert match("Milkshake powder", "diwali", path=table) is None
    assert match("Milk 1L", "diwali", path=table) is not None


def test_the_festival_scopes_the_match(table):
    """Sabudana moves at Maha Shivratri and is nothing special at Holi."""
    assert match("Sabudana 500g", "maha_shivaratri", path=table) is not None
    assert match("Sabudana 500g", "holi", path=table) is None


# --------------------------------------------------------------------------- direction


def test_a_suppressed_category_gets_a_multiplier_below_one(table):
    """Maha Shivratri raises fasting foods and suppresses onion; the table says both."""
    found = match("Red Onion 1kg", "maha_shivaratri", path=table)
    assert found is not None
    assert found.row.suppresses
    assert found.multiplier == 0.8


def test_the_same_festival_can_lift_one_product_and_suppress_another(table):
    up = match("Sabudana 500g", "maha_shivaratri", path=table)
    down = match("Garlic 250g", "maha_shivaratri", path=table)
    assert up.multiplier > 1.0 > down.multiplier


def test_an_ambiguous_tie_returns_nothing_rather_than_a_coin_flip(tmp_path):
    """Two equally specific keywords disagreeing about direction is not a decision."""
    path = tmp_path / "tie.csv"
    path.write_text(
        "festival_key,festival,direction,category,items,suggested_multiplier,notes\n"
        "diwali,Diwali,up,a,ghee,2.0,up row\n"
        "diwali,Diwali,down,b,ghee,0.5,down row\n",
        encoding="utf-8",
    )
    load.cache_clear()
    assert match("Pure Ghee 1L", "diwali", path=str(path)) is None
    load.cache_clear()


def test_the_more_specific_keyword_wins(tmp_path):
    """ "onion masala" is a spice mix, not an onion."""
    path = tmp_path / "spec.csv"
    path.write_text(
        "festival_key,festival,direction,category,items,suggested_multiplier,notes\n"
        "diwali,Diwali,down,alliums,onion,0.6,plain onion\n"
        "diwali,Diwali,up,spices,onion masala,1.4,spice mix\n",
        encoding="utf-8",
    )
    load.cache_clear()
    found = match("Everest Onion Masala 100g", "diwali", path=str(path))
    assert found is not None and found.keyword == "onion masala"
    assert found.multiplier == 1.4
    load.cache_clear()


# --------------------------------------------------------------------------- provenance


def test_a_prior_uplift_is_labelled_as_a_suggestion(table):
    u = prior_uplift("Amul Paneer 200g", "diwali", path=table)
    assert u.source is Source.PRIOR
    assert u.multiplier == 2.0
    text = u.describe()
    assert "suggested" in text and "not measured from your sales" in text
    assert "your sales ran" not in text


def test_the_matched_keyword_is_carried_so_a_buyer_can_reject_it(table):
    """ "Milk Bikis" is a biscuit. The user has to be able to see why it was classified."""
    u = prior_uplift("Milk Bikis Biscuits", "diwali", path=table)
    assert u.source is Source.PRIOR
    assert "milk" in u.prior_reason


def test_an_unmatched_product_gets_no_prior_and_says_why(table):
    u = prior_uplift("SKU-88213", "diwali", path=table)
    assert not u.available
    assert "does not contain any of the items" in u.reason


# --------------------------------------------------------------------------- resolve


def test_a_short_history_falls_back_to_the_prior(table):
    """The 90-day shop gets a labelled suggestion rather than nothing at all."""
    short = series_over("2025-01-01", 120)
    u = resolve("Amul Paneer 200g", short, "diwali", path=table)
    assert u.source is Source.PRIOR
    assert u.multiplier == 2.0


def test_a_long_history_uses_the_measurement_and_ignores_the_prior(table):
    """A shop's own sales always outrank a table someone wrote down."""
    series = series_over("2023-06-01", 900)
    for day in (date(2023, 11, 12), date(2024, 11, 1), date(2025, 10, 20)):
        window = pd.date_range(day - timedelta(days=14), day - timedelta(days=1), freq="D")
        series.loc[series.index.isin(window)] *= 1.25

    u = resolve("Amul Paneer 200g", series, "diwali", path=table)
    assert u.source is Source.MEASURED
    assert u.multiplier == pytest.approx(1.25, rel=0.05)
    assert u.multiplier != 2.0  # the prior did not leak in


def test_an_unmatched_product_with_no_history_is_simply_unavailable(table):
    u = resolve("SKU-88213", series_over("2025-01-01", 120), "diwali", path=table)
    assert u.source is Source.UNAVAILABLE
    assert not u.available


# --------------------------------------------------------------------------- coverage


def test_coverage_reports_how_much_of_a_catalogue_the_table_can_speak_about(table):
    """ "we can suggest for 2 of your 4 products" is the honest headline."""
    report = coverage_report(
        ["Amul Paneer 200g", "Milk 1L", "SKU-88213", "AT-1L-BLU"], "diwali", path=table
    )
    assert report["n_skus"] == 4
    assert report["n_matched"] == 2
    assert set(report["unmatched"]) == {"SKU-88213", "AT-1L-BLU"}
    assert report["matched"]["Amul Paneer 200g"] == "paneer"
