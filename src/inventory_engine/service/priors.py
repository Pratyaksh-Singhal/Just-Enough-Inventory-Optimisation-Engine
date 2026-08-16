"""The reference demand table: what to suggest when there is nothing to measure.

Where this sits
---------------
:mod:`inventory_engine.service.uplift` measures a festival's effect from a SKU's own history
and is always preferred. It needs roughly 13 months of data. This module covers the gap for
everyone else -- a shop that signed up last month, or a product introduced last week -- by
matching the SKU's name against ``inventory_engine/data/india_festival_demand.csv`` and
offering that row's multiplier as a **suggestion**.

:func:`resolve` is the only function most callers want. It tries the measurement, falls back
to the prior, and labels which one it returned. Nothing here ever silently upgrades a
suggestion into a measurement.

Why keyword matching is a suggestion and not a decision
-------------------------------------------------------
"Amul Taaza 1L" is recognisably milk. "AT-1L-BLU" and "SKU-88213" are not, and *"Milk Bikis"*
is a biscuit that matches the keyword ``milk`` perfectly while behaving nothing like dairy at
Holi. The failure is silent and expensive: double stock on the wrong product, no error
anywhere.

So every match carries the keyword that produced it. The intended presentation is
"matched **paneer** -> Holi, suggested 1.6x", which a buyer can reject at a glance. That is a
different thing from a taxonomy applied on their behalf, and it is the only form of name
inference this codebase will do.

More than groceries
-------------------
The table used to be food only, which made the arithmetic true and useless: the thing a
shopper buys *because* it is Diwali is a diya, and the thing they buy because it is Holi is
gulal. Those categories are in here now, and they behave differently from food -- a larger
multiple, because the baseline week sells almost none, and a shallower cost of being wrong,
because a leftover string of lights keeps until next year while leftover paneer does not.
The ``notes`` column says so per row, since the multiplier alone cannot.

Direction
---------
A festival is not one blanket lift. Maha Shivratri raises fasting foods and *suppresses*
onion, garlic and non-veg, so the table carries one row per category per direction and
multipliers below 1.0 are ordinary. A SKU matching both directions at the same festival is
resolved by specificity, and a genuine tie is reported as ambiguous rather than guessed --
"onion" and "onion masala" want different answers and a coin flip is not one of them.

Two festivals that are only ever suggestions
--------------------------------------------
Independence Day and Republic Day are :attr:`~inventory_engine.service.festivals.Source.
prior_only`. Their rows cover snacks, soft drinks and flags and nothing else: a civic
holiday does not move atta, and giving it a food-staple row would attach a multiplier to
every grocery in the catalogue on the strength of a long weekend.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from inventory_engine.service.festivals import SOURCES
from inventory_engine.service.uplift import Source, Uplift

#: The reference table. Data, not code, so it can be corrected without a release.
#:
#: Resolved relative to the *package*, not to a repository root. It used to be
#: ``PROJECT_ROOT / "data" / ...``, where ``PROJECT_ROOT`` is ``parents[2]`` of
#: ``config.py`` -- the repo root from a source checkout, and ``/usr/local/lib/python3.11``
#: from site-packages. The deployed service therefore looked for the table one directory
#: above the standard library and raised ``FileNotFoundError`` on every forecast. Caught by
#: the first end-to-end run against the deployment, not by any test, because every test
#: runs from a checkout where the old path happened to be right.
#:
#: Living inside the package means it ships with the wheel, which is what the sentence
#: below has always claimed.
DEFAULT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "india_festival_demand.csv"
)

#: Keywords shorter than this are not matched. "og" or "hi" inside a product code would
#: otherwise match half a catalogue.
MIN_KEYWORD_CHARS: Final = 3


@dataclass(frozen=True)
class PriorRow:
    """One row of the reference table: a festival, a direction, and the items it applies to."""

    festival_key: str
    festival_name: str
    direction: str
    category: str
    keywords: tuple[str, ...]
    multiplier: float
    notes: str

    @property
    def suppresses(self) -> bool:
        """Whether this row describes demand falling rather than rising."""
        return self.direction == "down"


@dataclass(frozen=True)
class PriorMatch:
    """A matched row, with the keyword that caused the match.

    The keyword is not decoration. It is what lets a buyer see *why* their product was
    classified, and reject it when the answer is "because 'Milk Bikis' contains 'milk'".
    """

    row: PriorRow
    keyword: str

    @property
    def multiplier(self) -> float:
        """The suggested multiplier."""
        return self.row.multiplier

    def describe(self) -> str:
        """One line naming the evidence, which is a keyword and not a measurement."""
        direction = "lower" if self.row.suppresses else "higher"
        return (
            f"matched '{self.keyword}' -> {self.row.festival_name}: reference suggests "
            f"{self.row.multiplier:.2f}x ({direction} demand, {self.row.category})"
        )


def _normalise(text: str) -> list[str]:
    """Lowercase word tokens, with punctuation and digits-with-units stripped to spaces."""
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _phrase_at(tokens: list[str], phrase: list[str]) -> bool:
    """Whether ``phrase`` appears as a contiguous run inside ``tokens``.

    Contiguous rather than "all words present anywhere": "dry fruits" should match
    *Haldiram Dry Fruits Mix* and not *dry roasted peanuts with fruit juice*.
    """
    if not phrase or len(phrase) > len(tokens):
        return False
    return any(tokens[i : i + len(phrase)] == phrase for i in range(len(tokens) - len(phrase) + 1))


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple[PriorRow, ...]:
    """Read the reference table.

    Raises:
        FileNotFoundError: If the table is missing. It ships with the repository, so its
            absence is a packaging fault worth failing on rather than degrading past.
        ValueError: If a row names a festival the calendar has no dates for, or uses a
            direction other than up/down. Both would produce a suggestion that could never
            be attached to a date, which is worse than a missing row.

    """
    source = Path(path) if path else DEFAULT_PATH
    if not source.is_file():
        raise FileNotFoundError(f"festival demand table not found at {source}")

    rows: list[PriorRow] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            key = (raw.get("festival_key") or "").strip()
            direction = (raw.get("direction") or "").strip().lower()
            if key not in SOURCES:
                raise ValueError(
                    f"{source.name} line {line_no}: festival_key {key!r} has no dates in the "
                    "calendar. Add it to festivals.SOURCES or remove the row -- a suggestion "
                    "that cannot be attached to a date can never be shown."
                )
            if direction not in {"up", "down"}:
                raise ValueError(
                    f"{source.name} line {line_no}: direction must be 'up' or 'down', "
                    f"got {direction!r}"
                )
            keywords = tuple(
                k.strip().lower()
                for k in (raw.get("items") or "").split(",")
                if len(k.strip()) >= MIN_KEYWORD_CHARS
            )
            rows.append(
                PriorRow(
                    festival_key=key,
                    festival_name=(raw.get("festival") or key).strip(),
                    direction=direction,
                    category=(raw.get("category") or "").strip(),
                    keywords=keywords,
                    multiplier=float(raw["suggested_multiplier"]),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    return tuple(rows)


def match(sku: str, festival_key: str, *, path: str | None = None) -> PriorMatch | None:
    """Best keyword match for ``sku`` at ``festival_key``, or ``None``.

    The longest matching keyword wins, because a longer phrase is a more specific claim:
    *onion masala* should not be classified by *onion*. A tie between two rows pulling in
    opposite directions is ambiguous and returns ``None`` -- the caller then has no prior,
    which is the correct answer when the table genuinely cannot tell.
    """
    tokens = _normalise(sku)
    if not tokens:
        return None

    best: list[tuple[int, PriorRow, str]] = []
    for row in load(path):
        if row.festival_key != festival_key:
            continue
        for keyword in row.keywords:
            phrase = _normalise(keyword)
            if _phrase_at(tokens, phrase):
                best.append((sum(len(p) for p in phrase), row, keyword))

    if not best:
        return None

    top = max(score for score, _, _ in best)
    winners = [(row, kw) for score, row, kw in best if score == top]
    directions = {row.direction for row, _ in winners}
    if len(directions) > 1:
        # Equally specific keywords disagreeing about whether demand rises or falls. Acting
        # on either would be a coin flip presented as advice.
        return None
    row, keyword = winners[0]
    return PriorMatch(row, keyword)


def prior_uplift(sku: str, festival_key: str, *, path: str | None = None) -> Uplift:
    """Return a prior-sourced :class:`~inventory_engine.service.uplift.Uplift`.

    Unavailable when the product name matches nothing in the table.
    """
    festival_name = SOURCES[festival_key].name if festival_key in SOURCES else festival_key
    found = match(sku, festival_key, path=path)
    if found is None:
        return Uplift(
            sku,
            festival_key,
            festival_name,
            reason=(
                "no measurement and no reference match -- the product name does not contain "
                "any of the items this festival is known to move."
            ),
        )
    return Uplift(
        sku,
        festival_key,
        festival_name,
        source=Source.PRIOR,
        prior_multiplier=found.multiplier,
        prior_reason=f"{found.describe()}. {found.row.notes}".strip(),
    )


def resolve(sku, series, festival_key, *, region: str = "IN", path: str | None = None) -> Uplift:
    """Measured uplift if the history supports one, otherwise the reference suggestion.

    The one function most callers want. The order is the whole point and is not
    configurable: a shop's own sales always outrank a table someone wrote down, and a
    suggestion is only ever a stand-in for a measurement that could not be made.

    Args:
        sku: Product identifier, used both for the result and for keyword matching.
        series: Daily units, date-indexed.
        festival_key: Which festival.
        region: Calendar region.
        path: Override the reference table, for tests.

    """
    from inventory_engine.service.uplift import measure_sku

    measured = measure_sku(sku, series, festival_key, region=region)
    if measured.source is Source.MEASURED:
        return measured
    return prior_uplift(sku, festival_key, path=path)


def coverage_report(skus: list[str], festival_key: str, *, path: str | None = None) -> dict:
    """How many of ``skus`` the reference table can speak about at all.

    Useful before showing anything: "we can suggest for 12 of your 60 products" is the
    honest headline, and it is also the number that says whether keyword matching is
    carrying its weight for this catalogue or not.
    """
    matched = {s: m for s in skus if (m := match(s, festival_key, path=path)) is not None}
    return {
        "festival_key": festival_key,
        "n_skus": len(skus),
        "n_matched": len(matched),
        "matched": {s: m.keyword for s, m in matched.items()},
        "unmatched": sorted(set(skus) - set(matched)),
    }
