"""The reference demand table: what to suggest when there is nothing to measure."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from inventory_engine.service.festivals import SOURCES
from inventory_engine.service.uplift import Source, Uplift

#: The reference table. Data, not code, so it can be corrected without a release. Resolved relative
#: to the *package*, not to a repository root. It used to be ``PROJECT_ROOT / "data" / ...``, where
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
    """A matched row, with the keyword that caused the match."""

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
    """Whether ``phrase`` appears as a contiguous run inside ``tokens``."""
    if not phrase or len(phrase) > len(tokens):
        return False
    return any(tokens[i : i + len(phrase)] == phrase for i in range(len(tokens) - len(phrase) + 1))


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple[PriorRow, ...]:
    """Read the reference table."""
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
            multiplier = float(raw["suggested_multiplier"])
            if not multiplier > 0:
                raise ValueError(
                    f"{source.name} line {line_no}: suggested_multiplier must be greater "
                    f"than zero, got {multiplier!r} -- a zero or negative multiplier would "
                    "drive the order to nothing while the row still reads as advice."
                )
            # The words come from `direction` and the arithmetic from `multiplier`, so a row where
            # they disagree tells the buyer demand falls while raising their order.
            expected_up = multiplier > 1.0
            expected_down = multiplier < 1.0
            if (direction == "up" and not expected_up) or (
                direction == "down" and not expected_down
            ):
                raise ValueError(
                    f"{source.name} line {line_no}: direction {direction!r} contradicts "
                    f"suggested_multiplier {multiplier!r}. 'up' needs a multiplier above "
                    "1.0 and 'down' needs one below it."
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
                    multiplier=multiplier,
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    return tuple(rows)


def match(sku: str, festival_key: str, *, path: str | None = None) -> PriorMatch | None:
    """Best keyword match for ``sku`` at ``festival_key``, or ``None``."""
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
    """Return a prior-sourced :class:`~inventory_engine.service.uplift.Uplift`."""
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
    """Measured uplift if the history supports one, otherwise the reference suggestion."""
    from inventory_engine.service.uplift import measure_sku

    measured = measure_sku(sku, series, festival_key, region=region)
    if measured.source is Source.MEASURED:
        return measured
    return prior_uplift(sku, festival_key, path=path)


def coverage_report(skus: list[str], festival_key: str, *, path: str | None = None) -> dict:
    """How many of ``skus`` the reference table can speak about at all."""
    matched = {s: m for s in skus if (m := match(s, festival_key, path=path)) is not None}
    return {
        "festival_key": festival_key,
        "n_skus": len(skus),
        "n_matched": len(matched),
        "matched": {s: m.keyword for s, m in matched.items()},
        "unmatched": sorted(set(skus) - set(matched)),
    }
