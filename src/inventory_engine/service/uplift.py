"""What a festival actually did to *this* SKU — measured first, assumed only as a fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from inventory_engine.service.festivals import (
    DEFAULT_REGION,
    SOURCES,
    Festival,
    is_partial,
    is_prior_only,
    partial_coverage,
    windows_in,
)

#: Days either side of a festival window excluded from its baseline. Demand right after a
#: festival is usually depressed, and counting those days as "normal" inflates the ratio.
BUFFER_DAYS: Final = 3

#: Days of surrounding history each side used to form the baseline, outside the buffer.
BASELINE_DAYS: Final = 28

#: Minimum baseline observations before a ratio is worth quoting. Below this the denominator
#: is a couple of days and the ratio is noise wearing a decimal point.
MIN_BASELINE_OBS: Final = 6

#: Minimum observations in the run-up window.
MIN_LEAD_OBS: Final = 3


class Source(StrEnum):
    """Where a multiplier came from. Never inferred from context — always carried."""

    MEASURED = "measured"
    PRIOR = "prior"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Occurrence:
    """One past instance of a festival, for one SKU."""

    year: int
    baseline_mean: float
    lead_mean: float
    lead_ratio: float
    day_mean: float
    day_ratio: float
    n_lead: int
    n_baseline: int
    closed_days: int = 0


@dataclass(frozen=True)
class Uplift:
    """The effect of one festival on one SKU, with its provenance attached."""

    sku: str
    festival_key: str
    festival_name: str
    source: Source = Source.UNAVAILABLE
    occurrences: tuple[Occurrence, ...] = ()
    prior_multiplier: float | None = None
    prior_reason: str = ""
    reason: str = ""
    #: The calendar dates this festival in only some years, so a measurement rests on the subset it
    #: happens to have.
    partial_calendar: bool = False

    @property
    def available(self) -> bool:
        """Whether any multiplier at all can be offered."""
        return self.source is not Source.UNAVAILABLE

    @property
    def multiplier(self) -> float:
        """The number to act on, whatever its source."""
        if self.source is Source.MEASURED and self.occurrences:
            return float(np.median([o.lead_ratio for o in self.occurrences]))
        if self.source is Source.PRIOR and self.prior_multiplier is not None:
            return float(self.prior_multiplier)
        return float("nan")

    @property
    def spread(self) -> tuple[float, float]:
        """Lowest and highest measured run-up ratio, or ``(nan, nan)`` for a prior."""
        if self.source is not Source.MEASURED or not self.occurrences:
            return float("nan"), float("nan")
        ratios = [o.lead_ratio for o in self.occurrences]
        return min(ratios), max(ratios)

    @property
    def n_years(self) -> int:
        """How many past instances the measurement rests on. Zero for a prior."""
        return len(self.occurrences)

    @property
    def closed_on_the_day(self) -> bool:
        """Whether the shop appears to shut for this festival in every year observed."""
        return bool(self.occurrences) and all(o.closed_days > 0 for o in self.occurrences)

    def describe(self) -> str:
        """One line a buyer can read, naming the evidence behind the number."""
        if self.source is Source.UNAVAILABLE:
            return f"{self.festival_name}: {self.reason}"

        if self.source is Source.PRIOR:
            return (
                f"{self.festival_name}: suggested {self.multiplier:.2f}x — a reference figure "
                f"for this kind of product, not measured from your sales. {self.prior_reason}"
            ).strip()

        low, high = self.spread
        shut = " Your shop recorded no sales on the day itself." if self.closed_on_the_day else ""
        gappy = f" {self._partial_note()}" if self.partial_calendar else ""
        if self.n_years == 1:
            return (
                f"{self.festival_name}: your sales ran {self.multiplier:.2f}x normal in the "
                f"run-up last year. One year of evidence — a hint, not a forecast.{shut}{gappy}"
            )
        return (
            f"{self.festival_name}: your sales ran {self.multiplier:.2f}x normal in the run-up "
            f"(range {low:.2f}-{high:.2f}x across {self.n_years} years).{shut}{gappy}"
        )

    def _partial_note(self) -> str:
        """Return the caveat for a festival our calendar can only date in some years."""
        return partial_coverage().get(self.festival_key, "Our calendar has gaps for this festival.")


def _baseline_mask(index: pd.DatetimeIndex, festival: Festival, weekdays: set[int]) -> np.ndarray:
    """Nearby days on matching weekdays, outside the window and its buffer."""
    outer_start = festival.window_start - timedelta(days=BUFFER_DAYS + BASELINE_DAYS)
    outer_end = festival.window_end + timedelta(days=BUFFER_DAYS + BASELINE_DAYS)
    inner_start = festival.window_start - timedelta(days=BUFFER_DAYS)
    inner_end = festival.window_end + timedelta(days=BUFFER_DAYS)

    as_date = index.date
    near = (as_date >= outer_start) & (as_date <= outer_end)
    outside = (as_date < inner_start) | (as_date > inner_end)
    return near & outside & np.isin(index.dayofweek, list(weekdays))


def measure(series: pd.Series, festival: Festival) -> Occurrence | None:
    """Measure one occurrence, or ``None`` when the history cannot support it."""
    index = pd.DatetimeIndex(series.index)
    as_date = index.date

    lead = (as_date >= festival.window_start) & (as_date < festival.day)
    day = (as_date >= festival.day) & (as_date <= festival.window_end)

    lead_values = series[lead].dropna()
    if len(lead_values) < MIN_LEAD_OBS:
        return None

    weekdays = set(index[lead].dayofweek.tolist())
    baseline_values = series[_baseline_mask(index, festival, weekdays)].dropna()
    if len(baseline_values) < MIN_BASELINE_OBS:
        return None

    baseline_mean = float(baseline_values.mean())
    if baseline_mean <= 0:
        return None

    day_values = series[day].dropna()
    day_mean = float(day_values.mean()) if len(day_values) else float("nan")
    lead_mean = float(lead_values.mean())

    return Occurrence(
        year=festival.day.year,
        baseline_mean=baseline_mean,
        lead_mean=lead_mean,
        lead_ratio=lead_mean / baseline_mean,
        day_mean=day_mean,
        day_ratio=day_mean / baseline_mean if np.isfinite(day_mean) else float("nan"),
        n_lead=len(lead_values),
        n_baseline=len(baseline_values),
        closed_days=int((day_values == 0).sum()) if len(day_values) else 0,
    )


#: Used when a festival is flagged partial in SOURCES but has no row in PARTIAL.
PARTIAL_FALLBACK: Final = "Our calendar has gaps for this festival."


def measure_sku(
    sku: str, series: pd.Series, festival_key: str, *, region: str = DEFAULT_REGION
) -> Uplift:
    """Measure one festival's effect on one SKU across every year its history covers."""
    if is_prior_only(festival_key):
        # Independence Day and Republic Day. Refused here rather than at each call site, so
        # one place decides and there is no way to route around it.
        return Uplift(
            sku,
            festival_key,
            SOURCES[festival_key].name,
            reason=(
                "not measured from your own sales by design. This is a small effect on a "
                "narrow set of products — snacks, soft drinks and flags — and a ratio "
                "fitted to it would mostly be measuring noise. The reference figure is "
                "used instead."
            ),
        )

    gappy = is_partial(festival_key)
    observed = series.dropna()
    if observed.empty:
        return Uplift(sku, festival_key, festival_key, reason="no observations")

    first, last = observed.index.min().date(), observed.index.max().date()
    instances = [f for f in windows_in(first, last, region) if f.key == festival_key]
    name = instances[0].name if instances else festival_key

    if not instances:
        months = round((last - first).days / 30.4)
        # A gappy calendar and a short history produce the same empty list and are not the
        # same problem: "you need more data" is wrong advice for a year we cannot date.
        why = (
            f"not measurable — your history covers {months} month(s), and our calendar has "
            f"no date for this festival inside it. "
            f"{partial_coverage().get(festival_key, PARTIAL_FALLBACK)}"
            if gappy
            else (
                f"not measurable — your history covers {months} month(s) and does not "
                "include this festival. An annual festival needs about 13 months of data "
                "before its effect can be measured from your own sales."
            )
        )
        return Uplift(sku, festival_key, name, reason=why, partial_calendar=gappy)

    measured = tuple(m for f in instances if (m := measure(observed, f)) is not None)
    if not measured:
        return Uplift(
            sku,
            festival_key,
            name,
            reason=(
                "not measurable — the festival falls inside your history but there are too "
                "few recorded days around it to compare against a normal week."
            ),
            partial_calendar=gappy,
        )
    return Uplift(
        sku,
        festival_key,
        name,
        source=Source.MEASURED,
        occurrences=measured,
        partial_calendar=gappy,
    )


def measure_all(sku: str, series: pd.Series, *, region: str = DEFAULT_REGION) -> list[Uplift]:
    """Every festival this SKU's history covers, largest deviation from normal first."""
    observed = series.dropna()
    if observed.empty:
        return []
    first, last = observed.index.min().date(), observed.index.max().date()
    keys = {f.key for f in windows_in(first, last, region)}
    measured = [measure_sku(sku, series, key, region=region) for key in sorted(keys)]
    return sorted(
        (u for u in measured if u.source is Source.MEASURED),
        key=lambda u: -abs(u.multiplier - 1.0),
    )


def relevant_for(
    last_observed: date, horizon: int, *, region: str = DEFAULT_REGION
) -> list[Festival]:
    """Festivals whose window overlaps the forecast horizon."""
    start = last_observed + timedelta(days=1)
    return windows_in(start, last_observed + timedelta(days=horizon), region)
