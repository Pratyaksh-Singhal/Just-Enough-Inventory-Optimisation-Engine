"""The data gate — refuse honestly rather than guess.

Why a gate exists at all
------------------------
Tier 1's quick calculator works on whatever it is given: it takes the empirical distribution
of H-day totals from the user's own history and reads a quantile off it. That degrades
gracefully, because with four observations it is visibly a rough estimate and the page says
so.

Tier 2 does not degrade gracefully. It fits a model, backtests it on rolling origins, and
returns a number with a measured error attached. On 21 days of history that error estimate
is itself noise, and the output would be *more* confident and *less* trustworthy than the
calculator's. A confident number from insufficient history is worse than a refusal, so the
gate refuses -- and names the shortfall, so the refusal is actionable rather than a wall.

What the gate does not do
-------------------------
It never pads, interpolates, or resamples a series onto a regular grid to make it pass.
Every transformation it *does* apply is counted and reported back to the caller:

* duplicate ``(sku, date)`` rows are summed -- the obvious read of a transaction-level
  export -- and the merge count is reported
* negative units are treated as returns and clipped to zero, and the count is reported
* rows with an unparseable date are dropped, and the count is reported

None of those are silent. A caller can always reconstruct what happened to their file.

Thresholds
----------
:data:`MIN_HISTORY_DAYS` (90) is the span the brief specifies, and it is a floor rather than
a recommendation: at 90 days and a 28-day horizon the rolling-origin backtest can only fit
two folds, so the fold spread is a range over two numbers. See
:func:`inventory_engine.service.folds.fold_count_for`.

:data:`MIN_OBSERVATIONS` (20) is separate from the span on purpose. A SKU can span 200 days
with 6 rows in it; the span says the history is old enough and the count says there is
something in it.

:data:`MAX_GAP_DAYS` (14) is the "roughly contiguous" check. A fortnight of missing rows in
a daily series is either a delisting, a stockout, or an export bug, and in all three cases
the days are not zero-demand days -- treating them as such would bias every quantile down.
The gate flags the gap rather than deciding which of the three it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

import pandas as pd

#: Columns the service cannot proceed without.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("sku", "date", "units_sold")

#: Optional. Absent or null prices fall back to
#: :data:`inventory_engine.optimize.costs.FALLBACK_PRICE`, exactly as tier 1 does -- the
#: order *quantity* does not depend on price at all (price cancels out of the critical
#: ratio), only the money columns do.
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("unit_price",)

#: Accepted spellings for each canonical column. Deliberately short: tier 1's calculator
#: does loose substring matching because a wrong guess there costs the user one glance at a
#: table, whereas here it would silently forecast the wrong column for twenty minutes on a
#: worker. Whatever mapping is chosen is echoed back in :attr:`GateReport.column_mapping`,
#: so it is visible rather than assumed.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "sku": ("sku", "product", "product_id", "item", "item_id"),
    "date": ("date", "day", "ds", "order_date", "sale_date"),
    "units_sold": ("units_sold", "units", "qty", "quantity", "sales", "demand", "y"),
    "unit_price": ("unit_price", "price", "sell_price", "selling_price"),
}

MIN_HISTORY_DAYS: Final = 90
MIN_OBSERVATIONS: Final = 20
MAX_GAP_DAYS: Final = 14

#: Above this share of unparseable dates the file is rejected outright rather than
#: per-SKU: it means the date column is in a format pandas cannot read, not that a few rows
#: are dirty, and reporting it 300 times per SKU would bury the actual problem.
MAX_UNPARSEABLE_DATE_SHARE: Final = 0.5


@dataclass(frozen=True)
class SkuVerdict:
    """The gate's decision about one SKU, with the numbers behind it.

    Attributes:
        sku: The product identifier as it appeared in the file.
        admitted: Whether this SKU can support a real forecast.
        n_days: Calendar span, ``last_date - first_date + 1``. This is the number the
            90-day threshold is checked against -- not the row count, which can be larger
            (duplicates) or much smaller (gaps).
        n_obs: Rows surviving date parsing, after duplicate dates are merged.
        n_nonnull: Rows whose ``units_sold`` is not null. Zero is *not* null: a
            zero-demand day is an observation, and 61.6% of the M5 panel this project was
            built on consists of them.
        first_date: Earliest observation.
        last_date: Latest observation.
        max_gap_days: Largest run of consecutive missing calendar days inside the span.
        reasons: Why the SKU was excluded. Empty when ``admitted`` is True.

    """

    sku: str
    admitted: bool
    n_days: int
    n_obs: int
    n_nonnull: int
    first_date: date | None
    last_date: date | None
    max_gap_days: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateReport:
    """The gate's verdict on a whole upload.

    Attributes:
        admitted: SKUs that passed.
        rejected: SKUs that failed, each carrying its own reasons.
        fatal: Set when the file is unusable as a whole -- missing columns, unreadable
            dates, no rows. When this is set both SKU lists are empty, because the gate
            could not get far enough to form a per-SKU opinion.
        column_mapping: Which file column was read as which canonical column. Echoed back
            so an alias match is visible to the caller rather than assumed.
        rows_read: Data rows in the file, before any cleaning.
        rows_dropped_unparseable_date: Rows discarded because the date could not be read.
        rows_merged_duplicate_date: Rows absorbed by summing duplicate ``(sku, date)``
            pairs. ``0`` means every ``(sku, date)`` was unique.
        negative_units_clipped: Rows whose ``units_sold`` was below zero and was treated as
            a return, i.e. clipped to zero.

    """

    admitted: tuple[SkuVerdict, ...] = ()
    rejected: tuple[SkuVerdict, ...] = ()
    fatal: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    rows_read: int = 0
    rows_dropped_unparseable_date: int = 0
    rows_merged_duplicate_date: int = 0
    negative_units_clipped: int = 0

    @property
    def usable(self) -> bool:
        """Whether anything at all can be forecast from this upload."""
        return self.fatal is None and bool(self.admitted)

    def summary(self) -> str:
        """One-line human summary, suitable for a log line or an API message."""
        if self.fatal:
            return f"rejected: {self.fatal}"
        return (
            f"{len(self.admitted)} SKU(s) admitted, {len(self.rejected)} excluded "
            f"from {self.rows_read} rows"
        )

    def warnings(self) -> tuple[str, ...]:
        """Every transformation the gate applied, stated plainly. Empty when it applied none."""
        out = []
        if self.rows_dropped_unparseable_date:
            out.append(
                f"{self.rows_dropped_unparseable_date} row(s) dropped: date could not be read."
            )
        if self.rows_merged_duplicate_date:
            out.append(
                f"{self.rows_merged_duplicate_date} row(s) had a date already seen for the "
                "same product; their units were added together."
            )
        if self.negative_units_clipped:
            out.append(
                f"{self.negative_units_clipped} row(s) had negative units, read as returns "
                "and counted as zero demand."
            )
        return tuple(out)


def resolve_columns(columns: list[str]) -> tuple[dict[str, str], str | None]:
    """Map the file's headers onto canonical names.

    Args:
        columns: Header row as it appears in the file.

    Returns:
        ``(mapping, fatal)`` where ``mapping`` is canonical name -> file column, and
        ``fatal`` is a message naming every missing required column, or ``None``.

    """
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[canonical] = lowered[alias]
                break

    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        wanted = ", ".join(f"'{c}'" for c in missing)
        found = ", ".join(f"'{c}'" for c in columns) or "(no columns)"
        accepted = "; ".join(
            f"{k}: " + "/".join(v) for k, v in COLUMN_ALIASES.items() if k in missing
        )
        return mapping, (
            f"Required column(s) not found: {wanted}. Your file has: {found}. "
            f"Rename a column to one of the accepted names ({accepted})."
        )
    return mapping, None


def _reasons_for(
    n_days: int, n_obs: int, n_nonnull: int, max_gap_days: int
) -> tuple[str, ...]:
    """Every threshold this SKU fails, each naming the required and the actual value."""
    reasons = []
    if n_days < MIN_HISTORY_DAYS:
        reasons.append(f"{MIN_HISTORY_DAYS} days of history needed, {n_days} found")
    if n_nonnull < MIN_OBSERVATIONS:
        reasons.append(
            f"{MIN_OBSERVATIONS} recorded days needed, {n_nonnull} found"
            + (f" (from {n_obs} rows)" if n_obs != n_nonnull else "")
        )
    if max_gap_days > MAX_GAP_DAYS:
        reasons.append(
            f"a {max_gap_days}-day stretch with no rows at all; gaps over "
            f"{MAX_GAP_DAYS} days are not treated as zero-demand days"
        )
    return tuple(reasons)


def _max_gap(dates: pd.Series) -> int:
    """Longest run of consecutive missing calendar days inside the observed span."""
    if len(dates) < 2:
        return 0
    ordered = dates.sort_values().drop_duplicates()
    # diff() of 1 day means contiguous; a diff of n days leaves n-1 days missing.
    return int((ordered.diff().dt.days.max() or 1) - 1)


def evaluate(frame: pd.DataFrame) -> GateReport:
    """Apply the gate to a parsed upload.

    Pure: no I/O, no database, no config. This is the whole decision, and
    ``tests/test_gate.py`` exercises it directly with hand-built frames -- including a
    deliberately-too-small one -- so the refusal behaviour is proven without a server, a
    queue or a Postgres running.

    Args:
        frame: The CSV as read, with its original column names.

    Returns:
        A :class:`GateReport`. Check ``.fatal`` first, then ``.admitted``.

    """
    mapping, fatal = resolve_columns(list(frame.columns))
    if fatal:
        return GateReport(fatal=fatal, column_mapping=mapping, rows_read=len(frame))

    rows_read = len(frame)
    if rows_read == 0:
        return GateReport(
            fatal="The file has a header row but no data rows.",
            column_mapping=mapping,
        )

    work = pd.DataFrame(
        {
            "sku": frame[mapping["sku"]].astype("string").str.strip(),
            "date": pd.to_datetime(frame[mapping["date"]], errors="coerce", format="mixed"),
            "units_sold": pd.to_numeric(frame[mapping["units_sold"]], errors="coerce"),
        }
    )

    unparseable = int(work["date"].isna().sum())
    if unparseable > rows_read * MAX_UNPARSEABLE_DATE_SHARE:
        sample = frame[mapping["date"]].dropna().astype(str).head(3).tolist()
        return GateReport(
            fatal=(
                f"{unparseable} of {rows_read} dates could not be read, so the date column "
                f"is in a format this service does not recognise. Examples from your file: "
                f"{', '.join(repr(s) for s in sample) or '(none)'}. Use YYYY-MM-DD."
            ),
            column_mapping=mapping,
            rows_read=rows_read,
        )
    work = work.dropna(subset=["date"])
    work = work[work["sku"].notna() & (work["sku"] != "")]
    if work.empty:
        return GateReport(
            fatal="No rows survived reading: every row is missing a product or a usable date.",
            column_mapping=mapping,
            rows_read=rows_read,
            rows_dropped_unparseable_date=unparseable,
        )

    negatives = int((work["units_sold"] < 0).sum())
    work.loc[work["units_sold"] < 0, "units_sold"] = 0.0

    # Duplicate (sku, date) is summed rather than rejected: it is what a transaction-level
    # export looks like when one product sold twice in a day. Nulls stay null -- summing a
    # group that is entirely null must not turn "no record" into "zero sold".
    before = len(work)
    grouped = work.groupby(["sku", "date"], as_index=False)["units_sold"].sum(min_count=1)
    merged = before - len(grouped)

    verdicts = []
    for sku, group in grouped.groupby("sku", sort=True):
        dates = group["date"]
        first, last = dates.min(), dates.max()
        n_days = int((last - first).days) + 1
        n_obs = len(group)
        n_nonnull = int(group["units_sold"].notna().sum())
        gap = _max_gap(dates)
        reasons = _reasons_for(n_days, n_obs, n_nonnull, gap)
        verdicts.append(
            SkuVerdict(
                sku=str(sku),
                admitted=not reasons,
                n_days=n_days,
                n_obs=n_obs,
                n_nonnull=n_nonnull,
                first_date=first.date(),
                last_date=last.date(),
                max_gap_days=gap,
                reasons=reasons,
            )
        )

    return GateReport(
        admitted=tuple(v for v in verdicts if v.admitted),
        rejected=tuple(v for v in verdicts if not v.admitted),
        column_mapping=mapping,
        rows_read=rows_read,
        rows_dropped_unparseable_date=unparseable,
        rows_merged_duplicate_date=merged,
        negative_units_clipped=negatives,
    )


def refusal_message(report: GateReport) -> str:
    """Return the message shown when nothing in the upload can be forecast.

    Names the specific shortfall per SKU and points at the quick calculator, which genuinely
    does work on this data -- the point of the gate is to redirect, not to stonewall.
    """
    if report.fatal:
        return report.fatal

    worst = sorted(report.rejected, key=lambda v: -v.n_days)[:5]
    detail = "; ".join(f"{v.sku}: {', '.join(v.reasons)}" for v in worst)
    more = f" (and {len(report.rejected) - len(worst)} more)" if len(report.rejected) > 5 else ""
    return (
        f"None of the {len(report.rejected)} product(s) in this file have enough history "
        f"for a backtested forecast. {detail}{more}. "
        "Use the quick calculator instead -- it works on short history because it reads a "
        "quantile straight off your own demand, with no model to validate."
    )
