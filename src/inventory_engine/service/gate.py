"""The data gate — refuse honestly rather than guess."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

import pandas as pd

#: Columns the service cannot proceed without.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("sku", "date", "units_sold")

#: Optional. Absent or null prices fall back to
#: :data:`inventory_engine.optimize.costs.FALLBACK_PRICE`, exactly as tier 1 does -- the order
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("unit_price",)

#: Accepted spellings for each canonical column.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "sku": ("sku", "product", "product_id", "item", "item_id"),
    "date": ("date", "day", "ds", "order_date", "sale_date"),
    "units_sold": ("units_sold", "units", "qty", "quantity", "sales", "demand", "y"),
    "unit_price": ("unit_price", "price", "sell_price", "selling_price"),
}

MIN_HISTORY_DAYS: Final = 90
MIN_OBSERVATIONS: Final = 20
MAX_GAP_DAYS: Final = 14

#: Above this share of unparseable dates the file is rejected outright rather than per-SKU: it
#: means the date column is in a format pandas cannot read, not that a few rows are dirty, and
MAX_UNPARSEABLE_DATE_SHARE: Final = 0.5


@dataclass(frozen=True)
class SkuVerdict:
    """The gate's decision about one SKU, with the numbers behind it."""

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
    """The gate's verdict on a whole upload."""

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
    """Map the file's headers onto canonical names."""
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


def _reasons_for(n_days: int, n_obs: int, n_nonnull: int, max_gap_days: int) -> tuple[str, ...]:
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
    """Apply the gate to a parsed upload."""
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

    # Duplicate (sku, date) is summed rather than rejected: it is what a transaction-level export
    # looks like when one product sold twice in a day.
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
    """Return the message shown when nothing in the upload can be forecast."""
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
