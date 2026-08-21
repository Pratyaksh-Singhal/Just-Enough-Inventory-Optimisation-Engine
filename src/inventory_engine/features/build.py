"""Phase 2 feature builder: ``fact_sales`` + ``dim_calendar`` -> ``feature_panel``."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from inventory_engine.config import HORIZON, WAREHOUSE_PATH
from inventory_engine.data.schema import DIM_CALENDAR, FACT_SALES

FEATURE_PANEL: Final = "feature_panel"

#: Series identity. Not features in themselves; E4 uses them as categoricals.
KEY_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
)

LAG_DAYS: Final[tuple[int, ...]] = (7, 14, 28, 365)
ROLL_WINDOWS: Final[tuple[int, ...]] = (7, 28, 90)
#: (suffix, DuckDB aggregate). ``std`` is sample stddev: a 1-observation window is
#: undefined rather than zero, and NULL is the honest answer there.
ROLL_AGGS: Final[tuple[tuple[str, str], ...]] = (
    ("mean", "avg"),
    ("std", "stddev_samp"),
    ("max", "max"),
)

#: Every feature derived from ``units``. All are horizon-shifted.
UNITS_DERIVED: Final[frozenset[str]] = frozenset(
    [f"units_lag_{k}" for k in LAG_DAYS]
    + [f"units_roll_{suffix}_{w}" for w in ROLL_WINDOWS for suffix, _ in ROLL_AGGS]
    + ["days_since_last_sale", "zero_share_90"]
)

#: Every feature a retailer genuinely knows before the target date.
KNOWN_IN_ADVANCE: Final[frozenset[str]] = frozenset(
    {
        "wday",
        "month",
        "week_of_year",
        "is_weekend",
        "snap",
        "event_type",
        "days_since_event",
        "days_to_event",
        "price",
        "price_rel_28",
        "price_changed",
        "is_listed",
    }
)

_SERIES: Final = "PARTITION BY item_id, store_id ORDER BY date"


def feature_columns() -> tuple[str, ...]:
    """Return every feature column name, sorted, excluding keys and the target."""
    return tuple(sorted(UNITS_DERIVED | KNOWN_IN_ADVANCE))


@dataclass(frozen=True)
class FeatureReport:
    """Summary of a feature build."""

    horizon: int
    rows: int
    series: int
    n_features: int
    date_min: str
    date_max: str
    rows_dropped_pre_listing: int
    null_shares: tuple[tuple[str, float], ...]
    elapsed_seconds: float

    def render(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            f"  horizon         {self.horizon} days",
            f"  rows            {self.rows:,}"
            f"  ({self.rows_dropped_pre_listing:,} dropped as pre-listing)",
            f"  series          {self.series:,}",
            f"  features        {self.n_features}"
            f"  ({len(UNITS_DERIVED)} units-derived, {len(KNOWN_IN_ADVANCE)} known-in-advance)",
            f"  date range      {self.date_min} -> {self.date_max}",
            f"  elapsed         {self.elapsed_seconds:.1f}s",
        ]
        nulls = [(n, s) for n, s in self.null_shares if s > 0]
        if nulls:
            lines.append("")
            lines.append("  null shares (expected: warm-up before the window fills)")
            for name, share in nulls:
                lines.append(f"    {name:<24} {share:.1%}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- The only two places
# `units` is read.


def _lag_sql(expr: str, lag: int, horizon: int) -> str:
    """Value of ``expr`` ``lag`` days before the forecast origin."""
    return f"lag({expr}, {horizon + lag}) OVER ({_SERIES})"


def _rolling_sql(agg: str, expr: str, window: int, horizon: int) -> str:
    """Aggregate ``expr`` over ``window`` days ending at the forecast origin."""
    return (
        f"{agg}({expr}) OVER ({_SERIES} "
        f"ROWS BETWEEN {horizon + window - 1} PRECEDING AND {horizon} PRECEDING)"
    )


def _units_feature_sql(horizon: int) -> list[str]:
    """Emit every units-derived feature, each via :func:`_lag_sql` or :func:`_rolling_sql`."""
    parts = [f"{_lag_sql('units', k, horizon)} AS units_lag_{k}" for k in LAG_DAYS]
    parts += [
        f"{_rolling_sql(agg, 'units', w, horizon)} AS units_roll_{suffix}_{w}"
        for w in ROLL_WINDOWS
        for suffix, agg in ROLL_AGGS
    ]
    # Share of zero-sales days in the trailing 90 days before the origin.
    parts.append(
        f"{_rolling_sql('avg', 'CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END', 90, horizon)}"
        " AS zero_share_90"
    )
    # Days from the last non-zero sale to the origin.
    last_sale = _rolling_sql("max", "CASE WHEN units > 0 THEN date END", 10**6, horizon)
    parts.append(f"date_diff('day', {last_sale}, date) - {horizon} AS days_since_last_sale")
    return parts


def _known_in_advance_sql() -> list[str]:
    """Emit calendar and price features. Deliberately unshifted -- see module docstring."""
    return [
        "c.wday",
        "c.month",
        "c.week_of_year",
        "c.is_weekend",
        "f.snap",
        "c.event_type",
        "c.days_since_event",
        "c.days_to_event",
        "f.price",
        # Promo proxy: today's price against the recent norm. Trailing 28 days including
        # today, all of which the retailer set itself.
        "f.price / nullif(avg(f.price) OVER"
        f" ({_SERIES} ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 0) AS price_rel_28",
        f"(f.price IS DISTINCT FROM lag(f.price, 1) OVER ({_SERIES})) AS price_changed",
        "(f.price IS NOT NULL) AS is_listed",
    ]


# --------------------------------------------------------------------------- Build
# ---------------------------------------------------------------------------.


def _assert_rectangular(con: duckdb.DuckDBPyConnection) -> None:
    """Fail unless every series has a row for every date in the panel."""
    rows, series, dates = con.execute(f"""
        SELECT count(*), count(DISTINCT (item_id, store_id)), count(DISTINCT date)
        FROM {FACT_SALES}
    """).fetchone()
    if rows != series * dates:
        raise ValueError(
            f"{FACT_SALES} is not rectangular: {rows:,} rows but {series:,} series"
            f" x {dates:,} dates = {series * dates:,}. Row-based window frames assume one"
            " row per series per day; rebuild the warehouse before building features."
        )


def _build_calendar_features(con: duckdb.DuckDBPyConnection) -> None:
    """Derive date-level features, including distance to the nearest event."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cal_feat AS
        WITH events AS (
            SELECT date FROM {DIM_CALENDAR} WHERE event_name_1 IS NOT NULL
        )
        SELECT
            c.date,
            c.wday,
            c.month,
            CAST(weekofyear(c.date) AS TINYINT)      AS week_of_year,
            (c.wday IN (1, 2))                       AS is_weekend,
            coalesce(c.event_type_1, 'none')         AS event_type,
            date_diff('day',
                (SELECT max(e.date) FROM events e WHERE e.date <= c.date), c.date
            )                                        AS days_since_event,
            date_diff('day', c.date,
                (SELECT min(e.date) FROM events e WHERE e.date >= c.date)
            )                                        AS days_to_event
        FROM {DIM_CALENDAR} c
    """)


def build_features(
    con: duckdb.DuckDBPyConnection,
    horizon: int = HORIZON,
    *,
    table: str = FEATURE_PANEL,
) -> FeatureReport:
    """Build the feature panel from ``fact_sales`` and ``dim_calendar``."""
    if horizon < 1:
        raise ValueError(
            f"horizon must be >= 1, got {horizon}. A horizon of 0 would frame window"
            " features up to the target date itself, which is exactly the leak this"
            " module exists to prevent."
        )
    _assert_rectangular(con)
    started = time.perf_counter()

    _build_calendar_features(con)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE first_listed AS
        SELECT item_id, store_id, min(date) AS first_listed_date
        FROM {FACT_SALES} WHERE price IS NOT NULL
        GROUP BY 1, 2
    """)

    select_parts = [
        *(f"f.{c}" for c in KEY_COLUMNS),
        f"{horizon} AS horizon",
        "f.units",
        *_units_feature_sql(horizon),
        *_known_in_advance_sql(),
    ]

    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"""
        CREATE TABLE {table} AS
        WITH featured AS (
            SELECT {", ".join(select_parts)}
            FROM {FACT_SALES} f
            JOIN cal_feat c USING (date)
        )
        SELECT featured.*
        FROM featured
        JOIN first_listed l USING (item_id, store_id)
        -- Rows before an item was ever listed are not zero demand; they are the absence
        -- of a product. Training on them teaches the model to forecast shelf gaps.
        WHERE featured.date >= l.first_listed_date
    """)
    # Enforces the (date, item, store) grain the rest of the project assumes.
    con.execute(f"CREATE UNIQUE INDEX idx_{table}_grain ON {table} (date, item_id, store_id)")

    return _summarise(con, table, horizon, time.perf_counter() - started)


def _summarise(
    con: duckdb.DuckDBPyConnection, table: str, horizon: int, elapsed: float
) -> FeatureReport:
    """Compute post-build summary statistics."""
    rows, series, dmin, dmax = con.execute(f"""
        SELECT count(*), count(DISTINCT (item_id, store_id)),
               CAST(min(date) AS VARCHAR), CAST(max(date) AS VARCHAR)
        FROM {table}
    """).fetchone()
    (total,) = con.execute(f"SELECT count(*) FROM {FACT_SALES}").fetchone()

    names = feature_columns()
    null_sql = ", ".join(f"avg(CASE WHEN {n} IS NULL THEN 1.0 ELSE 0.0 END)" for n in names)
    shares = con.execute(f"SELECT {null_sql} FROM {table}").fetchone()

    return FeatureReport(
        horizon=horizon,
        rows=rows,
        series=series,
        n_features=len(names),
        date_min=dmin,
        date_max=dmax,
        rows_dropped_pre_listing=total - rows,
        null_shares=tuple(zip(names, shares, strict=True)),
        elapsed_seconds=elapsed,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``build-features [--horizon N] [--db-path PATH]``."""
    parser = argparse.ArgumentParser(description="Build the feature panel (Phase 2).")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help="Forecast horizon in days. Every units feature is shifted by this much.",
    )
    parser.add_argument("--table", default=FEATURE_PANEL)
    args = parser.parse_args(argv)

    if not args.db_path.is_file():
        print(
            f"\nNo warehouse at {args.db_path}. Run `build-warehouse` first.\n",
            file=sys.stderr,
        )
        return 1

    con = duckdb.connect(str(args.db_path))
    try:
        report = build_features(con, args.horizon, table=args.table)
    except ValueError as exc:
        print(f"\nFeature build failed: {exc}\n", file=sys.stderr)
        return 1
    finally:
        con.close()

    print(f"\n{args.table} built in {args.db_path}\n{report.render()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
