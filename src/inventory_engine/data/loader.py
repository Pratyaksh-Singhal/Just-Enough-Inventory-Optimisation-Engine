"""Phase 1 loader: raw M5 CSVs -> long-format DuckDB warehouse.

Design notes
------------
The whole ETL runs as DuckDB SQL against the CSVs on disk. There is no pandas
in the loading path. ``sales_train_evaluation.csv`` is a 122 MB wide table of
1,947 columns; melting it in pandas means materialising the full frame in RAM
before the scope filter can shrink it. DuckDB pushes the filter down to the CSV
scan and streams the ``UNPIVOT``, so peak memory stays flat and the whole build
is a single transaction that either produces a valid warehouse or nothing.

The target table is created from an explicit DDL with ``NOT NULL`` and primary
key constraints, then inserted into. That is deliberate: the insert doubles as a
data-quality assertion. If the calendar join fans out, or a series has duplicate
(date, item, store) rows, the load fails loudly instead of quietly producing a
panel that every downstream phase would misread.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from inventory_engine.config import (
    BACKTEST_DAYS,
    DATA_DIR,
    DEFAULT_SCOPE,
    KAGGLE_DATASET_URL,
    RAW_CALENDAR_FILE,
    RAW_PRICES_FILE,
    RAW_SALES_FILE,
    REQUIRED_RAW_FILES,
    STRATIFY_WINDOW_DAYS,
    WAREHOUSE_PATH,
    Scope,
)
from inventory_engine.data.schema import (
    DIM_CALENDAR,
    DIM_CALENDAR_DDL,
    FACT_SALES,
    FACT_SALES_DDL,
)

VALID_STATES: frozenset[str] = frozenset({"CA", "TX", "WI"})

# Identity columns in the wide sales file; everything else is a d_N day column.
_ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


class MissingRawDataError(FileNotFoundError):
    """Raised when the M5 extracts are absent from the expected directory."""


@dataclass(frozen=True)
class StratumSummary:
    """One (department, intermittency stratum) cell of the sampled universe."""

    dept_id: str
    stratum: int
    items: int
    min_zero_share: float
    mean_zero_share: float
    max_zero_share: float


@dataclass(frozen=True)
class LoadReport:
    """Summary of a warehouse build, for logging and for the README."""

    scope: str
    rows: int
    series: int
    items: int
    stores: int
    depts: int
    date_min: str
    date_max: str
    zero_unit_share: float
    null_price_share: float
    elapsed_seconds: float
    strata: tuple[StratumSummary, ...] = ()

    def render(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            f"  scope           {self.scope}",
            f"  rows            {self.rows:,}",
            f"  series          {self.series:,}  ({self.items:,} items"
            f" x {self.stores} stores, {self.depts} depts)",
            f"  date range      {self.date_min} -> {self.date_max}",
            f"  zero-unit days  {self.zero_unit_share:.1%}",
            f"  null prices     {self.null_price_share:.1%}"
            "  (item not listed at that store that week)",
            f"  elapsed         {self.elapsed_seconds:.1f}s",
        ]
        if self.strata:
            lines.append("")
            lines.append("  sampled universe by intermittency stratum")
            lines.append(f"    {'dept':<9} {'stratum':>7} {'items':>6}  zero-day share")
            for s in self.strata:
                label = {1: "dense", 2: "mid", 3: "sparse"}.get(s.stratum, str(s.stratum))
                lines.append(
                    f"    {s.dept_id:<9} {label:>7} {s.items:>6}"
                    f"  {s.min_zero_share:.0%}-{s.max_zero_share:.0%}"
                    f" (mean {s.mean_zero_share:.0%})"
                )
        return "\n".join(lines)

    def strata_markdown(self) -> str:
        """Render the stratum breakdown as a markdown table for the README."""
        if not self.strata:
            return "_No stratified sample applied; all items retained._"
        header = (
            "| Dept | Stratum | Items | Zero-day share (min-max) | Mean |\n|---|---|---:|---|---:|"
        )
        names = {1: "dense", 2: "mid", 3: "sparse"}
        rows = [
            f"| {s.dept_id} | {names.get(s.stratum, s.stratum)} | {s.items} |"
            f" {s.min_zero_share:.0%}-{s.max_zero_share:.0%} | {s.mean_zero_share:.0%} |"
            for s in self.strata
        ]
        return "\n".join([header, *rows])


def _sql_literal(path: Path) -> str:
    """Quote a filesystem path for inline use in a DuckDB SQL string."""
    return "'" + str(path).replace("'", "''") + "'"


def verify_raw_files(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Check that every required M5 extract is present.

    Args:
        data_dir: Directory expected to contain the raw Kaggle CSVs.

    Returns:
        Mapping of filename to resolved path.

    Raises:
        MissingRawDataError: If any required file is absent, with instructions
            naming the exact files and the directory they belong in.

    """
    resolved = {name: data_dir / name for name in REQUIRED_RAW_FILES}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise MissingRawDataError(
            "Missing M5 data file(s): "
            + ", ".join(missing)
            + f"\n\nExpected them in: {data_dir}"
            + f"\nDownload from:   {KAGGLE_DATASET_URL}"
            + "\n\nThe competition download is a zip; extract it and copy "
            + ", ".join(REQUIRED_RAW_FILES)
            + " into the directory above. No renaming needed."
        )
    return resolved


def _load_calendar(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    """Load the full M5 calendar, including days beyond the sales window."""
    con.execute(DIM_CALENDAR_DDL)
    con.execute(f"""
        INSERT INTO {DIM_CALENDAR}
        SELECT
            CAST(date AS DATE),
            d,
            CAST(wm_yr_wk AS INTEGER),
            CAST(wday AS TINYINT),
            weekday,
            CAST(month AS TINYINT),
            CAST(year AS SMALLINT),
            NULLIF(event_name_1, ''),
            NULLIF(event_type_1, ''),
            NULLIF(event_name_2, ''),
            NULLIF(event_type_2, ''),
            CAST(snap_CA AS TINYINT),
            CAST(snap_TX AS TINYINT),
            CAST(snap_WI AS TINYINT)
        FROM read_csv({_sql_literal(path)}, header = true, all_varchar = true)
    """)


def _select_series(con: duckdb.DuckDBPyConnection, path: Path, scope: Scope) -> None:
    """Materialise the scoped wide sales rows into a temp table ``sales_wide``."""
    filters = ["state_id = ?", "cat_id = ?"]
    params: list[object] = [scope.state_id, scope.cat_id]
    if scope.dept_ids:
        placeholders = ", ".join("?" for _ in scope.dept_ids)
        filters.append(f"dept_id IN ({placeholders})")
        params.extend(scope.dept_ids)

    con.execute(
        f"""
        CREATE TEMP TABLE sales_wide AS
        SELECT * FROM read_csv({_sql_literal(path)}, header = true)
        WHERE {" AND ".join(filters)}
        """,
        params,
    )
    if con.execute("SELECT count(*) FROM sales_wide").fetchone()[0] == 0:
        raise ValueError(
            f"Scope matched zero series ({scope.describe()}). "
            "Check the state/category/department codes against the raw file."
        )


def _unpivot_to_long(con: duckdb.DuckDBPyConnection) -> None:
    """Melt the d_1..d_N day columns into (d, units) rows."""
    excluded = ", ".join(_ID_COLUMNS)
    con.execute(f"""
        CREATE TEMP TABLE sales_long AS
        UNPIVOT sales_wide
        ON COLUMNS(* EXCLUDE ({excluded}))
        INTO NAME d VALUE units
    """)


def _load_prices(con: duckdb.DuckDBPyConnection, prices_path: Path) -> None:
    """Materialise the scoped slice of ``sell_prices.csv`` into a temp table."""
    con.execute(f"""
        CREATE TEMP TABLE prices AS
        SELECT store_id, item_id, CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
               CAST(sell_price AS DOUBLE) AS sell_price
        FROM read_csv({_sql_literal(prices_path)}, header = true)
        WHERE item_id IN (SELECT DISTINCT item_id FROM sales_wide)
          AND store_id IN (SELECT DISTINCT store_id FROM sales_wide)
    """)


def _apply_stratified_sample(
    con: duckdb.DuckDBPyConnection, scope: Scope
) -> tuple[StratumSummary, ...]:
    """Draw ``scope.items_per_dept`` items per department, stratified by intermittency.

    Sampling by volume rank would quietly select the problem away: the densest
    SKUs are the easy ones, and Phase 3's Croston/TSB work only earns its place
    if genuinely intermittent series survive scoping. So each department's items
    are split into ``scope.n_strata`` equal-count bands by their share of
    zero-sales days, and an equal quota is drawn from each band.

    Two details matter for defensibility:

    * The intermittency statistic is measured over a
      :data:`~inventory_engine.config.STRATIFY_WINDOW_DAYS` window that ends
      before the backtest region starts. Selecting SKUs on statistics computed
      over the evaluation period is selection leakage, even offline.
    * Selection order comes from ``hash(item_id || seed)``, not ``random()``.
      A hash is deterministic across DuckDB versions, platforms and thread
      counts, so the sampled universe is reproducible in a way a PRNG seed is
      not guaranteed to be.

    Args:
        con: Open connection holding ``sales_wide``, ``sales_long``, ``prices``.
        scope: Scope carrying the item budget, stratum count and seed.

    Returns:
        Per (department, stratum) summaries of the drawn sample.

    """
    quota, remainder = divmod(scope.items_per_dept, scope.n_strata)

    con.execute(
        f"""
        CREATE TEMP TABLE item_intermittency AS
        WITH bounds AS (
            SELECT
                max(c.date) - INTERVAL {BACKTEST_DAYS} DAY AS ref_end,
                max(c.date) - INTERVAL {BACKTEST_DAYS + STRATIFY_WINDOW_DAYS - 1} DAY
                    AS ref_start
            FROM sales_long l JOIN {DIM_CALENDAR} c USING (d)
        ),
        -- An item is eligible only if it was actually on shelf during the
        -- window. A never-listed SKU is 100% zero-days for a reason that has
        -- nothing to do with demand, and would pollute the sparse stratum.
        listed AS (
            SELECT DISTINCT p.item_id
            FROM prices p
            JOIN {DIM_CALENDAR} c USING (wm_yr_wk)
            CROSS JOIN bounds b
            WHERE c.date BETWEEN b.ref_start AND b.ref_end
              AND p.sell_price IS NOT NULL
        )
        SELECT
            w.dept_id,
            w.item_id,
            avg(CASE WHEN l.units = 0 THEN 1.0 ELSE 0.0 END) AS zero_share
        FROM sales_long l
        JOIN sales_wide w USING (id)
        JOIN {DIM_CALENDAR} c USING (d)
        CROSS JOIN bounds b
        WHERE c.date BETWEEN b.ref_start AND b.ref_end
          AND w.item_id IN (SELECT item_id FROM listed)
        GROUP BY 1, 2
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE sampled_items AS
        WITH strat AS (
            SELECT dept_id, item_id, zero_share,
                   ntile({scope.n_strata}) OVER (
                       PARTITION BY dept_id ORDER BY zero_share, item_id
                   ) AS stratum
            FROM item_intermittency
        ),
        ranked AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY dept_id, stratum
                       ORDER BY hash(item_id || '#' || {scope.sampling_seed})
                   ) AS pick
            FROM strat
        )
        SELECT dept_id, stratum, item_id, zero_share
        FROM ranked
        WHERE pick <= CASE WHEN stratum <= {remainder} THEN {quota + 1} ELSE {quota} END
        """
    )

    _top_up_short_departments(con, scope)

    con.execute("DELETE FROM sales_wide WHERE item_id NOT IN (SELECT item_id FROM sampled_items)")
    con.execute("DELETE FROM sales_long WHERE item_id NOT IN (SELECT item_id FROM sampled_items)")

    rows = con.execute("""
        SELECT dept_id, stratum, count(*), min(zero_share), avg(zero_share), max(zero_share)
        FROM sampled_items GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()
    return tuple(StratumSummary(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows)


def _top_up_short_departments(con: duckdb.DuckDBPyConnection, scope: Scope) -> None:
    """Backfill departments whose strata could not meet their quota.

    A department with fewer items than ``items_per_dept`` cannot be filled, and
    a thin stratum leaves the department short. Rather than silently returning a
    smaller sample, pull the shortfall from that department's unselected items
    in the same deterministic hash order.
    """
    short = con.execute(
        """
        SELECT dept_id, ? - count(*) AS shortfall
        FROM sampled_items GROUP BY 1 HAVING count(*) < ?
        """,
        [scope.items_per_dept, scope.items_per_dept],
    ).fetchall()

    for dept_id, shortfall in short:
        con.execute(
            f"""
            INSERT INTO sampled_items
            SELECT dept_id,
                   ntile({scope.n_strata}) OVER (ORDER BY zero_share, item_id) AS stratum,
                   item_id, zero_share
            FROM (
                SELECT i.*,
                       row_number() OVER (ORDER BY hash(i.item_id || '#' || {scope.sampling_seed}))
                           AS pick
                FROM item_intermittency i
                WHERE i.dept_id = ?
                  AND i.item_id NOT IN (SELECT item_id FROM sampled_items)
            )
            WHERE pick <= ?
            """,
            [dept_id, shortfall],
        )


def _build_fact(con: duckdb.DuckDBPyConnection, scope: Scope) -> None:
    """Join long sales to calendar and prices, then insert into ``fact_sales``."""
    snap_column = f"snap_{scope.state_id}"

    con.execute(FACT_SALES_DDL)
    con.execute(f"""
        INSERT INTO {FACT_SALES}
        SELECT
            c.date,
            w.item_id,
            w.store_id,
            w.dept_id,
            w.cat_id,
            w.state_id,
            CAST(l.units AS INTEGER)  AS units,
            p.sell_price              AS price,
            CAST(c.{snap_column} AS TINYINT) AS snap,
            c.event_name_1            AS event
        FROM sales_long l
        JOIN sales_wide w USING (id)
        JOIN {DIM_CALENDAR} c USING (d)
        LEFT JOIN prices p
               ON p.store_id = w.store_id
              AND p.item_id  = w.item_id
              AND p.wm_yr_wk = c.wm_yr_wk
    """)


def _summarise(
    con: duckdb.DuckDBPyConnection,
    scope: Scope,
    elapsed: float,
    strata: tuple[StratumSummary, ...] = (),
) -> LoadReport:
    """Compute the post-load summary statistics."""
    row = con.execute(f"""
        SELECT
            count(*),
            count(DISTINCT (item_id, store_id)),
            count(DISTINCT item_id),
            count(DISTINCT store_id),
            count(DISTINCT dept_id),
            CAST(min(date) AS VARCHAR),
            CAST(max(date) AS VARCHAR),
            avg(CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END),
            avg(CASE WHEN price IS NULL THEN 1.0 ELSE 0.0 END)
        FROM {FACT_SALES}
    """).fetchone()
    return LoadReport(
        scope=scope.describe(),
        rows=row[0],
        series=row[1],
        items=row[2],
        stores=row[3],
        depts=row[4],
        date_min=row[5],
        date_max=row[6],
        zero_unit_share=row[7],
        null_price_share=row[8],
        elapsed_seconds=elapsed,
        strata=strata,
    )


def build_warehouse(
    data_dir: Path = DATA_DIR,
    db_path: Path = WAREHOUSE_PATH,
    scope: Scope = DEFAULT_SCOPE,
    *,
    overwrite: bool = False,
) -> LoadReport:
    """Build the DuckDB warehouse from the raw M5 CSVs.

    Args:
        data_dir: Directory holding the three raw Kaggle CSVs.
        db_path: Destination DuckDB file. Created if absent.
        scope: Which slice of M5 to load. See :class:`~inventory_engine.config.Scope`.
        overwrite: Drop and rebuild ``fact_sales``/``dim_calendar`` if they exist.
            Without this, building over a populated warehouse raises.

    Returns:
        A :class:`LoadReport` describing what landed.

    Raises:
        MissingRawDataError: If any raw CSV is absent.
        ValueError: If the scope matches no series, or the warehouse is already
            populated and ``overwrite`` is False.

    """
    # Validate the scope before touching disk, so an unusable scope fails with a
    # message about the scope rather than as a downstream "matched zero series".
    if scope.state_id not in VALID_STATES:
        raise ValueError(
            f"Unknown state_id {scope.state_id!r}; expected one of {sorted(VALID_STATES)}"
        )

    files = verify_raw_files(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    con = duckdb.connect(str(db_path))
    in_transaction = False
    try:
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        clashes = existing & {FACT_SALES, DIM_CALENDAR}
        if clashes and not overwrite:
            raise ValueError(
                f"{db_path} already contains {sorted(clashes)}. "
                "Pass overwrite=True (or --overwrite) to rebuild."
            )

        con.execute("BEGIN TRANSACTION")
        in_transaction = True
        con.execute(f"DROP TABLE IF EXISTS {FACT_SALES}")
        con.execute(f"DROP TABLE IF EXISTS {DIM_CALENDAR}")

        _load_calendar(con, files[RAW_CALENDAR_FILE])
        _select_series(con, files[RAW_SALES_FILE], scope)
        _unpivot_to_long(con)
        _load_prices(con, files[RAW_PRICES_FILE])
        strata: tuple[StratumSummary, ...] = ()
        if scope.items_per_dept is not None:
            strata = _apply_stratified_sample(con, scope)
        _build_fact(con, scope)
        con.execute("COMMIT")
        in_transaction = False

        return _summarise(con, scope, time.perf_counter() - started, strata)
    except Exception:
        # Roll back only if we actually opened a transaction; otherwise the
        # rollback itself raises and masks the real error.
        if in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``build-warehouse [--overwrite] [--dept ...]``."""
    parser = argparse.ArgumentParser(description="Build the M5 DuckDB warehouse (Phase 1).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--state", default=DEFAULT_SCOPE.state_id)
    parser.add_argument("--category", default=DEFAULT_SCOPE.cat_id)
    parser.add_argument(
        "--dept",
        action="append",
        dest="depts",
        help="Restrict to a department (repeatable), e.g. --dept FOODS_1",
    )
    parser.add_argument(
        "--items-per-dept",
        type=int,
        default=DEFAULT_SCOPE.items_per_dept,
        help="Items to sample per department, stratified by intermittency.",
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Keep every item in scope instead of sampling.",
    )
    parser.add_argument("--strata", type=int, default=DEFAULT_SCOPE.n_strata)
    parser.add_argument("--seed", type=int, default=DEFAULT_SCOPE.sampling_seed)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    scope = Scope(
        state_id=args.state,
        cat_id=args.category,
        dept_ids=tuple(args.depts) if args.depts else None,
        items_per_dept=None if args.all_items else args.items_per_dept,
        n_strata=args.strata,
        sampling_seed=args.seed,
    )

    try:
        report = build_warehouse(
            data_dir=args.data_dir,
            db_path=args.db_path,
            scope=scope,
            overwrite=args.overwrite,
        )
    except (MissingRawDataError, ValueError) as exc:
        print(f"\nBuild failed: {exc}\n", file=sys.stderr)
        return 1

    print(f"\nWarehouse built at {args.db_path}\n{report.render()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
