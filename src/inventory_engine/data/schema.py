"""Table contracts for the DuckDB warehouse.

Phases communicate through these tables rather than through function calls, so each one is
a contract: a phase is done when its tables exist and match the shape declared here. See
``docs/BACKLOG.md`` for which phase writes and reads each.

``fact_sales``
    The long-format panel every downstream phase reads. One row per
    (date, item_id, store_id). This is the shape the brief specifies and the
    only table feature engineering should touch.

``dim_calendar``
    The *full* M5 calendar, including the 28 days that extend past the end of
    the sales data. Feature engineering needs those future rows to compute
    "days until next event" for observations near the end of the training
    window; truncating the calendar to the sales range would silently produce
    NULLs there. This table is future-facing by design and is the one place
    where reading "ahead" of ``t`` is legitimate, because event calendars are
    genuinely known in advance.

``dim_item_stratum``
    Each item's intermittency band. Written once by Phase 1 and read by every phase that
    reports accuracy per band, so the classification is never recomputed with a second
    copy of the window logic that could drift from the first.

``forecast``
    Every forecast in the project -- baselines, the GBM, point and quantile, pre- and
    post-reconciliation -- in one table, discriminated by columns.

``backtest_fold_metrics``
    Fold-level scores, long format.
"""

from __future__ import annotations

from typing import Final

FACT_SALES: Final = "fact_sales"
DIM_CALENDAR: Final = "dim_calendar"
DIM_ITEM_STRATUM: Final = "dim_item_stratum"
FORECAST: Final = "forecast"
BACKTEST_METRICS: Final = "backtest_fold_metrics"

#: Column order of ``fact_sales``, exactly as specified in the project brief.
FACT_SALES_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    "units",
    "price",
    "snap",
    "event",
)

FACT_SALES_DDL: Final = f"""
CREATE TABLE {FACT_SALES} (
    date      DATE     NOT NULL,
    item_id   VARCHAR  NOT NULL,
    store_id  VARCHAR  NOT NULL,
    dept_id   VARCHAR  NOT NULL,
    cat_id    VARCHAR  NOT NULL,
    state_id  VARCHAR  NOT NULL,
    units     INTEGER  NOT NULL,
    price     DOUBLE,            -- NULL => item not listed at this store that week
    snap      TINYINT  NOT NULL, -- SNAP benefit day for this item's state
    event     VARCHAR,           -- primary calendar event name, NULL if none
    PRIMARY KEY (date, item_id, store_id)
);
"""

DIM_CALENDAR_DDL: Final = f"""
CREATE TABLE {DIM_CALENDAR} (
    date          DATE    NOT NULL PRIMARY KEY,
    d             VARCHAR NOT NULL,
    wm_yr_wk      INTEGER NOT NULL,
    wday          TINYINT NOT NULL,
    weekday       VARCHAR NOT NULL,
    month         TINYINT NOT NULL,
    year          SMALLINT NOT NULL,
    event_name_1  VARCHAR,
    event_type_1  VARCHAR,
    event_name_2  VARCHAR,
    event_type_2  VARCHAR,
    snap_CA       TINYINT NOT NULL,
    snap_TX       TINYINT NOT NULL,
    snap_WI       TINYINT NOT NULL
);
"""

#: One table carries every forecast in the project. Phases discriminate on columns rather
#: than on table names, so adding a model never forces a downstream schema change.
#:
#: No primary key: a (model, fold, series, target_date) legitimately appears more than once
#: -- once as a base forecast and again reconciled, and quantile rows multiply it further.
#: Uniqueness is asserted per-write, where the intended grain is known.
FORECAST_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {FORECAST} (
    run_id       VARCHAR NOT NULL,  -- MLflow run id, or 'baseline:<name>'
    model_name   VARCHAR NOT NULL,
    fold         INTEGER,           -- 0..4; NULL for production forecasts
    origin_date  DATE    NOT NULL,  -- last date of training data
    target_date  DATE    NOT NULL,
    horizon      INTEGER NOT NULL,  -- target_date - origin_date, in days
    level        VARCHAR NOT NULL,  -- 'item_store' | 'dept_store' | 'store' | 'state' ...
    item_id      VARCHAR,           -- NULL above item level
    store_id     VARCHAR,           -- NULL above store level
    dept_id      VARCHAR,
    quantile     DOUBLE,            -- NULL for a point forecast
    yhat         DOUBLE  NOT NULL,
    reconciled   BOOLEAN NOT NULL DEFAULT FALSE
);
"""

#: Long format so a new metric never needs a migration, and the dashboard can pivot
#: whichever way it likes.
BACKTEST_METRICS_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {BACKTEST_METRICS} (
    run_id      VARCHAR NOT NULL,
    model_name  VARCHAR NOT NULL,
    fold        INTEGER NOT NULL,
    level       VARCHAR NOT NULL,
    stratum     VARCHAR,            -- 'dense' | 'mid' | 'sparse' | NULL for overall
    horizon     INTEGER,            -- NULL when aggregated across the horizon
    metric      VARCHAR NOT NULL,
    value       DOUBLE,
    n_series    INTEGER,            -- series contributing
    n_excluded  INTEGER             -- series with no defined scale, excluded
);
"""
