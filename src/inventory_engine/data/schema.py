"""Table contracts for the DuckDB warehouse."""

from __future__ import annotations

from typing import Final

FACT_SALES: Final = "fact_sales"
DIM_CALENDAR: Final = "dim_calendar"
DIM_ITEM_STRATUM: Final = "dim_item_stratum"
FORECAST: Final = "forecast"
BACKTEST_METRICS: Final = "backtest_fold_metrics"
FEATURE_IMPORTANCE: Final = "feature_importance"

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

#: One table carries every forecast in the project.
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

#: SHAP and gain importance, precomputed. SHAP over 1M rows is far too slow to run per
#: API request, so E9's "why this forecast?" panel reads this table instead.
FEATURE_IMPORTANCE_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {FEATURE_IMPORTANCE} (
    run_id      VARCHAR NOT NULL,
    model_name  VARCHAR NOT NULL,
    fold        INTEGER,
    feature     VARCHAR NOT NULL,
    method      VARCHAR NOT NULL,  -- 'shap_mean_abs' | 'gain' | 'split'
    value       DOUBLE  NOT NULL
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
