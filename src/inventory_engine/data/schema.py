"""Table contracts for the DuckDB warehouse.

Two tables, deliberately:

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
"""

from __future__ import annotations

from typing import Final

FACT_SALES: Final = "fact_sales"
DIM_CALENDAR: Final = "dim_calendar"

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
