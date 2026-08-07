"""Synthetic warehouses for the feature tests.

Two builders, for two different jobs:

:func:`random_panel`
    Six series spanning dense to highly intermittent demand. Used by the leakage gate,
    where realistic variety matters more than predictable values.

:func:`deterministic_panel`
    One series with hand-chosen units. Used by the correctness tests, where the expected
    value of every feature has to be computable by hand.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.data.schema import DIM_CALENDAR_DDL, FACT_SALES_DDL
from inventory_engine.features.build import FEATURE_PANEL, build_features

START = date(2013, 1, 5)


def ts(days_from_start: int) -> pd.Timestamp:
    """Panel date as a Timestamp. DuckDB returns DATE as datetime64; compare like for like."""
    return pd.Timestamp(START + timedelta(days=days_from_start))


def _calendar(n_days: int, event_every: int = 40) -> pd.DataFrame:
    dates = [START + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates)),
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "wm_yr_wk": [11101 + i // 7 for i in range(n_days)],
            "wday": [((d.weekday() + 2) % 7) + 1 for d in dates],
            "weekday": [d.strftime("%A") for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
            "event_name_1": [("Holiday" if i % event_every == 0 else None) for i in range(n_days)],
            "event_type_1": [("National" if i % event_every == 0 else None) for i in range(n_days)],
            "event_name_2": [None] * n_days,
            "event_type_2": [None] * n_days,
            "snap_CA": [1 if (i % 31) < 10 else 0 for i in range(n_days)],
            "snap_TX": [0] * n_days,
            "snap_WI": [0] * n_days,
        }
    )


def _series_rows(
    cal: pd.DataFrame, item_id: str, store_id: str, units: np.ndarray, price
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": cal["date"],
            "item_id": item_id,
            "store_id": store_id,
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "state_id": "CA",
            "units": units.astype("int32"),
            "price": price,
            "snap": cal["snap_CA"].to_numpy().astype("int8"),
            "event": cal["event_name_1"].to_numpy(),
        }
    )


def random_panel(
    n_days: int = 520, n_series: int = 6, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar and rectangular fact panel spanning dense to intermittent demand.

    The intermittency spread matters: ``days_since_last_sale`` and ``zero_share_90``
    depend on the zero/non-zero *pattern* rather than magnitudes, and a panel of uniformly
    dense series would not exercise them.
    """
    rng = np.random.default_rng(seed)
    cal = _calendar(n_days)
    rows = []
    for s in range(n_series):
        p_zero = 0.05 + 0.85 * s / (n_series - 1)
        units = np.where(rng.random(n_days) < p_zero, 0, rng.integers(1, 25, n_days))
        price = np.round(2.0 + 0.25 * np.sin(np.arange(n_days) / 14.0), 2).astype(object)
        # One series is unlisted for its first 30 days, exercising the pre-listing drop.
        if s == 1:
            price[:30] = None
        rows.append(
            _series_rows(cal, f"FOODS_1_{s:03d}", "CA_1" if s % 2 == 0 else "CA_2", units, price)
        )
    return cal, pd.concat(rows, ignore_index=True)


#: Hand-chosen units for the correctness tests. Zeros are placed so that
#: ``days_since_last_sale`` has a non-trivial answer at several offsets.
DETERMINISTIC_UNITS: tuple[int, ...] = tuple(0 if i % 5 == 0 else (i % 11) + 1 for i in range(120))


def deterministic_panel() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """One series with known units, for asserting exact feature values.

    Returns:
        ``(calendar, fact, units)``.
    """
    units = np.array(DETERMINISTIC_UNITS, dtype="int32")
    cal = _calendar(len(units), event_every=25)
    price = np.where(np.arange(len(units)) < 60, 2.00, 3.00).astype(object)
    fact = _series_rows(cal, "FOODS_1_000", "CA_1", units, price)
    return cal, fact, units


def warehouse(fact: pd.DataFrame, cal: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Materialise a synthetic warehouse in memory."""
    con = duckdb.connect(":memory:")
    con.execute(DIM_CALENDAR_DDL)
    con.execute(FACT_SALES_DDL)
    con.register("cal_df", cal)
    con.register("fact_df", fact)
    con.execute("INSERT INTO dim_calendar SELECT * FROM cal_df")
    con.execute("INSERT INTO fact_sales SELECT * FROM fact_df")
    return con


def features(fact: pd.DataFrame, cal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build features over the given panel and return them sorted by grain."""
    con = warehouse(fact, cal)
    try:
        build_features(con, horizon)
        return con.execute(f"SELECT * FROM {FEATURE_PANEL} ORDER BY item_id, store_id, date").df()
    finally:
        con.close()
