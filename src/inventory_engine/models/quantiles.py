"""Monotone quantile reads."""

from __future__ import annotations

from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.data.schema import FORECAST

#: Columns identifying one forecast row across quantile levels.
GRAIN: Final[tuple[str, ...]] = ("fold", "item_id", "store_id", "target_date")


def monotonize(frame: pd.DataFrame, *, value: str = "yhat") -> pd.DataFrame:
    """Return ``frame`` with quantile forecasts sorted ascending within each row."""
    if frame.empty:
        return frame.copy()

    wide = frame.pivot_table(index=list(GRAIN), columns="quantile", values=value)
    levels = sorted(wide.columns)
    ordered = np.sort(wide[levels].to_numpy(), axis=1)
    fixed = pd.DataFrame(ordered, index=wide.index, columns=levels)
    # Naming the column axis makes `stack` produce a correctly named level directly.
    fixed.columns.name = "quantile"
    return fixed.stack().rename(value).reset_index()


def crossing_rate(
    frame: pd.DataFrame, *, value: str = "yhat", within: tuple[float, ...] | None = None
) -> tuple[int, int]:
    """Count rows whose quantile forecasts are not monotonically non-decreasing."""
    if frame.empty:
        return 0, 0
    wide = frame.pivot_table(index=list(GRAIN), columns="quantile", values=value)
    levels = sorted(wide.columns if within is None else [q for q in within if q in wide.columns])
    if len(levels) < 2:
        return 0, 0
    values = wide[levels].to_numpy()
    return int((np.diff(values, axis=1) < -1e-9).any(axis=1).sum()), int(len(wide))


def read_quantiles(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    *,
    fold: int | None = None,
    reconciled: bool = False,
    monotone: bool = True,
) -> pd.DataFrame:
    """Read stored quantile forecasts, ordered by default."""
    clauses = ["model_name = ?", "quantile IS NOT NULL", "reconciled = ?"]
    params: list[object] = [model_name, reconciled]
    if fold is not None:
        clauses.append("fold = ?")
        params.append(fold)

    frame = con.execute(
        f"""
        SELECT fold, item_id, store_id, target_date, quantile, yhat
        FROM {FORECAST} WHERE {" AND ".join(clauses)}
        """,
        params,
    ).df()
    return monotonize(frame) if monotone else frame
