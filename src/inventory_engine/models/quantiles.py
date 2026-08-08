"""Monotone quantile reads.

LightGBM fits each quantile level as an independent model, and nothing constrains those
fits to be ordered. Roughly 1.3% of rows come out with q0.90 above q0.95 — squarely inside
the CR 0.5–0.95 band E7's newsvendor rule selects from, so a critical ratio landing there
would read a lower stocking quantity for a *higher* service level. Silently wrong, in the
direction that causes stockouts.

The fix is applied at **read** time, not at write time. Stored forecasts keep the raw fitted
values, so the crossing rate stays measurable and auditable; every consumer that needs
ordered quantiles calls :func:`monotonize` or :func:`read_quantiles` and gets them.

Sorting, not isotonic regression
--------------------------------
The operator here is a **rearrangement** — sort each row's quantile values ascending — and
that choice is deliberate. Chernozhukov, Fernández-Val and Galichon (2010) prove
rearrangement of a crossed quantile curve *weakly reduces* pinball loss at every level: it
can only help. Pool-adjacent-violators isotonic regression would instead minimise squared
deviation from the raw predictions, which carries no such guarantee for the loss these
forecasts are actually judged by. Both produce monotone output; only one is provably
non-worsening for the metric that matters, so the empirical check in
:func:`crossing_rate` is paired with a pinball comparison rather than taken on faith.
"""

from __future__ import annotations

from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.data.schema import FORECAST

#: Columns identifying one forecast row across quantile levels.
GRAIN: Final[tuple[str, ...]] = ("fold", "item_id", "store_id", "target_date")


def monotonize(frame: pd.DataFrame, *, value: str = "yhat") -> pd.DataFrame:
    """Return ``frame`` with quantile forecasts sorted ascending within each row.

    Args:
        frame: Long-format quantile forecasts with :data:`GRAIN` columns, a ``quantile``
            column and a value column.
        value: Name of the forecast column.

    Returns:
        The same rows, with values rearranged so that quantile order is respected. Row
        count and grain are unchanged; only the assignment of values to levels moves.

    """
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
    """Count rows whose quantile forecasts are not monotonically non-decreasing.

    Args:
        frame: Long-format quantile forecasts.
        value: Name of the forecast column.
        within: Restrict the check to these quantile levels. E7 selects inside
            CR 0.5–0.95, so the rate over that sub-range is the one that governs whether
            an ordering decision can be wrong.

    Returns:
        ``(crossings, rows_checked)``.

    """
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
    """Read stored quantile forecasts, ordered by default.

    This is the accessor E7 should use. Reading ``forecast`` directly returns the raw
    fitted values, crossings included.

    Args:
        con: Open warehouse connection.
        model_name: Model to read.
        fold: Restrict to one fold; ``None`` reads all.
        reconciled: Whether to read post-MinT rows.
        monotone: Apply the rearrangement. ``False`` returns raw values, which is what the
            crossing-rate audit uses.

    Returns:
        Long-format quantile forecasts.

    """
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
