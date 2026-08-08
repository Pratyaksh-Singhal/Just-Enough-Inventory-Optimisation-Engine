"""E7-S2 — newsvendor order quantities from the forecast distribution.

Which forecast this consumes, and why
-------------------------------------
**Base LightGBM quantiles at ``item_store``, deliberately not the MinT-reconciled ones.**

This is a decision, not a default, and it is worth stating because the reconciled forecasts
were computed more recently and would be the lazy pick.

E6 measured what reconciliation costs: every aggregate level improves, and the bottom level
degrades ~1.5% on MASE (1.0192 -> 1.0348) and ~0.5% on RMSSE. MinT trades bottom-level
accuracy for cross-level coherence. That is a good trade for a planning view, where a
regional number that disagrees with the sum of its stores is useless.

It is the wrong trade here. **Orders are placed at ``item_store``.** Every unit of stockout
and every unit of spoilage is realised at that grain, so the cost function should optimise
against the most accurate forecast available *there* — which is the unreconciled one.
Coherence with the store total buys nothing at the moment of deciding how many units of one
SKU to put on one shelf.

So: newsvendor consumes base forecasts; E9's aggregate and planning views consume
reconciled ones. Both are in the ``forecast`` table, discriminated by ``reconciled``.

Interpolating the critical ratio
--------------------------------
LightGBM fits a fixed grid of quantiles, but ``CR`` is a continuous function of the cost
assumptions and will rarely land on a fitted level. The order quantity is therefore linearly
interpolated between the two bracketing quantiles.

The grid was extended downward to 0.10 for exactly this reason: with fresh-food spoilage,
``CR`` lands between 0.25 and 0.63, **below** the 0.5 the brief's original grid started at.
Every order would have been clamped to the median, and the headline finding — that optimal
service level is far below 95% — could not have been demonstrated at all.
"""

from __future__ import annotations

from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.data.schema import FACT_SALES, FORECAST
from inventory_engine.models.quantiles import monotonize
from inventory_engine.optimize.costs import FALLBACK_PRICE, CostModel

#: The forecast E7 orders against. See the module docstring -- this is an explicit choice.
USE_RECONCILED: Final = False
ORDER_LEVEL: Final = "item_store"
ORDER_POLICY_TABLE: Final = "order_policy"

ORDER_POLICY_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {ORDER_POLICY_TABLE} (
    model_name     VARCHAR NOT NULL,
    fold           INTEGER NOT NULL,
    item_id        VARCHAR NOT NULL,
    store_id       VARCHAR NOT NULL,
    target_date    DATE    NOT NULL,
    price          DOUBLE,
    cu             DOUBLE,
    co             DOUBLE,
    critical_ratio DOUBLE,
    order_qty      DOUBLE,
    demand         DOUBLE
);
"""


def interpolate_quantile(levels: np.ndarray, values: np.ndarray, target: float) -> np.ndarray:
    """Interpolate the forecast at quantile ``target`` from a fitted grid.

    Args:
        levels: Fitted quantile levels, ascending. Shape ``(n_levels,)``.
        values: Forecasts at those levels. Shape ``(n_rows, n_levels)``.
        target: The quantile to interpolate, typically the critical ratio.

    Returns:
        Interpolated forecasts, shape ``(n_rows,)``.

    Raises:
        ValueError: If ``target`` falls outside the fitted grid, which would mean
            extrapolating a tail the model was never asked to estimate.

    """
    levels = np.asarray(levels, dtype=float)
    if not levels.size or target < levels.min() - 1e-9 or target > levels.max() + 1e-9:
        raise ValueError(
            f"critical ratio {target:.4f} is outside the fitted quantile grid "
            f"[{levels.min():.2f}, {levels.max():.2f}]. Extrapolating here would invent a "
            "tail the model never estimated; refit with a wider grid instead."
        )
    # Row-wise linear interpolation across the quantile axis.
    return np.array([np.interp(target, levels, row) for row in np.asarray(values, dtype=float)])


def load_quantile_panel(
    con: duckdb.DuckDBPyConnection,
    model_name: str = "lgbm",
    *,
    reconciled: bool = USE_RECONCILED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load monotonized quantile forecasts joined to realised demand and price.

    Returns:
        ``(panel, levels)`` where ``panel`` is wide over quantile levels and carries
        ``demand`` and ``price``, and ``levels`` is the ascending fitted grid.

    """
    raw = con.execute(
        f"""
        SELECT fold, item_id, store_id, target_date, quantile, yhat
        FROM {FORECAST}
        WHERE model_name = ? AND level = ? AND quantile IS NOT NULL AND reconciled = ?
        """,
        [model_name, ORDER_LEVEL, reconciled],
    ).df()
    if raw.empty:
        raise ValueError(
            f"no quantile forecasts for {model_name} (reconciled={reconciled}); "
            "run `run-gbm` first."
        )

    ordered = monotonize(raw)
    wide = ordered.pivot_table(
        index=["fold", "item_id", "store_id", "target_date"], columns="quantile", values="yhat"
    ).reset_index()

    actuals = con.execute(
        f"""
        SELECT item_id, store_id, date AS target_date,
               CAST(units AS DOUBLE) AS demand, price
        FROM {FACT_SALES}
        """
    ).df()
    panel = wide.merge(actuals, on=["item_id", "store_id", "target_date"], how="inner")
    panel["price"] = panel["price"].fillna(FALLBACK_PRICE)

    levels = np.array(sorted(c for c in wide.columns if isinstance(c, float)), dtype=float)
    return panel, levels


def order_quantities(panel: pd.DataFrame, levels: np.ndarray, costs: CostModel) -> pd.DataFrame:
    """Compute the newsvendor order quantity for every forecast row.

    Args:
        panel: Wide quantile panel from :func:`load_quantile_panel`.
        levels: Fitted quantile grid.
        costs: The cost assumptions.

    Returns:
        ``panel`` with ``cu``, ``co``, ``critical_ratio`` and ``order_qty`` added.

    """
    out = panel.copy()
    cr = costs.critical_ratio()
    out["cu"] = costs.understock_cost(out["price"])
    out["co"] = costs.overstock_cost(out["price"])
    out["critical_ratio"] = cr
    out["order_qty"] = np.clip(interpolate_quantile(levels, out[levels].to_numpy(), cr), 0.0, None)
    return out


def persist_policy(
    con: duckdb.DuckDBPyConnection, orders: pd.DataFrame, model_name: str, *, replace: bool = True
) -> int:
    """Write order quantities to ``order_policy``."""
    con.execute(ORDER_POLICY_DDL)
    if replace:
        con.execute(f"DELETE FROM {ORDER_POLICY_TABLE} WHERE model_name = ?", [model_name])

    frame = orders.assign(model_name=model_name)[
        [
            "model_name",
            "fold",
            "item_id",
            "store_id",
            "target_date",
            "price",
            "cu",
            "co",
            "critical_ratio",
            "order_qty",
            "demand",
        ]
    ]
    con.register("policy_df", frame)
    con.execute(f"INSERT INTO {ORDER_POLICY_TABLE} SELECT * FROM policy_df")
    con.unregister("policy_df")
    return len(frame)
