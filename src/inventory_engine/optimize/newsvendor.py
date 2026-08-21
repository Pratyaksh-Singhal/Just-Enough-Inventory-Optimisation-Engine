"""E7-S2 — newsvendor order quantities from the forecast distribution."""

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
    """Interpolate the forecast at quantile ``target`` from a fitted grid."""
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
    """Load monotonized quantile forecasts joined to realised demand and price."""
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
    """Compute the newsvendor order quantity for every forecast row."""
    out = panel.copy()
    cr = costs.critical_ratio()
    out["cu"] = costs.understock_cost(out["price"])
    out["co"] = costs.overstock_cost(out["price"])
    out["critical_ratio"] = cr
    out["order_qty"] = np.clip(interpolate_quantile(levels, out[levels].to_numpy(), cr), 0.0, None)
    return out


def quantile_bin_probabilities(levels: np.ndarray) -> np.ndarray:
    """Probability mass assigned to each fitted quantile under the midpoint rule."""
    levels = np.asarray(levels, dtype=float)
    edges = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2.0, [1.0]])
    return np.diff(edges)


def expected_cost_from_distribution(
    levels: np.ndarray, values: np.ndarray, order_qty: float, costs: CostModel, price: float
) -> float:
    """Return the expected newsvendor cost of ``order_qty``."""
    probs = quantile_bin_probabilities(levels)
    values = np.asarray(values, dtype=float)
    shortfall = np.maximum(values - order_qty, 0.0)
    leftover = np.maximum(order_qty - values, 0.0)
    unit_cost = costs.unit_cost(price)
    per_scenario = shortfall * costs.understock_cost(price) + leftover * unit_cost * (
        costs.spoilage_rate + costs.holding_rate
    )
    return float(np.sum(probs * per_scenario))


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
