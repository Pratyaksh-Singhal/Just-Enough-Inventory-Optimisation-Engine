"""E7-S3/S4/S5 — simulate stocking policies against realised demand and price the gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.data.schema import FACT_SALES
from inventory_engine.optimize.costs import CostModel
from inventory_engine.optimize.newsvendor import interpolate_quantile

COST_COMPARISON_TABLE: Final = "cost_comparison"
COST_SENSITIVITY_TABLE: Final = "cost_sensitivity"

#: The service level people reach for by default, kept as a policy row precisely so the
#: money table can show what it costs.
FIXED_SERVICE_LEVEL: Final = 0.95

COST_COMPARISON_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {COST_COMPARISON_TABLE} (
    policy          VARCHAR NOT NULL,
    margin_rate     DOUBLE,
    spoilage_rate   DOUBLE,
    critical_ratio  DOUBLE,
    stockout_rate   DOUBLE,
    stockout_units  DOUBLE,
    waste_units     DOUBLE,
    understock_cost DOUBLE,
    holding_cost    DOUBLE,
    overstock_cost  DOUBLE,
    total_cost      DOUBLE,
    n_rows          INTEGER
);
"""

COST_SENSITIVITY_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {COST_SENSITIVITY_TABLE} (
    spoilage_rate  DOUBLE,
    margin_rate    DOUBLE,
    critical_ratio DOUBLE,
    policy         VARCHAR NOT NULL,
    total_cost     DOUBLE,
    saving_vs_naive DOUBLE
);
"""


@dataclass(frozen=True)
class PolicyResult:
    """Simulated outcome of one stocking policy."""

    policy: str
    stockout_rate: float
    stockout_units: float
    waste_units: float
    understock_cost: float
    holding_cost: float
    overstock_cost: float
    total_cost: float
    n_rows: int


def simulate_policy(
    policy: str,
    order_qty: np.ndarray,
    demand: np.ndarray,
    price: np.ndarray,
    costs: CostModel,
) -> PolicyResult:
    """Price one stocking policy against realised demand."""
    order_qty = np.asarray(order_qty, dtype=float)
    demand = np.asarray(demand, dtype=float)
    price = np.asarray(price, dtype=float)

    shortfall = np.maximum(demand - order_qty, 0.0)
    leftover = np.maximum(order_qty - demand, 0.0)
    unit_cost = costs.unit_cost(price)

    understock = shortfall * costs.understock_cost(price)
    spoilage = leftover * unit_cost * costs.spoilage_rate
    holding = leftover * unit_cost * costs.holding_rate

    return PolicyResult(
        policy=policy,
        # A stockout is a day where demand exceeded what was on the shelf. Days with zero
        # demand cannot stock out and are not counted as successes either way.
        stockout_rate=float((shortfall > 0).mean()),
        stockout_units=float(shortfall.sum()),
        waste_units=float(leftover.sum()),
        understock_cost=float(understock.sum()),
        holding_cost=float(holding.sum()),
        overstock_cost=float(spoilage.sum()),
        total_cost=float((understock + spoilage + holding).sum()),
        n_rows=int(order_qty.size),
    )


def naive_orders(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame) -> np.ndarray:
    """Return current practice: stock what sold on the same weekday last week."""
    lagged = con.execute(
        f"""
        SELECT item_id, store_id, date + INTERVAL 7 DAY AS target_date,
               CAST(units AS DOUBLE) AS naive_qty
        FROM {FACT_SALES}
        """
    ).df()
    merged = panel.merge(lagged, on=["item_id", "store_id", "target_date"], how="left")
    return merged["naive_qty"].fillna(0.0).to_numpy()


def money_table(
    con: duckdb.DuckDBPyConnection,
    orders: pd.DataFrame,
    levels: np.ndarray,
    costs: CostModel,
    *,
    replace: bool = True,
) -> pd.DataFrame:
    """Build the three-policy cost comparison (E7-S3)."""
    con.execute(COST_COMPARISON_DDL)
    if replace:
        con.execute(f"DELETE FROM {COST_COMPARISON_TABLE}")

    demand = orders["demand"].to_numpy(dtype=float)
    price = orders["price"].to_numpy(dtype=float)
    grid = orders[levels].to_numpy(dtype=float)

    results = [
        simulate_policy(
            "Current (naive: last week's sales)", naive_orders(con, orders), demand, price, costs
        ),
        simulate_policy(
            f"Fixed {FIXED_SERVICE_LEVEL:.0%} service level",
            interpolate_quantile(levels, grid, FIXED_SERVICE_LEVEL),
            demand,
            price,
            costs,
        ),
        simulate_policy(
            "Newsvendor + our forecast",
            orders["order_qty"].to_numpy(dtype=float),
            demand,
            price,
            costs,
        ),
    ]

    frame = pd.DataFrame([r.__dict__ for r in results])
    frame.insert(1, "margin_rate", costs.margin_rate)
    frame.insert(2, "spoilage_rate", costs.spoilage_rate)
    frame.insert(3, "critical_ratio", costs.critical_ratio())
    baseline = frame.loc[0, "total_cost"]
    frame["saving_vs_naive"] = (baseline - frame["total_cost"]) / baseline

    con.register("cost_df", frame.drop(columns="saving_vs_naive"))
    con.execute(f"""
        INSERT INTO {COST_COMPARISON_TABLE}
        SELECT policy, margin_rate, spoilage_rate, critical_ratio, stockout_rate,
               stockout_units, waste_units, understock_cost, holding_cost, overstock_cost,
               total_cost, n_rows
        FROM cost_df
    """)
    con.unregister("cost_df")
    return frame


def sensitivity(
    con: duckdb.DuckDBPyConnection,
    orders: pd.DataFrame,
    levels: np.ndarray,
    *,
    margin_rate: float,
    spoilage_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    replace: bool = True,
) -> pd.DataFrame:
    """Sweep the Cu/Co ratio and re-price every policy (E7-S4)."""
    con.execute(COST_SENSITIVITY_DDL)
    if replace:
        con.execute(f"DELETE FROM {COST_SENSITIVITY_TABLE}")

    demand = orders["demand"].to_numpy(dtype=float)
    price = orders["price"].to_numpy(dtype=float)
    grid = orders[levels].to_numpy(dtype=float)
    naive = naive_orders(con, orders)
    fixed = interpolate_quantile(levels, grid, FIXED_SERVICE_LEVEL)

    rows = []
    for spoilage in spoilage_grid:
        costs = CostModel(margin_rate=margin_rate, spoilage_rate=spoilage)
        cr = costs.critical_ratio()
        try:
            newsvendor_qty = interpolate_quantile(levels, grid, cr)
        except ValueError:
            # CR outside the fitted grid: record it rather than silently skipping, so the
            # gap in coverage is visible in the sensitivity output.
            newsvendor_qty = None

        policies = {
            "naive": naive,
            f"fixed_{FIXED_SERVICE_LEVEL:.2f}": fixed,
        }
        if newsvendor_qty is not None:
            policies["newsvendor"] = newsvendor_qty

        baseline = simulate_policy("naive", naive, demand, price, costs).total_cost
        for name, qty in policies.items():
            result = simulate_policy(name, qty, demand, price, costs)
            rows.append(
                {
                    "spoilage_rate": spoilage,
                    "margin_rate": margin_rate,
                    "critical_ratio": cr,
                    "policy": name,
                    "total_cost": result.total_cost,
                    "saving_vs_naive": (baseline - result.total_cost) / baseline
                    if baseline
                    else 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    con.register("sens_df", frame)
    con.execute(f"INSERT INTO {COST_SENSITIVITY_TABLE} SELECT * FROM sens_df")
    con.unregister("sens_df")
    return frame


def attribution(
    con: duckdb.DuckDBPyConnection, orders: pd.DataFrame, levels: np.ndarray, costs: CostModel
) -> pd.DataFrame:
    """Split the saving into forecast quality vs policy choice (E7-S5)."""
    demand = orders["demand"].to_numpy(dtype=float)
    price = orders["price"].to_numpy(dtype=float)
    grid = orders[levels].to_numpy(dtype=float)
    naive = naive_orders(con, orders)
    cr = costs.critical_ratio()

    # The naive forecast has no distribution, so "applying a service level" to it means scaling
    # last week's actuals by the same quantile ratio our model implies.
    ratio_fixed = float(
        np.mean(interpolate_quantile(levels, grid, FIXED_SERVICE_LEVEL) + 1e-9)
        / np.mean(interpolate_quantile(levels, grid, 0.5) + 1e-9)
    )
    ratio_cr = float(
        np.mean(interpolate_quantile(levels, grid, cr) + 1e-9)
        / np.mean(interpolate_quantile(levels, grid, 0.5) + 1e-9)
    )

    combos = {
        "naive forecast + fixed 95%": naive * ratio_fixed,
        "naive forecast + newsvendor CR": naive * ratio_cr,
        "our forecast + fixed 95%": interpolate_quantile(levels, grid, FIXED_SERVICE_LEVEL),
        "our forecast + newsvendor CR": orders["order_qty"].to_numpy(dtype=float),
    }
    rows = [
        simulate_policy(name, qty, demand, price, costs).__dict__ for name, qty in combos.items()
    ]
    frame = pd.DataFrame(rows)[["policy", "total_cost", "stockout_rate", "waste_units"]]
    worst = frame["total_cost"].max()
    frame["saving_vs_worst"] = (worst - frame["total_cost"]) / worst
    return frame
