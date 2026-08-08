"""CLI: build order policies, the money table, sensitivity and attribution (E7)."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import duckdb
import pandas as pd

from inventory_engine.config import WAREHOUSE_PATH
from inventory_engine.optimize.costs import (
    DEFAULT_MARGIN_RATE,
    DEFAULT_SPOILAGE_RATE,
    CostModel,
)
from inventory_engine.optimize.newsvendor import (
    USE_RECONCILED,
    load_quantile_panel,
    order_quantities,
    persist_policy,
)
from inventory_engine.optimize.simulate import attribution, money_table, sensitivity


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-optimize``."""
    parser = argparse.ArgumentParser(description="Newsvendor optimization layer (E7).")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_RATE)
    parser.add_argument("--spoilage", type=float, default=DEFAULT_SPOILAGE_RATE)
    args = parser.parse_args(argv)

    if not args.db_path.is_file():
        print(f"\nNo warehouse at {args.db_path}. Run `build-warehouse` first.\n", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    costs = CostModel(margin_rate=args.margin, spoilage_rate=args.spoilage)

    con = duckdb.connect(str(args.db_path))
    try:
        source = "MinT-reconciled" if USE_RECONCILED else "base (unreconciled)"
        print(f"\nCost assumptions: {costs.describe()}")
        print(f"Forecast source:  {source} {args.model} quantiles at item_store level")
        print("  -> orders are placed at item_store, so the cost function optimises against")
        print("     the most accurate forecast at that grain. See E6's trade-off table.\n")

        panel, levels = load_quantile_panel(con, args.model)
        print(f"Quantile grid: {[round(v, 2) for v in levels]}")
        orders = order_quantities(panel, levels, costs)
        rows = persist_policy(con, orders, args.model)
        print(f"Order policy rows: {rows:,}\n")

        print("=== THE MONEY TABLE ===")
        table = money_table(con, orders, levels, costs)
        print(
            table[
                [
                    "policy",
                    "stockout_rate",
                    "waste_units",
                    "holding_cost",
                    "total_cost",
                    "saving_vs_naive",
                ]
            ].to_string(index=False)
        )

        print("\n=== Sensitivity: spoilage rate -> critical ratio -> total cost ===")
        sens = sensitivity(con, orders, levels, margin_rate=args.margin)
        pivot = sens.pivot_table(
            index=["spoilage_rate", "critical_ratio"], columns="policy", values="total_cost"
        )
        print(pivot.round(0).to_string())

        print("\n=== Savings attribution: forecast quality vs policy choice ===")
        print(attribution(con, orders, levels, costs).to_string(index=False))
        print()
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
