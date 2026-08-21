"""E5 CLI: the canonical backtest report."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import duckdb

from inventory_engine.backtest.folds import (
    assert_no_training_leak,
    describe_folds,
    make_folds,
    panel_bounds,
)
from inventory_engine.backtest.score import (
    score_forecasts,
    score_quantile_forecasts,
    summarise,
    summarise_by_horizon,
    summarise_by_stratum,
)
from inventory_engine.config import HORIZON, N_FOLDS, WAREHOUSE_PATH

#: The model carried forward into E6 and E7. The stratum-routing hybrid was investigated
#: and deliberately not adopted -- see the README and E7-S6.
PRODUCTION_MODEL = "lgbm"


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-backtest``."""
    parser = argparse.ArgumentParser(description="Score every model (E5).")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args(argv)

    if not args.db_path.is_file():
        print(f"\nNo warehouse at {args.db_path}. Run `build-warehouse` first.\n", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore")
    con = duckdb.connect(str(args.db_path))
    try:
        first, last = panel_bounds(con)
        folds = make_folds(last, args.folds, args.horizon, first_date=first)
        assert_no_training_leak(folds)
        print(f"\nRolling-origin folds\n{describe_folds(folds)}\n")

        score_forecasts(con, folds)
        score_quantile_forecasts(con, folds)

        for metric in ("mase", "rmsse", "bias"):
            print(f"=== {metric.upper()} across {len(folds)} folds ===")
            print(summarise(con, metric).to_string(index=False))
            print()

        print("=== MASE by intermittency stratum ===")
        print(summarise_by_stratum(con, "mase").to_string(index=False))

        print("\n=== pinball loss on monotonized quantiles ===")
        print(
            con.execute("""
                SELECT model_name, metric,
                       round(avg(value), 5) AS mean,
                       round(stddev_samp(value), 5) AS std
                FROM backtest_fold_metrics
                WHERE metric LIKE 'pinball_q%'
                GROUP BY 1, 2 ORDER BY 2, 3
            """)
            .df()
            .to_string(index=False)
        )

        print(f"\n=== {PRODUCTION_MODEL} MASE decay by horizon (weeks 1-4) ===")
        decay = summarise_by_horizon(con, "mase", models=(PRODUCTION_MODEL,))
        if not decay.empty:
            decay["week"] = ((decay["horizon"] - 1) // 7) + 1
            print(decay.groupby("week")["mean"].mean().round(4).to_string())
        print()
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
