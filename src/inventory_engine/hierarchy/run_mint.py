"""CLI: reconcile forecasts with MinT and report the before/after coherence table."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import duckdb

from inventory_engine.backtest.folds import assert_no_training_leak, make_folds, panel_bounds
from inventory_engine.config import HORIZON, N_FOLDS, WAREHOUSE_PATH
from inventory_engine.hierarchy.evaluate import level_comparison, score_levels
from inventory_engine.hierarchy.mint import MINT_METHOD, MODEL_NAME, reconcile


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-mint``."""
    parser = argparse.ArgumentParser(description="MinT hierarchical reconciliation (E6).")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--method", default=MINT_METHOD)
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

        print(f"\nReconciling {args.model} with MinT ({args.method})...")
        print(reconcile(con, folds, model_name=args.model, method=args.method).render())

        print("\nScoring every level, before and after...")
        score_levels(con, folds, model_name=args.model)

        for metric in ("rmsse", "mase"):
            print(f"\n=== {metric.upper()} by level: base vs MinT ===")
            print(level_comparison(con, metric).to_string())

        print("\n=== WRMSSE (M5 official, now that the hierarchy exists) ===")
        print(
            con.execute("""
                SELECT model_name,
                       round(avg(value), 5) AS mean,
                       round(stddev_samp(value), 5) AS std
                FROM backtest_fold_metrics WHERE metric = 'wrmsse'
                GROUP BY 1 ORDER BY 2
            """)
            .df()
            .to_string(index=False)
        )

        print("\n=== coherence: |parent - sum(children)|, before vs after ===")
        print(
            con.execute("""
                SELECT parent_level, child_level,
                       round(avg(CASE WHEN NOT reconciled THEN mean_abs_gap END), 4) AS mean_before,
                       round(max(CASE WHEN NOT reconciled THEN max_abs_gap END), 4) AS max_before,
                       max(CASE WHEN reconciled THEN max_abs_gap END) AS max_after
                FROM coherence_check GROUP BY 1, 2
                ORDER BY 1
            """)
            .df()
            .to_string(index=False)
        )
        print()
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
