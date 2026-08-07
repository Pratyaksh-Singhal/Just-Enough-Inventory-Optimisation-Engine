"""CLI: fit every baseline on every fold, score them, print the comparison table (E3-S4)."""

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
from inventory_engine.backtest.score import score_forecasts, summarise, summarise_by_stratum
from inventory_engine.config import HORIZON, N_FOLDS, WAREHOUSE_PATH
from inventory_engine.models.baselines import BASELINE_MODELS, run_baselines


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-baselines``."""
    parser = argparse.ArgumentParser(description="Fit and score the E3 baselines.")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BASELINE_MODELS),
        choices=list(BASELINE_MODELS),
    )
    parser.add_argument("--skip-fit", action="store_true", help="Score existing forecasts only.")
    args = parser.parse_args(argv)

    if not args.db_path.is_file():
        print(f"\nNo warehouse at {args.db_path}. Run `build-warehouse` first.\n", file=sys.stderr)
        return 1

    # statsforecast is noisy about convergence on sparse series; the failures that matter
    # are counted and reported below rather than read off stderr.
    warnings.filterwarnings("ignore")

    con = duckdb.connect(str(args.db_path))
    try:
        first, last = panel_bounds(con)
        folds = make_folds(last, args.folds, args.horizon, first_date=first)
        assert_no_training_leak(folds)
        print(f"\nRolling-origin folds\n{describe_folds(folds)}\n")

        if not args.skip_fit:
            print("Fitting baselines...")
            for run in run_baselines(con, folds, tuple(args.models)):
                print(run.render())
            print()

        print("Scoring...")
        score_forecasts(con, folds)

        for metric in ("mase", "rmsse", "bias"):
            print(f"\n=== {metric.upper()} across {len(folds)} folds ===")
            print(summarise(con, metric).to_string(index=False))

        print("\n=== MASE by intermittency stratum ===")
        print(summarise_by_stratum(con, "mase").to_string(index=False))
        print()
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
