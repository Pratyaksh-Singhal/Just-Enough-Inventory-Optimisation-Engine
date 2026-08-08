"""CLI: train the global GBM, score it, and compare it honestly against the baselines."""

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
from inventory_engine.models.gbm import QUANTILES, train_and_forecast


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-gbm``."""
    parser = argparse.ArgumentParser(description="Train and score the E4 global GBM.")
    parser.add_argument("--db-path", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-quantiles", action="store_true")
    parser.add_argument("--skip-fit", action="store_true", help="Score existing rows only.")
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

        if not args.skip_fit:
            print("Training global GBM...")
            run = train_and_forecast(
                con,
                folds,
                quantiles=() if args.no_quantiles else QUANTILES,
                track=not args.no_mlflow,
            )
            print(run.render())
            print()

        print("Scoring against the baselines...")
        score_forecasts(con, folds)

        for metric in ("mase", "rmsse", "bias"):
            print(f"\n=== {metric.upper()} across {len(folds)} folds ===")
            print(summarise(con, metric).to_string(index=False))

        print("\n=== MASE by intermittency stratum ===")
        print(summarise_by_stratum(con, "mase").to_string(index=False))

        print("\n=== top SHAP features (fold 4) ===")
        print(
            con.execute("""
                SELECT feature, round(value, 4) AS mean_abs_shap
                FROM feature_importance
                WHERE method = 'shap_mean_abs' AND fold = (SELECT max(fold) FROM feature_importance)
                ORDER BY value DESC LIMIT 12
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
