"""CLI: assemble the stratum-aware hybrid and score it against every other model."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import duckdb

from inventory_engine.backtest.folds import assert_no_training_leak, make_folds, panel_bounds
from inventory_engine.backtest.score import score_forecasts, summarise, summarise_by_stratum
from inventory_engine.config import HORIZON, N_FOLDS, WAREHOUSE_PATH
from inventory_engine.models.hybrid import build_hybrid
from inventory_engine.models.quantiles import crossing_rate, monotonize, read_quantiles

CR_BAND = (0.5, 0.9, 0.95)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-hybrid``."""
    parser = argparse.ArgumentParser(description="Build and score the stratum-aware hybrid.")
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

        print("\nAssembling stratum-aware hybrid (ETS on sparse, LightGBM on dense/mid)...")
        print(build_hybrid(con, folds).render())

        print("\nScoring every model on identical folds...")
        score_forecasts(con, folds)

        for metric in ("mase", "rmsse", "bias"):
            print(f"\n=== {metric.upper()} across {len(folds)} folds ===")
            print(summarise(con, metric).to_string(index=False))

        print("\n=== MASE by intermittency stratum ===")
        print(summarise_by_stratum(con, "mase").to_string(index=False))

        print("\n=== quantile crossings, CR 0.5-0.95 band ===")
        for model in ("lgbm", "hybrid"):
            raw = read_quantiles(con, model, monotone=False)
            if raw.empty:
                continue
            before, n = crossing_rate(raw, within=CR_BAND)
            after, _ = crossing_rate(monotonize(raw), within=CR_BAND)
            print(
                f"  {model:<8} raw {before:,}/{n:,} ({before / n:.4%})"
                f"  ->  monotonized {after:,} ({after / n:.4%})"
            )
        print()
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
