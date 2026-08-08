"""E8-S4 — nightly precompute job: refresh forecasts without ever blocking the API.

The problem this exists to solve
---------------------------------
DuckDB's single-writer model (see ``deps.py``) means a write connection needs the file to
itself. The API opens and closes a connection per request, so in principle the write window
is available between requests -- but "in principle" is not "safely", especially on Windows,
where a rename over an open file handle can fail outright rather than merely block.

The fix is to never write to the live file at all. The job builds a **complete copy** of the
warehouse at a shadow path, runs every refresh step against the shadow, and only then
attempts an atomic ``os.replace`` onto the canonical path. Two consequences fall out of that
design directly:

* **Idempotent and safely re-runnable** (an explicit E8-S4 requirement): every run starts
  from a fresh copy of the last good warehouse, so there is no accumulated state to
  reconcile between runs.
* **Failure leaves the previous good results in place** (also required): if any refresh
  step raises, or the swap itself fails because a reader still has the old file open, the
  shadow file is discarded and the live file is untouched. There is no partial-write state
  for a request to observe.

What "refresh" actually runs
-----------------------------
The real pipeline -- GBM training, MinT reconciliation, the optimizer -- already exists as
the ``run-gbm`` / ``run-mint`` / ``run-optimize`` CLIs from Phases 4, 6 and 7. This module
does not reimplement them; it sequences them against the shadow copy. The step list is
injected rather than hard-coded so tests can substitute fast fakes instead of a 13-minute
GBM fit.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from inventory_engine.config import WAREHOUSE_PATH

RefreshStep = Callable[[duckdb.DuckDBPyConnection], None]


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of one nightly refresh attempt."""

    swapped: bool
    steps_run: int
    elapsed_seconds: float
    error: str | None = None

    def render(self) -> str:
        """One-line summary."""
        if self.swapped:
            return f"  refreshed  {self.steps_run} steps  {self.elapsed_seconds:.1f}s"
        return f"  NOT swapped -- {self.error}  ({self.elapsed_seconds:.1f}s elapsed)"


def default_steps() -> tuple[RefreshStep, ...]:
    """Return the real pipeline: GBM, MinT, optimizer.

    Imported lazily -- see the module docstring.
    """
    from inventory_engine.backtest.folds import assert_no_training_leak, make_folds, panel_bounds
    from inventory_engine.hierarchy.mint import reconcile
    from inventory_engine.models.gbm import train_and_forecast
    from inventory_engine.optimize.costs import DEFAULT_COST_MODEL
    from inventory_engine.optimize.newsvendor import (
        load_quantile_panel,
        order_quantities,
        persist_policy,
    )
    from inventory_engine.optimize.simulate import money_table, sensitivity

    def _folds(con: duckdb.DuckDBPyConnection):
        first, last = panel_bounds(con)
        folds = make_folds(last, first_date=first)
        assert_no_training_leak(folds)
        return folds

    def step_gbm(con: duckdb.DuckDBPyConnection) -> None:
        train_and_forecast(con, _folds(con), track=False)

    def step_mint(con: duckdb.DuckDBPyConnection) -> None:
        reconcile(con, _folds(con))

    def step_optimize(con: duckdb.DuckDBPyConnection) -> None:
        panel, levels = load_quantile_panel(con)
        orders = order_quantities(panel, levels, DEFAULT_COST_MODEL)
        persist_policy(con, orders, "lgbm")
        money_table(con, orders, levels, DEFAULT_COST_MODEL)
        sensitivity(con, orders, levels, margin_rate=DEFAULT_COST_MODEL.margin_rate)

    return (step_gbm, step_mint, step_optimize)


def nightly_refresh(
    steps: Sequence[RefreshStep] | None = None,
    *,
    live_path: Path = WAREHOUSE_PATH,
    max_swap_attempts: int = 5,
    swap_retry_seconds: float = 0.5,
) -> RefreshResult:
    """Run a refresh against a shadow copy and atomically swap it into place.

    Args:
        steps: Refresh steps to run, each given an open read-write connection to the
            shadow copy. Defaults to :func:`default_steps`.
        live_path: The warehouse the API reads. Never written to directly.
        max_swap_attempts: Retries for the final swap, in case a reader's connection is
            open at the exact moment of the attempt.
        swap_retry_seconds: Delay between swap attempts.

    Returns:
        A :class:`RefreshResult`. ``swapped=False`` means the live warehouse is untouched;
        check ``.error`` for why.

    """
    steps = tuple(steps) if steps is not None else default_steps()
    started = time.perf_counter()

    if not live_path.is_file():
        return RefreshResult(
            False, 0, time.perf_counter() - started, "no live warehouse to refresh from"
        )

    shadow_path = live_path.with_suffix(".shadow.duckdb")
    shadow_path.unlink(missing_ok=True)
    shutil.copy2(live_path, shadow_path)

    ran = 0
    try:
        con = duckdb.connect(str(shadow_path))
        try:
            for step in steps:
                step(con)
                ran += 1
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not re-raised
        shadow_path.unlink(missing_ok=True)
        return RefreshResult(
            False, ran, time.perf_counter() - started, f"refresh step failed: {exc}"
        )

    for attempt in range(max_swap_attempts):
        try:
            os.replace(shadow_path, live_path)
            return RefreshResult(True, ran, time.perf_counter() - started)
        except OSError as exc:
            if attempt == max_swap_attempts - 1:
                shadow_path.unlink(missing_ok=True)
                return RefreshResult(
                    False,
                    ran,
                    time.perf_counter() - started,
                    f"swap failed after {max_swap_attempts} attempts: {exc}",
                )
            time.sleep(swap_retry_seconds)

    # Unreachable, but keeps the type checker honest about every path returning.
    shadow_path.unlink(missing_ok=True)
    return RefreshResult(False, ran, time.perf_counter() - started, "swap loop exited unexpectedly")
