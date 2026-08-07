"""E3 — baselines. The bar every later model has to clear.

A model without a baseline is a number without meaning, so these run first and on exactly
the folds E4 will use. Four families, chosen because each is the honest competitor for a
different part of this panel:

``seasonal_naive``
    Last week, same weekday. The dumbest defensible forecast. On daily retail data with a
    strong day-of-week cycle it is much harder to beat than it looks, and any model that
    cannot clear it is not earning its complexity.

``croston`` / ``tsb``
    Built for zero-inflated demand. TSB is included alongside Croston specifically because
    Croston does not decay its estimate during a long zero run -- it keeps forecasting the
    last observed demand size forever. On a panel that is 61.6% zeros, that is not an
    academic distinction.

``ets``
    The classical statistical bar, via AutoETS.

Everything writes to the shared ``forecast`` table with ``reconciled = FALSE``, so E5
scores baselines and the GBM through one code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.data.schema import FACT_SALES, FORECAST, FORECAST_DDL

#: Every baseline this module can run. ``seasonal_naive`` is implemented here rather than
#: pulled from statsforecast so its definition is inspectable in one place -- it is the
#: number every later result is quoted against.
BASELINE_MODELS: Final[tuple[str, ...]] = ("seasonal_naive", "croston", "tsb", "ets")

#: Day-of-week cycle. Retail demand's dominant seasonality at daily grain.
SEASON_LENGTH: Final = 7

LEVEL_ITEM_STORE: Final = "item_store"


@dataclass(frozen=True)
class BaselineRun:
    """Outcome of running one baseline across all folds."""

    model_name: str
    rows_written: int
    series: int
    fit_seconds: float
    failures: int

    def render(self) -> str:
        """One-line summary."""
        note = f"  ({self.failures} series failed to fit)" if self.failures else ""
        return (
            f"  {self.model_name:<16} {self.rows_written:>9,} rows"
            f"  {self.series:>5} series  {self.fit_seconds:>7.1f}s{note}"
        )


def seasonal_naive(
    train: np.ndarray, horizon: int, season_length: int = SEASON_LENGTH
) -> np.ndarray:
    """Forecast each target day as the same weekday in the last complete training week.

    For a 28-day horizon this reuses the final ``season_length`` training days four times
    over, which is exactly the brief's "last week, same weekday". No averaging, no
    smoothing -- the point of this baseline is that it has no parameters to tune and so
    cannot be quietly improved until it flatters the comparison.

    Args:
        train: Training actuals for one series, in date order.
        horizon: Days to forecast.
        season_length: Seasonal cycle, 7 for day-of-week.

    Returns:
        Forecasts of length ``horizon``.

    Raises:
        ValueError: If ``train`` is shorter than one full season.

    """
    train = np.asarray(train, dtype=float)
    if train.size < season_length:
        raise ValueError(
            f"seasonal naive needs at least {season_length} training observations, got {train.size}"
        )
    last_week = train[-season_length:]
    return last_week[np.arange(horizon) % season_length]


def _statsforecast_models(names: tuple[str, ...]):
    """Instantiate the requested statsforecast model objects.

    Imported lazily: statsforecast pulls in numba and costs a few seconds to import, which
    should not be paid by anyone merely importing this module for ``seasonal_naive``.
    """
    from statsforecast.models import TSB, AutoETS, CrostonOptimized

    available = {
        "croston": lambda: CrostonOptimized(),
        # alpha_d / alpha_p are the demand and probability smoothing parameters. 0.2 is the
        # conventional default; TSB has no optimised variant in statsforecast, so this is a
        # stated assumption rather than a fitted value.
        "tsb": lambda: TSB(alpha_d=0.2, alpha_p=0.2),
        "ets": lambda: AutoETS(season_length=SEASON_LENGTH),
    }
    return [available[n]() for n in names if n in available]


def _training_frame(con: duckdb.DuckDBPyConnection, origin: pd.Timestamp) -> pd.DataFrame:
    """Long-format training panel up to and including ``origin``.

    Shaped as statsforecast expects: ``unique_id``, ``ds``, ``y``.
    """
    return con.execute(
        f"""
        SELECT item_id || '|' || store_id AS unique_id, date AS ds, CAST(units AS DOUBLE) AS y
        FROM {FACT_SALES}
        WHERE date <= ?
        ORDER BY unique_id, ds
        """,
        [origin],
    ).df()


def _write(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    """Append forecast rows, returning the number written."""
    if frame.empty:
        return 0
    con.register("fc_df", frame)
    con.execute(f"""
        INSERT INTO {FORECAST}
        SELECT run_id, model_name, fold, origin_date, target_date, horizon, level,
               item_id, store_id, dept_id, quantile, yhat, reconciled
        FROM fc_df
    """)
    con.unregister("fc_df")
    return len(frame)


def _forecast_rows(
    predictions: pd.DataFrame, fold: Fold, model_name: str, dept_lookup: dict[str, str]
) -> pd.DataFrame:
    """Reshape a wide statsforecast prediction frame into ``forecast`` table rows."""
    ids = predictions["unique_id"].str.split("|", n=1, expand=True)
    target = pd.to_datetime(predictions["ds"])
    return pd.DataFrame(
        {
            "run_id": f"baseline:{model_name}",
            "model_name": model_name,
            "fold": fold.index,
            "origin_date": pd.Timestamp(fold.origin_date),
            "target_date": target,
            "horizon": (target - pd.Timestamp(fold.origin_date)).dt.days,
            "level": LEVEL_ITEM_STORE,
            "item_id": ids[0],
            "store_id": ids[1],
            "dept_id": ids[0].map(dept_lookup),
            "quantile": np.nan,
            # Demand cannot be negative, and ETS in particular will happily forecast below
            # zero on a sparse series. Clipping here rather than at scoring time keeps the
            # stored forecast the same one the optimizer in E7 would actually order against.
            "yhat": predictions["yhat"].clip(lower=0.0),
            "reconciled": False,
        }
    )


def _run_seasonal_naive(
    con: duckdb.DuckDBPyConnection, fold: Fold, dept_lookup: dict[str, str]
) -> pd.DataFrame:
    """Seasonal naive across every series for one fold."""
    train = _training_frame(con, pd.Timestamp(fold.origin_date))
    out = []
    for unique_id, group in train.groupby("unique_id", sort=True):
        yhat = seasonal_naive(group["y"].to_numpy(), fold.horizon)
        out.append(
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.to_datetime(fold.target_dates()),
                    "yhat": yhat,
                }
            )
        )
    return _forecast_rows(pd.concat(out, ignore_index=True), fold, "seasonal_naive", dept_lookup)


def _run_statsforecast(
    con: duckdb.DuckDBPyConnection,
    fold: Fold,
    model_names: tuple[str, ...],
    dept_lookup: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Fit the requested statsforecast models for one fold and reshape their output."""
    from statsforecast import StatsForecast

    train = _training_frame(con, pd.Timestamp(fold.origin_date))
    models = _statsforecast_models(model_names)
    # n_jobs=1 deliberately. statsforecast's process pool re-imports scipy and numba in
    # every worker; at 1.3M training rows that exhausted the Windows paging file and the
    # pool died with BrokenProcessPool. The models are numba-JIT'd, so single-process is
    # fast enough here and the failure mode is worse than the speedup.
    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    wide = sf.forecast(df=train, h=fold.horizon)
    if wide.index.name == "unique_id":
        wide = wide.reset_index()

    # statsforecast names output columns after the model class, not our slugs.
    column_for = {
        "croston": "CrostonOptimized",
        "tsb": "TSB",
        "ets": "AutoETS",
    }
    frames = {}
    for name in model_names:
        col = column_for.get(name)
        if col is None or col not in wide.columns:
            continue
        preds = wide[["unique_id", "ds", col]].rename(columns={col: "yhat"})
        # A model that fails to converge on a series yields NaN. Those rows are dropped
        # here and counted by the caller, rather than being silently filled with zeros --
        # a zero forecast is a real prediction and would flatter the model's bias.
        preds = preds.dropna(subset=["yhat"])
        frames[name] = _forecast_rows(preds, fold, name, dept_lookup)
    return frames


def run_baselines(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    models: tuple[str, ...] = BASELINE_MODELS,
    *,
    replace: bool = True,
) -> tuple[BaselineRun, ...]:
    """Fit every baseline on every fold and write forecasts to the ``forecast`` table.

    Args:
        con: Open warehouse connection.
        folds: Rolling-origin folds from :func:`~inventory_engine.backtest.folds.make_folds`.
        models: Which baselines to run. See :data:`BASELINE_MODELS`.
        replace: Delete any existing rows for these models first, so re-running is
            idempotent rather than accumulating duplicates.

    Returns:
        One :class:`BaselineRun` per model.

    Raises:
        ValueError: If an unknown model name is requested.

    """
    unknown = set(models) - set(BASELINE_MODELS)
    if unknown:
        raise ValueError(f"unknown baseline(s): {sorted(unknown)}; expected {BASELINE_MODELS}")

    con.execute(FORECAST_DDL)
    if replace:
        placeholders = ", ".join("?" for _ in models)
        con.execute(f"DELETE FROM {FORECAST} WHERE model_name IN ({placeholders})", list(models))

    dept_lookup = dict(
        con.execute(f"SELECT DISTINCT item_id, dept_id FROM {FACT_SALES}").fetchall()
    )
    expected_per_fold = con.execute(
        f"SELECT count(DISTINCT (item_id, store_id)) FROM {FACT_SALES}"
    ).fetchone()[0]

    written = dict.fromkeys(models, 0)
    elapsed = dict.fromkeys(models, 0.0)
    sf_models = tuple(m for m in models if m != "seasonal_naive")

    for fold in folds:
        if "seasonal_naive" in models:
            started = time.perf_counter()
            rows = _run_seasonal_naive(con, fold, dept_lookup)
            elapsed["seasonal_naive"] += time.perf_counter() - started
            written["seasonal_naive"] += _write(con, rows)

        if sf_models:
            started = time.perf_counter()
            frames = _run_statsforecast(con, fold, sf_models, dept_lookup)
            # One fit call produces all statsforecast models, so the wall time is shared
            # rather than attributable per model. Split evenly and say so.
            share = (time.perf_counter() - started) / max(len(sf_models), 1)
            for name, frame in frames.items():
                elapsed[name] += share
                written[name] += _write(con, frame)

    runs = []
    for name in models:
        series = con.execute(
            f"SELECT count(DISTINCT (item_id, store_id)) FROM {FORECAST} WHERE model_name = ?",
            [name],
        ).fetchone()[0]
        runs.append(
            BaselineRun(
                model_name=name,
                rows_written=written[name],
                series=series,
                fit_seconds=elapsed[name],
                failures=expected_per_fold - series,
            )
        )
    return tuple(runs)
