"""E3 — baselines. The bar every later model has to clear."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.data.schema import FACT_SALES, FORECAST, FORECAST_DDL

#: Every baseline this module can run.
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
    """Forecast each target day as the same weekday in the last complete training week."""
    train = np.asarray(train, dtype=float)
    if train.size < season_length:
        raise ValueError(
            f"seasonal naive needs at least {season_length} training observations, got {train.size}"
        )
    last_week = train[-season_length:]
    return last_week[np.arange(horizon) % season_length]


def _statsforecast_models(names: tuple[str, ...]):
    """Instantiate the requested statsforecast model objects."""
    from statsforecast.models import TSB, AutoETS, CrostonOptimized

    available = {
        "croston": lambda: CrostonOptimized(),
        # alpha_d / alpha_p are the demand and probability smoothing parameters.
        "tsb": lambda: TSB(alpha_d=0.2, alpha_p=0.2),
        "ets": lambda: AutoETS(season_length=SEASON_LENGTH),
    }
    return [available[n]() for n in names if n in available]


def _training_frame(con: duckdb.DuckDBPyConnection, origin: pd.Timestamp) -> pd.DataFrame:
    """Long-format training panel up to and including ``origin``."""
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
            # Demand cannot be negative, and ETS in particular will happily forecast below zero on
            # a sparse series.
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
    # n_jobs=1 deliberately.
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
        # A model that fails to converge on a series yields NaN.
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
    """Fit every baseline on every fold and write forecasts to the ``forecast`` table."""
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
