"""E5-S3/S4/S5 — score stored forecasts and persist fold-level metrics.

Reads the shared ``forecast`` table, so baselines and the GBM go through one code path and
"model A beats model B" cannot become "model A was scored differently".

Reports **mean and spread across folds**, never mean alone. A model that wins on average
while losing badly on one fold is a different proposition from one that wins consistently,
and only the second is safe to put behind an ordering decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.backtest.metrics import naive_scale, naive_scale_squared, pinball
from inventory_engine.data.schema import (
    BACKTEST_METRICS,
    BACKTEST_METRICS_DDL,
    DIM_ITEM_STRATUM,
    FACT_SALES,
    FORECAST,
)
from inventory_engine.models.quantiles import monotonize

LEVEL_ITEM_STORE = "item_store"
METRICS = ("mase", "rmsse", "bias")

#: Column order of ``backtest_fold_metrics``, shared by the point and quantile scorers.
_METRIC_COLUMNS = (
    "run_id",
    "model_name",
    "fold",
    "level",
    "stratum",
    "horizon",
    "metric",
    "value",
    "n_series",
    "n_excluded",
)


@dataclass(frozen=True)
class ModelSummary:
    """Cross-fold summary for one model and metric."""

    model_name: str
    metric: str
    mean: float
    std: float
    worst: float
    best: float
    n_folds: int


def _series_scales(con: duckdb.DuckDBPyConnection, origin: date) -> pd.DataFrame:
    """Per-series naive scales from training data up to ``origin``.

    Computed once per fold and shared by every model scored on that fold, so the
    denominator is provably identical across models rather than incidentally so.
    """
    train = con.execute(
        f"""
        SELECT item_id, store_id, units
        FROM {FACT_SALES} WHERE date <= ? ORDER BY item_id, store_id, date
        """,
        [origin],
    ).df()
    rows = []
    for (item_id, store_id), group in train.groupby(["item_id", "store_id"], sort=False):
        y = group["units"].to_numpy(dtype=float)
        rows.append((item_id, store_id, naive_scale(y), naive_scale_squared(y)))
    return pd.DataFrame(rows, columns=["item_id", "store_id", "scale", "scale_sq"])


def _fold_frame(con: duckdb.DuckDBPyConnection, fold: Fold) -> pd.DataFrame:
    """Join one fold's forecasts to actuals, scales and strata."""
    forecasts = con.execute(
        f"""
        SELECT model_name, item_id, store_id, target_date, horizon, yhat
        FROM {FORECAST}
        WHERE fold = ? AND level = ? AND quantile IS NULL AND reconciled = FALSE
        """,
        [fold.index, LEVEL_ITEM_STORE],
    ).df()
    if forecasts.empty:
        return forecasts

    actuals = con.execute(
        f"""
        SELECT item_id, store_id, date AS target_date, CAST(units AS DOUBLE) AS units
        FROM {FACT_SALES} WHERE date BETWEEN ? AND ?
        """,
        [fold.test_start, fold.test_end],
    ).df()
    strata = con.execute(f"SELECT item_id, stratum_name FROM {DIM_ITEM_STRATUM}").df()

    merged = (
        forecasts.merge(actuals, on=["item_id", "store_id", "target_date"], how="inner")
        .merge(_series_scales(con, fold.origin_date), on=["item_id", "store_id"], how="left")
        .merge(strata, on="item_id", how="left")
    )
    merged["error"] = merged["yhat"] - merged["units"]
    return merged


def _per_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a fold frame to one row per (model, series) with each metric."""
    grouped = frame.groupby(["model_name", "item_id", "store_id"], sort=False)
    out = grouped.agg(
        abs_err=("error", lambda e: float(np.mean(np.abs(e)))),
        sq_err=("error", lambda e: float(np.mean(e**2))),
        bias=("error", "mean"),
        scale=("scale", "first"),
        scale_sq=("scale_sq", "first"),
        stratum_name=("stratum_name", "first"),
    ).reset_index()
    out["mase"] = out["abs_err"] / out["scale"]
    out["rmsse"] = np.sqrt(out["sq_err"] / out["scale_sq"])
    return out.replace([np.inf, -np.inf], np.nan)


def _emit(rows: list[tuple], model: str, fold: int, stratum, horizon, series: pd.DataFrame) -> None:
    """Append one aggregated row per metric."""
    for metric in METRICS:
        values = series[metric].to_numpy(dtype=float)
        finite = np.isfinite(values)
        rows.append(
            (
                # These rows record a scoring pass, not a training run; the true producing
                # run_id lives on the forecast rows themselves.
                f"scored:{model}",
                model,
                fold,
                LEVEL_ITEM_STORE,
                stratum,
                horizon,
                metric,
                float(values[finite].mean()) if finite.any() else None,
                int(finite.sum()),
                int((~finite).sum()),
            )
        )


def score_forecasts(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    *,
    replace: bool = True,
) -> pd.DataFrame:
    """Score every stored base forecast and write ``backtest_fold_metrics``.

    Args:
        con: Open warehouse connection.
        folds: The folds the forecasts were produced on.
        replace: Clear existing metric rows first, so re-running is idempotent.

    Returns:
        The metric rows written, as a DataFrame.

    """
    con.execute(BACKTEST_METRICS_DDL)
    if replace:
        # Delete exactly what this function writes: point metrics, at the forecasting
        # grain, for base (non-reconciled) models. Nothing else.
        #
        # This scope has been wrong twice. First it deleted the whole table, which wiped
        # score_quantile_forecasts' pinball rows. Narrowing it to "everything except
        # pinball" fixed that symptom but not the cause -- it still swept up E6's
        # aggregate-level rows and the WRMSSE figure, so re-running the GBM silently
        # destroyed the reconciliation metrics. An allowlist of what this function owns
        # cannot have that failure mode; a denylist of what it recognises always can.
        con.execute(
            f"""
            DELETE FROM {BACKTEST_METRICS}
            WHERE level = ? AND metric IN ('mase', 'rmsse', 'bias')
              AND NOT ends_with(model_name, '_mint')
            """,
            [LEVEL_ITEM_STORE],
        )

    rows: list[tuple] = []
    for fold in folds:
        frame = _fold_frame(con, fold)
        if frame.empty:
            continue
        series = _per_series(frame)
        for model, block in series.groupby("model_name", sort=True):
            _emit(rows, model, fold.index, None, None, block)
            for stratum, sub in block.groupby("stratum_name", sort=True):
                _emit(rows, model, fold.index, stratum, None, sub)

        # Per-horizon: a single day's scaled absolute error, so accuracy decay over the
        # 28-day window is visible rather than averaged away.
        frame = frame.merge(
            series[["model_name", "item_id", "store_id"]].drop_duplicates(),
            on=["model_name", "item_id", "store_id"],
        )
        frame["mase"] = frame["error"].abs() / frame["scale"]
        frame["rmsse"] = np.sqrt(frame["error"] ** 2 / frame["scale_sq"])
        frame["bias"] = frame["error"]
        frame = frame.replace([np.inf, -np.inf], np.nan)
        for (model, horizon), sub in frame.groupby(["model_name", "horizon"], sort=True):
            _emit(rows, model, fold.index, None, int(horizon), sub)

    written = pd.DataFrame(rows, columns=_METRIC_COLUMNS)
    if not written.empty:
        con.register("metrics_df", written)
        con.execute(f"INSERT INTO {BACKTEST_METRICS} SELECT * FROM metrics_df")
        con.unregister("metrics_df")
    return written


def score_quantile_forecasts(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    *,
    monotone: bool = True,
) -> pd.DataFrame:
    """Score stored quantile forecasts with pinball loss, per level and fold.

    Quantiles are monotonized before scoring by default, because that is how E7 reads them:
    scoring the raw crossed values would report a loss for a forecast nothing downstream
    actually uses.

    Args:
        con: Open warehouse connection.
        folds: The folds the forecasts were produced on.
        monotone: Apply rearrangement before scoring. ``False`` scores raw fitted values,
            which is what the before/after audit uses.

    Returns:
        The metric rows written.

    """
    con.execute(BACKTEST_METRICS_DDL)
    rows: list[tuple] = []

    for fold in folds:
        quantiles = con.execute(
            f"""
            SELECT model_name, fold, item_id, store_id, target_date, quantile, yhat
            FROM {FORECAST}
            WHERE fold = ? AND level = ? AND quantile IS NOT NULL AND reconciled = FALSE
            """,
            [fold.index, LEVEL_ITEM_STORE],
        ).df()
        if quantiles.empty:
            continue

        actuals = con.execute(
            f"""
            SELECT item_id, store_id, date AS target_date, CAST(units AS DOUBLE) AS units
            FROM {FACT_SALES} WHERE date BETWEEN ? AND ?
            """,
            [fold.test_start, fold.test_end],
        ).df()

        for model, block in quantiles.groupby("model_name", sort=True):
            ordered = monotonize(block.drop(columns="model_name")) if monotone else block
            merged = ordered.merge(actuals, on=["item_id", "store_id", "target_date"])
            for q, level_rows in merged.groupby("quantile", sort=True):
                rows.append(
                    (
                        f"scored:{model}",
                        model,
                        fold.index,
                        LEVEL_ITEM_STORE,
                        None,
                        None,
                        f"pinball_q{q:g}",
                        pinball(
                            level_rows["units"].to_numpy(),
                            level_rows["yhat"].to_numpy(),
                            float(q),
                        ),
                        int(len(level_rows)),
                        0,
                    )
                )

    written = pd.DataFrame(rows, columns=_METRIC_COLUMNS)
    if not written.empty:
        con.execute(f"DELETE FROM {BACKTEST_METRICS} WHERE metric LIKE 'pinball_q%'")
        con.register("qmetrics_df", written)
        con.execute(f"INSERT INTO {BACKTEST_METRICS} SELECT * FROM qmetrics_df")
        con.unregister("qmetrics_df")
    return written


def summarise_by_horizon(
    con: duckdb.DuckDBPyConnection, metric: str = "mase", models: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Cross-fold mean per forecast horizon, so accuracy decay is visible."""
    clause = ""
    params: list[object] = [metric]
    if models:
        clause = f" AND model_name IN ({', '.join('?' for _ in models)})"
        params.extend(models)
    return con.execute(
        f"""
        SELECT horizon, model_name, round(avg(value), 4) AS mean
        FROM {BACKTEST_METRICS}
        WHERE metric = ? AND horizon IS NOT NULL AND stratum IS NULL{clause}
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        params,
    ).df()


def summarise(con: duckdb.DuckDBPyConnection, metric: str = "mase") -> pd.DataFrame:
    """Cross-fold summary for one metric: mean, spread, worst and best fold."""
    return con.execute(
        f"""
        SELECT model_name,
               round(avg(value), 4)                     AS mean,
               round(stddev_samp(value), 4)             AS std,
               round(min(value), 4)                     AS best_fold,
               round(max(value), 4)                     AS worst_fold,
               count(*)                                 AS folds,
               min(n_series)                            AS min_series_scored,
               max(n_excluded)                          AS max_series_excluded
        FROM {BACKTEST_METRICS}
        WHERE metric = ? AND stratum IS NULL AND horizon IS NULL
        GROUP BY 1 ORDER BY mean
        """,
        [metric],
    ).df()


def summarise_by_stratum(con: duckdb.DuckDBPyConnection, metric: str = "mase") -> pd.DataFrame:
    """Cross-fold mean per intermittency band.

    The breakdown that matters most for E3: methods built for intermittent demand should
    earn their place on the sparse band specifically, not on the panel average.
    """
    return con.execute(
        f"""
        SELECT stratum,
               model_name,
               round(avg(value), 4)         AS mean,
               round(stddev_samp(value), 4) AS std
        FROM {BACKTEST_METRICS}
        WHERE metric = ? AND stratum IS NOT NULL AND horizon IS NULL
        GROUP BY 1, 2 ORDER BY 1, mean
        """,
        [metric],
    ).df()
