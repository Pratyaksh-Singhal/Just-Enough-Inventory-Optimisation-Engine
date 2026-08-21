"""E6-S5 and WRMSSE — score forecasts at every hierarchy level, before and after MinT."""

from __future__ import annotations

from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.backtest.metrics import naive_scale, naive_scale_squared
from inventory_engine.data.schema import (
    BACKTEST_METRICS,
    BACKTEST_METRICS_DDL,
    FACT_SALES,
    FORECAST,
)
from inventory_engine.hierarchy.mint import HIERARCHY_SPEC, LEVEL_NAMES, _level_of, build_hierarchy

#: Trailing window used to weight series by dollar share, per the M5 convention.
WEIGHT_WINDOW_DAYS: Final = 28


def _actuals_by_level(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return aggregated actuals for every series in the hierarchy."""
    y_df, _, _ = build_hierarchy(con)
    y_df["level"] = y_df["unique_id"].map(_level_of)
    return y_df


def _series_weights(con: duckdb.DuckDBPyConnection, y_df: pd.DataFrame, fold: Fold) -> pd.Series:
    """Dollar-sales share per series over the 28 days before the fold origin."""
    start = pd.Timestamp(fold.origin_date) - pd.Timedelta(days=WEIGHT_WINDOW_DAYS - 1)
    revenue = con.execute(
        f"""
        SELECT state_id, store_id, dept_id, item_id,
               sum(units * coalesce(price, 0)) AS revenue
        FROM {FACT_SALES} WHERE date BETWEEN ? AND ?
        GROUP BY 1, 2, 3, 4
        """,
        [start.date(), fold.origin_date],
    ).df()

    weights: dict[str, float] = {}
    for depth, spec in enumerate(HIERARCHY_SPEC):
        grouped = revenue.groupby(spec, as_index=False)["revenue"].sum()
        ids = grouped[spec].astype(str).agg("/".join, axis=1)
        total = grouped["revenue"].sum()
        share = grouped["revenue"] / total if total > 0 else 0.0
        # Each level's weights sum to 1, then levels are averaged, so no level dominates
        # merely by having more series in it.
        weights.update(dict(zip(ids, share / len(HIERARCHY_SPEC), strict=True)))
        _ = depth
    return pd.Series(weights, name="weight")


def score_levels(
    con: duckdb.DuckDBPyConnection, folds: tuple[Fold, ...], model_name: str = "lgbm"
) -> pd.DataFrame:
    """Score base and reconciled forecasts at every hierarchy level."""
    con.execute(BACKTEST_METRICS_DDL)
    # Delete exactly what this function writes: aggregate-level rows for the base model, every
    # level for the reconciled model, and the hierarchy-wide WRMSSE.
    con.execute(
        f"""
        DELETE FROM {BACKTEST_METRICS}
        WHERE ends_with(model_name, '_mint')
           OR (model_name = ? AND level <> ?)
           OR (model_name = ? AND metric = 'wrmsse')
        """,
        [model_name, "item_store", model_name],
    )

    y_df = _actuals_by_level(con)
    actual_wide = y_df.pivot(index="ds", columns="unique_id", values="y")
    rows: list[tuple] = []
    wrmsse_rows: list[tuple] = []

    for fold in folds:
        train_end = pd.Timestamp(fold.origin_date)
        scales = {}
        for uid in actual_wide.columns:
            history = actual_wide.loc[actual_wide.index <= train_end, uid].to_numpy(dtype=float)
            scales[uid] = (naive_scale(history), naive_scale_squared(history))

        weights = _series_weights(con, y_df, fold)
        forecasts = con.execute(
            f"""
            SELECT level, reconciled, item_id, store_id, dept_id, target_date, yhat
            FROM {FORECAST}
            WHERE model_name = ? AND fold = ? AND quantile IS NULL
            """,
            [model_name, fold.index],
        ).df()
        if forecasts.empty:
            continue
        forecasts["unique_id"] = _rebuild_ids(forecasts)

        for is_rec, block in forecasts.groupby("reconciled"):
            label = f"{model_name}_mint" if is_rec else model_name
            per_series = []
            for uid, group in block.groupby("unique_id"):
                scale, scale_sq = scales.get(uid, (np.nan, np.nan))
                actual = actual_wide.loc[group["target_date"].to_numpy(), uid].to_numpy(float)
                pred = group["yhat"].to_numpy(float)
                per_series.append(
                    {
                        "unique_id": uid,
                        "level": _level_of(uid),
                        "mase": np.mean(np.abs(actual - pred)) / scale
                        if np.isfinite(scale)
                        else np.nan,
                        "rmsse": np.sqrt(np.mean((actual - pred) ** 2) / scale_sq)
                        if np.isfinite(scale_sq)
                        else np.nan,
                        "bias": float(np.mean(pred - actual)),
                    }
                )
            frame = pd.DataFrame(per_series)
            for level, sub in frame.groupby("level"):
                # E5's scorer already owns (base model, item_store); writing it again here would
                # duplicate the key and make any later divergence between the two paths average
                if not is_rec and level == "item_store":
                    continue
                for metric in ("mase", "rmsse", "bias"):
                    values = sub[metric].to_numpy(dtype=float)
                    finite = np.isfinite(values)
                    rows.append(
                        (
                            f"scored:{label}",
                            label,
                            fold.index,
                            level,
                            None,
                            None,
                            metric,
                            float(values[finite].mean()) if finite.any() else None,
                            int(finite.sum()),
                            int((~finite).sum()),
                        )
                    )

            frame["weight"] = frame["unique_id"].map(weights).fillna(0.0)
            usable = frame[np.isfinite(frame["rmsse"])]
            wrmsse = float((usable["rmsse"] * usable["weight"]).sum())
            wrmsse_rows.append(
                (
                    f"scored:{label}",
                    label,
                    fold.index,
                    "hierarchy",
                    None,
                    None,
                    "wrmsse",
                    wrmsse,
                    int(len(usable)),
                    int(len(frame) - len(usable)),
                )
            )

    written = pd.DataFrame(rows + wrmsse_rows, columns=_COLUMNS)
    if not written.empty:
        con.register("lvl_df", written)
        con.execute(f"INSERT INTO {BACKTEST_METRICS} SELECT * FROM lvl_df")
        con.unregister("lvl_df")
    return written


def _rebuild_ids(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct hierarchy ids from the forecast table's split columns."""
    parts = frame[["store_id", "dept_id", "item_id"]].fillna("")
    return (
        "CA"
        + parts["store_id"].map(lambda v: f"/{v}" if v else "")
        + parts["dept_id"].map(lambda v: f"/{v}" if v else "")
        + parts["item_id"].map(lambda v: f"/{v}" if v else "")
    )


_COLUMNS: Final[tuple[str, ...]] = (
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


def level_comparison(
    con: duckdb.DuckDBPyConnection, metric: str = "rmsse", model_name: str = "lgbm"
) -> pd.DataFrame:
    """Before/after accuracy per hierarchy level, with the delta."""
    reconciled_name = f"{model_name}_mint"
    frame = con.execute(
        f"""
        SELECT level, model_name, round(avg(value), 4) AS mean
        FROM {BACKTEST_METRICS}
        WHERE metric = ?
          AND stratum IS NULL AND horizon IS NULL
          AND model_name IN (?, ?)
          AND level IN ({", ".join("?" for _ in LEVEL_NAMES)})
        GROUP BY 1, 2
        """,
        [metric, model_name, reconciled_name, *LEVEL_NAMES],
    ).df()
    if frame.empty:
        return frame
    wide = frame.pivot(index="level", columns="model_name", values="mean")
    if model_name in wide.columns and reconciled_name in wide.columns:
        wide["delta"] = (wide[reconciled_name] - wide[model_name]).round(4)
        wide["verdict"] = np.where(
            wide["delta"].isna(), "-", np.where(wide["delta"] < 0, "MinT better", "MinT worse")
        )
    order = {name: i for i, name in enumerate(LEVEL_NAMES)}
    return wide.reindex(sorted(wide.index, key=lambda level: order.get(level, 99)))
