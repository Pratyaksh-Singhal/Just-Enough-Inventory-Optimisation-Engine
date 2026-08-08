"""E4 — the global LightGBM model.

One model across all 720 series, with series identity as a feature, rather than 720
per-series models. On a panel where the sparse third sells on fewer than one day in five,
per-series fitting has almost nothing to learn from; a global model lets those series
borrow the day-of-week and event structure the dense ones establish.

**Why not deep learning.** At 720 series of daily data a global GBM wins on accuracy,
trains in minutes rather than hours, needs no GPU, and its feature attributions are
directly inspectable — which matters because E7 has to defend an ordering decision, not
just a forecast. An LSTM or N-BEATS here would be a larger, slower, less explainable model
fit to less data than it wants. This is a stated engineering judgement, and E4-S5 reports
the honest comparison either way.

The horizon-shift payoff
------------------------
No train/predict windowing logic is needed here, because E2 already did it. Every row of
``feature_panel`` targeting date ``t`` has features computed only from data at or before
``t - horizon``. So for a fold with origin ``o``:

* training rows are simply ``date <= o``
* prediction rows are simply ``o < date <= o + horizon``

and the prediction rows' features are, by construction, derived from data no later than
``o``. Correct by the panel's design rather than by a windowing routine that could drift
from it.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.data.schema import (
    FEATURE_IMPORTANCE,
    FEATURE_IMPORTANCE_DDL,
    FORECAST,
    FORECAST_DDL,
)
from inventory_engine.features.build import FEATURE_PANEL, feature_columns

MODEL_NAME: Final = "lgbm"
LEVEL_ITEM_STORE: Final = "item_store"

#: Series identity, handed to LightGBM as native categoricals. This is what makes the
#: model "global": one fit, with the series it is predicting as an input.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("item_id", "store_id", "dept_id")

#: Every column LightGBM must treat as categorical. ``event_type`` is a string feature
#: from the calendar rather than series identity, but needs the same handling.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (*CATEGORICAL_FEATURES, "event_type")

#: Quantile levels E7's newsvendor layer selects between.
QUANTILES: Final[tuple[float, ...]] = (0.5, 0.9, 0.95, 0.99)

#: Untuned, and deliberately so. The question E4 answers is "does a global GBM beat the
#: baselines", not "what is the best possible GBM"; tuning against the same folds used to
#: report the result would make the comparison meaningless. Hyperparameter search on an
#: inner split is listed under next steps.
BASE_PARAMS: Final[dict] = {
    "objective": "tweedie",
    # Tweedie handles zero-inflated non-negative counts, which is what 61.6%-zero retail
    # demand is. It was the objective behind the strongest M5 solutions. Chosen from that
    # literature rather than tuned here, and labelled as such.
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_boost_round": 500,
    "verbose": -1,
    "seed": 20260808,
    "num_threads": 4,
}


@dataclass(frozen=True)
class GbmRun:
    """Outcome of training the global model across all folds."""

    model_name: str
    rows_written: int
    series: int
    folds: int
    train_seconds: float
    quantile_crossings: int = 0
    quantile_rows: int = 0
    mlflow_run_ids: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Multi-line summary."""
        lines = [
            f"  {self.model_name:<16} {self.rows_written:>9,} rows"
            f"  {self.series:>5} series  {self.folds} folds  {self.train_seconds:>7.1f}s",
        ]
        if self.quantile_rows:
            share = self.quantile_crossings / self.quantile_rows
            lines.append(
                f"  quantile crossings {self.quantile_crossings:,} / {self.quantile_rows:,}"
                f" ({share:.2%}) -- reported, not silently sorted"
            )
        return "\n".join(lines)


def _git_sha() -> str:
    """Return the current commit, so an MLflow run traces back to the code behind it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_panel(con: duckdb.DuckDBPyConnection, table: str = FEATURE_PANEL) -> pd.DataFrame:
    """Read the feature panel and type the categoricals for LightGBM.

    Categories are fixed across the whole panel rather than per fold, so a series absent
    from one fold's training data still maps to the same code in another's.
    """
    panel = con.execute(f"SELECT * FROM {table}").df()
    for column in CATEGORICAL_COLUMNS:
        panel[column] = panel[column].astype("category")
    return panel


def feature_matrix(panel: pd.DataFrame) -> list[str]:
    """Feature column names for training.

    ``cat_id`` and ``state_id`` are excluded: the Phase 1 scope fixes them to FOODS and CA,
    so they are constant and carry no signal. ``horizon`` is excluded for the same reason —
    the panel is built at a single horizon. Both would be free variance for the model to
    chase if left in.
    """
    return [*feature_columns(), *CATEGORICAL_FEATURES]


def _predict_rows(
    predictions: np.ndarray,
    rows: pd.DataFrame,
    fold: Fold,
    run_id: str,
    quantile: float | None,
) -> pd.DataFrame:
    """Shape model output into ``forecast`` table rows."""
    return pd.DataFrame(
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "fold": fold.index,
            "origin_date": pd.Timestamp(fold.origin_date),
            "target_date": rows["date"].to_numpy(),
            "horizon": (
                pd.to_datetime(rows["date"]) - pd.Timestamp(fold.origin_date)
            ).dt.days.to_numpy(),
            "level": LEVEL_ITEM_STORE,
            "item_id": rows["item_id"].astype(str).to_numpy(),
            "store_id": rows["store_id"].astype(str).to_numpy(),
            "dept_id": rows["dept_id"].astype(str).to_numpy(),
            "quantile": quantile,
            # Demand cannot be negative. Clipped at write time so the stored forecast is
            # the one E7 would actually order against.
            "yhat": np.clip(predictions, 0.0, None),
            "reconciled": False,
        }
    )


def _count_quantile_crossings(frame: pd.DataFrame) -> tuple[int, int]:
    """Count rows where fitted quantiles are non-monotonic.

    Independently fitted quantile models are not constrained to be ordered, so q0.9 can
    land above q0.95. E7 selects a quantile by critical ratio and would silently pick a
    nonsense number where that happens, so the rate is measured and reported rather than
    quietly sorted away.

    Returns:
        ``(crossings, rows_checked)``.

    """
    wide = frame.pivot_table(
        index=["fold", "item_id", "store_id", "target_date"],
        columns="quantile",
        values="yhat",
    )
    levels = [q for q in QUANTILES if q in wide.columns]
    if len(levels) < 2:
        return 0, 0
    values = wide[levels].to_numpy()
    crossings = int((np.diff(values, axis=1) < -1e-9).any(axis=1).sum())
    return crossings, int(len(wide))


def train_and_forecast(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    *,
    quantiles: tuple[float, ...] = QUANTILES,
    params: dict | None = None,
    track: bool = True,
    shap_sample: int = 20_000,
    replace: bool = True,
) -> GbmRun:
    """Train the global GBM per fold, write point and quantile forecasts.

    Args:
        con: Open warehouse connection with ``feature_panel`` populated.
        folds: Rolling-origin folds; the same ones the baselines used.
        quantiles: Quantile levels to fit alongside the point model.
        params: Override :data:`BASE_PARAMS`.
        track: Log runs to MLflow.
        shap_sample: Rows sampled for SHAP. SHAP over the full panel is far too slow, and
            importance is stable well below this size.
        replace: Clear existing ``lgbm`` rows first, so re-running is idempotent.

    Returns:
        A :class:`GbmRun` summary.

    """
    import lightgbm as lgb

    settings = {**BASE_PARAMS, **(params or {})}
    n_rounds = settings.pop("num_boost_round", 500)

    con.execute(FORECAST_DDL)
    con.execute(FEATURE_IMPORTANCE_DDL)
    if replace:
        con.execute(f"DELETE FROM {FORECAST} WHERE model_name = ?", [MODEL_NAME])
        con.execute(f"DELETE FROM {FEATURE_IMPORTANCE} WHERE model_name = ?", [MODEL_NAME])

    panel = load_panel(con)
    features = feature_matrix(panel)
    started = time.perf_counter()
    written = 0
    run_ids: list[str] = []
    quantile_frames: list[pd.DataFrame] = []

    for fold in folds:
        origin = pd.Timestamp(fold.origin_date)
        train = panel[panel["date"] <= origin]
        test = panel[(panel["date"] > origin) & (panel["date"] <= pd.Timestamp(fold.test_end))]
        if train.empty or test.empty:
            continue

        train_set = lgb.Dataset(
            train[features],
            label=train["units"],
            categorical_feature=list(CATEGORICAL_COLUMNS),
            free_raw_data=False,
        )

        run_id = f"lgbm:fold{fold.index}"
        point_model = lgb.train({**settings}, train_set, num_boost_round=n_rounds)
        written += _write(
            con, _predict_rows(point_model.predict(test[features]), test, fold, run_id, None)
        )

        for q in quantiles:
            q_params = {**settings, "objective": "quantile", "alpha": q}
            q_params.pop("tweedie_variance_power", None)
            q_model = lgb.train(q_params, train_set, num_boost_round=n_rounds)
            rows = _predict_rows(q_model.predict(test[features]), test, fold, run_id, q)
            quantile_frames.append(rows)
            written += _write(con, rows)

        _record_importance(con, point_model, features, fold, run_id, train, shap_sample)
        if track:
            _log_mlflow(run_id, fold, settings, n_rounds, features, point_model, quantiles)
        run_ids.append(run_id)

    crossings, checked = (
        _count_quantile_crossings(pd.concat(quantile_frames, ignore_index=True))
        if quantile_frames
        else (0, 0)
    )
    series = con.execute(
        f"SELECT count(DISTINCT (item_id, store_id)) FROM {FORECAST} WHERE model_name = ?",
        [MODEL_NAME],
    ).fetchone()[0]

    return GbmRun(
        model_name=MODEL_NAME,
        rows_written=written,
        series=series,
        folds=len(run_ids),
        train_seconds=time.perf_counter() - started,
        quantile_crossings=crossings,
        quantile_rows=checked,
        mlflow_run_ids=tuple(run_ids),
    )


def _write(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> int:
    """Append forecast rows."""
    if frame.empty:
        return 0
    con.register("gbm_df", frame)
    con.execute(f"""
        INSERT INTO {FORECAST}
        SELECT run_id, model_name, fold, origin_date, target_date, horizon, level,
               item_id, store_id, dept_id, quantile, yhat, reconciled
        FROM gbm_df
    """)
    con.unregister("gbm_df")
    return len(frame)


def _record_importance(
    con: duckdb.DuckDBPyConnection,
    model,
    features: list[str],
    fold: Fold,
    run_id: str,
    train: pd.DataFrame,
    shap_sample: int,
) -> None:
    """Persist gain, split and mean-|SHAP| importance for E9's explainability panel."""
    rows = [
        (run_id, MODEL_NAME, fold.index, name, method, float(value))
        for method, importances in (
            ("gain", model.feature_importance("gain")),
            ("split", model.feature_importance("split")),
        )
        for name, value in zip(features, importances, strict=True)
    ]

    sample = train.sample(min(shap_sample, len(train)), random_state=BASE_PARAMS["seed"])
    contributions = model.predict(sample[features], pred_contrib=True)
    # pred_contrib appends a bias column; drop it before pairing with feature names.
    mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
    rows.extend(
        (run_id, MODEL_NAME, fold.index, name, "shap_mean_abs", float(value))
        for name, value in zip(features, mean_abs, strict=True)
    )

    frame = pd.DataFrame(
        rows, columns=["run_id", "model_name", "fold", "feature", "method", "value"]
    )
    con.register("imp_df", frame)
    con.execute(f"INSERT INTO {FEATURE_IMPORTANCE} SELECT * FROM imp_df")
    con.unregister("imp_df")


def _log_mlflow(
    run_id: str,
    fold: Fold,
    settings: dict,
    n_rounds: int,
    features: list[str],
    model,
    quantiles: tuple[float, ...],
) -> None:
    """Log one fold's run so it is reproducible from the logged config alone."""
    import mlflow

    mlflow.set_experiment("inventory-optimization-engine")
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({**settings, "num_boost_round": n_rounds})
        mlflow.log_params(
            {
                "fold": fold.index,
                "origin_date": fold.origin_date.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "horizon": fold.horizon,
                "n_features": len(features),
                "quantiles": ",".join(str(q) for q in quantiles),
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_text("\n".join(features), "features.txt")
        mlflow.log_metric("best_iteration", model.current_iteration())
