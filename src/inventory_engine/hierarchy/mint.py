"""E6 — MinT reconciliation, the technical differentiator.

The problem
-----------
Forecast each level of a retail hierarchy independently and the numbers will not add up.
The 720 SKU forecasts will not sum to the store forecasts; the store forecasts will not sum
to the state forecast. That is not a rounding nuisance — it means a replenishment planner
and a regional manager reading the same system get answers that contradict each other, and
only one of them can be acted on.

MinT (Minimum Trace, Wickramasuriya et al. 2019) finds the coherent forecast that minimises
the trace of the reconciliation error covariance. Unlike bottom-up (which throws away the
aggregate forecasts) or top-down (which throws away the bottom ones), it uses information
from every level and is guaranteed to produce forecasts that sum exactly.

The hierarchy, and a deviation from the brief
---------------------------------------------
The brief specifies ``State -> Store -> Category -> Dept -> Item``. Phase 1's scope fixes
``cat_id = FOODS``, which makes the Category level **identical** to the Store level: every
store has exactly one category, so aggregating to store-x-category returns the same four
series as aggregating to store.

Including it anyway would put duplicate rows in the summing matrix `S`, leaving the
residual covariance rank-deficient and MinT's inverse ill-conditioned — a numerical problem
created purely by encoding a level that carries no information. So the Category level is
**dropped**, and the deviation is recorded here rather than silently applied. On a scope
spanning multiple categories it would be restored unchanged.

What is left is a genuine four-level tree over 737 series.

Base forecasts
--------------
MinT needs a base forecast at *every* level, and those forecasts must be produced
**independently** — their mutual incoherence is exactly what reconciliation corrects and
what the before/after table measures.

Bottom level (item x store) uses the production LightGBM forecasts from E4. Aggregate
levels use AutoETS, and that is a considered choice rather than expedience: aggregating
hundreds of SKUs washes out the intermittency that made the bottom level hard, leaving
dense, strongly seasonal series that a classical method handles well. Building the full
E2 feature panel at each aggregation level would be a large amount of work to arrive at a
worse fit on easier data.
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
from inventory_engine.models.baselines import SEASON_LENGTH

#: Nested aggregation levels. Category is deliberately absent -- see the module docstring.
HIERARCHY_SPEC: Final[list[list[str]]] = [
    ["state_id"],
    ["state_id", "store_id"],
    ["state_id", "store_id", "dept_id"],
    ["state_id", "store_id", "dept_id", "item_id"],
]

#: Human-readable name per spec entry, used as the ``level`` column in ``forecast``.
LEVEL_NAMES: Final[tuple[str, ...]] = ("state", "store", "store_dept", "item_store")

#: MinT covariance estimator, and this choice needed a correction mid-build.
#:
#: ``mint_shrink`` (Schafer-Strimmer shrinkage toward a diagonal target) is the textbook
#: default and is what this module first used. It cannot be used here: it estimates the
#: residual covariance from **in-sample** one-step residuals, which requires fitted values
#: for all 737 series in every fold. E4 persists forecasts, not models, so those residuals
#: do not exist, and regenerating them would mean refitting the GBM on every training
#: window purely to obtain them.
#:
#: ``wls_struct`` is the honest alternative rather than a silent fallback. It weights each
#: node by the number of bottom series aggregated into it — structural information taken
#: from ``S`` alone — which is the case Wickramasuriya et al. (2019) propose it for when
#: residuals are unavailable. ``ols`` (identity covariance) is scored alongside it so the
#: estimator choice is evidenced rather than asserted.
MINT_METHOD: Final = "wls_struct"

#: Estimators compared in the report. Both need only ``S``, so both are available here.
COMPARED_METHODS: Final[tuple[str, ...]] = ("ols", "wls_struct")

MODEL_NAME: Final = "lgbm"
COHERENCE_TABLE: Final = "coherence_check"

COHERENCE_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS {COHERENCE_TABLE} (
    model_name   VARCHAR NOT NULL,
    fold         INTEGER NOT NULL,
    parent_level VARCHAR NOT NULL,
    child_level  VARCHAR NOT NULL,
    reconciled   BOOLEAN NOT NULL,
    mean_abs_gap DOUBLE,
    max_abs_gap  DOUBLE,
    n_checked    INTEGER
);
"""


@dataclass(frozen=True)
class ReconciliationRun:
    """Outcome of reconciling one model across all folds."""

    model_name: str
    rows_written: int
    series: int
    levels: int
    max_gap_before: float
    max_gap_after: float
    elapsed_seconds: float

    def render(self) -> str:
        """Multi-line summary."""
        return "\n".join(
            [
                f"  {self.model_name:<16} {self.rows_written:>9,} reconciled rows"
                f"  {self.series} series across {self.levels} levels"
                f"  {self.elapsed_seconds:>6.1f}s",
                f"  max coherence gap  before {self.max_gap_before:>12,.4f}"
                f"   after {self.max_gap_after:.2e}",
            ]
        )


def build_hierarchy(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate actuals up the tree and build the summing matrix.

    Args:
        con: Open warehouse connection.

    Returns:
        ``(Y_df, S_df, tags)`` as produced by ``hierarchicalforecast.utils.aggregate``.

    """
    from hierarchicalforecast.utils import aggregate

    panel = con.execute(f"""
        SELECT state_id, store_id, dept_id, item_id, date AS ds, CAST(units AS DOUBLE) AS y
        FROM {FACT_SALES} ORDER BY date
    """).df()
    return aggregate(panel, HIERARCHY_SPEC)


def validate_hierarchy(y_df: pd.DataFrame, s_df: pd.DataFrame, tags: dict) -> None:
    """Fail unless aggregated actuals genuinely sum up the tree.

    The summing matrix is the object every reconciled number depends on. If it is wrong,
    MinT still returns coherent-looking output — coherent with respect to the wrong
    structure. So it is checked against the actuals rather than trusted.

    Raises:
        ValueError: If any level's aggregated actuals fail to match the sum of its children.

    """
    wide = y_df.pivot(index="ds", columns="unique_id", values="y")
    bottom_ids = list(tags[LEVEL_KEY_BY_NAME["item_store"]])

    for parent_name, child_name in (("state", "store"), ("store", "store_dept")):
        parents = tags[LEVEL_KEY_BY_NAME[parent_name]]
        children = tags[LEVEL_KEY_BY_NAME[child_name]]
        for parent in parents:
            kids = [c for c in children if c.startswith(f"{parent}/")]
            if not kids:
                raise ValueError(f"{parent} at level {parent_name} has no children")
            gap = (wide[parent] - wide[kids].sum(axis=1)).abs().max()
            if gap > 1e-6:
                raise ValueError(
                    f"aggregated actuals do not sum: {parent} differs from the sum of its"
                    f" {len(kids)} children by {gap:.6f}. The summing matrix is wrong."
                )

    if len(bottom_ids) != len(s_df.columns) - 1:
        raise ValueError(
            f"S has {len(s_df.columns) - 1} bottom columns but the tree has"
            f" {len(bottom_ids)} bottom series"
        )


#: Map friendly level names onto the keys ``aggregate`` produces.
LEVEL_KEY_BY_NAME: Final[dict[str, str]] = {
    "state": "state_id",
    "store": "state_id/store_id",
    "store_dept": "state_id/store_id/dept_id",
    "item_store": "state_id/store_id/dept_id/item_id",
}


def _level_of(unique_id: str) -> str:
    """Infer a series' level from how many path segments its id has."""
    return LEVEL_NAMES[unique_id.count("/")]


def base_forecasts(
    con: duckdb.DuckDBPyConnection, y_df: pd.DataFrame, tags: dict, fold: Fold
) -> pd.DataFrame:
    """Produce independent base forecasts at every level for one fold.

    Bottom level reuses the stored LightGBM forecasts; aggregate levels are fit fresh with
    AutoETS. They are deliberately not made to agree — that incoherence is what E6-S4
    measures before and after.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    origin = pd.Timestamp(fold.origin_date)
    aggregate_ids = [uid for uid in y_df["unique_id"].unique() if _level_of(uid) != "item_store"]
    train = y_df[(y_df["ds"] <= origin) & (y_df["unique_id"].isin(aggregate_ids))]

    sf = StatsForecast(models=[AutoETS(season_length=SEASON_LENGTH)], freq="D", n_jobs=1)
    upper = sf.forecast(df=train, h=fold.horizon)
    if upper.index.name == "unique_id":
        upper = upper.reset_index()
    upper = upper.rename(columns={"AutoETS": MODEL_NAME})[["unique_id", "ds", MODEL_NAME]]

    bottom = con.execute(
        f"""
        SELECT d.state_id || '/' || d.store_id || '/' || d.dept_id || '/' || d.item_id
                   AS unique_id,
               f.target_date AS ds, f.yhat AS {MODEL_NAME}
        FROM {FORECAST} f
        JOIN (SELECT DISTINCT item_id, store_id, dept_id, state_id FROM {FACT_SALES}) d
          ON d.item_id = f.item_id AND d.store_id = f.store_id
        WHERE f.model_name = ? AND f.fold = ? AND f.quantile IS NULL AND f.reconciled = FALSE
        """,
        [MODEL_NAME, fold.index],
    ).df()

    bottom = _fill_unlisted_series(bottom, tags, fold)
    combined = pd.concat([upper, bottom], ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"])
    return combined


def _fill_unlisted_series(bottom: pd.DataFrame, tags: dict, fold: Fold) -> pd.DataFrame:
    """Give every bottom series a forecast, filling not-yet-listed ones with zero.

    MinT needs a base forecast for **every** column of the summing matrix; a gap is not
    something it can reconcile around. LightGBM produces no rows for a SKU that was not yet
    on shelf during a fold, because E2 drops pre-listing rows from the feature panel — so
    `FOODS_3_595` at two stores is missing from fold 0, which is enough to abort the whole
    reconciliation.

    Zero is the correct fill, not a placeholder: an item with no shelf presence has no
    demand, which is exactly what the raw actuals show for those dates. This is the fix for
    the data-source asymmetry recorded as open in E4 — E6 turned it from numerically inert
    into structurally fatal, which is the reason to fix it properly rather than intersect
    the series and quietly shrink the hierarchy.
    """
    expected = list(tags[LEVEL_KEY_BY_NAME["item_store"]])
    targets = pd.to_datetime(pd.Series(fold.target_dates()))
    full = pd.MultiIndex.from_product([expected, targets], names=["unique_id", "ds"])

    filled = (
        bottom.assign(ds=pd.to_datetime(bottom["ds"]))
        .set_index(["unique_id", "ds"])
        .reindex(full)
        .reset_index()
    )
    filled[MODEL_NAME] = filled[MODEL_NAME].fillna(0.0)
    return filled


def coherence_gap(frame: pd.DataFrame, tags: dict, value: str) -> pd.DataFrame:
    """Measure how far each parent is from the sum of its children.

    Args:
        frame: Forecasts with ``unique_id``, ``ds`` and a value column.
        tags: Level tags from ``aggregate``.
        value: Column holding the forecast.

    Returns:
        One row per (parent_level, child_level) with mean and max absolute gap.

    """
    wide = frame.pivot_table(index="ds", columns="unique_id", values=value)
    rows = []
    for parent_name, child_name in (
        ("state", "store"),
        ("store", "store_dept"),
        ("store_dept", "item_store"),
    ):
        parents = [p for p in tags[LEVEL_KEY_BY_NAME[parent_name]] if p in wide.columns]
        children = [c for c in tags[LEVEL_KEY_BY_NAME[child_name]] if c in wide.columns]
        gaps = []
        for parent in parents:
            kids = [c for c in children if c.startswith(f"{parent}/")]
            if kids:
                gaps.append((wide[parent] - wide[kids].sum(axis=1)).abs())
        if not gaps:
            continue
        stacked = pd.concat(gaps)
        rows.append(
            {
                "parent_level": parent_name,
                "child_level": child_name,
                "mean_abs_gap": float(stacked.mean()),
                "max_abs_gap": float(stacked.max()),
                "n_checked": int(stacked.size),
            }
        )
    return pd.DataFrame(rows)


def reconcile(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    *,
    model_name: str = MODEL_NAME,
    method: str = MINT_METHOD,
    replace: bool = True,
) -> ReconciliationRun:
    """Reconcile base forecasts with MinT and persist the coherence evidence.

    Args:
        con: Open warehouse connection.
        folds: Rolling-origin folds.
        model_name: Model whose bottom-level forecasts to reconcile.
        method: MinT covariance estimator.
        replace: Clear existing reconciled rows first.

    Returns:
        A :class:`ReconciliationRun` summary.

    Raises:
        ValueError: If the hierarchy fails validation or no base forecasts exist.

    """
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTrace

    con.execute(FORECAST_DDL)
    con.execute(COHERENCE_DDL)
    if replace:
        con.execute(
            f"DELETE FROM {FORECAST} WHERE model_name = ? AND reconciled = TRUE", [model_name]
        )
        # Aggregate-level base rows are also owned by this function and would otherwise
        # accumulate on every re-run. Scoped by level so E4's bottom-level forecasts --
        # which this function reads rather than produces -- are left untouched.
        con.execute(
            f"""
            DELETE FROM {FORECAST}
            WHERE model_name = ? AND reconciled = FALSE AND level <> 'item_store'
            """,
            [model_name],
        )
        con.execute(f"DELETE FROM {COHERENCE_TABLE} WHERE model_name = ?", [model_name])

    started = time.perf_counter()
    y_df, s_df, tags = build_hierarchy(con)
    validate_hierarchy(y_df, s_df, tags)

    hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method=method)])
    written = 0
    gaps_before: list[float] = []
    gaps_after: list[float] = []
    coherence_rows: list[pd.DataFrame] = []

    for fold in folds:
        y_hat = base_forecasts(con, y_df, tags, fold)
        if y_hat.empty:
            continue
        train = y_df[y_df["ds"] <= pd.Timestamp(fold.origin_date)]

        reconciled = hrec.reconcile(Y_hat_df=y_hat, Y_df=train, S_df=s_df, tags=tags)
        out_col = next(c for c in reconciled.columns if "MinTrace" in c)

        before = coherence_gap(y_hat, tags, MODEL_NAME)
        after = coherence_gap(reconciled, tags, out_col)
        for frame, is_rec in ((before, False), (after, True)):
            frame = frame.assign(model_name=model_name, fold=fold.index, reconciled=is_rec)
            coherence_rows.append(frame)
        gaps_before.append(before["max_abs_gap"].max())
        gaps_after.append(after["max_abs_gap"].max())

        # Persist the aggregate-level *base* forecasts too. Without them the before/after
        # accuracy comparison in E6-S5 could only be recomputed, never queried.
        _write_forecasts(
            con,
            y_hat[y_hat["unique_id"].map(_level_of) != "item_store"],
            MODEL_NAME,
            fold,
            model_name,
            reconciled=False,
            run_id=f"base_agg:{model_name}",
        )
        written += _write_forecasts(
            con,
            reconciled,
            out_col,
            fold,
            model_name,
            reconciled=True,
            run_id=f"mint:{method}",
        )

    if not coherence_rows:
        raise ValueError(
            f"no base forecasts found for {model_name}; run `run-gbm` before reconciling."
        )

    coherence = pd.concat(coherence_rows, ignore_index=True)[
        [
            "model_name",
            "fold",
            "parent_level",
            "child_level",
            "reconciled",
            "mean_abs_gap",
            "max_abs_gap",
            "n_checked",
        ]
    ]
    con.register("coherence_df", coherence)
    con.execute(f"INSERT INTO {COHERENCE_TABLE} SELECT * FROM coherence_df")
    con.unregister("coherence_df")

    return ReconciliationRun(
        model_name=model_name,
        rows_written=written,
        series=int(y_df["unique_id"].nunique()),
        levels=len(LEVEL_NAMES),
        max_gap_before=float(np.max(gaps_before)),
        max_gap_after=float(np.max(gaps_after)),
        elapsed_seconds=time.perf_counter() - started,
    )


def _write_forecasts(
    con: duckdb.DuckDBPyConnection,
    source: pd.DataFrame,
    value_col: str,
    fold: Fold,
    model_name: str,
    *,
    reconciled: bool,
    run_id: str,
) -> int:
    """Write hierarchy-shaped forecasts into ``forecast``."""
    if source.empty:
        return 0
    frame = source[["unique_id", "ds", value_col]].copy()
    frame["level"] = frame["unique_id"].map(_level_of)
    parts = frame["unique_id"].str.split("/", expand=True)
    frame["store_id"] = parts[1] if parts.shape[1] > 1 else None
    frame["dept_id"] = parts[2] if parts.shape[1] > 2 else None
    frame["item_id"] = parts[3] if parts.shape[1] > 3 else None
    frame["run_id"] = run_id
    frame["model_name"] = model_name
    frame["fold"] = fold.index
    frame["origin_date"] = pd.Timestamp(fold.origin_date)
    frame["target_date"] = pd.to_datetime(frame["ds"])
    frame["horizon"] = (frame["target_date"] - pd.Timestamp(fold.origin_date)).dt.days
    frame["quantile"] = np.nan
    # MinT is unconstrained and can push a low forecast below zero; demand cannot be
    # negative and E7 orders against this number.
    frame["yhat"] = np.clip(frame[value_col].to_numpy(), 0.0, None)
    frame["reconciled"] = reconciled

    con.register("rec_df", frame)
    con.execute(f"""
        INSERT INTO {FORECAST}
        SELECT run_id, model_name, fold, origin_date, target_date, horizon, level,
               item_id, store_id, dept_id, quantile, yhat, reconciled
        FROM rec_df
    """)
    con.unregister("rec_df")
    return len(frame)
