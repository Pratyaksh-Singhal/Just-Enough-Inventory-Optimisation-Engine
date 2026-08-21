"""Stratum-aware model selection: AutoETS on sparse series, LightGBM on dense and mid."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import Fold
from inventory_engine.data.schema import DIM_ITEM_STRATUM, FACT_SALES, FORECAST, FORECAST_DDL
from inventory_engine.models.baselines import SEASON_LENGTH
from inventory_engine.models.gbm import QUANTILES

MODEL_NAME: Final = "hybrid"
LEVEL_ITEM_STORE: Final = "item_store"

#: Which model serves which intermittency band, decided by the E3/E4 backtests.
ROUTING: Final[dict[str, str]] = {"dense": "lgbm", "mid": "lgbm", "sparse": "ets"}

#: Quantile level -> (AutoETS prediction-interval width, which bound to read).
_INTERVAL_FOR: Final[dict[float, tuple[int, str]]] = {
    0.1: (80, "lo"),
    0.25: (50, "lo"),
    0.75: (50, "hi"),
    0.9: (80, "hi"),
    0.95: (90, "hi"),
    0.99: (98, "hi"),
}


@dataclass(frozen=True)
class HybridRun:
    """Outcome of assembling the hybrid forecast."""

    rows_written: int
    series: int
    sparse_series: int
    routed_from: dict[str, int]
    elapsed_seconds: float

    def render(self) -> str:
        """Multi-line summary."""
        routes = "  ".join(f"{k}->{v:,} rows" for k, v in sorted(self.routed_from.items()))
        return "\n".join(
            [
                f"  {MODEL_NAME:<16} {self.rows_written:>9,} rows"
                f"  {self.series:>5} series  {self.elapsed_seconds:>7.1f}s",
                f"  routing  {routes}",
                f"  sparse stratum: {self.sparse_series} series served by AutoETS",
            ]
        )


def stratum_routing(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Map every item to the model that serves its intermittency band."""
    strata = con.execute(f"SELECT item_id, stratum_name FROM {DIM_ITEM_STRATUM}").df()
    strata["source_model"] = strata["stratum_name"].map(ROUTING)
    if strata["source_model"].isna().any():
        unknown = sorted(strata.loc[strata["source_model"].isna(), "stratum_name"].unique())
        raise ValueError(f"no routing rule for stratum(s): {unknown}; expected {sorted(ROUTING)}")
    return strata


def _ets_quantiles(
    con: duckdb.DuckDBPyConnection, folds: tuple[Fold, ...], items: list[str]
) -> pd.DataFrame:
    """Refit AutoETS on sparse series with prediction intervals, as quantile rows."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    if not items:
        return pd.DataFrame()

    placeholders = ", ".join("?" for _ in items)
    frames = []
    for fold in folds:
        train = con.execute(
            f"""
            SELECT item_id || '|' || store_id AS unique_id, date AS ds,
                   CAST(units AS DOUBLE) AS y
            FROM {FACT_SALES}
            WHERE date <= ? AND item_id IN ({placeholders})
            ORDER BY unique_id, ds
            """,
            [fold.origin_date, *items],
        ).df()

        sf = StatsForecast(models=[AutoETS(season_length=SEASON_LENGTH)], freq="D", n_jobs=1)
        widths = sorted({width for width, _ in _INTERVAL_FOR.values()})
        wide = sf.forecast(df=train, h=fold.horizon, level=widths)
        if wide.index.name == "unique_id":
            wide = wide.reset_index()

        for q in QUANTILES:
            if q == 0.5:
                column = "AutoETS"
            else:
                width, bound = _INTERVAL_FOR[q]
                column = f"AutoETS-{bound}-{width}"
            if column not in wide.columns:
                continue
            part = wide[["unique_id", "ds", column]].rename(columns={column: "yhat"})
            part = part.dropna(subset=["yhat"])
            ids = part["unique_id"].str.split("|", n=1, expand=True)
            frames.append(
                pd.DataFrame(
                    {
                        "fold": fold.index,
                        "origin_date": pd.Timestamp(fold.origin_date),
                        "target_date": pd.to_datetime(part["ds"]).to_numpy(),
                        "item_id": ids[0].to_numpy(),
                        "store_id": ids[1].to_numpy(),
                        "quantile": q,
                        "yhat": np.clip(part["yhat"].to_numpy(), 0.0, None),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_hybrid(
    con: duckdb.DuckDBPyConnection,
    folds: tuple[Fold, ...],
    *,
    replace: bool = True,
) -> HybridRun:
    """Assemble the hybrid forecast from stored per-model rows."""
    con.execute(FORECAST_DDL)
    if replace:
        con.execute(f"DELETE FROM {FORECAST} WHERE model_name = ?", [MODEL_NAME])

    started = time.perf_counter()
    routing = stratum_routing(con)
    con.register("routing_df", routing)

    # Point forecasts: pick each series' row from the model its stratum routes to.
    con.execute(f"""
        INSERT INTO {FORECAST}
        SELECT 'hybrid:' || f.model_name, '{MODEL_NAME}', f.fold, f.origin_date,
               f.target_date, f.horizon, f.level, f.item_id, f.store_id, f.dept_id,
               f.quantile, f.yhat, f.reconciled
        FROM {FORECAST} f
        JOIN routing_df r ON r.item_id = f.item_id AND r.source_model = f.model_name
        WHERE f.quantile IS NULL AND f.reconciled = FALSE
    """)
    routed = dict(
        con.execute(
            f"""
            SELECT run_id, count(*) FROM {FORECAST}
            WHERE model_name = '{MODEL_NAME}' AND quantile IS NULL GROUP BY 1
            """
        ).fetchall()
    )

    # Quantiles: LightGBM for the bands it serves.
    con.execute(f"""
        INSERT INTO {FORECAST}
        SELECT 'hybrid:lgbm', '{MODEL_NAME}', f.fold, f.origin_date, f.target_date,
               f.horizon, f.level, f.item_id, f.store_id, f.dept_id, f.quantile, f.yhat,
               f.reconciled
        FROM {FORECAST} f
        JOIN routing_df r ON r.item_id = f.item_id
        WHERE f.model_name = 'lgbm' AND f.quantile IS NOT NULL AND f.reconciled = FALSE
          AND r.source_model = 'lgbm'
    """)

    # Quantiles: AutoETS intervals for the sparse band.
    sparse_items = routing.loc[routing["source_model"] == "ets", "item_id"].tolist()
    ets_q = _ets_quantiles(con, folds, sparse_items)
    if not ets_q.empty:
        depts = con.execute(f"SELECT DISTINCT item_id, dept_id FROM {FACT_SALES}").df()
        ets_q = ets_q.merge(depts, on="item_id", how="left")
        ets_q["horizon"] = (
            pd.to_datetime(ets_q["target_date"]) - pd.to_datetime(ets_q["origin_date"])
        ).dt.days
        ets_q["run_id"] = "hybrid:ets"
        ets_q["model_name"] = MODEL_NAME
        ets_q["level"] = LEVEL_ITEM_STORE
        ets_q["reconciled"] = False
        con.register("ets_q_df", ets_q)
        con.execute(f"""
            INSERT INTO {FORECAST}
            SELECT run_id, model_name, fold, origin_date, target_date, horizon, level,
                   item_id, store_id, dept_id, quantile, yhat, reconciled
            FROM ets_q_df
        """)
        con.unregister("ets_q_df")
    con.unregister("routing_df")

    total, series = con.execute(
        f"""
        SELECT count(*), count(DISTINCT (item_id, store_id))
        FROM {FORECAST} WHERE model_name = ?
        """,
        [MODEL_NAME],
    ).fetchone()
    if total == 0:
        raise ValueError(
            "hybrid produced no rows; run `run-baselines` and `run-gbm` first so there "
            "are lgbm and ets forecasts to route between."
        )

    return HybridRun(
        rows_written=total,
        series=series,
        sparse_series=len(sparse_items) * 4,
        routed_from=routed,
        elapsed_seconds=time.perf_counter() - started,
    )
