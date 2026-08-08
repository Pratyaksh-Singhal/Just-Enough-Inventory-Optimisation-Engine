"""Regenerate ``dashboard/index.html`` from the warehouse.

Queries every table the page shows, computes the exact cost sweep, and inlines the result
into the template as JSON. Run after any pipeline change that moves the numbers.
"""

from __future__ import annotations

import json
import pathlib
import sys

import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_engine.config import WAREHOUSE_PATH  # noqa: E402
from inventory_engine.models.quantiles import monotonize  # noqa: E402

MINT_SUFFIX = "_mint"


def _sweep(con: duckdb.DuckDBPyConnection) -> dict:
    """Precompute shortfall/leftover unit totals per service level.

    Order quantity depends only on the critical ratio, so these two totals are all the page
    needs: ``cost(Cu, Co) = shortfall * Cu + leftover * Co``, exactly. That is what lets the
    slider be exact arithmetic rather than an interpolated approximation.
    """
    raw = con.execute("""
        SELECT fold, item_id, store_id, target_date, quantile, yhat FROM forecast
        WHERE model_name = 'lgbm' AND level = 'item_store' AND reconciled = FALSE
          AND quantile IS NOT NULL
    """).df()
    wide = monotonize(raw).pivot_table(
        index=["fold", "item_id", "store_id", "target_date"], columns="quantile", values="yhat"
    ).reset_index()
    actuals = con.execute("""
        SELECT item_id, store_id, date AS target_date, CAST(units AS DOUBLE) AS demand
        FROM fact_sales
    """).df()
    panel = wide.merge(actuals, on=["item_id", "store_id", "target_date"], how="inner")

    levels = np.array(sorted(c for c in wide.columns if isinstance(c, float)))
    grid = panel[levels].to_numpy(float)
    demand = panel["demand"].to_numpy(float)

    def qty_at(cr: float) -> np.ndarray:
        i = np.clip(np.searchsorted(levels, cr) - 1, 0, len(levels) - 2)
        w = (cr - levels[i]) / (levels[i + 1] - levels[i])
        return np.clip(grid[:, i] + w * (grid[:, i + 1] - grid[:, i]), 0, None)

    def totals(order: np.ndarray) -> dict:
        return {
            "short": float(np.maximum(demand - order, 0).sum()),
            "left": float(np.maximum(order - demand, 0).sum()),
            "stockout_rate": float((demand > order).mean()),
        }

    lagged = con.execute("""
        SELECT item_id, store_id, date + INTERVAL 7 DAY AS target_date,
               CAST(units AS DOUBLE) AS naive_qty
        FROM fact_sales
    """).df()
    naive_qty = (
        panel.merge(lagged, on=["item_id", "store_id", "target_date"], how="left")["naive_qty"]
        .fillna(0)
        .to_numpy()
    )
    return {
        "sweep": [
            {"cr": float(cr), **totals(qty_at(cr))}
            for cr in np.round(np.arange(0.10, 0.991, 0.01), 3)
        ],
        "fixed95": totals(qty_at(0.95)),
        "naive": totals(naive_qty),
        "n_decisions": int(len(panel)),
    }


def _examples(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One representative series per intermittency band, with its forecast interval."""
    out = []
    for stratum in ("dense", "mid", "sparse"):
        item, store = con.execute(
            """
            SELECT f.item_id, f.store_id FROM forecast f
            JOIN dim_item_stratum s ON s.item_id = f.item_id
            WHERE s.stratum_name = ? AND f.model_name = 'lgbm' AND f.fold = 4
              AND f.level = 'item_store' AND f.reconciled = FALSE AND f.quantile IS NULL
            LIMIT 1
            """,
            [stratum],
        ).fetchone()
        bands = con.execute(
            """
            SELECT CAST(target_date AS VARCHAR) AS d, quantile, yhat FROM forecast
            WHERE item_id = ? AND store_id = ? AND model_name = 'lgbm' AND fold = 4
              AND level = 'item_store' AND reconciled = FALSE AND quantile IS NOT NULL
            """,
            [item, store],
        ).df().pivot_table(index="d", columns="quantile", values="yhat")
        out.append(
            {
                "stratum": stratum,
                "item": item,
                "store": store,
                "zero_share": float(
                    con.execute(
                        "SELECT zero_share FROM dim_item_stratum WHERE item_id = ?", [item]
                    ).fetchone()[0]
                ),
                "hist": con.execute(
                    """
                    SELECT CAST(date AS VARCHAR) AS d, units FROM fact_sales
                    WHERE item_id = ? AND store_id = ?
                      AND date BETWEEN DATE '2016-02-25' AND DATE '2016-05-22'
                    ORDER BY date
                    """,
                    [item, store],
                ).df().to_dict("records"),
                "point": con.execute(
                    """
                    SELECT CAST(target_date AS VARCHAR) AS d, yhat FROM forecast
                    WHERE item_id = ? AND store_id = ? AND model_name = 'lgbm' AND fold = 4
                      AND level = 'item_store' AND reconciled = FALSE AND quantile IS NULL
                    ORDER BY target_date
                    """,
                    [item, store],
                ).df().to_dict("records"),
                "lo": [{"d": i, "v": float(r[0.1])} for i, r in bands.iterrows()],
                "hi": [{"d": i, "v": float(r[0.9])} for i, r in bands.iterrows()],
            }
        )
    return out


def build() -> dict:
    """Query the warehouse and assemble every figure the dashboard renders."""
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        data: dict = {"money": con.execute("SELECT * FROM cost_comparison").df().to_dict("records")}
        data.update(_sweep(con))
        data["folds"] = con.execute(
            """
            SELECT model_name, fold, metric, round(value, 4) AS value FROM backtest_fold_metrics
            WHERE stratum IS NULL AND horizon IS NULL AND level = 'item_store'
              AND metric IN ('mase', 'rmsse', 'bias') AND NOT ends_with(model_name, ?)
            ORDER BY metric, model_name, fold
            """,
            [MINT_SUFFIX],
        ).df().to_dict("records")
        data["strata"] = con.execute("""
            SELECT stratum, model_name, round(avg(value), 4) AS mean FROM backtest_fold_metrics
            WHERE metric = 'mase' AND stratum IS NOT NULL AND horizon IS NULL GROUP BY 1, 2
        """).df().to_dict("records")
        for key, metric in (("levels", "rmsse"), ("levels_mase", "mase")):
            data[key] = con.execute(
                """
                SELECT level, model_name, round(avg(value), 4) AS mean FROM backtest_fold_metrics
                WHERE metric = ? AND stratum IS NULL AND horizon IS NULL
                  AND model_name IN ('lgbm', 'lgbm_mint')
                  AND level IN ('state', 'store', 'store_dept', 'item_store')
                GROUP BY 1, 2
                """,
                [metric],
            ).df().to_dict("records")
        data["wrmsse"] = con.execute("""
            SELECT model_name, round(avg(value), 5) AS mean FROM backtest_fold_metrics
            WHERE metric = 'wrmsse' GROUP BY 1
        """).df().to_dict("records")
        data["coherence"] = con.execute("""
            SELECT parent_level, child_level,
                   round(max(CASE WHEN NOT reconciled THEN max_abs_gap END), 3) AS max_before,
                   max(CASE WHEN reconciled THEN max_abs_gap END) AS max_after
            FROM coherence_check GROUP BY 1, 2
        """).df().to_dict("records")
        data["shap"] = con.execute("""
            SELECT feature, round(value, 4) AS value FROM feature_importance
            WHERE method = 'shap_mean_abs' AND fold = (SELECT max(fold) FROM feature_importance)
            ORDER BY value DESC LIMIT 15
        """).df().to_dict("records")
        data["strata_profile"] = con.execute("""
            SELECT dept_id, stratum_name, count(*) AS items,
                   round(min(zero_share), 2) AS lo, round(max(zero_share), 2) AS hi
            FROM dim_item_stratum GROUP BY 1, 2 ORDER BY 1, min(zero_share)
        """).df().to_dict("records")
        data["examples"] = _examples(con)
        return data
    finally:
        con.close()


def main() -> int:
    """Regenerate ``dashboard/index.html`` from the template and the warehouse."""
    template = (ROOT / "dashboard" / "index.template.html").read_text(encoding="utf-8")
    payload = json.dumps(build(), separators=(",", ":"))
    out = ROOT / "dashboard" / "index.html"
    out.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
