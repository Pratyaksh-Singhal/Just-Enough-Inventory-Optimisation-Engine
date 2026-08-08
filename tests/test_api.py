"""E8-S6 — API contract tests against a synthetic fixture warehouse.

Every table the API reads (`forecast`, `backtest_fold_metrics`, `dim_item_stratum`,
`cost_comparison`, `cost_sensitivity`) is built directly in a temp DuckDB file, so these
tests run in milliseconds and never touch the real 300MB warehouse.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from inventory_engine.data.schema import (
    BACKTEST_METRICS_DDL,
    DIM_CALENDAR_DDL,
    FACT_SALES_DDL,
    FORECAST_DDL,
)

ITEM = "FOODS_1_001"
STORE = "CA_1"
DEPT = "FOODS_1"
STATE = "CA"
ORIGIN = date(2016, 4, 24)
FOLD = 4
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _build_fixture_warehouse(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute(DIM_CALENDAR_DDL)
    con.execute(FACT_SALES_DDL)
    con.execute(FORECAST_DDL)
    con.execute(BACKTEST_METRICS_DDL)
    con.execute("""
        CREATE TABLE dim_item_stratum (
            item_id VARCHAR, dept_id VARCHAR, zero_share DOUBLE,
            stratum INTEGER, stratum_name VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE cost_comparison (
            policy VARCHAR, margin_rate DOUBLE, spoilage_rate DOUBLE, critical_ratio DOUBLE,
            stockout_rate DOUBLE, stockout_units DOUBLE, waste_units DOUBLE,
            understock_cost DOUBLE, holding_cost DOUBLE, overstock_cost DOUBLE,
            total_cost DOUBLE, n_rows INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE cost_sensitivity (
            spoilage_rate DOUBLE, margin_rate DOUBLE, critical_ratio DOUBLE,
            policy VARCHAR, total_cost DOUBLE, saving_vs_naive DOUBLE
        )
    """)

    # fact_sales: enough history for a plausible price/units row, two items so /hierarchy
    # has more than one leaf.
    rows = []
    for item in (ITEM, "FOODS_1_002"):
        for i in range(10):
            rows.append(
                (ORIGIN - timedelta(days=i), item, STORE, DEPT, "FOODS", STATE, 3, 2.50, 0, None)
            )
    con.executemany("INSERT INTO fact_sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # forecast: point + a deliberately CROSSED quantile grid for (ITEM, STORE, fold=4,
    # horizon=1), so the monotonization contract is actually exercised, not just assumed.
    target = ORIGIN + timedelta(days=1)
    crossed_yhat = {0.1: 1.0, 0.25: 2.0, 0.5: 3.0, 0.75: 2.5, 0.9: 5.0, 0.95: 4.5, 0.99: 6.0}
    forecast_rows = [
        (
            "run:lgbm",
            "lgbm",
            FOLD,
            ORIGIN,
            target,
            1,
            "item_store",
            ITEM,
            STORE,
            DEPT,
            None,
            3.2,
            False,
        ),
    ]
    for q, yhat in crossed_yhat.items():
        forecast_rows.append(
            (
                "run:lgbm",
                "lgbm",
                FOLD,
                ORIGIN,
                target,
                1,
                "item_store",
                ITEM,
                STORE,
                DEPT,
                q,
                yhat,
                False,
            )
        )
    # Second store, clean (non-crossed) grid, so /optimize's per-store aggregation is
    # exercised across more than one store.
    for q, yhat in (
        (0.1, 0.5),
        (0.25, 1.0),
        (0.5, 2.0),
        (0.75, 3.0),
        (0.9, 4.0),
        (0.95, 4.5),
        (0.99, 5.0),
    ):
        forecast_rows.append(
            (
                "run:lgbm",
                "lgbm",
                FOLD,
                ORIGIN,
                target,
                1,
                "item_store",
                ITEM,
                "CA_2",
                DEPT,
                q,
                yhat,
                False,
            )
        )
    # A different horizon for the same item/store/fold, to prove /optimize does not mix it
    # into the horizon=1 curve (the bug this endpoint was rewritten to avoid).
    for q in QUANTILES:
        forecast_rows.append(
            (
                "run:lgbm",
                "lgbm",
                FOLD,
                ORIGIN,
                ORIGIN + timedelta(days=2),
                2,
                "item_store",
                ITEM,
                STORE,
                DEPT,
                q,
                999.0,
                False,
            )
        )

    con.executemany(
        """INSERT INTO forecast
           (run_id, model_name, fold, origin_date, target_date, horizon, level,
            item_id, store_id, dept_id, quantile, yhat, reconciled)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        forecast_rows,
    )

    # backtest_fold_metrics: a couple of folds so /backtest has spread to return.
    metric_rows = [
        ("scored:lgbm", "lgbm", f, "item_store", None, None, "mase", 1.0 + f * 0.01, 100, 0)
        for f in range(3)
    ]
    con.executemany(
        """INSERT INTO backtest_fold_metrics
           (run_id, model_name, fold, level, stratum, horizon, metric, value, n_series, n_excluded)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        metric_rows,
    )

    con.execute(
        "INSERT INTO cost_comparison VALUES ('Newsvendor + our forecast', 0.3, 0.6, 0.41, "
        "0.48, 100.0, 20.0, 500.0, 10.0, 300.0, 810.0, 1000)"
    )
    con.execute("INSERT INTO cost_sensitivity VALUES (0.6, 0.3, 0.41, 'newsvendor', 810.0, 0.29)")
    con.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a fresh fixture warehouse, isolated from the real one."""
    db_path = tmp_path / "fixture.duckdb"
    _build_fixture_warehouse(db_path)
    monkeypatch.setattr("inventory_engine.api.deps.WAREHOUSE_PATH", db_path)

    # The app module is already imported; re-import isn't needed since deps.get_connection
    # reads WAREHOUSE_PATH at call time via the module reference, not a captured value.
    from inventory_engine.api.app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_ok_with_data(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["forecast_rows"] > 0


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------


def test_forecast_returns_point_and_all_quantiles(client):
    r = client.post("/forecast", json={"sku": ITEM, "store": STORE, "horizon": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["point_forecast"] == pytest.approx(3.2)
    assert {q["quantile"] for q in body["quantiles"]} == set(QUANTILES)


def test_forecast_quantiles_are_monotonized_not_raw(client):
    """The fixture's stored grid is deliberately crossed (q0.75 < q0.5, q0.95 < q0.9).

    If this endpoint served raw values, the response would reproduce that disorder. It
    must not -- the API is a consumer of the same monotonize() contract E4/E5 established.
    """
    r = client.post("/forecast", json={"sku": ITEM, "store": STORE, "horizon": 1})
    values = [q["yhat"] for q in sorted(r.json()["quantiles"], key=lambda q: q["quantile"])]
    assert values == sorted(values)
    # And it must not have silently discarded any of the crossed values -- rearrangement,
    # not truncation.
    assert sorted(values) == sorted([1.0, 2.0, 3.0, 2.5, 5.0, 4.5, 6.0])


def test_forecast_unknown_sku_is_404_with_useful_message(client):
    r = client.post("/forecast", json={"sku": "DOES_NOT_EXIST", "store": STORE, "horizon": 1})
    assert r.status_code == 404
    assert "DOES_NOT_EXIST" in r.json()["detail"]


def test_forecast_rejects_horizon_out_of_range(client):
    for bad in (0, 29, -1):
        r = client.post("/forecast", json={"sku": ITEM, "store": STORE, "horizon": bad})
        assert r.status_code == 422


def test_forecast_missing_fields_are_422_not_500(client):
    r = client.post("/forecast", json={"sku": ITEM})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /optimize
# ---------------------------------------------------------------------------


def test_optimize_computes_the_requested_critical_ratio(client):
    r = client.post("/optimize", json={"sku": ITEM, "cu": 3.0, "co": 4.0})
    assert r.status_code == 200
    assert r.json()["critical_ratio"] == pytest.approx(3.0 / 7.0)


def test_optimize_aggregates_across_every_store_carrying_the_sku(client):
    """Fixture has the item at both STORE (crossed grid) and CA_2 (clean grid)."""
    r = client.post("/optimize", json={"sku": ITEM, "cu": 3.0, "co": 4.0})
    stores = {s["store"] for s in r.json()["by_store"]}
    assert stores == {STORE, "CA_2"}
    total = sum(s["order_qty"] for s in r.json()["by_store"])
    assert r.json()["total_order_qty"] == pytest.approx(total)


def test_optimize_does_not_mix_horizons(client):
    """The fixture plants a horizon=2 row with yhat=999 for the same item/store/fold.

    An earlier version of this endpoint pulled every horizon for the SKU without a date
    filter, mixing 28 days of quantile curves into one array before interpolating. If that
    regressed, this order quantity would be enormous (contaminated by the 999 sentinel).
    """
    r = client.post("/optimize", json={"sku": ITEM, "cu": 3.0, "co": 4.0})
    for store in r.json()["by_store"]:
        assert store["order_qty"] < 100, "order quantity contaminated by a different horizon"


def test_optimize_rejects_non_positive_costs(client):
    for cu, co in ((0.0, 4.0), (-1.0, 4.0), (3.0, 0.0), (3.0, -1.0)):
        r = client.post("/optimize", json={"sku": ITEM, "cu": cu, "co": co})
        assert r.status_code == 422


def test_optimize_unknown_sku_is_404(client):
    r = client.post("/optimize", json={"sku": "NOPE", "cu": 3.0, "co": 4.0})
    assert r.status_code == 404


def test_optimize_high_ratio_within_grid_orders_toward_the_top(client):
    """Cu > Co: understocking is more expensive, so order above the median."""
    r = client.post("/optimize", json={"sku": ITEM, "cu": 9.0, "co": 1.0})
    assert r.json()["critical_ratio"] == pytest.approx(0.9)
    median = client.post("/optimize", json={"sku": ITEM, "cu": 1.0, "co": 1.0}).json()
    assert r.json()["total_order_qty"] > median["total_order_qty"]


def test_optimize_ratio_outside_fitted_grid_is_422_not_500(client):
    """Cu >> Co pushes CR toward 1.0, past the top fitted level (0.99).

    interpolate_quantile refuses to extrapolate a tail the model never estimated -- this
    endpoint's first version let that ValueError propagate as an unhandled 500 instead of
    telling the caller their Cu/Co choice is outside what the model can answer.
    """
    r = client.post("/optimize", json={"sku": ITEM, "cu": 1000.0, "co": 0.01})
    assert r.status_code == 422
    assert "outside the fitted quantile grid" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /backtest
# ---------------------------------------------------------------------------


def test_backtest_returns_every_fold_not_a_pre_averaged_summary(client):
    r = client.get("/backtest", params={"model_name": "lgbm", "metric": "mase"})
    assert r.status_code == 200
    assert len(r.json()) == 3  # one row per fold in the fixture
    assert {row["fold"] for row in r.json()} == {0, 1, 2}


def test_backtest_filters_are_optional(client):
    r = client.get("/backtest")
    assert r.status_code == 200
    assert len(r.json()) >= 3


# ---------------------------------------------------------------------------
# /hierarchy
# ---------------------------------------------------------------------------


def test_hierarchy_is_a_nested_state_store_dept_item_tree(client):
    r = client.get("/hierarchy")
    assert r.status_code == 200
    tree = r.json()
    assert len(tree) == 1  # one state in the fixture
    assert tree[0]["level"] == "state"
    store = tree[0]["children"][0]
    assert store["level"] == "store"
    dept = store["children"][0]
    assert dept["level"] == "dept"
    items = {c["id"].rsplit("/", 1)[-1] for c in dept["children"]}
    assert items == {ITEM, "FOODS_1_002"}


# ---------------------------------------------------------------------------
# /cost-comparison, /cost-sensitivity
# ---------------------------------------------------------------------------


def test_cost_comparison_serves_the_money_table(client):
    r = client.get("/cost-comparison")
    assert r.status_code == 200
    assert r.json()[0]["policy"] == "Newsvendor + our forecast"


def test_cost_sensitivity_serves_the_sweep(client):
    r = client.get("/cost-sensitivity")
    assert r.status_code == 200
    assert r.json()[0]["spoilage_rate"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Never trains on request
# ---------------------------------------------------------------------------


def test_no_handler_imports_a_trainer():
    """Static check: the app module must not import training-time dependencies.

    A handler that started calling GBM-fitting or MinT-reconciling code would violate
    "never train on request" in a way that's easy to miss in review but easy to catch here:
    those libraries have no legitimate reason to be imported by request-handling code.
    """
    module = sys.modules.get("inventory_engine.api.app")
    assert module is not None, "import the app module before running this test"
    source = Path(module.__file__).read_text(encoding="utf-8")
    for banned in (
        "import lightgbm",
        "import statsforecast",
        "import hierarchicalforecast",
        "import mlflow",
    ):
        assert banned not in source, f"app.py must not import {banned.split()[-1]}"
