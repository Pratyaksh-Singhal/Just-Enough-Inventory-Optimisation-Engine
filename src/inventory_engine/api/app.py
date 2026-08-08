"""E8 — the API. Every handler reads precomputed tables; none trains or fits anything.

That constraint is checked, not just claimed: ``test_no_handler_imports_a_trainer`` in
``tests/test_api.py`` asserts this module does not import ``lightgbm``, ``statsforecast``
or ``hierarchicalforecast`` -- the training-time dependencies -- so a future handler that
started calling ``run-gbm`` logic would fail a test before it shipped.
"""

from __future__ import annotations

import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from inventory_engine.api.deps import get_connection
from inventory_engine.api.schemas import (
    BacktestMetricRow,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HierarchyNode,
    OptimizeRequest,
    OptimizeResponse,
    QuantilePoint,
    StoreOrder,
)
from inventory_engine.data.schema import BACKTEST_METRICS, FACT_SALES, FORECAST
from inventory_engine.models.quantiles import monotonize
from inventory_engine.optimize.newsvendor import interpolate_quantile, quantile_bin_probabilities

app = FastAPI(
    title="Inventory Optimization Engine API",
    description="Serves precomputed forecasts, backtest metrics and order policies. "
    "Never trains or fits a model on request.",
    version="1.0.0",
)

#: The model every read endpoint serves. Matches E7's forecast-source decision: base,
#: unreconciled forecasts at item_store, because that is what orders are placed against.
PRODUCTION_MODEL = "lgbm"
PRODUCTION_LEVEL = "item_store"


@app.get("/health", response_model=HealthResponse)
def health(con=Depends(get_connection)) -> HealthResponse:
    """Liveness/readiness probe: the warehouse is present and has forecast rows."""
    (n,) = con.execute(f"SELECT count(*) FROM {FORECAST}").fetchone()
    return HealthResponse(
        status="ok" if n > 0 else "degraded", warehouse_present=True, forecast_rows=n
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, con=Depends(get_connection)) -> ForecastResponse:
    """Point + quantile forecast for one SKU/store at a given horizon.

    Reads precomputed rows only. There is no live production forecast (E8-S4's nightly job
    owns that gap); this serves the most recently completed backtest fold, which is the
    closest analogue this project has to "the current forecast" until a real clock-driven
    run exists.
    """
    latest_fold = con.execute(
        f"SELECT max(fold) FROM {FORECAST} "
        "WHERE model_name = ? AND level = ? AND reconciled = FALSE",
        [PRODUCTION_MODEL, PRODUCTION_LEVEL],
    ).fetchone()[0]
    if latest_fold is None:
        raise HTTPException(status_code=503, detail="no forecasts precomputed yet")

    rows = con.execute(
        f"""
        SELECT fold, item_id, store_id, origin_date, target_date, quantile, yhat
        FROM {FORECAST}
        WHERE model_name = ? AND level = ? AND reconciled = FALSE AND fold = ?
          AND item_id = ? AND store_id = ? AND horizon = ?
        """,
        [PRODUCTION_MODEL, PRODUCTION_LEVEL, latest_fold, req.sku, req.store, req.horizon],
    ).df()
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no forecast for sku={req.sku!r} store={req.store!r} horizon={req.horizon}; "
            "check the SKU/store exist in scope (see GET /hierarchy)",
        )

    point = rows[rows["quantile"].isna()]
    if point.empty:
        raise HTTPException(status_code=500, detail="point forecast missing for this row set")
    # Fitted quantiles are independent per-level models and are not guaranteed ordered --
    # E4/E5 measured a ~1.3% crossing rate inside the CR band. Every consumer of stored
    # quantiles reads them through monotonize() rather than raw; this endpoint is no
    # exception, or the dashboard would occasionally render q0.9 above q0.95.
    quantile_rows = monotonize(rows[rows["quantile"].notna()]).sort_values("quantile")

    return ForecastResponse(
        sku=req.sku,
        store=req.store,
        horizon=req.horizon,
        origin_date=rows.iloc[0]["origin_date"],
        target_date=rows.iloc[0]["target_date"],
        model_name=PRODUCTION_MODEL,
        point_forecast=float(point.iloc[0]["yhat"]),
        quantiles=[
            QuantilePoint(quantile=float(r.quantile), yhat=float(r.yhat))
            for r in quantile_rows.itertuples()
        ],
        note="most recent completed backtest fold, served as the current forecast",
    )


#: The newsvendor problem is inherently single-period, and the brief's request shape has no
#: date or horizon field to pick one. horizon=1 -- "tomorrow", relative to the latest
#: completed fold's origin -- is the only reading that both matches the classic newsvendor
#: framing and needs no field the brief didn't ask for.
OPTIMIZE_HORIZON = 1


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest, con=Depends(get_connection)) -> OptimizeResponse:
    """Recommended order quantity and expected cost for a SKU, at caller-chosen Cu/Co.

    Aggregated across every store that carries the SKU: the brief's request shape omits
    store, and summing the per-store newsvendor recommendation is the natural reading of
    "how many units of this SKU should be ordered". Cost assumptions arrive per-request as
    a ratio (Cu, Co); only the interpolation is computed here, over quantiles that were
    already fit -- no training, no refitting.

    Pinned to :data:`OPTIMIZE_HORIZON`. Fetching every horizon for the SKU without a date
    filter would mix 28 days' worth of quantile curves per store into one array before
    interpolating -- a first draft of this endpoint did exactly that, and it produced
    plausible-looking numbers over a structurally meaningless (duplicated, unsorted)
    demand distribution. Caught by inspecting row counts, not by the output looking wrong.
    """
    latest_fold = con.execute(
        f"SELECT max(fold) FROM {FORECAST} "
        "WHERE model_name = ? AND level = ? AND reconciled = FALSE",
        [PRODUCTION_MODEL, PRODUCTION_LEVEL],
    ).fetchone()[0]
    if latest_fold is None:
        raise HTTPException(status_code=503, detail="no forecasts precomputed yet")

    rows = con.execute(
        f"""
        SELECT fold, item_id, store_id, target_date, quantile, yhat
        FROM {FORECAST}
        WHERE model_name = ? AND level = ? AND reconciled = FALSE AND fold = ?
          AND item_id = ? AND horizon = ? AND quantile IS NOT NULL
        """,
        [PRODUCTION_MODEL, PRODUCTION_LEVEL, latest_fold, req.sku, OPTIMIZE_HORIZON],
    ).df()
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no priced quantile forecasts for sku={req.sku!r}; check it exists in scope",
        )

    cr = req.cu / (req.cu + req.co)
    ordered = monotonize(rows)

    by_store: list[StoreOrder] = []
    for store_id, group in ordered.groupby("store_id"):
        group = group.sort_values("quantile")
        levels = group["quantile"].to_numpy()
        values = group["yhat"].to_numpy()

        try:
            qty = float(np.clip(interpolate_quantile(levels, values[None, :], cr)[0], 0.0, None))
        except ValueError as exc:
            # A caller-chosen Cu/Co can legitimately push CR outside the fitted grid (e.g.
            # Cu >> Co drives CR -> 1). That is a request the model cannot answer, not a
            # server fault; surface it as a client error with the grid bounds attached
            # rather than a bare 500.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        expected = _expected_cost(levels, values, qty, req.cu, req.co)
        by_store.append(StoreOrder(store=store_id, order_qty=qty, expected_cost=expected))

    return OptimizeResponse(
        sku=req.sku,
        critical_ratio=cr,
        total_order_qty=sum(s.order_qty for s in by_store),
        total_expected_cost=sum(s.expected_cost for s in by_store),
        by_store=sorted(by_store, key=lambda s: s.store),
        note="expected_cost is an estimate from the fitted quantile distribution, not a "
        "backtest result against realised demand",
    )


def _expected_cost(
    levels: np.ndarray, values: np.ndarray, order_qty: float, cu: float, co: float
) -> float:
    """Return the expected newsvendor cost of ``order_qty`` under the fitted quantile distribution.

    Unlike :func:`inventory_engine.optimize.newsvendor.expected_cost_from_distribution`,
    ``cu``/``co`` here are the caller's raw per-unit costs from the request body, not rates
    scaled by a ``CostModel`` and price -- the brief's ``/optimize`` request shape supplies
    them directly.
    """
    probs = quantile_bin_probabilities(levels)
    shortfall = np.maximum(values - order_qty, 0.0)
    leftover = np.maximum(order_qty - values, 0.0)
    return float(np.sum(probs * (shortfall * cu + leftover * co)))


@app.get("/backtest", response_model=list[BacktestMetricRow])
def backtest(
    model_name: str | None = None,
    metric: str | None = None,
    level: str = PRODUCTION_LEVEL,
    con=Depends(get_connection),
) -> list[BacktestMetricRow]:
    """Fold-level metrics -- every fold, not just the mean.

    Returning per-fold rows rather than a pre-aggregated summary is deliberate: the project
    ground rule is to report spread alongside the mean, and a client can always average
    what it's given, but cannot recover the spread from an average alone.
    """
    clauses = ["stratum IS NULL", "level = ?"]
    params: list[object] = [level]
    if model_name:
        clauses.append("model_name = ?")
        params.append(model_name)
    if metric:
        clauses.append("metric = ?")
        params.append(metric)

    rows = con.execute(
        f"SELECT * FROM {BACKTEST_METRICS} WHERE {' AND '.join(clauses)} "
        "ORDER BY model_name, metric, fold",
        params,
    ).df()
    return [BacktestMetricRow(**r) for r in rows.to_dict(orient="records")]


@app.get("/hierarchy", response_model=list[HierarchyNode])
def hierarchy(con=Depends(get_connection)) -> list[HierarchyNode]:
    """Drill-down tree: State -> Store -> Dept -> Item.

    Built from `fact_sales` distinct combinations with plain GROUP BY -- not
    `hierarchicalforecast.aggregate`, which builds the full summing matrix and is overkill
    for a UI tree with no reconciliation happening in this request.
    """
    rows = con.execute(
        f"SELECT DISTINCT state_id, store_id, dept_id, item_id FROM {FACT_SALES} "
        "ORDER BY 1, 2, 3, 4"
    ).df()

    root: dict[str, HierarchyNode] = {}
    for state_id, store_id, dept_id, item_id in rows.itertuples(index=False):
        state = root.setdefault(state_id, HierarchyNode(id=state_id, name=state_id, level="state"))
        store = _find_or_add(state.children, f"{state_id}/{store_id}", store_id, "store")
        dept = _find_or_add(store.children, f"{state_id}/{store_id}/{dept_id}", dept_id, "dept")
        _find_or_add(dept.children, f"{state_id}/{store_id}/{dept_id}/{item_id}", item_id, "item")
    return list(root.values())


def _find_or_add(
    children: list[HierarchyNode], node_id: str, name: str, level: str
) -> HierarchyNode:
    """Return the child with ``node_id``, appending a new one if absent."""
    for child in children:
        if child.id == node_id:
            return child
    node = HierarchyNode(id=node_id, name=name, level=level)  # type: ignore[arg-type]
    children.append(node)
    return node


@app.get("/cost-sensitivity")
def cost_sensitivity(con=Depends(get_connection)) -> list[dict]:
    """Precomputed Cu/Co sensitivity sweep, for E9's cost simulator curve.

    Serving the E7 sweep directly means the dashboard slider never round-trips through a
    fresh computation per pixel -- see E9-S4's "precompute the sweep" requirement.
    """
    return (
        con.execute("SELECT * FROM cost_sensitivity ORDER BY spoilage_rate, policy")
        .df()
        .to_dict(orient="records")
    )


@app.get("/cost-comparison")
def cost_comparison(con=Depends(get_connection)) -> list[dict]:
    """Return the money table, as served data rather than a markdown table."""
    return con.execute("SELECT * FROM cost_comparison").df().to_dict(orient="records")
