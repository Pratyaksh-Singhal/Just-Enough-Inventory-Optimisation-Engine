"""E8-S1/S2/S3 — request and response models."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Body for ``POST /forecast``."""

    sku: str = Field(..., description="M5 item_id, e.g. FOODS_1_001")
    store: str = Field(..., description="M5 store_id, e.g. CA_1")
    horizon: int = Field(..., ge=1, le=28, description="Days ahead, 1-28")


class QuantilePoint(BaseModel):
    """One fitted quantile level and its forecast."""

    quantile: float
    yhat: float


class ForecastResponse(BaseModel):
    """Point and quantile forecast for one SKU/store/horizon."""

    sku: str
    store: str
    horizon: int
    origin_date: date
    target_date: date
    model_name: str
    point_forecast: float
    quantiles: list[QuantilePoint]
    note: str = Field(
        description="No live production forecast run exists (E8-S4 owns that); this is "
        "the most recent completed backtest fold, served as the current forecast."
    )


class OptimizeRequest(BaseModel):
    """Body for ``POST /optimize``.

    Deliberately store-less, matching the brief's request shape. A SKU can be listed at
    several stores; see :func:`inventory_engine.api.app.optimize` for how that is resolved.
    """

    sku: str
    cu: float = Field(..., gt=0, description="Understock cost per unit (lost margin)")
    co: float = Field(..., gt=0, description="Overstock cost per unit (spoilage + holding)")


class StoreOrder(BaseModel):
    """One store's contribution to an aggregated SKU-level order."""

    store: str
    order_qty: float
    expected_cost: float


class OptimizeResponse(BaseModel):
    """Recommended order quantity and expected cost for a SKU, aggregated across stores."""

    sku: str
    critical_ratio: float
    total_order_qty: float
    total_expected_cost: float
    by_store: list[StoreOrder]
    note: str = Field(
        description="Priced for the next single period (horizon=1 -- the classic "
        "newsvendor framing) at the most recently completed fold. expected_cost is "
        "computed from the fitted quantile distribution, not from realised demand -- an "
        "estimate for a not-yet-observed period, not a backtest result. See "
        "docs/backlog/epic-08-api.md."
    )


class BacktestMetricRow(BaseModel):
    """One row of ``backtest_fold_metrics``."""

    model_name: str
    fold: int
    level: str
    stratum: str | None
    horizon: int | None
    metric: str
    value: float | None
    n_series: int | None
    n_excluded: int | None


class HierarchyNode(BaseModel):
    """One node of the drill-down tree."""

    id: str
    name: str
    level: Literal["state", "store", "dept", "item"]
    children: list[HierarchyNode] = Field(default_factory=list)


HierarchyNode.model_rebuild()


class HealthResponse(BaseModel):
    """``GET /health``."""

    status: Literal["ok", "degraded"]
    warehouse_present: bool
    forecast_rows: int
