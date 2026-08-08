"""E7 — cost model, newsvendor selection, and policy simulation.

The arithmetic here decides how many units go on a shelf, so it is asserted against
hand-computed values rather than eyeballed against a plot.
"""

from __future__ import annotations

import numpy as np
import pytest

from inventory_engine.models.gbm import QUANTILES
from inventory_engine.optimize.costs import CostModel
from inventory_engine.optimize.newsvendor import USE_RECONCILED, interpolate_quantile
from inventory_engine.optimize.simulate import FIXED_SERVICE_LEVEL, simulate_policy

CHEAP_TO_HOLD = CostModel(margin_rate=0.30, spoilage_rate=0.0, holding_rate=0.0)
PERISHABLE = CostModel(margin_rate=0.30, spoilage_rate=1.0, holding_rate=0.0)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_understock_costs_the_margin_not_the_price():
    """The retailer keeps the unit it failed to sell; only the margin is lost."""
    assert CostModel(margin_rate=0.30).understock_cost(10.0) == pytest.approx(3.0)


def test_overstock_costs_the_unit_not_the_price():
    """A spoiled unit costs what it cost to buy, not what it would have sold for."""
    assert PERISHABLE.overstock_cost(10.0) == pytest.approx(7.0)


def test_perishables_push_the_service_level_far_below_95_percent():
    """The counterintuitive headline: high spoilage means stocking *less*, not more."""
    cr = PERISHABLE.critical_ratio()
    assert cr == pytest.approx(0.30 / (0.30 + 0.70))
    assert cr < 0.5, "a thin-margin perishable should not be stocked to the median"
    assert cr < FIXED_SERVICE_LEVEL


def test_non_perishables_push_the_service_level_up():
    """With nothing to lose from leftovers, stock generously."""
    assert CHEAP_TO_HOLD.critical_ratio() == pytest.approx(1.0)


def test_critical_ratio_is_price_invariant():
    """Both costs scale with price, so the ordering quantile does not depend on it."""
    costs = CostModel()
    assert costs.critical_ratio(1.0) == pytest.approx(costs.critical_ratio(97.0))


def test_higher_spoilage_lowers_the_critical_ratio():
    ratios = [CostModel(spoilage_rate=s).critical_ratio() for s in (0.0, 0.3, 0.6, 1.0)]
    assert ratios == sorted(ratios, reverse=True)


@pytest.mark.parametrize(
    "kwargs",
    [{"margin_rate": 0.0}, {"margin_rate": 1.0}, {"spoilage_rate": 1.5}, {"holding_rate": -0.1}],
)
def test_degenerate_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CostModel(**kwargs)


# ---------------------------------------------------------------------------
# Quantile interpolation
# ---------------------------------------------------------------------------

LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def test_interpolation_returns_the_fitted_value_on_a_grid_point():
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert interpolate_quantile(LEVELS, values, 0.5)[0] == pytest.approx(3.0)


def test_interpolation_is_linear_between_grid_points():
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    # Halfway between 0.25 and 0.5 in level-space is halfway between 2.0 and 3.0.
    assert interpolate_quantile(LEVELS, values, 0.375)[0] == pytest.approx(2.5)


def test_interpolation_handles_many_rows():
    values = np.tile(np.arange(7, dtype=float), (100, 1))
    assert interpolate_quantile(LEVELS, values, 0.5).shape == (100,)


def test_critical_ratio_outside_the_grid_is_refused():
    """Extrapolating invents a tail the model never estimated; better to fail loudly."""
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    with pytest.raises(ValueError, match="outside the fitted quantile grid"):
        interpolate_quantile(LEVELS, values, 0.02)


def test_grid_covers_realistic_fresh_food_critical_ratios():
    """The brief's original {0.5, 0.9, 0.95, 0.99} could not express its own premise.

    Fresh-food CRs land between roughly 0.25 and 0.63, below the old grid's floor, so every
    order would have been clamped at the median.
    """
    assert min(QUANTILES) <= 0.25, "grid must reach below the median for perishables"
    realistic = [
        CostModel(margin_rate=m, spoilage_rate=s).critical_ratio()
        for m in (0.25, 0.30, 0.40)
        for s in (0.4, 0.6, 1.0)
    ]
    assert all(min(QUANTILES) <= cr <= max(QUANTILES) for cr in realistic)


# ---------------------------------------------------------------------------
# Policy simulation
# ---------------------------------------------------------------------------


def test_perfect_stocking_costs_nothing():
    demand = np.array([3.0, 5.0, 0.0])
    result = simulate_policy("perfect", demand, demand, np.full(3, 10.0), PERISHABLE)
    assert result.total_cost == pytest.approx(0.0)
    assert result.stockout_rate == pytest.approx(0.0)
    assert result.waste_units == pytest.approx(0.0)


def test_understocking_is_priced_as_lost_margin():
    result = simulate_policy(
        "short", np.array([1.0]), np.array([4.0]), np.array([10.0]), PERISHABLE
    )
    assert result.stockout_units == pytest.approx(3.0)
    assert result.understock_cost == pytest.approx(3.0 * 3.0)  # 3 units x 30% of 10
    assert result.stockout_rate == pytest.approx(1.0)


def test_overstocking_is_priced_as_spoilage():
    result = simulate_policy("over", np.array([6.0]), np.array([2.0]), np.array([10.0]), PERISHABLE)
    assert result.waste_units == pytest.approx(4.0)
    assert result.overstock_cost == pytest.approx(4.0 * 7.0)  # 4 units x unit cost 7
    assert result.stockout_rate == pytest.approx(0.0)


def test_simulation_uses_realised_demand_not_the_forecast():
    """Scoring a policy against its own forecast would measure nothing.

    A wildly biased order should be punished by the actuals, not excused by agreeing with
    the model that produced it.
    """
    demand = np.array([2.0, 2.0])
    biased = np.array([50.0, 50.0])
    assert simulate_policy("biased", biased, demand, np.full(2, 10.0), PERISHABLE).total_cost > 0


def test_zero_demand_days_cannot_stock_out():
    result = simulate_policy(
        "idle", np.array([1.0, 1.0]), np.zeros(2), np.full(2, 10.0), PERISHABLE
    )
    assert result.stockout_rate == pytest.approx(0.0)
    assert result.waste_units == pytest.approx(2.0)


def test_holding_and_spoilage_are_reported_separately():
    costs = CostModel(margin_rate=0.30, spoilage_rate=0.5, holding_rate=0.1)
    result = simulate_policy("mix", np.array([5.0]), np.array([1.0]), np.array([10.0]), costs)
    assert result.overstock_cost == pytest.approx(4.0 * 7.0 * 0.5)
    assert result.holding_cost == pytest.approx(4.0 * 7.0 * 0.1)
    assert result.total_cost == pytest.approx(result.overstock_cost + result.holding_cost)


# ---------------------------------------------------------------------------
# The forecast-source decision
# ---------------------------------------------------------------------------


def test_newsvendor_consumes_unreconciled_forecasts():
    """Explicit decision, not a default.

    MinT trades bottom-level accuracy for cross-level coherence. Orders are placed at
    item_store, so the cost function optimises against the most accurate forecast at that
    grain. Reconciled forecasts serve the dashboard's aggregate planning views instead.
    """
    assert USE_RECONCILED is False
