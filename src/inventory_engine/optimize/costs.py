"""E7-S1 — the cost model. Every number here is an assumption, and is labelled as one.

M5 ships unit *prices*, not margins, spoilage rates or holding costs. Everything below the
price is therefore invented. That does not make the result meaningless — the newsvendor
conclusion is driven by the **ratio** Cu/Co rather than either level, and E7-S4 sweeps that
ratio across its plausible range so the finding can be judged rather than taken on trust.
But it does mean no figure in this module should be quoted as a measurement.

The two costs
-------------
``Cu`` — **understock**. One unit of demand arrives, the shelf is empty, the sale is lost.
The cost is the lost gross margin, not the price: the retailer keeps the unit it did not
sell.

``Co`` — **overstock**. One unit is stocked and not sold. For fresh food most of that unit
is written off, and the cost approaches the full unit cost rather than a small carrying
charge. This is what drives the counterintuitive result: when overstocking costs nearly a
whole unit and understocking costs only the margin, the optimal service level is *low*.

Why the answer is not 95%
-------------------------
``CR = Cu / (Cu + Co)``. With a 30% gross margin and 60% of unsold units spoiling::

    Cu = price x 0.30
    Co = price x 0.70 x 0.60 = price x 0.42
    CR = 0.30 / 0.72 = 0.417

So the cost-optimal service level is roughly **42%**, not the 95% that gets picked by
default. Ordering to a 95th percentile on a perishable thin-margin item is not cautious;
it is systematically expensive, and E7-S3 prices exactly how expensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Gross margin as a fraction of shelf price. **Assumption.** US grocery gross margins run
#: roughly 25-35%; fresh food sits toward the lower end.
DEFAULT_MARGIN_RATE: Final = 0.30

#: Fraction of unsold units written off rather than carried or marked down.
#: **Assumption**, and the single most influential one in the model. 1.0 would be a
#: same-day-perishable; 0.0 a tin that keeps forever. 0.6 represents fresh food where most
#: unsold stock is lost but some is recovered through markdown.
DEFAULT_SPOILAGE_RATE: Final = 0.60

#: Daily holding cost as a fraction of unit cost — storage, capital, shrink. **Assumption**,
#: and deliberately small: for perishables it is dominated by spoilage.
DEFAULT_HOLDING_RATE: Final = 0.02

#: Fallback shelf price where M5 has none. **Assumption**, used only for rows whose price is
#: NULL; those rows are unlisted anyway and carry near-zero demand.
FALLBACK_PRICE: Final = 3.00


@dataclass(frozen=True)
class CostModel:
    """Newsvendor cost parameters. Every field is a stated assumption, not a measurement.

    Attributes:
        margin_rate: Gross margin as a fraction of shelf price.
        spoilage_rate: Fraction of unsold units written off.
        holding_rate: Daily holding cost as a fraction of unit cost.

    """

    margin_rate: float = DEFAULT_MARGIN_RATE
    spoilage_rate: float = DEFAULT_SPOILAGE_RATE
    holding_rate: float = DEFAULT_HOLDING_RATE

    def __post_init__(self) -> None:
        """Reject parameter values that would make the critical ratio meaningless."""
        if not 0.0 < self.margin_rate < 1.0:
            raise ValueError(f"margin_rate must be in (0, 1), got {self.margin_rate}")
        if not 0.0 <= self.spoilage_rate <= 1.0:
            raise ValueError(f"spoilage_rate must be in [0, 1], got {self.spoilage_rate}")
        if self.holding_rate < 0.0:
            raise ValueError(f"holding_rate must be >= 0, got {self.holding_rate}")

    def unit_cost(self, price: float) -> float:
        """Wholesale cost of one unit."""
        return price * (1.0 - self.margin_rate)

    def understock_cost(self, price: float) -> float:
        """``Cu`` — lost gross margin on one unit of unmet demand."""
        return price * self.margin_rate

    def overstock_cost(self, price: float) -> float:
        """``Co`` — spoilage plus holding on one unit left unsold."""
        cost = self.unit_cost(price)
        return cost * self.spoilage_rate + cost * self.holding_rate

    def critical_ratio(self, price: float = 1.0) -> float:
        """``CR = Cu / (Cu + Co)`` — the quantile of demand to stock to.

        Price-invariant under this model, because both costs scale linearly with price. The
        argument is kept so a per-item cost model can be substituted later without changing
        every call site.
        """
        cu = self.understock_cost(price)
        co = self.overstock_cost(price)
        total = cu + co
        return cu / total if total > 0 else 0.5

    def describe(self) -> str:
        """One-line summary, with the assumption flag attached."""
        return (
            f"margin={self.margin_rate:.0%} spoilage={self.spoilage_rate:.0%} "
            f"holding={self.holding_rate:.0%}/day -> CR={self.critical_ratio():.4f} "
            "(all values are assumptions)"
        )


DEFAULT_COST_MODEL: Final = CostModel()
