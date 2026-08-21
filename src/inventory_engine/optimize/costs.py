"""E7-S1 — the cost model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Gross margin as a fraction of shelf price. **Assumption.** US grocery gross margins run
#: roughly 25-35%; fresh food sits toward the lower end.
DEFAULT_MARGIN_RATE: Final = 0.30

#: Fraction of unsold units written off rather than carried or marked down.
DEFAULT_SPOILAGE_RATE: Final = 0.60

#: Daily holding cost as a fraction of unit cost — storage, capital, shrink. **Assumption**,
#: and deliberately small: for perishables it is dominated by spoilage.
DEFAULT_HOLDING_RATE: Final = 0.02

#: Fallback shelf price where M5 has none. **Assumption**, used only for rows whose price is
#: NULL; those rows are unlisted anyway and carry near-zero demand.
FALLBACK_PRICE: Final = 3.00


@dataclass(frozen=True)
class CostModel:
    """Newsvendor cost parameters."""

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
        """``CR = Cu / (Cu + Co)`` — the quantile of demand to stock to."""
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
