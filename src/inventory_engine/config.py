"""Central configuration: filesystem paths, dataset scope, and shared constants.

Everything that another module might want to tune lives here so that scope
decisions stay auditable in one place rather than scattered through SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"

RAW_SALES_FILE = "sales_train_evaluation.csv"
RAW_CALENDAR_FILE = "calendar.csv"
RAW_PRICES_FILE = "sell_prices.csv"

REQUIRED_RAW_FILES: tuple[str, ...] = (
    RAW_SALES_FILE,
    RAW_CALENDAR_FILE,
    RAW_PRICES_FILE,
)

KAGGLE_DATASET_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """The slice of M5 this project models.

    Scoping is an explicit engineering decision, not an accident: the full M5
    dataset is 30,490 series, which turns every backtest into an overnight job
    and buries the optimization work under data engineering. Restricting to one
    state and one category keeps the hierarchy genuinely multi-level
    (State -> Store -> Category -> Dept -> Item) while staying small enough to
    iterate on in minutes.

    Attributes:
        state_id: M5 state code to keep.
        cat_id: M5 category code to keep.
        dept_ids: Optional department filter within ``cat_id``. ``None`` keeps
            every department. Used to trade series count against breadth.
        items_per_dept: Optional per-department item budget. ``None`` keeps every
            item. When set, items are drawn by *stratified* sample (see
            :data:`N_STRATA`) rather than by volume rank.
        n_strata: Number of intermittency strata to sample equally from.
        sampling_seed: Salt for the deterministic item hash. Fixed so that the
            sampled universe is byte-identical on every rebuild and machine.

    """

    state_id: str = "CA"
    cat_id: str = "FOODS"
    dept_ids: tuple[str, ...] | None = None
    items_per_dept: int | None = 60
    n_strata: int = 3
    sampling_seed: int = 20260807

    def describe(self) -> str:
        """Return a one-line human-readable summary of the scope."""
        parts = [f"state={self.state_id}", f"cat={self.cat_id}"]
        parts.append(f"depts={','.join(self.dept_ids)}" if self.dept_ids else "depts=all")
        if self.items_per_dept is not None:
            parts.append(
                f"stratified{self.items_per_dept}/dept"
                f" x{self.n_strata}strata seed={self.sampling_seed}"
            )
        else:
            parts.append("items=all")
        return " ".join(parts)


DEFAULT_SCOPE = Scope()


# ---------------------------------------------------------------------------
# Dataset facts (M5 constants, not tunables)
# ---------------------------------------------------------------------------

#: Number of daily observations in ``sales_train_evaluation.csv``.
N_TRAIN_DAYS = 1941

#: The M5 calendar extends past the sales data so that "days until next event"
#: features remain computable at the end of the training window.
N_CALENDAR_DAYS = 1969

#: Forecast horizon used throughout backtesting and the API.
HORIZON = 28

#: Number of rolling-origin backtest folds.
N_FOLDS = 5

#: Days reserved at the end of the panel for rolling-origin backtesting.
BACKTEST_DAYS = N_FOLDS * HORIZON

#: Trailing window used to measure intermittency when stratifying the item
#: sample. The window ends *before* BACKTEST_DAYS begins: choosing which SKUs to
#: model using statistics computed over the evaluation period would be selection
#: leakage, even though scoping happens once and offline.
STRATIFY_WINDOW_DAYS = 90
