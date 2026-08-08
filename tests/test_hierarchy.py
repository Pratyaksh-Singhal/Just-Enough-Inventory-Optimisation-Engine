"""E6 — hierarchy structure and coherence.

The summing matrix is the object every reconciled number depends on. If it is wrong, MinT
still returns coherent-looking output — coherent with respect to the wrong tree — so these
assert the structure rather than trusting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_engine.hierarchy.mint import (
    HIERARCHY_SPEC,
    LEVEL_KEY_BY_NAME,
    LEVEL_NAMES,
    MINT_METHOD,
    _fill_unlisted_series,
    _level_of,
    coherence_gap,
)


class _Fold:
    """Minimal stand-in for a Fold, carrying only what the fill helper reads."""

    def __init__(self, dates):
        self._dates = dates

    def target_dates(self):
        return self._dates


# ---------------------------------------------------------------------------
# Hierarchy structure
# ---------------------------------------------------------------------------


def test_spec_is_strictly_nested():
    """Each level must extend the one above it, or the tree is not a tree."""
    for shallower, deeper in zip(HIERARCHY_SPEC, HIERARCHY_SPEC[1:], strict=False):
        assert deeper[: len(shallower)] == shallower
        assert len(deeper) == len(shallower) + 1


def test_category_level_is_deliberately_absent():
    """Scope fixes cat_id = FOODS, making a Category level identical to Store.

    Including it would put duplicate rows in the summing matrix and leave the covariance
    rank-deficient. The deviation from the brief is intentional and documented.
    """
    flattened = [col for level in HIERARCHY_SPEC for col in level]
    assert "cat_id" not in flattened
    assert "state_id" in flattened and "store_id" in flattened


def test_level_names_align_with_spec():
    assert len(LEVEL_NAMES) == len(HIERARCHY_SPEC)
    assert LEVEL_NAMES[-1] == "item_store", "the bottom level is the forecasting grain"
    assert set(LEVEL_KEY_BY_NAME) == set(LEVEL_NAMES)


@pytest.mark.parametrize(
    ("unique_id", "expected"),
    [
        ("CA", "state"),
        ("CA/CA_1", "store"),
        ("CA/CA_1/FOODS_1", "store_dept"),
        ("CA/CA_1/FOODS_1/FOODS_1_001", "item_store"),
    ],
)
def test_level_inferred_from_path_depth(unique_id, expected):
    assert _level_of(unique_id) == expected


def test_mint_method_needs_no_insample_residuals():
    """`mint_shrink` requires fitted values E4 does not persist; `wls_struct` needs only S."""
    assert MINT_METHOD == "wls_struct"


# ---------------------------------------------------------------------------
# Coherence measurement
# ---------------------------------------------------------------------------


def _tree_frame(state, stores, value_col="yhat"):
    rows = [{"unique_id": "CA", "ds": pd.Timestamp("2016-01-04"), value_col: state}]
    for name, v in stores.items():
        rows.append({"unique_id": f"CA/{name}", "ds": pd.Timestamp("2016-01-04"), value_col: v})
    return pd.DataFrame(rows)


TAGS = {
    "state_id": np.array(["CA"]),
    "state_id/store_id": np.array(["CA/CA_1", "CA/CA_2"]),
    "state_id/store_id/dept_id": np.array([]),
    "state_id/store_id/dept_id/item_id": np.array([]),
}


def test_coherent_tree_reports_zero_gap():
    frame = _tree_frame(10.0, {"CA_1": 6.0, "CA_2": 4.0})
    gaps = coherence_gap(frame, TAGS, "yhat")
    assert gaps.loc[0, "max_abs_gap"] == pytest.approx(0.0)


def test_incoherent_tree_reports_the_gap():
    """Independently forecast levels do not add up; that gap is what MinT removes."""
    frame = _tree_frame(10.0, {"CA_1": 6.0, "CA_2": 7.0})
    gaps = coherence_gap(frame, TAGS, "yhat")
    assert gaps.loc[0, "max_abs_gap"] == pytest.approx(3.0)
    assert gaps.loc[0, "n_checked"] == 1


# ---------------------------------------------------------------------------
# The unlisted-series fill
# ---------------------------------------------------------------------------


def test_unlisted_series_are_filled_with_zero():
    """MinT needs every column of S present; a not-yet-listed SKU has no forecast rows.

    Zero is correct rather than a placeholder: no shelf presence means no demand, which is
    what the raw actuals show for those dates.
    """
    dates = [pd.Timestamp("2016-01-04"), pd.Timestamp("2016-01-05")]
    tags = {LEVEL_KEY_BY_NAME["item_store"]: np.array(["CA/CA_1/F/A", "CA/CA_1/F/B"])}
    partial = pd.DataFrame({"unique_id": ["CA/CA_1/F/A"] * 2, "ds": dates, "lgbm": [3.0, 4.0]})

    filled = _fill_unlisted_series(partial, tags, _Fold(dates))

    assert len(filled) == 4, "every series x date must be present"
    missing = filled[filled["unique_id"] == "CA/CA_1/F/B"]
    assert (missing["lgbm"] == 0.0).all()
    present = filled[filled["unique_id"] == "CA/CA_1/F/A"].sort_values("ds")
    assert present["lgbm"].tolist() == [3.0, 4.0], "existing forecasts must be untouched"


def test_fill_is_a_noop_when_nothing_is_missing():
    dates = [pd.Timestamp("2016-01-04")]
    tags = {LEVEL_KEY_BY_NAME["item_store"]: np.array(["CA/CA_1/F/A"])}
    complete = pd.DataFrame({"unique_id": ["CA/CA_1/F/A"], "ds": dates, "lgbm": [5.0]})
    filled = _fill_unlisted_series(complete, tags, _Fold(dates))
    assert len(filled) == 1
    assert filled["lgbm"].tolist() == [5.0]
