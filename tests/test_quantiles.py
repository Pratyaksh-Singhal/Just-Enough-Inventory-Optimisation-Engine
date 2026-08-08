"""Monotone quantile reads — the fix for LightGBM's independently-fitted quantile crossings."""

from __future__ import annotations

import pandas as pd
import pytest

from inventory_engine.models.quantiles import GRAIN, crossing_rate, monotonize


def _frame(rows: dict[float, list[float]]) -> pd.DataFrame:
    out = []
    for q, values in rows.items():
        for i, v in enumerate(values):
            out.append(
                {
                    "fold": 0,
                    "item_id": f"I{i}",
                    "store_id": "CA_1",
                    "target_date": pd.Timestamp("2016-01-04"),
                    "quantile": q,
                    "yhat": v,
                }
            )
    return pd.DataFrame(out)


CROSSED = _frame({0.5: [1.0, 1.0], 0.9: [9.0, 2.0], 0.95: [3.0, 3.0], 0.99: [4.0, 4.0]})
CLEAN = _frame({0.5: [1.0, 2.0], 0.9: [2.0, 3.0], 0.95: [3.0, 4.0], 0.99: [4.0, 5.0]})


def test_crossings_are_detected_before_the_fix():
    assert crossing_rate(CROSSED) == (1, 2)


def test_monotonize_eliminates_crossings():
    assert crossing_rate(monotonize(CROSSED)) == (0, 2)


def test_monotonize_leaves_clean_rows_untouched():
    fixed = monotonize(CLEAN).sort_values([*GRAIN, "quantile"]).reset_index(drop=True)
    original = CLEAN.sort_values([*GRAIN, "quantile"]).reset_index(drop=True)
    pd.testing.assert_series_equal(fixed["yhat"], original["yhat"], check_names=False)


def test_rearrangement_preserves_the_multiset_of_values():
    """Rearrangement reassigns values across levels; it must not invent or drop any.

    This is what makes the Chernozhukov non-worsening result apply — the forecast values
    themselves are unchanged, only which quantile level each is attached to.
    """
    before = sorted(CROSSED[CROSSED["item_id"] == "I0"]["yhat"].tolist())
    after = sorted(monotonize(CROSSED).query("item_id == 'I0'")["yhat"].tolist())
    assert before == after


def test_row_count_and_grain_are_preserved():
    fixed = monotonize(CROSSED)
    assert len(fixed) == len(CROSSED)
    assert set(fixed.columns) >= {*GRAIN, "quantile", "yhat"}


def test_crossing_rate_can_be_restricted_to_the_newsvendor_band():
    """E7 selects inside CR 0.5-0.95; a crossing at 0.99 cannot affect its decision."""
    only_high = _frame({0.5: [1.0], 0.9: [2.0], 0.95: [3.0], 0.99: [0.0]})
    assert crossing_rate(only_high) == (1, 1)
    assert crossing_rate(only_high, within=(0.5, 0.9, 0.95)) == (0, 1)


def test_single_level_reports_nothing_checked():
    """One level has no ordering to violate, so no rows were meaningfully audited.

    Returning ``(0, 2)`` here would read as "2 rows verified clean" and overstate the
    audit; ``(0, 0)`` says plainly that the check did not apply.
    """
    assert crossing_rate(_frame({0.5: [1.0, 2.0]})) == (0, 0)


def test_empty_input_is_handled():
    empty = CLEAN.iloc[0:0]
    assert crossing_rate(empty) == (0, 0)
    assert monotonize(empty).empty


@pytest.mark.parametrize("bad", [[3.0, 2.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0]])
def test_arbitrary_disorder_is_fully_sorted(bad):
    frame = _frame(dict(zip([0.5, 0.9, 0.95, 0.99], [[v] for v in bad], strict=True)))
    fixed = monotonize(frame).sort_values("quantile")
    assert fixed["yhat"].tolist() == sorted(bad)
