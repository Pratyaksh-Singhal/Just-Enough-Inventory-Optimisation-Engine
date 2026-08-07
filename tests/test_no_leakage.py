"""E2-S6 — the leakage gate. No story in E3 or E4 starts until this file passes.

Leakage is the highest-risk failure mode in this project because of how it fails: it does
not crash, it does not look wrong, and it produces backtest numbers that simply evaporate
in production.

So the central test here does not inspect any individual feature's definition. It
**perturbs the future and checks the past did not move**:

1. build the feature panel
2. corrupt actuals at and after some date ``C``
3. rebuild
4. every feature on a row targeting ``t < C + horizon`` must be bit-identical

That argument holds for any feature, including ones added later by someone who never read
this file. A per-feature assertion would only test the features someone remembered to
write an assertion for.

Two supporting tests keep it honest:

* a **sensitivity** check, so a builder that emitted constants could not pass by doing
  nothing, and
* an **exact boundary** probe on a single corrupted cell, which pins the window frame to
  ``horizon PRECEDING`` rather than merely "somewhere in the past".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_engine.features.build import (
    KNOWN_IN_ADVANCE,
    UNITS_DERIVED,
    build_features,
    feature_columns,
)
from synthetic import features, random_panel, ts, warehouse

HORIZON = 28


def _features(fact: pd.DataFrame, cal: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return features(fact, cal, horizon)


@pytest.fixture(scope="module")
def panel():
    return random_panel()


@pytest.fixture(scope="module")
def baseline(panel):
    cal, fact = panel
    return _features(fact, cal)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_corrupting_the_future_does_not_move_the_past(panel, baseline):
    """No feature targeting t may reflect any observation at or after t - horizon."""
    cal, fact = panel
    cutoff = ts(300)

    corrupted = fact.copy()
    # 9999 rather than a scaling factor: this flips zeros to non-zeros, so the
    # intermittency features are perturbed too, not just the magnitude-based ones.
    corrupted.loc[corrupted["date"] >= cutoff, "units"] = np.int32(9999)
    after = _features(corrupted, cal)

    # A feature on a row targeting t reads no later than t - horizon, so anything
    # targeting before cutoff + horizon must be untouched.
    boundary = ts(300 + HORIZON)
    cols = list(feature_columns())
    before_a = baseline.loc[baseline["date"] < boundary, cols].reset_index(drop=True)
    before_b = after.loc[after["date"] < boundary, cols].reset_index(drop=True)

    assert not before_a.empty, "fixture produced no rows before the boundary"
    changed = [c for c in cols if not before_a[c].equals(before_b[c])]
    assert not changed, (
        f"features leaked future data: {changed}. A row targeting date t must not reflect"
        f" any actual at or after t - {HORIZON}."
    )


def test_the_gate_can_actually_fail(panel, baseline):
    """Guard against a vacuous gate: past the boundary, features must move.

    The cutoff is deliberately earlier than the one in the gate above. ``units_lag_365``
    on a 28-day horizon reads 393 days behind its target, so with a late cutoff it could
    never reach the corrupted region inside this panel and would appear "immune" for
    reasons that have nothing to do with the code. The gate wants a late cutoff to leave a
    large clean region to verify; this test wants an early one so the longest lag can
    respond.
    """
    cal, fact = panel
    cutoff = ts(100)
    corrupted = fact.copy()
    corrupted.loc[corrupted["date"] >= cutoff, "units"] = np.int32(9999)
    after = _features(corrupted, cal)

    boundary = ts(100 + HORIZON)
    cols = list(feature_columns())
    a = baseline.loc[baseline["date"] >= boundary, cols].reset_index(drop=True)
    b = after.loc[after["date"] >= boundary, cols].reset_index(drop=True)

    moved = {c for c in cols if not a[c].equals(b[c])}
    assert moved >= UNITS_DERIVED, (
        "every units-derived feature should respond to corrupted actuals past the"
        f" boundary; these did not: {sorted(UNITS_DERIVED - moved)}. If a feature never"
        " moves, the gate above proves nothing about it."
    )


def test_window_frame_ends_exactly_at_horizon(panel, baseline):
    """Pin the frame to `horizon PRECEDING`, not merely 'somewhere in the past'.

    Corrupt one cell at date D. A rolling feature targeting D + horizon reads exactly up
    to D and must change; the same feature targeting D + horizon - 1 reads only up to
    D - 1 and must not. An off-by-one in the frame breaks one of these two assertions.
    """
    cal, fact = panel
    d = ts(250)
    item, store = "FOODS_1_000", "CA_1"

    corrupted = fact.copy()
    mask = (
        (corrupted["date"] == d) & (corrupted["item_id"] == item) & (corrupted["store_id"] == store)
    )
    assert mask.sum() == 1
    corrupted.loc[mask, "units"] = np.int32(5000)
    after = _features(corrupted, cal)

    def row(df, target):
        sel = df[(df["item_id"] == item) & (df["store_id"] == store) & (df["date"] == target)]
        assert len(sel) == 1, f"expected exactly one row targeting {target}"
        return sel.iloc[0]

    on_edge = ts(250 + HORIZON)
    just_inside = ts(250 + HORIZON - 1)

    assert (
        row(baseline, on_edge)["units_roll_mean_7"] != row(after, on_edge)["units_roll_mean_7"]
    ), "the newest observation a feature may read is t - horizon; it is not being read"
    assert (
        row(baseline, just_inside)["units_roll_mean_7"]
        == row(after, just_inside)["units_roll_mean_7"]
    ), "the frame reaches one day too far forward -- off-by-one in ROWS BETWEEN"

    # Lags are origin-relative: units_lag_7 targeting t reads t - horizon - 7.
    lag_edge = ts(250 + HORIZON + 7)
    assert row(baseline, lag_edge)["units_lag_7"] != row(after, lag_edge)["units_lag_7"]
    assert (
        row(baseline, ts(250 + HORIZON + 6))["units_lag_7"]
        == row(after, ts(250 + HORIZON + 6))["units_lag_7"]
    )


# ---------------------------------------------------------------------------
# The known-in-advance exception must stay small and deliberate
# ---------------------------------------------------------------------------


def test_known_in_advance_allowlist_is_pinned():
    """Widening the unshifted set must be a deliberate, reviewed edit.

    The perturbation gate corrupts `units`, so it cannot catch a leak introduced through a
    column sourced from somewhere else. This hard-coded expectation is what stops the
    allowlist growing quietly: adding a column here forces a reviewer to justify why a
    retailer genuinely knows it before the target date.
    """
    assert {
        # Calendar: published years ahead.
        "wday",
        "month",
        "week_of_year",
        "is_weekend",
        "event_type",
        "days_since_event",
        "days_to_event",
        # SNAP benefit schedule: set by statute, known in advance.
        "snap",
        # Prices: the retailer sets its own future shelf prices.
        "price",
        "price_rel_28",
        "price_changed",
        "is_listed",
    } == KNOWN_IN_ADVANCE


def test_every_column_is_classified(baseline):
    """No feature may exist outside the units-derived / known-in-advance split."""
    keys = {"date", "item_id", "store_id", "dept_id", "cat_id", "state_id", "horizon", "units"}
    produced = set(baseline.columns) - keys
    declared = UNITS_DERIVED | KNOWN_IN_ADVANCE
    assert produced == declared, (
        f"unclassified columns: {sorted(produced - declared)};"
        f" declared but missing: {sorted(declared - produced)}"
    )
    assert not (UNITS_DERIVED & KNOWN_IN_ADVANCE), "a feature cannot be in both sets"


def test_no_feature_is_a_copy_of_the_target(baseline):
    """A feature perfectly tracking the target is leakage that survived everything above."""
    numeric = baseline.select_dtypes(include=[np.number])
    target = baseline["units"].astype(float)
    suspects = []
    for col in feature_columns():
        if col not in numeric.columns:
            continue
        series = numeric[col].astype(float)
        pair = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(pair) < 30 or pair["x"].nunique() < 2:
            continue
        if abs(pair["x"].corr(pair["y"])) > 0.99:
            suspects.append(col)
    assert not suspects, f"suspiciously perfect correlation with the target: {suspects}"


# ---------------------------------------------------------------------------
# Preconditions the leakage guarantee rests on
# ---------------------------------------------------------------------------


def test_horizon_zero_is_rejected(panel):
    cal, fact = panel
    con = warehouse(fact, cal)
    try:
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            build_features(con, 0)
    finally:
        con.close()


def test_non_rectangular_panel_is_rejected(panel):
    """Row-based frames only equal day-based frames on a dense panel; check, don't assume."""
    cal, fact = panel
    gapped = fact.drop(fact.index[100]).reset_index(drop=True)
    con = warehouse(gapped, cal)
    try:
        with pytest.raises(ValueError, match="not rectangular"):
            build_features(con, HORIZON)
    finally:
        con.close()


def test_shift_scales_with_horizon(panel):
    """A larger horizon must push every units feature further back, not keep it fixed."""
    cal, fact = panel
    short = _features(fact, cal, horizon=7)
    long = _features(fact, cal, horizon=28)
    key = ["item_id", "store_id", "date"]
    merged = short.merge(long, on=key, suffixes=("_h7", "_h28"))
    tail = merged[merged["date"] > ts(400)]
    assert not tail.empty
    assert not tail["units_lag_7_h7"].equals(tail["units_lag_7_h28"]), (
        "features are identical at horizon 7 and 28, so the horizon is not being applied"
    )


def test_grain_is_unique(panel, baseline):
    assert not baseline.duplicated(subset=["date", "item_id", "store_id"]).any()


def test_pre_listing_rows_are_dropped(baseline):
    """An unlisted item is an absent product, not zero demand.

    FOODS_1_001 has no price for its first 30 days, so those rows must not survive into
    training data -- they record the absence of a product, not an absence of demand.
    """
    unlisted = baseline[baseline["item_id"] == "FOODS_1_001"]
    assert not unlisted.empty, "fixture: the series itself should still be present"
    assert unlisted["date"].min() == ts(30)
    # Every other series was listed from day one and must be untouched.
    listed = baseline[baseline["item_id"] == "FOODS_1_000"]
    assert listed["date"].min() == ts(0)
