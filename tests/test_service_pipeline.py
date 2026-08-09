"""The model, the baseline, the choice between them, and the order that falls out.

Most of these avoid fitting LightGBM: the arithmetic that decides an order quantity is
separable from the boosting, and testing it directly keeps the suite fast. The two tests
that do fit are marked so they can be deselected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_engine.optimize.costs import FALLBACK_PRICE, CostModel
from inventory_engine.service.pipeline import (
    BASE_QUANTILES,
    BASELINE_NAME,
    MODEL_NAME,
    MethodScore,
    _monotonize_rows,
    baseline_daily,
    baseline_total,
    choose_method,
    forecast_sku,
    levels_for,
    order_from_distribution,
    resolve_price,
)


def daily(days=200, start="2025-01-01", base=8.0, seed=1) -> pd.Series:
    """A reproducible daily series with a weekday cycle."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D")
    weekday = base + 0.5 * base * (index.dayofweek >= 5)
    return pd.Series(np.maximum(0, weekday + rng.normal(0, 1.0, days)).round(), index=index)


# --------------------------------------------------------------------------- the grid


def test_the_critical_ratio_is_always_inside_the_fitted_grid():
    """Removes the whole extrapolation failure mode, for any cost inputs a caller can send."""
    for margin, spoil in [(0.30, 0.60), (0.90, 0.0), (0.01, 1.0), (0.5, 0.5)]:
        cr = CostModel(margin_rate=margin, spoilage_rate=spoil).critical_ratio()
        levels = levels_for(cr)
        assert min(levels) <= round(cr, 4) <= max(levels)
        assert round(cr, 4) in levels
        assert list(levels) == sorted(levels)


def test_the_grid_still_contains_the_base_quantiles():
    levels = levels_for(0.4087)
    assert set(BASE_QUANTILES) <= set(levels)


def test_a_critical_ratio_already_on_the_grid_does_not_duplicate_it():
    levels = levels_for(0.5)
    assert len(levels) == len(set(levels)) == len(BASE_QUANTILES)


# --------------------------------------------------------------------------- monotonicity


def test_crossed_quantiles_are_rearranged_and_clipped():
    raw = np.array([[5.0, 3.0, 9.0, -2.0]])
    fixed = _monotonize_rows(raw)
    assert list(fixed[0]) == [0.0, 3.0, 5.0, 9.0]


def test_rearrangement_preserves_row_count():
    raw = np.random.default_rng(0).normal(size=(17, 8))
    assert _monotonize_rows(raw).shape == (17, 8)


# --------------------------------------------------------------------------- baselines


def test_the_total_baseline_is_the_empirical_quantile_of_window_totals():
    """This is tier 1's rule verbatim; the tiers must not drift apart on it."""
    series = daily(days=120)
    levels = (0.1, 0.5, 0.9)
    got = baseline_total(series, 28, levels)[0]

    values = series.to_numpy(float)
    totals = [values[i : i + 28].sum() for i in range(len(values) - 27)]
    assert got == pytest.approx(np.quantile(totals, levels))


def test_the_total_baseline_is_not_moved_far_by_one_anomalous_final_week():
    """Regression: the seasonal-naive version projected a spike week x4 into the order.

    On real M5 data that produced an order of 974 units at a 41% service level against a
    399-unit average -- two and a half times average demand while claiming to under-stock.
    """
    series = daily(days=200, base=8.0)
    spiked = series.copy()
    spiked.iloc[-7:] *= 5

    calm = baseline_total(series, 28, (0.4087,))[0, 0]
    surged = baseline_total(spiked, 28, (0.4087,))[0, 0]
    assert surged < calm * 1.5


def test_the_daily_baseline_is_ordered_and_non_negative():
    band = baseline_daily(daily(days=120), 28, (0.1, 0.5, 0.9))
    assert band.shape == (28, 3)
    assert (np.diff(band, axis=1) >= -1e-9).all()
    assert (band >= 0).all()


def test_the_daily_baseline_median_repeats_the_last_week():
    """Seasonal naive is 'same weekday, last week', and must stay literally that."""
    series = daily(days=120)
    band = baseline_daily(series, 14, (0.5,))
    last_week = series.to_numpy(float)[-7:]
    # The median offset is the residual median, not necessarily zero, so compare the shape.
    assert np.corrcoef(band[:7, 0], last_week)[0, 1] > 0.99


def test_a_series_shorter_than_the_horizon_still_returns_a_distribution():
    short = daily(days=10)
    out = baseline_total(short, 28, (0.1, 0.5, 0.9))
    assert out.shape == (1, 3)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- selection


def score(name, pinball, mase=None, folds=3):
    """A MethodScore with the given per-fold numbers."""
    pin = tuple([pinball] * folds)
    ms = tuple([mase if mase is not None else pinball] * folds)
    return MethodScore(name, ms, pin)


def test_the_baseline_is_served_when_it_wins_and_the_response_says_so():
    method, reason = choose_method(
        score(MODEL_NAME, 90.0), score(BASELINE_NAME, 50.0), n_folds=3
    )
    assert method == BASELINE_NAME
    assert "baseline beat the model" in reason
    assert "50.000" in reason and "90.000" in reason


def test_the_model_is_served_when_it_wins():
    method, reason = choose_method(
        score(MODEL_NAME, 40.0), score(BASELINE_NAME, 70.0), n_folds=5
    )
    assert method == MODEL_NAME
    assert "model beat the simple baseline" in reason


def test_a_tie_goes_to_the_baseline():
    """The simpler method wins ties: equal evidence does not justify the extra machinery."""
    method, _ = choose_method(score(MODEL_NAME, 50.0), score(BASELINE_NAME, 50.0), n_folds=3)
    assert method == BASELINE_NAME


def test_a_narrow_gap_is_reported_as_too_close_to_call_not_as_a_win():
    """Regression: a 2% single-fold gap was announced as "the baseline beat the model".

    Real numbers from an end-to-end run on the tier 1 demo CSV: pinball 34.515 vs 35.194,
    one fold, while MASE said the model was clearly better. Calling that a victory
    overstates the evidence in whichever direction the noise happened to fall.
    """
    method, reason = choose_method(
        score(MODEL_NAME, 35.194), score(BASELINE_NAME, 34.515), n_folds=1
    )
    assert method == BASELINE_NAME
    assert "Too close to call" in reason
    assert "beat the model" not in reason


def test_a_narrow_gap_the_other_way_is_also_a_draw():
    """Direction must not matter, or the rule is just a thumb on the scale."""
    method, reason = choose_method(
        score(MODEL_NAME, 34.515), score(BASELINE_NAME, 35.194), n_folds=3
    )
    assert method == BASELINE_NAME
    assert "Too close to call" in reason


def test_a_clear_margin_is_still_called_a_win():
    _, reason = choose_method(score(MODEL_NAME, 40.0), score(BASELINE_NAME, 70.0), n_folds=5)
    assert "Too close to call" not in reason
    assert "model beat the simple baseline" in reason


def test_disagreement_between_the_two_metrics_is_stated_not_hidden():
    model = MethodScore(MODEL_NAME, (2.0, 2.0, 2.0), (40.0, 40.0, 40.0))
    baseline = MethodScore(BASELINE_NAME, (1.0, 1.0, 1.0), (70.0, 70.0, 70.0))
    method, reason = choose_method(model, baseline, n_folds=3)
    assert method == MODEL_NAME
    assert "the two measures disagree" in reason
    assert "better day-to-day" in reason


def test_an_unfittable_model_falls_back_and_says_which_method_ran():
    model = MethodScore(MODEL_NAME, (float("nan"),) * 3, (float("nan"),) * 3)
    method, reason = choose_method(model, score(BASELINE_NAME, 50.0), n_folds=3)
    assert method == BASELINE_NAME
    assert "could not be fitted" in reason


def test_the_reason_quotes_folds_compared_not_the_fold_budget():
    """Regression: 'mean over 3 folds' was printed when only 1 fold scored the model."""
    model = MethodScore(MODEL_NAME, (float("nan"), float("nan"), 1.2), (np.nan, np.nan, 40.0))
    baseline = score(BASELINE_NAME, 70.0, folds=3)
    _, reason = choose_method(model, baseline, n_folds=3)
    assert "over 1 backtest fold(s)" in reason
    assert "only be fitted in 1 of 3 folds" in reason


def test_n_scored_counts_only_finite_folds():
    assert MethodScore("m", (), (1.0, float("nan"), 2.0)).n_scored == 2


# --------------------------------------------------------------------------- the order


def test_a_low_critical_ratio_orders_below_the_median():
    """The headline finding of tier 1: with spoilage, the optimal stock level is low.

    The tolerance is not slack. ``levels_for`` puts the critical ratio on the grid rounded
    to four decimals, while the interpolation uses the exact ratio, so the answer lands a
    hair off the grid point -- 95.001 rather than 95.0 here. Rounding the ratio used for
    the economics instead would be worse: it would move the number the buyer acts on to
    make a test tidy.
    """
    levels = (0.1, 0.4087, 0.5, 0.9)
    totals = np.array([[80.0, 95.0, 100.0, 130.0]])
    costs = CostModel()  # CR ~ 0.40871
    qty, _ = order_from_distribution(levels, totals, costs, price=3.0)
    assert qty == pytest.approx(95.0, abs=0.01)
    assert qty < 100.0


def test_a_high_critical_ratio_orders_above_the_median():
    cr = CostModel(margin_rate=0.8, spoilage_rate=0.05).critical_ratio()
    levels = levels_for(cr)
    totals = np.asarray([np.linspace(50, 200, len(levels))])
    qty, _ = order_from_distribution(levels, totals, CostModel(margin_rate=0.8, spoilage_rate=0.05), 3.0)
    assert qty > np.interp(0.5, levels, totals[0])


def test_the_order_is_never_negative():
    levels = (0.1, 0.4087, 0.5, 0.9)
    totals = np.array([[-5.0, -1.0, 0.0, 2.0]])
    qty, _ = order_from_distribution(levels, totals, CostModel(), price=3.0)
    assert qty >= 0.0


def test_expected_cost_is_reported_in_money():
    levels = (0.1, 0.4087, 0.5, 0.9)
    totals = np.array([[80.0, 95.0, 100.0, 130.0]])
    _, cost = order_from_distribution(levels, totals, CostModel(), price=4.0)
    assert cost > 0


# --------------------------------------------------------------------------- pricing


def test_the_latest_quoted_price_wins_not_the_mean():
    frame = pd.DataFrame({"unit_price": [1.0, 1.0, 9.0]})
    price, fallback = resolve_price(frame)
    assert price == 9.0 and fallback is False


def test_a_missing_price_column_falls_back_to_the_shared_constant():
    price, fallback = resolve_price(pd.DataFrame({"units_sold": [1, 2]}))
    assert price == FALLBACK_PRICE and fallback is True


def test_an_all_null_price_column_falls_back_and_is_flagged():
    price, fallback = resolve_price(pd.DataFrame({"unit_price": [None, np.nan]}))
    assert price == FALLBACK_PRICE and fallback is True


def test_zero_prices_are_ignored_as_unusable():
    price, fallback = resolve_price(pd.DataFrame({"unit_price": [0.0, 0.0]}))
    assert price == FALLBACK_PRICE and fallback is True


# --------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_forecast_sku_returns_a_complete_chartable_result():
    series = daily(days=400)
    frame = pd.DataFrame(
        {"date": series.index, "units_sold": series.to_numpy(), "unit_price": 2.5}
    )
    result = forecast_sku("WIDGET", frame, horizon=28, costs=CostModel())

    assert result.method_used in (MODEL_NAME, BASELINE_NAME)
    assert result.method_reason
    assert result.order_qty > 0
    assert result.unit_price == 2.5 and result.price_is_fallback is False

    chart = result.series
    assert len(chart["forecast"]) == 28
    assert chart["history"]
    assert chart["band"]["label"].startswith("Where sales landed")
    # The band must bracket the point everywhere, or the chart draws a line outside its own
    # shaded region.
    for point in chart["forecast"]:
        assert point["lo"] <= point["point"] <= point["hi"]
    assert chart["order"]["daily_rate"] == pytest.approx(result.order_qty / 28, rel=1e-3)


@pytest.mark.slow
def test_a_flat_series_orders_close_to_its_own_level():
    """Sanity bound: constant demand of 10/day over 28 days is about 280 units."""
    index = pd.date_range("2025-01-01", periods=300, freq="D")
    frame = pd.DataFrame({"date": index, "units_sold": 10.0, "unit_price": 1.0})
    result = forecast_sku("FLAT", frame, horizon=28, costs=CostModel())
    assert 200 < result.order_qty < 320
