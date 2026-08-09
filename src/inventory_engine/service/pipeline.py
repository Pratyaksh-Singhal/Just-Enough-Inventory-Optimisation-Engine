"""Worker-side: fit, backtest, choose honestly, and turn the distribution into an order.

This module imports LightGBM. Nothing that serves an HTTP request may import *this* --
``tests/test_service_layering`` enforces it.

Two grains, because two different questions
-------------------------------------------
**Daily** forecasts are what the chart draws and what MASE is quoted on, so the accuracy
figure is comparable to tier 1's and to anything else the user has seen.

**The horizon total** is what the order decision is made at. The newsvendor problem is
single-period: the question is "how many units cover the next H days", and the quantile of
a sum is not the sum of quantiles, so summing 28 daily q0.42 forecasts would understate the
spread badly. Tier 1's calculator frames it as an H-day total for the same reason.

Choosing between the model and the baseline
-------------------------------------------
Selection is on **pinball loss at the critical ratio, at the total grain**, because that
number is literally the loss function of the quantity being ordered. A model can win on
MASE -- a point-accuracy metric -- while producing a worse q0.42, and it is q0.42 that
becomes the purchase order.

MASE is reported alongside, at the daily grain, because it is the interpretable one: 1.0
means "as good as naive". When the two metrics disagree the response says so rather than
quoting whichever flatters the model.

If the baseline wins, the baseline is what gets served, and ``method_reason`` says by how
much. A service that silently returns the loser of its own comparison is worse than one
with no comparison at all, because it looks validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from inventory_engine.backtest.folds import make_folds
from inventory_engine.backtest.metrics import mase, naive_scale, pinball
from inventory_engine.models.baselines import SEASON_LENGTH, seasonal_naive
from inventory_engine.optimize.costs import FALLBACK_PRICE, CostModel
from inventory_engine.optimize.newsvendor import (
    expected_cost_from_distribution,
    interpolate_quantile,
)
from inventory_engine.service.features import (
    FEATURE_COLUMNS,
    WARMUP_DAYS,
    supervised_frame,
    to_daily,
    total_frame,
)
from inventory_engine.service.folds import fold_count_for, spread_caveat

#: Quantile grid. Matches ``models.gbm.QUANTILES`` -- extended down to 0.10 because
#: fresh-food critical ratios land near 0.42, below the 0.5 a default grid would start at.
#: The critical ratio itself is added at fit time (see :func:`levels_for`), so the order
#: quantity is always interpolated inside the fitted grid and never extrapolated.
BASE_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

#: The interval the chart shades, labelled in plain language rather than as "q10-q90".
BAND_LOW: Final = 0.1
BAND_HIGH: Final = 0.9

#: Small-data parameters, and deliberately nothing like tier 1's. The global M5 model
#: trains on ~1M rows and can afford 63 leaves; a per-SKU model here sees 1,700 rows at the
#: gate's minimum. 63 leaves on 1,700 rows memorises the training window and reports a
#: beautiful in-sample fit that the rolling-origin backtest then demolishes. Untuned, for
#: the same reason tier 1 leaves its GBM untuned: tuning against the folds used to report
#: the result would make the baseline comparison meaningless.
SMALL_DATA_PARAMS: Final[dict] = {
    "objective": "quantile",
    "learning_rate": 0.06,
    "num_leaves": 7,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "num_boost_round": 200,
    "verbose": -1,
    "seed": 20260809,
    "num_threads": 2,
}

#: Below this many usable daily rows the model is not fitted and the baseline is served
#: outright. Fitting eight quantile models on fewer rows than that produces a
#: confident-looking curve with nothing behind it.
MIN_TRAIN_ROWS: Final = 60

#: Same idea at the total grain, which is far scarcer: one row per origin rather than one
#: per (origin, horizon) pair, so a training window yielding 200 daily rows yields ~7 total
#: rows. This is the binding constraint on short uploads, and it is why a 90-day file with a
#: 28-day horizon is served by the baseline: fold 0 trains on ~34 days, leaving 6 origins.
#: That is stated in the response rather than hidden -- and a shorter horizon (a weekly
#: order cycle) clears it comfortably on the same file.
MIN_TOTAL_ROWS: Final = 24

#: Ceiling on daily training rows per fit, enforced by striding origins in
#: :func:`_training_origins`. 8,000 rows fits in about a second per quantile level, which
#: keeps a 50-SKU upload inside the job timeout with room to spare.
MAX_TRAIN_ROWS: Final = 8_000

MODEL_NAME: Final = "quantile_gbm"
BASELINE_NAME: Final = "seasonal_naive"

#: Relative margin in pinball loss below which the two methods are called a draw rather
#: than a win for either.
#:
#: Without this the service reported "the baseline beat the model" off a 2% gap measured on
#: a single fold -- 34.515 against 35.194 on a real 120-day upload -- while MASE said the
#: model was clearly the better forecast (0.830 against 1.082). Both statements were
#: arithmetically true and the conclusion was not supported by either. The simpler method
#: still wins a draw, since equal evidence does not justify the extra machinery; what
#: changes is that the response calls it a draw instead of a victory.
DECISIVE_MARGIN: Final = 0.05


def levels_for(critical_ratio: float) -> tuple[float, ...]:
    """Return the quantile grid to fit, guaranteed to bracket ``critical_ratio``.

    Adding the critical ratio to the grid removes an entire failure mode. Without it a
    caller choosing a 90% margin and no spoilage drives CR to 0.998, outside the fitted
    grid, and :func:`~inventory_engine.optimize.newsvendor.interpolate_quantile` correctly
    refuses to extrapolate a tail nobody estimated -- returning a 422 for a request that is
    perfectly reasonable. Fitting the level that was actually asked for answers it instead.
    """
    return tuple(sorted({*BASE_QUANTILES, round(float(critical_ratio), 4)}))


@dataclass(frozen=True)
class MethodScore:
    """One method's backtest performance on one SKU."""

    name: str
    mase_folds: tuple[float, ...] = ()
    pinball_folds: tuple[float, ...] = ()

    @property
    def mase_mean(self) -> float:
        """Mean MASE over scorable folds."""
        return _nanmean(self.mase_folds)

    @property
    def mase_spread(self) -> float:
        """Half the fold range -- the plus-or-minus quoted beside the mean."""
        finite = [v for v in self.mase_folds if np.isfinite(v)]
        return (max(finite) - min(finite)) / 2 if len(finite) > 1 else float("nan")

    @property
    def pinball_mean(self) -> float:
        """Mean pinball loss at the critical ratio, over scorable folds."""
        return _nanmean(self.pinball_folds)

    @property
    def n_scored(self) -> int:
        """Folds this method actually produced a score on.

        Distinct from the job's fold count, and the distinction matters. A 120-day upload
        gets three folds, but the model cannot be fitted in the two earliest of them --
        their training windows are too short -- so its "mean over 3 folds" is one number
        wearing a plural. The caveat shown to the user is driven by this, not by the
        nominal count.
        """
        return sum(1 for v in self.pinball_folds if np.isfinite(v))


@dataclass
class SkuForecast:
    """Everything the API needs to return for one SKU."""

    sku: str
    method_used: str
    method_reason: str
    n_folds: int
    model: MethodScore
    baseline: MethodScore
    critical_ratio: float
    order_qty: float
    expected_cost: float | None
    unit_price: float
    price_is_fallback: bool
    series: dict = field(default_factory=dict)


def _nanmean(values) -> float:
    """Mean of the finite entries, or NaN when there are none."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# --------------------------------------------------------------------------- the model


def _fit_one(train: pd.DataFrame, level: float):
    """Fit a single quantile model. Imported lazily so importing this module is cheap."""
    import lightgbm as lgb

    params = {**SMALL_DATA_PARAMS, "alpha": level}
    dataset = lgb.Dataset(train[list(FEATURE_COLUMNS)], label=train["y"], free_raw_data=False)
    return lgb.train(params, dataset, num_boost_round=params["num_boost_round"])


def fit_quantile_models(train: pd.DataFrame, levels: tuple[float, ...]) -> dict[float, object]:
    """Fit one model per quantile level.

    Independent fits, exactly as tier 1 does -- and with the same consequence: nothing
    constrains them to come out ordered. :func:`_monotonize_rows` rearranges at read time.
    """
    usable = train.dropna(subset=["y", *FEATURE_COLUMNS[:1]])
    return {level: _fit_one(usable, level) for level in levels}


def predict_quantiles(
    models: dict[float, object], features: pd.DataFrame, levels: tuple[float, ...]
) -> np.ndarray:
    """Predict every level for every row, rearranged into ascending order.

    Returns:
        Array of shape ``(n_rows, n_levels)``, non-negative and non-decreasing across the
        level axis.

    """
    raw = np.column_stack(
        [models[level].predict(features[list(FEATURE_COLUMNS)]) for level in levels]
    )
    return _monotonize_rows(raw)


def _monotonize_rows(values: np.ndarray) -> np.ndarray:
    """Sort each row ascending and clip at zero.

    The same rearrangement tier 1 applies in ``models.quantiles.monotonize``, done here
    with a bare sort because that function's grain is hardcoded to the M5 key
    ``(fold, item_id, store_id, target_date)`` and this data has no such columns. The
    justification is unchanged: Chernozhukov, Fernandez-Val and Galichon (2010) show
    rearranging a crossed quantile curve weakly reduces pinball loss at every level, so it
    cannot make the forecast worse by the metric it is judged on.
    """
    return np.clip(np.sort(np.asarray(values, dtype=float), axis=1), 0.0, None)


# --------------------------------------------------------------------------- the baseline


def baseline_daily(train: pd.Series, horizon: int, levels: tuple[float, ...]) -> np.ndarray:
    """Seasonal-naive point path plus an empirical residual distribution.

    Seasonal naive alone is a point forecast, and a point forecast cannot be compared on
    pinball loss or turned into an order quantity. The spread comes from the method's own
    in-sample errors: ``e_t = y_t - y_{t-7}``, whose quantiles say how wrong "same weekday
    last week" has actually been on this series.

    Returns:
        Array of shape ``(horizon, n_levels)``.

    """
    values = train.dropna().to_numpy(dtype=float)
    point = seasonal_naive(values, horizon, SEASON_LENGTH)
    residuals = values[SEASON_LENGTH:] - values[:-SEASON_LENGTH]
    if residuals.size == 0:
        residuals = np.zeros(1)
    offsets = np.quantile(residuals, levels)
    return _monotonize_rows(point[:, None] + offsets[None, :])


def baseline_total(train: pd.Series, horizon: int, levels: tuple[float, ...]) -> np.ndarray:
    """Empirical quantiles of the horizon-day totals this series has actually produced.

    This is **tier 1's rule, verbatim** -- the browser calculator's ``windowTotals`` followed
    by an empirical ``quantile``. Using it here makes the tier 2 comparison answer the
    question a user actually has: *does fitting a model beat what the quick calculator
    already gives me, on my own data?*

    Why not seasonal naive at this grain
    ------------------------------------
    The first version of this function did use seasonal naive: project the last 7 days four
    times over, then add the quantiles of its rolling-origin residuals. Running it on real
    M5 data showed why that is wrong for a *total*. ``FOODS_3_086-CA_3`` happens to end on a
    week selling 249 units against a 100-unit average week, so seasonal naive projected 996
    for the next 28 days on a series whose mean 28-day total is 399 -- and the order
    quantity came out at 974 units at a **41%** service level, which is to say two and a
    half times the average demand while claiming to be deliberately under-stocking.

    Seasonal naive was not malfunctioning; carrying the last week forward is its definition,
    and it is a reasonable daily forecast. But a single anomalous final week should not
    scale a month's purchase order, and the empirical distribution of realised totals cannot
    be moved that far by one week.

    Known limitation, stated: the windows overlap, so the totals are autocorrelated and the
    tails are a little tighter than independent sampling would give. Tier 1 has exactly the
    same property, which is part of why the two remain comparable.

    Returns:
        Array of shape ``(1, n_levels)``.

    """
    values = train.dropna().to_numpy(dtype=float)
    if values.size < horizon:
        # Too short for even one complete window: fall back to scaling the mean, which is
        # a weak forecast but an honest one, rather than inventing a distribution.
        mean = float(values.mean()) if values.size else 0.0
        return _monotonize_rows(np.full((1, len(levels)), mean * horizon))

    totals = np.array(
        [values[i : i + horizon].sum() for i in range(len(values) - horizon + 1)], dtype=float
    )
    return _monotonize_rows(np.asarray([np.quantile(totals, levels)]))


# --------------------------------------------------------------------------- backtest


def backtest_sku(
    series: pd.Series, horizon: int, critical_ratio: float
) -> tuple[MethodScore, MethodScore, int]:
    """Score the model and the baseline on rolling origins over the user's own history.

    Rolling origin, never a random split and never a single holdout -- the discipline tier
    1 states in ``backtest/folds.py``, applied to a series whose length the user chose
    rather than one this project curated. The fold count comes from
    :func:`~inventory_engine.service.folds.fold_count_for`, so a 90-day series gets the two
    folds it can support instead of five folds with fortnight-long training windows.

    Only the two levels that are actually scored get fitted per fold -- the median for MASE
    and the critical ratio for pinball -- rather than the whole grid. The remaining levels
    exist to draw the chart's band and to interpolate the order quantity, neither of which
    the backtest measures, so fitting them five times over is pure cost. On a two-year
    series this took one SKU from 23s to under 10s, which is the difference between a
    50-SKU upload finishing inside the job timeout and not.

    Returns:
        ``(model_score, baseline_score, n_folds)``. ``n_folds`` of zero means the series
        could not be backtested at this horizon at all.

    """
    n_folds = fold_count_for(len(series), horizon)
    if n_folds == 0:
        return MethodScore(MODEL_NAME), MethodScore(BASELINE_NAME), 0

    scored = tuple(sorted({0.5, round(float(critical_ratio), 4)}))
    folds = make_folds(series.index[-1].date(), n_folds=n_folds, horizon=horizon)
    model_mase, model_pin, base_mase, base_pin = [], [], [], []

    for fold in folds:
        train = series.loc[: pd.Timestamp(fold.origin_date)]
        actual = series.loc[pd.Timestamp(fold.test_start) : pd.Timestamp(fold.test_end)].to_numpy(
            dtype=float
        )
        if np.isnan(actual).all() or train.dropna().size <= SEASON_LENGTH:
            continue

        scale = naive_scale(train.dropna().to_numpy(dtype=float))
        observed = ~np.isnan(actual)

        realised_total = actual[observed].sum()
        base_daily = baseline_daily(train, horizon, scored)
        base_tot = baseline_total(train, horizon, scored)
        base_mase.append(mase(actual[observed], base_daily[observed, _at(scored, 0.5)], scale))
        base_pin.append(
            pinball(
                [realised_total],
                [float(base_tot[0, _at(scored, critical_ratio)])],
                critical_ratio,
            )
        )

        fitted = _fit_fold(train, horizon, scored)
        if fitted is None:
            model_mase.append(float("nan"))
            model_pin.append(float("nan"))
            continue
        daily_pred, total_pred = fitted
        model_mase.append(mase(actual[observed], daily_pred[observed, _at(scored, 0.5)], scale))
        model_pin.append(
            pinball(
                [realised_total],
                [float(total_pred[0, _at(scored, critical_ratio)])],
                critical_ratio,
            )
        )

    return (
        MethodScore(MODEL_NAME, tuple(model_mase), tuple(model_pin)),
        MethodScore(BASELINE_NAME, tuple(base_mase), tuple(base_pin)),
        n_folds,
    )


def _at(levels: tuple[float, ...], level: float) -> int:
    """Index of ``level`` in the fitted grid, nearest match."""
    return int(np.argmin(np.abs(np.asarray(levels) - level)))


def _training_origins(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Origins to build daily training rows from, strided when there are too many.

    The daily frame is one row per ``(origin, horizon)`` pair, so it grows as
    ``n_days x horizon``: a two-year series at a 28-day horizon is ~19,600 rows, fitted
    five times over during the backtest. That was 24 seconds for a single SKU, which puts a
    50-product upload past the job timeout.

    Adjacent origins differ by one day and produce near-identical feature rows, so striding
    costs very little information for a large saving. The full date range is kept -- this
    thins the sampling, it does not truncate history, which would quietly discard the
    oldest seasonality.
    """
    available = train.index[WARMUP_DAYS:]
    if len(available) * horizon <= MAX_TRAIN_ROWS:
        return pd.DatetimeIndex(available)
    stride = int(np.ceil(len(available) * horizon / MAX_TRAIN_ROWS))
    # Anchored at the end so the most recent origin is always included: it is the one the
    # production forecast is actually made from.
    return pd.DatetimeIndex(available[::-1][::stride][::-1])


def _fit_fold(
    train: pd.Series, horizon: int, levels: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit both grains on ``train`` and predict from its final origin.

    Returns ``None`` when the training window is too short to fit on, which the caller
    records as an unscorable fold rather than as a zero.
    """
    daily_rows = supervised_frame(train, horizon, origins=_training_origins(train, horizon))
    total_rows = total_frame(train, horizon)
    trainable_daily = daily_rows.dropna(subset=["y"])
    trainable_total = total_rows.dropna(subset=["y"])
    if len(trainable_daily) < MIN_TRAIN_ROWS or len(trainable_total) < MIN_TOTAL_ROWS:
        return None

    origin = train.index[-1]
    daily_models = fit_quantile_models(trainable_daily, levels)
    total_models = fit_quantile_models(trainable_total, levels)

    daily_future = supervised_frame(train, horizon, origins=pd.DatetimeIndex([origin]))
    total_future = total_rows[total_rows["origin"] == origin]
    if daily_future.empty or total_future.empty:
        return None

    return (
        predict_quantiles(daily_models, daily_future, levels),
        predict_quantiles(total_models, total_future, levels),
    )


# --------------------------------------------------------------------------- selection


def choose_method(model: MethodScore, baseline: MethodScore, n_folds: int) -> tuple[str, str]:
    """Pick the method to serve, and say why in a sentence a buyer can read.

    The caveat is driven by how many folds the **comparison** actually ran on, not by the
    fold budget the history allows. Those differ whenever the model could not be fitted in
    the earliest folds, and quoting the larger number would overstate the evidence behind
    the choice.

    Returns:
        ``(method_name, reason)``.

    """
    m_pin, b_pin = model.pinball_mean, baseline.pinball_mean
    m_mase, b_mase = model.mase_mean, baseline.mase_mean
    compared = min(model.n_scored, baseline.n_scored)
    caveat = spread_caveat(compared) or ""
    thin = (
        f" (the model could only be fitted in {compared} of {n_folds} folds; the earlier "
        "folds have too little training history)"
        if 0 < compared < n_folds
        else ""
    )

    if not np.isfinite(m_pin):
        return BASELINE_NAME, (
            f"The model could not be fitted on this product's history in any of the "
            f"{n_folds} backtest fold(s), so the pattern in your own sales is used instead. "
            f"{spread_caveat(baseline.n_scored) or ''}"
        ).strip()

    scale = max(abs(m_pin), abs(b_pin), 1e-9)
    if abs(m_pin - b_pin) / scale < DECISIVE_MARGIN:
        return BASELINE_NAME, (
            f"Too close to call: the model and the simple baseline scored within "
            f"{DECISIVE_MARGIN:.0%} of each other on your own data (ordering loss "
            f"{m_pin:.3f} vs {b_pin:.3f} over {compared} backtest fold(s){thin}), which is "
            f"not enough to prefer one. The simpler method is used. Day-to-day accuracy, "
            f"MASE: model {m_mase:.3f} vs baseline {b_mase:.3f}. {caveat}"
        ).strip()

    if b_pin < m_pin:
        return BASELINE_NAME, (
            f"The simple baseline beat the model on your own data, so the baseline is what "
            f"is used here. Ordering loss over {compared} backtest fold(s){thin}: baseline "
            f"{b_pin:.3f} vs model {m_pin:.3f} (lower is better). "
            f"Day-to-day accuracy, MASE: baseline {b_mase:.3f} vs model {m_mase:.3f}. "
            f"{caveat}"
        ).strip()

    disagreement = ""
    if np.isfinite(m_mase) and np.isfinite(b_mase) and b_mase < m_mase:
        disagreement = (
            " Note the two measures disagree: the baseline is better day-to-day "
            f"(MASE {b_mase:.3f} vs {m_mase:.3f}) while the model is better at the quantity "
            "that actually gets ordered. The ordering measure is the one used to choose, "
            "because it is the loss function of the number on the purchase order."
        )
    return MODEL_NAME, (
        f"The model beat the simple baseline on your own data over {compared} backtest "
        f"fold(s){thin}. Ordering loss: model {m_pin:.3f} vs baseline {b_pin:.3f} (lower is "
        f"better). Day-to-day accuracy, MASE: model {m_mase:.3f} vs baseline {b_mase:.3f}."
        f"{disagreement} {caveat}"
    ).strip()


# --------------------------------------------------------------------------- the order


def order_from_distribution(
    levels: tuple[float, ...], total_quantiles: np.ndarray, costs: CostModel, price: float
) -> tuple[float, float]:
    """Turn the horizon-total distribution into an order quantity and its expected cost.

    Both numbers come from ``optimize/`` rather than being recomputed here: the critical
    ratio from :meth:`~inventory_engine.optimize.costs.CostModel.critical_ratio`, the
    quantity from :func:`~inventory_engine.optimize.newsvendor.interpolate_quantile`, and
    the cost from
    :func:`~inventory_engine.optimize.newsvendor.expected_cost_from_distribution`. Tier 1
    and tier 2 therefore cannot drift apart on the arithmetic that matters most.

    Returns:
        ``(order_qty, expected_cost)``.

    """
    cr = costs.critical_ratio()
    raw = interpolate_quantile(np.asarray(levels), total_quantiles, cr)[0]
    qty = float(np.clip(raw, 0.0, None))
    cost = expected_cost_from_distribution(
        np.asarray(levels), total_quantiles[0], qty, costs, price
    )
    return qty, cost


def resolve_price(frame: pd.DataFrame) -> tuple[float, bool]:
    """Latest quoted price for this SKU, falling back to the shared constant.

    Latest rather than mean: the order being costed is a future one, and retail data
    reprices. Identical to tier 1's rule, and the fallback is the same
    :data:`~inventory_engine.optimize.costs.FALLBACK_PRICE` so the two tiers cannot quote
    different money for the same unpriced product.

    Returns:
        ``(price, is_fallback)``.

    """
    if "unit_price" not in frame.columns:
        return FALLBACK_PRICE, True
    priced = pd.to_numeric(frame["unit_price"], errors="coerce").dropna()
    priced = priced[priced > 0]
    return (float(priced.iloc[-1]), False) if len(priced) else (FALLBACK_PRICE, True)


# --------------------------------------------------------------------------- entry point


def forecast_sku(
    sku: str, frame: pd.DataFrame, horizon: int, costs: CostModel, *, history_days: int = 120
) -> SkuForecast:
    """Fit, backtest, choose, and price one SKU.

    Args:
        sku: Product identifier, for the result row.
        frame: This SKU's rows, with ``date``, ``units_sold`` and optionally ``unit_price``.
        horizon: Days the order must cover.
        costs: The caller's cost assumptions.
        history_days: How much history to hand back for the chart.

    Returns:
        A fully populated :class:`SkuForecast`, including the chart series.

    """
    series = to_daily(frame)
    cr = costs.critical_ratio()
    levels = levels_for(cr)
    price, is_fallback = resolve_price(frame)

    model_score, baseline_score, n_folds = backtest_sku(series, horizon, cr)
    method, reason = choose_method(model_score, baseline_score, n_folds)

    fitted = _fit_fold(series, horizon, levels) if method == MODEL_NAME else None
    if fitted is None:
        method = BASELINE_NAME
        daily = baseline_daily(series, horizon, levels)
        total = baseline_total(series, horizon, levels)
    else:
        daily, total = fitted

    qty, cost = order_from_distribution(levels, total, costs, price)
    return SkuForecast(
        sku=sku,
        method_used=method,
        method_reason=reason,
        n_folds=n_folds,
        model=model_score,
        baseline=baseline_score,
        critical_ratio=cr,
        order_qty=qty,
        expected_cost=cost,
        unit_price=price,
        price_is_fallback=is_fallback,
        series=_chart_series(series, daily, levels, horizon, qty, history_days),
    )


def _chart_series(
    series: pd.Series,
    daily: np.ndarray,
    levels: tuple[float, ...],
    horizon: int,
    order_qty: float,
    history_days: int,
) -> dict:
    """Assemble the JSON blob the chart draws.

    ``daily_rate`` rather than the raw order quantity: the order covers the whole horizon,
    so plotting its total as a reference line on a per-day axis would put a line at 140 on
    a chart whose values are around 5. The rate is the same decision expressed in the
    chart's own units.
    """
    tail = series.dropna().tail(history_days)
    last = series.index[-1]
    dates = [last + pd.Timedelta(days=h) for h in range(1, horizon + 1)]
    lo, mid, hi = _at(levels, BAND_LOW), _at(levels, 0.5), _at(levels, BAND_HIGH)

    return {
        "history": [{"d": d.strftime("%Y-%m-%d"), "v": float(v)} for d, v in tail.items()],
        "forecast": [
            {
                "d": d.strftime("%Y-%m-%d"),
                "point": round(float(daily[i, mid]), 3),
                "lo": round(float(daily[i, lo]), 3),
                "hi": round(float(daily[i, hi]), 3),
            }
            for i, d in enumerate(dates)
        ],
        "band": {
            "low_level": BAND_LOW,
            "high_level": BAND_HIGH,
            "label": f"Where sales landed {round((BAND_HIGH - BAND_LOW) * 100)}% of the time",
        },
        "order": {
            "total": round(order_qty, 2),
            "daily_rate": round(order_qty / horizon, 3),
            "label": f"Order covers {horizon} days",
        },
    }
