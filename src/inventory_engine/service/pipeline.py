"""Worker-side: fit, backtest, choose honestly, and turn the distribution into an order."""

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
from inventory_engine.service.adjust import FestivalPlan, plan_for
from inventory_engine.service.features import (
    FEATURE_COLUMNS,
    WARMUP_DAYS,
    supervised_frame,
    to_daily,
    total_frame,
)
from inventory_engine.service.folds import fold_count_for, spread_caveat

#: Quantile grid. Matches ``models.gbm.QUANTILES`` -- extended down to 0.10 because fresh-food
#: critical ratios land near 0.42, below the 0.5 a default grid would start at. The critical ratio
BASE_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

#: The interval the chart shades, labelled in plain language rather than as "q10-q90".
BAND_LOW: Final = 0.1
BAND_HIGH: Final = 0.9

#: Small-data parameters, and deliberately nothing like tier 1's.
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

#: Below this many usable daily rows the model is not fitted and the baseline is served outright.
MIN_TRAIN_ROWS: Final = 60

#: Same idea at the total grain, which is far scarcer: one row per origin rather than one per
#: (origin, horizon) pair, so a training window yielding 200 daily rows yields ~7 total rows.
MIN_TOTAL_ROWS: Final = 24

#: Ceiling on daily training rows per fit, enforced by striding origins in
#: :func:`_training_origins`.
MAX_TRAIN_ROWS: Final = 8_000

MODEL_NAME: Final = "quantile_gbm"
BASELINE_NAME: Final = "seasonal_naive"

#: Relative margin in pinball loss below which the two methods are called a draw rather than a win
#: for either.
DECISIVE_MARGIN: Final = 0.05


def levels_for(critical_ratio: float) -> tuple[float, ...]:
    """Return the quantile grid to fit, guaranteed to bracket ``critical_ratio``."""
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
        """Folds this method actually produced a score on."""
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
    #: What the newsvendor asked for, before any festival adjustment.
    order_qty_before_festival: float = 0.0
    #: The festival decision for this SKU: which of the three states, what matched it, and
    #: the sentence explaining it. See :mod:`inventory_engine.service.adjust`.
    festival: dict = field(default_factory=dict)


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
    """Fit one model per quantile level."""
    usable = train.dropna(subset=["y", *FEATURE_COLUMNS[:1]])
    return {level: _fit_one(usable, level) for level in levels}


def predict_quantiles(
    models: dict[float, object], features: pd.DataFrame, levels: tuple[float, ...]
) -> np.ndarray:
    """Predict every level for every row, rearranged into ascending order."""
    raw = np.column_stack(
        [models[level].predict(features[list(FEATURE_COLUMNS)]) for level in levels]
    )
    return _monotonize_rows(raw)


def _monotonize_rows(values: np.ndarray) -> np.ndarray:
    """Sort each row ascending and clip at zero."""
    return np.clip(np.sort(np.asarray(values, dtype=float), axis=1), 0.0, None)


# --------------------------------------------------------------------------- the baseline


def baseline_daily(train: pd.Series, horizon: int, levels: tuple[float, ...]) -> np.ndarray:
    """Seasonal-naive point path plus an empirical residual distribution."""
    values = train.dropna().to_numpy(dtype=float)
    point = seasonal_naive(values, horizon, SEASON_LENGTH)
    residuals = values[SEASON_LENGTH:] - values[:-SEASON_LENGTH]
    if residuals.size == 0:
        residuals = np.zeros(1)
    offsets = np.quantile(residuals, levels)
    return _monotonize_rows(point[:, None] + offsets[None, :])


def baseline_total(train: pd.Series, horizon: int, levels: tuple[float, ...]) -> np.ndarray:
    """Empirical quantiles of the horizon-day totals this series has actually produced."""
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
    series: pd.Series, horizon: int, critical_ratio: float, region: str | None = None
) -> tuple[MethodScore, MethodScore, int]:
    """Score the model and the baseline on rolling origins over the user's own history."""
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

        fitted = _fit_fold(train, horizon, scored, region)
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
    """Origins to build daily training rows from, strided when there are too many."""
    available = train.index[WARMUP_DAYS:]
    if len(available) * horizon <= MAX_TRAIN_ROWS:
        return pd.DatetimeIndex(available)
    stride = int(np.ceil(len(available) * horizon / MAX_TRAIN_ROWS))
    # Anchored at the end so the most recent origin is always included: it is the one the
    # production forecast is actually made from.
    return pd.DatetimeIndex(available[::-1][::stride][::-1])


def _fit_fold(
    train: pd.Series, horizon: int, levels: tuple[float, ...], region: str | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit both grains on ``train`` and predict from its final origin."""
    daily_rows = supervised_frame(
        train, horizon, origins=_training_origins(train, horizon), region=region
    )
    total_rows = total_frame(train, horizon, region=region)
    trainable_daily = daily_rows.dropna(subset=["y"])
    trainable_total = total_rows.dropna(subset=["y"])
    if len(trainable_daily) < MIN_TRAIN_ROWS or len(trainable_total) < MIN_TOTAL_ROWS:
        return None

    origin = train.index[-1]
    daily_models = fit_quantile_models(trainable_daily, levels)
    total_models = fit_quantile_models(trainable_total, levels)

    daily_future = supervised_frame(
        train, horizon, origins=pd.DatetimeIndex([origin]), region=region
    )
    total_future = total_rows[total_rows["origin"] == origin]
    if daily_future.empty or total_future.empty:
        return None

    return (
        predict_quantiles(daily_models, daily_future, levels),
        predict_quantiles(total_models, total_future, levels),
    )


# --------------------------------------------------------------------------- selection


def choose_method(model: MethodScore, baseline: MethodScore, n_folds: int) -> tuple[str, str]:
    """Pick the method to serve, and say why in a sentence a buyer can read."""
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
    """Turn the horizon-total distribution into an order quantity and its expected cost."""
    cr = costs.critical_ratio()
    raw = interpolate_quantile(np.asarray(levels), total_quantiles, cr)[0]
    qty = float(np.clip(raw, 0.0, None))
    cost = expected_cost_from_distribution(
        np.asarray(levels), total_quantiles[0], qty, costs, price
    )
    return qty, cost


def resolve_price(frame: pd.DataFrame) -> tuple[float, bool]:
    """Latest quoted price for this SKU, falling back to the shared constant."""
    if "unit_price" not in frame.columns:
        return FALLBACK_PRICE, True
    priced = pd.to_numeric(frame["unit_price"], errors="coerce").dropna()
    priced = priced[priced > 0]
    return (float(priced.iloc[-1]), False) if len(priced) else (FALLBACK_PRICE, True)


# --------------------------------------------------------------------------- entry point


def forecast_sku(
    sku: str,
    frame: pd.DataFrame,
    horizon: int,
    costs: CostModel,
    *,
    history_days: int = 120,
    region: str | None = None,
) -> SkuForecast:
    """Fit, backtest, choose, and price one SKU."""
    series = to_daily(frame)
    cr = costs.critical_ratio()
    levels = levels_for(cr)
    price, is_fallback = resolve_price(frame)

    model_score, baseline_score, n_folds = backtest_sku(series, horizon, cr, region)
    method, reason = choose_method(model_score, baseline_score, n_folds)

    fitted = _fit_fold(series, horizon, levels, region) if method == MODEL_NAME else None
    if fitted is None:
        method = BASELINE_NAME
        daily = baseline_daily(series, horizon, levels)
        total = baseline_total(series, horizon, levels)
    else:
        daily, total = fitted

    forecast_qty, _ = order_from_distribution(levels, total, costs, price)
    plan = plan_for(sku, series, horizon=horizon, region=region)
    qty = plan.apply(forecast_qty)
    # Costed at the quantity actually recommended, not at the one before the adjustment.
    cost = expected_cost_from_distribution(np.asarray(levels), total[0], qty, costs, price)

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
        series=_chart_series(series, daily, levels, horizon, qty, history_days, plan),
        order_qty_before_festival=forecast_qty,
        festival=plan.to_dict(),
    )


def _chart_series(
    series: pd.Series,
    daily: np.ndarray,
    levels: tuple[float, ...],
    horizon: int,
    order_qty: float,
    history_days: int,
    plan: FestivalPlan | None = None,
) -> dict:
    """Assemble the JSON blob the chart draws."""
    plan = plan or FestivalPlan.unchanged()
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
        # The festival run-ups the chart shades, so the days that were adjusted are the days the
        # reader can see.
        "festival_windows": [
            {
                "key": m.festival_key,
                "name": m.festival_name,
                "multiplier": round(m.multiplier, 3),
                "days": m.days_in_window,
            }
            for m in plan.matches
        ]
        if plan.adjusts
        else [],
    }
