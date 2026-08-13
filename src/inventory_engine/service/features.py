"""Features for a single user-supplied series.

Tier 1's feature builder (``features/build.py``) reads the M5 panel: SNAP flags, event
calendars, department and store ids, weekly repricing. None of that exists in a CSV with
four columns, so this builds the largest feature set derivable from ``(date, units)`` alone
and nothing more.

The framing: origin features, horizon as a column
-------------------------------------------------
Every training row is one ``(origin, target)`` pair. Features are computed **at the origin**
-- the last day whose sales are known -- and the number of days from origin to target is
itself a feature. One model therefore covers the whole horizon, rather than 28 separate
direct-h models or a recursive loop that feeds its own errors back in.

This is the same shape tier 1's GBM uses, and it is chosen for the same reason: the leak is
structurally impossible rather than merely avoided. A feature cannot see past the origin
because it is never given a row later than the origin to look at.

Why no target-date lags
-----------------------
``units[target - 7]`` would be a wonderfully predictive feature and is unavailable at
prediction time for any target more than 7 days out. Every lag here is measured backwards
from the *origin*, so a feature that exists in training exists in production.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

#: Lags measured backwards from the origin, in days.
LAGS: Final[tuple[int, ...]] = (1, 2, 3, 7, 14, 28)

#: Trailing windows for rolling statistics, ending at the origin.
WINDOWS: Final[tuple[int, ...]] = (7, 14, 28, 91)

#: Days of history before the first usable training row.
#:
#: The longest **lag**, not the longest window. A lag is genuinely undefined before its
#: offset -- ``lag_28`` needs 28 prior days and there is no honest substitute -- whereas the
#: rolling statistics carry ``min_periods`` and are defined from a quarter of their window
#: onward, so a 91-day mean starts producing values on day 23 over however much history
#: exists.
#:
#: Setting this to ``max(WINDOWS)`` instead cost the model every short series: at 91 it
#: exceeded the training window of every fold on a 120-day upload, ``supervised_frame``
#: returned nothing, and the baseline was served by walkover with the reason "the model
#: could not be fitted" on data that could comfortably fit one. Found by running the
#: pipeline on real M5 series, not by a unit test -- ``test_pipeline`` now pins it.
#:
#: 28 also lines up with :data:`inventory_engine.service.folds.MIN_TRAIN_DAYS`, so the fold
#: budget and the feature budget agree about what "enough history" means.
WARMUP_DAYS: Final = max(LAGS)

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "horizon",
    "target_dow",
    "target_dom",
    "target_month",
    "target_is_weekend",
    *(f"lag_{lag}" for lag in LAGS),
    *(f"roll_mean_{w}" for w in WINDOWS),
    *(f"roll_std_{w}" for w in WINDOWS),
    *(f"roll_zero_share_{w}" for w in WINDOWS),
    "same_dow_mean_4",
    "trend_index",
    # Calendar features, measured on the TARGET date rather than the origin. That is the
    # documented exception to the origin rule: unlike a lag, a festival date is genuinely
    # known in advance, exactly as tier 1's dim_calendar extends past its sales data.
    # Knowing the target is three days before Diwali needs nothing the buyer would not have
    # when placing the order.
    "days_to_festival",
    "days_since_festival",
    "in_festival_window",
)


def to_daily(frame: pd.DataFrame) -> pd.Series:
    """Collapse one SKU's rows into a date-indexed daily series.

    Reindexed onto a complete daily calendar between the first and last observation, so
    that "lag 7" means seven *days* rather than seven rows. Missing days become ``NaN``,
    not zero: the gate has already refused any series whose gaps are large enough for that
    distinction to change the answer, and within a tolerated gap a NaN keeps the rolling
    statistics honest where a zero would drag them down.

    Args:
        frame: Rows for one SKU with ``date`` and ``units_sold`` columns.

    Returns:
        Daily series indexed by date, ascending.

    """
    series = (
        frame.assign(date=pd.to_datetime(frame["date"]))
        .groupby("date")["units_sold"]
        .sum(min_count=1)
        .sort_index()
    )
    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full)


def _origin_features(series: pd.Series) -> pd.DataFrame:
    """Per-origin features: everything knowable on the day the order is placed.

    Indexed by origin date. Every column is computed from ``series`` up to and including
    that date, using ``shift`` where a window would otherwise include the origin's own
    value in a way that a production caller could not reproduce.
    """
    values = series.astype(float)
    out = pd.DataFrame(index=series.index)

    for lag in LAGS:
        out[f"lag_{lag}"] = values.shift(lag - 1)

    for window in WINDOWS:
        rolling = values.rolling(window, min_periods=max(2, window // 4))
        out[f"roll_mean_{window}"] = rolling.mean()
        out[f"roll_std_{window}"] = rolling.std()
        out[f"roll_zero_share_{window}"] = (
            (values == 0).astype(float).rolling(window, min_periods=max(2, window // 4)).mean()
        )

    out["trend_index"] = np.arange(len(series), dtype=float)
    return out


def festival_features(target: pd.Timestamp, region: str | None) -> dict[str, float]:
    """Calendar proximity for one target date.

    ``region=None`` means no calendar, and the features come back at their neutral values
    -- maximum distance, not in a window. They are still emitted rather than omitted so the
    feature matrix has one shape regardless of region, which keeps a model fitted with a
    calendar loadable against data without one.
    """
    from inventory_engine.service.festivals import HORIZON_DAYS, distance

    if region is None:
        return {
            "days_to_festival": float(HORIZON_DAYS),
            "days_since_festival": float(HORIZON_DAYS),
            "in_festival_window": 0.0,
        }
    to_next, since_prev, inside = distance(target.date(), region)
    return {
        "days_to_festival": float(to_next),
        "days_since_festival": float(since_prev),
        "in_festival_window": float(inside),
    }


def _same_dow_mean(series: pd.Series, origin: pd.Timestamp, target_dow: int, n: int = 4) -> float:
    """Mean units on ``target_dow`` over the last ``n`` such weekdays at or before ``origin``.

    The single most useful feature on daily retail demand, and the one seasonal naive is
    built on -- included explicitly so the model starts from the baseline's own information
    rather than having to rediscover it.
    """
    history = series.loc[:origin]
    matching = history[history.index.dayofweek == target_dow].dropna()
    if matching.empty:
        return float("nan")
    return float(matching.tail(n).mean())


def supervised_frame(
    series: pd.Series,
    horizon: int,
    *,
    origins: pd.DatetimeIndex | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    """Build ``(origin, target)`` training rows from a daily series.

    Args:
        series: Daily series from :func:`to_daily`.
        horizon: Maximum days ahead to build rows for. One row per origin per ``h`` in
            ``1..horizon``.
        region: Festival calendar to read proximity from. ``None`` emits the calendar
            features at neutral values, so the matrix keeps one shape either way.
        origins: Restrict to these origins. ``None`` uses every origin with enough warmup
            behind it and at least one in-range target ahead of it.

    Returns:
        A frame with :data:`FEATURE_COLUMNS`, plus ``origin``, ``target`` and ``y``. Rows
        whose target is unobserved are kept with ``y`` as ``NaN`` -- prediction needs them;
        training drops them.

    """
    per_origin = _origin_features(series)
    usable = series.index[WARMUP_DAYS:]
    chosen = usable if origins is None else pd.DatetimeIndex(origins)

    rows = []
    for origin in chosen:
        if origin not in per_origin.index:
            continue
        base = per_origin.loc[origin]
        for h in range(1, horizon + 1):
            target = origin + pd.Timedelta(days=h)
            rows.append(
                {
                    "origin": origin,
                    "target": target,
                    "horizon": h,
                    "target_dow": target.dayofweek,
                    "target_dom": target.day,
                    "target_month": target.month,
                    "target_is_weekend": int(target.dayofweek >= 5),
                    **festival_features(origin + pd.Timedelta(days=1), region),
                    "same_dow_mean_4": _same_dow_mean(series, origin, target.dayofweek),
                    **festival_features(target, region),
                    "y": float(series.get(target, np.nan)),
                    **base.to_dict(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "origin", "target", "y"])
    return pd.DataFrame(rows)


def total_frame(series: pd.Series, horizon: int, *, region: str | None = None) -> pd.DataFrame:
    """Build one row per origin whose target is the ``horizon``-day forward total.

    This is the grain the order decision is actually made at. The newsvendor problem is
    single-period: the question is not "how many will sell next Tuesday" but "how many
    should be on the shelf to cover the next H days", and the quantile of a sum is not the
    sum of quantiles. Tier 1's calculator frames it the same way, via ``windowTotals`` --
    the two tiers answer the same question so their numbers can be compared.

    Args:
        series: Daily series from :func:`to_daily`.
        horizon: Days the order must cover.
        region: Festival calendar to read proximity from, or ``None``.

    Returns:
        Frame with :data:`FEATURE_COLUMNS` (``horizon`` constant) plus ``origin`` and ``y``,
        where ``y`` is the realised total over ``origin+1 .. origin+horizon``.

    """
    per_origin = _origin_features(series)
    values = series.astype(float)
    # Forward-looking on purpose: this is the training *target*, not a feature. Reversing,
    # rolling, and reversing back gives the sum over the H days strictly after each origin.
    forward = values[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)

    rows = []
    for origin in series.index[WARMUP_DAYS:]:
        base = per_origin.loc[origin]
        rows.append(
            {
                "origin": origin,
                "horizon": horizon,
                # The order covers origin+1 onwards, so the first covered day is the target
                # the calendar features describe.
                **festival_features(origin + pd.Timedelta(days=1), region),
                "target_dow": (origin + pd.Timedelta(days=1)).dayofweek,
                "target_dom": (origin + pd.Timedelta(days=1)).day,
                "target_month": (origin + pd.Timedelta(days=1)).month,
                "target_is_weekend": int((origin + pd.Timedelta(days=1)).dayofweek >= 5),
                "same_dow_mean_4": _same_dow_mean(
                    series, origin, (origin + pd.Timedelta(days=1)).dayofweek
                ),
                "y": float(forward.get(origin, np.nan)),
                **base.to_dict(),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "origin", "y"])
    return pd.DataFrame(rows)
