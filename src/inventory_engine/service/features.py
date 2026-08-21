"""Features for a single user-supplied series."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

#: Lags measured backwards from the origin, in days.
LAGS: Final[tuple[int, ...]] = (1, 2, 3, 7, 14, 28)

#: Trailing windows for rolling statistics, ending at the origin.
WINDOWS: Final[tuple[int, ...]] = (7, 14, 28, 91)

#: Days of history before the first usable training row.
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
    # Calendar features, measured on the TARGET date rather than the origin.
    "days_to_festival",
    "days_since_festival",
    "in_festival_window",
)


def to_daily(frame: pd.DataFrame) -> pd.Series:
    """Collapse one SKU's rows into a date-indexed daily series."""
    series = (
        frame.assign(date=pd.to_datetime(frame["date"]))
        .groupby("date")["units_sold"]
        .sum(min_count=1)
        .sort_index()
    )
    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full)


def _origin_features(series: pd.Series) -> pd.DataFrame:
    """Per-origin features: everything knowable on the day the order is placed."""
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
    """Calendar proximity for one target date."""
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
    """Mean units on ``target_dow`` over the last ``n`` such weekdays at or before ``origin``."""
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
    """Build ``(origin, target)`` training rows from a daily series."""
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
    """Build one row per origin whose target is the ``horizon``-day forward total."""
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
