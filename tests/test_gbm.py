"""E4 — global GBM wiring.

The model's *accuracy* is E5's business, reported honestly in the epic doc. What these
tests pin is the wiring around it: that the panel split can't leak, that quantile crossings
are counted rather than hidden, and that the constant columns Phase 1's scope creates stay
out of the feature set.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from inventory_engine.backtest.folds import make_folds
from inventory_engine.features.build import feature_columns
from inventory_engine.models.gbm import (
    BASE_PARAMS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_FEATURES,
    QUANTILES,
    _count_quantile_crossings,
    _predict_rows,
    feature_matrix,
)

LAST = date(2016, 5, 22)


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------


def test_series_identity_is_a_feature():
    """This is what makes the model global rather than 720 separate models."""
    features = feature_matrix(pd.DataFrame())
    for column in CATEGORICAL_FEATURES:
        assert column in features


def test_constant_scope_columns_are_excluded():
    """cat_id and state_id are fixed by Phase 1's scope, so they carry no signal.

    Leaving them in gives the model free variance to chase and inflates the feature count
    for nothing.
    """
    features = feature_matrix(pd.DataFrame())
    assert "cat_id" not in features
    assert "state_id" not in features
    assert "horizon" not in features, "panel is built at a single horizon; constant"


def test_target_is_not_a_feature():
    """The most direct leak available: predicting units from units."""
    assert "units" not in feature_matrix(pd.DataFrame())


def test_every_engineered_feature_reaches_the_model():
    """Nothing E2 built should be silently dropped on the way in."""
    features = set(feature_matrix(pd.DataFrame()))
    assert set(feature_columns()) <= features


def test_event_type_is_declared_categorical():
    """A string column LightGBM cannot consume as a float; the first full fit failed on it."""
    assert "event_type" in CATEGORICAL_COLUMNS
    assert set(CATEGORICAL_FEATURES) < set(CATEGORICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Quantile monotonicity
# ---------------------------------------------------------------------------


def _quantile_frame(values: dict[float, list[float]]) -> pd.DataFrame:
    rows = []
    for q, series in values.items():
        for i, v in enumerate(series):
            rows.append(
                {
                    "fold": 0,
                    "item_id": f"I{i}",
                    "store_id": "CA_1",
                    "target_date": pd.Timestamp("2016-01-04"),
                    "quantile": q,
                    "yhat": v,
                }
            )
    return pd.DataFrame(rows)


def test_monotonic_quantiles_report_no_crossings():
    frame = _quantile_frame({0.5: [1.0, 2.0], 0.9: [2.0, 3.0], 0.95: [3.0, 4.0], 0.99: [4.0, 5.0]})
    crossings, checked = _count_quantile_crossings(frame)
    assert (crossings, checked) == (0, 2)


def test_crossed_quantiles_are_detected():
    """q0.9 above q0.95 means E7 would select a nonsense number by critical ratio."""
    frame = _quantile_frame({0.5: [1.0, 1.0], 0.9: [9.0, 2.0], 0.95: [3.0, 3.0], 0.99: [4.0, 4.0]})
    crossings, checked = _count_quantile_crossings(frame)
    assert crossings == 1
    assert checked == 2


def test_crossing_check_needs_at_least_two_levels():
    crossings, checked = _count_quantile_crossings(_quantile_frame({0.5: [1.0]}))
    assert (crossings, checked) == (0, 0)


def test_quantile_levels_match_the_newsvendor_layer():
    """E7 selects the CR-th quantile by interpolating between these."""
    assert QUANTILES == (0.5, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Forecast row shaping
# ---------------------------------------------------------------------------


@pytest.fixture
def rows():
    fold = make_folds(LAST, 5, 28)[0]
    targets = [fold.test_start, fold.test_start + timedelta(days=5), fold.test_end]
    return fold, pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(targets)),
            "item_id": ["FOODS_1_001"] * 3,
            "store_id": ["CA_1"] * 3,
            "dept_id": ["FOODS_1"] * 3,
        }
    )


def test_horizon_counts_from_the_fold_origin(rows):
    fold, frame = rows
    out = _predict_rows(np.array([1.0, 2.0, 3.0]), frame, fold, "run", None)
    assert out["horizon"].tolist() == [1, 6, 28]


def test_negative_predictions_are_clipped_at_write_time(rows):
    """Demand cannot be negative, and E7 orders against the stored number."""
    fold, frame = rows
    out = _predict_rows(np.array([-4.0, 0.0, 2.5]), frame, fold, "run", None)
    assert out["yhat"].tolist() == [0.0, 0.0, 2.5]


def test_point_forecast_has_null_quantile(rows):
    fold, frame = rows
    out = _predict_rows(np.array([1.0, 1.0, 1.0]), frame, fold, "run", None)
    assert out["quantile"].isna().all()


def test_quantile_forecast_records_its_level(rows):
    fold, frame = rows
    out = _predict_rows(np.array([1.0, 1.0, 1.0]), frame, fold, "run", 0.95)
    assert (out["quantile"] == 0.95).all()


def test_rows_are_written_as_unreconciled(rows):
    """E6 adds reconciled rows alongside; base forecasts must stay distinguishable."""
    fold, frame = rows
    out = _predict_rows(np.array([1.0, 1.0, 1.0]), frame, fold, "run", None)
    assert not out["reconciled"].any()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_objective_suits_zero_inflated_counts():
    """Tweedie, taken from the M5 literature and labelled untuned rather than searched."""
    assert BASE_PARAMS["objective"] == "tweedie"
    assert 1.0 < BASE_PARAMS["tweedie_variance_power"] < 2.0


def test_seed_is_fixed():
    assert isinstance(BASE_PARAMS["seed"], int)
