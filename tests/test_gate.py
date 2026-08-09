"""The data gate: does it refuse, and does it say why in numbers.

These tests are pure -- no server, no queue, no Postgres. The gate is the one piece of tier
2 whose behaviour a user meets directly on a bad file, so it is proven in isolation before
any infrastructure exists to hide behind.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from inventory_engine.service.folds import fold_count_for, spread_caveat
from inventory_engine.service.gate import (
    MAX_GAP_DAYS,
    MIN_HISTORY_DAYS,
    MIN_OBSERVATIONS,
    evaluate,
    refusal_message,
    resolve_columns,
)


def series(sku="WIDGET", days=120, start="2025-01-01", units=5, freq="D", price=2.50):
    """A clean daily series for one SKU, long enough to pass by default."""
    dates = pd.date_range(start, periods=days, freq=freq)
    return pd.DataFrame(
        {
            "sku": sku,
            "date": dates.strftime("%Y-%m-%d"),
            "units_sold": units,
            "unit_price": price,
        }
    )


# --------------------------------------------------------------------------- columns


def test_missing_required_column_is_fatal_and_names_it():
    frame = series().drop(columns=["units_sold"])
    report = evaluate(frame)
    assert report.fatal is not None
    assert "'units_sold'" in report.fatal
    assert not report.admitted and not report.rejected


def test_aliases_resolve_and_the_mapping_is_reported_back():
    frame = series().rename(columns={"sku": "Product", "units_sold": "QTY", "date": "Day"})
    report = evaluate(frame)
    assert report.fatal is None
    assert report.column_mapping["sku"] == "Product"
    assert report.column_mapping["units_sold"] == "QTY"
    assert report.column_mapping["date"] == "Day"


def test_unit_price_is_optional():
    report = evaluate(series().drop(columns=["unit_price"]))
    assert report.fatal is None
    assert len(report.admitted) == 1


def test_resolve_columns_lists_every_missing_column_at_once():
    _, fatal = resolve_columns(["date"])
    assert "'sku'" in fatal and "'units_sold'" in fatal


# --------------------------------------------------------------------------- thresholds


def test_the_deliberately_too_small_upload_is_refused_with_the_actual_number():
    """21 days of history: the case the brief calls out by name."""
    report = evaluate(series(days=21))
    assert not report.usable
    assert len(report.rejected) == 1
    assert report.rejected[0].n_days == 21
    assert "90 days of history needed, 21 found" in report.rejected[0].reasons[0]

    message = refusal_message(report)
    assert "90 days of history needed, 21 found" in message
    assert "quick calculator" in message


def test_exactly_at_the_threshold_is_admitted():
    report = evaluate(series(days=MIN_HISTORY_DAYS))
    assert len(report.admitted) == 1
    assert report.admitted[0].n_days == MIN_HISTORY_DAYS


def test_one_day_under_the_threshold_is_refused():
    report = evaluate(series(days=MIN_HISTORY_DAYS - 1))
    assert len(report.rejected) == 1


def test_a_long_span_with_too_few_rows_fails_the_observation_check():
    """200-day span, 8 rows. The span passes; the count is what catches it."""
    dates = pd.date_range("2025-01-01", periods=8, freq="28D")
    frame = pd.DataFrame({"sku": "SPARSE", "date": dates, "units_sold": 3})
    report = evaluate(frame)
    assert len(report.rejected) == 1
    verdict = report.rejected[0]
    assert verdict.n_days > MIN_HISTORY_DAYS
    assert verdict.n_nonnull == 8
    assert any(f"{MIN_OBSERVATIONS} recorded days needed, 8 found" in r for r in verdict.reasons)


def test_zero_demand_days_count_as_observations():
    """61.6% of the M5 panel is zeros. A zero is data; only a null is missing."""
    frame = series(days=120, units=0)
    report = evaluate(frame)
    assert len(report.admitted) == 1
    assert report.admitted[0].n_nonnull == 120


def test_null_units_do_not_count_as_observations():
    """A 120-day span with 15 recorded days fails on the count, not the span."""
    frame = series(days=120)
    frame.loc[15:, "units_sold"] = None
    report = evaluate(frame)
    verdict = report.rejected[0]
    assert verdict.n_days == 120
    assert verdict.n_obs == 120
    assert verdict.n_nonnull == 15
    assert any(f"{MIN_OBSERVATIONS} recorded days needed, 15 found" in r for r in verdict.reasons)


def test_a_large_gap_is_flagged_rather_than_filled():
    early = series(days=60, start="2025-01-01")
    late = series(days=60, start="2025-05-01")
    report = evaluate(pd.concat([early, late], ignore_index=True))
    verdict = (report.admitted + report.rejected)[0]
    assert verdict.max_gap_days > MAX_GAP_DAYS
    assert not verdict.admitted
    assert any("no rows at all" in r for r in verdict.reasons)


def test_a_small_gap_is_tolerated():
    frame = series(days=120)
    frame = frame.drop(index=range(40, 45))  # a 5-day hole
    report = evaluate(frame)
    assert len(report.admitted) == 1
    assert report.admitted[0].max_gap_days == 5


# --------------------------------------------------------------------------- mixed files


def test_a_failing_sku_is_excluded_and_the_rest_proceed():
    frame = pd.concat([series("GOOD", days=120), series("THIN", days=30)], ignore_index=True)
    report = evaluate(frame)
    assert report.usable
    assert [v.sku for v in report.admitted] == ["GOOD"]
    assert [v.sku for v in report.rejected] == ["THIN"]
    assert "90 days of history needed, 30 found" in report.rejected[0].reasons[0]


def test_when_every_sku_fails_the_message_names_several_of_them():
    frame = pd.concat(
        [series("A", days=21), series("B", days=14), series("C", days=45)], ignore_index=True
    )
    report = evaluate(frame)
    assert not report.usable
    message = refusal_message(report)
    for sku in ("A", "B", "C"):
        assert sku in message
    assert "quick calculator" in message


# --------------------------------------------------------------------------- cleaning


def test_duplicate_dates_are_summed_and_the_count_is_reported():
    frame = pd.concat([series(days=120), series(days=120)], ignore_index=True)
    report = evaluate(frame)
    assert report.rows_merged_duplicate_date == 120
    assert report.admitted[0].n_obs == 120
    assert any("added together" in w for w in report.warnings())


def test_negative_units_are_clipped_and_the_count_is_reported():
    frame = series(days=120)
    frame.loc[0:4, "units_sold"] = -2
    report = evaluate(frame)
    assert report.negative_units_clipped == 5
    assert any("returns" in w for w in report.warnings())


def test_a_few_unreadable_dates_are_dropped_and_counted():
    frame = series(days=120)
    frame.loc[0:2, "date"] = "not-a-date"
    report = evaluate(frame)
    assert report.rows_dropped_unparseable_date == 3
    assert any("could not be read" in w for w in report.warnings())


def test_mostly_unreadable_dates_is_fatal_rather_than_per_sku():
    frame = series(days=120)
    frame["date"] = "31/02/2025 garbage"
    report = evaluate(frame)
    assert report.fatal is not None
    assert "YYYY-MM-DD" in report.fatal


def test_an_empty_file_is_fatal():
    frame = pd.DataFrame(columns=["sku", "date", "units_sold"])
    report = evaluate(frame)
    assert report.fatal is not None
    assert "no data rows" in report.fatal


def test_gate_reads_a_real_csv_end_to_end():
    csv = series(days=120).to_csv(index=False)
    report = evaluate(pd.read_csv(io.StringIO(csv)))
    assert report.usable
    assert report.rows_read == 120


# --------------------------------------------------------------------------- fold budget


@pytest.mark.parametrize(
    ("n_days", "horizon", "expected"),
    [
        (90, 28, 2),  # the gate's own minimum: two folds, and we say so
        (90, 7, 5),  # a short horizon fits the full five
        (1941, 28, 5),  # tier 1's panel, capped at MAX_FOLDS
        (40, 28, 0),  # admitted by span but unbacktestable at this horizon
        (56, 28, 1),
    ],
)
def test_fold_count_is_derived_from_the_history_actually_present(n_days, horizon, expected):
    assert fold_count_for(n_days, horizon) == expected


def test_thin_fold_counts_carry_a_caveat_and_healthy_ones_do_not():
    assert spread_caveat(1) is not None
    assert spread_caveat(2) is not None
    assert spread_caveat(5) is None
