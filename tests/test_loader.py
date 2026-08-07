"""Phase 1 loader tests.

These build a miniature M5 from synthetic CSVs rather than touching the real
300 MB extracts, so the suite runs in about a second and works on a clean
checkout with no Kaggle download.

Two fixtures:

``raw_dir``
    Four days, six series. Enough to assert the joins, filters and null
    handling. Too short for the stratified sampler, which needs a reference
    window behind the backtest region, so these tests pass ``items_per_dept=None``.

``sampling_dir``
    260 days with a hand-built intermittency gradient, used to test the
    stratified sample itself.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest

from inventory_engine.config import BACKTEST_DAYS, STRATIFY_WINDOW_DAYS, Scope
from inventory_engine.data.loader import (
    MissingRawDataError,
    build_warehouse,
    verify_raw_files,
)

ALL_ITEMS = Scope(items_per_dept=None)

# --------------------------------------------------------------------------
# Small fixture: joins, filters, null handling
# --------------------------------------------------------------------------

# Four calendar days, but only three days of sales: the fourth exercises the
# requirement that dim_calendar extends past the end of the sales window.
CALENDAR_CSV = """date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI
2011-01-29,11101,Saturday,1,1,2011,d_1,,,,,1,0,0
2011-01-30,11101,Sunday,2,1,2011,d_2,SuperBowl,Sporting,,,0,1,0
2011-02-05,11102,Saturday,1,2,2011,d_3,,,,,0,0,1
2011-02-06,11102,Sunday,2,2,2011,d_4,,,,,1,0,0
"""

# CA/FOODS rows are in scope. The HOBBIES row and the TX row must be dropped.
SALES_CSV = """id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3
FOODS_1_001_CA_1_evaluation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,5,0,7
FOODS_1_001_CA_2_evaluation,FOODS_1_001,FOODS_1,FOODS,CA_2,CA,1,2,3
FOODS_1_002_CA_1_evaluation,FOODS_1_002,FOODS_1,FOODS,CA_1,CA,0,0,1
FOODS_2_001_CA_1_evaluation,FOODS_2_001,FOODS_2,FOODS,CA_1,CA,9,9,9
HOBBIES_1_001_CA_1_evaluation,HOBBIES_1_001,HOBBIES_1,HOBBIES,CA_1,CA,4,4,4
FOODS_1_001_TX_1_evaluation,FOODS_1_001,FOODS_1,FOODS,TX_1,TX,8,8,8
"""

# Deliberately missing FOODS_1_002 in week 11101 -> that day must land as NULL price.
PRICES_CSV = """store_id,item_id,wm_yr_wk,sell_price
CA_1,FOODS_1_001,11101,2.50
CA_1,FOODS_1_001,11102,3.00
CA_2,FOODS_1_001,11101,2.50
CA_2,FOODS_1_001,11102,2.50
CA_1,FOODS_1_002,11102,1.25
CA_1,FOODS_2_001,11101,4.00
CA_1,FOODS_2_001,11102,4.00
"""


@pytest.fixture
def raw_dir(tmp_path):
    """Write a synthetic M5 extract into a temp directory."""
    (tmp_path / "calendar.csv").write_text(CALENDAR_CSV, encoding="utf-8")
    (tmp_path / "sales_train_evaluation.csv").write_text(SALES_CSV, encoding="utf-8")
    (tmp_path / "sell_prices.csv").write_text(PRICES_CSV, encoding="utf-8")
    return tmp_path


@pytest.fixture
def warehouse(raw_dir, tmp_path):
    """Build a warehouse from the synthetic extract and return (path, report)."""
    db_path = tmp_path / "test.duckdb"
    report = build_warehouse(data_dir=raw_dir, db_path=db_path, scope=ALL_ITEMS)
    return db_path, report


def query(db_path, sql):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_verify_raw_files_names_what_is_missing(tmp_path):
    (tmp_path / "calendar.csv").write_text(CALENDAR_CSV, encoding="utf-8")
    with pytest.raises(MissingRawDataError) as exc:
        verify_raw_files(tmp_path)
    message = str(exc.value)
    assert "sales_train_evaluation.csv" in message
    assert "sell_prices.csv" in message
    assert "calendar.csv" not in message.split("Expected them in")[0]
    assert str(tmp_path) in message
    assert "kaggle.com" in message


def test_scope_filters_to_state_and_category(warehouse):
    db_path, report = warehouse
    assert report.series == 4
    assert report.items == 3
    assert report.stores == 2
    assert report.rows == 12  # 4 series x 3 days
    states, cats = query(db_path, "SELECT DISTINCT state_id, cat_id FROM fact_sales")[0]
    assert (states, cats) == ("CA", "FOODS")


def test_calendar_extends_past_sales_window(warehouse):
    db_path, _ = warehouse
    (n_cal,) = query(db_path, "SELECT count(*) FROM dim_calendar")[0]
    (max_sales,) = query(db_path, "SELECT max(date) FROM fact_sales")[0]
    (max_cal,) = query(db_path, "SELECT max(date) FROM dim_calendar")[0]
    assert n_cal == 4
    assert max_cal > max_sales, "future calendar rows are needed for days-to-event features"


def test_units_land_on_the_right_dates(warehouse):
    db_path, _ = warehouse
    rows = query(
        db_path,
        """SELECT CAST(date AS VARCHAR), units FROM fact_sales
           WHERE item_id = 'FOODS_1_001' AND store_id = 'CA_1' ORDER BY date""",
    )
    assert rows == [("2011-01-29", 5), ("2011-01-30", 0), ("2011-02-05", 7)]


def test_price_is_null_when_item_not_listed_that_week(warehouse):
    db_path, _ = warehouse
    rows = query(
        db_path,
        """SELECT CAST(date AS VARCHAR), price FROM fact_sales
           WHERE item_id = 'FOODS_1_002' AND store_id = 'CA_1' ORDER BY date""",
    )
    assert rows == [("2011-01-29", None), ("2011-01-30", None), ("2011-02-05", 1.25)]


def test_snap_comes_from_the_scoped_state(warehouse):
    db_path, _ = warehouse
    rows = query(
        db_path,
        """SELECT CAST(date AS VARCHAR), snap FROM fact_sales
           WHERE item_id = 'FOODS_1_001' AND store_id = 'CA_1' ORDER BY date""",
    )
    # snap_CA is 1,0,0 across d_1..d_3; snap_TX/snap_WI must not leak in.
    assert [r[1] for r in rows] == [1, 0, 0]


def test_event_is_null_when_absent(warehouse):
    db_path, _ = warehouse
    rows = query(db_path, "SELECT DISTINCT CAST(date AS VARCHAR), event FROM fact_sales ORDER BY 1")
    assert rows == [("2011-01-29", None), ("2011-01-30", "SuperBowl"), ("2011-02-05", None)]


def test_no_duplicate_series_days(warehouse):
    db_path, _ = warehouse
    (dupes,) = query(
        db_path,
        """SELECT count(*) FROM (
             SELECT date, item_id, store_id FROM fact_sales
             GROUP BY 1,2,3 HAVING count(*) > 1)""",
    )[0]
    assert dupes == 0


def test_dept_filter(raw_dir, tmp_path):
    report = build_warehouse(
        data_dir=raw_dir,
        db_path=tmp_path / "dept.duckdb",
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=None),
    )
    assert report.depts == 1
    assert report.series == 3


def test_rebuild_requires_overwrite(raw_dir, tmp_path):
    db_path = tmp_path / "twice.duckdb"
    build_warehouse(data_dir=raw_dir, db_path=db_path, scope=ALL_ITEMS)
    with pytest.raises(ValueError, match="overwrite"):
        build_warehouse(data_dir=raw_dir, db_path=db_path, scope=ALL_ITEMS)
    report = build_warehouse(data_dir=raw_dir, db_path=db_path, scope=ALL_ITEMS, overwrite=True)
    assert report.rows == 12


def test_empty_scope_raises(raw_dir, tmp_path):
    with pytest.raises(ValueError, match="zero series"):
        build_warehouse(
            data_dir=raw_dir,
            db_path=tmp_path / "empty.duckdb",
            scope=Scope(cat_id="HOUSEHOLD", items_per_dept=None),
        )


def test_unknown_state_rejected(raw_dir, tmp_path):
    with pytest.raises(ValueError, match="Unknown state_id"):
        build_warehouse(
            data_dir=raw_dir,
            db_path=tmp_path / "bad.duckdb",
            scope=Scope(state_id="NY", items_per_dept=None),
        )


# --------------------------------------------------------------------------
# Larger fixture: stratified sampling
# --------------------------------------------------------------------------

N_DAYS = 260
START = date(2011, 1, 29)
STORES = ("CA_1", "CA_2")

# FOODS_1 carries the intermittency gradient. FOODS_2 is deliberately smaller
# than any sane item budget, to exercise the short-department path.
DENSE_ITEMS = [f"FOODS_1_{i:03d}" for i in range(30)]
LATE_BLOOMER = "FOODS_1_900"  # sells only inside the backtest region
UNLISTED = "FOODS_1_901"  # sells, but never has a price row
SMALL_DEPT_ITEMS = [f"FOODS_2_{i:03d}" for i in range(4)]


def _sales_pattern(item_index: int) -> list[int]:
    """Sell every ``1 + item_index // 3``-th day: a monotone zero-density ramp."""
    period = 1 + item_index // 3
    return [3 if day % period == 0 else 0 for day in range(N_DAYS)]


def _write_sampling_extract(directory):
    """Write a 260-day synthetic extract with a controlled intermittency spread."""
    day_cols = [f"d_{i + 1}" for i in range(N_DAYS)]

    cal_lines = [
        "date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,"
        "event_name_2,event_type_2,snap_CA,snap_TX,snap_WI"
    ]
    for i in range(N_DAYS):
        d = START + timedelta(days=i)
        cal_lines.append(
            f"{d.isoformat()},{11101 + i // 7},{d.strftime('%A')},{d.weekday() + 1},"
            f"{d.month},{d.year},d_{i + 1},,,,,0,0,0"
        )

    sales_lines = ["id,item_id,dept_id,cat_id,store_id,state_id," + ",".join(day_cols)]

    def add(item_id, dept_id, units):
        for store in STORES:
            sales_lines.append(
                f"{item_id}_{store}_evaluation,{item_id},{dept_id},FOODS,{store},CA,"
                + ",".join(str(u) for u in units)
            )

    for idx, item in enumerate(DENSE_ITEMS):
        add(item, "FOODS_1", _sales_pattern(idx))
    # Silent until the backtest region begins, then dense. If the stratifier
    # measured intermittency over the evaluation period it would call this item
    # dense; measured correctly, it is maximally sparse.
    add(LATE_BLOOMER, "FOODS_1", [0] * (N_DAYS - BACKTEST_DAYS) + [5] * BACKTEST_DAYS)
    add(UNLISTED, "FOODS_1", [4] * N_DAYS)
    for idx, item in enumerate(SMALL_DEPT_ITEMS):
        add(item, "FOODS_2", _sales_pattern(idx * 3))

    price_lines = ["store_id,item_id,wm_yr_wk,sell_price"]
    priced = [*DENSE_ITEMS, LATE_BLOOMER, *SMALL_DEPT_ITEMS]
    for week in range(11101, 11101 + (N_DAYS // 7) + 1):
        for item in priced:
            for store in STORES:
                price_lines.append(f"{store},{item},{week},2.00")

    (directory / "calendar.csv").write_text("\n".join(cal_lines) + "\n", encoding="utf-8")
    (directory / "sales_train_evaluation.csv").write_text(
        "\n".join(sales_lines) + "\n", encoding="utf-8"
    )
    (directory / "sell_prices.csv").write_text("\n".join(price_lines) + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def sampling_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return _write_sampling_extract(raw)


def sampled_items(db_path):
    return {r[0] for r in query(db_path, "SELECT DISTINCT item_id FROM fact_sales")}


def test_stratified_sample_hits_the_item_budget(sampling_dir, tmp_path):
    db_path = tmp_path / "strat.duckdb"
    report = build_warehouse(
        data_dir=sampling_dir,
        db_path=db_path,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=6, n_strata=3),
    )
    assert report.items == 6
    assert report.series == 12  # 6 items x 2 stores
    assert [s.items for s in report.strata] == [2, 2, 2]


def test_strata_are_ordered_by_intermittency(sampling_dir, tmp_path):
    # Budget set high enough to admit nearly the whole department, so this
    # asserts how items are *classified* rather than which two happened to be
    # drawn -- a 2-item draw per stratum is too noisy to assert a mean on.
    report = build_warehouse(
        data_dir=sampling_dir,
        db_path=tmp_path / "order.duckdb",
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=30, n_strata=3),
    )
    means = [s.mean_zero_share for s in report.strata]
    assert means == sorted(means), "stratum 1 must be densest, stratum 3 sparsest"
    assert means[0] < 0.55, "the dense stratum must actually contain dense series"
    assert means[-1] > 0.80, "the sparse stratum must actually contain intermittent series"
    # Bands must not overlap: every stratum-1 item is denser than every stratum-3 item.
    assert report.strata[0].max_zero_share <= report.strata[-1].min_zero_share


def test_sampling_is_deterministic_across_rebuilds(sampling_dir, tmp_path):
    scope = Scope(dept_ids=("FOODS_1",), items_per_dept=6)
    first = tmp_path / "a.duckdb"
    second = tmp_path / "b.duckdb"
    build_warehouse(data_dir=sampling_dir, db_path=first, scope=scope)
    build_warehouse(data_dir=sampling_dir, db_path=second, scope=scope)
    assert sampled_items(first) == sampled_items(second)


def test_seed_changes_the_sample(sampling_dir, tmp_path):
    a = tmp_path / "seed_a.duckdb"
    b = tmp_path / "seed_b.duckdb"
    build_warehouse(
        data_dir=sampling_dir,
        db_path=a,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=6, sampling_seed=1),
    )
    build_warehouse(
        data_dir=sampling_dir,
        db_path=b,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=6, sampling_seed=2),
    )
    assert sampled_items(a) != sampled_items(b)


def test_intermittency_is_measured_before_the_backtest_region(sampling_dir, tmp_path):
    """The stratifier must not see the evaluation period.

    LATE_BLOOMER is silent for the whole reference window and dense for the
    entire backtest region. It therefore belongs in the sparsest stratum; if it
    lands anywhere else, the stratification window has drifted forward into data
    the model is supposed to be evaluated on.
    """
    # Sanity-check the fixture itself: silent across the reference window ...
    unsampled = tmp_path / "leak_all.duckdb"
    build_warehouse(
        data_dir=sampling_dir,
        db_path=unsampled,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=None),
    )
    last = query(unsampled, "SELECT max(date) FROM fact_sales")[0][0]
    ref_end = last - timedelta(days=BACKTEST_DAYS)
    ref_start = last - timedelta(days=BACKTEST_DAYS + STRATIFY_WINDOW_DAYS - 1)
    con = duckdb.connect(str(unsampled), read_only=True)
    try:
        ref_zero, eval_zero = con.execute(
            """SELECT
                 avg(CASE WHEN units = 0 AND date BETWEEN ? AND ? THEN 1.0
                          WHEN date BETWEEN ? AND ? THEN 0.0 END),
                 avg(CASE WHEN units = 0 AND date > ? THEN 1.0
                          WHEN date > ? THEN 0.0 END)
               FROM fact_sales WHERE item_id = ?""",
            [ref_start, ref_end, ref_start, ref_end, ref_end, ref_end, LATE_BLOOMER],
        ).fetchone()
    finally:
        con.close()
    assert ref_zero == 1.0, "fixture: late bloomer must be silent in the reference window"
    assert eval_zero == 0.0, "fixture: late bloomer must be dense in the backtest region"

    # ... therefore the sampler must classify it as maximally sparse.
    sampled = tmp_path / "leak_sampled.duckdb"
    report = build_warehouse(
        data_dir=sampling_dir,
        db_path=sampled,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=31, n_strata=3),
    )
    assert LATE_BLOOMER in sampled_items(sampled)
    sparsest = report.strata[-1]
    assert sparsest.max_zero_share == 1.0
    denser_strata_max = max(s.max_zero_share for s in report.strata[:-1])
    assert denser_strata_max < 1.0, (
        "a series silent for the whole reference window must not be graded as dense; "
        "if it is, the stratification window has drifted into the backtest region"
    )


def test_never_listed_items_are_excluded(sampling_dir, tmp_path):
    db_path = tmp_path / "unlisted.duckdb"
    build_warehouse(
        data_dir=sampling_dir,
        db_path=db_path,
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=30),
    )
    assert UNLISTED not in sampled_items(db_path), (
        "an item with no price history was never on shelf; its zeros are not demand signal"
    )


def test_small_department_returns_everything_it_has(sampling_dir, tmp_path):
    db_path = tmp_path / "small.duckdb"
    report = build_warehouse(
        data_dir=sampling_dir,
        db_path=db_path,
        scope=Scope(dept_ids=("FOODS_2",), items_per_dept=6, n_strata=3),
    )
    assert report.items == len(SMALL_DEPT_ITEMS)


def test_strata_markdown_renders(sampling_dir, tmp_path):
    report = build_warehouse(
        data_dir=sampling_dir,
        db_path=tmp_path / "md.duckdb",
        scope=Scope(dept_ids=("FOODS_1",), items_per_dept=6),
    )
    table = report.strata_markdown()
    assert table.startswith("| Dept |")
    assert "dense" in table and "sparse" in table
