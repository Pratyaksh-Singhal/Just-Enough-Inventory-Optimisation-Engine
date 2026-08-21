"""Run the built dashboard's own JavaScript and check what it computes.

Every other check on this page reads it as text. A page can parse, carry every element id
its code queries, and still compute the wrong number -- which is exactly what happened: a
CSV cell reading ``N/A`` became a genuine zero-sales day, and dates in ``DD/MM/YYYY`` were
ordered as text, so every rolling window was summed over a shuffled history. Neither is
visible to any check that does not execute the code.

Node is not a dependency of this project, so these skip when it is absent -- the same
arrangement ``scripts/build_dashboard.py`` uses for its syntax check and
``tests/test_service_postgres.py`` uses for a database. CI runs on Linux with node
available, so they run there.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "dashboard" / "index.html"
DRIVER = ROOT / "tests" / "js" / "page_checks.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed; the page's JS cannot be run"
)


@pytest.fixture(scope="module")
def ran() -> dict:
    """Execute the page's functions once and share the results."""
    if not PAGE.is_file():
        pytest.skip(f"no built page at {PAGE}; run scripts/build_dashboard.py")
    done = subprocess.run(
        [shutil.which("node"), str(DRIVER), str(PAGE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"page_checks.js failed:\n{done.stderr}"
    return json.loads(done.stdout)


# --------------------------------------------------------------------- reading a file


def test_the_page_still_has_its_three_script_blocks(ran):
    """A dropped block is silent: the page renders and half the controls do nothing."""
    assert ran["blocks"] == 3


def test_an_unreadable_units_cell_is_skipped_not_read_as_zero(ran):
    """``N/A`` is a missing figure. Reading it as a sale of none invents demand data.

    ``Number('')`` is ``0`` and finite, so stripping the non-digits out of ``N/A`` used to
    leave a row that passed every check as a real zero-sales day -- pulling the demand
    quantile, and therefore the order quantity, down, while ``skipped`` stayed at zero so
    the warning that exists to report it never fired.
    """
    got = ran["unreadable_units_are_skipped"]
    assert got["skipped"] == 1
    assert got["rows"] == 1
    assert got["first_units"] == 5


def test_a_row_with_fewer_columns_than_its_header_is_skipped(ran):
    assert ran["ragged_row_is_skipped"] == {"rows": 1, "skipped": 1}


def test_an_empty_units_cell_is_skipped(ran):
    assert ran["blank_cell_is_skipped"] == {"rows": 1, "skipped": 1}


def test_a_written_zero_is_still_a_real_day_of_no_sales(ran):
    """The other half of the rule: a typed ``0`` is data and must survive."""
    got = ran["zero_is_a_real_sale_of_none"]
    assert got == {"rows": 2, "skipped": 0, "first_units": 0}


# --------------------------------------------------------------------- column mapping


def test_a_date_column_is_not_claimed_by_the_units_matcher(ran):
    """``sales_date`` contains "sales", and the units matcher used to reach it first.

    The file then read every unit figure out of a date string, every row failed to parse,
    and the page reported "check the sales column contains numbers" about a file with a
    perfectly good ``units_sold`` column.
    """
    row = ran["columns"]["sales_date_is_a_date"]
    assert "error" not in row
    assert row["date"] == "2026-01-01"
    assert row["units"] == 5


def test_a_store_column_is_not_claimed_by_the_product_matcher(ran):
    """``store_name`` contains "name"; every product collapsed into one series."""
    row = ran["columns"]["store_name_is_a_store"]
    assert row["sku"] == "Milk"
    assert row["store"] == "S1"


def test_unit_price_is_not_claimed_by_the_units_matcher(ran):
    row = ran["columns"]["unit_price_beats_units"]
    assert row["units"] == 5
    assert row["price"] == 2.5


def test_the_documented_aliases_all_resolve(ran):
    row = ran["columns"]["aliases"]
    assert row == {"sku": "A", "date": "2026-01-01", "store": "S1", "units": 5, "price": 2.5}


@pytest.mark.parametrize(
    "case,needle",
    [
        ("missing_units_column_is_named", "sales column"),
        ("missing_sku_column_is_named", "product column"),
        ("header_only_is_refused", "Not enough rows"),
    ],
)
def test_a_file_that_cannot_be_read_says_which_column_is_missing(ran, case, needle):
    """The refusal has to name the fix, not just decline."""
    assert needle in ran[case]["error"]


# --------------------------------------------------------------------- dates


@pytest.mark.parametrize(
    "text,key",
    [
        ("2026-01-05", 20260105),
        ("5 Jan 2026", 20260105),
        ("13/01/2026", 20260113),
        ("05/01/2026", 20260105),
        ("2026/1/5", 20260105),
    ],
)
def test_every_recognised_date_format_lands_on_one_scale(ran, text, key):
    """All branches return YYYYMMDD.

    Mixing scales would sort every date read by one branch ahead of every date read by
    another, which is a worse failure than not parsing at all.
    """
    assert ran["dates"][text] == key


@pytest.mark.parametrize("text", ["week 3", "banana", "", "period 12"])
def test_a_period_label_is_not_mistaken_for_a_date(ran, text):
    """``Date.parse('week 3')`` returns a real timestamp; the file's own order is safer."""
    assert ran["unparseable_dates"][text] is None


def test_day_first_dates_sort_chronologically(ran):
    """Sorted as text, ``1/5`` ``10/1`` ``2/2`` stays in written order and every rolling
    window is then computed over a shuffled history."""
    assert ran["sorted_ddmmyyyy"] == ["10/1/2026", "2/2/2026", "1/5/2026"]


# --------------------------------------------------------------------- degenerate maths


def test_a_history_shorter_than_the_horizon_yields_no_windows(ran):
    """Not an error and not a partial window -- there is simply nothing to average."""
    assert ran["window_totals_shorter_than_horizon"] == 0


def test_a_history_exactly_the_horizon_yields_one_window(ran):
    assert ran["window_totals_exactly_horizon"] == 1


def test_quantiles_of_degenerate_samples_stay_finite(ran):
    """An empty, single-valued or constant sample must not produce NaN in the order list."""
    assert ran["quantile_of_empty"] == 0
    assert ran["quantile_of_one"] == 4
    assert ran["quantile_of_identical"] == 3
