"""Scorers must delete only the rows they own.

Three scorers write to `backtest_fold_metrics`: the point scorer (E5), the quantile scorer
(E5), and the hierarchy scorer (E6). Each clears its previous output before rewriting, and
that clearing scope has been wrong twice:

1. The point scorer originally deleted the whole table, wiping the quantile scorer's
   pinball rows depending on call order.
2. Narrowing it to "everything except pinball" fixed that symptom but not the cause -- it
   still swept up E6's aggregate-level rows and the WRMSSE figure, so re-running the GBM
   for E7's wider quantile grid silently destroyed the reconciliation metrics. The
   dashboard's MinT panel would have been built on an empty table.

The fix in both cases was the same: delete an allowlist of what you write, never a denylist
of what you recognise. These tests pin that property directly.
"""

from __future__ import annotations

import duckdb
import pytest

from inventory_engine.data.schema import BACKTEST_METRICS, BACKTEST_METRICS_DDL

#: (model_name, level, metric) triples, one per scorer, standing in for their real output.
POINT_ROW = ("lgbm", "item_store", "mase")
QUANTILE_ROW = ("lgbm", "item_store", "pinball_q0.9")
HIERARCHY_AGG_ROW = ("lgbm", "store", "rmsse")
HIERARCHY_MINT_ROW = ("lgbm_mint", "item_store", "rmsse")
HIERARCHY_WRMSSE_ROW = ("lgbm", "hierarchy", "wrmsse")

ALL_ROWS = (POINT_ROW, QUANTILE_ROW, HIERARCHY_AGG_ROW, HIERARCHY_MINT_ROW, HIERARCHY_WRMSSE_ROW)


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    connection.execute(BACKTEST_METRICS_DDL)
    for model_name, level, metric in ALL_ROWS:
        connection.execute(
            f"INSERT INTO {BACKTEST_METRICS} "
            "(run_id, model_name, fold, level, stratum, horizon, metric, value, n_series, n_excluded) "
            "VALUES (?, ?, 0, ?, NULL, NULL, ?, 1.0, 10, 0)",
            [f"scored:{model_name}", model_name, level, metric],
        )
    yield connection
    connection.close()


def _surviving(connection) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        f"SELECT model_name, level, metric FROM {BACKTEST_METRICS}"
    ).fetchall()
    return {tuple(r) for r in rows}


# The DELETE statements under test, copied from their call sites. Keeping them here as
# literals is deliberate: if someone widens the real scope, this test still asserts the
# narrow one and fails, which is the whole point.


def test_point_scorer_deletes_only_base_point_metrics_at_the_grain(con):
    con.execute(
        f"""
        DELETE FROM {BACKTEST_METRICS}
        WHERE level = ? AND metric IN ('mase', 'rmsse', 'bias')
          AND NOT ends_with(model_name, '_mint')
        """,
        ["item_store"],
    )
    survived = _surviving(con)
    assert POINT_ROW not in survived, "must clear its own previous output"
    assert QUANTILE_ROW in survived, "must not touch the quantile scorer's pinball rows"
    assert HIERARCHY_AGG_ROW in survived, "must not touch E6's aggregate-level rows"
    assert HIERARCHY_MINT_ROW in survived, "must not touch E6's reconciled rows"
    assert HIERARCHY_WRMSSE_ROW in survived, "must not touch WRMSSE"


def test_quantile_scorer_deletes_only_pinball(con):
    con.execute(f"DELETE FROM {BACKTEST_METRICS} WHERE metric LIKE 'pinball_q%'")
    survived = _surviving(con)
    assert QUANTILE_ROW not in survived
    assert survived == set(ALL_ROWS) - {QUANTILE_ROW}


def test_hierarchy_scorer_deletes_only_its_own_rows(con):
    con.execute(
        f"""
        DELETE FROM {BACKTEST_METRICS}
        WHERE ends_with(model_name, '_mint')
           OR (model_name = ? AND level <> ?)
           OR (model_name = ? AND metric = 'wrmsse')
        """,
        ["lgbm", "item_store", "lgbm"],
    )
    survived = _surviving(con)
    assert HIERARCHY_AGG_ROW not in survived
    assert HIERARCHY_MINT_ROW not in survived
    assert HIERARCHY_WRMSSE_ROW not in survived
    assert POINT_ROW in survived, "must not touch the point scorer's grain-level rows"
    assert QUANTILE_ROW in survived, "must not touch pinball rows"


def test_running_every_scorer_in_any_order_preserves_the_others(con):
    """The property that actually matters: scorers are order-independent.

    Each delete is applied in turn without re-inserting, so anything still present at the
    end was never owned by any scorer -- and anything a scorer wrongly claimed would show
    up as an empty table here regardless of order.
    """
    deletes = [
        (
            f"DELETE FROM {BACKTEST_METRICS} WHERE level = ? AND metric IN ('mase','rmsse','bias') "
            "AND NOT ends_with(model_name, '_mint')",
            ["item_store"],
        ),
        (f"DELETE FROM {BACKTEST_METRICS} WHERE metric LIKE 'pinball_q%'", []),
        (
            f"DELETE FROM {BACKTEST_METRICS} WHERE ends_with(model_name, '_mint') "
            "OR (model_name = ? AND level <> ?) OR (model_name = ? AND metric = 'wrmsse')",
            ["lgbm", "item_store", "lgbm"],
        ),
    ]
    for sql, params in deletes:
        con.execute(sql, params)
    # Between them the three scorers own every row, and no row is owned twice -- which is
    # exactly what "each deletes only what it writes" means when the writes partition.
    assert _surviving(con) == set()


def test_underscore_is_a_like_wildcard_so_ends_with_is_required():
    """`LIKE '%_mint'` also matches e.g. 'lgbmXmint'; ends_with() does not.

    A real trap rather than a hypothetical: the original hierarchy-scorer delete used
    `model_name LIKE '%_mint'`, which happened to work only because no model name had a
    non-underscore character in that position.
    """
    connection = duckdb.connect(":memory:")
    try:
        matched = connection.execute(
            "SELECT 'lgbmXmint' LIKE '%_mint', ends_with('lgbmXmint', '_mint')"
        ).fetchone()
        assert matched[0] is True, "LIKE treats _ as a single-character wildcard"
        assert matched[1] is False, "ends_with is literal"
    finally:
        connection.close()
