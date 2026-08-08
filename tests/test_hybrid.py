"""Stratum-aware routing: ETS on sparse, LightGBM on dense and mid."""

from __future__ import annotations

import duckdb
import pytest

from inventory_engine.data.schema import DIM_ITEM_STRATUM
from inventory_engine.models.hybrid import ROUTING, stratum_routing


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    connection.execute(f"""
        CREATE TABLE {DIM_ITEM_STRATUM} (
            item_id VARCHAR, dept_id VARCHAR, zero_share DOUBLE,
            stratum INTEGER, stratum_name VARCHAR
        )
    """)
    connection.execute(f"""
        INSERT INTO {DIM_ITEM_STRATUM} VALUES
            ('A', 'FOODS_1', 0.10, 1, 'dense'),
            ('B', 'FOODS_1', 0.50, 2, 'mid'),
            ('C', 'FOODS_1', 0.90, 3, 'sparse')
    """)
    yield connection
    connection.close()


def test_sparse_routes_to_ets(con):
    """The backtest finding this whole module exists to act on."""
    routing = stratum_routing(con).set_index("item_id")
    assert routing.loc["C", "source_model"] == "ets"


def test_dense_and_mid_route_to_lightgbm(con):
    routing = stratum_routing(con).set_index("item_id")
    assert routing.loc["A", "source_model"] == "lgbm"
    assert routing.loc["B", "source_model"] == "lgbm"


def test_every_item_is_routed(con):
    routing = stratum_routing(con)
    assert len(routing) == 3
    assert routing["source_model"].notna().all()


def test_unknown_stratum_fails_loudly(con):
    """Silently dropping an unrouted stratum would shrink the forecast without warning."""
    con.execute(f"INSERT INTO {DIM_ITEM_STRATUM} VALUES ('D', 'FOODS_1', 0.99, 4, 'glacial')")
    with pytest.raises(ValueError, match="no routing rule"):
        stratum_routing(con)


def test_routing_covers_the_three_sampled_strata():
    """Phase 1 samples exactly three bands; each needs a rule."""
    assert set(ROUTING) == {"dense", "mid", "sparse"}


def test_routing_is_not_a_single_global_model():
    """If every stratum routed to one model the hybrid would be pointless."""
    assert len(set(ROUTING.values())) > 1
