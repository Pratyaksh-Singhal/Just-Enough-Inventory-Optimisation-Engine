"""Record the festival decision alongside each order quantity.

Two columns on ``forecast_results``:

* ``order_qty_before_festival`` -- what the newsvendor asked for, so both numbers are
  always available and the adjustment can be undone by eye.
* ``festival`` -- the whole decision as JSONB: which of the three states, the factor, and
  every match with the keyword and category behind it.

Both are additive and neither is back-filled. Existing rows predate the feature; copying
``order_qty`` into the new column would assert that nothing was adjusted, which is not
something those rows know. ``festival`` defaults to an empty object, which the response
layer reads as "no festival information", distinct from "no festival nearby".

Revision ID: 0002_festival_adjustment
Revises: 0001_initial_schema
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_festival_adjustment"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the festival columns."""
    op.add_column(
        "forecast_results",
        sa.Column("order_qty_before_festival", sa.Float(), nullable=True),
    )
    op.add_column(
        "forecast_results",
        sa.Column(
            "festival",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    """Drop them again. The order quantities themselves are untouched either way."""
    op.drop_column("forecast_results", "festival")
    op.drop_column("forecast_results", "order_qty_before_festival")
