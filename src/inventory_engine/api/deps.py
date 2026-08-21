"""E8-S5 — connection handling for DuckDB's single-writer model."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
from fastapi import HTTPException

from inventory_engine.config import WAREHOUSE_PATH


def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """FastAPI dependency: a short-lived read-only connection, closed after the request."""
    if not WAREHOUSE_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"warehouse not found at {WAREHOUSE_PATH}; nothing precomputed yet",
        )
    try:
        con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    except duckdb.IOException as exc:
        raise HTTPException(
            status_code=503,
            detail="warehouse is being refreshed; retry in a moment",
        ) from exc
    try:
        yield con
    finally:
        con.close()
