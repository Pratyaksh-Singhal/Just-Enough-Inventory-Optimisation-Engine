"""E8-S5 — connection handling for DuckDB's single-writer model.

DuckDB allows any number of concurrent readers, but a writer needs **exclusive** access —
no other connection, reading or writing, may be open on the file at the same time. A long-
lived global connection held by the API process would therefore permanently block the
nightly precompute job (E8-S4) from ever acquiring a write handle.

The fix is to never hold one. Every request opens a fresh ``read_only=True`` connection,
uses it, and closes it before the response is returned. The connection's lifetime is
milliseconds, not the process lifetime, so the write-exclusion window the nightly job needs
is almost always available, and precompute.py's atomic swap (rather than an in-place write)
is what makes "almost always" acceptable — see precompute.py's docstring.

Opening a DuckDB connection is not free, but it is fast relative to a network round trip;
E8-S5 measures the actual overhead this design costs.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
from fastapi import HTTPException

from inventory_engine.config import WAREHOUSE_PATH


def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """FastAPI dependency: a short-lived read-only connection, closed after the request.

    Raises:
        HTTPException: 503 if the warehouse file is missing, or if DuckDB cannot open it
            because a writer currently holds it — a transient, retryable state during the
            nightly job's atomic swap, not a client error.

    """
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
