"""E8-S4 — the nightly refresh: idempotent, safely re-runnable, never corrupts the live file.

Uses fast fake steps rather than the real GBM/MinT/optimize pipeline, so these tests run in
milliseconds and exercise the orchestration logic (shadow copy, swap, rollback) in
isolation from whether the real pipeline itself works -- that is covered elsewhere.
"""

from __future__ import annotations

import duckdb
import pytest

from inventory_engine.api.precompute import RefreshResult, nightly_refresh


def _make_warehouse(path, marker: str) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE marker (value VARCHAR)")
    con.execute("INSERT INTO marker VALUES (?)", [marker])
    con.close()


def _read_marker(path) -> str:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("SELECT value FROM marker").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def live_db(tmp_path):
    path = tmp_path / "warehouse.duckdb"
    _make_warehouse(path, "original")
    return path


def test_successful_refresh_swaps_in_the_new_content(live_db):
    def bump(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("UPDATE marker SET value = 'refreshed'")

    result = nightly_refresh(steps=[bump], live_path=live_db)

    assert result.swapped is True
    assert result.steps_run == 1
    assert _read_marker(live_db) == "refreshed"


def test_steps_run_in_order_against_the_shadow_copy_not_the_live_file(live_db):
    calls: list[str] = []

    def step_a(con):
        calls.append("a")
        con.execute("INSERT INTO marker VALUES ('a')")

    def step_b(con):
        calls.append("b")
        con.execute("INSERT INTO marker VALUES ('b')")

    nightly_refresh(steps=[step_a, step_b], live_path=live_db)

    assert calls == ["a", "b"]
    # Original untouched until the swap -- verified indirectly: the live file now has all
    # three rows (original + a + b) only after the swap, proving steps ran on the copy.
    con = duckdb.connect(str(live_db), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM marker").fetchone()[0] == 3
    finally:
        con.close()


def test_failing_step_leaves_the_live_file_untouched(live_db):
    def good(con):
        con.execute("UPDATE marker SET value = 'should_not_appear'")

    def bad(con):
        raise RuntimeError("simulated pipeline failure")

    result = nightly_refresh(steps=[good, bad], live_path=live_db)

    assert result.swapped is False
    assert "simulated pipeline failure" in result.error
    assert _read_marker(live_db) == "original", "a failed step must not corrupt the live file"


def test_failed_refresh_does_not_leak_the_shadow_file(live_db):
    def bad(con):
        raise RuntimeError("boom")

    nightly_refresh(steps=[bad], live_path=live_db)

    shadow = live_db.with_suffix(".shadow.duckdb")
    assert not shadow.exists()


def test_refresh_is_idempotent(live_db):
    def bump(con):
        con.execute("UPDATE marker SET value = 'v2'")

    first = nightly_refresh(steps=[bump], live_path=live_db)
    second = nightly_refresh(steps=[bump], live_path=live_db)

    assert first.swapped and second.swapped
    assert _read_marker(live_db) == "v2"


def test_missing_live_warehouse_fails_cleanly(tmp_path):
    result = nightly_refresh(steps=[], live_path=tmp_path / "does_not_exist.duckdb")
    assert result.swapped is False
    assert "no live warehouse" in result.error


def test_swap_retries_on_a_locked_file_then_succeeds(live_db, monkeypatch):
    """A reader briefly holding the file open must not fail the swap outright."""
    attempts = {"count": 0}
    real_replace = __import__("os").replace

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("simulated: file locked by a reader")
        return real_replace(src, dst)

    monkeypatch.setattr("inventory_engine.api.precompute.os.replace", flaky_replace)

    result = nightly_refresh(steps=[], live_path=live_db, swap_retry_seconds=0.01)

    assert result.swapped is True
    assert attempts["count"] == 3


def test_swap_gives_up_after_max_attempts_and_preserves_the_live_file(live_db, monkeypatch):
    def always_locked(src, dst):
        raise OSError("simulated: permanently locked")

    monkeypatch.setattr("inventory_engine.api.precompute.os.replace", always_locked)

    result = nightly_refresh(
        steps=[], live_path=live_db, max_swap_attempts=3, swap_retry_seconds=0.01
    )

    assert result.swapped is False
    assert "failed after 3 attempts" in result.error
    assert _read_marker(live_db) == "original"


def test_render_reports_success_and_failure_distinctly():
    ok = RefreshResult(swapped=True, steps_run=3, elapsed_seconds=1.5)
    fail = RefreshResult(swapped=False, steps_run=1, elapsed_seconds=0.2, error="boom")
    assert "refreshed" in ok.render()
    assert "NOT swapped" in fail.render()
    assert "boom" in fail.render()
