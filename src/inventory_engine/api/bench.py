"""E8-S5 — measure request latency against the real warehouse, for the README/E10."""

from __future__ import annotations

import argparse
import statistics
import time

import duckdb

from inventory_engine.config import WAREHOUSE_PATH


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def run_bench(n: int = 200) -> dict[str, dict[str, float]]:
    """Time ``/health``, ``/forecast`` and ``/optimize`` over ``n`` requests each."""
    from fastapi.testclient import TestClient

    from inventory_engine.api.app import app

    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    sku, store = con.execute(
        "SELECT item_id, store_id FROM forecast "
        "WHERE model_name = 'lgbm' AND level = 'item_store' AND reconciled = FALSE LIMIT 1"
    ).fetchone()
    con.close()

    results: dict[str, dict[str, float]] = {}
    with TestClient(app) as client:
        calls = {
            "GET /health": lambda: client.get("/health"),
            "POST /forecast": lambda: client.post(
                "/forecast", json={"sku": sku, "store": store, "horizon": 7}
            ),
            "POST /optimize": lambda: client.post(
                "/optimize", json={"sku": sku, "cu": 3.0, "co": 4.0}
            ),
        }
        for name, call in calls.items():
            call()  # warm-up: first DuckDB connection pays a one-time cost
            timings = []
            for _ in range(n):
                started = time.perf_counter()
                response = call()
                timings.append((time.perf_counter() - started) * 1000)
                assert response.status_code == 200, response.text
            results[name] = {
                "p50_ms": statistics.median(timings),
                "p95_ms": _percentile(timings, 0.95),
                "p99_ms": _percentile(timings, 0.99),
                "max_ms": max(timings),
            }
    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-api-bench``."""
    parser = argparse.ArgumentParser(description="Measure API latency (E8-S5).")
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args(argv)

    results = run_bench(args.n)
    print(f"\n{args.n} requests per endpoint, in-process (no network hop):\n")
    for name, stats in results.items():
        print(
            f"  {name:<16} p50 {stats['p50_ms']:6.2f}ms  p95 {stats['p95_ms']:6.2f}ms"
            f"  p99 {stats['p99_ms']:6.2f}ms  max {stats['max_ms']:6.2f}ms"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
