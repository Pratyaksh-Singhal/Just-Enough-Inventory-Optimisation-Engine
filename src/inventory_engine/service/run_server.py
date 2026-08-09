"""``run-service`` — start the tier 2 API.

Separate entry point from ``run-api`` (tier 1). Both can run at once, and
:data:`DEFAULT_PORT` is what makes that true: tier 1's ``run-api`` defaults to 8000, so
tier 2 defaults to 8001. They collided until a ``run-api`` already listening on 8000
refused this process a socket -- the two tiers are meant to be independent, and sharing a
default port is the one way to make them not be.
"""

from __future__ import annotations

import argparse

import uvicorn

#: One past tier 1's, so `run-api` and `run-service` coexist without a flag.
DEFAULT_PORT = 8001


def main() -> int:
    """Start the tier 2 FastAPI app under uvicorn."""
    parser = argparse.ArgumentParser(description="Run the Full Forecast service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "inventory_engine.service.app:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
