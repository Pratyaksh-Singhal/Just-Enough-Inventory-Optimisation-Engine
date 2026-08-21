"""``run-service`` — start the tier 2 API."""

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
