"""CLI: run the API with uvicorn."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-api``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Inventory Optimization Engine API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    uvicorn.run("inventory_engine.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
