"""CLI: run the nightly precompute job once (E8-S4)."""

from __future__ import annotations

import sys

from inventory_engine.api.precompute import nightly_refresh


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``run-nightly-refresh``."""
    result = nightly_refresh()
    print(result.render())
    if not result.swapped:
        print("live warehouse untouched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
