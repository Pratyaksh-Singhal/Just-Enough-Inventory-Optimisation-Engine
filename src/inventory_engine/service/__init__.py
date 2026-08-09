"""Tier 2 — the backend forecast service.

Separate from ``inventory_engine.api`` (tier 1), which serves precomputed M5 results out of
DuckDB and is not modified by anything in this package. Tier 2 accepts a user's own CSV,
stores it, and runs a real per-SKU forecast on a worker.

The split that matters
----------------------
``routers/`` enqueues. ``worker.py`` trains. Nothing in ``routers/`` may import a model
library, and ``tests/test_service_layering.py`` asserts it by AST scan rather than trusting
review -- the same discipline ``test_no_handler_imports_a_trainer`` established for tier 1,
widened from one file to the whole request-handling surface.
"""
