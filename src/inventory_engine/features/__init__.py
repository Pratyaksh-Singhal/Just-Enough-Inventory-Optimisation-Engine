"""Leakage-free feature engineering over the DuckDB panel."""

from inventory_engine.features.build import (
    FEATURE_PANEL,
    KNOWN_IN_ADVANCE,
    UNITS_DERIVED,
    FeatureReport,
    build_features,
    feature_columns,
)

__all__ = [
    "FEATURE_PANEL",
    "KNOWN_IN_ADVANCE",
    "UNITS_DERIVED",
    "FeatureReport",
    "build_features",
    "feature_columns",
]
