"""Request handlers.

Nothing in this package may import a model-fitting library. ``tests/test_service_layering``
asserts it by AST scan over every module here, so the rule survives a new file being added
by someone who never read this docstring.
"""
