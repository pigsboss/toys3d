# src/toys3d/geometrics.py
"""
Deprecated legacy module.

The geometrics implementation has moved into the package:

    src/toys3d/geometrics/
        __init__.py
        euclidean.py
        discrete.py
        topology.py
        utilities.py

This file was kept only as a temporary marker during migration.
It will be removed after regression tests pass.
"""

raise ImportError(
    "Do not import 'toys3d.geometrics' from the legacy geometrics.py file. "
    "Use the 'toys3d.geometrics' package instead."
)
