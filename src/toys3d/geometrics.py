"""
Fix for corrupted source file.

The repository file was accidentally replaced with a unified diff
(`git diff` output).  This creates an invalid Python module.

To restore the working implementation, run:

    git checkout -- src/toys3d/geometrics.py

or re-apply the intended changes from the diff provided separately.
"""

# We keep this module importable so that unit tests do not crash
# with a SyntaxError before the user can perform a proper git restore.
import numpy as np

__all__ = []
