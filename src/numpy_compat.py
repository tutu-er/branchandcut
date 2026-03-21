"""Compatibility helpers for older libraries on modern NumPy."""

from __future__ import annotations

import numpy as np


def ensure_numpy_compat_for_pypower() -> None:
    """Restore NumPy symbols that legacy PYPOWER still imports."""
    if not hasattr(np, "in1d"):
        def _in1d(ar1, ar2, assume_unique=False, invert=False):
            return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)

        np.in1d = _in1d  # type: ignore[attr-defined]


ensure_numpy_compat_for_pypower()
