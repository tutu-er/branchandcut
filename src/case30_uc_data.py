from __future__ import annotations

import copy

import numpy as np

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower.case30


# UC-oriented cost data for the 6 thermal units in standard case30.
# Columns follow MATPOWER / PYPOWER gencost:
# [model, startup, shutdown, ncost, c2, c1, c0]
CASE30_UC_GENCOST = np.array([
    [2, 120.0, 60.0, 3, 0.0200, 2.00, 8.0],
    [2, 110.0, 55.0, 3, 0.0175, 1.75, 7.0],
    [2, 85.0,  40.0, 3, 0.0625, 1.00, 5.0],
    [2, 150.0, 75.0, 3, 0.0085, 3.25, 9.0],
    [2, 70.0,  35.0, 3, 0.0250, 3.00, 4.0],
    [2, 90.0,  45.0, 3, 0.0300, 3.10, 4.5],
], dtype=float)


def apply_case30_uc_costs(ppc: dict) -> dict:
    """Return a copy of case30 ppc with UC-oriented nonzero startup/shutdown costs."""
    ppc_uc = copy.deepcopy(ppc)
    gencost = np.asarray(ppc_uc["gencost"], dtype=float)
    if gencost.shape != CASE30_UC_GENCOST.shape:
        raise ValueError(
            f"Unexpected case30 gencost shape {gencost.shape}, "
            f"expected {CASE30_UC_GENCOST.shape}"
        )
    ppc_uc["gencost"] = CASE30_UC_GENCOST.copy()
    return ppc_uc


def get_case30_uc_ppc() -> dict:
    """Load standard PYPOWER case30 and overwrite costs with UC-oriented values."""
    return apply_case30_uc_costs(pypower.case30.case30())
