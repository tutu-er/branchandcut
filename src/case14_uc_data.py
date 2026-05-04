from __future__ import annotations

import copy

import numpy as np

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower.case14
from pypower.idx_brch import RATE_A, RATE_B, RATE_C
from pypower.idx_gen import PMAX, PMIN


# UC-oriented cost data for the 5 thermal units in standard case14.
# Columns follow MATPOWER / PYPOWER gencost:
# [model, startup, shutdown, ncost, c2, c1, c0]
CASE14_UC_GENCOST = np.array(
    [
        [2, 180.0, 90.0, 3, 0.0100, 1.70, 22.0],
        [2, 130.0, 65.0, 3, 0.0180, 2.05, 14.0],
        [2, 95.0, 45.0, 3, 0.0300, 2.65, 8.0],
        [2, 70.0, 35.0, 3, 0.0420, 3.20, 5.0],
        [2, 55.0, 25.0, 3, 0.0550, 3.80, 3.0],
    ],
    dtype=float,
)


def _ensure_branch_limits(ppc: dict, default_limit: float = 95.0) -> None:
    branch = np.asarray(ppc["branch"], dtype=float)
    for col in (RATE_A, RATE_B, RATE_C):
        mask = branch[:, col] <= 0.0
        branch[mask, col] = float(default_limit)
    # Keep several corridors mildly congestible without making the case brittle.
    branch[: min(4, branch.shape[0]), RATE_A] = np.minimum(
        branch[: min(4, branch.shape[0]), RATE_A],
        np.asarray([80.0, 80.0, 75.0, 70.0], dtype=float)[: min(4, branch.shape[0])],
    )
    branch[:, RATE_B] = np.maximum(branch[:, RATE_B], branch[:, RATE_A])
    branch[:, RATE_C] = np.maximum(branch[:, RATE_C], branch[:, RATE_A])
    ppc["branch"] = branch


def apply_case14_uc_parameters(ppc: dict) -> dict:
    """Return a UC-ready case14 copy with nonzero UC costs and light metadata."""
    ppc_uc = copy.deepcopy(ppc)
    gen = np.asarray(ppc_uc["gen"], dtype=float)
    if gen.shape[0] != CASE14_UC_GENCOST.shape[0]:
        raise ValueError(
            f"Unexpected case14 generator count {gen.shape[0]}, "
            f"expected {CASE14_UC_GENCOST.shape[0]}"
        )

    # Standard case14 has several zero PMIN units.  A small PMIN helps UC expose
    # meaningful fixed-cost and minimum-output tradeoffs while preserving feasibility.
    gen[:, PMIN] = np.maximum(gen[:, PMIN], np.asarray([25.0, 12.0, 8.0, 8.0, 6.0]))
    gen[:, PMAX] = np.maximum(gen[:, PMAX], gen[:, PMIN] + 10.0)
    ppc_uc["gen"] = gen
    ppc_uc["gencost"] = CASE14_UC_GENCOST.copy()

    ppc_uc["uc_min_up_time_h"] = np.asarray([4.0, 3.0, 2.0, 2.0, 1.0], dtype=float)
    ppc_uc["uc_min_down_time_h"] = np.asarray([4.0, 3.0, 2.0, 2.0, 1.0], dtype=float)
    ppc_uc["uc_ramp_up_mw_per_h"] = np.maximum(0.45 * gen[:, PMAX], gen[:, PMIN])
    ppc_uc["uc_ramp_down_mw_per_h"] = np.maximum(0.45 * gen[:, PMAX], gen[:, PMIN])
    _ensure_branch_limits(ppc_uc)
    return ppc_uc


def get_case14_uc_ppc() -> dict:
    """Load standard PYPOWER case14 and apply UC-oriented parameters."""
    return apply_case14_uc_parameters(pypower.case14.case14())
