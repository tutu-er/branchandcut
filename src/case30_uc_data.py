from __future__ import annotations

import copy

import numpy as np

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower.case30
from pypower.idx_brch import RATE_A, RATE_B, RATE_C
from pypower.idx_gen import PMAX, PMIN


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

# A lighter 4-unit version of case30.  It keeps the 30-bus network and load
# distribution, but removes two small peaking units so active-set sampling and
# surrogate training stay closer to the case3lite computational footprint.
CASE30LITE_KEEP_GENERATORS = np.array([0, 1, 2, 3], dtype=int)
CASE30LITE_UC_GENCOST = np.array([
    [2, 140.0, 70.0, 3, 0.0180, 1.85, 12.0],
    [2, 120.0, 60.0, 3, 0.0200, 2.05, 10.0],
    [2, 85.0,  40.0, 3, 0.0450, 2.70, 6.0],
    [2, 70.0,  35.0, 3, 0.0600, 3.30, 4.0],
], dtype=float)

CASE30LITE_PERTURBED_UNIT_ID = 0
CASE30LITE_PERTURBED_PMAX_SCALE = 0.85
CASE30LITE_PERTURBED_RAMP_SCALE = 0.75
CASE30LITE_PERTURBED_LINEAR_COST_SCALE = 1.18
CASE30LITE_PERTURBED_QUAD_COST_SCALE = 1.12
CASE30LITE_PERTURBED_STARTUP_SCALE = 1.10
CASE30LITE_PERTURBED_SHUTDOWN_SCALE = 1.10


def _ensure_branch_limits(ppc: dict, default_limit: float = 75.0) -> None:
    branch = np.asarray(ppc["branch"], dtype=float)
    for col in (RATE_A, RATE_B, RATE_C):
        mask = branch[:, col] <= 0.0
        branch[mask, col] = float(default_limit)
    # Leave a few corridors tight enough that DCPF terms can matter, but avoid
    # the standard case30 zero-rate degeneracy.
    if branch.shape[0] >= 6:
        branch[:6, RATE_A] = np.minimum(
            branch[:6, RATE_A],
            np.asarray([65.0, 65.0, 60.0, 60.0, 55.0, 55.0], dtype=float),
        )
    branch[:, RATE_B] = np.maximum(branch[:, RATE_B], branch[:, RATE_A])
    branch[:, RATE_C] = np.maximum(branch[:, RATE_C], branch[:, RATE_A])
    ppc["branch"] = branch


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
    _ensure_branch_limits(ppc_uc, default_limit=90.0)
    return ppc_uc


def get_case30_uc_ppc() -> dict:
    """Load standard PYPOWER case30 and overwrite costs with UC-oriented values."""
    return apply_case30_uc_costs(pypower.case30.case30())


def get_case30lite_uc_ppc() -> dict:
    """Return a compact UC-ready case30 variant with 4 thermal units."""
    ppc_uc = copy.deepcopy(pypower.case30.case30())
    keep = CASE30LITE_KEEP_GENERATORS
    gen = np.asarray(ppc_uc["gen"], dtype=float)[keep].copy()
    gen[:, PMIN] = np.maximum(gen[:, PMIN], np.asarray([18.0, 18.0, 8.0, 6.0]))
    gen[:, PMAX] = np.maximum(gen[:, PMAX], np.asarray([95.0, 90.0, 55.0, 45.0]))
    ppc_uc["gen"] = gen
    ppc_uc["gencost"] = CASE30LITE_UC_GENCOST.copy()
    ppc_uc["uc_min_up_time_h"] = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
    ppc_uc["uc_min_down_time_h"] = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
    ppc_uc["uc_ramp_up_mw_per_h"] = np.maximum(0.50 * gen[:, PMAX], gen[:, PMIN])
    ppc_uc["uc_ramp_down_mw_per_h"] = np.maximum(0.50 * gen[:, PMAX], gen[:, PMIN])
    _ensure_branch_limits(ppc_uc, default_limit=75.0)
    return ppc_uc


def get_case30lite_perturbed_uc_ppc(
    unit_id: int = CASE30LITE_PERTURBED_UNIT_ID,
    pmax_scale: float = CASE30LITE_PERTURBED_PMAX_SCALE,
    ramp_scale: float = CASE30LITE_PERTURBED_RAMP_SCALE,
    linear_cost_scale: float = CASE30LITE_PERTURBED_LINEAR_COST_SCALE,
    quad_cost_scale: float = CASE30LITE_PERTURBED_QUAD_COST_SCALE,
    startup_scale: float = CASE30LITE_PERTURBED_STARTUP_SCALE,
    shutdown_scale: float = CASE30LITE_PERTURBED_SHUTDOWN_SCALE,
) -> dict:
    """Return case30lite with one generator moderately perturbed.

    The default perturbation makes unit 0 slightly less capable and more
    expensive, which is enough to shift active sets without changing topology.
    """
    ppc_uc = get_case30lite_uc_ppc()
    g = int(unit_id)
    gen = np.asarray(ppc_uc["gen"], dtype=float).copy()
    gencost = np.asarray(ppc_uc["gencost"], dtype=float).copy()
    if g < 0 or g >= gen.shape[0]:
        raise ValueError(f"unit_id={g} out of range for case30lite with {gen.shape[0]} units")

    old_pmax = float(gen[g, PMAX])
    gen[g, PMAX] = max(float(gen[g, PMIN]) + 1.0, old_pmax * float(pmax_scale))
    ppc_uc["gen"] = gen

    gencost[g, 1] *= float(startup_scale)
    gencost[g, 2] *= float(shutdown_scale)
    gencost[g, 4] *= float(quad_cost_scale)
    gencost[g, 5] *= float(linear_cost_scale)
    ppc_uc["gencost"] = gencost

    for key in ("uc_ramp_up_mw_per_h", "uc_ramp_down_mw_per_h"):
        arr = np.asarray(ppc_uc[key], dtype=float).copy()
        arr[g] = max(float(gen[g, PMIN]), float(arr[g]) * float(ramp_scale))
        ppc_uc[key] = arr

    ppc_uc["case30lite_perturbation"] = {
        "base_case": "case30lite",
        "unit_id": g,
        "pmax_scale": float(pmax_scale),
        "ramp_scale": float(ramp_scale),
        "linear_cost_scale": float(linear_cost_scale),
        "quad_cost_scale": float(quad_cost_scale),
        "startup_scale": float(startup_scale),
        "shutdown_scale": float(shutdown_scale),
    }
    return ppc_uc
