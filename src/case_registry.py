from __future__ import annotations

from pathlib import Path

import numpy as np

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower.case14
import pypower.case39
from pypower.idx_bus import PD

from src.case3_uc_data import get_case3_uc_ppc, get_case3lite_uc_ppc
from src.case30_uc_data import get_case30_uc_ppc
from src.mti118_data_loader import load_case118_ppc_with_mti_limits


CASE3_LOAD_SCALE = 1.40
CASE3LITE_LOAD_SCALE = 1.28
CASE30_LOAD_SCALE = 1.15
LOAD_PROFILE_PATH = Path(__file__).resolve().parent / "load.csv"


def _load_system_profile() -> np.ndarray:
    load_profile_raw = np.loadtxt(LOAD_PROFILE_PATH, delimiter=",", dtype=float)
    if load_profile_raw.ndim == 1:
        return load_profile_raw
    return np.sum(load_profile_raw, axis=0)


def _resample_profile(system_profile: np.ndarray, horizon: int) -> np.ndarray:
    if system_profile.size == horizon:
        return system_profile
    if system_profile.size % horizon == 0:
        group_size = system_profile.size // horizon
        return system_profile.reshape(horizon, group_size).sum(axis=1)
    src_grid = np.linspace(0.0, 1.0, system_profile.size)
    dst_grid = np.linspace(0.0, 1.0, horizon)
    return np.interp(dst_grid, src_grid, system_profile)


def build_scaled_base_load(ppc: dict, horizon: int, scale: float = 1.0) -> np.ndarray:
    base_bus_load = np.asarray(ppc["bus"][:, PD], dtype=float)
    system_profile = _load_system_profile()
    horizon_profile = _resample_profile(system_profile, horizon)
    normalized_profile = horizon_profile / max(np.max(horizon_profile), 1e-9)
    return base_bus_load[:, None] * normalized_profile[None, :] * scale


def build_case3_base_load(horizon: int, scale: float = CASE3_LOAD_SCALE) -> tuple[dict, np.ndarray]:
    ppc = get_case3_uc_ppc()
    return ppc, build_scaled_base_load(ppc, horizon, scale=scale)


def build_case3lite_base_load(horizon: int, scale: float = CASE3LITE_LOAD_SCALE) -> tuple[dict, np.ndarray]:
    ppc = get_case3lite_uc_ppc()
    return ppc, build_scaled_base_load(ppc, horizon, scale=scale)


def build_case30_base_load(horizon: int, scale: float = CASE30_LOAD_SCALE) -> tuple[dict, np.ndarray]:
    ppc = get_case30_uc_ppc()
    return ppc, build_scaled_base_load(ppc, horizon, scale=scale)


def get_case_ppc(case_name: str) -> dict:
    ppc_map = {
        "case3": get_case3_uc_ppc,
        "case3lite": get_case3lite_uc_ppc,
        "case14": pypower.case14.case14,
        "case30": get_case30_uc_ppc,
        "case39": pypower.case39.case39,
        "case118": load_case118_ppc_with_mti_limits,
    }
    if case_name not in ppc_map:
        raise ValueError(f"Unsupported case_name: {case_name}")
    return ppc_map[case_name]()
