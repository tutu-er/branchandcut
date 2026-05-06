"""Utilities for UC time-coupled generator parameters."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pypower.idx_gen import GEN_BUS, PMAX, PMIN


def get_custom_generator_array(ppc_raw: dict, ng: int, key: str):
    values = ppc_raw.get(key) if isinstance(ppc_raw, dict) else None
    if values is None:
        return None
    values = np.asarray(values)
    if values.shape[0] != ng:
        return None

    raw_gen = np.asarray(ppc_raw.get("gen", []))
    if raw_gen.shape[0] != ng:
        return values

    order = np.argsort(raw_gen[:, GEN_BUS], kind="stable")
    return values[order]


def get_min_up_down_steps_from_ppc(
    ppc_raw: dict,
    ng: int,
    horizon: int,
    T_delta: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    max_window = max(int(horizon) - 1, 0)
    default_steps = min(max(int(4 * T_delta), 1), max_window)
    min_up_h = get_custom_generator_array(ppc_raw, ng, "uc_min_up_time_h")
    min_down_h = get_custom_generator_array(ppc_raw, ng, "uc_min_down_time_h")

    if min_up_h is None or min_down_h is None:
        min_up = np.full(ng, default_steps, dtype=int)
        min_down = np.full(ng, default_steps, dtype=int)
    else:
        min_up = np.maximum(np.ceil(np.asarray(min_up_h, dtype=float) / T_delta).astype(int), 1)
        min_down = np.maximum(np.ceil(np.asarray(min_down_h, dtype=float) / T_delta).astype(int), 1)
        if max_window > 0:
            min_up = np.minimum(min_up, max_window)
            min_down = np.minimum(min_down, max_window)
        else:
            min_up[:] = 0
            min_down[:] = 0

    Ton = int(np.max(min_up)) if min_up.size else 0
    Toff = int(np.max(min_down)) if min_down.size else 0
    return min_up.astype(int), min_down.astype(int), Ton, Toff


def get_ramp_limits_from_ppc(
    ppc_raw: dict,
    gen: np.ndarray,
    T_delta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ng = int(gen.shape[0])
    default_up = 0.4 * gen[:, PMAX] / T_delta
    default_down = 0.4 * gen[:, PMAX] / T_delta
    default_up_co = 0.3 * gen[:, PMAX]
    default_down_co = 0.3 * gen[:, PMAX]

    ramp_up_h = get_custom_generator_array(ppc_raw, ng, "uc_ramp_up_mw_per_h")
    ramp_down_h = get_custom_generator_array(ppc_raw, ng, "uc_ramp_down_mw_per_h")
    if ramp_up_h is None or ramp_down_h is None:
        return default_up, default_down, default_up_co, default_down_co

    Ru = np.maximum(np.asarray(ramp_up_h, dtype=float) * T_delta, default_up)
    Rd = np.maximum(np.asarray(ramp_down_h, dtype=float) * T_delta, default_down)
    Ru_co = np.maximum(Ru, gen[:, PMIN])
    Rd_co = np.maximum(Rd, gen[:, PMIN])
    return Ru, Rd, Ru_co, Rd_co

