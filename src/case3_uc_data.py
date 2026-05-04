from __future__ import annotations

import copy

import numpy as np


def get_case3_uc_ppc() -> dict:
    """Return a compact 3-bus / 3-generator UC-ready test case.

    The coefficients are chosen so the 24h load profile can trigger
    multiple commitment regimes instead of collapsing to a single pattern.
    Compared with the initial toy setting, this version is intentionally
    more aggressive: higher peak load, tighter line limits, and a stronger
    economic split between the flexible unit, baseload unit, and peaker.
    """
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": np.array(
            [
                [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
                [2, 2, 78.0, 26.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
                [3, 2, 72.0, 24.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
            ],
            dtype=float,
        ),
        "gen": np.array(
            [
                [1, 72.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 130.0, 60.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 52.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 100.0, 15.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 20.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 55.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "branch": np.array(
            [
                [1, 2, 0.0100, 0.1200, 0.020, 85.0, 85.0, 85.0, 0.0, 0.0, 1, -360.0, 360.0],
                [1, 3, 0.0125, 0.1500, 0.025, 60.0, 60.0, 60.0, 0.0, 0.0, 1, -360.0, 360.0],
                [2, 3, 0.0075, 0.1000, 0.015, 45.0, 45.0, 45.0, 0.0, 0.0, 1, -360.0, 360.0],
            ],
            dtype=float,
        ),
        "gencost": np.array(
            [
                [2, 150.0, 75.0, 3, 0.0100, 1.70, 130.0],
                [2, 30.0, 15.0, 3, 0.0140, 2.30, 6.0],
                [2, 12.0, 6.0, 3, 0.0550, 5.00, 1.5],
            ],
            dtype=float,
        ),
        "uc_min_up_time_h": np.array([4.0, 2.0, 1.0], dtype=float),
        "uc_min_down_time_h": np.array([4.0, 2.0, 1.0], dtype=float),
    }


def get_case3lite_uc_ppc() -> dict:
    """Return a milder 3-bus / 3-generator UC case that still yields mixed commitment regimes.

    Compared with ``case3``, this variant eases the network and capacity pressure:
    lower effective peak load, slightly looser line limits, and a less punitive peaker.
    The goal is to keep multiple commitment patterns while avoiding the most aggressive
    congestion/capacity behavior of the current ``case3`` setting.
    """
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": np.array(
            [
                [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
                [2, 2, 84.0, 28.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
                [3, 2, 66.0, 22.0, 0.0, 0.0, 1, 1.00, 0.0, 230.0, 1, 1.05, 0.95],
            ],
            dtype=float,
        ),
        "gen": np.array(
            [
                [1, 60.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 135.0, 58.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 40.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 102.0, 15.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 15.0, 0.0, 100.0, -100.0, 1.00, 100.0, 1, 50.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "branch": np.array(
            [
                [1, 2, 0.0100, 0.1200, 0.020, 100.0, 100.0, 100.0, 0.0, 0.0, 1, -360.0, 360.0],
                [1, 3, 0.0125, 0.1500, 0.025, 70.0, 70.0, 70.0, 0.0, 0.0, 1, -360.0, 360.0],
                [2, 3, 0.0075, 0.1000, 0.015, 50.0, 50.0, 50.0, 0.0, 0.0, 1, -360.0, 360.0],
            ],
            dtype=float,
        ),
        "gencost": np.array(
            [
                [2, 120.0, 60.0, 3, 0.0110, 1.90, 40.0],
                [2, 28.0, 14.0, 3, 0.0140, 2.35, 6.0],
                [2, 10.0, 5.0, 3, 0.0400, 4.20, 2.0],
            ],
            dtype=float,
        ),
        "uc_min_up_time_h": np.array([3.0, 2.0, 1.0], dtype=float),
        "uc_min_down_time_h": np.array([3.0, 2.0, 1.0], dtype=float),
    }


def apply_case3_uc_costs(ppc: dict) -> dict:
    """Overwrite gencost with the canonical case3 UC cost table."""
    ppc_uc = copy.deepcopy(ppc)
    ppc_uc["gencost"] = get_case3_uc_ppc()["gencost"].copy()
    return ppc_uc
