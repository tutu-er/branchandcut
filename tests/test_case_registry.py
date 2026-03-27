from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypower")

from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF

from src.case_registry import build_case3_base_load, build_case3lite_base_load, get_case_ppc


def test_case3_ppc_structure_and_ptdf():
    ppc = get_case_ppc("case3")
    assert ppc["bus"].shape == (3, 13)
    assert ppc["gen"].shape[0] == 3
    assert ppc["branch"].shape[0] == 3
    assert ppc["gencost"].shape[0] == 3

    ppc_int = ext2int(ppc)
    ptdf = makePTDF(ppc_int["baseMVA"], ppc_int["bus"], ppc_int["branch"])
    assert ptdf.shape == (ppc_int["branch"].shape[0], ppc_int["bus"].shape[0])
    assert np.all(np.isfinite(ptdf))


def test_case3_base_load_matches_horizon_and_capacity():
    ppc, base_load = build_case3_base_load(horizon=24)
    assert base_load.shape == (ppc["bus"].shape[0], 24)
    assert np.all(base_load >= 0.0)

    total_load = np.sum(base_load, axis=0)
    total_pmax = float(np.sum(ppc["gen"][:, 8]))
    assert np.max(total_load) < total_pmax
    assert np.max(total_load) > np.min(total_load)


def test_case3lite_ppc_and_base_load_are_available():
    ppc = get_case_ppc("case3lite")
    assert ppc["bus"].shape == (3, 13)
    assert ppc["gen"].shape[0] == 3
    assert ppc["branch"].shape[0] == 3

    ppc, base_load = build_case3lite_base_load(horizon=24)
    assert base_load.shape == (ppc["bus"].shape[0], 24)
    total_load = np.sum(base_load, axis=0)
    total_pmax = float(np.sum(ppc["gen"][:, 8]))
    assert np.max(total_load) < total_pmax
    assert np.max(total_load) > np.min(total_load)
