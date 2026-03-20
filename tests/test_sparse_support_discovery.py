import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sparse_constraint_templates import (
    SparseConstraintTemplateLibrary,
    build_sparse_template_library,
    template_library_to_bcd_union_constraints,
)
from src.sparse_support_discovery import (
    SparseSupportDiscoveryResult,
    discover_sparse_supports,
    extract_support_discovery_samples_from_agent,
)


def _make_samples():
    samples = []
    # 让 x[0,1], x[1,2], x[2,0] 成为高价值变量
    load_profiles = [
        np.array([50.0, 80.0, 60.0]),
        np.array([52.0, 82.0, 58.0]),
        np.array([48.0, 78.0, 63.0]),
        np.array([55.0, 85.0, 65.0]),
        np.array([47.0, 76.0, 61.0]),
        np.array([53.0, 88.0, 66.0]),
    ]
    for idx, total in enumerate(load_profiles):
        pd_data = np.vstack([0.6 * total, 0.4 * total])
        x_true = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        x_lp = x_true.copy()
        x_lp[0, 1] = 0.45 if idx % 2 == 0 else 0.35
        x_lp[1, 2] = 0.55 if idx % 2 == 0 else 0.65
        x_lp[2, 0] = 0.40 if idx % 3 == 0 else 0.30
        samples.append({"pd_data": pd_data, "x_lp": x_lp, "x_true": x_true})
    return samples


def test_discover_sparse_supports_selects_high_value_variables():
    result = discover_sparse_supports(
        _make_samples(),
        top_k_variables=6,
        max_groups=3,
        group_size=3,
    )

    picked = {(item.variable.unit_id, item.variable.time_index) for item in result.selected_variables}
    assert (0, 1) in picked
    assert (1, 2) in picked
    assert result.support_groups


def test_template_library_round_trip_and_bcd_conversion(tmp_path):
    samples = _make_samples()
    result = discover_sparse_supports(samples, top_k_variables=5, max_groups=2, group_size=3)
    library = build_sparse_template_library(result, samples, max_templates=2)
    assert library.templates

    path = tmp_path / "sparse_x_templates.json"
    library.to_json(path)
    restored = SparseConstraintTemplateLibrary.from_json(path)
    assert len(restored.templates) == len(library.templates)

    union_constraints = template_library_to_bcd_union_constraints(restored, start_branch_id=5000)
    assert union_constraints
    assert union_constraints[0]["branch_id"] >= 5000
    assert "time_index" in union_constraints[0]["nonzero_pg_coefficients"][0]


def test_extract_support_discovery_samples_from_agent():
    class DummyAgent:
        def __init__(self, samples):
            self.active_set_data = [{"pd_data": item["pd_data"]} for item in samples]
            self.x = np.asarray([item["x_lp"] for item in samples], dtype=float)
            self.x_opt = np.asarray([item["x_true"] for item in samples], dtype=float)

    samples = _make_samples()
    agent = DummyAgent(samples)
    extracted = extract_support_discovery_samples_from_agent(agent)
    assert len(extracted) == len(samples)
    assert np.allclose(extracted[0]["x_lp"], samples[0]["x_lp"])
