import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sparse_surrogate_mining import (
    SparseSurrogateLibrary,
    build_mining_samples_from_agent,
    build_pg_window_candidates,
    compute_rhs_value,
    evaluate_library_on_samples,
    greedy_select_candidates,
    mine_sparse_surrogate_library,
    mine_sparse_surrogate_library_from_agent,
)


def _make_samples():
    samples = []
    total_load_profiles = [
        np.array([60.0, 80.0, 75.0, 55.0]),
        np.array([65.0, 90.0, 82.0, 58.0]),
        np.array([58.0, 78.0, 70.0, 52.0]),
        np.array([70.0, 95.0, 88.0, 60.0]),
        np.array([62.0, 85.0, 80.0, 57.0]),
    ]
    for total in total_load_profiles:
        pd_data = np.vstack([0.55 * total, 0.45 * total])

        pg_true = np.zeros((2, 4), dtype=float)
        pg_true[0, :] = np.array([
            0.35 * total[0],
            0.70 * total[1],
            0.66 * total[2],
            0.30 * total[3],
        ])
        pg_true[1, :] = total - pg_true[0, :]

        pg_lp = pg_true.copy()
        pg_lp[0, 1] += 12.0
        pg_lp[0, 2] += 9.0
        pg_lp[1, 1] -= 12.0
        pg_lp[1, 2] -= 9.0

        x_true = (pg_true > 1e-6).astype(float)
        x_lp = np.clip(x_true + np.array([[0.0, -0.2, -0.15, 0.0], [0.0, 0.2, 0.15, 0.0]]), 0.0, 1.0)

        samples.append(
            {
                "pd_data": pd_data,
                "pg_lp": pg_lp,
                "pg_true": pg_true,
                "x_lp": x_lp,
                "x_true": x_true,
            }
        )
    return samples


def test_mine_sparse_surrogate_library_finds_discriminative_pg_window():
    samples = _make_samples()
    library = mine_sparse_surrogate_library(
        samples,
        max_selected=6,
        window_sizes=(1, 2),
    )

    assert library.constraints
    assert library.metadata["n_selected_constraints"] <= 6

    selected_windows = {
        (tuple(candidate.metadata.get("unit_ids", [])), tuple(candidate.metadata.get("time_indices", [])))
        for candidate in library.constraints
    }
    assert ((0,), (1,)) in selected_windows or ((0,), (1, 2)) in selected_windows

    best = max(
        library.constraints,
        key=lambda candidate: candidate.metrics.score if candidate.metrics else float("-inf"),
    )
    assert best.metrics is not None
    assert best.metrics.true_satisfaction >= 0.95
    assert best.metrics.violation_gap_mean > 0.0


def test_sparse_library_round_trip_and_rhs_evaluation(tmp_path):
    samples = _make_samples()
    library = mine_sparse_surrogate_library(
        samples,
        max_selected=4,
        window_sizes=(1,),
    )
    assert library.constraints

    path = tmp_path / "sparse_library.json"
    library.to_json(path)
    restored = SparseSurrogateLibrary.from_json(path)

    assert len(restored.constraints) == len(library.constraints)
    rhs_value = compute_rhs_value(restored.constraints[0], samples[0]["pd_data"])
    assert np.isfinite(rhs_value)


def test_greedy_selection_reduces_duplicate_activation_patterns():
    candidates = build_pg_window_candidates(ng=1, T=3, window_sizes=(1,))
    for idx, candidate in enumerate(candidates[:2]):
        candidate.metrics = type("Metrics", (), {
            "true_satisfaction": 1.0,
            "violation_gap_mean": 2.0,
            "score": 5.0 - idx,
        })()
        candidate.lp_activation_pattern = [1, 0, 1, 0]
    candidates = candidates[:2]

    selected = greedy_select_candidates(
        candidates,
        max_selected=5,
        redundancy_threshold=0.8,
    )
    assert len(selected) == 1


def test_agent_adapter_builds_samples_and_library():
    class DummyAgent:
        def __init__(self, samples):
            self.active_set_data = []
            self.x = []
            self.pg = []
            self.x_opt = []
            for item in samples:
                self.active_set_data.append({"pd_data": item["pd_data"], "pg_true": item["pg_true"]})
                self.x.append(item["x_lp"])
                self.pg.append(item["pg_lp"])
                self.x_opt.append(item["x_true"])
            self.x = np.asarray(self.x, dtype=float)
            self.pg = np.asarray(self.pg, dtype=float)
            self.x_opt = np.asarray(self.x_opt, dtype=float)

    samples = _make_samples()
    agent = DummyAgent(samples)
    mining_samples = build_mining_samples_from_agent(agent)
    assert len(mining_samples) == len(samples)

    library = mine_sparse_surrogate_library_from_agent(agent, max_selected=3, window_sizes=(1,))
    summary = evaluate_library_on_samples(library, mining_samples)
    assert library.constraints
    assert summary["n_constraints"] >= 1
