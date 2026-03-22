from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower.case30
from pypower.idx_bus import PD

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.ActiveSetLearner import ActiveSetLearner
from src.active_set_learner_parallel import ParallelActiveSetLearner
from src.case30_uc_data import get_case30_uc_ppc
from src.mti118_data_loader import (
    build_case118_daily_samples,
    load_case118_ppc_with_mti_limits,
)


def build_case30_base_load(horizon: int) -> tuple[dict, np.ndarray]:
    ppc = get_case30_uc_ppc()
    base_bus_load = np.asarray(ppc["bus"][:, PD], dtype=float)
    load_matrix = np.repeat(base_bus_load[:, None], horizon, axis=1)
    return ppc, load_matrix


def perturb_sample(sample: dict, scale_low: float, scale_high: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    load = np.asarray(sample["load_data"], dtype=float)
    renewable = np.asarray(sample.get("renewable_data", np.zeros_like(load)), dtype=float)

    perturbed = dict(sample)
    perturbed["load_data"] = load * rng.uniform(scale_low, scale_high, size=load.shape)
    perturbed["pd_data"] = perturbed["load_data"].copy()
    perturbed["renewable_data"] = renewable * rng.uniform(scale_low, scale_high, size=renewable.shape)
    return perturbed


def expand_case118_scenarios(
    scenarios: list[dict],
    perturb_repeats: int,
    scale_low: float,
    scale_high: float,
    seed: int,
) -> list[dict]:
    if perturb_repeats <= 0:
        return scenarios

    expanded: list[dict] = []
    for sample in scenarios:
        sample_id = int(sample.get("sample_id", len(expanded)))
        expanded.append(dict(sample))
        for rep in range(perturb_repeats):
            new_sample = perturb_sample(
                sample,
                scale_low=scale_low,
                scale_high=scale_high,
                seed=seed + sample_id * 1000 + rep,
            )
            new_sample["sample_id"] = f"{sample_id}_perturb_{rep + 1}"
            expanded.append(new_sample)
    return expanded


def build_learner(
    parallel: bool,
    alpha: float,
    delta: float,
    epsilon: float,
    ppc,
    t_delta: float,
    pd_data,
    case_name: str,
    n_workers: int,
    gurobi_threads: int,
    verbose_solver: bool,
):
    learner_cls = ParallelActiveSetLearner if parallel else ActiveSetLearner
    kwargs = {
        "alpha": alpha,
        "delta": delta,
        "epsilon": epsilon,
        "ppc": ppc,
        "T_delta": t_delta,
        "Pd": pd_data,
        "case_name": case_name,
        "verbose_solver": verbose_solver,
    }
    if parallel:
        kwargs["n_workers"] = n_workers
        kwargs["gurobi_threads"] = gurobi_threads
    return learner_cls(**kwargs)


def run_case30(
    horizon: int = 24,
    max_samples: int = 200,
    alpha: float = 0.75,
    delta: float = 0.15,
    epsilon: float = 0.15,
    t_delta: float = 1.0,
    parallel: bool = False,
    n_workers: int = 4,
    gurobi_threads: int = 2,
    verbose_solver: bool = False,
    output: str | None = None,
) -> str:
    ppc, base_load = build_case30_base_load(horizon)
    learner = build_learner(
        parallel=parallel,
        alpha=alpha,
        delta=delta,
        epsilon=epsilon,
        ppc=ppc,
        t_delta=t_delta,
        pd_data=base_load,
        case_name="case30",
        n_workers=n_workers,
        gurobi_threads=gurobi_threads,
        verbose_solver=verbose_solver,
    )
    active_sets = learner.run(max_samples=max_samples)
    print(f"case30 active sets: {len(active_sets)}", flush=True)
    return learner.save_active_sets_json(filename=output)


def run_case118(
    market: str = "DA",
    horizon: int = 24,
    aggregate_thermal_by_bus: bool = True,
    max_days: int | None = None,
    max_samples: int | None = None,
    perturb_repeats: int = 0,
    scale_low: float = 0.95,
    scale_high: float = 1.05,
    seed: int = 42,
    alpha: float = 0.70,
    delta: float = 0.05,
    epsilon: float = 0.10,
    t_delta: float = 1.0,
    parallel: bool = True,
    n_workers: int = 4,
    gurobi_threads: int = 2,
    verbose_solver: bool = False,
    output: str | None = None,
) -> str:
    ppc = load_case118_ppc_with_mti_limits(
        aggregate_thermal_by_bus=aggregate_thermal_by_bus,
    )
    scenarios = build_case118_daily_samples(
        market=market,
        horizon=horizon,
        max_days=max_days,
    )
    scenarios = expand_case118_scenarios(
        scenarios,
        perturb_repeats=perturb_repeats,
        scale_low=scale_low,
        scale_high=scale_high,
        seed=seed,
    )

    learner = build_learner(
        parallel=parallel,
        alpha=alpha,
        delta=delta,
        epsilon=epsilon,
        ppc=ppc,
        t_delta=t_delta,
        pd_data=None,
        case_name="case118",
        n_workers=n_workers,
        gurobi_threads=gurobi_threads,
        verbose_solver=verbose_solver,
    )
    active_sets = learner.run_on_precomputed_scenarios(
        scenarios,
        max_samples=max_samples,
    )
    print(f"case118 scenarios solved: {learner.M}", flush=True)
    print(f"case118 active sets: {len(active_sets)}", flush=True)
    return learner.save_active_sets_json(filename=output)


def main() -> None:
    case_name = "case118"

    if case_name == "case30":
        output_path = run_case30(
            horizon=24,
            max_samples=200,
            alpha=0.70,
            delta=0.05,
            epsilon=0.10,
            t_delta=1.0,
            parallel=False,
            n_workers=4,
            gurobi_threads=2,
            verbose_solver=True,
            output=None,
        )
    elif case_name == "case118":
        output_path = run_case118(
            market="DA",
            horizon=24,
            aggregate_thermal_by_bus=True,
            max_days=None,
            max_samples=None,
            perturb_repeats=0,
            scale_low=0.95,
            scale_high=1.05,
            seed=42,
            alpha=0.75,
            delta=0.15,
            epsilon=0.15,
            t_delta=1.0,
            parallel=False,
            n_workers=4,
            gurobi_threads=2,
            verbose_solver=False,
            output=None,
        )
    else:
        raise ValueError(f"Unsupported case_name: {case_name}")

    print(f"saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
