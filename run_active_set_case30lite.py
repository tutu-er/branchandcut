#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build UC active-set samples for the lightweight case30 preset."""

from __future__ import annotations

import argparse

import numpy as np

from src.ActiveSetLearner import ActiveSetLearner
from src.active_set_learner_parallel import ParallelActiveSetLearner
from src.case_registry import CASE30LITE_LOAD_SCALE, build_case30lite_base_load


CASE_NAME = "case30lite"
HORIZON = 24
LOAD_SCALE = CASE30LITE_LOAD_SCALE
MAX_SAMPLES = 500
TARGET_SAMPLES = 500
LOAD_PERTURB_LOW = 0.92
LOAD_PERTURB_HIGH = 1.08
SYSTEM_LOAD_SCALE_LOW = 1.00
SYSTEM_LOAD_SCALE_HIGH = 1.00
TEMPORAL_WAVE_AMPLITUDE = 0.02
TEMPORAL_WAVE_CYCLES_LOW = 0.75
TEMPORAL_WAVE_CYCLES_HIGH = 1.75
ALPHA = 0.72
DELTA = 0.12
EPSILON = 0.12
T_DELTA = 1.0
PARALLEL = False
N_WORKERS = 4
GUROBI_THREADS = 2
VERBOSE_SOLVER = False
OUTPUT_PATH: str | None = None


def _active_set_to_x(active_set, ng: int, horizon: int) -> np.ndarray:
    x = np.zeros((ng, horizon), dtype=int)
    for item in active_set:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        pos, value = item
        if isinstance(pos, tuple) and len(pos) == 2:
            g, t = int(pos[0]), int(pos[1])
            if 0 <= g < ng and 0 <= t < horizon:
                x[g, t] = int(value)
    return x


def _print_commitment_report(samples: list, ng: int, horizon: int) -> None:
    if not samples:
        return
    matrices = []
    for sample in samples:
        active_set = sample["active_set"] if isinstance(sample, dict) else sample[1]
        matrices.append(_active_set_to_x(active_set, ng, horizon))
    stack = np.stack(matrices, axis=0)
    unique = {tuple(x.reshape(-1).tolist()) for x in stack}
    on_rate = stack.mean(axis=(0, 2))
    transition_rate = np.abs(stack[:, :, 1:] - stack[:, :, :-1]).mean(axis=(0, 2))
    print("-" * 72, flush=True)
    print(f"commitment diversity: unique={len(unique)} / samples={len(samples)}", flush=True)
    print("unit | on_rate | transition_rate", flush=True)
    for g in range(ng):
        print(f"{g:4d} | {on_rate[g]:7.3f} | {transition_rate[g]:15.3f}", flush=True)
    print("-" * 72, flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=TARGET_SAMPLES)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--load-scale", type=float, default=LOAD_SCALE)
    p.add_argument("--load-perturb-low", type=float, default=LOAD_PERTURB_LOW)
    p.add_argument("--load-perturb-high", type=float, default=LOAD_PERTURB_HIGH)
    p.add_argument("--system-scale-low", type=float, default=SYSTEM_LOAD_SCALE_LOW)
    p.add_argument("--system-scale-high", type=float, default=SYSTEM_LOAD_SCALE_HIGH)
    p.add_argument("--temporal-wave", type=float, default=TEMPORAL_WAVE_AMPLITUDE)
    p.add_argument("--wave-cycles-low", type=float, default=TEMPORAL_WAVE_CYCLES_LOW)
    p.add_argument("--wave-cycles-high", type=float, default=TEMPORAL_WAVE_CYCLES_HIGH)
    p.add_argument("--parallel", action="store_true", default=PARALLEL)
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--output", type=str, default=OUTPUT_PATH)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ppc, base_load = build_case30lite_base_load(args.horizon, scale=args.load_scale)
    learner_cls = ParallelActiveSetLearner if args.parallel else ActiveSetLearner
    kwargs = dict(
        alpha=ALPHA,
        delta=DELTA,
        epsilon=EPSILON,
        ppc=ppc,
        T_delta=T_DELTA,
        Pd=base_load,
        case_name=CASE_NAME,
        verbose_solver=VERBOSE_SOLVER,
        load_perturbation_low=args.load_perturb_low,
        load_perturbation_high=args.load_perturb_high,
        system_load_scale_low=args.system_scale_low,
        system_load_scale_high=args.system_scale_high,
        temporal_wave_amplitude=args.temporal_wave,
        temporal_wave_cycles_low=args.wave_cycles_low,
        temporal_wave_cycles_high=args.wave_cycles_high,
    )
    if args.parallel:
        kwargs.update(n_workers=max(1, args.workers), gurobi_threads=GUROBI_THREADS)
    learner = learner_cls(**kwargs)
    print(
        f"{CASE_NAME} load sampling | base_scale={args.load_scale} | "
        f"bus=[{args.load_perturb_low}, {args.load_perturb_high}] | "
        f"system=[{args.system_scale_low}, {args.system_scale_high}] | "
        f"temporal_wave={args.temporal_wave} cycles=[{args.wave_cycles_low}, {args.wave_cycles_high}]",
        flush=True,
    )
    active_sets = learner.run(max_samples=args.max_samples, target_samples=args.samples)
    print(f"{CASE_NAME} active sets: {len(active_sets)}", flush=True)
    _print_commitment_report(learner.samples, ppc["gen"].shape[0], base_load.shape[1])
    output_path = learner.save_active_sets_json(filename=args.output)
    print(f"saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
