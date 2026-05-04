#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build UC active-set samples for the UC-ready IEEE case14 preset."""

from __future__ import annotations

import argparse

from src.ActiveSetLearner import ActiveSetLearner
from src.active_set_learner_parallel import ParallelActiveSetLearner
from src.case_registry import CASE14_LOAD_SCALE, build_case14_base_load


CASE_NAME = "case14"
HORIZON = 24
LOAD_SCALE = CASE14_LOAD_SCALE
MAX_SAMPLES = 600
TARGET_SAMPLES = 600
ALPHA = 0.72
DELTA = 0.10
EPSILON = 0.12
T_DELTA = 1.0
PARALLEL = False
N_WORKERS = 4
GUROBI_THREADS = 2
VERBOSE_SOLVER = False
OUTPUT_PATH: str | None = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=TARGET_SAMPLES)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--load-scale", type=float, default=LOAD_SCALE)
    p.add_argument("--parallel", action="store_true", default=PARALLEL)
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--output", type=str, default=OUTPUT_PATH)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ppc, base_load = build_case14_base_load(args.horizon, scale=args.load_scale)
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
    )
    if args.parallel:
        kwargs.update(n_workers=max(1, args.workers), gurobi_threads=GUROBI_THREADS)
    learner = learner_cls(**kwargs)
    active_sets = learner.run(max_samples=args.max_samples, target_samples=args.samples)
    print(f"{CASE_NAME} active sets: {len(active_sets)}", flush=True)
    output_path = learner.save_active_sets_json(filename=args.output)
    print(f"saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
