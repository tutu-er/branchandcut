#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train main BCD and subproblem surrogate models for case3lite."""

from __future__ import annotations

import argparse

import run_training as rt


CASE_NAME = "case3lite"
TRAIN_TARGET = "both"  # "both" | "surrogate" | "bcd"
ACTIVE_SETS_FILE: str | None = None
MAX_SAMPLES = 100
BCD_MAX_ITER = 120
SUBPROBLEM_MAX_ITER = 180
DUAL_EPOCHS = 180
UNIT_PREDICTOR_EPOCHS = 700
NN_EPOCHS = 16
N_WORKERS_SAMPLE = 2
N_WORKERS_UNIT = 1
N_WORKERS_BCD = 1
SUBPROBLEM_LP_BACKEND = "cvxpy_highs"
BCD_LP_BACKEND = "gurobi"
SURROGATE_CONSTRAINT_STRATEGY = "all_templates_sign4_plus_single"
UNIT_PREDICTOR_LOAD_PATH: str | None = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("both", "surrogate", "bcd"), default=TRAIN_TARGET)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--bcd-iter", type=int, default=BCD_MAX_ITER)
    p.add_argument("--sub-iter", type=int, default=SUBPROBLEM_MAX_ITER)
    p.add_argument("--sample-workers", type=int, default=N_WORKERS_SAMPLE)
    p.add_argument("--unit-predictor", type=str, default=UNIT_PREDICTOR_LOAD_PATH)
    p.add_argument("--no-unit-predictor", action="store_true")
    p.add_argument("--no-auto-latest-unit-predictor", action="store_true")
    return p.parse_args()


def _configure_iterations(bcd_iter: int, sub_iter: int) -> None:
    rt.MAX_ITER = max(int(bcd_iter), int(sub_iter))
    rt.BCD_MAX_ITER = max(1, int(bcd_iter))
    rt.SUBPROBLEM_MAX_ITER = max(1, int(sub_iter))
    rt.BCD_UNIT_PREDICTOR_WARMUP_ROUNDS = max(0, round(rt.BCD_MAX_ITER * 0.10))
    rt.BCD_THETA_CONSTRAINT_DELAY_ROUNDS = max(0, round(rt.BCD_MAX_ITER * 0.10))
    rt.DUAL_DECAY_ROUND = max(1, round(rt.BCD_MAX_ITER / 8))
    rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.10))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.25))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.50))
    rt.SUBPROBLEM_PG_COST_START_ROUND = max(0, rt.SUBPROBLEM_MAX_ITER // 4)


def main() -> None:
    args = _parse_args()
    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.target
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = None
    rt.DUAL_EPOCHS = DUAL_EPOCHS
    rt.UNIT_PREDICTOR_EPOCHS = UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_SIZE = 64
    rt.UNIT_PREDICTOR_LR = 1.5e-3
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = 0.0
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = [512, 256, 128]
    rt.UNIT_PREDICTOR_NET_VARIANT = "tcn_shared_film"
    rt.UNIT_PREDICTOR_TCN_CHANNELS = 128
    rt.UNIT_PREDICTOR_TCN_DEPTH = 8
    rt.UNIT_PREDICTOR_DROPOUT = 0.0
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = True
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.30
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.12
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.02
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR = 0.12
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.08
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.80
    rt.NN_EPOCHS = NN_EPOCHS
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = N_WORKERS_UNIT
    rt.N_WORKERS_BCD = N_WORKERS_BCD
    rt.SUBPROBLEM_LP_BACKEND = SUBPROBLEM_LP_BACKEND
    rt.BCD_LP_BACKEND = BCD_LP_BACKEND
    rt.SURROGATE_CONSTRAINT_STRATEGY = SURROGATE_CONSTRAINT_STRATEGY
    rt.USE_UNIT_PREDICTOR = not bool(args.no_unit_predictor)
    rt.BCD_USE_UNIT_PREDICTOR = rt.USE_UNIT_PREDICTOR
    rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = not bool(args.no_auto_latest_unit_predictor)
    rt.UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_MODEL_FILE = None
    rt.SURROGATE_MODEL_DIR = None
    _configure_iterations(args.bcd_iter, args.sub_iter)

    print("=" * 72, flush=True)
    print(
        f"case3lite training | target={rt.MODE} | max_samples={rt.MAX_SAMPLES} | "
        f"bcd_iter={rt.BCD_MAX_ITER} | sub_iter={rt.SUBPROBLEM_MAX_ITER} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'}",
        flush=True,
    )
    print(
        f"unit_predictor={rt.USE_UNIT_PREDICTOR} | "
        f"load_path={rt.UNIT_PREDICTOR_LOAD_PATH or 'auto-latest'} | "
        f"auto_latest={rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE}",
        flush=True,
    )
    print("=" * 72, flush=True)
    rt.main()


if __name__ == "__main__":
    main()
