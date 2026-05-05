#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one perturbed case30lite generator for the generalization test."""

from __future__ import annotations

import argparse

import run_training as rt
from src.case30_uc_data import CASE30LITE_PERTURBED_UNIT_ID


CASE_NAME = "case30lite_perturbed"
TRAIN_TARGET = "surrogate"
ACTIVE_SETS_FILE: str | None = None
PERTURBED_UNIT_ID = CASE30LITE_PERTURBED_UNIT_ID
MAX_SAMPLES = 100
SUBPROBLEM_MAX_ITER = 60
DUAL_EPOCHS = 160
UNIT_PREDICTOR_EPOCHS = 0
UNIT_PREDICTOR_TRAIN_EPOCHS = 500
NN_EPOCHS = 4
N_WORKERS_SAMPLE = 1
N_WORKERS_UNIT = 1
SUBPROBLEM_LP_BACKEND = "cvxpy_highs"
SURROGATE_CONSTRAINT_STRATEGY = "all_templates_sign4_plus_single"
UNIT_PREDICTOR_LOAD_PATH = (
    "result/surrogate_models/unit_predictor_case30lite_20260504_132021/unit_predictor.pth"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--unit-id", type=int, default=PERTURBED_UNIT_ID)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--sub-iter", type=int, default=SUBPROBLEM_MAX_ITER)
    p.add_argument("--dual-epochs", type=int, default=DUAL_EPOCHS)
    p.add_argument("--nn-epochs", type=int, default=NN_EPOCHS)
    p.add_argument("--sample-workers", type=int, default=N_WORKERS_SAMPLE)
    p.add_argument("--unit-predictor", type=str, default=UNIT_PREDICTOR_LOAD_PATH)
    p.add_argument("--unit-predictor-epochs", type=int, default=None)
    p.add_argument(
        "--train-unit-predictor",
        action="store_true",
        help="Also train a fresh standalone unit predictor on the perturbed active set.",
    )
    return p.parse_args()


def _configure_iterations(sub_iter: int) -> None:
    rt.MAX_ITER = max(1, int(sub_iter))
    rt.SUBPROBLEM_MAX_ITER = max(1, int(sub_iter))
    rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.10))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.25))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.50))
    rt.SUBPROBLEM_PG_COST_START_ROUND = max(0, rt.SUBPROBLEM_MAX_ITER // 4)


def main() -> None:
    args = _parse_args()

    rt.CASE_NAME = CASE_NAME
    rt.MODE = TRAIN_TARGET
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = [int(args.unit_id)]
    rt.DUAL_EPOCHS = max(1, int(args.dual_epochs))
    rt.NN_EPOCHS = max(1, int(args.nn_epochs))
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = N_WORKERS_UNIT
    rt.SUBPROBLEM_LP_BACKEND = SUBPROBLEM_LP_BACKEND
    rt.SURROGATE_CONSTRAINT_STRATEGY = SURROGATE_CONSTRAINT_STRATEGY
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.SURROGATE_CONTINUE_TRAINING = False
    rt.SURROGATE_MODEL_DIR = None
    rt.BCD_MODEL_FILE = None

    rt.USE_UNIT_PREDICTOR = True
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = True
    rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor or None
    if args.unit_predictor_epochs is not None:
        rt.UNIT_PREDICTOR_EPOCHS = max(0, int(args.unit_predictor_epochs))
    else:
        rt.UNIT_PREDICTOR_EPOCHS = UNIT_PREDICTOR_TRAIN_EPOCHS if args.train_unit_predictor else UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_SIZE = 64
    rt.UNIT_PREDICTOR_LR = 1.5e-3
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = 0.0
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = [512, 256, 128]
    rt.UNIT_PREDICTOR_DROPOUT = 0.0
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = True
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.22
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.07
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.02
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.035
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.70

    _configure_iterations(args.sub_iter)

    print("=" * 72, flush=True)
    print(
        f"{CASE_NAME} single-unit training | unit={rt.UNIT_IDS[0]} | "
        f"max_samples={rt.MAX_SAMPLES} | sub_iter={rt.SUBPROBLEM_MAX_ITER} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'}",
        flush=True,
    )
    print(
        f"unit_predictor={rt.UNIT_PREDICTOR_LOAD_PATH or 'auto-latest'} | "
        f"unit_predictor_epochs={rt.UNIT_PREDICTOR_EPOCHS}",
        flush=True,
    )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
