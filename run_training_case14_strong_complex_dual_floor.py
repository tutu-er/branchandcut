#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train case14 with stronger dual floors for complex surrogate constraints."""

from __future__ import annotations

import argparse

import run_training as rt
import run_training_case14 as base
import run_training_case3lite_strong_complex_dual_floor as strong


CASE_NAME = base.CASE_NAME
TRAIN_TARGET = base.TRAIN_TARGET
ACTIVE_SETS_FILE = base.ACTIVE_SETS_FILE
RESUME_SURROGATE_DIR = base.RESUME_SURROGATE_DIR
MAX_SAMPLES = base.MAX_SAMPLES
BCD_MAX_ITER = base.BCD_MAX_ITER
SUBPROBLEM_MAX_ITER = base.SUBPROBLEM_MAX_ITER
DUAL_EPOCHS = base.DUAL_EPOCHS
UNIT_PREDICTOR_EPOCHS = base.UNIT_PREDICTOR_EPOCHS
NN_EPOCHS = base.NN_EPOCHS
N_WORKERS_SAMPLE = base.N_WORKERS_SAMPLE
N_WORKERS_UNIT = base.N_WORKERS_UNIT
N_WORKERS_BCD = base.N_WORKERS_BCD
SUBPROBLEM_LP_BACKEND = base.SUBPROBLEM_LP_BACKEND
BCD_LP_BACKEND = base.BCD_LP_BACKEND
BCD_GUROBI_THREADS = 1
SURROGATE_CONSTRAINT_STRATEGY = base.SURROGATE_CONSTRAINT_STRATEGY
UNIT_PREDICTOR_LOAD_PATH = (
    "result/surrogate_models/unit_predictor_case14_20260504_132051/unit_predictor.pth"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("both", "surrogate", "bcd"), default=TRAIN_TARGET)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--surrogate-model-dir", "--resume-dir", dest="surrogate_model_dir",
                   type=str, default=RESUME_SURROGATE_DIR)
    p.add_argument("--skip-existing-units", action="store_true")
    p.add_argument("--retrain-existing", action="store_true")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--bcd-iter", type=int, default=BCD_MAX_ITER)
    p.add_argument("--sub-iter", type=int, default=SUBPROBLEM_MAX_ITER)
    p.add_argument("--sample-workers", type=int, default=N_WORKERS_SAMPLE)
    p.add_argument("--bcd-workers", type=int, default=N_WORKERS_BCD)
    p.add_argument("--bcd-backend", choices=("gurobi", "cvxpy_highs"), default=BCD_LP_BACKEND)
    p.add_argument("--gurobi-threads", type=int, default=BCD_GUROBI_THREADS)
    p.add_argument("--unit-predictor", type=str, default=UNIT_PREDICTOR_LOAD_PATH)
    p.add_argument("--no-unit-predictor", action="store_true")
    p.add_argument("--no-auto-latest-unit-predictor", action="store_true")
    p.add_argument("--theta-dual-floor", type=float, default=strong.THETA_DUAL_FLOOR)
    p.add_argument("--zeta-dual-floor", type=float, default=strong.ZETA_DUAL_FLOOR)
    p.add_argument("--sign4-dual-floor", type=float, default=strong.SIGN4_DUAL_FLOOR)
    p.add_argument("--theta-floor-frac", type=float, default=strong.THETA_FLOOR_FRACTION)
    p.add_argument("--sign4-individual-frac", type=float, default=strong.SIGN4_INDIVIDUAL_FRACTION)
    p.add_argument("--sign4-group-frac", type=float, default=strong.SIGN4_GROUP_FRACTION)
    p.add_argument("--sign4-delay-rounds", type=int, default=strong.SIGN4_DELAY_ROUNDS)
    p.add_argument("--sign4-curriculum-rounds", type=int, default=strong.SIGN4_CURRICULUM_ROUNDS)
    p.add_argument("--sign4-initial-scale", type=float, default=strong.SIGN4_INITIAL_SCALE)
    p.add_argument("--sign4-final-scale", type=float, default=strong.SIGN4_FINAL_SCALE)
    p.add_argument("--single-mu-cap-weight", type=float, default=strong.SINGLE_MU_CAP_WEIGHT)
    p.add_argument("--single-mu-cap-initial-weight", type=float,
                   default=strong.SINGLE_MU_CAP_INITIAL_WEIGHT)
    p.add_argument("--single-mu-cap-final-weight", type=float, default=None)
    p.add_argument("--single-mu-cap-initial", type=float, default=strong.SINGLE_MU_CAP_INITIAL)
    p.add_argument("--single-mu-cap-final", type=float, default=strong.SINGLE_MU_CAP_FINAL)
    p.add_argument("--single-mu-cap-start-frac", type=float,
                   default=strong.SINGLE_MU_CAP_START_FRACTION)
    p.add_argument("--single-mu-cap-end-frac", type=float,
                   default=strong.SINGLE_MU_CAP_END_FRACTION)
    p.add_argument("--c-pg-start-round", type=int, default=strong.SUBPROBLEM_PG_COST_START_ROUND)
    p.add_argument("--c-pg-nn-epochs", type=int, default=strong.SUBPROBLEM_PG_COST_NN_EPOCHS)
    p.add_argument("--c-pg-direct-epochs", type=int,
                   default=strong.SUBPROBLEM_C_PG_DIRECT_EPOCHS)
    p.add_argument("--c-pg-direct-batch-strategy", choices=("full-batch", "mini-batch"),
                   default=strong.SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY)
    p.add_argument("--c-pg-direct-batch-size", type=int,
                   default=strong.SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE)
    p.add_argument("--unit-ids", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.target
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = strong._parse_unit_ids(args.unit_ids)
    rt.DUAL_EPOCHS = DUAL_EPOCHS
    rt.UNIT_PREDICTOR_EPOCHS = UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_SIZE = 64
    rt.UNIT_PREDICTOR_LR = 1.5e-3
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = 0.0
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = [512, 256, 128]
    rt.UNIT_PREDICTOR_DROPOUT = 0.0
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = True
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.25
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.08
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.02
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.04
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.70
    rt.NN_EPOCHS = NN_EPOCHS
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = N_WORKERS_UNIT
    rt.N_WORKERS_BCD = max(1, int(args.bcd_workers))
    rt.SUBPROBLEM_LP_BACKEND = SUBPROBLEM_LP_BACKEND
    rt.BCD_LP_BACKEND = args.bcd_backend
    rt.BCD_GUROBI_THREADS = None if int(args.gurobi_threads) <= 0 else int(args.gurobi_threads)
    rt.SURROGATE_CONSTRAINT_STRATEGY = SURROGATE_CONSTRAINT_STRATEGY
    rt.USE_UNIT_PREDICTOR = not bool(args.no_unit_predictor)
    rt.BCD_USE_UNIT_PREDICTOR = rt.USE_UNIT_PREDICTOR
    rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = not bool(args.no_auto_latest_unit_predictor)
    rt.UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_MODEL_FILE = None
    rt.SURROGATE_MODEL_DIR = args.surrogate_model_dir
    rt.SURROGATE_CONTINUE_TRAINING = bool(args.surrogate_model_dir)
    rt.SURROGATE_SKIP_EXISTING_UNITS = (
        bool(args.skip_existing_units)
        or (bool(args.surrogate_model_dir) and not bool(args.retrain_existing))
    )
    rt.BCD_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.BCD_DUAL_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT = 0.0

    base._configure_iterations(args.bcd_iter, args.sub_iter)
    strong._configure_strong_complex_dual_floors(args)
    rt.SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY = strong.SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY
    rt.SUBPROBLEM_MAIN_DIRECT_EPOCHS = max(1, int(strong.SUBPROBLEM_MAIN_DIRECT_EPOCHS))
    rt.SUBPROBLEM_PG_COST_START_ROUND = max(0, int(args.c_pg_start_round))
    rt.SUBPROBLEM_PG_COST_NN_EPOCHS = max(1, int(args.c_pg_nn_epochs))
    rt.SUBPROBLEM_C_PG_DIRECT_EPOCHS = max(1, int(args.c_pg_direct_epochs))
    rt.SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY = args.c_pg_direct_batch_strategy
    rt.SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE = max(1, int(args.c_pg_direct_batch_size))

    print("=" * 72, flush=True)
    print(
        f"case14 strong-complex-dual-floor training | target={rt.MODE} | "
        f"max_samples={rt.MAX_SAMPLES} | bcd_iter={rt.BCD_MAX_ITER} | "
        f"sub_iter={rt.SUBPROBLEM_MAX_ITER} | units={rt.UNIT_IDS or 'all'} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'}",
        flush=True,
    )
    print(
        f"theta_floor={rt.BCD_MU_DUAL_FLOOR_INIT} until round {rt.DUAL_DECAY_ROUND} | "
        f"sign4_floor={rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT} "
        f"individual_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND}, "
        f"group_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND}",
        flush=True,
    )
    print(
        f"single_mu_cap_weight={rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT}"
        f"->{rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT} | "
        f"cap={rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL}->{rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL} | "
        f"rounds={rt.SUBPROBLEM_SINGLE_MU_CAP_START_ROUND}"
        f"..{rt.SUBPROBLEM_SINGLE_MU_CAP_END_ROUND}",
        flush=True,
    )
    print("=" * 72, flush=True)
    rt.main()


if __name__ == "__main__":
    main()
