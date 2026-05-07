#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train case3lite with stronger dual floors for complex surrogate constraints.

This preset is a sibling of ``run_training_case3lite.py``.  It keeps the same
case3lite data/model defaults, but strengthens the dual lower bounds used by
the main theta constraints and subproblem sign4 constraints.
"""

from __future__ import annotations

import argparse

import run_training as rt
import run_training_case3lite as base


CASE_NAME = base.CASE_NAME
TRAIN_TARGET = base.TRAIN_TARGET
ACTIVE_SETS_FILE = base.ACTIVE_SETS_FILE
MAX_SAMPLES = base.MAX_SAMPLES
BCD_MAX_ITER = base.BCD_MAX_ITER
SUBPROBLEM_MAX_ITER = base.SUBPROBLEM_MAX_ITER
N_WORKERS_SAMPLE = base.N_WORKERS_SAMPLE
UNIT_PREDICTOR_LOAD_PATH = base.UNIT_PREDICTOR_LOAD_PATH
N_WORKERS_BCD = base.N_WORKERS_BCD
BCD_LP_BACKEND = base.BCD_LP_BACKEND
BCD_GUROBI_THREADS = 1

# Stronger-than-default complex-constraint dual floors.
# - theta_dual_floor is passed to BCD_MU_DUAL_FLOOR_INIT, i.e. the main-model
#   transmission/theta dual family.
# - sign4_dual_floor is passed to SUBPROBLEM_MU_DUAL_FLOOR_INIT, i.e. the
#   subproblem surrogate/sign4 constraint dual family.
THETA_DUAL_FLOOR = 5.00
ZETA_DUAL_FLOOR = 0.00
SIGN4_DUAL_FLOOR = 5.00
THETA_FLOOR_FRACTION = 0.40
SIGN4_INDIVIDUAL_FRACTION = 0.50
SIGN4_GROUP_FRACTION = 0.85


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("both", "surrogate", "bcd"), default=TRAIN_TARGET)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
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
    p.add_argument("--theta-dual-floor", type=float, default=THETA_DUAL_FLOOR)
    p.add_argument("--zeta-dual-floor", type=float, default=ZETA_DUAL_FLOOR)
    p.add_argument("--sign4-dual-floor", type=float, default=SIGN4_DUAL_FLOOR)
    p.add_argument("--theta-floor-frac", type=float, default=THETA_FLOOR_FRACTION)
    p.add_argument("--sign4-individual-frac", type=float, default=SIGN4_INDIVIDUAL_FRACTION)
    p.add_argument("--sign4-group-frac", type=float, default=SIGN4_GROUP_FRACTION)
    return p.parse_args()


def _clip_fraction(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _configure_strong_complex_dual_floors(args: argparse.Namespace) -> None:
    bcd_iter = max(1, int(args.bcd_iter))
    sub_iter = max(1, int(args.sub_iter))
    theta_floor_frac = _clip_fraction(args.theta_floor_frac)
    sign4_individual_frac = _clip_fraction(args.sign4_individual_frac)
    sign4_group_frac = max(sign4_individual_frac, _clip_fraction(args.sign4_group_frac))

    rt.BCD_MU_DUAL_FLOOR_INIT = max(0.0, float(args.theta_dual_floor))
    rt.BCD_ITA_DUAL_FLOOR_INIT = max(0.0, float(args.zeta_dual_floor))
    rt.DUAL_DECAY_ROUND = max(1, round(bcd_iter * theta_floor_frac))
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = 4

    rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT = max(0.0, float(args.sign4_dual_floor))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, round(sub_iter * sign4_individual_frac))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND,
        round(sub_iter * sign4_group_frac),
    )
    rt.SUBPROBLEM_SIGN4_DELAY_ROUNDS = 0
    rt.SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS = 0
    rt.SUBPROBLEM_SIGN4_INITIAL_SCALE = 1.0
    rt.SUBPROBLEM_SIGN4_FINAL_SCALE = 1.0
    rt.SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = True
    rt.SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE = "sign4_only"


def main() -> None:
    args = _parse_args()

    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.target
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = None

    rt.DUAL_EPOCHS = base.DUAL_EPOCHS
    rt.UNIT_PREDICTOR_EPOCHS = base.UNIT_PREDICTOR_EPOCHS
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

    rt.NN_EPOCHS = base.NN_EPOCHS
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = base.N_WORKERS_UNIT
    rt.N_WORKERS_BCD = max(1, int(args.bcd_workers))
    rt.SUBPROBLEM_LP_BACKEND = base.SUBPROBLEM_LP_BACKEND
    rt.BCD_LP_BACKEND = args.bcd_backend
    rt.BCD_GUROBI_THREADS = None if int(args.gurobi_threads) <= 0 else int(args.gurobi_threads)
    rt.SURROGATE_CONSTRAINT_STRATEGY = base.SURROGATE_CONSTRAINT_STRATEGY
    rt.USE_UNIT_PREDICTOR = not bool(args.no_unit_predictor)
    rt.BCD_USE_UNIT_PREDICTOR = rt.USE_UNIT_PREDICTOR
    rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = not bool(args.no_auto_latest_unit_predictor)
    rt.UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_MODEL_FILE = None
    rt.SURROGATE_MODEL_DIR = None
    rt.BCD_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.BCD_DUAL_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT = 0.0

    base._configure_iterations(args.bcd_iter, args.sub_iter)
    _configure_strong_complex_dual_floors(args)

    print("=" * 72, flush=True)
    print(
        f"case3lite strong-complex-dual-floor training | target={rt.MODE} | "
        f"max_samples={rt.MAX_SAMPLES} | bcd_iter={rt.BCD_MAX_ITER} | "
        f"sub_iter={rt.SUBPROBLEM_MAX_ITER} | active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'}",
        flush=True,
    )
    print(
        f"bcd_backend={rt.BCD_LP_BACKEND} | bcd_workers={rt.N_WORKERS_BCD} | "
        f"gurobi_threads={rt.BCD_GUROBI_THREADS if rt.BCD_GUROBI_THREADS is not None else 'auto'}",
        flush=True,
    )
    print(
        f"quadratic_prox=off | bcd_pg_prox={rt.BCD_PG_BLOCK_PROX_WEIGHT} | "
        f"bcd_dual_prox={rt.BCD_DUAL_BLOCK_PROX_WEIGHT} | "
        f"sub_pg_prox={rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT} | "
        f"sub_dual_prox={rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT}",
        flush=True,
    )
    print(
        f"theta_floor={rt.BCD_MU_DUAL_FLOOR_INIT} until round {rt.DUAL_DECAY_ROUND} | "
        f"zeta_floor={rt.BCD_ITA_DUAL_FLOOR_INIT} | "
        f"sign4_floor={rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT} "
        f"individual_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND}, "
        f"group_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND}",
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
