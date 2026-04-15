#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 training entrypoint with curated presets.

Switch ``TRAIN_TARGET`` below to run either:
- ``main_bcd``: main-problem BCD training
- ``subproblem_bcd``: subproblem surrogate/BCD training

This wrapper reuses ``run_training.py`` and only overrides the configuration
needed for the refined case118 dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import run_training as rt


TRAIN_TARGET = "main_bcd"  # "main_bcd" | "subproblem_bcd"
# 子问题求解预设：
# - "desktop": 偏保守（适合本地 Windows/笔记本）
# - "server":  更激进（适合服务器并行 + HiGHS）
SUBPROBLEM_SOLVE_PRESET = "desktop"  # "desktop" | "server"

ROOT = Path(__file__).resolve().parent
CASE118_ACTIVE_SET_JSON = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260408_132932_active_set_like_refined.json"
)


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _configure_common() -> None:
    rt.CASE_NAME = "case118"
    rt.RUN_FP = False
    rt.ENABLE_SPARSE_SUPPORTS = False
    rt.ACTIVE_SETS_FILE = CASE118_ACTIVE_SET_JSON
    rt.MAX_SAMPLES = None
    rt.T_DELTA = 1.0

    rt.BCD_MODEL_FILE = None
    rt.BCD_CONTINUE_TRAINING = False
    rt.SURROGATE_MODEL_DIR = None
    rt.SURROGATE_CONTINUE_TRAINING = False
    rt.BCD_THETA_TRAINING_STAGES = None

    rt.THETA_HOT_START_STRATEGY = "dcpf_relative"
    rt.ZETA_HOT_START_STRATEGY = "zero"
    rt.BCD_LAMBDA_INIT_STRATEGY = "ed_on_x_opt"


def _configure_main_bcd() -> None:
    workers = min(2, _cpu_count())

    rt.MODE = "bcd"
    rt.N_WORKERS_BCD = workers

    # Use the full refined case118 set; keep iterations moderate because each
    # sample is a 39-unit, 24-period UC trajectory.
    rt.MAX_ITER = 120
    rt.BCD_MAX_ITER = 120
    rt.SUBPROBLEM_MAX_ITER = 120
    rt.NN_EPOCHS = 6
    rt.DUAL_DECAY_ROUND = 24
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = 6

    rt.BCD_NN_SIZE = "medium"
    rt.BCD_NN_BATCH_STRATEGY = "full-batch"
    rt.BCD_NN_BATCH_SIZE = 8
    rt.BCD_NN_SHUFFLE = True
    rt.BCD_NN_LR = 2e-5
    rt.BCD_ENABLE_DROPOUT_DURING_NN_TRAINING = True

    rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = 24
    rt.BCD_THETA_TRAINING_STAGES = [
        {"max_constraints_per_time_slot": 8, "iterations": 40},
        {"max_constraints_per_time_slot": 16, "iterations": 40},
        {"max_constraints_per_time_slot": 24, "iterations": 40},
    ]
    rt.BCD_GAMMA_BASE = 1e-3
    rt.BCD_RHO_PRIMAL_INIT = 1e-3
    rt.BCD_RHO_DUAL_INIT = 1e-3
    rt.BCD_RHO_DUAL_PG_INIT = 1.0
    rt.BCD_RHO_DUAL_X_INIT = 1e-3
    rt.BCD_RHO_DUAL_COC_INIT = 1e1
    rt.BCD_RHO_BINARY_INIT = 1e2
    rt.BCD_RHO_OPT_INIT = 1e-3


def _configure_subproblem_bcd() -> None:
    cpu = _cpu_count()

    rt.MODE = "surrogate"
    rt.UNIT_IDS = None

    if SUBPROBLEM_SOLVE_PRESET == "server":
        # 服务器：让子问题 LP 用 HiGHS（cvxpy + highspy），并增大并行度
        rt.SUBPROBLEM_LP_BACKEND = "cvxpy_highs"

        # unit 维度并行 + sample 维度并行都开，但避免把机器打满（留一点给系统/IO）
        rt.N_WORKERS_UNIT = min(4, cpu)
        rt.N_WORKERS_SAMPLE = max(1, min(32, cpu - 2))
        rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    else:
        # 本地：unit 串行（Windows 更稳），sample 轻度并行
        rt.N_WORKERS_UNIT = 1
        rt.N_WORKERS_SAMPLE = min(3, cpu)
        rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
        rt.SUBPROBLEM_LP_BACKEND = "gurobi"

    # Dual predictor + per-unit subproblem training.
    rt.DUAL_EPOCHS = 120
    rt.DUAL_BATCH_SIZE = 16
    rt.DUAL_BATCH_STRATEGY = "mini-batch"
    rt.DUAL_SHUFFLE = True
    rt.DUAL_LR = 3e-4

    rt.NN_EPOCHS = 5
    rt.MAX_ITER = 80
    rt.SUBPROBLEM_MAX_ITER = 80

    rt.SUBPROBLEM_NN_SIZE = "medium"
    rt.SUBPROBLEM_C_PG_NN_SIZE = "medium"
    rt.SUBPROBLEM_NN_BATCH_STRATEGY = "full-batch"
    rt.SUBPROBLEM_NN_BATCH_SIZE = 8
    rt.SUBPROBLEM_NN_SHUFFLE = True
    rt.SUBPROBLEM_NN_LR = 3e-4

    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
    rt.SUBPROBLEM_GAMMA_BASE = 1e-3
    rt.SUBPROBLEM_RHO_PRIMAL_INIT = 1e-1
    rt.SUBPROBLEM_RHO_DUAL_INIT = 1e-3
    rt.SUBPROBLEM_RHO_DUAL_PG_INIT = 1e-1
    rt.SUBPROBLEM_RHO_DUAL_X_INIT = 1e-1
    rt.SUBPROBLEM_RHO_DUAL_COC_INIT = 1e1
    rt.SUBPROBLEM_RHO_BINARY_INIT = 1e2
    rt.SUBPROBLEM_RHO_OPT_INIT = 1e-1

    rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT = 1.0
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = 20
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = 40
    rt.SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL = 10
    rt.SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS = 2
    rt.SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE = 0.67

    rt.SUBPROBLEM_PG_COST_NN_EPOCHS = 8
    rt.SUBPROBLEM_PG_COST_START_ROUND = 40
    rt.SUBPROBLEM_X_COST_NN_LR = 5e-6
    rt.SUBPROBLEM_PG_COST_LR = 5e-5
    rt.SUBPROBLEM_PG_COST_SURR_LR = 1e-4


def _validate_inputs() -> None:
    path = ROOT / CASE118_ACTIVE_SET_JSON
    if not path.exists():
        raise FileNotFoundError(f"case118 active set json not found: {path}")


def main() -> None:
    _validate_inputs()
    _configure_common()

    if TRAIN_TARGET == "main_bcd":
        _configure_main_bcd()
    elif TRAIN_TARGET == "subproblem_bcd":
        _configure_subproblem_bcd()
    else:
        raise ValueError(
            f"Unsupported TRAIN_TARGET={TRAIN_TARGET!r}; "
            "expected 'main_bcd' or 'subproblem_bcd'."
        )

    print("=" * 72, flush=True)
    print(f"run_training_case118.py -> target={TRAIN_TARGET}", flush=True)
    print(f"active_set_json={CASE118_ACTIVE_SET_JSON}", flush=True)
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
