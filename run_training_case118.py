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
MAIN_BCD_SOLVE_PRESET = "gurobi"  # "gurobi" | "cvxpy_highs"
# 子问题求解预设：
# - "desktop": 偏保守（适合本地 Windows/笔记本）
# - "server":  更激进（适合服务器并行 + HiGHS）
SUBPROBLEM_SOLVE_PRESET = "desktop"  # "desktop" | "server"

ROOT = Path(__file__).resolve().parent
CASE118_ACTIVE_SET_JSON = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025.json"
)


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _validate_main_bcd_theta_schedule(
    *,
    max_iter: int,
    max_constraints_per_time_slot: int,
    theta_training_stages: list[dict] | None,
) -> None:
    if not theta_training_stages:
        return

    total_iterations = 0
    previous_limit = 0
    for stage_idx, stage in enumerate(theta_training_stages):
        if not isinstance(stage, dict):
            raise ValueError(
                f"theta_training_stages[{stage_idx}] must be a dict, got {type(stage).__name__}"
            )

        limit = int(stage["max_constraints_per_time_slot"])
        iterations = int(stage["iterations"])
        if limit <= 0:
            raise ValueError(
                f"theta_training_stages[{stage_idx}] has non-positive limit={limit}"
            )
        if iterations <= 0:
            raise ValueError(
                f"theta_training_stages[{stage_idx}] has non-positive iterations={iterations}"
            )
        if limit > max_constraints_per_time_slot:
            raise ValueError(
                f"theta_training_stages[{stage_idx}] limit={limit} exceeds "
                f"BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT={max_constraints_per_time_slot}"
            )
        if limit < previous_limit:
            raise ValueError(
                f"theta_training_stages must be non-decreasing by limit; "
                f"stage {stage_idx} has {limit} after {previous_limit}"
            )
        previous_limit = limit
        total_iterations += iterations

    if total_iterations != max_iter:
        raise ValueError(
            f"theta_training_stages iterations sum to {total_iterations}, "
            f"but max_iter={max_iter}"
        )
    if previous_limit != max_constraints_per_time_slot:
        raise ValueError(
            "The final theta stage must reach "
            f"BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT={max_constraints_per_time_slot}, "
            f"got {previous_limit}"
        )


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
    rt.BCD_GUROBI_THREADS = None

    rt.THETA_HOT_START_STRATEGY = "dcpf_relative"
    rt.ZETA_HOT_START_STRATEGY = "zero"
    rt.BCD_LAMBDA_INIT_STRATEGY = "ed_on_x_opt"


def _configure_main_bcd() -> None:
    rt.MODE = "bcd"
    cpu = _cpu_count()

    if MAIN_BCD_SOLVE_PRESET == "gurobi":
        rt.BCD_LP_BACKEND = "gurobi"
        # Gurobi academic license: 2 concurrent sessions maximum.
        # With Method=2 (barrier/IPM) each LP model is multi-threaded, so
        # allocate cpu//2 threads per model to saturate all cores with 2 workers.
        rt.N_WORKERS_BCD = min(2, cpu)
        rt.BCD_GUROBI_THREADS = max(1, cpu // rt.N_WORKERS_BCD)
        rt.BCD_GUROBI_LP_METHOD = 2   # barrier (IPM) – multi-threaded LP
    elif MAIN_BCD_SOLVE_PRESET == "cvxpy_highs":
        rt.BCD_LP_BACKEND = "cvxpy_highs"
        # HiGHS has no concurrent-model license limit.
        # Run one worker per 2 CPU cores (each worker keeps one persistent CVXPY
        # problem alive; warm_start=True avoids re-canonicalization every iter).
        # Give each HiGHS solve 2 threads so total ≈ cpu cores used.
        rt.N_WORKERS_BCD = max(1, cpu // 2)
        rt.BCD_GUROBI_THREADS = None
        rt.BCD_GUROBI_LP_METHOD = -1
        # 2 threads per HiGHS solve: barrier can use them; simplex falls back to 1.
        # Total cores used = N_WORKERS_BCD × bcd_highs_threads ≈ cpu.
        rt.BCD_HIGHS_THREADS = 2
    else:
        raise ValueError(
            f"Unsupported MAIN_BCD_SOLVE_PRESET={MAIN_BCD_SOLVE_PRESET!r}; "
            "expected 'gurobi' or 'cvxpy_highs'."
        )

    # Keep a longer outer BCD horizon for case118. The baseline BCD NN learning
    # rate needs more than 120 outer rounds to fully exploit staged theta
    # training on the 366-sample refined dataset.
    rt.MAX_ITER = 180
    rt.BCD_MAX_ITER = 180
    rt.SUBPROBLEM_MAX_ITER = 180
    rt.NN_EPOCHS = 6
    rt.DUAL_DECAY_ROUND = 36
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = 6

    rt.BCD_NN_SIZE = "medium"
    rt.BCD_NN_BATCH_STRATEGY = "full-batch"
    rt.BCD_NN_BATCH_SIZE = 8
    rt.BCD_NN_SHUFFLE = True
    rt.BCD_NN_LR = 5e-5
    rt.BCD_ENABLE_DROPOUT_DURING_NN_TRAINING = True

    rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = 24
    rt.BCD_THETA_TRAINING_STAGES = [
        {"max_constraints_per_time_slot": 8, "iterations": 36},
        {"max_constraints_per_time_slot": 16, "iterations": 54},
        {"max_constraints_per_time_slot": 24, "iterations": 90},
    ]
    _validate_main_bcd_theta_schedule(
        max_iter=rt.BCD_MAX_ITER,
        max_constraints_per_time_slot=rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT,
        theta_training_stages=rt.BCD_THETA_TRAINING_STAGES,
    )
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
    if TRAIN_TARGET == "main_bcd":
        print(
            "main_bcd_preset="
            f"{MAIN_BCD_SOLVE_PRESET}, "
            f"backend={rt.BCD_LP_BACKEND}, "
            f"n_workers_bcd={rt.N_WORKERS_BCD}, "
            f"cpu_count={_cpu_count()}, "
            f"gurobi_threads={rt.BCD_GUROBI_THREADS}, "
            f"bcd_max_iter={rt.BCD_MAX_ITER}, "
            f"nn_epochs={rt.NN_EPOCHS}",
            flush=True,
        )
    elif TRAIN_TARGET == "subproblem_bcd":
        print(
            "subproblem_preset="
            f"{SUBPROBLEM_SOLVE_PRESET}, "
            f"backend={rt.SUBPROBLEM_LP_BACKEND}, "
            f"n_workers_sample={rt.N_WORKERS_SAMPLE}, "
            f"n_workers_unit={rt.N_WORKERS_UNIT}",
            flush=True,
        )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
