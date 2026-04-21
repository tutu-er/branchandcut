#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 training entrypoint with curated presets.

Switch ``TRAIN_TARGET`` below to run either:
- ``main_bcd``: main-problem BCD training
- ``subproblem_bcd``: subproblem surrogate/BCD training
- ``dual_predictor``: only train ``dual_predictor.pth`` (``run_training.SURROGATE_DUAL_PREDICTOR_ONLY``)

Run from repo root::

    python run_training_case118.py

This wrapper reuses ``run_training.py`` and only overrides the configuration
needed for the refined case118 dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import run_training as rt


TRAIN_TARGET = "dual_predictor"  # "main_bcd" | "subproblem_bcd" | "dual_predictor"
MAIN_BCD_SOLVE_PRESET = "gurobi"  # "gurobi" | "cvxpy_highs"
# 子问题求解预设：
# - "desktop": 偏保守（适合本地 Windows/笔记本）
# - "server":  更激进（适合服务器并行 + HiGHS）
SUBPROBLEM_SOLVE_PRESET = "desktop"  # "desktop" | "server"

ROOT = Path(__file__).resolve().parent

# ── 对偶预测器「新设定」（仅用负荷/可再生作输入；不将启停作为特征）────────────────
# 在 `_configure_common` 中写入 `run_training`，凡经本入口的 case118 训练均生效。
CASE118_DUAL_PREDICTOR_NET_VARIANT = "temporal_conv"  # "mlp" | "temporal_conv"
CASE118_DUAL_PREDICTOR_NORMALIZE_TARGETS = True
CASE118_DUAL_PREDICTOR_COSINE_LOSS_WEIGHT = 0.12
CASE118_DUAL_PREDICTOR_SMOOTH_L1_BETA = 2.0

# 原始 refined active set（含 lambda_power_balance / lambda_dcpf_upper/lower）
CASE118_ACTIVE_SET_JSON_FULL = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025.json"
)
# 电价裁剪精简版：仅含 lambda_pg_electricity_price，范围 [0, 2×max_unit_linear_cost]
# 由 clip_dual_prices_case118.py 生成
CASE118_ACTIVE_SET_JSON_PRICE_CLIPPED = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025"
    "_price_only_clipped.json"
)

# 当前激活的 active set（切换此变量即可）
CASE118_ACTIVE_SET_JSON = CASE118_ACTIVE_SET_JSON_PRICE_CLIPPED

# 轻量并行入口（见 run_training_case118_subproblem_bcd_light.py）：在 subproblem 预设之后覆盖 rt
SUBPROBLEM_LIGHT_MAX_SAMPLES: int | None = None
SUBPROBLEM_LIGHT_N_WORKERS_UNIT: int | None = None
SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE: int | None = None
# 仅训练部分机组时设为列表（如 [0, 1, 5]）；None 表示全部机组（与 run_training.UNIT_IDS 一致）
CASE118_SUBPROBLEM_UNIT_IDS: list[int] | None = [0, 1, 2]

# ── Case118 子问题 c_pg（发电边际修正头）────────────────────────────────
# 与 118 节点系统、裁剪电价 λ 的典型量级及较长子问题 BCD 外循环对齐：
# - 略增大 pg_cost_scale 倍率，避免 tanh 饱和导致无法闭合 pg 驻点残差；
# - c_pg 分支用 large 宽度，便于拟合时段相关的 pg_const；
# - 略提前启用 c_pg，使外循环后半段有更多轮次专门压 obj_dual_pg；
# - surr_lr 为 BCD 内 c_pg 步实际使用的 Adam 学习率（见 uc_NN_subproblem.iter_with_c_pg_nn）。
CASE118_SUBPROBLEM_C_PG_NN_SIZE = "large"
CASE118_SUBPROBLEM_PG_COST_SCALE_MULTIPLIER = 2.75
CASE118_SUBPROBLEM_PG_COST_NN_EPOCHS = 14
CASE118_SUBPROBLEM_PG_COST_START_ROUND = 32
CASE118_SUBPROBLEM_PG_COST_LR = 1e-4
CASE118_SUBPROBLEM_PG_COST_SURR_LR = 2e-4
CASE118_SUBPROBLEM_PG_COST_REG_DEADBAND = 1.0
CASE118_SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS = 1e-5
CASE118_SUBPROBLEM_PG_COST_BATCH_STRATEGY: str | None = "mini-batch"
CASE118_SUBPROBLEM_PG_COST_BATCH_SIZE: int | None = 32
CASE118_SUBPROBLEM_PG_COST_SHUFFLE: bool | None = True
CASE118_SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS: bool = True
CASE118_SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER: float = 1.0
CASE118_SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP: float = 10.0
CASE118_SUBPROBLEM_RHO_DUAL_PG_INIT = 0.15
CASE118_SUBPROBLEM_LOSS_RATIO_DUAL_PG = 1.25

# ── 单机组 0/1 变量预测器（Case118 子问题训练专用，可切换开关）────────────
# 注意：仅当 constraint_generation_strategy 包含 single-time 段时才生效
#        （all_single_time / all_templates_sign4_plus_single）。
CASE118_USE_UNIT_PREDICTOR = True
CASE118_UNIT_PREDICTOR_EPOCHS = 200
CASE118_UNIT_PREDICTOR_BATCH_STRATEGY = "full-batch"
CASE118_UNIT_PREDICTOR_BATCH_SIZE = 32
CASE118_UNIT_PREDICTOR_SHUFFLE = True
CASE118_UNIT_PREDICTOR_LR = 1e-3
CASE118_UNIT_PREDICTOR_HIDDEN_DIMS = [256, 128]
CASE118_UNIT_PREDICTOR_FINETUNE_LR = 1e-5
CASE118_UNIT_PREDICTOR_WEIGHT_DECAY = 1e-4
# UnitPredictor（距离最小化）结构与损失配置
CASE118_UNIT_PREDICTOR_NET_VARIANT = "tcn_shared_film"
CASE118_UNIT_PREDICTOR_TCN_CHANNELS = 64
CASE118_UNIT_PREDICTOR_TCN_DEPTH = 6
CASE118_UNIT_PREDICTOR_DROPOUT = 0.1
# 预训练阶段：用纯 MSE 对齐 x_hat（避免拉长 epoch 导致 unit0/2 过拟合）
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_BCE = 0.0
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 1.0
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_L1 = 0.0
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TV = 0.0
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.25
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.05
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR = 0.2
CASE118_UNIT_PREDICTOR_STD_FLOOR_SCALE = 0.5
CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.15
CASE118_UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.8
# 可选：仅对 unit=1 做 TV 微调（从实验结果看能显著降低距离且不拖累其它机组）
CASE118_UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS = 240
CASE118_UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT = 0.02
# 自动 TV 微调：针对其它机组防“均值塌陷”
CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV = True
CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS = 120
CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD = 0.02
CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT = 0.02
# 若已用独立脚本训练好 unit_predictor，可在此指定目录/文件供主流程直接加载使用
# 例如: "result/surrogate_models/unit_predictor_case118_20260421_173107"
CASE118_UNIT_PREDICTOR_LOAD_PATH: str | None = "result/surrogate_models/unit_predictor_case118_20260421_173107"

# BCD 对偶符号松弛间隔（dual floor active 期间每隔 k 轮松弛一次符号）
# 该值写入 run_training.BCD_DUAL_SIGN_RELAX_INTERVAL，影响 main_bcd / subproblem_bcd / dual_predictor 三种入口下的 BCD/子问题训练。
CASE118_BCD_DUAL_SIGN_RELAX_INTERVAL = 10


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
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_THETA_TRAINING_STAGES = None
    rt.BCD_GUROBI_THREADS = None

    rt.THETA_HOT_START_STRATEGY = "dcpf_relative"
    rt.ZETA_HOT_START_STRATEGY = "zero"
    rt.BCD_LAMBDA_INIT_STRATEGY = "ed_on_x_opt"

    # 统一设置 BCD 对偶符号松弛间隔（main/subproblem 预设若无显式覆盖则沿用此值）
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = int(CASE118_BCD_DUAL_SIGN_RELAX_INTERVAL)

    rt.DUAL_PREDICTOR_NET_VARIANT = CASE118_DUAL_PREDICTOR_NET_VARIANT
    rt.DUAL_PREDICTOR_NORMALIZE_TARGETS = CASE118_DUAL_PREDICTOR_NORMALIZE_TARGETS
    rt.DUAL_PREDICTOR_COSINE_LOSS_WEIGHT = CASE118_DUAL_PREDICTOR_COSINE_LOSS_WEIGHT
    rt.DUAL_PREDICTOR_SMOOTH_L1_BETA = CASE118_DUAL_PREDICTOR_SMOOTH_L1_BETA


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
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = int(CASE118_BCD_DUAL_SIGN_RELAX_INTERVAL)

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


def _configure_dual_predictor() -> None:
    """仅训练对偶变量预测器 NN（写出 ``dual_predictor.pth``），不跑各机组子问题代理。"""
    rt.MODE = "surrogate"
    rt.RUN_FP = False
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = True
    rt.SURROGATE_MODEL_DIR = None
    rt.SURROGATE_CONTINUE_TRAINING = False

    # 续训：把目录指到已有 ``dual_predictor.pth``，并打开 ``SURROGATE_CONTINUE_TRAINING``。
    # rt.SURROGATE_MODEL_DIR = "result/surrogate_models/your_run"
    # rt.SURROGATE_CONTINUE_TRAINING = True

    if SUBPROBLEM_SOLVE_PRESET == "server":
        rt.SUBPROBLEM_LP_BACKEND = "cvxpy_highs"
    else:
        rt.SUBPROBLEM_LP_BACKEND = "gurobi"

    rt.DUAL_EPOCHS = 320
    rt.DUAL_BATCH_SIZE = 16
    rt.DUAL_BATCH_STRATEGY = "mini-batch"
    rt.DUAL_SHUFFLE = True
    rt.DUAL_LR = 3e-4
    # 对偶预测器新设定见文件顶部 CASE118_DUAL_* 与 _configure_common


def _configure_subproblem_bcd() -> None:
    cpu = _cpu_count()

    rt.MODE = "surrogate"
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.UNIT_IDS = None
    if CASE118_SUBPROBLEM_UNIT_IDS is not None:
        rt.UNIT_IDS = list(CASE118_SUBPROBLEM_UNIT_IDS)

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
    rt.DUAL_EPOCHS = 320
    rt.DUAL_BATCH_SIZE = 16
    rt.DUAL_BATCH_STRATEGY = "mini-batch"
    rt.DUAL_SHUFFLE = True
    rt.DUAL_LR = 3e-4
    # 对偶预测器新设定见文件顶部 CASE118_DUAL_* 与 _configure_common

    rt.NN_EPOCHS = 5
    rt.MAX_ITER = 80
    rt.SUBPROBLEM_MAX_ITER = 80

    rt.SUBPROBLEM_NN_SIZE = "medium"
    rt.SUBPROBLEM_C_PG_NN_SIZE = CASE118_SUBPROBLEM_C_PG_NN_SIZE
    rt.SUBPROBLEM_NN_BATCH_STRATEGY = "full-batch"
    rt.SUBPROBLEM_NN_BATCH_SIZE = 8
    rt.SUBPROBLEM_NN_SHUFFLE = True
    rt.SUBPROBLEM_NN_LR = 3e-4

    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
    rt.SUBPROBLEM_GAMMA_BASE = 1e-3
    rt.SUBPROBLEM_RHO_PRIMAL_INIT = 1e-1
    rt.SUBPROBLEM_RHO_DUAL_INIT = 1e-3
    rt.SUBPROBLEM_RHO_DUAL_PG_INIT = CASE118_SUBPROBLEM_RHO_DUAL_PG_INIT
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

    rt.SUBPROBLEM_LOSS_RATIO_DUAL_PG = CASE118_SUBPROBLEM_LOSS_RATIO_DUAL_PG

    rt.SUBPROBLEM_PG_COST_NN_EPOCHS = CASE118_SUBPROBLEM_PG_COST_NN_EPOCHS
    rt.SUBPROBLEM_PG_COST_START_ROUND = CASE118_SUBPROBLEM_PG_COST_START_ROUND
    rt.SUBPROBLEM_PG_COST_SCALE_MULTIPLIER = CASE118_SUBPROBLEM_PG_COST_SCALE_MULTIPLIER
    rt.SUBPROBLEM_X_COST_NN_LR = 5e-6
    rt.SUBPROBLEM_PG_COST_LR = CASE118_SUBPROBLEM_PG_COST_LR
    rt.SUBPROBLEM_PG_COST_SURR_LR = CASE118_SUBPROBLEM_PG_COST_SURR_LR
    rt.SUBPROBLEM_PG_COST_REG_DEADBAND = CASE118_SUBPROBLEM_PG_COST_REG_DEADBAND
    rt.SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS = CASE118_SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS
    rt.SUBPROBLEM_PG_COST_BATCH_STRATEGY = CASE118_SUBPROBLEM_PG_COST_BATCH_STRATEGY
    rt.SUBPROBLEM_PG_COST_BATCH_SIZE = CASE118_SUBPROBLEM_PG_COST_BATCH_SIZE
    rt.SUBPROBLEM_PG_COST_SHUFFLE = CASE118_SUBPROBLEM_PG_COST_SHUFFLE
    rt.SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS = CASE118_SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS
    rt.SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER = CASE118_SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER
    rt.SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP = CASE118_SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP

    rt.USE_UNIT_PREDICTOR = CASE118_USE_UNIT_PREDICTOR
    rt.UNIT_PREDICTOR_EPOCHS = CASE118_UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_STRATEGY = CASE118_UNIT_PREDICTOR_BATCH_STRATEGY
    rt.UNIT_PREDICTOR_BATCH_SIZE = CASE118_UNIT_PREDICTOR_BATCH_SIZE
    rt.UNIT_PREDICTOR_SHUFFLE = CASE118_UNIT_PREDICTOR_SHUFFLE
    rt.UNIT_PREDICTOR_LR = CASE118_UNIT_PREDICTOR_LR
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = CASE118_UNIT_PREDICTOR_HIDDEN_DIMS
    rt.UNIT_PREDICTOR_FINETUNE_LR = CASE118_UNIT_PREDICTOR_FINETUNE_LR
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = CASE118_UNIT_PREDICTOR_WEIGHT_DECAY
    rt.UNIT_PREDICTOR_NET_VARIANT = CASE118_UNIT_PREDICTOR_NET_VARIANT
    rt.UNIT_PREDICTOR_TCN_CHANNELS = CASE118_UNIT_PREDICTOR_TCN_CHANNELS
    rt.UNIT_PREDICTOR_TCN_DEPTH = CASE118_UNIT_PREDICTOR_TCN_DEPTH
    rt.UNIT_PREDICTOR_DROPOUT = CASE118_UNIT_PREDICTOR_DROPOUT
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = False
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BCE = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_BCE
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_MSE
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_L1 = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_L1
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TV
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR
    rt.UNIT_PREDICTOR_STD_FLOOR_SCALE = CASE118_UNIT_PREDICTOR_STD_FLOOR_SCALE
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = CASE118_UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = CASE118_UNIT_PREDICTOR_TV_FLOOR_SCALE
    rt.UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS = CASE118_UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS
    rt.UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT = CASE118_UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT
    rt.UNIT_PREDICTOR_AUTO_EXTRA_TV = CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV
    rt.UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS = CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS
    rt.UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD = CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD
    rt.UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT = CASE118_UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT
    rt.UNIT_PREDICTOR_LOAD_PATH = CASE118_UNIT_PREDICTOR_LOAD_PATH


def _validate_inputs() -> None:
    path = ROOT / CASE118_ACTIVE_SET_JSON
    if not path.exists():
        raise FileNotFoundError(f"case118 active set json not found: {path}")


def _apply_subproblem_light_runtime_overrides() -> bool:
    """在 ``_configure_subproblem_bcd`` 之后应用轻量覆盖（仅当对应 *_LIGHT_* 非 None）。"""
    changed = False
    if SUBPROBLEM_LIGHT_MAX_SAMPLES is not None:
        rt.MAX_SAMPLES = max(1, int(SUBPROBLEM_LIGHT_MAX_SAMPLES))
        changed = True
    if SUBPROBLEM_LIGHT_N_WORKERS_UNIT is not None:
        rt.N_WORKERS_UNIT = max(1, int(SUBPROBLEM_LIGHT_N_WORKERS_UNIT))
        changed = True
    if SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE is not None:
        w = max(1, int(SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE))
        rt.N_WORKERS_SAMPLE = w
        rt.N_WORKERS_SUBPROBLEM = w
        changed = True
    return changed


def main() -> None:
    _validate_inputs()
    _configure_common()

    light_overrides = False
    if TRAIN_TARGET == "main_bcd":
        _configure_main_bcd()
    elif TRAIN_TARGET == "subproblem_bcd":
        _configure_subproblem_bcd()
        light_overrides = _apply_subproblem_light_runtime_overrides()
    elif TRAIN_TARGET == "dual_predictor":
        _configure_dual_predictor()
        light_overrides = _apply_subproblem_light_runtime_overrides()
    else:
        raise ValueError(
            f"Unsupported TRAIN_TARGET={TRAIN_TARGET!r}; "
            "expected 'main_bcd', 'subproblem_bcd', or 'dual_predictor'."
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
            f"n_workers_unit={rt.N_WORKERS_UNIT}, "
            f"unit_ids={rt.UNIT_IDS!r}",
            flush=True,
        )
        print(
            "dual_predictor_preset: "
            f"net={rt.DUAL_PREDICTOR_NET_VARIANT}, "
            f"normalize_targets={rt.DUAL_PREDICTOR_NORMALIZE_TARGETS}, "
            f"cosine_w={rt.DUAL_PREDICTOR_COSINE_LOSS_WEIGHT}, "
            f"smooth_l1_beta={rt.DUAL_PREDICTOR_SMOOTH_L1_BETA}",
            flush=True,
        )
        print(
            "case118_c_pg: "
            f"c_pg_size={CASE118_SUBPROBLEM_C_PG_NN_SIZE}, "
            f"pg_cost_scale_mult={CASE118_SUBPROBLEM_PG_COST_SCALE_MULTIPLIER}, "
            f"pg_cost_start_round={CASE118_SUBPROBLEM_PG_COST_START_ROUND}, "
            f"pg_cost_nn_epochs={CASE118_SUBPROBLEM_PG_COST_NN_EPOCHS}, "
            f"rho_dual_pg_init={CASE118_SUBPROBLEM_RHO_DUAL_PG_INIT}, "
            f"loss_ratio_dual_pg={CASE118_SUBPROBLEM_LOSS_RATIO_DUAL_PG}, "
            f"pg_cost_surr_lr={CASE118_SUBPROBLEM_PG_COST_SURR_LR}, "
            f"pg_cost_batch={CASE118_SUBPROBLEM_PG_COST_BATCH_STRATEGY}/"
            f"{CASE118_SUBPROBLEM_PG_COST_BATCH_SIZE}, "
            f"pg_cost_w={CASE118_SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS}",
            flush=True,
        )
        if light_overrides:
            print(
                "subproblem_light_overrides: "
                f"max_samples={rt.MAX_SAMPLES}, "
                f"n_workers_unit={rt.N_WORKERS_UNIT}, "
                f"n_workers_sample={rt.N_WORKERS_SAMPLE}",
                flush=True,
            )
    elif TRAIN_TARGET == "dual_predictor":
        print(
            "dual_predictor_only: "
            f"SURROGATE_DUAL_PREDICTOR_ONLY={rt.SURROGATE_DUAL_PREDICTOR_ONLY}, "
            f"epochs={rt.DUAL_EPOCHS}, batch={rt.DUAL_BATCH_SIZE}, "
            f"strategy={rt.DUAL_BATCH_STRATEGY}, lr={rt.DUAL_LR}, "
            f"continue={rt.SURROGATE_CONTINUE_TRAINING}, "
            f"model_dir={rt.SURROGATE_MODEL_DIR}",
            flush=True,
        )
        print(
            "dual_predictor_preset: "
            f"net={rt.DUAL_PREDICTOR_NET_VARIANT}, "
            f"normalize_targets={rt.DUAL_PREDICTOR_NORMALIZE_TARGETS}, "
            f"cosine_w={rt.DUAL_PREDICTOR_COSINE_LOSS_WEIGHT}, "
            f"smooth_l1_beta={rt.DUAL_PREDICTOR_SMOOTH_L1_BETA}",
            flush=True,
        )
        if light_overrides:
            print(
                "subproblem_light_overrides (samples): "
                f"max_samples={rt.MAX_SAMPLES}",
                flush=True,
            )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
