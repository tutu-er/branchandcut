#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本（多模式）
- surrogate: V3 三时段代理约束训练（uc_NN_subproblem）
- bcd:       BCD 主代理训练（uc_NN_BCD，Agent_NN_BCD）
- sparse:    稀疏支持集发现 → sparse BCD 主代理训练
- both:      BCD 训练 → surrogate 训练 → 联合 BCD 训练

可选标志 RUN_FP=True：训练后运行 feasibility_pump 可行性泵测试
（bcd / sparse 模式不支持 RUN_FP，请改用 both 模式）

修改顶部的 MODE / RUN_FP / SURROGATE_DUAL_PREDICTOR_ONLY 等变量切换执行模式。
"""

import os
import sys
import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

load_active_set_from_json = None
Agent_NN_BCD = None
ParallelAgent_NN_BCD = None


def ensure_bcd_modules_imported() -> None:
    global load_active_set_from_json, Agent_NN_BCD, ParallelAgent_NN_BCD
    if (
        load_active_set_from_json is not None
        and Agent_NN_BCD is not None
        and ParallelAgent_NN_BCD is not None
    ):
        return
    try:
        from uc_NN_BCD import load_active_set_from_json as _load_active_set_from_json, Agent_NN_BCD as _Agent_NN_BCD
        from uc_NN_BCD_parallel import ParallelAgent_NN_BCD as _ParallelAgent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)
    load_active_set_from_json = _load_active_set_from_json
    Agent_NN_BCD = _Agent_NN_BCD
    ParallelAgent_NN_BCD = _ParallelAgent_NN_BCD

# ──────────────────────── 依赖检查 ────────────────────────

def check_and_install_dependencies(required_subproblem_backend: str | None = None):
    backend = "gurobi" if required_subproblem_backend is None else str(required_subproblem_backend).strip().lower()
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'pypower': 'PYPOWER',
    }
    if backend != "cvxpy_highs":
        dependencies['gurobipy'] = 'gurobipy'
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"[OK] {import_name}")
        except ImportError:
            missing.append(package_name)
            print(f"[MISS] {import_name} 未安装")

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        response = input("是否自动安装? (y/n): ")
        if response.strip().lower() == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("[OK] 安装完成")
            return True
        else:
            print("请手动安装后重试")
            return False
    if backend == "cvxpy_highs":
        optional_missing = []
        for import_name, package_name in {"cvxpy": "cvxpy", "highspy": "highspy"}.items():
            try:
                __import__(import_name)
                print(f"[OK] {import_name}")
            except ImportError:
                optional_missing.append(package_name)
                print(f"[MISS] {import_name}")
        if optional_missing:
            print(
                "\ncvxpy_highs backend requires optional packages: "
                + ", ".join(optional_missing)
            )
            print(f"Install command: {sys.executable} -m pip install {' '.join(optional_missing)}")
            return False
    return True

# ──────────────────────── 模式配置 ────────────────────────
#
#   'surrogate' - V3 三时段代理约束训练
#   'bcd'       - BCD 主代理训练（Agent_NN_BCD）
#   'sparse'    - 稀疏支持集发现 → sparse BCD 训练
#   'both'      - BCD 训练 → surrogate 训练 → 联合 BCD 训练
#
MODE = 'surrogate'
ENABLE_SPARSE_SUPPORTS = False
RUN_FP = False

# 顶部集中配置区：训练相关参数统一在这里调整
CASE_NAME = 'case3lite'      # 'case3' / 'case3lite' / 'case14' / 'case30' / 'case30lite' / 'case30lite_perturbed' / 'case39' / 'case118'
MAX_SAMPLES = 100            # None = 使用全部样本
T_DELTA = 1.0
DUAL_EPOCHS = 200
DUAL_BATCH_SIZE = 8
DUAL_BATCH_STRATEGY = 'full-batch'   # 'full-batch' / 'mini-batch'
DUAL_SHUFFLE = True
DUAL_LR = 5e-4
# 对偶预测器（所有经 run_surrogate 的子问题/surrogate 训练默认启用；不满足 2*nb*T 时自动回退 MLP）
DUAL_PREDICTOR_NET_VARIANT = 'temporal_conv'  # 'mlp' | 'temporal_conv'
DUAL_PREDICTOR_NORMALIZE_TARGETS = True
DUAL_PREDICTOR_COSINE_LOSS_WEIGHT = 0.12
DUAL_PREDICTOR_SMOOTH_L1_BETA = 2.0
# 单机组 0/1 变量预测器：为 True 时在 BCD 之前预训练 per-unit predictor，
# 并在 all_single_time / all_templates_sign4_plus_single 策略下用其派生 single-time (alpha, delta)。
USE_UNIT_PREDICTOR = True
UNIT_PREDICTOR_EPOCHS = 200
UNIT_PREDICTOR_BATCH_STRATEGY = 'full-batch'   # 'full-batch' / 'mini-batch'
UNIT_PREDICTOR_BATCH_SIZE = 32
UNIT_PREDICTOR_SHUFFLE = True
UNIT_PREDICTOR_LR = 1e-3
UNIT_PREDICTOR_HIDDEN_DIMS = [256, 128]
# UnitPredictor 结构与“距离型”损失配置（默认保持兼容：mlp + BCE）
UNIT_PREDICTOR_NET_VARIANT = "mlp"   # "mlp" | "resmlp" | "tconv" | "tcn"
UNIT_PREDICTOR_TCN_CHANNELS = 64
UNIT_PREDICTOR_TCN_DEPTH = 6
UNIT_PREDICTOR_TCONV_CHANNELS = 64
UNIT_PREDICTOR_TCONV_DEPTH = 4
UNIT_PREDICTOR_DROPOUT = 0.1
UNIT_PREDICTOR_ENABLE_POS_WEIGHT = False
UNIT_PREDICTOR_POS_WEIGHT_CLIP = 20.0
UNIT_PREDICTOR_LOSS_WEIGHT_BCE = 1.0
UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.0
UNIT_PREDICTOR_LOSS_WEIGHT_L1 = 0.0
UNIT_PREDICTOR_LOSS_WEIGHT_TV = 0.0
UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.0
UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.0
UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR = 0.0
UNIT_PREDICTOR_STD_FLOOR_SCALE = 0.5
UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.0
UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.8
# 可选：直接加载外部训练好的 unit_predictor（覆盖 SURROGATE_MODEL_DIR 的继续训练逻辑）
# - None: 按原逻辑（若 SURROGATE_MODEL_DIR 下存在 unit_predictor.pth 则继续训练）
# - file: 直接加载该文件
# - dir: 读取 <dir>/LATEST.txt 或 <dir>/unit_predictor.pth
# - LATEST.txt: 读取其中指向的模型路径
UNIT_PREDICTOR_LOAD_PATH = None
# 当 UNIT_PREDICTOR_LOAD_PATH 为空时，自动查找
# result/surrogate_models/unit_predictor_<CASE_NAME>_*/unit_predictor.pth。
# 对 case3lite，这可直接复用 run_unit_predictor_case3lite.py 的 standalone 输出。
UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = True
# 若加载的 unit_predictor checkpoint 含 metadata，则用其中的 net_variant / TCN 宽度等
# 覆盖当前配置，避免用默认 MLP 去加载 tcn_shared_film checkpoint。
UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True
# 可选：仅对 unit=1 追加 TV 微调（在预训练后对单个机组继续训练，避免拖累其它机组）
UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS = 0
UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT = 0.02
# 自动挑选“变化不足”的机组做 TV 微调
UNIT_PREDICTOR_AUTO_EXTRA_TV = False
UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS = 160
UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD = 0.02
UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT = 0.02
UNIT_PREDICTOR_FINETUNE_LR = 1e-5        # BCD 阶段对 predictor 的微调学习率
UNIT_PREDICTOR_WEIGHT_DECAY = 1e-4
MAX_ITER = 300             # backward-compatible shared fallback
BCD_MAX_ITER = MAX_ITER
SUBPROBLEM_MAX_ITER = MAX_ITER
NN_EPOCHS = 4
UNIT_IDS = [1]              # None = 所有机组；或如 [0, 1, 2]
FP_TEST_SAMPLES = 3
# Windows 本地默认使用更保守的线程并发，优先稳定性和内存占用。
LOCAL_CPU_COUNT = max(1, os.cpu_count() or 1)
N_WORKERS_BCD = min(4, max(1, LOCAL_CPU_COUNT // 2))
# subproblem 并行参数（建议只开一层：要么机组级，要么样本级）
# - N_WORKERS_UNIT:   机组级并行（跨进程），>1 时启用 `train_all_surrogates_parallel`
# - N_WORKERS_SAMPLE: 样本级并行（机组内），>1 时使用 `ParallelSubproblemSurrogateTrainer`
N_WORKERS_UNIT = 1
N_WORKERS_SAMPLE = min(6, LOCAL_CPU_COUNT, max(2, LOCAL_CPU_COUNT // 2))
# 兼容旧变量名（历史配置仍可用）
N_WORKERS_SUBPROBLEM = N_WORKERS_SAMPLE
BCD_LP_BACKEND = 'gurobi'   # 'gurobi' / 'cvxpy_highs'
BCD_GUROBI_THREADS = None   # None = let agent decide / backend default
BCD_GUROBI_LP_METHOD = -1   # -1=auto, 0=primal simplex, 1=dual simplex, 2=barrier(IPM)
BCD_HIGHS_THREADS = 1       # HiGHS threads per BCD LP solve (>1 safe only for thread-pool path)
SUBPROBLEM_LP_BACKEND = 'cvxpy_highs'   # 'gurobi' / 'cvxpy_highs'
SUBPROBLEM_SOLVER_BIN_PREPEND = None
JOINT_MAX_ITER = 10
JOINT_NN_EPOCHS = 5
JOINT_SURR_NN_EPOCHS = 5
JOINT_DUAL_DECAY_ROUND = 0
JOINT_DUAL_SIGN_RELAX_INTERVAL = 4
ACTIVE_SETS_FILE = "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"  # None = 自动查找最新
BCD_MODEL_FILE = None
BCD_CONTINUE_TRAINING = False
SURROGATE_MODEL_DIR = None
SURROGATE_CONTINUE_TRAINING = False
SURROGATE_SKIP_EXISTING_UNITS = False
# surrogate 模式：为 True 时只训练/微调 dual_predictor（写出 dual_predictor.pth），跳过各机组子问题代理
SURROGATE_DUAL_PREDICTOR_ONLY = False
# 若非空，则 metrics JSON 文件名为 training_metrics_<CASE_NAME>_<METRICS_NAME_TAG>_<timestamp>.json
METRICS_NAME_TAG: str = ""
SPARSE_TOP_K_VARIABLES = 20
SPARSE_MAX_GROUPS = 5
SPARSE_GROUP_SIZE = 3
SURROGATE_CONSTRAINT_STRATEGY = 'all_templates_sign4_plus_single'  # 'sensitive' / 'all' / 'all_templates_sign4' / 'all_single_time' / 'all_templates_sign4_plus_single'
BCD_LAMBDA_INIT_STRATEGY = 'ed_on_x_opt'   # 'lp_relaxation' / 'ed_on_x_opt'
THETA_HOT_START_STRATEGY = 'dcpf_relative'   # 'dcpf_relative' / 'gaussian'
ZETA_HOT_START_STRATEGY = 'zero'             # 'zero' / 'gaussian'
THETA_GAUSSIAN_STD = 0.01
ZETA_GAUSSIAN_STD = 0.01
BCD_ENABLE_DROPOUT_DURING_NN_TRAINING = True
BCD_NN_SIZE = 'medium'   # 'tiny' / 'small' / 'medium' / 'large'
BCD_NN_BATCH_STRATEGY = 'full-batch'   # 'full-batch' / 'mini-batch'
BCD_NN_BATCH_SIZE = 4
BCD_NN_SHUFFLE = True
BCD_NN_LR = 5e-5
BCD_DIRECT_NN_EPOCHS = 20
BCD_DIRECT_NN_LR = 2e-3
BCD_DIRECT_NN_ETA_MIN_RATIO = 0.20
BCD_DIRECT_NN_BATCH_STRATEGY = "mini-batch"
BCD_DIRECT_NN_BATCH_SIZE = 16
BCD_DIRECT_NN_SHUFFLE = True
BCD_DIRECT_NN_LOSS = "mse"
BCD_DIRECT_NN_GRAD_CLIP = 3.0
BCD_DIRECT_NN_LOG_INTERVAL = 10
BCD_DIRECT_NN_FEATURE_NOISE_STD = 0.005
BCD_DIRECT_NN_ADAM_WD = 1e-5
BCD_DIRECT_NN_TARGET_BLEND = 0.75
BCD_DIRECT_NN_DUAL_EQ_WEIGHT = 1.0
BCD_DIRECT_NN_ACTIVE_OPT_WEIGHT = 0.8
BCD_DIRECT_NN_INACTIVE_MARGIN_WEIGHT = 0.08
BCD_DIRECT_NN_INACTIVE_MARGIN = 0.15
BCD_DIRECT_NN_ANCHOR_WEIGHT = 0.18
BCD_DIRECT_NN_THETA_ANCHOR_WEIGHT = 0.22
BCD_DIRECT_NN_ZETA_ANCHOR_WEIGHT = 0.22
BCD_DIRECT_NN_RHS_ANCHOR_WEIGHT = 0.12
BCD_DIRECT_NN_DUAL_ACTIVE_THRESHOLD = 1e-7
BCD_DIRECT_NN_PROXY_KKT_LOSS_WEIGHT = 0.008
BCD_DIRECT_NN_COEFF_BOUND = 2.0
BCD_DIRECT_NN_RHS_BOUND = 3.0
BCD_RHO_PRIMAL_INIT = 1e-3
BCD_RHO_DUAL_INIT = 1e-3
BCD_RHO_DUAL_PG_INIT = 1
BCD_RHO_DUAL_X_INIT = 1e-3
BCD_RHO_DUAL_COC_INIT = 1e1
BCD_RHO_BINARY_INIT = 1.0
BCD_RHO_OPT_INIT = 1e-3
BCD_LOSS_RATIO_PRIMAL = 1.0
BCD_LOSS_RATIO_DUAL_X = 2e0
BCD_LOSS_RATIO_OPT = 1.0
BCD_LOSS_RATIO_REG = 1.0
BCD_NN_SMOOTH_ABS_EPS = 1e-5
BCD_RESTORE_RHO_FROM_CHECKPOINT = False
BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = 20
BCD_THETA_TRAINING_STAGES = None
BCD_THETA_CURRICULUM_ROUNDS = 0
BCD_THETA_INITIAL_SCALE = 1.0
BCD_THETA_FINAL_SCALE = 1.0
BCD_THETA_CURRICULUM_DELAY_ROUNDS = 0
BCD_USE_UNIT_PREDICTOR = USE_UNIT_PREDICTOR
BCD_UNIT_PREDICTOR_WARMUP_ROUNDS = round(BCD_MAX_ITER * 0.10)
BCD_THETA_CONSTRAINT_DELAY_ROUNDS = round(BCD_MAX_ITER * 0.10)
BCD_GAMMA_BASE = 1e-3
DUAL_DECAY_ROUND = round(BCD_MAX_ITER/8)
BCD_DUAL_SIGN_RELAX_INTERVAL = 4
BCD_MU_DUAL_FLOOR_INIT = 0
BCD_ITA_DUAL_FLOOR_INIT = 0
BCD_ZETA_ITA_CAP_PENALTY_WEIGHT = 0.0
BCD_ZETA_ITA_CAP_INITIAL_WEIGHT = None
BCD_ZETA_ITA_CAP_FINAL_WEIGHT = None
BCD_ZETA_ITA_CAP_INITIAL = None
BCD_ZETA_ITA_CAP_FINAL = None
BCD_ZETA_ITA_CAP_START_ROUND = 0
BCD_ZETA_ITA_CAP_END_ROUND = 0
SUBPROBLEM_RHO_PRIMAL_INIT = 1e-1
SUBPROBLEM_RHO_DUAL_INIT = 1e-3
SUBPROBLEM_RHO_DUAL_PG_INIT = 1e-1
SUBPROBLEM_RHO_DUAL_X_INIT = 1e-1
SUBPROBLEM_RHO_DUAL_COC_INIT = 1e1
SUBPROBLEM_RHO_BINARY_INIT = 1.0
SUBPROBLEM_RHO_BINARY_MAX = 1e4
SUBPROBLEM_RHO_OPT_INIT = 1e-1
SUBPROBLEM_LOSS_RATIO_PRIMAL = 1.0
SUBPROBLEM_LOSS_RATIO_DUAL_PG = 1.0
SUBPROBLEM_LOSS_RATIO_DUAL_X = 2e0
SUBPROBLEM_NN_SMOOTH_ABS_EPS = 1e-5
SUBPROBLEM_NN_DUAL_TERM_INTERVAL = 1  # None=NN训练时完全屏蔽dual项；k=每k轮NN才启用一次dual项
SUBPROBLEM_LOSS_RATIO_OPT = 1.0
SUBPROBLEM_LOSS_RATIO_REG = 1.0
SUBPROBLEM_GAMMA_BASE = 1e-3
SUBPROBLEM_MU_DUAL_FLOOR_INIT = 2
SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = round(SUBPROBLEM_MAX_ITER/4)
SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = round(SUBPROBLEM_MAX_ITER/2)
# predictor override 热身轮数：前 N 轮禁用 single-time 段覆写，让 surrogate_net 独立 bootstrap mu
# 设为 0 保持原有行为（Plan B 实现前该参数由训练流程读取但尚未透传给 trainer）
SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = round(SUBPROBLEM_MAX_ITER * 0.10)
SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS = 0
SUBPROBLEM_SIGN4_INITIAL_SCALE = 1.0
SUBPROBLEM_SIGN4_FINAL_SCALE = 1.0
SUBPROBLEM_SIGN4_DELAY_ROUNDS = 0
SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = None  # None=auto: enable for sign4 strategies
SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS = 1e-6
SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE = "sign4_only"
SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR = 1e-9
SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL = 20
SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS = 3
SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE = 0.72
SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS = 0
SUBPROBLEM_NN_BATCH_STRATEGY = 'full-batch'   # 'full-batch' / 'mini-batch'
SUBPROBLEM_NN_SIZE = 'medium'   # 'tiny' / 'small' / 'medium' / 'large'
SUBPROBLEM_C_PG_NN_SIZE = 'medium'   # 'tiny' / 'small' / 'medium' / 'large'
SUBPROBLEM_NN_BATCH_SIZE = 4
SUBPROBLEM_NN_SHUFFLE = True
SUBPROBLEM_NN_LR = 5e-4
# NN-main（可微 KKT）细化：与 direct-NN-main 相同每轮 BCD 内 CosineAnnealingLR；外循环基准 lr 按
# nn_main_lr_late_scale 递减，类比子问题 LP 中 rho 累积后的局部精炼（见 subproblem_lp_solver 加权目标）。
SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO = 0.08
SUBPROBLEM_NN_MAIN_LR_LATE_SCALE = 0.42
SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY = 1e-4
SUBPROBLEM_NN_MAIN_GRAD_CLIP = 0.85
# 仅缩小 NN-main（KKT）步长，direct-NN-main 仍用 SUBPROBLEM_NN_LR
SUBPROBLEM_NN_MAIN_KKT_LR_SCALE = 1.0
# True：子问题 LP/BCD 中忽略启停货币成本，并 **忽略无负荷（固定）发电成本项 b*x**（gencost[:,-1]）。
SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
SUBPROBLEM_X_COST_NN_LR = 1e-5
SUBPROBLEM_PG_COST_NN_EPOCHS = 64
SUBPROBLEM_PG_COST_START_ROUND = max(0, SUBPROBLEM_MAX_ITER // 4)
SUBPROBLEM_PG_COST_SCALE_MULTIPLIER = 2
SUBPROBLEM_PG_COST_LR = 1e-4
SUBPROBLEM_PG_COST_SURR_LR = 2e-4
SUBPROBLEM_PG_COST_REG_DEADBAND = 0.35
SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT = 1.0
SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS = 1e-5
SUBPROBLEM_PG_COST_BATCH_STRATEGY: str | None = "mini-batch"
SUBPROBLEM_PG_COST_BATCH_SIZE: int | None = 32
SUBPROBLEM_PG_COST_SHUFFLE: bool | None = True
SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS: bool = True
SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER: float = 1.0
SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP: float = 10.0
# None：由 SubproblemSurrogateTrainer 在 n_samples==1 时自动收紧 c_pg 正则与 Adam weight decay
SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE: float | None = None
SUBPROBLEM_PG_COST_C_PG_ADAM_WD: float | None = None

# NN_subproblem 新训练策略：先用解析/KKT proxy 直接目标预训练，再进入原可微 KKT loss 微调。
# epochs=0 可关闭对应 direct-target 阶段。
SUBPROBLEM_MAIN_DIRECT_EPOCHS: int = 120
SUBPROBLEM_MAIN_DIRECT_LR: float = 2e-3
SUBPROBLEM_MAIN_DIRECT_COST_LR: float = 8e-4
SUBPROBLEM_MAIN_DIRECT_ETA_MIN_RATIO: float = 0.20
SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY: str = "mini-batch"
SUBPROBLEM_MAIN_DIRECT_BATCH_SIZE: int = 16
SUBPROBLEM_MAIN_DIRECT_SHUFFLE: bool = True
SUBPROBLEM_MAIN_DIRECT_LOSS: str = "mse"
SUBPROBLEM_MAIN_DIRECT_GRAD_CLIP: float = 3.0
SUBPROBLEM_MAIN_DIRECT_LOG_INTERVAL: int = 10
SUBPROBLEM_MAIN_DIRECT_TARGET_CHECK_INTERVAL: int = 5
SUBPROBLEM_MAIN_DIRECT_MAE_TARGET: float | None = 0.076
SUBPROBLEM_MAIN_DIRECT_FEATURE_NOISE_STD: float = 0.005
SUBPROBLEM_MAIN_DIRECT_ADAM_WD: float = 1e-5
SUBPROBLEM_MAIN_DIRECT_TARGET_BLEND: float = 0.75
SUBPROBLEM_MAIN_DIRECT_DUAL_EQ_WEIGHT: float = 1.0
SUBPROBLEM_MAIN_DIRECT_ACTIVE_OPT_WEIGHT: float = 0.8
SUBPROBLEM_MAIN_DIRECT_INACTIVE_MARGIN_WEIGHT: float = 0.08
SUBPROBLEM_MAIN_DIRECT_INACTIVE_MARGIN: float = 0.15
SUBPROBLEM_MAIN_DIRECT_ANCHOR_WEIGHT: float = 0.18
SUBPROBLEM_MAIN_DIRECT_COEFF_ANCHOR_WEIGHT: float = 0.22
SUBPROBLEM_MAIN_DIRECT_DELTA_ANCHOR_WEIGHT: float = 0.12
SUBPROBLEM_MAIN_DIRECT_COST_ANCHOR_WEIGHT: float = 0.10
SUBPROBLEM_MAIN_DIRECT_MU_ACTIVE_THRESHOLD: float = 1e-7
SUBPROBLEM_MAIN_DIRECT_COEFF_LOSS_WEIGHT: float = 1.0
SUBPROBLEM_MAIN_DIRECT_DELTA_LOSS_WEIGHT: float = 0.65
SUBPROBLEM_MAIN_DIRECT_COST_LOSS_WEIGHT: float = 0.85
SUBPROBLEM_MAIN_DIRECT_PROXY_KKT_LOSS_WEIGHT: float = 0.008

SUBPROBLEM_C_PG_DIRECT_EPOCHS: int = 420
SUBPROBLEM_C_PG_DIRECT_LR: float = 6e-3
SUBPROBLEM_C_PG_DIRECT_ETA_MIN_RATIO: float = 0.25
SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY: str = "mini-batch"
SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE: int = 16
SUBPROBLEM_C_PG_DIRECT_SHUFFLE: bool = True
SUBPROBLEM_C_PG_DIRECT_LOSS: str = "mse"
SUBPROBLEM_C_PG_DIRECT_OBJ_DUAL_PG_TARGET: float | None = 45.0
SUBPROBLEM_C_PG_DIRECT_BETA: float = 1.0
SUBPROBLEM_C_PG_DIRECT_GRAD_CLIP: float = 5.0
SUBPROBLEM_C_PG_DIRECT_LOG_INTERVAL: int = 25
SUBPROBLEM_C_PG_DIRECT_TARGET_CHECK_INTERVAL: int = 25
SUBPROBLEM_C_PG_DIRECT_FEATURE_NOISE_STD: float = 0.005
SUBPROBLEM_C_PG_DIRECT_ADAM_WD: float = 1e-5

# 迭代间输出差异正则：温和抑制相邻 BCD 轮次输出跳变；可按算例继续调节
BCD_ITER_DELTA_REG_WEIGHT = 1e-4
BCD_ITER_DELTA_REG_DEADBAND = 0.05
SUBPROBLEM_ITER_DELTA_REG_WEIGHT = 5e-5
SUBPROBLEM_ITER_DELTA_REG_DEADBAND = 0.10
BCD_PG_BLOCK_PROX_WEIGHT = 0
BCD_DUAL_BLOCK_PROX_WEIGHT = 0
SUBPROBLEM_PG_BLOCK_PROX_WEIGHT = 0
SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT = 0
SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT = 0.0
SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT = None
SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT = None
SUBPROBLEM_SINGLE_MU_CAP_INITIAL = None
SUBPROBLEM_SINGLE_MU_CAP_FINAL = None
SUBPROBLEM_SINGLE_MU_CAP_START_ROUND = 0
SUBPROBLEM_SINGLE_MU_CAP_END_ROUND = 0
SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP = False

SUBPROBLEM_SOLVER_PATH_PREPENDED = False
def _bootstrap_runtime_environment() -> None:
    """Run-once bootstrap for the main process only.

    IMPORTANT: keep this out of module top-level so Windows multiprocessing
    (spawn) won't repeatedly run dependency checks in every worker process.
    """
    global SUBPROBLEM_SOLVER_PATH_PREPENDED
    if SUBPROBLEM_SOLVER_BIN_PREPEND:
        prepend_path = str(Path(SUBPROBLEM_SOLVER_BIN_PREPEND))
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = prepend_path + (os.pathsep + current_path if current_path else "")
        SUBPROBLEM_SOLVER_PATH_PREPENDED = True

    if not check_and_install_dependencies(SUBPROBLEM_LP_BACKEND):
        sys.exit(1)

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# 添加 src/ 到模块搜索路径
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    import pypower.case118
    from uc_NN_subproblem import (
        train_dual_predictor_from_data,
        train_unit_predictor_from_data,
        SingleUnitBinaryPredictorTrainer,
        SubproblemSurrogateTrainer,
        ActiveSetReader,
        load_trained_models,
        resolve_constraint_offsets_from_trainer,
        normalize_constraint_generation_strategy,
    )
    from uc_NN_subproblem_parallel import ParallelSubproblemSurrogateTrainer
    from case_registry import get_case_ppc
    from mti118_data_loader import load_case118_ppc_with_mti_limits
    from dataset_json_utils import load_v3_active_set_json
    from subproblem_lp_solver import get_cvxpy_highs_status, is_lp_backend_available
    from training_logger import TrainingLogger
    from training_visualizer import TrainingVisualizer
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

# BCD 模式额外导入
if MODE in ('bcd', 'sparse', 'both'):
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
        from uc_NN_BCD_parallel import ParallelAgent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)

# 可行性泵额外导入
if RUN_FP:
    try:
        from feasibility_pump import recover_integer_solution
    except ImportError as e:
        print(f"feasibility_pump 模块导入失败: {e}")
        sys.exit(1)

# ──────────────────────── 工具函数 / 辅助函数 ────────────────────────

def _looks_mojibake(msg: object) -> bool:
    text = str(msg)
    mojibake_markers = ("æ", "ç", "é", "è", "å", "œ", "Ž", "ˆ", "™", "€", "‚")
    return any(marker in text for marker in mojibake_markers)


def log(msg):
    msg = str(msg)
    if not msg.isascii():
        return
    if _looks_mojibake(msg):
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


SUPPORTED_NN_SIZE_OPTIONS = ('tiny', 'small', 'medium', 'large')
BCD_NN_HIDDEN_DIM_OPTIONS = {
    'tiny': [8, 16],
    'small': [16, 32],
    'medium': [24, 48],
    'large': [32, 64],
}
SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS = {
    'tiny': [32, 32],
    'small': [64, 64],
    'medium': [96, 96],
    'large': [128, 128],
}


def normalize_nn_size_option(size_option: str, label: str) -> str:
    resolved = str(size_option).strip().lower()
    if resolved not in SUPPORTED_NN_SIZE_OPTIONS:
        raise ValueError(
            f"{label} must be one of {SUPPORTED_NN_SIZE_OPTIONS}, got {size_option!r}"
        )
    return resolved


def resolve_nn_hidden_dims(size_option: str, dim_options: dict[str, list[int]], label: str) -> tuple[str, list[int]]:
    resolved_size = normalize_nn_size_option(size_option, label)
    return resolved_size, list(dim_options[resolved_size])


def build_subproblem_main_direct_train_config() -> dict:
    return {
        "direct_epochs": SUBPROBLEM_MAIN_DIRECT_EPOCHS,
        "direct_lr": SUBPROBLEM_MAIN_DIRECT_LR,
        "direct_cost_lr": SUBPROBLEM_MAIN_DIRECT_COST_LR,
        "direct_eta_min_ratio": SUBPROBLEM_MAIN_DIRECT_ETA_MIN_RATIO,
        "direct_batch_strategy": SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY,
        "direct_batch_size": SUBPROBLEM_MAIN_DIRECT_BATCH_SIZE,
        "direct_shuffle": SUBPROBLEM_MAIN_DIRECT_SHUFFLE,
        "direct_loss": SUBPROBLEM_MAIN_DIRECT_LOSS,
        "direct_grad_clip": SUBPROBLEM_MAIN_DIRECT_GRAD_CLIP,
        "direct_log_interval": SUBPROBLEM_MAIN_DIRECT_LOG_INTERVAL,
        "direct_target_check_interval": SUBPROBLEM_MAIN_DIRECT_TARGET_CHECK_INTERVAL,
        "direct_mae_target": SUBPROBLEM_MAIN_DIRECT_MAE_TARGET,
        "feature_noise_std": SUBPROBLEM_MAIN_DIRECT_FEATURE_NOISE_STD,
        "adam_weight_decay": SUBPROBLEM_MAIN_DIRECT_ADAM_WD,
        "target_blend": SUBPROBLEM_MAIN_DIRECT_TARGET_BLEND,
        "dual_eq_weight": SUBPROBLEM_MAIN_DIRECT_DUAL_EQ_WEIGHT,
        "active_opt_weight": SUBPROBLEM_MAIN_DIRECT_ACTIVE_OPT_WEIGHT,
        "inactive_margin_weight": SUBPROBLEM_MAIN_DIRECT_INACTIVE_MARGIN_WEIGHT,
        "inactive_margin": SUBPROBLEM_MAIN_DIRECT_INACTIVE_MARGIN,
        "anchor_weight": SUBPROBLEM_MAIN_DIRECT_ANCHOR_WEIGHT,
        "coeff_anchor_weight": SUBPROBLEM_MAIN_DIRECT_COEFF_ANCHOR_WEIGHT,
        "delta_anchor_weight": SUBPROBLEM_MAIN_DIRECT_DELTA_ANCHOR_WEIGHT,
        "cost_anchor_weight": SUBPROBLEM_MAIN_DIRECT_COST_ANCHOR_WEIGHT,
        "mu_active_threshold": SUBPROBLEM_MAIN_DIRECT_MU_ACTIVE_THRESHOLD,
        "coeff_loss_weight": SUBPROBLEM_MAIN_DIRECT_COEFF_LOSS_WEIGHT,
        "delta_loss_weight": SUBPROBLEM_MAIN_DIRECT_DELTA_LOSS_WEIGHT,
        "cost_loss_weight": SUBPROBLEM_MAIN_DIRECT_COST_LOSS_WEIGHT,
        "proxy_kkt_loss_weight": SUBPROBLEM_MAIN_DIRECT_PROXY_KKT_LOSS_WEIGHT,
    }


def build_subproblem_c_pg_direct_train_config() -> dict:
    return {
        "direct_epochs": SUBPROBLEM_C_PG_DIRECT_EPOCHS,
        "direct_lr": SUBPROBLEM_C_PG_DIRECT_LR,
        "direct_eta_min_ratio": SUBPROBLEM_C_PG_DIRECT_ETA_MIN_RATIO,
        "direct_batch_strategy": SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY,
        "direct_batch_size": SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE,
        "direct_shuffle": SUBPROBLEM_C_PG_DIRECT_SHUFFLE,
        "direct_loss": SUBPROBLEM_C_PG_DIRECT_LOSS,
        "direct_obj_dual_pg_target": SUBPROBLEM_C_PG_DIRECT_OBJ_DUAL_PG_TARGET,
        "direct_beta": SUBPROBLEM_C_PG_DIRECT_BETA,
        "direct_grad_clip": SUBPROBLEM_C_PG_DIRECT_GRAD_CLIP,
        "direct_log_interval": SUBPROBLEM_C_PG_DIRECT_LOG_INTERVAL,
        "direct_target_check_interval": SUBPROBLEM_C_PG_DIRECT_TARGET_CHECK_INTERVAL,
        "feature_noise_std": SUBPROBLEM_C_PG_DIRECT_FEATURE_NOISE_STD,
        "adam_weight_decay": SUBPROBLEM_C_PG_DIRECT_ADAM_WD,
    }


def build_bcd_direct_train_config() -> dict:
    return {
        "direct_epochs": BCD_DIRECT_NN_EPOCHS,
        "direct_lr": BCD_DIRECT_NN_LR,
        "direct_eta_min_ratio": BCD_DIRECT_NN_ETA_MIN_RATIO,
        "direct_batch_strategy": BCD_DIRECT_NN_BATCH_STRATEGY,
        "direct_batch_size": BCD_DIRECT_NN_BATCH_SIZE,
        "direct_shuffle": BCD_DIRECT_NN_SHUFFLE,
        "direct_loss": BCD_DIRECT_NN_LOSS,
        "direct_grad_clip": BCD_DIRECT_NN_GRAD_CLIP,
        "direct_log_interval": BCD_DIRECT_NN_LOG_INTERVAL,
        "feature_noise_std": BCD_DIRECT_NN_FEATURE_NOISE_STD,
        "adam_weight_decay": BCD_DIRECT_NN_ADAM_WD,
        "target_blend": BCD_DIRECT_NN_TARGET_BLEND,
        "dual_eq_weight": BCD_DIRECT_NN_DUAL_EQ_WEIGHT,
        "active_opt_weight": BCD_DIRECT_NN_ACTIVE_OPT_WEIGHT,
        "inactive_margin_weight": BCD_DIRECT_NN_INACTIVE_MARGIN_WEIGHT,
        "inactive_margin": BCD_DIRECT_NN_INACTIVE_MARGIN,
        "anchor_weight": BCD_DIRECT_NN_ANCHOR_WEIGHT,
        "theta_anchor_weight": BCD_DIRECT_NN_THETA_ANCHOR_WEIGHT,
        "zeta_anchor_weight": BCD_DIRECT_NN_ZETA_ANCHOR_WEIGHT,
        "rhs_anchor_weight": BCD_DIRECT_NN_RHS_ANCHOR_WEIGHT,
        "dual_active_threshold": BCD_DIRECT_NN_DUAL_ACTIVE_THRESHOLD,
        "proxy_kkt_loss_weight": BCD_DIRECT_NN_PROXY_KKT_LOSS_WEIGHT,
        "coeff_bound": BCD_DIRECT_NN_COEFF_BOUND,
        "rhs_bound": BCD_DIRECT_NN_RHS_BOUND,
    }


def load_json_data(data_file: Path) -> list:
    """加载 JSON 数据文件并规范化为 v3 所需格式。"""
    return load_v3_active_set_json(data_file, announce=log)


def inject_bcd_lambda(all_samples: list, bcd_lambdas: list, T: int) -> None:
    """将 BCD 求解的全局对偶变量注入样本，供 dual predictor 使用。

    Args:
        all_samples: v3 格式样本列表，注入后每条样本含 'lambda' 字段。
        bcd_lambdas: Agent_NN_BCD.lambda_ 列表，每项为全局对偶变量 dict。
        T: 时段数，用于生成零向量默认值。
    """
    for i, sample in enumerate(all_samples):
        if i >= len(bcd_lambdas):
            break
        lam_dict = bcd_lambdas[i]
        sample['lambda'] = {
            'lambda_power_balance': np.asarray(
                lam_dict.get('lambda_power_balance', np.zeros(T)),
                dtype=float,
            ).tolist(),
            'lambda_dcpf_upper': np.asarray(
                lam_dict.get('lambda_dcpf_upper', np.zeros((0, T))),
                dtype=float,
            ).tolist(),
            'lambda_dcpf_lower': np.asarray(
                lam_dict.get('lambda_dcpf_lower', np.zeros((0, T))),
                dtype=float,
            ).tolist(),
        }
        if 'lambda_pg_electricity_price' in lam_dict:
            sample['lambda_pg_electricity_price'] = np.asarray(
                lam_dict['lambda_pg_electricity_price'],
                dtype=float,
            ).tolist()


def _file_timestamp_key(path: Path) -> str:
    """从文件名中提取 YYYYMMDD_HHMMSS 时间戳作为排序键；若无则回退到文件修改时间。"""
    m = re.search(r'(\d{8}_\d{6})', path.stem)
    if m:
        return m.group(1)
    return str(path.stat().st_mtime)


def pick_data_file(result_dir: Path, case_name: str) -> Path:
    """按优先级查找最合适的数据文件，优先选择时间戳最新的文件。"""
    specific = sorted(result_dir.glob(f'active_sets_{case_name}_*.json'),
                      key=_file_timestamp_key)
    if specific:
        return specific[-1]
    any_files = sorted(result_dir.glob('active_sets_*.json'),
                       key=_file_timestamp_key)
    if any_files:
        log(f"未找到 {case_name} 专属文件，使用: {any_files[-1].name}")
        return any_files[-1]
    return None


def resolve_existing_path(path_value: str | None, label: str) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def create_bcd_agent(ppc, all_samples, T_DELTA, *,
                     n_workers: int = 4,
                     lp_backend: str = 'gurobi',
                     gurobi_threads: int | None = None,
                     external_sparse_templates=None,
                     lambda_init_strategy: str = 'lp_relaxation',
                     max_theta_constraints_per_time_slot: int = 10,
                     theta_hot_start_strategy: str = 'dcpf_relative',
                     zeta_hot_start_strategy: str = 'zero',
                     theta_gaussian_std: float = 0.01,
                     zeta_gaussian_std: float = 0.01,
                     enable_dropout_during_nn_training: bool = True,
                     rho_primal_init: float = 1e-2,
                     rho_dual_init: float = 1e-2,
                     rho_dual_pg_init: float | None = None,
                     rho_dual_x_init: float | None = None,
                     rho_dual_coc_init: float | None = None,
                     rho_binary_init: float = 1.0,
                     rho_opt_init: float = 1e-2,
                     loss_ratio_primal: float = 1.0,
                     loss_ratio_dual_x: float = 1.0,
                     loss_ratio_opt: float = 1.0,
                     loss_ratio_reg: float = 1.0,
                     gamma_base: float = 1e-2,
                     mu_dual_floor_init: float = 0.1,
                     ita_dual_floor_init: float = 0.1,
                     nn_hidden_dims: list[int] | None = None,
                     nn_batch_strategy: str = 'full-batch',
                     nn_batch_size: int = 4,
                     nn_shuffle: bool = True,
                     nn_learning_rate: float = 5e-5,
                     nn_smooth_abs_eps: float = BCD_NN_SMOOTH_ABS_EPS,
                     direct_train_config: dict | None = None,
                     pg_block_prox_weight: float = BCD_PG_BLOCK_PROX_WEIGHT,
                     dual_block_prox_weight: float = BCD_DUAL_BLOCK_PROX_WEIGHT,
                     iter_delta_reg_weight: float = BCD_ITER_DELTA_REG_WEIGHT,
                     iter_delta_reg_deadband: float = BCD_ITER_DELTA_REG_DEADBAND):
    ensure_bcd_modules_imported()
    if external_sparse_templates is not None and n_workers > 1:
        log("璀﹀憡: external_sparse_templates 褰撳墠浠呮敮鎸佷覆琛?Agent_NN_BCD锛屽皢蹇界暐 n_workers > 1")
        n_workers = 1

    agent_kwargs = dict(
        lp_backend=lp_backend,
        gurobi_threads=gurobi_threads,
        lambda_init_strategy=lambda_init_strategy,
        max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
        theta_hot_start_strategy=theta_hot_start_strategy,
        zeta_hot_start_strategy=zeta_hot_start_strategy,
        theta_gaussian_std=theta_gaussian_std,
        zeta_gaussian_std=zeta_gaussian_std,
        enable_dropout_during_nn_training=enable_dropout_during_nn_training,
        rho_primal_init=rho_primal_init,
        rho_dual_init=rho_dual_init,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        rho_binary_init=rho_binary_init,
        rho_opt_init=rho_opt_init,
        gamma_base=gamma_base,
        mu_dual_floor_init=mu_dual_floor_init,
        ita_dual_floor_init=ita_dual_floor_init,
        nn_hidden_dims=nn_hidden_dims,
        nn_learning_rate=nn_learning_rate,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        nn_smooth_abs_eps=nn_smooth_abs_eps,
        direct_train_config=direct_train_config,
        pg_block_prox_weight=pg_block_prox_weight,
        dual_block_prox_weight=dual_block_prox_weight,
        loss_ratio_primal=loss_ratio_primal,
        loss_ratio_dual_x=loss_ratio_dual_x,
        loss_ratio_opt=loss_ratio_opt,
        loss_ratio_reg=loss_ratio_reg,
        iter_delta_reg_weight=iter_delta_reg_weight,
        iter_delta_reg_deadband=iter_delta_reg_deadband,
    )
    if external_sparse_templates is not None:
        agent_kwargs['external_sparse_templates'] = external_sparse_templates

    if n_workers <= 1:
        log("浣跨敤涓茶 Agent_NN_BCD")
        return Agent_NN_BCD(ppc, all_samples, T_DELTA, **agent_kwargs)

    log(f"浣跨敤骞惰 ParallelAgent_NN_BCD (n_workers={n_workers})")
    agent_kwargs['n_workers'] = n_workers
    return ParallelAgent_NN_BCD(ppc, all_samples, T_DELTA, **agent_kwargs)


def create_subproblem_trainer(ppc, all_samples, T_DELTA, unit_id: int, *,
                              n_workers: int = 4,
                              lambda_predictor=None,
                              lp_backend: str = 'gurobi',
                              constraint_generation_strategy: str = 'sensitive',
                              rho_primal_init: float = 1e-3,
                              rho_dual_init: float = 1e-3,
                              rho_dual_pg_init: float | None = None,
                              rho_dual_x_init: float | None = None,
                              rho_dual_coc_init: float | None = None,
                              rho_binary_init: float = 1.0,
                              rho_binary_max: float = SUBPROBLEM_RHO_BINARY_MAX,
                              rho_opt_init: float = 1e-3,
                              loss_ratio_primal: float = 1.0,
                              loss_ratio_dual_pg: float = 1.0,
                              loss_ratio_dual_x: float = 1.0,
                              nn_dual_term_interval: int | None = 1,
                              loss_ratio_opt: float = 1.0,
                              loss_ratio_reg: float = 1.0,
                              subproblem_gamma_base: float = 1e-3,
                              mu_lower_bound_init: float = 0.1,
                              mu_individual_lower_bound_round: int = 3,
                              mu_group_lower_bound_round: int = 50,
                              mu_signed_round_interval: int | None = None,
                              mu_sign_hysteresis_rounds: int = 2,
                              mu_sign_flip_min_share: float = 0.67,
                              x_bound_dual_zero_rounds: int = 0,
                              subproblem_nn_hidden_dims: list[int] | None = None,
                              subproblem_c_pg_nn_hidden_dims: list[int] | None = None,
                              subproblem_nn_batch_strategy: str = 'full-batch',
                              subproblem_nn_batch_size: int = 4,
                              subproblem_nn_shuffle: bool = True,
                              subproblem_nn_learning_rate: float = 1e-4,
                              subproblem_nn_main_eta_min_ratio: float = SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO,
                              subproblem_nn_main_lr_late_scale: float = SUBPROBLEM_NN_MAIN_LR_LATE_SCALE,
                              subproblem_nn_main_adam_weight_decay: float = SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY,
                              subproblem_nn_main_grad_clip: float = SUBPROBLEM_NN_MAIN_GRAD_CLIP,
                              subproblem_nn_main_kkt_lr_scale: float = SUBPROBLEM_NN_MAIN_KKT_LR_SCALE,
                              ignore_startup_shutdown_costs: bool = False,
                              subproblem_cost_learning_rate: float = 1e-5,
                              subproblem_nn_smooth_abs_eps: float = SUBPROBLEM_NN_SMOOTH_ABS_EPS,
                              pg_cost_nn_epochs: int | None = None,
                              pg_cost_start_round: int = 3,
                              pg_cost_scale_multiplier: float = 1.2,
                              pg_cost_lr: float = 2e-5,
                              pg_cost_surr_lr: float = 5e-5,
                              pg_cost_reg_deadband: float = 0.25,
                              pg_cost_softbound_weight: float = SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT,
                  pg_cost_smooth_abs_eps: float = 1e-6,
                  pg_cost_batch_strategy: str | None = None,
                  pg_cost_batch_size: int | None = None,
                  pg_cost_shuffle: bool | None = None,
                  pg_cost_use_sample_weights: bool = True,
                  pg_cost_sample_weight_power: float = 1.0,
                  pg_cost_sample_weight_clip: float = 10.0,
                  pg_cost_single_sample_reg_scale: float | None = None,
                  pg_cost_c_pg_adam_weight_decay: float | None = None,
                  main_direct_train_config: dict | None = None,
                  c_pg_direct_train_config: dict | None = None,
                  pg_block_prox_weight: float = SUBPROBLEM_PG_BLOCK_PROX_WEIGHT,
                              dual_block_prox_weight: float = SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT,
                              single_mu_cap_penalty_weight: float = SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT,
                              single_mu_cap_initial_weight: float | None = SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT,
                              single_mu_cap_final_weight: float | None = SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT,
                              single_mu_cap_initial: float | None = SUBPROBLEM_SINGLE_MU_CAP_INITIAL,
                              single_mu_cap_final: float | None = SUBPROBLEM_SINGLE_MU_CAP_FINAL,
                              single_mu_cap_start_round: int = SUBPROBLEM_SINGLE_MU_CAP_START_ROUND,
                              single_mu_cap_end_round: int = SUBPROBLEM_SINGLE_MU_CAP_END_ROUND,
                              iter_delta_reg_weight: float = SUBPROBLEM_ITER_DELTA_REG_WEIGHT,
                              iter_delta_reg_deadband: float = SUBPROBLEM_ITER_DELTA_REG_DEADBAND,
                              unit_predictor: 'SingleUnitBinaryPredictorTrainer | None' = None,
                              use_unit_predictor: bool = False,
                              predictor_warmup_rounds: int = 0,
                              sign4_curriculum_rounds: int = 0,
                              sign4_initial_scale: float = 1.0,
                              sign4_final_scale: float = 1.0,
                              sign4_delay_rounds: int = 0,
                              enable_surrogate_delta_reference_lift: bool | None = None,
                              surrogate_delta_reference_eps: float = 1e-6,
                              surrogate_delta_reference_scope: str = "sign4_only",
                              surrogate_delta_reference_min_abs_factor: float = 1e-9,
                              unit_predictor_finetune_lr: float = 1e-5,
                              unit_predictor_weight_decay: float = 1e-4,
                              case_name: str | None = None):
    trainer_kwargs = dict(
        lambda_predictor=lambda_predictor,
        lp_backend=lp_backend,
        constraint_generation_strategy=constraint_generation_strategy,
        rho_primal_init=rho_primal_init,
        rho_dual_init=rho_dual_init,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        rho_binary_init=rho_binary_init,
        rho_binary_max=rho_binary_max,
        rho_opt_init=rho_opt_init,
        loss_ratio_primal=loss_ratio_primal,
        loss_ratio_dual_pg=loss_ratio_dual_pg,
        loss_ratio_dual_x=loss_ratio_dual_x,
        nn_dual_term_interval=nn_dual_term_interval,
        loss_ratio_opt=loss_ratio_opt,
        loss_ratio_reg=loss_ratio_reg,
        gamma_base=subproblem_gamma_base,
                mu_lower_bound_init=mu_lower_bound_init,
                mu_individual_lower_bound_round=mu_individual_lower_bound_round,
                mu_group_lower_bound_round=mu_group_lower_bound_round,
                mu_signed_round_interval=mu_signed_round_interval,
                mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
                mu_sign_flip_min_share=mu_sign_flip_min_share,
                x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
                nn_hidden_dims=subproblem_nn_hidden_dims,
                pg_cost_hidden_dims=subproblem_c_pg_nn_hidden_dims,
        nn_batch_strategy=subproblem_nn_batch_strategy,
        nn_batch_size=subproblem_nn_batch_size,
        nn_shuffle=subproblem_nn_shuffle,
        nn_learning_rate=subproblem_nn_learning_rate,
        nn_main_eta_min_ratio=subproblem_nn_main_eta_min_ratio,
        nn_main_lr_late_scale=subproblem_nn_main_lr_late_scale,
        nn_main_adam_weight_decay=subproblem_nn_main_adam_weight_decay,
        nn_main_grad_clip=subproblem_nn_main_grad_clip,
        nn_main_kkt_lr_scale=subproblem_nn_main_kkt_lr_scale,
        ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
        cost_learning_rate=subproblem_cost_learning_rate,
        nn_smooth_abs_eps=subproblem_nn_smooth_abs_eps,
        pg_cost_nn_epochs=pg_cost_nn_epochs,
        pg_cost_start_round=pg_cost_start_round,
        pg_cost_scale_multiplier=pg_cost_scale_multiplier,
        pg_cost_lr=pg_cost_lr,
        pg_cost_surr_lr=pg_cost_surr_lr,
        pg_cost_reg_deadband=pg_cost_reg_deadband,
        pg_cost_softbound_weight=pg_cost_softbound_weight,
        pg_cost_smooth_abs_eps=pg_cost_smooth_abs_eps,
        pg_cost_batch_strategy=pg_cost_batch_strategy,
        pg_cost_batch_size=pg_cost_batch_size,
        pg_cost_shuffle=pg_cost_shuffle,
        pg_cost_use_sample_weights=pg_cost_use_sample_weights,
        pg_cost_sample_weight_power=pg_cost_sample_weight_power,
        pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
        pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
        pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
        main_direct_train_config=main_direct_train_config,
        c_pg_direct_train_config=c_pg_direct_train_config,
        pg_block_prox_weight=pg_block_prox_weight,
        dual_block_prox_weight=dual_block_prox_weight,
        single_mu_cap_penalty_weight=single_mu_cap_penalty_weight,
        single_mu_cap_initial_weight=single_mu_cap_initial_weight,
        single_mu_cap_final_weight=single_mu_cap_final_weight,
        single_mu_cap_initial=single_mu_cap_initial,
        single_mu_cap_final=single_mu_cap_final,
        single_mu_cap_start_round=single_mu_cap_start_round,
        single_mu_cap_end_round=single_mu_cap_end_round,
        iter_delta_reg_weight=iter_delta_reg_weight,
        iter_delta_reg_deadband=iter_delta_reg_deadband,
        unit_predictor=unit_predictor,
        use_unit_predictor=use_unit_predictor,
        predictor_warmup_rounds=predictor_warmup_rounds,
        sign4_curriculum_rounds=sign4_curriculum_rounds,
        sign4_initial_scale=sign4_initial_scale,
        sign4_final_scale=sign4_final_scale,
        sign4_delay_rounds=sign4_delay_rounds,
        enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
        surrogate_delta_reference_eps=surrogate_delta_reference_eps,
        surrogate_delta_reference_scope=surrogate_delta_reference_scope,
        surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
        unit_predictor_finetune_lr=unit_predictor_finetune_lr,
        unit_predictor_weight_decay=unit_predictor_weight_decay,
        case_name=case_name,
    )
    if n_workers <= 1:
        return SubproblemSurrogateTrainer(ppc, all_samples, T_DELTA, unit_id, **trainer_kwargs)
    trainer_kwargs['n_workers'] = n_workers
    return ParallelSubproblemSurrogateTrainer(ppc, all_samples, T_DELTA, unit_id, **trainer_kwargs)


# ──────────────────────── 模式实现 ────────────────────────

def run_surrogate(ppc, all_samples, T_DELTA, UNIT_IDS,
                  DUAL_EPOCHS, DUAL_BATCH_SIZE, SUBPROBLEM_MAX_ITER, NN_EPOCHS, save_dir,
                  n_workers: int = 4, logger: 'TrainingLogger | None' = None,
                  load_dir: str | None = None,
                  lp_backend: str = 'gurobi',
                  dual_batch_strategy: str = 'full-batch',
                  dual_shuffle: bool = True,
                  dual_learning_rate: float = 5e-4,
                  subproblem_nn_size: str = 'medium',
                  subproblem_nn_hidden_dims: list[int] | None = None,
                  subproblem_c_pg_nn_size: str = 'medium',
                  subproblem_c_pg_nn_hidden_dims: list[int] | None = None,
                  constraint_generation_strategy: str = 'sensitive',
                  rho_primal_init: float = 1e-3,
                  rho_dual_init: float = 1e-3,
                  rho_dual_pg_init: float | None = None,
                  rho_dual_x_init: float | None = None,
                  rho_dual_coc_init: float | None = None,
                  rho_binary_init: float = 1.0,
                  rho_binary_max: float = SUBPROBLEM_RHO_BINARY_MAX,
                  rho_opt_init: float = 1e-3,
                  loss_ratio_primal: float = 1.0,
                  loss_ratio_dual_pg: float = 1.0,
                  loss_ratio_dual_x: float = 1.0,
                  nn_dual_term_interval: int | None = 1,
                  loss_ratio_opt: float = 1.0,
                  loss_ratio_reg: float = 1.0,
                  subproblem_gamma_base: float = 1e-3,
                  mu_lower_bound_init: float = 0.1,
                  mu_individual_lower_bound_round: int = 3,
                  mu_group_lower_bound_round: int = 50,
                  mu_signed_round_interval: int | None = None,
                  mu_sign_hysteresis_rounds: int = 2,
                  mu_sign_flip_min_share: float = 0.67,
                  x_bound_dual_zero_rounds: int = 0,
                  subproblem_nn_batch_strategy: str = 'full-batch',
                  subproblem_nn_batch_size: int = 4,
                  subproblem_nn_shuffle: bool = True,
                  subproblem_nn_learning_rate: float = 1e-4,
                  subproblem_nn_main_eta_min_ratio: float = SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO,
                  subproblem_nn_main_lr_late_scale: float = SUBPROBLEM_NN_MAIN_LR_LATE_SCALE,
                  subproblem_nn_main_adam_weight_decay: float = SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY,
                  subproblem_nn_main_grad_clip: float = SUBPROBLEM_NN_MAIN_GRAD_CLIP,
                  subproblem_nn_main_kkt_lr_scale: float = SUBPROBLEM_NN_MAIN_KKT_LR_SCALE,
                  ignore_startup_shutdown_costs: bool = False,
                  subproblem_cost_learning_rate: float = 1e-5,
                  subproblem_nn_smooth_abs_eps: float = SUBPROBLEM_NN_SMOOTH_ABS_EPS,
                  pg_cost_nn_epochs: int | None = None,
                  pg_cost_start_round: int = 3,
                  pg_cost_scale_multiplier: float = 1.2,
                  pg_cost_lr: float = 2e-5,
                  pg_cost_surr_lr: float = 5e-5,
                  pg_cost_reg_deadband: float = 0.25,
                  pg_cost_softbound_weight: float = SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT,
                  pg_cost_smooth_abs_eps: float = 1e-6,
                  pg_cost_batch_strategy: str | None = None,
                  pg_cost_batch_size: int | None = None,
                  pg_cost_shuffle: bool | None = None,
                  pg_cost_use_sample_weights: bool = True,
                  pg_cost_sample_weight_power: float = 1.0,
                  pg_cost_sample_weight_clip: float = 10.0,
                  pg_cost_single_sample_reg_scale: float | None = None,
                  pg_cost_c_pg_adam_weight_decay: float | None = None,
                  main_direct_train_config: dict | None = None,
                  c_pg_direct_train_config: dict | None = None,
                  pg_block_prox_weight: float = SUBPROBLEM_PG_BLOCK_PROX_WEIGHT,
                  dual_block_prox_weight: float = SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT,
                  single_mu_cap_penalty_weight: float = SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT,
                  single_mu_cap_initial_weight: float | None = SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT,
                  single_mu_cap_final_weight: float | None = SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT,
                  single_mu_cap_initial: float | None = SUBPROBLEM_SINGLE_MU_CAP_INITIAL,
                  single_mu_cap_final: float | None = SUBPROBLEM_SINGLE_MU_CAP_FINAL,
                  single_mu_cap_start_round: int = SUBPROBLEM_SINGLE_MU_CAP_START_ROUND,
                  single_mu_cap_end_round: int = SUBPROBLEM_SINGLE_MU_CAP_END_ROUND,
                  override_loaded_single_mu_cap: bool = SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP,
                  iter_delta_reg_weight: float = SUBPROBLEM_ITER_DELTA_REG_WEIGHT,
                  iter_delta_reg_deadband: float = SUBPROBLEM_ITER_DELTA_REG_DEADBAND,
                  dual_predictor_only: bool = False,
                  dual_net_variant: str = 'temporal_conv',
                  dual_normalize_targets: bool = True,
                  dual_cosine_loss_weight: float = 0.12,
                  dual_smooth_l1_beta: float = 2.0,
                  use_unit_predictor: bool = False,
                  unit_predictor_epochs: int = 200,
                  unit_predictor_batch_strategy: str = 'full-batch',
                  unit_predictor_batch_size: int = 32,
                  unit_predictor_shuffle: bool = True,
                  unit_predictor_lr: float = 1e-3,
                  unit_predictor_hidden_dims: list[int] | None = None,
                  unit_predictor_net_variant: str = "mlp",
                  unit_predictor_tcn_channels: int = 64,
                  unit_predictor_tcn_depth: int = 6,
                  unit_predictor_tconv_channels: int = 64,
                  unit_predictor_tconv_depth: int = 4,
                  unit_predictor_dropout: float = 0.1,
                  unit_predictor_enable_pos_weight: bool = False,
                  unit_predictor_pos_weight_clip: float = 20.0,
                  unit_predictor_loss_weight_bce: float = 1.0,
                  unit_predictor_loss_weight_mse: float = 0.0,
                  unit_predictor_loss_weight_l1: float = 0.0,
                  unit_predictor_loss_weight_tv: float = 0.0,
                  unit_predictor_loss_weight_transition: float = 0.0,
                  unit_predictor_loss_weight_binarize: float = 0.0,
                  unit_predictor_loss_weight_std_floor: float = 0.0,
                  unit_predictor_std_floor_scale: float = 0.5,
                  unit_predictor_loss_weight_tv_floor: float = 0.0,
                  unit_predictor_tv_floor_scale: float = 0.8,
                  unit_predictor_load_path: str | None = None,
                  unit_predictor_unit1_extra_tv_epochs: int = 0,
                  unit_predictor_unit1_extra_tv_weight: float = 0.02,
                  unit_predictor_auto_extra_tv: bool = False,
                  unit_predictor_auto_extra_tv_epochs: int = 160,
                  unit_predictor_auto_extra_tv_tv_threshold: float = 0.02,
                  unit_predictor_auto_extra_tv_weight: float = 0.02,
                  predictor_warmup_rounds: int = 0,
                  sign4_curriculum_rounds: int = 0,
                  sign4_initial_scale: float = 1.0,
                  sign4_final_scale: float = 1.0,
                  sign4_delay_rounds: int = 0,
                  enable_surrogate_delta_reference_lift: bool | None = None,
                  surrogate_delta_reference_eps: float = 1e-6,
                  surrogate_delta_reference_scope: str = "sign4_only",
                  surrogate_delta_reference_min_abs_factor: float = 1e-9,
                  unit_predictor_finetune_lr: float = 1e-5,
                  unit_predictor_weight_decay: float = 1e-4,
                  skip_existing_units: bool = False,
                  case_name: str | None = None):
    """V3 代理约束训练（样本级并行），返回 (dual_predictor, trainers)。"""
    import os
    from pypower.ext2int import ext2int

    def _apply_runtime_single_mu_cap_config(trainer) -> None:
        trainer.single_mu_cap_initial_weight = (
            0.0 if single_mu_cap_initial_weight is None
            else max(float(single_mu_cap_initial_weight), 0.0)
        )
        final_weight = (
            single_mu_cap_penalty_weight
            if single_mu_cap_final_weight is None
            else single_mu_cap_final_weight
        )
        trainer.single_mu_cap_final_weight = max(float(final_weight), 0.0)
        trainer.single_mu_cap_penalty_weight = trainer.single_mu_cap_final_weight
        trainer.single_mu_cap_initial = (
            None if single_mu_cap_initial is None else max(float(single_mu_cap_initial), 0.0)
        )
        trainer.single_mu_cap_final = (
            None if single_mu_cap_final is None else max(float(single_mu_cap_final), 0.0)
        )
        trainer.single_mu_cap_start_round = max(int(single_mu_cap_start_round), 0)
        trainer.single_mu_cap_end_round = max(
            int(single_mu_cap_end_round),
            trainer.single_mu_cap_start_round,
        )

    def _resolve_unit_predictor_load_path(text: str | None) -> str | None:
        if text is None:
            return None
        p = str(text).strip()
        if not p:
            return None
        # directory: prefer LATEST.txt then unit_predictor.pth
        if os.path.isdir(p):
            latest = os.path.join(p, "LATEST.txt")
            if os.path.exists(latest):
                try:
                    target = open(latest, "r", encoding="utf-8").read().strip()
                    if target:
                        return target
                except Exception:
                    pass
            cand = os.path.join(p, "unit_predictor.pth")
            return cand if os.path.exists(cand) else None
        # file: if LATEST.txt, follow it
        if os.path.isfile(p) and os.path.basename(p).lower() == "latest.txt":
            try:
                target = open(p, "r", encoding="utf-8").read().strip()
                return target if target else None
            except Exception:
                return None
        return p if os.path.exists(p) else None

    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    unit_ids = UNIT_IDS if UNIT_IDS is not None else list(range(ng))
    lp_backend = str(lp_backend).strip().lower()

    # 在 cvxpy+HiGHS 路径下，启用“失败样本强诊断并中断”，便于一次定位根因。
    if lp_backend == "cvxpy_highs":
        os.environ.setdefault("STRICT_CVXPY_HIGHS", "1")

    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log(f"开始并行代理训练: {n_samples} 样本，{len(unit_ids)} 机组，"
        f"n_workers={n_workers}，dual_epochs={DUAL_EPOCHS}，"
        f"subproblem_iter={SUBPROBLEM_MAX_ITER}，nn_epochs={NN_EPOCHS}，"
        f"constraint_strategy={constraint_generation_strategy}")
    log(
        f"dual_nn: batch_strategy={dual_batch_strategy}, batch_size={DUAL_BATCH_SIZE}, "
        f"shuffle={dual_shuffle}, lr={dual_learning_rate}"
    )
    log(
        f"dual_predictor: net_variant={dual_net_variant}, normalize_targets={dual_normalize_targets}, "
        f"cosine_loss_w={dual_cosine_loss_weight}, smooth_l1_beta={dual_smooth_l1_beta}"
    )
    log(
        f"subproblem_nn: batch_strategy={subproblem_nn_batch_strategy}, "
        f"batch_size={subproblem_nn_batch_size}, shuffle={subproblem_nn_shuffle}, "
        f"size={subproblem_nn_size}, hidden_dims={subproblem_nn_hidden_dims}, "
        f"c_pg_size={subproblem_c_pg_nn_size}, c_pg_hidden_dims={subproblem_c_pg_nn_hidden_dims}, "
        f"ignore_startup_shutdown_costs={ignore_startup_shutdown_costs}, "
        f"main_lr={subproblem_nn_learning_rate}, x_cost_lr={subproblem_cost_learning_rate}, "
        f"nn_main_eta_min={subproblem_nn_main_eta_min_ratio}, "
        f"nn_main_lr_late_scale={subproblem_nn_main_lr_late_scale}, "
        f"nn_main_adam_wd={subproblem_nn_main_adam_weight_decay}, "
        f"nn_main_grad_clip={subproblem_nn_main_grad_clip}, "
        f"nn_main_kkt_lr_scale={subproblem_nn_main_kkt_lr_scale}, "
        f"nn_smooth_eps={subproblem_nn_smooth_abs_eps}, "
        f"c_pg_epochs={pg_cost_nn_epochs}, c_pg_surr_lr={pg_cost_surr_lr}, "
        f"c_pg_softbound_w={pg_cost_softbound_weight}, "
        f"c_pg_smooth_eps={pg_cost_smooth_abs_eps}, "
        f"c_pg_batch={pg_cost_batch_strategy or '(inherit subproblem_nn)'}/"
        f"{pg_cost_batch_size or '(inherit)'}, c_pg_shuffle={pg_cost_shuffle}, "
        f"c_pg_sample_w={pg_cost_use_sample_weights}, "
        f"c_pg_sw_power={pg_cost_sample_weight_power}, c_pg_sw_clip={pg_cost_sample_weight_clip}, "
        f"main_direct_epochs={int((main_direct_train_config or {}).get('direct_epochs', 0) or 0)}, "
        f"c_pg_direct_epochs={int((c_pg_direct_train_config or {}).get('direct_epochs', 0) or 0)}, "
        f"predictor_warmup={predictor_warmup_rounds}, "
        f"sign4_curriculum={sign4_initial_scale}->{sign4_final_scale}/"
        f"{sign4_curriculum_rounds}, sign4_delay={sign4_delay_rounds}"
    )
    log(
        f"iter_delta_reg: subproblem_weight={iter_delta_reg_weight}, "
        f"subproblem_deadband={iter_delta_reg_deadband}"
    )
    log(
        f"subproblem_prox: pg_block={pg_block_prox_weight}, "
        f"dual_block={dual_block_prox_weight}"
    )
    backend_status = get_cvxpy_highs_status()
    log(
        f"subproblem_lp_backend={lp_backend}, "
        f"path_prepended={SUBPROBLEM_SOLVER_PATH_PREPENDED}, "
        f"cvxpy={backend_status['cvxpy_available']}, "
        f"highspy={backend_status['highspy_available']}, "
        f"highs_solver={backend_status['highs_solver_available']}"
    )
    if lp_backend == 'cvxpy_highs' and not is_lp_backend_available(lp_backend):
        raise RuntimeError(
            "SUBPROBLEM_LP_BACKEND='cvxpy_highs' but cvxpy/highspy/HiGHS is unavailable."
        )
    log(
        f"subproblem_loss_ratio: primal={loss_ratio_primal}, dual_pg={loss_ratio_dual_pg}, "
        f"dual_x={loss_ratio_dual_x}, opt={loss_ratio_opt}, reg={loss_ratio_reg}"
    )
    log(
        f"subproblem_rho_binary: init={rho_binary_init}, max={rho_binary_max} "
        f"(cap independent of rho_max=10 for primal/dual/opt)"
    )
    print("=" * 70)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 步骤 1：对偶变量预测器（串行，NN 训练无需并行化）
    dual_save_path = os.path.join(save_dir, 'dual_predictor.pth') if save_dir else None
    load_path = resolve_existing_path(load_dir, 'surrogate model dir') if load_dir else None
    dual_checkpoint_path = load_path / 'dual_predictor.pth' if load_path else None
    if dual_checkpoint_path is not None and dual_checkpoint_path.exists():
        log(f"从已有模型继续训练 dual predictor: {dual_checkpoint_path}")
        dual_predictor = load_trained_models(
            ppc,
            all_samples,
            T_DELTA,
            load_dir=str(load_path),
            unit_ids=[],
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            case_name=case_name,
        )[0]
        dual_predictor.train(
            num_epochs=DUAL_EPOCHS,
            batch_size=DUAL_BATCH_SIZE,
            batch_strategy=dual_batch_strategy,
            shuffle=dual_shuffle,
            learning_rate=dual_learning_rate,
        )
        if dual_save_path:
            dual_predictor.save(dual_save_path)
    else:
        dual_predictor = train_dual_predictor_from_data(
            ppc, all_samples, T_delta=T_DELTA,
            num_epochs=DUAL_EPOCHS,
            batch_size=DUAL_BATCH_SIZE,
            batch_strategy=dual_batch_strategy,
            shuffle=dual_shuffle,
            learning_rate=dual_learning_rate,
            save_path=dual_save_path,
            dual_net_variant=dual_net_variant,
            dual_normalize_targets=dual_normalize_targets,
            dual_cosine_loss_weight=dual_cosine_loss_weight,
            dual_smooth_l1_beta=dual_smooth_l1_beta,
        )

    if dual_predictor_only:
        log("dual_predictor_only=True：已结束对偶预测器训练，跳过子问题代理约束训练")
        return dual_predictor, {}

    # 步骤 1.5：单机组 0/1 预测器预训练（可选）
    unit_predictor = None
    unit_predictor_save_path = None
    if use_unit_predictor:
        if constraint_generation_strategy not in (
            'all_single_time', 'all_templates_sign4_plus_single',
        ):
            log(
                f"[unit_predictor] constraint_strategy={constraint_generation_strategy} 不包含 single-time 段，"
                "跳过单机组 0/1 预测器"
            )
        else:
            unit_predictor_save_path = (
                os.path.join(save_dir, 'unit_predictor.pth') if save_dir else None
            )
            resolved_external = _resolve_unit_predictor_load_path(unit_predictor_load_path)
            if unit_predictor_load_path and not resolved_external:
                raise FileNotFoundError(
                    f"UNIT_PREDICTOR_LOAD_PATH was set but no checkpoint was found: {unit_predictor_load_path}"
                )
            unit_predictor_load_path = resolved_external or (
                str(load_path / 'unit_predictor.pth')
                if load_path and (load_path / 'unit_predictor.pth').exists()
                else None
            )
            log(
                f"[unit_predictor] 预训练单机组 0/1 预测器: "
                f"units={unit_ids}, epochs={unit_predictor_epochs}, "
                f"batch={unit_predictor_batch_size}/{unit_predictor_batch_strategy}, "
                f"lr={unit_predictor_lr}, hidden_dims={unit_predictor_hidden_dims}, "
                f"finetune_lr={unit_predictor_finetune_lr}"
            )
            # Explicit external checkpoints are used as-is; auto-discovered checkpoints
            # from load_dir are treated as warm starts and may continue training.  If an
            # explicit checkpoint is incompatible, train_unit_predictor_from_data raises
            # instead of silently running with random weights.
            effective_epochs = 0 if resolved_external else int(unit_predictor_epochs)
            unit_predictor = train_unit_predictor_from_data(
                ppc,
                all_samples,
                T_delta=T_DELTA,
                unit_ids=unit_ids,
                hidden_dims=unit_predictor_hidden_dims,
                num_epochs=effective_epochs,
                batch_size=unit_predictor_batch_size,
                batch_strategy=unit_predictor_batch_strategy,
                shuffle=unit_predictor_shuffle,
                learning_rate=unit_predictor_lr,
                weight_decay=unit_predictor_weight_decay,
                net_variant=unit_predictor_net_variant,
                tcn_channels=unit_predictor_tcn_channels,
                tcn_depth=unit_predictor_tcn_depth,
                tconv_channels=unit_predictor_tconv_channels,
                tconv_depth=unit_predictor_tconv_depth,
                dropout=unit_predictor_dropout,
                enable_pos_weight=unit_predictor_enable_pos_weight,
                pos_weight_clip=unit_predictor_pos_weight_clip,
                loss_weight_bce=unit_predictor_loss_weight_bce,
                loss_weight_mse=unit_predictor_loss_weight_mse,
                loss_weight_l1=unit_predictor_loss_weight_l1,
                loss_weight_tv=unit_predictor_loss_weight_tv,
                loss_weight_transition=unit_predictor_loss_weight_transition,
                loss_weight_binarize=unit_predictor_loss_weight_binarize,
                loss_weight_std_floor=unit_predictor_loss_weight_std_floor,
                std_floor_scale=unit_predictor_std_floor_scale,
                loss_weight_tv_floor=unit_predictor_loss_weight_tv_floor,
                tv_floor_scale=unit_predictor_tv_floor_scale,
                save_path=unit_predictor_save_path,
                load_path=unit_predictor_load_path,
            )

            # 可选：仅对 unit=1 追加 TV 微调（避免 TV 拖累其它机组）
            try:
                extra_epochs = int(unit_predictor_unit1_extra_tv_epochs)
            except Exception:
                extra_epochs = 0
            if (
                unit_predictor is not None
                and extra_epochs > 0
                and any(int(x) == 1 for x in (unit_ids or []))
            ):
                log(
                    f"[unit_predictor] unit=1 extra TV finetune: "
                    f"epochs={extra_epochs}, tv_w={float(unit_predictor_unit1_extra_tv_weight)}"
                )
                unit_predictor.train_unit(
                    unit_id=1,
                    num_epochs=extra_epochs,
                    batch_size=unit_predictor_batch_size,
                    batch_strategy=unit_predictor_batch_strategy,
                    shuffle=unit_predictor_shuffle,
                    learning_rate=unit_predictor_lr,
                    enable_scheduler=True,
                    scheduler_patience=10,
                    scheduler_factor=0.5,
                    scheduler_min_lr=1e-5,
                    enable_pos_weight=False,
                    pos_weight_clip=unit_predictor_pos_weight_clip,
                    loss_weight_bce=0.0,
                    loss_weight_mse=1.0,
                    loss_weight_l1=0.0,
                    loss_weight_tv=float(unit_predictor_unit1_extra_tv_weight),
                    loss_weight_transition=0.25,
                    loss_weight_binarize=0.05,
                    loss_weight_std_floor=0.2,
                    std_floor_scale=unit_predictor_std_floor_scale,
                    loss_weight_tv_floor=unit_predictor_loss_weight_tv_floor,
                    tv_floor_scale=unit_predictor_tv_floor_scale,
                )

            # 自动 TV 微调：挑选预测变化不足（tv 很小）的机组，追加一段 TV+transition+std_floor 微调
            if unit_predictor is not None and bool(unit_predictor_auto_extra_tv):
                try:
                    auto_epochs = max(0, int(unit_predictor_auto_extra_tv_epochs))
                except Exception:
                    auto_epochs = 0
                tv_thr = float(unit_predictor_auto_extra_tv_tv_threshold)
                tv_w = float(unit_predictor_auto_extra_tv_weight)
                if auto_epochs > 0:
                    # 计算每台机组在训练集上的预测 tv（不依赖阈值）
                    import numpy as _np
                    import torch as _torch

                    X = _np.array([unit_predictor._extract_features(i) for i in range(unit_predictor.n_samples)])
                    X_tensor = _torch.tensor(X, dtype=_torch.float32, device=unit_predictor.device)
                    low_tv_units: list[int] = []
                    for g in unit_ids:
                        g_int = int(g)
                        net = unit_predictor.get_network(g_int)
                        with _torch.no_grad():
                            logits = net(X_tensor).detach()
                            probs = _torch.sigmoid(logits)
                            tv_val = float(_torch.mean(_torch.abs(probs[:, 1:] - probs[:, :-1])).item())
                        if tv_val < tv_thr:
                            low_tv_units.append(g_int)
                    if low_tv_units:
                        log(
                            "[unit_predictor] auto extra TV finetune: "
                            f"units={low_tv_units}, epochs={auto_epochs}, tv_w={tv_w}, tv_thr={tv_thr}"
                        )
                        for g_int in low_tv_units:
                            unit_predictor.train_unit(
                                unit_id=g_int,
                                num_epochs=auto_epochs,
                                batch_size=unit_predictor_batch_size,
                                batch_strategy=unit_predictor_batch_strategy,
                                shuffle=unit_predictor_shuffle,
                                learning_rate=unit_predictor_lr,
                                enable_scheduler=False,
                                scheduler_patience=10,
                                scheduler_factor=0.5,
                                scheduler_min_lr=1e-5,
                                enable_pos_weight=False,
                                pos_weight_clip=unit_predictor_pos_weight_clip,
                                loss_weight_bce=0.0,
                                loss_weight_mse=1.0,
                                loss_weight_l1=0.0,
                                loss_weight_tv=tv_w,
                                loss_weight_transition=unit_predictor_loss_weight_transition,
                                loss_weight_binarize=unit_predictor_loss_weight_binarize,
                                loss_weight_std_floor=unit_predictor_loss_weight_std_floor,
                                std_floor_scale=unit_predictor_std_floor_scale,
                                loss_weight_tv_floor=max(float(unit_predictor_loss_weight_tv_floor), float(tv_w)),
                                tv_floor_scale=unit_predictor_tv_floor_scale,
                            )

    # 步骤 2：训练代理约束（支持机组级并行或样本级并行）
    # B1：机组级并行（Level 1）
    unit_ids_to_train = list(unit_ids)
    if load_path is not None and bool(skip_existing_units):
        import shutil

        existing_units = []
        missing_units = []
        for g in unit_ids:
            src_path = load_path / f"surrogate_unit_{int(g)}.pth"
            if src_path.exists():
                existing_units.append(int(g))
                if save_dir:
                    dst_path = Path(save_dir) / src_path.name
                    if src_path.resolve() != dst_path.resolve():
                        shutil.copy2(src_path, dst_path)
            else:
                missing_units.append(int(g))
        unit_ids_to_train = missing_units
        if existing_units:
            log(
                f"resume surrogate: reuse existing units={existing_units}; "
                f"train missing units={unit_ids_to_train}"
            )
        if not unit_ids_to_train:
            model_dir_for_load = str(Path(save_dir)) if save_dir else str(load_path)
            log("resume surrogate: all requested unit models already exist; skip subproblem training")
            dual_predictor_loaded, trainers = load_trained_models(
                ppc,
                all_samples,
                T_DELTA,
                load_dir=model_dir_for_load,
                unit_ids=unit_ids,
                lp_backend=lp_backend,
                constraint_generation_strategy=constraint_generation_strategy,
                ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
                case_name=case_name,
            )
            if logger is not None:
                for trainer in trainers.values():
                    trainer.logger = logger
            if unit_predictor is not None:
                for trainer in trainers.values():
                    trainer.unit_predictor = unit_predictor
                    trainer.use_unit_predictor = True
            return dual_predictor_loaded, trainers

    if N_WORKERS_UNIT > 1:
        from uc_NN_subproblem_parallel import train_all_surrogates_parallel

        if unit_predictor is not None and not unit_predictor_save_path:
            raise RuntimeError(
                "unit_predictor is enabled with unit-level multiprocessing, but no save_dir/path "
                "is available for child workers to load the predictor checkpoint"
            )
        log(
            f"  启用机组级并行: unit_workers={N_WORKERS_UNIT}, sample_workers={N_WORKERS_SAMPLE}"
        )
        _ = train_all_surrogates_parallel(
            ppc,
            all_samples,
            T_delta=T_DELTA,
            lambda_predictor=dual_predictor,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            unit_ids=unit_ids_to_train,
            max_iter=SUBPROBLEM_MAX_ITER,
            nn_epochs=NN_EPOCHS,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_binary_max=rho_binary_max,
            rho_opt_init=rho_opt_init,
            gamma_base=subproblem_gamma_base,
            mu_lower_bound_init=mu_lower_bound_init,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            mu_signed_round_interval=mu_signed_round_interval,
            mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
            mu_sign_flip_min_share=mu_sign_flip_min_share,
            x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            nn_learning_rate=subproblem_nn_learning_rate,
            cost_learning_rate=subproblem_cost_learning_rate,
            nn_batch_strategy=subproblem_nn_batch_strategy,
            nn_batch_size=subproblem_nn_batch_size,
            nn_shuffle=subproblem_nn_shuffle,
            nn_smooth_abs_eps=subproblem_nn_smooth_abs_eps,
            nn_main_eta_min_ratio=subproblem_nn_main_eta_min_ratio,
            nn_main_lr_late_scale=subproblem_nn_main_lr_late_scale,
            nn_main_adam_weight_decay=subproblem_nn_main_adam_weight_decay,
            nn_main_grad_clip=subproblem_nn_main_grad_clip,
            nn_main_kkt_lr_scale=subproblem_nn_main_kkt_lr_scale,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            nn_dual_term_interval=nn_dual_term_interval,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            pg_cost_nn_epochs=pg_cost_nn_epochs,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            pg_cost_batch_strategy=pg_cost_batch_strategy,
            pg_cost_batch_size=pg_cost_batch_size,
            pg_cost_shuffle=pg_cost_shuffle,
            pg_cost_use_sample_weights=pg_cost_use_sample_weights,
            pg_cost_sample_weight_power=pg_cost_sample_weight_power,
            pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
            pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
            pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
            main_direct_train_config=main_direct_train_config,
            c_pg_direct_train_config=c_pg_direct_train_config,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            single_mu_cap_penalty_weight=single_mu_cap_penalty_weight,
            single_mu_cap_initial_weight=single_mu_cap_initial_weight,
            single_mu_cap_final_weight=single_mu_cap_final_weight,
            single_mu_cap_initial=single_mu_cap_initial,
            single_mu_cap_final=single_mu_cap_final,
            single_mu_cap_start_round=single_mu_cap_start_round,
            single_mu_cap_end_round=single_mu_cap_end_round,
            save_dir=save_dir,
            n_workers=N_WORKERS_UNIT,
            sample_n_workers=N_WORKERS_SAMPLE,
            use_sample_parallel=(N_WORKERS_SAMPLE > 1),
            unit_predictor_path=unit_predictor_save_path,
            use_unit_predictor=(unit_predictor is not None),
            predictor_warmup_rounds=predictor_warmup_rounds,
            sign4_curriculum_rounds=sign4_curriculum_rounds,
            sign4_initial_scale=sign4_initial_scale,
            sign4_final_scale=sign4_final_scale,
            sign4_delay_rounds=sign4_delay_rounds,
            enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
            surrogate_delta_reference_eps=surrogate_delta_reference_eps,
            surrogate_delta_reference_scope=surrogate_delta_reference_scope,
            surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            unit_predictor_hidden_dims=unit_predictor_hidden_dims,
            unit_predictor_net_variant=unit_predictor_net_variant,
            unit_predictor_tcn_channels=unit_predictor_tcn_channels,
            unit_predictor_tcn_depth=unit_predictor_tcn_depth,
            unit_predictor_tconv_channels=unit_predictor_tconv_channels,
            unit_predictor_tconv_depth=unit_predictor_tconv_depth,
            unit_predictor_dropout=unit_predictor_dropout,
            case_name=case_name,
        )

        # train_all_surrogates_parallel 返回的是 state_dict；为了保持 run_surrogate 的返回形态
        # （trainer 对象，用于 print_surrogate_results / 下游流程），这里从 save_dir 重新加载。
        dual_predictor_loaded, trainers = load_trained_models(
            ppc,
            all_samples,
            T_DELTA,
            load_dir=str(save_dir),
            unit_ids=unit_ids,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            case_name=case_name,
        )
        if logger is not None:
            for trainer in trainers.values():
                trainer.logger = logger
        if unit_predictor is not None:
            for trainer in trainers.values():
                trainer.unit_predictor = unit_predictor
                trainer.use_unit_predictor = True
        return dual_predictor_loaded, trainers

    trainers = {}
    for i, g in enumerate(unit_ids_to_train):
        if n_workers <= 1:
            log(f"  机组 {g} ({i+1}/{len(unit_ids)}) — 串行")
            trainer = SubproblemSurrogateTrainer(
                ppc, all_samples, T_DELTA, g,
                lambda_predictor=dual_predictor,
                lp_backend=lp_backend,
                constraint_generation_strategy=constraint_generation_strategy,
                rho_primal_init=rho_primal_init,
        rho_dual_init=rho_dual_init,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        rho_binary_init=rho_binary_init,
        rho_binary_max=rho_binary_max,
        rho_opt_init=rho_opt_init,
                loss_ratio_primal=loss_ratio_primal,
                loss_ratio_dual_pg=loss_ratio_dual_pg,
                loss_ratio_dual_x=loss_ratio_dual_x,
                nn_dual_term_interval=nn_dual_term_interval,
                loss_ratio_opt=loss_ratio_opt,
                loss_ratio_reg=loss_ratio_reg,
                gamma_base=subproblem_gamma_base,
                mu_lower_bound_init=mu_lower_bound_init,
                mu_individual_lower_bound_round=mu_individual_lower_bound_round,
                mu_group_lower_bound_round=mu_group_lower_bound_round,
                mu_signed_round_interval=mu_signed_round_interval,
                mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
                mu_sign_flip_min_share=mu_sign_flip_min_share,
                x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
                nn_hidden_dims=subproblem_nn_hidden_dims,
                pg_cost_hidden_dims=subproblem_c_pg_nn_hidden_dims,
                nn_batch_strategy=subproblem_nn_batch_strategy,
                nn_batch_size=subproblem_nn_batch_size,
                nn_shuffle=subproblem_nn_shuffle,
                nn_learning_rate=subproblem_nn_learning_rate,
                nn_main_eta_min_ratio=subproblem_nn_main_eta_min_ratio,
                nn_main_lr_late_scale=subproblem_nn_main_lr_late_scale,
                nn_main_adam_weight_decay=subproblem_nn_main_adam_weight_decay,
        nn_main_grad_clip=subproblem_nn_main_grad_clip,
        nn_main_kkt_lr_scale=subproblem_nn_main_kkt_lr_scale,
        ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
                cost_learning_rate=subproblem_cost_learning_rate,
                nn_smooth_abs_eps=subproblem_nn_smooth_abs_eps,
                pg_cost_start_round=pg_cost_start_round,
                pg_cost_scale_multiplier=pg_cost_scale_multiplier,
                pg_cost_lr=pg_cost_lr,
                pg_cost_surr_lr=pg_cost_surr_lr,
                pg_cost_reg_deadband=pg_cost_reg_deadband,
                pg_cost_softbound_weight=pg_cost_softbound_weight,
                pg_cost_nn_epochs=pg_cost_nn_epochs,
                pg_cost_smooth_abs_eps=pg_cost_smooth_abs_eps,
                pg_cost_batch_strategy=pg_cost_batch_strategy,
                pg_cost_batch_size=pg_cost_batch_size,
                pg_cost_shuffle=pg_cost_shuffle,
                pg_cost_use_sample_weights=pg_cost_use_sample_weights,
                pg_cost_sample_weight_power=pg_cost_sample_weight_power,
                pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
                pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
                pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
                main_direct_train_config=main_direct_train_config,
                c_pg_direct_train_config=c_pg_direct_train_config,
                pg_block_prox_weight=pg_block_prox_weight,
                dual_block_prox_weight=dual_block_prox_weight,
                single_mu_cap_penalty_weight=single_mu_cap_penalty_weight,
                single_mu_cap_initial_weight=single_mu_cap_initial_weight,
                single_mu_cap_final_weight=single_mu_cap_final_weight,
                single_mu_cap_initial=single_mu_cap_initial,
                single_mu_cap_final=single_mu_cap_final,
                single_mu_cap_start_round=single_mu_cap_start_round,
                single_mu_cap_end_round=single_mu_cap_end_round,
                iter_delta_reg_weight=iter_delta_reg_weight,
                iter_delta_reg_deadband=iter_delta_reg_deadband,
                unit_predictor=unit_predictor,
                use_unit_predictor=(unit_predictor is not None),
                predictor_warmup_rounds=predictor_warmup_rounds,
                sign4_curriculum_rounds=sign4_curriculum_rounds,
                sign4_initial_scale=sign4_initial_scale,
                sign4_final_scale=sign4_final_scale,
                sign4_delay_rounds=sign4_delay_rounds,
                enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
                surrogate_delta_reference_eps=surrogate_delta_reference_eps,
                surrogate_delta_reference_scope=surrogate_delta_reference_scope,
                surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
                unit_predictor_finetune_lr=unit_predictor_finetune_lr,
                unit_predictor_weight_decay=unit_predictor_weight_decay,
                case_name=case_name,
            )
        else:
            log(f"  机组 {g} ({i+1}/{len(unit_ids)}) — 样本级并行 n_workers={n_workers}")
            trainer = ParallelSubproblemSurrogateTrainer(
                ppc, all_samples, T_DELTA, g,
                lambda_predictor=dual_predictor,
                lp_backend=lp_backend,
                constraint_generation_strategy=constraint_generation_strategy,
                rho_primal_init=rho_primal_init,
                rho_dual_init=rho_dual_init,
                rho_dual_pg_init=rho_dual_pg_init,
                rho_dual_x_init=rho_dual_x_init,
                rho_dual_coc_init=rho_dual_coc_init,
                rho_binary_init=rho_binary_init,
                rho_binary_max=rho_binary_max,
                rho_opt_init=rho_opt_init,
                loss_ratio_primal=loss_ratio_primal,
                loss_ratio_dual_pg=loss_ratio_dual_pg,
                loss_ratio_dual_x=loss_ratio_dual_x,
                nn_dual_term_interval=nn_dual_term_interval,
                loss_ratio_opt=loss_ratio_opt,
                loss_ratio_reg=loss_ratio_reg,
                gamma_base=subproblem_gamma_base,
                mu_lower_bound_init=mu_lower_bound_init,
                mu_individual_lower_bound_round=mu_individual_lower_bound_round,
                mu_group_lower_bound_round=mu_group_lower_bound_round,
                mu_signed_round_interval=mu_signed_round_interval,
                mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
                mu_sign_flip_min_share=mu_sign_flip_min_share,
                x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
                nn_hidden_dims=subproblem_nn_hidden_dims,
                pg_cost_hidden_dims=subproblem_c_pg_nn_hidden_dims,
                nn_batch_strategy=subproblem_nn_batch_strategy,
                nn_batch_size=subproblem_nn_batch_size,
                nn_shuffle=subproblem_nn_shuffle,
                nn_learning_rate=subproblem_nn_learning_rate,
                nn_main_eta_min_ratio=subproblem_nn_main_eta_min_ratio,
                nn_main_lr_late_scale=subproblem_nn_main_lr_late_scale,
                nn_main_adam_weight_decay=subproblem_nn_main_adam_weight_decay,
        nn_main_grad_clip=subproblem_nn_main_grad_clip,
        nn_main_kkt_lr_scale=subproblem_nn_main_kkt_lr_scale,
        ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
                cost_learning_rate=subproblem_cost_learning_rate,
                nn_smooth_abs_eps=subproblem_nn_smooth_abs_eps,
                pg_cost_start_round=pg_cost_start_round,
                pg_cost_scale_multiplier=pg_cost_scale_multiplier,
                pg_cost_lr=pg_cost_lr,
                pg_cost_surr_lr=pg_cost_surr_lr,
                pg_cost_reg_deadband=pg_cost_reg_deadband,
                pg_cost_softbound_weight=pg_cost_softbound_weight,
                pg_cost_nn_epochs=pg_cost_nn_epochs,
                pg_cost_smooth_abs_eps=pg_cost_smooth_abs_eps,
                pg_cost_batch_strategy=pg_cost_batch_strategy,
                pg_cost_batch_size=pg_cost_batch_size,
                pg_cost_shuffle=pg_cost_shuffle,
                pg_cost_use_sample_weights=pg_cost_use_sample_weights,
                pg_cost_sample_weight_power=pg_cost_sample_weight_power,
                pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
                pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
                pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
                main_direct_train_config=main_direct_train_config,
                c_pg_direct_train_config=c_pg_direct_train_config,
                pg_block_prox_weight=pg_block_prox_weight,
                dual_block_prox_weight=dual_block_prox_weight,
                single_mu_cap_penalty_weight=single_mu_cap_penalty_weight,
                single_mu_cap_initial_weight=single_mu_cap_initial_weight,
                single_mu_cap_final_weight=single_mu_cap_final_weight,
                single_mu_cap_initial=single_mu_cap_initial,
                single_mu_cap_final=single_mu_cap_final,
                single_mu_cap_start_round=single_mu_cap_start_round,
                single_mu_cap_end_round=single_mu_cap_end_round,
                iter_delta_reg_weight=iter_delta_reg_weight,
                iter_delta_reg_deadband=iter_delta_reg_deadband,
                unit_predictor=unit_predictor,
                use_unit_predictor=(unit_predictor is not None),
                predictor_warmup_rounds=predictor_warmup_rounds,
                sign4_curriculum_rounds=sign4_curriculum_rounds,
                sign4_initial_scale=sign4_initial_scale,
                sign4_final_scale=sign4_final_scale,
                sign4_delay_rounds=sign4_delay_rounds,
                enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
                surrogate_delta_reference_eps=surrogate_delta_reference_eps,
                surrogate_delta_reference_scope=surrogate_delta_reference_scope,
                surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
                unit_predictor_finetune_lr=unit_predictor_finetune_lr,
                unit_predictor_weight_decay=unit_predictor_weight_decay,
                case_name=case_name,
                n_workers=n_workers,
            )
        if load_path is not None:
            trainer_checkpoint_path = load_path / f'surrogate_unit_{g}.pth'
            if trainer_checkpoint_path.exists():
                log(f"继续训练 surrogate checkpoint: {trainer_checkpoint_path}")
                trainer.load(str(trainer_checkpoint_path))
                trainer.nn_smooth_abs_eps = max(0.0, float(subproblem_nn_smooth_abs_eps))
                trainer.pg_cost_smooth_abs_eps = max(0.0, float(pg_cost_smooth_abs_eps))
                if bool(override_loaded_single_mu_cap):
                    _apply_runtime_single_mu_cap_config(trainer)
                    log(
                        "override loaded single_mu_cap: "
                        f"weight={trainer.single_mu_cap_initial_weight}"
                        f"->{trainer.single_mu_cap_final_weight}, "
                        f"cap={trainer.single_mu_cap_initial}"
                        f"->{trainer.single_mu_cap_final}, "
                        f"rounds={trainer.single_mu_cap_start_round}"
                        f"..{trainer.single_mu_cap_end_round}"
                    )
        if logger is not None:
            trainer.logger = logger
        trainer.iter(
            max_iter=SUBPROBLEM_MAX_ITER,
            nn_epochs=NN_EPOCHS,
            pg_cost_nn_epochs=pg_cost_nn_epochs,
            nn_batch_strategy=subproblem_nn_batch_strategy,
            nn_batch_size=subproblem_nn_batch_size,
            nn_shuffle=subproblem_nn_shuffle,
            nn_learning_rate=subproblem_nn_learning_rate,
            cost_learning_rate=subproblem_cost_learning_rate,
            pg_cost_surr_learning_rate=pg_cost_surr_lr,
        )
        trainers[g] = trainer
        if save_dir:
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))

    if load_path is not None and bool(skip_existing_units):
        model_dir_for_load = str(Path(save_dir)) if save_dir else str(load_path)
        dual_predictor_loaded, trainers = load_trained_models(
            ppc,
            all_samples,
            T_DELTA,
            load_dir=model_dir_for_load,
            unit_ids=unit_ids,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            case_name=case_name,
        )
        if logger is not None:
            for trainer in trainers.values():
                trainer.logger = logger
        if unit_predictor is not None:
            for trainer in trainers.values():
                trainer.unit_predictor = unit_predictor
                trainer.use_unit_predictor = True
        return dual_predictor_loaded, trainers

    return dual_predictor, trainers


def load_surrogate(ppc, all_samples, T_DELTA, UNIT_IDS, load_dir,
                   logger: 'TrainingLogger | None' = None,
                   lp_backend: str | None = None,
                   constraint_generation_strategy: str | None = None,
                   ignore_startup_shutdown_costs: bool | None = None,
                   case_name: str | None = None):
    """加载已有 dual_predictor 和 subproblem surrogate 模型。"""
    load_path = Path(load_dir)
    if not load_path.is_absolute():
        load_path = Path(__file__).parent / load_path
    if not load_path.exists():
        raise FileNotFoundError(f"surrogate 模型目录不存在: {load_path}")

    log(f"从已有目录加载 subproblem 模型，跳过 subproblem 训练: {load_path}")
    predictor_strategy = normalize_constraint_generation_strategy(
        constraint_generation_strategy or SURROGATE_CONSTRAINT_STRATEGY
    )
    load_unit_predictor = bool(USE_UNIT_PREDICTOR) and predictor_strategy in {
        'all_single_time',
        'all_templates_sign4_plus_single',
    }
    unit_predictor_checkpoint = None
    if load_unit_predictor:
        raw_predictor_path = UNIT_PREDICTOR_LOAD_PATH
        if raw_predictor_path:
            p = Path(raw_predictor_path)
            if not p.is_absolute():
                p = Path(__file__).parent / p
            if p.is_dir():
                latest = p / 'LATEST.txt'
                if latest.is_file():
                    target = Path(latest.read_text(encoding='utf-8').strip())
                    p = target if target.is_absolute() else p / target
                else:
                    p = p / 'unit_predictor.pth'
            if not p.exists():
                raise FileNotFoundError(f"UNIT_PREDICTOR_LOAD_PATH was set but no checkpoint was found: {p}")
            unit_predictor_checkpoint = str(p)
        else:
            unit_predictor_checkpoint = str(load_path / 'unit_predictor.pth')

    dual_predictor, trainers = load_trained_models(
        ppc,
        all_samples,
        T_DELTA,
        load_dir=str(load_path),
        unit_ids=UNIT_IDS,
        lp_backend=lp_backend,
        constraint_generation_strategy=constraint_generation_strategy,
        ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
        case_name=case_name,
        unit_predictor_path=unit_predictor_checkpoint,
        use_unit_predictor=load_unit_predictor,
        unit_predictor_hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS,
        unit_predictor_net_variant=UNIT_PREDICTOR_NET_VARIANT,
        unit_predictor_tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS,
        unit_predictor_tcn_depth=UNIT_PREDICTOR_TCN_DEPTH,
        unit_predictor_tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS,
        unit_predictor_tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH,
        unit_predictor_dropout=UNIT_PREDICTOR_DROPOUT,
        unit_predictor_finetune_lr=UNIT_PREDICTOR_FINETUNE_LR,
        unit_predictor_weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY,
    )
    if logger is not None:
        for trainer in trainers.values():
            trainer.logger = logger
    return dual_predictor, trainers


def print_surrogate_results(trainers, all_samples):
    """打印代理训练结果摘要。"""
    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log("训练结果验证")
    print("=" * 70)

    for unit_id, trainer in trainers.items():
        T = trainer.T
        nc = trainer.num_coupling_constraints
        print(f"\n机组 {unit_id}:")
        print(f"  alpha_values shape: {trainer.alpha_values.shape}  "
              f"(期望: ({n_samples}, {nc}))")
        print(f"  beta_values  shape: {trainer.beta_values.shape}")
        print(f"  gamma_values shape: {trainer.gamma_values.shape}")
        print(f"  delta_values shape: {trainer.delta_values.shape}  (RHS，非负)")

        print(f"  样本0 时序约束示例（最多5条）:")
        x0 = trainer.x[0]
        offsets0 = resolve_constraint_offsets_from_trainer(trainer, 0, nc)
        timesteps0 = list(getattr(trainer, 'sensitive_timesteps', [[]])[0]) if getattr(trainer, 'sensitive_timesteps', None) else list(range(nc))
        for k in range(min(5, nc, len(timesteps0), len(offsets0))):
            t = int(timesteps0[k])
            offsets = tuple(int(v) for v in offsets0[k])
            a = trainer.alpha_values[0, k]
            b = trainer.beta_values[0, k]
            g = trainer.gamma_values[0, k]
            d = trainer.delta_values[0, k]

            terms = []
            lhs = 0.0
            if 0 in offsets and 0 <= t < T:
                terms.append(f"{a:.3f}*x[{t}]")
                lhs += a * x0[t]
            if 1 in offsets and 0 <= t + 1 < T:
                terms.append(f"{b:.3f}*x[{t+1}]")
                lhs += b * x0[t + 1]
            if 2 in offsets and 0 <= t + 2 < T:
                terms.append(f"{g:.3f}*x[{t+2}]")
                lhs += g * x0[t + 2]
            if not terms:
                continue

            viol = max(0.0, lhs - d)
            offsets_str = ",".join(str(v) for v in offsets)
            print(
                f"    k={k}, t={t}, offsets=({offsets_str}): "
                f"{' + '.join(terms)} <= {d:.3f}  "
                f"(lhs={lhs:.3f}, viol={viol:.4f})"
            )

        integrality = float(np.sum(x0 * (1 - x0)))
        print(f"  整数性指标(样本0): {integrality:.6f}  (0=完全整数)")


def run_bcd(ppc, all_samples: list, T_DELTA, MAX_ITER, bcd_model_dir,
            case_name: str = 'case', timestamp: str = '', n_workers: int = 4, NN_EPOCHS: int = 10, DUAL_DECAY_ROUND: int = 10,
            DUAL_SIGN_RELAX_INTERVAL: int | None = None,
            lp_backend: str = 'gurobi',
            gurobi_threads: int | None = None,
            gurobi_lp_method: int = -1,
            bcd_highs_threads: int = 1,
            logger: 'TrainingLogger | None' = None,
            load_model_path: str | None = None,
            restore_rho_from_checkpoint: bool = False,
            external_sparse_templates=None,
            lambda_init_strategy: str = 'lp_relaxation',
            max_theta_constraints_per_time_slot: int = 10,
            theta_training_stages=None,
            theta_curriculum_rounds: int = 0,
            theta_initial_scale: float = 1.0,
            theta_final_scale: float = 1.0,
            theta_curriculum_delay_rounds: int = 0,
            theta_hot_start_strategy: str = 'dcpf_relative',
            zeta_hot_start_strategy: str = 'zero',
            theta_gaussian_std: float = 0.01,
            zeta_gaussian_std: float = 0.01,
            use_unit_predictor: bool = False,
            unit_predictor_epochs: int = 0,
            unit_predictor_batch_strategy: str = 'full-batch',
            unit_predictor_batch_size: int = 32,
            unit_predictor_shuffle: bool = True,
            unit_predictor_lr: float = 1e-3,
            unit_predictor_hidden_dims: list[int] | None = None,
            unit_predictor_net_variant: str = "mlp",
            unit_predictor_tcn_channels: int = 64,
            unit_predictor_tcn_depth: int = 6,
            unit_predictor_tconv_channels: int = 64,
            unit_predictor_tconv_depth: int = 4,
            unit_predictor_dropout: float = 0.1,
            unit_predictor_enable_pos_weight: bool = False,
            unit_predictor_pos_weight_clip: float = 20.0,
            unit_predictor_loss_weight_bce: float = 1.0,
            unit_predictor_loss_weight_mse: float = 0.0,
            unit_predictor_loss_weight_l1: float = 0.0,
            unit_predictor_loss_weight_tv: float = 0.0,
            unit_predictor_loss_weight_transition: float = 0.0,
            unit_predictor_loss_weight_binarize: float = 0.0,
            unit_predictor_loss_weight_std_floor: float = 0.0,
            unit_predictor_std_floor_scale: float = 0.5,
            unit_predictor_loss_weight_tv_floor: float = 0.0,
            unit_predictor_tv_floor_scale: float = 0.8,
            unit_predictor_load_path: str | None = None,
            unit_predictor_warmup_rounds: int = 0,
            unit_predictor_finetune_lr: float = 1e-5,
            unit_predictor_weight_decay: float = 1e-4,
            theta_constraint_delay_rounds: int = 0,
            enable_dropout_during_nn_training: bool = True,
            rho_primal_init: float = 1e-2,
            rho_dual_init: float = 1e-2,
            rho_dual_pg_init: float | None = None,
            rho_dual_x_init: float | None = None,
            rho_dual_coc_init: float | None = None,
            rho_binary_init: float = 1.0,
            rho_opt_init: float = 1e-2,
            loss_ratio_primal: float = 1.0,
            loss_ratio_dual_x: float = 1.0,
            loss_ratio_opt: float = 1.0,
            loss_ratio_reg: float = 1.0,
            gamma_base: float = 1e-2,
            mu_dual_floor_init: float = 0.1,
            ita_dual_floor_init: float = 0.1,
            zeta_ita_cap_penalty_weight: float = 0.0,
            zeta_ita_cap_initial_weight: float | None = None,
            zeta_ita_cap_final_weight: float | None = None,
            zeta_ita_cap_initial: float | None = None,
            zeta_ita_cap_final: float | None = None,
            zeta_ita_cap_start_round: int = 0,
            zeta_ita_cap_end_round: int = 0,
            iter_delta_reg_weight: float = BCD_ITER_DELTA_REG_WEIGHT,
            iter_delta_reg_deadband: float = BCD_ITER_DELTA_REG_DEADBAND,
            nn_size: str = 'medium',
            nn_hidden_dims: list[int] | None = None,
            nn_batch_strategy: str = 'full-batch',
            nn_batch_size: int = 4,
            nn_shuffle: bool = True,
            nn_learning_rate: float = 5e-5,
            nn_smooth_abs_eps: float = BCD_NN_SMOOTH_ABS_EPS,
            direct_train_config: dict | None = None,
            pg_block_prox_weight: float = BCD_PG_BLOCK_PROX_WEIGHT,
            dual_block_prox_weight: float = BCD_DUAL_BLOCK_PROX_WEIGHT):
    ensure_bcd_modules_imported()
    log("Mode: main-problem BCD training (Agent_NN_BCD)")
    log(f"Using {len(all_samples)} samples")
    """BCD 主代理训练（样本级并行），返回 ParallelAgent_NN_BCD 实例。"""
    log("模式: BCD 主代理训练（Agent_NN_BCD）")
    log(f"使用 {len(all_samples)} 个样本")
    log(
        f"theta热启动={theta_hot_start_strategy}, "
        f"zeta热启动={zeta_hot_start_strategy}, "
        f"nn_dropout={'on' if enable_dropout_during_nn_training else 'off'}, "
        f"nn_size={nn_size}, nn_hidden_dims={nn_hidden_dims}, "
        f"nn_batch={nn_batch_strategy}, nn_batch_size={nn_batch_size}, "
        f"nn_shuffle={nn_shuffle}, nn_lr={nn_learning_rate}, "
        f"nn_smooth_eps={nn_smooth_abs_eps}"
    )
    log(
        f"rho_init: primal={rho_primal_init}, dual={rho_dual_init}, "
        f"dual_pg={rho_dual_pg_init if rho_dual_pg_init is not None else rho_dual_init}, "
        f"dual_x={rho_dual_x_init if rho_dual_x_init is not None else rho_dual_init}, "
        f"dual_coc={rho_dual_coc_init if rho_dual_coc_init is not None else rho_dual_init}, "
        f"binary={rho_binary_init}, opt={rho_opt_init}"
    )
    log(
        f"bcd_loss_ratio: primal={loss_ratio_primal}, dual_x={loss_ratio_dual_x}, "
        f"opt={loss_ratio_opt}, reg={loss_ratio_reg}"
    )
    log(
        f"iter_delta_reg: bcd_weight={iter_delta_reg_weight}, "
        f"bcd_deadband={iter_delta_reg_deadband}"
    )
    log(
        f"bcd_prox: pg_block={pg_block_prox_weight}, "
        f"dual_block={dual_block_prox_weight}"
    )
    log(
        f"bcd_schedule: max_iter={MAX_ITER}, nn_epochs_per_iter={NN_EPOCHS}, "
        f"total_nn_epochs={MAX_ITER * NN_EPOCHS}"
    )
    log(
        f"bcd_direct_nn: epochs={int((direct_train_config or {}).get('direct_epochs', 0) or 0)}, "
        f"lr={(direct_train_config or {}).get('direct_lr', None)}, "
        f"batch={(direct_train_config or {}).get('direct_batch_strategy', None)}/"
        f"{(direct_train_config or {}).get('direct_batch_size', None)}"
    )
    log(
        f"bcd_solver: lp_backend={lp_backend}, n_workers={n_workers}, "
        f"gurobi_threads={gurobi_threads}"
    )
    if theta_training_stages:
        log(f"theta staged training: {theta_training_stages}")
    if theta_curriculum_rounds or theta_curriculum_delay_rounds or theta_initial_scale != 1.0 or theta_final_scale != 1.0:
        log(
            f"theta curriculum: delay={theta_curriculum_delay_rounds}, "
            f"rounds={theta_curriculum_rounds}, scale={theta_initial_scale}->{theta_final_scale}"
        )
    if (
        (zeta_ita_cap_penalty_weight or 0.0) > 0.0
        or (zeta_ita_cap_initial_weight or 0.0) > 0.0
        or (zeta_ita_cap_final_weight or 0.0) > 0.0
    ):
        log(
            f"zeta ita soft-cap: weight={zeta_ita_cap_initial_weight if zeta_ita_cap_initial_weight is not None else zeta_ita_cap_penalty_weight}"
            f"->{zeta_ita_cap_final_weight if zeta_ita_cap_final_weight is not None else zeta_ita_cap_penalty_weight}, "
            f"cap={zeta_ita_cap_initial}->{zeta_ita_cap_final}, "
            f"rounds={zeta_ita_cap_start_round}->{zeta_ita_cap_end_round}"
        )
    if n_workers <= 1:
        log(f"Initializing serial Agent_NN_BCD, max_iter={MAX_ITER}")
    else:
        log(f"Initializing ParallelAgent_NN_BCD, max_iter={MAX_ITER}, n_workers={n_workers}")

    unit_predictor_obj = None
    if use_unit_predictor:
        resolved_unit_predictor_path = _resolve_unit_predictor_checkpoint_path(unit_predictor_load_path)
        if resolved_unit_predictor_path is not None and not resolved_unit_predictor_path.is_file():
            raise FileNotFoundError(
                f"BCD unit_predictor checkpoint was requested but not found: {resolved_unit_predictor_path}"
            )
        unit_predictor_save_path = str(bcd_model_dir / 'unit_predictor.pth')
        effective_unit_predictor_epochs = 0 if resolved_unit_predictor_path else int(unit_predictor_epochs)
        log(
            f"[BCD][unit_predictor] enabled, load_path={resolved_unit_predictor_path}, "
            f"epochs={effective_unit_predictor_epochs}, save={unit_predictor_save_path}, "
            f"warmup={unit_predictor_warmup_rounds}, theta_delay={theta_constraint_delay_rounds}"
        )
        unit_predictor_obj = train_unit_predictor_from_data(
            ppc,
            all_samples,
            T_DELTA,
            num_epochs=effective_unit_predictor_epochs,
            hidden_dims=unit_predictor_hidden_dims,
            batch_size=unit_predictor_batch_size,
            batch_strategy=unit_predictor_batch_strategy,
            shuffle=unit_predictor_shuffle,
            learning_rate=unit_predictor_lr,
            weight_decay=unit_predictor_weight_decay,
            net_variant=unit_predictor_net_variant,
            tcn_channels=unit_predictor_tcn_channels,
            tcn_depth=unit_predictor_tcn_depth,
            tconv_channels=unit_predictor_tconv_channels,
            tconv_depth=unit_predictor_tconv_depth,
            dropout=unit_predictor_dropout,
            enable_pos_weight=unit_predictor_enable_pos_weight,
            pos_weight_clip=unit_predictor_pos_weight_clip,
            loss_weight_bce=unit_predictor_loss_weight_bce,
            loss_weight_mse=unit_predictor_loss_weight_mse,
            loss_weight_l1=unit_predictor_loss_weight_l1,
            loss_weight_tv=unit_predictor_loss_weight_tv,
            loss_weight_transition=unit_predictor_loss_weight_transition,
            loss_weight_binarize=unit_predictor_loss_weight_binarize,
            loss_weight_std_floor=unit_predictor_loss_weight_std_floor,
            std_floor_scale=unit_predictor_std_floor_scale,
            loss_weight_tv_floor=unit_predictor_loss_weight_tv_floor,
            tv_floor_scale=unit_predictor_tv_floor_scale,
            save_path=unit_predictor_save_path,
            load_path=str(resolved_unit_predictor_path) if resolved_unit_predictor_path else None,
        )

    print("\n" + "=" * 70)
    print("=" * 70)

    if external_sparse_templates is not None and n_workers > 1:
        log("Warning: external_sparse_templates currently forces serial Agent_NN_BCD; overriding n_workers to 1")

    if n_workers <= 1:
        log("Using serial Agent_NN_BCD")
    else:
        log(f"Using ParallelAgent_NN_BCD (n_workers={n_workers})")

    if external_sparse_templates is not None and n_workers > 1:
        log("Warning: external_sparse_templates only supports serial Agent_NN_BCD; forcing n_workers=1")
        n_workers = 1

    if n_workers <= 1:
        agent = Agent_NN_BCD(
            ppc,
            all_samples,
            T_DELTA,
            lp_backend=lp_backend,
            gurobi_threads=gurobi_threads,
            gurobi_lp_method=gurobi_lp_method,
            bcd_highs_threads=bcd_highs_threads,
            external_sparse_templates=external_sparse_templates,
            lambda_init_strategy=lambda_init_strategy,
            max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
            theta_curriculum_rounds=theta_curriculum_rounds,
            theta_initial_scale=theta_initial_scale,
            theta_final_scale=theta_final_scale,
            theta_curriculum_delay_rounds=theta_curriculum_delay_rounds,
            theta_hot_start_strategy=theta_hot_start_strategy,
            zeta_hot_start_strategy=zeta_hot_start_strategy,
            theta_gaussian_std=theta_gaussian_std,
            zeta_gaussian_std=zeta_gaussian_std,
            unit_predictor=unit_predictor_obj,
            use_unit_predictor=(unit_predictor_obj is not None),
            unit_predictor_warmup_rounds=unit_predictor_warmup_rounds,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            theta_constraint_delay_rounds=theta_constraint_delay_rounds,
            enable_dropout_during_nn_training=enable_dropout_during_nn_training,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_dual_floor_init=mu_dual_floor_init,
            ita_dual_floor_init=ita_dual_floor_init,
            zeta_ita_cap_penalty_weight=zeta_ita_cap_penalty_weight,
            zeta_ita_cap_initial_weight=zeta_ita_cap_initial_weight,
            zeta_ita_cap_final_weight=zeta_ita_cap_final_weight,
            zeta_ita_cap_initial=zeta_ita_cap_initial,
            zeta_ita_cap_final=zeta_ita_cap_final,
            zeta_ita_cap_start_round=zeta_ita_cap_start_round,
            zeta_ita_cap_end_round=zeta_ita_cap_end_round,
            dual_sign_relax_interval=DUAL_SIGN_RELAX_INTERVAL,
            nn_hidden_dims=nn_hidden_dims,
            nn_learning_rate=nn_learning_rate,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            nn_smooth_abs_eps=nn_smooth_abs_eps,
            direct_train_config=direct_train_config,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_x=loss_ratio_dual_x,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            iter_delta_reg_weight=iter_delta_reg_weight,
            iter_delta_reg_deadband=iter_delta_reg_deadband,
        )
    else:
        agent = ParallelAgent_NN_BCD(
            ppc,
            all_samples,
            T_DELTA,
            lp_backend=lp_backend,
            gurobi_threads=gurobi_threads,
            gurobi_lp_method=gurobi_lp_method,
            bcd_highs_threads=bcd_highs_threads,
            lambda_init_strategy=lambda_init_strategy,
            max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
            theta_curriculum_rounds=theta_curriculum_rounds,
            theta_initial_scale=theta_initial_scale,
            theta_final_scale=theta_final_scale,
            theta_curriculum_delay_rounds=theta_curriculum_delay_rounds,
            theta_hot_start_strategy=theta_hot_start_strategy,
            zeta_hot_start_strategy=zeta_hot_start_strategy,
            theta_gaussian_std=theta_gaussian_std,
            zeta_gaussian_std=zeta_gaussian_std,
            unit_predictor=unit_predictor_obj,
            use_unit_predictor=(unit_predictor_obj is not None),
            unit_predictor_warmup_rounds=unit_predictor_warmup_rounds,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            theta_constraint_delay_rounds=theta_constraint_delay_rounds,
            enable_dropout_during_nn_training=enable_dropout_during_nn_training,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_dual_floor_init=mu_dual_floor_init,
            ita_dual_floor_init=ita_dual_floor_init,
            zeta_ita_cap_penalty_weight=zeta_ita_cap_penalty_weight,
            zeta_ita_cap_initial_weight=zeta_ita_cap_initial_weight,
            zeta_ita_cap_final_weight=zeta_ita_cap_final_weight,
            zeta_ita_cap_initial=zeta_ita_cap_initial,
            zeta_ita_cap_final=zeta_ita_cap_final,
            zeta_ita_cap_start_round=zeta_ita_cap_start_round,
            zeta_ita_cap_end_round=zeta_ita_cap_end_round,
            dual_sign_relax_interval=DUAL_SIGN_RELAX_INTERVAL,
            nn_hidden_dims=nn_hidden_dims,
            nn_learning_rate=nn_learning_rate,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            nn_smooth_abs_eps=nn_smooth_abs_eps,
            direct_train_config=direct_train_config,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_x=loss_ratio_dual_x,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            iter_delta_reg_weight=iter_delta_reg_weight,
            iter_delta_reg_deadband=iter_delta_reg_deadband,
            n_workers=n_workers,
        )

    print("\n" + "=" * 70)
    log("Starting BCD iterative training")
    print("=" * 70)

    if load_model_path is not None:
        log(f"Continuing training from existing BCD checkpoint: {load_model_path}")
        agent.load_model_parameters(
            str(load_model_path),
            restore_rho_state=restore_rho_from_checkpoint,
        )
        agent.theta_constraint_delay_rounds = max(int(theta_constraint_delay_rounds), 0)
    if logger is not None:
        agent.logger = logger
    agent.iter(
        max_iter=MAX_ITER,
        dual_decay_round=DUAL_DECAY_ROUND,
        dual_sign_relax_interval=DUAL_SIGN_RELAX_INTERVAL,
        nn_epochs=NN_EPOCHS,
        theta_training_stages=theta_training_stages,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        nn_learning_rate=nn_learning_rate,
        direct_train_config=direct_train_config,
    )

    # 保存模型（含算例名和时间戳）
    suffix = f'_{case_name}_{timestamp}' if timestamp else f'_{case_name}'
    save_path = str(bcd_model_dir / f'bcd_model{suffix}.pth')
    try:
        agent.save_model_parameters(save_path)
        log(f"BCD 模型参数保存至: {save_path}")
    except Exception as e:
        log(f"模型保存失败（非致命）: {e}")

    return agent


def build_sparse_template_library_from_bcd_agent(
    agent,
    top_k_variables: int,
    max_groups: int,
    group_size: int,
    sparse_dir: Path,
    case_name: str,
    timestamp: str,
):
    """从 BCD 初始化结果中发现高价值 x[g,t] 变量，并构建 sparse 模板库。"""
    from sparse_support_discovery import (
        discover_sparse_supports,
        extract_support_discovery_samples_from_agent,
    )
    from sparse_constraint_templates import build_sparse_template_library

    log("开始 sparse 支持集发现")
    sparse_samples = extract_support_discovery_samples_from_agent(agent)
    discovery_result = discover_sparse_supports(
        sparse_samples,
        top_k_variables=top_k_variables,
        max_groups=max_groups,
        group_size=group_size,
    )
    template_library = build_sparse_template_library(
        discovery_result,
        sparse_samples,
        max_templates=max_groups,
    )

    sparse_dir.mkdir(parents=True, exist_ok=True)
    support_path = sparse_dir / f'sparse_supports_{case_name}_{timestamp}.json'
    template_path = sparse_dir / f'sparse_templates_{case_name}_{timestamp}.json'
    discovery_result.to_json(support_path)
    template_library.to_json(template_path)

    log(f"  已选变量数: {len(discovery_result.selected_variables)}")
    log(f"  已构造模板数: {len(template_library.templates)}")
    log(f"  sparse supports 保存至: {support_path}")
    log(f"  sparse templates 保存至: {template_path}")
    return discovery_result, template_library


def run_sparse_bcd(ppc, all_samples: list, T_DELTA, MAX_ITER, bcd_model_dir,
                   case_name: str = 'case', timestamp: str = '',
                   NN_EPOCHS: int = 10, DUAL_DECAY_ROUND: int = 10,
                   DUAL_SIGN_RELAX_INTERVAL: int | None = None,
                   top_k_variables: int = 20, max_groups: int = 5, group_size: int = 3,
                   logger: 'TrainingLogger | None' = None,
                   lambda_init_strategy: str = 'lp_relaxation',
                   max_theta_constraints_per_time_slot: int = 10,
                   theta_hot_start_strategy: str = 'dcpf_relative',
                   zeta_hot_start_strategy: str = 'zero',
                   theta_gaussian_std: float = 0.01,
                   zeta_gaussian_std: float = 0.01,
                   enable_dropout_during_nn_training: bool = True,
                   rho_primal_init: float = 1e-2,
                   rho_dual_init: float = 1e-2,
                   rho_dual_pg_init: float | None = None,
                   rho_dual_x_init: float | None = None,
                   rho_dual_coc_init: float | None = None,
                   rho_binary_init: float = 1.0,
                   rho_opt_init: float = 1e-2,
                   loss_ratio_primal: float = 1.0,
                   loss_ratio_dual_x: float = 1.0,
                   loss_ratio_opt: float = 1.0,
                   loss_ratio_reg: float = 1.0,
                   gamma_base: float = 1e-2,
                   mu_dual_floor_init: float = 0.1,
                   ita_dual_floor_init: float = 0.1,
                   iter_delta_reg_weight: float = BCD_ITER_DELTA_REG_WEIGHT,
                   iter_delta_reg_deadband: float = BCD_ITER_DELTA_REG_DEADBAND,
                   nn_size: str = 'medium',
                   nn_hidden_dims: list[int] | None = None,
                   nn_batch_strategy: str = 'full-batch',
                   nn_batch_size: int = 4,
                   nn_shuffle: bool = True,
                   nn_learning_rate: float = 5e-5,
                   pg_block_prox_weight: float = BCD_PG_BLOCK_PROX_WEIGHT,
                   dual_block_prox_weight: float = BCD_DUAL_BLOCK_PROX_WEIGHT):
    """
    sparse 模式：
    1. 用普通 Agent_NN_BCD 初始化，拿到 x_lp/x_true
    2. 离线发现高价值 x[g,t] 变量并构建 sparse 模板
    3. 用 external_sparse_templates 启动真正的 sparse-BCD 训练
    """
    log("模式: Sparse 支持集发现 → Sparse BCD 训练")
    log(f"使用 {len(all_samples)} 个样本")
    print("\n" + "=" * 70)
    log("Step 1/3: 初始化 bootstrap Agent_NN_BCD 用于 sparse 变量发现")
    print("=" * 70)

    bootstrap_agent = Agent_NN_BCD(
        ppc,
        all_samples,
        T_DELTA,
        lambda_init_strategy=lambda_init_strategy,
        max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
        theta_hot_start_strategy=theta_hot_start_strategy,
        zeta_hot_start_strategy=zeta_hot_start_strategy,
        theta_gaussian_std=theta_gaussian_std,
        zeta_gaussian_std=zeta_gaussian_std,
        enable_dropout_during_nn_training=enable_dropout_during_nn_training,
        rho_primal_init=rho_primal_init,
        rho_dual_init=rho_dual_init,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        rho_binary_init=rho_binary_init,
        rho_opt_init=rho_opt_init,
        loss_ratio_primal=loss_ratio_primal,
        loss_ratio_dual_x=loss_ratio_dual_x,
        loss_ratio_opt=loss_ratio_opt,
        loss_ratio_reg=loss_ratio_reg,
        gamma_base=gamma_base,
        mu_dual_floor_init=mu_dual_floor_init,
        ita_dual_floor_init=ita_dual_floor_init,
        iter_delta_reg_weight=iter_delta_reg_weight,
        iter_delta_reg_deadband=iter_delta_reg_deadband,
        pg_block_prox_weight=pg_block_prox_weight,
        dual_block_prox_weight=dual_block_prox_weight,
        nn_hidden_dims=nn_hidden_dims,
        nn_learning_rate=nn_learning_rate,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        direct_train_config=build_bcd_direct_train_config(),
    )
    sparse_dir = Path(__file__).parent / 'result' / 'sparse_templates'
    discovery_result, template_library = build_sparse_template_library_from_bcd_agent(
        bootstrap_agent,
        top_k_variables=top_k_variables,
        max_groups=max_groups,
        group_size=group_size,
        sparse_dir=sparse_dir,
        case_name=case_name,
        timestamp=timestamp,
    )

    print("\n" + "=" * 70)
    log("Step 2/3: 使用 sparse 模板启动 BCD 训练")
    print("=" * 70)
    agent = run_bcd(
        ppc,
        all_samples,
        T_DELTA,
        MAX_ITER,
        bcd_model_dir,
        case_name=case_name,
        timestamp=timestamp,
        n_workers=1,
        NN_EPOCHS=NN_EPOCHS,
        DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
        DUAL_SIGN_RELAX_INTERVAL=BCD_DUAL_SIGN_RELAX_INTERVAL,
        lp_backend=BCD_LP_BACKEND,
        gurobi_threads=BCD_GUROBI_THREADS,
        logger=logger,
        external_sparse_templates=template_library,
        lambda_init_strategy=lambda_init_strategy,
        max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
        theta_hot_start_strategy=theta_hot_start_strategy,
        zeta_hot_start_strategy=zeta_hot_start_strategy,
        theta_gaussian_std=theta_gaussian_std,
        zeta_gaussian_std=zeta_gaussian_std,
        enable_dropout_during_nn_training=enable_dropout_during_nn_training,
        rho_primal_init=rho_primal_init,
        rho_dual_init=rho_dual_init,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        rho_binary_init=rho_binary_init,
        rho_opt_init=rho_opt_init,
        loss_ratio_primal=loss_ratio_primal,
        loss_ratio_dual_x=loss_ratio_dual_x,
        loss_ratio_opt=loss_ratio_opt,
        loss_ratio_reg=loss_ratio_reg,
        gamma_base=gamma_base,
        mu_dual_floor_init=mu_dual_floor_init,
        ita_dual_floor_init=ita_dual_floor_init,
        iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
        iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
        pg_block_prox_weight=pg_block_prox_weight,
        dual_block_prox_weight=dual_block_prox_weight,
        nn_size=nn_size,
        nn_hidden_dims=nn_hidden_dims,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        nn_learning_rate=nn_learning_rate,
    )

    return agent, discovery_result, template_library


def run_feasibility_pump_test(ppc, all_samples, dual_predictor, trainers,
                               T_DELTA, FP_TEST_SAMPLES):
    """对多个样本运行可行性泵并汇总结果。"""
    test_n = min(FP_TEST_SAMPLES, len(all_samples))
    print("\n" + "=" * 70)
    log(f"可行性泵测试: {test_n} 个样本")
    print("=" * 70)

    results = []
    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']   # (nb, T) net load compatibility field
        log(f"  样本 {i + 1}/{test_n}，pd_data shape={pd_data.shape}")
        try:
            x_result, success = recover_integer_solution(
                sample, trainers, dual_predictor, ppc, T_DELTA,
                scenario_bank=all_samples,
                verbose=True,
            )
        except Exception as e:
            log(f"    异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((i, False, None))
            continue

        status = "成功" if success else "失败"
        log(f"    可行性泵结果: {status}")
        results.append((i, success, x_result))

    n_success = sum(1 for _, s, _ in results if s)
    print("\n" + "=" * 70)
    log(f"可行性泵完成: {n_success}/{test_n} 样本找到可行解")
    print("=" * 70)
    return results


def _case_env_prefix(case_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(case_name).upper())


def _resolve_unit_predictor_checkpoint_path(raw_path: str | os.PathLike | None) -> Path | None:
    """Resolve file/dir/LATEST.txt forms used by standalone UnitPredictor scripts."""
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    root = Path(__file__).parent
    p = Path(text)
    if not p.is_absolute():
        p = root / p

    if p.is_dir():
        latest = p / "LATEST.txt"
        if latest.is_file():
            target = latest.read_text(encoding="utf-8").strip()
            if target:
                t = Path(target)
                if t.is_absolute():
                    return t.resolve()
                cand = (p / t).resolve()
                if cand.exists():
                    return cand
                return (root / t).resolve()
        return (p / "unit_predictor.pth").resolve()

    if p.is_file() and p.name.lower() == "latest.txt":
        target = p.read_text(encoding="utf-8").strip()
        if not target:
            return None
        t = Path(target)
        if t.is_absolute():
            return t.resolve()
        cand = (p.parent / t).resolve()
        if cand.exists():
            return cand
        return (root / t).resolve()

    return p.resolve()


def _latest_standalone_unit_predictor_ckpt(case_name: str) -> Path | None:
    base = Path(__file__).parent / "result" / "surrogate_models"
    if not base.is_dir():
        return None
    prefix = f"unit_predictor_{case_name}_"
    best: tuple[float, Path] | None = None
    for d in base.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        ckpt = d / "unit_predictor.pth"
        if not ckpt.is_file():
            continue
        mtime = ckpt.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, ckpt)
    return best[1].resolve() if best else None


def _read_unit_predictor_metadata(checkpoint_path: Path) -> dict:
    try:
        import torch

        state = torch.load(str(checkpoint_path), map_location="cpu")
        meta = state.get("metadata", {}) if isinstance(state, dict) else {}
        return dict(meta or {})
    except Exception as exc:
        log(f"[unit_predictor] 无法读取 checkpoint metadata（将沿用当前配置）: {checkpoint_path} ({exc})")
        return {}


def _apply_unit_predictor_metadata_config(metadata: dict) -> bool:
    if not metadata:
        return False
    global UNIT_PREDICTOR_HIDDEN_DIMS
    global UNIT_PREDICTOR_NET_VARIANT
    global UNIT_PREDICTOR_TCN_CHANNELS
    global UNIT_PREDICTOR_TCN_DEPTH
    global UNIT_PREDICTOR_TCONV_CHANNELS
    global UNIT_PREDICTOR_TCONV_DEPTH
    global UNIT_PREDICTOR_DROPOUT

    changed = False
    if metadata.get("hidden_dims") is not None:
        UNIT_PREDICTOR_HIDDEN_DIMS = list(metadata["hidden_dims"])
        changed = True
    if metadata.get("net_variant") is not None:
        UNIT_PREDICTOR_NET_VARIANT = str(metadata["net_variant"])
        changed = True
    if metadata.get("tcn_channels") is not None:
        UNIT_PREDICTOR_TCN_CHANNELS = int(metadata["tcn_channels"])
        changed = True
    if metadata.get("tcn_depth") is not None:
        UNIT_PREDICTOR_TCN_DEPTH = int(metadata["tcn_depth"])
        changed = True
    if metadata.get("tconv_channels") is not None:
        UNIT_PREDICTOR_TCONV_CHANNELS = int(metadata["tconv_channels"])
        changed = True
    if metadata.get("tconv_depth") is not None:
        UNIT_PREDICTOR_TCONV_DEPTH = int(metadata["tconv_depth"])
        changed = True
    if metadata.get("dropout") is not None:
        UNIT_PREDICTOR_DROPOUT = float(metadata["dropout"])
        changed = True
    return changed


def _configure_unit_predictor_load_path_from_runtime() -> None:
    """Apply case/env/standalone UnitPredictor runtime defaults before config snapshot."""
    global UNIT_PREDICTOR_LOAD_PATH

    if not bool(USE_UNIT_PREDICTOR):
        return

    case_prefix = _case_env_prefix(CASE_NAME)
    env_specific = os.environ.get(f"{case_prefix}_UNIT_PREDICTOR_LOAD_PATH", "").strip()
    env_generic = os.environ.get("UNIT_PREDICTOR_LOAD_PATH", "").strip()

    configured = UNIT_PREDICTOR_LOAD_PATH
    source = "configured"
    if env_specific:
        configured = env_specific
        source = f"env:{case_prefix}_UNIT_PREDICTOR_LOAD_PATH"
    elif env_generic:
        configured = env_generic
        source = "env:UNIT_PREDICTOR_LOAD_PATH"

    resolved = _resolve_unit_predictor_checkpoint_path(configured)
    if resolved is None and bool(UNIT_PREDICTOR_AUTO_LATEST_STANDALONE):
        resolved = _latest_standalone_unit_predictor_ckpt(CASE_NAME)
        source = "auto_latest_standalone"

    if resolved is None:
        return

    UNIT_PREDICTOR_LOAD_PATH = str(resolved)
    if bool(UNIT_PREDICTOR_LOAD_METADATA_CONFIG) and resolved.is_file():
        metadata = _read_unit_predictor_metadata(resolved)
        if _apply_unit_predictor_metadata_config(metadata):
            log(
                "[unit_predictor] 已按 checkpoint metadata 对齐配置: "
                f"source={source}, path={resolved}, "
                f"net={UNIT_PREDICTOR_NET_VARIANT}, "
                f"tcn=({UNIT_PREDICTOR_TCN_CHANNELS},{UNIT_PREDICTOR_TCN_DEPTH}), "
                f"tconv=({UNIT_PREDICTOR_TCONV_CHANNELS},{UNIT_PREDICTOR_TCONV_DEPTH}), "
                f"dropout={UNIT_PREDICTOR_DROPOUT}"
            )
            return

    log(f"[unit_predictor] 使用 standalone checkpoint: source={source}, path={resolved}")


# ──────────────────────── 主函数 ────────────────────────

def main():
    _bootstrap_runtime_environment()
    _configure_unit_predictor_load_path_from_runtime()
    start_time = time.time()

    print("=" * 70)
    print(f"Training script mode: {MODE}")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────
    # 顶部集中配置区：本函数内只做派生值整理
    CONSTRAINT_GENERATION_STRATEGY = SURROGATE_CONSTRAINT_STRATEGY
    THETA_WARM_START_STRATEGY = THETA_HOT_START_STRATEGY
    ZETA_WARM_START_STRATEGY = ZETA_HOT_START_STRATEGY
    THETA_WARM_START_GAUSSIAN_STD = THETA_GAUSSIAN_STD
    ZETA_WARM_START_GAUSSIAN_STD = ZETA_GAUSSIAN_STD
    BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE = BCD_ENABLE_DROPOUT_DURING_NN_TRAINING
    BCD_NN_BATCH_STRATEGY_VALUE = BCD_NN_BATCH_STRATEGY
    BCD_NN_BATCH_SIZE_VALUE = BCD_NN_BATCH_SIZE
    BCD_NN_SHUFFLE_VALUE = BCD_NN_SHUFFLE
    BCD_NN_LR_VALUE = BCD_NN_LR
    BCD_NN_SIZE_VALUE, BCD_NN_HIDDEN_DIMS_VALUE = resolve_nn_hidden_dims(
        BCD_NN_SIZE,
        BCD_NN_HIDDEN_DIM_OPTIONS,
        'BCD_NN_SIZE',
    )
    DUAL_BATCH_STRATEGY_VALUE = DUAL_BATCH_STRATEGY
    DUAL_SHUFFLE_VALUE = DUAL_SHUFFLE
    DUAL_LR_VALUE = DUAL_LR
    DUAL_PREDICTOR_NET_VARIANT_VALUE = DUAL_PREDICTOR_NET_VARIANT
    DUAL_PREDICTOR_NORMALIZE_TARGETS_VALUE = DUAL_PREDICTOR_NORMALIZE_TARGETS
    DUAL_PREDICTOR_COSINE_LOSS_WEIGHT_VALUE = DUAL_PREDICTOR_COSINE_LOSS_WEIGHT
    DUAL_PREDICTOR_SMOOTH_L1_BETA_VALUE = DUAL_PREDICTOR_SMOOTH_L1_BETA
    USE_UNIT_PREDICTOR_VALUE = USE_UNIT_PREDICTOR
    UNIT_PREDICTOR_EPOCHS_VALUE = UNIT_PREDICTOR_EPOCHS
    UNIT_PREDICTOR_BATCH_STRATEGY_VALUE = UNIT_PREDICTOR_BATCH_STRATEGY
    UNIT_PREDICTOR_BATCH_SIZE_VALUE = UNIT_PREDICTOR_BATCH_SIZE
    UNIT_PREDICTOR_SHUFFLE_VALUE = UNIT_PREDICTOR_SHUFFLE
    UNIT_PREDICTOR_LR_VALUE = UNIT_PREDICTOR_LR
    UNIT_PREDICTOR_HIDDEN_DIMS_VALUE = UNIT_PREDICTOR_HIDDEN_DIMS
    UNIT_PREDICTOR_NET_VARIANT_VALUE = UNIT_PREDICTOR_NET_VARIANT
    UNIT_PREDICTOR_TCN_CHANNELS_VALUE = UNIT_PREDICTOR_TCN_CHANNELS
    UNIT_PREDICTOR_TCN_DEPTH_VALUE = UNIT_PREDICTOR_TCN_DEPTH
    UNIT_PREDICTOR_TCONV_CHANNELS_VALUE = UNIT_PREDICTOR_TCONV_CHANNELS
    UNIT_PREDICTOR_TCONV_DEPTH_VALUE = UNIT_PREDICTOR_TCONV_DEPTH
    UNIT_PREDICTOR_DROPOUT_VALUE = UNIT_PREDICTOR_DROPOUT
    UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE = UNIT_PREDICTOR_ENABLE_POS_WEIGHT
    UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE = UNIT_PREDICTOR_POS_WEIGHT_CLIP
    UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_BCE
    UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_MSE
    UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_L1
    UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_TV
    UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION
    UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE
    UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR
    UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE = UNIT_PREDICTOR_STD_FLOOR_SCALE
    UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE = UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR
    UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE = UNIT_PREDICTOR_TV_FLOOR_SCALE
    UNIT_PREDICTOR_LOAD_PATH_VALUE = UNIT_PREDICTOR_LOAD_PATH
    UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS_VALUE = UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS
    UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT_VALUE = UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT
    UNIT_PREDICTOR_AUTO_EXTRA_TV_VALUE = UNIT_PREDICTOR_AUTO_EXTRA_TV
    UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS_VALUE = UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS
    UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD_VALUE = UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD
    UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT_VALUE = UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT
    UNIT_PREDICTOR_FINETUNE_LR_VALUE = UNIT_PREDICTOR_FINETUNE_LR
    UNIT_PREDICTOR_WEIGHT_DECAY_VALUE = UNIT_PREDICTOR_WEIGHT_DECAY
    BCD_MAX_ITER_VALUE = BCD_MAX_ITER
    SUBPROBLEM_MAX_ITER_VALUE = SUBPROBLEM_MAX_ITER
    BCD_LAMBDA_INIT_STRATEGY_VALUE = BCD_LAMBDA_INIT_STRATEGY
    BCD_RHO_PRIMAL_INIT_VALUE = BCD_RHO_PRIMAL_INIT
    BCD_RHO_DUAL_INIT_VALUE = BCD_RHO_DUAL_INIT
    BCD_RHO_DUAL_PG_INIT_VALUE = BCD_RHO_DUAL_PG_INIT
    BCD_RHO_DUAL_X_INIT_VALUE = BCD_RHO_DUAL_X_INIT
    BCD_RHO_DUAL_COC_INIT_VALUE = BCD_RHO_DUAL_COC_INIT
    BCD_RHO_BINARY_INIT_VALUE = BCD_RHO_BINARY_INIT
    BCD_RHO_OPT_INIT_VALUE = BCD_RHO_OPT_INIT
    BCD_LOSS_RATIO_PRIMAL_VALUE = BCD_LOSS_RATIO_PRIMAL
    BCD_LOSS_RATIO_DUAL_X_VALUE = BCD_LOSS_RATIO_DUAL_X
    BCD_LOSS_RATIO_OPT_VALUE = BCD_LOSS_RATIO_OPT
    BCD_LOSS_RATIO_REG_VALUE = BCD_LOSS_RATIO_REG
    BCD_NN_SMOOTH_ABS_EPS_VALUE = BCD_NN_SMOOTH_ABS_EPS
    BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE = BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT
    BCD_THETA_TRAINING_STAGES_VALUE = BCD_THETA_TRAINING_STAGES
    BCD_THETA_CURRICULUM_ROUNDS_VALUE = BCD_THETA_CURRICULUM_ROUNDS
    BCD_THETA_INITIAL_SCALE_VALUE = BCD_THETA_INITIAL_SCALE
    BCD_THETA_FINAL_SCALE_VALUE = BCD_THETA_FINAL_SCALE
    BCD_THETA_CURRICULUM_DELAY_ROUNDS_VALUE = BCD_THETA_CURRICULUM_DELAY_ROUNDS
    BCD_USE_UNIT_PREDICTOR_VALUE = BCD_USE_UNIT_PREDICTOR
    BCD_UNIT_PREDICTOR_WARMUP_ROUNDS_VALUE = BCD_UNIT_PREDICTOR_WARMUP_ROUNDS
    BCD_THETA_CONSTRAINT_DELAY_ROUNDS_VALUE = BCD_THETA_CONSTRAINT_DELAY_ROUNDS
    BCD_LP_BACKEND_VALUE = BCD_LP_BACKEND
    BCD_GUROBI_THREADS_VALUE = BCD_GUROBI_THREADS
    BCD_GUROBI_LP_METHOD_VALUE = BCD_GUROBI_LP_METHOD
    BCD_HIGHS_THREADS_VALUE = BCD_HIGHS_THREADS
    BCD_GAMMA_BASE_VALUE = BCD_GAMMA_BASE
    BCD_MU_DUAL_FLOOR_INIT_VALUE = BCD_MU_DUAL_FLOOR_INIT
    BCD_ITA_DUAL_FLOOR_INIT_VALUE = BCD_ITA_DUAL_FLOOR_INIT
    BCD_ZETA_ITA_CAP_PENALTY_WEIGHT_VALUE = BCD_ZETA_ITA_CAP_PENALTY_WEIGHT
    BCD_ZETA_ITA_CAP_INITIAL_WEIGHT_VALUE = BCD_ZETA_ITA_CAP_INITIAL_WEIGHT
    BCD_ZETA_ITA_CAP_FINAL_WEIGHT_VALUE = BCD_ZETA_ITA_CAP_FINAL_WEIGHT
    BCD_ZETA_ITA_CAP_INITIAL_VALUE = BCD_ZETA_ITA_CAP_INITIAL
    BCD_ZETA_ITA_CAP_FINAL_VALUE = BCD_ZETA_ITA_CAP_FINAL
    BCD_ZETA_ITA_CAP_START_ROUND_VALUE = BCD_ZETA_ITA_CAP_START_ROUND
    BCD_ZETA_ITA_CAP_END_ROUND_VALUE = BCD_ZETA_ITA_CAP_END_ROUND
    BCD_DUAL_SIGN_RELAX_INTERVAL_VALUE = BCD_DUAL_SIGN_RELAX_INTERVAL
    BCD_PG_BLOCK_PROX_WEIGHT_VALUE = BCD_PG_BLOCK_PROX_WEIGHT
    BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE = BCD_DUAL_BLOCK_PROX_WEIGHT
    SUBPROBLEM_RHO_PRIMAL_INIT_VALUE = SUBPROBLEM_RHO_PRIMAL_INIT
    SUBPROBLEM_RHO_DUAL_INIT_VALUE = SUBPROBLEM_RHO_DUAL_INIT
    SUBPROBLEM_RHO_DUAL_PG_INIT_VALUE = SUBPROBLEM_RHO_DUAL_PG_INIT
    SUBPROBLEM_RHO_DUAL_X_INIT_VALUE = SUBPROBLEM_RHO_DUAL_X_INIT
    SUBPROBLEM_RHO_DUAL_COC_INIT_VALUE = SUBPROBLEM_RHO_DUAL_COC_INIT
    SUBPROBLEM_RHO_BINARY_INIT_VALUE = SUBPROBLEM_RHO_BINARY_INIT
    SUBPROBLEM_RHO_BINARY_MAX_VALUE = SUBPROBLEM_RHO_BINARY_MAX
    SUBPROBLEM_RHO_OPT_INIT_VALUE = SUBPROBLEM_RHO_OPT_INIT
    SUBPROBLEM_LOSS_RATIO_PRIMAL_VALUE = SUBPROBLEM_LOSS_RATIO_PRIMAL
    SUBPROBLEM_LOSS_RATIO_DUAL_PG_VALUE = SUBPROBLEM_LOSS_RATIO_DUAL_PG
    SUBPROBLEM_LOSS_RATIO_DUAL_X_VALUE = SUBPROBLEM_LOSS_RATIO_DUAL_X
    SUBPROBLEM_NN_SMOOTH_ABS_EPS_VALUE = SUBPROBLEM_NN_SMOOTH_ABS_EPS
    SUBPROBLEM_NN_DUAL_TERM_INTERVAL_VALUE = SUBPROBLEM_NN_DUAL_TERM_INTERVAL
    SUBPROBLEM_LOSS_RATIO_OPT_VALUE = SUBPROBLEM_LOSS_RATIO_OPT
    SUBPROBLEM_LOSS_RATIO_REG_VALUE = SUBPROBLEM_LOSS_RATIO_REG
    SUBPROBLEM_GAMMA_BASE_VALUE = SUBPROBLEM_GAMMA_BASE
    SUBPROBLEM_MU_DUAL_FLOOR_INIT_VALUE = SUBPROBLEM_MU_DUAL_FLOOR_INIT
    SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND_VALUE = SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND
    SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND_VALUE = SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND
    SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS_VALUE = SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS
    SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS_VALUE = SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS
    SUBPROBLEM_SIGN4_INITIAL_SCALE_VALUE = SUBPROBLEM_SIGN4_INITIAL_SCALE
    SUBPROBLEM_SIGN4_FINAL_SCALE_VALUE = SUBPROBLEM_SIGN4_FINAL_SCALE
    SUBPROBLEM_SIGN4_DELAY_ROUNDS_VALUE = SUBPROBLEM_SIGN4_DELAY_ROUNDS
    SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT_VALUE = SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT
    SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS_VALUE = SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS
    SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE_VALUE = SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE
    SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR_VALUE = SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR
    SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL_VALUE = SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL
    SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS_VALUE = SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS
    SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE_VALUE = SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE
    SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS_VALUE = SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS
    SUBPROBLEM_NN_BATCH_STRATEGY_VALUE = SUBPROBLEM_NN_BATCH_STRATEGY
    SUBPROBLEM_NN_BATCH_SIZE_VALUE = SUBPROBLEM_NN_BATCH_SIZE
    SUBPROBLEM_NN_SHUFFLE_VALUE = SUBPROBLEM_NN_SHUFFLE
    SUBPROBLEM_NN_LR_VALUE = SUBPROBLEM_NN_LR
    SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO_VALUE = SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO
    SUBPROBLEM_NN_MAIN_LR_LATE_SCALE_VALUE = SUBPROBLEM_NN_MAIN_LR_LATE_SCALE
    SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY_VALUE = SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY
    SUBPROBLEM_NN_MAIN_GRAD_CLIP_VALUE = SUBPROBLEM_NN_MAIN_GRAD_CLIP
    SUBPROBLEM_NN_MAIN_KKT_LR_SCALE_VALUE = SUBPROBLEM_NN_MAIN_KKT_LR_SCALE
    SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS_VALUE = SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS
    SUBPROBLEM_NN_SIZE_VALUE, SUBPROBLEM_NN_HIDDEN_DIMS_VALUE = resolve_nn_hidden_dims(
        SUBPROBLEM_NN_SIZE,
        SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS,
        'SUBPROBLEM_NN_SIZE',
    )
    SUBPROBLEM_C_PG_NN_SIZE_VALUE, SUBPROBLEM_C_PG_NN_HIDDEN_DIMS_VALUE = resolve_nn_hidden_dims(
        SUBPROBLEM_C_PG_NN_SIZE,
        SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS,
        'SUBPROBLEM_C_PG_NN_SIZE',
    )
    SUBPROBLEM_X_COST_NN_LR_VALUE = SUBPROBLEM_X_COST_NN_LR
    SUBPROBLEM_PG_COST_NN_EPOCHS_VALUE = SUBPROBLEM_PG_COST_NN_EPOCHS
    SUBPROBLEM_PG_COST_START_ROUND_VALUE = SUBPROBLEM_PG_COST_START_ROUND
    SUBPROBLEM_PG_COST_SCALE_MULTIPLIER_VALUE = SUBPROBLEM_PG_COST_SCALE_MULTIPLIER
    SUBPROBLEM_PG_COST_LR_VALUE = SUBPROBLEM_PG_COST_LR
    SUBPROBLEM_PG_COST_SURR_LR_VALUE = SUBPROBLEM_PG_COST_SURR_LR
    SUBPROBLEM_PG_COST_REG_DEADBAND_VALUE = SUBPROBLEM_PG_COST_REG_DEADBAND
    SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT_VALUE = SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT
    SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS_VALUE = SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS
    SUBPROBLEM_PG_COST_BATCH_STRATEGY_VALUE = SUBPROBLEM_PG_COST_BATCH_STRATEGY
    SUBPROBLEM_PG_COST_BATCH_SIZE_VALUE = SUBPROBLEM_PG_COST_BATCH_SIZE
    SUBPROBLEM_PG_COST_SHUFFLE_VALUE = SUBPROBLEM_PG_COST_SHUFFLE
    SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS_VALUE = SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS
    SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER_VALUE = SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER
    SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP_VALUE = SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP
    SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE_VALUE = SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE
    SUBPROBLEM_PG_COST_C_PG_ADAM_WD_VALUE = SUBPROBLEM_PG_COST_C_PG_ADAM_WD
    SUBPROBLEM_MAIN_DIRECT_TRAIN_CONFIG_VALUE = build_subproblem_main_direct_train_config()
    SUBPROBLEM_C_PG_DIRECT_TRAIN_CONFIG_VALUE = build_subproblem_c_pg_direct_train_config()
    BCD_DIRECT_TRAIN_CONFIG_VALUE = build_bcd_direct_train_config()
    SUBPROBLEM_PG_BLOCK_PROX_WEIGHT_VALUE = SUBPROBLEM_PG_BLOCK_PROX_WEIGHT
    SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT_VALUE = SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT
    SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT_VALUE = SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT
    SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT_VALUE = SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT
    SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT_VALUE = SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT
    SUBPROBLEM_SINGLE_MU_CAP_INITIAL_VALUE = SUBPROBLEM_SINGLE_MU_CAP_INITIAL
    SUBPROBLEM_SINGLE_MU_CAP_FINAL_VALUE = SUBPROBLEM_SINGLE_MU_CAP_FINAL
    SUBPROBLEM_SINGLE_MU_CAP_START_ROUND_VALUE = SUBPROBLEM_SINGLE_MU_CAP_START_ROUND
    SUBPROBLEM_SINGLE_MU_CAP_END_ROUND_VALUE = SUBPROBLEM_SINGLE_MU_CAP_END_ROUND
    SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP_VALUE = SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP

    # 创建训练指标收集器
    logger = TrainingLogger()

    data_dir = Path(__file__).parent / 'result' / 'active_set'
    bcd_model_dir = Path(__file__).parent / 'result' / 'bcd_models'
    surrogate_model_dir = Path(__file__).parent / 'result' / 'surrogate_models'
    data_dir.mkdir(parents=True, exist_ok=True)
    bcd_model_dir.mkdir(parents=True, exist_ok=True)
    surrogate_model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = str(surrogate_model_dir / f'subproblem_models_{CASE_NAME}_{timestamp}')

    # ── 加载 PyPower 案例 ────────────────────────────────
    log(f"Loading PyPower case: {CASE_NAME}")
    supported_cases = ['case3', 'case3lite', 'case14', 'case30', 'case30lite', 'case30lite_perturbed', 'case39', 'case118']
    if CASE_NAME not in supported_cases:
        print(f"Unknown case: {CASE_NAME}. Supported cases: {supported_cases}")
        sys.exit(1)
    ppc = get_case_ppc(CASE_NAME)
    n_units = ppc['gen'].shape[0]
    n_buses = ppc['bus'].shape[0]
    log(f"Case summary: units={n_units}, buses={n_buses}")
    log(
        f"NN size config: BCD={BCD_NN_SIZE_VALUE} {BCD_NN_HIDDEN_DIMS_VALUE}, "
        f"subproblem={SUBPROBLEM_NN_SIZE_VALUE} {SUBPROBLEM_NN_HIDDEN_DIMS_VALUE}, "
        f"subproblem_c_pg={SUBPROBLEM_C_PG_NN_SIZE_VALUE} {SUBPROBLEM_C_PG_NN_HIDDEN_DIMS_VALUE}"
    )
    log(
        f"BCD direct-NN config: epochs={BCD_DIRECT_TRAIN_CONFIG_VALUE.get('direct_epochs')}, "
        f"lr={BCD_DIRECT_TRAIN_CONFIG_VALUE.get('direct_lr')}, "
        f"batch={BCD_DIRECT_TRAIN_CONFIG_VALUE.get('direct_batch_strategy')}/"
        f"{BCD_DIRECT_TRAIN_CONFIG_VALUE.get('direct_batch_size')}"
    )
    log(
        f"Loss ratio config: BCD(primal={BCD_LOSS_RATIO_PRIMAL_VALUE}, dual_x={BCD_LOSS_RATIO_DUAL_X_VALUE}, "
        f"opt={BCD_LOSS_RATIO_OPT_VALUE}, reg={BCD_LOSS_RATIO_REG_VALUE}); "
        f"subproblem(primal={SUBPROBLEM_LOSS_RATIO_PRIMAL_VALUE}, dual_pg={SUBPROBLEM_LOSS_RATIO_DUAL_PG_VALUE}, "
        f"dual_x={SUBPROBLEM_LOSS_RATIO_DUAL_X_VALUE}, nn_dual_interval={SUBPROBLEM_NN_DUAL_TERM_INTERVAL_VALUE}, "
        f"opt={SUBPROBLEM_LOSS_RATIO_OPT_VALUE}, "
        f"reg={SUBPROBLEM_LOSS_RATIO_REG_VALUE})"
    )
    log(
        f"Iteration config: bcd_max_iter={BCD_MAX_ITER_VALUE}, "
        f"subproblem_max_iter={SUBPROBLEM_MAX_ITER_VALUE}, joint_max_iter={JOINT_MAX_ITER}"
    )
    log(
        f"UnitPredictor config: use={USE_UNIT_PREDICTOR_VALUE}, "
        f"load_path={UNIT_PREDICTOR_LOAD_PATH_VALUE!r}, "
        f"auto_latest={UNIT_PREDICTOR_AUTO_LATEST_STANDALONE}, "
        f"net={UNIT_PREDICTOR_NET_VARIANT_VALUE}, "
        f"tcn=({UNIT_PREDICTOR_TCN_CHANNELS_VALUE},{UNIT_PREDICTOR_TCN_DEPTH_VALUE}), "
        f"tconv=({UNIT_PREDICTOR_TCONV_CHANNELS_VALUE},{UNIT_PREDICTOR_TCONV_DEPTH_VALUE}), "
        f"dropout={UNIT_PREDICTOR_DROPOUT_VALUE}, "
        f"predictor_warmup_rounds={SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS_VALUE}, "
        f"bcd_use={BCD_USE_UNIT_PREDICTOR_VALUE}, "
        f"bcd_warmup={BCD_UNIT_PREDICTOR_WARMUP_ROUNDS_VALUE}, "
        f"bcd_theta_delay={BCD_THETA_CONSTRAINT_DELAY_ROUNDS_VALUE}"
    )

    # ── 查找数据文件 ─────────────────────────────────────
    if ACTIVE_SETS_FILE is not None:
        data_file = Path(ACTIVE_SETS_FILE)
        if not data_file.is_absolute():
            data_file = Path(__file__).parent / data_file
        if not data_file.exists():
            log(f"Error: specified active-set file does not exist: {data_file}")
            sys.exit(1)
        log(f"Using explicit active-set file: {data_file}")
    else:
        data_file = pick_data_file(data_dir, CASE_NAME)
    if data_file is None:
        log(f"Error: no JSON active-set file found for {CASE_NAME} under {data_dir}")
        log("Run ActiveSetLearner first or place the active-set JSON file under result/active_set")
        log(f"Expected filename pattern: active_sets_{CASE_NAME}_*.json")
        sys.exit(1)

    # ── 执行模式分支 ─────────────────────────────────────
    try:
        if MODE == 'bcd':
            # BCD 通过 ActiveSetReader 加载（含 unit_commitment_matrix）
            log(f"Loading data through ActiveSetReader: {data_file.name}")
            ensure_bcd_modules_imported()
            all_samples_bcd = load_active_set_from_json(str(data_file))
            if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
                all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
            run_bcd(ppc, all_samples_bcd, T_DELTA, BCD_MAX_ITER_VALUE, bcd_model_dir,
                    case_name=CASE_NAME, timestamp=timestamp, n_workers=N_WORKERS_BCD, NN_EPOCHS=NN_EPOCHS, DUAL_DECAY_ROUND=DUAL_DECAY_ROUND, DUAL_SIGN_RELAX_INTERVAL=BCD_DUAL_SIGN_RELAX_INTERVAL_VALUE,
                    lp_backend=BCD_LP_BACKEND_VALUE,
                    gurobi_threads=BCD_GUROBI_THREADS_VALUE,
                    gurobi_lp_method=BCD_GUROBI_LP_METHOD_VALUE,
                    bcd_highs_threads=BCD_HIGHS_THREADS_VALUE,
                    logger=logger,
                    load_model_path=str(resolve_existing_path(BCD_MODEL_FILE, 'BCD model file')) if BCD_CONTINUE_TRAINING and BCD_MODEL_FILE is not None else None,
                    restore_rho_from_checkpoint=BCD_RESTORE_RHO_FROM_CHECKPOINT,
                    lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                    max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                    theta_training_stages=BCD_THETA_TRAINING_STAGES_VALUE,
                    theta_curriculum_rounds=BCD_THETA_CURRICULUM_ROUNDS_VALUE,
                    theta_initial_scale=BCD_THETA_INITIAL_SCALE_VALUE,
                    theta_final_scale=BCD_THETA_FINAL_SCALE_VALUE,
                    theta_curriculum_delay_rounds=BCD_THETA_CURRICULUM_DELAY_ROUNDS_VALUE,
                    theta_hot_start_strategy=THETA_WARM_START_STRATEGY,
                    zeta_hot_start_strategy=ZETA_WARM_START_STRATEGY,
                    theta_gaussian_std=THETA_WARM_START_GAUSSIAN_STD,
                    zeta_gaussian_std=ZETA_WARM_START_GAUSSIAN_STD,
                    use_unit_predictor=BCD_USE_UNIT_PREDICTOR_VALUE,
                    unit_predictor_epochs=UNIT_PREDICTOR_EPOCHS_VALUE,
                    unit_predictor_batch_strategy=UNIT_PREDICTOR_BATCH_STRATEGY_VALUE,
                    unit_predictor_batch_size=UNIT_PREDICTOR_BATCH_SIZE_VALUE,
                    unit_predictor_shuffle=UNIT_PREDICTOR_SHUFFLE_VALUE,
                    unit_predictor_lr=UNIT_PREDICTOR_LR_VALUE,
                    unit_predictor_hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS_VALUE,
                    unit_predictor_net_variant=UNIT_PREDICTOR_NET_VARIANT_VALUE,
                    unit_predictor_tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS_VALUE,
                    unit_predictor_tcn_depth=UNIT_PREDICTOR_TCN_DEPTH_VALUE,
                    unit_predictor_tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS_VALUE,
                    unit_predictor_tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH_VALUE,
                    unit_predictor_dropout=UNIT_PREDICTOR_DROPOUT_VALUE,
                    unit_predictor_enable_pos_weight=UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE,
                    unit_predictor_pos_weight_clip=UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE,
                    unit_predictor_loss_weight_bce=UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE,
                    unit_predictor_loss_weight_mse=UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE,
                    unit_predictor_loss_weight_l1=UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE,
                    unit_predictor_loss_weight_tv=UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE,
                    unit_predictor_loss_weight_transition=UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE,
                    unit_predictor_loss_weight_binarize=UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE,
                    unit_predictor_loss_weight_std_floor=UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE,
                    unit_predictor_std_floor_scale=UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE,
                    unit_predictor_loss_weight_tv_floor=UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE,
                    unit_predictor_tv_floor_scale=UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE,
                    unit_predictor_load_path=UNIT_PREDICTOR_LOAD_PATH_VALUE,
                    unit_predictor_warmup_rounds=BCD_UNIT_PREDICTOR_WARMUP_ROUNDS_VALUE,
                    unit_predictor_finetune_lr=UNIT_PREDICTOR_FINETUNE_LR_VALUE,
                    unit_predictor_weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY_VALUE,
                    theta_constraint_delay_rounds=BCD_THETA_CONSTRAINT_DELAY_ROUNDS_VALUE,
                    enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                    rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                    rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                    rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                    rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                    rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                    rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                    rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                    loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                    loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                    loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                    loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                    gamma_base=BCD_GAMMA_BASE_VALUE,
                    mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                    ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                    zeta_ita_cap_penalty_weight=BCD_ZETA_ITA_CAP_PENALTY_WEIGHT_VALUE,
                    zeta_ita_cap_initial_weight=BCD_ZETA_ITA_CAP_INITIAL_WEIGHT_VALUE,
                    zeta_ita_cap_final_weight=BCD_ZETA_ITA_CAP_FINAL_WEIGHT_VALUE,
                    zeta_ita_cap_initial=BCD_ZETA_ITA_CAP_INITIAL_VALUE,
                    zeta_ita_cap_final=BCD_ZETA_ITA_CAP_FINAL_VALUE,
                    zeta_ita_cap_start_round=BCD_ZETA_ITA_CAP_START_ROUND_VALUE,
                    zeta_ita_cap_end_round=BCD_ZETA_ITA_CAP_END_ROUND_VALUE,
                    iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                    iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                    nn_size=BCD_NN_SIZE_VALUE,
                    nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                    nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                    nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                    nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                    nn_learning_rate=BCD_NN_LR_VALUE,
                    nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                    direct_train_config=BCD_DIRECT_TRAIN_CONFIG_VALUE,
                    pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                    dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE)
            if RUN_FP:
                log("警告: bcd 模式不支持 RUN_FP（需要 trainers），请改用 both 模式")

        elif MODE == 'sparse':
            log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
            ensure_bcd_modules_imported()
            all_samples_bcd = load_active_set_from_json(str(data_file))
            if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
                all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]

            if N_WORKERS_BCD > 1:
                log("警告: sparse 模式当前仅支持串行 Agent_NN_BCD，将忽略 N_WORKERS_BCD > 1")

            run_sparse_bcd(
                ppc,
                all_samples_bcd,
                T_DELTA,
                BCD_MAX_ITER_VALUE,
                bcd_model_dir,
                case_name=CASE_NAME,
                timestamp=timestamp,
                NN_EPOCHS=NN_EPOCHS,
                DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
                DUAL_SIGN_RELAX_INTERVAL=BCD_DUAL_SIGN_RELAX_INTERVAL,
                top_k_variables=SPARSE_TOP_K_VARIABLES,
                max_groups=SPARSE_MAX_GROUPS,
                group_size=SPARSE_GROUP_SIZE,
                logger=logger,
                lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                theta_hot_start_strategy=THETA_WARM_START_STRATEGY,
                zeta_hot_start_strategy=ZETA_WARM_START_STRATEGY,
                theta_gaussian_std=THETA_WARM_START_GAUSSIAN_STD,
                zeta_gaussian_std=ZETA_WARM_START_GAUSSIAN_STD,
                rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                gamma_base=BCD_GAMMA_BASE_VALUE,
                mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                nn_size=BCD_NN_SIZE_VALUE,
                nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                nn_learning_rate=BCD_NN_LR_VALUE,
                nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
            )
            if RUN_FP:
                log("警告: sparse 模式暂不支持 RUN_FP（需要 trainers），请改用 both 模式或单独接入模板库")

        elif MODE == 'surrogate':
            # 加载并规范化样本（v3 格式）
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            if SURROGATE_MODEL_DIR is not None and not SURROGATE_CONTINUE_TRAINING:
                dual_predictor, trainers = load_surrogate(
                    ppc, all_samples, T_DELTA, UNIT_IDS,
                    SURROGATE_MODEL_DIR, logger=logger,
                    lp_backend=SUBPROBLEM_LP_BACKEND,
                    constraint_generation_strategy=CONSTRAINT_GENERATION_STRATEGY,
                    ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS_VALUE,
                    case_name=CASE_NAME,
                )
            else:
                dual_predictor, trainers = run_surrogate(
                    ppc, all_samples, T_DELTA, UNIT_IDS,
                    DUAL_EPOCHS, DUAL_BATCH_SIZE, SUBPROBLEM_MAX_ITER_VALUE, NN_EPOCHS, save_dir,
                    n_workers=N_WORKERS_SAMPLE, logger=logger,
                    load_dir=str(resolve_existing_path(SURROGATE_MODEL_DIR, 'surrogate model dir')) if SURROGATE_CONTINUE_TRAINING and SURROGATE_MODEL_DIR is not None else None,
                    lp_backend=SUBPROBLEM_LP_BACKEND,
                    dual_batch_strategy=DUAL_BATCH_STRATEGY_VALUE,
                    dual_shuffle=DUAL_SHUFFLE_VALUE,
                    dual_learning_rate=DUAL_LR_VALUE,
                    constraint_generation_strategy=CONSTRAINT_GENERATION_STRATEGY,
                    rho_primal_init=SUBPROBLEM_RHO_PRIMAL_INIT_VALUE,
                    rho_dual_init=SUBPROBLEM_RHO_DUAL_INIT_VALUE,
                    rho_dual_pg_init=SUBPROBLEM_RHO_DUAL_PG_INIT_VALUE,
                    rho_dual_x_init=SUBPROBLEM_RHO_DUAL_X_INIT_VALUE,
                    rho_dual_coc_init=SUBPROBLEM_RHO_DUAL_COC_INIT_VALUE,
                    rho_binary_init=SUBPROBLEM_RHO_BINARY_INIT_VALUE,
                    rho_binary_max=SUBPROBLEM_RHO_BINARY_MAX_VALUE,
                    rho_opt_init=SUBPROBLEM_RHO_OPT_INIT_VALUE,
                    loss_ratio_primal=SUBPROBLEM_LOSS_RATIO_PRIMAL_VALUE,
                    loss_ratio_dual_pg=SUBPROBLEM_LOSS_RATIO_DUAL_PG_VALUE,
                    loss_ratio_dual_x=SUBPROBLEM_LOSS_RATIO_DUAL_X_VALUE,
                    nn_dual_term_interval=SUBPROBLEM_NN_DUAL_TERM_INTERVAL_VALUE,
                    loss_ratio_opt=SUBPROBLEM_LOSS_RATIO_OPT_VALUE,
                    loss_ratio_reg=SUBPROBLEM_LOSS_RATIO_REG_VALUE,
                    subproblem_gamma_base=SUBPROBLEM_GAMMA_BASE_VALUE,
                    mu_lower_bound_init=SUBPROBLEM_MU_DUAL_FLOOR_INIT_VALUE,
                    mu_individual_lower_bound_round=SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND_VALUE,
                    mu_group_lower_bound_round=SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND_VALUE,
                    mu_signed_round_interval=SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL_VALUE,
                    mu_sign_hysteresis_rounds=SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS_VALUE,
                    mu_sign_flip_min_share=SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE_VALUE,
                    x_bound_dual_zero_rounds=SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS_VALUE,
                    subproblem_nn_size=SUBPROBLEM_NN_SIZE_VALUE,
                    subproblem_nn_hidden_dims=SUBPROBLEM_NN_HIDDEN_DIMS_VALUE,
                    subproblem_c_pg_nn_size=SUBPROBLEM_C_PG_NN_SIZE_VALUE,
                    subproblem_c_pg_nn_hidden_dims=SUBPROBLEM_C_PG_NN_HIDDEN_DIMS_VALUE,
                    subproblem_nn_batch_strategy=SUBPROBLEM_NN_BATCH_STRATEGY_VALUE,
                    subproblem_nn_batch_size=SUBPROBLEM_NN_BATCH_SIZE_VALUE,
                    subproblem_nn_shuffle=SUBPROBLEM_NN_SHUFFLE_VALUE,
                    subproblem_nn_learning_rate=SUBPROBLEM_NN_LR_VALUE,
                    subproblem_nn_main_eta_min_ratio=SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO_VALUE,
                    subproblem_nn_main_lr_late_scale=SUBPROBLEM_NN_MAIN_LR_LATE_SCALE_VALUE,
                    subproblem_nn_main_adam_weight_decay=SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY_VALUE,
                    subproblem_nn_main_grad_clip=SUBPROBLEM_NN_MAIN_GRAD_CLIP_VALUE,
                    subproblem_nn_main_kkt_lr_scale=SUBPROBLEM_NN_MAIN_KKT_LR_SCALE_VALUE,
                    ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS_VALUE,
                    subproblem_cost_learning_rate=SUBPROBLEM_X_COST_NN_LR_VALUE,
                    subproblem_nn_smooth_abs_eps=SUBPROBLEM_NN_SMOOTH_ABS_EPS_VALUE,
                    pg_cost_nn_epochs=SUBPROBLEM_PG_COST_NN_EPOCHS_VALUE,
                    pg_cost_start_round=SUBPROBLEM_PG_COST_START_ROUND_VALUE,
                    pg_cost_scale_multiplier=SUBPROBLEM_PG_COST_SCALE_MULTIPLIER_VALUE,
                    pg_cost_lr=SUBPROBLEM_PG_COST_LR_VALUE,
                    pg_cost_surr_lr=SUBPROBLEM_PG_COST_SURR_LR_VALUE,
                    pg_cost_reg_deadband=SUBPROBLEM_PG_COST_REG_DEADBAND_VALUE,
                    pg_cost_softbound_weight=SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT_VALUE,
                    pg_cost_smooth_abs_eps=SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS_VALUE,
                    pg_cost_batch_strategy=SUBPROBLEM_PG_COST_BATCH_STRATEGY_VALUE,
                    pg_cost_batch_size=SUBPROBLEM_PG_COST_BATCH_SIZE_VALUE,
                    pg_cost_shuffle=SUBPROBLEM_PG_COST_SHUFFLE_VALUE,
                    pg_cost_use_sample_weights=SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS_VALUE,
                    pg_cost_sample_weight_power=SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER_VALUE,
                    pg_cost_sample_weight_clip=SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP_VALUE,
                    pg_cost_single_sample_reg_scale=SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE_VALUE,
                    pg_cost_c_pg_adam_weight_decay=SUBPROBLEM_PG_COST_C_PG_ADAM_WD_VALUE,
                    main_direct_train_config=SUBPROBLEM_MAIN_DIRECT_TRAIN_CONFIG_VALUE,
                    c_pg_direct_train_config=SUBPROBLEM_C_PG_DIRECT_TRAIN_CONFIG_VALUE,
                    pg_block_prox_weight=SUBPROBLEM_PG_BLOCK_PROX_WEIGHT_VALUE,
                    dual_block_prox_weight=SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                    single_mu_cap_penalty_weight=SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT_VALUE,
                    single_mu_cap_initial_weight=SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT_VALUE,
                    single_mu_cap_final_weight=SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT_VALUE,
                    single_mu_cap_initial=SUBPROBLEM_SINGLE_MU_CAP_INITIAL_VALUE,
                    single_mu_cap_final=SUBPROBLEM_SINGLE_MU_CAP_FINAL_VALUE,
                    single_mu_cap_start_round=SUBPROBLEM_SINGLE_MU_CAP_START_ROUND_VALUE,
                    single_mu_cap_end_round=SUBPROBLEM_SINGLE_MU_CAP_END_ROUND_VALUE,
                    override_loaded_single_mu_cap=SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP_VALUE,
                    iter_delta_reg_weight=SUBPROBLEM_ITER_DELTA_REG_WEIGHT,
                    iter_delta_reg_deadband=SUBPROBLEM_ITER_DELTA_REG_DEADBAND,
                    dual_predictor_only=SURROGATE_DUAL_PREDICTOR_ONLY,
                    dual_net_variant=DUAL_PREDICTOR_NET_VARIANT_VALUE,
                    dual_normalize_targets=DUAL_PREDICTOR_NORMALIZE_TARGETS_VALUE,
                    dual_cosine_loss_weight=DUAL_PREDICTOR_COSINE_LOSS_WEIGHT_VALUE,
                    dual_smooth_l1_beta=DUAL_PREDICTOR_SMOOTH_L1_BETA_VALUE,
                    use_unit_predictor=USE_UNIT_PREDICTOR_VALUE,
                    unit_predictor_epochs=UNIT_PREDICTOR_EPOCHS_VALUE,
                    unit_predictor_batch_strategy=UNIT_PREDICTOR_BATCH_STRATEGY_VALUE,
                    unit_predictor_batch_size=UNIT_PREDICTOR_BATCH_SIZE_VALUE,
                    unit_predictor_shuffle=UNIT_PREDICTOR_SHUFFLE_VALUE,
                    unit_predictor_lr=UNIT_PREDICTOR_LR_VALUE,
                    unit_predictor_hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS_VALUE,
                    unit_predictor_net_variant=UNIT_PREDICTOR_NET_VARIANT_VALUE,
                    unit_predictor_tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS_VALUE,
                    unit_predictor_tcn_depth=UNIT_PREDICTOR_TCN_DEPTH_VALUE,
                    unit_predictor_tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS_VALUE,
                    unit_predictor_tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH_VALUE,
                    unit_predictor_dropout=UNIT_PREDICTOR_DROPOUT_VALUE,
                    unit_predictor_enable_pos_weight=UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE,
                    unit_predictor_pos_weight_clip=UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE,
                    unit_predictor_loss_weight_bce=UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE,
                    unit_predictor_loss_weight_mse=UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE,
                    unit_predictor_loss_weight_l1=UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE,
                    unit_predictor_loss_weight_tv=UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE,
                    unit_predictor_loss_weight_transition=UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE,
                    unit_predictor_loss_weight_binarize=UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE,
                    unit_predictor_loss_weight_std_floor=UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE,
                    unit_predictor_std_floor_scale=UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE,
                    unit_predictor_loss_weight_tv_floor=UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE,
                    unit_predictor_tv_floor_scale=UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE,
                    unit_predictor_load_path=UNIT_PREDICTOR_LOAD_PATH_VALUE,
                    unit_predictor_unit1_extra_tv_epochs=UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS_VALUE,
                    unit_predictor_unit1_extra_tv_weight=UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT_VALUE,
                    unit_predictor_auto_extra_tv=UNIT_PREDICTOR_AUTO_EXTRA_TV_VALUE,
                    unit_predictor_auto_extra_tv_epochs=UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS_VALUE,
                    unit_predictor_auto_extra_tv_tv_threshold=UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD_VALUE,
                    unit_predictor_auto_extra_tv_weight=UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT_VALUE,
                    predictor_warmup_rounds=SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS_VALUE,
                    sign4_curriculum_rounds=SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS_VALUE,
                    sign4_initial_scale=SUBPROBLEM_SIGN4_INITIAL_SCALE_VALUE,
                    sign4_final_scale=SUBPROBLEM_SIGN4_FINAL_SCALE_VALUE,
                    sign4_delay_rounds=SUBPROBLEM_SIGN4_DELAY_ROUNDS_VALUE,
                    enable_surrogate_delta_reference_lift=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT_VALUE,
                    surrogate_delta_reference_eps=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS_VALUE,
                    surrogate_delta_reference_scope=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE_VALUE,
                    surrogate_delta_reference_min_abs_factor=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR_VALUE,
                    skip_existing_units=SURROGATE_SKIP_EXISTING_UNITS,
                    unit_predictor_finetune_lr=UNIT_PREDICTOR_FINETUNE_LR_VALUE,
                    unit_predictor_weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY_VALUE,
                    case_name=CASE_NAME,
                )
            if trainers:
                print_surrogate_results(trainers, all_samples)
            elif SURROGATE_DUAL_PREDICTOR_ONLY:
                log("SURROGATE_DUAL_PREDICTOR_ONLY：无子问题代理，跳过 print_surrogate_results")

            if RUN_FP:
                if not trainers:
                    log("警告: SURROGATE_DUAL_PREDICTOR_ONLY 且无子问题代理，跳过 RUN_FP")
                else:
                    run_feasibility_pump_test(
                        ppc, all_samples, dual_predictor, trainers,
                        T_DELTA, FP_TEST_SAMPLES,
                    )

        elif MODE == 'both':
            # Step 1: BCD 训练（或从已有模型加载跳过）
            log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
            ensure_bcd_modules_imported()
            all_samples_bcd = load_active_set_from_json(str(data_file))
            if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
                all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]

            sparse_template_library = None
            if ENABLE_SPARSE_SUPPORTS:
                log("both 模式: 启用 sparse 支持集发现")
                if N_WORKERS_BCD > 1:
                    log("警告: both + sparse 当前仅支持串行 Agent_NN_BCD，将忽略 N_WORKERS_BCD > 1")
                _bootstrap_agent = Agent_NN_BCD(
                    ppc,
                    all_samples_bcd,
                    T_DELTA,
                    lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                    max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                    enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                    rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                    rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                    rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                    rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                    rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                    rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                    rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                    loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                    loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                    loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                    loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                    gamma_base=BCD_GAMMA_BASE_VALUE,
                    mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                    ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                    iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                    iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                    pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                    dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                    nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                )
                _, sparse_template_library = build_sparse_template_library_from_bcd_agent(
                    _bootstrap_agent,
                    top_k_variables=SPARSE_TOP_K_VARIABLES,
                    max_groups=SPARSE_MAX_GROUPS,
                    group_size=SPARSE_GROUP_SIZE,
                    sparse_dir=Path(__file__).parent / 'result' / 'sparse_templates',
                    case_name=CASE_NAME,
                    timestamp=timestamp,
                )

            if BCD_MODEL_FILE is not None and not BCD_CONTINUE_TRAINING:
                # 从已有模型加载，跳过 BCD 训练
                bcd_path = Path(BCD_MODEL_FILE)
                if not bcd_path.is_absolute():
                    bcd_path = Path(__file__).parent / bcd_path
                if not bcd_path.exists():
                    log(f"错误: 指定的 BCD 模型文件不存在: {bcd_path}")
                    sys.exit(1)
                log(f"从已有模型加载 BCD，跳过 BCD 训练: {bcd_path}")
                if ENABLE_SPARSE_SUPPORTS:
                    agent = Agent_NN_BCD(
                        ppc,
                        all_samples_bcd,
                        T_DELTA,
                        external_sparse_templates=sparse_template_library,
                        lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                        max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                        enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                        rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                        rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                        rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                        rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                        rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                        rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                        rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                        loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                        loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                        loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                        loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                        gamma_base=BCD_GAMMA_BASE_VALUE,
                        mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                        ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                        iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                        iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                        pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                        dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                        nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                        nn_learning_rate=BCD_NN_LR_VALUE,
                        nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                        nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                        nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                        nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                    )
                elif N_WORKERS_BCD <= 1:
                    agent = Agent_NN_BCD(
                        ppc,
                        all_samples_bcd,
                        T_DELTA,
                        lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                        max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                        enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                        rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                        rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                        rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                        rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                        rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                        rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                        rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                        loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                        loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                        loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                        loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                        gamma_base=BCD_GAMMA_BASE_VALUE,
                        mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                        ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                        iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                        iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                        pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                        dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                        nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                        nn_learning_rate=BCD_NN_LR_VALUE,
                        nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                        nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                        nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                        nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                    )
                else:
                    agent = ParallelAgent_NN_BCD(
                        ppc, all_samples_bcd, T_DELTA,
                        lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                        max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                        enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                        rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                        rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                        rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                        rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                        rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                        rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                        rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                        loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                        loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                        loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                        loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                        gamma_base=BCD_GAMMA_BASE_VALUE,
                        mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                        ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                        iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                        iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                        pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                        dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                        nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                        nn_learning_rate=BCD_NN_LR_VALUE,
                        nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                        nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                        nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                        nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                        n_workers=N_WORKERS_BCD,
                    )
                if BCD_USE_UNIT_PREDICTOR_VALUE:
                    resolved_bcd_unit_predictor = _resolve_unit_predictor_checkpoint_path(
                        UNIT_PREDICTOR_LOAD_PATH_VALUE
                    )
                    if resolved_bcd_unit_predictor is None or not resolved_bcd_unit_predictor.is_file():
                        sibling_predictor = bcd_path.parent / 'unit_predictor.pth'
                        resolved_bcd_unit_predictor = (
                            sibling_predictor.resolve()
                            if sibling_predictor.is_file()
                            else None
                        )
                    if resolved_bcd_unit_predictor is not None:
                        log(f"[BCD][unit_predictor] 加载 checkpoint 配套 predictor: {resolved_bcd_unit_predictor}")
                        bcd_unit_predictor_obj = train_unit_predictor_from_data(
                            ppc,
                            all_samples_bcd,
                            T_DELTA,
                            num_epochs=0,
                            hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS_VALUE,
                            batch_size=UNIT_PREDICTOR_BATCH_SIZE_VALUE,
                            batch_strategy=UNIT_PREDICTOR_BATCH_STRATEGY_VALUE,
                            shuffle=UNIT_PREDICTOR_SHUFFLE_VALUE,
                            learning_rate=UNIT_PREDICTOR_LR_VALUE,
                            weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY_VALUE,
                            net_variant=UNIT_PREDICTOR_NET_VARIANT_VALUE,
                            tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS_VALUE,
                            tcn_depth=UNIT_PREDICTOR_TCN_DEPTH_VALUE,
                            tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS_VALUE,
                            tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH_VALUE,
                            dropout=UNIT_PREDICTOR_DROPOUT_VALUE,
                            enable_pos_weight=UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE,
                            pos_weight_clip=UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE,
                            loss_weight_bce=UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE,
                            loss_weight_mse=UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE,
                            loss_weight_l1=UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE,
                            loss_weight_tv=UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE,
                            loss_weight_transition=UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE,
                            loss_weight_binarize=UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE,
                            loss_weight_std_floor=UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE,
                            std_floor_scale=UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE,
                            loss_weight_tv_floor=UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE,
                            tv_floor_scale=UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE,
                            save_path=None,
                            load_path=str(resolved_bcd_unit_predictor),
                        )
                        agent.unit_predictor = bcd_unit_predictor_obj
                        agent.use_unit_predictor = True
                        agent.unit_predictor_warmup_rounds = BCD_UNIT_PREDICTOR_WARMUP_ROUNDS_VALUE
                        agent.unit_predictor_finetune_lr = UNIT_PREDICTOR_FINETUNE_LR_VALUE
                        agent.unit_predictor_weight_decay = UNIT_PREDICTOR_WEIGHT_DECAY_VALUE
                        agent.theta_constraint_delay_rounds = BCD_THETA_CONSTRAINT_DELAY_ROUNDS_VALUE
                    else:
                        log("[BCD][unit_predictor] 警告: 已启用 BCD_USE_UNIT_PREDICTOR，但未找到可加载的 predictor checkpoint")
                agent.load_model_parameters(
                    str(bcd_path),
                    restore_rho_state=BCD_RESTORE_RHO_FROM_CHECKPOINT,
                )
                log("BCD 模型加载成功，跳过训练")
            else:
                agent = run_bcd(
                    ppc,
                    all_samples_bcd,
                    T_DELTA,
                    BCD_MAX_ITER_VALUE,
                    bcd_model_dir,
                    case_name=CASE_NAME,
                    timestamp=timestamp,
                    n_workers=N_WORKERS_BCD if not ENABLE_SPARSE_SUPPORTS else 1,
                    NN_EPOCHS=NN_EPOCHS,
                    DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
                    DUAL_SIGN_RELAX_INTERVAL=BCD_DUAL_SIGN_RELAX_INTERVAL_VALUE,
                    lp_backend=BCD_LP_BACKEND_VALUE,
                    gurobi_threads=BCD_GUROBI_THREADS_VALUE,
                    logger=logger,
                    load_model_path=str(resolve_existing_path(BCD_MODEL_FILE, 'BCD model file')) if BCD_CONTINUE_TRAINING and BCD_MODEL_FILE is not None else None,
                    restore_rho_from_checkpoint=BCD_RESTORE_RHO_FROM_CHECKPOINT,
                    external_sparse_templates=sparse_template_library,
                    lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY_VALUE,
                    max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT_VALUE,
                    theta_training_stages=BCD_THETA_TRAINING_STAGES_VALUE,
                    theta_curriculum_rounds=BCD_THETA_CURRICULUM_ROUNDS_VALUE,
                    theta_initial_scale=BCD_THETA_INITIAL_SCALE_VALUE,
                    theta_final_scale=BCD_THETA_FINAL_SCALE_VALUE,
                    theta_curriculum_delay_rounds=BCD_THETA_CURRICULUM_DELAY_ROUNDS_VALUE,
                    theta_hot_start_strategy=THETA_WARM_START_STRATEGY,
                    zeta_hot_start_strategy=ZETA_WARM_START_STRATEGY,
                    theta_gaussian_std=THETA_WARM_START_GAUSSIAN_STD,
                    zeta_gaussian_std=ZETA_WARM_START_GAUSSIAN_STD,
                    use_unit_predictor=BCD_USE_UNIT_PREDICTOR_VALUE,
                    unit_predictor_epochs=UNIT_PREDICTOR_EPOCHS_VALUE,
                    unit_predictor_batch_strategy=UNIT_PREDICTOR_BATCH_STRATEGY_VALUE,
                    unit_predictor_batch_size=UNIT_PREDICTOR_BATCH_SIZE_VALUE,
                    unit_predictor_shuffle=UNIT_PREDICTOR_SHUFFLE_VALUE,
                    unit_predictor_lr=UNIT_PREDICTOR_LR_VALUE,
                    unit_predictor_hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS_VALUE,
                    unit_predictor_net_variant=UNIT_PREDICTOR_NET_VARIANT_VALUE,
                    unit_predictor_tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS_VALUE,
                    unit_predictor_tcn_depth=UNIT_PREDICTOR_TCN_DEPTH_VALUE,
                    unit_predictor_tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS_VALUE,
                    unit_predictor_tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH_VALUE,
                    unit_predictor_dropout=UNIT_PREDICTOR_DROPOUT_VALUE,
                    unit_predictor_enable_pos_weight=UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE,
                    unit_predictor_pos_weight_clip=UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE,
                    unit_predictor_loss_weight_bce=UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE,
                    unit_predictor_loss_weight_mse=UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE,
                    unit_predictor_loss_weight_l1=UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE,
                    unit_predictor_loss_weight_tv=UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE,
                    unit_predictor_loss_weight_transition=UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE,
                    unit_predictor_loss_weight_binarize=UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE,
                    unit_predictor_loss_weight_std_floor=UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE,
                    unit_predictor_std_floor_scale=UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE,
                    unit_predictor_loss_weight_tv_floor=UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE,
                    unit_predictor_tv_floor_scale=UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE,
                    unit_predictor_load_path=UNIT_PREDICTOR_LOAD_PATH_VALUE,
                    unit_predictor_warmup_rounds=BCD_UNIT_PREDICTOR_WARMUP_ROUNDS_VALUE,
                    unit_predictor_finetune_lr=UNIT_PREDICTOR_FINETUNE_LR_VALUE,
                    unit_predictor_weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY_VALUE,
                    theta_constraint_delay_rounds=BCD_THETA_CONSTRAINT_DELAY_ROUNDS_VALUE,
                    enable_dropout_during_nn_training=BCD_ENABLE_DROPOUT_DURING_NN_TRAINING_VALUE,
                    rho_primal_init=BCD_RHO_PRIMAL_INIT_VALUE,
                    rho_dual_init=BCD_RHO_DUAL_INIT_VALUE,
                    rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT_VALUE,
                    rho_dual_x_init=BCD_RHO_DUAL_X_INIT_VALUE,
                    rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT_VALUE,
                    rho_binary_init=BCD_RHO_BINARY_INIT_VALUE,
                    rho_opt_init=BCD_RHO_OPT_INIT_VALUE,
                    loss_ratio_primal=BCD_LOSS_RATIO_PRIMAL_VALUE,
                    loss_ratio_dual_x=BCD_LOSS_RATIO_DUAL_X_VALUE,
                    loss_ratio_opt=BCD_LOSS_RATIO_OPT_VALUE,
                    loss_ratio_reg=BCD_LOSS_RATIO_REG_VALUE,
                    gamma_base=BCD_GAMMA_BASE_VALUE,
                    mu_dual_floor_init=BCD_MU_DUAL_FLOOR_INIT_VALUE,
                    ita_dual_floor_init=BCD_ITA_DUAL_FLOOR_INIT_VALUE,
                    zeta_ita_cap_penalty_weight=BCD_ZETA_ITA_CAP_PENALTY_WEIGHT_VALUE,
                    zeta_ita_cap_initial_weight=BCD_ZETA_ITA_CAP_INITIAL_WEIGHT_VALUE,
                    zeta_ita_cap_final_weight=BCD_ZETA_ITA_CAP_FINAL_WEIGHT_VALUE,
                    zeta_ita_cap_initial=BCD_ZETA_ITA_CAP_INITIAL_VALUE,
                    zeta_ita_cap_final=BCD_ZETA_ITA_CAP_FINAL_VALUE,
                    zeta_ita_cap_start_round=BCD_ZETA_ITA_CAP_START_ROUND_VALUE,
                    zeta_ita_cap_end_round=BCD_ZETA_ITA_CAP_END_ROUND_VALUE,
                    iter_delta_reg_weight=BCD_ITER_DELTA_REG_WEIGHT,
                    iter_delta_reg_deadband=BCD_ITER_DELTA_REG_DEADBAND,
                    pg_block_prox_weight=BCD_PG_BLOCK_PROX_WEIGHT_VALUE,
                    dual_block_prox_weight=BCD_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                    nn_size=BCD_NN_SIZE_VALUE,
                    nn_hidden_dims=BCD_NN_HIDDEN_DIMS_VALUE,
                    nn_batch_strategy=BCD_NN_BATCH_STRATEGY_VALUE,
                    nn_batch_size=BCD_NN_BATCH_SIZE_VALUE,
                    nn_shuffle=BCD_NN_SHUFFLE_VALUE,
                    nn_learning_rate=BCD_NN_LR_VALUE,
                    nn_smooth_abs_eps=BCD_NN_SMOOTH_ABS_EPS_VALUE,
                    direct_train_config=BCD_DIRECT_TRAIN_CONFIG_VALUE,
                )

            # Step 2: 加载 v3 格式样本（subproblem 独立训练，不注入 BCD 对偶变量）
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            # Step 3: surrogate 训练（或从已有模型加载跳过）
            # BCD 的对偶变量仅在后续联合训练中使用，不注入 subproblem 训练
            if SURROGATE_MODEL_DIR is not None and not SURROGATE_CONTINUE_TRAINING:
                dual_predictor, trainers = load_surrogate(
                    ppc, all_samples, T_DELTA, UNIT_IDS,
                    SURROGATE_MODEL_DIR, logger=logger,
                    lp_backend=SUBPROBLEM_LP_BACKEND,
                    constraint_generation_strategy=CONSTRAINT_GENERATION_STRATEGY,
                    ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS_VALUE,
                    case_name=CASE_NAME,
                )
            else:
                dual_predictor, trainers = run_surrogate(
                    ppc, all_samples, T_DELTA, UNIT_IDS,
                    DUAL_EPOCHS, DUAL_BATCH_SIZE, SUBPROBLEM_MAX_ITER_VALUE, NN_EPOCHS, save_dir,
                    n_workers=N_WORKERS_SAMPLE, logger=logger,
                    load_dir=str(resolve_existing_path(SURROGATE_MODEL_DIR, 'surrogate model dir')) if SURROGATE_CONTINUE_TRAINING and SURROGATE_MODEL_DIR is not None else None,
                    lp_backend=SUBPROBLEM_LP_BACKEND,
                    dual_batch_strategy=DUAL_BATCH_STRATEGY_VALUE,
                    dual_shuffle=DUAL_SHUFFLE_VALUE,
                    dual_learning_rate=DUAL_LR_VALUE,
                    constraint_generation_strategy=CONSTRAINT_GENERATION_STRATEGY,
                    rho_primal_init=SUBPROBLEM_RHO_PRIMAL_INIT_VALUE,
                    rho_dual_init=SUBPROBLEM_RHO_DUAL_INIT_VALUE,
                    rho_dual_pg_init=SUBPROBLEM_RHO_DUAL_PG_INIT_VALUE,
                    rho_dual_x_init=SUBPROBLEM_RHO_DUAL_X_INIT_VALUE,
                    rho_dual_coc_init=SUBPROBLEM_RHO_DUAL_COC_INIT_VALUE,
                    rho_binary_init=SUBPROBLEM_RHO_BINARY_INIT_VALUE,
                    rho_binary_max=SUBPROBLEM_RHO_BINARY_MAX_VALUE,
                    rho_opt_init=SUBPROBLEM_RHO_OPT_INIT_VALUE,
                    loss_ratio_primal=SUBPROBLEM_LOSS_RATIO_PRIMAL_VALUE,
                    loss_ratio_dual_pg=SUBPROBLEM_LOSS_RATIO_DUAL_PG_VALUE,
                    loss_ratio_dual_x=SUBPROBLEM_LOSS_RATIO_DUAL_X_VALUE,
                    nn_dual_term_interval=SUBPROBLEM_NN_DUAL_TERM_INTERVAL_VALUE,
                    loss_ratio_opt=SUBPROBLEM_LOSS_RATIO_OPT_VALUE,
                    loss_ratio_reg=SUBPROBLEM_LOSS_RATIO_REG_VALUE,
                    subproblem_gamma_base=SUBPROBLEM_GAMMA_BASE_VALUE,
                    mu_lower_bound_init=SUBPROBLEM_MU_DUAL_FLOOR_INIT_VALUE,
                    mu_individual_lower_bound_round=SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND_VALUE,
                    mu_group_lower_bound_round=SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND_VALUE,
                    mu_signed_round_interval=SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL_VALUE,
                    mu_sign_hysteresis_rounds=SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS_VALUE,
                    mu_sign_flip_min_share=SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE_VALUE,
                    x_bound_dual_zero_rounds=SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS_VALUE,
                    subproblem_nn_size=SUBPROBLEM_NN_SIZE_VALUE,
                    subproblem_nn_hidden_dims=SUBPROBLEM_NN_HIDDEN_DIMS_VALUE,
                    subproblem_c_pg_nn_size=SUBPROBLEM_C_PG_NN_SIZE_VALUE,
                    subproblem_c_pg_nn_hidden_dims=SUBPROBLEM_C_PG_NN_HIDDEN_DIMS_VALUE,
                    subproblem_nn_batch_strategy=SUBPROBLEM_NN_BATCH_STRATEGY_VALUE,
                    subproblem_nn_batch_size=SUBPROBLEM_NN_BATCH_SIZE_VALUE,
                    subproblem_nn_shuffle=SUBPROBLEM_NN_SHUFFLE_VALUE,
                    subproblem_nn_learning_rate=SUBPROBLEM_NN_LR_VALUE,
                    subproblem_nn_main_eta_min_ratio=SUBPROBLEM_NN_MAIN_ETA_MIN_RATIO_VALUE,
                    subproblem_nn_main_lr_late_scale=SUBPROBLEM_NN_MAIN_LR_LATE_SCALE_VALUE,
                    subproblem_nn_main_adam_weight_decay=SUBPROBLEM_NN_MAIN_ADAM_WEIGHT_DECAY_VALUE,
                    subproblem_nn_main_grad_clip=SUBPROBLEM_NN_MAIN_GRAD_CLIP_VALUE,
                    subproblem_nn_main_kkt_lr_scale=SUBPROBLEM_NN_MAIN_KKT_LR_SCALE_VALUE,
                    ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS_VALUE,
                    subproblem_cost_learning_rate=SUBPROBLEM_X_COST_NN_LR_VALUE,
                    subproblem_nn_smooth_abs_eps=SUBPROBLEM_NN_SMOOTH_ABS_EPS_VALUE,
                    pg_cost_nn_epochs=SUBPROBLEM_PG_COST_NN_EPOCHS_VALUE,
                    pg_cost_start_round=SUBPROBLEM_PG_COST_START_ROUND_VALUE,
                    pg_cost_scale_multiplier=SUBPROBLEM_PG_COST_SCALE_MULTIPLIER_VALUE,
                    pg_cost_lr=SUBPROBLEM_PG_COST_LR_VALUE,
                    pg_cost_surr_lr=SUBPROBLEM_PG_COST_SURR_LR_VALUE,
                    pg_cost_reg_deadband=SUBPROBLEM_PG_COST_REG_DEADBAND_VALUE,
                    pg_cost_softbound_weight=SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT_VALUE,
                    pg_cost_smooth_abs_eps=SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS_VALUE,
                    pg_cost_batch_strategy=SUBPROBLEM_PG_COST_BATCH_STRATEGY_VALUE,
                    pg_cost_batch_size=SUBPROBLEM_PG_COST_BATCH_SIZE_VALUE,
                    pg_cost_shuffle=SUBPROBLEM_PG_COST_SHUFFLE_VALUE,
                    pg_cost_use_sample_weights=SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS_VALUE,
                    pg_cost_sample_weight_power=SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER_VALUE,
                    pg_cost_sample_weight_clip=SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP_VALUE,
                    pg_cost_single_sample_reg_scale=SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE_VALUE,
                    pg_cost_c_pg_adam_weight_decay=SUBPROBLEM_PG_COST_C_PG_ADAM_WD_VALUE,
                    main_direct_train_config=SUBPROBLEM_MAIN_DIRECT_TRAIN_CONFIG_VALUE,
                    c_pg_direct_train_config=SUBPROBLEM_C_PG_DIRECT_TRAIN_CONFIG_VALUE,
                    pg_block_prox_weight=SUBPROBLEM_PG_BLOCK_PROX_WEIGHT_VALUE,
                    dual_block_prox_weight=SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT_VALUE,
                    single_mu_cap_penalty_weight=SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT_VALUE,
                    single_mu_cap_initial_weight=SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT_VALUE,
                    single_mu_cap_final_weight=SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT_VALUE,
                    single_mu_cap_initial=SUBPROBLEM_SINGLE_MU_CAP_INITIAL_VALUE,
                    single_mu_cap_final=SUBPROBLEM_SINGLE_MU_CAP_FINAL_VALUE,
                    single_mu_cap_start_round=SUBPROBLEM_SINGLE_MU_CAP_START_ROUND_VALUE,
                    single_mu_cap_end_round=SUBPROBLEM_SINGLE_MU_CAP_END_ROUND_VALUE,
                    override_loaded_single_mu_cap=SUBPROBLEM_OVERRIDE_LOADED_SINGLE_MU_CAP_VALUE,
                    iter_delta_reg_weight=SUBPROBLEM_ITER_DELTA_REG_WEIGHT,
                    iter_delta_reg_deadband=SUBPROBLEM_ITER_DELTA_REG_DEADBAND,
                    dual_predictor_only=False,
                    dual_net_variant=DUAL_PREDICTOR_NET_VARIANT_VALUE,
                    dual_normalize_targets=DUAL_PREDICTOR_NORMALIZE_TARGETS_VALUE,
                    dual_cosine_loss_weight=DUAL_PREDICTOR_COSINE_LOSS_WEIGHT_VALUE,
                    dual_smooth_l1_beta=DUAL_PREDICTOR_SMOOTH_L1_BETA_VALUE,
                    use_unit_predictor=USE_UNIT_PREDICTOR_VALUE,
                    unit_predictor_epochs=UNIT_PREDICTOR_EPOCHS_VALUE,
                    unit_predictor_batch_strategy=UNIT_PREDICTOR_BATCH_STRATEGY_VALUE,
                    unit_predictor_batch_size=UNIT_PREDICTOR_BATCH_SIZE_VALUE,
                    unit_predictor_shuffle=UNIT_PREDICTOR_SHUFFLE_VALUE,
                    unit_predictor_lr=UNIT_PREDICTOR_LR_VALUE,
                    unit_predictor_hidden_dims=UNIT_PREDICTOR_HIDDEN_DIMS_VALUE,
                    unit_predictor_net_variant=UNIT_PREDICTOR_NET_VARIANT_VALUE,
                    unit_predictor_tcn_channels=UNIT_PREDICTOR_TCN_CHANNELS_VALUE,
                    unit_predictor_tcn_depth=UNIT_PREDICTOR_TCN_DEPTH_VALUE,
                    unit_predictor_tconv_channels=UNIT_PREDICTOR_TCONV_CHANNELS_VALUE,
                    unit_predictor_tconv_depth=UNIT_PREDICTOR_TCONV_DEPTH_VALUE,
                    unit_predictor_dropout=UNIT_PREDICTOR_DROPOUT_VALUE,
                    unit_predictor_enable_pos_weight=UNIT_PREDICTOR_ENABLE_POS_WEIGHT_VALUE,
                    unit_predictor_pos_weight_clip=UNIT_PREDICTOR_POS_WEIGHT_CLIP_VALUE,
                    unit_predictor_loss_weight_bce=UNIT_PREDICTOR_LOSS_WEIGHT_BCE_VALUE,
                    unit_predictor_loss_weight_mse=UNIT_PREDICTOR_LOSS_WEIGHT_MSE_VALUE,
                    unit_predictor_loss_weight_l1=UNIT_PREDICTOR_LOSS_WEIGHT_L1_VALUE,
                    unit_predictor_loss_weight_tv=UNIT_PREDICTOR_LOSS_WEIGHT_TV_VALUE,
                    unit_predictor_loss_weight_transition=UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION_VALUE,
                    unit_predictor_loss_weight_binarize=UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE_VALUE,
                    unit_predictor_loss_weight_std_floor=UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR_VALUE,
                    unit_predictor_std_floor_scale=UNIT_PREDICTOR_STD_FLOOR_SCALE_VALUE,
                    unit_predictor_loss_weight_tv_floor=UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR_VALUE,
                    unit_predictor_tv_floor_scale=UNIT_PREDICTOR_TV_FLOOR_SCALE_VALUE,
                    unit_predictor_load_path=UNIT_PREDICTOR_LOAD_PATH_VALUE,
                    unit_predictor_unit1_extra_tv_epochs=UNIT_PREDICTOR_UNIT1_EXTRA_TV_EPOCHS_VALUE,
                    unit_predictor_unit1_extra_tv_weight=UNIT_PREDICTOR_UNIT1_EXTRA_TV_WEIGHT_VALUE,
                    unit_predictor_auto_extra_tv=UNIT_PREDICTOR_AUTO_EXTRA_TV_VALUE,
                    unit_predictor_auto_extra_tv_epochs=UNIT_PREDICTOR_AUTO_EXTRA_TV_EPOCHS_VALUE,
                    unit_predictor_auto_extra_tv_tv_threshold=UNIT_PREDICTOR_AUTO_EXTRA_TV_TV_THRESHOLD_VALUE,
                    unit_predictor_auto_extra_tv_weight=UNIT_PREDICTOR_AUTO_EXTRA_TV_WEIGHT_VALUE,
                    predictor_warmup_rounds=SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS_VALUE,
                    sign4_curriculum_rounds=SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS_VALUE,
                    sign4_initial_scale=SUBPROBLEM_SIGN4_INITIAL_SCALE_VALUE,
                    sign4_final_scale=SUBPROBLEM_SIGN4_FINAL_SCALE_VALUE,
                    sign4_delay_rounds=SUBPROBLEM_SIGN4_DELAY_ROUNDS_VALUE,
                    enable_surrogate_delta_reference_lift=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT_VALUE,
                    surrogate_delta_reference_eps=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS_VALUE,
                    surrogate_delta_reference_scope=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE_VALUE,
                    surrogate_delta_reference_min_abs_factor=SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR_VALUE,
                    skip_existing_units=SURROGATE_SKIP_EXISTING_UNITS,
                    unit_predictor_finetune_lr=UNIT_PREDICTOR_FINETUNE_LR_VALUE,
                    unit_predictor_weight_decay=UNIT_PREDICTOR_WEIGHT_DECAY_VALUE,
                    case_name=CASE_NAME,
                )
            print_surrogate_results(trainers, all_samples)

            # Step 4: 联合BCD训练（theta/zeta + surrogate 约束，BCD迭代）
            # BCD 的对偶变量（agent.lambda_/mu/ita）在此阶段被联合训练器使用
            from joint_trainer import JointLPTrainer
            log("联合BCD训练: pg块→dual块→theta/zeta NN→surrogate NN→cal_viol")
            joint_trainer = JointLPTrainer(agent, trainers)
            joint_trainer.logger = logger
            joint_trainer.iter(
                max_iter=JOINT_MAX_ITER,
                dual_decay_round=JOINT_DUAL_DECAY_ROUND,
                nn_epochs=JOINT_NN_EPOCHS,
                surr_nn_epochs=JOINT_SURR_NN_EPOCHS,
            )

            # Step 5: 可选 FP 测试
            if RUN_FP:
                run_feasibility_pump_test(
                    ppc, all_samples, dual_predictor, trainers,
                    T_DELTA, FP_TEST_SAMPLES,
                )

        else:
            log(f"未知模式: '{MODE}'，可选: 'surrogate' | 'bcd' | 'sparse' | 'both'")
            sys.exit(1)

    except Exception as e:
        log(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 保存指标 & 生成图表 ────────────────────────────────
    figures_dir = Path(__file__).parent / 'result' / 'figures'
    _metrics_tag_raw = str(globals().get("METRICS_NAME_TAG", "") or "").strip()
    if _metrics_tag_raw:
        import re as _re_mtag
        _metrics_tag_safe = _re_mtag.sub(r"\s+", "_", _metrics_tag_raw)
        _metrics_tag_safe = _re_mtag.sub(r'[<>:"/\\|?*\x00-\x1f]', "", _metrics_tag_safe)[:80].strip(
            "_"
        )
        if _metrics_tag_safe:
            metrics_path = (
                Path(__file__).parent
                / "result"
                / f"training_metrics_{CASE_NAME}_{_metrics_tag_safe}_{timestamp}.json"
            )
        else:
            metrics_path = (
                Path(__file__).parent / "result" / f"training_metrics_{CASE_NAME}_{timestamp}.json"
            )
    else:
        metrics_path = Path(__file__).parent / "result" / f"training_metrics_{CASE_NAME}_{timestamp}.json"
    try:
        logger.save(metrics_path)
        log(f"训练指标已保存: {metrics_path}")
        viz = TrainingVisualizer(logger)
        viz.plot_all(figures_dir, trainers=locals().get('trainers'))
    except Exception as e:
        log(f"图表生成失败（非致命）: {e}")
        import traceback
        traceback.print_exc()

    # ── 汇总 ─────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    log(f"完成！模式={MODE}，耗时 {total_time / 60:.1f} 分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
