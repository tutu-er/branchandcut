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

修改顶部的 MODE / RUN_FP 变量切换执行模式。
"""

import sys
import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

# ──────────────────────── 依赖检查 ────────────────────────

def check_and_install_dependencies():
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gurobipy': 'gurobipy',
        'pypower': 'PYPOWER',
    }
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
    return True

if not check_and_install_dependencies():
    sys.exit(1)

# ──────────────────────── 模式配置 ────────────────────────
#
#   'surrogate' - V3 三时段代理约束训练
#   'bcd'       - BCD 主代理训练（Agent_NN_BCD）
#   'sparse'    - 稀疏支持集发现 → sparse BCD 训练
#   'both'      - BCD 训练 → surrogate 训练 → 联合 BCD 训练
#
MODE   = 'both'
ENABLE_SPARSE_SUPPORTS = False
RUN_FP = True        # True → 训练后运行 feasibility_pump 测试（bcd/sparse 模式不支持）

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# 添加 src/ 到模块搜索路径
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    from uc_NN_subproblem import (
        train_dual_predictor_from_data,
        SubproblemSurrogateTrainer,
        ActiveSetReader,
    )
    from uc_NN_subproblem_parallel import ParallelSubproblemSurrogateTrainer
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

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json_data(data_file: Path) -> list:
    """加载 JSON 数据文件并规范化为 v3 所需格式。"""
    log(f"加载数据文件: {data_file.name}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_samples = data.get('all_samples', [])
    if not all_samples:
        raise ValueError("JSON 文件中没有样本数据 (all_samples 为空)")

    log(f"  原始样本数: {len(all_samples)}")

    for sample in all_samples:
        if isinstance(sample.get('pd_data'), list):
            sample['pd_data'] = np.array(sample['pd_data'], dtype=float)

    return all_samples


def inject_bcd_lambda(all_samples: list, bcd_lambdas: list, T: int) -> None:
    """将 BCD 求解的功率平衡对偶变量注入样本，供 v3 dual predictor 使用。

    Args:
        all_samples: v3 格式样本列表，注入后每条样本含 'lambda' 字段。
        bcd_lambdas: Agent_NN_BCD.lambda_ 列表，每项为含 'lambda_power_balance' 的 dict。
        T: 时段数，用于生成零向量默认值。
    """
    for i, sample in enumerate(all_samples):
        if i >= len(bcd_lambdas):
            break
        lam_dict = bcd_lambdas[i]
        pb = lam_dict.get('lambda_power_balance', np.zeros(T))
        sample['lambda'] = np.asarray(pb, dtype=float)


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


# ──────────────────────── 模式实现 ────────────────────────

def run_surrogate(ppc, all_samples, T_DELTA, UNIT_IDS,
                  DUAL_EPOCHS, DUAL_BATCH_SIZE, MAX_ITER, NN_EPOCHS, save_dir,
                  n_workers: int = 4, logger: 'TrainingLogger | None' = None):
    """V3 代理约束训练（样本级并行），返回 (dual_predictor, trainers)。"""
    import os
    from pypower.ext2int import ext2int

    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    unit_ids = UNIT_IDS if UNIT_IDS is not None else list(range(ng))

    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log(f"开始并行代理训练: {n_samples} 样本，{len(unit_ids)} 机组，"
        f"n_workers={n_workers}，dual_epochs={DUAL_EPOCHS}，"
        f"bcd_iter={MAX_ITER}，nn_epochs={NN_EPOCHS}")
    print("=" * 70)

    # 步骤 1：对偶变量预测器（串行，NN 训练无需并行化）
    dual_save_path = os.path.join(save_dir, 'dual_predictor.pth') if save_dir else None
    dual_predictor = train_dual_predictor_from_data(
        ppc, all_samples, T_delta=T_DELTA,
        num_epochs=DUAL_EPOCHS, batch_size=DUAL_BATCH_SIZE,
        save_path=dual_save_path,
    )

    # 步骤 2：逐机组训练代理约束（n_workers<=1 串行，否则样本级并行）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    trainers = {}
    for i, g in enumerate(unit_ids):
        if n_workers <= 1:
            log(f"  机组 {g} ({i+1}/{len(unit_ids)}) — 串行")
            trainer = SubproblemSurrogateTrainer(
                ppc, all_samples, T_DELTA, g,
                lambda_predictor=dual_predictor,
            )
        else:
            log(f"  机组 {g} ({i+1}/{len(unit_ids)}) — 样本级并行 n_workers={n_workers}")
            trainer = ParallelSubproblemSurrogateTrainer(
                ppc, all_samples, T_DELTA, g,
                lambda_predictor=dual_predictor,
                n_workers=n_workers,
            )
        if logger is not None:
            trainer.logger = logger
        trainer.iter(max_iter=MAX_ITER, nn_epochs=NN_EPOCHS)
        trainers[g] = trainer
        if save_dir:
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))

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
        for t in range(min(5, nc)):
            if t + 2 >= T:
                break
            a = trainer.alpha_values[0, t]
            b = trainer.beta_values[0, t]
            g = trainer.gamma_values[0, t]
            d = trainer.delta_values[0, t]
            lhs = a * x0[t] + b * x0[t + 1] + g * x0[t + 2]
            viol = max(0.0, lhs - d)
            print(f"    t={t}: {a:.3f}*x[{t}] + {b:.3f}*x[{t+1}] "
                  f"+ {g:.3f}*x[{t+2}] <= {d:.3f}  "
                  f"(lhs={lhs:.3f}, viol={viol:.4f})")

        integrality = float(np.sum(x0 * (1 - x0)))
        print(f"  整数性指标(样本0): {integrality:.6f}  (0=完全整数)")


def run_bcd(ppc, all_samples: list, T_DELTA, MAX_ITER, bcd_model_dir,
            case_name: str = 'case', timestamp: str = '', n_workers: int = 4, NN_EPOCHS: int = 10, DUAL_DECAY_ROUND: int = 10,
            logger: 'TrainingLogger | None' = None,
            external_sparse_templates=None):
    """BCD 主代理训练（样本级并行），返回 ParallelAgent_NN_BCD 实例。"""
    log("模式: BCD 主代理训练（Agent_NN_BCD）")
    log(f"使用 {len(all_samples)} 个样本")

    print("\n" + "=" * 70)
    if n_workers <= 1:
        log(f"初始化 Agent_NN_BCD（串行），max_iter={MAX_ITER}")
    else:
        log(f"初始化 ParallelAgent_NN_BCD，max_iter={MAX_ITER}，n_workers={n_workers}")
    print("=" * 70)

    if external_sparse_templates is not None and n_workers > 1:
        log("警告: external_sparse_templates 当前仅支持串行 Agent_NN_BCD，将忽略 n_workers > 1")
        n_workers = 1

    if n_workers <= 1:
        log("使用串行 Agent_NN_BCD")
        agent = Agent_NN_BCD(
            ppc,
            all_samples,
            T_DELTA,
            external_sparse_templates=external_sparse_templates,
        )
    else:
        log(f"使用并行 ParallelAgent_NN_BCD (n_workers={n_workers})")
        agent = ParallelAgent_NN_BCD(ppc, all_samples, T_DELTA, n_workers=n_workers)

    print("\n" + "=" * 70)
    log("开始 BCD 迭代训练")
    print("=" * 70)

    if logger is not None:
        agent.logger = logger
    agent.iter(max_iter=MAX_ITER, dual_decay_round=DUAL_DECAY_ROUND, nn_epochs=NN_EPOCHS)

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
                   top_k_variables: int = 20, max_groups: int = 5, group_size: int = 3,
                   logger: 'TrainingLogger | None' = None):
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

    bootstrap_agent = Agent_NN_BCD(ppc, all_samples, T_DELTA)
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
        logger=logger,
        external_sparse_templates=template_library,
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
        pd_data = sample['pd_data']   # (nb, T)
        log(f"  样本 {i + 1}/{test_n}，pd_data shape={pd_data.shape}")
        try:
            x_result, success = recover_integer_solution(
                pd_data, trainers, dual_predictor, ppc, T_DELTA,
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


# ──────────────────────── 主函数 ────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print(f"训练脚本  模式: {MODE}")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────
    CASE_NAME       = 'case30'      # 'case14' / 'case30' / 'case39'
    MAX_SAMPLES     = 4           # 最多使用多少个样本（None=全部）
    T_DELTA         = 1.0
    DUAL_EPOCHS     = 50
    DUAL_BATCH_SIZE = 8
    MAX_ITER        = 20            # 迭代次数（BCD / surrogate BCD 轮数）
    DUAL_DECAY_ROUND= 10
    NN_EPOCHS       = 10            # surrogate 模式每次 BCD 迭代的 NN 训练轮数
    UNIT_IDS        = None          # None = 所有机组；或如 [0, 1, 2]
    FP_TEST_SAMPLES = 3             # feasibility_pump 模式：测试样本数
    N_WORKERS_BCD   = 2             # 样本级并行线程数；1 = 串行（BCD 建议先用串行），>1 = 线程并行
    N_WORKERS_SUBPROBLEM = 2             # 样本级并行线程数；1 = 串行（BCD 建议先用串行），>1 = 线程并行
    JOINT_MAX_ITER  = 10            # 联合BCD训练外层迭代次数
    JOINT_NN_EPOCHS = 5             # 联合BCD训练每轮theta/zeta NN训练epoch数
    JOINT_SURR_NN_EPOCHS = 5        # 联合BCD训练每轮surrogate NN训练epoch数
    JOINT_DUAL_DECAY_ROUND = 10     # 联合BCD训练dual_para_bound衰减轮次
    ACTIVE_SETS_FILE = None          # 指定 active_sets JSON 文件路径（None=自动查找最新）
    BCD_MODEL_FILE   = None          # 指定已有 BCD 模型 .pth 文件路径（None=从头训练；both 模式下可跳过 BCD 训练）
    SPARSE_TOP_K_VARIABLES = 20      # sparse 支持发现：保留的高价值 x[g,t] 变量数量
    SPARSE_MAX_GROUPS = 5            # sparse 支持发现：构造的支持集模板数量上限
    SPARSE_GROUP_SIZE = 3            # sparse 支持发现：每条模板最多包含多少个参与变量

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
    log(f"加载 PyPower 案例: {CASE_NAME}")
    ppc_map = {
        'case14': pypower.case14.case14,
        'case30': pypower.case30.case30,
        'case39': pypower.case39.case39,
    }
    if CASE_NAME not in ppc_map:
        print(f"未知案例: {CASE_NAME}，可选: {list(ppc_map)}")
        sys.exit(1)
    ppc = ppc_map[CASE_NAME]()
    n_units = ppc['gen'].shape[0]
    n_buses = ppc['bus'].shape[0]
    log(f"  {n_units} 机组，{n_buses} 节点")

    # ── 查找数据文件 ─────────────────────────────────────
    if ACTIVE_SETS_FILE is not None:
        data_file = Path(ACTIVE_SETS_FILE)
        if not data_file.is_absolute():
            data_file = Path(__file__).parent / data_file
        if not data_file.exists():
            log(f"错误: 指定的文件不存在: {data_file}")
            sys.exit(1)
        log(f"使用指定文件: {data_file}")
    else:
        data_file = pick_data_file(data_dir, CASE_NAME)
    if data_file is None:
        log(f"错误: 在 {data_dir} 中未找到 {CASE_NAME} 的 JSON 数据文件。")
        log("请先运行 ActiveSetLearner 生成数据，或在 result/ 目录下放置")
        log(f"命名为 active_sets_{CASE_NAME}_*.json 的数据文件后重试。")
        sys.exit(1)

    # ── 执行模式分支 ─────────────────────────────────────
    try:
        if MODE == 'bcd':
            # BCD 通过 ActiveSetReader 加载（含 unit_commitment_matrix）
            log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
            all_samples_bcd = load_active_set_from_json(str(data_file))
            if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
                all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
            run_bcd(ppc, all_samples_bcd, T_DELTA, MAX_ITER, bcd_model_dir,
                    case_name=CASE_NAME, timestamp=timestamp, n_workers=N_WORKERS_BCD, NN_EPOCHS=NN_EPOCHS, DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
                    logger=logger)
            if RUN_FP:
                log("警告: bcd 模式不支持 RUN_FP（需要 trainers），请改用 both 模式")

        elif MODE == 'sparse':
            log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
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
                MAX_ITER,
                bcd_model_dir,
                case_name=CASE_NAME,
                timestamp=timestamp,
                NN_EPOCHS=NN_EPOCHS,
                DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
                top_k_variables=SPARSE_TOP_K_VARIABLES,
                max_groups=SPARSE_MAX_GROUPS,
                group_size=SPARSE_GROUP_SIZE,
                logger=logger,
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

            dual_predictor, trainers = run_surrogate(
                ppc, all_samples, T_DELTA, UNIT_IDS,
                DUAL_EPOCHS, DUAL_BATCH_SIZE, MAX_ITER, NN_EPOCHS, save_dir,
                n_workers=N_WORKERS_SUBPROBLEM, logger=logger,
            )
            print_surrogate_results(trainers, all_samples)

            if RUN_FP:
                run_feasibility_pump_test(
                    ppc, all_samples, dual_predictor, trainers,
                    T_DELTA, FP_TEST_SAMPLES,
                )

        elif MODE == 'both':
            # Step 1: BCD 训练（或从已有模型加载跳过）
            log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
            all_samples_bcd = load_active_set_from_json(str(data_file))
            if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
                all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]

            sparse_template_library = None
            if ENABLE_SPARSE_SUPPORTS:
                log("both 模式: 启用 sparse 支持集发现")
                if N_WORKERS_BCD > 1:
                    log("警告: both + sparse 当前仅支持串行 Agent_NN_BCD，将忽略 N_WORKERS_BCD > 1")
                _bootstrap_agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)
                _, sparse_template_library = build_sparse_template_library_from_bcd_agent(
                    _bootstrap_agent,
                    top_k_variables=SPARSE_TOP_K_VARIABLES,
                    max_groups=SPARSE_MAX_GROUPS,
                    group_size=SPARSE_GROUP_SIZE,
                    sparse_dir=Path(__file__).parent / 'result' / 'sparse_templates',
                    case_name=CASE_NAME,
                    timestamp=timestamp,
                )

            if BCD_MODEL_FILE is not None:
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
                    )
                elif N_WORKERS_BCD <= 1:
                    agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)
                else:
                    agent = ParallelAgent_NN_BCD(ppc, all_samples_bcd, T_DELTA, n_workers=N_WORKERS_BCD)
                agent.load_model_parameters(str(bcd_path))
                log("BCD 模型加载成功，跳过训练")
            else:
                agent = run_bcd(
                    ppc,
                    all_samples_bcd,
                    T_DELTA,
                    MAX_ITER,
                    bcd_model_dir,
                    case_name=CASE_NAME,
                    timestamp=timestamp,
                    n_workers=N_WORKERS_BCD if not ENABLE_SPARSE_SUPPORTS else 1,
                    NN_EPOCHS=NN_EPOCHS,
                    DUAL_DECAY_ROUND=DUAL_DECAY_ROUND,
                    logger=logger,
                    external_sparse_templates=sparse_template_library,
                )

            # Step 2: 加载 v3 格式样本（subproblem 独立训练，不注入 BCD 对偶变量）
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            # Step 3: surrogate 训练（与 surrogate 模式一致，自行 LP 求解对偶变量）
            # BCD 的对偶变量仅在后续联合训练中使用，不注入 subproblem 训练
            dual_predictor, trainers = run_surrogate(
                ppc, all_samples, T_DELTA, UNIT_IDS,
                DUAL_EPOCHS, DUAL_BATCH_SIZE, MAX_ITER, NN_EPOCHS, save_dir,
                n_workers=N_WORKERS_SUBPROBLEM, logger=logger,
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
    metrics_path = Path(__file__).parent / 'result' / f'training_metrics_{CASE_NAME}_{timestamp}.json'
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
