#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本（多模式）
- surrogate: V3 三时段代理约束训练（uc_NN_subproblem_v3）
- bcd:       BCD 主代理训练（uc_NN_BCD，Agent_NN_BCD）
- both:      BCD 训练 → 提取功率平衡对偶变量注入样本 → surrogate 训练

可选标志 RUN_FP=True：训练后运行 feasibility_pump 可行性泵测试
（bcd 模式不支持 RUN_FP，请改用 both 模式）

修改顶部的 MODE / RUN_FP 变量切换执行模式。
"""

import sys
import subprocess
import time
import json
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
#   'both'      - BCD 训练 → lambda 注入 → surrogate 训练
#
MODE   = 'both'
RUN_FP = True        # True → 训练后运行 feasibility_pump 测试（bcd 模式不支持）

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# 添加 src/ 到模块搜索路径
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    from uc_NN_subproblem_v3 import (
        train_dual_predictor_from_data,
        train_subproblem_surrogate_from_data,
        train_complete_model,
        ActiveSetReader,
    )
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

# BCD 模式额外导入
if MODE in ('bcd', 'both'):
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
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


def pick_data_file(result_dir: Path, case_name: str) -> Path:
    """按优先级查找最合适的数据文件。"""
    specific = sorted(result_dir.glob(f'active_sets_{case_name}_*.json'))
    if specific:
        return specific[-1]
    any_files = sorted(result_dir.glob('active_sets_*.json'))
    if any_files:
        log(f"未找到 {case_name} 专属文件，使用: {any_files[-1].name}")
        return any_files[-1]
    return None


# ──────────────────────── 模式实现 ────────────────────────

def run_surrogate(ppc, all_samples, T_DELTA, UNIT_IDS,
                  DUAL_EPOCHS, DUAL_BATCH_SIZE, MAX_ITER, NN_EPOCHS, save_dir):
    """V3 代理约束训练，返回 (dual_predictor, trainers)。"""
    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log(f"开始代理训练: {n_samples} 样本，dual_epochs={DUAL_EPOCHS}，"
        f"bcd_iter={MAX_ITER}，nn_epochs={NN_EPOCHS}")
    print("=" * 70)

    dual_predictor, trainers = train_complete_model(
        ppc, all_samples, T_delta=T_DELTA,
        unit_ids=UNIT_IDS,
        dual_epochs=DUAL_EPOCHS,
        dual_batch_size=DUAL_BATCH_SIZE,
        surrogate_max_iter=MAX_ITER,
        surrogate_nn_epochs=NN_EPOCHS,
        save_dir=save_dir,
    )
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


def run_bcd(ppc, data_file, MAX_SAMPLES, T_DELTA, MAX_ITER, result_dir,
            case_name: str = 'case', timestamp: str = ''):
    """BCD 主代理训练，返回 Agent_NN_BCD 实例。"""
    log("模式: BCD 主代理训练（Agent_NN_BCD）")
    log(f"通过 ActiveSetReader 加载数据: {data_file.name}")

    all_samples_bcd = load_active_set_from_json(str(data_file))
    if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
        log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples_bcd)}）")
        all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
    log(f"  使用 {len(all_samples_bcd)} 个样本")

    print("\n" + "=" * 70)
    log(f"初始化 Agent_NN_BCD，max_iter={MAX_ITER}")
    print("=" * 70)

    agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)

    print("\n" + "=" * 70)
    log("开始 BCD 迭代训练")
    print("=" * 70)

    agent.iter(max_iter=MAX_ITER)

    # 保存模型（含算例名和时间戳）
    suffix = f'_{case_name}_{timestamp}' if timestamp else f'_{case_name}'
    save_path = str(result_dir / f'bcd_model{suffix}.pth')
    try:
        agent.save_model_parameters(save_path)
        log(f"BCD 模型参数保存至: {save_path}")
    except Exception as e:
        log(f"模型保存失败（非致命）: {e}")

    return agent


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
    MAX_SAMPLES     = None            # 最多使用多少个样本（None=全部）
    T_DELTA         = 1.0
    DUAL_EPOCHS     = 50
    DUAL_BATCH_SIZE = 8
    MAX_ITER        = 20            # 迭代次数（BCD / surrogate BCD 轮数）
    NN_EPOCHS       = 20            # surrogate 模式每次 BCD 迭代的 NN 训练轮数
    UNIT_IDS        = None          # None = 所有机组；或如 [0, 1, 2]
    FP_TEST_SAMPLES = 3             # feasibility_pump 模式：测试样本数

    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = str(result_dir / f'subproblem_models_{CASE_NAME}_{timestamp}')

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
    data_file = pick_data_file(result_dir, CASE_NAME)
    if data_file is None:
        log(f"错误: 在 {result_dir} 中未找到 {CASE_NAME} 的 JSON 数据文件。")
        log("请先运行 ActiveSetLearner 生成数据，或在 result/ 目录下放置")
        log(f"命名为 active_sets_{CASE_NAME}_*.json 的数据文件后重试。")
        sys.exit(1)

    # ── 执行模式分支 ─────────────────────────────────────
    try:
        if MODE == 'bcd':
            # BCD 通过 ActiveSetReader 加载（含 unit_commitment_matrix）
            run_bcd(ppc, data_file, MAX_SAMPLES, T_DELTA, MAX_ITER, result_dir,
                    case_name=CASE_NAME, timestamp=timestamp)
            if RUN_FP:
                log("警告: bcd 模式不支持 RUN_FP（需要 trainers），请改用 both 模式")

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
            )
            print_surrogate_results(trainers, all_samples)

            if RUN_FP:
                run_feasibility_pump_test(
                    ppc, all_samples, dual_predictor, trainers,
                    T_DELTA, FP_TEST_SAMPLES,
                )

        elif MODE == 'both':
            # Step 1: BCD 训练
            agent = run_bcd(ppc, data_file, MAX_SAMPLES, T_DELTA, MAX_ITER, result_dir,
                            case_name=CASE_NAME, timestamp=timestamp)

            # Step 2: 加载 v3 格式样本
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            # Step 3: 注入 BCD 求解的功率平衡对偶变量
            if hasattr(agent, 'lambda_') and agent.lambda_:
                log(f"注入 BCD lambda（{len(agent.lambda_)} 条）→ 样本 'lambda' 字段")
                inject_bcd_lambda(all_samples, agent.lambda_, T_from_data)
            else:
                log("警告: agent.lambda_ 为空，surrogate 训练将自行求解 LP 获取对偶变量")

            # Step 4: surrogate 训练（使用 BCD 的 lambda 初始化）
            dual_predictor, trainers = run_surrogate(
                ppc, all_samples, T_DELTA, UNIT_IDS,
                DUAL_EPOCHS, DUAL_BATCH_SIZE, MAX_ITER, NN_EPOCHS, save_dir,
            )
            print_surrogate_results(trainers, all_samples)

            # Step 5: 可选 FP 测试
            if RUN_FP:
                run_feasibility_pump_test(
                    ppc, all_samples, dual_predictor, trainers,
                    T_DELTA, FP_TEST_SAMPLES,
                )

        else:
            log(f"未知模式: '{MODE}'，可选: 'surrogate' | 'bcd' | 'both'")
            sys.exit(1)

    except Exception as e:
        log(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 汇总 ─────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    log(f"完成！模式={MODE}，耗时 {total_time / 60:.1f} 分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
