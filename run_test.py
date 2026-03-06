#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本（多模式）
- surrogate: 加载已训练的 V3 代理约束模型，输出参数摘要，可选运行可行性泵
- bcd:       加载已训练的 BCD 神经网络模型，报告参数统计

修改顶部的 MODE / MODEL_DIR / BCD_MODEL_PATH 等变量切换执行模式。
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
#   'surrogate' - 加载 V3 代理约束模型并测试
#   'bcd'       - 加载 BCD 神经网络模型并报告参数统计
#
MODE      = 'surrogate'
RUN_FP    = True        # surrogate 模式：是否运行可行性泵测试
CASE_NAME = 'case30'   # 'case14' / 'case30' / 'case39'

# surrogate 模式：指定已训练模型目录（训练时输出的带时间戳路径）
MODEL_DIR = 'result/subproblem_models_case30_20240101_120000'

# bcd 模式：指定已训练 BCD 模型 .pth 文件路径
BCD_MODEL_PATH = 'result/bcd_model_case30_20240101_120000.pth'

TEST_SAMPLES = 3   # 测试/评估样本数

# ──────────────────────── 导入 ────────────────────────

import numpy as np

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    from uc_NN_subproblem_v3 import (
        load_trained_models,
        ActiveSetReader,
    )
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

if MODE == 'bcd':
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)

if RUN_FP and MODE == 'surrogate':
    try:
        from feasibility_pump import recover_integer_solution
    except ImportError as e:
        print(f"feasibility_pump 模块导入失败: {e}")
        sys.exit(1)

# ──────────────────────── 工具函数 ────────────────────────


def log(msg: str) -> None:
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


def print_surrogate_results(trainers: dict, all_samples: list) -> None:
    """打印代理训练结果摘要。"""
    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log("模型参数摘要")
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

        x0 = trainer.x[0]
        print(f"  样本0 时序约束示例（最多5条）:")
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


def run_fp_test(ppc, all_samples: list, dual_predictor, trainers: dict,
                T_DELTA: float, n_test: int) -> list:
    """对多个样本运行可行性泵并汇总结果。"""
    test_n = min(n_test, len(all_samples))
    print("\n" + "=" * 70)
    log(f"可行性泵测试: {test_n} 个样本")
    print("=" * 70)

    results = []
    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
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


def test_surrogate(ppc, all_samples: list, T_DELTA: float,
                   model_dir: str, unit_ids) -> None:
    """加载 surrogate 模型，打印参数摘要，可选运行 FP。"""
    print("\n" + "=" * 70)
    log(f"加载 surrogate 模型: {model_dir}")
    print("=" * 70)

    if not Path(model_dir).exists():
        log(f"错误: 模型目录不存在: {model_dir}")
        log("请先运行 run_training.py 生成模型，或修改 MODEL_DIR 配置")
        sys.exit(1)

    dual_predictor, trainers = load_trained_models(
        ppc, all_samples, T_DELTA,
        load_dir=model_dir,
        unit_ids=unit_ids,
    )

    log(f"已加载 {len(trainers)} 个机组的代理约束模型")
    print_surrogate_results(trainers, all_samples[:TEST_SAMPLES])

    if RUN_FP:
        run_fp_test(ppc, all_samples, dual_predictor, trainers, T_DELTA, TEST_SAMPLES)


def test_bcd(ppc, data_file: Path, bcd_model_path: str,
             MAX_SAMPLES, T_DELTA: float) -> None:
    """加载 BCD 模型，初始化 agent，报告参数统计。"""
    print("\n" + "=" * 70)
    log(f"加载 BCD 模型: {bcd_model_path}")
    print("=" * 70)

    model_path = Path(bcd_model_path)
    if not model_path.exists():
        log(f"错误: BCD 模型文件不存在: {bcd_model_path}")
        log("请先运行 run_training.py MODE='bcd' 生成模型，或修改 BCD_MODEL_PATH 配置")
        sys.exit(1)

    log(f"通过 load_active_set_from_json 加载数据: {data_file.name}")
    all_samples_bcd = load_active_set_from_json(str(data_file))
    if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
        all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
    log(f"  使用 {len(all_samples_bcd)} 个样本")

    agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)
    agent.load_model_parameters(str(model_path))
    log("BCD 模型加载成功")

    # 报告 theta / zeta 参数统计
    print("\n" + "=" * 70)
    log("BCD 模型参数统计")
    print("=" * 70)

    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        import torch
        total_params = sum(p.numel() for p in agent.theta_net.parameters())
        log(f"  theta_net 参数量: {total_params:,}")

    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        import torch
        total_params = sum(p.numel() for p in agent.zeta_net.parameters())
        log(f"  zeta_net  参数量: {total_params:,}")

    if hasattr(agent, 'theta_values') and agent.theta_values:
        n_theta = len(agent.theta_values)
        vals = list(agent.theta_values.values())
        log(f"  theta_values 数量: {n_theta}，"
            f"均值={np.mean(vals):.4f}，标准差={np.std(vals):.4f}")

    if hasattr(agent, 'zeta_values') and agent.zeta_values:
        n_zeta = len(agent.zeta_values)
        vals = list(agent.zeta_values.values())
        log(f"  zeta_values  数量: {n_zeta}，"
            f"均值={np.mean(vals):.4f}，标准差={np.std(vals):.4f}")

    log("BCD 测试完成（如需评估解质量，请使用 run_training.py MODE='both' + RUN_FP=True）")


# ──────────────────────── 主函数 ────────────────────────


def main():
    start_time = time.time()

    print("=" * 70)
    print(f"测试脚本  模式: {MODE}  算例: {CASE_NAME}")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────
    MAX_SAMPLES = None   # 最多使用多少个样本（None=全部）
    T_DELTA     = 1.0
    UNIT_IDS    = None   # None = 所有机组；或如 [0, 1, 2]

    result_dir = Path(__file__).parent / 'result'

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
        if MODE == 'surrogate':
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            # 将相对路径解析为绝对路径
            model_dir = str((Path(__file__).parent / MODEL_DIR).resolve())
            test_surrogate(ppc, all_samples, T_DELTA, model_dir, UNIT_IDS)

        elif MODE == 'bcd':
            bcd_path = str((Path(__file__).parent / BCD_MODEL_PATH).resolve())
            test_bcd(ppc, data_file, bcd_path, MAX_SAMPLES, T_DELTA)

        else:
            log(f"未知模式: '{MODE}'，可选: 'surrogate' | 'bcd'")
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
