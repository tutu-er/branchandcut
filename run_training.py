#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 三时段耦合约束训练脚本
- 支持从 JSON 数据文件加载，或自动生成测试数据
- 使用 uc_NN_subproblem_v3 中的训练函数
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

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    from uc_NN_subproblem_v3 import (
        generate_test_data,
        train_dual_predictor_from_data,
        train_subproblem_surrogate_from_data,
        train_complete_model,
        ActiveSetReader,
    )
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

# ──────────────────────── 工具函数 ────────────────────────

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

    # 规范化每个样本
    for sample in all_samples:
        # pd_data: list → numpy array
        if isinstance(sample.get('pd_data'), list):
            sample['pd_data'] = np.array(sample['pd_data'], dtype=float)
        # lambda: dict → 确保 v3 可以解析（保持原样，v3 内部用 _extract_lambda_power_balance 处理）
        # active_set 保持原样（v3 _initialize_solve 可处理整数索引列表，不提取 x 就用全零）

    return all_samples


def pick_data_file(result_dir: Path, case_name: str) -> Path:
    """按优先级查找最合适的数据文件。"""
    # 优先找 case 专属文件
    specific = sorted(result_dir.glob(f'active_sets_{case_name}_*.json'))
    if specific:
        return specific[-1]   # 取最新的
    # 退而求其次找任意文件
    any_files = sorted(result_dir.glob('active_sets_*.json'))
    if any_files:
        log(f"未找到 {case_name} 专属文件，使用: {any_files[-1].name}")
        return any_files[-1]
    return None


# ──────────────────────── 主函数 ────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print("V3 三时段耦合约束训练脚本")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────
    CASE_NAME       = 'case30'      # 'case14' / 'case30' / 'case39'
    USE_JSON_DATA   = True          # True=从 JSON 加载；False=自动生成测试数据
    MAX_SAMPLES     = 30            # 最多使用多少个样本（None=全部）
    T_DELTA         = 1.0
    DUAL_EPOCHS     = 50
    DUAL_BATCH_SIZE = 8
    MAX_ITER        = 20            # BCD 迭代次数
    NN_EPOCHS       = 50            # 每次 BCD 迭代中 NN 训练轮数
    UNIT_IDS        = None          # None = 所有机组；或如 [0, 1, 2]

    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(exist_ok=True)
    save_dir = str(result_dir / 'subproblem_models')

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

    # ── 准备数据 ─────────────────────────────────────────
    if USE_JSON_DATA:
        data_file = pick_data_file(result_dir, CASE_NAME)
        if data_file is None:
            log("未找到 JSON 数据文件，改为自动生成测试数据")
            USE_JSON_DATA = False

    if USE_JSON_DATA:
        all_samples = load_json_data(data_file)
        if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
            log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
            all_samples = all_samples[:MAX_SAMPLES]
        T_from_data = all_samples[0]['pd_data'].shape[1]
        log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")
    else:
        T_GEN = 8
        N_GEN = 20
        log(f"自动生成 {N_GEN} 个测试样本 (T={T_GEN})")
        all_samples = generate_test_data(ppc, T=T_GEN, n_samples=N_GEN)

    n_samples = len(all_samples)

    # ── 训练 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    log(f"开始完整训练: {n_samples} 样本，dual_epochs={DUAL_EPOCHS}，"
        f"bcd_iter={MAX_ITER}，nn_epochs={NN_EPOCHS}")
    print("=" * 70)

    try:
        dual_predictor, trainers = train_complete_model(
            ppc, all_samples, T_delta=T_DELTA,
            unit_ids=UNIT_IDS,
            dual_epochs=DUAL_EPOCHS,
            dual_batch_size=DUAL_BATCH_SIZE,
            surrogate_max_iter=MAX_ITER,
            surrogate_nn_epochs=NN_EPOCHS,
            save_dir=save_dir,
        )
    except Exception as e:
        log(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 验证结果（V3 参数语义）──────────────────────────
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

        # 显示样本0的前几条约束（V3：alpha*x[t] + beta*x[t+1] + gamma*x[t+2] <= delta）
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

        # 整数性指标
        integrality = float(np.sum(x0 * (1 - x0)))
        print(f"  整数性指标(样本0): {integrality:.6f}  (0=完全整数)")

    # ── 汇总 ─────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    log(f"训练完成！共 {len(trainers)} 个机组，"
        f"耗时 {total_time/60:.1f} 分钟")
    log(f"模型保存至: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
