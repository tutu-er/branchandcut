#!/usr/bin/env python3
"""
时序耦合约束训练脚本 - 带依赖检查
"""

import sys
import subprocess

# 检查并安装依赖
def check_and_install_dependencies():
    """检查依赖并尝试安装"""
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gurobipy': 'gurobipy',
        'pypower': 'PYPOWER'
    }
    
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {import_name} 已安装")
        except ImportError:
            missing.append(package_name)
            print(f"✗ {import_name} 未安装")
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        response = input("是否自动安装？(y/n): ")
        if response.lower() == 'y':
            print("正在安装...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✓ 依赖安装完成")
            return True
        else:
            print("请手动安装依赖后重试")
            return False
    return True

if not check_and_install_dependencies():
    sys.exit(1)

# 导入模块
import numpy as np
import json
from pathlib import Path

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import pypower.case14 as case14
    from uc_NN_subproblem import (
        train_dual_predictor_from_data,
        train_subproblem_surrogate_from_data,
        ActiveSetReader
    )
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

def main():
    print("=" * 70)
    print("🚀 时序耦合约束训练脚本")
    print("=" * 70)
    
    # 1. 选择数据文件
    result_dir = Path(__file__).parent / 'result'
    json_files = list(result_dir.glob('active_sets_*.json'))
    
    if not json_files:
        print("❌ 未找到数据文件")
        return
    
    print(f"\n📦 找到 {len(json_files)} 个数据文件:")
    for i, f in enumerate(json_files[:5]):
        print(f"  {i+1}. {f.name}")
    
    # 使用第一个文件
    data_file = json_files[0]
    print(f"\n✓ 使用数据文件: {data_file.name}")
    
    # 2. 加载数据
    print("\n📊 加载数据...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    all_samples = data['all_samples']
    n_samples = len(all_samples)
    print(f"✓ 加载 {n_samples} 个样本")
    
    # 限制样本数量加快测试
    max_samples = 10
    if n_samples > max_samples:
        print(f"  (限制为前{max_samples}个样本进行快速测试)")
        all_samples = all_samples[:max_samples]
    
    # 转换数据格式：list -> numpy array
    print("  转换数据格式...")
    for sample in all_samples:
        sample['pd_data'] = np.array(sample['pd_data'])
    print("  ✓ 数据格式转换完成")
    
    # 3. 加载PyPower案例
    print("\n🔌 加载PyPower案例...")
    ppc = case14.case14()
    print(f"✓ case14: {ppc['gen'].shape[0]}个机组, {ppc['bus'].shape[0]}个节点")
    
    # 4. 训练对偶变量预测器
    print("\n" + "=" * 70)
    print("第1阶段：训练对偶变量预测器")
    print("=" * 70)
    
    try:
        lambda_predictor = train_dual_predictor_from_data(
            ppc, all_samples, T_delta=1.0,
            num_epochs=10,
            batch_size=min(4, len(all_samples)),
            save_path='result/dual_predictor.pth'
        )
        print("\n✅ 对偶变量预测器训练完成")
    except Exception as e:
        print(f"\n❌ 对偶变量预测器训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 训练时序耦合代理约束
    print("\n" + "=" * 70)
    print("第2阶段：训练时序耦合代理约束")
    print("=" * 70)
    
    # 选择一个机组进行训练
    unit_id = 0
    print(f"\n📍 训练机组 {unit_id}...")
    
    try:
        trainer = train_subproblem_surrogate_from_data(
            ppc, all_samples, unit_id=unit_id,
            T_delta=1.0, lambda_predictor=lambda_predictor,
            max_iter=5,  # 快速测试用较少迭代
            nn_epochs=5,
            save_path=f'result/temporal_coupling_unit{unit_id}.pth'
        )
        print(f"\n✅ 机组{unit_id}训练完成")
        
        # 6. 验证结果
        print("\n" + "=" * 70)
        print("验证结果")
        print("=" * 70)
        
        T = trainer.T
        print(f"\n📏 参数形状检查:")
        print(f"  alpha_values: {trainer.alpha_values.shape} (期望: {(len(all_samples), T-1)})")
        print(f"  beta_values: {trainer.beta_values.shape} (期望: {(len(all_samples), T-1)})")
        print(f"  gamma_values: {trainer.gamma_values.shape} (期望: {(len(all_samples), T-1)})")
        
        # 显示第一个样本的约束
        print(f"\n🎯 样本0的时序耦合约束:")
        for t in range(min(5, T-1)):  # 只显示前5个
            alpha_t = trainer.alpha_values[0, t]
            beta_t = trainer.beta_values[0, t]
            gamma_t = trainer.gamma_values[0, t]
            x_t = trainer.x[0, t]
            x_t1 = trainer.x[0, t+1]
            lhs = alpha_t * x_t + beta_t * x_t1
            viol = max(0, lhs - gamma_t)
            print(f"  t={t}: {alpha_t:.3f}*x[{t}] + {beta_t:.3f}*x[{t+1}] ≤ {gamma_t:.3f}")
            print(f"        lhs={lhs:.3f}, viol={viol:.6f}, x[{t}]={x_t:.3f}, x[{t+1}]={x_t1:.3f}")
        
        # 计算整数性
        x_vals = trainer.x[0]
        integrality = np.sum(x_vals * (1 - x_vals))
        print(f"\n📐 整数性指标: {integrality:.6f} (越小越好，0=完全整数)")
        
        print("\n" + "=" * 70)
        print("✅ 训练完成！")
        print("=" * 70)
        
        # 保存位置
        save_path = Path('result') / f'temporal_coupling_unit{unit_id}.pth'
        print(f"\n💾 模型已保存至: {save_path}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
