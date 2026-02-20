#!/usr/bin/env python3
"""
V3训练脚本（简化版，先测试网络和敏感时段识别）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import json
import torch
import pypower.case14 as case14

# 测试导入V3模块
try:
    from uc_NN_subproblem_v3 import (
        SubproblemSurrogateNet,
        identify_sensitive_timesteps
    )
    print("✅ V3模块导入成功")
except ImportError as e:
    print(f"❌ V3模块导入失败: {e}")
    sys.exit(1)

def test_network():
    """测试V3网络架构"""
    print("\n" + "="*70)
    print("🧪 测试V3网络架构")
    print("="*70)
    
    T = 24
    n_load = 9
    input_dim = n_load * T + T  # 216 + 24 = 240
    max_constraints = 15
    
    # 创建网络
    net = SubproblemSurrogateNet(
        input_dim=input_dim,
        T=T,
        max_constraints=max_constraints
    )
    
    print(f"\n网络参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  时段数: {T}")
    print(f"  最大约束数: {max_constraints}")
    
    # 测试前向传播
    batch_size = 8
    x_input = torch.randn(batch_size, input_dim)
    
    alphas, betas, gammas, deltas = net(x_input)
    
    print(f"\n输出形状:")
    print(f"  alphas: {alphas.shape}")  # (8, 15)
    print(f"  betas: {betas.shape}")    # (8, 15)
    print(f"  gammas: {gammas.shape}")  # (8, 15)
    print(f"  deltas: {deltas.shape}")  # (8, 15)
    
    # 检查delta是否非负
    assert (deltas >= 0).all(), "Delta should be non-negative"
    print(f"\n✅ Delta非负性检查通过")
    print(f"  Delta范围: [{deltas.min():.4f}, {deltas.max():.4f}]")
    
    # 参数统计
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n网络总参数: {total_params:,}")

def test_sensitive_identification():
    """测试敏感时段识别"""
    print("\n" + "="*70)
    print("🧪 测试敏感时段识别")
    print("="*70)
    
    # 模拟不同场景的x值
    scenarios = {
        "全整数": np.array([1,1,1,0,0,0,1,1,1,0,0,0] * 2),  # 完全0/1
        "部分分数": np.array([1,1,0.7,0.3,0,0,1,0.8,0.2,0,0,0] * 2),  # 有分数值
        "全分数": np.array([0.5] * 24),  # 全是0.5
        "混合": np.concatenate([np.ones(8), np.array([0.9,0.7,0.5,0.3,0.1]), np.zeros(11)])
    }
    
    for name, x_vals in scenarios.items():
        sensitive = identify_sensitive_timesteps(x_vals, max_constraints=15)
        
        # 计算整数性
        integrality = np.sum(x_vals * (1 - x_vals))
        
        print(f"\n场景: {name}")
        print(f"  整数性: {integrality:.4f}")
        print(f"  敏感时段数: {len(sensitive)}")
        print(f"  敏感时段: {sensitive}")
        
        if len(sensitive) > 0:
            # 显示敏感时段的x值
            print(f"  敏感时段x值样例: ", end="")
            for t in sensitive[:5]:
                window = x_vals[t:t+3]
                print(f"t{t}={window} ", end="")
            print()

def test_with_real_data():
    """用真实数据测试"""
    print("\n" + "="*70)
    print("🧪 用真实数据测试V3组件")
    print("="*70)
    
    # 加载数据
    data_file = Path('result/active_sets_20250803_025149.json')
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    ppc = case14.case14()
    sample = data['all_samples'][0]
    pd_data = np.array(sample['pd_data'])
    
    print(f"\n数据信息:")
    print(f"  pd_data shape: {pd_data.shape}")
    print(f"  机组数: {ppc['gen'].shape[0]}")
    
    # 模拟一个LP松弛解
    np.random.seed(42)
    x_lp = np.random.rand(24)  # 模拟LP松弛结果
    
    # 识别敏感时段
    sensitive = identify_sensitive_timesteps(x_lp, max_constraints=15)
    
    print(f"\n模拟LP松弛解:")
    print(f"  整数性: {np.sum(x_lp * (1-x_lp)):.4f}")
    print(f"  敏感时段数: {len(sensitive)}")
    print(f"  约束覆盖率: {len(sensitive)}/{24-2} = {len(sensitive)/(24-2)*100:.1f}%")

if __name__ == '__main__':
    print("="*70)
    print("🚀 V3改进版组件测试")
    print("="*70)
    
    # 测试网络
    test_network()
    
    # 测试敏感时段识别
    test_sensitive_identification()
    
    # 用真实数据测试
    test_with_real_data()
    
    print("\n" + "="*70)
    print("✅ V3组件测试完成")
    print("="*70)
    print("\n下一步: 修改BCD训练方法以支持三时段约束")
