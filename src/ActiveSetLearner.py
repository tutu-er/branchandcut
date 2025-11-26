import numpy as np
from collections import defaultdict
import cvxpy as cp
import pandas as pd
import pypower
import pypower.case14
import pypower.case9
import pypower.idx_bus
import json
from datetime import datetime

from pathlib import Path
import io
import sys
import re

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
from src.ed_cvxpy import EconomicDispatchCVXPY
from src.uc_cvxpy import UnitCommitmentModelCVXPY

from src.uc_gurobipy import UnitCommitmentModel
from src.ed_gurobipy import EconomicDispatchGurobi

class ActiveSetLearner:
    def __init__(self, alpha=0.05, delta=0.01, epsilon=0.04, ppc=None, T_delta=4, Pd=None):
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.observed_active_sets = set()
        self.samples = []
        self.M = 1
        self.ppc = ppc
        self.T_delta = T_delta
        self.Pd = Pd
        self._cal_W()  # 计算初始窗口大小

    def _cal_W(self):
        from scipy.stats import beta
        """计算窗口大小W"""
        for n in range(1, 10000):
            k = max(int((self.alpha-self.epsilon) * n),1)  # 最多1%的成功样本
            upper_bound = beta.ppf(1 - self.delta, k + 1, n - k)
            if upper_bound < self.alpha:
                print(f"Required sample size: {n}, k = {k}, Upper Bound = {upper_bound:.4f}")
                break
        self.W = n
    
    def _generate_random_Pd(self, rng=42):
        np.random.seed(rng)  # 设置随机种子
        perturb = np.random.uniform(0.95, 1.05, self.Pd.shape)
        Pd_perturbed = self.Pd * perturb
        return Pd_perturbed

    def _solve_optimization(self, Pd):
        """求解优化问题并返回活动集"""
        # uc = UnitCommitmentModel(self.ppc, Pd, self.T_delta)
        uc = UnitCommitmentModelCVXPY(self.ppc, Pd, self.T_delta)
        pg_sol, x_sol, total_cost = uc.solve()
        # ed = EconomicDispatchGurobi(self.ppc, Pd, self.T_delta, x_sol)
        ed = EconomicDispatchCVXPY(self.ppc, Pd, self.T_delta, x_sol)
        pg_sol, total_cost = ed.solve()
        # 将x_sol转为(序号,值)的list
        x_sol_list = [((i, j), int(x_sol[i, j])) for i in range(x_sol.shape[0]) for j in range(x_sol.shape[1])]
        # 确定活动约束（仅考虑不等式约束）
        active = []
        active.extend(x_sol_list)
        for i in range(len(ed.constraints)):
            dual_val = ed.constraints[i].dual_value
            dual_val = np.atleast_1d(dual_val)  # Ensure it's always an array
            for j in range(ed.constraints[i].size):
                if abs(dual_val[j]) > 1e-6:
                    active.append(i)
        return frozenset(active)  # 用不可变类型存储活动集

    def save_active_sets_json(self, filename=None):
        """保存活动集为JSON格式，紧凑格式减少换行"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"active_sets_{timestamp}.json"
        
        # 创建result目录（如果不存在）
        result_dir = Path('result')
        result_dir.mkdir(exist_ok=True)
        
        # 将frozenset转换为list便于JSON序列化
        active_sets_list = [list(active_set) for active_set in self.observed_active_sets]
        
        # 计算活动集大小分布统计
        sizes = [len(active_set) for active_set in self.observed_active_sets]
        size_stats = {}
        if sizes:
            size_stats = {
                'min_size': min(sizes),
                'max_size': max(sizes),
                'avg_size': float(np.mean(sizes)),
                'std_size': float(np.std(sizes))
            }
        
        # 直接保存所有样本的Pd数据和对应活动集（一一对应）
        all_samples = []
        for i, (pd_data, active_set) in enumerate(self.samples):
            all_samples.append({
                'sample_id': i,
                'pd_data': pd_data.tolist(),
                'active_set': list(active_set)
            })
        
        data = {
            'metadata': {
                'total_active_sets': len(self.observed_active_sets),
                'total_samples': len(self.samples),
                'timestamp': timestamp,
                'size_statistics': size_stats
            },
            'parameters': {
                'alpha': self.alpha,
                'delta': self.delta,
                'epsilon': self.epsilon,
                'T_delta': self.T_delta,
                'W': self.W
            },
            'unique_active_sets': active_sets_list,
            'all_samples': all_samples
        }
        
        # 使用Path对象构建完整路径，使用紧凑格式（无缩进，紧凑分隔符）
        filepath = result_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
        
        print(f"Active sets和对应Pd数据已保存为JSON文件（紧凑格式）: {filepath}")
        return str(filepath)

    def save_active_sets_mapping_json(self, filename=None):
        """保存活动集映射关系为JSON格式（轻量级）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"active_sets_mapping_{timestamp}.json"
        
        # 创建result目录（如果不存在）
        result_dir = Path('result')
        result_dir.mkdir(exist_ok=True)
        
        # 生成映射关系
        mapping = {f"样本{i+1}": {"Pd": sample[0].tolist(), "活动集": list(sample[1])} for i, sample in enumerate(self.samples)}
        
        # 保存为JSON文件
        filepath = result_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"活动集映射关系已保存为JSON文件: {filepath}")
        return str(filepath)

    def run(self, max_samples=22000):
        """严格按照DiscoverMass算法伪代码实现"""
        # 初始化
        epsilon = self.epsilon
        alpha = self.alpha
        M = 1
        O = set()
        samples = []
        iter_count = 0
        # 计算理论最大M
        M_max = max_samples
        WM = self.W  # 初始窗口大小
        while True:
            # 计算窗口大小
            iter_count += 1
            print(f"迭代{iter_count}: 当前窗口WM={WM}, 当前M={M}")
            # 采样
            for idx in range(WM):
                if M >= M_max:
                    break
                Pd = self._generate_random_Pd(rng=idx)
                active_set = self._solve_optimization(Pd)
                samples.append((Pd, active_set))
                # 进度条显示
                bar_len = 30
                percent = (idx + 1) / WM
                filled_len = int(bar_len * percent)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\r  采样进度: |{bar}| {percent:.0%}", end='')
            print()
            # 计算发现率
            window_samples = samples[-WM:]
            new_active_sets = set()
            for _, active_set in window_samples:
                if active_set not in O:
                    new_active_sets.add(active_set)
            O.update(new_active_sets)
            RM_W = len(new_active_sets) / WM
            print(f"  发现率RM_W={RM_W:.4f}，目标发现率R={alpha - epsilon:.4f}，累计活动集数={len(O)}")
            # 检查停止条件
            if RM_W < alpha - epsilon or M >= M_max:
                print("  停止条件触发，算法终止。")
                break
            M = M + 1
        self.samples = samples
        self.observed_active_sets = O
        self.M = M
        return O

# 使用示例
if __name__ == "__main__":
    # ppc = get_case39_pypower()
    
    load_df = pd.read_csv('src/load.csv', header=None)
    Pd_base = load_df.values
    Pd_base = np.sum(Pd_base[:, ::4], axis=0)  # 假设每4个时段合并为一个
    
    ppc = pypower.case9.case9()
    
    Pd = ppc['bus'][:, pypower.idx_bus.PD]  # 假设Pd为bus数据中的Pd列
    Pd = Pd[:, None] * Pd_base[None, :] / np.max(Pd_base)  # 归一化负荷
    
    ppc['branch'][:, pypower.idx_brch.RATE_A] = ppc['branch'][:, pypower.idx_brch.RATE_A]
    
    learner = ActiveSetLearner(alpha=0.50, delta=0.05, epsilon=0.20, ppc=ppc, T_delta=1, Pd=Pd)
    active_sets = learner.run(max_samples=200)
    
    print(f"发现的活动集数量: {len(active_sets)}")
    print("示例活动集:", list(active_sets)[:3])

    # 保存完整数据（包含Pd数值）
    json_filename = learner.save_active_sets_json()
    print(f"完整JSON文件已保存: {json_filename}")
    
    # 保存轻量级映射关系
    # mapping_filename = learner.save_active_sets_mapping_json()
    # print(f"映射关系JSON文件已保存: {mapping_filename}")

    # 验证新样本
    Pd = learner._generate_random_Pd()
    test_active_set = learner._solve_optimization(Pd)
    
    # 使用集合预测（简单示例）
    if test_active_set in learner.observed_active_sets:
        print("成功预测活动集!")
    else:
        print("需要进一步学习的新活动集")
