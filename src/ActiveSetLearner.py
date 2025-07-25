import numpy as np
from collections import defaultdict
import cvxpy as cp
import pandas as pd
import pypower
import pypower.case14
import pypower.case9
import pypower.idx_bus

from pathlib import Path
import io
import sys
import re

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
from src.ed_cvxpy import EconomicDispatchCVXPY
from src.uc_cvxpy import UnitCommitmentModelCVXPY

class ActiveSetLearner:
    def __init__(self, alpha=0.05, delta=0.01, epsilon=0.04, gamma=2, ppc=None, T_delta=4, Pd=None):
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma
        self.c = 2 * gamma / epsilon**2
        self.observed_active_sets = set()
        self.samples = []
        self.M = 1
        self.M_base = 1 + (gamma / delta / (gamma - 1)) ** (1 / (gamma - 1))
        self.ppc = ppc
        self.T_delta = T_delta
        self.Pd = Pd
        self.W_min = 100 # 根据论文设置的初始窗口大小

    def _generate_random_Pd(self):
        perturb = np.random.uniform(0.95, 1.05, self.Pd.shape)
        Pd_perturbed = self.Pd * perturb
        return Pd_perturbed

    def _solve_optimization(self, Pd):
        """求解优化问题并返回活动集"""
        uc = UnitCommitmentModelCVXPY(self.ppc, Pd, self.T_delta)
        pg_sol, x_sol, total_cost = uc.solve()
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
                    active.append((i, j))
        return frozenset(active)  # 用不可变类型存储活动集

    def run(self, max_samples=22000):
        """严格按照DiscoverMass算法伪代码实现"""
        # 初始化
        gamma = self.gamma
        delta = self.delta
        epsilon = self.epsilon
        c = self.c
        alpha = self.alpha
        W_min = self.W_min
        M = 1
        O = set()
        samples = []
        iter_count = 0
        # 计算理论最大M
        M_max = max_samples
        while True:
            # 计算窗口大小
            WM = 130
            iter_count += 1
            print(f"迭代{iter_count}: 当前窗口WM={WM}, 当前M={M}")
            # 采样
            for idx in range(WM):
                if M >= M_max:
                    break
                Pd = self._generate_random_Pd()
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
    
    learner = ActiveSetLearner(ppc=ppc, T_delta=1, Pd=Pd)
    active_sets = learner.run(max_samples=200)
    
    print(f"发现的活动集数量: {len(active_sets)}")
    print("示例活动集:", list(active_sets)[:3])

    # 验证新样本
    Pd = learner._generate_random_Pd()
    test_active_set = learner._solve_optimization(Pd)
    
    # 使用集合预测（简单示例）
    if test_active_set in learner.observed_active_sets:
        print("成功预测活动集!")
    else:
        print("需要进一步学习的新活动集")
