import numpy as np
from collections import defaultdict
import pandas as pd
from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import pypower
import pypower.case39
import pypower.case30
import pypower.case14
import pypower.case9
import pypower.idx_bus
from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS
import json
from datetime import datetime
import time

from pathlib import Path
import io
import sys
import re
import contextlib

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
from src.case3_uc_data import get_case3_uc_ppc, get_case3lite_uc_ppc
from src.case14_uc_data import get_case14_uc_ppc
from src.case30_uc_data import get_case30_uc_ppc, get_case30lite_uc_ppc
from src.uc_gurobipy import UnitCommitmentModel
from src.ed_gurobipy import EconomicDispatchGurobi
from src.scenario_utils import normalize_sample_arrays


def extract_ed_dual_bundle(ppc, ed_model, T: int) -> dict:
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    bus = ppc_int['bus']
    branch = ppc_int['branch']
    nl = branch.shape[0]
    ng = gen.shape[0]
    nb = bus.shape[0]

    lambda_power_balance = np.zeros(T, dtype=float)
    lambda_dcpf_upper = np.zeros((nl, T), dtype=float)
    lambda_dcpf_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        constr = ed_model.model.getConstrByName(f'power_balance_{t}')
        lambda_power_balance[t] = float(constr.Pi) if constr is not None else 0.0
        for l in range(nl):
            cu = ed_model.model.getConstrByName(f'flow_upper_{l}_{t}')
            cl = ed_model.model.getConstrByName(f'flow_lower_{l}_{t}')
            lambda_dcpf_upper[l, t] = float(cu.Pi) if cu is not None else 0.0
            lambda_dcpf_lower[l, t] = max(0.0, -(float(cl.Pi) if cl is not None else 0.0))

    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0
    ptdf_g = makePTDF(ppc_int['baseMVA'], bus, branch) @ G
    lambda_pg_effective = np.zeros((ng, T), dtype=float)
    congestion = lambda_dcpf_upper - lambda_dcpf_lower
    for t in range(T):
        lambda_pg_effective[:, t] = (
            lambda_power_balance[t] - ptdf_g.T @ congestion[:, t]
        )

    return {
        'lambda_power_balance': lambda_power_balance.tolist(),
        'lambda_dcpf_upper': lambda_dcpf_upper.tolist(),
        'lambda_dcpf_lower': lambda_dcpf_lower.tolist(),
        'lambda_pg_effective': lambda_pg_effective.tolist(),
    }


class ActiveSetLearner:
    def __init__(self, alpha=0.05, delta=0.01, epsilon=0.04, ppc=None, T_delta=4, Pd=None,
                 case_name=None, renewable_data=None, verbose_solver=False,
                 load_perturbation_low=0.95, load_perturbation_high=1.05,
                 system_load_scale_low=1.0, system_load_scale_high=1.0,
                 temporal_wave_amplitude=0.0, temporal_wave_cycles_low=0.75,
                 temporal_wave_cycles_high=1.75):
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.observed_active_sets = set()
        self.samples = []
        self.M = 1
        if ppc is None and case_name == 'case3':
            ppc = get_case3_uc_ppc()
        if ppc is None and case_name == 'case3lite':
            ppc = get_case3lite_uc_ppc()
        if ppc is None and case_name == 'case14':
            ppc = get_case14_uc_ppc()
        if ppc is None and case_name == 'case30':
            ppc = get_case30_uc_ppc()
        if ppc is None and case_name == 'case30lite':
            ppc = get_case30lite_uc_ppc()
        self.ppc = ppc
        self.T_delta = T_delta
        self.Pd = Pd
        self.renewable_data = renewable_data
        self.case_name = case_name
        self.verbose_solver = verbose_solver
        self.load_perturbation_low = float(load_perturbation_low)
        self.load_perturbation_high = float(load_perturbation_high)
        self.system_load_scale_low = float(system_load_scale_low)
        self.system_load_scale_high = float(system_load_scale_high)
        self.temporal_wave_amplitude = float(temporal_wave_amplitude)
        self.temporal_wave_cycles_low = float(temporal_wave_cycles_low)
        self.temporal_wave_cycles_high = float(temporal_wave_cycles_high)
        self._cal_W()  # 计算初始窗口大小

    @staticmethod
    def _resolve_sample_goal(max_samples, target_samples):
        if target_samples is None:
            return max_samples, False
        if target_samples <= 0:
            raise ValueError("target_samples must be a positive integer")
        if max_samples is not None and target_samples > max_samples:
            raise ValueError("target_samples cannot exceed max_samples")
        return target_samples, True

    def _cal_W(self):
        from scipy.stats import beta
        """计算窗口大小W"""
        for n in range(1, 10000):
            k = max(int((self.alpha-self.epsilon) * n),1)  # 最多1%的成功样本
            upper_bound = beta.ppf(1 - self.delta, k + 1, n - k)
            if upper_bound < self.alpha:
                print(f"Required sample size: {n}, k = {k}, Upper Bound = {upper_bound:.4f}", flush=True)
                break
        self.W = n
    
    def _generate_random_Pd(self, rng=42):
        rng_obj = np.random.default_rng(int(rng))
        low = min(self.load_perturbation_low, self.load_perturbation_high)
        high = max(self.load_perturbation_low, self.load_perturbation_high)
        perturb = rng_obj.uniform(low, high, self.Pd.shape)

        scale_low = min(self.system_load_scale_low, self.system_load_scale_high)
        scale_high = max(self.system_load_scale_low, self.system_load_scale_high)
        system_scale = rng_obj.uniform(scale_low, scale_high)

        Pd_perturbed = np.asarray(self.Pd, dtype=float) * perturb * system_scale
        wave_amp = max(0.0, float(self.temporal_wave_amplitude))
        if wave_amp > 0.0 and Pd_perturbed.ndim == 2 and Pd_perturbed.shape[1] > 1:
            T = Pd_perturbed.shape[1]
            cycles_low = min(self.temporal_wave_cycles_low, self.temporal_wave_cycles_high)
            cycles_high = max(self.temporal_wave_cycles_low, self.temporal_wave_cycles_high)
            cycles = rng_obj.uniform(cycles_low, cycles_high)
            phase = rng_obj.uniform(0.0, 2.0 * np.pi)
            t = np.arange(T, dtype=float)
            wave = 1.0 + wave_amp * np.sin(2.0 * np.pi * cycles * t / max(T, 1) + phase)
            wave = np.maximum(wave, 0.05)
            wave = wave / max(float(np.mean(wave)), 1e-9)
            Pd_perturbed = Pd_perturbed * wave[None, :]
        return Pd_perturbed

    def _solve_optimization(self, Pd, renewable_data=None):
        """求解优化问题并返回活动集和对偶变量λ

        注意：求解MILP获取活动集（包含x），然后用x求解LP获取λ
        """
        # Step 1: 求解 UC (MILP) 获取机组状态 x
        uc = UnitCommitmentModel(
            self.ppc,
            Pd,
            self.T_delta,
            renewable_data=renewable_data,
            verbose=self.verbose_solver,
        )
        pg_sol, x_sol, total_cost = uc.solve()
        if x_sol is None:
            raise RuntimeError(f"UC solve failed with status={uc.model.status}")

        # Step 2: 用 x 求解 ED (LP) 获取对偶变量 λ
        ed = EconomicDispatchGurobi(
            self.ppc,
            Pd,
            self.T_delta,
            x_sol,
            renewable_data=renewable_data,
            verbose=self.verbose_solver,
        )
        pg_sol, total_cost = ed.solve()
        if pg_sol is None:
            raise RuntimeError(f"ED solve failed with status={ed.model.status}")

        # Step 3: 提取功率平衡约束的对偶变量 λ（前T个约束）
        T = Pd.shape[1]
        lambda_vals = extract_ed_dual_bundle(self.ppc, ed, T)

        # Step 4: 构建活动集（只用二进制变量 x，不含 LP 活跃约束索引）
        # 将x_sol转为[[g,t],value]的list（保持JSON格式一致）
        x_sol_list = [[[i, j], int(x_sol[i, j])] for i in range(x_sol.shape[0]) for j in range(x_sol.shape[1])]
        active = list(x_sol_list)

        # 转换为可哈希的frozenset（用于集合操作）
        def make_hashable(item):
            if isinstance(item, list):
                return tuple(tuple(x) if isinstance(x, list) else x for x in item)
            return item

        return frozenset(make_hashable(item) for item in active), lambda_vals

    def save_active_sets_json(self, filename=None):
        """保存活动集为JSON格式，紧凑格式减少换行"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if filename is None:
            case_tag = self.case_name if self.case_name else 'unknown'
            T = self.Pd.shape[1] if self.Pd is not None else 0
            n = len(self.samples)
            filename = f"active_sets_{case_tag}_T{T}_n{n}_{timestamp}.json"

        # 创建result/active_set目录（如果不存在）
        result_dir = Path('result') / 'active_set'
        result_dir.mkdir(parents=True, exist_ok=True)

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

        # 直接保存所有样本的Pd数据、活动集和对偶变量（一一对应）
        all_samples = []
        for i, sample in enumerate(self.samples):
            if isinstance(sample, dict):
                sample = normalize_sample_arrays(sample)
                pd_data = sample['pd_data']
                load_data = sample.get('load_data')
                renewable_data = sample.get('renewable_data')
                active_set = sample['active_set']
                lambda_vals = sample['lambda']
            else:
                pd_data, active_set, lambda_vals = sample
                load_data = pd_data
                renewable_data = None
            all_samples.append({
                'sample_id': i,
                'pd_data': pd_data.tolist(),
                'load_data': load_data.tolist() if load_data is not None else pd_data.tolist(),
                'active_set': list(active_set),
                'lambda': lambda_vals  # 新增：保存对偶变量λ
            })
            if renewable_data is not None:
                all_samples[-1]['renewable_data'] = renewable_data.tolist()

        data = {
            'metadata': {
                'case_name': self.case_name,
                'total_active_sets': len(self.observed_active_sets),
                'total_samples': len(self.samples),
                'T': self.Pd.shape[1] if self.Pd is not None else None,
                'timestamp': timestamp,
                'size_statistics': size_stats
            },
            'parameters': {
                'alpha': self.alpha,
                'delta': self.delta,
                'epsilon': self.epsilon,
                'T_delta': self.T_delta,
                'W': self.W,
                'load_sampling': {
                    'load_perturbation_low': self.load_perturbation_low,
                    'load_perturbation_high': self.load_perturbation_high,
                    'system_load_scale_low': self.system_load_scale_low,
                    'system_load_scale_high': self.system_load_scale_high,
                    'temporal_wave_amplitude': self.temporal_wave_amplitude,
                    'temporal_wave_cycles_low': self.temporal_wave_cycles_low,
                    'temporal_wave_cycles_high': self.temporal_wave_cycles_high,
                }
            },
            'unique_active_sets': active_sets_list,
            'all_samples': all_samples
        }
        
        # 使用Path对象构建完整路径，使用紧凑格式（无缩进，紧凑分隔符）
        filename_path = Path(filename)
        if filename_path.parent == Path('.'):
            filepath = result_dir / filename_path
        else:
            filepath = filename_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
        
        print(f"Active sets和对应Pd数据已保存为JSON文件（紧凑格式）: {filepath}", flush=True)
        return str(filepath)

    def save_active_sets_mapping_json(self, filename=None):
        """保存活动集映射关系为JSON格式（轻量级）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            case_tag = self.case_name if self.case_name else 'unknown'
            T = self.Pd.shape[1] if self.Pd is not None else 0
            n = len(self.samples)
            filename = f"active_sets_mapping_{case_tag}_T{T}_n{n}_{timestamp}.json"
        
        # 创建result/active_set目录（如果不存在）
        result_dir = Path('result') / 'active_set'
        result_dir.mkdir(parents=True, exist_ok=True)

        # 生成映射关系
        mapping = {}
        for i, sample in enumerate(self.samples):
            if isinstance(sample, dict):
                sample = normalize_sample_arrays(sample)
                mapping[f"样本{i+1}"] = {
                    "Pd": sample['pd_data'].tolist(),
                    "Load": sample['load_data'].tolist(),
                    "活动集": list(sample['active_set']),
                }
                if 'renewable_data' in sample:
                    mapping[f"鏍锋湰{i+1}"]["Renewable"] = sample['renewable_data'].tolist()
            else:
                mapping[f"样本{i+1}"] = {"Pd": sample[0].tolist(), "活动集": list(sample[1])}
        
        # 保存为JSON文件
        filename_path = Path(filename)
        if filename_path.parent == Path('.'):
            filepath = result_dir / filename_path
        else:
            filepath = filename_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"活动集映射关系已保存为JSON文件: {filepath}", flush=True)
        return str(filepath)

    def run(self, max_samples=22000, target_samples=None):
        """严格按照DiscoverMass算法伪代码实现"""
        # 初始化
        epsilon = self.epsilon
        alpha = self.alpha
        M = 1
        O = set()
        samples = []
        iter_count = 0
        # 计算理论最大M
        sample_goal, force_target_samples = self._resolve_sample_goal(
            max_samples,
            target_samples,
        )
        WM = self.W  # 初始窗口大小
        global_seed_counter = 0
        while True:
            # 计算窗口大小
            iter_count += 1
            print(f"迭代{iter_count}: 当前窗口WM={WM}, 当前M={M}", flush=True)
            # 采样
            actual_wm = min(WM, sample_goal - len(samples))
            if actual_wm <= 0:
                break
            for idx in range(actual_wm):
                seed = global_seed_counter + idx
                Pd = self._generate_random_Pd(rng=seed)
                renewable_data = None if self.renewable_data is None else np.asarray(self.renewable_data, dtype=float)
                try:
                    # 调试时允许显示 Gurobi 日志；默认静默避免打断进度条
                    if self.verbose_solver:
                        active_set, lambda_vals = self._solve_optimization(Pd, renewable_data=renewable_data)
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            active_set, lambda_vals = self._solve_optimization(Pd, renewable_data=renewable_data)
                    samples.append({
                        'sample_id': len(samples),
                        'load_data': Pd,
                        'pd_data': Pd,
                        'active_set': active_set,
                        'lambda': lambda_vals,
                    })
                    if renewable_data is not None:
                        samples[-1]['renewable_data'] = renewable_data
                except Exception as e:
                    # 在终端显式打印失败样本，便于排查不可行或求解异常
                    print(f"  样本 idx={idx} 求解失败: {e}", flush=True)
                    continue
                # 进度条显示
                bar_len = 30
                percent = (idx + 1) / actual_wm
                filled_len = int(bar_len * percent)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\r  采样进度: |{bar}| {percent:.0%}", end='', flush=True)
            global_seed_counter += actual_wm
            print(flush=True)
            # 计算发现率
            window_samples = samples[-actual_wm:]
            new_active_sets = set()
            for sample in window_samples:
                active_set = sample['active_set'] if isinstance(sample, dict) else sample[1]
                if active_set not in O:
                    new_active_sets.add(active_set)
            O.update(new_active_sets)
            RM_W = len(new_active_sets) / max(actual_wm, 1)
            print(f"  发现率RM_W={RM_W:.4f}，目标发现率R={alpha - epsilon:.4f}，累计活动集数={len(O)}", flush=True)
            # 检查停止条件
            if len(samples) >= sample_goal:
                print("  Reached target sample count, stopping.", flush=True)
                break
            if (not force_target_samples) and RM_W < alpha - epsilon:
                print("  停止条件触发，算法终止。", flush=True)
                break
            M = M + 1
        self.samples = samples
        self.observed_active_sets = O
        self.M = M
        return O

    def run_fixed_samples(self, num_samples: int):
        return self.run(target_samples=num_samples)

    def run_on_precomputed_scenarios(self, scenarios, max_samples=None):
        """对预先构造的场景列表逐个求解并收集 active sets。"""
        samples = []
        observed = set()
        limit = len(scenarios) if max_samples is None else min(max_samples, len(scenarios))
        t_start = time.time()
        for idx, sample in enumerate(scenarios[:limit]):
            sample = normalize_sample_arrays(dict(sample))
            if self.verbose_solver:
                active_set, lambda_vals = self._solve_optimization(
                    sample['load_data'],
                    renewable_data=sample['renewable_data'],
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    active_set, lambda_vals = self._solve_optimization(
                        sample['load_data'],
                        renewable_data=sample['renewable_data'],
                    )
            sample['sample_id'] = idx
            sample['active_set'] = active_set
            sample['lambda'] = lambda_vals
            samples.append(sample)
            observed.add(active_set)
            done_count = idx + 1
            bar_len = 30
            percent = done_count / max(limit, 1)
            filled_len = int(bar_len * percent)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            elapsed = time.time() - t_start
            eta = elapsed / done_count * (limit - done_count) if done_count > 0 else 0.0
            print(
                f"\r  场景求解进度: |{bar}| {done_count}/{limit} ({percent:.0%}) ETA: {eta:.0f}s",
                end='',
                flush=True,
            )

        if limit > 0:
            print(flush=True)

        self.samples = samples
        self.observed_active_sets = observed
        self.M = len(samples)
        return observed

# 使用示例
if __name__ == "__main__":
    from src.mti118_data_loader import (
        build_case118_daily_samples,
        load_case118_ppc_with_mti_limits,
    )

    ppc = load_case118_ppc_with_mti_limits()
    scenarios = build_case118_daily_samples(max_days=5)
    
    learner = ActiveSetLearner(
        alpha=0.70, delta=0.05, epsilon=0.10,
        ppc=ppc, T_delta=1, Pd=None, case_name='case118'
    )
    active_sets = learner.run_on_precomputed_scenarios(scenarios, max_samples=5)
    
    print(f"发现的活动集数量: {len(active_sets)}", flush=True)
    print("示例活动集:", list(active_sets)[:3], flush=True)

    # 保存完整数据（包含Pd数值）
    json_filename = learner.save_active_sets_json()
    print(f"完整JSON文件已保存: {json_filename}", flush=True)
    
    # 保存轻量级映射关系
    # mapping_filename = learner.save_active_sets_mapping_json()
    # print(f"映射关系JSON文件已保存: {mapping_filename}")

    # 验证新样本
    sample0 = scenarios[0]
    test_active_set, test_lambda = learner._solve_optimization(
        sample0['load_data'],
        renewable_data=sample0['renewable_data'],
    )
    
    # 使用集合预测（简单示例）
    if test_active_set in learner.observed_active_sets:
        print("成功预测活动集!", flush=True)
    else:
        print("需要进一步学习的新活动集", flush=True)
