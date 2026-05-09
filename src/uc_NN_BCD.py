"""
结合BCD迭代和神经网络更新的混合方法
- x和对偶变量采用BCD方法迭代（参考uc_dfsm_bcd.py）
- theta和zeta变量采用神经网络更新（参考uc_NN.py）
- 约束构建采用直接优化系数的形式（参考uc_NN.py）
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import io
import time
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    cp = None
    CVXPY_AVAILABLE = False

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将无法使用神经网络功能", flush=True)

# 导入必要的工具函数
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX
from pypower.idx_brch import RATE_A
from pypower.makePTDF import makePTDF
try:
    from scenario_utils import (
        get_feature_vector_from_sample,
        get_sample_net_load,
        normalize_sample_arrays,
    )
except ImportError:
    from src.scenario_utils import (
        get_feature_vector_from_sample,
        get_sample_net_load,
        normalize_sample_arrays,
    )

# 导入ED求解器
try:
    from ed_gurobipy import EconomicDispatchGurobi
    ED_GUROBI_AVAILABLE = True
except ImportError:
    try:
        from src.ed_gurobipy import EconomicDispatchGurobi
        ED_GUROBI_AVAILABLE = True
    except ImportError:
        ED_GUROBI_AVAILABLE = False
        print("警告: ed_gurobipy未安装，将无法使用ED问题求解功能", flush=True)

# 导入pypower用于测试
try:
    import pypower
    import pypower.case39
    import pypower.case14
    import pypower.case30
    PYPOWER_AVAILABLE = True
except ImportError:
    PYPOWER_AVAILABLE = False
    print("警告: pypower未安装，测试代码可能无法运行", flush=True)

try:
    from sparse_constraint_templates import template_library_to_bcd_union_constraints
except ImportError:
    from src.sparse_constraint_templates import template_library_to_bcd_union_constraints

try:
    from subproblem_lp_solver import (
        LP_BACKEND_GUROBI,
        LP_BACKEND_CVXPY_HIGHS,
        assert_lp_backend_available,
        normalize_lp_backend,
        _cvxpy_pi_to_gurobi_pi,
        _problem_is_optimal,
        _solve_with_cvxpy_highs,
        _sum_scalar_terms,
    )
except ImportError:
    from src.subproblem_lp_solver import (
        LP_BACKEND_GUROBI,
        LP_BACKEND_CVXPY_HIGHS,
        assert_lp_backend_available,
        normalize_lp_backend,
        _cvxpy_pi_to_gurobi_pi,
        _problem_is_optimal,
        _solve_with_cvxpy_highs,
        _sum_scalar_terms,
    )

# 设置输出缓冲
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

SUPPORTED_THETA_HOT_START_STRATEGIES = {"dcpf_relative", "gaussian"}
SUPPORTED_ZETA_HOT_START_STRATEGIES = {"zero", "gaussian"}
SUPPORTED_LAMBDA_INIT_STRATEGIES = {"lp_relaxation", "ed_on_x_opt"}
SUPPORTED_NN_BATCH_STRATEGIES = {"full-batch", "mini-batch"}


def normalize_theta_hot_start_strategy(strategy: str | None) -> str:
    strategy_norm = "dcpf_relative" if strategy is None else str(strategy).strip().lower()
    if strategy_norm not in SUPPORTED_THETA_HOT_START_STRATEGIES:
        raise ValueError(
            f"Unsupported theta_hot_start_strategy: {strategy}. "
            f"Supported: {sorted(SUPPORTED_THETA_HOT_START_STRATEGIES)}"
        )
    return strategy_norm


def normalize_zeta_hot_start_strategy(strategy: str | None) -> str:
    strategy_norm = "zero" if strategy is None else str(strategy).strip().lower()
    if strategy_norm not in SUPPORTED_ZETA_HOT_START_STRATEGIES:
        raise ValueError(
            f"Unsupported zeta_hot_start_strategy: {strategy}. "
            f"Supported: {sorted(SUPPORTED_ZETA_HOT_START_STRATEGIES)}"
        )
    return strategy_norm


def normalize_lambda_init_strategy(strategy: str | None) -> str:
    strategy_norm = "lp_relaxation" if strategy is None else str(strategy).strip().lower()
    if strategy_norm not in SUPPORTED_LAMBDA_INIT_STRATEGIES:
        raise ValueError(
            f"Unsupported lambda_init_strategy: {strategy}. "
            f"Supported: {sorted(SUPPORTED_LAMBDA_INIT_STRATEGIES)}"
        )
    return strategy_norm


def normalize_nn_batch_strategy(strategy: str | None) -> str:
    strategy_norm = "full-batch" if strategy is None else str(strategy).strip().lower()
    if strategy_norm not in SUPPORTED_NN_BATCH_STRATEGIES:
        raise ValueError(
            f"Unsupported nn_batch_strategy: {strategy}. "
            f"Supported: {sorted(SUPPORTED_NN_BATCH_STRATEGIES)}"
        )
    return strategy_norm


def normalize_nn_hidden_dims(hidden_dims: List[int] | Tuple[int, ...] | None, default_hidden_dims: List[int]) -> List[int]:
    dims = default_hidden_dims if hidden_dims is None else hidden_dims
    normalized = [int(dim) for dim in dims]
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError(f"nn_hidden_dims must be a non-empty sequence of positive integers, got {hidden_dims}")
    return normalized


def build_mlp_with_dropout(input_dim: int, hidden_dims: List[int], output_dim: int, dropout_p: float = 0.1) -> 'nn.Sequential':
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Dropout(dropout_p))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

# ========================== ActiveSetReader 和 load_active_set_from_json ==========================
# 从uc_NN.py复制，保持文件独立性

class ActiveSetReader:
    """读取和解析活动集JSON文件的工具类"""
    
    def __init__(self, json_filepath: str):
        """
        初始化活动集读取器
        
        Args:
            json_filepath: JSON文件路径
        """
        self.json_filepath = Path(json_filepath)
        self.data = self._load_json()
        
    def _load_json(self) -> Dict:
        """加载JSON文件"""
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON文件未找到: {self.json_filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"JSON文件格式错误: {self.json_filepath}")
    
    def get_sample_data(self, sample_id: int) -> Optional[Dict]:
        """
        获取指定样本的完整数据
        
        Args:
            sample_id: 样本ID
            
        Returns:
            包含样本数据的字典
        """
        samples = self.data.get('all_samples', [])
        if 0 <= sample_id < len(samples):
            return samples[sample_id]
        else:
            print(f"样本ID {sample_id} 超出范围 [0, {len(samples)-1}]", flush=True)
            return None
    
    def get_total_samples_count(self) -> int:
        """
        获取总样本数量
        
        Returns:
            总样本数量
        """
        samples = self.data.get('all_samples', [])
        return len(samples)
    
    def load_all_samples(self) -> List[Dict]:
        """
        加载所有样本的活动集数据
        
        Returns:
            包含所有样本数据的列表
        """
        all_samples_data = []
        total_samples = self.get_total_samples_count()
        raw_samples = self.data.get('all_samples', [])
        has_dataset_renewable = any(
            'renewable_data' in sample and np.any(np.abs(np.asarray(sample['renewable_data'], dtype=float)) > 1e-9)
            for sample in raw_samples
        )
        print(f"[ActiveSet] Loading {total_samples} samples...", flush=True)
        
        print(f"开始加载 {total_samples} 个样本的数据...", flush=True)
        
        for sample_id in range(total_samples):
            try:
                sample = self.get_sample_data(sample_id)
                if sample is None:
                    raise ValueError(f"样本 {sample_id} 不存在")
                if not has_dataset_renewable:
                    sample = dict(sample)
                    sample.pop('renewable_data', None)
                sample = normalize_sample_arrays(dict(sample))

                active_constraints, active_variables, pd_data = self.extract_active_constraints_and_variables(sample_id)
                unit_commitment = self.get_unit_commitment_matrix(sample_id)
                
                sample_data = {
                    'sample_id': sample_id,
                    'active_constraints': active_constraints,
                    'active_variables': active_variables,
                    'pd_data': pd_data,
                    'load_data': np.array(sample.get('load_data', pd_data), dtype=float),
                    'unit_commitment_matrix': unit_commitment
                }
                if has_dataset_renewable and 'renewable_data' in sample:
                    sample_data['renewable_data'] = np.array(sample['renewable_data'], dtype=float)
                
                # 读取对偶变量（如果存在）
                if sample and 'lambda' in sample:
                    sample_data['lambda'] = sample['lambda']
                
                all_samples_data.append(sample_data)
                
                if (sample_id + 1) % 10 == 0:
                    print(f"[ActiveSet] loaded_samples={sample_id + 1}/{total_samples}", flush=True)
                    print(f"已加载 {sample_id + 1}/{total_samples} 个样本", flush=True)
                    
            except Exception as e:
                print(f"加载样本 {sample_id} 时出错: {e}", flush=True)
                # 添加空数据以保持索引一致性
                all_samples_data.append({
                    'sample_id': sample_id,
                    'active_constraints': [],
                    'active_variables': [],
                    'pd_data': np.array([]),
                    'load_data': np.array([]),
                    'renewable_data': np.array([]),
                    'unit_commitment_matrix': np.array([]),
                    'error': str(e)
                })
        
        print(f"✓ 完成加载所有样本数据", flush=True)
        print("[ActiveSet] Finished loading all samples", flush=True)
        return all_samples_data
    
    def extract_active_constraints_and_variables(self, sample_id: int) -> Tuple[List, List, np.ndarray]:
        """
        提取指定样本的起作用约束、变量和对应的Pd数据
        
        Args:
            sample_id: 样本ID
            
        Returns:
            tuple: (active_constraints, active_variables, pd_data)
        """
        sample = self.get_sample_data(sample_id)
        if sample is None:
            return [], [], np.array([])
        
        active_set = sample['active_set']
        sample = normalize_sample_arrays(dict(sample))
        pd_data = sample['pd_data']
        
        # 分离约束和变量
        active_constraints = []  # 起作用的约束
        active_variables = []    # 活动变量（主要是二进制变量）
        
        for item in active_set:
            if isinstance(item, list) and len(item) == 2:
                if isinstance(item[0], list) and len(item[0]) == 2:
                    # 二进制变量格式 [[unit_id, time_slot], value]
                    active_variables.append({
                        'type': 'binary_variable',
                        'unit_id': item[0][0],
                        'time_slot': item[0][1],
                        'value': item[1],
                        'variable_name': f'x[{item[0][0]},{item[0][1]}]'
                    })
                else:
                    # 约束格式 [constraint_id, dual_value]
                    active_constraints.append({
                        'type': 'constraint',
                        'constraint_id': item[0],
                        'dual_value': item[1] if len(item) > 1 else None,
                        'constraint_name': f'constraint_{item[0]}'
                    })
            else:
                # 其他格式的约束
                active_constraints.append({
                    'type': 'constraint',
                    'constraint_id': item,
                    'dual_value': None,
                    'constraint_name': f'constraint_{item}'
                })
        
        return active_constraints, active_variables, pd_data
    
    def get_unit_commitment_matrix(self, sample_id: int) -> np.ndarray:
        """
        获取机组启停状态矩阵
        
        Args:
            sample_id: 样本ID
            
        Returns:
            机组启停状态矩阵 (ng, T)
        """
        _, active_variables, pd_data = self.extract_active_constraints_and_variables(sample_id)
        
        if not active_variables:
            return np.array([])
        
        # 确定矩阵大小
        max_unit = max([var['unit_id'] for var in active_variables if var['type'] == 'binary_variable']) + 1
        max_time = max([var['time_slot'] for var in active_variables if var['type'] == 'binary_variable']) + 1
        
        # 初始化矩阵
        unit_commitment = np.zeros((max_unit, max_time), dtype=int)
        
        # 填充矩阵
        for var in active_variables:
            if var['type'] == 'binary_variable':
                unit_commitment[var['unit_id'], var['time_slot']] = var['value']
        
        return unit_commitment


def load_active_set_from_json(json_filepath: str, sample_id: Optional[int] = None):
    """
    从JSON文件加载活动集数据
    
    Args:
        json_filepath: JSON文件路径
        sample_id: 要加载的样本ID，如果为None则加载所有样本
        
    Returns:
        包含活动约束、变量和Pd数据的字典
        - 当sample_id不为None时：返回单个样本的数据
        - 当sample_id为None时：返回所有样本的数据列表
    """
    reader = ActiveSetReader(json_filepath)
    
    if sample_id is not None:
        # 加载单个样本
        active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
        unit_commitment = reader.get_unit_commitment_matrix(sample_id)
        print(f"[ActiveSet] Loaded sample {sample_id}", flush=True)
        print(f"[ActiveSet] active_constraints={len(active_constraints)}", flush=True)
        print(f"[ActiveSet] active_variables={len(active_variables)}", flush=True)
        print(f"[ActiveSet] pd_shape={pd_data.shape}", flush=True)
        
        print(f"=== 加载活动集数据 (样本 {sample_id}) ===", flush=True)
        print(f"活动约束数量: {len(active_constraints)}", flush=True)
        print(f"活动变量数量: {len(active_variables)}", flush=True)
        print(f"Pd数据形状: {pd_data.shape}", flush=True)

        sample_data = {
            'sample_id': sample_id,
            'active_constraints': active_constraints,
            'active_variables': active_variables,
            'pd_data': pd_data,
            'unit_commitment_matrix': unit_commitment,
            'single_sample': True
        }
        
        # 读取对偶变量（如果存在）
        sample = reader.get_sample_data(sample_id)
        if sample and 'lambda' in sample:
            sample_data['lambda'] = sample['lambda']
        
        return sample_data
    else:
        # 加载所有样本
        all_samples_data = reader.load_all_samples()
        print("=== Loaded all active-set samples ===", flush=True)
        print(f"[ActiveSet] total_samples={len(all_samples_data)}", flush=True)
        
        print(f"=== 加载所有活动集数据 ===", flush=True)
        print(f"总样本数量: {len(all_samples_data)}", flush=True)
        
        return all_samples_data


def _get_uc_matrix_from_sample(sample: dict, ng: int, T: int) -> Optional[np.ndarray]:
    """从样本数据中提取 unit_commitment 矩阵 (ng, T)。
    优先读 unit_commitment_matrix，若无则从 active_set 解析，返回 np.ndarray(ng, T) 或 None。
    """
    uc = sample.get('unit_commitment_matrix', None)
    if isinstance(uc, np.ndarray) and uc.ndim == 2 and uc.shape == (ng, T):
        return uc
    # 从 active_set 解析
    if 'active_set' in sample:
        x_sol = np.zeros((ng, T))
        for item in sample['active_set']:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                if 0 <= g < ng and 0 <= t < T:
                    x_sol[g, t] = item[1]
        return x_sol
    return None


def _get_custom_generator_array_from_ppc(ppc_raw, ng: int, key: str) -> Optional[np.ndarray]:
    """Read optional per-generator arrays from raw ppc and align them to ext2int generator order."""
    if not isinstance(ppc_raw, dict):
        return None

    values = ppc_raw.get(key)
    if values is None:
        return None

    values = np.asarray(values, dtype=float)
    if values.shape[0] != ng:
        return None

    raw_gen = np.asarray(ppc_raw.get('gen'))
    if raw_gen.shape[0] != ng:
        return values

    order = np.argsort(raw_gen[:, GEN_BUS], kind='stable')
    return values[order]


def _get_ramp_limits_from_ppc(ppc_raw, gen: np.ndarray, T_delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-generator ramp limits, using optional raw ppc metadata when available."""
    default_up = 0.4 * gen[:, PMAX] / T_delta
    default_down = 0.4 * gen[:, PMAX] / T_delta
    default_up_co = 0.3 * gen[:, PMAX]
    default_down_co = 0.3 * gen[:, PMAX]

    ramp_up_h = _get_custom_generator_array_from_ppc(
        ppc_raw, gen.shape[0], 'uc_ramp_up_mw_per_h')
    ramp_down_h = _get_custom_generator_array_from_ppc(
        ppc_raw, gen.shape[0], 'uc_ramp_down_mw_per_h')

    if ramp_up_h is None or ramp_down_h is None:
        return default_up, default_down, default_up_co, default_down_co

    Ru = np.asarray(ramp_up_h, dtype=float) * T_delta
    Rd = np.asarray(ramp_down_h, dtype=float) * T_delta
    Ru = np.maximum(Ru, default_up)
    Rd = np.maximum(Rd, default_down)
    Ru_co = np.maximum(Ru, gen[:, PMIN])
    Rd_co = np.maximum(Rd, gen[:, PMIN])
    return Ru, Rd, Ru_co, Rd_co


def _get_min_up_down_steps_from_ppc(ppc_raw, ng: int, T: int, T_delta: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-generator min up/down steps, aligned with the UC MIP wrapper."""
    default_steps = max(int(4 * T_delta), 1)
    min_up_h = _get_custom_generator_array_from_ppc(ppc_raw, ng, 'uc_min_up_time_h')
    min_down_h = _get_custom_generator_array_from_ppc(ppc_raw, ng, 'uc_min_down_time_h')
    if min_up_h is None or min_down_h is None:
        min_up = np.full(ng, default_steps, dtype=int)
        min_down = np.full(ng, default_steps, dtype=int)
    else:
        min_up = np.maximum(np.ceil(np.asarray(min_up_h, dtype=float) / T_delta).astype(int), 1)
        min_down = np.maximum(np.ceil(np.asarray(min_down_h, dtype=float) / T_delta).astype(int), 1)
    min_up = np.minimum(min_up, max(int(T), 1))
    min_down = np.minimum(min_down, max(int(T), 1))
    return min_up, min_down, int(np.max(min_up)), int(np.max(min_down))


class Agent_NN_BCD:
    """
    结合BCD迭代和神经网络更新的混合方法
    
    - x和对偶变量：使用BCD方法迭代（iter_with_pg_block, iter_with_dual_block）
    - theta和zeta：使用神经网络更新（类似uc_NN.py）
    - 约束构建：直接优化系数（类似uc_NN.py）
    """
    
    def __init__(
        self,
        ppc,
        active_set_data,
        T_delta,
        union_analysis=None,
        external_sparse_templates=None,
        lambda_init_strategy: str = "lp_relaxation",
        max_theta_constraints_per_time_slot: int = 10,
        theta_curriculum_rounds: int = 0,
        theta_initial_scale: float = 1.0,
        theta_final_scale: float = 1.0,
        theta_curriculum_delay_rounds: int = 0,
        theta_hot_start_strategy: str = "dcpf_relative",
        zeta_hot_start_strategy: str = "zero",
        theta_gaussian_std: float = 0.01,
        zeta_gaussian_std: float = 0.01,
        unit_predictor=None,
        use_unit_predictor: bool = False,
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
        dual_sign_relax_interval: int | None = None,
        lp_backend: str = LP_BACKEND_GUROBI,
        gurobi_threads: int | None = None,
        gurobi_lp_method: int = -1,
        bcd_highs_threads: int = 1,
        nn_hidden_dims: List[int] | None = None,
        nn_learning_rate: float = 5e-5,
        nn_batch_strategy: str = "full-batch",
        nn_batch_size: int = 4,
        nn_shuffle: bool = True,
        loss_ratio_primal: float = 1.0,
        loss_ratio_dual_x: float = 1.0,
        loss_ratio_opt: float = 1.0,
        loss_ratio_reg: float = 1.0,
        nn_smooth_abs_eps: float = 1e-6,
        direct_train_config: dict | None = None,
        iter_delta_reg_weight: float = 1e-4,
        iter_delta_reg_deadband: float = 0.05,
        pg_block_prox_weight: float = 2e-2,
        dual_block_prox_weight: float = 1e-2,
    ):
        self.ppc = ppc
        self.ppc_raw = ppc
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        self.Ru, self.Rd, self.Ru_co, self.Rd_co = _get_ramp_limits_from_ppc(
            self.ppc_raw, self.gen, self.T_delta
        )
        
        self.iter_number = 0
        
        self.penalty_factor = 1e7
        
        # 对偶变量的下界约束
        self.dual_para_bound = 0.1  # mu和ita的最小值
        self.dual_para_bound_quit_iteration = 50
        self.mu_dual_floor_init = float(mu_dual_floor_init)
        self.ita_dual_floor_init = float(ita_dual_floor_init)
        self.zeta_ita_cap_penalty_weight = max(float(zeta_ita_cap_penalty_weight or 0.0), 0.0)
        self.zeta_ita_cap_initial_weight = (
            None if zeta_ita_cap_initial_weight is None else max(float(zeta_ita_cap_initial_weight), 0.0)
        )
        self.zeta_ita_cap_final_weight = (
            None if zeta_ita_cap_final_weight is None else max(float(zeta_ita_cap_final_weight), 0.0)
        )
        self.zeta_ita_cap_initial = None if zeta_ita_cap_initial is None else max(float(zeta_ita_cap_initial), 0.0)
        self.zeta_ita_cap_final = None if zeta_ita_cap_final is None else max(float(zeta_ita_cap_final), 0.0)
        self.zeta_ita_cap_start_round = max(int(zeta_ita_cap_start_round), 0)
        self.zeta_ita_cap_end_round = max(int(zeta_ita_cap_end_round), self.zeta_ita_cap_start_round)
        if dual_sign_relax_interval is None:
            self.dual_sign_relax_interval = 2
        else:
            self.dual_sign_relax_interval = max(int(dual_sign_relax_interval), 0)
        self.dual_para_bound = self.mu_dual_floor_init
        
        # BCD迭代参数
        self.rho_primal = float(rho_primal_init)
        self.rho_binary = float(rho_binary_init)
        self.rho_opt = float(rho_opt_init)
        self.gamma_base = float(gamma_base)   # gamma 缩放基准
        self.gamma_dual_component_scale = 3.0
        self.rho_max = 10.0     # rho 上限
        self.theta_reg_weight = 1e-4   # theta L1 正则化权重（[-1, 1] 死区外生效）
        self.zeta_reg_weight = 1e-4    # zeta L1 正则化权重（[-1, 1] 死区外生效）
        self.gurobi_mip_gap = 1e-4
        self.gurobi_feasibility_tol = 1e-5
        self.gurobi_optimality_tol = 1e-5
        self.gurobi_int_feas_tol = 1e-5
        self.max_theta_constraints_per_time_slot = int(max_theta_constraints_per_time_slot)
        self.theta_curriculum_rounds = max(int(theta_curriculum_rounds), 0)
        self.theta_initial_scale = max(float(theta_initial_scale), 0.0)
        self.theta_final_scale = max(float(theta_final_scale), 0.0)
        self.theta_curriculum_delay_rounds = max(int(theta_curriculum_delay_rounds), 0)
        
        # 约束违反惩罚项的权重和epsilon参数
        self.constraint_violation_weight = 0
        self.constraint_violation_epsilon = 1e-3
        
        # 是否启用theta/zeta约束
        self.enable_theta_constraints = True
        self.enable_zeta_constraints = True
        self.use_per_variable_zeta_constraints = True
        
        # 是否使用Fischer-Burmeister函数处理互补松弛条件
        self.use_fischer_burmeister_for_loss = False
        
        # 处理单个样本或多个样本的情况
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
            
        self.ng = self.gen.shape[0]
        self.nl = self.branch.shape[0]
        (
            self.min_up_steps,
            self.min_down_steps,
            self.Ton,
            self.Toff,
        ) = _get_min_up_down_steps_from_ppc(self.ppc_raw, self.ng, self.T, self.T_delta)
        self._generator_incidence_matrix = self._build_generator_incidence_matrix()
        self._ptdf_matrix = makePTDF(self.baseMVA, self.bus, self.branch)
        self._branch_limit = np.asarray(self.branch[:, RATE_A], dtype=float)
        self._ptdf_g = self._ptdf_matrix @ self._generator_incidence_matrix
        self.theta_constraint_direction_signs = np.ones((self.nl, self.T), dtype=float)
        self.zeta_constraint_direction_signs = np.ones((self.ng, self.T), dtype=float)
        self._theta_limited_union_analysis_cache = {}
        self._last_reported_theta_stage_signature = None
        self._lp_backend = normalize_lp_backend(lp_backend)
        self.gurobi_threads = None if gurobi_threads is None else max(1, int(gurobi_threads))
        self.gurobi_lp_method = int(gurobi_lp_method)
        self.bcd_highs_threads = max(1, int(bcd_highs_threads))
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            assert_lp_backend_available(self._lp_backend)

        self.active_set_data = active_set_data
        self.external_sparse_templates = external_sparse_templates
        self.lambda_init_strategy = normalize_lambda_init_strategy(lambda_init_strategy)
        self.theta_hot_start_strategy = normalize_theta_hot_start_strategy(theta_hot_start_strategy)
        self.zeta_hot_start_strategy = normalize_zeta_hot_start_strategy(zeta_hot_start_strategy)
        self.theta_gaussian_std = float(theta_gaussian_std)
        self.zeta_gaussian_std = float(zeta_gaussian_std)
        self.unit_predictor = unit_predictor
        self.use_unit_predictor = bool(use_unit_predictor)
        self.unit_predictor_warmup_rounds = max(int(unit_predictor_warmup_rounds), 0)
        self.unit_predictor_finetune_lr = max(float(unit_predictor_finetune_lr), 1e-10)
        self.unit_predictor_weight_decay = max(float(unit_predictor_weight_decay), 0.0)
        self.theta_constraint_delay_rounds = max(int(theta_constraint_delay_rounds), 0)
        self._unit_predictor_optimizer = None
        self._unit_predictor_optimizer_lr = None
        self._theta_delay_reported_round = None
        self.enable_dropout_during_nn_training = bool(enable_dropout_during_nn_training)
        self.nn_hidden_dims = normalize_nn_hidden_dims(nn_hidden_dims, [64, 128])
        self.nn_learning_rate = float(nn_learning_rate)
        if self.nn_learning_rate <= 0:
            raise ValueError(f"nn_learning_rate must be positive, got {nn_learning_rate}")
        self.nn_batch_strategy = normalize_nn_batch_strategy(nn_batch_strategy)
        self.nn_batch_size = max(1, int(nn_batch_size))
        self.nn_shuffle = bool(nn_shuffle)
        self.loss_ratio_primal = float(loss_ratio_primal)
        self.loss_ratio_dual_x = float(loss_ratio_dual_x)
        self.loss_ratio_opt = float(loss_ratio_opt)
        self.loss_ratio_reg = float(loss_ratio_reg)
        self.nn_smooth_abs_eps = max(float(nn_smooth_abs_eps), 0.0)
        self.direct_train_config = dict(direct_train_config or {})
        self.iter_delta_reg_weight = float(iter_delta_reg_weight)
        self.iter_delta_reg_deadband = max(float(iter_delta_reg_deadband), 0.0)
        self.pg_block_prox_weight = max(float(pg_block_prox_weight), 0.0)
        self.dual_block_prox_weight = max(float(dual_block_prox_weight), 0.0)
        rho_dual_pg_base = rho_dual_init if rho_dual_pg_init is None else rho_dual_pg_init
        rho_dual_x_base = rho_dual_init if rho_dual_x_init is None else rho_dual_x_init
        rho_dual_coc_base = rho_dual_init if rho_dual_coc_init is None else rho_dual_coc_init
        self.rho_dual_pg = float(rho_dual_pg_base)
        self.rho_dual_x = float(rho_dual_x_base)
        self.rho_dual_coc = float(rho_dual_coc_base)
        self._sync_rho_dual_summary()
        
        # 初始化theta和zeta变量字典
        self.theta_vars = {}
        self.zeta_vars = {}

        # 迭代间输出差异正则：上一代 theta/zeta（per-sample）缓存
        self._prev_theta_values_list = None
        self._prev_zeta_values_list = None
        
        # 初始化求解（获得初始x和lambda）
        self.pg, self.x, self.x_opt, self.coc, self.cpower, self.lambda_ = self.initialize_solve()
        
        # 如果没有提供union_analysis，则基于x_init创建
        if union_analysis is None:
            self._current_union_analysis = self._create_union_analysis_from_x_init(self.x, self.lambda_)
        else:
            self._current_union_analysis = union_analysis
        
        # 创建theta和zeta变量
        self.add_theta_variables_for_branches(self._current_union_analysis)
        self.add_zeta_variables_for_units(self._current_union_analysis)
        
        # 初始化theta和zeta值（直接优化系数，使用随机初始化）
        self.theta_values_list, self.mu = self.initialize_theta_values(self._current_union_analysis)
        self.zeta_values_list, self.ita = self.initialize_zeta_values(self._current_union_analysis)
        # 兼容别名（供 _init_neural_network 取 var_names、feasibility_pump 等）
        self.theta_values = self.theta_values_list[0]
        self.zeta_values = self.zeta_values_list[0]
        
        # 初始化神经网络模型（用于更新theta和zeta）
        if TORCH_AVAILABLE:
            self._init_neural_network()
            self._generate_initial_values_from_nn()
        else:
            self.theta_net = None
            self.zeta_net = None
            self.device = None
            # 回退：用随机初始化填充占位值
        self._apply_hot_start_initial_values(self._current_union_analysis)
        self._apply_unit_predictor_initial_values()

        # ----- Persistent Gurobi model cache -----
        # 每个样本只建一次模型，后续迭代通过 setObjective / chgCoeff 更新
        self._pg_models: dict = {}          # sample_id -> gp.Model
        self._pg_vars: dict = {}            # sample_id -> vars_dict
        self._pg_model_union_id: dict = {}  # sample_id -> id(union_analysis)

        self._dual_models: dict = {}          # sample_id -> gp.Model
        self._dual_vars: dict = {}            # sample_id -> vars_dict
        self._dual_model_state: dict = {}     # sample_id -> (floor_active, sign_relax_round)
        self._dual_model_union_id: dict = {}  # sample_id -> id(union_analysis)

    def _sync_rho_dual_summary(self) -> None:
        self.rho_dual = float(np.mean([
            self.rho_dual_pg,
            self.rho_dual_x,
            self.rho_dual_coc,
        ]))

    def _min_up_horizon(self, g: int) -> int:
        return int(self.min_up_steps[g])

    def _min_down_horizon(self, g: int) -> int:
        return int(self.min_down_steps[g])

    def _zero_unused_min_up_entries(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float).copy()
        for g in range(self.ng):
            arr[g, self._min_up_horizon(g):, :] = 0.0
        return arr

    def _zero_unused_min_down_entries(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float).copy()
        for g in range(self.ng):
            arr[g, self._min_down_horizon(g):, :] = 0.0
        return arr

    def _extract_min_up_gurobi_values(self, vars_dict, width: int) -> np.ndarray:
        values = np.zeros((self.ng, int(width), self.T), dtype=float)
        for g in range(self.ng):
            for tau in range(min(int(width), self._min_up_horizon(g))):
                for t in range(self.T):
                    values[g, tau, t] = float(vars_dict[g, tau, t].X)
        return values

    def _extract_min_down_gurobi_values(self, vars_dict, width: int) -> np.ndarray:
        values = np.zeros((self.ng, int(width), self.T), dtype=float)
        for g in range(self.ng):
            for tau in range(min(int(width), self._min_down_horizon(g))):
                for t in range(self.T):
                    values[g, tau, t] = float(vars_dict[g, tau, t].X)
        return values

    def _smooth_abs(self, tensor: torch.Tensor, eps: float | None = None) -> torch.Tensor:
        resolved_eps = self.nn_smooth_abs_eps if eps is None else max(float(eps), 0.0)
        if resolved_eps <= 0:
            return torch.abs(tensor)
        return torch.sqrt(tensor * tensor + resolved_eps) - np.sqrt(resolved_eps)

    def _smooth_relu(self, tensor: torch.Tensor, eps: float | None = None) -> torch.Tensor:
        resolved_eps = self.nn_smooth_abs_eps if eps is None else max(float(eps), 0.0)
        if resolved_eps <= 0:
            return torch.relu(tensor)
        sqrt_eps = np.sqrt(resolved_eps)
        smoothed = 0.5 * (tensor + torch.sqrt(tensor * tensor + resolved_eps) - sqrt_eps)
        return torch.clamp_min(smoothed, 0.0)

    def _smooth_deadband_excess(
        self,
        tensor: torch.Tensor,
        deadband: float,
        eps: float | None = None,
    ) -> torch.Tensor:
        return self._smooth_relu(
            self._smooth_abs(tensor, self.nn_smooth_abs_eps) - float(deadband),
            eps=eps,
        )

    def _get_regularization_scale(self) -> float:
        min_rho = min(
            self.rho_primal,
            self.rho_dual_pg,
            self.rho_dual_x,
            self.rho_dual_coc,
            self.rho_opt,
        )
        return float(np.sqrt(max(min_rho, 0.0)))

    def _add_squared_distance_term(self, expr, reference_value: float, scale: float):
        reference_value = float(reference_value)
        scale = max(float(scale), 1e-6)
        diff = (expr - reference_value) / scale
        return diff * diff

    def _build_pg_block_prox_obj(self, model, sample_id: int, pg, x, coc):
        prox_obj = gp.LinExpr()
        if self.pg_block_prox_weight <= 0:
            return prox_obj

        prev_pg = self.pg[sample_id]
        prev_x = self.x[sample_id]
        prev_coc = self.coc[sample_id]
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]

        for g in range(self.ng):
            pg_scale = max(float(self.gen[g, PMAX]), 1.0)
            coc_scale = max(float(start_cost[g]), float(shut_cost[g]), 1.0)
            for t in range(self.T):
                prox_obj += self._add_squared_distance_term(
                    pg[g, t],
                    prev_pg[g, t],
                    pg_scale,
                )
                prox_obj += self._add_squared_distance_term(
                    x[g, t],
                    prev_x[g, t],
                    1.0,
                )
            for t in range(self.T - 1):
                prox_obj += self._add_squared_distance_term(
                    coc[g, t],
                    prev_coc[g, t],
                    coc_scale,
                )

        return prox_obj

    def _build_pg_block_prox_expr_cvxpy(self, sample_id: int, pg, x, coc):
        if self.pg_block_prox_weight <= 0:
            return 0.0

        prev_pg = np.asarray(self.pg[sample_id], dtype=float)
        prev_x = np.asarray(self.x[sample_id], dtype=float)
        prev_coc = np.asarray(self.coc[sample_id], dtype=float)
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]

        terms = []
        for g in range(self.ng):
            pg_scale = max(float(self.gen[g, PMAX]), 1.0)
            coc_scale = max(float(start_cost[g]), float(shut_cost[g]), 1.0)
            for t in range(self.T):
                terms.append(cp.square((pg[g, t] - prev_pg[g, t]) / pg_scale))
                terms.append(cp.square(x[g, t] - prev_x[g, t]))
            for t in range(self.T - 1):
                terms.append(cp.square((coc[g, t] - prev_coc[g, t]) / coc_scale))

        return _sum_scalar_terms(terms)

    def _build_dual_block_prox_obj(
        self,
        model,
        sample_id: int,
        lambda_power_balance,
        lambda_pg_lower,
        lambda_pg_upper,
        lambda_ramp_up,
        lambda_ramp_down,
        lambda_min_on,
        lambda_min_off,
        lambda_start_cost,
        lambda_shut_cost,
        lambda_coc_nonneg,
        lambda_dcpf_upper,
        lambda_dcpf_lower,
        lambda_x_upper,
        lambda_x_lower,
        mu,
        ita,
        Ton: int,
        Toff: int,
    ):
        prox_obj = gp.LinExpr()
        if self.dual_block_prox_weight <= 0:
            return prox_obj

        prev_lambda = self.lambda_[sample_id] if self.lambda_[sample_id] is not None else self._create_empty_lambda_dict()
        prev_mu = self.mu[sample_id]
        prev_ita = self.ita[sample_id]

        for t in range(self.T):
            prev_val = float(prev_lambda['lambda_power_balance'][t])
            prox_obj += self._add_squared_distance_term(
                lambda_power_balance[t],
                prev_val,
                max(1.0, abs(prev_val)),
            )

        for g in range(self.ng):
            for t in range(self.T):
                for var_name, var_container in (
                    ('lambda_pg_lower', lambda_pg_lower),
                    ('lambda_pg_upper', lambda_pg_upper),
                    ('lambda_x_upper', lambda_x_upper),
                    ('lambda_x_lower', lambda_x_lower),
                ):
                    prev_val = float(prev_lambda[var_name][g, t])
                    prox_obj += self._add_squared_distance_term(
                        var_container[g, t],
                        prev_val,
                        max(1.0, abs(prev_val)),
                    )
                prev_ita_val = float(prev_ita[g, t])
                prox_obj += self._add_squared_distance_term(
                    ita[g, t],
                    prev_ita_val,
                    max(1.0, abs(prev_ita_val)),
                )

            for t in range(self.T - 1):
                for var_name, var_container in (
                    ('lambda_ramp_up', lambda_ramp_up),
                    ('lambda_ramp_down', lambda_ramp_down),
                    ('lambda_start_cost', lambda_start_cost),
                    ('lambda_shut_cost', lambda_shut_cost),
                    ('lambda_coc_nonneg', lambda_coc_nonneg),
                ):
                    prev_val = float(prev_lambda[var_name][g, t])
                    prox_obj += self._add_squared_distance_term(
                        var_container[g, t],
                        prev_val,
                        max(1.0, abs(prev_val)),
                    )

            for tau in range(self._min_up_horizon(g)):
                for t in range(self.T):
                    prev_val = float(prev_lambda['lambda_min_on'][g, tau, t])
                    prox_obj += self._add_squared_distance_term(
                        lambda_min_on[g, tau, t],
                        prev_val,
                        max(1.0, abs(prev_val)),
                    )
            for tau in range(self._min_down_horizon(g)):
                for t in range(self.T):
                    prev_val = float(prev_lambda['lambda_min_off'][g, tau, t])
                    prox_obj += self._add_squared_distance_term(
                        lambda_min_off[g, tau, t],
                        prev_val,
                        max(1.0, abs(prev_val)),
                    )

        for l in range(self.nl):
            for t in range(self.T):
                for var_name, var_container in (
                    ('lambda_dcpf_upper', lambda_dcpf_upper),
                    ('lambda_dcpf_lower', lambda_dcpf_lower),
                ):
                    prev_val = float(prev_lambda[var_name][l, t])
                    prox_obj += self._add_squared_distance_term(
                        var_container[l, t],
                        prev_val,
                        max(1.0, abs(prev_val)),
                    )
                prev_mu_val = float(prev_mu[l, t])
                prox_obj += self._add_squared_distance_term(
                    mu[l, t],
                    prev_mu_val,
                    max(1.0, abs(prev_mu_val)),
                )

        return prox_obj

    def _solve_lp_with_fixed_x(self, Pd: np.ndarray, x_vals: np.ndarray):
        """以固定 x 矩阵构建纯 LP，求解后返回连续变量解和对偶变量字典。

        Args:
            Pd: 负荷矩阵 (nb, T)
            x_vals: 固定的机组启停矩阵 (ng, T)，取值 0 或 1

        Returns:
            (pg_sol, coc_sol, cpower_sol, lambda_dict) 或 (None, None, None, None) 若失败
        """
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            try:
                return self._solve_lp_with_fixed_x_cvxpy_highs(Pd, x_vals)
            except Exception as e:
                print(f"[ActiveSet] sample {sample_id} failed to load: {e}", flush=True)
                print(f"❌ _solve_lp_with_fixed_x cvxpy_highs 失败: {e}", flush=True)
                return None, None, None, None
        try:
            lp = gp.Model('lp_fixed_x')
            self._apply_fast_gurobi_tolerances(lp, mip=False)

            pg = lp.addVars(self.ng, self.T, lb=0, name='pg')
            coc = lp.addVars(self.ng, self.T - 1, lb=0, name='coc')
            cpower = lp.addVars(self.ng, self.T, lb=0, name='cpower')

            # 功率平衡约束
            for t in range(self.T):
                lp.addConstr(
                    gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]),
                    name=f'power_balance_{t}'
                )

            # 发电上下限约束（x 为常数）
            for g in range(self.ng):
                for t in range(self.T):
                    xgt = float(x_vals[g, t])
                    lp.addConstr(pg[g, t] >= self.gen[g, PMIN] * xgt, name=f'pg_lower_{g}_{t}')
                    lp.addConstr(pg[g, t] <= self.gen[g, PMAX] * xgt, name=f'pg_upper_{g}_{t}')

            # 爬坡约束（x 为常数）
            for g in range(self.ng):
                for t in range(1, self.T):
                    rhs_up = self.Ru[g] * float(x_vals[g, t - 1]) + self.Ru_co[g] * (1 - float(x_vals[g, t - 1]))
                    rhs_dn = self.Rd[g] * float(x_vals[g, t]) + self.Rd_co[g] * (1 - float(x_vals[g, t]))
                    lp.addConstr(pg[g, t] - pg[g, t - 1] <= rhs_up, name=f'ramp_up_{g}_{t}')
                    lp.addConstr(pg[g, t - 1] - pg[g, t] <= rhs_dn, name=f'ramp_down_{g}_{t}')

            # 启停成本约束（x 为常数）
            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    xg_t = float(x_vals[g, t])
                    xg_t1 = float(x_vals[g, t - 1])
                    lp.addConstr(coc[g, t - 1] >= start_cost[g] * (xg_t - xg_t1), name=f'start_cost_{g}_{t}')
                    lp.addConstr(coc[g, t - 1] >= shut_cost[g] * (xg_t1 - xg_t), name=f'shut_cost_{g}_{t}')
                    lp.addConstr(coc[g, t - 1] >= 0, name=f'coc_nonneg_{g}_{t}')

            # 发电成本约束（x 为常数）
            for t in range(self.T):
                for g in range(self.ng):
                    xgt = float(x_vals[g, t])
                    lp.addConstr(
                        cpower[g, t] >= self.gencost[g, -2] / self.T_delta * pg[g, t] +
                        self.gencost[g, -1] / self.T_delta * xgt,
                        name=f'cpower_{g}_{t}'
                    )

            # 目标函数
            obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                   gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T - 1)))
            lp.setObjective(obj, GRB.MINIMIZE)
            lp.optimize()

            if lp.status == GRB.OPTIMAL:
                pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
                coc_sol = np.array([[coc[g, t].X for t in range(self.T - 1)] for g in range(self.ng)])
                cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
                lambda_dict = self.extract_dual_variables_as_arrays(lp)
                return pg_sol, coc_sol, cpower_sol, lambda_dict
            else:
                return None, None, None, None
        except Exception as e:
            print(f"❌ _solve_lp_with_fixed_x 失败: {e}", flush=True)
            return None, None, None, None

    def _solve_lp_relaxation(self, Pd: np.ndarray):
        """求解 UC 的 LP 松弛（x ∈ [0,1]），用于获取连续初始点和对偶变量。

        Args:
            Pd: 负荷矩阵 (nb, T)

        Returns:
            (x_lp, pg_sol, coc_sol, cpower_sol, lambda_dict) 或全 None（若失败）
        """
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            try:
                return self._solve_lp_relaxation_cvxpy_highs(Pd)
            except Exception as e:
                print(f"❌ _solve_lp_relaxation cvxpy_highs 失败: {e}", flush=True)
                return None, None, None, None, None
        try:
            lp = gp.Model('lp_relaxation')
            self._apply_fast_gurobi_tolerances(lp, mip=False)

            pg = lp.addVars(self.ng, self.T, lb=0, name='pg')
            x = lp.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
            coc = lp.addVars(self.ng, self.T - 1, lb=0, name='coc')
            cpower = lp.addVars(self.ng, self.T, lb=0, name='cpower')

            # 功率平衡约束
            for t in range(self.T):
                lp.addConstr(
                    gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]),
                    name=f'power_balance_{t}'
                )

            # 发电上下限约束
            for g in range(self.ng):
                for t in range(self.T):
                    lp.addConstr(pg[g, t] >= self.gen[g, PMIN] * x[g, t], name=f'pg_lower_{g}_{t}')
                    lp.addConstr(pg[g, t] <= self.gen[g, PMAX] * x[g, t], name=f'pg_upper_{g}_{t}')

            # 爬坡约束
            for g in range(self.ng):
                for t in range(1, self.T):
                    lp.addConstr(pg[g, t] - pg[g, t - 1] <= self.Ru[g] * x[g, t - 1] + self.Ru_co[g] * (1 - x[g, t - 1]),
                                 name=f'ramp_up_{g}_{t}')
                    lp.addConstr(pg[g, t - 1] - pg[g, t] <= self.Rd[g] * x[g, t] + self.Rd_co[g] * (1 - x[g, t]),
                                 name=f'ramp_down_{g}_{t}')

            # 最小开关机时间约束（松弛版本）
            Ton = self.Ton
            Toff = self.Toff
            for g in range(self.ng):
                for t in range(1, self._min_up_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        lp.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + t], name=f'min_on_{g}_{t}_{t1}')
            for g in range(self.ng):
                for t in range(1, self._min_down_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        lp.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + t], name=f'min_off_{g}_{t}_{t1}')

            # 启停成本约束
            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    lp.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]), name=f'start_cost_{g}_{t}')
                    lp.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]), name=f'shut_cost_{g}_{t}')
                    lp.addConstr(coc[g, t - 1] >= 0, name=f'coc_nonneg_{g}_{t}')

            # 发电成本约束
            for t in range(self.T):
                for g in range(self.ng):
                    lp.addConstr(
                        cpower[g, t] >= self.gencost[g, -2] / self.T_delta * pg[g, t] +
                        self.gencost[g, -1] / self.T_delta * x[g, t],
                        name=f'cpower_{g}_{t}'
                    )

            # 目标函数
            obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                   gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T - 1)))
            lp.setObjective(obj, GRB.MINIMIZE)
            lp.optimize()

            if lp.status == GRB.OPTIMAL:
                x_lp = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
                pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
                coc_sol = np.array([[coc[g, t].X for t in range(self.T - 1)] for g in range(self.ng)])
                cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
                lambda_dict = self.extract_dual_variables_as_arrays(lp)
                return x_lp, pg_sol, coc_sol, cpower_sol, lambda_dict
            else:
                print(f"警告: LP 松弛求解失败，状态: {lp.status}", flush=True)
                return None, None, None, None, None
        except Exception as e:
            print(f"❌ _solve_lp_relaxation 失败: {e}", flush=True)
            return None, None, None, None, None

    def _normalize_lambda_from_data(self, raw_lambda) -> Optional[dict]:
        """将 JSON 中的 lambda 数据规范化为 numpy 数组格式。

        Args:
            raw_lambda: active_set_data[sample_id].get('lambda', None)

        Returns:
            规范化的 lambda 字典（与 _create_empty_lambda_dict 格式相同），
            或 None（raw_lambda 为 None 或键不完整）。
        """
        if raw_lambda is None:
            return None

        required_keys = [
            'lambda_power_balance', 'lambda_pg_lower', 'lambda_pg_upper',
            'lambda_ramp_up', 'lambda_ramp_down',
        ]
        # 检查关键字段是否存在
        for key in required_keys:
            if key not in raw_lambda:
                return None

        try:
            Ton = self.Ton
            Toff = self.Toff
            result = {}

            def _to_array(v, shape):
                arr = np.array(v, dtype=np.float64)
                if arr.shape == shape:
                    return arr
                return None

            result['lambda_power_balance'] = _to_array(raw_lambda['lambda_power_balance'], (self.T,))
            result['lambda_pg_lower'] = _to_array(raw_lambda['lambda_pg_lower'], (self.ng, self.T))
            result['lambda_pg_upper'] = _to_array(raw_lambda['lambda_pg_upper'], (self.ng, self.T))
            result['lambda_ramp_up'] = _to_array(raw_lambda['lambda_ramp_up'], (self.ng, self.T - 1))
            result['lambda_ramp_down'] = _to_array(raw_lambda['lambda_ramp_down'], (self.ng, self.T - 1))

            # 检查必需字段是否形状正确
            for key in required_keys:
                if result[key] is None:
                    return None

            # 可选字段，缺失则用零填充
            def _opt(key, shape):
                if key in raw_lambda:
                    arr = _to_array(raw_lambda[key], shape)
                    return arr if arr is not None else np.zeros(shape)
                return np.zeros(shape)

            result['lambda_min_on'] = _opt('lambda_min_on', (self.ng, Ton, self.T))
            result['lambda_min_off'] = _opt('lambda_min_off', (self.ng, Toff, self.T))
            result['lambda_start_cost'] = _opt('lambda_start_cost', (self.ng, self.T - 1))
            result['lambda_shut_cost'] = _opt('lambda_shut_cost', (self.ng, self.T - 1))
            result['lambda_coc_nonneg'] = _opt('lambda_coc_nonneg', (self.ng, self.T - 1))
            result['lambda_cpower'] = _opt('lambda_cpower', (self.ng, self.T))
            result['lambda_dcpf_upper'] = _opt('lambda_dcpf_upper', (self.nl, self.T))
            result['lambda_dcpf_lower'] = _opt('lambda_dcpf_lower', (self.nl, self.T))
            result['lambda_x_upper'] = _opt('lambda_x_upper', (self.ng, self.T))
            result['lambda_x_lower'] = _opt('lambda_x_lower', (self.ng, self.T))
            return result
        except Exception as e:
            print(f"警告: _normalize_lambda_from_data 转换失败: {e}", flush=True)
            return None

    def _zero_x_bound_lambda_terms(self, lambda_dict: Optional[dict]) -> dict:
        if lambda_dict is None:
            lambda_dict = self._create_empty_lambda_dict()
        lambda_clean = {
            key: np.array(val, copy=True) if isinstance(val, np.ndarray) else val
            for key, val in lambda_dict.items()
        }
        lambda_clean['lambda_x_upper'] = np.zeros((self.ng, self.T))
        lambda_clean['lambda_x_lower'] = np.zeros((self.ng, self.T))
        return lambda_clean

    def _build_initial_lambda(
        self,
        Pd: np.ndarray,
        x_opt: np.ndarray,
        lp_lambda: Optional[dict],
        raw_lambda,
    ) -> dict:
        if self.lambda_init_strategy == 'lp_relaxation':
            if lp_lambda is not None:
                return lp_lambda
            lambda_data = self._normalize_lambda_from_data(raw_lambda)
            return lambda_data if lambda_data is not None else self._create_empty_lambda_dict()

        if self.lambda_init_strategy == 'ed_on_x_opt':
            _, _, _, ed_lambda = self._solve_lp_with_fixed_x(Pd, x_opt)
            if ed_lambda is not None:
                return self._zero_x_bound_lambda_terms(ed_lambda)

            if lp_lambda is not None:
                return self._zero_x_bound_lambda_terms(lp_lambda)

            lambda_data = self._normalize_lambda_from_data(raw_lambda)
            if lambda_data is not None:
                return self._zero_x_bound_lambda_terms(lambda_data)

            return self._create_empty_lambda_dict()

        raise ValueError(f"Unsupported lambda_init_strategy: {self.lambda_init_strategy}")

    def _solve_milp_for_x_opt(self, Pd: np.ndarray) -> Optional[np.ndarray]:
        """求解 MILP 获取真值 x（仅返回 x，不返回对偶变量）。

        Args:
            Pd: 负荷矩阵 (nb, T)

        Returns:
            x_milp 的 numpy 数组 (ng, T)，或 None（求解失败）。
        """
        try:
            model = gp.Model('milp_x_opt')
            self._apply_fast_gurobi_tolerances(model, mip=True)

            pg = model.addVars(self.ng, self.T, lb=0, name='pg')
            x = model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
            coc = model.addVars(self.ng, self.T - 1, lb=0, name='coc')
            cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')

            for t in range(self.T):
                model.addConstr(
                    gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]),
                    name=f'power_balance_{t}'
                )
            for g in range(self.ng):
                for t in range(self.T):
                    model.addConstr(pg[g, t] >= self.gen[g, PMIN] * x[g, t], name=f'pg_lower_{g}_{t}')
                    model.addConstr(pg[g, t] <= self.gen[g, PMAX] * x[g, t], name=f'pg_upper_{g}_{t}')

            for g in range(self.ng):
                for t in range(1, self.T):
                    model.addConstr(pg[g, t] - pg[g, t - 1] <= self.Ru[g] * x[g, t - 1] + self.Ru_co[g] * (1 - x[g, t - 1]),
                                    name=f'ramp_up_{g}_{t}')
                    model.addConstr(pg[g, t - 1] - pg[g, t] <= self.Rd[g] * x[g, t] + self.Rd_co[g] * (1 - x[g, t]),
                                    name=f'ramp_down_{g}_{t}')

            Ton = self.Ton
            Toff = self.Toff
            for g in range(self.ng):
                for t in range(1, self._min_up_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + t], name=f'min_on_{g}_{t}_{t1}')
            for g in range(self.ng):
                for t in range(1, self._min_down_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + t], name=f'min_off_{g}_{t}_{t1}')

            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    model.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]), name=f'start_cost_{g}_{t}')
                    model.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]), name=f'shut_cost_{g}_{t}')
                    model.addConstr(coc[g, t - 1] >= 0, name=f'coc_nonneg_{g}_{t}')

            for t in range(self.T):
                for g in range(self.ng):
                    model.addConstr(cpower[g, t] >= self.gencost[g, -2] / self.T_delta * pg[g, t] +
                                    self.gencost[g, -1] / self.T_delta * x[g, t],
                                    name=f'cpower_{g}_{t}')

            primal_obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                          gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T - 1)))
            model.setObjective(primal_obj, GRB.MINIMIZE)
            model.setParam("Presolve", 2)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                return np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            else:
                print(f"警告: _solve_milp_for_x_opt 求解失败，状态: {model.status}", flush=True)
                return None
        except Exception as e:
            print(f"❌ _solve_milp_for_x_opt 失败: {e}", flush=True)
            return None

    def initialize_solve(self):
        """初始化求解：self.x 取 LP 松弛解，self.x_opt 取 MILP/数据集真值。

        流程（每个样本）：
        1. 求 LP 松弛 → x_lp（self.x 初值）+ 对偶变量（备用）
        2. 尝试从数据中规范化对偶变量；若无则用 LP 松弛对偶
        3. 获取真值 x_opt：优先数据集，否则求 MILP，最终回退到 x_lp

        Returns:
            (pg_sol, x_sol, x_opt_sol, coc_sol, cpower_sol, lambda_sol)
        """
        pg_sol = []
        x_sol = []
        x_opt_sol = []
        coc_sol = []
        cpower_sol = []
        lambda_sol = []

        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']

            # A. 求 LP 松弛 → x_lp 及候选对偶
            x_lp, pg_s, coc_s, cpower_s, lp_lambda = self._solve_lp_relaxation(Pd)

            if x_lp is None:
                print(f"[BCD:init] sample {sample_id} LP relaxation failed; using zero fallback", flush=True)
                print(f"[BCD:init] sample {sample_id} LP relaxation failed; using zero fallback", flush=True)
                pg_sol.append(np.zeros((self.ng, self.T)))
                x_sol.append(np.zeros((self.ng, self.T)))
                x_opt_sol.append(np.zeros((self.ng, self.T), dtype=np.float64))
                coc_sol.append(np.zeros((self.ng, self.T - 1)))
                cpower_sol.append(np.zeros((self.ng, self.T)))
                lambda_sol.append(self._create_empty_lambda_dict())
                continue

            # B. 对偶变量：优先使用数据中已有的，否则用 LP 松弛结果
            raw_lambda = self.active_set_data[sample_id].get('lambda', None)
            lambda_s = self._normalize_lambda_from_data(raw_lambda)
            if lambda_s is None:
                lambda_s = lp_lambda if lp_lambda is not None else self._create_empty_lambda_dict()

            # C. 获取真值 x_opt：优先数据集，否则求 MILP
            x_opt = _get_uc_matrix_from_sample(self.active_set_data[sample_id], self.ng, self.T)
            if x_opt is None:
                x_milp = self._solve_milp_for_x_opt(Pd)
                x_opt = x_milp if x_milp is not None else x_lp.copy()

            # D. 存储：self.x ← x_lp（连续），self.x_opt ← x_opt（真值）
            pg_sol.append(pg_s)
            x_sol.append(x_lp)
            x_opt_sol.append(x_opt.astype(np.float64))
            coc_sol.append(coc_s)
            cpower_sol.append(cpower_s)
            lambda_sol.append(lambda_s)

        pg_sol = np.array(pg_sol)
        x_sol = np.array(x_sol)
        x_opt_sol = np.array(x_opt_sol)
        coc_sol = np.array(coc_sol)
        cpower_sol = np.array(cpower_sol)

        return pg_sol, x_sol, x_opt_sol, coc_sol, cpower_sol, lambda_sol

    def initialize_solve(self):
        """Initialize LP-relaxation primal values, x-opt, and initial lambda values."""
        pg_sol = []
        x_sol = []
        x_opt_sol = []
        coc_sol = []
        cpower_sol = []
        lambda_sol = []
        print(
            f"[BCD:init] Starting initial LP/MILP solves for {self.n_samples} samples "
            f"(backend={self._lp_backend}, gurobi_threads={self.gurobi_threads})",
            flush=True,
        )

        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']

            x_lp, pg_s, coc_s, cpower_s, lp_lambda = self._solve_lp_relaxation(Pd)

            if x_lp is None:
                print(f"[BCD:init] sample {sample_id} LP relaxation failed; using zero fallback", flush=True)
                pg_sol.append(np.zeros((self.ng, self.T)))
                x_sol.append(np.zeros((self.ng, self.T)))
                x_opt_sol.append(np.zeros((self.ng, self.T), dtype=np.float64))
                coc_sol.append(np.zeros((self.ng, self.T - 1)))
                cpower_sol.append(np.zeros((self.ng, self.T)))
                lambda_sol.append(self._create_empty_lambda_dict())
                continue

            x_opt = _get_uc_matrix_from_sample(self.active_set_data[sample_id], self.ng, self.T)
            if x_opt is None:
                x_milp = self._solve_milp_for_x_opt(Pd)
                x_opt = x_milp if x_milp is not None else x_lp.copy()

            raw_lambda = self.active_set_data[sample_id].get('lambda', None)
            lambda_s = self._build_initial_lambda(Pd, x_opt, lp_lambda, raw_lambda)

            pg_sol.append(pg_s)
            x_sol.append(x_lp)
            x_opt_sol.append(x_opt.astype(np.float64))
            coc_sol.append(coc_s)
            cpower_sol.append(cpower_s)
            lambda_sol.append(lambda_s)
            if (sample_id + 1) % 25 == 0 or sample_id == self.n_samples - 1:
                print(
                    f"[BCD:init] processed_samples={sample_id + 1}/{self.n_samples}",
                    flush=True,
                )

        pg_sol = np.array(pg_sol)
        x_sol = np.array(x_sol)
        x_opt_sol = np.array(x_opt_sol)
        coc_sol = np.array(coc_sol)
        cpower_sol = np.array(cpower_sol)
        print("[BCD:init] Initialization complete", flush=True)

        return pg_sol, x_sol, x_opt_sol, coc_sol, cpower_sol, lambda_sol
    
    def extract_dual_variables(self, model):
        """
        通过约束名称提取对偶变量（完全复制自uc_NN.py）
        
        Args:
            model: 求解后的Gurobi模型
            
        Returns:
            Dict: 包含对偶变量的字典
        """
        implicit_duals = {}
        
        try:
            # 1. 功率平衡约束的对偶变量
            implicit_duals['power_balance'] = {}
            for t in range(self.T):
                constr = model.getConstrByName(f'power_balance_{t}')
                if constr is not None:
                    try:
                        implicit_duals['power_balance'][t] = constr.Pi
                    except (AttributeError, gp.GurobiError):
                        implicit_duals['power_balance'][t] = 0.0
            
            # 2. 发电上下限约束的对偶变量
            implicit_duals['pg_lower'] = {}
            implicit_duals['pg_upper'] = {}
            for g in range(self.ng):
                implicit_duals['pg_lower'][g] = {}
                implicit_duals['pg_upper'][g] = {}
                for t in range(self.T):
                    # 下限约束
                    constr_lower = model.getConstrByName(f'pg_lower_{g}_{t}')
                    if constr_lower is not None:
                        try:
                            implicit_duals['pg_lower'][g][t] = constr_lower.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['pg_lower'][g][t] = 0.0
                    
                    # 上限约束
                    constr_upper = model.getConstrByName(f'pg_upper_{g}_{t}')
                    if constr_upper is not None:
                        try:
                            implicit_duals['pg_upper'][g][t] = constr_upper.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['pg_upper'][g][t] = 0.0
            
            # 3. 爬坡约束的对偶变量
            implicit_duals['ramp_up'] = {}
            implicit_duals['ramp_down'] = {}
            for g in range(self.ng):
                implicit_duals['ramp_up'][g] = {}
                implicit_duals['ramp_down'][g] = {}
                for t in range(1, self.T):
                    # 上爬坡约束
                    constr_ramp_up = model.getConstrByName(f'ramp_up_{g}_{t}')
                    if constr_ramp_up is not None:
                        try:
                            implicit_duals['ramp_up'][g][t-1] = constr_ramp_up.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['ramp_up'][g][t-1] = 0.0
                    
                    # 下爬坡约束
                    constr_ramp_down = model.getConstrByName(f'ramp_down_{g}_{t}')
                    if constr_ramp_down is not None:
                        try:
                            implicit_duals['ramp_down'][g][t-1] = constr_ramp_down.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['ramp_down'][g][t-1] = 0.0

            # 4. 最小开机/关机时间约束的对偶变量
            implicit_duals['min_on'] = {}
            implicit_duals['min_off'] = {}
            
            Ton = self.Ton
            Toff = self.Toff
            
            for g in range(self.ng):
                implicit_duals['min_on'][g] = {}
                implicit_duals['min_off'][g] = {}
                
                # 最小开机时间约束
                for tau in range(1, self._min_up_horizon(g) + 1):
                    for t1 in range(self.T - tau):
                        cname_on = f'min_on_{g}_{tau}_{t1}'
                        constr_on = model.getConstrByName(cname_on)
                        if constr_on is not None:
                            if tau not in implicit_duals['min_on'][g]:
                                implicit_duals['min_on'][g][tau] = {}
                            try:
                                implicit_duals['min_on'][g][tau][t1] = constr_on.Pi
                            except (AttributeError, gp.GurobiError):
                                implicit_duals['min_on'][g][tau][t1] = 0.0
                
                # 最小关机时间约束
                for tau in range(1, self._min_down_horizon(g) + 1):
                    for t1 in range(self.T - tau):
                        cname_off = f'min_off_{g}_{tau}_{t1}'
                        constr_off = model.getConstrByName(cname_off)
                        if constr_off is not None:
                            if tau not in implicit_duals['min_off'][g]:
                                implicit_duals['min_off'][g][tau] = {}
                            try:
                                implicit_duals['min_off'][g][tau][t1] = constr_off.Pi
                            except (AttributeError, gp.GurobiError):
                                implicit_duals['min_off'][g][tau][t1] = 0.0
            
            # 5. 启停成本约束的对偶变量
            implicit_duals['start_cost'] = {}
            implicit_duals['shut_cost'] = {}
            implicit_duals['coc_nonneg'] = {}
            
            for g in range(self.ng):
                implicit_duals['start_cost'][g] = {}
                implicit_duals['shut_cost'][g] = {}
                implicit_duals['coc_nonneg'][g] = {}
                
                for t in range(1, self.T):
                    # 启动成本约束
                    constr_start = model.getConstrByName(f'start_cost_{g}_{t}')
                    if constr_start is not None:
                        try:
                            implicit_duals['start_cost'][g][t-1] = constr_start.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['start_cost'][g][t-1] = 0.0
                    
                    # 关机成本约束
                    constr_shut = model.getConstrByName(f'shut_cost_{g}_{t}')
                    if constr_shut is not None:
                        try:
                            implicit_duals['shut_cost'][g][t-1] = constr_shut.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['shut_cost'][g][t-1] = 0.0
                    
                    # 非负约束
                    constr_nonneg = model.getConstrByName(f'coc_nonneg_{g}_{t}')
                    if constr_nonneg is not None:
                        try:
                            implicit_duals['coc_nonneg'][g][t-1] = constr_nonneg.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['coc_nonneg'][g][t-1] = 0.0
            
            # 6. 发电成本约束的对偶变量
            implicit_duals['cpower'] = {}
            
            for g in range(self.ng):
                implicit_duals['cpower'][g] = {}
                for t in range(self.T):
                    constr_cpower = model.getConstrByName(f'cpower_{g}_{t}')
                    if constr_cpower is not None:
                        try:
                            implicit_duals['cpower'][g][t] = constr_cpower.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['cpower'][g][t] = 0.0
            
            # 7. DCPF潮流约束的对偶变量
            implicit_duals['dcpf_upper'] = {}
            implicit_duals['dcpf_lower'] = {}
            
            for l in range(self.nl):
                implicit_duals['dcpf_upper'][l] = {}
                implicit_duals['dcpf_lower'][l] = {}
                for t in range(self.T):
                    # 潮流上限约束
                    constr_flow_upper = model.getConstrByName(f'flow_upper_{l}_{t}')
                    if constr_flow_upper is not None:
                        try:
                            implicit_duals['dcpf_upper'][l][t] = constr_flow_upper.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['dcpf_upper'][l][t] = 0.0
                    
                    # 潮流下限约束
                    constr_flow_lower = model.getConstrByName(f'flow_lower_{l}_{t}')
                    if constr_flow_lower is not None:
                        try:
                            implicit_duals['dcpf_lower'][l][t] = constr_flow_lower.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['dcpf_lower'][l][t] = 0.0
            
            # 8. x变量上下界约束的对偶变量
            implicit_duals['x_upper'] = {}
            implicit_duals['x_lower'] = {}
            
            for g in range(self.ng):
                implicit_duals['x_upper'][g] = {}
                implicit_duals['x_lower'][g] = {}
                for t in range(self.T):
                    # x上界约束
                    constr_x_upper = model.getConstrByName(f'x_upper_{g}_{t}')
                    if constr_x_upper is not None:
                        try:
                            implicit_duals['x_upper'][g][t] = constr_x_upper.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['x_upper'][g][t] = 0.0
                    
                    # x下界约束
                    constr_x_lower = model.getConstrByName(f'x_lower_{g}_{t}')
                    if constr_x_lower is not None:
                        try:
                            implicit_duals['x_lower'][g][t] = constr_x_lower.Pi
                        except (AttributeError, gp.GurobiError):
                            implicit_duals['x_lower'][g][t] = 0.0

            return implicit_duals

        except Exception as e:
            print(f"❌ 对偶变量提取过程中出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_dual_variables_as_arrays(self, model):
        """
        提取对偶变量并转换为numpy数组格式（完全复制自uc_NN.py）
        
        Args:
            model: 求解后的Gurobi模型
            
        Returns:
            Dict: 包含numpy数组格式的对偶变量
        """
        try:
            # 先提取所有对偶变量
            implicit_duals_dict = self.extract_dual_variables(model)
            
            # 转换为与lambda_sol相同的numpy数组格式
            lambda_sol_implicit = {}
            
            # 1. 功率平衡约束对偶变量: shape (T,)
            if 'power_balance' in implicit_duals_dict:
                lambda_sol_implicit['lambda_power_balance'] = np.array([
                    implicit_duals_dict['power_balance'].get(t, 0) for t in range(self.T)
                ])
            else:
                lambda_sol_implicit['lambda_power_balance'] = np.zeros(self.T)
            
            # 2. 发电上下限约束对偶变量: shape (ng, T)
            if 'pg_lower' in implicit_duals_dict and 'pg_upper' in implicit_duals_dict:
                lambda_sol_implicit['lambda_pg_lower'] = np.array([
                    [abs(implicit_duals_dict['pg_lower'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_pg_upper'] = np.array([
                    [abs(implicit_duals_dict['pg_upper'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            else:
                lambda_sol_implicit['lambda_pg_lower'] = np.zeros((self.ng, self.T))
                lambda_sol_implicit['lambda_pg_upper'] = np.zeros((self.ng, self.T))
            
            # 3. 爬坡约束对偶变量: shape (ng, T-1)
            if 'ramp_up' in implicit_duals_dict and 'ramp_down' in implicit_duals_dict:
                lambda_sol_implicit['lambda_ramp_up'] = np.array([
                    [abs(implicit_duals_dict['ramp_up'].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_ramp_down'] = np.array([
                    [abs(implicit_duals_dict['ramp_down'].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                    for g in range(self.ng)
                ])
            else:
                lambda_sol_implicit['lambda_ramp_up'] = np.zeros((self.ng, self.T-1))
                lambda_sol_implicit['lambda_ramp_down'] = np.zeros((self.ng, self.T-1))
            
            # 4. 最小开关机时间约束对偶变量: shape (ng, Ton/Toff, T)
            Ton = self.Ton
            Toff = self.Toff
            
            if 'min_on' in implicit_duals_dict:
                lambda_sol_implicit['lambda_min_on'] = np.zeros((self.ng, Ton, self.T))
                for g in range(self.ng):
                    for tau in range(self._min_up_horizon(g)):
                        for t in range(self.T):
                            val = implicit_duals_dict['min_on'].get(g, {}).get(tau+1, {}).get(t, 0)
                            lambda_sol_implicit['lambda_min_on'][g, tau, t] = abs(val)
            else:
                lambda_sol_implicit['lambda_min_on'] = np.zeros((self.ng, Ton, self.T))
            
            if 'min_off' in implicit_duals_dict:
                lambda_sol_implicit['lambda_min_off'] = np.zeros((self.ng, Toff, self.T))
                for g in range(self.ng):
                    for tau in range(self._min_down_horizon(g)):
                        for t in range(self.T):
                            val = implicit_duals_dict['min_off'].get(g, {}).get(tau+1, {}).get(t, 0)
                            lambda_sol_implicit['lambda_min_off'][g, tau, t] = abs(val)
            else:
                lambda_sol_implicit['lambda_min_off'] = np.zeros((self.ng, Toff, self.T))
            
            # 5. 启停成本约束对偶变量: shape (ng, T-1)
            constraint_names = ['start_cost', 'shut_cost', 'coc_nonneg']
            lambda_names = ['lambda_start_cost', 'lambda_shut_cost', 'lambda_coc_nonneg']
            
            for constraint_name, lambda_name in zip(constraint_names, lambda_names):
                if constraint_name in implicit_duals_dict:
                    lambda_sol_implicit[lambda_name] = np.array([
                        [abs(implicit_duals_dict[constraint_name].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                        for g in range(self.ng)
                    ])
                else:
                    lambda_sol_implicit[lambda_name] = np.zeros((self.ng, self.T-1))
            
            # 6. 发电成本约束对偶变量: shape (ng, T)
            if 'cpower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_cpower'] = np.array([
                    [abs(implicit_duals_dict['cpower'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            else:
                lambda_sol_implicit['lambda_cpower'] = np.zeros((self.ng, self.T))
            
            # 7. DCPF约束对偶变量: shape (nl, T)
            if 'dcpf_upper' in implicit_duals_dict and 'dcpf_lower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_dcpf_upper'] = np.array([
                    [abs(implicit_duals_dict['dcpf_upper'].get(l, {}).get(t, 0)) for t in range(self.T)] 
                    for l in range(self.nl)
                ])
                lambda_sol_implicit['lambda_dcpf_lower'] = np.array([
                    [abs(implicit_duals_dict['dcpf_lower'].get(l, {}).get(t, 0)) for t in range(self.T)] 
                    for l in range(self.nl)
                ])
            else:
                lambda_sol_implicit['lambda_dcpf_upper'] = np.zeros((self.nl, self.T))
                lambda_sol_implicit['lambda_dcpf_lower'] = np.zeros((self.nl, self.T))

            # 8. x变量上下界约束对偶变量: shape (ng, T)
            if 'x_upper' in implicit_duals_dict and 'x_lower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_x_upper'] = np.array([
                    [abs(implicit_duals_dict['x_upper'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_x_lower'] = np.array([
                    [abs(implicit_duals_dict['x_lower'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            else:
                lambda_sol_implicit['lambda_x_upper'] = np.zeros((self.ng, self.T))
                lambda_sol_implicit['lambda_x_lower'] = np.zeros((self.ng, self.T))
            
            return lambda_sol_implicit
            
        except Exception as e:
            print(f"❌ 对偶变量数组转换失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 返回空的对偶变量字典（使用零值）
            return self._create_empty_lambda_dict()
    
    def _create_empty_lambda_dict(self):
        """
        创建空的对偶变量字典（所有值设为0）
        """
        Ton = self.Ton
        Toff = self.Toff
        return {
            'lambda_power_balance': np.zeros(self.T),
            'lambda_pg_lower': np.zeros((self.ng, self.T)),
            'lambda_pg_upper': np.zeros((self.ng, self.T)),
            'lambda_ramp_up': np.zeros((self.ng, self.T-1)),
            'lambda_ramp_down': np.zeros((self.ng, self.T-1)),
            'lambda_min_on': np.zeros((self.ng, Ton, self.T)),
            'lambda_min_off': np.zeros((self.ng, Toff, self.T)),
            'lambda_start_cost': np.zeros((self.ng, self.T-1)),
            'lambda_shut_cost': np.zeros((self.ng, self.T-1)),
            'lambda_coc_nonneg': np.zeros((self.ng, self.T-1)),
            'lambda_cpower': np.zeros((self.ng, self.T)),
            'lambda_dcpf_upper': np.zeros((self.nl, self.T)),
            'lambda_dcpf_lower': np.zeros((self.nl, self.T)),
            'lambda_x_upper': np.zeros((self.ng, self.T)),
            'lambda_x_lower': np.zeros((self.ng, self.T))
        }

    def _apply_fast_gurobi_tolerances(self, model, mip=True):
        model.Params.OutputFlag = 0
        if self.gurobi_threads is not None:
            model.Params.Threads = self.gurobi_threads
        if not mip and self.gurobi_lp_method != -1:
            model.Params.Method = self.gurobi_lp_method
        model.Params.FeasibilityTol = self.gurobi_feasibility_tol
        model.Params.OptimalityTol = self.gurobi_optimality_tol
        if mip:
            model.Params.MIPGap = self.gurobi_mip_gap
            model.Params.IntFeasTol = self.gurobi_int_feas_tol

    def _require_cvxpy_highs_main_backend(self) -> None:
        if self._lp_backend != LP_BACKEND_CVXPY_HIGHS:
            raise RuntimeError(f"Main BCD lp_backend={self._lp_backend!r} is not cvxpy_highs")
        assert_lp_backend_available(self._lp_backend)
        if not CVXPY_AVAILABLE:
            raise RuntimeError("cvxpy is unavailable for main BCD cvxpy_highs backend")

    def _build_generator_incidence_matrix(self) -> np.ndarray:
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1.0
        return G

    def _solve_lp_with_fixed_x_cvxpy_highs(self, Pd: np.ndarray, x_vals: np.ndarray):
        self._require_cvxpy_highs_main_backend()

        Pd = np.asarray(Pd, dtype=float)
        x_vals = np.asarray(x_vals, dtype=float)
        a = (self.gencost[:, -2] / self.T_delta).reshape(self.ng, 1)
        b = (self.gencost[:, -1] / self.T_delta).reshape(self.ng, 1)
        pg = cp.Variable((self.ng, self.T))
        coc = cp.Variable((self.ng, max(self.T - 1, 0)))
        cpower = cp.Variable((self.ng, self.T))
        constraints = [pg >= 0, cpower >= 0]
        if self.T > 1:
            constraints.append(coc >= 0)

        power_balance_cons = []
        pg_lower_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]
        pg_upper_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]
        ramp_up_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        ramp_down_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        start_cost_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        shut_cost_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        coc_nonneg_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        cpower_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]

        for t in range(self.T):
            cons = cp.sum(pg[:, t]) == float(np.sum(Pd[:, t]))
            constraints.append(cons)
            power_balance_cons.append(cons)

        for g in range(self.ng):
            for t in range(self.T):
                cons = pg[g, t] >= self.gen[g, PMIN] * x_vals[g, t]
                constraints.append(cons)
                pg_lower_cons[g][t] = cons
                cons = pg[g, t] <= self.gen[g, PMAX] * x_vals[g, t]
                constraints.append(cons)
                pg_upper_cons[g][t] = cons

        for g in range(self.ng):
            for t in range(1, self.T):
                rhs_up = self.Ru[g] * x_vals[g, t - 1] + self.Ru_co[g] * (1.0 - x_vals[g, t - 1])
                cons = pg[g, t] - pg[g, t - 1] <= rhs_up
                constraints.append(cons)
                ramp_up_cons[g][t - 1] = cons
                rhs_dn = self.Rd[g] * x_vals[g, t] + self.Rd_co[g] * (1.0 - x_vals[g, t])
                cons = pg[g, t - 1] - pg[g, t] <= rhs_dn
                constraints.append(cons)
                ramp_down_cons[g][t - 1] = cons

        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for g in range(self.ng):
            for t in range(1, self.T):
                cons = coc[g, t - 1] >= start_cost[g] * (x_vals[g, t] - x_vals[g, t - 1])
                constraints.append(cons)
                start_cost_cons[g][t - 1] = cons
                cons = coc[g, t - 1] >= shut_cost[g] * (x_vals[g, t - 1] - x_vals[g, t])
                constraints.append(cons)
                shut_cost_cons[g][t - 1] = cons
                cons = coc[g, t - 1] >= 0.0
                constraints.append(cons)
                coc_nonneg_cons[g][t - 1] = cons

        for g in range(self.ng):
            for t in range(self.T):
                cons = cpower[g, t] >= a[g, 0] * pg[g, t] + b[g, 0] * x_vals[g, t]
                constraints.append(cons)
                cpower_cons[g][t] = cons

        objective = cp.sum(cpower) + (cp.sum(coc) if self.T > 1 else 0.0)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        _solve_with_cvxpy_highs(problem, verbose=False, threads=self.bcd_highs_threads)
        if not _problem_is_optimal(problem):
            return None, None, None, None

        lambda_dict = self._create_empty_lambda_dict()
        lambda_dict['lambda_power_balance'] = np.array(
            [_cvxpy_pi_to_gurobi_pi(cons, 'eq') for cons in power_balance_cons],
            dtype=float,
        )
        lambda_dict['lambda_pg_lower'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(pg_lower_cons[g][t], 'ge')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )
        lambda_dict['lambda_pg_upper'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(pg_upper_cons[g][t], 'le')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )
        if self.T > 1:
            lambda_dict['lambda_ramp_up'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(ramp_up_cons[g][t], 'le')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_ramp_down'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(ramp_down_cons[g][t], 'le')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_start_cost'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(start_cost_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_shut_cost'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(shut_cost_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_coc_nonneg'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(coc_nonneg_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
        lambda_dict['lambda_cpower'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(cpower_cons[g][t], 'ge')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )

        pg_sol = np.asarray(pg.value, dtype=float)
        coc_sol = np.asarray(coc.value, dtype=float) if self.T > 1 else np.zeros((self.ng, 0), dtype=float)
        cpower_sol = np.asarray(cpower.value, dtype=float)
        return pg_sol, coc_sol, cpower_sol, lambda_dict

    def _solve_lp_relaxation_cvxpy_highs(self, Pd: np.ndarray):
        self._require_cvxpy_highs_main_backend()

        Pd = np.asarray(Pd, dtype=float)
        a = (self.gencost[:, -2] / self.T_delta).reshape(self.ng, 1)
        b = (self.gencost[:, -1] / self.T_delta).reshape(self.ng, 1)
        pg = cp.Variable((self.ng, self.T))
        x = cp.Variable((self.ng, self.T))
        coc = cp.Variable((self.ng, max(self.T - 1, 0)))
        cpower = cp.Variable((self.ng, self.T))
        constraints = [pg >= 0, x >= 0, x <= 1, cpower >= 0]
        if self.T > 1:
            constraints.append(coc >= 0)

        power_balance_cons = []
        pg_lower_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]
        pg_upper_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]
        ramp_up_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        ramp_down_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        min_on_cons = {}
        min_off_cons = {}
        start_cost_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        shut_cost_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        coc_nonneg_cons = [[None for _ in range(max(self.T - 1, 0))] for _ in range(self.ng)]
        cpower_cons = [[None for _ in range(self.T)] for _ in range(self.ng)]

        for t in range(self.T):
            cons = cp.sum(pg[:, t]) == float(np.sum(Pd[:, t]))
            constraints.append(cons)
            power_balance_cons.append(cons)

        for g in range(self.ng):
            for t in range(self.T):
                cons = pg[g, t] >= self.gen[g, PMIN] * x[g, t]
                constraints.append(cons)
                pg_lower_cons[g][t] = cons
                cons = pg[g, t] <= self.gen[g, PMAX] * x[g, t]
                constraints.append(cons)
                pg_upper_cons[g][t] = cons

        for g in range(self.ng):
            for t in range(1, self.T):
                cons = pg[g, t] - pg[g, t - 1] <= self.Ru[g] * x[g, t - 1] + self.Ru_co[g] * (1 - x[g, t - 1])
                constraints.append(cons)
                ramp_up_cons[g][t - 1] = cons
                cons = pg[g, t - 1] - pg[g, t] <= self.Rd[g] * x[g, t] + self.Rd_co[g] * (1 - x[g, t])
                constraints.append(cons)
                ramp_down_cons[g][t - 1] = cons

        Ton = self.Ton
        Toff = self.Toff
        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    cons = x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau]
                    constraints.append(cons)
                    min_on_cons[g, tau - 1, t1] = cons
            for tau in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    cons = -x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau]
                    constraints.append(cons)
                    min_off_cons[g, tau - 1, t1] = cons

        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for g in range(self.ng):
            for t in range(1, self.T):
                cons = coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1])
                constraints.append(cons)
                start_cost_cons[g][t - 1] = cons
                cons = coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t])
                constraints.append(cons)
                shut_cost_cons[g][t - 1] = cons
                cons = coc[g, t - 1] >= 0.0
                constraints.append(cons)
                coc_nonneg_cons[g][t - 1] = cons

        for g in range(self.ng):
            for t in range(self.T):
                cons = cpower[g, t] >= a[g, 0] * pg[g, t] + b[g, 0] * x[g, t]
                constraints.append(cons)
                cpower_cons[g][t] = cons

        objective = cp.sum(cpower) + (cp.sum(coc) if self.T > 1 else 0.0)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        _solve_with_cvxpy_highs(problem, verbose=False, threads=self.bcd_highs_threads)
        if not _problem_is_optimal(problem):
            return None, None, None, None, None

        lambda_dict = self._create_empty_lambda_dict()
        lambda_dict['lambda_power_balance'] = np.array(
            [_cvxpy_pi_to_gurobi_pi(cons, 'eq') for cons in power_balance_cons],
            dtype=float,
        )
        lambda_dict['lambda_pg_lower'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(pg_lower_cons[g][t], 'ge')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )
        lambda_dict['lambda_pg_upper'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(pg_upper_cons[g][t], 'le')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )
        if self.T > 1:
            lambda_dict['lambda_ramp_up'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(ramp_up_cons[g][t], 'le')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_ramp_down'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(ramp_down_cons[g][t], 'le')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_start_cost'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(start_cost_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_shut_cost'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(shut_cost_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
            lambda_dict['lambda_coc_nonneg'] = np.array(
                [[abs(_cvxpy_pi_to_gurobi_pi(coc_nonneg_cons[g][t], 'ge')) for t in range(self.T - 1)] for g in range(self.ng)],
                dtype=float,
            )
        for g in range(self.ng):
            for tau in range(self._min_up_horizon(g)):
                for t1 in range(self.T - (tau + 1)):
                    lambda_dict['lambda_min_on'][g, tau, t1] = abs(_cvxpy_pi_to_gurobi_pi(min_on_cons[g, tau, t1], 'le'))
            for tau in range(self._min_down_horizon(g)):
                for t1 in range(self.T - (tau + 1)):
                    lambda_dict['lambda_min_off'][g, tau, t1] = abs(_cvxpy_pi_to_gurobi_pi(min_off_cons[g, tau, t1], 'le'))
        lambda_dict['lambda_cpower'] = np.array(
            [[abs(_cvxpy_pi_to_gurobi_pi(cpower_cons[g][t], 'ge')) for t in range(self.T)] for g in range(self.ng)],
            dtype=float,
        )

        x_lp = np.asarray(x.value, dtype=float)
        pg_sol = np.asarray(pg.value, dtype=float)
        coc_sol = np.asarray(coc.value, dtype=float) if self.T > 1 else np.zeros((self.ng, 0), dtype=float)
        cpower_sol = np.asarray(cpower.value, dtype=float)
        return x_lp, pg_sol, coc_sol, cpower_sol, lambda_dict
    
    def _create_union_analysis_from_x_init(self, x_init, lambda_init):
        """创建union_analysis（参考uc_NN.py）"""
        # 找到非整数变量
        fractional_variables = []
        for sample_id in range(self.n_samples):
            for g in range(self.ng):
                for t in range(self.T):
                    x_val = x_init[sample_id, g, t]
                    if not (np.abs(x_val) < 1e-6 or np.abs(x_val - 1.0) < 1e-6):
                        fractional_variables.append({
                            'sample_id': sample_id,
                            'unit_id': g,
                            'time_slot': t,
                            'x_value': x_val
                        })
        
        print(f"发现 {len(fractional_variables)} 个非整数/非正确变量")

        union_constraints = []
        union_zeta_constraints = []
        
        enable_theta = getattr(self, 'enable_theta_constraints', True)
        if enable_theta:
            if getattr(self, 'external_sparse_templates', None) is not None:
                union_constraints = template_library_to_bcd_union_constraints(
                    self.external_sparse_templates
                )
            else:
                union_constraints = self._compute_dcpf_constraints_for_fractional_times(
                    fractional_variables, lambda_init
                )
            
            # 手动为每个时段添加M=4个包含所有机组的约束
            # M = 4
            # manual_constraints = self._add_manual_constraints_all_units(M)
            # union_constraints.extend(manual_constraints)
            manual_constraints = []
        
        enable_zeta = getattr(self, 'enable_zeta_constraints', True)
        if enable_zeta:
            use_per_variable_zeta = getattr(self, 'use_per_variable_zeta_constraints', False)
            if use_per_variable_zeta:
                union_zeta_constraints = self._create_per_variable_zeta_constraints()
            else:
                union_zeta_constraints = self._compute_specialized_constraints_of_balance_node(
                    fractional_variables
                )
        
        manual_constraints_count = len(manual_constraints) if enable_theta else 0
        print(f"生成 {len(union_constraints)} 个theta约束 (包含 {manual_constraints_count} 个手动添加的约束), 生成 {len(union_zeta_constraints)} 个zeta约束")
        
        return {
            'union_constraints': union_constraints,
            'union_zeta_constraints': union_zeta_constraints,
        }

    def _theta_member_time_index(self, constraint_info, coeff_info):
        """theta 成员变量的显式时间索引；旧模板回退到 constraint 的 time_slot。"""
        return int(coeff_info.get('time_index', constraint_info.get('time_slot', 0)))

    def _theta_var_name(self, branch_id, unit_id, member_time):
        return f'theta_branch_{branch_id}_unit_{unit_id}_time_{member_time}'

    def _theta_rhs_name(self, branch_id, anchor_time):
        return f'theta_rhs_branch_{branch_id}_time_{anchor_time}'

    def _zeta_var_name(self, unit_id, time_slot):
        return f'zeta_unit_{unit_id}_time_{time_slot}'

    def _zeta_rhs_name(self, unit_id, time_slot):
        return f'zeta_rhs_unit_{unit_id}_time_{time_slot}'

    def _iter_theta_lookup_entries(self, union_analysis=None, dedupe_branch_time=True):
        """迭代 theta 查找表所需元组，显式区分 anchor/member 时间。"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        if not (self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis):
            return

        seen_branch_time: set[tuple[int, int]] = set()
        for constraint_info in union_analysis['union_constraints']:
            branch_id = int(constraint_info['branch_id'])
            anchor_time = int(constraint_info.get('time_slot', 0))

            key = (branch_id, anchor_time)
            if dedupe_branch_time and key in seen_branch_time:
                continue
            seen_branch_time.add(key)

            for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                unit_id = int(coeff_info['unit_id'])
                member_time = self._theta_member_time_index(constraint_info, coeff_info)
                theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                yield unit_id, member_time, branch_id, anchor_time, theta_name

    def _build_theta_value_lookup(self, sample_theta, union_analysis=None):
        """(unit_id, member_time) -> [(branch_id, anchor_time, theta_value), ...]"""
        theta_lookup: dict = defaultdict(list)
        if sample_theta is None:
            return {}

        for unit_id, member_time, branch_id, anchor_time, theta_name in (
            self._iter_theta_lookup_entries(union_analysis=union_analysis, dedupe_branch_time=True) or []
        ):
            theta_lookup[(unit_id, member_time)].append(
                (branch_id, anchor_time, sample_theta.get(theta_name, 0.0))
            )

        return dict(theta_lookup)

    def _build_theta_index_lookup(self, union_analysis=None):
        """(unit_id, member_time) -> [(theta_idx, branch_id, anchor_time), ...]"""
        theta_lookup: dict = defaultdict(list)
        if not hasattr(self, '_theta_name_to_idx'):
            return {}

        for unit_id, member_time, branch_id, anchor_time, theta_name in (
            self._iter_theta_lookup_entries(union_analysis=union_analysis, dedupe_branch_time=True) or []
        ):
            theta_idx = self._theta_name_to_idx.get(theta_name, -1)
            if theta_idx >= 0:
                theta_lookup[(unit_id, member_time)].append((theta_idx, branch_id, anchor_time))

        return dict(theta_lookup)

    def _iter_zeta_lookup_entries(self, union_analysis=None):
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        if not (self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis):
            return

        for constraint in union_analysis['union_zeta_constraints']:
            unit_id = int(constraint['unit_id'])
            time_slot = int(constraint['time_slot'])
            yield unit_id, time_slot, self._zeta_var_name(unit_id, time_slot)

    def _build_zeta_value_lookup(self, sample_zeta, union_analysis=None):
        """(unit_id, time_slot) -> [zeta_value, ...]"""
        zeta_lookup: dict = defaultdict(list)
        if sample_zeta is None:
            return {}

        for unit_id, time_slot, zeta_name in self._iter_zeta_lookup_entries(union_analysis=union_analysis) or []:
            zeta_lookup[(unit_id, time_slot)].append(sample_zeta.get(zeta_name, 0.0))

        return dict(zeta_lookup)

    def _build_zeta_index_lookup(self, union_analysis=None):
        """(unit_id, time_slot) -> [zeta_idx, ...]"""
        zeta_lookup: dict = defaultdict(list)
        if not hasattr(self, '_zeta_name_to_idx'):
            return {}

        for unit_id, time_slot, zeta_name in self._iter_zeta_lookup_entries(union_analysis=union_analysis) or []:
            zeta_idx = self._zeta_name_to_idx.get(zeta_name, -1)
            if zeta_idx >= 0:
                zeta_lookup[(unit_id, time_slot)].append(zeta_idx)

        return dict(zeta_lookup)
    
    def _build_x_dual_stationarity_expr(
        self,
        g,
        t,
        lambda_cpower,
        lambda_x_upper,
        lambda_x_lower,
        lambda_pg_lower,
        lambda_pg_upper,
        lambda_ramp_down,
        lambda_ramp_up,
        lambda_min_on,
        lambda_min_off,
        lambda_start_cost,
        lambda_shut_cost,
        fixed_cost,
        pmin,
        pmax,
        rd_delta,
        ru_delta,
        start_cost,
        shut_cost,
        Ton,
        Toff,
    ):
        """Source-of-truth for the x-stationarity term used by dual_x."""
        dual_expr = fixed_cost * lambda_cpower[g, t]
        dual_expr = dual_expr + lambda_x_upper[g, t] - lambda_x_lower[g, t]
        dual_expr = dual_expr + pmin * lambda_pg_lower[g, t]
        dual_expr = dual_expr - pmax * lambda_pg_upper[g, t]

        if t > 0:
            dual_expr = dual_expr + rd_delta * lambda_ramp_down[g, t - 1]
        if t < self.T - 1:
            dual_expr = dual_expr + ru_delta * lambda_ramp_up[g, t]

        ton_g = min(int(Ton), self._min_up_horizon(g))
        toff_g = min(int(Toff), self._min_down_horizon(g))

        for tau in range(1, ton_g + 1):
            for t1 in range(self.T - tau):
                if t == t1 + 1:
                    dual_expr = dual_expr + lambda_min_on[g, tau - 1, t1]
                if t == t1:
                    dual_expr = dual_expr - lambda_min_on[g, tau - 1, t1]
                if t == t1 + tau:
                    dual_expr = dual_expr - lambda_min_on[g, tau - 1, t1]

        for tau in range(1, toff_g + 1):
            for t1 in range(self.T - tau):
                if t == t1 + 1:
                    dual_expr = dual_expr - lambda_min_off[g, tau - 1, t1]
                if t == t1:
                    dual_expr = dual_expr + lambda_min_off[g, tau - 1, t1]
                if t == t1 + tau:
                    dual_expr = dual_expr + lambda_min_off[g, tau - 1, t1]

        if t > 0:
            dual_expr = dual_expr + start_cost * lambda_start_cost[g, t - 1]
            dual_expr = dual_expr - shut_cost * lambda_shut_cost[g, t - 1]
        if t < self.T - 1:
            dual_expr = dual_expr - start_cost * lambda_start_cost[g, t]
            dual_expr = dual_expr + shut_cost * lambda_shut_cost[g, t]

        return dual_expr

    def _compute_dcpf_constraints_for_fractional_times(self, fractional_variables, lambda_init):
        """计算DCPF约束：per (branch, fractional_time) 构建，每条约束包含多个 generator。"""
        union_constraints = []
        nb = self.bus.shape[0]
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)

        # 收集所有分数时段（去重）
        fractional_times = sorted(set(var['time_slot'] for var in fractional_variables))

        branch_time_priority = {}
        for sample_id, sample in enumerate(self.active_set_data):
            sample_lambda = self._normalize_lambda_from_data(sample.get('lambda', None))
            if sample_lambda is None and lambda_init is not None and sample_id < len(lambda_init):
                sample_lambda = lambda_init[sample_id] if isinstance(lambda_init[sample_id], dict) else None
            if not isinstance(sample_lambda, dict):
                continue
            lam_up = np.asarray(sample_lambda.get('lambda_dcpf_upper', np.zeros((self.nl, self.T))), dtype=float)
            lam_dn = np.asarray(sample_lambda.get('lambda_dcpf_lower', np.zeros((self.nl, self.T))), dtype=float)
            if lam_up.shape != (self.nl, self.T):
                lam_up = np.zeros((self.nl, self.T), dtype=float)
            if lam_dn.shape != (self.nl, self.T):
                lam_dn = np.zeros((self.nl, self.T), dtype=float)
            branch_scores = np.abs(lam_up) + np.abs(lam_dn)
            for time_slot in fractional_times:
                for branch_id in range(self.nl):
                    key = (branch_id, time_slot)
                    score = float(branch_scores[branch_id, time_slot])
                    if score > branch_time_priority.get(key, 0.0):
                        branch_time_priority[key] = score

        # 对每个 (branch, fractional_time) 构建约束
        for time_slot in fractional_times:
            time_slot_constraints = []
            for branch_id in range(self.nl):
                nonzero_coefficients = []
                for g in range(self.ng):
                    bus_idx = int(self.gen[g, GEN_BUS])
                    if 0 <= bus_idx < nb:
                        coeff = PTDF[branch_id, bus_idx]
                        if abs(coeff) > 1e-3:
                            nonzero_coefficients.append({
                                'unit_id': g,
                                'branch_id': branch_id,
                                'coefficient': coeff,
                            })
                if nonzero_coefficients:
                    time_slot_constraints.append({
                        'branch_id': branch_id,
                        'time_slot': time_slot,
                        'constraint_type': 'dcpf',
                        'nonzero_pg_coefficients': nonzero_coefficients,
                        'constraint_name': f'dcpf_{branch_id}_{time_slot}',
                        'priority_score': branch_time_priority.get((branch_id, time_slot), 0.0),
                    })

            max_keep = self.max_theta_constraints_per_time_slot
            if max_keep is not None and max_keep > 0 and len(time_slot_constraints) > max_keep:
                time_slot_constraints.sort(
                    key=lambda item: (-float(item.get('priority_score', 0.0)), int(item['branch_id']))
                )
                time_slot_constraints = time_slot_constraints[:max_keep]

            for constraint in time_slot_constraints:
                constraint.pop('priority_score', None)
            union_constraints.extend(time_slot_constraints)

        return union_constraints
    
    def _add_manual_constraints_all_units(self, M=4):
        """手动添加包含所有机组的约束（参考uc_NN.py）"""
        manual_constraints = []
        max_branch_id = self.nl - 1
        
        for t in range(self.T):
            for m in range(M):
                virtual_branch_id = max_branch_id + 1 + m
                nonzero_coefficients = []
                for g in range(self.ng):
                    nonzero_coefficients.append({
                        'unit_id': g,
                        'branch_id': virtual_branch_id,
                        'coefficient': 1.0 / self.ng  # 归一化系数
                    })
                
                manual_constraints.append({
                    'branch_id': virtual_branch_id,
                    'time_slot': t,
                    'constraint_type': 'manual_all_units',
                    'nonzero_pg_coefficients': nonzero_coefficients,
                    'constraint_name': f'manual_all_units_{virtual_branch_id}_{t}'
                })
        
        return manual_constraints
    
    def _create_per_variable_zeta_constraints(self):
        """创建每个变量一一对应的zeta约束（参考uc_NN.py）"""
        per_variable_constraints = []
        
        for g in range(self.ng):
            for t in range(self.T):
                per_variable_constraints.append({
                    'unit_id': g,
                    'time_slot': t,
                    'constraint_type': 'per_variable_zeta',
                    'constraint_name': f'zeta_per_variable_{g}_{t}',
                    'is_per_variable': True
                })
        
        print(f"✓ 创建了 {len(per_variable_constraints)} 个每个变量一一对应的zeta约束")
        return per_variable_constraints
    
    def _compute_specialized_constraints_of_balance_node(self, fractional_variables):
        """计算平衡节点约束（参考uc_NN.py）"""
        union_constraints = []
        
        for var in fractional_variables:
            union_constraints.append({
                'time_slot': var['time_slot'],
                'unit_id': var['unit_id'],
                'constraint_type': 'balance_node_power',
                'constraint_name': f"balance_node_power_{var['unit_id']}_{var['time_slot']}"
            })
        
        return union_constraints

    def _build_theta_limited_union_analysis(self, union_analysis, max_constraints_per_time_slot: int | None):
        """Cache a union_analysis view that keeps only the first k theta constraints per time slot."""
        if (
            max_constraints_per_time_slot is None
            or max_constraints_per_time_slot <= 0
            or not union_analysis
            or 'union_constraints' not in union_analysis
        ):
            return union_analysis

        union_constraints = union_analysis['union_constraints']
        if not union_constraints:
            return union_analysis

        cache_key = (id(union_analysis), int(max_constraints_per_time_slot))
        cached = self._theta_limited_union_analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        limited_constraints = []
        per_time_slot_counts: dict[int, int] = {}
        for constraint_info in union_constraints:
            time_slot = int(constraint_info.get('time_slot', 0))
            used = per_time_slot_counts.get(time_slot, 0)
            if used >= max_constraints_per_time_slot:
                continue
            limited_constraints.append(constraint_info)
            per_time_slot_counts[time_slot] = used + 1

        if len(limited_constraints) == len(union_constraints):
            self._theta_limited_union_analysis_cache[cache_key] = union_analysis
            return union_analysis

        limited_union_analysis = dict(union_analysis)
        limited_union_analysis['union_constraints'] = limited_constraints
        self._theta_limited_union_analysis_cache[cache_key] = limited_union_analysis
        return limited_union_analysis

    def _normalize_theta_training_stages(self, theta_training_stages, max_iter: int) -> list[dict] | None:
        """Normalize staged theta-constraint curriculum definitions."""
        if not theta_training_stages:
            return None

        normalized = []
        unresolved_stage_indices = []
        assigned_iterations = 0

        for stage_idx, raw_stage in enumerate(theta_training_stages):
            if isinstance(raw_stage, dict):
                limit = raw_stage.get(
                    'max_constraints_per_time_slot',
                    raw_stage.get('constraint_limit', raw_stage.get('limit')),
                )
                iterations = raw_stage.get(
                    'iterations',
                    raw_stage.get('max_iter', raw_stage.get('iters')),
                )
            else:
                limit = raw_stage
                iterations = None

            if limit is None:
                raise ValueError(
                    f"theta_training_stages[{stage_idx}] is missing max_constraints_per_time_slot"
                )
            limit = int(limit)
            if limit <= 0:
                raise ValueError(
                    f"theta_training_stages[{stage_idx}] must have a positive constraint limit, got {limit}"
                )

            if iterations is None:
                resolved_iterations = None
                unresolved_stage_indices.append(stage_idx)
            else:
                resolved_iterations = int(iterations)
                if resolved_iterations <= 0:
                    raise ValueError(
                        f"theta_training_stages[{stage_idx}] must have positive iterations, got {resolved_iterations}"
                    )
                assigned_iterations += resolved_iterations

            normalized.append({
                'stage_index': stage_idx,
                'max_constraints_per_time_slot': limit,
                'iterations': resolved_iterations,
            })

        if assigned_iterations > max_iter:
            raise ValueError(
                f"theta_training_stages assign {assigned_iterations} iterations, exceeding max_iter={max_iter}"
            )

        if unresolved_stage_indices:
            remaining = max_iter - assigned_iterations
            if remaining < len(unresolved_stage_indices):
                raise ValueError(
                    "theta_training_stages leaves too few iterations for stages without explicit durations"
                )
            base_iters = remaining // len(unresolved_stage_indices)
            extra_iters = remaining % len(unresolved_stage_indices)
            for offset, stage_idx in enumerate(unresolved_stage_indices):
                normalized[stage_idx]['iterations'] = base_iters + (1 if offset < extra_iters else 0)
        elif assigned_iterations < max_iter:
            normalized[-1]['iterations'] += max_iter - assigned_iterations

        iter_start = 0
        for stage in normalized:
            iter_end = iter_start + int(stage['iterations'])
            stage['iter_start'] = iter_start
            stage['iter_end'] = iter_end
            iter_start = iter_end

        return normalized

    def _resolve_active_theta_stage(self, iter_index: int, max_iter: int, theta_training_stages):
        normalized_stages = self._normalize_theta_training_stages(theta_training_stages, max_iter)
        if not normalized_stages:
            return None
        for stage in normalized_stages:
            if iter_index < stage['iter_end']:
                return stage
        return normalized_stages[-1]

    def _resolve_active_union_analysis(self, iter_index: int, max_iter: int, union_analysis, theta_training_stages):
        stage = self._resolve_active_theta_stage(iter_index, max_iter, theta_training_stages)
        if stage is None:
            return self._apply_theta_delay_to_union_analysis(union_analysis, iter_index), None
        active_union_analysis = self._build_theta_limited_union_analysis(
            union_analysis,
            stage['max_constraints_per_time_slot'],
        )
        active_union_analysis = self._apply_theta_delay_to_union_analysis(active_union_analysis, iter_index)
        return active_union_analysis, stage
    
    def initialize_theta_values(self, union_analysis=None):
        """初始化theta值（直接优化系数，参考uc_NN.py）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化theta值")
            return [{} for _ in range(self.n_samples)], np.zeros((self.n_samples, self.nl, self.T))
        
        union_constraints = union_analysis['union_constraints']
        
        initialization_values = {'theta_values': {}}

        for constraint in union_constraints:
            branch_id = constraint['branch_id']
            time_slot = constraint['time_slot']
            nonzero_coefficients = constraint.get('nonzero_pg_coefficients', [])

            for coeff_info in nonzero_coefficients:
                unit_id = coeff_info['unit_id']
                member_time = self._theta_member_time_index(constraint, coeff_info)
                var_name = self._theta_var_name(branch_id, unit_id, member_time)
                initialization_values['theta_values'][var_name] = 0.0

            theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
            initialization_values['theta_values'][theta_rhs_name] = constraint.get('initial_rhs', 0.0)
        
        # 初始化mu
        mu_init = np.ones((self.n_samples, self.nl, self.T), dtype=float) * 0.1
        
        base_dict = initialization_values['theta_values']
        return [base_dict.copy() for _ in range(self.n_samples)], mu_init

    def initialize_zeta_values(self, union_analysis=None):
        """初始化zeta值（直接优化系数，参考uc_NN.py）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化zeta值")
            return [{} for _ in range(self.n_samples)], np.zeros((self.n_samples, self.ng, self.T))
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        initialization_values = {'zeta_values': {}}

        for constraint in union_constraints:
            unit_id = constraint["unit_id"]
            time_slot = constraint["time_slot"]

            var_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            initialization_values['zeta_values'][var_name] = 0.0

            zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            initialization_values['zeta_values'][zeta_rhs_name] = 0.0
        
        # 初始化ita
        ita_init = np.ones((self.n_samples, self.ng, self.T), dtype=float) * 0.1
        
        base_dict = initialization_values['zeta_values']
        return [base_dict.copy() for _ in range(self.n_samples)], ita_init
    
    def _build_theta_hot_start_values(self, union_analysis=None, sample_id: int = 0):
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_constraints' not in union_analysis:
            return {}

        theta_values = {}
        rng = np.random.default_rng(42 + sample_id)
        strategy = self.theta_hot_start_strategy

        for constraint in union_analysis['union_constraints']:
            branch_id = constraint['branch_id']
            time_slot = constraint['time_slot']
            nonzero_coefficients = constraint.get('nonzero_pg_coefficients', [])

            if strategy == "dcpf_relative":
                coeff_scale = max(
                    (abs(float(coeff_info.get('coefficient', 0.0))) for coeff_info in nonzero_coefficients),
                    default=1.0,
                )
                coeff_scale = max(coeff_scale, 1e-8)
            else:
                coeff_scale = 1.0

            for coeff_info in nonzero_coefficients:
                unit_id = coeff_info['unit_id']
                member_time = self._theta_member_time_index(constraint, coeff_info)
                theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                if strategy == "dcpf_relative":
                    theta_values[theta_name] = float(coeff_info.get('coefficient', 0.0)) / coeff_scale
                else:
                    theta_values[theta_name] = float(rng.normal(0.0, self.theta_gaussian_std))

            theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
            if strategy == "dcpf_relative":
                rhs = float(constraint.get('initial_rhs', 1.0))
                rhs_scale = max(abs(rhs), 1e-8)
                theta_values[theta_rhs_name] = rhs / rhs_scale
            else:
                theta_values[theta_rhs_name] = float(rng.normal(1.0, self.theta_gaussian_std))

        return theta_values

    def _build_zeta_hot_start_values(self, union_analysis=None, sample_id: int = 0):
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return {}

        zeta_values = {}
        rng = np.random.default_rng(84 + sample_id)
        strategy = self.zeta_hot_start_strategy

        for constraint in union_analysis['union_zeta_constraints']:
            unit_id = constraint["unit_id"]
            time_slot = constraint["time_slot"]
            zeta_name = self._zeta_var_name(unit_id, time_slot)
            zeta_rhs_name = self._zeta_rhs_name(unit_id, time_slot)

            if strategy == "zero":
                zeta_values[zeta_name] = 0.0
                zeta_values[zeta_rhs_name] = 0.0
            else:
                zeta_values[zeta_name] = float(rng.normal(0.0, self.zeta_gaussian_std))
                zeta_values[zeta_rhs_name] = float(rng.normal(0.0, self.zeta_gaussian_std))

        return zeta_values

    def _apply_hot_start_initial_values(self, union_analysis=None):
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        self.theta_values_list = [
            self._build_theta_hot_start_values(union_analysis, sample_id=i)
            for i in range(self.n_samples)
        ]
        self.zeta_values_list = [
            self._build_zeta_hot_start_values(union_analysis, sample_id=i)
            for i in range(self.n_samples)
        ]

        self.theta_values = self.theta_values_list[0] if self.theta_values_list else {}
        self.zeta_values = self.zeta_values_list[0] if self.zeta_values_list else {}

        print(
            f"✓ theta/zeta 热启动已应用 "
            f"(theta={self.theta_hot_start_strategy}, zeta={self.zeta_hot_start_strategy})",
            flush=True,
        )

    def _unit_predictor_active(self) -> bool:
        if not getattr(self, 'use_unit_predictor', False):
            return False
        if getattr(self, 'unit_predictor', None) is None:
            return False
        return bool(TORCH_AVAILABLE)

    def _unit_predictor_finetune_active(self) -> bool:
        if not self._unit_predictor_active():
            return False
        return int(getattr(self, 'iter_number', 0)) >= int(getattr(self, 'unit_predictor_warmup_rounds', 0))

    def _unit_predictor_parameters(self) -> list:
        if not self._unit_predictor_active():
            return []
        params = []
        seen = set()
        for g in range(self.ng):
            net = self.unit_predictor.get_network(g)
            for p in net.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def _ensure_unit_predictor_optimizer(self, learning_rate: float | None = None) -> None:
        if not self._unit_predictor_finetune_active():
            self._unit_predictor_optimizer = None
            self._unit_predictor_optimizer_lr = None
            return
        resolved_lr = (
            self.unit_predictor_finetune_lr if learning_rate is None
            else max(float(learning_rate), 1e-10)
        )
        params = self._unit_predictor_parameters()
        if not params:
            self._unit_predictor_optimizer = None
            self._unit_predictor_optimizer_lr = None
            return
        if self._unit_predictor_optimizer is None or self._unit_predictor_optimizer_lr != resolved_lr:
            self._unit_predictor_optimizer = optim.Adam(
                params,
                lr=resolved_lr,
                weight_decay=self.unit_predictor_weight_decay,
            )
            self._unit_predictor_optimizer_lr = resolved_lr

    def _extract_unit_predictor_features(self, sample_id: int) -> np.ndarray:
        sample = normalize_sample_arrays(dict(self.active_set_data[sample_id]))
        features = np.asarray(get_feature_vector_from_sample(sample), dtype=np.float32)
        if self._unit_predictor_active():
            expected = int(getattr(self.unit_predictor, 'input_dim', features.size))
            if features.size == expected:
                return features
            net_load_features = np.asarray(get_sample_net_load(sample), dtype=np.float32).flatten()
            if net_load_features.size == expected:
                return net_load_features
            raise ValueError(
                f"UnitPredictor feature dimension mismatch for sample {sample_id}: "
                f"got {features.size}, net_load={net_load_features.size}, expected={expected}"
            )
        return features

    def _refresh_unit_predictor_feature_cache(self) -> None:
        if not self._unit_predictor_active() or self.device is None:
            self._unit_predictor_features_cache = None
            return
        self._unit_predictor_features_cache = torch.stack([
            torch.tensor(self._extract_unit_predictor_features(s), dtype=torch.float32)
            for s in range(self.n_samples)
        ]).to(self.device)

    def _unit_predictor_feature_tensor(self, sample_id: int) -> torch.Tensor:
        if not hasattr(self, '_unit_predictor_features_cache') or self._unit_predictor_features_cache is None:
            self._refresh_unit_predictor_feature_cache()
        return self._unit_predictor_features_cache[sample_id:sample_id + 1]

    def _apply_unit_predictor_zeta_override(
        self,
        zeta_tensor: torch.Tensor,
        sample_id: int,
        train_predictor: bool = False,
    ) -> torch.Tensor:
        """Replace zeta single-unit constraints with differentiable unit-predictor output.

        Mapping mirrors the subproblem single-time slice:
          x_hat = sigmoid(logits)
          zeta = 1 - 2*x_hat
          rhs  = x_hat*zeta

        So x_hat≈0 yields x<=0, while x_hat≈1 yields -x<=-1.
        """
        if not self._unit_predictor_active():
            return zeta_tensor
        if not self.enable_zeta_constraints:
            return zeta_tensor
        union_analysis = getattr(self, '_current_union_analysis', None)
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return zeta_tensor
        if zeta_tensor.numel() == 0:
            return zeta_tensor

        features_tensor = self._unit_predictor_feature_tensor(sample_id)
        replacements = {}
        for g in range(self.ng):
            net = self.unit_predictor.get_network(g)
            was_training = bool(net.training)
            if train_predictor:
                net.train()
                logits = self.unit_predictor.forward_logits(g, features_tensor)
            else:
                net.eval()
                with torch.no_grad():
                    logits = self.unit_predictor.forward_logits(g, features_tensor)
                if was_training:
                    net.train()
            x_hat = torch.sigmoid(logits).reshape(-1)
            zeta_pred = 1.0 - 2.0 * x_hat
            rhs_pred = x_hat * zeta_pred
            replacements[g] = (zeta_pred, rhs_pred)

        values = []
        for idx, name in enumerate(self.zeta_var_names):
            if name.startswith('zeta_rhs_unit_'):
                parsed = name[len('zeta_rhs_unit_'):]
                unit_text, _, time_text = parsed.partition('_time_')
                if unit_text.isdigit() and time_text.isdigit():
                    g = int(unit_text)
                    t = int(time_text)
                    if g in replacements and t < replacements[g][1].numel():
                        values.append(replacements[g][1][t])
                        continue
            elif name.startswith('zeta_unit_'):
                parsed = name[len('zeta_unit_'):]
                unit_text, _, time_text = parsed.partition('_time_')
                if unit_text.isdigit() and time_text.isdigit():
                    g = int(unit_text)
                    t = int(time_text)
                    if g in replacements and t < replacements[g][0].numel():
                        values.append(replacements[g][0][t])
                        continue
            values.append(zeta_tensor[idx])
        return torch.stack(values).to(device=zeta_tensor.device, dtype=zeta_tensor.dtype)

    def _apply_unit_predictor_initial_values(self) -> None:
        if not self._unit_predictor_active() or not hasattr(self, 'zeta_var_names'):
            return
        if not self.zeta_values_list:
            return
        self._refresh_unit_predictor_feature_cache()
        updated = []
        with torch.no_grad():
            for sample_id in range(self.n_samples):
                base_tensor = torch.tensor(
                    [float(self.zeta_values_list[sample_id].get(name, 0.0)) for name in self.zeta_var_names],
                    dtype=torch.float32,
                    device=self.device,
                )
                zeta_tensor = self._apply_unit_predictor_zeta_override(
                    base_tensor,
                    sample_id,
                    train_predictor=False,
                )
                updated.append(self._tensor_to_zeta_dict(zeta_tensor))
        self.zeta_values_list = updated
        self.zeta_values = self.zeta_values_list[0] if self.zeta_values_list else {}
        print("✓ BCD unit predictor 已用于初始化 zeta 代理约束", flush=True)

    def _apply_theta_delay_to_union_analysis(self, union_analysis, iter_index: int):
        delay = max(int(getattr(self, 'theta_constraint_delay_rounds', 0) or 0), 0)
        if delay <= 0 or int(iter_index) >= delay:
            return union_analysis
        if not union_analysis or 'union_constraints' not in union_analysis:
            return union_analysis
        if self._theta_delay_reported_round != int(iter_index):
            print(
                f"[BCD][theta-delay] iter={int(iter_index) + 1}, "
                f"theta constraints disabled until iter {delay}",
                flush=True,
            )
            self._theta_delay_reported_round = int(iter_index)
        delayed = dict(union_analysis)
        delayed['union_constraints'] = []
        return delayed

    def _current_theta_curriculum_scale(self) -> float:
        delay = max(int(getattr(self, 'theta_curriculum_delay_rounds', 0) or 0), 0)
        current_iter = max(int(getattr(self, 'iter_number', 0) or 0), 0)
        if current_iter < delay:
            return 0.0
        start_scale = max(float(getattr(self, 'theta_initial_scale', 1.0) or 0.0), 0.0)
        final_scale = max(float(getattr(self, 'theta_final_scale', 1.0) or 0.0), 0.0)
        rounds = max(int(getattr(self, 'theta_curriculum_rounds', 0) or 0), 0)
        if rounds <= 0:
            return final_scale
        frac = min(max((current_iter - delay) / max(rounds, 1), 0.0), 1.0)
        return start_scale + frac * (final_scale - start_scale)

    def _current_zeta_ita_cap(self) -> tuple[float | None, float]:
        base_weight = max(float(getattr(self, 'zeta_ita_cap_penalty_weight', 0.0) or 0.0), 0.0)
        initial_weight = getattr(self, 'zeta_ita_cap_initial_weight', None)
        final_weight = getattr(self, 'zeta_ita_cap_final_weight', None)
        if initial_weight is None:
            initial_weight = base_weight
        if final_weight is None:
            final_weight = base_weight
        initial_weight = max(float(initial_weight or 0.0), 0.0)
        final_weight = max(float(final_weight or 0.0), 0.0)

        initial_cap = getattr(self, 'zeta_ita_cap_initial', None)
        final_cap = getattr(self, 'zeta_ita_cap_final', None)
        if initial_cap is None and final_cap is None:
            return None, 0.0
        if initial_cap is None:
            initial_cap = final_cap
        if final_cap is None:
            final_cap = initial_cap

        start_round = max(int(getattr(self, 'zeta_ita_cap_start_round', 0) or 0), 0)
        end_round = max(int(getattr(self, 'zeta_ita_cap_end_round', start_round) or start_round), start_round)
        current_iter = max(int(getattr(self, 'iter_number', 0) or 0), 0)
        if end_round <= start_round:
            frac = 1.0 if current_iter >= start_round else 0.0
        else:
            frac = min(max((current_iter - start_round) / (end_round - start_round), 0.0), 1.0)

        cap = float(initial_cap) + frac * (float(final_cap) - float(initial_cap))
        weight = initial_weight + frac * (final_weight - initial_weight)
        return max(cap, 0.0), max(weight, 0.0)

    def add_theta_variables_for_branches(self, union_analysis=None):
        """为参数化约束添加theta变量（参考uc_NN.py）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建theta变量")
            return
        
        union_constraints = union_analysis['union_constraints']
        self.theta_vars = {}
        
        for constraint in union_constraints:
            branch_id = constraint['branch_id']
            time_slot = constraint['time_slot']
            nonzero_coefficients = constraint.get('nonzero_pg_coefficients', [])
            
            for coeff_info in nonzero_coefficients:
                unit_id = coeff_info['unit_id']
                member_time = self._theta_member_time_index(constraint, coeff_info)
                var_name = self._theta_var_name(branch_id, unit_id, member_time)
                self.theta_vars[var_name] = {
                    'branch_id': branch_id,
                    'unit_id': unit_id,
                    'time_slot': member_time,
                    'var_name': var_name,
                    'value': 0.0
                }
            
            # 创建右端项变量
            theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
            self.theta_vars[theta_rhs_name] = {
                'branch_id': branch_id,
                'time_slot': time_slot,
                'var_name': theta_rhs_name,
                'value': constraint.get('initial_rhs', 1.0)
            }
        
        print(f"✓ 创建了 {len(self.theta_vars)} 个theta变量")
    
    def add_zeta_variables_for_units(self, union_analysis=None):
        """为参数化约束添加zeta变量（参考uc_NN.py）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建zeta变量")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        self.zeta_vars = {}
        
        for constraint in union_constraints:
            unit_id = constraint["unit_id"]
            time_slot = constraint["time_slot"]
            
            var_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            self.zeta_vars[var_name] = {
                'unit_id': unit_id,
                'time_slot': time_slot,
                'var_name': var_name,
                'value': 0.0
            }
            
            # 创建右端项变量
            zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            self.zeta_vars[zeta_rhs_name] = {
                'unit_id': unit_id,
                'time_slot': time_slot,
                'var_name': zeta_rhs_name,
                'value': 1.0
            }
        
        print(f"✓ 创建了 {len(self.zeta_vars)} 个zeta变量")
    
    def _init_neural_network(self):
        """初始化神经网络模型（参考uc_NN.py）"""
        if not TORCH_AVAILABLE:
            return
        
        # 检测设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            self.device = torch.device('cpu')
            print("⚠ 未检测到GPU，将使用CPU", flush=True)
        
        # 计算输入特征维度
        sample_features = self._extract_features(0)
        input_dim = len(sample_features)
        
        # 计算输出维度
        self.theta_output_dim = len(self.theta_values)
        self.zeta_output_dim = len(self.zeta_values)
        
        # 创建theta/zeta网络，隐藏层规模由外部配置控制
        self.theta_net = build_mlp_with_dropout(
            input_dim,
            self.nn_hidden_dims,
            self.theta_output_dim,
        )
        self.zeta_net = build_mlp_with_dropout(
            input_dim,
            self.nn_hidden_dims,
            self.zeta_output_dim,
        )
        # 移动到设备
        if self.device:
            self.theta_net = self.theta_net.to(self.device)
            self.zeta_net = self.zeta_net.to(self.device)
        
        # 创建优化器（只优化theta和zeta网络，mu和ita在loss中通过Gurobi优化）
        all_params = list(self.theta_net.parameters()) + list(self.zeta_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.nn_learning_rate, weight_decay=0.0)
        
        # 保存变量名列表
        self.theta_var_names = list(self.theta_values.keys())
        self.zeta_var_names = list(self.zeta_values.keys())

        # 预建名称→索引映射，避免热路径中重复创建
        self._theta_name_to_idx = {name: idx for idx, name in enumerate(self.theta_var_names)}
        self._zeta_name_to_idx = {name: idx for idx, name in enumerate(self.zeta_var_names)}
        # union_analysis 缓存失效标记
        self._cached_union_analysis_id = None

        print(f"✓ 初始化神经网络:", flush=True)
        print(f"  - 设备: {self.device}", flush=True)
        print(f"  - 输入维度: {input_dim}", flush=True)
        print(f"  - 隐藏层: {self.nn_hidden_dims}", flush=True)
        print(f"  - theta输出维度: {self.theta_output_dim}", flush=True)
        print(f"  - zeta输出维度: {self.zeta_output_dim}", flush=True)

        # 预计算常量张量（Tier 1）和特征缓存（Tier 2）
        self._precompute_constant_tensors()
        self._features_cache = torch.stack([
            torch.tensor(np.array(self._extract_features(s)), dtype=torch.float32)
            for s in range(self.n_samples)
        ]).to(self.device)  # shape: (n_samples, input_dim)

    def _generate_initial_values_from_nn(self):
        """用未训练的神经网络 forward pass 生成每个 sample 的 theta/zeta 初值。"""
        self.theta_net.eval()
        self.zeta_net.eval()

        with torch.no_grad():
            # 批量推理所有 sample
            theta_out = self.theta_net(self._features_cache)  # (n_samples, theta_dim)
            zeta_out = self.zeta_net(self._features_cache)    # (n_samples, zeta_dim)

        for i in range(self.n_samples):
            self.theta_values_list[i] = self._tensor_to_theta_dict(theta_out[i])
            self.zeta_values_list[i] = self._tensor_to_zeta_dict(
                self._apply_unit_predictor_zeta_override(zeta_out[i], i, train_predictor=False)
            )

        self.theta_values = self.theta_values_list[0]
        self.zeta_values = self.zeta_values_list[0]

        self.theta_net.train()
        self.zeta_net.train()

        vals = list(self.theta_values.values())
        zvals = list(self.zeta_values.values())
        print(f"✓ 用NN forward pass生成theta/zeta初值 "
              f"(theta: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}; "
              f"zeta: mean={np.mean(zvals):.4f}, std={np.std(zvals):.4f})", flush=True)

    def refresh_theta_zeta_values_from_networks(self):
        """Refresh per-sample theta/zeta caches using the current loaded networks."""
        if not TORCH_AVAILABLE or self.theta_net is None or self.zeta_net is None:
            return

        if not hasattr(self, "_features_cache") or self._features_cache is None:
            self._refresh_feature_cache()

        theta_was_training = self.theta_net.training
        zeta_was_training = self.zeta_net.training
        self.theta_net.eval()
        self.zeta_net.eval()

        with torch.no_grad():
            theta_out = self.theta_net(self._features_cache)
            zeta_out = self.zeta_net(self._features_cache)

        self.theta_values_list = [
            self._tensor_to_theta_dict(theta_out[i])
            for i in range(self.n_samples)
        ]
        self.zeta_values_list = [
            self._tensor_to_zeta_dict(
                self._apply_unit_predictor_zeta_override(zeta_out[i], i, train_predictor=False)
            )
            for i in range(self.n_samples)
        ]
        self.theta_values = self.theta_values_list[0] if self.theta_values_list else {}
        self.zeta_values = self.zeta_values_list[0] if self.zeta_values_list else {}

        if theta_was_training:
            self.theta_net.train()
        if zeta_was_training:
            self.zeta_net.train()

    def _precompute_constant_tensors(self):
        """预计算训练全程不变的常量张量，避免每次 loss 调用时重建（Tier 1）。"""
        dev = self.device
        self._ct = {
            'gen_PMIN':      torch.tensor(self.gen[:, PMIN], dtype=torch.float32, device=dev),
            'gen_PMAX':      torch.tensor(self.gen[:, PMAX], dtype=torch.float32, device=dev),
            'gencost_fixed': torch.tensor(self.gencost[:, -1] / self.T_delta, dtype=torch.float32, device=dev),
            'start_cost':    torch.tensor(self.gencost[:, 1], dtype=torch.float32, device=dev),
            'shut_cost':     torch.tensor(self.gencost[:, 2], dtype=torch.float32, device=dev),
            'Ru':            torch.tensor(self.Ru, dtype=torch.float32, device=dev),
            'Rd':            torch.tensor(self.Rd, dtype=torch.float32, device=dev),
            'Ru_co':         torch.tensor(self.Ru_co, dtype=torch.float32, device=dev),
            'Rd_co':         torch.tensor(self.Rd_co, dtype=torch.float32, device=dev),
        }
        self._empty_tensor = torch.zeros(0, device=dev)
        # 热路径常量张量：避免循环内 torch.tensor(scalar) 触发 CUDA 分配
        self._const_zero = torch.tensor(0.0, device=dev)
        self._const_one = torch.tensor(1.0, device=dev)
        self._default_mu_tensor = torch.tensor(
            getattr(self, 'mu_dual_floor_init', getattr(self, 'dual_para_bound', 0.1)),
            dtype=torch.float32,
            device=dev,
        )

    def _current_dual_decay_round(self) -> int:
        dual_decay_round = getattr(self, 'dual_decay_round', None)
        if dual_decay_round is None:
            dual_decay_round = getattr(self, 'dual_para_bound_quit_iteration', 50)
        return max(int(dual_decay_round), 0)

    def _is_dual_floor_active(self) -> bool:
        return self.iter_number < self._current_dual_decay_round()

    def _is_dual_sign_relaxation_round(self) -> bool:
        interval = int(getattr(self, 'dual_sign_relax_interval', 0) or 0)
        return (
            interval > 0
            and self._is_dual_floor_active()
            and max(float(getattr(self, 'mu_dual_floor_init', 0.0)), float(getattr(self, 'ita_dual_floor_init', 0.0))) > 0.0
            and ((self.iter_number + 1) % interval == 0)
        )

    def _sync_parametric_direction_strategy_state(self) -> None:
        self.theta_constraint_direction_signs = np.asarray(self.theta_constraint_direction_signs, dtype=float)
        self.zeta_constraint_direction_signs = np.asarray(self.zeta_constraint_direction_signs, dtype=float)

    def _get_theta_constraint_direction(self, branch_id: int, time_slot: int) -> float:
        signs = np.asarray(getattr(self, 'theta_constraint_direction_signs', None), dtype=float)
        if signs.shape != (self.nl, self.T):
            self.theta_constraint_direction_signs = np.ones((self.nl, self.T), dtype=float)
            signs = self.theta_constraint_direction_signs
        if 0 <= branch_id < self.nl and 0 <= time_slot < self.T:
            return float(signs[branch_id, time_slot])
        return 1.0

    def _get_zeta_constraint_direction(self, unit_id: int, time_slot: int) -> float:
        signs = np.asarray(getattr(self, 'zeta_constraint_direction_signs', None), dtype=float)
        if signs.shape != (self.ng, self.T):
            self.zeta_constraint_direction_signs = np.ones((self.ng, self.T), dtype=float)
            signs = self.zeta_constraint_direction_signs
        if 0 <= unit_id < self.ng and 0 <= time_slot < self.T:
            return float(signs[unit_id, time_slot])
        return 1.0

    def _resolve_direction_signs_from_signed_duals(
        self,
        dual_values: np.ndarray,
        current_signs: np.ndarray,
        tol: float = 1e-8,
    ) -> np.ndarray:
        dual_arr = np.asarray(dual_values, dtype=float)
        if dual_arr.size == 0:
            return np.asarray(current_signs, dtype=float).copy()
        pos_count = np.sum(dual_arr > tol, axis=0)
        neg_count = np.sum(dual_arr < -tol, axis=0)
        resolved = np.asarray(current_signs, dtype=float).copy()
        resolved[pos_count > neg_count] = 1.0
        resolved[neg_count > pos_count] = -1.0
        return resolved

    def _finalize_signed_dual_values(
        self,
        dual_values: np.ndarray,
        direction_signs: np.ndarray,
        abs_floor: float,
    ) -> np.ndarray:
        dual_abs = np.abs(np.asarray(dual_values, dtype=float))
        if abs_floor > 0:
            dual_abs = np.maximum(dual_abs, abs_floor)
        return dual_abs

    def _direction_corrected_theta_values(self, theta_values: dict | None) -> dict:
        if theta_values is None:
            return {}
        corrected = dict(theta_values)
        union_analysis = getattr(self, '_current_union_analysis', None)
        if not (self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis):
            return corrected
        for constraint_info in union_analysis['union_constraints']:
            branch_id = constraint_info['branch_id']
            time_slot = constraint_info['time_slot']
            sign = self._get_theta_constraint_direction(branch_id, time_slot)
            if sign == 1.0:
                continue
            for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                unit_id = coeff_info['unit_id']
                member_time = self._theta_member_time_index(constraint_info, coeff_info)
                theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                if theta_name in corrected:
                    corrected[theta_name] = float(corrected[theta_name]) * sign
            theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
            if theta_rhs_name in corrected:
                corrected[theta_rhs_name] = float(corrected[theta_rhs_name]) * sign
        return corrected

    def _direction_corrected_zeta_values(self, zeta_values: dict | None) -> dict:
        if zeta_values is None:
            return {}
        corrected = dict(zeta_values)
        union_analysis = getattr(self, '_current_union_analysis', None)
        if not (self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis):
            return corrected
        for constraint in union_analysis['union_zeta_constraints']:
            unit_id = constraint['unit_id']
            time_slot = constraint['time_slot']
            sign = self._get_zeta_constraint_direction(unit_id, time_slot)
            if sign == 1.0:
                continue
            zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            if zeta_name in corrected:
                corrected[zeta_name] = float(corrected[zeta_name]) * sign
            zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            if zeta_rhs_name in corrected:
                corrected[zeta_rhs_name] = float(corrected[zeta_rhs_name]) * sign
        return corrected

    def _preprocess_union_analysis_cache(self, union_analysis) -> None:
        """将 union_analysis 预处理为结构化元组，消除 loss 热路径中的字符串格式化与字典扫描。

        结果写入:
            _cached_theta_constraints: list[(branch_id, time_slot, constraint_type,
                                            coeff_list[(unit_id, member_time, theta_idx)], rhs_idx)]
            _dual_theta_lookup: dict[(unit_id, member_time)] -> list[(theta_idx, branch_id, anchor_time)]
            _cached_zeta_constraints: list[(unit_id, time_slot, zeta_idx, rhs_idx)]
            _dual_zeta_lookup: dict[(unit_id, time_slot)] -> list[zeta_idx]
        """
        cached_theta: list = []
        seen_branch_time: set = set()

        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']

                key = (branch_id, time_slot)
                if key in seen_branch_time:
                    continue
                seen_branch_time.add(key)

                constraint_type = constraint_info.get('constraint_type', 'dcpf')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']

                coeff_list: list = []
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    theta_idx = self._theta_name_to_idx.get(theta_name, -1)
                    if theta_idx >= 0:
                        coeff_list.append((unit_id, member_time, theta_idx))

                theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
                rhs_idx = self._theta_name_to_idx.get(theta_rhs_name, -1)
                cached_theta.append((branch_id, time_slot, constraint_type, coeff_list, rhs_idx))

        cached_zeta: list = []

        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            for constraint in union_analysis['union_zeta_constraints']:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                zeta_name = self._zeta_var_name(unit_id, time_slot)
                zeta_idx = self._zeta_name_to_idx.get(zeta_name, -1)
                zeta_rhs_name = self._zeta_rhs_name(unit_id, time_slot)
                rhs_idx = self._zeta_name_to_idx.get(zeta_rhs_name, -1)
                if zeta_idx >= 0:
                    cached_zeta.append((unit_id, time_slot, zeta_idx, rhs_idx))

        self._cached_theta_constraints = cached_theta
        self._dual_theta_lookup: dict = self._build_theta_index_lookup(union_analysis)
        self._cached_zeta_constraints = cached_zeta
        self._dual_zeta_lookup: dict = self._build_zeta_index_lookup(union_analysis)
        self._cached_union_analysis_id = id(union_analysis)

    def _refresh_iter_tensor_cache(self):
        """每次 BCD 迭代整批转换 x/mu/ita/lambda，避免 loss 内逐张量创建（Tier 3）。"""
        dev = self.device
        self._x_cache   = torch.tensor(self.x,   dtype=torch.float32, device=dev)   # (n_samples, ng, T)
        self._mu_cache  = torch.tensor(self.mu,  dtype=torch.float32, device=dev)   # (n_samples, nl, T)
        self._ita_cache = torch.tensor(self.ita, dtype=torch.float32, device=dev)   # (n_samples, ng, T)
        self._lambda_cache = []
        for s in range(self.n_samples):
            d = self.lambda_[s]
            self._lambda_cache.append({
                k: torch.tensor(np.array(v), dtype=torch.float32, device=dev)
                for k, v in d.items()
                if isinstance(v, (list, np.ndarray)) and len(v) > 0
            })

    def _extract_features(self, sample_id):
        """从样本中提取特征，并兼容旧 checkpoint 的输入维度。"""
        sample = normalize_sample_arrays(dict(self.active_set_data[sample_id]))
        features = np.asarray(get_feature_vector_from_sample(sample), dtype=np.float32)

        expected_dim = self._get_expected_feature_dim()
        if expected_dim is None or features.size == expected_dim:
            return features

        net_load_features = np.asarray(get_sample_net_load(sample), dtype=np.float32).flatten()
        if net_load_features.size == expected_dim:
            return net_load_features

        raise ValueError(
            f"Feature dimension mismatch for sample {sample_id}: "
            f"got {features.size}, net_load={net_load_features.size}, expected={expected_dim}"
        )

    def _get_expected_feature_dim(self) -> Optional[int]:
        """Infer the expected NN input dimension from the loaded networks."""
        for net in (getattr(self, "theta_net", None), getattr(self, "zeta_net", None)):
            if net is None:
                continue
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    return int(module.in_features)
        return None

    def _refresh_feature_cache(self) -> None:
        """Rebuild cached features to match the currently loaded network input dimension."""
        if not TORCH_AVAILABLE or self.device is None:
            return
        self._features_cache = torch.stack([
            torch.tensor(np.asarray(self._extract_features(s)), dtype=torch.float32)
            for s in range(self.n_samples)
        ]).to(self.device)
        self._refresh_unit_predictor_feature_cache()
    
    def _tensor_to_theta_dict(self, theta_tensor):
        """将theta张量转换为字典（参考uc_NN.py）"""
        if not TORCH_AVAILABLE or theta_tensor is None:
            return self.theta_values.copy()
        values = theta_tensor.detach().cpu().numpy()
        return {name: float(val) for name, val in zip(self.theta_var_names, values)}
    
    def _tensor_to_zeta_dict(self, zeta_tensor):
        """将zeta张量转换为字典（参考uc_NN.py）"""
        if not TORCH_AVAILABLE or zeta_tensor is None:
            return self.zeta_values.copy()
        values = zeta_tensor.detach().cpu().numpy()
        return {name: float(val) for name, val in zip(self.zeta_var_names, values)}

    def _vector_to_theta_dict(self, values) -> dict:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return {name: float(arr[i]) for i, name in enumerate(self.theta_var_names[: arr.size])}

    def _vector_to_zeta_dict(self, values) -> dict:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return {name: float(arr[i]) for i, name in enumerate(self.zeta_var_names[: arr.size])}
    
    def _build_pg_cvxpy_persistent(self, sample_id: int):
        """Build a persistent CVXPY PG-block problem using cp.Parameter for every
        coefficient that changes between BCD iterations (lambda values, rho weights).

        The returned dict contains:
          'problem' – cp.Problem (built once, re-solved with warm_start=True)
          'params'  – dict of cp.Parameter objects to update each iteration
          'vars'    – dict of cp.Variable objects for reading solutions
        """
        Pd          = np.asarray(self.active_set_data[sample_id]['pd_data'], dtype=float)
        G           = self._generator_incidence_matrix
        PTDF        = self._ptdf_matrix
        branch_limit = self._branch_limit
        a  = (self.gencost[:, -2] / self.T_delta).reshape(self.ng, 1)
        b  = (self.gencost[:, -1] / self.T_delta).reshape(self.ng, 1)
        start_cost = self.gencost[:, 1]
        shut_cost  = self.gencost[:, 2]
        Ton  = self.Ton
        Toff = self.Toff

        # ── Decision variables ──────────────────────────────────────────────
        pg  = cp.Variable((self.ng, self.T))
        x   = cp.Variable((self.ng, self.T))
        coc = cp.Variable((self.ng, max(self.T - 1, 0)))
        constraints = [pg >= 0, x >= 0, x <= 1]
        if self.T > 1:
            constraints.append(coc >= 0)

        # ── Scalar rho parameters ───────────────────────────────────────────
        p_rho_primal = cp.Parameter(nonneg=True)
        p_rho_binary = cp.Parameter(nonneg=True)

        # ── Lambda magnitude parameters (nonneg=True: we store |lambda|) ───
        p_lambda_pb  = cp.Parameter(self.T,               nonneg=True)
        p_lambda_pgl = cp.Parameter((self.ng, self.T),    nonneg=True)
        p_lambda_pgu = cp.Parameter((self.ng, self.T),    nonneg=True)
        p_lambda_xl  = cp.Parameter((self.ng, self.T),    nonneg=True)
        p_lambda_xu  = cp.Parameter((self.ng, self.T),    nonneg=True)
        p_lambda_dcu = cp.Parameter((self.nl, self.T),    nonneg=True)
        p_lambda_dcl = cp.Parameter((self.nl, self.T),    nonneg=True)
        p_lambda_mon  = cp.Parameter((self.ng, Ton,  self.T), nonneg=True)
        p_lambda_moff = cp.Parameter((self.ng, Toff, self.T), nonneg=True)
        if self.T > 1:
            p_lambda_ru  = cp.Parameter((self.ng, self.T - 1), nonneg=True)
            p_lambda_rd  = cp.Parameter((self.ng, self.T - 1), nonneg=True)
            p_lambda_sc  = cp.Parameter((self.ng, self.T - 1), nonneg=True)
            p_lambda_shc = cp.Parameter((self.ng, self.T - 1), nonneg=True)
            p_lambda_coc = cp.Parameter((self.ng, self.T - 1), nonneg=True)
        else:
            p_lambda_ru = p_lambda_rd = p_lambda_sc = p_lambda_shc = p_lambda_coc = None

        # uc_matrix: fixed from dataset, or parameter if only runtime self.x is available
        uc_const = _get_uc_matrix_from_sample(self.active_set_data[sample_id], self.ng, self.T)
        if uc_const is not None:
            uc_const = np.asarray(uc_const, dtype=float)
            p_uc_matrix = None
            obj_binary = cp.sum(cp.abs(x - uc_const))
        else:
            p_uc_matrix = cp.Parameter((self.ng, self.T))
            obj_binary = cp.sum(cp.abs(x - p_uc_matrix))

        # ── Structural residual expressions (structure never changes) ───────
        pg_lower_expr  = cp.multiply(self.gen[:, PMIN].reshape(self.ng, 1), x) - pg
        pg_upper_expr  = pg - cp.multiply(self.gen[:, PMAX].reshape(self.ng, 1), x)

        # ── Objective terms ─────────────────────────────────────────────────
        obj_primal_terms = []
        obj_opt_terms    = []

        # Power balance (need per-slot |pb_expr| for element-wise lambda multiply)
        pb_abs_list = []
        for t in range(self.T):
            pb_expr = cp.sum(pg[:, t]) - float(np.sum(Pd[:, t]))
            pb_abs_list.append(cp.abs(pb_expr))
            obj_primal_terms.append(pb_abs_list[-1])
        obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_pb, cp.hstack(pb_abs_list))))

        # PG bounds
        obj_primal_terms.append(cp.sum(cp.pos(pg_lower_expr)))
        obj_primal_terms.append(cp.sum(cp.pos(pg_upper_expr)))
        obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_pgl, cp.abs(pg_lower_expr))))
        obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_pgu, cp.abs(pg_upper_expr))))
        obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_xl,  x)))
        obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_xu,  1 - x)))

        # Ramp
        if self.T > 1:
            ru_expr = (pg[:, 1:] - pg[:, :-1]
                       - cp.multiply(self.Ru.reshape(self.ng, 1), x[:, :-1])
                       - cp.multiply(self.Ru_co.reshape(self.ng, 1), 1 - x[:, :-1]))
            rd_expr = (pg[:, :-1] - pg[:, 1:]
                       - cp.multiply(self.Rd.reshape(self.ng, 1), x[:, 1:])
                       - cp.multiply(self.Rd_co.reshape(self.ng, 1), 1 - x[:, 1:]))
            obj_primal_terms.append(cp.sum(cp.pos(ru_expr)))
            obj_primal_terms.append(cp.sum(cp.pos(rd_expr)))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_ru, cp.abs(ru_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_rd, cp.abs(rd_expr))))

        # Min-on / min-off
        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                expr = x[g, 1:self.T - tau + 1] - x[g, :self.T - tau] - x[g, tau:]
                obj_primal_terms.append(cp.sum(cp.pos(expr)))
                obj_opt_terms.append(
                    cp.sum(cp.multiply(p_lambda_mon[g, tau - 1, :self.T - tau], cp.abs(expr))))
            for tau in range(1, self._min_down_horizon(g) + 1):
                expr = -x[g, 1:self.T - tau + 1] + x[g, :self.T - tau] - (1 - x[g, tau:])
                obj_primal_terms.append(cp.sum(cp.pos(expr)))
                obj_opt_terms.append(
                    cp.sum(cp.multiply(p_lambda_moff[g, tau - 1, :self.T - tau], cp.abs(expr))))

        # Start/shut costs + coc
        if self.T > 1:
            sc_expr  = cp.multiply(start_cost.reshape(self.ng, 1), x[:, 1:] - x[:, :-1]) - coc
            shc_expr = cp.multiply(shut_cost.reshape(self.ng, 1), x[:, :-1] - x[:, 1:]) - coc
            obj_primal_terms.append(cp.sum(cp.pos(sc_expr)))
            obj_primal_terms.append(cp.sum(cp.pos(shc_expr)))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_sc,  cp.abs(sc_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_shc, cp.abs(shc_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_coc, coc)))

        # DCPF
        for t in range(self.T):
            flow = PTDF @ (G @ pg[:, t] - Pd[:, t])
            dcpf_up  = flow - branch_limit
            dcpf_lo  = -flow - branch_limit
            obj_primal_terms.append(cp.sum(cp.pos(dcpf_up)))
            obj_primal_terms.append(cp.sum(cp.pos(dcpf_lo)))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_dcu[:, t], cp.abs(dcpf_up))))
            obj_opt_terms.append(cp.sum(cp.multiply(p_lambda_dcl[:, t], cp.abs(dcpf_lo))))

        cpower_expr = cp.multiply(a, pg) + cp.multiply(b, x)

        objective = cp.Minimize(
            p_rho_binary * obj_binary
            + p_rho_primal * _sum_scalar_terms(obj_primal_terms)
            + _sum_scalar_terms(obj_opt_terms)
        )
        problem = cp.Problem(objective, constraints)

        params = dict(
            p_rho_primal=p_rho_primal, p_rho_binary=p_rho_binary,
            p_lambda_pb=p_lambda_pb,
            p_lambda_pgl=p_lambda_pgl, p_lambda_pgu=p_lambda_pgu,
            p_lambda_xl=p_lambda_xl,   p_lambda_xu=p_lambda_xu,
            p_lambda_dcu=p_lambda_dcu, p_lambda_dcl=p_lambda_dcl,
            p_lambda_mon=p_lambda_mon, p_lambda_moff=p_lambda_moff,
            p_lambda_ru=p_lambda_ru,   p_lambda_rd=p_lambda_rd,
            p_lambda_sc=p_lambda_sc,   p_lambda_shc=p_lambda_shc,
            p_lambda_coc=p_lambda_coc,
            p_uc_matrix=p_uc_matrix,
        )
        vars_dict = {'pg': pg, 'x': x, 'coc': coc, 'cpower_expr': cpower_expr}
        return {'problem': problem, 'params': params, 'vars': vars_dict}

    def _build_pg_cvxpy_numeric(self, sample_id: int,
                                theta_values=None,
                                zeta_values=None,
                                union_analysis=None):
        """Build the CVXPY PG-block with numeric objective weights.

        CVXPY rejects products such as ``Parameter * abs(affine_expr)`` on some
        versions.  The PG block changes lambda/rho weights every BCD iteration,
        so for HiGHS we rebuild this LP with plain NumPy constants instead of
        caching a parameterized objective.
        """
        Pd = np.asarray(self.active_set_data[sample_id]['pd_data'], dtype=float)
        G = self._generator_incidence_matrix
        PTDF = self._ptdf_matrix
        branch_limit = self._branch_limit
        a = (self.gencost[:, -2] / self.T_delta).reshape(self.ng, 1)
        b = (self.gencost[:, -1] / self.T_delta).reshape(self.ng, 1)
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        lam = self.lambda_[sample_id]
        theta_values = theta_values or {}
        zeta_values = zeta_values or {}
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        pg = cp.Variable((self.ng, self.T))
        x = cp.Variable((self.ng, self.T))
        coc = cp.Variable((self.ng, max(self.T - 1, 0)))
        cpower = cp.Variable((self.ng, self.T), nonneg=True)
        constraints = [pg >= 0, x >= 0, x <= 1]
        if self.T > 1:
            constraints.append(coc >= 0)

        uc_const = _get_uc_matrix_from_sample(self.active_set_data[sample_id], self.ng, self.T)
        if uc_const is None:
            uc_const = np.asarray(self.x[sample_id], dtype=float)
        else:
            uc_const = np.asarray(uc_const, dtype=float)
        obj_binary = cp.sum(cp.abs(x - uc_const))

        pg_lower_expr = cp.multiply(self.gen[:, PMIN].reshape(self.ng, 1), x) - pg
        pg_upper_expr = pg - cp.multiply(self.gen[:, PMAX].reshape(self.ng, 1), x)

        obj_primal_terms = []
        obj_opt_terms = []

        w_pb = np.abs(lam['lambda_power_balance'])
        pb_abs_list = []
        for t in range(self.T):
            pb_expr = cp.sum(pg[:, t]) - float(np.sum(Pd[:, t]))
            pb_abs_list.append(cp.abs(pb_expr))
            obj_primal_terms.append(pb_abs_list[-1])
        obj_opt_terms.append(cp.sum(cp.multiply(w_pb, cp.hstack(pb_abs_list))))

        obj_primal_terms.append(cp.sum(cp.pos(pg_lower_expr)))
        obj_primal_terms.append(cp.sum(cp.pos(pg_upper_expr)))
        obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_pg_lower']), cp.abs(pg_lower_expr))))
        obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_pg_upper']), cp.abs(pg_upper_expr))))
        obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_x_lower']), x)))
        obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_x_upper']), 1 - x)))

        if self.T > 1:
            ru_expr = (
                pg[:, 1:] - pg[:, :-1]
                - cp.multiply(self.Ru.reshape(self.ng, 1), x[:, :-1])
                - cp.multiply(self.Ru_co.reshape(self.ng, 1), 1 - x[:, :-1])
            )
            rd_expr = (
                pg[:, :-1] - pg[:, 1:]
                - cp.multiply(self.Rd.reshape(self.ng, 1), x[:, 1:])
                - cp.multiply(self.Rd_co.reshape(self.ng, 1), 1 - x[:, 1:])
            )
            obj_primal_terms.append(cp.sum(cp.pos(ru_expr)))
            obj_primal_terms.append(cp.sum(cp.pos(rd_expr)))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_ramp_up']), cp.abs(ru_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_ramp_down']), cp.abs(rd_expr))))

        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                expr = x[g, 1:self.T - tau + 1] - x[g, :self.T - tau] - x[g, tau:]
                obj_primal_terms.append(cp.sum(cp.pos(expr)))
                weights = np.abs(lam['lambda_min_on'][g, tau - 1, :self.T - tau])
                obj_opt_terms.append(cp.sum(cp.multiply(weights, cp.abs(expr))))
            for tau in range(1, self._min_down_horizon(g) + 1):
                expr = -x[g, 1:self.T - tau + 1] + x[g, :self.T - tau] - (1 - x[g, tau:])
                obj_primal_terms.append(cp.sum(cp.pos(expr)))
                weights = np.abs(lam['lambda_min_off'][g, tau - 1, :self.T - tau])
                obj_opt_terms.append(cp.sum(cp.multiply(weights, cp.abs(expr))))

        if self.T > 1:
            sc_expr = cp.multiply(start_cost.reshape(self.ng, 1), x[:, 1:] - x[:, :-1]) - coc
            shc_expr = cp.multiply(shut_cost.reshape(self.ng, 1), x[:, :-1] - x[:, 1:]) - coc
            obj_primal_terms.append(cp.sum(cp.pos(sc_expr)))
            obj_primal_terms.append(cp.sum(cp.pos(shc_expr)))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_start_cost']), cp.abs(sc_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_shut_cost']), cp.abs(shc_expr))))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_coc_nonneg']), coc)))

        for t in range(self.T):
            flow = PTDF @ (G @ pg[:, t] - Pd[:, t])
            dcpf_up = flow - branch_limit
            dcpf_lo = -flow - branch_limit
            obj_primal_terms.append(cp.sum(cp.pos(dcpf_up)))
            obj_primal_terms.append(cp.sum(cp.pos(dcpf_lo)))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_dcpf_upper'][:, t]), cp.abs(dcpf_up))))
            obj_opt_terms.append(cp.sum(cp.multiply(np.abs(lam['lambda_dcpf_lower'][:, t]), cp.abs(dcpf_lo))))

        theta_scale = self._current_theta_curriculum_scale()
        if theta_scale > 0 and self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                lhs_expr = 0.0
                for coeff_info in constraint_info['nonzero_pg_coefficients']:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    lhs_expr += theta_scale * float(theta_values.get(theta_name, 0.0)) * x[unit_id, member_time]

                rhs = theta_scale * float(theta_values.get(self._theta_rhs_name(branch_id, time_slot), 1.0))
                direction = float(self._get_theta_constraint_direction(branch_id, time_slot))
                residual = lhs_expr - rhs
                obj_primal_terms.append(cp.pos(direction * residual))
                if hasattr(self, 'mu') and sample_id < len(self.mu) and branch_id < self.nl:
                    mu_weight = abs(float(self.mu[sample_id, branch_id, time_slot]))
                else:
                    mu_weight = float(getattr(self, 'dual_para_bound', 0.1))
                obj_opt_terms.append(mu_weight * cp.abs(residual))

        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            for constraint_info in union_analysis['union_zeta_constraints']:
                unit_id = constraint_info['unit_id']
                time_slot = constraint_info['time_slot']
                zeta_name = self._zeta_var_name(unit_id, time_slot)
                rhs_name = self._zeta_rhs_name(unit_id, time_slot)
                residual = float(zeta_values.get(zeta_name, 0.0)) * x[unit_id, time_slot] - float(
                    zeta_values.get(rhs_name, 1.0)
                )
                direction = float(self._get_zeta_constraint_direction(unit_id, time_slot))
                obj_primal_terms.append(cp.pos(direction * residual))
                if hasattr(self, 'ita') and sample_id < len(self.ita):
                    ita_weight = abs(float(self.ita[sample_id, unit_id, time_slot]))
                else:
                    ita_weight = float(getattr(
                        self,
                        'ita_dual_floor_init',
                        getattr(self, 'dual_para_bound', 0.1),
                    ))
                obj_opt_terms.append(ita_weight * cp.abs(residual))

        cpower_expr = cp.multiply(a, pg) + cp.multiply(b, x)
        constraints.append(cpower == cpower_expr)
        obj_prox = self._build_pg_block_prox_expr_cvxpy(sample_id, pg, x, coc)
        obj_primal = _sum_scalar_terms(obj_primal_terms)
        obj_opt = _sum_scalar_terms(obj_opt_terms)
        objective = cp.Minimize(
            float(self.rho_binary) * obj_binary
            + float(self.rho_primal) * obj_primal
            + float(self.rho_opt) * obj_opt
            + float(self.pg_block_prox_weight) * obj_prox
        )
        problem = cp.Problem(objective, constraints)
        return {
            'problem': problem,
            'vars': {'pg': pg, 'x': x, 'coc': coc, 'cpower': cpower},
            'objs': {
                'obj_primal': obj_primal,
                'obj_binary': obj_binary,
                'obj_opt': obj_opt,
                'obj_prox': obj_prox,
            },
        }

    def _iter_with_pg_block_cvxpy_highs(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        self._require_cvxpy_highs_main_backend()
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        cache = self._build_pg_cvxpy_numeric(
            sample_id,
            theta_values=theta_values,
            zeta_values=zeta_values,
            union_analysis=union_analysis,
        )
        problem = cache['problem']
        v = cache['vars']
        objs = cache.get('objs', {})
        _solve_with_cvxpy_highs(
            problem, verbose=False,
            threads=self.bcd_highs_threads,
            warm_start=False,
        )
        if not _problem_is_optimal(problem):
            print(f"PG block cvxpy_highs failed, status={problem.status}", flush=True)
            return None, None, None, None

        pg_sol = np.asarray(v['pg'].value, dtype=float)
        x_sol = np.asarray(v['x'].value, dtype=float)
        cpower_sol = np.asarray(v['cpower'].value, dtype=float)
        coc_sol = (
            np.asarray(v['coc'].value, dtype=float)
            if self.T > 1 else np.zeros((self.ng, 0), dtype=float)
        )
        if sample_id <= 2:
            def _cvx_value(expr):
                try:
                    if expr is None:
                        return 0.0
                    value = getattr(expr, 'value', expr)
                    if value is None:
                        return 0.0
                    return float(np.asarray(value, dtype=float).reshape(-1)[0])
                except Exception:
                    return 0.0

            print(
                f"pg_block, sample_id: {sample_id}, "
                f"backend: cvxpy_highs, status: {problem.status}, "
                f"obj_primal: {_cvx_value(objs.get('obj_primal')):.4f}, "
                f"obj_binary: {_cvx_value(objs.get('obj_binary')):.4f}, "
                f"obj_opt: {_cvx_value(objs.get('obj_opt')):.4f}, "
                f"obj_prox: {_cvx_value(objs.get('obj_prox')):.4f}, "
                f"objective: {float(problem.value):.4f}",
                flush=True,
            )
        return pg_sol, x_sol, cpower_sol, coc_sol

    def _iter_with_dual_block_cvxpy_highs(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """HiGHS/CVXPY implementation of the dual block.

        Minimises over (lambda, mu, ita) the same objective as the persistent
        Gurobi model in iter_with_dual_block:

            rho_dual_pg  * ||KKT_pg||_1
          + rho_dual_x   * ||KKT_x||_1      (includes theta/zeta contributions)
          + rho_dual_coc * ||KKT_coc||_1
          + rho_opt      * <violation_magnitudes, lambdas>
          + dual_block_prox_weight * ||delta_lambda / scale||_2^2

        All lambda variables are non-negative (except lambda_power_balance which
        is free). mu/ita bounds depend on the current floor/sign_relax state.
        """
        self._require_cvxpy_highs_main_backend()
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        Pd           = np.asarray(self.active_set_data[sample_id]['pd_data'], dtype=float)
        PTDF_G       = self._ptdf_g          # (nl, ng)
        G            = self._generator_incidence_matrix  # (nb, ng)
        branch_limit = self._branch_limit    # (nl,)
        Ru   = self.Ru;   Rd   = self.Rd
        Ru_co= self.Ru_co; Rd_co= self.Rd_co
        start_cost   = self.gencost[:, 1]
        shut_cost    = self.gencost[:, 2]
        Ton  = self.Ton
        Toff = self.Toff

        # ---- floor / sign-relax flags ----
        floor_active     = self._is_dual_floor_active()
        sign_relax_round = self._is_dual_sign_relaxation_round()

        # ================================================================
        # Decision variables
        # ================================================================
        lambda_pb   = cp.Variable(self.T)                                    # free
        lambda_pgl  = cp.Variable((self.ng, self.T),   nonneg=True)
        lambda_pgu  = cp.Variable((self.ng, self.T),   nonneg=True)
        lambda_ru   = cp.Variable((self.ng, self.T-1), nonneg=True) if self.T > 1 else None
        lambda_rd   = cp.Variable((self.ng, self.T-1), nonneg=True) if self.T > 1 else None

        class _MinTimeDualVarBlock:
            """2D/1D CVXPY variable wrapper with legacy 3-index access."""

            def __init__(self, ng: int, width: int, horizon: int, name: str):
                self.ng = int(ng)
                self.width = int(width)
                self.horizon = int(horizon)
                self._vars = [
                    [
                        cp.Variable(self.horizon, nonneg=True, name=f"{name}_{g}_{tau}")
                        for tau in range(self.width)
                    ]
                    for g in range(self.ng)
                ]

            def __getitem__(self, key):
                if not isinstance(key, tuple):
                    raise TypeError("min-time dual variables require tuple indexing")
                if len(key) == 3:
                    g, tau, t = key
                    return self._vars[int(g)][int(tau)][int(t)]
                if len(key) == 2:
                    g, tau = key
                    return self._vars[int(g)][int(tau)]
                raise IndexError(key)

            def value_array(self) -> np.ndarray:
                out = np.zeros((self.ng, self.width, self.horizon), dtype=float)
                for g in range(self.ng):
                    for tau in range(self.width):
                        val = self._vars[g][tau].value
                        if val is not None:
                            out[g, tau, :] = np.asarray(val, dtype=float)
                return out

        lambda_mon = _MinTimeDualVarBlock(self.ng, Ton, self.T, "lambda_min_on")
        lambda_moff = _MinTimeDualVarBlock(self.ng, Toff, self.T, "lambda_min_off")
        lambda_sc   = cp.Variable((self.ng, self.T-1), nonneg=True) if self.T > 1 else None
        lambda_shc  = cp.Variable((self.ng, self.T-1), nonneg=True) if self.T > 1 else None
        lambda_coc_nonneg = cp.Variable((self.ng, self.T-1), nonneg=True) if self.T > 1 else None
        lambda_cpower = cp.Variable((self.ng, self.T), nonneg=True)
        lambda_dcu  = cp.Variable((self.nl, self.T), nonneg=True)
        lambda_dcl  = cp.Variable((self.nl, self.T), nonneg=True)
        lambda_xu   = cp.Variable((self.ng, self.T), nonneg=True)
        lambda_xl   = cp.Variable((self.ng, self.T), nonneg=True)

        # mu/ita: free when sign-relax, otherwise non-negative
        if floor_active and sign_relax_round:
            mu  = cp.Variable((self.nl, self.T))
            ita = cp.Variable((self.ng, self.T))
        else:
            mu  = cp.Variable((self.nl, self.T), nonneg=True)
            ita = cp.Variable((self.ng, self.T), nonneg=True)

        mu_abs  = cp.Variable((self.nl, self.T), nonneg=True)
        ita_abs = cp.Variable((self.ng, self.T), nonneg=True)

        # ================================================================
        # Constraints
        # ================================================================
        constraints = [
            # cpower fixed to 1 (KKT stationarity for cpower variable)
            lambda_cpower == 1.0,
            # abs linearisation
            mu_abs  >=  mu,
            mu_abs  >= -mu,
            ita_abs >=  ita,
            ita_abs >= -ita,
        ]

        mu_floor  = float(getattr(self, 'mu_dual_floor_init',  0.0)) if floor_active else 0.0
        ita_floor = float(getattr(self, 'ita_dual_floor_init', 0.0)) if floor_active else 0.0
        if mu_floor > 0:
            if sign_relax_round:
                constraints.append(mu_abs >= mu_floor)
            else:
                constraints.append(mu >= mu_floor)
        if ita_floor > 0:
            if sign_relax_round:
                constraints.append(ita_abs >= ita_floor)
            else:
                constraints.append(ita >= ita_floor)
        zeta_cap_penalty_obj = gp.LinExpr()
        zeta_cap, zeta_cap_weight = self._current_zeta_ita_cap()
        zeta_cap_penalty = 0.0
        if zeta_cap is not None and zeta_cap_weight > 0:
            zeta_ita_excess = cp.Variable((self.ng, self.T), nonneg=True)
            constraints.append(zeta_ita_excess >= ita_abs - float(zeta_cap))
            zeta_cap_penalty = float(zeta_cap_weight) * cp.sum(zeta_ita_excess)

        # ================================================================
        # KKT stationarity expressions
        # ================================================================
        tv = theta_values or {}
        zv = zeta_values or {}
        theta_scale = self._current_theta_curriculum_scale()

        # -- pg stationarity: cost_g/T - lambda_pb[t] - lambda_pgl + lambda_pgu
        #                     + ramp terms + PTDF_G^T (lambda_dcu - lambda_dcl) --
        de_pg_list = []
        for g in range(self.ng):
            cost_g = float(self.gencost[g, -2]) / self.T_delta
            for t in range(self.T):
                de = cost_g - lambda_pb[t] - lambda_pgl[g, t] + lambda_pgu[g, t]
                if self.T > 1:
                    if t > 0:
                        de = de + lambda_ru[g, t-1] - lambda_rd[g, t-1]
                    if t < self.T - 1:
                        de = de - lambda_ru[g, t]   + lambda_rd[g, t]
                # DCPF dual contribution (only non-zero PTDF entries)
                for l in range(self.nl):
                    coeff = float(PTDF_G[l, g])
                    if abs(coeff) > 1e-10:
                        de = de + coeff * (lambda_dcu[l, t] - lambda_dcl[l, t])
                de_pg_list.append(de)

        # -- x stationarity: reuse the same source-of-truth helper --
        de_x_list = []
        for g in range(self.ng):
            pmin = float(self.gen[g, PMIN]); pmax = float(self.gen[g, PMAX])
            rd_delta = float(Rd_co[g] - Rd[g]); ru_delta = float(Ru_co[g] - Ru[g])
            for t in range(self.T):
                de = self._build_x_dual_stationarity_expr(
                    g=g, t=t,
                    lambda_cpower=lambda_cpower,
                    lambda_x_upper=lambda_xu,
                    lambda_x_lower=lambda_xl,
                    lambda_pg_lower=lambda_pgl,
                    lambda_pg_upper=lambda_pgu,
                    lambda_ramp_down=lambda_rd,
                    lambda_ramp_up=lambda_ru,
                    lambda_min_on=lambda_mon,
                    lambda_min_off=lambda_moff,
                    lambda_start_cost=lambda_sc,
                    lambda_shut_cost=lambda_shc,
                    fixed_cost=float(self.gencost[g, -1]) / self.T_delta,
                    pmin=pmin, pmax=pmax,
                    rd_delta=rd_delta, ru_delta=ru_delta,
                    start_cost=float(start_cost[g]),
                    shut_cost=float(shut_cost[g]),
                    Ton=Ton, Toff=Toff,
                )
                # theta contributions: direction * theta_val * mu[bid, ts]
                if theta_scale > 0 and self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
                    for ci in union_analysis['union_constraints']:
                        bid  = ci['branch_id']
                        ts   = ci['time_slot']
                        direc = self._get_theta_constraint_direction(bid, ts)
                        for coeff_info in ci['nonzero_pg_coefficients']:
                            uid = coeff_info['unit_id']
                            mt  = self._theta_member_time_index(ci, coeff_info)
                            if uid != g or mt != t:
                                continue
                            tname = self._theta_var_name(bid, uid, mt)
                            theta_val = theta_scale * float(tv.get(tname, 0.0))
                            if bid < self.nl and abs(theta_val) > 1e-12:
                                de = de + direc * theta_val * mu[bid, ts]
                # zeta contributions: direction * zeta_val * ita[uid, ts]
                if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
                    for ci in union_analysis['union_zeta_constraints']:
                        uid = ci['unit_id']
                        ts  = ci['time_slot']
                        if uid != g or ts != t:
                            continue
                        zname  = f'zeta_unit_{uid}_time_{ts}'
                        direc  = self._get_zeta_constraint_direction(uid, ts)
                        zeta_val = float(zv.get(zname, 0.0))
                        if abs(zeta_val) > 1e-12:
                            de = de + direc * zeta_val * ita[uid, ts]
                de_x_list.append(de)

        # -- coc stationarity: 1 - lambda_sc[g,t] - lambda_shc[g,t] - lambda_coc[g,t] --
        de_coc_list = []
        if self.T > 1:
            for g in range(self.ng):
                for t in range(self.T - 1):
                    de = 1 - lambda_sc[g, t] - lambda_shc[g, t] - lambda_coc_nonneg[g, t]
                    de_coc_list.append(de)

        obj_dual_pg  = cp.sum([cp.abs(d) for d in de_pg_list])  if de_pg_list  else 0.0
        obj_dual_x   = cp.sum([cp.abs(d) for d in de_x_list])   if de_x_list   else 0.0
        obj_dual_coc = cp.sum([cp.abs(d) for d in de_coc_list]) if de_coc_list else 0.0

        # ================================================================
        # obj_opt: violation magnitudes (scalars) × lambda variables
        # Mirrors _update_dual_model_objective exactly.
        # ================================================================
        opt_terms = []
        pg_arr = self.pg[sample_id]   # (ng, T)
        x_arr  = self.x[sample_id]    # (ng, T)
        coc_arr= self.coc[sample_id]  # (ng, T-1)

        # auxiliary: |lambda_pb[t]| is handled as lambda_pb_abs in Gurobi;
        # in CVXPY we use cp.abs(lambda_pb) which CVXPY expands to auxiliary vars.
        for t in range(self.T):
            pb_viol = abs(float(np.sum(pg_arr[:, t])) - float(np.sum(Pd[:, t])))
            if pb_viol > 1e-10:
                opt_terms.append(pb_viol * cp.abs(lambda_pb[t]))
            for g in range(self.ng):
                pgl_v = abs(float(pg_arr[g, t]) - self.gen[g, PMIN] * float(x_arr[g, t]))
                if pgl_v > 1e-10:
                    opt_terms.append(pgl_v * lambda_pgl[g, t])
                pgu_v = abs(self.gen[g, PMAX] * float(x_arr[g, t]) - float(pg_arr[g, t]))
                if pgu_v > 1e-10:
                    opt_terms.append(pgu_v * lambda_pgu[g, t])
                xl_v = abs(float(x_arr[g, t]))
                if xl_v > 1e-10:
                    opt_terms.append(xl_v * lambda_xl[g, t])
                xu_v = abs(float(x_arr[g, t]) - 1.0)
                if xu_v > 1e-10:
                    opt_terms.append(xu_v * lambda_xu[g, t])

        if self.T > 1:
            for t in range(1, self.T):
                for g in range(self.ng):
                    ru_v = abs(float(pg_arr[g, t]) - float(pg_arr[g, t-1])
                               - (Ru[g]*float(x_arr[g, t-1]) + Ru_co[g]*(1-float(x_arr[g, t-1]))))
                    if ru_v > 1e-10:
                        opt_terms.append(ru_v * lambda_ru[g, t-1])
                    rd_v = abs(float(pg_arr[g, t-1]) - float(pg_arr[g, t])
                               - (Rd[g]*float(x_arr[g, t]) + Rd_co[g]*(1-float(x_arr[g, t]))))
                    if rd_v > 1e-10:
                        opt_terms.append(rd_v * lambda_rd[g, t-1])

        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    v = abs(float(x_arr[g, t1+1]) - float(x_arr[g, t1]) - float(x_arr[g, t1+tau]))
                    if v > 1e-10:
                        opt_terms.append(v * lambda_mon[g, tau-1, t1])
            for tau in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    v = abs(-float(x_arr[g, t1+1]) + float(x_arr[g, t1]) - 1.0 + float(x_arr[g, t1+tau]))
                    if v > 1e-10:
                        opt_terms.append(v * lambda_moff[g, tau-1, t1])

        if self.T > 1:
            for t in range(1, self.T):
                for g in range(self.ng):
                    coc_v = abs(float(coc_arr[g, t-1]))
                    if coc_v > 1e-10:
                        opt_terms.append(coc_v * lambda_coc_nonneg[g, t-1])
                    sc_v = abs(float(coc_arr[g, t-1])
                               - start_cost[g]*(float(x_arr[g, t]) - float(x_arr[g, t-1])))
                    if sc_v > 1e-10:
                        opt_terms.append(sc_v * lambda_sc[g, t-1])
                    shc_v = abs(float(coc_arr[g, t-1])
                                - shut_cost[g]*(float(x_arr[g, t-1]) - float(x_arr[g, t])))
                    if shc_v > 1e-10:
                        opt_terms.append(shc_v * lambda_shc[g, t-1])

        # DCPF flow violation × lambda_dcu / lambda_dcl
        for t in range(self.T):
            flow_t = self._ptdf_matrix @ (G @ pg_arr[:, t] - Pd[:, t])
            for l in range(self.nl):
                dcu_v = abs(float(flow_t[l]) - float(branch_limit[l]))
                if dcu_v > 1e-10:
                    opt_terms.append(dcu_v * lambda_dcu[l, t])
                dcl_v = abs(float(flow_t[l]) + float(branch_limit[l]))
                if dcl_v > 1e-10:
                    opt_terms.append(dcl_v * lambda_dcl[l, t])

        # theta/zeta parametric constraint violations × mu_abs / ita_abs
        if theta_scale > 0 and self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            for ci in union_analysis['union_constraints']:
                bid = ci['branch_id']
                ts  = ci['time_slot']
                rhs_name = self._theta_rhs_name(bid, ts)
                rhs  = theta_scale * float(tv.get(rhs_name, 1.0))
                lhs  = 0.0
                for coeff_info in ci['nonzero_pg_coefficients']:
                    uid   = coeff_info['unit_id']
                    mt    = self._theta_member_time_index(ci, coeff_info)
                    tname = self._theta_var_name(bid, uid, mt)
                    lhs  += theta_scale * float(tv.get(tname, 0.0)) * float(x_arr[uid, mt])
                para_abs = abs(lhs - rhs)
                if para_abs > 1e-10:
                    if bid < self.nl:
                        opt_terms.append(para_abs * mu_abs[bid, ts])
                    else:
                        opt_terms.append(para_abs * float(getattr(self, 'dual_para_bound', 0.1)))

        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            for ci in union_analysis['union_zeta_constraints']:
                uid      = ci['unit_id']
                ts       = ci['time_slot']
                rhs_name = f'zeta_rhs_unit_{uid}_time_{ts}'
                rhs      = float(zv.get(rhs_name, 1.0))
                zname    = f'zeta_unit_{uid}_time_{ts}'
                zeta_val = float(zv.get(zname, 0.0))
                para_abs = abs(zeta_val * float(x_arr[uid, ts]) - rhs)
                if para_abs > 1e-10:
                    opt_terms.append(para_abs * ita_abs[uid, ts])

        obj_opt = _sum_scalar_terms(opt_terms) if opt_terms else 0.0

        # ================================================================
        # Prox term (QP): penalise deviation from previous iterate
        # ================================================================
        prox_terms = []
        if self.dual_block_prox_weight > 0:
            prev_lam = (self.lambda_[sample_id]
                        if self.lambda_[sample_id] is not None
                        else self._create_empty_lambda_dict())
            prev_mu  = self.mu[sample_id]
            prev_ita = self.ita[sample_id]

            for t in range(self.T):
                pv = float(prev_lam['lambda_power_balance'][t])
                sc = max(1.0, abs(pv))
                prox_terms.append(cp.square((lambda_pb[t] - pv) / sc))

            for g in range(self.ng):
                for t in range(self.T):
                    for key, var in (('lambda_pg_lower', lambda_pgl),
                                     ('lambda_pg_upper', lambda_pgu),
                                     ('lambda_x_upper',  lambda_xu),
                                     ('lambda_x_lower',  lambda_xl)):
                        pv = float(prev_lam[key][g, t])
                        sc = max(1.0, abs(pv))
                        prox_terms.append(cp.square((var[g, t] - pv) / sc))
                    pv = float(prev_ita[g, t])
                    sc = max(1.0, abs(pv))
                    prox_terms.append(cp.square((ita[g, t] - pv) / sc))
                if self.T > 1:
                    for t in range(self.T - 1):
                        for key, var in (('lambda_ramp_up',    lambda_ru),
                                         ('lambda_ramp_down',  lambda_rd),
                                         ('lambda_start_cost', lambda_sc),
                                         ('lambda_shut_cost',  lambda_shc),
                                         ('lambda_coc_nonneg', lambda_coc_nonneg)):
                            pv = float(prev_lam[key][g, t])
                            sc = max(1.0, abs(pv))
                            prox_terms.append(cp.square((var[g, t] - pv) / sc))
                    for tau in range(self._min_up_horizon(g)):
                        for t in range(self.T):
                            pv = float(prev_lam['lambda_min_on'][g, tau, t])
                            sc = max(1.0, abs(pv))
                            prox_terms.append(cp.square((lambda_mon[g, tau, t] - pv) / sc))
                    for tau in range(self._min_down_horizon(g)):
                        for t in range(self.T):
                            pv = float(prev_lam['lambda_min_off'][g, tau, t])
                            sc = max(1.0, abs(pv))
                            prox_terms.append(cp.square((lambda_moff[g, tau, t] - pv) / sc))

            for l in range(self.nl):
                for t in range(self.T):
                    for key, var in (('lambda_dcpf_upper', lambda_dcu),
                                     ('lambda_dcpf_lower', lambda_dcl)):
                        pv = float(prev_lam[key][l, t])
                        sc = max(1.0, abs(pv))
                        prox_terms.append(cp.square((var[l, t] - pv) / sc))
                    pv = float(prev_mu[l, t])
                    sc = max(1.0, abs(pv))
                    prox_terms.append(cp.square((mu[l, t] - pv) / sc))

        obj_prox = cp.sum(prox_terms) if prox_terms else 0.0

        # ================================================================
        # Build and solve
        # ================================================================
        objective = cp.Minimize(
            self.rho_dual_pg  * obj_dual_pg
            + self.rho_dual_x  * obj_dual_x
            + self.rho_dual_coc* obj_dual_coc
            + self.rho_opt     * obj_opt
            + zeta_cap_penalty
            + self.dual_block_prox_weight * obj_prox
        )
        problem = cp.Problem(objective, constraints)
        _solve_with_cvxpy_highs(problem, verbose=False, threads=self.bcd_highs_threads)

        if not _problem_is_optimal(problem):
            print(f"❌ Dual block cvxpy_highs failed, status={problem.status}", flush=True)
            return None, None, None

        # ================================================================
        # Extract solution
        # ================================================================
        def _v(arr):
            """Return ndarray copy; fall back to zeros on None value."""
            if arr is None:
                return np.zeros(0)
            v = arr.value
            return np.array(v, dtype=float) if v is not None else np.zeros(np.shape(arr))

        lambda_sol = {
            'lambda_power_balance': _v(lambda_pb),
            'lambda_pg_lower':      _v(lambda_pgl),
            'lambda_pg_upper':      _v(lambda_pgu),
            'lambda_ramp_up':       _v(lambda_ru)  if self.T > 1 else np.zeros((self.ng, 0)),
            'lambda_ramp_down':     _v(lambda_rd)  if self.T > 1 else np.zeros((self.ng, 0)),
            'lambda_min_on':        self._zero_unused_min_up_entries(lambda_mon.value_array()),
            'lambda_min_off':       self._zero_unused_min_down_entries(lambda_moff.value_array()),
            'lambda_start_cost':    _v(lambda_sc)  if self.T > 1 else np.zeros((self.ng, 0)),
            'lambda_shut_cost':     _v(lambda_shc) if self.T > 1 else np.zeros((self.ng, 0)),
            'lambda_coc_nonneg':    _v(lambda_coc_nonneg) if self.T > 1 else np.zeros((self.ng, 0)),
            'lambda_cpower':        _v(lambda_cpower),
            'lambda_dcpf_upper':    _v(lambda_dcu),
            'lambda_dcpf_lower':    _v(lambda_dcl),
            'lambda_x_upper':       _v(lambda_xu),
            'lambda_x_lower':       _v(lambda_xl),
        }
        mu_sol  = _v(mu)
        ita_sol = _v(ita)

        if sample_id <= 2:
            print(
                f"[BCD][cvxpy_highs] dual_block sample_id={sample_id}, "
                f"status={problem.status}",
                flush=True,
            )
        return lambda_sol, mu_sol, ita_sol

    # ------------------------------------------------------------------
    # Persistent pg-block model helpers
    # ------------------------------------------------------------------

    def _build_pg_model(self, sample_id: int, union_analysis=None):
        """一次性建立 pg 块 Gurobi 模型（变量 + 约束结构）。
        返回 (model, vars_dict)；vars_dict 保存所有变量引用及 theta/zeta 更新信息。
        """
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('pg_block')
        model.Params.OutputFlag = 0

        pg      = model.addVars(self.ng, self.T,   lb=0,                         name='pg')
        x       = model.addVars(self.ng, self.T,   vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc     = model.addVars(self.ng, self.T-1, lb=0,                         name='coc')
        cpower  = model.addVars(self.ng, self.T,   lb=0,                         name='cpower')

        power_balance_viol = model.addVars(self.T,              lb=0, name='power_balance_viol')
        pg_lower_viol      = model.addVars(self.ng, self.T,     lb=0, name='pg_lower_viol')
        pg_upper_viol      = model.addVars(self.ng, self.T,     lb=0, name='pg_upper_viol')
        pg_lower_abs       = model.addVars(self.ng, self.T,     lb=0, name='pg_lower_abs')
        pg_upper_abs       = model.addVars(self.ng, self.T,     lb=0, name='pg_upper_abs')
        ramp_up_viol       = model.addVars(self.ng, self.T-1,   lb=0, name='ramp_up_viol')
        ramp_down_viol     = model.addVars(self.ng, self.T-1,   lb=0, name='ramp_down_viol')
        ramp_up_abs        = model.addVars(self.ng, self.T-1,   lb=0, name='ramp_up_abs')
        ramp_down_abs      = model.addVars(self.ng, self.T-1,   lb=0, name='ramp_down_abs')

        Ton  = self.Ton
        Toff = self.Toff
        min_on_viol   = model.addVars(self.ng, Ton,  self.T,   lb=0, name='min_on_viol')
        min_off_viol  = model.addVars(self.ng, Toff, self.T,   lb=0, name='min_off_viol')
        min_on_abs    = model.addVars(self.ng, Ton,  self.T,   lb=0, name='min_on_abs')
        min_off_abs   = model.addVars(self.ng, Toff, self.T,   lb=0, name='min_off_abs')
        start_cost_viol = model.addVars(self.ng, self.T-1, lb=0, name='start_cost_viol')
        shut_cost_viol  = model.addVars(self.ng, self.T-1, lb=0, name='shut_cost_viol')
        start_cost_abs  = model.addVars(self.ng, self.T-1, lb=0, name='start_cost_abs')
        shut_cost_abs   = model.addVars(self.ng, self.T-1, lb=0, name='shut_cost_abs')
        dcpf_upper_viol = model.addVars(self.nl, self.T,   lb=0, name='dcpf_upper_viol')
        dcpf_upper_abs  = model.addVars(self.nl, self.T,   lb=0, name='dcpf_upper_abs')
        dcpf_lower_viol = model.addVars(self.nl, self.T,   lb=0, name='dcpf_lower_viol')
        dcpf_lower_abs  = model.addVars(self.nl, self.T,   lb=0, name='dcpf_lower_abs')
        x_binary_dev    = model.addVars(self.ng, self.T,   lb=0, name='x_binary_dev')

        # 静态目标项（结构不变，仅需在 setObjective 时乘以当前 rho）
        obj_primal = gp.LinExpr()
        obj_binary = gp.LinExpr()

        Ru    = self.Ru;  Rd    = self.Rd
        Ru_co = self.Ru_co;  Rd_co = self.Rd_co

        # 功率平衡约束
        for t in range(self.T):
            pb_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_viol[t] >= pb_expr,  name=f'pb_pos_{t}')
            model.addConstr(power_balance_viol[t] >= -pb_expr, name=f'pb_neg_{t}')
            obj_primal += power_balance_viol[t]

            for g in range(self.ng):
                lower_e = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                upper_e = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_lower_viol[g, t] >= lower_e,  name=f'pg_lo_viol_{g}_{t}')
                model.addConstr(pg_upper_viol[g, t] >= upper_e,  name=f'pg_up_viol_{g}_{t}')
                model.addConstr(pg_lower_abs[g, t]  >= lower_e,  name=f'pg_lo_abs1_{g}_{t}')
                model.addConstr(pg_lower_abs[g, t]  >= -lower_e, name=f'pg_lo_abs2_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t]  >= upper_e,  name=f'pg_up_abs1_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t]  >= -upper_e, name=f'pg_up_abs2_{g}_{t}')
                obj_primal += pg_lower_viol[g, t] + pg_upper_viol[g, t]

        # 爬坡约束
        for t in range(1, self.T):
            for g in range(self.ng):
                ru_e = pg[g,t] - pg[g,t-1] - Ru[g]*x[g,t-1] - Ru_co[g]*(1-x[g,t-1])
                rd_e = pg[g,t-1] - pg[g,t] - Rd[g]*x[g,t] - Rd_co[g]*(1-x[g,t])
                model.addConstr(ramp_up_viol[g,t-1]   >= ru_e,  name=f'ru_viol_{g}_{t}')
                model.addConstr(ramp_up_abs[g,t-1]    >= ru_e,  name=f'ru_abs1_{g}_{t}')
                model.addConstr(ramp_up_abs[g,t-1]    >= -ru_e, name=f'ru_abs2_{g}_{t}')
                model.addConstr(ramp_down_viol[g,t-1] >= rd_e,  name=f'rd_viol_{g}_{t}')
                model.addConstr(ramp_down_abs[g,t-1]  >= rd_e,  name=f'rd_abs1_{g}_{t}')
                model.addConstr(ramp_down_abs[g,t-1]  >= -rd_e, name=f'rd_abs2_{g}_{t}')
                obj_primal += ramp_up_viol[g,t-1] + ramp_down_viol[g,t-1]

        # 最小开/关机时间约束
        for g in range(self.ng):
            for t in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - t):
                    e = x[g,t1+1] - x[g,t1] - x[g,t1+t]
                    model.addConstr(min_on_viol[g,t-1,t1] >= e,  name=f'mon_viol_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g,t-1,t1]  >= e,  name=f'mon_abs1_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g,t-1,t1]  >= -e, name=f'mon_abs2_{g}_{t}_{t1}')
                    obj_primal += min_on_viol[g,t-1,t1]
            for t in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - t):
                    e = -x[g,t1+1] + x[g,t1] - (1 - x[g,t1+t])
                    model.addConstr(min_off_viol[g,t-1,t1] >= e,  name=f'moff_viol_{g}_{t}_{t1}')
                    model.addConstr(min_off_abs[g,t-1,t1]  >= e,  name=f'moff_abs1_{g}_{t}_{t1}')
                    model.addConstr(min_off_abs[g,t-1,t1]  >= -e, name=f'moff_abs2_{g}_{t}_{t1}')
                    obj_primal += min_off_viol[g,t-1,t1]

        # 启停成本约束
        start_cost = self.gencost[:, 1]
        shut_cost  = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                sc_e  = start_cost[g] * (x[g,t] - x[g,t-1]) - coc[g,t-1]
                shc_e = shut_cost[g]  * (x[g,t-1] - x[g,t]) - coc[g,t-1]
                model.addConstr(start_cost_viol[g,t-1] >= sc_e,   name=f'sc_viol_{g}_{t}')
                model.addConstr(start_cost_abs[g,t-1]  >= sc_e,   name=f'sc_abs1_{g}_{t}')
                model.addConstr(start_cost_abs[g,t-1]  >= -sc_e,  name=f'sc_abs2_{g}_{t}')
                model.addConstr(shut_cost_viol[g,t-1]  >= shc_e,  name=f'shc_viol_{g}_{t}')
                model.addConstr(shut_cost_abs[g,t-1]   >= shc_e,  name=f'shc_abs1_{g}_{t}')
                model.addConstr(shut_cost_abs[g,t-1]   >= -shc_e, name=f'shc_abs2_{g}_{t}')
                obj_primal += start_cost_viol[g,t-1] + shut_cost_viol[g,t-1]

        # 发电成本定义
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(
                    cpower[g,t] == self.gencost[g,-2]/self.T_delta * pg[g,t]
                                 + self.gencost[g,-1]/self.T_delta * x[g,t],
                    name=f'cpower_{g}_{t}')

        # 潮流约束
        G           = self._generator_incidence_matrix
        PTDF        = self._ptdf_matrix
        branch_limit = self._branch_limit
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr(dcpf_upper_viol[l,t] >= flow[l] - branch_limit[l],  name=f'dcpf_up_viol_{l}_{t}')
                model.addConstr(dcpf_lower_viol[l,t] >= -flow[l] - branch_limit[l], name=f'dcpf_lo_viol_{l}_{t}')
                model.addConstr(dcpf_upper_abs[l,t]  >= flow[l] - branch_limit[l],  name=f'dcpf_up_abs1_{l}_{t}')
                model.addConstr(dcpf_upper_abs[l,t]  >= -flow[l]+branch_limit[l],   name=f'dcpf_up_abs2_{l}_{t}')
                model.addConstr(dcpf_lower_abs[l,t]  >= -flow[l]-branch_limit[l],   name=f'dcpf_lo_abs1_{l}_{t}')
                model.addConstr(dcpf_lower_abs[l,t]  >= flow[l]+branch_limit[l],    name=f'dcpf_lo_abs2_{l}_{t}')
                obj_primal += dcpf_upper_viol[l,t] + dcpf_lower_viol[l,t]

        # 二进制偏差约束（使用数据中的静态 unit_commitment_matrix）
        ucm = _get_uc_matrix_from_sample(self.active_set_data[sample_id], self.ng, self.T)
        if ucm is None:
            ucm = self.x[sample_id]
        for g in range(self.ng):
            for t in range(self.T):
                dev = x[g,t] - float(ucm[g,t])
                model.addConstr(x_binary_dev[g,t] >= dev,  name=f'xdev_pos_{g}_{t}')
                model.addConstr(x_binary_dev[g,t] >= -dev, name=f'xdev_neg_{g}_{t}')
                obj_binary += x_binary_dev[g,t]

        # 参数化 theta/zeta 约束（结构固定，系数通过 chgCoeff 更新）
        theta_update_infos = []
        zeta_update_infos  = []
        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            theta_update_infos = self._build_pg_model_parametric_theta(
                model, x, sample_id, union_analysis, PTDF, branch_limit)
            for info in theta_update_infos:
                obj_primal += info['viol_var']
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            zeta_update_infos = self._build_pg_model_parametric_zeta(
                model, x, sample_id, union_analysis)
            for info in zeta_update_infos:
                obj_primal += info['viol_var']

        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.update()

        vars_dict = {
            'pg': pg, 'x': x, 'coc': coc, 'cpower': cpower,
            'power_balance_viol': power_balance_viol,
            'pg_lower_abs': pg_lower_abs, 'pg_upper_abs': pg_upper_abs,
            'ramp_up_abs': ramp_up_abs, 'ramp_down_abs': ramp_down_abs,
            'min_on_abs': min_on_abs, 'min_off_abs': min_off_abs,
            'start_cost_abs': start_cost_abs, 'shut_cost_abs': shut_cost_abs,
            'dcpf_upper_abs': dcpf_upper_abs, 'dcpf_lower_abs': dcpf_lower_abs,
            'x_binary_dev': x_binary_dev,
            'obj_primal': obj_primal,
            'obj_binary': obj_binary,
            'Ton': Ton, 'Toff': Toff,
            'theta_update_infos': theta_update_infos,
            'zeta_update_infos': zeta_update_infos,
        }
        return model, vars_dict

    def _build_pg_model_parametric_theta(self, model, x, sample_id, union_analysis, PTDF, branch_limit):
        """为 theta 参数化约束添加变量和约束，返回 update_infos 列表（供 chgCoeff 使用）。"""
        if not union_analysis or 'union_constraints' not in union_analysis:
            return []
        update_infos = []
        for ci in union_analysis['union_constraints']:
            branch_id  = ci['branch_id']
            time_slot  = ci['time_slot']
            direction  = self._get_theta_constraint_direction(branch_id, time_slot)
            rhs_name   = self._theta_rhs_name(branch_id, time_slot)
            terms = []
            for coeff_info in ci['nonzero_pg_coefficients']:
                unit_id     = coeff_info['unit_id']
                member_time = self._theta_member_time_index(ci, coeff_info)
                theta_name  = self._theta_var_name(branch_id, unit_id, member_time)
                terms.append((x[unit_id, member_time], theta_name))

            viol_var = model.addVar(lb=0, name=f'para_viol_{branch_id}_{time_slot}')
            abs_var  = model.addVar(lb=0, name=f'para_abs_{branch_id}_{time_slot}')

            # 以零系数建立约束，后续 chgCoeff 填入真实值
            # 约束形式：viol_var - direction*Σ(theta*x) >= -direction*rhs
            c_viol    = model.addConstr(viol_var    >= 0.0, name=f'para_viol_c_{branch_id}_{time_slot}')
            c_abs_pos = model.addConstr(abs_var     >= 0.0, name=f'para_abs_pos_{branch_id}_{time_slot}')
            c_abs_neg = model.addConstr(abs_var     >= 0.0, name=f'para_abs_neg_{branch_id}_{time_slot}')

            update_infos.append({
                'branch_id': branch_id, 'time_slot': time_slot,
                'direction': direction, 'rhs_name': rhs_name,
                'terms': terms,
                'c_viol': c_viol, 'c_abs_pos': c_abs_pos, 'c_abs_neg': c_abs_neg,
                'viol_var': viol_var, 'abs_var': abs_var,
            })
        return update_infos

    def _build_pg_model_parametric_zeta(self, model, x, sample_id, union_analysis):
        """为 zeta 参数化约束添加变量和约束，返回 update_infos 列表。"""
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return []
        update_infos = []
        for ci in union_analysis['union_zeta_constraints']:
            unit_id   = ci['unit_id']
            time_slot = ci['time_slot']
            direction = self._get_zeta_constraint_direction(unit_id, time_slot)
            rhs_name  = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            terms = [(x[unit_id, time_slot], zeta_name)]

            viol_var = model.addVar(lb=0, name=f'zeta_viol_{unit_id}_{time_slot}')
            abs_var  = model.addVar(lb=0, name=f'zeta_abs_{unit_id}_{time_slot}')

            c_viol    = model.addConstr(viol_var >= 0.0, name=f'zeta_viol_c_{unit_id}_{time_slot}')
            c_abs_pos = model.addConstr(abs_var  >= 0.0, name=f'zeta_abs_pos_{unit_id}_{time_slot}')
            c_abs_neg = model.addConstr(abs_var  >= 0.0, name=f'zeta_abs_neg_{unit_id}_{time_slot}')

            update_infos.append({
                'unit_id': unit_id, 'time_slot': time_slot,
                'direction': direction, 'rhs_name': rhs_name,
                'terms': terms,
                'c_viol': c_viol, 'c_abs_pos': c_abs_pos, 'c_abs_neg': c_abs_neg,
                'viol_var': viol_var, 'abs_var': abs_var,
            })
        return update_infos

    def _apply_pg_model_parametric_update(self, model, theta_values, zeta_values,
                                          theta_update_infos, zeta_update_infos):
        """通过 chgCoeff 更新 theta/zeta 参数化约束的系数和 RHS。"""
        tv = theta_values or {}
        for info in theta_update_infos:
            d   = info['direction']
            rhs = float(tv.get(info['rhs_name'], 1.0))
            # 清除旧的变量系数再设新值（通过 chgCoeff 直接覆盖）
            lhs_expr = gp.LinExpr()
            for x_var, theta_name in info['terms']:
                theta = float(tv.get(theta_name, 0.0))
                lhs_expr += theta * x_var
            # viol: viol_var - direction*lhs >= -direction*rhs
            # → coeff of x_var in LHS: -direction*theta
            for x_var, theta_name in info['terms']:
                theta = float(tv.get(theta_name, 0.0))
                model.chgCoeff(info['c_viol'],    x_var,  -d * theta)
                model.chgCoeff(info['c_abs_pos'], x_var,  -theta)
                model.chgCoeff(info['c_abs_neg'], x_var,   theta)
            info['c_viol'].RHS    = -d * rhs
            info['c_abs_pos'].RHS = -rhs
            info['c_abs_neg'].RHS =  rhs

        zv = zeta_values or {}
        for info in zeta_update_infos:
            d   = info['direction']
            rhs = float(zv.get(info['rhs_name'], 1.0))
            for x_var, zeta_name in info['terms']:
                zeta = float(zv.get(zeta_name, 0.0))
                model.chgCoeff(info['c_viol'],    x_var,  -d * zeta)
                model.chgCoeff(info['c_abs_pos'], x_var,  -zeta)
                model.chgCoeff(info['c_abs_neg'], x_var,   zeta)
            info['c_viol'].RHS    = -d * rhs
            info['c_abs_pos'].RHS = -rhs
            info['c_abs_neg'].RHS =  rhs

    def _update_pg_model_objective(self, sample_id: int, model, vars_dict,
                                   theta_values=None, zeta_values=None, union_analysis=None):
        """每次迭代：更新 theta/zeta 约束系数，然后重建目标函数并调用 setObjective。"""
        # 1. 更新参数化约束系数（chgCoeff + RHS）
        if vars_dict['theta_update_infos'] or vars_dict['zeta_update_infos']:
            self._apply_pg_model_parametric_update(
                model, theta_values, zeta_values,
                vars_dict['theta_update_infos'],
                vars_dict['zeta_update_infos'],
            )

        pg  = vars_dict['pg']
        x   = vars_dict['x']
        coc = vars_dict['coc']
        power_balance_viol = vars_dict['power_balance_viol']
        pg_lower_abs       = vars_dict['pg_lower_abs']
        pg_upper_abs       = vars_dict['pg_upper_abs']
        ramp_up_abs        = vars_dict['ramp_up_abs']
        ramp_down_abs      = vars_dict['ramp_down_abs']
        min_on_abs         = vars_dict['min_on_abs']
        min_off_abs        = vars_dict['min_off_abs']
        start_cost_abs     = vars_dict['start_cost_abs']
        shut_cost_abs      = vars_dict['shut_cost_abs']
        dcpf_upper_abs     = vars_dict['dcpf_upper_abs']
        dcpf_lower_abs     = vars_dict['dcpf_lower_abs']
        Ton  = vars_dict['Ton']
        Toff = vars_dict['Toff']
        lam  = self.lambda_[sample_id]

        # 2. 重建 obj_opt（动态对偶系数）
        obj_opt = gp.LinExpr()
        for t in range(self.T):
            obj_opt += power_balance_viol[t] * abs(lam['lambda_power_balance'][t])
            for g in range(self.ng):
                obj_opt += pg_lower_abs[g,t] * abs(lam['lambda_pg_lower'][g,t])
                obj_opt += pg_upper_abs[g,t] * abs(lam['lambda_pg_upper'][g,t])
                obj_opt += x[g,t]        * abs(lam['lambda_x_lower'][g,t])
                obj_opt += (1 - x[g,t]) * abs(lam['lambda_x_upper'][g,t])
        for t in range(1, self.T):
            for g in range(self.ng):
                obj_opt += ramp_up_abs[g,t-1]   * abs(lam['lambda_ramp_up'][g,t-1])
                obj_opt += ramp_down_abs[g,t-1] * abs(lam['lambda_ramp_down'][g,t-1])
        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    obj_opt += min_on_abs[g,tau-1,t1] * abs(lam['lambda_min_on'][g,tau-1,t1])
            for tau in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    obj_opt += min_off_abs[g,tau-1,t1] * abs(lam['lambda_min_off'][g,tau-1,t1])
        for t in range(1, self.T):
            for g in range(self.ng):
                obj_opt += start_cost_abs[g,t-1] * abs(lam['lambda_start_cost'][g,t-1])
                obj_opt += shut_cost_abs[g,t-1]  * abs(lam['lambda_shut_cost'][g,t-1])
                obj_opt += coc[g,t-1]            * abs(lam['lambda_coc_nonneg'][g,t-1])
        for t in range(self.T):
            for l in range(self.branch.shape[0]):
                obj_opt += dcpf_upper_abs[l,t] * abs(lam['lambda_dcpf_upper'][l,t])
                obj_opt += dcpf_lower_abs[l,t] * abs(lam['lambda_dcpf_lower'][l,t])

        # theta/zeta abs_var × |mu/ita|
        for info in vars_dict['theta_update_infos']:
            bid, ts = info['branch_id'], info['time_slot']
            if bid < self.nl:
                mu_v = abs(float(self.mu[sample_id, bid, ts]))
            else:
                mu_v = getattr(self, 'dual_para_bound', 0.1)
            obj_opt += info['abs_var'] * mu_v
        for info in vars_dict['zeta_update_infos']:
            uid, ts = info['unit_id'], info['time_slot']
            obj_opt += info['abs_var'] * abs(float(self.ita[sample_id, uid, ts]))

        # 3. Prox 项（QP）
        obj_prox = self._build_pg_block_prox_obj(model, sample_id, pg, x, coc)

        # 4. 设置完整目标函数
        model.setObjective(
            self.rho_binary  * vars_dict['obj_binary']
            + self.rho_primal * vars_dict['obj_primal']
            + self.rho_opt    * obj_opt
            + self.pg_block_prox_weight * obj_prox,
            GRB.MINIMIZE,
        )
        vars_dict['obj_opt'] = obj_opt
        vars_dict['obj_prox'] = obj_prox

    # ------------------------------------------------------------------

    def iter_with_pg_block(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """
        迭代PG块（完整实现，参考uc_dfsm_bcd.py）
        更新x, pg等原始变量
        """
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            return self._iter_with_pg_block_cvxpy_highs(sample_id, theta_values, zeta_values, union_analysis)

        # --- 复用或新建 persistent 模型 ---
        union_id = id(union_analysis)
        if (sample_id not in self._pg_models
                or self._pg_model_union_id.get(sample_id) != union_id):
            if sample_id in self._pg_models:
                try:
                    self._pg_models[sample_id].dispose()
                except Exception:
                    pass
            model, vars_dict = self._build_pg_model(sample_id, union_analysis)
            self._pg_models[sample_id]         = model
            self._pg_vars[sample_id]           = vars_dict
            self._pg_model_union_id[sample_id] = union_id
        else:
            model     = self._pg_models[sample_id]
            vars_dict = self._pg_vars[sample_id]

        # --- 每次迭代只更新目标函数 ---
        self._update_pg_model_objective(
            sample_id, model, vars_dict, theta_values, zeta_values, union_analysis)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            pg      = vars_dict['pg']
            x       = vars_dict['x']
            cpower  = vars_dict['cpower']
            coc     = vars_dict['coc']
            pg_sol     = np.array([[pg[g,t].X      for t in range(self.T)]   for g in range(self.ng)])
            x_sol      = np.array([[x[g,t].X       for t in range(self.T)]   for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g,t].X  for t in range(self.T)]   for g in range(self.ng)])
            coc_sol    = np.array([[coc[g,t].X     for t in range(self.T-1)] for g in range(self.ng)])

            if sample_id <= 2:
                obj_primal_v = vars_dict['obj_primal'].getValue()
                obj_binary_v = vars_dict['obj_binary'].getValue()
                obj_opt_v = vars_dict.get('obj_opt', 0.0)
                obj_opt_v = obj_opt_v.getValue() if hasattr(obj_opt_v, 'getValue') else float(obj_opt_v)
                obj_prox_v = vars_dict.get('obj_prox', 0.0)
                obj_prox_v = obj_prox_v.getValue() if hasattr(obj_prox_v, 'getValue') else float(obj_prox_v)
                print(
                    f"pg_block, sample_id: {sample_id}, "
                    f"backend: gurobi, status: {model.status}, "
                    f"obj_primal: {obj_primal_v:.4f}, "
                    f"obj_binary: {obj_binary_v:.4f}, "
                    f"obj_opt: {obj_opt_v:.4f}, "
                    f"obj_prox: {obj_prox_v:.4f}, "
                    f"objective: {model.ObjVal:.4f}",
                    flush=True,
                )
            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ PG块模型求解失败，状态: {model.status}", flush=True)
            return None, None, None, None
    
    # ------------------------------------------------------------------
    # Persistent dual-block model helpers
    # ------------------------------------------------------------------

    def _build_dual_model(self, sample_id: int, union_analysis=None,
                          floor_active: bool = False, sign_relax_round: bool = False):
        """一次性建立 dual 块 Gurobi 模型（变量 + 约束结构）。
        返回 (model, vars_dict)。theta/zeta 的 mu 贡献系数以零初始化，
        后续通过 chgCoeff 更新。
        """
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('dual_block')
        model.Params.OutputFlag = 0

        lambda_power_balance = model.addVars(self.T,           lb=-GRB.INFINITY, name='lambda_pb')
        lambda_pg_lower      = model.addVars(self.ng, self.T,  lb=0, name='lambda_pg_lower')
        lambda_pg_upper      = model.addVars(self.ng, self.T,  lb=0, name='lambda_pg_upper')
        lambda_ramp_up       = model.addVars(self.ng, self.T-1,lb=0, name='lambda_ramp_up')
        lambda_ramp_down     = model.addVars(self.ng, self.T-1,lb=0, name='lambda_ramp_down')

        Ton  = self.Ton
        Toff = self.Toff
        lambda_min_on    = model.addVars(self.ng, Ton,  self.T,   lb=0, name='lambda_min_on')
        lambda_min_off   = model.addVars(self.ng, Toff, self.T,   lb=0, name='lambda_min_off')
        lambda_start_cost= model.addVars(self.ng, self.T-1,        lb=0, name='lambda_sc')
        lambda_shut_cost = model.addVars(self.ng, self.T-1,        lb=0, name='lambda_shc')
        lambda_coc_nonneg= model.addVars(self.ng, self.T-1,        lb=0, name='lambda_coc')
        lambda_cpower    = model.addVars(self.ng, self.T,           lb=0, name='lambda_cpower')
        lambda_dcpf_upper= model.addVars(self.nl, self.T,           lb=0, name='lambda_dcpf_up')
        lambda_dcpf_lower= model.addVars(self.nl, self.T,           lb=0, name='lambda_dcpf_lo')
        lambda_x_upper   = model.addVars(self.ng, self.T,           lb=0, name='lambda_x_up')
        lambda_x_lower   = model.addVars(self.ng, self.T,           lb=0, name='lambda_x_lo')

        mu_lb  = -GRB.INFINITY if (floor_active and sign_relax_round) else 0.0
        ita_lb = -GRB.INFINITY if (floor_active and sign_relax_round) else 0.0
        mu     = model.addVars(self.nl, self.T, lb=mu_lb,  name='mu')
        ita    = model.addVars(self.ng, self.T, lb=ita_lb, name='ita')
        mu_abs = model.addVars(self.nl, self.T, lb=0,      name='mu_abs')
        ita_abs= model.addVars(self.ng, self.T, lb=0,      name='ita_abs')
        zeta_ita_cap_excess = model.addVars(self.ng, self.T, lb=0, name='zeta_ita_cap_excess')
        zeta_ita_cap_constrs = {}
        mu_max = model.addVar(lb=0, name='mu_max')
        ita_max= model.addVar(lb=0, name='ita_max')
        deadband = 100

        for l in range(self.nl):
            for t in range(self.T):
                model.addConstr(mu_abs[l,t] >= mu[l,t],  name=f'mu_abs_pos_{l}_{t}')
                model.addConstr(mu_abs[l,t] >= -mu[l,t], name=f'mu_abs_neg_{l}_{t}')
                if floor_active:
                    if sign_relax_round:
                        model.addConstr(mu_abs[l,t] >= self.mu_dual_floor_init, name=f'mu_abs_lb_{l}_{t}')
                    else:
                        model.addConstr(mu[l,t] >= self.mu_dual_floor_init, name=f'mu_lb_{l}_{t}')
                model.addConstr(mu_max >= mu_abs[l,t] - deadband, name=f'mu_max_{l}_{t}')
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(ita_abs[g,t] >= ita[g,t],  name=f'ita_abs_pos_{g}_{t}')
                model.addConstr(ita_abs[g,t] >= -ita[g,t], name=f'ita_abs_neg_{g}_{t}')
                if floor_active:
                    if sign_relax_round:
                        model.addConstr(ita_abs[g,t] >= self.ita_dual_floor_init, name=f'ita_abs_lb_{g}_{t}')
                    else:
                        model.addConstr(ita[g,t] >= self.ita_dual_floor_init, name=f'ita_lb_{g}_{t}')
                zeta_ita_cap_constrs[g, t] = model.addConstr(
                    zeta_ita_cap_excess[g, t] - ita_abs[g, t] >= 0.0,
                    name=f'zeta_ita_cap_excess_lb_{g}_{t}',
                )
                model.addConstr(ita_max >= ita_abs[g,t] - deadband, name=f'ita_max_{g}_{t}')

        PTDF_G = self._ptdf_g
        PTDF   = self._ptdf_matrix
        branch_limit = self._branch_limit
        Ru  = self.Ru;  Rd  = self.Rd
        Ru_co = self.Ru_co; Rd_co = self.Rd_co
        start_cost = self.gencost[:, 1]
        shut_cost  = self.gencost[:, 2]

        obj_dual_pg  = gp.LinExpr()
        obj_dual_x   = gp.LinExpr()
        obj_dual_coc = gp.LinExpr()

        # pg 驻点约束（全静态系数）
        for g in range(self.ng):
            for t in range(self.T):
                de = self.gencost[g,-2] / self.T_delta
                de -= lambda_power_balance[t]
                de -= lambda_pg_lower[g,t]
                de += lambda_pg_upper[g,t]
                if t > 0:
                    de += lambda_ramp_up[g,t-1]
                    de -= lambda_ramp_down[g,t-1]
                if t < self.T - 1:
                    de -= lambda_ramp_up[g,t]
                    de += lambda_ramp_down[g,t]
                ptdfg_col = PTDF_G[:, g]
                for l in range(self.branch.shape[0]):
                    de += ptdfg_col[l] * (lambda_dcpf_upper[l,t] - lambda_dcpf_lower[l,t])
                abs_v = model.addVar(lb=0, name=f'dabs_pg_{g}_{t}')
                model.addConstr(abs_v >= de,  name=f'dabs_pg_pos_{g}_{t}')
                model.addConstr(abs_v >= -de, name=f'dabs_pg_neg_{g}_{t}')
                obj_dual_pg += abs_v

        # x 驻点约束（theta/zeta 贡献以零系数初始化，后续 chgCoeff 更新）
        dual_x_theta_terms = []  # for chgCoeff updates
        for g in range(self.ng):
            for t in range(self.T):
                de = self._build_x_dual_stationarity_expr(
                    g=g, t=t,
                    lambda_cpower=lambda_cpower,
                    lambda_x_upper=lambda_x_upper,
                    lambda_x_lower=lambda_x_lower,
                    lambda_pg_lower=lambda_pg_lower,
                    lambda_pg_upper=lambda_pg_upper,
                    lambda_ramp_down=lambda_ramp_down,
                    lambda_ramp_up=lambda_ramp_up,
                    lambda_min_on=lambda_min_on,
                    lambda_min_off=lambda_min_off,
                    lambda_start_cost=lambda_start_cost,
                    lambda_shut_cost=lambda_shut_cost,
                    fixed_cost=self.gencost[g,-1]/self.T_delta,
                    pmin=self.gen[g,PMIN], pmax=self.gen[g,PMAX],
                    rd_delta=Rd_co[g]-Rd[g], ru_delta=Ru_co[g]-Ru[g],
                    start_cost=start_cost[g], shut_cost=shut_cost[g],
                    Ton=Ton, Toff=Toff,
                )
                # 收集 theta 贡献信息（但不加入 de，初始系数为 0）
                theta_terms = []
                if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
                    for ci in union_analysis['union_constraints']:
                        bid  = ci['branch_id']
                        ts   = ci['time_slot']
                        direc = self._get_theta_constraint_direction(bid, ts)
                        for coeff_info in ci['nonzero_pg_coefficients']:
                            uid  = coeff_info['unit_id']
                            mt   = self._theta_member_time_index(ci, coeff_info)
                            if uid != g or mt != t:
                                continue
                            tname = self._theta_var_name(bid, uid, mt)
                            if bid < self.nl:
                                theta_terms.append((mu[bid, ts], direc, tname))
                            else:
                                default_mu = getattr(self, 'dual_para_bound', 0.1)
                                de += 0.0  # handled as constant; skip chgCoeff for virtual branch

                zeta_terms = []
                if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
                    for ci in union_analysis['union_zeta_constraints']:
                        uid  = ci['unit_id']
                        ts   = ci['time_slot']
                        if uid != g or ts != t:
                            continue
                        zname = f'zeta_unit_{uid}_time_{ts}'
                        direc  = self._get_zeta_constraint_direction(uid, ts)
                        zeta_terms.append((ita[uid, ts], direc, zname))

                abs_v = model.addVar(lb=0, name=f'dabs_x_{g}_{t}')
                c_pos = model.addConstr(abs_v >= de,  name=f'dabs_x_pos_{g}_{t}')
                c_neg = model.addConstr(abs_v >= -de, name=f'dabs_x_neg_{g}_{t}')
                obj_dual_x += abs_v

                if theta_terms or zeta_terms:
                    dual_x_theta_terms.append({
                        'g': g, 't': t,
                        'c_pos': c_pos, 'c_neg': c_neg,
                        'theta_terms': theta_terms,
                        'zeta_terms':  zeta_terms,
                    })

        # cpower 驻点（等式约束）
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(lambda_cpower[g,t] == 1, name=f'dcpower_{g}_{t}')

        # coc 驻点
        for g in range(self.ng):
            for t in range(self.T-1):
                de = 1 - lambda_start_cost[g,t] - lambda_shut_cost[g,t] - lambda_coc_nonneg[g,t]
                abs_v = model.addVar(lb=0, name=f'dabs_coc_{g}_{t}')
                model.addConstr(abs_v >= de,  name=f'dabs_coc_pos_{g}_{t}')
                model.addConstr(abs_v >= -de, name=f'dabs_coc_neg_{g}_{t}')
                obj_dual_coc += abs_v

        # 为 obj_opt 预声明 lambda_power_balance_abs 变量（结构静态）
        lambda_pb_abs = model.addVars(self.T, lb=0, name='lambda_pb_abs')
        for t in range(self.T):
            model.addConstr(lambda_pb_abs[t] >= lambda_power_balance[t],  name=f'lpb_abs_pos_{t}')
            model.addConstr(lambda_pb_abs[t] >= -lambda_power_balance[t], name=f'lpb_abs_neg_{t}')

        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.update()

        vars_dict = {
            'lambda_power_balance': lambda_power_balance,
            'lambda_pg_lower': lambda_pg_lower, 'lambda_pg_upper': lambda_pg_upper,
            'lambda_ramp_up': lambda_ramp_up,   'lambda_ramp_down': lambda_ramp_down,
            'lambda_min_on': lambda_min_on,     'lambda_min_off': lambda_min_off,
            'lambda_start_cost': lambda_start_cost, 'lambda_shut_cost': lambda_shut_cost,
            'lambda_coc_nonneg': lambda_coc_nonneg,
            'lambda_cpower': lambda_cpower,
            'lambda_dcpf_upper': lambda_dcpf_upper, 'lambda_dcpf_lower': lambda_dcpf_lower,
            'lambda_x_upper': lambda_x_upper,   'lambda_x_lower': lambda_x_lower,
            'mu': mu, 'ita': ita, 'mu_abs': mu_abs, 'ita_abs': ita_abs,
            'zeta_ita_cap_excess': zeta_ita_cap_excess,
            'zeta_ita_cap_constrs': zeta_ita_cap_constrs,
            'mu_max': mu_max, 'ita_max': ita_max,
            'lambda_pb_abs': lambda_pb_abs,
            'obj_dual_pg': obj_dual_pg, 'obj_dual_x': obj_dual_x, 'obj_dual_coc': obj_dual_coc,
            'Ton': Ton, 'Toff': Toff,
            'dual_x_theta_terms': dual_x_theta_terms,
            'Pd': Pd,
        }
        return model, vars_dict

    def _apply_dual_model_theta_chgcoeff(self, model, theta_values, zeta_values, dual_x_theta_terms):
        """通过 chgCoeff 更新 x 驻点约束中 theta/zeta 的 mu/ita 系数。"""
        tv = theta_values or {}
        zv = zeta_values or {}
        theta_scale = self._current_theta_curriculum_scale()
        for info in dual_x_theta_terms:
            c_pos = info['c_pos']
            c_neg = info['c_neg']
            for mu_var, direction, tname in info['theta_terms']:
                theta = theta_scale * float(tv.get(tname, 0.0))
                # de includes direction*theta*mu[b,t]
                # abs_v >= de  →  coeff of mu in LHS: -direction*theta
                model.chgCoeff(c_pos, mu_var, -direction * theta)
                model.chgCoeff(c_neg, mu_var,  direction * theta)
            for ita_var, direction, zname in info['zeta_terms']:
                zeta = float(zv.get(zname, 0.0))
                model.chgCoeff(c_pos, ita_var, -direction * zeta)
                model.chgCoeff(c_neg, ita_var,  direction * zeta)

    def _update_dual_model_objective(self, sample_id: int, model, vars_dict,
                                     theta_values=None, zeta_values=None, union_analysis=None):
        """每次迭代：更新 theta/zeta 驻点系数，重建 obj_opt 和 obj_prox，调用 setObjective。"""
        # 1. 更新 x 驻点约束的 theta/zeta 系数
        if vars_dict['dual_x_theta_terms']:
            self._apply_dual_model_theta_chgcoeff(
                model, theta_values, zeta_values, vars_dict['dual_x_theta_terms'])

        Pd = vars_dict['Pd']
        lam_pb   = vars_dict['lambda_power_balance']
        lam_pgl  = vars_dict['lambda_pg_lower']
        lam_pgu  = vars_dict['lambda_pg_upper']
        lam_ru   = vars_dict['lambda_ramp_up']
        lam_rd   = vars_dict['lambda_ramp_down']
        lam_mon  = vars_dict['lambda_min_on']
        lam_moff = vars_dict['lambda_min_off']
        lam_sc   = vars_dict['lambda_start_cost']
        lam_shc  = vars_dict['lambda_shut_cost']
        lam_coc  = vars_dict['lambda_coc_nonneg']
        lam_dcu  = vars_dict['lambda_dcpf_upper']
        lam_dcl  = vars_dict['lambda_dcpf_lower']
        lam_xu   = vars_dict['lambda_x_upper']
        lam_xl   = vars_dict['lambda_x_lower']
        lam_pb_abs = vars_dict['lambda_pb_abs']
        mu       = vars_dict['mu']
        ita      = vars_dict['ita']
        mu_abs   = vars_dict['mu_abs']
        ita_abs  = vars_dict['ita_abs']
        mu_max   = vars_dict['mu_max']
        ita_max  = vars_dict['ita_max']
        Ton  = vars_dict['Ton']
        Toff = vars_dict['Toff']
        PTDF         = self._ptdf_matrix
        G            = self._generator_incidence_matrix
        branch_limit = self._branch_limit
        start_cost   = self.gencost[:, 1]
        shut_cost    = self.gencost[:, 2]
        Ru   = self.Ru;  Rd   = self.Rd
        Ru_co= self.Ru_co; Rd_co= self.Rd_co

        # 2. 重建 obj_opt（动态：使用当前 self.pg/x/coc 的违反量作为标量系数）
        obj_opt = gp.LinExpr()

        for t in range(self.T):
            pb_viol = abs(sum(float(self.pg[sample_id, g, t]) for g in range(self.ng)) - np.sum(Pd[:, t]))
            if pb_viol > 1e-10:
                obj_opt += pb_viol * lam_pb_abs[t]
            for g in range(self.ng):
                pgl_v = abs(float(self.pg[sample_id,g,t]) - self.gen[g,PMIN]*float(self.x[sample_id,g,t]))
                if pgl_v > 1e-10:
                    obj_opt += pgl_v * lam_pgl[g,t]
                pgu_v = abs(self.gen[g,PMAX]*float(self.x[sample_id,g,t]) - float(self.pg[sample_id,g,t]))
                if pgu_v > 1e-10:
                    obj_opt += pgu_v * lam_pgu[g,t]

        for t in range(1, self.T):
            for g in range(self.ng):
                ru_v = abs(float(self.pg[sample_id,g,t]) - float(self.pg[sample_id,g,t-1])
                           - (Ru[g]*float(self.x[sample_id,g,t-1]) + Ru_co[g]*(1-float(self.x[sample_id,g,t-1]))))
                if ru_v > 1e-10:
                    obj_opt += ru_v * lam_ru[g,t-1]
                rd_v = abs(float(self.pg[sample_id,g,t-1]) - float(self.pg[sample_id,g,t])
                           - (Rd[g]*float(self.x[sample_id,g,t]) + Rd_co[g]*(1-float(self.x[sample_id,g,t]))))
                if rd_v > 1e-10:
                    obj_opt += rd_v * lam_rd[g,t-1]

        for g in range(self.ng):
            for tau in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    v = abs(float(self.x[sample_id,g,t1+1]) - float(self.x[sample_id,g,t1]) - float(self.x[sample_id,g,t1+tau]))
                    if v > 1e-10:
                        obj_opt += v * lam_mon[g,tau-1,t1]
            for tau in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - tau):
                    v = abs(-float(self.x[sample_id,g,t1+1]) + float(self.x[sample_id,g,t1]) - 1 + float(self.x[sample_id,g,t1+tau]))
                    if v > 1e-10:
                        obj_opt += v * lam_moff[g,tau-1,t1]

        for t in range(1, self.T):
            for g in range(self.ng):
                coc_v = abs(float(self.coc[sample_id,g,t-1]))
                if coc_v > 1e-10:
                    obj_opt += coc_v * lam_coc[g,t-1]
                sc_v = abs(float(self.coc[sample_id,g,t-1]) - start_cost[g]*(float(self.x[sample_id,g,t]) - float(self.x[sample_id,g,t-1])))
                if sc_v > 1e-10:
                    obj_opt += sc_v * lam_sc[g,t-1]
                shc_v = abs(float(self.coc[sample_id,g,t-1]) - shut_cost[g]*(float(self.x[sample_id,g,t-1]) - float(self.x[sample_id,g,t])))
                if shc_v > 1e-10:
                    obj_opt += shc_v * lam_shc[g,t-1]

        for t in range(self.T):
            flow = PTDF @ (G @ np.array([float(self.pg[sample_id,g,t]) for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                dcu_v = abs(flow[l] - branch_limit[l])
                dcl_v = abs(flow[l] + branch_limit[l])
                if dcu_v > 1e-10:
                    obj_opt += dcu_v * lam_dcu[l,t]
                if dcl_v > 1e-10:
                    obj_opt += dcl_v * lam_dcl[l,t]

        for t in range(self.T):
            for g in range(self.ng):
                xl_v = abs(float(self.x[sample_id,g,t]))
                if xl_v > 1e-10:
                    obj_opt += xl_v * lam_xl[g,t]
                xu_v = abs(float(self.x[sample_id,g,t]) - 1)
                if xu_v > 1e-10:
                    obj_opt += xu_v * lam_xu[g,t]

        # theta/zeta obj_opt 项（用 self.x[s] 标量计算 lhs - rhs）
        tv = theta_values or {}
        zv = zeta_values or {}
        theta_scale = self._current_theta_curriculum_scale()
        if theta_scale > 0 and self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            for ci in union_analysis['union_constraints']:
                bid = ci['branch_id']
                ts  = ci['time_slot']
                rhs_name = self._theta_rhs_name(bid, ts)
                rhs = theta_scale * float(tv.get(rhs_name, 1.0))
                lhs = 0.0
                for coeff_info in ci['nonzero_pg_coefficients']:
                    uid  = coeff_info['unit_id']
                    mt   = self._theta_member_time_index(ci, coeff_info)
                    tname = self._theta_var_name(bid, uid, mt)
                    theta = theta_scale * float(tv.get(tname, 0.0))
                    lhs  += theta * float(self.x[sample_id, uid, mt])
                para_abs = abs(lhs - rhs)
                if para_abs > 1e-10:
                    if bid < self.nl:
                        obj_opt += para_abs * mu_abs[bid, ts]
                    else:
                        obj_opt += para_abs * getattr(self, 'dual_para_bound', 0.1)
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            for ci in union_analysis['union_zeta_constraints']:
                uid  = ci['unit_id']
                ts   = ci['time_slot']
                rhs_name = f'zeta_rhs_unit_{uid}_time_{ts}'
                rhs = float(zv.get(rhs_name, 1.0))
                zname = f'zeta_unit_{uid}_time_{ts}'
                zeta = float(zv.get(zname, 0.0))
                lhs  = zeta * float(self.x[sample_id, uid, ts])
                para_abs = abs(lhs - rhs)
                if para_abs > 1e-10:
                    obj_opt += para_abs * ita_abs[uid, ts]

        # 3. Prox 项（QP）
        zeta_cap_penalty_obj = gp.LinExpr()
        zeta_cap, zeta_cap_weight = self._current_zeta_ita_cap()
        if zeta_cap is not None:
            for constr in vars_dict.get('zeta_ita_cap_constrs', {}).values():
                constr.RHS = -float(zeta_cap)
            if zeta_cap_weight > 0:
                zeta_ita_cap_excess = vars_dict.get('zeta_ita_cap_excess')
                if zeta_ita_cap_excess is not None:
                    for g in range(self.ng):
                        for t in range(self.T):
                            zeta_cap_penalty_obj += float(zeta_cap_weight) * zeta_ita_cap_excess[g, t]

        obj_dual_prox = self._build_dual_block_prox_obj(
            model, sample_id,
            lam_pb, lam_pgl, lam_pgu, lam_ru, lam_rd,
            lam_mon, lam_moff, lam_sc, lam_shc, lam_coc,
            lam_dcu, lam_dcl, lam_xu, lam_xl,
            mu, ita, Ton, Toff,
        )

        penal_mu  = 0 * mu_max
        penal_ita = 0 * ita_max

        model.setObjective(
            self.rho_dual_pg  * vars_dict['obj_dual_pg']
            + self.rho_dual_x  * vars_dict['obj_dual_x']
            + self.rho_dual_coc* vars_dict['obj_dual_coc']
            + self.rho_opt     * obj_opt
            + zeta_cap_penalty_obj
            + self.dual_block_prox_weight * obj_dual_prox
            + penal_mu + penal_ita,
            GRB.MINIMIZE,
        )

    # ------------------------------------------------------------------

    def iter_with_dual_block(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """
        迭代对偶块（完整实现，参考uc_dfsm_bcd.py）
        更新对偶变量lambda, mu, ita
        """
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            return self._iter_with_dual_block_cvxpy_highs(sample_id, theta_values, zeta_values, union_analysis)

        # 确定当前 floor / sign_relax 状态
        dual_decay_round_ = self._current_dual_decay_round()
        floor_active      = self.iter_number < dual_decay_round_
        sign_relax_round  = self._is_dual_sign_relaxation_round()
        current_state     = (floor_active, sign_relax_round)

        # --- 复用或新建 persistent 模型 ---
        union_id = id(union_analysis)
        need_rebuild = (
            sample_id not in self._dual_models
            or self._dual_model_state.get(sample_id) != current_state
            or self._dual_model_union_id.get(sample_id) != union_id
        )
        if need_rebuild:
            if sample_id in self._dual_models:
                try:
                    self._dual_models[sample_id].dispose()
                except Exception:
                    pass
            model, vars_dict = self._build_dual_model(
                sample_id, union_analysis, floor_active, sign_relax_round)
            self._dual_models[sample_id]      = model
            self._dual_vars[sample_id]        = vars_dict
            self._dual_model_state[sample_id] = current_state
            self._dual_model_union_id[sample_id] = union_id
        else:
            model     = self._dual_models[sample_id]
            vars_dict = self._dual_vars[sample_id]

        # --- 每次迭代只更新目标函数（含 theta/zeta chgCoeff）---
        self._update_dual_model_objective(
            sample_id, model, vars_dict, theta_values, zeta_values, union_analysis)

        model.optimize()

        Ton  = vars_dict['Ton']
        Toff = vars_dict['Toff']
        lam_pb  = vars_dict['lambda_power_balance']
        lam_pgl = vars_dict['lambda_pg_lower']
        lam_pgu = vars_dict['lambda_pg_upper']
        lam_ru  = vars_dict['lambda_ramp_up']
        lam_rd  = vars_dict['lambda_ramp_down']
        lam_mon = vars_dict['lambda_min_on']
        lam_moff= vars_dict['lambda_min_off']
        lam_sc  = vars_dict['lambda_start_cost']
        lam_shc = vars_dict['lambda_shut_cost']
        lam_coc = vars_dict['lambda_coc_nonneg']
        lam_cp  = vars_dict['lambda_cpower']
        lam_dcu = vars_dict['lambda_dcpf_upper']
        lam_dcl = vars_dict['lambda_dcpf_lower']
        lam_xu  = vars_dict['lambda_x_upper']
        lam_xl  = vars_dict['lambda_x_lower']
        mu      = vars_dict['mu']
        ita     = vars_dict['ita']

        Pd = self.active_set_data[sample_id]['pd_data']

        if model.status == GRB.OPTIMAL:
            lambda_sol = {
                'lambda_power_balance': np.array([lam_pb[t].X  for t in range(self.T)]),
                'lambda_pg_lower':      np.array([[lam_pgl[g,t].X  for t in range(self.T)]   for g in range(self.ng)]),
                'lambda_pg_upper':      np.array([[lam_pgu[g,t].X  for t in range(self.T)]   for g in range(self.ng)]),
                'lambda_ramp_up':       np.array([[lam_ru[g,t].X   for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_ramp_down':     np.array([[lam_rd[g,t].X   for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_min_on':        self._extract_min_up_gurobi_values(lam_mon, Ton),
                'lambda_min_off':       self._extract_min_down_gurobi_values(lam_moff, Toff),
                'lambda_start_cost':    np.array([[lam_sc[g,t].X   for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_shut_cost':     np.array([[lam_shc[g,t].X  for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_coc_nonneg':    np.array([[lam_coc[g,t].X  for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_cpower':        np.array([[lam_cp[g,t].X   for t in range(self.T)]   for g in range(self.ng)]),
                'lambda_dcpf_upper':    np.array([[lam_dcu[l,t].X  for t in range(self.T)]   for l in range(self.nl)]),
                'lambda_dcpf_lower':    np.array([[lam_dcl[l,t].X  for t in range(self.T)]   for l in range(self.nl)]),
                'lambda_x_upper':       np.array([[lam_xu[g,t].X   for t in range(self.T)]   for g in range(self.ng)]),
                'lambda_x_lower':       np.array([[lam_xl[g,t].X   for t in range(self.T)]   for g in range(self.ng)]),
            }
            mu_sol  = np.array([[mu[l,t].X   for t in range(self.T)] for l in range(self.nl)])
            ita_sol = np.array([[ita[g,t].X  for t in range(self.T)] for g in range(self.ng)])

            if sample_id <= 2:
                obj_dual_v = (vars_dict['obj_dual_pg'].getValue()
                              + vars_dict['obj_dual_x'].getValue()
                              + vars_dict['obj_dual_coc'].getValue())
                print(
                    f"dual_block, sample_id: {sample_id}, obj_dual: {obj_dual_v:.4f}",
                    flush=True,
                )
            return lambda_sol, mu_sol, ita_sol
        else:
            print(f"❌ 对偶块模型求解失败，状态: {model.status}", flush=True)
            return None, None, None

    def _iter_with_dual_block_original(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """保留原始实现以备参考（不再被主流程调用）。"""
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            return self._iter_with_dual_block_cvxpy_highs(sample_id, theta_values, zeta_values, union_analysis)
        model = gp.Model('iter_with_dual_block')
        model.Params.OutputFlag = 0
        Pd = self.active_set_data[sample_id]['pd_data']
        
        # 功率平衡约束的对偶变量（无符号限制）
        lambda_power_balance = model.addVars(self.T, lb=-GRB.INFINITY, name='lambda_power_balance')
        lambda_pg_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_lower')
        lambda_pg_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_upper')
        lambda_ramp_up = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_up')
        lambda_ramp_down = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_down')
        
        Ton = self.Ton
        Toff = self.Toff
        lambda_min_on = model.addVars(self.ng, Ton, self.T, lb=0, name='lambda_min_on')
        lambda_min_off = model.addVars(self.ng, Toff, self.T, lb=0, name='lambda_min_off')
        lambda_start_cost = model.addVars(self.ng, self.T-1, lb=0, name='lambda_start_cost')
        lambda_shut_cost = model.addVars(self.ng, self.T-1, lb=0, name='lambda_shut_cost')
        lambda_coc_nonneg = model.addVars(self.ng, self.T-1, lb=0, name='lambda_coc_nonneg')
        lambda_cpower = model.addVars(self.ng, self.T, lb=0, name='lambda_cpower')
        lambda_dcpf_upper = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_upper')
        lambda_dcpf_lower = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_lower')
        lambda_x_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_x_upper')
        lambda_x_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_x_lower')
        
        # mu和ita变量（注意：ita的维度是(ng, T)）
        
        dual_decay_round_ = self._current_dual_decay_round()
        floor_active = self.iter_number < dual_decay_round_
        sign_relax_round = self._is_dual_sign_relaxation_round()

        mu_lb = -GRB.INFINITY if (floor_active and sign_relax_round) else 0.0
        ita_lb = -GRB.INFINITY if (floor_active and sign_relax_round) else 0.0
        mu = model.addVars(self.nl, self.T, lb=mu_lb, name='mu')
        ita = model.addVars(self.ng, self.T, lb=ita_lb, name='ita')
        mu_abs = model.addVars(self.nl, self.T, lb=0, name='mu_abs')
        ita_abs = model.addVars(self.ng, self.T, lb=0, name='ita_abs')

        mu_max = model.addVar(lb=0, name='mu_max')
        ita_max = model.addVar(lb=0, name='ita_max')
        deadband = 100
        
        for l in range(self.nl):
            for t in range(self.T):
                model.addConstr(mu_abs[l, t] >= mu[l, t], name=f'mu_abs_pos_{l}_{t}')
                model.addConstr(mu_abs[l, t] >= -mu[l, t], name=f'mu_abs_neg_{l}_{t}')
                if floor_active:
                    if sign_relax_round:
                        model.addConstr(mu_abs[l, t] >= self.mu_dual_floor_init, name=f'mu_abs_lb_{l}_{t}')
                    else:
                        model.addConstr(mu[l, t] >= self.mu_dual_floor_init, name=f'mu_lb_{l}_{t}')
                model.addConstr(mu_max >= mu_abs[l, t] - deadband, name=f'mu_max_constr_{l}_{t}')
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(ita_abs[g, t] >= ita[g, t], name=f'ita_abs_pos_{g}_{t}')
                model.addConstr(ita_abs[g, t] >= -ita[g, t], name=f'ita_abs_neg_{g}_{t}')
                if floor_active:
                    if sign_relax_round:
                        model.addConstr(ita_abs[g, t] >= self.ita_dual_floor_init, name=f'ita_abs_lb_{g}_{t}')
                    else:
                        model.addConstr(ita[g, t] >= self.ita_dual_floor_init, name=f'ita_lb_{g}_{t}')
                model.addConstr(ita_max >= ita_abs[g, t] - deadband, name=f'ita_max_constr_{g}_{t}')
        
        penalty_factor = 0
        penal_mu = penalty_factor * mu_max
        penal_ita = penalty_factor * ita_max

        G = self._generator_incidence_matrix
        PTDF = self._ptdf_matrix
        branch_limit = self._branch_limit
        PTDF_G = self._ptdf_g

        Ru = self.Ru
        Rd = self.Rd
        Ru_co = self.Ru_co
        Rd_co = self.Rd_co
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        obj_dual_pg = 0
        obj_dual_x = 0
        obj_dual_coc = 0
        obj_opt = 0
                
        # pg变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self.gencost[g, -2] / self.T_delta
                dual_expr -= lambda_power_balance[t]
                dual_expr -= lambda_pg_lower[g, t]
                dual_expr += lambda_pg_upper[g, t]
                
                if t > 0:
                    dual_expr += lambda_ramp_up[g, t-1]
                    dual_expr -= lambda_ramp_down[g, t-1]
                if t < self.T - 1:
                    dual_expr -= lambda_ramp_up[g, t]
                    dual_expr += lambda_ramp_down[g, t]
                
                ptdfg_col = PTDF_G[:, g]
                for l in range(self.branch.shape[0]):
                    pg_coeff = ptdfg_col[l]
                    dual_expr += pg_coeff * (lambda_dcpf_upper[l, t] - lambda_dcpf_lower[l, t])
                
                dual_expr_pg_abs = model.addVar(lb=0, name=f'dual_expr_abs_pg_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= dual_expr, name=f'dual_expr_abs_pg_pos_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= -dual_expr, name=f'dual_expr_abs_pg_neg_{g}_{t}')
                obj_dual_pg += dual_expr_pg_abs
        
        # x变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self._build_x_dual_stationarity_expr(
                    g=g,
                    t=t,
                    lambda_cpower=lambda_cpower,
                    lambda_x_upper=lambda_x_upper,
                    lambda_x_lower=lambda_x_lower,
                    lambda_pg_lower=lambda_pg_lower,
                    lambda_pg_upper=lambda_pg_upper,
                    lambda_ramp_down=lambda_ramp_down,
                    lambda_ramp_up=lambda_ramp_up,
                    lambda_min_on=lambda_min_on,
                    lambda_min_off=lambda_min_off,
                    lambda_start_cost=lambda_start_cost,
                    lambda_shut_cost=lambda_shut_cost,
                    fixed_cost=self.gencost[g, -1] / self.T_delta,
                    pmin=self.gen[g, PMIN],
                    pmax=self.gen[g, PMAX],
                    rd_delta=Rd_co[g] - Rd[g],
                    ru_delta=Ru_co[g] - Ru[g],
                    start_cost=start_cost[g],
                    shut_cost=shut_cost[g],
                    Ton=Ton,
                    Toff=Toff,
                )

                # 添加参数化约束的对偶贡献（theta相关）
                if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
                    dual_expr_para = self._add_parametric_constraints_dual_block_const_to_model(
                        model, g, t, mu, sample_id, theta_values, union_analysis
                    )
                    dual_expr += dual_expr_para

                # 添加参数化约束的对偶贡献（zeta相关）
                if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
                    dual_expr_para = self._add_parametric_balance_power_constraints_dual_block_const_to_model(
                        model, g, t, ita, sample_id, zeta_values, union_analysis
                    )
                    dual_expr += dual_expr_para
                                
                dual_expr_x_abs = model.addVar(lb=0, name=f'dual_expr_abs_x_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= dual_expr, name=f'dual_expr_abs_x_pos_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= -dual_expr, name=f'dual_expr_abs_x_neg_{g}_{t}')
                obj_dual_x += dual_expr_x_abs

        # cpower变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(lambda_cpower[g, t] == 1, name=f'dual_expr_cpower_solid_{g}_{t}')
        
        # coc变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T-1):
                dual_expr = 1
                dual_expr -= lambda_start_cost[g, t]
                dual_expr -= lambda_shut_cost[g, t]
                dual_expr -= lambda_coc_nonneg[g, t]
                
                dual_expr_coc_abs = model.addVar(lb=0, name=f'dual_expr_abs_coc_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= dual_expr, name=f'dual_expr_abs_coc_pos_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= -dual_expr, name=f'dual_expr_abs_coc_neg_{g}_{t}')
                obj_dual_coc += dual_expr_coc_abs

        obj_dual = obj_dual_pg + obj_dual_x + obj_dual_coc

        # 原问题约束违反量
        for t in range(self.T):
            lambda_power_balance_abs = model.addVar(lb=0, name=f'lambda_power_balance_{t}')
            model.addConstr(lambda_power_balance_abs >= lambda_power_balance[t], name=f'lambda_power_balance_pos_{t}')
            model.addConstr(lambda_power_balance_abs >= -lambda_power_balance[t], name=f'lambda_power_balance_neg_{t}')
            power_balance_viol = abs(sum(self.pg[sample_id, g, t] for g in range(self.ng)) - np.sum(Pd[:, t]))
            if power_balance_viol > 1e-10:
                obj_opt += power_balance_viol * lambda_power_balance_abs

            for g in range(self.ng):
                pg_lower_viol = abs(self.pg[sample_id, g, t] - self.gen[g, PMIN] * self.x[sample_id, g, t])
                if pg_lower_viol > 1e-10:
                    obj_opt += pg_lower_viol * lambda_pg_lower[g, t]
                pg_upper_viol = abs(self.gen[g, PMAX] * self.x[sample_id, g, t] - self.pg[sample_id, g, t])
                if pg_upper_viol > 1e-10:
                    obj_opt += pg_upper_viol * lambda_pg_upper[g, t]

        for t in range(1, self.T):
            for g in range(self.ng):
                ramp_up_viol = abs(self.pg[sample_id, g, t] - self.pg[sample_id, g, t-1] - (Ru[g] * self.x[sample_id, g, t-1] + Ru_co[g] * (1 - self.x[sample_id, g, t-1])))
                if ramp_up_viol > 1e-10:
                    obj_opt += ramp_up_viol * lambda_ramp_up[g, t-1]
                ramp_down_viol = abs(self.pg[sample_id, g, t-1] - self.pg[sample_id, g, t] - (Rd[g] * self.x[sample_id, g, t] + Rd_co[g] * (1 - self.x[sample_id, g, t])))
                if ramp_down_viol > 1e-10:
                    obj_opt += ramp_down_viol * lambda_ramp_down[g, t-1]

        for g in range(self.ng):
            for t in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - t):
                    min_on_viol = abs(self.x[sample_id, g, t1+1] - self.x[sample_id, g, t1] - self.x[sample_id, g, t1+t])
                    if min_on_viol > 1e-10:
                        obj_opt += min_on_viol * lambda_min_on[g, t-1, t1]

        for g in range(self.ng):
            for t in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - t):
                    min_off_viol = abs(-self.x[sample_id, g, t1+1] + self.x[sample_id, g, t1] - 1 + self.x[sample_id, g, t1+t])
                    if min_off_viol > 1e-10:
                        obj_opt += min_off_viol * lambda_min_off[g, t-1, t1]

        for t in range(1, self.T):
            for g in range(self.ng):
                coc_viol = abs(self.coc[sample_id, g, t-1])
                if coc_viol > 1e-10:
                    obj_opt += coc_viol * lambda_coc_nonneg[g, t-1]
                start_cost_viol = abs(self.coc[sample_id, g, t-1] - start_cost[g] * (self.x[sample_id, g, t] - self.x[sample_id, g, t-1]))
                if start_cost_viol > 1e-10:
                    obj_opt += start_cost_viol * lambda_start_cost[g, t-1]
                shut_cost_viol = abs(self.coc[sample_id, g, t-1] - shut_cost[g] * (self.x[sample_id, g, t-1] - self.x[sample_id, g, t]))
                if shut_cost_viol > 1e-10:
                    obj_opt += shut_cost_viol * lambda_shut_cost[g, t-1]

        for t in range(self.T):
            flow = PTDF @ (G @ np.array([self.pg[sample_id, g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                dcpf_upper_viol = abs(flow[l] - branch_limit[l])
                dcpf_lower_viol = abs(flow[l] + branch_limit[l])
                if dcpf_upper_viol > 1e-10:
                    obj_opt += dcpf_upper_viol * lambda_dcpf_upper[l, t]
                if dcpf_lower_viol > 1e-10:
                    obj_opt += dcpf_lower_viol * lambda_dcpf_lower[l, t]

        for t in range(self.T):
            for g in range(self.ng):
                x_lower_viol = abs(self.x[sample_id, g, t])
                if x_lower_viol > 1e-10:
                    obj_opt += x_lower_viol * lambda_x_lower[g, t]
                x_upper_viol = abs(self.x[sample_id, g, t] - 1)
                if x_upper_viol > 1e-10:
                    obj_opt += x_upper_viol * lambda_x_upper[g, t]
        
        # 添加参数化约束的obj_opt项（参考uc_dfsm_bcd.py）
        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis and theta_values is not None:
            model, obj_opt_para = self._add_parametric_obj_dual_block(
                model, self.x[sample_id, :, :], mu, mu_abs, sample_id, theta_values, union_analysis, PTDF=PTDF, branch_limit=branch_limit
            )
            obj_opt += obj_opt_para
        
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis and zeta_values is not None:
            model, obj_opt_para = self._add_parametric_balance_power_obj_dual_block(
                model, self.x[sample_id, :, :], ita, ita_abs, sample_id, zeta_values, union_analysis
            )
            obj_opt += obj_opt_para
   
        # 设置目标函数
        obj_dual_prox = self._build_dual_block_prox_obj(
            model,
            sample_id,
            lambda_power_balance,
            lambda_pg_lower,
            lambda_pg_upper,
            lambda_ramp_up,
            lambda_ramp_down,
            lambda_min_on,
            lambda_min_off,
            lambda_start_cost,
            lambda_shut_cost,
            lambda_coc_nonneg,
            lambda_dcpf_upper,
            lambda_dcpf_lower,
            lambda_x_upper,
            lambda_x_lower,
            mu,
            ita,
            Ton,
            Toff,
        )
        total_objective = (
            self.rho_dual_pg * obj_dual_pg
            + self.rho_dual_x * obj_dual_x
            + self.rho_dual_coc * obj_dual_coc
            + self.rho_opt * obj_opt
            + self.dual_block_prox_weight * obj_dual_prox
            + penal_mu
            + penal_ita
        )
        model.setObjective(total_objective, GRB.MINIMIZE)

        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            lambda_sol = {
                'lambda_power_balance': np.array([lambda_power_balance[t].X for t in range(self.T)]),
                'lambda_pg_lower': np.array([[lambda_pg_lower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_pg_upper': np.array([[lambda_pg_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_ramp_up': np.array([[lambda_ramp_up[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_ramp_down': np.array([[lambda_ramp_down[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_min_on': self._extract_min_up_gurobi_values(lambda_min_on, Ton),
                'lambda_min_off': self._extract_min_down_gurobi_values(lambda_min_off, Toff),
                'lambda_start_cost': np.array([[lambda_start_cost[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_shut_cost': np.array([[lambda_shut_cost[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_coc_nonneg': np.array([[lambda_coc_nonneg[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_cpower': np.array([[lambda_cpower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_dcpf_upper': np.array([[lambda_dcpf_upper[l, t].X for t in range(self.T)] for l in range(self.nl)]),
                'lambda_dcpf_lower': np.array([[lambda_dcpf_lower[l, t].X for t in range(self.T)] for l in range(self.nl)]),
                'lambda_x_upper': np.array([[lambda_x_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_x_lower': np.array([[lambda_x_lower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            }
            
            mu_sol = np.array([[mu[l, t].X for t in range(self.T)] for l in range(self.nl)])
            ita_sol = np.array([[ita[g, t].X for t in range(self.T)] for g in range(self.ng)])
            
            if sample_id <= 2:
                print(
                    f"dual_block, sample_id: {sample_id}, "
                    f"obj_dual_pg: {obj_dual_pg.getValue()}, "
                    f"obj_dual_x: {obj_dual_x.getValue()}, "
                    f"obj_dual_coc: {obj_dual_coc.getValue()}, "
                    f"obj_dual: {obj_dual.getValue()}, obj_opt: {obj_opt.getValue()}, "
                    f"obj_dual_prox: {obj_dual_prox.getValue() if hasattr(obj_dual_prox, 'getValue') else 0.0}",
                    flush=True,
                )
            
            return lambda_sol, mu_sol, ita_sol
        else:
            print(f"❌ 对偶块模型求解失败，状态: {model.status}", flush=True)
            return None, None, None
    
    def iter_with_theta_zeta_neural_network(
        self,
        union_analysis=None,
        num_epochs=1,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
    ):
        """
        使用神经网络更新theta和zeta（参考uc_NN.py的train方法）
        使用loss_function_differentiable进行训练，mini-batch梯度累积模式
        """
        if not TORCH_AVAILABLE or self.theta_net is None or self.zeta_net is None:
            print("警告: 神经网络不可用，跳过theta/zeta更新", flush=True)
            return self.theta_values_list, self.zeta_values_list

        if union_analysis is None:
            union_analysis = self._current_union_analysis

        resolved_batch_strategy = normalize_nn_batch_strategy(
            self.nn_batch_strategy if batch_strategy is None else batch_strategy
        )
        resolved_shuffle = self.nn_shuffle if shuffle is None else bool(shuffle)
        resolved_learning_rate = self.nn_learning_rate if learning_rate is None else float(learning_rate)
        if resolved_learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {resolved_learning_rate}")
        if resolved_batch_strategy == "full-batch":
            resolved_batch_size = max(1, self.n_samples)
        else:
            base_batch_size = self.nn_batch_size if batch_size is None else int(batch_size)
            resolved_batch_size = max(1, base_batch_size)

        self.theta_optimizer = optim.Adam(
            self.theta_net.parameters(),
            lr=resolved_learning_rate,
            weight_decay=0.0,
        )
        self.zeta_optimizer = optim.Adam(
            self.zeta_net.parameters(),
            lr=resolved_learning_rate,
            weight_decay=0.0,
        )
        self.theta_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.theta_optimizer,
            T_max=max(int(num_epochs), 1),
            eta_min=resolved_learning_rate * 0.05,
        )
        self.zeta_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.zeta_optimizer,
            T_max=max(int(num_epochs), 1),
            eta_min=resolved_learning_rate * 0.05,
        )
        predictor_train_active = self._unit_predictor_finetune_active()
        if predictor_train_active:
            self._ensure_unit_predictor_optimizer()
            for g in range(self.ng):
                self.unit_predictor.get_network(g).train()
        predictor_override_active = self._unit_predictor_active()
        all_params = list(self.theta_net.parameters()) + list(self.zeta_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=resolved_learning_rate, weight_decay=0.0)
        if self.enable_dropout_during_nn_training:
            self.theta_net.train()
            self.zeta_net.train()
        else:
            # Keep gradients enabled while disabling stochastic dropout masks.
            self.theta_net.eval()
            self.zeta_net.eval()

        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_component_sums = {
                'obj_primal': None,
                'obj_dual_x': None,
                'obj_opt': None,
                'primal_term': None,
                'dual_x_term': None,
                'opt_term': None,
                'reg_term': None,
            }
            self.theta_optimizer.zero_grad()
            self.zeta_optimizer.zero_grad()
            if predictor_train_active and self._unit_predictor_optimizer is not None:
                self._unit_predictor_optimizer.zero_grad()
            batch_count = 0
            need_epoch_breakdown = self.n_samples > 0 and (epoch == 0 or epoch == num_epochs - 1)
            sample_indices = np.arange(self.n_samples, dtype=int)
            if resolved_shuffle and self.n_samples > 1:
                np.random.shuffle(sample_indices)

            for sample_pos, sample_id in enumerate(sample_indices):
                batch_start = (sample_pos // resolved_batch_size) * resolved_batch_size
                actual_batch_size = min(resolved_batch_size, self.n_samples - batch_start)

                # 从缓存取特征（view，无拷贝）
                features_tensor = self._features_cache[sample_id:sample_id+1]

                # 前向传播
                theta_output = self.theta_net(features_tensor)
                zeta_output = self.zeta_net(features_tensor)
                theta_tensor = theta_output[0]
                zeta_tensor = zeta_output[0]
                if predictor_override_active:
                    zeta_tensor = self._apply_unit_predictor_zeta_override(
                        zeta_tensor,
                        sample_id,
                        train_predictor=predictor_train_active,
                    )

                # 计算可微分的loss + scaled backward
                if need_epoch_breakdown:
                    differentiable_loss, loss_components = self.loss_function_differentiable(
                        sample_id,
                        theta_tensor,
                        zeta_tensor,
                        union_analysis,
                        device=self.device,
                        return_components=True,
                    )
                    for key in epoch_component_sums:
                        if epoch_component_sums[key] is None:
                            epoch_component_sums[key] = loss_components[key]
                        else:
                            epoch_component_sums[key] = epoch_component_sums[key] + loss_components[key]
                else:
                    differentiable_loss = self.loss_function_differentiable(
                        sample_id,
                        theta_tensor,
                        zeta_tensor,
                        union_analysis,
                        device=self.device,
                    )
                (differentiable_loss / actual_batch_size).backward()
                epoch_total_loss += differentiable_loss.detach().cpu().item()
                batch_count += 1

                # batch 满或 epoch 结束：clip + step
                if batch_count == resolved_batch_size or sample_pos == self.n_samples - 1:
                    torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.zeta_net.parameters(), max_norm=1.0)
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        torch.nn.utils.clip_grad_norm_(self._unit_predictor_parameters(), max_norm=1.0)
                    self.theta_optimizer.step()
                    self.zeta_optimizer.step()
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        self._unit_predictor_optimizer.step()
                    self.theta_optimizer.zero_grad()
                    self.zeta_optimizer.zero_grad()
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        self._unit_predictor_optimizer.zero_grad()
                    batch_count = 0

            self.theta_scheduler.step()
            self.zeta_scheduler.step()

            if self.n_samples > 0 and (epoch == 0 or epoch == num_epochs - 1):
                avg_loss = epoch_total_loss / self.n_samples
                avg_components = {
                    key: float((value / self.n_samples).cpu().item())
                    for key, value in epoch_component_sums.items()
                }
                # print(
                #     f"[NN-theta/zeta] epoch {epoch+1}/{num_epochs}, avg_loss = {avg_loss:.6f}, "
                #     f"avg_terms(primal={avg_components['primal_term']:.6f}, "
                #     f"dual_x={avg_components['dual_x_term']:.6f}, "
                #     f"opt={avg_components['opt_term']:.6f}, "
                #     f"reg={avg_components['reg_term']:.6f})",
                #     flush=True,
                # )
                print(
                    f"[NN-theta/zeta] epoch {epoch+1}/{num_epochs}, avg_loss = {avg_loss:.6f}, "
                    f"obj_primal={avg_components['obj_primal']:.6f}, "
                    f"obj_dual_x={avg_components['obj_dual_x']:.6f}, "
                    f"obj_opt={avg_components['obj_opt']:.6f}, "
                    f"reg={avg_components['reg_term']:.6f}",
                    flush=True,
                )                
                self._last_nn_loss_breakdown = {
                    'avg_loss': avg_loss,
                    **avg_components,
                }

        # 记录最终 epoch loss 供 logger 使用
        if self.n_samples > 0:
            self._last_nn_loss = epoch_total_loss / self.n_samples

        # 更新theta和zeta值（per-sample 生成）
        self.theta_net.eval()
        self.zeta_net.eval()
        theta_values_new_list = []
        zeta_values_new_list = []
        with torch.no_grad():
            for sample_id in range(self.n_samples):
                features_tensor = self._features_cache[sample_id:sample_id + 1]
                theta_output = self.theta_net(features_tensor)
                zeta_output = self.zeta_net(features_tensor)
                theta_values_new_list.append(self._tensor_to_theta_dict(theta_output[0]))
                zeta_tensor = self._apply_unit_predictor_zeta_override(
                    zeta_output[0],
                    sample_id,
                    train_predictor=False,
                )
                zeta_values_new_list.append(self._tensor_to_zeta_dict(zeta_tensor))

        return theta_values_new_list, zeta_values_new_list
    
    def iter(
        self,
        max_iter=20,
        dual_decay_round=10,
        dual_sign_relax_interval: int | None = None,
        nn_epochs=10,
        union_analysis=None,
        theta_training_stages=None,
        nn_batch_strategy: str | None = None,
        nn_batch_size: int | None = None,
        nn_shuffle: bool | None = None,
        nn_learning_rate: float | None = None,
        direct_train_config: dict | None = None,
    ):
        """
        主迭代循环（参考uc_dfsm_bcd.py）
        - 迭代PG块（更新x, pg）
        - 迭代对偶块（更新lambda, mu, ita）
        - 使用神经网络更新theta和zeta
        """
        if not hasattr(self, 'logger'):
            self.logger = None
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        self.dual_decay_round = dual_decay_round
        if dual_sign_relax_interval is not None:
            self.dual_sign_relax_interval = max(int(dual_sign_relax_interval), 0)

        gamma = self.gamma_base / (self.n_samples * max_iter)

        for i in range(max_iter):
            self.iter_number = i
            self._sync_parametric_direction_strategy_state()
            print(f"[BCD] Iteration {i+1}/{max_iter}", flush=True)
            active_union_analysis, active_theta_stage = self._resolve_active_union_analysis(
                i,
                max_iter,
                union_analysis,
                theta_training_stages,
            )
            if active_theta_stage is None:
                self._last_reported_theta_stage_signature = None
            else:
                active_theta_count = len(active_union_analysis.get('union_constraints', []))
                stage_signature = (
                    int(active_theta_stage['stage_index']),
                    int(active_theta_stage['max_constraints_per_time_slot']),
                    int(active_theta_count),
                )
                if stage_signature != self._last_reported_theta_stage_signature:
                    normalized_stages = self._normalize_theta_training_stages(theta_training_stages, max_iter) or []
                    print(
                        f"[BCD][theta-stage] stage={active_theta_stage['stage_index'] + 1}/{len(normalized_stages)}, "
                        f"iter_range={active_theta_stage['iter_start'] + 1}-{active_theta_stage['iter_end']}, "
                        f"active_limit_per_time={active_theta_stage['max_constraints_per_time_slot']}, "
                        f"active_theta_constraints={active_theta_count}",
                        flush=True,
                    )
                    self._last_reported_theta_stage_signature = stage_signature

            # 1. 迭代PG块
            EPS = 1e-10
            for sample_id in range(self.n_samples):
                pg_sol, x_sol, cpower_sol, coc_sol = self.iter_with_pg_block(
                    sample_id=sample_id,
                    theta_values=self.theta_values_list[sample_id],
                    zeta_values=self.zeta_values_list[sample_id],
                    union_analysis=active_union_analysis
                )
                if pg_sol is None:
                    print(f"[BCD] PG block failed for sample {sample_id}", flush=True)
                    print("[BCD] Stopping after PG block failure", flush=True)
                    break
                
                # 数值过滤
                pg_sol = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                x_sol = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                x_sol = np.where(np.abs(x_sol - 1) < EPS, 1, x_sol)
                
                self.pg[sample_id, :, :] = pg_sol
                self.x[sample_id, :, :] = x_sol
                self.cpower[sample_id, :, :] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
                self.coc[sample_id, :, :] = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)
            
            # 2. 迭代对偶块
            sign_relax_round = self._is_dual_sign_relaxation_round()
            mu_floor = self.mu_dual_floor_init if self._is_dual_floor_active() else 0.0
            ita_floor = self.ita_dual_floor_init if self._is_dual_floor_active() else 0.0
            raw_mu_solutions = np.zeros((self.n_samples, self.nl, self.T), dtype=float)
            raw_ita_solutions = np.zeros((self.n_samples, self.ng, self.T), dtype=float)
            solved_mask = np.zeros(self.n_samples, dtype=bool)
            for sample_id in range(self.n_samples):
                lambda_sol, mu_sol, ita_sol = self.iter_with_dual_block(
                    sample_id=sample_id,
                    theta_values=self.theta_values_list[sample_id],
                    zeta_values=self.zeta_values_list[sample_id],
                    union_analysis=active_union_analysis
                )
                if lambda_sol is None or mu_sol is None:
                    print(f"[BCD] Dual block failed for sample {sample_id}", flush=True)
                    print("[BCD] Stopping after dual block failure", flush=True)
                    break
                
                self.lambda_[sample_id] = lambda_sol
                raw_mu_solutions[sample_id] = mu_sol
                raw_ita_solutions[sample_id] = ita_sol
                solved_mask[sample_id] = True
            if sign_relax_round and np.any(solved_mask):
                self.theta_constraint_direction_signs = self._resolve_direction_signs_from_signed_duals(
                    raw_mu_solutions[solved_mask],
                    self.theta_constraint_direction_signs,
                )
                self.zeta_constraint_direction_signs = self._resolve_direction_signs_from_signed_duals(
                    raw_ita_solutions[solved_mask],
                    self.zeta_constraint_direction_signs,
                )
            for sample_id in range(self.n_samples):
                if not solved_mask[sample_id]:
                    continue
                mu_signed = self._finalize_signed_dual_values(
                    raw_mu_solutions[sample_id],
                    self.theta_constraint_direction_signs,
                    mu_floor,
                )
                ita_signed = self._finalize_signed_dual_values(
                    raw_ita_solutions[sample_id],
                    self.zeta_constraint_direction_signs,
                    ita_floor,
                )
                self.mu[sample_id, :, :] = np.where(np.abs(mu_signed) < EPS, 0, mu_signed)
                self.ita[sample_id, :, :] = np.where(np.abs(ita_signed) < EPS, 0, ita_signed)
            
            # 3. 刷新迭代级张量缓存（整批转换，避免 loss 内逐张量创建）
            if TORCH_AVAILABLE and hasattr(self, 'device'):
                self._refresh_iter_tensor_cache()
            (
                obj_primal_pre,
                obj_dual_pg_pre,
                obj_dual_x_pre,
                obj_dual_coc_pre,
                obj_dual_pre,
                obj_opt_pre,
            ) = self.cal_viol_components(union_analysis=active_union_analysis)
            EPS_PRE = 1e-12
            obj_primal_pre = obj_primal_pre if abs(obj_primal_pre) >= EPS_PRE else 0.0
            obj_dual_pg_pre = obj_dual_pg_pre if abs(obj_dual_pg_pre) >= EPS_PRE else 0.0
            obj_dual_x_pre = obj_dual_x_pre if abs(obj_dual_x_pre) >= EPS_PRE else 0.0
            obj_dual_coc_pre = obj_dual_coc_pre if abs(obj_dual_coc_pre) >= EPS_PRE else 0.0
            obj_dual_pre = obj_dual_pre if abs(obj_dual_pre) >= EPS_PRE else 0.0
            obj_opt_pre = obj_opt_pre if abs(obj_opt_pre) >= EPS_PRE else 0.0
            nn_metrics_pre = self.cal_nn_logging_components(union_analysis=active_union_analysis)

            # 4. 使用神经网络更新theta和zeta
            print(
                f"[BCD][NN-metric][before] obj_primal={nn_metrics_pre['obj_primal']:.6f}, "
                f"obj_dual_x={nn_metrics_pre['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_pre['obj_opt']:.6f}, "
                f"reg={nn_metrics_pre['reg']:.6f}",
                flush=True,
            )
            #    先缓存上一代输出，供“迭代间差异正则”使用（默认权重为0时不影响训练）
            if self.iter_delta_reg_weight > 0:
                self._prev_theta_values_list = [dict(v) for v in self.theta_values_list]
                self._prev_zeta_values_list = [dict(v) for v in self.zeta_values_list]
            self.iter_with_direct_theta_zeta_targets(
                union_analysis=active_union_analysis,
                config=direct_train_config,
            )
            theta_values_new, zeta_values_new = self.iter_with_theta_zeta_neural_network(
                union_analysis=active_union_analysis,
                num_epochs=nn_epochs,
                batch_strategy=nn_batch_strategy,
                batch_size=nn_batch_size,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
            )
            if theta_values_new is None or zeta_values_new is None:
                print("[BCD] Theta/Zeta neural update failed", flush=True)
                print("[BCD] Stopping after Theta/Zeta neural update failure", flush=True)
                break
            self.theta_values_list = theta_values_new
            self.zeta_values_list = zeta_values_new
            self.theta_values = self.theta_values_list[0]
            self.zeta_values = self.zeta_values_list[0]
            print(f"[BCD] Iteration {i+1}/{max_iter} complete", flush=True)
            
            print(f"[BCD] Iteration {i+1}/{max_iter} succeeded", flush=True)
            
            # 计算违反量（参考uc_dfsm_bcd.py）
            (
                obj_primal,
                obj_dual_pg,
                obj_dual_x,
                obj_dual_coc,
                obj_dual,
                obj_opt,
            ) = self.cal_viol_components(union_analysis=active_union_analysis)
            
            # 简单数值过滤：绝对值过小的值设为0
            EPS = 1e-12
            obj_primal = obj_primal if abs(obj_primal) >= EPS else 0.0
            obj_dual_pg = obj_dual_pg if abs(obj_dual_pg) >= EPS else 0.0
            obj_dual_x = obj_dual_x if abs(obj_dual_x) >= EPS else 0.0
            obj_dual_coc = obj_dual_coc if abs(obj_dual_coc) >= EPS else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= EPS else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= EPS else 0.0
            
            nn_metrics_after = self.cal_nn_logging_components(union_analysis=active_union_analysis)
            print(
                f"[BCD][NN-metric][after] obj_primal={nn_metrics_after['obj_primal']:.6f}, "
                f"obj_dual_x={nn_metrics_after['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_after['obj_opt']:.6f}, "
                f"reg={nn_metrics_after['reg']:.6f}",
                flush=True,
            )
            self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
            obj_binary = self.cal_obj_binary_gap()
            obj_binary = obj_binary if abs(obj_binary) >= EPS else 0.0
            self.rho_binary = min(self.rho_binary + gamma * obj_binary, self.rho_max)
            gamma_dual = gamma * self.gamma_dual_component_scale
            self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
            self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
            self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
            self._sync_rho_dual_summary()
            self.rho_opt = min(self.rho_opt + gamma * obj_opt, self.rho_max)
            print(
                f"[BCD][rho] primal={self.rho_primal}, binary={self.rho_binary}, "
                f"dual_pg={self.rho_dual_pg}, "
                f"dual_x={self.rho_dual_x}, dual_coc={self.rho_dual_coc}, "
                f"dual={self.rho_dual}, opt={self.rho_opt}",
                flush=True,
            )
            print(
                f"当前惩罚参数: ρ_primal={self.rho_primal}, "
                f"ρ_dual_pg={self.rho_dual_pg}, ρ_dual_x={self.rho_dual_x}, "
                f"ρ_dual_coc={self.rho_dual_coc}, ρ_dual={self.rho_dual}, "
                f"ρ_opt={self.rho_opt}",
                flush=True,
            )
            print("--------------------------------", flush=True)

            # logger 钩子
            if self.logger is not None:
                nn_loss = getattr(self, '_last_nn_loss', None)
                self.logger.log_bcd_iter(
                    iter=i, obj_primal=obj_primal, obj_dual=obj_dual, obj_opt=obj_opt,
                    rho_primal=self.rho_primal, rho_dual=self.rho_dual, rho_opt=self.rho_opt,
                    obj_dual_pg=obj_dual_pg,
                    obj_dual_x=obj_dual_x,
                    obj_dual_coc=obj_dual_coc,
                    rho_dual_pg=self.rho_dual_pg,
                    rho_dual_x=self.rho_dual_x,
                    rho_dual_coc=self.rho_dual_coc,
                    nn_loss=nn_loss,
                )
                self.logger.snapshot('bcd', i, x=self.x[0], pg=self.pg[0], lambda_=self.lambda_[0])

            time.sleep(1)
        
        return self.theta_values_list, self.zeta_values_list

    def cal_viol_components(self, union_analysis=None):
        """
        计算约束违反量，并拆分 dual 违反量为 pg/x/coc 三部分。

        Returns:
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        obj_primal = 0
        obj_dual_pg = 0
        obj_dual_x = 0
        obj_dual_coc = 0
        obj_opt = 0
        
        # 预计算常量
        Ton = self.Ton
        Toff = self.Toff
        Ru = self.Ru
        Rd = self.Rd
        Ru_co = self.Ru_co
        Rd_co = self.Rd_co
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        # 潮流约束相关矩阵
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]
        # 预计算 PTDF @ G，避免在循环内重复计算
        PTDF_G = PTDF @ G  # shape: (nl, ng)

        for sample_id in range(self.n_samples):
            # per-sample theta/zeta
            sample_theta = self.theta_values_list[sample_id] if hasattr(self, 'theta_values_list') else self.theta_values
            sample_zeta = self.zeta_values_list[sample_id] if hasattr(self, 'zeta_values_list') else self.zeta_values

            # 预建 theta/zeta 查找表（per-sample）
            theta_lookup = self._build_theta_value_lookup(sample_theta, union_analysis=union_analysis)
            zeta_lookup = self._build_zeta_value_lookup(sample_zeta, union_analysis=union_analysis)
            Pd = self.active_set_data[sample_id]['pd_data']
            pg = self.pg[sample_id, :, :]
            x = self.x[sample_id, :, :]
            coc = self.coc[sample_id, :, :]
            cpower = self.cpower[sample_id, :, :]
            
            # 功率平衡约束
            for t in range(self.T):
                power_balance_expr = np.sum(pg[:, t]) - np.sum(Pd[:, t])
                obj_primal += abs(power_balance_expr)
                if sample_id < len(self.lambda_) and 'lambda_power_balance' in self.lambda_[sample_id]:
                    obj_opt += abs(power_balance_expr) * abs(self.lambda_[sample_id]['lambda_power_balance'][t])
                
                # 发电上下限约束
                for g in range(self.ng):
                    pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                    pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                    obj_primal += max(0, pg_lower_expr) + max(0, pg_upper_expr)
                    if sample_id < len(self.lambda_) and 'lambda_pg_lower' in self.lambda_[sample_id]:
                        obj_opt += abs(pg_lower_expr) * abs(self.lambda_[sample_id]['lambda_pg_lower'][g, t])
                    if sample_id < len(self.lambda_) and 'lambda_pg_upper' in self.lambda_[sample_id]:
                        obj_opt += abs(pg_upper_expr) * abs(self.lambda_[sample_id]['lambda_pg_upper'][g, t])
                    
                    # x变量约束违反
                    if sample_id < len(self.lambda_) and 'lambda_x_lower' in self.lambda_[sample_id]:
                        obj_opt += x[g, t] * abs(self.lambda_[sample_id]['lambda_x_lower'][g, t])
                    if sample_id < len(self.lambda_) and 'lambda_x_upper' in self.lambda_[sample_id]:
                        obj_opt += (1 - x[g, t]) * abs(self.lambda_[sample_id]['lambda_x_upper'][g, t])
            
            # 爬坡约束
            for t in range(1, self.T):
                for g in range(self.ng):
                    ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                    ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                    obj_primal += max(0, ramp_up_expr) + max(0, ramp_down_expr)
                    if sample_id < len(self.lambda_) and 'lambda_ramp_up' in self.lambda_[sample_id]:
                        obj_opt += abs(ramp_up_expr) * abs(self.lambda_[sample_id]['lambda_ramp_up'][g, t-1])
                    if sample_id < len(self.lambda_) and 'lambda_ramp_down' in self.lambda_[sample_id]:
                        obj_opt += abs(ramp_down_expr) * abs(self.lambda_[sample_id]['lambda_ramp_down'][g, t-1])
            
            # 最小开关机时间约束
            for g in range(self.ng):
                for t in range(1, self._min_up_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                        obj_primal += max(0, min_on_expr)
                        if sample_id < len(self.lambda_) and 'lambda_min_on' in self.lambda_[sample_id]:
                            obj_opt += abs(min_on_expr) * abs(self.lambda_[sample_id]['lambda_min_on'][g, t-1, t1])
            
            for g in range(self.ng):
                for t in range(1, self._min_down_horizon(g) + 1):
                    for t1 in range(self.T - t):
                        min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                        obj_primal += max(0, min_off_expr)
                        if sample_id < len(self.lambda_) and 'lambda_min_off' in self.lambda_[sample_id]:
                            obj_opt += abs(min_off_expr) * abs(self.lambda_[sample_id]['lambda_min_off'][g, t-1, t1])
            
            # 启停成本约束（对齐参考 uc_dfsm_bcd.py:4014-4027）
            for t in range(1, self.T):
                for g in range(self.ng):
                    start_cost_expr = start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1]
                    obj_primal += max(0, start_cost_expr)

                    shut_cost_expr = shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1]
                    obj_primal += max(0, shut_cost_expr)

                    if sample_id < len(self.lambda_) and 'lambda_start_cost' in self.lambda_[sample_id]:
                        obj_opt += abs(start_cost_expr) * abs(self.lambda_[sample_id]['lambda_start_cost'][g, t-1])
                    if sample_id < len(self.lambda_) and 'lambda_shut_cost' in self.lambda_[sample_id]:
                        obj_opt += abs(shut_cost_expr) * abs(self.lambda_[sample_id]['lambda_shut_cost'][g, t-1])

                    if sample_id < len(self.lambda_) and 'lambda_coc_nonneg' in self.lambda_[sample_id]:
                        obj_opt += coc[g, t-1] * abs(self.lambda_[sample_id]['lambda_coc_nonneg'][g, t-1])
            
            # 发电成本约束（对齐参考 uc_dfsm_bcd.py:4028-4034）
            for t in range(self.T):
                for g in range(self.ng):
                    cpower_expr = self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t] - cpower[g, t]
                    obj_primal += max(0, cpower_expr)
                    if sample_id < len(self.lambda_) and 'lambda_cpower' in self.lambda_[sample_id]:
                        obj_opt += abs(cpower_expr) * abs(self.lambda_[sample_id]['lambda_cpower'][g, t])
            
            # 潮流约束
            for t in range(self.T):
                flow = PTDF @ (G @ pg[:, t] - Pd[:, t])
                for l in range(self.branch.shape[0]):
                    dcpf_upper_viol = max(0, flow[l] - branch_limit[l])
                    dcpf_lower_viol = max(0, -flow[l] - branch_limit[l])
                    obj_primal += dcpf_upper_viol + dcpf_lower_viol
                    
                    abs_dcpf_upper_viol = abs(flow[l] - branch_limit[l])
                    abs_dcpf_lower_viol = abs(-flow[l] - branch_limit[l])
                    if sample_id < len(self.lambda_) and 'lambda_dcpf_upper' in self.lambda_[sample_id]:
                        obj_opt += abs_dcpf_upper_viol * abs(self.lambda_[sample_id]['lambda_dcpf_upper'][l, t])
                    if sample_id < len(self.lambda_) and 'lambda_dcpf_lower' in self.lambda_[sample_id]:
                        obj_opt += abs_dcpf_lower_viol * abs(self.lambda_[sample_id]['lambda_dcpf_lower'][l, t])
            
            # 参数化约束违反量（theta和zeta）
            if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis and sample_theta is not None:
                union_constraints = union_analysis['union_constraints']
                for constraint_info in union_constraints:
                    branch_id = constraint_info['branch_id']
                    time_slot = constraint_info['time_slot']
                    nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                    
                    lhs_expr = 0.0
                    for coeff_info in nonzero_coefficients:
                        unit_id = coeff_info['unit_id']
                        member_time = self._theta_member_time_index(constraint_info, coeff_info)
                        theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                        theta = sample_theta.get(theta_name, 0.0)
                        lhs_expr += theta * x[unit_id, member_time]
                    
                    theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
                    theta_rhs = sample_theta.get(theta_rhs_name, 1.0)
                    direction = self._get_theta_constraint_direction(branch_id, time_slot)
                    violation = direction * (lhs_expr - theta_rhs)
                    abs_violation = abs(violation)
                    obj_primal += max(0, violation)
                    if branch_id < self.nl and sample_id < len(self.mu):
                        obj_opt += abs_violation * abs(self.mu[sample_id, branch_id, time_slot])
                    else:
                        obj_opt += abs_violation * getattr(self, 'dual_para_bound', 0.1)
            
            if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis and sample_zeta is not None:
                union_zeta_constraints = union_analysis['union_zeta_constraints']
                for constraint in union_zeta_constraints:
                    unit_id = constraint['unit_id']
                    time_slot = constraint['time_slot']
                    
                    zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                    zeta = sample_zeta.get(zeta_name, 0.0)
                    lhs_expr = zeta * x[unit_id, time_slot]
                    
                    zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                    zeta_rhs = sample_zeta.get(zeta_rhs_name, 1.0)
                    direction = self._get_zeta_constraint_direction(unit_id, time_slot)
                    violation = direction * (lhs_expr - zeta_rhs)
                    abs_violation = abs(violation)
                    obj_primal += max(0, violation)
                    if sample_id < len(self.ita):
                        obj_opt += abs_violation * abs(self.ita[sample_id, unit_id, time_slot])
            
            # === 对偶约束的违反量 obj_dual（拆分为 pg/x/coc 三部分） ===
            # pg 变量的对偶约束
            if sample_id < len(self.lambda_):
                for g in range(self.ng):
                    for t in range(self.T):
                        # 基础项：发电成本系数 * lambda_cpower
                        dual_expr = self.gencost[g, -2] / self.T_delta * self.lambda_[sample_id]['lambda_cpower'][g, t]

                        # 功率平衡约束贡献
                        dual_expr -= self.lambda_[sample_id]['lambda_power_balance'][t]

                        # 发电上下限约束贡献
                        dual_expr -= self.lambda_[sample_id]['lambda_pg_lower'][g, t]
                        dual_expr += self.lambda_[sample_id]['lambda_pg_upper'][g, t]

                        # 爬坡约束贡献
                        if t > 0:  # 当前时段的爬坡约束
                            dual_expr += self.lambda_[sample_id]['lambda_ramp_up'][g, t-1]
                            dual_expr -= self.lambda_[sample_id]['lambda_ramp_down'][g, t-1]
                        if t < self.T - 1:  # 下一时段的爬坡约束
                            dual_expr -= self.lambda_[sample_id]['lambda_ramp_up'][g, t]
                            dual_expr += self.lambda_[sample_id]['lambda_ramp_down'][g, t]

                        # DCPF约束贡献 - 使用预计算的 PTDF_G（向量化）
                        ptdfg_col = PTDF_G[:, g]  # (nl,)
                        dual_expr += float(ptdfg_col @ (
                            self.lambda_[sample_id]['lambda_dcpf_upper'][:, t]
                            - self.lambda_[sample_id]['lambda_dcpf_lower'][:, t]
                        ))

                        # 对偶约束：梯度 = 0
                        obj_dual_pg += abs(dual_expr)

                # x 变量的对偶约束
                for g in range(self.ng):
                    for t in range(self.T):
                        # 基础项：固定成本 * lambda_cpower
                        dual_expr = self._build_x_dual_stationarity_expr(
                            g=g,
                            t=t,
                            lambda_cpower=self.lambda_[sample_id]['lambda_cpower'],
                            lambda_x_upper=self.lambda_[sample_id]['lambda_x_upper'],
                            lambda_x_lower=self.lambda_[sample_id]['lambda_x_lower'],
                            lambda_pg_lower=self.lambda_[sample_id]['lambda_pg_lower'],
                            lambda_pg_upper=self.lambda_[sample_id]['lambda_pg_upper'],
                            lambda_ramp_down=self.lambda_[sample_id]['lambda_ramp_down'],
                            lambda_ramp_up=self.lambda_[sample_id]['lambda_ramp_up'],
                            lambda_min_on=self.lambda_[sample_id]['lambda_min_on'],
                            lambda_min_off=self.lambda_[sample_id]['lambda_min_off'],
                            lambda_start_cost=self.lambda_[sample_id]['lambda_start_cost'],
                            lambda_shut_cost=self.lambda_[sample_id]['lambda_shut_cost'],
                            fixed_cost=self.gencost[g, -1] / self.T_delta,
                            pmin=self.gen[g, PMIN],
                            pmax=self.gen[g, PMAX],
                            rd_delta=Rd_co[g] - Rd[g],
                            ru_delta=Ru_co[g] - Ru[g],
                            start_cost=start_cost[g],
                            shut_cost=shut_cost[g],
                            Ton=Ton,
                            Toff=Toff,
                        )

                        # x 上下界约束贡献

                        # 发电上下限约束贡献

                        # 爬坡约束贡献

                        # 最小开机时间约束贡献（直接索引，O(Ton) 替代 O(Ton*T)）

                        # 最小关机时间约束贡献（直接索引，O(Toff) 替代 O(Toff*T)）

                        # 启停成本约束贡献

                        # theta 相关的对偶约束贡献（O(1) 查找，表在循环外预建）
                        for _branch_id, _anchor_time, _theta_val in theta_lookup.get((g, t), []):
                            if _branch_id < self.nl:
                                dual_expr += self._get_theta_constraint_direction(_branch_id, _anchor_time) * _theta_val * self.mu[sample_id, _branch_id, _anchor_time]
                            else:
                                dual_expr += _theta_val * getattr(self, 'dual_para_bound', 0.1)

                        # zeta 相关的对偶约束贡献（O(1) 查找）
                        for _zeta_val in zeta_lookup.get((g, t), []):
                            dual_expr += self._get_zeta_constraint_direction(g, t) * _zeta_val * self.ita[sample_id, g, t]

                        # 对偶约束：梯度 = 0
                        obj_dual_x += abs(dual_expr)

                #coc变量的对偶约束
                for g in range(self.ng):
                    for t in range(self.T-1):
                        dual_expr = 1
                        dual_expr -= self.lambda_[sample_id]['lambda_start_cost'][g, t]
                        dual_expr -= self.lambda_[sample_id]['lambda_shut_cost'][g, t]
                        dual_expr -= self.lambda_[sample_id]['lambda_coc_nonneg'][g, t]
                        obj_dual_coc += abs(dual_expr)

        obj_dual = obj_dual_pg + obj_dual_x + obj_dual_coc
        return obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt

    def cal_viol(self, union_analysis=None):
        """
        计算约束违反量（参考uc_dfsm_bcd.py）
        返回obj_primal, obj_dual, obj_opt
        """
        obj_primal, _, _, _, obj_dual, obj_opt = self.cal_viol_components(
            union_analysis=union_analysis
        )
        return obj_primal, obj_dual, obj_opt

    def cal_obj_binary_gap(self) -> float:
        """Match PG-block obj_binary: sum |x - reference commitment| over samples."""
        total = 0.0
        for sample_id in range(self.n_samples):
            uc_matrix = _get_uc_matrix_from_sample(
                self.active_set_data[sample_id],
                self.ng,
                self.T,
            )
            if uc_matrix is None:
                continue
            total += float(np.sum(np.abs(np.asarray(self.x[sample_id], dtype=float) - uc_matrix)))
        return total

    def cal_nn_logging_components(self, union_analysis=None) -> dict[str, float]:
        if not TORCH_AVAILABLE or self.n_samples <= 0:
            return {
                'obj_primal': 0.0,
                'obj_dual_x': 0.0,
                'obj_opt': 0.0,
                'reg': 0.0,
            }
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        self._refresh_iter_tensor_cache()

        totals = {
            'obj_primal': 0.0,
            'obj_dual_x': 0.0,
            'obj_opt': 0.0,
            'reg': 0.0,
        }
        for sample_id in range(self.n_samples):
            theta_tensor = torch.tensor(
                [float(self.theta_values_list[sample_id].get(name, 0.0)) for name in self.theta_var_names],
                dtype=torch.float32,
                device=self.device,
            )
            zeta_tensor = torch.tensor(
                [float(self.zeta_values_list[sample_id].get(name, 0.0)) for name in self.zeta_var_names],
                dtype=torch.float32,
                device=self.device,
            )
            _, components = self.loss_function_differentiable(
                sample_id,
                theta_tensor,
                zeta_tensor,
                union_analysis=union_analysis,
                device=self.device,
                return_components=True,
            )
            totals['obj_primal'] += float(components['obj_primal'].cpu().item())
            totals['obj_dual_x'] += float(components['obj_dual_x'].cpu().item())
            totals['obj_opt'] += float(components['obj_opt'].cpu().item())
            totals['reg'] += float(components['reg_term'].cpu().item())
        return totals

    def _bcd_direct_target_bounds(self, config: dict) -> tuple[np.ndarray, np.ndarray]:
        theta_dim = len(self.theta_var_names)
        zeta_dim = len(self.zeta_var_names)
        coeff_bound = float(config.get("coeff_bound", 2.0))
        rhs_bound = float(config.get("rhs_bound", 3.0))
        lower = np.full(theta_dim + zeta_dim, -coeff_bound, dtype=np.float64)
        upper = np.full(theta_dim + zeta_dim, coeff_bound, dtype=np.float64)
        for i, name in enumerate(self.theta_var_names):
            if "_rhs_" in name or name.startswith("theta_rhs_"):
                lower[i] = -rhs_bound
                upper[i] = rhs_bound
        offset = theta_dim
        for i, name in enumerate(self.zeta_var_names):
            if "_rhs_" in name or name.startswith("zeta_rhs_"):
                lower[offset + i] = -rhs_bound
                upper[offset + i] = rhs_bound
        return lower, upper

    def _x_stationarity_inherent_np(self, sample_id: int, g: int, t: int) -> float:
        lam = self.lambda_[sample_id]
        return float(
            self._build_x_dual_stationarity_expr(
                g=g,
                t=t,
                lambda_cpower=np.asarray(lam['lambda_cpower'], dtype=float),
                lambda_x_upper=np.asarray(lam['lambda_x_upper'], dtype=float),
                lambda_x_lower=np.asarray(lam['lambda_x_lower'], dtype=float),
                lambda_pg_lower=np.asarray(lam['lambda_pg_lower'], dtype=float),
                lambda_pg_upper=np.asarray(lam['lambda_pg_upper'], dtype=float),
                lambda_ramp_down=np.asarray(lam['lambda_ramp_down'], dtype=float),
                lambda_ramp_up=np.asarray(lam['lambda_ramp_up'], dtype=float),
                lambda_min_on=np.asarray(lam['lambda_min_on'], dtype=float),
                lambda_min_off=np.asarray(lam['lambda_min_off'], dtype=float),
                lambda_start_cost=np.asarray(lam['lambda_start_cost'], dtype=float),
                lambda_shut_cost=np.asarray(lam['lambda_shut_cost'], dtype=float),
                fixed_cost=float(self.gencost[g, -1] / self.T_delta),
                pmin=float(self.gen[g, PMIN]),
                pmax=float(self.gen[g, PMAX]),
                rd_delta=float(self.Rd_co[g] - self.Rd[g]),
                ru_delta=float(self.Ru_co[g] - self.Ru[g]),
                start_cost=float(self.gencost[g, 1]),
                shut_cost=float(self.gencost[g, 2]),
                Ton=self.Ton,
                Toff=self.Toff,
            )
        )

    def _build_bcd_direct_targets(self, union_analysis, config: dict) -> dict[str, np.ndarray | float]:
        if id(union_analysis) != self._cached_union_analysis_id:
            self._preprocess_union_analysis_cache(union_analysis)

        theta_dim = len(self.theta_var_names)
        zeta_dim = len(self.zeta_var_names)
        total_dim = theta_dim + zeta_dim
        lower, upper = self._bcd_direct_target_bounds(config)
        target_blend = min(max(float(config.get("target_blend", 0.75)), 0.0), 1.0)
        dual_w = float(config.get("dual_eq_weight", 1.0))
        active_w = float(config.get("active_opt_weight", 0.8))
        inactive_w = float(config.get("inactive_margin_weight", 0.0))
        inactive_margin = float(config.get("inactive_margin", 0.0))
        anchor = float(config.get("anchor_weight", 0.15))
        theta_anchor = float(config.get("theta_anchor_weight", anchor))
        zeta_anchor = float(config.get("zeta_anchor_weight", anchor))
        rhs_anchor = float(config.get("rhs_anchor_weight", anchor))
        active_threshold = float(config.get("dual_active_threshold", 1e-7))

        theta_targets = np.zeros((self.n_samples, theta_dim), dtype=np.float32)
        zeta_targets = np.zeros((self.n_samples, zeta_dim), dtype=np.float32)
        residuals: list[float] = []

        for sample_id in range(self.n_samples):
            y0 = np.concatenate([
                np.asarray([self.theta_values_list[sample_id].get(name, 0.0) for name in self.theta_var_names], dtype=np.float64),
                np.asarray([self.zeta_values_list[sample_id].get(name, 0.0) for name in self.zeta_var_names], dtype=np.float64),
            ])
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            weights: list[float] = []
            x_val = np.asarray(self.x[sample_id], dtype=np.float64)
            mu_val = np.asarray(self.mu[sample_id], dtype=np.float64)
            ita_val = np.asarray(self.ita[sample_id], dtype=np.float64)

            for g in range(self.ng):
                for t in range(self.T):
                    row = np.zeros(total_dim, dtype=np.float64)
                    for theta_idx, branch_id, anchor_time in self._dual_theta_lookup.get((g, t), []):
                        if branch_id < self.nl:
                            dual_scale = float(mu_val[branch_id, anchor_time])
                        else:
                            dual_scale = float(getattr(self, "dual_para_bound", 0.1))
                        row[theta_idx] += self._get_theta_constraint_direction(branch_id, anchor_time) * dual_scale
                    for zeta_idx in self._dual_zeta_lookup.get((g, t), []):
                        row[theta_dim + zeta_idx] += self._get_zeta_constraint_direction(g, t) * float(ita_val[g, t])
                    if np.any(np.abs(row) > 1e-12):
                        rows.append(row)
                        rhs.append(-self._x_stationarity_inherent_np(sample_id, g, t))
                        weights.append(max(dual_w, 0.0))

            for branch_id, time_slot, _ctype, coeff_list, rhs_idx in self._cached_theta_constraints:
                row = np.zeros(total_dim, dtype=np.float64)
                direction = self._get_theta_constraint_direction(branch_id, time_slot)
                for unit_id, member_time, theta_idx in coeff_list:
                    row[theta_idx] += direction * float(x_val[unit_id, member_time])
                if rhs_idx >= 0:
                    row[rhs_idx] -= direction
                dual_abs = abs(float(mu_val[branch_id, time_slot])) if branch_id < self.nl else float(getattr(self, "dual_para_bound", 0.1))
                if dual_abs > active_threshold and active_w > 0:
                    rows.append(row)
                    rhs.append(0.0)
                    weights.append(active_w)
                elif inactive_w > 0:
                    rows.append(row)
                    rhs.append(-inactive_margin)
                    weights.append(inactive_w)

            for unit_id, time_slot, zeta_idx, rhs_idx in self._cached_zeta_constraints:
                row = np.zeros(total_dim, dtype=np.float64)
                direction = self._get_zeta_constraint_direction(unit_id, time_slot)
                row[theta_dim + zeta_idx] += direction * float(x_val[unit_id, time_slot])
                if rhs_idx >= 0:
                    row[theta_dim + rhs_idx] -= direction
                dual_abs = abs(float(ita_val[unit_id, time_slot]))
                if dual_abs > active_threshold and active_w > 0:
                    rows.append(row)
                    rhs.append(0.0)
                    weights.append(active_w)
                elif inactive_w > 0:
                    rows.append(row)
                    rhs.append(-inactive_margin)
                    weights.append(inactive_w)

            if rows:
                mat = np.vstack(rows)
                vec = np.asarray(rhs, dtype=np.float64)
                w = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 0.0))
                mat_w = mat * w[:, None]
                vec_w = vec * w
                anchor_diag = np.full(total_dim, max(anchor, 1e-8), dtype=np.float64)
                for idx, name in enumerate(self.theta_var_names):
                    anchor_diag[idx] = max(rhs_anchor if "_rhs_" in name or name.startswith("theta_rhs_") else theta_anchor, 1e-8)
                for idx, name in enumerate(self.zeta_var_names):
                    anchor_diag[theta_dim + idx] = max(rhs_anchor if "_rhs_" in name or name.startswith("zeta_rhs_") else zeta_anchor, 1e-8)
                lhs = mat_w.T @ mat_w + np.diag(anchor_diag)
                rhs_vec = mat_w.T @ vec_w + anchor_diag * y0
                try:
                    y = np.linalg.solve(lhs, rhs_vec)
                except np.linalg.LinAlgError:
                    y = np.linalg.lstsq(lhs, rhs_vec, rcond=None)[0]
                residuals.append(float(np.mean(np.abs(mat @ y - vec))) if vec.size else 0.0)
            else:
                y = y0.copy()
                residuals.append(0.0)
            y = np.clip(y, lower, upper)
            y = (1.0 - target_blend) * y0 + target_blend * y
            theta_targets[sample_id] = y[:theta_dim]
            zeta_targets[sample_id] = y[theta_dim:]

        return {
            "theta": theta_targets,
            "zeta": zeta_targets,
            "proxy_residual_mean": float(np.mean(residuals)) if residuals else 0.0,
        }

    def iter_with_direct_theta_zeta_targets(self, union_analysis=None, config: dict | None = None):
        config = dict(self.direct_train_config if config is None else config)
        epochs = int(config.get("direct_epochs", 0) or 0)
        if not TORCH_AVAILABLE or epochs <= 0 or self.theta_net is None or self.zeta_net is None:
            return None
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        resolved_batch_strategy = normalize_nn_batch_strategy(config.get("direct_batch_strategy", self.nn_batch_strategy))
        if resolved_batch_strategy == "full-batch":
            batch_size = max(1, self.n_samples)
        else:
            batch_size = max(1, int(config.get("direct_batch_size", self.nn_batch_size)))
        shuffle = bool(config.get("direct_shuffle", self.nn_shuffle))
        lr = float(config.get("direct_lr", max(self.nn_learning_rate * 20.0, 1e-4)))
        eta_min_ratio = max(float(config.get("direct_eta_min_ratio", 0.10)), 0.0)
        grad_clip = float(config.get("direct_grad_clip", 3.0))
        wd = float(config.get("adam_weight_decay", 0.0) or 0.0)
        loss_kind = str(config.get("direct_loss", "mse")).strip().lower()
        beta = float(config.get("direct_beta", 1.0))
        log_interval = int(config.get("direct_log_interval", 25) or 0)
        proxy_kkt_w = float(config.get("proxy_kkt_loss_weight", 0.0))
        feature_noise_std = max(float(config.get("feature_noise_std", 0.0) or 0.0), 0.0)

        np_targets = self._build_bcd_direct_targets(union_analysis, config)
        target_theta = torch.tensor(np_targets["theta"], dtype=torch.float32, device=self.device)
        target_zeta = torch.tensor(np_targets["zeta"], dtype=torch.float32, device=self.device)
        theta_scale = torch.clamp(torch.mean(torch.abs(target_theta)), min=0.25)
        zeta_scale = torch.clamp(torch.mean(torch.abs(target_zeta)), min=0.25)
        print(
            f"[BCD][direct-NN] epochs={epochs}, direct_loss={loss_kind}, "
            f"proxy_residual_mean={np_targets['proxy_residual_mean']:.6f}",
            flush=True,
        )

        def group_loss(pred, target, scale):
            err = (pred - target) / scale
            if loss_kind in ("mse", "l2"):
                return torch.sum(err * err)
            return torch.nn.functional.smooth_l1_loss(err, torch.zeros_like(err), beta=beta, reduction="sum")

        self.theta_net.train()
        self.zeta_net.train()
        theta_optimizer = optim.AdamW(self.theta_net.parameters(), lr=lr, weight_decay=wd)
        zeta_optimizer = optim.AdamW(self.zeta_net.parameters(), lr=lr, weight_decay=wd)
        theta_scheduler = optim.lr_scheduler.CosineAnnealingLR(theta_optimizer, T_max=max(epochs, 1), eta_min=lr * eta_min_ratio)
        zeta_scheduler = optim.lr_scheduler.CosineAnnealingLR(zeta_optimizer, T_max=max(epochs, 1), eta_min=lr * eta_min_ratio)
        last_avg = None
        for epoch in range(epochs):
            epoch_loss = 0.0
            sample_indices = np.arange(self.n_samples, dtype=int)
            if shuffle and self.n_samples > 1:
                np.random.shuffle(sample_indices)
            theta_optimizer.zero_grad()
            zeta_optimizer.zero_grad()
            for sample_pos, _sample_id in enumerate(sample_indices):
                batch_start = (sample_pos // batch_size) * batch_size
                is_batch_end = ((sample_pos + 1) % batch_size == 0) or sample_pos == self.n_samples - 1
                if not is_batch_end:
                    continue
                batch_ids = sample_indices[batch_start: sample_pos + 1]
                batch_index = torch.as_tensor(batch_ids, dtype=torch.long, device=self.device)
                features_tensor = self._features_cache.index_select(0, batch_index)
                if feature_noise_std > 0:
                    features_tensor = features_tensor + torch.randn_like(features_tensor) * feature_noise_std
                pred_theta = self.theta_net(features_tensor)
                pred_zeta = self.zeta_net(features_tensor)
                loss = group_loss(pred_theta, target_theta.index_select(0, batch_index), theta_scale)
                loss = loss + group_loss(pred_zeta, target_zeta.index_select(0, batch_index), zeta_scale)
                if proxy_kkt_w > 0:
                    for local_pos, sid in enumerate(batch_ids):
                        zeta_tensor = pred_zeta[local_pos]
                        if self._unit_predictor_active():
                            zeta_tensor = self._apply_unit_predictor_zeta_override(
                                zeta_tensor,
                                int(sid),
                                train_predictor=False,
                            )
                        loss = loss + proxy_kkt_w * self.loss_function_differentiable(
                            int(sid),
                            pred_theta[local_pos],
                            zeta_tensor,
                            union_analysis=union_analysis,
                            device=self.device,
                        )
                (loss / max(1, len(batch_ids))).backward()
                epoch_loss += float(loss.detach().cpu().item())
                torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), max_norm=grad_clip)
                torch.nn.utils.clip_grad_norm_(self.zeta_net.parameters(), max_norm=grad_clip)
                theta_optimizer.step()
                zeta_optimizer.step()
                theta_optimizer.zero_grad()
                zeta_optimizer.zero_grad()
            theta_scheduler.step()
            zeta_scheduler.step()
            last_avg = epoch_loss / max(self.n_samples, 1)
            if epoch == 0 or epoch == epochs - 1 or (log_interval > 0 and (epoch + 1) % log_interval == 0):
                print(
                    f"  [BCD][direct-NN] epoch {epoch+1:>4}/{epochs}, "
                    f"avg_target_loss={last_avg:.6f}, lr={theta_optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        self.theta_net.eval()
        self.zeta_net.eval()
        self.refresh_theta_zeta_values_from_networks()
        return {"avg_target_loss": None if last_avg is None else float(last_avg)}
    
    def loss_function_differentiable(self, sample_id, theta_tensor, zeta_tensor,
                                     union_analysis=None, device=None,
                                     return_components: bool = False):
        """
        可微分的loss函数，用于神经网络训练（完全复制自uc_NN.py）
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装，无法使用可微分loss函数")
        
        if device is None:
            device = self.device if hasattr(self, 'device') else torch.device('cpu')
        
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        # 从 Tier1/Tier3 缓存读取，避免每次调用重建张量
        # ============================================
        # 重要：使用BCD迭代得到的变量值（从iter_with_pg_block和iter_with_dual_block得到）
        # ============================================
        # x: Tier3 缓存（view，无拷贝）
        x = self._x_cache[sample_id]  # (ng, T)

        # lambda: Tier3 缓存
        lc = self._lambda_cache[sample_id]
        et = self._empty_tensor  # 空张量占位
        lambda_cpower     = lc.get('lambda_cpower',     et)
        lambda_pg_lower   = lc.get('lambda_pg_lower',   et)
        lambda_pg_upper   = lc.get('lambda_pg_upper',   et)
        lambda_ramp_down  = lc.get('lambda_ramp_down',  et)
        lambda_ramp_up    = lc.get('lambda_ramp_up',    et)
        lambda_min_on     = lc.get('lambda_min_on',     et)
        lambda_min_off    = lc.get('lambda_min_off',    et)
        lambda_start_cost = lc.get('lambda_start_cost', et)
        lambda_shut_cost  = lc.get('lambda_shut_cost',  et)
        lambda_x_upper    = lc.get('lambda_x_upper',    et)
        lambda_x_lower    = lc.get('lambda_x_lower',    et)

        # Tier1 常量张量
        ct = self._ct
        gen_PMIN      = ct['gen_PMIN']
        gen_PMAX      = ct['gen_PMAX']
        gencost_fixed = ct['gencost_fixed']
        start_cost    = ct['start_cost']
        shut_cost     = ct['shut_cost']
        Ton = self.Ton
        Toff = self.Toff
        Ru    = ct['Ru']
        Rd    = ct['Rd']
        Ru_co = ct['Ru_co']
        Rd_co = ct['Rd_co']

        # mu, ita: Tier3 缓存（view，无拷贝）
        mu  = self._mu_cache[sample_id]   # (nl, T)
        ita = self._ita_cache[sample_id]  # (ng, T)
        
        use_fb_for_loss = getattr(self, 'use_fischer_burmeister_for_loss', False)
        
        final_loss = self._compute_loss_with_lambda(
            sample_id, theta_tensor, zeta_tensor, mu, ita,
            lambda_x_upper.detach(), lambda_x_lower.detach(),
            lambda_min_on.detach(), lambda_min_off.detach(),
            lambda_start_cost.detach(), lambda_shut_cost.detach(),
            x, lambda_cpower, lambda_pg_lower, lambda_pg_upper,
            lambda_ramp_down, lambda_ramp_up,
            gen_PMIN, gen_PMAX, gencost_fixed, start_cost, shut_cost,
            Ru, Rd, Ru_co, Rd_co, Ton, Toff, union_analysis, device,
            use_fischer_burmeister=use_fb_for_loss,
            return_components=return_components,
        )
        
        return final_loss
    
    def _compute_loss_with_lambda(self, sample_id, theta_tensor, zeta_tensor, mu_tensor, ita_tensor,
                                   lambda_x_upper, lambda_x_lower, lambda_min_on, lambda_min_off,
                                   lambda_start_cost, lambda_shut_cost,
                                   x, lambda_cpower, lambda_pg_lower, lambda_pg_upper,
                                   lambda_ramp_down, lambda_ramp_up,
                                   gen_PMIN, gen_PMAX, gencost_fixed, start_cost, shut_cost,
                                   Ru, Rd, Ru_co, Rd_co, Ton, Toff, union_analysis, device,
                                   use_fischer_burmeister=False,
                                   return_components: bool = False):
        """
        计算包含Lambda变量的loss（完全复制自uc_NN.py）
        """
        # mu_tensor和ita_tensor已经是(nl, T)和(ng, T)的形状，不需要reshape
        # 但如果传入的是1D张量，则reshape
        if mu_tensor.dim() == 1:
            mu = mu_tensor.view(self.nl, self.T)
        else:
            mu = mu_tensor  # 已经是(nl, T)的形状

        if ita_tensor.dim() == 1:
            ita = ita_tensor.view(self.ng, self.T)
        else:
            ita = ita_tensor  # 已经是(ng, T)的形状

        # 如果 union_analysis 对象发生变化，重建约束元数据缓存
        if id(union_analysis) != self._cached_union_analysis_id:
            self._preprocess_union_analysis_cache(union_analysis)

        obj_primal = torch.tensor(0.0, device=device, requires_grad=True)
        obj_opt = torch.tensor(0.0, device=device, requires_grad=True)
        obj_dual_x = torch.tensor(0.0, device=device, requires_grad=True)

        # 预缓存常量张量（避免循环内 torch.tensor 分配）
        _one = self._const_one
        _default_mu = self._default_mu_tensor

        # 计算theta相关的参数化约束损失（使用预处理缓存，无字符串格式化）
        theta_constraint_scale = self._current_theta_curriculum_scale()
        for branch_id, time_slot, _ctype, coeff_list, rhs_idx in self._cached_theta_constraints:
            lhs_expr = 0.0
            for unit_id, member_time, theta_idx in coeff_list:
                lhs_expr = lhs_expr + theta_constraint_scale * theta_tensor[theta_idx] * x[unit_id, member_time]

            theta_rhs = theta_constraint_scale * (theta_tensor[rhs_idx] if rhs_idx >= 0 else _one)

            direction = self._get_theta_constraint_direction(branch_id, time_slot)
            violation = direction * (lhs_expr - theta_rhs)
            obj_primal = obj_primal + self._smooth_relu(violation)

            if branch_id < self.nl:
                obj_opt = obj_opt + self._smooth_abs(violation) * torch.abs(mu[branch_id, time_slot])
            else:
                obj_opt = obj_opt + self._smooth_abs(violation) * _default_mu

        # 计算zeta相关的参数化约束损失（使用预处理缓存）
        for unit_id, time_slot, zeta_idx, rhs_idx in self._cached_zeta_constraints:
            zeta = zeta_tensor[zeta_idx]
            lhs_expr = zeta * x[unit_id, time_slot]

            zeta_rhs = zeta_tensor[rhs_idx] if rhs_idx >= 0 else _one

            direction = self._get_zeta_constraint_direction(unit_id, time_slot)
            violation = direction * (lhs_expr - zeta_rhs)
            obj_primal = obj_primal + self._smooth_relu(violation)
            obj_opt = obj_opt + self._smooth_abs(violation) * torch.abs(ita[unit_id, time_slot])

        # theta/zeta 仅影响 x 驻点条件，因此可微 dual loss 只包含 obj_dual_x。
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self._build_x_dual_stationarity_expr(
                    g=g,
                    t=t,
                    lambda_cpower=lambda_cpower,
                    lambda_x_upper=lambda_x_upper,
                    lambda_x_lower=lambda_x_lower,
                    lambda_pg_lower=lambda_pg_lower,
                    lambda_pg_upper=lambda_pg_upper,
                    lambda_ramp_down=lambda_ramp_down,
                    lambda_ramp_up=lambda_ramp_up,
                    lambda_min_on=lambda_min_on,
                    lambda_min_off=lambda_min_off,
                    lambda_start_cost=lambda_start_cost,
                    lambda_shut_cost=lambda_shut_cost,
                    fixed_cost=gencost_fixed[g],
                    pmin=gen_PMIN[g],
                    pmax=gen_PMAX[g],
                    rd_delta=Rd_co[g] - Rd[g],
                    ru_delta=Ru_co[g] - Ru[g],
                    start_cost=start_cost[g],
                    shut_cost=shut_cost[g],
                    Ton=Ton,
                    Toff=Toff,
                )

                # 参数化约束的对偶贡献（theta相关）——使用查找表，O(1) 替代全量扫描
                for theta_idx, branch_id, anchor_time in self._dual_theta_lookup.get((g, t), []):
                    theta = theta_constraint_scale * theta_tensor[theta_idx]
                    direction = self._get_theta_constraint_direction(branch_id, anchor_time)
                    if branch_id < self.nl:
                        dual_expr = dual_expr + direction * theta * mu[branch_id, anchor_time]
                    else:
                        dual_expr = dual_expr + direction * theta * _default_mu

                # 参数化约束的对偶贡献（zeta相关）——使用查找表
                for zeta_idx in self._dual_zeta_lookup.get((g, t), []):
                    direction = self._get_zeta_constraint_direction(g, t)
                    dual_expr = dual_expr + direction * zeta_tensor[zeta_idx] * ita[g, t]

                obj_dual_x = obj_dual_x + self._smooth_abs(dual_expr)
        
        # L1 正则化：[-1, 1] 死区内不惩罚，仅惩罚超出死区的幅值
        theta_deadzone_excess = self._smooth_deadband_excess(theta_tensor, 1.0)
        zeta_deadzone_excess = self._smooth_deadband_excess(zeta_tensor, 1.0)
        reg_base = (
            self.theta_reg_weight * torch.sum(theta_deadzone_excess)
            + self.zeta_reg_weight * torch.sum(zeta_deadzone_excess)
        )
        reg_loss = reg_base * self._get_regularization_scale()

        # 迭代间差异正则：控制输出系数与上一代差异不过大（deadband 外二次惩罚）
        if self.iter_delta_reg_weight > 0 and self._prev_theta_values_list is not None:
            prev_theta_dict = self._prev_theta_values_list[sample_id]
            prev_zeta_dict = self._prev_zeta_values_list[sample_id]
            prev_theta = torch.tensor(
                [float(prev_theta_dict.get(name, 0.0)) for name in self.theta_var_names],
                dtype=theta_tensor.dtype,
                device=device,
            )
            prev_zeta = torch.tensor(
                [float(prev_zeta_dict.get(name, 0.0)) for name in self.zeta_var_names],
                dtype=zeta_tensor.dtype,
                device=device,
            )
            theta_diff = theta_tensor - prev_theta
            zeta_diff = zeta_tensor - prev_zeta
            if self.iter_delta_reg_deadband > 0:
                theta_excess = self._smooth_deadband_excess(
                    theta_diff,
                    self.iter_delta_reg_deadband,
                )
                zeta_excess = self._smooth_deadband_excess(
                    zeta_diff,
                    self.iter_delta_reg_deadband,
                )
                iter_delta_reg = torch.sum(theta_excess ** 2) + torch.sum(zeta_excess ** 2)
            else:
                iter_delta_reg = torch.sum(theta_diff ** 2) + torch.sum(zeta_diff ** 2)
            reg_loss = reg_loss + self.iter_delta_reg_weight * iter_delta_reg

        primal_term = self.loss_ratio_primal * obj_primal * self.rho_primal
        dual_x_term = self.loss_ratio_dual_x * obj_dual_x * self.rho_dual_x
        opt_term = self.loss_ratio_opt * obj_opt * self.rho_opt
        scaled_reg_loss = self.loss_ratio_reg * reg_loss
        loss = primal_term + dual_x_term + opt_term + scaled_reg_loss

        if not return_components:
            return loss

        components = {
            'obj_primal': obj_primal.detach(),
            'obj_dual_x': obj_dual_x.detach(),
            'obj_opt': obj_opt.detach(),
            'primal_term': primal_term.detach(),
            'dual_x_term': dual_x_term.detach(),
            'opt_term': opt_term.detach(),
            'reg_term': scaled_reg_loss.detach(),
        }
        return loss, components
      
    def _add_parametric_penalties_pg_block(self, model, x, sample_id, theta_values=None, union_analysis=None, PTDF=None, branch_limit=None):
        """
        添加包含theta参数的DCPF罚项（参考uc_dfsm_bcd.py，但使用直接优化系数的形式）
        
        Args:
            model: Gurobi模型
            x: 二进制变量
            sample_id: 样本ID
            theta_values: theta参数值字典
            union_analysis: 并集约束分析结果
            PTDF: 预计算的PTDF矩阵
            branch_limit: 预计算的线路容量限制
            
        Returns:
            model, obj_primal, obj_opt
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return model, gp.LinExpr(), gp.LinExpr()
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        # 如果没有传入预计算的矩阵，则计算
        if PTDF is None:
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                if 0 <= bus_idx < nb:
                    G[bus_idx, g] = 1
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        if branch_limit is None:
            branch_limit = self.branch[:, RATE_A]
        
        try:
            constraint_count = 0
            obj_primal = gp.LinExpr()
            obj_opt = gp.LinExpr()
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'dcpf')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                # 构建左端项表达式 - 直接使用theta值
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    original_coeff = 0
                    
                    # 直接获取theta变量值
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    theta = self._current_theta_curriculum_scale() * theta_values.get(theta_name, 0.0)
                    
                    # 直接使用theta值作为系数
                    parametric_coeff = original_coeff + theta
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, member_time]
                
                # 构建右端项 - 从theta_values字典中获取
                theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
                if theta_rhs_name in theta_values:
                    parametric_rhs = self._current_theta_curriculum_scale() * theta_values[theta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                direction = self._get_theta_constraint_direction(branch_id, time_slot)
                # 添加违反量和绝对值变量
                parametric_rhs_viol = model.addVar(lb=0, name=f'parametric_rhs_viol_{branch_id}_{time_slot}')
                parametric_rhs_abs = model.addVar(lb=0, name=f'parametric_rhs_abs_{branch_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol >= direction * (lhs_expr - parametric_rhs), name=f'parametric_rhs_viol_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= lhs_expr - parametric_rhs, name=f'parametric_rhs_abs1_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= -lhs_expr + parametric_rhs, name=f'parametric_rhs_abs2_{branch_id}_{time_slot}')

                obj_primal += parametric_rhs_viol

                # 获取mu值（如果存在）
                if hasattr(self, 'mu') and sample_id < len(self.mu) and branch_id < self.nl:
                    mu_val = self.mu[sample_id, branch_id, time_slot]
                    obj_opt += parametric_rhs_abs * abs(mu_val)
                else:
                    # 使用默认mu值
                    default_mu = getattr(self, 'dual_para_bound', 0.1)
                    obj_opt += parametric_rhs_abs * default_mu

            model.update()
            
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return model, gp.LinExpr(), gp.LinExpr()
    
    def _add_parametric_balance_power_penalties_pg_block(self, model, x, sample_id, zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的平衡功率罚项（参考uc_dfsm_bcd.py，但使用直接优化系数的形式）
        
        Args:
            model: Gurobi模型
            x: 二进制变量
            sample_id: 样本ID
            zeta_values: zeta参数值字典
            union_analysis: 并集约束分析结果
            
        Returns:
            model, obj_primal, obj_opt
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return model, gp.LinExpr(), gp.LinExpr()
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            zeta_values = {}
        
        try:
            obj_primal = gp.LinExpr()
            obj_opt = gp.LinExpr()
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 0

                # 构建左端项表达式 - 直接使用zeta值
                lhs_expr = 0
                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)

                # 直接使用zeta值作为系数
                parametric_coeff = original_coeff + zeta

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项 - 从zeta_values字典中获取
                zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                if zeta_rhs_name in zeta_values:
                    parametric_rhs = zeta_values[zeta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                direction = self._get_zeta_constraint_direction(unit_id, time_slot)
                # 添加违反量和绝对值变量
                parametric_rhs_viol = model.addVar(lb=0, name=f'parametric_balance_power_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVar(lb=0, name=f'parametric_balance_power_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol >= direction * (lhs_expr - parametric_rhs), name=f'parametric_balance_power_rhs_viol_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_rhs_abs1_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= -lhs_expr + parametric_rhs, name=f'parametric_balance_power_rhs_abs2_{unit_id}_{time_slot}')
                
                obj_primal += parametric_rhs_viol

                # 获取ita值（如果存在）
                if hasattr(self, 'ita') and sample_id < len(self.ita):
                    ita_val = self.ita[sample_id, unit_id, time_slot]
                    obj_opt += parametric_rhs_abs * abs(ita_val)
                else:
                    # 使用默认ita值
                    default_ita = getattr(
                        self,
                        'ita_dual_floor_init',
                        getattr(self, 'dual_para_bound', 0.1),
                    )
                    obj_opt += parametric_rhs_abs * default_ita

            model.update()
            
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return model, gp.LinExpr(), gp.LinExpr()
    
    def _add_parametric_penalties_pg_block_const_to_model(self, model, x, sample_id, theta_values, union_analysis):
        """添加theta参数化约束到模型（严格按照uc_NN.py的_add_parametric_penalties_pg_block_solid）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        try:
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'dcpf')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                # 构建左端项表达式 - 直接使用theta值，不再使用多项式参数化
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    original_coeff = 0
                    
                    # 直接获取theta变量值（不再使用多项式）
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    theta = self._current_theta_curriculum_scale() * theta_values.get(theta_name, 0.0)
                    
                    # 直接使用theta值作为系数
                    parametric_coeff = original_coeff + theta
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, member_time]
                
                # 构建右端项 - 从theta_values字典中获取
                theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
                if theta_values is not None and theta_rhs_name in theta_values:
                    parametric_rhs = self._current_theta_curriculum_scale() * theta_values[theta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                direction = self._get_theta_constraint_direction(branch_id, time_slot)
                model.addConstr(direction * lhs_expr <= direction * parametric_rhs, name=f'parametric_solid_{branch_id}_{time_slot}')

            model.update()
            
            return model
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _add_parametric_balance_power_penalties_pg_block_const_to_model(self, model, x, sample_id, zeta_values, union_analysis):
        """添加zeta参数化约束到模型（严格按照uc_NN.py的_add_parametric_balance_power_penalties_pg_block_solid）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            zeta_values = {}
        
        try:            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 0

                # 构建左端项表达式 - 直接使用zeta值，不再使用多项式参数化
                lhs_expr = 0
                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)

                # 直接使用zeta值作为系数
                parametric_coeff = original_coeff + zeta

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项 - 从zeta_values字典中获取
                zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                if zeta_values is not None and zeta_rhs_name in zeta_values:
                    parametric_rhs = zeta_values[zeta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                direction = self._get_zeta_constraint_direction(unit_id, time_slot)
                model.addConstr(direction * lhs_expr <= direction * parametric_rhs, name=f'parametric_balance_power_solid_{unit_id}_{time_slot}')

            model.update()
            
            return model
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _add_parametric_constraints_dual_block_const_to_model(self, model, g_id, t_id, mu, sample_id, theta_values, union_analysis):
        """添加theta参数化约束的对偶贡献到模型（严格按照uc_NN.py的_add_parametric_constraints_dual_block_const）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return 0.0
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        try:
            dual_expr = 0.0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'dcpf')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                # 直接使用theta值，不再使用多项式参数化
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    if unit_id != g_id or member_time != t_id:
                        continue
                    original_coeff = 0
                    
                    # 直接获取theta变量值（不再使用多项式）
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    theta = self._current_theta_curriculum_scale() * theta_values.get(theta_name, 0.0)
                    
                    # 直接使用theta值作为系数
                    parametric_coeff = original_coeff + theta
                    
                    # 检查branch_id是否在有效范围内（真实支路范围）
                    if branch_id < self.nl:
                        dual_expr += self._get_theta_constraint_direction(branch_id, time_slot) * parametric_coeff * mu[branch_id, time_slot]
                    else:
                        # 对于手动约束（虚拟branch_id），使用默认mu值
                        default_mu = getattr(self, 'dual_para_bound', 0.1)
                        dual_expr += parametric_coeff * default_mu

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()   
            return 0.0
    
    def _add_parametric_balance_power_constraints_dual_block_const_to_model(self, model, g_id, t_id, ita, sample_id, zeta_values, union_analysis):
        """添加zeta参数化约束的对偶贡献到模型（严格按照uc_NN.py的_add_parametric_balance_power_constraints_dual_block_const）"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return 0.0

        union_constraints = union_analysis['union_zeta_constraints']

        if zeta_values is None:
            zeta_values = {}
        
        try:            
            dual_expr = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                
                if time_slot != t_id or unit_id != g_id:
                    continue

                original_coeff = 0
                
                # 直接使用zeta值，不再使用多项式参数化
                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)
                
                parametric_coeff = original_coeff + zeta
                    
                dual_expr += self._get_zeta_constraint_direction(unit_id, time_slot) * parametric_coeff * ita[unit_id, time_slot]

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()   
            return 0.0
    
    def _add_parametric_obj_dual_block(self, model, x, mu, mu_abs, sample_id, theta_values=None, union_analysis=None, PTDF=None, branch_limit=None):
        """
        添加包含theta参数的DCPF罚项到obj_opt（参考uc_dfsm_bcd.py，但使用直接优化系数的形式）
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return model, gp.LinExpr()
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        if PTDF is None:
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                if 0 <= bus_idx < nb:
                    G[bus_idx, g] = 1
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        if branch_limit is None:
            branch_limit = self.branch[:, RATE_A]
        
        try:
            obj_opt = gp.LinExpr()
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'dcpf')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                
                # 构建左端项表达式 - 直接使用theta值
                lhs_expr = 0.0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    member_time = self._theta_member_time_index(constraint_info, coeff_info)
                    original_coeff = 0.0
                    
                    # 直接获取theta变量值
                    theta_name = self._theta_var_name(branch_id, unit_id, member_time)
                    theta = self._current_theta_curriculum_scale() * theta_values.get(theta_name, 0.0)
                    
                    # 直接使用theta值作为系数
                    parametric_coeff = original_coeff + theta
                    
                    # 添加到左端项（使用当前的x值，numpy数组）
                    lhs_expr += parametric_coeff * x[unit_id, member_time]
                
                # 构建右端项
                theta_rhs_name = self._theta_rhs_name(branch_id, time_slot)
                if theta_rhs_name in theta_values:
                    parametric_rhs = self._current_theta_curriculum_scale() * theta_values[theta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                # 过滤数值精度问题
                if parametric_rhs_abs > 1e-10:
                    if branch_id < self.nl:
                        obj_opt += parametric_rhs_abs * mu_abs[branch_id, time_slot]
                    else:
                        # 对于手动约束（虚拟branch_id），使用默认mu值
                        default_mu = getattr(self, 'dual_para_bound', 0.1)
                        obj_opt += parametric_rhs_abs * default_mu

            model.update()

            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return model, gp.LinExpr()
    
    def _add_parametric_balance_power_obj_dual_block(self, model, x, ita, ita_abs, sample_id, zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的平衡功率罚项到obj_opt（参考uc_dfsm_bcd.py，但使用直接优化系数的形式）
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return model, gp.LinExpr()
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            zeta_values = {}

        try:
            obj_opt = gp.LinExpr()
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 0.0

                # 构建左端项表达式 - 直接使用zeta值
                lhs_expr = 0.0
                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)

                # 直接使用zeta值作为系数
                parametric_coeff = original_coeff + zeta

                # 添加到左端项（使用当前的x值，numpy数组）
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项
                zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                if zeta_rhs_name in zeta_values:
                    parametric_rhs = zeta_values[zeta_rhs_name]
                else:
                    parametric_rhs = 1.0  # 默认值

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                # 过滤数值精度问题
                if parametric_rhs_abs > 1e-10:
                    obj_opt += parametric_rhs_abs * ita_abs[unit_id, time_slot]

            model.update()

            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return model, gp.LinExpr()

    def heuristic_sol_x_spec(self, sample_id, x_LP, x_LP_refined):
        gap = 0.4
        x_heu = np.round(x_LP_refined).astype(int)
        x_heu[np.logical_and(x_LP <= gap, x_LP_refined <= gap)] = 0
        x_heu[np.logical_and(x_LP <= gap, x_LP_refined >= gap)] = 1

        x_heu[np.logical_and(x_LP >= 1 - gap, x_LP_refined <= 1 - gap)] = 0
        x_heu[np.logical_and(x_LP >= 1 - gap, x_LP_refined >= 1 - gap)] = 1

        mask_middle = np.logical_and(x_LP > gap, x_LP < 1 - gap)
        x_heu[np.logical_and(mask_middle, x_LP_refined <= x_LP)] = 0
        x_heu[np.logical_and(mask_middle, x_LP_refined >= x_LP)] = 1
        
        return x_heu
    
    def analyse_surrogate_model_totle(self):
        """
        分析代理模型的总体性能（参考uc_NN.py）
        比较有无theta约束时的LP解与整数解的差异
        """
        differ_LP = 0
        differ_LP_refined = 0
        differ_LP_heu = 0
        differ_LP_heu_refined = 0
        differ_spec_heu = 0
        
        for sample_id in range(self.n_samples):
            # 求解不含theta约束的LP
            sol_LP = self.solve_LP_without_theta_constraints(sample_id)
            if sol_LP is None:
                continue
            x_LP = sol_LP[1]
            
            # 求解含theta约束的LP
            sol_LP_refined = self.solve_LP_with_theta_constraints(sample_id, self._current_union_analysis)
            if sol_LP_refined is None:
                continue
            x_LP_refined = sol_LP_refined[1]
            
            # 计算与整数解的差异
            x_int = _get_uc_matrix_from_sample(
                self.active_set_data[sample_id], self.ng, self.T)
            if x_int is None:
                print(f"警告: 样本 {sample_id} 缺少 unit_commitment_matrix，跳过", flush=True)
                continue
            
            differ_LP += np.sum(np.abs(x_LP - x_int))
            differ_LP_refined += np.sum(np.abs(x_LP_refined - x_int))
            
            differ_LP_heu += np.sum(np.abs(np.round(x_LP).astype(int) - x_int))
            differ_LP_heu_refined += np.sum(np.abs(np.round(x_LP_refined).astype(int) - x_int))
            
            differ_spec_heu_sample = np.sum(np.abs(self.heuristic_sol_x_spec(sample_id, x_LP, x_LP_refined) - self.active_set_data[sample_id]['unit_commitment_matrix']))
            differ_spec_heu += differ_spec_heu_sample
            
            if sample_id == 1:
                x_true = x_int
                x_round1 = np.round(x_LP).astype(int)
                x_round2 = np.round(x_LP_refined).astype(int)
                x_round3 = np.round(self.heuristic_sol_x_spec(sample_id, x_LP, x_LP_refined)).astype(int)
                self.plot_sample0_binary_comparison(x_true, x_round1, x_round2,
                                                    labels=("True optimum", "LP Optimized", "BCD-Neural Optimized"),
                                                    save_path="result/figures/sample0_binary_comparison.png",
                                                    show=True)
                
                # 求解ED问题并比较最优值
                if ED_GUROBI_AVAILABLE:
                    print("\n" + "="*60, flush=True)
                    print("Sample 0: ED问题求解结果比较", flush=True)
                    print("="*60, flush=True)
                    
                    Pd_sample0 = self.active_set_data[sample_id]['pd_data']
                    
                    # 求解x_true对应的ED问题
                    try:
                        ed_true = EconomicDispatchGurobi(self.ppc, Pd_sample0, self.T_delta, x_true)
                        ed_true.model.Params.OutputFlag = 0
                        ed_true.model.optimize()
                        if ed_true.model.status == GRB.OPTIMAL:
                            obj_ed_true = ed_true.model.ObjVal
                            print(f"✓ x_true (真实最优解) 的ED最优值: {obj_ed_true:.6f}", flush=True)
                        else:
                            obj_ed_true = None
                            print(f"✗ x_true 的ED问题求解失败，状态: {ed_true.model.status}", flush=True)
                    except Exception as e:
                        obj_ed_true = None
                        print(f"✗ x_true 的ED问题求解出错: {e}", flush=True)
                    
                    # 求解x_round1对应的ED问题
                    try:
                        ed_round1 = EconomicDispatchGurobi(self.ppc, Pd_sample0, self.T_delta, x_round1)
                        ed_round1.model.Params.OutputFlag = 0
                        ed_round1.model.optimize()
                        if ed_round1.model.status == GRB.OPTIMAL:
                            obj_ed_round1 = ed_round1.model.ObjVal
                            print(f"✓ x_round1 (LP Optimized) 的ED最优值: {obj_ed_round1:.6f}", flush=True)
                        else:
                            obj_ed_round1 = None
                            print(f"✗ x_round1 的ED问题求解失败，状态: {ed_round1.model.status}", flush=True)
                    except Exception as e:
                        obj_ed_round1 = None
                        print(f"✗ x_round1 的ED问题求解出错: {e}", flush=True)
                    
                    # 求解x_round2对应的ED问题
                    try:
                        ed_round2 = EconomicDispatchGurobi(self.ppc, Pd_sample0, self.T_delta, x_round2)
                        ed_round2.model.Params.OutputFlag = 0
                        ed_round2.model.optimize()
                        if ed_round2.model.status == GRB.OPTIMAL:
                            obj_ed_round2 = ed_round2.model.ObjVal
                            print(f"✓ x_round2 (BCD-Neural Optimized) 的ED最优值: {obj_ed_round2:.6f}", flush=True)
                        else:
                            obj_ed_round2 = None
                            print(f"✗ x_round2 的ED问题求解失败，状态: {ed_round2.model.status}", flush=True)
                    except Exception as e:
                        obj_ed_round2 = None
                        print(f"✗ x_round2 的ED问题求解出错: {e}", flush=True)
                        
                    # 求解x_round3对应的ED问题
                    try:
                        ed_round3 = EconomicDispatchGurobi(self.ppc, Pd_sample0, self.T_delta, x_round3)
                        ed_round3.model.Params.OutputFlag = 0
                        ed_round3.model.optimize()
                        if ed_round3.model.status == GRB.OPTIMAL:
                            obj_ed_round3 = ed_round3.model.ObjVal
                            print(f"✓ x_round3 (BCD-Neural Optimized) 的ED最优值: {obj_ed_round3:.6f}", flush=True)
                        else:
                            obj_ed_round3 = None
                            print(f"✗ x_round3 的ED问题求解失败，状态: {ed_round3.model.status}", flush=True)
                    except Exception as e:
                        obj_ed_round3 = None
                        print(f"✗ x_round3 的ED问题求解出错: {e}", flush=True)                    
        
        print(f"最优间隙（不含theta约束）: {differ_LP}", flush=True)
        print(f"最优间隙（含theta约束）: {differ_LP_refined}", flush=True)
        print(f"最优间隙（四舍五入不含theta约束）: {differ_LP_heu}", flush=True)
        print(f"最优间隙（四舍五入含theta约束）: {differ_LP_heu_refined}", flush=True)
        print(f"特殊解恢复恢复间隙: {differ_spec_heu}", flush=True)
    
    def solve_LP_without_theta_constraints(self, sample_id, union_analysis=None):
        """
        求解不含theta约束的LP问题（参考uc_NN.py）
        """
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('LP_without_theta')
        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 功率平衡约束
        for t in range(self.T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(self.ng)) == gp.quicksum(Pd[:, t]),
                name=f'power_balance_{t}'
            )
        
        # 发电上下限约束
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(pg[g, t] >= self.gen[g, PMIN] * x[g, t], name=f'pg_lower_{g}_{t}')
                model.addConstr(pg[g, t] <= self.gen[g, PMAX] * x[g, t], name=f'pg_upper_{g}_{t}')
        
        # 爬坡约束
        Ru = self.Ru
        Rd = self.Rd
        Ru_co = self.Ru_co
        Rd_co = self.Rd_co
        for g in range(self.ng):
            for t in range(1, self.T):
                model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1]), 
                              name=f'ramp_up_{g}_{t}')
                model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t]), 
                              name=f'ramp_down_{g}_{t}')
        
        # 最小开关机时间约束
        Ton = self.Ton
        Toff = self.Toff
        for g in range(self.ng):
            for t in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - t):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+t], name=f'min_on_{g}_{t}_{t1}')
        for g in range(self.ng):
            for t in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - t):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+t], name=f'min_off_{g}_{t}_{t1}')
        
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]), name=f'start_cost_{g}_{t}')
                model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]), name=f'shut_cost_{g}_{t}')
                model.addConstr(coc[g, t-1] >= 0, name=f'coc_nonneg_{g}_{t}')
        
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] >= self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t],
                              name=f'cpower_{g}_{t}')
        
        # 目标函数
        primal_obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                     gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
        model.setObjective(primal_obj, GRB.MINIMIZE)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            
            if sample_id <= 2:
                print(f"solve_LP_without_theta_constraints, sample_id: {sample_id}, obj: {model.ObjVal}", flush=True)
            
            return pg_sol, x_sol, coc_sol, cpower_sol
        else:
            return None
    
    def solve_LP_with_theta_constraints(self, sample_id, union_analysis=None):
        """
        求解含theta约束的LP问题（参考uc_dfsm_bcd.py，使用罚项形式）
        使用神经网络生成的theta和zeta值
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('LP_with_theta')
        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 构建约束和目标函数项（参考uc_dfsm_bcd.py）
        obj = 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_expr == 0, name=f'power_balance_{t}')
            
            # 发电上下限约束
            for g in range(self.ng):
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_expr <= 0, name=f'pg_lower_{g}_{t}')
                
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_expr <= 0, name=f'pg_upper_{g}_{t}')
        
        # 爬坡约束
        Ru = self.Ru
        Rd = self.Rd
        Ru_co = self.Ru_co
        Rd_co = self.Rd_co
        
        for t in range(1, self.T):
            for g in range(self.ng):
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_expr <= 0, name=f'ramp_up_{g}_{t}')
                
                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_expr <= 0, name=f'ramp_down_{g}_{t}')
        
        # 最小开机时间和最小关机时间约束
        Ton = self.Ton
        Toff = self.Toff
        for g in range(self.ng):
            for t in range(1, self._min_up_horizon(g) + 1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_expr <= 0, name=f'min_on_{g}_{t}_{t1}')
        
        for g in range(self.ng):
            for t in range(1, self._min_down_horizon(g) + 1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                    model.addConstr(min_off_expr <= 0, name=f'min_off_{g}_{t}_{t1}')
        
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(start_cost[g] * (x[g, t] - x[g, t-1]) <= coc[g, t-1], name=f'start_cost_{g}_{t}')
                model.addConstr(shut_cost[g] * (x[g, t-1] - x[g, t]) <= coc[g, t-1], name=f'shut_cost_{g}_{t}')
                obj += coc[g, t-1]
        
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')
                obj += cpower[g, t]
        
        # 潮流约束
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr(flow[l] - branch_limit[l] <= 0, name=f'dcpf_upper_{l}_{t}')
                model.addConstr(-flow[l] - branch_limit[l] <= 0, name=f'dcpf_lower_{l}_{t}')
        
        # 添加参数化约束的罚项（参考uc_dfsm_bcd.py）
        obj_primal = None
        features = self._extract_features(sample_id)
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)
        if self.device:
            features_tensor = features_tensor.to(self.device)

        # 推理阶段关闭dropout等随机性：临时切换到eval模式
        theta_was_training = self.theta_net.training if self.theta_net is not None else False
        zeta_was_training = self.zeta_net.training if self.zeta_net is not None else False
        if self.theta_net is not None:
            self.theta_net.eval()
        if self.zeta_net is not None:
            self.zeta_net.eval()

        with torch.no_grad():
            theta_out = self.theta_net(features_tensor)
            zeta_out = self.zeta_net(features_tensor)

        # 恢复之前的训练/评估状态（以防后续继续训练）
        if self.theta_net is not None and theta_was_training:
            self.theta_net.train()
        if self.zeta_net is not None and zeta_was_training:
            self.zeta_net.train()

        # 将网络输出转换为字典形式的theta / zeta 参数（使用已有变量名顺序）
        theta_arr = theta_out.detach().cpu().numpy().flatten()
        theta = {name: float(val) for name, val in zip(self.theta_var_names, theta_arr)}

        model, parametric_obj_primal, _ = self._add_parametric_penalties_pg_block(
            model, x, sample_id, theta, union_analysis, PTDF=PTDF, branch_limit=branch_limit
        )

        if obj_primal is None:
            obj_primal = parametric_obj_primal
        else:
            obj_primal += parametric_obj_primal

        zeta_arr = zeta_out.detach().cpu().numpy().flatten()
        zeta = {name: float(val) for name, val in zip(self.zeta_var_names, zeta_arr)}
        model, parametric_obj_primal, _ = self._add_parametric_balance_power_penalties_pg_block(
            model, x, sample_id, zeta, union_analysis
        )
        if obj_primal is None:
            obj_primal = parametric_obj_primal
        else:
            obj_primal += parametric_obj_primal
        
        # 设置目标函数（参考uc_dfsm_bcd.py，将罚项乘以权重加入目标函数）
        if obj_primal is not None:
            penalty_weight = self.penalty_factor  # 默认权重60
            obj_model = obj + penalty_weight * obj_primal
        else:
            obj_model = obj
        
        model.setObjective(obj_model, GRB.MINIMIZE)
        
        self._apply_fast_gurobi_tolerances(model, mip=False)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])
            
            if sample_id <= 2:
                print(f"solve_LP_with_theta_constraints, sample_id: {sample_id}, obj: {obj.getValue()}, penalty:{penalty_weight*obj_primal.getValue()}", flush=True)
                
            
            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}", flush=True)
            return None
    
    def save_theta_values(self, filepath: str, ensure_dir: bool = True) -> None:
        """
        将 self.theta_values 和 self.zeta_values 保存为 JSON 文件（参考uc_NN.py）
        """
        import json
        import os
        
        if not hasattr(self, 'theta_values') or self.theta_values is None:
            raise RuntimeError("theta_values 未初始化，无法保存。")
        
        if ensure_dir:
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        
        # 将可能的 numpy 类型转换为 Python 原生类型
        theta_serializable = {
            str(k): float(v)
            for k, v in self._direction_corrected_theta_values(self.theta_values).items()
        }
        zeta_serializable = (
            {
                str(k): float(v)
                for k, v in self._direction_corrected_zeta_values(self.zeta_values).items()
            }
            if hasattr(self, 'zeta_values') and self.zeta_values else {}
        )

        data = {
            'theta_values': theta_serializable,
            'zeta_values': zeta_serializable,
            'theta_constraint_direction_signs': self.theta_constraint_direction_signs.tolist(),
            'zeta_constraint_direction_signs': self.zeta_constraint_direction_signs.tolist(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ theta_values 和 zeta_values 已保存到: {filepath}", flush=True)

    def save_theta_zeta_values_list(self, filepath: str, ensure_dir: bool = True) -> None:
        """Save per-sample theta/zeta caches for post-training analysis."""
        import json
        import os

        if not hasattr(self, 'theta_values_list') or self.theta_values_list is None:
            raise RuntimeError("theta_values_list 未初始化，无法保存。")
        if not hasattr(self, 'zeta_values_list') or self.zeta_values_list is None:
            raise RuntimeError("zeta_values_list 未初始化，无法保存。")

        if ensure_dir:
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        sample_ids = []
        if hasattr(self, 'active_set_data') and self.active_set_data is not None:
            for i, sample in enumerate(self.active_set_data):
                if isinstance(sample, dict):
                    sample_ids.append(sample.get('sample_id', i))
                else:
                    sample_ids.append(i)

        data = {
            'sample_ids': sample_ids,
            'theta_var_names': list(getattr(self, 'theta_var_names', [])),
            'zeta_var_names': list(getattr(self, 'zeta_var_names', [])),
            'theta_values': {
                str(k): float(v)
                for k, v in self._direction_corrected_theta_values(self.theta_values or {}).items()
            },
            'zeta_values': {
                str(k): float(v)
                for k, v in self._direction_corrected_zeta_values(self.zeta_values or {}).items()
            },
            'theta_values_list': [
                {
                    str(k): float(v)
                    for k, v in self._direction_corrected_theta_values(values).items()
                }
                for values in self.theta_values_list
            ],
            'zeta_values_list': [
                {
                    str(k): float(v)
                    for k, v in self._direction_corrected_zeta_values(values).items()
                }
                for values in self.zeta_values_list
            ],
            'theta_constraint_direction_signs': self.theta_constraint_direction_signs.tolist(),
            'zeta_constraint_direction_signs': self.zeta_constraint_direction_signs.tolist(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ theta_values_list 和 zeta_values_list 已保存到: {filepath}", flush=True)
    
    def load_theta_values(self, filepath: str) -> dict:
        """
        从 JSON 文件加载 theta_values 和 zeta_values（参考uc_NN.py）
        """
        import json
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"theta 文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tv = data.get('theta_values', {})
        zv = data.get('zeta_values', {})
        
        # 保证值为 float
        self.theta_values = {str(k): float(v) for k, v in tv.items()}
        if zv:
            self.zeta_values = {str(k): float(v) for k, v in zv.items()}
        theta_signs = data.get('theta_constraint_direction_signs')
        zeta_signs = data.get('zeta_constraint_direction_signs')
        if theta_signs is not None:
            self.theta_constraint_direction_signs = np.asarray(theta_signs, dtype=float)
        if zeta_signs is not None:
            self.zeta_constraint_direction_signs = np.asarray(zeta_signs, dtype=float)
        
        print(f"✓ theta_values 和 zeta_values 已从文件加载: {filepath}，theta变量数量: {len(self.theta_values)}，zeta变量数量: {len(self.zeta_values) if hasattr(self, 'zeta_values') else 0}", flush=True)
        return self.theta_values
    
    def save_model_parameters(self, filepath: str, ensure_dir: bool = True) -> None:
        """
        保存神经网络模型参数（theta_net, zeta_net 和优化器状态）。
        
        Args:
            filepath: 保存文件路径（例如 'result/nn_params.pth'）
            ensure_dir: 如为 True，则自动创建目录
        """
        if not TORCH_AVAILABLE or self.theta_net is None or self.zeta_net is None:
            raise RuntimeError("神经网络未初始化，无法保存模型参数。")
        
        import os
        import torch
        
        if ensure_dir:
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        
        state = {
            "theta_net_state_dict": self.theta_net.state_dict(),
            "zeta_net_state_dict": self.zeta_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "theta_var_names": getattr(self, "theta_var_names", None),
            "zeta_var_names": getattr(self, "zeta_var_names", None),
            "nn_hidden_dims": getattr(self, "nn_hidden_dims", None),
            "device": str(self.device) if hasattr(self, "device") and self.device is not None else "cpu",
            "lambda_": self.lambda_,
            "mu": self.mu,
            "ita": self.ita,
            "theta_constraint_direction_signs": self.theta_constraint_direction_signs,
            "zeta_constraint_direction_signs": self.zeta_constraint_direction_signs,
            "dual_sign_relax_interval": self.dual_sign_relax_interval,
            "use_unit_predictor": self.use_unit_predictor,
            "unit_predictor_warmup_rounds": self.unit_predictor_warmup_rounds,
            "unit_predictor_finetune_lr": self.unit_predictor_finetune_lr,
            "unit_predictor_weight_decay": self.unit_predictor_weight_decay,
            "theta_constraint_delay_rounds": self.theta_constraint_delay_rounds,
            "rho_primal": self.rho_primal,
            "rho_binary": self.rho_binary,
            "rho_dual": self.rho_dual,
            "rho_dual_pg": self.rho_dual_pg,
            "rho_dual_x": self.rho_dual_x,
            "rho_dual_coc": self.rho_dual_coc,
            "rho_opt": self.rho_opt,
            "gamma_dual_component_scale": self.gamma_dual_component_scale,
            "loss_ratio_primal": self.loss_ratio_primal,
            "loss_ratio_dual_x": self.loss_ratio_dual_x,
            "loss_ratio_opt": self.loss_ratio_opt,
            "loss_ratio_reg": self.loss_ratio_reg,
            "nn_smooth_abs_eps": self.nn_smooth_abs_eps,
            "pg_block_prox_weight": self.pg_block_prox_weight,
            "dual_block_prox_weight": self.dual_block_prox_weight,
        }
        
        torch.save(state, filepath)
        print(f"✓ 模型参数已保存到: {filepath}", flush=True)
    
    @staticmethod
    def _rebuild_sequential_from_state_dict(sd: dict) -> 'nn.Sequential':
        """从 state_dict 的 weight shape 推断 Linear 层结构，重建 nn.Sequential。

        Args:
            sd: 网络的 state_dict，键形如 '0.weight', '0.bias', '1.weight' 等。

        Returns:
            按保存时结构重建的 nn.Sequential（含 LeakyReLU + Dropout 中间层）。
        """
        import torch.nn as nn_local

        # 收集所有 Linear 层的 (layer_idx, out_features, in_features)
        linear_layers: list[tuple[int, int, int]] = []
        for key, tensor in sd.items():
            if key.endswith('.weight') and tensor.dim() == 2:
                idx = int(key.split('.')[0])
                out_f, in_f = tensor.shape
                linear_layers.append((idx, out_f, in_f))
        linear_layers.sort(key=lambda x: x[0])

        # 重建与保存时相同的 Sequential 结构
        layers: list[nn_local.Module] = []
        for i, (idx, out_f, in_f) in enumerate(linear_layers):
            layers.append(nn_local.Linear(in_f, out_f))
            # 非最后一层：加 LeakyReLU + Dropout（与 _init_neural_network 一致）
            if i < len(linear_layers) - 1:
                layers.append(nn_local.LeakyReLU(0.01))
                layers.append(nn_local.Dropout(0.1))
        return nn_local.Sequential(*layers)

    def load_model_parameters(
        self,
        filepath: str,
        map_location: str = "cpu",
        restore_rho_state: bool = True,
    ) -> None:
        """从文件加载神经网络模型参数（theta_net, zeta_net 和优化器状态）。

        与 _init_neural_network 不同，此方法直接从 checkpoint 的 state_dict
        推断网络结构并重建，避免因当前数据维度不同导致的 size mismatch。

        Args:
            filepath: 保存的 .pth 文件路径
            map_location: 加载到的设备（例如 'cpu' 或 'cuda:0'）
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装，无法加载模型参数。")

        import os
        import torch

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型参数文件不存在: {filepath}")

        state = torch.load(filepath, map_location=map_location, weights_only=False)

        # 从 state_dict 重建网络结构（而非调用 _init_neural_network）
        self.theta_net = self._rebuild_sequential_from_state_dict(
            state["theta_net_state_dict"]
        )
        self.zeta_net = self._rebuild_sequential_from_state_dict(
            state["zeta_net_state_dict"]
        )

        self.theta_net.load_state_dict(state["theta_net_state_dict"])
        self.zeta_net.load_state_dict(state["zeta_net_state_dict"])

        # 设备设置
        device_str = state.get("device", "cpu")
        if torch.cuda.is_available() and "cuda" in device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cpu")
        self.theta_net = self.theta_net.to(self.device)
        self.zeta_net = self.zeta_net.to(self.device)
        self._refresh_feature_cache()

        # 重建优化器
        all_params = list(self.theta_net.parameters()) + list(self.zeta_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.nn_learning_rate, weight_decay=0.0)
        if state.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except (ValueError, KeyError):
                pass  # optimizer 结构变化时忽略

        # 同步变量名
        if state.get("theta_var_names") is not None:
            self.theta_var_names = state["theta_var_names"]
        if state.get("zeta_var_names") is not None:
            self.zeta_var_names = state["zeta_var_names"]
        self.nn_hidden_dims = normalize_nn_hidden_dims(
            state.get("nn_hidden_dims"),
            [
                module.out_features
                for module in self.theta_net
                if isinstance(module, nn.Linear)
            ][:-1],
        )
        self.nn_smooth_abs_eps = max(
            float(state.get("nn_smooth_abs_eps", self.nn_smooth_abs_eps)),
            0.0,
        )

        # 恢复对偶变量（向后兼容旧checkpoint）
        if "lambda_" in state:
            self.lambda_ = state["lambda_"]
        if "mu" in state:
            self.mu = state["mu"]
        if "ita" in state:
            self.ita = state["ita"]
        if "theta_constraint_direction_signs" in state:
            self.theta_constraint_direction_signs = np.asarray(state["theta_constraint_direction_signs"], dtype=float)
        if "zeta_constraint_direction_signs" in state:
            self.zeta_constraint_direction_signs = np.asarray(state["zeta_constraint_direction_signs"], dtype=float)
        self.dual_sign_relax_interval = max(int(state.get("dual_sign_relax_interval", self.dual_sign_relax_interval)), 0)
        self.unit_predictor_warmup_rounds = max(
            int(state.get("unit_predictor_warmup_rounds", self.unit_predictor_warmup_rounds)),
            0,
        )
        self.unit_predictor_finetune_lr = max(
            float(state.get("unit_predictor_finetune_lr", self.unit_predictor_finetune_lr)),
            1e-10,
        )
        self.unit_predictor_weight_decay = max(
            float(state.get("unit_predictor_weight_decay", self.unit_predictor_weight_decay)),
            0.0,
        )
        self.theta_constraint_delay_rounds = max(
            int(state.get("theta_constraint_delay_rounds", self.theta_constraint_delay_rounds)),
            0,
        )
        if restore_rho_state:
            if "rho_primal" in state:
                self.rho_primal = state["rho_primal"]
            self.rho_binary = state.get("rho_binary", self.rho_binary)
            self.rho_dual_pg = state.get("rho_dual_pg", state.get("rho_dual", self.rho_dual_pg))
            self.rho_dual_x = state.get("rho_dual_x", state.get("rho_dual", self.rho_dual_x))
            self.rho_dual_coc = state.get("rho_dual_coc", state.get("rho_dual", self.rho_dual_coc))
            self._sync_rho_dual_summary()
            if "rho_opt" in state:
                self.rho_opt = state["rho_opt"]
        self.gamma_dual_component_scale = state.get(
            "gamma_dual_component_scale",
            self.gamma_dual_component_scale,
        )
        self.loss_ratio_primal = float(state.get("loss_ratio_primal", self.loss_ratio_primal))
        self.loss_ratio_dual_x = float(state.get("loss_ratio_dual_x", self.loss_ratio_dual_x))
        self.loss_ratio_opt = float(state.get("loss_ratio_opt", self.loss_ratio_opt))
        self.loss_ratio_reg = float(state.get("loss_ratio_reg", self.loss_ratio_reg))
        self.pg_block_prox_weight = max(
            float(state.get("pg_block_prox_weight", self.pg_block_prox_weight)),
            0.0,
        )
        self.dual_block_prox_weight = max(
            float(state.get("dual_block_prox_weight", self.dual_block_prox_weight)),
            0.0,
        )

        self.refresh_theta_zeta_values_from_networks()

        print(f"✓ 模型参数已从文件加载: {filepath}", flush=True)

    def plot_sample0_binary_comparison(
        self,
        x_true: np.ndarray,
        x_round1: np.ndarray,
        x_round2: np.ndarray,
        labels: tuple = ("Ground Truth", "Standard Rounding", "BCD-Neural Optimized"),
        save_path: str = "result/figures/binary_comparison_pro.pdf",
        show: bool = True,
    ) -> None:
        # --- 1. 数据转换 ---
        ng, T = self.ng, self.T
        m_true = np.array(x_true).reshape(ng, T)
        m_r1 = np.array(x_round1).reshape(ng, T)
        m_r2 = np.array(x_round2).reshape(ng, T)

        diff1 = np.abs(m_true - m_r1)
        diff2 = np.abs(m_true - m_r2)
        
        # 统计匹配率 
        rate1 = (m_true == m_r1).mean() * 100
        rate2 = (m_true == m_r2).mean() * 100

        # --- 2. 颜色定义 ---
        # 主图颜色：深蓝色表示机组开启 (1)，浅灰色表示关闭 (0)
        on_color = "#2c3e50"
        off_color = "#ecf0f1"
        cmap_main = ListedColormap([off_color, on_color])

        # 误差图颜色：深红色表示错误 (1)，浅黄色表示正确 (0)
        err_color = "#b33939"
        corr_color = "#f7f1e3"
        cmap_diff = ListedColormap([corr_color, err_color])

        # --- 3. 布局设置 ---
        plt.rcParams['font.family'] = 'serif'
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), 
                                gridspec_kw={'width_ratios': [1, 1], 'hspace': 0.4, 'wspace': 0.12})
        
        data_list = [m_true, m_r1, m_r2]
        diff_list = [None, diff1, diff2]
        titles = [f"A. {labels[0]}", f"B. {labels[1]}", f"C. {labels[2]}"]

        # --- 4. 循环绘图 ---
        for i in range(3):
            # 左侧：状态图
            ax_m = axes[i, 0]
            sns.heatmap(data_list[i], ax=ax_m, cmap=cmap_main, cbar=False, 
                        linewidths=0.1, linecolor='white', xticklabels=5, yticklabels=5)
            ax_m.set_title(titles[i], loc='left', fontsize=12, fontweight='bold')
            ax_m.set_ylabel("Unit Index")
            
            # 右侧：误差图或图例说明
            ax_e = axes[i, 1]
            if i == 0:
                ax_e.axis('off')
                # 手动创建与图表颜色完全一致的图例说明
                leg_on = mpatches.Patch(color=on_color, label='Unit ON (1)')
                leg_off = mpatches.Patch(color=off_color, label='Unit OFF (0)')
                leg_err = mpatches.Patch(color=err_color, label='Mismatch (Error)')
                leg_corr = mpatches.Patch(color=corr_color, label='Correct Match')
                
                ax_e.legend(handles=[leg_on, leg_off, leg_corr, leg_err], 
                            loc='center', fontsize=10, title="Legend (Color Map)", 
                            title_fontsize=11, frameon=True, edgecolor='gray')
            else:
                sns.heatmap(diff_list[i], ax=ax_e, cmap=cmap_diff, cbar=False, 
                            linewidths=0.1, linecolor='white', xticklabels=5, yticklabels=5)
                # 在子图标题中标注正确率提升 
                acc_text = f"Accuracy: {rate1 if i==1 else rate2:.2f}%"
                ax_e.set_title(acc_text, loc='right', fontsize=10, 
                            color=err_color if i==1 else 'green', fontweight='bold')

        axes[2, 0].set_xlabel("Time Slots (h)")
        axes[2, 1].set_xlabel("Time Slots (h)")

        # --- 5. 保存 ---
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已修正图例颜色并保存: {save_path}")

        if show: plt.show()
        else: plt.close()
    
    # ========================== 测试代码段 ==========================
def _active_set_reader_load_all_samples_clean(self) -> List[Dict]:
    all_samples_data = []
    total_samples = self.get_total_samples_count()
    raw_samples = self.data.get('all_samples', [])
    has_dataset_renewable = any(
        'renewable_data' in sample and np.any(np.abs(np.asarray(sample['renewable_data'], dtype=float)) > 1e-9)
        for sample in raw_samples
    )

    print(f"[ActiveSet] Loading {total_samples} samples...", flush=True)

    for sample_id in range(total_samples):
        try:
            sample = self.get_sample_data(sample_id)
            if sample is None:
                raise ValueError(f"Sample {sample_id} does not exist")
            if not has_dataset_renewable:
                sample = dict(sample)
                sample.pop('renewable_data', None)
            sample = normalize_sample_arrays(dict(sample))

            active_constraints, active_variables, pd_data = self.extract_active_constraints_and_variables(sample_id)
            unit_commitment = self.get_unit_commitment_matrix(sample_id)

            sample_data = {
                'sample_id': sample_id,
                'active_constraints': active_constraints,
                'active_variables': active_variables,
                'pd_data': pd_data,
                'load_data': np.array(sample.get('load_data', pd_data), dtype=float),
                'unit_commitment_matrix': unit_commitment,
            }
            if has_dataset_renewable and 'renewable_data' in sample:
                sample_data['renewable_data'] = np.array(sample['renewable_data'], dtype=float)
            if sample and 'lambda' in sample:
                sample_data['lambda'] = sample['lambda']

            all_samples_data.append(sample_data)
            if (sample_id + 1) % 10 == 0 or sample_id == total_samples - 1:
                print(f"[ActiveSet] loaded_samples={sample_id + 1}/{total_samples}", flush=True)
        except Exception as e:
            print(f"[ActiveSet] sample {sample_id} failed to load: {e}", flush=True)
            all_samples_data.append({
                'sample_id': sample_id,
                'active_constraints': [],
                'active_variables': [],
                'pd_data': np.array([]),
                'load_data': np.array([]),
                'renewable_data': np.array([]),
                'unit_commitment_matrix': np.array([]),
                'error': str(e),
            })

    print("[ActiveSet] Finished loading all samples", flush=True)
    return all_samples_data


ActiveSetReader.load_all_samples = _active_set_reader_load_all_samples_clean


def load_active_set_from_json(json_filepath: str, sample_id: Optional[int] = None):
    reader = ActiveSetReader(json_filepath)

    if sample_id is not None:
        active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
        unit_commitment = reader.get_unit_commitment_matrix(sample_id)

        print(f"[ActiveSet] Loaded sample {sample_id}", flush=True)
        print(f"[ActiveSet] active_constraints={len(active_constraints)}", flush=True)
        print(f"[ActiveSet] active_variables={len(active_variables)}", flush=True)
        print(f"[ActiveSet] pd_shape={pd_data.shape}", flush=True)

        sample_data = {
            'sample_id': sample_id,
            'active_constraints': active_constraints,
            'active_variables': active_variables,
            'pd_data': pd_data,
            'unit_commitment_matrix': unit_commitment,
            'single_sample': True,
        }
        sample = reader.get_sample_data(sample_id)
        if sample and 'lambda' in sample:
            sample_data['lambda'] = sample['lambda']
        return sample_data

    all_samples_data = reader.load_all_samples()
    print("=== Loaded all active-set samples ===", flush=True)
    print(f"[ActiveSet] total_samples={len(all_samples_data)}", flush=True)
    return all_samples_data


if __name__ == "__main__":
    if not PYPOWER_AVAILABLE:
        print("错误: pypower未安装，无法运行测试代码", flush=True)
        exit(1)
    
    # 加载active_set数据
    json_file = "result/active_set/active_sets_20251221_161355.json"  # case39
    # json_file = "result/active_set/active_sets_20251221_182502.json"  # case14
    # json_file = "result/active_set/active_sets_case30_20251223_002959.json"
    
    active_set_data = load_active_set_from_json(json_file)
    
    # 只使用第一个样本进行测试（可以修改为使用更多样本）
    active_set_data = active_set_data
    
    # 准备ppc数据
    case = 'case30'
    ppc = pypower.case39.case39()
    ppc['branch'][:, pypower.idx_brch.RATE_A] = ppc['branch'][:, pypower.idx_brch.RATE_A]
    ppc['gencost'][:, -2] = np.array([[0.3, 0.28, 0.33, 0.35, 0.2, 0.34, 0.22, 0.28, 0.32, 0.36]])
    T_delta = 1
    
    # 创建模型对象
    print("=" * 60, flush=True)
    print("初始化 Agent_NN_BCD 模型", flush=True)
    print("=" * 60, flush=True)
    iter_bcd = Agent_NN_BCD(ppc, active_set_data=active_set_data, T_delta=T_delta)
    
    # 运行BCD迭代（结合神经网络更新theta和zeta）
    # print("\n" + "=" * 60, flush=True)
    # print("开始BCD迭代（结合神经网络更新）", flush=True)
    # print("=" * 60, flush=True)
    # iter_bcd.iter(max_iter=20)  # 运行20次BCD迭代
    
    # 可选：加载已保存的theta和zeta值
    # iter_bcd.load_theta_values('result/theta_zeta_values_final_20251222_161834.json')
    iter_bcd.load_model_parameters(f'result/net/model_parameters_final_20251222_193649_case39.pth')
    
    # 分析代理模型性能
    print("\n" + "=" * 60, flush=True)
    print("分析代理模型性能", flush=True)
    print("=" * 60, flush=True)
    iter_bcd.analyse_surrogate_model_totle()
    
    # 保存theta和zeta值
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'result/theta_zeta/theta_zeta_values_final_{timestamp}.json'
    iter_bcd.save_theta_values(save_path)
    
    save_path = f'result/net/model_parameters_final_{timestamp}_{case}.pth'
    iter_bcd.save_model_parameters(save_path)
    
    print("\n" + "=" * 60, flush=True)
    print("测试完成！", flush=True)
    print("=" * 60, flush=True)
