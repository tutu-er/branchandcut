import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import io
import time
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import os
import matplotlib.pyplot as plt

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将无法使用神经网络功能", flush=True)

# 导入必要的工具函数
from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A
try:
    from scenario_utils import (
        get_feature_vector,
        get_feature_vector_from_sample,
        get_sample_load_data,
        get_sample_net_load,
        get_sample_renewable_data,
        normalize_sample_arrays,
    )
    from case_registry import get_case_ppc
except ImportError:
    from src.scenario_utils import (
        get_feature_vector,
        get_feature_vector_from_sample,
        get_sample_load_data,
        get_sample_net_load,
        get_sample_renewable_data,
        normalize_sample_arrays,
    )
    from src.case_registry import get_case_ppc

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

# 设置输出缓冲（用 reconfigure 原地修改，避免替换 stdout 导致 buffer 被 GC 关闭）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def _load_demo_ppc(case_name: str):
    try:
        return get_case_ppc(case_name)
    except ValueError as exc:
        raise ValueError(f"Unknown case_name: {case_name}") from exc


SUPPORTED_NN_BATCH_STRATEGIES = {"full-batch", "mini-batch"}


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


# ========================== 数据加载工具 ==========================

class ActiveSetReader:
    """读取和解析活动集JSON文件的工具类"""
    
    def __init__(self, json_filepath: str):
        self.json_filepath = Path(json_filepath)
        self.data = self._load_json()
        
    def _load_json(self) -> Dict:
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON文件未找到: {self.json_filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"JSON文件格式错误: {self.json_filepath}")
    
    def get_sample_data(self, sample_id: int) -> Optional[Dict]:
        samples = self.data.get('all_samples', [])
        if 0 <= sample_id < len(samples):
            return samples[sample_id]
        return None
    
    def get_total_samples_count(self) -> int:
        return len(self.data.get('all_samples', []))
    
    def load_all_samples(self) -> List[Dict]:
        all_samples_data = []
        total_samples = self.get_total_samples_count()
        raw_samples = self.data.get('all_samples', [])
        has_dataset_renewable = any(
            'renewable_data' in sample and np.any(np.abs(np.asarray(sample['renewable_data'], dtype=float)) > 1e-9)
            for sample in raw_samples
        )

        for sample_id in range(total_samples):
            try:
                sample = self.get_sample_data(sample_id) or {}
                if not has_dataset_renewable:
                    sample = dict(sample)
                    sample.pop('renewable_data', None)
                active_constraints, active_variables, pd_data = self.extract_active_constraints_and_variables(sample_id)
                unit_commitment = self.get_unit_commitment_matrix(sample_id)
                
                sample_data = {
                    'sample_id': sample_id,
                    'active_constraints': active_constraints,
                    'active_variables': active_variables,
                    'pd_data': pd_data,
                    'unit_commitment_matrix': unit_commitment
                }

                if 'load_data' in sample:
                    sample_data['load_data'] = np.array(sample['load_data'], dtype=float)
                if has_dataset_renewable and 'renewable_data' in sample:
                    sample_data['renewable_data'] = np.array(sample['renewable_data'], dtype=float)
                
                if sample and 'lambda' in sample:
                    sample_data['lambda'] = sample['lambda']
                
                all_samples_data.append(sample_data)
                    
            except Exception as e:
                print(f"加载样本 {sample_id} 时出错: {e}", flush=True)
                all_samples_data.append({
                    'sample_id': sample_id,
                    'pd_data': np.array([]),
                    'load_data': np.array([]),
                    'renewable_data': np.array([]),
                    'error': str(e)
                })
        
        return all_samples_data
    
    def extract_active_constraints_and_variables(self, sample_id: int) -> Tuple[List, List, np.ndarray]:
        sample = self.get_sample_data(sample_id)
        if sample is None:
            return [], [], np.array([])
        
        sample = normalize_sample_arrays(dict(sample))
        active_set = sample.get('active_set', [])
        pd_data = sample.get('pd_data', np.array([]))
        
        active_constraints = []
        active_variables = []
        
        for item in active_set:
            if isinstance(item, list) and len(item) == 2:
                if isinstance(item[0], list) and len(item[0]) == 2:
                    active_variables.append({
                        'type': 'binary_variable',
                        'unit_id': item[0][0],
                        'time_slot': item[0][1],
                        'value': item[1]
                    })
                else:
                    active_constraints.append({
                        'constraint_id': item[0],
                        'dual_value': item[1]
                    })
        
        return active_constraints, active_variables, pd_data
    
    def get_unit_commitment_matrix(self, sample_id: int) -> np.ndarray:
        _, active_variables, _ = self.extract_active_constraints_and_variables(sample_id)
        
        if not active_variables:
            return np.array([])
        
        binary_vars = [v for v in active_variables if v.get('type') == 'binary_variable']
        if not binary_vars:
            return np.array([])
            
        max_unit = max([v['unit_id'] for v in binary_vars]) + 1
        max_time = max([v['time_slot'] for v in binary_vars]) + 1
        
        unit_commitment = np.zeros((max_unit, max_time), dtype=int)
        for var in binary_vars:
            unit_commitment[var['unit_id'], var['time_slot']] = var['value']
        
        return unit_commitment


def _extract_lambda_power_balance(lambda_field, T: int) -> np.ndarray:
    """从样本的 lambda 字段中提取功率平衡对偶变量 (T,) 数组。
    支持三种格式：
      - list/ndarray: 直接使用
      - dict with 'lambda_power_balance' key: 提取该字段
      - dict without that key: 取第一个 list 值
    """
    if isinstance(lambda_field, dict):
        if 'lambda_power_balance' in lambda_field:
            arr = np.array(lambda_field['lambda_power_balance'], dtype=float)
        else:
            # 取第一个 list 类型的值
            for v in lambda_field.values():
                if isinstance(v, list):
                    arr = np.array(v, dtype=float)
                    break
            else:
                return np.zeros(T)
    else:
        arr = np.array(lambda_field, dtype=float)
    # 确保长度匹配 T
    if arr.ndim != 1 or len(arr) != T:
        return np.zeros(T)
    return arr


def _extract_lambda_matrix(
    lambda_field,
    key: str,
    shape: Tuple[int, int],
) -> np.ndarray:
    if not isinstance(lambda_field, dict) or key not in lambda_field:
        return np.zeros(shape, dtype=float)
    arr = np.array(lambda_field[key], dtype=float)
    if arr.shape != shape:
        return np.zeros(shape, dtype=float)
    return arr


def _pack_global_dual_targets(
    lambda_power_balance: np.ndarray,
    lambda_dcpf_upper: np.ndarray,
    lambda_dcpf_lower: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(lambda_power_balance, dtype=float).reshape(-1),
            np.asarray(lambda_dcpf_upper, dtype=float).reshape(-1),
            np.asarray(lambda_dcpf_lower, dtype=float).reshape(-1),
        ]
    )


def _unpack_global_dual_targets(
    packed_targets: np.ndarray,
    T: int,
    nl: int,
    generator_injection_sensitivity: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    packed = np.asarray(packed_targets, dtype=float).reshape(-1)
    full_size = T + 2 * nl * T

    if packed.size == T:
        lambda_power_balance = packed.copy()
        lambda_dcpf_upper = np.zeros((nl, T), dtype=float)
        lambda_dcpf_lower = np.zeros((nl, T), dtype=float)
    elif packed.size == full_size:
        offset = 0
        lambda_power_balance = packed[offset:offset + T]
        offset += T
        lambda_dcpf_upper = packed[offset:offset + nl * T].reshape(nl, T)
        offset += nl * T
        lambda_dcpf_lower = packed[offset:offset + nl * T].reshape(nl, T)
    else:
        raise ValueError(
            f"Packed dual target has size {packed.size}, expected {T} or {full_size}"
        )

    payload = {
        'lambda_power_balance': lambda_power_balance,
        'lambda_dcpf_upper': lambda_dcpf_upper,
        'lambda_dcpf_lower': lambda_dcpf_lower,
    }
    if generator_injection_sensitivity is not None:
        payload['lambda_pg_effective'] = _combine_pg_duals(
            lambda_power_balance,
            lambda_dcpf_upper,
            lambda_dcpf_lower,
            generator_injection_sensitivity,
        )
    return payload


def _build_generator_injection_sensitivity(ppc) -> np.ndarray:
    ppc_int = ext2int(ppc)
    bus = ppc_int['bus']
    gen = ppc_int['gen']
    branch = ppc_int['branch']
    nb = bus.shape[0]
    ng = gen.shape[0]
    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0
    PTDF = makePTDF(ppc_int['baseMVA'], bus, branch)
    return PTDF @ G


def _get_custom_generator_array_from_ppc(ppc_raw, ng: int, key: str) -> Optional[np.ndarray]:
    """Read optional per-generator metadata from raw ppc and align it to ext2int order."""
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
    """Return per-generator ramp limits, preferring optional raw ppc metadata when present."""
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


def _combine_pg_duals(
    lambda_power_balance: np.ndarray,
    lambda_dcpf_upper: np.ndarray,
    lambda_dcpf_lower: np.ndarray,
    generator_injection_sensitivity: np.ndarray,
) -> np.ndarray:
    T = len(lambda_power_balance)
    ng = generator_injection_sensitivity.shape[1]
    combined = np.zeros((ng, T), dtype=float)
    congestion = lambda_dcpf_upper - lambda_dcpf_lower
    for t in range(T):
        combined[:, t] = (
            lambda_power_balance[t]
            - generator_injection_sensitivity.T @ congestion[:, t]
        )
    return combined


def _extract_effective_pg_dual(
    lambda_field,
    T: int,
    ng: int,
    nl: int,
    generator_injection_sensitivity: np.ndarray,
) -> np.ndarray:
    if isinstance(lambda_field, dict):
        direct = lambda_field.get('lambda_pg_electricity_price')
        if direct is None:
            direct = lambda_field.get('lambda_pg_effective')
        if direct is not None:
            arr = np.array(direct, dtype=float)
            if arr.shape == (ng, T):
                return arr
        lambda_pb = _extract_lambda_power_balance(lambda_field, T)
        lambda_dcpf_upper = _extract_lambda_matrix(
            lambda_field, 'lambda_dcpf_upper', (nl, T))
        lambda_dcpf_lower = _extract_lambda_matrix(
            lambda_field, 'lambda_dcpf_lower', (nl, T))
        return _combine_pg_duals(
            lambda_pb,
            lambda_dcpf_upper,
            lambda_dcpf_lower,
            generator_injection_sensitivity,
        )

    lambda_pb = _extract_lambda_power_balance(lambda_field, T)
    return np.tile(lambda_pb, (ng, 1))


def _has_complete_effective_pg_dual(
    lambda_field,
    T: int,
    ng: int,
    nl: int,
) -> bool:
    if not isinstance(lambda_field, dict):
        return False
    direct = lambda_field.get('lambda_pg_electricity_price')
    if direct is None:
        direct = lambda_field.get('lambda_pg_effective')
    if direct is not None:
        arr = np.array(direct, dtype=float)
        if arr.shape == (ng, T):
            return True
    lambda_pb = np.array(lambda_field.get('lambda_power_balance', []), dtype=float)
    lambda_du = np.array(lambda_field.get('lambda_dcpf_upper', []), dtype=float)
    lambda_dl = np.array(lambda_field.get('lambda_dcpf_lower', []), dtype=float)
    return (
        lambda_pb.shape == (T,)
        and lambda_du.shape == (nl, T)
        and lambda_dl.shape == (nl, T)
    )


def _has_global_dual_payload(lambda_field, T: int, nl: int) -> bool:
    if not isinstance(lambda_field, dict):
        return False
    lambda_pb = np.array(lambda_field.get('lambda_power_balance', []), dtype=float)
    lambda_du = np.array(lambda_field.get('lambda_dcpf_upper', []), dtype=float)
    lambda_dl = np.array(lambda_field.get('lambda_dcpf_lower', []), dtype=float)
    return (
        lambda_pb.shape == (T,)
        and lambda_du.shape == (nl, T)
        and lambda_dl.shape == (nl, T)
    )


def _recover_unit_commitment_matrix(sample: Dict, ng: int, T: int) -> np.ndarray:
    x_sol = np.zeros((ng, T), dtype=float)
    if 'active_set' in sample:
        active_set = sample['active_set']
        for item in active_set:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                value = item[1]
                if 0 <= g < ng and 0 <= t < T:
                    x_sol[g, t] = value
        return x_sol
    if 'unit_commitment_matrix' in sample:
        uc = np.asarray(sample['unit_commitment_matrix'], dtype=float)
        rows = min(uc.shape[0], ng)
        cols = min(uc.shape[1], T)
        x_sol[:rows, :cols] = uc[:rows, :cols]
    return x_sol


def _solve_global_dual_payload_from_ed(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    generator_injection_sensitivity: np.ndarray,
    renewable_data: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from src.ed_gurobipy import EconomicDispatchGurobi
    except ImportError:
        from ed_gurobipy import EconomicDispatchGurobi

    ed = EconomicDispatchGurobi(
        ppc,
        Pd,
        T_delta,
        x_sol,
        renewable_data=renewable_data,
        verbose=False,
    )
    pg_sol, _ = ed.solve()
    if pg_sol is None:
        raise RuntimeError(f"ED solve failed with status={ed.model.status}")

    T = Pd.shape[1]
    nl = ed.branch.shape[0]
    lambda_pb = np.zeros(T, dtype=float)
    lambda_dcpf_upper = np.zeros((nl, T), dtype=float)
    lambda_dcpf_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        constr = ed.model.getConstrByName(f'power_balance_{t}')
        # For the equality balance constraint, Gurobi's Pi already matches the
        # economic marginal price convention used by the subproblem:
        #   min cost - lambda^T pg, with stationarity a - lambda + ... = 0.
        lambda_pb[t] = float(constr.Pi) if constr is not None else 0.0
        for l in range(nl):
            cu = ed.model.getConstrByName(f'flow_upper_{l}_{t}')
            cl = ed.model.getConstrByName(f'flow_lower_{l}_{t}')
            # Recover canonical nonnegative multipliers for:
            #   flow_upper:  flow - limit <= 0
            #   flow_lower: -flow - limit <= 0
            # In the ED model these are encoded as:
            #   flow <= limit      -> Pi is nonpositive when binding
            #   flow >= -limit     -> Pi is nonnegative when binding
            # So the correct mapping is upper = -Pi, lower = +Pi.
            lambda_dcpf_upper[l, t] = max(
                0.0,
                -(float(cu.Pi) if cu is not None else 0.0),
            )
            lambda_dcpf_lower[l, t] = max(
                0.0,
                float(cl.Pi) if cl is not None else 0.0,
            )

    return {
        'lambda_power_balance': lambda_pb,
        'lambda_dcpf_upper': lambda_dcpf_upper,
        'lambda_dcpf_lower': lambda_dcpf_lower,
        'lambda_pg_effective': _combine_pg_duals(
            lambda_pb,
            lambda_dcpf_upper,
            lambda_dcpf_lower,
            generator_injection_sensitivity,
        ),
    }


def _solve_effective_pg_dual_from_ed(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    generator_injection_sensitivity: np.ndarray,
    renewable_data: np.ndarray | None = None,
) -> np.ndarray:
    return _solve_global_dual_payload_from_ed(
        ppc,
        Pd,
        T_delta,
        x_sol,
        generator_injection_sensitivity,
        renewable_data=renewable_data,
    )['lambda_pg_effective']


def _extract_pg_electricity_price_matrix(source, T: int, ng: int) -> Optional[np.ndarray]:
    """Extract the full per-unit electricity-price matrix when available."""
    if source is None:
        return None
    if isinstance(source, dict):
        direct = source.get('lambda_pg_electricity_price')
        if direct is None:
            direct = source.get('lambda_pg_effective')
        if direct is None:
            return None
        arr = np.asarray(direct, dtype=float)
    else:
        arr = np.asarray(source, dtype=float)
    if arr.shape == (ng, T):
        return arr
    if arr.shape == (T, ng):
        return arr.T
    return None


def _get_sample_pg_electricity_price_matrix(sample: Dict, T: int, ng: int) -> Optional[np.ndarray]:
    """Read cached electricity prices from a sample when they exist."""
    direct = _extract_pg_electricity_price_matrix(
        sample.get('lambda_pg_electricity_price'),
        T,
        ng,
    )
    if direct is not None:
        return direct
    return _extract_pg_electricity_price_matrix(sample.get('lambda'), T, ng)


def _solve_pg_electricity_price_from_ed(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    renewable_data: np.ndarray | None = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Solve ED with fixed commitment and recover the full pg stationarity price.

    The returned electricity price includes every dual term that enters the pg
    KKT condition in the fixed-x ED:
      - power balance
      - pg lower / upper bounds
      - ramp up / down
      - DC flow upper / lower
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    bus = ppc_int['bus']
    branch = ppc_int['branch']
    gencost = ppc_int['gencost']

    ng = gen.shape[0]
    nb = bus.shape[0]
    nl = branch.shape[0]
    T = Pd.shape[1]

    load_data = np.asarray(Pd, dtype=float)
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    if renewable_arr is not None and renewable_arr.shape != load_data.shape:
        raise ValueError(
            f"renewable_data shape {renewable_arr.shape} does not match load shape {load_data.shape}"
        )

    renewable_bus_ids = (
        np.where(np.any(renewable_arr > 1e-9, axis=1))[0]
        if renewable_arr is not None else np.array([], dtype=int)
    )
    nr = len(renewable_bus_ids)
    R = np.zeros((nb, nr), dtype=float)
    for r, bus_idx in enumerate(renewable_bus_ids):
        R[bus_idx, r] = 1.0

    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0

    PTDF = makePTDF(ppc_int['baseMVA'], bus, branch)
    PTDF_G = PTDF @ G
    branch_limit = branch[:, RATE_A]
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)

    model = gp.Model("subproblem_ed_price")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0

    pg = model.addVars(ng, T, lb=0.0, name='pg')
    cpower = model.addVars(ng, T, lb=0.0, name='cpower')
    p_ren = model.addVars(nr, T, lb=0.0, name='p_ren') if nr > 0 else None

    for t in range(T):
        renewable_supply = gp.quicksum(p_ren[r, t] for r in range(nr)) if nr > 0 else 0.0
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + renewable_supply == float(np.sum(load_data[:, t])),
            name=f'power_balance_{t}',
        )
        for g in range(ng):
            model.addConstr(
                pg[g, t] >= float(gen[g, PMIN] * x_sol[g, t]),
                name=f'pg_lower_{g}_{t}',
            )
            model.addConstr(
                pg[g, t] <= float(gen[g, PMAX] * x_sol[g, t]),
                name=f'pg_upper_{g}_{t}',
            )
            model.addConstr(
                cpower[g, t] >= float(gencost[g, -2] / T_delta) * pg[g, t]
                + float(gencost[g, -1] / T_delta * x_sol[g, t]),
                name=f'cpower_{g}_{t}',
            )
        for r, bus_idx in enumerate(renewable_bus_ids):
            model.addConstr(
                p_ren[r, t] <= float(renewable_arr[bus_idx, t]),
                name=f'ren_upper_{r}_{t}',
            )

    for t in range(1, T):
        for g in range(ng):
            model.addConstr(
                pg[g, t] - pg[g, t - 1]
                <= float(Ru[g] * x_sol[g, t - 1] + Ru_co[g] * (1 - x_sol[g, t - 1])),
                name=f'ramp_up_{g}_{t}',
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t]
                <= float(Rd[g] * x_sol[g, t] + Rd_co[g] * (1 - x_sol[g, t])),
                name=f'ramp_down_{g}_{t}',
            )

    for t in range(T):
        thermal_injection = G @ np.array([pg[g, t] for g in range(ng)], dtype=object)
        renewable_injection = (
            R @ np.array([p_ren[r, t] for r in range(nr)], dtype=object)
            if nr > 0 else 0.0
        )
        flow = PTDF @ (thermal_injection + renewable_injection - load_data[:, t])
        for l in range(nl):
            model.addConstr(flow[l] <= float(branch_limit[l]), name=f'flow_upper_{l}_{t}')
            model.addConstr(flow[l] >= -float(branch_limit[l]), name=f'flow_lower_{l}_{t}')

    model.setObjective(
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"ED solve for electricity price failed with status={model.status}")

    lambda_power_balance = np.zeros(T, dtype=float)
    lambda_pg_lower = np.zeros((ng, T), dtype=float)
    lambda_pg_upper = np.zeros((ng, T), dtype=float)
    lambda_ramp_up = np.zeros((ng, T - 1), dtype=float)
    lambda_ramp_down = np.zeros((ng, T - 1), dtype=float)
    lambda_flow_upper = np.zeros((nl, T), dtype=float)
    lambda_flow_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        lambda_power_balance[t] = float(model.getConstrByName(f'power_balance_{t}').Pi)
        for g in range(ng):
            lambda_pg_lower[g, t] = float(model.getConstrByName(f'pg_lower_{g}_{t}').Pi)
            lambda_pg_upper[g, t] = float(model.getConstrByName(f'pg_upper_{g}_{t}').Pi)
        for l in range(nl):
            lambda_flow_upper[l, t] = float(model.getConstrByName(f'flow_upper_{l}_{t}').Pi)
            lambda_flow_lower[l, t] = float(model.getConstrByName(f'flow_lower_{l}_{t}').Pi)

    for t in range(1, T):
        for g in range(ng):
            lambda_ramp_up[g, t - 1] = float(model.getConstrByName(f'ramp_up_{g}_{t}').Pi)
            lambda_ramp_down[g, t - 1] = float(model.getConstrByName(f'ramp_down_{g}_{t}').Pi)

    effective = np.zeros((ng, T), dtype=float)
    lambda_ramp_contrib = np.zeros((ng, T), dtype=float)
    lambda_flow_contrib = np.zeros((ng, T), dtype=float)
    for g in range(ng):
        for t in range(T):
            ramp_contrib = 0.0
            if t > 0:
                ramp_contrib += lambda_ramp_up[g, t - 1]
                ramp_contrib -= lambda_ramp_down[g, t - 1]
            if t < T - 1:
                ramp_contrib -= lambda_ramp_up[g, t]
                ramp_contrib += lambda_ramp_down[g, t]

            flow_contrib = float(
                np.dot(PTDF_G[:, g], lambda_flow_upper[:, t] + lambda_flow_lower[:, t])
            )
            effective[g, t] = (
                lambda_power_balance[t]
                + lambda_pg_lower[g, t]
                + lambda_pg_upper[g, t]
                + ramp_contrib
                + flow_contrib
            )
            lambda_ramp_contrib[g, t] = ramp_contrib
            lambda_flow_contrib[g, t] = flow_contrib

    return {
        'lambda_pg_electricity_price': effective,
        'lambda_power_balance': lambda_power_balance,
        'lambda_pg_lower': lambda_pg_lower,
        'lambda_pg_upper': lambda_pg_upper,
        'lambda_ramp_up': lambda_ramp_up,
        'lambda_ramp_down': lambda_ramp_down,
        'lambda_flow_upper': lambda_flow_upper,
        'lambda_flow_lower': lambda_flow_lower,
        'lambda_ramp_contrib': lambda_ramp_contrib,
        'lambda_flow_contrib': lambda_flow_contrib,
    }


def load_active_set_from_json(json_filepath: str, sample_id: Optional[int] = None):
    """从JSON文件加载活动集数据"""
    reader = ActiveSetReader(json_filepath)
    
    if sample_id is not None:
        active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
        unit_commitment = reader.get_unit_commitment_matrix(sample_id)
        return {
            'sample_id': sample_id,
            'active_constraints': active_constraints,
            'active_variables': active_variables,
            'pd_data': pd_data,
            'unit_commitment_matrix': unit_commitment
        }
    else:
        return reader.load_all_samples()


def generate_test_data(ppc, T: int = 8, n_samples: int = 10, seed: int = 42) -> List[Dict]:
    """
    生成测试用的活动集数据

    Args:
        ppc: PyPower案例数据
        T: 时段数
        n_samples: 样本数量
        seed: 随机种子

    Returns:
        活动集数据列表
    """
    ppc_int = ext2int(ppc)
    nb = ppc_int['bus'].shape[0]
    ng = ppc_int['gen'].shape[0]

    active_set_data = []

    for sample_id in range(n_samples):
        np.random.seed(seed + sample_id)

        # 生成随机负荷数据（带有日变化曲线）
        base_load = np.random.uniform(50, 150, nb)
        time_factor = 1 + 0.3 * np.sin(np.linspace(0, 2*np.pi, T)) + 0.1 * np.random.randn(T)
        pd_data = np.outer(base_load, time_factor)
        pd_data = np.maximum(pd_data, 10)  # 确保负荷为正

        # 生成随机的机组启停状态（满足部分约束）
        unit_commitment = np.zeros((ng, T), dtype=int)
        for g in range(ng):
            on_probability = 0.6 + 0.3 * np.random.rand()
            for t in range(T):
                if np.random.rand() < on_probability:
                    unit_commitment[g, t] = 1

        sample = {
            'sample_id': sample_id,
            'pd_data': pd_data,
            'unit_commitment_matrix': unit_commitment,
            'active_constraints': [],
            'active_variables': []
        }
        active_set_data.append(sample)

    print(f"✓ 生成了 {n_samples} 个测试样本 (T={T}, nb={nb}, ng={ng})", flush=True)
    return active_set_data


# ========================== 第一部分：对偶变量预测网络 ==========================

class DualVariablePredictorNet(nn.Module):
    """
    从Pd数据预测对偶变量的神经网络
    
    输入: Pd数据展平 (nb * T,)
    输出: 功率平衡约束的对偶变量 λ (T,)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(DualVariablePredictorNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class DualVariablePredictorTrainer:
    """
    对偶变量预测网络的独立训练器
    
    功能：训练神经网络从Pd数据预测功率平衡约束的对偶变量λ
    训练方式：监督学习（MSE损失）
    """
    
    def __init__(self, ppc, active_set_data, T_delta, device=None):
        self.ppc = ppc
        self.ppc_raw = ppc
        ppc_int = ext2int(ppc)
        self.baseMVA = ppc_int['baseMVA']
        self.bus = ppc_int['bus']
        self.gen = ppc_int['gen']
        self.branch = ppc_int['branch']
        self.gencost = ppc_int['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
            
        self.ng = self.gen.shape[0]
        self.nb = self.bus.shape[0]
        self.nl = self.branch.shape[0]
        self.active_set_data = active_set_data
        self.generator_injection_sensitivity = _build_generator_injection_sensitivity(self.ppc)
        self.generator_injection_sensitivity = _build_generator_injection_sensitivity(self.ppc)
        self.generator_injection_sensitivity = _build_generator_injection_sensitivity(ppc)
        
        # 设置设备
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # 输入输出维度
        first_sample = active_set_data[0] if isinstance(active_set_data, list) else active_set_data
        self.input_dim = len(get_feature_vector_from_sample(dict(first_sample)))
        self.output_dim = self.T
        
        # 初始化网络
        if TORCH_AVAILABLE:
            self.networks = [
                DualVariablePredictorNet(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim
                ).to(self.device)
                for _ in range(self.ng)
            ]
            self.optimizers = [
                optim.Adam(network.parameters(), lr=1e-3)
                for network in self.networks
            ]
        
        # 求解原始问题获取对偶变量真值
        self.lambda_true = self._solve_for_true_dual_variables()
        
        print(f"✓ 对偶变量预测训练器初始化完成", flush=True)
        print(f"  - 输入维度: {self.input_dim}, 输出维度: {self.output_dim}", flush=True)
    
    def _solve_for_true_dual_variables(self) -> np.ndarray:
        """获取功率平衡约束的对偶变量真值λ
        
        优化策略：
        1. 如果JSON有lambda → 直接读取（0次求解）
        2. 如果JSON没有lambda → 从active_set提取x，求解ED（LP）获取（不需要MILP）
        """
        lambda_true = []
        needs_solve = []
        
        # 检查JSON中是否已有lambda
        for sample_id in range(self.n_samples):
            if 'lambda' in self.active_set_data[sample_id] and \
               self.active_set_data[sample_id]['lambda'] is not None:
                lam = _extract_lambda_power_balance(
                    self.active_set_data[sample_id]['lambda'], self.T)
                if np.any(lam != 0) or True:  # 即使全零也接受，避免重复求解
                    lambda_true.append(lam)
                else:
                    needs_solve.append(sample_id)
            else:
                needs_solve.append(sample_id)
        
        # 如果所有样本都有lambda，直接返回
        if not needs_solve:
            print(f"✓ 从数据中读取了 {len(lambda_true)} 个样本的 lambda 真值", flush=True)
            return np.array(lambda_true)
        
        # 否则通过求解LP获取缺失的lambda
        print(f"⚠ {len(needs_solve)} 个样本缺少 lambda，从 active_set 提取 x 并求解 LP...", flush=True)
        
        # 构建完整的lambda_true字典（按sample_id索引）
        lambda_dict = {}
        already_loaded = [i for i in range(self.n_samples) if i not in needs_solve]
        for idx, sample_id in enumerate(already_loaded):
            lambda_dict[sample_id] = lambda_true[idx]

        for sample_id in needs_solve:
            Pd = self.active_set_data[sample_id]['pd_data']

            # 恢复x矩阵：优先用 active_set，否则用 unit_commitment_matrix
            x_sol = np.zeros((self.ng, self.T))
            if 'active_set' in self.active_set_data[sample_id]:
                active_set = self.active_set_data[sample_id]['active_set']
                for item in active_set:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                        g, t = item[0]
                        value = item[1]
                        x_sol[g, t] = value
            elif 'unit_commitment_matrix' in self.active_set_data[sample_id]:
                uc = self.active_set_data[sample_id]['unit_commitment_matrix']
                x_sol[:uc.shape[0], :] = uc

            # 用x求解ED（LP），获取对偶变量
            from ed_cvxpy import EconomicDispatchCVXPY
            ed = EconomicDispatchCVXPY(self.ppc, Pd, self.T_delta, x_sol)
            pg_sol, total_cost = ed.solve()

            # 提取功率平衡约束的对偶变量λ（前T个约束）
            lambda_sample = np.zeros(self.T)
            for t in range(self.T):
                dual_val = ed.constraints[t].dual_value
                if dual_val is None:
                    lambda_sample[t] = 0.0
                else:
                    lambda_sample[t] = float(dual_val)

            lambda_dict[sample_id] = lambda_sample

        print(f"✓ 成功获取所有 {self.n_samples} 个样本的 lambda（{len(already_loaded)} 个从JSON读取，{len(needs_solve)} 个求解LP获得）", flush=True)

        # 按sample_id顺序返回
        return np.array([lambda_dict[i] for i in range(self.n_samples)])
    
    def _extract_features(self, sample_id: int) -> np.ndarray:
        """提取Pd数据作为特征"""
        return get_feature_vector_from_sample(dict(self.active_set_data[sample_id]))
    
    def train(self, num_epochs: int = 100, batch_size: int = 8):
        """训练对偶变量预测网络"""
        if not TORCH_AVAILABLE:
            print("警告: PyTorch不可用", flush=True)
            return
        
        print(f"开始训练对偶变量预测网络 (epochs={num_epochs})...", flush=True)
        
        # 准备数据
        X = np.array([self._extract_features(i) for i in range(self.n_samples)])
        Y = self.lambda_true
        
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5)
        self.network.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                self.optimizer.zero_grad()
                pred = self.network(batch_X)
                loss = criterion(pred, batch_Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataset)
            scheduler.step(epoch_loss)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}", flush=True)
        
        print(f"✓ 对偶变量预测网络训练完成", flush=True)
    
    def predict(self, pd_data: np.ndarray | dict, renewable_data: np.ndarray | None = None) -> np.ndarray:
        """预测对偶变量"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")
        
        self.network.eval()
        if isinstance(pd_data, dict):
            features = get_feature_vector_from_sample(dict(pd_data))
        else:
            features = get_feature_vector(pd_data, renewable_data=renewable_data)
        pd_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            lambda_pred = self.network(pd_tensor.unsqueeze(0)).squeeze(0)
        
        return lambda_pred.cpu().numpy()
    
    def save(self, filepath: str):
        """保存模型"""
        if TORCH_AVAILABLE:
            # #region agent log
            import json as _json_debug; _log_path = r'd:\0-python_workspace\branchandcut\.cursor\debug.log'; _log_data = {"location": "uc_NN_subproblem.py:save:427", "message": "DualVariablePredictorTrainer.save called", "data": {"filepath": filepath, "cwd": os.getcwd(), "abs_filepath": os.path.abspath(filepath)}, "timestamp": int(__import__('time').time()*1000), "sessionId": "debug-session", "hypothesisId": "A"}; open(_log_path, 'a', encoding='utf-8').write(_json_debug.dumps(_log_data) + '\n')
            # #endregion
            dirpath = os.path.dirname(os.path.abspath(filepath))
            # #region agent log
            _log_data2 = {"location": "uc_NN_subproblem.py:save:430", "message": "Checking dirpath", "data": {"dirpath": dirpath, "exists": os.path.exists(dirpath)}, "timestamp": int(__import__('time').time()*1000), "sessionId": "debug-session", "hypothesisId": "B"}; open(_log_path, 'a', encoding='utf-8').write(_json_debug.dumps(_log_data2) + '\n')
            # #endregion
            if dirpath and not os.path.exists(dirpath):
                # #region agent log
                _log_data3 = {"location": "uc_NN_subproblem.py:save:433", "message": "Creating directory", "data": {"dirpath": dirpath}, "timestamp": int(__import__('time').time()*1000), "sessionId": "debug-session", "hypothesisId": "C"}; open(_log_path, 'a', encoding='utf-8').write(_json_debug.dumps(_log_data3) + '\n')
                # #endregion
                os.makedirs(dirpath, exist_ok=True)
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, filepath)
            print(f"✓ 对偶预测模型已保存: {filepath}", flush=True)
    
    def load(self, filepath: str):
        """加载模型"""
        if TORCH_AVAILABLE:
            state = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(state['network_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            print(f"✓ 对偶预测模型已加载: {filepath}", flush=True)


# ========================== 第二部分：子问题代理约束训练（BCD方式） ==========================

class ResBlock(nn.Module):
    """残差块：Linear → LN → LeakyReLU → Linear → LN + skip"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU(0.01)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return self.act(out + identity)


class SubproblemSurrogateNet(nn.Module):
    """
    单机组子问题的代理约束网络 - V3三时段耦合版本

    输入: Pd数据 + 对偶变量λ + 机组静态参数 (pd_dim + T + 6)
    输出: 三时段耦合约束参数 (alphas, betas, gammas, deltas)
          约束形式: alpha_t * x_t + beta_t * x_{t+1} + gamma_t * x_{t+2} <= delta_t

    改进点：
    - 残差连接 + LayerNorm 稳定训练
    - 输出头增加隐层增强表达力
    - 机组静态参数作为额外特征
    """

    def __init__(
        self,
        input_dim: int,
        T: int,
        max_constraints: int = 20,
        hidden_dims: List[int] = None,
        x_cost_scale: float = 1.0,
        pg_cost_scale: float = 1.0,
        coupling_coeff_scale: float = 2.0,
        delta_base: float = 3.0,
        delta_scale: float = 2.0,
    ):
        super(SubproblemSurrogateNet, self).__init__()

        self.T = T
        self.max_constraints = max_constraints
        self.x_cost_scale = float(max(x_cost_scale, 1e-6))
        self.pg_cost_scale = float(max(pg_cost_scale, 1e-6))
        self.coupling_coeff_scale = float(max(coupling_coeff_scale, 1e-6))
        self.delta_base = float(delta_base)
        self.delta_scale = float(max(delta_scale, 1e-6))

        if hidden_dims is None:
            hidden_dims = [512, 256, 256]

        # 输入投影 + 残差块
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        res_blocks = []
        prev_dim = hidden_dims[0]
        for hd in hidden_dims:
            res_blocks.append(ResBlock(prev_dim, hd))
            prev_dim = hd
        self.res_blocks = nn.ModuleList(res_blocks)

        feat_dim = prev_dim
        head_hidden = max(feat_dim // 2, 64)
        head_mid = max(head_hidden // 2, 32)

        # 四个参数头（每个带两个隐层 + LayerNorm）
        self.alpha_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.max_constraints)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.max_constraints)
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.max_constraints)
        )
        self.delta_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.max_constraints)
        )

        # x 调整项头：输出 T 个值，后续经 tanh 缩放到启停成本量级
        self.cost_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.T)
        )
        # pg 调整项头：输出 T 个值，后续经 tanh 缩放到边际成本量级
        self.pg_cost_net = nn.Sequential(
            nn.Linear(feat_dim, head_hidden), nn.LayerNorm(head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(head_hidden, head_mid), nn.LayerNorm(head_mid), nn.LeakyReLU(0.01),
            nn.Linear(head_mid, self.T)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # delta头最后一个Linear的偏置初始化为0，使初始RHS中心落在delta_base
        delta_linears = [m for m in self.delta_net if isinstance(m, nn.Linear)]
        if delta_linears:
            last_linear = delta_linears[-1]
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def encode_features(self, x):
        features = self.input_proj(x)
        for block in self.res_blocks:
            features = block(features)
        return features

    def forward_main(self, x):
        """
        主代理网络前向传播。

        Returns:
            alphas/betas/gammas/deltas/costs
        """
        features = self.encode_features(x)
        alphas = torch.tanh(self.alpha_net(features)) * self.coupling_coeff_scale
        betas = torch.tanh(self.beta_net(features)) * self.coupling_coeff_scale
        gammas = torch.tanh(self.gamma_net(features)) * self.coupling_coeff_scale
        deltas = self.delta_base + torch.tanh(self.delta_net(features)) * self.delta_scale
        costs = torch.tanh(self.cost_net(features)) * self.x_cost_scale
        return alphas, betas, gammas, deltas, costs

    def forward_pg_cost(self, x):
        """c_pg 单独前向传播。"""
        features = self.encode_features(x)
        pg_costs = torch.tanh(self.pg_cost_net(features)) * self.pg_cost_scale
        return pg_costs

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 (batch_size, input_dim)

        Returns:
            alphas: t时段系数 (batch_size, max_constraints)
            betas: t+1时段系数 (batch_size, max_constraints)
            gammas: t+2时段系数 (batch_size, max_constraints)
            deltas: 右端项 (batch_size, max_constraints)
            costs: x 调整项 (batch_size, T)
            pg_costs: pg 调整项 (batch_size, T)
        """
        alphas, betas, gammas, deltas, costs = self.forward_main(x)
        pg_costs = self.forward_pg_cost(x)
        return alphas, betas, gammas, deltas, costs, pg_costs


CONSTRAINT_STRATEGY_ALL = "all"
CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4 = "all_templates_sign4"
CONSTRAINT_STRATEGY_ALL_SINGLE_TIME = "all_single_time"
CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE = "all_templates_sign4_plus_single"
CONSTRAINT_STRATEGY_ALIAS_MAP = {
    "all_templates_rhs3": CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
}
SURROGATE_TRIPLE_WINDOW_OFFSETS = (0, 1, 2)
SURROGATE_SINGLE_TIME_OFFSETS = (0,)
SUPPORTED_CONSTRAINT_GENERATION_STRATEGIES = {
    "sensitive",
    CONSTRAINT_STRATEGY_ALL,
    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
    CONSTRAINT_STRATEGY_ALL_SINGLE_TIME,
    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
}


def normalize_constraint_generation_strategy(strategy: str | None) -> str:
    strategy_norm = "sensitive" if strategy is None else str(strategy).strip().lower()
    strategy_norm = CONSTRAINT_STRATEGY_ALIAS_MAP.get(strategy_norm, strategy_norm)
    if strategy_norm not in SUPPORTED_CONSTRAINT_GENERATION_STRATEGIES:
        raise ValueError(
            f"Unsupported constraint_generation_strategy: {strategy}. "
            f"Supported: {sorted(SUPPORTED_CONSTRAINT_GENERATION_STRATEGIES)}"
        )
    return strategy_norm


def _normalize_constraint_offsets(offsets) -> tuple[int, ...]:
    normalized = []
    if offsets is None:
        return SURROGATE_TRIPLE_WINDOW_OFFSETS
    for offset in offsets:
        try:
            offset_int = int(offset)
        except (TypeError, ValueError):
            continue
        if 0 <= offset_int <= 2 and offset_int not in normalized:
            normalized.append(offset_int)
    return tuple(normalized) if normalized else SURROGATE_TRIPLE_WINDOW_OFFSETS


def iterate_surrogate_constraint_terms(
    timestep: int,
    offsets,
    alpha_value,
    beta_value,
    gamma_value,
    horizon: int,
):
    active_offsets = _normalize_constraint_offsets(offsets)
    if 0 in active_offsets and 0 <= timestep < horizon:
        yield timestep, alpha_value
    if 1 in active_offsets and 0 <= timestep + 1 < horizon:
        yield timestep + 1, beta_value
    if 2 in active_offsets and 0 <= timestep + 2 < horizon:
        yield timestep + 2, gamma_value


def build_surrogate_constraint_expression(
    x_values,
    timestep: int,
    offsets,
    alpha_value,
    beta_value,
    gamma_value,
    horizon: int,
):
    expr = 0
    for time_idx, coeff in iterate_surrogate_constraint_terms(
        timestep,
        offsets,
        alpha_value,
        beta_value,
        gamma_value,
        horizon,
    ):
        expr += coeff * x_values[time_idx]
    return expr


def resolve_constraint_offsets_from_trainer(
    trainer,
    sample_id: int | None,
    n_constraints: int,
):
    if n_constraints <= 0:
        return []

    offsets_by_sample = getattr(trainer, 'surrogate_constraint_offsets', None)
    if (
        isinstance(offsets_by_sample, list)
        and sample_id is not None
        and 0 <= sample_id < len(offsets_by_sample)
    ):
        resolved = [
            _normalize_constraint_offsets(offsets)
            for offsets in list(offsets_by_sample[sample_id])
        ]
        if len(resolved) >= n_constraints:
            return resolved[:n_constraints]

    strategy = normalize_constraint_generation_strategy(
        getattr(trainer, 'constraint_generation_strategy', 'sensitive') or 'sensitive'
    )
    default_offsets = (
        SURROGATE_SINGLE_TIME_OFFSETS
        if strategy == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME
        else SURROGATE_TRIPLE_WINDOW_OFFSETS
    )
    if 'resolved' in locals() and resolved:
        return resolved + [default_offsets] * (n_constraints - len(resolved))
    return [default_offsets] * n_constraints


def identify_sensitive_timesteps(x_vals, threshold_low=0.1, threshold_high=0.9, max_constraints=20):
    """
    识别整数性差的敏感时段（用于三时段约束）
    
    Args:
        x_vals: (T,) 时段变量值
        threshold_low: 下阈值，低于此值认为接近0
        threshold_high: 上阈值，高于此值认为接近1
        max_constraints: 最大约束数量
    
    Returns:
        sensitive_timesteps: 需要生成约束的时段索引列表（长度≤max_constraints）
    """
    T = len(x_vals)
    sensitive = []
    
    # 三时段约束需要t, t+1, t+2都存在
    for t in range(T - 2):
        # 检查三时段窗口是否有整数性问题
        window = x_vals[t:t+3]
        # 如果窗口内任意变量在(0.1, 0.9)区间，标记为敏感
        if any(threshold_low < x < threshold_high for x in window):
            sensitive.append(t)
    
    # 限制约束数量
    if len(sensitive) > max_constraints:
        # 按整数性从差到好排序，保留最差的max_constraints个
        violations = []
        for t in sensitive:
            window = x_vals[t:t+3]
            # 整数性：sum(x*(1-x))，越大越差
            viol = sum(x * (1-x) for x in window)
            violations.append((t, viol))

        # 按违反程度排序
        violations.sort(key=lambda item: item[1], reverse=True)
        sensitive = [t for t, _ in violations[:max_constraints]]
        sensitive.sort()  # 按时间顺序排列

    # 全整数回退：若无分数时段，按 |x[t]-0.5| 升序选最多 max_constraints 个三时段起点
    if len(sensitive) == 0:
        n_pick = min(max_constraints, T - 2)
        candidates = list(range(T - 2))
        candidates.sort(key=lambda t: min(abs(x_vals[t] - 0.5),
                                          abs(x_vals[t+1] - 0.5),
                                          abs(x_vals[t+2] - 0.5)))
        sensitive = sorted(candidates[:n_pick])

    return sensitive


def select_constraint_layout(
    x_vals,
    strategy: str = "sensitive",
    threshold_low: float = 0.1,
    threshold_high: float = 0.9,
    max_constraints: int = 20,
):
    strategy_norm = normalize_constraint_generation_strategy(strategy)
    T = len(x_vals)
    if T <= 0:
        return [], []
    if strategy_norm == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME:
        return list(range(T)), [SURROGATE_SINGLE_TIME_OFFSETS] * T
    if T < 3:
        return [], []
    if strategy_norm == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4:
        expanded_timesteps = []
        expanded_offsets = []
        for t in range(T - 2):
            expanded_timesteps.extend([t] * 4)
            expanded_offsets.extend([SURROGATE_TRIPLE_WINDOW_OFFSETS] * 4)
        return expanded_timesteps, expanded_offsets
    if strategy_norm == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
        expanded_timesteps = []
        expanded_offsets = []
        for t in range(T - 2):
            expanded_timesteps.extend([t] * 4)
            expanded_offsets.extend([SURROGATE_TRIPLE_WINDOW_OFFSETS] * 4)
        for t in range(T):
            expanded_timesteps.append(t)
            expanded_offsets.append(SURROGATE_SINGLE_TIME_OFFSETS)
        return expanded_timesteps, expanded_offsets
    if strategy_norm == CONSTRAINT_STRATEGY_ALL:
        timesteps = list(range(T - 2))
        return timesteps, [SURROGATE_TRIPLE_WINDOW_OFFSETS] * len(timesteps)
    timesteps = identify_sensitive_timesteps(
        x_vals,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        max_constraints=max_constraints,
    )
    return timesteps, [SURROGATE_TRIPLE_WINDOW_OFFSETS] * len(timesteps)


def select_constraint_timesteps(
    x_vals,
    strategy: str = "sensitive",
    threshold_low: float = 0.1,
    threshold_high: float = 0.9,
    max_constraints: int = 20,
):
    timesteps, _ = select_constraint_layout(
        x_vals,
        strategy=strategy,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        max_constraints=max_constraints,
    )
    return timesteps


class SubproblemSurrogateTrainer:
    """
    单机组子问题代理约束的BCD训练器
    
    训练方式与uc_NN_BCD.py一致：
    1. iter_with_primal_block: 固定代理约束参数，求解子问题更新原始变量(pg, x)
    2. iter_with_dual_block: 固定原始变量，求解对偶问题更新对偶变量(mu)
    3. iter_with_surrogate_nn: 使用可微分loss函数训练神经网络更新代理约束参数
    
    拉格朗日松弛子问题形式：
        min  cost_g(pg, x) - λᵀ × pg + sum_t(mu * max(0, alpha_t * x_t - beta))
        s.t. pg_min * x <= pg <= pg_max * x
             爬坡约束
             最小开关机时间约束
             启停成本约束
    """
    
    def __init__(self, ppc, active_set_data, T_delta, unit_id: int, 
                 lambda_predictor: DualVariablePredictorTrainer = None, 
                 max_constraints: int = 20,
                 constraint_generation_strategy: str = "sensitive",
                 rho_primal_init: float = 1e-3,
                 rho_dual_init: float = 1e-3,
                 rho_dual_pg_init: float | None = None,
                 rho_dual_x_init: float | None = None,
                 rho_dual_coc_init: float | None = None,
                 rho_binary_init: float = 1.0,
                 rho_opt_init: float = 1e-3,
                 gamma_base: float = 1e-3,
                 mu_lower_bound_init: float = 0.1,
                 mu_individual_lower_bound_round: int = 3,
                 mu_group_lower_bound_round: int = 50,
                 mu_signed_round_interval: int | None = None,
                 x_bound_dual_zero_rounds: int = 0,
                 pg_cost_start_round: int = 3,
                 pg_cost_scale_multiplier: float = 1.2,
                 nn_hidden_dims: List[int] = None,
                 nn_learning_rate: float = 1e-4,
                 cost_learning_rate: float = 1e-5,
                 pg_cost_lr: float = 2e-5,
                 pg_cost_surr_lr: float = 5e-5,
                 nn_batch_strategy: str = "full-batch",
                 nn_batch_size: int = 4,
                 nn_shuffle: bool = True,
                 pg_cost_nn_epochs: int | None = None,
                 pg_cost_reg_deadband: float = 0.5,
                 nn_smooth_abs_eps: float = 1e-6,
                 pg_cost_smooth_abs_eps: float = 1e-6,
                 iter_delta_reg_weight: float = 5e-5,
                 iter_delta_reg_deadband: float = 0.10,
                 loss_ratio_primal: float = 1.0,
                 loss_ratio_dual_pg: float = 1.0,
                 loss_ratio_dual_x: float = 1.0,
                 nn_dual_term_interval: int | None = 1,
                 loss_ratio_opt: float = 1.0,
                 loss_ratio_reg: float = 1.0,
                 pg_block_prox_weight: float = 2e-2,
                 dual_block_prox_weight: float = 1e-2,
                 ignore_startup_shutdown_costs: bool = False,
                 device=None):
        """
        初始化单机组子问题代理约束训练器 - V3三时段版本
        
        Args:
            ppc: PyPower案例数据
            active_set_data: 活动集数据
            T_delta: 时间间隔
            unit_id: 机组索引
            lambda_predictor: 已训练的对偶变量预测器（可选）
            max_constraints: 最大约束数量（敏感时段）
            device: 计算设备
        """
        self.ppc = ppc
        self.ppc_raw = ppc
        ppc_int = ext2int(ppc)
        self.baseMVA = ppc_int['baseMVA']
        self.bus = ppc_int['bus']
        self.gen = ppc_int['gen']
        self.branch = ppc_int['branch']
        self.gencost = ppc_int['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        self.unit_id = unit_id
        self.Ru_all, self.Rd_all, self.Ru_co_all, self.Rd_co_all = _get_ramp_limits_from_ppc(
            self.ppc_raw, self.gen, self.T_delta
        )
        self.constraint_generation_strategy = normalize_constraint_generation_strategy(
            constraint_generation_strategy
        )
        self.all_mode_group_size = 4 if self.constraint_generation_strategy in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        ) else 1
        self.template_rhs_jitter_scale = 2.0
        self.template_rhs_reg_deadband = 0.8
        self.coeff_reg_deadband = 0.35
        self.aux_cost_reg_deadband = 0.5
        self.pg_cost_reg_deadband = max(float(pg_cost_reg_deadband), 0.0)
        self.nn_smooth_abs_eps = max(float(nn_smooth_abs_eps), 0.0)
        self.pg_cost_smooth_abs_eps = max(float(pg_cost_smooth_abs_eps), 0.0)
        self.requested_max_constraints = max_constraints
        self.max_constraints = max_constraints  # V3新增
        
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
            
        self.ng = self.gen.shape[0]
        self.nb = self.bus.shape[0]
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4:
            self.max_constraints = max(self.all_mode_group_size * (self.T - 2), 0)
        elif self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
            self.max_constraints = max(self.all_mode_group_size * (self.T - 2) + self.T, 0)
        elif self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME:
            self.max_constraints = max(self.T, 0)
        elif self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL:
            self.max_constraints = max(self.T - 2, 0)
        else:
            self.max_constraints = self.requested_max_constraints
        
        # 获取实际pd_data的维度（可能只包含负荷节点）
        if isinstance(active_set_data, list):
            self.n_load = active_set_data[0]['pd_data'].shape[0]
        else:
            self.n_load = active_set_data['pd_data'].shape[0]
        
        self.active_set_data = active_set_data
        
        # 对偶变量预测器
        self.generator_injection_sensitivity = _build_generator_injection_sensitivity(self.ppc)
        self.lambda_predictor = lambda_predictor
        
        # 设备
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"[SubproblemTrainer Unit-{unit_id}] 使用 CUDA 设备: {torch.cuda.get_device_name(0)}", flush=True)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # BCD迭代参数
        self.rho_primal = float(rho_primal_init)
        self.rho_dual = float(rho_dual_init)
        self.rho_binary = float(rho_binary_init)
        self.rho_opt = float(rho_opt_init)
        self.gamma_base = float(gamma_base)
        self.gamma = self.gamma_base
        self.gamma_dual_component_scale = 3.0
        self.rho_max = 10.0
        self.reg_weight = 1e-4   # alpha/beta/gamma L2 正则化权重
        self.iter_delta_reg_weight = float(iter_delta_reg_weight)
        self.iter_delta_reg_deadband = max(float(iter_delta_reg_deadband), 0.0)
        self.mu_lower_bound = float(mu_lower_bound_init)
        rho_dual_pg_base = rho_dual_init if rho_dual_pg_init is None else rho_dual_pg_init
        rho_dual_x_base = rho_dual_init if rho_dual_x_init is None else rho_dual_x_init
        rho_dual_coc_base = rho_dual_init if rho_dual_coc_init is None else rho_dual_coc_init
        self.rho_dual_pg = float(rho_dual_pg_base)
        self.rho_dual_x = float(rho_dual_x_base)
        self.rho_dual_coc = float(rho_dual_coc_base)
        self.mu_individual_lower_bound_round = max(int(mu_individual_lower_bound_round), 0)
        self.mu_group_lower_bound_round = max(
            int(mu_group_lower_bound_round),
            self.mu_individual_lower_bound_round,
        )
        if mu_signed_round_interval is None:
            self.mu_signed_round_interval = 2
        else:
            self.mu_signed_round_interval = max(int(mu_signed_round_interval), 0)
        self.x_bound_dual_zero_rounds = max(int(x_bound_dual_zero_rounds), 0)
        self.pg_cost_start_round = max(int(pg_cost_start_round), 0)
        self.pg_cost_scale_multiplier = max(float(pg_cost_scale_multiplier), 1e-6)
        self.nn_hidden_dims = normalize_nn_hidden_dims(nn_hidden_dims, [256, 256])
        self.nn_learning_rate = max(float(nn_learning_rate), 1e-8)
        self.cost_learning_rate = max(float(cost_learning_rate), 1e-8)
        self.pg_cost_lr = max(float(pg_cost_lr), 1e-8)
        self.pg_cost_surr_lr = max(float(pg_cost_surr_lr), 1e-8)
        self.nn_batch_strategy = normalize_nn_batch_strategy(nn_batch_strategy)
        self.nn_batch_size = max(int(nn_batch_size), 1)
        self.nn_shuffle = bool(nn_shuffle)
        self.pg_cost_nn_epochs = None if pg_cost_nn_epochs is None else max(int(pg_cost_nn_epochs), 1)
        self.loss_ratio_primal = float(loss_ratio_primal)
        self.loss_ratio_dual_pg = float(loss_ratio_dual_pg)
        self.loss_ratio_dual_x = float(loss_ratio_dual_x)
        if nn_dual_term_interval is None:
            self.nn_dual_term_interval = None
        else:
            resolved_dual_interval = int(nn_dual_term_interval)
            if resolved_dual_interval <= 0:
                raise ValueError(
                    f"nn_dual_term_interval must be None or a positive integer, got {nn_dual_term_interval}"
                )
            self.nn_dual_term_interval = resolved_dual_interval
        self.loss_ratio_opt = float(loss_ratio_opt)
        self.loss_ratio_reg = float(loss_ratio_reg)
        self.pg_block_prox_weight = max(float(pg_block_prox_weight), 0.0)
        self.dual_block_prox_weight = max(float(dual_block_prox_weight), 0.0)
        self.ignore_startup_shutdown_costs = bool(ignore_startup_shutdown_costs)
        self.iter_number = 0
        self.x_cost_scale = 2.0
        self.pg_cost_scale = 1.0
        self._sync_rho_dual_summary()
        
        # 初始化原始变量和对偶变量存储
        self.pg = np.zeros((self.n_samples, self.T))
        self.x = np.zeros((self.n_samples, self.T))
        self.coc = np.zeros((self.n_samples, self.T-1))
        self.cpower = np.zeros((self.n_samples, self.T))
        
        # 三时段耦合约束，每个样本可能有不同数量的约束（≤max_constraints）
        # 初始化为max_constraints大小
        self.num_coupling_constraints = self.max_constraints
        self.template_rhs_base_vector = self._build_template_rhs_base_vector(self.num_coupling_constraints)
        self.surrogate_direction_signs = np.ones(self.num_coupling_constraints, dtype=float)
        self.mu = np.ones((self.n_samples, self.num_coupling_constraints), dtype=float) * self.mu_lower_bound

        # 固有约束的对偶变量（由dual block更新，用于NN loss的完整KKT驻点条件）
        self.lambda_inherent = [None] * self.n_samples

        # 存储每个样本的敏感时段索引
        self.sensitive_timesteps = [[] for _ in range(self.n_samples)]
        self.surrogate_constraint_offsets = [[] for _ in range(self.n_samples)]
        
        # 获取对偶变量λ
        self.lambda_vals = self._get_lambda_values()
        lambda_abs_mean = float(np.mean(np.abs(self.lambda_vals))) if self.lambda_vals.size > 0 else 0.0
        base_pg_cost_scale = max(lambda_abs_mean / 5.0, 1e-3)
        self.pg_cost_scale = base_pg_cost_scale * self.pg_cost_scale_multiplier
        
        # 初始化三时段耦合代理约束参数（占位，后续由NN填充）
        self.alpha_values = np.zeros((self.n_samples, self.num_coupling_constraints))
        self.beta_values = np.zeros((self.n_samples, self.num_coupling_constraints))
        self.gamma_values = np.zeros((self.n_samples, self.num_coupling_constraints))
        self.delta_values = np.full((self.n_samples, self.num_coupling_constraints), 3.0)
        self.cost_values = np.zeros((self.n_samples, self.T))
        self.pg_cost_values = np.zeros((self.n_samples, self.T))

        # 初始化神经网络，并用forward pass生成初值
        if TORCH_AVAILABLE:
            self._init_neural_network()
            self._generate_initial_values_from_nn()
        else:
            # 回退：保持zeros/ones默认值
            pass

        self._apply_initial_surrogate_templates()
        
        # 初始化求解
        self._initialize_solve()

        # 迭代间输出差异正则：用于抑制 NN 输出在相邻 BCD 迭代间剧烈跳变
        self._prev_alpha_values = None
        self._prev_beta_values = None
        self._prev_gamma_values = None
        self._prev_delta_values = None
        self._prev_cost_values = None
        self._prev_pg_cost_values = None
        
        print(f"✓ 机组{unit_id}子问题代理约束训练器初始化完成", flush=True)
    
    def _get_lambda_values(self) -> np.ndarray:
        """获取对偶变量λ"""
        if self.lambda_predictor is not None:
            # 使用预测器
            lambda_vals = []
            for sample_id in range(self.n_samples):
                sample = self.active_set_data[sample_id]
                lambda_pred = self.lambda_predictor.predict(sample)
                lambda_vals.append(lambda_pred)
            return np.array(lambda_vals)
        else:
            # 使用真值（需要先求解原问题）
            return self._solve_for_lambda()
    
    def _solve_for_lambda(self) -> np.ndarray:
        """获取对偶变量λ（优先从数据读取，否则从active_set提取x求解LP）"""
        lambda_vals = []
        needs_solve = []
        
        # 检查JSON中是否已有lambda
        for sample_id in range(self.n_samples):
            if 'lambda' in self.active_set_data[sample_id] and \
               self.active_set_data[sample_id]['lambda'] is not None:
                lambda_vals.append(_extract_lambda_power_balance(
                    self.active_set_data[sample_id]['lambda'], self.T))
            else:
                needs_solve.append(sample_id)
        
        # 如果所有样本都有lambda，直接返回
        if not needs_solve:
            return np.array(lambda_vals)
        
        # 否则通过求解LP获取缺失的lambda
        print(f"⚠ {len(needs_solve)} 个样本缺少 lambda，从 active_set 提取 x 并求解 LP...", flush=True)
        
        # 构建完整的lambda_vals字典（按sample_id索引）
        already_loaded_ids = [i for i in range(self.n_samples) if i not in needs_solve]
        lambda_dict = {sid: lambda_vals[idx] for idx, sid in enumerate(already_loaded_ids)}

        for sample_id in needs_solve:
            Pd = self.active_set_data[sample_id]['pd_data']

            # 恢复x矩阵：优先用 active_set，否则用 unit_commitment_matrix
            x_sol = np.zeros((self.ng, self.T))
            if 'active_set' in self.active_set_data[sample_id]:
                active_set = self.active_set_data[sample_id]['active_set']
                for item in active_set:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                        g, t = item[0]
                        value = item[1]
                        x_sol[g, t] = value
            elif 'unit_commitment_matrix' in self.active_set_data[sample_id]:
                uc = self.active_set_data[sample_id]['unit_commitment_matrix']
                x_sol[:uc.shape[0], :] = uc

            # 用x求解ED（LP），获取对偶变量
            from ed_cvxpy import EconomicDispatchCVXPY
            ed = EconomicDispatchCVXPY(self.ppc, Pd, self.T_delta, x_sol)
            pg_sol, total_cost = ed.solve()

            # 提取功率平衡约束的对偶变量λ（前T个约束）
            lambda_sample = np.zeros(self.T)
            for t in range(self.T):
                dual_val = ed.constraints[t].dual_value
                if dual_val is None:
                    lambda_sample[t] = 0.0
                else:
                    lambda_sample[t] = float(dual_val)

            lambda_dict[sample_id] = lambda_sample

        # 按sample_id顺序返回
        return np.array([lambda_dict[i] for i in range(self.n_samples)])
    
    def _init_neural_network(self):
        """初始化代理约束神经网络 - V3版本"""
        input_dim = len(self._extract_features(0))
        delta_base = 0.0 if self._uses_template_rhs_bases() else 3.0
        delta_scale = self.template_rhs_jitter_scale if self._uses_template_rhs_bases() else 3.0

        self.surrogate_net = SubproblemSurrogateNet(
            input_dim=input_dim,
            T=self.T,
            max_constraints=self.max_constraints,
            hidden_dims=self.nn_hidden_dims,
            x_cost_scale=self.x_cost_scale,
            pg_cost_scale=self.pg_cost_scale,
            delta_base=delta_base,
            delta_scale=delta_scale,
        ).to(self.device)

        # 分离参数组：主优化器管理 backbone + alpha/beta/gamma/delta 头
        aux_params = set(self.surrogate_net.cost_net.parameters()) | set(self.surrogate_net.pg_cost_net.parameters())
        main_params = [p for p in self.surrogate_net.parameters() if p not in aux_params]
        self.optimizer = optim.Adam(main_params, lr=self.nn_learning_rate)
        # x / pg 调整项头拆分优化，便于仅对 c_pg 提高学习率并延迟启用
        self.cost_optimizer = optim.Adam(
            self.surrogate_net.cost_net.parameters(),
            lr=self.cost_learning_rate,
        )
        self.pg_cost_optimizer = optim.Adam(
            self.surrogate_net.pg_cost_net.parameters(),
            lr=self.pg_cost_lr,
        )

        print(f"  - 代理约束网络输入维度: {input_dim}", flush=True)
        print(f"  - 隐藏层: {self.nn_hidden_dims}", flush=True)
        print(f"  - 约束生成策略: {self.constraint_generation_strategy}", flush=True)
        print(f"  - 最大约束数量: {self.max_constraints}", flush=True)
        print(f"  - x_cost_scale={self.x_cost_scale:.4f}, pg_cost_scale={self.pg_cost_scale:.4f}", flush=True)
        print(f"  - delta_base={delta_base:.4f}, delta_scale={delta_scale:.4f}", flush=True)
        print(
            f"  - coeff_reg_deadband={self.coeff_reg_deadband:.4f}, "
            f"delta_reg_deadband={self.template_rhs_reg_deadband:.4f}",
            flush=True,
        )
        print(
            f"  - c_pg_start_round={self.pg_cost_start_round}, "
            f"main_nn_lr={self.nn_learning_rate:.2e}, x_cost_nn_lr={self.cost_learning_rate:.2e}, "
            f"c_pg_lr={self.pg_cost_lr:.2e}, c_pg_surr_lr={self.pg_cost_surr_lr:.2e}, "
            f"c_pg_epochs={self.pg_cost_nn_epochs}, c_pg_deadband={self.pg_cost_reg_deadband:.4f}, "
            f"nn_smooth_eps={self.nn_smooth_abs_eps:.1e}, "
            f"c_pg_smooth_eps={self.pg_cost_smooth_abs_eps:.1e}",
            flush=True,
        )
        print(f"  - nn_dual_term_interval={self.nn_dual_term_interval}", flush=True)

    def _pg_costs_active(self) -> bool:
        return self.iter_number >= self.pg_cost_start_round

    def _sync_rho_dual_summary(self) -> None:
        self.rho_dual = float(np.mean([
            self.rho_dual_pg,
            self.rho_dual_x,
            self.rho_dual_coc,
        ]))

    def _nn_dual_terms_active(self) -> bool:
        if self.nn_dual_term_interval is None:
            return False
        if self.nn_dual_term_interval <= 1:
            return True
        return ((self.iter_number + 1) % self.nn_dual_term_interval) == 0

    def _add_squared_distance_term(self, expr, reference_value: float, scale: float):
        reference_value = float(reference_value)
        scale = max(float(scale), 1e-6)
        diff = (expr - reference_value) / scale
        return diff * diff

    def _smooth_abs(self, tensor: torch.Tensor, eps: float) -> torch.Tensor:
        eps = max(float(eps), 0.0)
        if eps <= 0:
            return torch.abs(tensor)
        return torch.sqrt(tensor * tensor + eps) - np.sqrt(eps)

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

    def _build_primal_block_prox_obj(self, model, sample_id: int, pg, x, coc):
        prox_obj = gp.LinExpr()
        if self.pg_block_prox_weight <= 0:
            return prox_obj

        prev_pg = self.pg[sample_id]
        prev_x = self.x[sample_id]
        prev_coc = self.coc[sample_id]
        g = self.unit_id
        pg_scale = max(float(self.gen[g, PMAX]), 1.0)
        coc_scale = 1.0 if self.ignore_startup_shutdown_costs else max(float(self.gencost[g, 1]), float(self.gencost[g, 2]), 1.0)

        for t in range(self.T):
            prox_obj += self._add_squared_distance_term(
                pg[t],
                prev_pg[t],
                pg_scale,
            )
            prox_obj += self._add_squared_distance_term(
                x[t],
                prev_x[t],
                1.0,
            )
        for t in range(self.T - 1):
            prox_obj += self._add_squared_distance_term(
                coc[t],
                prev_coc[t],
                coc_scale,
            )

        return prox_obj

    def _empty_lambda_inherent(self, Ton: int, Toff: int) -> dict:
        return {
            'lambda_pg_lower': np.zeros(self.T, dtype=float),
            'lambda_pg_upper': np.zeros(self.T, dtype=float),
            'lambda_ramp_up': np.zeros(max(self.T - 1, 0), dtype=float),
            'lambda_ramp_down': np.zeros(max(self.T - 1, 0), dtype=float),
            'lambda_min_on': np.array(
                [np.zeros(self.T - tau, dtype=float) for tau in range(1, Ton + 1)],
                dtype=object,
            ),
            'lambda_min_off': np.array(
                [np.zeros(self.T - tau, dtype=float) for tau in range(1, Toff + 1)],
                dtype=object,
            ),
            'lambda_start_cost': np.zeros(max(self.T - 1, 0), dtype=float),
            'lambda_shut_cost': np.zeros(max(self.T - 1, 0), dtype=float),
            'lambda_coc_nonneg': np.zeros(max(self.T - 1, 0), dtype=float),
            'lambda_x_upper': np.zeros(self.T, dtype=float),
            'lambda_x_lower': np.zeros(self.T, dtype=float),
        }

    def _build_dual_block_prox_obj(
        self,
        model,
        sample_id: int,
        lam_pg_lower,
        lam_pg_upper,
        lam_ramp_up,
        lam_ramp_down,
        lam_start_cost,
        lam_shut_cost,
        lam_coc_nonneg,
        lam_x_upper,
        lam_x_lower,
        lam_min_on,
        lam_min_off,
        mu,
        Ton: int,
        Toff: int,
    ):
        prox_obj = gp.LinExpr()
        if self.dual_block_prox_weight <= 0:
            return prox_obj

        prev_lambda = self.lambda_inherent[sample_id]
        if prev_lambda is None:
            prev_lambda = self._empty_lambda_inherent(Ton, Toff)
        prev_mu = self.mu[sample_id]

        for t in range(self.T):
            for var_name, var_container in (
                ('lambda_pg_lower', lam_pg_lower),
                ('lambda_pg_upper', lam_pg_upper),
                ('lambda_x_upper', lam_x_upper),
                ('lambda_x_lower', lam_x_lower),
            ):
                prev_val = float(prev_lambda[var_name][t])
                prox_obj += self._add_squared_distance_term(
                    var_container[t],
                    prev_val,
                    max(1.0, abs(prev_val)),
                )

        for t in range(self.T - 1):
            for var_name, var_container in (
                ('lambda_ramp_up', lam_ramp_up),
                ('lambda_ramp_down', lam_ramp_down),
                ('lambda_start_cost', lam_start_cost),
                ('lambda_shut_cost', lam_shut_cost),
                ('lambda_coc_nonneg', lam_coc_nonneg),
            ):
                prev_val = float(prev_lambda[var_name][t])
                prox_obj += self._add_squared_distance_term(
                    var_container[t],
                    prev_val,
                    max(1.0, abs(prev_val)),
                )

        for tau in range(1, Ton + 1):
            for t1 in range(self.T - tau):
                prev_val = float(prev_lambda['lambda_min_on'][tau - 1][t1])
                prox_obj += self._add_squared_distance_term(
                    lam_min_on[tau - 1, t1],
                    prev_val,
                    max(1.0, abs(prev_val)),
                )

        for tau in range(1, Toff + 1):
            for t1 in range(self.T - tau):
                prev_val = float(prev_lambda['lambda_min_off'][tau - 1][t1])
                prox_obj += self._add_squared_distance_term(
                    lam_min_off[tau - 1, t1],
                    prev_val,
                    max(1.0, abs(prev_val)),
                )

        for k in range(self.num_coupling_constraints):
            prev_val = float(prev_mu[k])
            prox_obj += self._add_squared_distance_term(
                mu[k],
                prev_val,
                max(1.0, abs(prev_val)),
            )

        return prox_obj

    def _uses_template_rhs_bases(self) -> bool:
        return self.constraint_generation_strategy in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        )

    def _build_all_template_patterns_and_rhs(self) -> tuple[np.ndarray, np.ndarray]:
        patterns = np.array(
            [
                [1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        # RHS 基准 = 各模板在 x∈{0,1} 下的最大 LHS，保证所有整数组合初始可行
        rhs = np.array([2.0, 1.0, 1.0, 0.0], dtype=float)
        return patterns, rhs

    def _build_template_rhs_base_vector(self, size: int) -> np.ndarray:
        if not self._uses_template_rhs_bases() or size <= 0:
            return np.zeros(max(size, 0), dtype=float)
        _, rhs = self._build_all_template_patterns_and_rhs()
        n_sign4 = self.all_mode_group_size * max(self.T - 2, 0)
        n_groups = n_sign4 // self.all_mode_group_size
        if n_groups <= 0:
            return np.zeros(size, dtype=float)
        sign4_part = np.tile(rhs, n_groups)
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
            single_part = np.ones(max(size - n_sign4, 0), dtype=float)
            return np.concatenate([sign4_part, single_part])[:size]
        return sign4_part[:size]

    def _postprocess_delta_tensor(self, deltas_tensor: torch.Tensor) -> torch.Tensor:
        if not self._uses_template_rhs_bases():
            return deltas_tensor
        if deltas_tensor.ndim == 2:
            base = torch.tensor(
                self.template_rhs_base_vector[:deltas_tensor.shape[1]],
                dtype=deltas_tensor.dtype,
                device=deltas_tensor.device,
            ).unsqueeze(0)
            return base + deltas_tensor
        base = torch.tensor(
            self.template_rhs_base_vector[:deltas_tensor.shape[0]],
            dtype=deltas_tensor.dtype,
            device=deltas_tensor.device,
        )
        return base + deltas_tensor

    def _deadband_quadratic(
        self,
        tensor: torch.Tensor,
        deadband: float,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if deadband <= 0:
            ref = tensor if center is None else (tensor - center)
            return torch.sum(ref ** 2)
        ref = tensor if center is None else (tensor - center)
        excess = self._smooth_deadband_excess(ref, deadband)
        return torch.sum(excess ** 2)

    def _iter_delta_regularization(
        self,
        current: torch.Tensor,
        prev: torch.Tensor | None,
        deadband: float,
    ) -> torch.Tensor:
        """惩罚当前输出与上一代输出的差异（deadband 外二次惩罚）。"""
        if prev is None:
            return torch.tensor(0.0, dtype=current.dtype, device=current.device)
        diff = current - prev
        if deadband <= 0:
            return torch.sum(diff ** 2)
        excess = self._smooth_deadband_excess(diff, deadband)
        return torch.sum(excess ** 2)

    def _gate_pg_cost_tensor(self, pg_costs_tensor: torch.Tensor) -> torch.Tensor:
        if self._pg_costs_active():
            return pg_costs_tensor
        return torch.zeros_like(pg_costs_tensor)

    def _set_c_pg_training_mode(self, enabled: bool) -> None:
        """冻结主代理参数，仅训练 c_pg head；退出时恢复全部可训练。"""
        for param in self.surrogate_net.parameters():
            param.requires_grad_(not enabled)
        if enabled:
            for param in self.surrogate_net.pg_cost_net.parameters():
                param.requires_grad_(True)

    def _resolve_nn_batch_config(
        self,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
    ) -> tuple[str, int, bool]:
        resolved_batch_strategy = normalize_nn_batch_strategy(
            self.nn_batch_strategy if batch_strategy is None else batch_strategy
        )
        resolved_shuffle = self.nn_shuffle if shuffle is None else bool(shuffle)
        if resolved_batch_strategy == "full-batch":
            resolved_batch_size = max(1, self.n_samples)
        else:
            base_batch_size = self.nn_batch_size if batch_size is None else int(batch_size)
            resolved_batch_size = max(1, base_batch_size)
        return resolved_batch_strategy, resolved_batch_size, resolved_shuffle

    def _refresh_cached_surrogate_outputs(self, zero_auxiliary: bool = False) -> None:
        """Recompute cached surrogate parameters from the current network weights."""
        if not TORCH_AVAILABLE or self.n_samples <= 0:
            return

        was_training = self.surrogate_net.training
        self.surrogate_net.eval()

        features_list = [self._extract_features(s) for s in range(self.n_samples)]
        feat_tensor = torch.tensor(
            np.array(features_list),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            alphas, betas, gammas, deltas, costs = self.surrogate_net.forward_main(feat_tensor)
            pg_costs = self.surrogate_net.forward_pg_cost(feat_tensor)
            deltas = self._postprocess_delta_tensor(deltas)

            if zero_auxiliary:
                costs = torch.zeros_like(costs)
                pg_costs = torch.zeros_like(pg_costs)
            elif not self._pg_costs_active():
                pg_costs = torch.zeros_like(pg_costs)

        self.alpha_values = alphas.detach().cpu().numpy()
        self.beta_values = betas.detach().cpu().numpy()
        self.gamma_values = gammas.detach().cpu().numpy()
        self.delta_values = deltas.detach().cpu().numpy()
        self.cost_values = costs.detach().cpu().numpy()
        self.pg_cost_values = pg_costs.detach().cpu().numpy()

        if was_training:
            self.surrogate_net.train()

    def _weighted_violation_merit(
        self,
        components: Tuple[float, float, float, float, float, float],
    ) -> float:
        """Single merit used to accept or reject NN updates."""
        obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, _, obj_opt = components
        return float(
            self.loss_ratio_primal * self.rho_primal * obj_primal
            + self.loss_ratio_dual_pg * self.rho_dual_pg * obj_dual_pg
            + self.loss_ratio_dual_x * self.rho_dual_x * obj_dual_x
            + self.rho_dual_coc * obj_dual_coc
            + self.loss_ratio_opt * self.rho_opt * obj_opt
        )

    def _capture_nn_update_state(self) -> dict:
        """Snapshot current NN/cached state so a bad update can be rolled back."""
        state = {
            'surrogate_net_state_dict': copy.deepcopy(self.surrogate_net.state_dict()),
            'alpha_values': self.alpha_values.copy(),
            'beta_values': self.beta_values.copy(),
            'gamma_values': self.gamma_values.copy(),
            'delta_values': self.delta_values.copy(),
            'cost_values': self.cost_values.copy(),
            'pg_cost_values': self.pg_cost_values.copy(),
            'last_surr_nn_loss': getattr(self, '_last_surr_nn_loss', None),
            'last_pg_cost_nn_loss': getattr(self, '_last_pg_cost_nn_loss', None),
        }
        for attr_name in (
            '_surr_optimizer',
            '_surr_cost_optimizer',
            '_surr_scheduler',
            '_surr_pg_cost_optimizer',
            '_surr_pg_cost_scheduler',
        ):
            attr_value = getattr(self, attr_name, None)
            state[attr_name] = None if attr_value is None else copy.deepcopy(attr_value.state_dict())
        return state

    def _restore_nn_update_state(self, state: dict) -> None:
        """Restore the last accepted NN/cached state."""
        self.surrogate_net.load_state_dict(state['surrogate_net_state_dict'])
        self.alpha_values = state['alpha_values'].copy()
        self.beta_values = state['beta_values'].copy()
        self.gamma_values = state['gamma_values'].copy()
        self.delta_values = state['delta_values'].copy()
        self.cost_values = state['cost_values'].copy()
        self.pg_cost_values = state['pg_cost_values'].copy()
        self._last_surr_nn_loss = state.get('last_surr_nn_loss')
        self._last_pg_cost_nn_loss = state.get('last_pg_cost_nn_loss')

        for attr_name in (
            '_surr_optimizer',
            '_surr_cost_optimizer',
            '_surr_scheduler',
            '_surr_pg_cost_optimizer',
            '_surr_pg_cost_scheduler',
        ):
            attr_state = state.get(attr_name)
            attr_value = getattr(self, attr_name, None)
            if attr_state is None:
                setattr(self, attr_name, None)
            elif attr_value is not None:
                attr_value.load_state_dict(attr_state)

    def _generate_initial_values_from_nn(self):
        """用未训练的 SubproblemSurrogateNet forward pass 生成 alpha/beta/gamma/delta 初值。"""
        self._refresh_cached_surrogate_outputs(zero_auxiliary=True)

        print(f"  ✓ 用NN forward pass生成代理约束初值 "
              f"(alpha: mean={self.alpha_values.mean():.4f}; "
              f"delta: mean={self.delta_values.mean():.4f}; "
              f"cost/pg_cost: forced to 0)", flush=True)

    def _apply_initial_surrogate_templates(self):
        """Apply deterministic surrogate templates for expanded all-mode constraints."""
        if self.constraint_generation_strategy not in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        ) or self.num_coupling_constraints <= 0:
            return

        base_patterns, base_rhs = self._build_all_template_patterns_and_rhs()
        n_sign4 = self.all_mode_group_size * max(self.T - 2, 0)
        n_groups = n_sign4 // self.all_mode_group_size
        if n_groups <= 0:
            return

        # sign4 部分：覆盖前 n_sign4 条约束
        self.alpha_values[:, :n_sign4] = np.tile(base_patterns[:, 0], n_groups)
        self.beta_values[:, :n_sign4] = np.tile(base_patterns[:, 1], n_groups)
        self.gamma_values[:, :n_sign4] = np.tile(base_patterns[:, 2], n_groups)
        self.delta_values[:, :n_sign4] = np.tile(base_rhs, n_groups)
        # single_time 尾部保持 NN 初值（alpha 自由，beta/gamma=0，delta 由 NN 生成）

    def _uses_group_mu_lower_bound(self) -> bool:
        return self.constraint_generation_strategy in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        ) and self.all_mode_group_size > 1

    def _sync_surrogate_direction_strategy_state(self) -> None:
        self.surrogate_direction_signs = self._get_surrogate_direction_signs()

    def _get_surrogate_direction_signs(self, size: int | None = None) -> np.ndarray:
        signs = getattr(self, 'surrogate_direction_signs', None)
        if signs is None:
            signs = np.ones(self.num_coupling_constraints, dtype=float)
            self.surrogate_direction_signs = signs
        signs = np.asarray(signs, dtype=float)
        if signs.ndim != 1 or signs.size != self.num_coupling_constraints:
            signs = np.ones(self.num_coupling_constraints, dtype=float)
            self.surrogate_direction_signs = signs
        if size is None:
            return signs.copy()
        return signs[: int(size)].copy()

    def _is_mu_sign_relaxation_round(self) -> bool:
        interval = int(getattr(self, 'mu_signed_round_interval', 0) or 0)
        lb = float(self._current_mu_lower_bound_value())
        return (
            interval > 0
            and lb > 0.0
            and ((self.iter_number + 1) % interval == 0)
        )

    def _resolve_surrogate_direction_signs_from_mu(
        self,
        mu_values_by_sample: np.ndarray,
        tol: float = 1e-8,
    ) -> np.ndarray:
        base_signs = self._get_surrogate_direction_signs()
        mu_arr = np.asarray(mu_values_by_sample, dtype=float)
        if mu_arr.size == 0:
            return base_signs
        if mu_arr.ndim == 1:
            mu_arr = mu_arr.reshape(1, -1)
        if mu_arr.shape[-1] != self.num_coupling_constraints:
            return base_signs

        pos_count = np.sum(mu_arr > tol, axis=0)
        neg_count = np.sum(mu_arr < -tol, axis=0)
        resolved = base_signs.copy()
        resolved[pos_count > neg_count] = 1.0
        resolved[neg_count > pos_count] = -1.0
        return resolved

    def _apply_surrogate_direction_to_params(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        deltas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        signs = self._get_surrogate_direction_signs(len(alphas))
        return (
            np.asarray(alphas, dtype=float) * signs,
            np.asarray(betas, dtype=float) * signs,
            np.asarray(gammas, dtype=float) * signs,
            np.asarray(deltas, dtype=float) * signs,
        )

    def _constraint_offsets_for_sample(self, sample_id: int) -> list[tuple[int, ...]]:
        return resolve_constraint_offsets_from_trainer(
            self,
            sample_id,
            len(self.sensitive_timesteps[sample_id]),
        )

    def _get_mu_lower_bound_phase(self) -> str:
        if self.iter_number < self.mu_individual_lower_bound_round:
            return "individual"
        if self.iter_number < self.mu_group_lower_bound_round:
            return "group" if self._uses_group_mu_lower_bound() else "individual"
        return "none"

    def _current_mu_lower_bound_value(self) -> float:
        return self.mu_lower_bound if self._get_mu_lower_bound_phase() != "none" else 0.0

    def _force_zero_x_bound_duals(self) -> bool:
        return self.iter_number < self.x_bound_dual_zero_rounds

    def _apply_mu_lower_bound_policy(self, mu_values: np.ndarray, lb_mu: float) -> np.ndarray:
        """Preserve grouped lower-bound semantics for |mu| in all mode."""
        mu_arr = np.abs(np.asarray(mu_values, dtype=float).reshape(-1))
        if lb_mu <= 0:
            return mu_arr

        phase = self._get_mu_lower_bound_phase()
        if phase == "individual" or not self._uses_group_mu_lower_bound():
            return np.maximum(mu_arr, lb_mu)
        if phase == "none":
            return mu_arr

        for start in range(0, mu_arr.size, self.all_mode_group_size):
            stop = min(start + self.all_mode_group_size, mu_arr.size)
            if stop - start != self.all_mode_group_size:
                continue
            deficit = lb_mu - float(np.sum(mu_arr[start:stop]))
            if deficit > 0:
                mu_arr[start:stop] += deficit / self.all_mode_group_size
        return mu_arr

    def _finalize_mu_values(
        self,
        mu_values: np.ndarray,
        lb_mu: float,
        direction_signs: np.ndarray | None = None,
    ) -> np.ndarray:
        mu_abs = self._apply_mu_lower_bound_policy(mu_values, lb_mu)
        return mu_abs

    def _initialize_solve(self):
        """初始化求解：从active_set提取x，求解LP获取初始原始解和对偶变量（lambda_inherent）。

        求解不含代理约束的单机组原问题LP，同时提取约束对偶变量作为 lambda_inherent 的初始值，
        确保 BCD 第一次迭代时 obj_opt 已有有意义的对偶权重。
        """
        g    = self.unit_id
        Pmin = self.gen[g, PMIN]
        Pmax = self.gen[g, PMAX]
        Ru    = float(self.Ru_all[g])
        Rd    = float(self.Rd_all[g])
        Ru_co = float(self.Ru_co_all[g])
        Rd_co = float(self.Rd_co_all[g])
        a     = self.gencost[g, -2] / self.T_delta
        b     = self.gencost[g, -1] / self.T_delta
        sc    = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shc   = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton   = min(4, self.T)
        Toff  = min(4, self.T)

        def _get_pi(m, name):
            """安全获取约束对偶变量（负值截断为0）"""
            try:
                return max(0.0, m.getConstrByName(name).Pi)
            except Exception:
                return 0.0

        for sample_id in range(self.n_samples):
            lambda_val = self.lambda_vals[sample_id]

            # 恢复x：优先用 active_set，否则用 unit_commitment_matrix
            x_init = np.zeros(self.T)
            _x_source = 'none'
            if 'active_set' in self.active_set_data[sample_id]:
                active_set = self.active_set_data[sample_id]['active_set']
                for item in active_set:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                        g_idx, t = item[0]
                        value = item[1]
                        if g_idx == g:
                            x_init[t] = value
                _x_source = 'active_set'
            elif 'unit_commitment_matrix' in self.active_set_data[sample_id]:
                uc = self.active_set_data[sample_id]['unit_commitment_matrix']
                if isinstance(uc, np.ndarray) and uc.ndim == 2 and g < uc.shape[0]:
                    x_init = uc[g].astype(float)
                    _x_source = 'unit_commitment_matrix'

            if np.all(x_init == 0) and _x_source != 'none':
                print(f"  ⚠ 样本 {sample_id} 机组 {g}: x_init 全零（来源={_x_source}），"
                      f"该机组在所有时段均关机或数据缺失", flush=True)
            elif _x_source == 'none':
                print(f"  ⚠ 样本 {sample_id} 机组 {g}: 数据中既无 active_set 也无 "
                      f"unit_commitment_matrix，x_init 使用全零默认值", flush=True)

            # 求解单机组LP松弛，目标: cost - λᵀpg
            model = gp.Model('init_subproblem_LP')
            model.Params.OutputFlag = 0

            pg     = model.addVars(self.T,   lb=0,       name='pg')
            x      = model.addVars(self.T,   lb=0, ub=1, name='x')   # LP 松弛（连续）
            coc    = model.addVars(self.T-1, lb=0,       name='coc')
            cpower = model.addVars(self.T,   lb=0,       name='cpower')

            # 发电上下限（x 为松弛变量）
            for t in range(self.T):
                model.addConstr(pg[t] >= Pmin * x[t], name=f'pg_lower_{t}')
                model.addConstr(pg[t] <= Pmax * x[t], name=f'pg_upper_{t}')

            # 爬坡约束（与 dual block 一致：Ru_co=0.3*Pmax）
            for t in range(1, self.T):
                model.addConstr(
                    pg[t] - pg[t-1] <= Ru * x[t-1] + Ru_co * (1 - x[t-1]),
                    name=f'ramp_up_{t}')
                model.addConstr(
                    pg[t-1] - pg[t] <= Rd * x[t] + Rd_co * (1 - x[t]),
                    name=f'ramp_down_{t}')

            # 最小开机时间（LP 松弛形式）
            for tau in range(1, Ton + 1):
                for t1 in range(self.T - tau):
                    model.addConstr(
                        x[t1+1] - x[t1] <= x[t1+tau],
                        name=f'min_on_{tau}_{t1}')
            # 最小关机时间（LP 松弛形式）
            for tau in range(1, Toff + 1):
                for t1 in range(self.T - tau):
                    model.addConstr(
                        -x[t1+1] + x[t1] <= 1 - x[t1+tau],
                        name=f'min_off_{tau}_{t1}')

            # 启停成本
            for t in range(1, self.T):
                model.addConstr(coc[t-1] >= sc  * (x[t] - x[t-1]), name=f'start_cost_{t}')
                model.addConstr(coc[t-1] >= shc * (x[t-1] - x[t]), name=f'shut_cost_{t}')

            # 发电成本（等式约束，与 dual block 假设 lambda_cpower=1 一致）
            for t in range(self.T):
                model.addConstr(cpower[t] == a * pg[t] + b * x[t], name=f'cpower_{t}')

            # 目标: min cost - λᵀpg（用于提取有意义的对偶变量）
            obj = (gp.quicksum(cpower[t] for t in range(self.T)) +
                   gp.quicksum(coc[t]    for t in range(self.T-1)) -
                   gp.quicksum(lambda_val[t] * pg[t] for t in range(self.T)))
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                x_lp = np.array([x[t].X for t in range(self.T)])    # LP 松弛解（连续）
                self.pg[sample_id]     = np.array([pg[t].X     for t in range(self.T)])
                self.x[sample_id]      = x_lp                        # ← 连续松弛解
                # 将 JSON 整数解存入 sample，作为 iter_with_primal_block 的持久锚点
                # （不随 BCD 迭代更新，确保 x_true 始终指向原始 MILP 最优解）
                self.active_set_data[sample_id]['x_true'] = x_init.copy()

                # 识别敏感时段（分数解 → 优先覆盖；全整数 → 按距0.5升序补齐）
                (
                    self.sensitive_timesteps[sample_id],
                    self.surrogate_constraint_offsets[sample_id],
                ) = select_constraint_layout(
                    x_lp,
                    strategy=self.constraint_generation_strategy,
                    threshold_low=0.1, threshold_high=0.9,
                    max_constraints=self.max_constraints
                )
                self.coc[sample_id]    = np.array([coc[t].X    for t in range(self.T-1)])
                self.cpower[sample_id] = np.array([cpower[t].X for t in range(self.T)])

                # 提取约束对偶变量作为 lambda_inherent 初始值
                self.lambda_inherent[sample_id] = {
                    'lambda_pg_lower':   np.array([_get_pi(model, f'pg_lower_{t}')
                                                   for t in range(self.T)]),
                    'lambda_pg_upper':   np.array([_get_pi(model, f'pg_upper_{t}')
                                                   for t in range(self.T)]),
                    'lambda_ramp_up':    np.array([_get_pi(model, f'ramp_up_{t}')
                                                   for t in range(1, self.T)]),
                    'lambda_ramp_down':  np.array([_get_pi(model, f'ramp_down_{t}')
                                                   for t in range(1, self.T)]),
                    'lambda_min_on':     np.array([[_get_pi(model, f'min_on_{tau}_{t1}')
                                                    for t1 in range(self.T - tau)]
                                                   for tau in range(1, Ton+1)],  dtype=object),
                    'lambda_min_off':    np.array([[_get_pi(model, f'min_off_{tau}_{t1}')
                                                    for t1 in range(self.T - tau)]
                                                   for tau in range(1, Toff+1)], dtype=object),
                    'lambda_start_cost': np.array([_get_pi(model, f'start_cost_{t}')
                                                   for t in range(1, self.T)]),
                    'lambda_shut_cost':  np.array([_get_pi(model, f'shut_cost_{t}')
                                                   for t in range(1, self.T)]),
                    'lambda_coc_nonneg': np.zeros(self.T - 1),
                    'lambda_x_upper':    np.zeros(self.T),
                    'lambda_x_lower':    np.zeros(self.T),
                }
    
    def iter_with_primal_block(
        self,
        sample_id: int,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        deltas: np.ndarray,
        costs: np.ndarray = None,
        pg_costs: np.ndarray = None,
    ):
        """
        BCD迭代：原始块 - V3三时段耦合约束版本
        固定代理约束参数(alphas, betas, gammas, deltas)和对偶变量(mu)，更新原始变量(pg, x)

        时序耦合约束形式（T-2个）:
            alpha_t * x_t + beta_t * x_{t+1} + gamma_t * x_{t+2} <= delta_t  (t = 0..T-3)

        目标函数（参考BCD软约束形式）:
            min  rho_primal * Σ_{all} max(0, violation)   [原问题约束 + 耦合约束]
                 + rho_opt    * Σ_{all} |violation| * dual  [互补松弛]
                 + obj_binary

        原问题约束均以软约束形式处理（violation变量），与BCD一致。
        lambda_inherent 由 _initialize_solve 在初始化阶段从单机组LP提取，保证非None。
        c_x / c_pg 均不入原始块目标；相关调整仅通过 dual block / 驻点损失传递。
        """
        g = self.unit_id
        mu_vals = np.abs(self.mu[sample_id])   # (num_coupling_constraints,)
        lam_inh = self.lambda_inherent[sample_id]  # dict，由 _initialize_solve 保证非None

        alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
            alphas,
            betas,
            gammas,
            deltas,
        )

        Pmin    = self.gen[g, PMIN]
        Pmax    = self.gen[g, PMAX]
        a       = self.gencost[g, -2] / self.T_delta   # 线性发电成本系数
        b       = self.gencost[g, -1] / self.T_delta   # 无负荷成本系数
        Ru      = float(self.Ru_all[g])
        Rd      = float(self.Rd_all[g])
        Ru_co   = float(self.Ru_co_all[g])
        Rd_co   = float(self.Rd_co_all[g])
        sc      = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shc     = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton     = min(4, self.T)
        Toff    = min(4, self.T)

        model = gp.Model('primal_block_temporal')
        model.Params.OutputFlag = 0

        # --- 决策变量 ---
        pg     = model.addVars(self.T,   lb=0, name='pg')
        x      = model.addVars(self.T,   lb=0, ub=1, name='x')
        coc    = model.addVars(self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.T,   lb=0, name='cpower')

        x_true = self.active_set_data[sample_id].get('x_true', None)
        if x_true is None:
            x_true = self.x[sample_id]
        x_binary_dev = model.addVars(self.T, lb=0, name='x_binary_dev')

        # 原问题约束违反量/绝对值辅助变量（软约束形式）
        pg_lower_viol   = model.addVars(self.T,   lb=0, name='pg_lower_viol')
        pg_upper_viol   = model.addVars(self.T,   lb=0, name='pg_upper_viol')
        pg_lower_abs    = model.addVars(self.T,   lb=0, name='pg_lower_abs')
        pg_upper_abs    = model.addVars(self.T,   lb=0, name='pg_upper_abs')
        ramp_up_viol    = model.addVars(self.T-1, lb=0, name='ramp_up_viol')
        ramp_down_viol  = model.addVars(self.T-1, lb=0, name='ramp_down_viol')
        ramp_up_abs     = model.addVars(self.T-1, lb=0, name='ramp_up_abs')
        ramp_down_abs   = model.addVars(self.T-1, lb=0, name='ramp_down_abs')
        min_on_viol     = model.addVars(Ton,  self.T, lb=0, name='min_on_viol')
        min_off_viol    = model.addVars(Toff, self.T, lb=0, name='min_off_viol')
        min_on_abs      = model.addVars(Ton,  self.T, lb=0, name='min_on_abs')
        min_off_abs     = model.addVars(Toff, self.T, lb=0, name='min_off_abs')
        start_cost_viol = model.addVars(self.T-1, lb=0, name='start_cost_viol')
        shut_cost_viol  = model.addVars(self.T-1, lb=0, name='shut_cost_viol')
        start_cost_abs  = model.addVars(self.T-1, lb=0, name='start_cost_abs')
        shut_cost_abs   = model.addVars(self.T-1, lb=0, name='shut_cost_abs')

        # 代理耦合约束违反量/绝对值
        surrogate_viols    = model.addVars(self.num_coupling_constraints, lb=0, name='surrogate_viol')
        surrogate_abs_vals = model.addVars(self.num_coupling_constraints, lb=0, name='surrogate_abs')

        obj_primal = gp.LinExpr()
        obj_opt    = gp.LinExpr()
        obj_binary = gp.LinExpr()

        # --- x 偏差（硬约束，定义性）---
        for t in range(self.T):
            model.addConstr(x_binary_dev[t] >= x[t] - x_true[t], name=f'x_binary_dev_pos_{t}')
            model.addConstr(x_binary_dev[t] >= x_true[t] - x[t], name=f'x_binary_dev_neg_{t}')
            obj_binary += x_binary_dev[t]

        # --- 发电上下限（软约束）---
        for t in range(self.T):
            pg_lower_expr = Pmin * x[t] - pg[t]
            model.addConstr(pg_lower_viol[t] >= pg_lower_expr, name=f'pg_lower_viol_{t}')
            model.addConstr(pg_lower_abs[t]  >= pg_lower_expr, name=f'pg_lower_abs1_{t}')
            model.addConstr(pg_lower_abs[t]  >= -pg_lower_expr, name=f'pg_lower_abs2_{t}')
            obj_primal += pg_lower_viol[t]

            pg_upper_expr = pg[t] - Pmax * x[t]
            model.addConstr(pg_upper_viol[t] >= pg_upper_expr, name=f'pg_upper_viol_{t}')
            model.addConstr(pg_upper_abs[t]  >= pg_upper_expr, name=f'pg_upper_abs1_{t}')
            model.addConstr(pg_upper_abs[t]  >= -pg_upper_expr, name=f'pg_upper_abs2_{t}')
            obj_primal += pg_upper_viol[t]

            obj_opt += pg_lower_abs[t] * abs(float(lam_inh['lambda_pg_lower'][t]))
            obj_opt += pg_upper_abs[t] * abs(float(lam_inh['lambda_pg_upper'][t]))
            obj_opt += x[t]       * abs(float(lam_inh['lambda_x_lower'][t]))
            obj_opt += (1 - x[t]) * abs(float(lam_inh['lambda_x_upper'][t]))

        # --- 爬坡约束（软约束，Ru_co=0.3*Pmax 与 dual block 一致）---
        for t in range(1, self.T):
            ramp_up_expr = pg[t] - pg[t-1] - Ru * x[t-1] - Ru_co * (1 - x[t-1])
            model.addConstr(ramp_up_viol[t-1] >= ramp_up_expr, name=f'ramp_up_viol_{t}')
            model.addConstr(ramp_up_abs[t-1]  >= ramp_up_expr, name=f'ramp_up_abs1_{t}')
            model.addConstr(ramp_up_abs[t-1]  >= -ramp_up_expr, name=f'ramp_up_abs2_{t}')
            obj_primal += ramp_up_viol[t-1]
            obj_opt += ramp_up_abs[t-1]   * abs(float(lam_inh['lambda_ramp_up'][t-1]))

            ramp_down_expr = pg[t-1] - pg[t] - Rd * x[t] - Rd_co * (1 - x[t])
            model.addConstr(ramp_down_viol[t-1] >= ramp_down_expr, name=f'ramp_down_viol_{t}')
            model.addConstr(ramp_down_abs[t-1]  >= ramp_down_expr, name=f'ramp_down_abs1_{t}')
            model.addConstr(ramp_down_abs[t-1]  >= -ramp_down_expr, name=f'ramp_down_abs2_{t}')
            obj_primal += ramp_down_viol[t-1]
            obj_opt += ramp_down_abs[t-1] * abs(float(lam_inh['lambda_ramp_down'][t-1]))

        # --- 最小开关机时间（软约束）---
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                min_on_expr = x[t1+1] - x[t1] - x[t1+tau]
                model.addConstr(min_on_viol[tau-1, t1] >= min_on_expr, name=f'min_on_viol_{tau}_{t1}')
                model.addConstr(min_on_abs[tau-1, t1]  >= min_on_expr, name=f'min_on_abs1_{tau}_{t1}')
                model.addConstr(min_on_abs[tau-1, t1]  >= -min_on_expr, name=f'min_on_abs2_{tau}_{t1}')
                obj_primal += min_on_viol[tau-1, t1]
                obj_opt += min_on_abs[tau-1, t1] * abs(float(lam_inh['lambda_min_on'][tau-1][t1]))

        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                min_off_expr = -x[t1+1] + x[t1] - (1 - x[t1+tau])
                model.addConstr(min_off_viol[tau-1, t1] >= min_off_expr, name=f'min_off_viol_{tau}_{t1}')
                model.addConstr(min_off_abs[tau-1, t1]  >= min_off_expr, name=f'min_off_abs1_{tau}_{t1}')
                model.addConstr(min_off_abs[tau-1, t1]  >= -min_off_expr, name=f'min_off_abs2_{tau}_{t1}')
                obj_primal += min_off_viol[tau-1, t1]
                obj_opt += min_off_abs[tau-1, t1] * abs(float(lam_inh['lambda_min_off'][tau-1][t1]))

        # --- 启停成本（软约束）---
        for t in range(1, self.T):
            start_expr = sc * (x[t] - x[t-1]) - coc[t-1]
            model.addConstr(start_cost_viol[t-1] >= start_expr, name=f'start_cost_viol_{t}')
            model.addConstr(start_cost_abs[t-1]  >= start_expr, name=f'start_cost_abs1_{t}')
            model.addConstr(start_cost_abs[t-1]  >= -start_expr, name=f'start_cost_abs2_{t}')
            obj_primal += start_cost_viol[t-1]
            obj_opt += start_cost_abs[t-1] * abs(float(lam_inh['lambda_start_cost'][t-1]))

            shut_expr = shc * (x[t-1] - x[t]) - coc[t-1]
            model.addConstr(shut_cost_viol[t-1] >= shut_expr, name=f'shut_cost_viol_{t}')
            model.addConstr(shut_cost_abs[t-1]  >= shut_expr, name=f'shut_cost_abs1_{t}')
            model.addConstr(shut_cost_abs[t-1]  >= -shut_expr, name=f'shut_cost_abs2_{t}')
            obj_primal += shut_cost_viol[t-1]
            obj_opt += shut_cost_abs[t-1]  * abs(float(lam_inh['lambda_shut_cost'][t-1]))
            obj_opt += coc[t-1]            * abs(float(lam_inh['lambda_coc_nonneg'][t-1]))

        # --- 发电成本定义（等式约束，lambda_cpower=1；成本不显式入目标，与BCD一致）---
        for t in range(self.T):
            model.addConstr(cpower[t] == a * pg[t] + b * x[t], name=f'cpower_{t}')

        # --- 代理耦合约束（软约束，按 sensitive_timesteps 索引）---
        sensitive_t = self.sensitive_timesteps[sample_id]
        constraint_offsets = self._constraint_offsets_for_sample(sample_id)
        for k, t in enumerate(sensitive_t):
            coupling_lhs = build_surrogate_constraint_expression(
                x,
                t,
                constraint_offsets[k],
                alphas[k],
                betas[k],
                gammas[k],
                self.T,
            )
            model.addConstr(surrogate_viols[k]    >= coupling_lhs - deltas[k], name=f'coupling_viol_{k}')
            model.addConstr(surrogate_abs_vals[k] >= coupling_lhs - deltas[k], name=f'coupling_abs_pos_{k}')
            model.addConstr(surrogate_abs_vals[k] >= deltas[k] - coupling_lhs, name=f'coupling_abs_neg_{k}')
        obj_primal += gp.quicksum(surrogate_viols[k]    for k in range(len(sensitive_t)))
        obj_opt    += gp.quicksum(surrogate_abs_vals[k] * mu_vals[k]
                                  for k in range(len(sensitive_t)))

        # --- 目标函数 ---
        obj_prox = self._build_primal_block_prox_obj(model, sample_id, pg, x, coc)
        model.setObjective(
            self.rho_primal * obj_primal
            + self.rho_opt  * obj_opt
            + self.rho_binary * obj_binary
            + self.pg_block_prox_weight * obj_prox,
            GRB.MINIMIZE,
        )
        model.optimize()

        if model.status == GRB.OPTIMAL:
            if sample_id <= 2:
                print(f"primal_block, sample_id: {sample_id}, "
                      f"obj_primal: {obj_primal.getValue():.4f}, "
                      f"obj_opt: {obj_opt.getValue():.4f}, "
                      f"obj_binary: {obj_binary.getValue():.4f}, "
                      f"obj_prox: {obj_prox.getValue() if hasattr(obj_prox, 'getValue') else 0.0:.4f}", flush=True)

            pg_sol     = np.array([pg[t].X     for t in range(self.T)])
            x_sol      = np.array([x[t].X      for t in range(self.T)])
            coc_sol    = np.array([coc[t].X    for t in range(self.T-1)])
            cpower_sol = np.array([cpower[t].X for t in range(self.T)])
            return pg_sol, x_sol, coc_sol, cpower_sol
        else:
            print(f"警告: 原始块求解失败，状态: {model.status}", flush=True)
            return None, None, None, None
    
    def iter_with_dual_block(
        self,
        sample_id: int,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        deltas: np.ndarray,
        costs: np.ndarray = None,
        pg_costs: np.ndarray = None,
    ):
        """
        BCD迭代：对偶块 - V3三时段耦合约束完整版本
        固定原始变量(pg, x, coc)和代理约束参数，联合更新所有对偶变量：
          - 固有约束对偶变量 (lambda_pg_lower/upper, lambda_ramp_up/down,
            lambda_min_on/off, lambda_start/shut_cost, lambda_coc_nonneg,
            lambda_x_upper/lower)
          - 代理耦合约束对偶变量 (mu)

        目标：
            min  rho_dual * obj_dual + rho_opt * obj_opt

        obj_dual = Σ KKT驻点条件违反量（对 pg, x, coc 变量）
        obj_opt  = Σ 约束违反量 * 对应对偶变量（互补松弛条件）

        Returns:
            lambda_inherent_sol: dict，固有约束对偶变量
            mu_sol: (num_coupling_constraints,) 代理耦合对偶变量
        """
        g = self.unit_id
        pg_val  = self.pg[sample_id]    # (T,)
        x_val   = self.x[sample_id]     # (T,)
        coc_val = self.coc[sample_id]   # (T-1,)
        lambda_val = self.lambda_vals[sample_id]  # (T,)  电价对偶变量（外部给定）

        # 机组参数
        a    = self.gencost[g, -2] / self.T_delta   # 线性发电成本系数
        b    = self.gencost[g, -1] / self.T_delta   # 无负荷成本系数
        Pmin = self.gen[g, PMIN]
        Pmax = self.gen[g, PMAX]
        Ru    = float(self.Ru_all[g])
        Rd    = float(self.Rd_all[g])
        Ru_co = float(self.Ru_co_all[g])
        Rd_co = float(self.Rd_co_all[g])
        start_cost = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shut_cost  = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton  = min(4, self.T)
        Toff = min(4, self.T)

        phase = self._get_mu_lower_bound_phase()
        lb = self._current_mu_lower_bound_value()
        sign_relax_round = self._is_mu_sign_relaxation_round()
        direction_signs = self._get_surrogate_direction_signs()
        alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
            alphas,
            betas,
            gammas,
            deltas,
        )

        model = gp.Model('dual_block_v3')
        model.Params.OutputFlag = 0
        model.Params.NumericFocus = 2
        model.Params.MIPGap = 1e-6

        # ===== 声明固有约束对偶变量 =====
        lam_pg_lower    = model.addVars(self.T,      lb=0, name='lam_pg_lower')
        lam_pg_upper    = model.addVars(self.T,      lb=0, name='lam_pg_upper')
        lam_ramp_up     = model.addVars(self.T - 1,  lb=0, name='lam_ramp_up')
        lam_ramp_down   = model.addVars(self.T - 1,  lb=0, name='lam_ramp_down')
        lam_start_cost  = model.addVars(self.T - 1,  lb=0, name='lam_start_cost')
        lam_shut_cost   = model.addVars(self.T - 1,  lb=0, name='lam_shut_cost')
        lam_coc_nonneg  = model.addVars(self.T - 1,  lb=0, name='lam_coc_nonneg')
        x_bound_dual_ub = 0.0 if self._force_zero_x_bound_duals() else GRB.INFINITY
        lam_x_upper     = model.addVars(self.T,      lb=0, ub=x_bound_dual_ub, name='lam_x_upper')
        lam_x_lower     = model.addVars(self.T,      lb=0, ub=x_bound_dual_ub, name='lam_x_lower')

        # min_on / min_off：只为有效约束索引创建变量
        lam_min_on  = {}
        lam_min_off = {}
        for tau in range(1, Ton + 1):
            for t1 in range(self.T - tau):
                lam_min_on[tau - 1, t1]  = model.addVar(lb=0, name=f'lam_min_on_{tau-1}_{t1}')
        for tau in range(1, Toff + 1):
            for t1 in range(self.T - tau):
                lam_min_off[tau - 1, t1] = model.addVar(lb=0, name=f'lam_min_off_{tau-1}_{t1}')

        # 代理耦合约束对偶变量
        mu_lb = -GRB.INFINITY if (phase != "none" and sign_relax_round) else 0.0
        mu = model.addVars(self.num_coupling_constraints, lb=mu_lb, name='mu')
        mu_abs = model.addVars(self.num_coupling_constraints, lb=0, name='mu_abs')
        for k in range(self.num_coupling_constraints):
            model.addConstr(mu_abs[k] >= mu[k], name=f'mu_abs_pos_{k}')
            model.addConstr(mu_abs[k] >= -mu[k], name=f'mu_abs_neg_{k}')
            if phase == "individual" and lb > 0:
                if sign_relax_round:
                    model.addConstr(mu_abs[k] >= lb, name=f'mu_abs_lb_{k}')
                else:
                    model.addConstr(mu[k] >= lb, name=f'mu_lb_{k}')
        if phase == "group" and lb > 0 and self._uses_group_mu_lower_bound():
            for group_idx in range(self.num_coupling_constraints // self.all_mode_group_size):
                group_start = group_idx * self.all_mode_group_size
                group_stop = group_start + self.all_mode_group_size
                model.addConstr(
                    gp.quicksum((mu_abs[k] if sign_relax_round else mu[k]) for k in range(group_start, group_stop)) >= lb,
                    name=f'mu_group_lb_{group_idx}',
                )

        # lambda_cpower 由驻点条件固定为 1，不需要作为变量

        obj_dual_pg = 0
        obj_dual_x = 0
        obj_dual_coc = 0
        obj_opt  = 0

        # ===== obj_dual：KKT 驻点条件 =====

        # -- pg[t] 驻点：  a + c_pg[t] - lambda[t] - lam_pg_lower[t] + lam_pg_upper[t] + ramp_terms = 0
        for t in range(self.T):
            expr = a + (pg_costs[t] if pg_costs is not None else 0) - lambda_val[t]
            expr -= lam_pg_lower[t]
            expr += lam_pg_upper[t]
            if t > 0:
                expr += lam_ramp_up[t - 1]
                expr -= lam_ramp_down[t - 1]
            if t < self.T - 1:
                expr -= lam_ramp_up[t]
                expr += lam_ramp_down[t]
            abs_v = model.addVar(lb=0, name=f'abs_pg_{t}')
            model.addConstr(abs_v >= expr,  name=f'abs_pg_pos_{t}')
            model.addConstr(abs_v >= -expr, name=f'abs_pg_neg_{t}')
            obj_dual_pg += abs_v

        # -- x[t] 驻点：  b + Pmin*lam_pg_lower[t] - Pmax*lam_pg_upper[t]
        #                  + ramp_co_terms + min_on/off_terms + start/shut_terms
        #                  + coupling_surrogate_terms + lam_x_upper[t] - lam_x_lower[t] = 0
        
        for t in range(self.T):
            expr = b + (costs[t] if costs is not None else 0)
            expr += Pmin * lam_pg_lower[t]
            expr -= Pmax * lam_pg_upper[t]

            # 爬坡约束对x[t]的贡献（x[t]作为t时段的"上一时刻"）
            if t < self.T - 1:
                expr += (Ru_co - Ru) * lam_ramp_up[t]
            if t > 0:
                expr += (Rd_co - Rd) * lam_ramp_down[t - 1]

            # 最小开机时间约束
            for tau in range(1, Ton + 1):
                for t1 in range(self.T - tau):
                    k = lam_min_on[tau - 1, t1]
                    if t == t1 + 1:
                        expr += k
                    if t == t1:
                        expr -= k
                    if t == t1 + tau:
                        expr -= k

            # 最小停机时间约束
            for tau in range(1, Toff + 1):
                for t1 in range(self.T - tau):
                    k = lam_min_off[tau - 1, t1]
                    if t == t1 + 1:
                        expr -= k
                    if t == t1:
                        expr += k
                    if t == t1 + tau:
                        expr += k

            # 启停成本约束
            if t > 0:
                expr += start_cost * lam_start_cost[t - 1]
                expr -= shut_cost  * lam_shut_cost[t - 1]
            if t < self.T - 1:
                expr -= start_cost * lam_start_cost[t]
                expr += shut_cost  * lam_shut_cost[t]

            # 代理耦合约束对 x[t] 的贡献（按 sensitive_timesteps 索引）
            sensitive_t = self.sensitive_timesteps[sample_id]
            constraint_offsets = self._constraint_offsets_for_sample(sample_id)
            for k, ts in enumerate(sensitive_t):
                for time_idx, coeff in iterate_surrogate_constraint_terms(
                    ts,
                    constraint_offsets[k],
                    alphas[k],
                    betas[k],
                    gammas[k],
                    self.T,
                ):
                    if time_idx == t:
                        expr += coeff * mu[k]

            # x 变量界约束（x ∈ [0,1]）
            expr += lam_x_upper[t] - lam_x_lower[t]

            abs_v = model.addVar(lb=0, name=f'abs_x_{t}')
            model.addConstr(abs_v >= expr,  name=f'abs_x_pos_{t}')
            model.addConstr(abs_v >= -expr, name=f'abs_x_neg_{t}')
            obj_dual_x += abs_v

        # -- coc[t] 驻点：  1 - lam_start_cost[t] - lam_shut_cost[t] - lam_coc_nonneg[t] = 0
        for t in range(self.T - 1):
            expr = 1 - lam_start_cost[t] - lam_shut_cost[t] - lam_coc_nonneg[t]
            abs_v = model.addVar(lb=0, name=f'abs_coc_{t}')
            model.addConstr(abs_v >= expr,  name=f'abs_coc_pos_{t}')
            model.addConstr(abs_v >= -expr, name=f'abs_coc_neg_{t}')
            obj_dual_coc += abs_v

        obj_dual = obj_dual_pg + obj_dual_x + obj_dual_coc

        # ===== obj_opt：互补松弛条件（约束违反量 × 对偶变量）=====

        # pg_lower: pg[t] >= Pmin * x[t]
        for t in range(self.T):
            viol = abs(pg_val[t] - Pmin * x_val[t])
            if viol > 1e-10:
                obj_opt += viol * lam_pg_lower[t]

        # pg_upper: pg[t] <= Pmax * x[t]
        for t in range(self.T):
            viol = abs(Pmax * x_val[t] - pg_val[t])
            if viol > 1e-10:
                obj_opt += viol * lam_pg_upper[t]

        # ramp_up: pg[t] - pg[t-1] <= Ru*x[t-1] + Ru_co*(1-x[t-1])
        for t in range(1, self.T):
            limit = Ru * x_val[t - 1] + Ru_co * (1 - x_val[t - 1])
            viol  = abs(pg_val[t] - pg_val[t - 1] - limit)
            if viol > 1e-10:
                obj_opt += viol * lam_ramp_up[t - 1]

        # ramp_down: pg[t-1] - pg[t] <= Rd*x[t] + Rd_co*(1-x[t])
        for t in range(1, self.T):
            limit = Rd * x_val[t] + Rd_co * (1 - x_val[t])
            viol  = abs(pg_val[t - 1] - pg_val[t] - limit)
            if viol > 1e-10:
                obj_opt += viol * lam_ramp_down[t - 1]

        # min_on: x[t1+1] - x[t1] - x[t1+tau] <= 0
        for tau in range(1, Ton + 1):
            for t1 in range(self.T - tau):
                viol = abs(x_val[t1 + 1] - x_val[t1] - x_val[t1 + tau])
                if viol > 1e-10:
                    obj_opt += viol * lam_min_on[tau - 1, t1]

        # min_off: -x[t1+1] + x[t1] - 1 + x[t1+tau] <= 0
        for tau in range(1, Toff + 1):
            for t1 in range(self.T - tau):
                viol = abs(-x_val[t1 + 1] + x_val[t1] - 1 + x_val[t1 + tau])
                if viol > 1e-10:
                    obj_opt += viol * lam_min_off[tau - 1, t1]

        # start_cost: coc[t] >= start_cost*(x[t+1] - x[t])
        for t in range(self.T - 1):
            viol = abs(coc_val[t] - start_cost * (x_val[t + 1] - x_val[t]))
            if viol > 1e-10:
                obj_opt += viol * lam_start_cost[t]

        # shut_cost: coc[t] >= shut_cost*(x[t] - x[t+1])
        for t in range(self.T - 1):
            viol = abs(coc_val[t] - shut_cost * (x_val[t] - x_val[t + 1]))
            if viol > 1e-10:
                obj_opt += viol * lam_shut_cost[t]

        # coc_nonneg: coc[t] >= 0
        for t in range(self.T - 1):
            viol = abs(coc_val[t])
            if viol > 1e-10:
                obj_opt += viol * lam_coc_nonneg[t]

        # x_lower: x[t] >= 0
        for t in range(self.T):
            viol = abs(x_val[t])
            if viol > 1e-10:
                obj_opt += viol * lam_x_lower[t]

        # x_upper: x[t] <= 1
        for t in range(self.T):
            viol = abs(x_val[t] - 1)
            if viol > 1e-10:
                obj_opt += viol * lam_x_upper[t]

        # 代理耦合约束（按 sensitive_timesteps 索引）
        constraint_offsets = self._constraint_offsets_for_sample(sample_id)
        for k, t in enumerate(self.sensitive_timesteps[sample_id]):
            lhs = build_surrogate_constraint_expression(
                x_val,
                t,
                constraint_offsets[k],
                alphas[k],
                betas[k],
                gammas[k],
                self.T,
            )
            viol = abs(lhs - deltas[k])
            if viol > 1e-10:
                obj_opt += viol * mu_abs[k]

        # ===== 设置目标函数并求解 =====
        obj_dual_prox = self._build_dual_block_prox_obj(
            model,
            sample_id,
            lam_pg_lower,
            lam_pg_upper,
            lam_ramp_up,
            lam_ramp_down,
            lam_start_cost,
            lam_shut_cost,
            lam_coc_nonneg,
            lam_x_upper,
            lam_x_lower,
            lam_min_on,
            lam_min_off,
            mu,
            Ton,
            Toff,
        )
        model.setObjective(
            self.rho_dual_pg * obj_dual_pg
            + self.rho_dual_x * obj_dual_x
            + self.rho_dual_coc * obj_dual_coc
            + self.rho_opt * obj_opt
            + self.dual_block_prox_weight * obj_dual_prox,
            GRB.MINIMIZE
        )
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # 提取固有约束对偶变量
            lambda_inherent_sol = {
                'lambda_pg_lower':   np.array([lam_pg_lower[t].X   for t in range(self.T)]),
                'lambda_pg_upper':   np.array([lam_pg_upper[t].X   for t in range(self.T)]),
                'lambda_ramp_up':    np.array([lam_ramp_up[t].X    for t in range(self.T - 1)]),
                'lambda_ramp_down':  np.array([lam_ramp_down[t].X  for t in range(self.T - 1)]),
                'lambda_min_on':     np.array([[lam_min_on[tau - 1, t1].X
                                                for t1 in range(self.T - tau)]
                                               for tau in range(1, Ton + 1)], dtype=object),
                'lambda_min_off':    np.array([[lam_min_off[tau - 1, t1].X
                                                for t1 in range(self.T - tau)]
                                               for tau in range(1, Toff + 1)], dtype=object),
                'lambda_start_cost': np.array([lam_start_cost[t].X for t in range(self.T - 1)]),
                'lambda_shut_cost':  np.array([lam_shut_cost[t].X  for t in range(self.T - 1)]),
                'lambda_coc_nonneg': np.array([lam_coc_nonneg[t].X for t in range(self.T - 1)]),
                'lambda_x_upper':    np.array([lam_x_upper[t].X    for t in range(self.T)]),
                'lambda_x_lower':    np.array([lam_x_lower[t].X    for t in range(self.T)]),
            }
            mu_sol = np.array([mu[k].X for k in range(self.num_coupling_constraints)])

            if sample_id <= 2:
                print(f"  dual_block sample={sample_id}: "
                      f"obj_dual_pg={obj_dual_pg.getValue():.4f}, "
                      f"obj_dual_x={obj_dual_x.getValue():.4f}, "
                      f"obj_dual_coc={obj_dual_coc.getValue():.4f}, "
                      f"obj_dual={obj_dual.getValue():.4f}, "
                      f"obj_opt={obj_opt.getValue() if hasattr(obj_opt, 'getValue') else obj_opt:.4f}, "
                      f"obj_dual_prox={obj_dual_prox.getValue() if hasattr(obj_dual_prox, 'getValue') else 0.0:.4f}",
                      flush=True)

            return lambda_inherent_sol, mu_sol
        else:
            print(f"警告: 对偶块求解失败 sample={sample_id}，状态: {model.status}", flush=True)
            return None, None
    
    def _extract_features(self, sample_id: int) -> np.ndarray:
        """提取特征: [Pd, λ, unit_params]"""
        pd_flat = get_feature_vector_from_sample(dict(self.active_set_data[sample_id]))
        lambda_val = self.lambda_vals[sample_id]

        # 机组静态参数（归一化，训练和推理都一致可用）
        g = self.unit_id
        Pmax = self.gen[g, PMAX]
        unit_params = np.array([
            self.gen[g, PMIN] / (Pmax + 1e-8),                    # Pmin/Pmax ratio
            self.gencost[g, -2] / self.T_delta,                    # 边际成本 a
            self.gencost[g, -1] / self.T_delta / (Pmax + 1e-8),   # 归一化无负荷成本 b
            float(self.Ru_all[g]) / (Pmax + 1e-8),                # 归一化爬坡率 Ru
            0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1] / (Pmax + 1e-8),
            0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2] / (Pmax + 1e-8),
        ])

        return np.concatenate([pd_flat, lambda_val, unit_params])
    
    def loss_function_differentiable(self, sample_id: int, alphas_tensor: torch.Tensor,
                                     betas_tensor: torch.Tensor, gammas_tensor: torch.Tensor,
                                     deltas_tensor: torch.Tensor, costs_tensor: torch.Tensor,
                                     device,
                                     return_components: bool = False) -> torch.Tensor:
        """
        主代理网络 loss。

        这里只训练 alpha/beta/gamma/delta/c_x，显式剥离 c_pg 对应的驻点项。
        c_pg 使用独立训练器和独立 loss。
        """
        g = self.unit_id
        
        # 从BCD迭代得到的变量
        x_val   = torch.tensor(self.x[sample_id],   dtype=torch.float32, device=device)  # (T,)
        mu_vals = torch.tensor(self.mu[sample_id],  dtype=torch.float32, device=device)  # (num_coupling_constraints,)
        mu_abs_vals = torch.abs(mu_vals)
        lambda_val = torch.tensor(self.lambda_vals[sample_id], dtype=torch.float32, device=device)
        lam_inh = self.lambda_inherent[sample_id]  # dict or None
        direction_signs = torch.tensor(
            self._get_surrogate_direction_signs(len(alphas_tensor)),
            dtype=torch.float32,
            device=device,
        )
        signed_alphas = alphas_tensor * direction_signs
        signed_betas = betas_tensor * direction_signs
        signed_gammas = gammas_tensor * direction_signs
        signed_deltas = deltas_tensor * direction_signs
        
        # ========== 计算obj_primal ==========
        # V3三时段约束违反量（按 sensitive_timesteps 索引）
        obj_primal = torch.tensor(0.0, device=device, requires_grad=True)
        sensitive_t = self.sensitive_timesteps[sample_id]
        constraint_offsets = self._constraint_offsets_for_sample(sample_id)
        for k, t in enumerate(sensitive_t):
            coupling_lhs = build_surrogate_constraint_expression(
                x_val,
                t,
                constraint_offsets[k],
                signed_alphas[k],
                signed_betas[k],
                signed_gammas[k],
                self.T,
            )
            coupling_viol = self._smooth_relu(coupling_lhs - signed_deltas[k])
            obj_primal = obj_primal + coupling_viol

        # ========== 计算obj_opt ==========
        # V3互补松弛（按 sensitive_timesteps 索引）
        obj_opt = torch.tensor(0.0, device=device, requires_grad=True)
        for k, t in enumerate(sensitive_t):
            coupling_lhs = build_surrogate_constraint_expression(
                x_val,
                t,
                constraint_offsets[k],
                signed_alphas[k],
                signed_betas[k],
                signed_gammas[k],
                self.T,
            )
            coupling_abs = self._smooth_abs(
                coupling_lhs - signed_deltas[k],
                self.nn_smooth_abs_eps,
            )
            obj_opt = obj_opt + coupling_abs * mu_abs_vals[k]
        
        # ========== 计算obj_dual_x ==========
        # x[t]驻点：b + Pmin*lam_pg_lower[t] - Pmax*lam_pg_upper[t]
        #           + ramp_co_terms + min_on/off_terms + start/shut_terms
        #           + coupling_terms(alpha,beta,gamma,mu) + lam_x_upper[t] - lam_x_lower[t] = 0
        #
        # 固有项（常数，来自dual block存储的lambda_inherent）
        # 代理耦合项（含alpha,beta,gamma张量，提供NN梯度）
        g = self.unit_id
        b_val   = float(self.gencost[g, -1] / self.T_delta)
        Pmin_v  = float(self.gen[g, PMIN])
        Pmax_v  = float(self.gen[g, PMAX])
        Ru_v    = float(self.Ru_all[g])
        Rd_v    = float(self.Rd_all[g])
        Ru_co_v = float(self.Ru_co_all[g])
        Rd_co_v = float(self.Rd_co_all[g])
        start_c = 0.0 if self.ignore_startup_shutdown_costs else float(self.gencost[g, 1])
        shut_c  = 0.0 if self.ignore_startup_shutdown_costs else float(self.gencost[g, 2])
        Ton_l   = min(4, self.T)
        Toff_l  = min(4, self.T)
        obj_dual_x = torch.tensor(0.0, device=device, requires_grad=True)
        for t in range(self.T):
            # 固有约束贡献（常数部分）
            inherent_const = b_val
            if lam_inh is not None:
                lam_pgl = float(lam_inh['lambda_pg_lower'][t])
                lam_pgu = float(lam_inh['lambda_pg_upper'][t])
                inherent_const += Pmin_v * lam_pgl - Pmax_v * lam_pgu

                lam_ru = lam_inh['lambda_ramp_up']    # (T-1,)
                lam_rd = lam_inh['lambda_ramp_down']  # (T-1,)
                if t < self.T - 1:
                    inherent_const += (Ru_co_v - Ru_v) * float(lam_ru[t])
                if t > 0:
                    inherent_const += (Rd_co_v - Rd_v) * float(lam_rd[t - 1])

                lam_mon  = lam_inh['lambda_min_on']   # ragged: list indexed [tau_idx][t1]
                lam_moff = lam_inh['lambda_min_off']
                for tau in range(1, Ton_l + 1):
                    tau_row = lam_mon[tau - 1]  # (T-tau,)
                    for t1 in range(self.T - tau):
                        k = float(tau_row[t1])
                        if t == t1 + 1:
                            inherent_const += k
                        if t == t1:
                            inherent_const -= k
                        if t == t1 + tau:
                            inherent_const -= k
                for tau in range(1, Toff_l + 1):
                    tau_row = lam_moff[tau - 1]
                    for t1 in range(self.T - tau):
                        k = float(tau_row[t1])
                        if t == t1 + 1:
                            inherent_const -= k
                        if t == t1:
                            inherent_const += k
                        if t == t1 + tau:
                            inherent_const += k

                lam_sc  = lam_inh['lambda_start_cost']  # (T-1,)
                lam_shc = lam_inh['lambda_shut_cost']   # (T-1,)
                if t > 0:
                    inherent_const += start_c * float(lam_sc[t - 1])
                    inherent_const -= shut_c  * float(lam_shc[t - 1])
                if t < self.T - 1:
                    inherent_const -= start_c * float(lam_sc[t])
                    inherent_const += shut_c  * float(lam_shc[t])

                lam_xu = lam_inh['lambda_x_upper']  # (T,)
                lam_xl = lam_inh['lambda_x_lower']  # (T,)
                inherent_const += float(lam_xu[t]) - float(lam_xl[t])

            # 代理耦合约束贡献（含NN参数，可微分；按 sensitive_timesteps 索引）
            dual_expr = torch.tensor(inherent_const, dtype=torch.float32, device=device) + costs_tensor[t]
            for k, ts in enumerate(sensitive_t):
                for time_idx, coeff in iterate_surrogate_constraint_terms(
                    ts,
                    constraint_offsets[k],
                    alphas_tensor[k],
                    betas_tensor[k],
                    gammas_tensor[k],
                    self.T,
                ):
                    if time_idx == t:
                        dual_expr = dual_expr + coeff * mu_vals[k]

            obj_dual_x = obj_dual_x + self._smooth_abs(
                dual_expr,
                self.nn_smooth_abs_eps,
            )

        # 死区正则：限制幅值失控，但不给模板附近/小范围波动施加默认回拉。
        reg_loss = self.reg_weight * (
            self._deadband_quadratic(alphas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(betas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(gammas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(costs_tensor, self.aux_cost_reg_deadband)
        )
        if self._uses_template_rhs_bases():
            delta_base_tensor = torch.tensor(
                self.template_rhs_base_vector[:deltas_tensor.shape[0]],
                dtype=deltas_tensor.dtype,
                device=deltas_tensor.device,
            )
            reg_loss = reg_loss + self.reg_weight * self._deadband_quadratic(
                deltas_tensor,
                self.template_rhs_reg_deadband,
                center=delta_base_tensor,
            )
        else:
            reg_loss = reg_loss + self.reg_weight * self._deadband_quadratic(
                deltas_tensor,
                self.coeff_reg_deadband,
            )

        # 迭代间差异正则：抑制相邻 BCD 轮次 NN 输出跳变（可通过 iter_delta_reg_weight 控制）
        if self.iter_delta_reg_weight > 0 and self._prev_alpha_values is not None:
            nc = int(self.num_coupling_constraints)
            prev_alphas = torch.tensor(
                self._prev_alpha_values[sample_id][:nc],
                dtype=alphas_tensor.dtype,
                device=alphas_tensor.device,
            )
            prev_betas = torch.tensor(
                self._prev_beta_values[sample_id][:nc],
                dtype=betas_tensor.dtype,
                device=betas_tensor.device,
            )
            prev_gammas = torch.tensor(
                self._prev_gamma_values[sample_id][:nc],
                dtype=gammas_tensor.dtype,
                device=gammas_tensor.device,
            )
            prev_deltas = torch.tensor(
                self._prev_delta_values[sample_id][:nc],
                dtype=deltas_tensor.dtype,
                device=deltas_tensor.device,
            )
            prev_costs = torch.tensor(
                self._prev_cost_values[sample_id][: self.T],
                dtype=costs_tensor.dtype,
                device=costs_tensor.device,
            )
            iter_delta = (
                self._iter_delta_regularization(alphas_tensor, prev_alphas, self.iter_delta_reg_deadband)
                + self._iter_delta_regularization(betas_tensor, prev_betas, self.iter_delta_reg_deadband)
                + self._iter_delta_regularization(gammas_tensor, prev_gammas, self.iter_delta_reg_deadband)
                + self._iter_delta_regularization(deltas_tensor, prev_deltas, self.iter_delta_reg_deadband)
                + self._iter_delta_regularization(costs_tensor, prev_costs, self.iter_delta_reg_deadband)
            )
            reg_loss = reg_loss + self.iter_delta_reg_weight * iter_delta

        # 总损失：三项BCD目标 + 正则化
        dual_x_term = (
            self.loss_ratio_dual_x * self.rho_dual_x * obj_dual_x
            if self._nn_dual_terms_active()
            else torch.zeros((), dtype=obj_dual_x.dtype, device=obj_dual_x.device)
        )
        primal_term = self.loss_ratio_primal * self.rho_primal * obj_primal
        opt_term = self.loss_ratio_opt * self.rho_opt * obj_opt
        reg_term = self.loss_ratio_reg * reg_loss
        loss = (
            primal_term
            + dual_x_term
            + opt_term
            + reg_term
        )

        if not return_components:
            return loss

        components = {
            'obj_primal': obj_primal.detach(),
            'obj_dual_x': obj_dual_x.detach(),
            'obj_opt': obj_opt.detach(),
            'primal_term': primal_term.detach(),
            'dual_x_term': dual_x_term.detach(),
            'opt_term': opt_term.detach(),
            'reg_term': reg_term.detach(),
        }
        return loss, components

    def loss_function_c_pg_differentiable(
        self,
        sample_id: int,
        pg_costs_tensor: torch.Tensor,
        device,
        return_components: bool = False,
    ) -> torch.Tensor:
        """c_pg 单独 loss，只优化 pg 驻点项和 c_pg 自身正则。"""
        g = self.unit_id
        lambda_val = torch.tensor(
            self.lambda_vals[sample_id],
            dtype=torch.float32,
            device=device,
        )
        lam_inh = self.lambda_inherent[sample_id]
        obj_dual_pg = torch.tensor(0.0, device=device, requires_grad=True)
        a_val = float(self.gencost[g, -2] / self.T_delta)

        if lam_inh is not None:
            lam_ru = lam_inh['lambda_ramp_up']
            lam_rd = lam_inh['lambda_ramp_down']
            for t in range(self.T):
                pg_const = a_val - float(lambda_val[t])
                pg_const -= float(lam_inh['lambda_pg_lower'][t])
                pg_const += float(lam_inh['lambda_pg_upper'][t])
                if t > 0:
                    pg_const += float(lam_ru[t - 1])
                    pg_const -= float(lam_rd[t - 1])
                if t < self.T - 1:
                    pg_const -= float(lam_ru[t])
                    pg_const += float(lam_rd[t])
                residual = torch.tensor(pg_const, dtype=torch.float32, device=device) + pg_costs_tensor[t]
                obj_dual_pg = obj_dual_pg + self._smooth_abs(
                    residual,
                    self.pg_cost_smooth_abs_eps,
                )

        reg_loss = self.reg_weight * self._deadband_quadratic(
            pg_costs_tensor,
            self.pg_cost_reg_deadband,
        )
        if self.iter_delta_reg_weight > 0 and self._prev_pg_cost_values is not None:
            prev_pg_costs = torch.tensor(
                self._prev_pg_cost_values[sample_id][: self.T],
                dtype=pg_costs_tensor.dtype,
                device=pg_costs_tensor.device,
            )
            reg_loss = reg_loss + self.iter_delta_reg_weight * self._iter_delta_regularization(
                pg_costs_tensor,
                prev_pg_costs,
                self.iter_delta_reg_deadband,
            )
        dual_pg_term = self.loss_ratio_dual_pg * self.rho_dual_pg * obj_dual_pg
        reg_term = self.loss_ratio_reg * reg_loss
        loss = dual_pg_term + reg_term
        if not return_components:
            return loss
        components = {
            'obj_dual_pg': obj_dual_pg.detach(),
            'dual_pg_term': dual_pg_term.detach(),
            'reg_term': reg_term.detach(),
        }
        return loss, components

    def cal_nn_logging_components(self) -> dict[str, float]:
        if not TORCH_AVAILABLE or self.n_samples <= 0:
            return {
                'obj_primal': 0.0,
                'obj_dual_pg': 0.0,
                'obj_dual_x': 0.0,
                'obj_opt': 0.0,
                'reg_main': 0.0,
                'reg_pg': 0.0,
            }

        totals = {
            'obj_primal': 0.0,
            'obj_dual_pg': 0.0,
            'obj_dual_x': 0.0,
            'obj_opt': 0.0,
            'reg_main': 0.0,
            'reg_pg': 0.0,
        }
        for sample_id in range(self.n_samples):
            alphas_tensor = torch.tensor(self.alpha_values[sample_id], dtype=torch.float32, device=self.device)
            betas_tensor = torch.tensor(self.beta_values[sample_id], dtype=torch.float32, device=self.device)
            gammas_tensor = torch.tensor(self.gamma_values[sample_id], dtype=torch.float32, device=self.device)
            deltas_tensor = torch.tensor(self.delta_values[sample_id], dtype=torch.float32, device=self.device)
            costs_tensor = torch.tensor(self.cost_values[sample_id], dtype=torch.float32, device=self.device)
            pg_costs_tensor = torch.tensor(self.pg_cost_values[sample_id], dtype=torch.float32, device=self.device)

            _, main_components = self.loss_function_differentiable(
                sample_id,
                alphas_tensor,
                betas_tensor,
                gammas_tensor,
                deltas_tensor,
                costs_tensor,
                self.device,
                return_components=True,
            )
            _, pg_components = self.loss_function_c_pg_differentiable(
                sample_id,
                pg_costs_tensor,
                self.device,
                return_components=True,
            )
            totals['obj_primal'] += float(main_components['obj_primal'].cpu().item())
            totals['obj_dual_x'] += float(main_components['obj_dual_x'].cpu().item())
            totals['obj_opt'] += float(main_components['obj_opt'].cpu().item())
            totals['reg_main'] += float(main_components['reg_term'].cpu().item())
            totals['obj_dual_pg'] += float(pg_components['obj_dual_pg'].cpu().item())
            totals['reg_pg'] += float(pg_components['reg_term'].cpu().item())
        return totals
    
    def iter_with_surrogate_nn(
        self,
        num_epochs: int = 10,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
        cost_learning_rate: float | None = None,
    ):
        """
        BCD迭代：主代理网络训练。
        这里只更新 alpha/beta/gamma/delta/c_x，不包含 c_pg。
        """
        if not TORCH_AVAILABLE:
            return

        _, resolved_batch_size, resolved_shuffle = self._resolve_nn_batch_config(
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            shuffle=shuffle,
        )
        resolved_learning_rate = (
            self.nn_learning_rate if learning_rate is None else float(learning_rate)
        )
        resolved_cost_learning_rate = (
            self.cost_learning_rate if cost_learning_rate is None else float(cost_learning_rate)
        )
        if resolved_learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {resolved_learning_rate}")
        if resolved_cost_learning_rate <= 0:
            raise ValueError(
                f"cost_learning_rate must be positive, got {resolved_cost_learning_rate}"
            )

        # 前N轮BCD重建optimizer（适应剧烈变化），之后保持动量
        optimizer_persist_after = 5
        rebuild = (self.iter_number < optimizer_persist_after)
        current_main_lr = None
        if hasattr(self, '_surr_optimizer') and self._surr_optimizer is not None:
            current_main_lr = self._surr_optimizer.param_groups[0].get('lr')
        current_cost_lr = None
        if hasattr(self, '_surr_cost_optimizer') and self._surr_cost_optimizer is not None:
            current_cost_lr = self._surr_cost_optimizer.param_groups[0].get('lr')
        if (
            rebuild
            or not hasattr(self, '_surr_optimizer')
            or self._surr_optimizer is None
            or not hasattr(self, '_surr_cost_optimizer')
            or self._surr_cost_optimizer is None
            or current_main_lr != resolved_learning_rate
            or current_cost_lr != resolved_cost_learning_rate
        ):
            # 分离 x/pg 辅助成本头参数，主优化器不管理这些低学习率头
            aux_params = set(self.surrogate_net.cost_net.parameters()) | set(self.surrogate_net.pg_cost_net.parameters())
            main_params = [p for p in self.surrogate_net.parameters() if p not in aux_params]
            self._surr_optimizer = optim.Adam(
                main_params, lr=resolved_learning_rate, weight_decay=1e-4)
            self._surr_cost_optimizer = optim.Adam(
                self.surrogate_net.cost_net.parameters(),
                lr=resolved_cost_learning_rate,
                weight_decay=1e-4,
            )
            self._surr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._surr_optimizer, T_0=max(num_epochs, 1), T_mult=1)

        self.surrogate_net.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self._surr_optimizer.zero_grad()
            self._surr_cost_optimizer.zero_grad()
            batch_count = 0
            sample_indices = np.arange(self.n_samples, dtype=int)
            if resolved_shuffle and self.n_samples > 1:
                np.random.shuffle(sample_indices)

            for sample_pos, sample_id in enumerate(sample_indices):
                # 计算当前 batch 的实际大小（最后一个 batch 可能不满）
                batch_start = (sample_pos // resolved_batch_size) * resolved_batch_size
                actual_batch_size = min(resolved_batch_size, self.n_samples - batch_start)

                # 提取特征: [Pd, λ]
                features = self._extract_features(sample_id)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

                # V3前向传播：输出 (alphas, betas, gammas, deltas, c_x, c_pg)
                alphas_out, betas_out, gammas_out, deltas_out, costs_out = self.surrogate_net.forward_main(features_tensor)
                nc = self.num_coupling_constraints
                alphas_tensor = alphas_out.squeeze(0)[:nc]   # (num_coupling_constraints,)
                betas_tensor = betas_out.squeeze(0)[:nc]
                gammas_tensor = gammas_out.squeeze(0)[:nc]
                deltas_tensor = self._postprocess_delta_tensor(
                    deltas_out.squeeze(0)[:nc]
                )
                costs_tensor = costs_out.squeeze(0)[:self.T]  # (T,)

                # 主 loss 不再包含 c_pg 对应项
                loss = self.loss_function_differentiable(
                    sample_id, alphas_tensor, betas_tensor, gammas_tensor, deltas_tensor,
                    costs_tensor, self.device
                )
                (loss / actual_batch_size).backward()
                epoch_loss += loss.detach().cpu().item()
                batch_count += 1

                # batch 满或 epoch 结束：clip + step
                if batch_count == resolved_batch_size or sample_pos == self.n_samples - 1:
                    torch.nn.utils.clip_grad_norm_(self.surrogate_net.parameters(), max_norm=1.0)
                    self._surr_optimizer.step()
                    self._surr_cost_optimizer.step()
                    self._surr_optimizer.zero_grad()
                    self._surr_cost_optimizer.zero_grad()
                    batch_count = 0

            self._surr_scheduler.step()

            if epoch == 0 or epoch == num_epochs - 1:
                print(f"  [NN-main] epoch {epoch+1}/{num_epochs}, avg_loss = {epoch_loss/self.n_samples:.6f}", flush=True)

        # 记录最终 epoch loss 供 logger 使用
        if self.n_samples > 0:
            self._last_surr_nn_loss = epoch_loss / self.n_samples
            self._refresh_cached_surrogate_outputs()

    def iter_with_c_pg_nn(
        self,
        num_epochs: int = 10,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
    ):
        """
        BCD迭代：c_pg 单独训练器。
        仅更新 pg_cost_net，并使用独立 loss。
        """
        if not TORCH_AVAILABLE or not self._pg_costs_active():
            self._last_pg_cost_nn_loss = None
            return

        _, resolved_batch_size, resolved_shuffle = self._resolve_nn_batch_config(
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            shuffle=shuffle,
        )
        resolved_learning_rate = (
            self.pg_cost_surr_lr if learning_rate is None else float(learning_rate)
        )
        if resolved_learning_rate <= 0:
            raise ValueError(
                f"pg_cost_surr_learning_rate must be positive, got {resolved_learning_rate}"
            )

        optimizer_persist_after = 5
        rebuild = (self.iter_number < optimizer_persist_after)
        current_pg_cost_lr = None
        if hasattr(self, '_surr_pg_cost_optimizer') and self._surr_pg_cost_optimizer is not None:
            current_pg_cost_lr = self._surr_pg_cost_optimizer.param_groups[0].get('lr')
        if (
            rebuild
            or not hasattr(self, '_surr_pg_cost_optimizer')
            or self._surr_pg_cost_optimizer is None
            or current_pg_cost_lr != resolved_learning_rate
        ):
            self._surr_pg_cost_optimizer = optim.Adam(
                self.surrogate_net.pg_cost_net.parameters(),
                lr=resolved_learning_rate,
                weight_decay=1e-4,
            )
            self._surr_pg_cost_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._surr_pg_cost_optimizer, T_0=max(num_epochs, 1), T_mult=1)

        self.surrogate_net.train()
        self._set_c_pg_training_mode(True)
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                self._surr_pg_cost_optimizer.zero_grad()
                batch_count = 0
                sample_indices = np.arange(self.n_samples, dtype=int)
                if resolved_shuffle and self.n_samples > 1:
                    np.random.shuffle(sample_indices)

                for sample_pos, sample_id in enumerate(sample_indices):
                    batch_start = (sample_pos // resolved_batch_size) * resolved_batch_size
                    actual_batch_size = min(resolved_batch_size, self.n_samples - batch_start)

                    features = self._extract_features(sample_id)
                    features_tensor = torch.tensor(
                        features,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                    pg_costs_out = self.surrogate_net.forward_pg_cost(features_tensor)
                    pg_costs_tensor = self._gate_pg_cost_tensor(
                        pg_costs_out.squeeze(0)[:self.T]
                    )

                    loss = self.loss_function_c_pg_differentiable(
                        sample_id,
                        pg_costs_tensor,
                        self.device,
                    )
                    (loss / actual_batch_size).backward()
                    epoch_loss += loss.detach().cpu().item()
                    batch_count += 1

                    if batch_count == resolved_batch_size or sample_pos == self.n_samples - 1:
                        torch.nn.utils.clip_grad_norm_(
                            self.surrogate_net.pg_cost_net.parameters(),
                            max_norm=1.0,
                        )
                        self._surr_pg_cost_optimizer.step()
                        self._surr_pg_cost_optimizer.zero_grad()
                        batch_count = 0

                self._surr_pg_cost_scheduler.step()

                if epoch == 0 or epoch == num_epochs - 1:
                    print(
                        f"  [NN-c_pg] epoch {epoch+1}/{num_epochs}, avg_loss = {epoch_loss/self.n_samples:.6f}",
                        flush=True,
                    )

            if self.n_samples > 0:
                self._last_pg_cost_nn_loss = epoch_loss / self.n_samples
                self._refresh_cached_surrogate_outputs()
        finally:
            self._set_c_pg_training_mode(False)

    def cal_viol_components(self) -> Tuple[float, float, float, float, float, float]:
        """
        计算完整KKT违反量（与primal/dual block的目标函数完全对应）
          obj_primal: 所有约束（原问题+代理）的原始可行性违反
          obj_dual_pg: pg 驻点条件违反
          obj_dual_x:  x 驻点条件违反
          obj_dual_coc: coc 驻点条件违反
          obj_dual:   所有决策变量（pg, x, coc）的KKT驻点条件违反
          obj_opt:    所有约束-对偶变量对的互补松弛违反
        """
        obj_primal = 0.0
        obj_dual_pg = 0.0
        obj_dual_x = 0.0
        obj_dual_coc = 0.0
        obj_opt    = 0.0

        g = self.unit_id

        for sample_id in range(self.n_samples):
            x_val   = self.x[sample_id]
            pg_val  = self.pg[sample_id]
            coc_val = self.coc[sample_id]
            alphas  = self.alpha_values[sample_id]
            betas   = self.beta_values[sample_id]
            gammas  = self.gamma_values[sample_id]
            deltas  = self.delta_values[sample_id]
            mu_vals = np.abs(self.mu[sample_id])
            alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
                alphas,
                betas,
                gammas,
                deltas,
            )
            lam_inh = self.lambda_inherent[sample_id]
            lambda_val = self.lambda_vals[sample_id]   # 电价对偶变量

            a_v     = self.gencost[g, -2] / self.T_delta   # 线性发电成本系数
            b_v     = self.gencost[g, -1] / self.T_delta   # 无负荷成本系数
            Pmin_v  = float(self.gen[g, PMIN])
            Pmax_v  = float(self.gen[g, PMAX])
            Ru_v    = float(self.Ru_all[g])
            Rd_v    = float(self.Rd_all[g])
            Ru_co_v = float(self.Ru_co_all[g])
            Rd_co_v = float(self.Rd_co_all[g])
            sc_v    = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
            shc_v   = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
            Ton_v   = min(4, self.T)
            Toff_v  = min(4, self.T)

            # ================================================================
            # obj_primal：所有原始可行性违反
            # ================================================================

            # -- 代理耦合约束违反（按 sensitive_timesteps 索引）--
            sensitive_t = self.sensitive_timesteps[sample_id]
            constraint_offsets = self._constraint_offsets_for_sample(sample_id)
            for k, ts in enumerate(sensitive_t):
                coupling_lhs = build_surrogate_constraint_expression(
                    x_val,
                    ts,
                    constraint_offsets[k],
                    alphas[k],
                    betas[k],
                    gammas[k],
                    self.T,
                )
                obj_primal += max(0.0, coupling_lhs - deltas[k])

            # -- 原问题固有约束违反（与primal block的obj_primal对应）--
            for t in range(self.T):
                obj_primal += max(0.0, Pmin_v * x_val[t] - pg_val[t])   # pg >= Pmin*x
                obj_primal += max(0.0, pg_val[t] - Pmax_v * x_val[t])   # pg <= Pmax*x

            for t in range(1, self.T):
                obj_primal += max(0.0, pg_val[t] - pg_val[t-1]
                                       - Ru_v*x_val[t-1] - Ru_co_v*(1-x_val[t-1]))  # ramp_up
                obj_primal += max(0.0, pg_val[t-1] - pg_val[t]
                                       - Rd_v*x_val[t] - Rd_co_v*(1-x_val[t]))       # ramp_down
                obj_primal += max(0.0, sc_v  * (x_val[t] - x_val[t-1]) - coc_val[t-1])  # start_cost
                obj_primal += max(0.0, shc_v * (x_val[t-1] - x_val[t]) - coc_val[t-1])  # shut_cost

            for tau in range(1, Ton_v + 1):
                for t1 in range(self.T - tau):
                    obj_primal += max(0.0, x_val[t1+1] - x_val[t1] - x_val[t1+tau])   # min_on

            for tau in range(1, Toff_v + 1):
                for t1 in range(self.T - tau):
                    obj_primal += max(0.0, -x_val[t1+1] + x_val[t1] - (1-x_val[t1+tau]))  # min_off

            # ================================================================
            # obj_opt：所有互补松弛违反（与primal block的obj_opt对应）
            # ================================================================

            # -- 代理耦合约束互补松弛（按 sensitive_timesteps 索引）--
            for k, ts in enumerate(sensitive_t):
                coupling_lhs = build_surrogate_constraint_expression(
                    x_val,
                    ts,
                    constraint_offsets[k],
                    alphas[k],
                    betas[k],
                    gammas[k],
                    self.T,
                )
                obj_opt += abs(coupling_lhs - deltas[k]) * mu_vals[k]

            # -- 原问题固有约束互补松弛 --
            if lam_inh is not None:
                for t in range(self.T):
                    obj_opt += abs(Pmin_v*x_val[t] - pg_val[t])   * abs(float(lam_inh['lambda_pg_lower'][t]))
                    obj_opt += abs(pg_val[t] - Pmax_v*x_val[t])   * abs(float(lam_inh['lambda_pg_upper'][t]))
                    obj_opt += x_val[t]       * abs(float(lam_inh['lambda_x_lower'][t]))
                    obj_opt += (1-x_val[t])   * abs(float(lam_inh['lambda_x_upper'][t]))

                for t in range(1, self.T):
                    ru_expr    = pg_val[t] - pg_val[t-1] - Ru_v*x_val[t-1] - Ru_co_v*(1-x_val[t-1])
                    rd_expr    = pg_val[t-1] - pg_val[t] - Rd_v*x_val[t]   - Rd_co_v*(1-x_val[t])
                    start_expr = sc_v  * (x_val[t] - x_val[t-1]) - coc_val[t-1]
                    shut_expr  = shc_v * (x_val[t-1] - x_val[t]) - coc_val[t-1]
                    obj_opt += abs(ru_expr)    * abs(float(lam_inh['lambda_ramp_up'][t-1]))
                    obj_opt += abs(rd_expr)    * abs(float(lam_inh['lambda_ramp_down'][t-1]))
                    obj_opt += abs(start_expr) * abs(float(lam_inh['lambda_start_cost'][t-1]))
                    obj_opt += abs(shut_expr)  * abs(float(lam_inh['lambda_shut_cost'][t-1]))
                    obj_opt += coc_val[t-1]    * abs(float(lam_inh['lambda_coc_nonneg'][t-1]))

                for tau in range(1, Ton_v + 1):
                    for t1 in range(self.T - tau):
                        expr = x_val[t1+1] - x_val[t1] - x_val[t1+tau]
                        obj_opt += abs(expr) * abs(float(lam_inh['lambda_min_on'][tau-1][t1]))

                for tau in range(1, Toff_v + 1):
                    for t1 in range(self.T - tau):
                        expr = -x_val[t1+1] + x_val[t1] - (1-x_val[t1+tau])
                        obj_opt += abs(expr) * abs(float(lam_inh['lambda_min_off'][tau-1][t1]))

            # ================================================================
            # obj_dual：所有决策变量的KKT驻点条件违反
            # ================================================================

            # -- pg[t] 驻点条件（与dual block一致）--
            if lam_inh is not None:
                lam_ru = lam_inh['lambda_ramp_up']
                lam_rd = lam_inh['lambda_ramp_down']
                for t in range(self.T):
                    pg_stat = a_v + self.pg_cost_values[sample_id][t] - lambda_val[t]
                    pg_stat -= float(lam_inh['lambda_pg_lower'][t])
                    pg_stat += float(lam_inh['lambda_pg_upper'][t])
                    if t > 0:
                        pg_stat += float(lam_ru[t-1])
                        pg_stat -= float(lam_rd[t-1])
                    if t < self.T - 1:
                        pg_stat -= float(lam_ru[t])
                        pg_stat += float(lam_rd[t])
                    obj_dual_pg += abs(pg_stat)

            # -- x[t] 驻点条件（含固有约束项、代理耦合项和cost项）--
            for t in range(self.T):
                x_stat = b_v + self.cost_values[sample_id][t]
                if lam_inh is not None:
                    x_stat += Pmin_v * float(lam_inh['lambda_pg_lower'][t])
                    x_stat -= Pmax_v * float(lam_inh['lambda_pg_upper'][t])
                    lam_ru = lam_inh['lambda_ramp_up']
                    lam_rd = lam_inh['lambda_ramp_down']
                    if t < self.T - 1:
                        x_stat += (Ru_co_v - Ru_v) * float(lam_ru[t])
                    if t > 0:
                        x_stat += (Rd_co_v - Rd_v) * float(lam_rd[t-1])
                    lam_mon  = lam_inh['lambda_min_on']
                    lam_moff = lam_inh['lambda_min_off']
                    for tau in range(1, Ton_v + 1):
                        for t1 in range(self.T - tau):
                            k = float(lam_mon[tau-1][t1])
                            if t == t1 + 1:   x_stat += k
                            if t == t1:       x_stat -= k
                            if t == t1 + tau: x_stat -= k
                    for tau in range(1, Toff_v + 1):
                        for t1 in range(self.T - tau):
                            k = float(lam_moff[tau-1][t1])
                            if t == t1 + 1:   x_stat -= k
                            if t == t1:       x_stat += k
                            if t == t1 + tau: x_stat += k
                    lam_sc  = lam_inh['lambda_start_cost']
                    lam_shc = lam_inh['lambda_shut_cost']
                    if t > 0:
                        x_stat += sc_v  * float(lam_sc[t-1])
                        x_stat -= shc_v * float(lam_shc[t-1])
                    if t < self.T - 1:
                        x_stat -= sc_v  * float(lam_sc[t])
                        x_stat += shc_v * float(lam_shc[t])
                    x_stat += float(lam_inh['lambda_x_upper'][t]) - float(lam_inh['lambda_x_lower'][t])
                # 代理耦合约束对偶贡献（按 sensitive_timesteps 索引）
                sensitive_t = self.sensitive_timesteps[sample_id]
                for k, ts in enumerate(sensitive_t):
                    for time_idx, coeff in iterate_surrogate_constraint_terms(
                        ts,
                        constraint_offsets[k],
                        alphas[k],
                        betas[k],
                        gammas[k],
                        self.T,
                    ):
                        if time_idx == t:
                            x_stat += coeff * mu_vals[k]
                obj_dual_x += abs(x_stat)

            # -- coc[t] 驻点条件（与dual block一致）--
            if lam_inh is not None:
                lam_sc  = lam_inh['lambda_start_cost']
                lam_shc = lam_inh['lambda_shut_cost']
                lam_cn  = lam_inh['lambda_coc_nonneg']
                for t in range(self.T - 1):
                    coc_stat = 1.0 - float(lam_sc[t]) - float(lam_shc[t]) - float(lam_cn[t])
                    obj_dual_coc += abs(coc_stat)

        obj_dual = obj_dual_pg + obj_dual_x + obj_dual_coc
        return obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt

    def cal_obj_opt_breakdown(self) -> dict[str, float]:
        breakdown = {
            'surrogate': 0.0,
            'pg_lower': 0.0,
            'pg_upper': 0.0,
            'x_lower': 0.0,
            'x_upper': 0.0,
            'ramp_up': 0.0,
            'ramp_down': 0.0,
            'min_on': 0.0,
            'min_off': 0.0,
            'start_cost': 0.0,
            'shut_cost': 0.0,
            'coc_nonneg': 0.0,
        }

        g = self.unit_id

        for sample_id in range(self.n_samples):
            x_val = self.x[sample_id]
            pg_val = self.pg[sample_id]
            coc_val = self.coc[sample_id]
            alphas = self.alpha_values[sample_id]
            betas = self.beta_values[sample_id]
            gammas = self.gamma_values[sample_id]
            deltas = self.delta_values[sample_id]
            mu_vals = np.abs(self.mu[sample_id])
            alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
                alphas,
                betas,
                gammas,
                deltas,
            )
            lam_inh = self.lambda_inherent[sample_id]

            Pmin_v = float(self.gen[g, PMIN])
            Pmax_v = float(self.gen[g, PMAX])
            Ru_v = float(self.Ru_all[g])
            Rd_v = float(self.Rd_all[g])
            Ru_co_v = float(self.Ru_co_all[g])
            Rd_co_v = float(self.Rd_co_all[g])
            sc_v = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
            shc_v = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
            Ton_v = min(4, self.T)
            Toff_v = min(4, self.T)

            sensitive_t = self.sensitive_timesteps[sample_id]
            constraint_offsets = self._constraint_offsets_for_sample(sample_id)
            for k, ts in enumerate(sensitive_t):
                coupling_lhs = build_surrogate_constraint_expression(
                    x_val,
                    ts,
                    constraint_offsets[k],
                    alphas[k],
                    betas[k],
                    gammas[k],
                    self.T,
                )
                breakdown['surrogate'] += abs(coupling_lhs - deltas[k]) * mu_vals[k]

            if lam_inh is None:
                continue

            for t in range(self.T):
                breakdown['pg_lower'] += abs(Pmin_v * x_val[t] - pg_val[t]) * abs(float(lam_inh['lambda_pg_lower'][t]))
                breakdown['pg_upper'] += abs(pg_val[t] - Pmax_v * x_val[t]) * abs(float(lam_inh['lambda_pg_upper'][t]))
                breakdown['x_lower'] += x_val[t] * abs(float(lam_inh['lambda_x_lower'][t]))
                breakdown['x_upper'] += (1 - x_val[t]) * abs(float(lam_inh['lambda_x_upper'][t]))

            for t in range(1, self.T):
                ru_expr = pg_val[t] - pg_val[t-1] - Ru_v * x_val[t-1] - Ru_co_v * (1 - x_val[t-1])
                rd_expr = pg_val[t-1] - pg_val[t] - Rd_v * x_val[t] - Rd_co_v * (1 - x_val[t])
                start_expr = sc_v * (x_val[t] - x_val[t-1]) - coc_val[t-1]
                shut_expr = shc_v * (x_val[t-1] - x_val[t]) - coc_val[t-1]
                breakdown['ramp_up'] += abs(ru_expr) * abs(float(lam_inh['lambda_ramp_up'][t-1]))
                breakdown['ramp_down'] += abs(rd_expr) * abs(float(lam_inh['lambda_ramp_down'][t-1]))
                breakdown['start_cost'] += abs(start_expr) * abs(float(lam_inh['lambda_start_cost'][t-1]))
                breakdown['shut_cost'] += abs(shut_expr) * abs(float(lam_inh['lambda_shut_cost'][t-1]))
                breakdown['coc_nonneg'] += coc_val[t-1] * abs(float(lam_inh['lambda_coc_nonneg'][t-1]))

            for tau in range(1, Ton_v + 1):
                for t1 in range(self.T - tau):
                    expr = x_val[t1+1] - x_val[t1] - x_val[t1+tau]
                    breakdown['min_on'] += abs(expr) * abs(float(lam_inh['lambda_min_on'][tau-1][t1]))

            for tau in range(1, Toff_v + 1):
                for t1 in range(self.T - tau):
                    expr = -x_val[t1+1] + x_val[t1] - (1 - x_val[t1+tau])
                    breakdown['min_off'] += abs(expr) * abs(float(lam_inh['lambda_min_off'][tau-1][t1]))

        breakdown['total'] = float(sum(breakdown.values()))
        return {k: float(v) for k, v in breakdown.items()}

    def cal_viol(self) -> Tuple[float, float, float]:
        obj_primal, _, _, _, obj_dual, obj_opt = self.cal_viol_components()
        return obj_primal, obj_dual, obj_opt
    
    def iter(
        self,
        max_iter: int = 20,
        nn_epochs: int = 10,
        pg_cost_nn_epochs: int | None = None,
        nn_batch_strategy: str | None = None,
        nn_batch_size: int | None = None,
        nn_shuffle: bool | None = None,
        nn_learning_rate: float | None = None,
        cost_learning_rate: float | None = None,
        pg_cost_surr_learning_rate: float | None = None,
    ):
        """
        主BCD迭代循环 - V3三时段耦合约束版本
        """
        if not hasattr(self, 'logger'):
            self.logger = None
        print(f"开始BCD迭代训练 (机组{self.unit_id}, V3三时段耦合约束)...", flush=True)
        gamma = self.gamma_base / (self.n_samples * max(max_iter, 1))
        gamma_dual = gamma * self.gamma_dual_component_scale
        self.gamma = gamma
        resolved_pg_cost_nn_epochs = (
            self.pg_cost_nn_epochs if pg_cost_nn_epochs is None else max(int(pg_cost_nn_epochs), 1)
        )
        
        for i in range(max_iter):
            print(f"🔄 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i
            self._sync_surrogate_direction_strategy_state()
            
            EPS = 1e-10
            
            # 1. 原始块迭代（V3：传入5个参数）
            for sample_id in range(self.n_samples):
                alphas = self.alpha_values[sample_id]
                betas = self.beta_values[sample_id]
                gammas = self.gamma_values[sample_id]
                deltas = self.delta_values[sample_id]
                costs = self.cost_values[sample_id]
                pg_costs = self.pg_cost_values[sample_id]

                pg_sol, x_sol, coc_sol, cpower_sol = self.iter_with_primal_block(
                    sample_id, alphas, betas, gammas, deltas, costs, pg_costs
                )
                
                if pg_sol is not None:
                    self.pg[sample_id] = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                    self.x[sample_id] = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                    self.x[sample_id] = np.where(np.abs(self.x[sample_id] - 1) < EPS, 1, self.x[sample_id])
                    self.coc[sample_id] = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)
                    self.cpower[sample_id] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
            
            # 2. 对偶块迭代（V3：联合更新固有约束对偶变量和代理耦合对偶变量）
            lb_mu = self._current_mu_lower_bound_value()
            sign_relax_round = self._is_mu_sign_relaxation_round()
            raw_mu_solutions = np.zeros((self.n_samples, self.num_coupling_constraints), dtype=float)
            solved_mask = np.zeros(self.n_samples, dtype=bool)
            for sample_id in range(self.n_samples):
                alphas = self.alpha_values[sample_id]
                betas  = self.beta_values[sample_id]
                gammas = self.gamma_values[sample_id]
                deltas = self.delta_values[sample_id]
                costs  = self.cost_values[sample_id]
                pg_costs = self.pg_cost_values[sample_id]

                lambda_inherent_sol, mu_sol = self.iter_with_dual_block(
                    sample_id, alphas, betas, gammas, deltas, costs, pg_costs
                )
                if lambda_inherent_sol is not None:
                    self.lambda_inherent[sample_id] = lambda_inherent_sol
                    raw_mu_solutions[sample_id] = mu_sol
                    solved_mask[sample_id] = True
            if sign_relax_round and np.any(solved_mask):
                self.surrogate_direction_signs = self._resolve_surrogate_direction_signs_from_mu(
                    raw_mu_solutions[solved_mask]
                )
            direction_signs = self._get_surrogate_direction_signs()
            for sample_id in range(self.n_samples):
                if solved_mask[sample_id]:
                    self.mu[sample_id] = self._finalize_mu_values(
                        raw_mu_solutions[sample_id],
                        lb_mu,
                        direction_signs=direction_signs,
                    )
            
            # 计算违反量（NN更新前）
            _z = lambda v: v if abs(v) >= 1e-12 else 0.0
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = (
                _z(obj_primal), _z(obj_dual_pg), _z(obj_dual_x), _z(obj_dual_coc), _z(obj_dual), _z(obj_opt)
            )
            nn_metrics_pre = self.cal_nn_logging_components()
            print(
                f"[Unit-{self.unit_id}][NN-metric][before] "
                f"obj_primal={nn_metrics_pre['obj_primal']:.6f}, "
                f"obj_dual_pg={nn_metrics_pre['obj_dual_pg']:.6f}, "
                f"obj_dual_x={nn_metrics_pre['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_pre['obj_opt']:.6f}, "
                f"reg_main={nn_metrics_pre['reg_main']:.6f}, "
                f"reg_pg={nn_metrics_pre['reg_pg']:.6f}",
                flush=True,
            )
            merit_before_nn = self._weighted_violation_merit(
                (obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt)
            )

            # 3. 神经网络更新代理约束参数
            #    先缓存上一代输出，供“迭代间差异正则”使用（默认权重为0时不影响训练）
            self._prev_alpha_values = self.alpha_values.copy()
            self._prev_beta_values = self.beta_values.copy()
            self._prev_gamma_values = self.gamma_values.copy()
            self._prev_delta_values = self.delta_values.copy()
            self._prev_cost_values = self.cost_values.copy()
            self._prev_pg_cost_values = self.pg_cost_values.copy()
            nn_snapshot = self._capture_nn_update_state()
            self.iter_with_surrogate_nn(
                num_epochs=nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
                cost_learning_rate=cost_learning_rate,
            )
            self.iter_with_c_pg_nn(
                num_epochs=resolved_pg_cost_nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=pg_cost_surr_learning_rate,
            )
            self._refresh_cached_surrogate_outputs()

            # 计算违反量（NN更新后）
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = (
                _z(obj_primal), _z(obj_dual_pg), _z(obj_dual_x), _z(obj_dual_coc), _z(obj_dual), _z(obj_opt)
            )
            merit_after_nn = self._weighted_violation_merit(
                (obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt)
            )
            merit_tol = max(1e-8, 1e-4 * max(1.0, abs(merit_before_nn)))
            if merit_after_nn > merit_before_nn + merit_tol:
                self._restore_nn_update_state(nn_snapshot)
                obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
                obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = (
                    _z(obj_primal), _z(obj_dual_pg), _z(obj_dual_x), _z(obj_dual_coc), _z(obj_dual), _z(obj_opt)
                )
                merit_after_nn = self._weighted_violation_merit(
                    (obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt)
                )
                print(
                    f"[Unit-{self.unit_id}]   rollback NN update: "
                    f"merit_before={merit_before_nn:.6f}, merit_after={merit_after_nn:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[Unit-{self.unit_id}]   accept NN update: "
                    f"merit_before={merit_before_nn:.6f}, merit_after={merit_after_nn:.6f}",
                    flush=True,
                )
            nn_metrics_after = self.cal_nn_logging_components()
            print(
                f"[Unit-{self.unit_id}][NN-metric][after] "
                f"obj_primal={nn_metrics_after['obj_primal']:.6f}, "
                f"obj_dual_pg={nn_metrics_after['obj_dual_pg']:.6f}, "
                f"obj_dual_x={nn_metrics_after['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_after['obj_opt']:.6f}, "
                f"reg_main={nn_metrics_after['reg_main']:.6f}, "
                f"reg_pg={nn_metrics_after['reg_pg']:.6f}",
                flush=True,
            )

            # 前3次迭代冻结rho，之后再按累加式更新
            if i >= 3:
                self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
                self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
                self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
                self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
                self._sync_rho_dual_summary()
                self.rho_binary = min(self.rho_binary + gamma * obj_binary, self.rho_max)
                self.rho_opt    = min(self.rho_opt    + gamma * obj_opt,    self.rho_max)

            print(
                f"  ρ_primal={self.rho_primal:.4f}, ρ_dual_pg={self.rho_dual_pg:.4f}, "
                f"ρ_dual_x={self.rho_dual_x:.4f}, ρ_dual_coc={self.rho_dual_coc:.4f}, "
                f"ρ_dual={self.rho_dual:.4f}, ρ_opt={self.rho_opt:.4f}",
                flush=True,
            )
            print("  " + "-" * 40, flush=True)

            # logger 钩子
            if i == max_iter - 1:
                obj_opt_breakdown = self.cal_obj_opt_breakdown()
                print(
                    f"[Unit-{self.unit_id}][full][final] obj_opt_breakdown: "
                    f"surrogate={obj_opt_breakdown['surrogate']:.6f}, "
                    f"pg_lower={obj_opt_breakdown['pg_lower']:.6f}, "
                    f"pg_upper={obj_opt_breakdown['pg_upper']:.6f}, "
                    f"x_lower={obj_opt_breakdown['x_lower']:.6f}, "
                    f"x_upper={obj_opt_breakdown['x_upper']:.6f}, "
                    f"ramp_up={obj_opt_breakdown['ramp_up']:.6f}, "
                    f"ramp_down={obj_opt_breakdown['ramp_down']:.6f}, "
                    f"min_on={obj_opt_breakdown['min_on']:.6f}, "
                    f"min_off={obj_opt_breakdown['min_off']:.6f}, "
                    f"start_cost={obj_opt_breakdown['start_cost']:.6f}, "
                    f"shut_cost={obj_opt_breakdown['shut_cost']:.6f}, "
                    f"coc_nonneg={obj_opt_breakdown['coc_nonneg']:.6f}, "
                    f"total={obj_opt_breakdown['total']:.6f}",
                    flush=True,
                )

            if self.logger is not None:
                nn_loss = getattr(self, '_last_surr_nn_loss', None)
                self.logger.log_surrogate_iter(
                    unit_id=self.unit_id, iter=i,
                    obj_primal=obj_primal, obj_dual=obj_dual, obj_opt=obj_opt,
                    rho_primal=self.rho_primal, rho_dual=self.rho_dual, rho_opt=self.rho_opt,
                    obj_dual_pg=obj_dual_pg,
                    obj_dual_x=obj_dual_x,
                    obj_dual_coc=obj_dual_coc,
                    rho_dual_pg=self.rho_dual_pg,
                    rho_dual_x=self.rho_dual_x,
                    rho_dual_coc=self.rho_dual_coc,
                    alpha_mean=float(np.mean(self.alpha_values)),
                    beta_mean=float(np.mean(self.beta_values)),
                    gamma_mean=float(np.mean(self.gamma_values)),
                    delta_mean=float(np.mean(self.delta_values)),
                    mu_mean=float(np.mean(self.mu)),
                    nn_loss=nn_loss,
                )


        print(f"✓ 机组{self.unit_id} V3三时段耦合代理约束训练完成", flush=True)
    
    def get_surrogate_params(
        self,
        pd_data: np.ndarray | dict,
        lambda_val: np.ndarray,
        renewable_data: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取V3三时段耦合代理约束参数

        Returns:
            alphas: (max_constraints,) 第一时段系数
            betas: (max_constraints,) 第二时段系数
            gammas: (max_constraints,) 第三时段系数
            deltas: (max_constraints,) 右端项
            costs: (T,) x 调整项
            pg_costs: (T,) pg 调整项
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")

        self.surrogate_net.eval()

        lambda_val = np.asarray(lambda_val, dtype=float).reshape(-1)
        # 机组静态参数（与 _extract_features 保持一致）
        g = self.unit_id
        Pmax = self.gen[g, PMAX]
        unit_params = np.array([
            self.gen[g, PMIN] / (Pmax + 1e-8),
            self.gencost[g, -2] / self.T_delta,
            self.gencost[g, -1] / self.T_delta / (Pmax + 1e-8),
            float(self.Ru_all[g]) / (Pmax + 1e-8),
            0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1] / (Pmax + 1e-8),
            0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2] / (Pmax + 1e-8),
        ])
        expected_total_dim = int(self.surrogate_net.input_proj.in_features)
        expected_pd_dim = expected_total_dim - lambda_val.size - unit_params.size
        pd_flat = self._build_compatible_surrogate_feature_vector(
            pd_data,
            renewable_data=renewable_data,
            expected_pd_dim=expected_pd_dim,
        )
        features = np.concatenate([pd_flat, lambda_val, unit_params])
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            alphas, betas, gammas, deltas, costs = self.surrogate_net.forward_main(features_tensor)
            pg_costs = self.surrogate_net.forward_pg_cost(features_tensor)
            deltas = self._postprocess_delta_tensor(deltas.squeeze(0)).unsqueeze(0)

        alphas_np = alphas.squeeze(0).cpu().numpy()
        betas_np = betas.squeeze(0).cpu().numpy()
        gammas_np = gammas.squeeze(0).cpu().numpy()
        deltas_np = deltas.squeeze(0).cpu().numpy()
        alphas_np, betas_np, gammas_np, deltas_np = self._apply_surrogate_direction_to_params(
            alphas_np,
            betas_np,
            gammas_np,
            deltas_np,
        )

        return (alphas_np,
                betas_np,
                gammas_np,
                deltas_np,
                costs.squeeze(0).cpu().numpy(),
                pg_costs.squeeze(0).cpu().numpy())

    def _build_compatible_surrogate_feature_vector(
        self,
        pd_data: np.ndarray | dict,
        renewable_data: np.ndarray | None,
        expected_pd_dim: int,
    ) -> np.ndarray:
        """Build surrogate scenario features that match the loaded model input width.

        Old models were trained on net-load-only features, while newer branches may
        concatenate [load, renewable]. During FP inference we may receive richer
        sample dicts than the model was trained on, so select the candidate whose
        flattened length matches the network's expected scenario dimension.
        """
        candidates: list[tuple[str, np.ndarray]] = []

        def _add_row_reduced_candidates(name: str, matrix_2d: np.ndarray) -> None:
            matrix_2d = np.asarray(matrix_2d, dtype=float)
            if matrix_2d.ndim != 2:
                return
            rows, horizon = matrix_2d.shape
            if rows * horizon == expected_pd_dim:
                return
            if horizon <= 0 or expected_pd_dim % horizon != 0:
                return
            target_rows = expected_pd_dim // horizon
            if target_rows <= 0 or target_rows >= rows:
                return

            row_activity = np.sum(np.abs(matrix_2d), axis=1)
            active_order = np.argsort(-row_activity, kind="stable")[:target_rows]
            active_order = np.sort(active_order)
            candidates.append(
                (f"{name}_top_rows_{target_rows}", matrix_2d[active_order, :].reshape(-1))
            )
            candidates.append(
                (f"{name}_leading_rows_{target_rows}", matrix_2d[:target_rows, :].reshape(-1))
            )

        if isinstance(pd_data, dict):
            sample = normalize_sample_arrays(dict(pd_data))
            net_load_matrix = np.asarray(get_sample_net_load(sample), dtype=float)
            candidates.append(("net_load", net_load_matrix.reshape(-1)))
            _add_row_reduced_candidates("net_load", net_load_matrix)

            if "load_data" in sample:
                load_matrix = np.asarray(get_sample_load_data(sample), dtype=float)
                candidates.append(("load_only", load_matrix.reshape(-1)))
                _add_row_reduced_candidates("load_only", load_matrix)

            if "load_data" in sample and "renewable_data" in sample:
                load_matrix = np.asarray(get_sample_load_data(sample), dtype=float)
                renewable_matrix = np.asarray(get_sample_renewable_data(sample), dtype=float)
                load_plus_renew = np.concatenate(
                    [
                        load_matrix.reshape(-1),
                        renewable_matrix.reshape(-1),
                    ]
                )
                candidates.append(("load_plus_renewable", load_plus_renew))

            default_feature = np.asarray(get_feature_vector_from_sample(sample), dtype=float).reshape(-1)
            candidates.append(("default", default_feature))
        else:
            pd_arr = np.asarray(pd_data, dtype=float)
            candidates.append(("pd_only", np.asarray(get_feature_vector(pd_arr), dtype=float).reshape(-1)))
            _add_row_reduced_candidates("pd_only", pd_arr)

            if renewable_data is not None:
                candidates.append(
                    (
                        "pd_plus_renewable",
                        np.asarray(get_feature_vector(pd_arr, renewable_data=renewable_data), dtype=float).reshape(-1),
                    )
                )

        for _, candidate in candidates:
            if candidate.size == expected_pd_dim:
                return candidate

        default_name, default_candidate = candidates[0]
        raise ValueError(
            f"Surrogate feature dimension mismatch for unit {self.unit_id}: "
            f"expected scenario dim {expected_pd_dim}, got candidates "
            f"{[(name, arr.size) for name, arr in candidates]}; default={default_name}"
        )
    
    def save(self, filepath: str):
        """保存V3模型"""
        if TORCH_AVAILABLE:
            state = {
                'surrogate_net_state_dict': self.surrogate_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'alpha_values': self.alpha_values,
                'beta_values': self.beta_values,
                'gamma_values': self.gamma_values,
                'delta_values': self.delta_values,
                'cost_values': self.cost_values,
                'pg_cost_values': self.pg_cost_values,
                'x_cost_scale': self.x_cost_scale,
                'pg_cost_scale': self.pg_cost_scale,
                'mu': self.mu,
                'surrogate_direction_signs': self._get_surrogate_direction_signs(),
                'rho_primal': self.rho_primal,
                'rho_dual': self.rho_dual,
                'rho_dual_pg': self.rho_dual_pg,
                'rho_dual_x': self.rho_dual_x,
                'rho_dual_coc': self.rho_dual_coc,
                'rho_binary': self.rho_binary,
                'rho_opt': self.rho_opt,
                'gamma_dual_component_scale': self.gamma_dual_component_scale,
                'num_coupling_constraints': self.num_coupling_constraints,
                'max_constraints': self.max_constraints,
                'requested_max_constraints': self.requested_max_constraints,
                'constraint_generation_strategy': self.constraint_generation_strategy,
                'ignore_startup_shutdown_costs': self.ignore_startup_shutdown_costs,
                'mu_individual_lower_bound_round': self.mu_individual_lower_bound_round,
                'mu_group_lower_bound_round': self.mu_group_lower_bound_round,
                'mu_signed_round_interval': self.mu_signed_round_interval,
                'x_bound_dual_zero_rounds': self.x_bound_dual_zero_rounds,
                'pg_cost_start_round': self.pg_cost_start_round,
                'pg_cost_scale_multiplier': self.pg_cost_scale_multiplier,
                'nn_learning_rate': self.nn_learning_rate,
                'nn_hidden_dims': self.nn_hidden_dims,
                'cost_learning_rate': self.cost_learning_rate,
                'pg_cost_lr': self.pg_cost_lr,
                'pg_cost_surr_lr': self.pg_cost_surr_lr,
                'pg_cost_nn_epochs': self.pg_cost_nn_epochs,
                'nn_batch_strategy': self.nn_batch_strategy,
                'nn_batch_size': self.nn_batch_size,
                'nn_shuffle': self.nn_shuffle,
                'pg_cost_reg_deadband': self.pg_cost_reg_deadband,
                'nn_smooth_abs_eps': self.nn_smooth_abs_eps,
                'pg_cost_smooth_abs_eps': self.pg_cost_smooth_abs_eps,
                'loss_ratio_primal': self.loss_ratio_primal,
                'loss_ratio_dual_pg': self.loss_ratio_dual_pg,
                'loss_ratio_dual_x': self.loss_ratio_dual_x,
                'nn_dual_term_interval': self.nn_dual_term_interval,
                'loss_ratio_opt': self.loss_ratio_opt,
                'loss_ratio_reg': self.loss_ratio_reg,
                'pg_block_prox_weight': self.pg_block_prox_weight,
                'dual_block_prox_weight': self.dual_block_prox_weight,
                'template_rhs_jitter_scale': self.template_rhs_jitter_scale,
                'template_rhs_reg_deadband': self.template_rhs_reg_deadband,
                'coeff_reg_deadband': self.coeff_reg_deadband,
                'aux_cost_reg_deadband': self.aux_cost_reg_deadband,
                'lambda_inherent': self.lambda_inherent,
            }
            
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            
            torch.save(state, filepath)
            print(f"✓ V3三时段耦合代理约束模型已保存: {filepath}", flush=True)
    
    def load(self, filepath: str):
        """加载V3模型"""
        if TORCH_AVAILABLE:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            saved_hidden_dims = state.get('nn_hidden_dims')
            if saved_hidden_dims is not None:
                resolved_hidden_dims = normalize_nn_hidden_dims(saved_hidden_dims, self.nn_hidden_dims)
                if resolved_hidden_dims != self.nn_hidden_dims:
                    self.nn_hidden_dims = resolved_hidden_dims
                    self._init_neural_network()
            self.surrogate_net.load_state_dict(state['surrogate_net_state_dict'], strict=False)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alpha_values = state['alpha_values']
            self.beta_values = state['beta_values']
            self.gamma_values = state['gamma_values']
            self.delta_values = state.get('delta_values', np.full_like(self.gamma_values, 3.0))
            self.cost_values = state.get('cost_values', np.zeros((self.n_samples, self.T)))
            self.pg_cost_values = state.get('pg_cost_values', np.zeros((self.n_samples, self.T)))
            self.x_cost_scale = state.get('x_cost_scale', self.x_cost_scale)
            self.pg_cost_scale = state.get('pg_cost_scale', self.pg_cost_scale)
            self.mu = state['mu']
            loaded_direction_signs = state.get('surrogate_direction_signs')
            if loaded_direction_signs is None:
                self.surrogate_direction_signs = np.ones(self.num_coupling_constraints, dtype=float)
            else:
                self.surrogate_direction_signs = np.asarray(loaded_direction_signs, dtype=float).reshape(-1)
            self.rho_primal = state['rho_primal']
            self.rho_dual_pg = state.get('rho_dual_pg', state.get('rho_dual', self.rho_dual_pg))
            self.rho_dual_x = state.get('rho_dual_x', state.get('rho_dual', self.rho_dual_x))
            self.rho_dual_coc = state.get('rho_dual_coc', state.get('rho_dual', self.rho_dual_coc))
            self._sync_rho_dual_summary()
            self.rho_binary = state.get('rho_binary', self.rho_binary)
            self.rho_opt = state['rho_opt']
            self.gamma_dual_component_scale = state.get(
                'gamma_dual_component_scale',
                self.gamma_dual_component_scale,
            )
            saved_strategy = state.get('constraint_generation_strategy', 'sensitive')
            self.constraint_generation_strategy = normalize_constraint_generation_strategy(saved_strategy)
            self.ignore_startup_shutdown_costs = bool(
                state.get('ignore_startup_shutdown_costs', self.ignore_startup_shutdown_costs)
            )
            self.all_mode_group_size = (
                4
                if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4
                else 1
            )
            self.mu_individual_lower_bound_round = state.get(
                'mu_individual_lower_bound_round',
                self.mu_individual_lower_bound_round,
            )
            self.mu_group_lower_bound_round = max(
                state.get('mu_group_lower_bound_round', self.mu_group_lower_bound_round),
                self.mu_individual_lower_bound_round,
            )
            self.mu_signed_round_interval = max(
                int(state.get('mu_signed_round_interval', self.mu_signed_round_interval)),
                0,
            )
            self.x_bound_dual_zero_rounds = max(
                int(state.get('x_bound_dual_zero_rounds', self.x_bound_dual_zero_rounds)),
                0,
            )
            self.pg_cost_start_round = state.get('pg_cost_start_round', self.pg_cost_start_round)
            self.pg_cost_scale_multiplier = state.get(
                'pg_cost_scale_multiplier',
                self.pg_cost_scale_multiplier,
            )
            self.nn_learning_rate = state.get('nn_learning_rate', self.nn_learning_rate)
            self.nn_hidden_dims = normalize_nn_hidden_dims(
                state.get('nn_hidden_dims'),
                self.nn_hidden_dims,
            )
            self.cost_learning_rate = state.get('cost_learning_rate', self.cost_learning_rate)
            self.pg_cost_lr = state.get('pg_cost_lr', self.pg_cost_lr)
            self.pg_cost_surr_lr = state.get('pg_cost_surr_lr', self.pg_cost_surr_lr)
            loaded_pg_cost_nn_epochs = state.get('pg_cost_nn_epochs', self.pg_cost_nn_epochs)
            self.pg_cost_nn_epochs = (
                None if loaded_pg_cost_nn_epochs is None else max(int(loaded_pg_cost_nn_epochs), 1)
            )
            self.nn_batch_strategy = normalize_nn_batch_strategy(
                state.get('nn_batch_strategy', self.nn_batch_strategy)
            )
            self.nn_batch_size = max(int(state.get('nn_batch_size', self.nn_batch_size)), 1)
            self.nn_shuffle = bool(state.get('nn_shuffle', self.nn_shuffle))
            self.loss_ratio_primal = float(state.get('loss_ratio_primal', self.loss_ratio_primal))
            self.loss_ratio_dual_pg = float(state.get('loss_ratio_dual_pg', self.loss_ratio_dual_pg))
            self.loss_ratio_dual_x = float(state.get('loss_ratio_dual_x', self.loss_ratio_dual_x))
            loaded_nn_dual_term_interval = state.get('nn_dual_term_interval', self.nn_dual_term_interval)
            if loaded_nn_dual_term_interval is None:
                self.nn_dual_term_interval = None
            else:
                self.nn_dual_term_interval = max(int(loaded_nn_dual_term_interval), 1)
            self.loss_ratio_opt = float(state.get('loss_ratio_opt', self.loss_ratio_opt))
            self.loss_ratio_reg = float(state.get('loss_ratio_reg', self.loss_ratio_reg))
            self.pg_block_prox_weight = max(
                float(state.get('pg_block_prox_weight', self.pg_block_prox_weight)),
                0.0,
            )
            self.dual_block_prox_weight = max(
                float(state.get('dual_block_prox_weight', self.dual_block_prox_weight)),
                0.0,
            )
            self.pg_cost_reg_deadband = state.get(
                'pg_cost_reg_deadband',
                self.pg_cost_reg_deadband,
            )
            self.nn_smooth_abs_eps = max(
                float(state.get('nn_smooth_abs_eps', self.nn_smooth_abs_eps)),
                0.0,
            )
            self.pg_cost_smooth_abs_eps = max(
                float(state.get('pg_cost_smooth_abs_eps', self.pg_cost_smooth_abs_eps)),
                0.0,
            )
            self.template_rhs_jitter_scale = state.get(
                'template_rhs_jitter_scale',
                self.template_rhs_jitter_scale,
            )
            self.template_rhs_reg_deadband = state.get(
                'template_rhs_reg_deadband',
                self.template_rhs_reg_deadband,
            )
            self.coeff_reg_deadband = state.get('coeff_reg_deadband', self.coeff_reg_deadband)
            self.aux_cost_reg_deadband = state.get(
                'aux_cost_reg_deadband',
                self.aux_cost_reg_deadband,
            )
            self.surrogate_net.x_cost_scale = self.x_cost_scale
            self.surrogate_net.pg_cost_scale = self.pg_cost_scale
            if self._uses_template_rhs_bases():
                self.surrogate_net.delta_base = 0.0
                self.surrogate_net.delta_scale = self.template_rhs_jitter_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.nn_learning_rate
            for param_group in self.cost_optimizer.param_groups:
                param_group['lr'] = self.cost_learning_rate
            for param_group in self.pg_cost_optimizer.param_groups:
                param_group['lr'] = self.pg_cost_lr
            self.requested_max_constraints = state.get('requested_max_constraints', self.requested_max_constraints)
            self.max_constraints = state.get('max_constraints', self.max_constraints)
            self.num_coupling_constraints = state.get('num_coupling_constraints', self.num_coupling_constraints)
            self.template_rhs_base_vector = self._build_template_rhs_base_vector(self.num_coupling_constraints)
            self.surrogate_direction_signs = self._get_surrogate_direction_signs()
            if 'lambda_inherent' in state:
                self.lambda_inherent = state['lambda_inherent']
            print(f"✓ V3三时段耦合代理约束模型已加载: {filepath}", flush=True)


# ========================== 训练代码 ==========================

def _load_surrogate_model_metadata(filepath: str, device=None) -> dict:
    if not TORCH_AVAILABLE:
        return {}
    state = torch.load(filepath, map_location=device, weights_only=False)
    return {
        'constraint_generation_strategy': state.get('constraint_generation_strategy', 'sensitive'),
        'ignore_startup_shutdown_costs': bool(state.get('ignore_startup_shutdown_costs', False)),
        'max_constraints': state.get('max_constraints'),
        'requested_max_constraints': state.get('requested_max_constraints'),
        'num_coupling_constraints': state.get('num_coupling_constraints'),
        'mu_individual_lower_bound_round': state.get('mu_individual_lower_bound_round'),
        'mu_group_lower_bound_round': state.get('mu_group_lower_bound_round'),
        'mu_signed_round_interval': state.get('mu_signed_round_interval'),
        'surrogate_direction_signs': state.get('surrogate_direction_signs'),
        'nn_hidden_dims': state.get('nn_hidden_dims'),
    }


def train_dual_predictor_from_data(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                                    num_epochs: int = 100, batch_size: int = 8,
                                    batch_strategy: str = "full-batch",
                                    shuffle: bool = True,
                                    learning_rate: float = 1e-3,
                                    save_path: str = None, device=None) -> DualVariablePredictorTrainer:
    """
    训练对偶变量预测器
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据列表
        T_delta: 时间间隔
        num_epochs: 训练轮数
        batch_size: 批次大小
        save_path: 模型保存路径（可选）
        device: 计算设备
        
    Returns:
        训练好的对偶变量预测器
    """
    print("=" * 60, flush=True)
    print("训练对偶变量预测器", flush=True)
    print("=" * 60, flush=True)
    
    # 创建预测器
    predictor = DualVariablePredictorTrainer(
        ppc,
        active_set_data,
        T_delta,
        device=device,
        batch_strategy=batch_strategy,
        batch_size=batch_size,
        shuffle=shuffle,
        learning_rate=learning_rate,
    )
    
    # 训练
    predictor.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        batch_strategy=batch_strategy,
        shuffle=shuffle,
        learning_rate=learning_rate,
    )
    
    # 保存模型
    if save_path:
        predictor.save(save_path)
    
    return predictor


def train_subproblem_surrogate_from_data(ppc, active_set_data: List[Dict], unit_id: int,
                                          T_delta: float = 1.0, lambda_predictor=None,
                                          max_iter: int = 20, nn_epochs: int = 10,
                                          constraint_generation_strategy: str = "sensitive",
                                          rho_dual_pg_init: float | None = None,
                                          rho_dual_x_init: float | None = None,
                                          rho_dual_coc_init: float | None = None,
                                          loss_ratio_primal: float = 1.0,
                                          loss_ratio_dual_pg: float = 1.0,
                                          loss_ratio_dual_x: float = 1.0,
                                          loss_ratio_opt: float = 1.0,
                                          loss_ratio_reg: float = 1.0,
                                          mu_individual_lower_bound_round: int = 3,
                                          mu_group_lower_bound_round: int = 50,
                                          mu_signed_round_interval: int | None = None,
                                          pg_cost_start_round: int = 3,
                                          pg_cost_scale_multiplier: float = 1.2,
                                          nn_learning_rate: float = 1e-4,
                                          cost_learning_rate: float = 1e-5,
                                          pg_cost_lr: float = 2e-5,
                                          pg_cost_surr_lr: float = 5e-5,
                                          nn_batch_strategy: str = "full-batch",
                                          nn_batch_size: int = 4,
                                          nn_shuffle: bool = True,
                                          pg_cost_reg_deadband: float = 0.5,
                                          save_path: str = None, device=None) -> SubproblemSurrogateTrainer:
    """
    训练单机组子问题代理约束
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据列表
        unit_id: 机组ID
        T_delta: 时间间隔
        lambda_predictor: 已训练的对偶变量预测器（可选）
        max_iter: BCD最大迭代次数
        nn_epochs: 每次BCD迭代中神经网络训练轮数
        save_path: 模型保存路径（可选）
        device: 计算设备
        
    Returns:
        训练好的代理约束训练器
    """
    print("=" * 60, flush=True)
    print(f"训练机组{unit_id}子问题代理约束", flush=True)
    print("=" * 60, flush=True)
    
    # 创建训练器
    trainer = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta, unit_id,
        lambda_predictor=lambda_predictor,
        constraint_generation_strategy=constraint_generation_strategy,
        rho_dual_pg_init=rho_dual_pg_init,
        rho_dual_x_init=rho_dual_x_init,
        rho_dual_coc_init=rho_dual_coc_init,
        loss_ratio_primal=loss_ratio_primal,
        loss_ratio_dual_pg=loss_ratio_dual_pg,
        loss_ratio_dual_x=loss_ratio_dual_x,
        loss_ratio_opt=loss_ratio_opt,
        loss_ratio_reg=loss_ratio_reg,
        mu_individual_lower_bound_round=mu_individual_lower_bound_round,
        mu_group_lower_bound_round=mu_group_lower_bound_round,
        mu_signed_round_interval=mu_signed_round_interval,
        pg_cost_start_round=pg_cost_start_round,
        pg_cost_scale_multiplier=pg_cost_scale_multiplier,
        nn_learning_rate=nn_learning_rate,
        cost_learning_rate=cost_learning_rate,
        pg_cost_lr=pg_cost_lr,
        pg_cost_surr_lr=pg_cost_surr_lr,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        pg_cost_reg_deadband=pg_cost_reg_deadband,
        device=device
    )
    
    # BCD迭代训练
    trainer.iter(
        max_iter=max_iter,
        nn_epochs=nn_epochs,
        nn_batch_strategy=nn_batch_strategy,
        nn_batch_size=nn_batch_size,
        nn_shuffle=nn_shuffle,
        nn_learning_rate=nn_learning_rate,
        cost_learning_rate=cost_learning_rate,
        pg_cost_surr_learning_rate=pg_cost_surr_lr,
    )
    
    # 保存模型
    if save_path:
        trainer.save(save_path)
    
    return trainer


def train_all_subproblem_surrogates(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                                      lambda_predictor=None, unit_ids: List[int] = None,
                                      max_iter: int = 20, nn_epochs: int = 10,
                                      constraint_generation_strategy: str = "sensitive",
                                      rho_dual_pg_init: float | None = None,
                                      rho_dual_x_init: float | None = None,
                                      rho_dual_coc_init: float | None = None,
                                      loss_ratio_primal: float = 1.0,
                                      loss_ratio_dual_pg: float = 1.0,
                                      loss_ratio_dual_x: float = 1.0,
                                      loss_ratio_opt: float = 1.0,
                                      loss_ratio_reg: float = 1.0,
                                      mu_individual_lower_bound_round: int = 3,
                                      mu_group_lower_bound_round: int = 50,
                                      mu_signed_round_interval: int | None = None,
                                       pg_cost_start_round: int = 3,
                                       pg_cost_scale_multiplier: float = 1.2,
                                       nn_learning_rate: float = 1e-4,
                                       cost_learning_rate: float = 1e-5,
                                       pg_cost_lr: float = 2e-5,
                                       pg_cost_surr_lr: float = 5e-5,
                                       nn_batch_strategy: str = "full-batch",
                                       nn_batch_size: int = 4,
                                       nn_shuffle: bool = True,
                                       pg_cost_reg_deadband: float = 0.25,
                                       save_dir: str = None, device=None) -> Dict[int, SubproblemSurrogateTrainer]:
    """
    训练所有机组的子问题代理约束
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据列表
        T_delta: 时间间隔
        lambda_predictor: 已训练的对偶变量预测器（可选）
        unit_ids: 要训练的机组ID列表（默认为所有机组）
        max_iter: BCD最大迭代次数
        nn_epochs: 每次BCD迭代中神经网络训练轮数
        save_dir: 模型保存目录（可选）
        device: 计算设备
        
    Returns:
        所有机组的代理约束训练器字典 {unit_id: trainer}
    """
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    
    if unit_ids is None:
        unit_ids = list(range(ng))
    
    print("=" * 60, flush=True)
    print(f"训练所有机组代理约束 ({len(unit_ids)} 个机组)", flush=True)
    print("=" * 60, flush=True)
    
    trainers = {}
    
    for i, g in enumerate(unit_ids):
        print(f"\n>>> 机组 {g} ({i+1}/{len(unit_ids)}) <<<", flush=True)
        
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, g,
            lambda_predictor=lambda_predictor,
            constraint_generation_strategy=constraint_generation_strategy,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            mu_signed_round_interval=mu_signed_round_interval,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_scale_multiplier=pg_cost_scale_multiplier,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            pg_cost_reg_deadband=pg_cost_reg_deadband,
            device=device
        )
        
        trainer.iter(
            max_iter=max_iter,
            nn_epochs=nn_epochs,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            pg_cost_surr_learning_rate=pg_cost_surr_lr,
        )
        trainers[g] = trainer
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))
    
    print(f"\n✓ 所有机组代理约束训练完成", flush=True)
    return trainers


def train_complete_model(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                          unit_ids: List[int] = None,
                          dual_epochs: int = 100, dual_batch_size: int = 8,
                          dual_batch_strategy: str = "full-batch",
                          dual_shuffle: bool = True,
                          dual_learning_rate: float = 1e-3,
                          surrogate_max_iter: int = 20, surrogate_nn_epochs: int = 10,
                          constraint_generation_strategy: str = "sensitive",
                          surrogate_nn_batch_strategy: str = "full-batch",
                          surrogate_nn_batch_size: int = 4,
                          surrogate_nn_shuffle: bool = True,
                          surrogate_nn_learning_rate: float = 1e-4,
                          surrogate_cost_learning_rate: float = 1e-5,
                          surrogate_pg_cost_lr: float = 2e-5,
                          surrogate_pg_cost_surr_lr: float = 5e-5,
                          surrogate_pg_cost_reg_deadband: float = 0.25,
                          surrogate_loss_ratio_primal: float = 1.0,
                          surrogate_loss_ratio_dual_pg: float = 1.0,
                          surrogate_loss_ratio_dual_x: float = 1.0,
                          surrogate_loss_ratio_opt: float = 1.0,
                          surrogate_loss_ratio_reg: float = 1.0,
                          save_dir: str = None, device=None):
    """
    完整的训练流程：先训练对偶预测器，再训练所有机组的代理约束
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据列表
        T_delta: 时间间隔
        unit_ids: 要训练的机组ID列表（默认为所有机组）
        dual_epochs: 对偶预测器训练轮数
        dual_batch_size: 对偶预测器批次大小
        surrogate_max_iter: 代理约束BCD最大迭代次数
        surrogate_nn_epochs: 代理约束神经网络训练轮数
        save_dir: 模型保存目录（可选）
        device: 计算设备
        
    Returns:
        (dual_predictor, trainers) 元组
    """
    print("\n" + "=" * 60, flush=True)
    print("开始完整模型训练", flush=True)
    print("=" * 60, flush=True)
    
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    n_samples = len(active_set_data)
    
    if unit_ids is None:
        unit_ids = list(range(ng))
    
    print(f"\n配置信息:", flush=True)
    print(f"  - 样本数量: {n_samples}", flush=True)
    print(f"  - 机组数量: {ng} (训练{len(unit_ids)}个)", flush=True)
    print(f"  - 对偶预测器训练轮数: {dual_epochs}", flush=True)
    print(f"  - 代理约束BCD迭代次数: {surrogate_max_iter}", flush=True)
    print(f"  - 代理约束NN训练轮数/迭代: {surrogate_nn_epochs}", flush=True)
    
    # 步骤1: 训练对偶变量预测器
    print("\n" + "-" * 40, flush=True)
    print("【步骤1】训练对偶变量预测器", flush=True)
    print("-" * 40, flush=True)
    
    dual_save_path = os.path.join(save_dir, 'dual_predictor.pth') if save_dir else None
    dual_predictor = train_dual_predictor_from_data(
        ppc, active_set_data, T_delta,
        num_epochs=dual_epochs,
        batch_size=dual_batch_size,
        batch_strategy=dual_batch_strategy,
        shuffle=dual_shuffle,
        learning_rate=dual_learning_rate,
        save_path=dual_save_path, device=device
    )
    
    # 步骤2: 训练所有机组的代理约束
    print("\n" + "-" * 40, flush=True)
    print("【步骤2】训练机组代理约束", flush=True)
    print("-" * 40, flush=True)
    
    trainers = train_all_subproblem_surrogates(
        ppc, active_set_data, T_delta,
        lambda_predictor=dual_predictor, unit_ids=unit_ids,
        max_iter=surrogate_max_iter, nn_epochs=surrogate_nn_epochs,
        constraint_generation_strategy=constraint_generation_strategy,
        nn_batch_strategy=surrogate_nn_batch_strategy,
        nn_batch_size=surrogate_nn_batch_size,
        nn_shuffle=surrogate_nn_shuffle,
        nn_learning_rate=surrogate_nn_learning_rate,
        cost_learning_rate=surrogate_cost_learning_rate,
        pg_cost_lr=surrogate_pg_cost_lr,
        pg_cost_surr_lr=surrogate_pg_cost_surr_lr,
        pg_cost_reg_deadband=surrogate_pg_cost_reg_deadband,
        loss_ratio_primal=surrogate_loss_ratio_primal,
        loss_ratio_dual_pg=surrogate_loss_ratio_dual_pg,
        loss_ratio_dual_x=surrogate_loss_ratio_dual_x,
        loss_ratio_opt=surrogate_loss_ratio_opt,
        loss_ratio_reg=surrogate_loss_ratio_reg,
        save_dir=save_dir, device=device
    )
    
    print("\n" + "=" * 60, flush=True)
    print("完整模型训练完成!", flush=True)
    print("=" * 60, flush=True)
    
    return dual_predictor, trainers


def load_trained_models(ppc, active_set_data: List[Dict], T_delta: float,
                         load_dir: str, unit_ids: List[int] = None,
                         constraint_generation_strategy: str | None = None,
                         ignore_startup_shutdown_costs: bool | None = None,
                         device=None):
    """
    加载已训练的模型
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据列表
        T_delta: 时间间隔
        load_dir: 模型加载目录
        unit_ids: 要加载的机组ID列表
        device: 计算设备
        
    Returns:
        (dual_predictor, trainers) 元组
    """
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    
    if unit_ids is None:
        unit_ids = list(range(ng))
    
    print(f"从 {load_dir} 加载模型...", flush=True)
    
    # 加载对偶预测器
    dual_predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_delta, device)
    dual_path = os.path.join(load_dir, 'dual_predictor.pth')
    if os.path.exists(dual_path):
        dual_predictor.load(dual_path)
    else:
        print(f"警告: 未找到对偶预测器模型 {dual_path}", flush=True)
    
    # 加载代理约束模型
    trainers = {}
    for g in unit_ids:
        surrogate_path = os.path.join(load_dir, f'surrogate_unit_{g}.pth')
        if not os.path.exists(surrogate_path):
            print(f"警告: 未找到机组{g}代理约束模型 {surrogate_path}", flush=True)
            continue

        metadata = _load_surrogate_model_metadata(surrogate_path, device=device)
        saved_strategy = normalize_constraint_generation_strategy(
            metadata.get('constraint_generation_strategy', 'sensitive')
        )
        saved_ignore_startup_shutdown_costs = bool(
            metadata.get('ignore_startup_shutdown_costs', False)
        )
        requested_strategy = (
            saved_strategy
            if constraint_generation_strategy is None
            else normalize_constraint_generation_strategy(constraint_generation_strategy)
        )
        if requested_strategy != saved_strategy:
            raise ValueError(
                f"Surrogate model strategy mismatch for unit {g}: "
                f"requested={requested_strategy}, saved={saved_strategy}"
            )
        requested_ignore_startup_shutdown_costs = (
            saved_ignore_startup_shutdown_costs
            if ignore_startup_shutdown_costs is None
            else bool(ignore_startup_shutdown_costs)
        )
        if requested_ignore_startup_shutdown_costs != saved_ignore_startup_shutdown_costs:
            raise ValueError(
                f"Surrogate model startup/shutdown-cost setting mismatch for unit {g}: "
                f"requested={requested_ignore_startup_shutdown_costs}, "
                f"saved={saved_ignore_startup_shutdown_costs}"
            )

        requested_max_constraints = metadata.get('requested_max_constraints')
        saved_max_constraints = metadata.get('max_constraints')
        saved_num_constraints = metadata.get('num_coupling_constraints')
        trainer_max_constraints = requested_max_constraints
        if trainer_max_constraints is None:
            if saved_max_constraints is not None:
                trainer_max_constraints = saved_max_constraints
            elif saved_num_constraints is not None:
                trainer_max_constraints = saved_num_constraints
            else:
                trainer_max_constraints = 20

        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, g,
            lambda_predictor=dual_predictor,
            max_constraints=trainer_max_constraints,
            constraint_generation_strategy=requested_strategy,
            ignore_startup_shutdown_costs=requested_ignore_startup_shutdown_costs,
            nn_hidden_dims=metadata.get('nn_hidden_dims'),
            device=device,
        )
        trainer.load(surrogate_path)
        trainers[g] = trainer
    
    print(f"✓ 模型加载完成", flush=True)
    return dual_predictor, trainers


def evaluate_trained_models(dual_predictor: DualVariablePredictorTrainer,
                            trainers: Dict[int, SubproblemSurrogateTrainer],
                            active_set_data: List[Dict], n_eval_samples: int = 5):
    """
    评估已训练模型的效果
    
    Args:
        dual_predictor: 对偶变量预测器
        trainers: 代理约束训练器字典
        active_set_data: 活动集数据
        n_eval_samples: 评估样本数量
    """
    print("\n" + "=" * 60, flush=True)
    print("模型评估", flush=True)
    print("=" * 60, flush=True)
    
    n_eval = min(n_eval_samples, len(active_set_data))
    
    # 1. 评估对偶预测器
    print("\n--- 对偶变量预测器评估 ---", flush=True)
    total_mse = 0.0
    total_mae = 0.0
    
    for sample_id in range(n_eval):
        sample = active_set_data[sample_id]
        lambda_pred = np.asarray(dual_predictor.predict(sample), dtype=float).reshape(-1)
        lambda_true = np.asarray(dual_predictor.lambda_true[sample_id], dtype=float).reshape(-1)
        
        mse = np.mean((lambda_pred - lambda_true) ** 2)
        mae = np.mean(np.abs(lambda_pred - lambda_true))
        total_mse += mse
        total_mae += mae
    
    print(f"  平均MSE: {total_mse / n_eval:.6f}", flush=True)
    print(f"  平均MAE: {total_mae / n_eval:.6f}", flush=True)
    
    # 2. 评估代理约束
    print("\n--- 代理约束评估 ---", flush=True)
    
    for g, trainer in trainers.items():
        total_gap_without = 0.0
        total_gap_with = 0.0
        feasible_count = 0
        
        for sample_id in range(n_eval):
            lambda_val = trainer.lambda_vals[sample_id]
            alpha = trainer.alpha_values[sample_id]
            beta = trainer.beta_values[sample_id]
            
            x_true = active_set_data[sample_id].get('x_true', None)
            
            # 无代理约束
            x_without = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, None, None)
            gap_without = np.sum(np.abs(x_without - x_true))
            total_gap_without += gap_without
            
            # 有代理约束
            x_with = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, alpha, beta)
            gap_with = np.sum(np.abs(x_with - x_true))
            total_gap_with += gap_with
            
            # 真实解可行性
            unit_commitment = active_set_data[sample_id].get('unit_commitment_matrix', None)
            if unit_commitment is not None and g < unit_commitment.shape[0]:
                x_target = unit_commitment[g]
                if np.sum(alpha * x_target) <= beta + 1e-6:
                    feasible_count += 1
        
        avg_gap_without = total_gap_without / n_eval
        avg_gap_with = total_gap_with / n_eval
        gap_reduction = (avg_gap_without - avg_gap_with) / max(avg_gap_without, 1e-6) * 100
        
        print(f"\n  机组 {g}:", flush=True)
        print(f"    绝对间隙 (无代理): {avg_gap_without:.4f}", flush=True)
        print(f"    绝对间隙 (有代理): {avg_gap_with:.4f}", flush=True)
        print(f"    间隙减少: {gap_reduction:.2f}%", flush=True)


def train_from_json_file(json_filepath: str, ppc, T_delta: float = 1.0,
                          unit_ids: List[int] = None, save_dir: str = None,
                          dual_epochs: int = 100, surrogate_max_iter: int = 20,
                          surrogate_nn_epochs: int = 10, device=None):
    """
    从JSON文件加载数据并训练模型
    
    Args:
        json_filepath: JSON数据文件路径
        ppc: PyPower案例数据
        T_delta: 时间间隔
        unit_ids: 要训练的机组ID列表
        save_dir: 模型保存目录
        dual_epochs: 对偶预测器训练轮数
        surrogate_max_iter: 代理约束BCD迭代次数
        surrogate_nn_epochs: 代理约束NN训练轮数
        device: 计算设备
        
    Returns:
        (dual_predictor, trainers) 元组
    """
    print(f"从JSON文件加载数据: {json_filepath}", flush=True)
    
    # 加载数据
    active_set_data = load_active_set_from_json(json_filepath)
    print(f"加载了 {len(active_set_data)} 个样本", flush=True)
    
    # 训练模型
    dual_predictor, trainers = train_complete_model(
        ppc, active_set_data, T_delta,
        unit_ids=unit_ids,
        dual_epochs=dual_epochs,
        surrogate_max_iter=surrogate_max_iter,
        surrogate_nn_epochs=surrogate_nn_epochs,
        save_dir=save_dir,
        device=device
    )
    
    # 评估模型
    evaluate_trained_models(dual_predictor, trainers, active_set_data)
    
    return dual_predictor, trainers


# ========================== 测试代码 ==========================

def test_dual_predictor(ppc=None, active_set_data=None, save_path: str = None):
    """
    测试对偶变量预测器
    
    Args:
        ppc: PyPower案例数据（如果为None则使用case30）
        active_set_data: 活动集数据（如果为None则生成）
        save_path: 模型保存路径
        
    Returns:
        训练好的预测器
    """
    if not PYPOWER_AVAILABLE:
        print("pypower未安装，跳过测试", flush=True)
        return None
    
    print("\n" + "=" * 60)
    print("测试1: 对偶变量预测器训练")
    print("=" * 60)
    
    # 准备数据
    if ppc is None:
        ppc = pypower.case30.case30()
    
    # 创建并训练预测器
    print("\n--- 初始化预测器 ---", flush=True)
    predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_delta=1.0)
    
    print("\n--- 开始训练 ---", flush=True)
    predictor.train(num_epochs=100, batch_size=8)
    
    # 评估预测效果
    print("\n--- 评估预测效果 ---", flush=True)
    total_mse = 0.0
    total_mae = 0.0
    
    for sample_id in range(min(5, len(active_set_data))):
        test_pd = active_set_data[sample_id]['pd_data']
        lambda_pred = np.asarray(predictor.predict(test_pd), dtype=float).reshape(-1)
        lambda_true = np.asarray(predictor.lambda_true[sample_id], dtype=float).reshape(-1)
        
        mse = np.mean((lambda_pred - lambda_true) ** 2)
        mae = np.mean(np.abs(lambda_pred - lambda_true))
        total_mse += mse
        total_mae += mae
        
        if sample_id < 3:
            print(f"\n  样本 {sample_id}:", flush=True)
            print(f"    预测: {lambda_pred[:4]}... (前4个时段)", flush=True)
            print(f"    真值: {lambda_true[:4]}...", flush=True)
            print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}", flush=True)
    
    avg_mse = total_mse / min(5, len(active_set_data))
    avg_mae = total_mae / min(5, len(active_set_data))
    print(f"\n  平均MSE: {avg_mse:.6f}", flush=True)
    print(f"  平均MAE: {avg_mae:.6f}", flush=True)
    
    # 保存模型
    if save_path:
        predictor.save(save_path)
    
    print("\n✓ 对偶变量预测器测试完成", flush=True)
    return predictor


def test_subproblem_surrogate(ppc=None, active_set_data=None, lambda_predictor=None,
                              unit_id: int = 0, save_path: str = None):
    """
    测试子问题代理约束训练
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据
        lambda_predictor: 已训练的对偶变量预测器
        unit_id: 测试的机组ID
        save_path: 模型保存路径
        
    Returns:
        训练好的代理约束训练器
    """
    if not PYPOWER_AVAILABLE:
        print("pypower未安装，跳过测试", flush=True)
        return None
    
    print("\n" + "=" * 60)
    print(f"测试2: 机组{unit_id}子问题代理约束训练 (BCD方式)")
    print("=" * 60)
    
    # 准备数据
    if ppc is None:
        ppc = pypower.case30.case30()
    
    T = 8
    if active_set_data is None:
        active_set_data = generate_test_data(ppc, T=T, n_samples=15)
    
    # 创建训练器
    print("\n--- 初始化代理约束训练器 ---", flush=True)
    trainer = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta=1.0, unit_id=unit_id,
        lambda_predictor=lambda_predictor
    )
    
    # BCD迭代训练
    print("\n--- 开始BCD迭代训练 ---", flush=True)
    trainer.iter(max_iter=15, nn_epochs=8)
    
    # 评估代理约束效果
    print("\n--- 评估代理约束效果 ---", flush=True)
    evaluate_surrogate_effectiveness(trainer, active_set_data)
    
    # 保存模型
    if save_path:
        trainer.save(save_path)
    
    print(f"\n✓ 机组{unit_id}代理约束训练测试完成", flush=True)
    return trainer


def evaluate_surrogate_effectiveness(trainer: SubproblemSurrogateTrainer, active_set_data: List[Dict]):
    """
    评估代理约束的有效性
    
    比较有无代理约束时的LP松弛解质量
    """
    g = trainer.unit_id
    T = trainer.T
    
    total_integrality_gap_without = 0.0
    total_integrality_gap_with = 0.0
    total_constraint_violation = 0.0
    target_feasibility_rate = 0.0
    
    n_test = min(5, len(active_set_data))
    
    for sample_id in range(n_test):
        lambda_val = trainer.lambda_vals[sample_id]
        alpha = trainer.alpha_values[sample_id]
        beta = trainer.beta_values[sample_id]
        
        # 获取真实的机组状态
        unit_commitment = active_set_data[sample_id].get('unit_commitment_matrix', None)
        if unit_commitment is not None and g < unit_commitment.shape[0]:
            x_target = unit_commitment[g]
        else:
            x_target = None
        
        x_true = active_set_data[sample_id].get('x_true', None)
        # 1. 无代理约束的LP松弛
        x_without = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, None, None)
        integrality_gap_without = np.sum(np.abs(x_without - x_true))
        total_integrality_gap_without += integrality_gap_without
        
        # 2. 有代理约束的LP松弛
        x_with = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, alpha, beta)
        integrality_gap_with = np.sum(np.abs(x_with - x_true))
        total_integrality_gap_with += integrality_gap_with
        
        # 3. 代理约束违反量
        constraint_viol = max(0, np.sum(alpha * x_with) - beta)
        total_constraint_violation += constraint_viol
        
        if sample_id < 3:
            print(f"\n  样本 {sample_id}:", flush=True)
            print(f"    无代理约束绝对间隙: {integrality_gap_without:.4f}", flush=True)
            print(f"    有代理约束绝对间隙: {integrality_gap_with:.4f}", flush=True)
            print(f"    代理约束违反量: {constraint_viol:.6f}", flush=True)
    
    avg_gap_without = total_integrality_gap_without / n_test
    avg_gap_with = total_integrality_gap_with / n_test
    avg_violation = total_constraint_violation / n_test
    
    print(f"\n  === 总体评估 ===", flush=True)
    print(f"  平均整数性间隙 (无代理约束): {avg_gap_without:.4f}", flush=True)
    print(f"  平均整数性间隙 (有代理约束): {avg_gap_with:.4f}", flush=True)
    print(f"  间隙减少: {(avg_gap_without - avg_gap_with) / max(avg_gap_without, 1e-6) * 100:.2f}%", flush=True)
    print(f"  平均代理约束违反量: {avg_violation:.6f}", flush=True)


def solve_subproblem_LP_simple(trainer: SubproblemSurrogateTrainer, sample_id: int,
                               lambda_val: np.ndarray, alpha: np.ndarray, beta: float) -> np.ndarray:
    """
    求解简单的子问题LP松弛
    
    Returns:
        x的LP松弛解
    """
    g = trainer.unit_id
    T = trainer.T
    
    model = gp.Model('subproblem_LP_simple')
    model.Params.OutputFlag = 0
    
    pg = model.addVars(T, lb=0, name='pg')
    x = model.addVars(T, lb=0, ub=1, name='x')
    cpower = model.addVars(T, lb=0, name='cpower')
    
    # 发电上下限约束
    for t in range(T):
        model.addConstr(pg[t] >= trainer.gen[g, PMIN] * x[t])
        model.addConstr(pg[t] <= trainer.gen[g, PMAX] * x[t])
    
    # 爬坡约束
    Ru = 0.4 * trainer.gen[g, PMAX] / trainer.T_delta
    Rd = 0.4 * trainer.gen[g, PMAX] / trainer.T_delta
    for t in range(1, T):
        model.addConstr(pg[t] - pg[t-1] <= Ru * x[t-1] + trainer.gen[g, PMAX] * (1 - x[t-1]))
        model.addConstr(pg[t-1] - pg[t] <= Rd * x[t] + trainer.gen[g, PMAX] * (1 - x[t]))
    
    # 最小开关机时间约束
    Ton = min(4, T)
    Toff = min(4, T)
    for tau in range(1, Ton+1):
        for t1 in range(T - tau):
            model.addConstr(x[t1+1] - x[t1] <= x[t1+tau])
    for tau in range(1, Toff+1):
        for t1 in range(T - tau):
            model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau])
    
    # 发电成本
    for t in range(T):
        model.addConstr(cpower[t] >= trainer.gencost[g, -2]/trainer.T_delta * pg[t] + 
                      trainer.gencost[g, -1]/trainer.T_delta * x[t])
    
    # 代理约束
    if alpha is not None and beta is not None:
        model.addConstr(gp.quicksum(alpha[t] * x[t] for t in range(T)) <= beta)
    
    # 目标函数
    obj = gp.quicksum(cpower[t] for t in range(T))
    obj -= gp.quicksum(lambda_val[t] * pg[t] for t in range(T))
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return np.array([x[t].X for t in range(T)])
    else:
        return np.zeros(T)


def test_multi_unit_surrogate(ppc=None, active_set_data=None, lambda_predictor=None,
                              unit_ids: List[int] = None, save_dir: str = None):
    """
    测试多机组代理约束训练
    
    Args:
        ppc: PyPower案例数据
        active_set_data: 活动集数据
        lambda_predictor: 已训练的对偶变量预测器
        unit_ids: 要训练的机组ID列表
        save_dir: 模型保存目录
        
    Returns:
        训练好的代理约束训练器字典
    """
    if not PYPOWER_AVAILABLE:
        print("pypower未安装，跳过测试", flush=True)
        return None
    
    print("\n" + "=" * 60)
    print("测试3: 多机组代理约束训练")
    print("=" * 60)
    
    # 准备数据
    if ppc is None:
        ppc = pypower.case30.case30()
    
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    
    T = 8
    
    if unit_ids is None:
        unit_ids = list(range(min(3, ng)))  # 默认训练前3个机组
    
    trainers = {}
    
    for g in unit_ids:
        print(f"\n--- 机组 {g} ---", flush=True)
        
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta=1.0, unit_id=g,
            lambda_predictor=lambda_predictor
        )
        
        trainer.iter(max_iter=10, nn_epochs=5)
        trainers[g] = trainer
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))
    
    print(f"\n✓ 多机组代理约束训练完成 ({len(unit_ids)} 个机组)", flush=True)
    return trainers


def test_save_load(ppc=None, active_set_data=None):
    """
    测试模型保存和加载功能
    """
    if not PYPOWER_AVAILABLE or not TORCH_AVAILABLE:
        print("依赖未安装，跳过测试", flush=True)
        return
    
    print("\n" + "=" * 60)
    print("测试4: 模型保存和加载")
    print("=" * 60)
    
    # 准备数据
    if ppc is None:
        ppc = pypower.case30.case30()
    
    T = 8
    if active_set_data is None:
        active_set_data = generate_test_data(ppc, T=T, n_samples=10)
    
    # 创建临时目录
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. 训练并保存对偶预测器
        print("\n--- 训练对偶预测器 ---", flush=True)
        predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_delta=1.0)
        predictor.train(num_epochs=30)
        
        dual_path = os.path.join(temp_dir, 'dual_predictor.pth')
        predictor.save(dual_path)
        
        # 2. 加载对偶预测器并验证
        print("\n--- 加载并验证对偶预测器 ---", flush=True)
        predictor2 = DualVariablePredictorTrainer(ppc, active_set_data, T_delta=1.0)
        predictor2.load(dual_path)
        
        # 验证预测结果一致
        test_pd = active_set_data[0]['pd_data']
        pred1 = predictor.predict(test_pd)
        pred2 = predictor2.predict(test_pd)
        diff = np.max(np.abs(pred1 - pred2))
        print(f"  对偶预测器加载验证: 最大差异 = {diff:.8f}", flush=True)
        assert diff < 1e-5, "对偶预测器加载失败"
        
        # 3. 训练并保存代理约束
        print("\n--- 训练代理约束 ---", flush=True)
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta=1.0, unit_id=0,
            lambda_predictor=predictor
        )
        trainer.iter(max_iter=5, nn_epochs=3)
        
        surrogate_path = os.path.join(temp_dir, 'surrogate_unit_0.pth')
        trainer.save(surrogate_path)
        
        # 4. 加载代理约束并验证
        print("\n--- 加载并验证代理约束 ---", flush=True)
        trainer2 = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta=1.0, unit_id=0,
            lambda_predictor=predictor
        )
        trainer2.load(surrogate_path)
        
        # 验证代理约束参数一致
        alpha1, beta1, gamma1, delta1, costs1, pg_costs1 = trainer.get_surrogate_params(test_pd, trainer.lambda_vals[0])
        alpha2, beta2, gamma2, delta2, costs2, pg_costs2 = trainer2.get_surrogate_params(test_pd, trainer2.lambda_vals[0])
        diff_alpha = np.max(np.abs(alpha1 - alpha2))
        diff_beta = np.max(np.abs(beta1 - beta2))
        diff_gamma = np.max(np.abs(gamma1 - gamma2))
        diff_delta = np.max(np.abs(delta1 - delta2))
        diff_costs = np.max(np.abs(costs1 - costs2))
        diff_pg_costs = np.max(np.abs(pg_costs1 - pg_costs2))
        print(
            "  代理约束加载验证: "
            f"alpha差异 = {diff_alpha:.8f}, "
            f"beta差异 = {diff_beta:.8f}, "
            f"gamma差异 = {diff_gamma:.8f}, "
            f"delta差异 = {diff_delta:.8f}, "
            f"cost差异 = {diff_costs:.8f}",
            flush=True
        )
        assert max(diff_alpha, diff_beta, diff_gamma, diff_delta, diff_costs, diff_pg_costs) < 1e-5, "代理约束加载失败"
        
        print("\n✓ 模型保存和加载测试通过", flush=True)
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_end_to_end(case_name: str = 'case30', n_samples: int = 20, 
                    num_units: int = 3, save_dir: str = None):
    """
    端到端完整测试
    
    Args:
        case_name: PyPower案例名称 ('case14', 'case30', 'case39')
        n_samples: 样本数量
        num_units: 训练的机组数量
        save_dir: 模型保存目录
    """
    if not PYPOWER_AVAILABLE or not TORCH_AVAILABLE:
        print("依赖未安装，跳过测试", flush=True)
        return
    
    print("\n" + "=" * 60)
    print(f"端到端完整测试 ({case_name}, {n_samples}样本, {num_units}机组)")
    print("=" * 60)
    
    # 1. 加载案例
    if case_name == 'case14':
        ppc = pypower.case14.case14()
    elif case_name == 'case30':
        ppc = pypower.case30.case30()
    elif case_name == 'case39':
        ppc = pypower.case39.case39()
    elif case_name == 'case3' or case_name == 'case118':
        ppc = _load_demo_ppc(case_name)
    elif case_name == 'case3' or case_name == 'case118':
        ppc = _load_demo_ppc(case_name)
    else:
        print(f"未知案例: {case_name}", flush=True)
        return
    
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    num_units = min(num_units, ng)
    
    # 2. 生成数据
    print("\n【步骤1】生成测试数据", flush=True)
    active_set_data = generate_test_data(ppc, T=8, n_samples=n_samples)
    
    # 3. 训练对偶预测器
    print("\n【步骤2】训练对偶变量预测器", flush=True)
    dual_predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_delta=1.0)
    dual_predictor.train(num_epochs=100)
    
    # 4. 训练多机组代理约束
    print("\n【步骤3】训练多机组代理约束", flush=True)
    trainers = {}
    for g in range(num_units):
        print(f"\n  --- 机组 {g}/{num_units-1} ---", flush=True)
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta=1.0, unit_id=g,
            lambda_predictor=dual_predictor
        )
        trainer.iter(max_iter=15, nn_epochs=8)
        trainers[g] = trainer
    
    # 5. 评估整体效果
    print("\n【步骤4】整体效果评估", flush=True)
    total_gap_reduction = 0.0
    total_feasibility = 0.0
    
    for g, trainer in trainers.items():
        print(f"\n  机组 {g}:", flush=True)
        
        gap_without_sum = 0.0
        gap_with_sum = 0.0
        feasible_count = 0
        
        for sample_id in range(min(5, n_samples)):
            lambda_val = trainer.lambda_vals[sample_id]
            alpha = trainer.alpha_values[sample_id]
            beta = trainer.beta_values[sample_id]
            
            x_without = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, None, None)
            x_with = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, alpha, beta)
            
            gap_without = np.sum(x_without * (1 - x_without))
            gap_with = np.sum(x_with * (1 - x_with))
            
            gap_without_sum += gap_without
            gap_with_sum += gap_with
            
            # 检查真实解可行性
            unit_commitment = active_set_data[sample_id].get('unit_commitment_matrix', None)
            if unit_commitment is not None and g < unit_commitment.shape[0]:
                x_target = unit_commitment[g]
                if np.sum(alpha * x_target) <= beta + 1e-6:
                    feasible_count += 1
        
        n_test = min(5, n_samples)
        avg_gap_without = gap_without_sum / n_test
        avg_gap_with = gap_with_sum / n_test
        gap_reduction = (avg_gap_without - avg_gap_with) / max(avg_gap_without, 1e-6) * 100
        feasibility_rate = feasible_count / n_test * 100
        
        print(f"    整数性间隙减少: {gap_reduction:.2f}%", flush=True)
        print(f"    真实解可行率: {feasibility_rate:.1f}%", flush=True)
        
        total_gap_reduction += gap_reduction
        total_feasibility += feasibility_rate
    
    print(f"\n  === 平均结果 ===", flush=True)
    print(f"  平均整数性间隙减少: {total_gap_reduction / num_units:.2f}%", flush=True)
    print(f"  平均真实解可行率: {total_feasibility / num_units:.1f}%", flush=True)
    
    # 6. 保存模型
    if save_dir:
        print(f"\n【步骤5】保存模型到 {save_dir}", flush=True)
        os.makedirs(save_dir, exist_ok=True)
        
        dual_predictor.save(os.path.join(save_dir, 'dual_predictor.pth'))
        for g, trainer in trainers.items():
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))
        
        print("✓ 模型保存完成", flush=True)
    
    print("\n" + "=" * 60)
    print("端到端测试完成!")
    print("=" * 60)
    
    return dual_predictor, trainers


def main():
    """主函数"""
    print("=" * 60)
    print("子代理模型训练模块")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("错误: PyTorch未安装", flush=True)
        return
    
    if not PYPOWER_AVAILABLE:
        print("错误: pypower未安装", flush=True)
        return
    
    # 选择运行模式
    print("\n可用模式:")
    print("  === 训练模式 ===")
    print("  1. 完整训练 (对偶预测器 + 所有机组代理约束)")
    print("  2. 仅训练对偶变量预测器")
    print("  3. 仅训练指定机组代理约束")
    print("  === 测试模式 ===")
    print("  4. 对偶变量预测器测试")
    print("  5. 单机组代理约束测试")
    print("  6. 多机组代理约束测试")
    print("  7. 模型保存/加载测试")
    print("  8. 端到端完整测试")
    print("  9. 运行所有测试")
    
    # 默认运行完整训练
    mode = 1
    
    # 计算项目根目录（基于当前脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # src的父目录即为项目根目录
    result_dir = os.path.join(project_root, 'result', 'subproblem_models')
    
    # #region agent log
    import json as _json_debug; _log_path = r'd:\0-python_workspace\branchandcut\.cursor\debug.log'; _log_data = {"location": "uc_NN_subproblem.py:main:2147", "message": "Path calculation", "data": {"script_dir": script_dir, "project_root": project_root, "result_dir": result_dir, "cwd": os.getcwd()}, "timestamp": int(__import__('time').time()*1000), "sessionId": "debug-session", "hypothesisId": "D"}; open(_log_path, 'a', encoding='utf-8').write(_json_debug.dumps(_log_data) + '\n')
    # #endregion
    
    # ==================== 训练模式 ====================
    if mode == 1:
        # 完整训练
        print("\n>>> 完整训练模式 <<<\n")
        
        # 配置参数
        case_name = 'case30'  # 可选: 'case14', 'case30', 'case39'
        n_samples = 20
        T = 24
        T_delta = 1.0
        unit_ids = [0]  # None表示所有机组，或指定如 [0, 1, 2]
        save_dir = result_dir  # 使用绝对路径
        
        # 训练参数
        dual_epochs = 100
        dual_batch_size = 8
        surrogate_max_iter = 20
        surrogate_nn_epochs = 10
        
        # 加载案例
        if case_name == 'case14':
            ppc = pypower.case14.case14()
        elif case_name == 'case30':
            ppc = pypower.case30.case30()
        elif case_name == 'case39':
            ppc = pypower.case39.case39()
        elif case_name == 'case3' or case_name == 'case118':
            ppc = _load_demo_ppc(case_name)
        else:
            print(f"未知案例: {case_name}")
            return
        
        # 生成训练数据
        active_set_data = generate_test_data(ppc, T=T, n_samples=n_samples)
        
        # 完整训练
        dual_predictor, trainers = train_complete_model(
            ppc, active_set_data, T_delta,
            unit_ids=unit_ids,
            dual_epochs=dual_epochs,
            dual_batch_size=dual_batch_size,
            surrogate_max_iter=surrogate_max_iter,
            surrogate_nn_epochs=surrogate_nn_epochs,
            save_dir=save_dir
        )
        
        # 评估模型
        evaluate_trained_models(dual_predictor, trainers, active_set_data)
        
    elif mode == 2:
        # 仅训练对偶预测器
        print("\n>>> 仅训练对偶变量预测器 <<<\n")
        
        ppc = _load_demo_ppc('case30')
        active_set_data = generate_test_data(ppc, T=8, n_samples=20)
        
        predictor = train_dual_predictor_from_data(
            ppc, active_set_data, T_delta=1.0,
            num_epochs=100, batch_size=8,
            save_path=os.path.join(result_dir, 'dual_predictor.pth')
        )
        
    elif mode == 3:
        # 仅训练指定机组代理约束
        print("\n>>> 仅训练指定机组代理约束 <<<\n")
        
        ppc = _load_demo_ppc('case30')
        active_set_data = generate_test_data(ppc, T=8, n_samples=20)
        
        # 先训练对偶预测器
        predictor = train_dual_predictor_from_data(
            ppc, active_set_data, T_delta=1.0, num_epochs=100
        )
        
        # 训练指定机组
        unit_id = 0
        trainer = train_subproblem_surrogate_from_data(
            ppc, active_set_data, unit_id=unit_id, T_delta=1.0,
            lambda_predictor=predictor,
            max_iter=20, nn_epochs=10,
            save_path=os.path.join(result_dir, f'surrogate_unit_{unit_id}.pth')
        )
    
    # ==================== 测试模式 ====================
    elif mode == 4:
        test_dual_predictor()
        
    elif mode == 5:
        predictor = test_dual_predictor()
        test_subproblem_surrogate(lambda_predictor=predictor)
        
    elif mode == 6:
        predictor = test_dual_predictor()
        test_multi_unit_surrogate(lambda_predictor=predictor)
        
    elif mode == 7:
        test_save_load()
        
    elif mode == 8:
        test_end_to_end(case_name='case30', n_samples=15, num_units=3)
        
    elif mode == 9:
        # 运行所有测试
        print("\n>>> 运行所有测试 <<<\n")
        
        # 生成共享数据
        ppc = _load_demo_ppc('case30')
        active_set_data = generate_test_data(ppc, T=8, n_samples=15)
        
        # 测试1: 对偶预测器
        predictor = test_dual_predictor(ppc, active_set_data)
        
        # 测试2: 单机组代理约束
        test_subproblem_surrogate(ppc, active_set_data, predictor, unit_id=0)
        
        # 测试3: 多机组代理约束
        test_multi_unit_surrogate(ppc, active_set_data, predictor, unit_ids=[0, 1])
        
        # 测试4: 保存/加载
        test_save_load(ppc, active_set_data)
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
    
    else:
        print(f"未知模式: {mode}")


def run_training(case_name: str = 'case30', n_samples: int = 20, T: int = 8,
                 unit_ids: List[int] = None, save_dir: str = '../result/subproblem_models',
                 dual_epochs: int = 100, surrogate_max_iter: int = 20,
                 surrogate_nn_epochs: int = 10,
                 surrogate_loss_ratio_primal: float = 1.0,
                 surrogate_loss_ratio_dual_pg: float = 1.0,
                 surrogate_loss_ratio_dual_x: float = 1.0,
                 surrogate_loss_ratio_opt: float = 1.0,
                 surrogate_loss_ratio_reg: float = 1.0):
    """
    便捷的训练入口函数
    
    Args:
        case_name: PyPower案例名称 ('case14', 'case30', 'case39')
        n_samples: 样本数量
        T: 时段数
        unit_ids: 要训练的机组ID列表（None表示所有机组）
        save_dir: 模型保存目录
        dual_epochs: 对偶预测器训练轮数
        surrogate_max_iter: 代理约束BCD迭代次数
        surrogate_nn_epochs: 代理约束NN训练轮数
        surrogate_loss_ratio_primal: 主代理NN中 obj_primal 的额外倍率
        surrogate_loss_ratio_dual_pg: pg-cost NN中 obj_dual_pg 的额外倍率
        surrogate_loss_ratio_dual_x: 主代理NN中 obj_dual_x 的额外倍率
        surrogate_loss_ratio_opt: 主代理NN中 obj_opt 的额外倍率
        surrogate_loss_ratio_reg: NN正则项的额外倍率
        
    Returns:
        (dual_predictor, trainers) 元组
    """
    # 加载案例
    if case_name == 'case14':
        ppc = pypower.case14.case14()
    elif case_name == 'case30':
        ppc = pypower.case30.case30()
    elif case_name == 'case39':
        ppc = pypower.case39.case39()
    else:
        raise ValueError(f"未知案例: {case_name}")
    
    # 生成数据
    active_set_data = generate_test_data(ppc, T=T, n_samples=n_samples)
    
    # 训练模型
    dual_predictor, trainers = train_complete_model(
        ppc, active_set_data, T_delta=1.0,
        unit_ids=unit_ids,
        dual_epochs=dual_epochs,
        surrogate_max_iter=surrogate_max_iter,
        surrogate_nn_epochs=surrogate_nn_epochs,
        surrogate_loss_ratio_primal=surrogate_loss_ratio_primal,
        surrogate_loss_ratio_dual_pg=surrogate_loss_ratio_dual_pg,
        surrogate_loss_ratio_dual_x=surrogate_loss_ratio_dual_x,
        surrogate_loss_ratio_opt=surrogate_loss_ratio_opt,
        surrogate_loss_ratio_reg=surrogate_loss_ratio_reg,
        save_dir=save_dir
    )
    
    # 评估模型
    evaluate_trained_models(dual_predictor, trainers, active_set_data)
    
    return dual_predictor, trainers


if __name__ == "__main__":
    pass


def _dual_predictor_trainer_init(
    self,
    ppc,
    active_set_data,
    T_delta,
    device=None,
    batch_strategy: str = "full-batch",
    batch_size: int = 8,
    shuffle: bool = True,
    learning_rate: float = 1e-3,
):
    self.ppc = ppc
    ppc_int = ext2int(ppc)
    self.baseMVA = ppc_int['baseMVA']
    self.bus = ppc_int['bus']
    self.gen = ppc_int['gen']
    self.branch = ppc_int['branch']
    self.gencost = ppc_int['gencost']
    self.n_samples = len(active_set_data)
    self.T_delta = T_delta
    self.T = active_set_data[0]['pd_data'].shape[1] if isinstance(active_set_data, list) else active_set_data['pd_data'].shape[1]
    self.ng = self.gen.shape[0]
    self.nb = self.bus.shape[0]
    self.nl = self.branch.shape[0]
    self.active_set_data = active_set_data
    self.generator_injection_sensitivity = _build_generator_injection_sensitivity(ppc)

    if device is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[DualPredictor] 使用 CUDA 设备: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            self.device = torch.device('cpu')
            print("[DualPredictor] 使用 CPU 设备" + (
                "（CUDA 不可用）" if TORCH_AVAILABLE else "（PyTorch 未安装）"
            ), flush=True)
    else:
        self.device = device
        print(f"[DualPredictor] 使用指定设备: {self.device}", flush=True)

    first_sample = active_set_data[0] if isinstance(active_set_data, list) else active_set_data
    self.input_dim = len(get_feature_vector_from_sample(dict(first_sample)))
    self.output_dim = self.ng * self.T
    self._legacy_mode = None
    self.batch_strategy = normalize_nn_batch_strategy(batch_strategy)
    self.batch_size = max(int(batch_size), 1)
    self.shuffle = bool(shuffle)
    self.learning_rate = max(float(learning_rate), 1e-8)

    if TORCH_AVAILABLE:
        self.network = DualVariablePredictorNet(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    self.lambda_targets = self._solve_for_true_dual_variables()
    self.lambda_true = self.lambda_targets


def _dual_predictor_trainer_solve_true(self) -> np.ndarray:
    lambda_targets = {}
    lambda_payloads = {}
    for sample_id in range(self.n_samples):
        sample = self.active_set_data[sample_id]
        effective = _get_sample_pg_electricity_price_matrix(sample, self.T, self.ng)
        if effective is None:
            x_sol = _recover_unit_commitment_matrix(sample, self.ng, self.T)
            payload = _solve_pg_electricity_price_from_ed(
                self.ppc,
                sample['pd_data'],
                self.T_delta,
                x_sol,
                renewable_data=sample.get('renewable_data'),
                verbose=False,
            )
            effective = np.asarray(payload['lambda_pg_electricity_price'], dtype=float)
            sample['lambda_pg_electricity_price'] = effective.copy()
        else:
            effective = np.asarray(effective, dtype=float)
            payload = {
                'lambda_pg_electricity_price': effective,
            }
        lambda_payloads[sample_id] = payload
        lambda_targets[sample_id] = effective.reshape(-1)

    self.lambda_payloads = [lambda_payloads[i] for i in range(self.n_samples)]
    return np.array([lambda_targets[i] for i in range(self.n_samples)])


def _dual_predictor_trainer_extract_features(self, sample_id: int) -> np.ndarray:
    return get_feature_vector_from_sample(dict(self.active_set_data[sample_id]))


def _dual_predictor_trainer_train(
    self,
    num_epochs: int = 100,
    batch_size: int | None = None,
    batch_strategy: str | None = None,
    shuffle: bool | None = None,
    learning_rate: float | None = None,
):
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch unavailable", flush=True)
        return

    X = np.array([self._extract_features(i) for i in range(self.n_samples)])
    Y = self.lambda_targets
    X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    resolved_batch_strategy = normalize_nn_batch_strategy(
        self.batch_strategy if batch_strategy is None else batch_strategy
    )
    resolved_shuffle = self.shuffle if shuffle is None else bool(shuffle)
    resolved_learning_rate = (
        self.learning_rate if learning_rate is None else max(float(learning_rate), 1e-8)
    )
    if resolved_batch_strategy == "full-batch":
        resolved_batch_size = max(1, self.n_samples)
    else:
        resolved_batch_size = max(1, self.batch_size if batch_size is None else int(batch_size))
    self.optimizer = optim.Adam(self.network.parameters(), lr=resolved_learning_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=resolved_shuffle and self.n_samples > 1,
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
    self.network.train()

    print_interval = max(1, num_epochs // 10)
    print(f"[DualPredictor] 开始训练 (epochs={num_epochs}, batch_strategy={resolved_batch_strategy}, "
          f"batch_size={resolved_batch_size}, lr={resolved_learning_rate:.1e}, device={self.device})", flush=True)

    for _epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            self.optimizer.zero_grad()
            pred = self.network(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)

        if (_epoch + 1) % print_interval == 0 or _epoch == 0 or _epoch == num_epochs - 1:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  [DualPredictor] Epoch {_epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, LR: {current_lr:.1e}", flush=True)

    print(f"✓ 对偶变量预测器训练完成 (final_loss={epoch_loss:.6f})", flush=True)


def _dual_predictor_trainer_predict(self, pd_data, renewable_data=None, unit_id=None) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable")

    if isinstance(pd_data, dict):
        features = get_feature_vector_from_sample(dict(pd_data))
    else:
        features = get_feature_vector(pd_data, renewable_data=renewable_data)
    pd_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

    with torch.no_grad():
        if self._legacy_mode == 'per_unit_effective':
            if unit_id is not None:
                self.legacy_networks[unit_id].eval()
                pred = self.legacy_networks[unit_id](pd_tensor.unsqueeze(0)).squeeze(0)
                return pred.cpu().numpy()

            preds = []
            for network in self.legacy_networks:
                network.eval()
                preds.append(network(pd_tensor.unsqueeze(0)).squeeze(0).cpu().numpy())
            return np.array(preds, dtype=float)

        if self._legacy_mode == 'power_balance_only':
            self.network.eval()
            pred = self.network(pd_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
            if unit_id is not None:
                return pred
            return np.tile(pred, (self.ng, 1))

        if self._legacy_mode == 'global_dual_payload':
            self.network.eval()
            packed = self.network(pd_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
            payload = _unpack_global_dual_targets(
                packed,
                self.T,
                self.nl,
                self.generator_injection_sensitivity,
            )
            effective = np.asarray(payload['lambda_pg_effective'], dtype=float)
            if unit_id is not None:
                return effective[unit_id]
            return effective

        self.network.eval()
        packed = self.network(pd_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
        payload = packed.reshape(self.ng, self.T)
        if unit_id is not None:
            return payload[unit_id]
        return payload


def _dual_predictor_trainer_save(self, filepath: str):
    if TORCH_AVAILABLE:
        dirpath = os.path.dirname(os.path.abspath(filepath))
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        state = {
            'ng': self.ng,
            'nl': self.nl,
            'T': self.T,
        }
        if self._legacy_mode == 'per_unit_effective':
            state['network_state_dicts'] = [
                network.state_dict() for network in self.legacy_networks
            ]
            state['optimizer_state_dicts'] = [
                optimizer.state_dict() for optimizer in self.legacy_optimizers
            ]
        else:
            state['network_state_dict'] = self.network.state_dict()
            state['optimizer_state_dict'] = self.optimizer.state_dict()
            state['lambda_output_dim'] = self.T if self._legacy_mode == 'power_balance_only' else self.output_dim
        state['learning_rate'] = getattr(self, 'learning_rate', 1e-3)
        state['batch_strategy'] = getattr(self, 'batch_strategy', 'full-batch')
        state['batch_size'] = getattr(self, 'batch_size', 8)
        state['shuffle'] = getattr(self, 'shuffle', True)
        torch.save(state, filepath)


def _dual_predictor_trainer_load(self, filepath: str):
    if TORCH_AVAILABLE:
        state = torch.load(filepath, map_location=self.device)
        self.learning_rate = max(float(state.get('learning_rate', getattr(self, 'learning_rate', 1e-3))), 1e-8)
        self.batch_strategy = normalize_nn_batch_strategy(
            state.get('batch_strategy', getattr(self, 'batch_strategy', 'full-batch'))
        )
        self.batch_size = max(int(state.get('batch_size', getattr(self, 'batch_size', 8))), 1)
        self.shuffle = bool(state.get('shuffle', getattr(self, 'shuffle', True)))
        if 'network_state_dicts' in state:
            self._legacy_mode = 'per_unit_effective'
            self.legacy_networks = [
                DualVariablePredictorNet(self.input_dim, self.T).to(self.device)
                for _ in range(self.ng)
            ]
            self.legacy_optimizers = [
                optim.Adam(network.parameters(), lr=self.learning_rate)
                for network in self.legacy_networks
            ]
            for network, network_state in zip(self.legacy_networks, state['network_state_dicts']):
                network.load_state_dict(network_state)
            for optimizer, optimizer_state in zip(
                self.legacy_optimizers,
                state.get('optimizer_state_dicts', []),
            ):
                optimizer.load_state_dict(optimizer_state)
        else:
            legacy_dim = state.get('lambda_output_dim')
            if legacy_dim == self.output_dim:
                self._legacy_mode = None
                self.network.load_state_dict(state['network_state_dict'])
                if 'optimizer_state_dict' in state:
                    self.optimizer.load_state_dict(state['optimizer_state_dict'])
            elif legacy_dim == self.T + 2 * self.nl * self.T:
                self._legacy_mode = 'global_dual_payload'
                self.network = DualVariablePredictorNet(self.input_dim, legacy_dim).to(self.device)
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
                self.network.load_state_dict(state['network_state_dict'])
                if 'optimizer_state_dict' in state:
                    self.optimizer.load_state_dict(state['optimizer_state_dict'])
            else:
                self._legacy_mode = 'power_balance_only'
                self.network = DualVariablePredictorNet(self.input_dim, self.T).to(self.device)
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
                self.network.load_state_dict(state['network_state_dict'])
                if 'optimizer_state_dict' in state:
                    self.optimizer.load_state_dict(state['optimizer_state_dict'])


def _subproblem_get_lambda_values(self) -> np.ndarray:
    if self.lambda_predictor is not None:
        lambda_vals = []
        for sample_id in range(self.n_samples):
            sample = self.active_set_data[sample_id]
            predicted = self.lambda_predictor.predict(sample)
            effective = _extract_pg_electricity_price_matrix(predicted, self.T, self.ng)
            if effective is not None:
                lambda_vals.append(effective[self.unit_id])
                continue
            predicted_arr = np.asarray(predicted, dtype=float)
            if predicted_arr.shape == (self.T,):
                lambda_vals.append(predicted_arr)
            elif predicted_arr.shape == (self.ng, self.T):
                lambda_vals.append(predicted_arr[self.unit_id])
            elif predicted_arr.shape == (self.T, self.ng):
                lambda_vals.append(predicted_arr.T[self.unit_id])
            else:
                raise ValueError(
                    f"lambda_predictor must output lambda_pg_electricity_price with shape "
                    f"(ng, T) or (T,), got {predicted_arr.shape}"
                )
        return np.array(lambda_vals)
    return self._solve_for_lambda()


def _subproblem_solve_for_lambda(self) -> np.ndarray:
    lambda_vals = {}
    for sample_id in range(self.n_samples):
        sample = self.active_set_data[sample_id]
        effective = _get_sample_pg_electricity_price_matrix(sample, self.T, self.ng)
        if effective is None:
            x_sol = _recover_unit_commitment_matrix(sample, self.ng, self.T)
            payload = _solve_pg_electricity_price_from_ed(
                self.ppc,
                sample['pd_data'],
                self.T_delta,
                x_sol,
                renewable_data=sample.get('renewable_data'),
                verbose=False,
            )
            effective = payload['lambda_pg_electricity_price']
            sample['lambda_pg_electricity_price'] = effective.copy()
        lambda_vals[sample_id] = effective[self.unit_id]

    return np.array([lambda_vals[i] for i in range(self.n_samples)])


DualVariablePredictorTrainer.__init__ = _dual_predictor_trainer_init
DualVariablePredictorTrainer._solve_for_true_dual_variables = _dual_predictor_trainer_solve_true
DualVariablePredictorTrainer._extract_features = _dual_predictor_trainer_extract_features
DualVariablePredictorTrainer.train = _dual_predictor_trainer_train
DualVariablePredictorTrainer.predict = _dual_predictor_trainer_predict
DualVariablePredictorTrainer.save = _dual_predictor_trainer_save
DualVariablePredictorTrainer.load = _dual_predictor_trainer_load

SubproblemSurrogateTrainer._get_lambda_values = _subproblem_get_lambda_values
SubproblemSurrogateTrainer._solve_for_lambda = _subproblem_solve_for_lambda


if __name__ == "__main__":
    main()
