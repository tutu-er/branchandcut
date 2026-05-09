import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
    GUROBI_IMPORT_ERROR = None
except Exception as exc:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
    GUROBI_IMPORT_ERROR = exc
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
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _require_gurobi_available(context: str = "Gurobi code path") -> None:
    if GUROBI_AVAILABLE:
        return
    raise RuntimeError(
        f"{context} requires gurobipy, but gurobipy could not be imported: "
        f"{GUROBI_IMPORT_ERROR!r}"
    )

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 允许在无 torch 环境下 import 本模块（例如仅跑 LP/数据工具脚本）。
    # 注意：任何依赖 torch 的训练/推理路径在运行时仍会报错。
    print("警告: PyTorch未安装，将无法使用神经网络功能", flush=True)
    import types

    class _TorchStub:
        def __getattr__(self, name: str):
            raise ImportError("PyTorch is required for neural network components")

    torch = _TorchStub()  # type: ignore
    optim = _TorchStub()  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    nn = types.SimpleNamespace(Module=object)  # type: ignore

# 导入必要的工具函数
from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A
try:
    from subproblem_lp_solver import (
        LP_BACKEND_CVXPY_HIGHS,
        LP_BACKEND_GUROBI,
        assert_lp_backend_available,
        normalize_lp_backend,
        _recover_unit_x_from_sample,
        solve_dual_block as solve_dual_block_backend,
        solve_ed_electricity_price,
        solve_init_lp as solve_init_lp_backend,
        solve_primal_block as solve_primal_block_backend,
    )
except ImportError:
    from src.subproblem_lp_solver import (
        LP_BACKEND_CVXPY_HIGHS,
        LP_BACKEND_GUROBI,
        assert_lp_backend_available,
        normalize_lp_backend,
        _recover_unit_x_from_sample,
        solve_dual_block as solve_dual_block_backend,
        solve_ed_electricity_price,
        solve_init_lp as solve_init_lp_backend,
        solve_primal_block as solve_primal_block_backend,
    )
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
# 额外启用 write_through，尽量确保重定向/管道下也能及时落盘。
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True, write_through=True)
except Exception:
    pass


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


def _compute_subproblem_min_up_down_tau_max(
    case_name: Optional[str],
    ppc_raw,
    gen: np.ndarray,
    T: int,
    T_delta: float,
    unit_id: int,
) -> Tuple[int, int]:
    """子问题中最小开/关时间约束的 τ 上界（与 ``uc_gurobipy._get_min_up_down_time_steps`` 一致）。

    - ``case118``：使用 ``ppc['uc_min_up_time_h']`` / ``uc_min_down_time_h`` 按 ``T_delta`` 换算
      （``ceil``，至少 1），与全模型 UC 对齐。
    - 其余算例（含 ``case3``、``case30``、``None`` 等）：保持原 ``min(4, T)`` 紧凑建模。
    """
    min_up_h = _get_custom_generator_array_from_ppc(
        ppc_raw, gen.shape[0], "uc_min_up_time_h"
    )
    min_down_h = _get_custom_generator_array_from_ppc(
        ppc_raw, gen.shape[0], "uc_min_down_time_h"
    )
    if min_up_h is not None and min_down_h is not None:
        ton = int(np.maximum(np.ceil(float(min_up_h[unit_id]) / float(T_delta)), 1))
        toff = int(np.maximum(np.ceil(float(min_down_h[unit_id]) / float(T_delta)), 1))
        return ton, toff

    if str(case_name).strip().lower() == "case118":
        if min_up_h is None or min_down_h is None:
            raise ValueError(
                "case118 子问题需要 ppc 上包含 uc_min_up_time_h 与 uc_min_down_time_h "
                "（与 load_case118_ppc_with_mti_limits 一致）。"
            )
    cap = min(max(int(4 * float(T_delta)), 1), int(T))
    return cap, cap


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
    """全矩阵恢复：与 ``_recover_unit_x_from_sample`` 一致，先 UCM 再 active_set 覆盖。"""
    if 'unit_commitment_matrix' in sample:
        uc = np.asarray(sample['unit_commitment_matrix'], dtype=float)
        rows = min(uc.shape[0], ng)
        cols = min(uc.shape[1], T)
        x_sol = np.zeros((ng, T), dtype=float)
        x_sol[:rows, :cols] = uc[:rows, :cols]
    else:
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
    try:
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
    finally:
        # Release native Gurobi resources immediately; batch convert creates
        # hundreds of ED models and relying on GC alone can thrash memory and
        # look like a hang after the first WLS license banner.
        try:
            ed.model.dispose()
        except Exception:
            pass


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


def _get_effective_pg_prices_from_sample_or_dual_payload(
    sample: Dict,
    T: int,
    ng: int,
    nl: int,
    generator_injection_sensitivity: np.ndarray,
) -> Optional[np.ndarray]:
    """Resolve per-unit electricity prices from any cached sample payload."""
    direct = _get_sample_pg_electricity_price_matrix(sample, T, ng)
    if direct is not None:
        return np.asarray(direct, dtype=float)
    lambda_field = sample.get("lambda")
    if _has_complete_effective_pg_dual(lambda_field, T, ng, nl):
        return _extract_effective_pg_dual(
            lambda_field,
            T,
            ng,
            nl,
            generator_injection_sensitivity,
        )
    return None


def _solve_pg_electricity_price_from_ed(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    renewable_data: np.ndarray | None = None,
    verbose: bool = False,
    lp_backend: str = LP_BACKEND_GUROBI,
    ignore_fixed_generation_cost: bool = False,
) -> Dict[str, np.ndarray]:
    return solve_ed_electricity_price(
        ppc,
        Pd,
        T_delta,
        x_sol,
        renewable_data=renewable_data,
        verbose=verbose,
        lp_backend=lp_backend,
        ignore_fixed_generation_cost=ignore_fixed_generation_cost,
    )


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


class DualVariablePredictorNetTemporalConv(nn.Module):
    """沿时段 T 卷积编码 (load, renew) 拼成的 (2*nb, T)，再映射到 ng*T 维电价型对偶。

    要求输入为 ``get_feature_vector_from_sample`` 的拼接顺序：
    ``load_data.reshape(-1)`` 后接 ``renewable_data.reshape(-1)``，且二者形状均为 (nb, T)。
    """

    def __init__(
        self,
        nb: int,
        T: int,
        output_dim: int,
        hidden_ch: int = 192,
        n_stacks: int = 3,
    ):
        super().__init__()
        self.nb = nb
        self.T = T
        cin = 2 * nb
        layers: List[nn.Module] = []
        c = cin
        for _ in range(int(n_stacks)):
            layers += [
                nn.Conv1d(c, hidden_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_ch),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.05),
            ]
            c = hidden_ch
        self.body = nn.Sequential(*layers)
        flat = c * T
        hid2 = min(2048, max(512, flat // 2))
        self.head = nn.Sequential(
            nn.Linear(flat, hid2),
            nn.LayerNorm(hid2),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(hid2, output_dim),
        )
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        b = x_flat.shape[0]
        x = x_flat.reshape(b, 2 * self.nb, self.T)
        h = self.body(x)
        return self.head(h.reshape(b, -1))


def _dual_predictor_batch_cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - cosine(pred, target)，按 batch 样本分别计算再取均值。"""
    pn = torch.linalg.norm(pred, dim=1).clamp(min=eps)
    tn = torch.linalg.norm(target, dim=1).clamp(min=eps)
    cos = (pred * target).sum(dim=1) / (pn * tn)
    return (1.0 - cos).mean()


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
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
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


# ========================== 第一部分·补充：单机组 0/1 变量预测器 ==========================

class SingleUnitBinaryPredictorNet(nn.Module):
    """场景特征 → 单机组 T 维 0/1 logits。

    每个实例对应一台机组。训练监督标签来自 active_set / unit_commitment_matrix
    中对应机组的 0/1 序列（最优 MIP 解的整数部分）。
    """

    def __init__(self, input_dim: int, T: int, hidden_dims: List[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, int(T)))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class SingleUnitBinaryPredictorResMLPNet(nn.Module):
    """更稳定的残差 MLP：输入投影→多层残差块→输出 T logits。"""

    def __init__(
        self,
        input_dim: int,
        T: int,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(width)),
            nn.LayerNorm(int(width)),
            nn.LeakyReLU(0.01),
            nn.Dropout(float(dropout)),
        )
        blocks = []
        for _ in range(max(0, int(depth))):
            blocks.append(_ResidualMLPBlock(int(width), dropout=float(dropout)))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(int(width), int(T))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)


class SingleUnitBinaryPredictorTemporalConvNet(nn.Module):
    """时间卷积结构：把 (load, renewable) 的 (nb, T) 特征还原后沿时间做 1D Conv。

    输入特征假设来自 `scenario_utils.get_feature_vector_from_sample`：
    - new-format: concat([load.flatten(), renewable.flatten()])
      其中 load/renewable 形状为 (nb, T)，flatten 为行优先（先 bus 后 time）。

    输出：每个 time slot 一个 logits，形状 (B, T)。
    """

    def __init__(
        self,
        input_dim: int,
        T: int,
        channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.T = int(T)
        if self.T <= 0 or self.input_dim <= 0 or (self.input_dim % self.T) != 0:
            raise ValueError(f"TemporalConv expects input_dim % T == 0, got input_dim={input_dim}, T={T}")
        per_t = self.input_dim // self.T
        if per_t % 2 != 0:
            raise ValueError(f"TemporalConv expects per-time features even (2*nb), got per_t={per_t}")
        self.nb = per_t // 2
        self.in_channels = per_t

        ch = int(max(4, channels))
        d = int(max(1, depth))
        p = float(dropout)

        layers: List[nn.Module] = []
        layers.append(nn.Conv1d(self.in_channels, ch, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Dropout(p))
        for _ in range(d - 1):
            layers.append(nn.Conv1d(ch, ch, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(p))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(ch, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        B = x.shape[0]
        nb = self.nb
        T = self.T
        # restore load/renewable time-major features: (B, T, 2*nb)
        load = x[:, : nb * T].reshape(B, nb, T).permute(0, 2, 1)
        ren = x[:, nb * T : 2 * nb * T].reshape(B, nb, T).permute(0, 2, 1)
        feat = torch.cat([load, ren], dim=2)  # (B, T, 2*nb)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, 2*nb, T)
        y = self.body(feat)
        y = self.head(y)  # (B,1,T)
        return y.squeeze(1)  # (B,T)


class _TCNResBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float = 0.1):
        super().__init__()
        ch = int(channels)
        d = int(max(1, dilation))
        p = float(dropout)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=d, dilation=d)
        self.norm1 = nn.GroupNorm(8 if ch >= 8 else 1, ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=d, dilation=d)
        self.norm2 = nn.GroupNorm(8 if ch >= 8 else 1, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class SingleUnitBinaryPredictorTCNNet(nn.Module):
    """TCN（空洞残差卷积）：沿时间维度建模，更贴近 x_hat 与真值的距离优化。"""

    def __init__(
        self,
        input_dim: int,
        T: int,
        channels: int = 64,
        depth: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.T = int(T)
        if self.T <= 0 or self.input_dim <= 0 or (self.input_dim % self.T) != 0:
            raise ValueError(f"TCN expects input_dim % T == 0, got input_dim={input_dim}, T={T}")
        per_t = self.input_dim // self.T
        if per_t % 2 != 0:
            raise ValueError(f"TCN expects per-time features even (2*nb), got per_t={per_t}")
        self.nb = per_t // 2
        self.in_channels = per_t

        ch = int(max(8, channels))
        d = int(max(1, depth))
        p = float(dropout)

        self.in_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, ch, kernel_size=1),
            nn.GroupNorm(8 if ch >= 8 else 1, ch),
            nn.SiLU(),
        )
        blocks: List[nn.Module] = []
        for i in range(d):
            blocks.append(_TCNResBlock(ch, dilation=2**i, dropout=p))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        nb = self.nb
        T = self.T
        load = x[:, : nb * T].reshape(B, nb, T).permute(0, 2, 1)
        ren = x[:, nb * T : 2 * nb * T].reshape(B, nb, T).permute(0, 2, 1)
        feat = torch.cat([load, ren], dim=2).permute(0, 2, 1).contiguous()  # (B, 2*nb, T)
        y = self.in_proj(feat)
        y = self.blocks(y)
        y = self.head(y)
        return y.squeeze(1)


class _SharedTCNBackbone(nn.Module):
    """共享 TCN 骨干：输出 (B, ch, T) 的时序特征。

    - 标准：``input_dim = 2 * nb * T``（负荷 + 可再生拼接），每步 ``per_t = 2*nb`` 为偶数。
    - 兼容：``input_dim = nb * T`` 仅负荷/净负荷展平（无独立可再生）时 ``per_t = nb`` 可能为奇数
      （例如 case3lite 三母线）：在 ``forward`` 内与全零「可再生」槽拼接，卷积输入通道数 ``2*nb``。
    """

    def __init__(self, *, input_dim: int, T: int, channels: int = 64, depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.T = int(T)
        if self.T <= 0 or self.input_dim <= 0 or (self.input_dim % self.T) != 0:
            raise ValueError(f"SharedTCN expects input_dim % T == 0, got input_dim={input_dim}, T={T}")
        per_t = self.input_dim // self.T
        if per_t % 2 == 0:
            self.nb = per_t // 2
            self._net_load_only = False
            self.in_channels = per_t
        else:
            self.nb = per_t
            self._net_load_only = True
            self.in_channels = 2 * per_t

        ch = int(max(8, channels))
        d = int(max(1, depth))
        p = float(dropout)

        self.in_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, ch, kernel_size=1),
            nn.GroupNorm(8 if ch >= 8 else 1, ch),
            nn.SiLU(),
        )
        blocks: List[nn.Module] = []
        for i in range(d):
            blocks.append(_TCNResBlock(ch, dilation=2**i, dropout=p))
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = ch

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        nb = self.nb
        T = self.T
        if getattr(self, "_net_load_only", False):
            load = x.reshape(B, nb, T).permute(0, 2, 1)
            ren = torch.zeros_like(load)
        else:
            load = x[:, : nb * T].reshape(B, nb, T).permute(0, 2, 1)
            ren = x[:, nb * T : 2 * nb * T].reshape(B, nb, T).permute(0, 2, 1)
        feat = torch.cat([load, ren], dim=2).permute(0, 2, 1).contiguous()  # (B, 2*nb, T)
        y = self.in_proj(feat)
        y = self.blocks(y)
        return y  # (B, ch, T)


class SharedTCNFiLMNet(nn.Module):
    """方案 B（正确训练形态）：共享 backbone + unit embedding FiLM + per-unit head，输出 (B, ng, T)。"""

    def __init__(self, *, input_dim: int, T: int, ng: int, channels: int = 64, depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.ng = int(ng)
        self.backbone = _SharedTCNBackbone(input_dim=input_dim, T=T, channels=channels, depth=depth, dropout=dropout)
        ch = int(getattr(self.backbone, "out_channels", channels))
        self.ch = ch
        self.emb_dim = 16
        self.unit_embedding = nn.Embedding(self.ng, self.emb_dim)
        nn.init.normal_(self.unit_embedding.weight, mean=0.0, std=0.02)
        self.film = nn.Linear(self.emb_dim, 2 * ch)
        # per-unit head: logits_g(t) = sum_c W[g,c] * h_g[c,t] + b[g]
        self.head_w = nn.Parameter(torch.zeros(self.ng, ch))
        self.head_b = nn.Parameter(torch.zeros(self.ng))
        nn.init.kaiming_normal_(self.head_w, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h: (B, ch, T)
        h = self.backbone(x)
        B, ch, T = h.shape
        # FiLM params for all units: (ng, 2ch)
        gb = self.film(self.unit_embedding.weight)
        gamma, beta = torch.split(gb, ch, dim=1)  # (ng,ch)
        gamma = gamma.view(1, self.ng, ch, 1)
        beta = beta.view(1, self.ng, ch, 1)
        # broadcast h -> (B, ng, ch, T)
        h4 = h.unsqueeze(1)
        h_mod = h4 * (1.0 + gamma) + beta
        w = self.head_w.view(1, self.ng, ch, 1)
        b = self.head_b.view(1, self.ng, 1)
        logits = torch.sum(h_mod * w, dim=2) + b  # (B, ng, T)
        return logits


class _SharedTCNFiLMUnitWrapper(nn.Module):
    """单机组视角包装：从 shared net 输出中取 unit_id 对应的 (B,T)。"""

    def __init__(self, *, unit_id: int, shared: SharedTCNFiLMNet):
        super().__init__()
        self.unit_id = int(unit_id)
        self.shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)  # (B, ng, T)
        return y[:, self.unit_id, :]


class SingleUnitBinaryPredictorTrainer:
    """管理 ng 个独立 ``SingleUnitBinaryPredictorNet`` 的批量训练器。

    - 预训练阶段：BCEWithLogitsLoss 以 ``unit_commitment_matrix`` 作为监督标签。
    - BCD 阶段：被 ``SubproblemSurrogateTrainer`` 以小学习率做端到端微调。
    - 存储：按 ``unit_id`` 懒初始化（避免为未训练机组浪费显存）。
    - 输入特征仅为 ``scenario_utils.get_feature_vector_from_sample``（负荷/可再生或净负荷），
      与主 surrogate 网络的 ``[场景, λ, unit_params]`` 拼接向量分离；BCD 内不得混入 λ 与机组静态后缀。
    """

    def __init__(
        self,
        ppc,
        active_set_data,
        T_delta,
        unit_ids: List[int] | None = None,
        hidden_dims: List[int] | None = None,
        net_variant: str = "mlp",
        resmlp_width: int = 512,
        resmlp_depth: int = 4,
        tconv_channels: int = 64,
        tconv_depth: int = 4,
        tcn_channels: int = 64,
        tcn_depth: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_strategy: str = 'full-batch',
        batch_size: int = 32,
        shuffle: bool = True,
        device=None,
    ):
        self.ppc = ppc
        ppc_int = ext2int(ppc)
        self.gen = ppc_int['gen']
        self.bus = ppc_int['bus']
        self.branch = ppc_int['branch']
        self.gencost = ppc_int['gencost']
        self.baseMVA = ppc_int['baseMVA']
        self.ng = int(self.gen.shape[0])
        self.nb = int(self.bus.shape[0])
        self.nl = int(self.branch.shape[0])
        self.n_samples = len(active_set_data)
        self.T_delta = float(T_delta)
        self.active_set_data = active_set_data
        if isinstance(active_set_data, list):
            first_sample = active_set_data[0]
        else:
            first_sample = active_set_data
        self.T = int(first_sample['pd_data'].shape[1])

        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(
                    f"[UnitPredictor] 使用 CUDA 设备: {torch.cuda.get_device_name(0)}",
                    flush=True,
                )
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.input_dim = len(get_feature_vector_from_sample(dict(first_sample)))
        self.hidden_dims = normalize_nn_hidden_dims(hidden_dims, [256, 128])
        self.net_variant = str(net_variant or "mlp").strip().lower()
        self.resmlp_width = max(1, int(resmlp_width))
        self.resmlp_depth = max(0, int(resmlp_depth))
        self.tconv_channels = max(4, int(tconv_channels))
        self.tconv_depth = max(1, int(tconv_depth))
        self.tcn_channels = max(8, int(tcn_channels))
        self.tcn_depth = max(1, int(tcn_depth))
        self.dropout = float(dropout)
        self.learning_rate = max(float(learning_rate), 1e-8)
        self.weight_decay = max(float(weight_decay), 0.0)
        self.batch_strategy = normalize_nn_batch_strategy(batch_strategy)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        if unit_ids is None:
            self.unit_ids = list(range(self.ng))
        else:
            self.unit_ids = [int(g) for g in unit_ids]

        self.networks: Dict[int, nn.Module] = {}
        self.optimizers: Dict[int, 'optim.Optimizer'] = {}
        # 共享骨干（方案 B）：仅在 net_variant == "tcn_shared_film" 时启用
        self.shared_model: SharedTCNFiLMNet | None = None
        self.shared_optimizer: 'optim.Optimizer' | None = None
        if TORCH_AVAILABLE:
            for g in self.unit_ids:
                self._ensure_network(g)

        # 共享网络需要单一优化器（避免对同一参数维护多份 Adam 状态）
        if TORCH_AVAILABLE and self.net_variant == "tcn_shared_film":
            if self.shared_model is None:
                raise RuntimeError("shared_model is not initialized for tcn_shared_film")
            self.shared_optimizer = optim.Adam(
                self.shared_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        self.x_true = self._extract_x_labels()
        pos_rate = float(self.x_true.mean()) if self.x_true.size else 0.0
        print(
            f"[UnitPredictor] 训练器就绪: ng={self.ng}, T={self.T}, "
            f"input_dim={self.input_dim}, n_samples={self.n_samples}, "
            f"unit_ids={self.unit_ids}, net={self.net_variant}, "
            f"hidden_dims={self.hidden_dims}, "
            f"pos_rate={pos_rate:.3f}, device={self.device}",
            flush=True,
        )
    def _ensure_network(self, unit_id: int) -> None:
        if not TORCH_AVAILABLE:
            return
        g_int = int(unit_id)
        if g_int in self.networks:
            return
        if self.net_variant == "tcn_shared_film":
            # 共享 backbone + embedding + FiLM
            if self.shared_model is None:
                self.shared_model = SharedTCNFiLMNet(
                    input_dim=self.input_dim,
                    T=self.T,
                    ng=self.ng,
                    channels=self.tcn_channels,
                    depth=self.tcn_depth,
                    dropout=self.dropout,
                ).to(self.device)
            net = _SharedTCNFiLMUnitWrapper(unit_id=g_int, shared=self.shared_model).to(self.device)
            self.networks[g_int] = net
            # optimizer：共享模式下只使用 self.shared_optimizer
            if g_int not in self.optimizers:
                self.optimizers[g_int] = None  # type: ignore
            if g_int not in self.unit_ids:
                self.unit_ids.append(g_int)
            return
        if self.net_variant == "resmlp":
            net = SingleUnitBinaryPredictorResMLPNet(
                input_dim=self.input_dim,
                T=self.T,
                width=self.resmlp_width,
                depth=self.resmlp_depth,
                dropout=self.dropout,
            ).to(self.device)
        elif self.net_variant == "tcn":
            net = SingleUnitBinaryPredictorTCNNet(
                input_dim=self.input_dim,
                T=self.T,
                channels=self.tcn_channels,
                depth=self.tcn_depth,
                dropout=self.dropout,
            ).to(self.device)
        elif self.net_variant in ("tconv", "temporal_conv", "conv"):
            net = SingleUnitBinaryPredictorTemporalConvNet(
                input_dim=self.input_dim,
                T=self.T,
                channels=self.tconv_channels,
                depth=self.tconv_depth,
                dropout=self.dropout,
            ).to(self.device)
        else:
            net = SingleUnitBinaryPredictorNet(
                input_dim=self.input_dim, T=self.T, hidden_dims=self.hidden_dims,
            ).to(self.device)
        self.networks[g_int] = net
        self.optimizers[g_int] = optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if g_int not in self.unit_ids:
            self.unit_ids.append(g_int)

    def _extract_x_labels(self) -> np.ndarray:
        """还原 (n_samples, ng, T) 的 0/1 监督标签。优先用 active_set，回落 unit_commitment_matrix。"""
        labels = np.zeros((self.n_samples, self.ng, self.T), dtype=np.float32)
        for sample_id in range(self.n_samples):
            sample = self.active_set_data[sample_id]
            if 'active_set' in sample and sample['active_set'] is not None:
                for item in sample['active_set']:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                        g, t = item[0]
                        g_int = int(g)
                        t_int = int(t)
                        if 0 <= g_int < self.ng and 0 <= t_int < self.T:
                            labels[sample_id, g_int, t_int] = float(item[1])
            elif 'unit_commitment_matrix' in sample:
                uc = np.asarray(sample['unit_commitment_matrix'], dtype=float)
                if uc.ndim == 2:
                    rows = min(uc.shape[0], self.ng)
                    cols = min(uc.shape[1], self.T)
                    labels[sample_id, :rows, :cols] = uc[:rows, :cols]
        return labels

    def _extract_features(self, sample_id: int) -> np.ndarray:
        return get_feature_vector_from_sample(dict(self.active_set_data[sample_id]))

    def train_unit(
        self,
        unit_id: int,
        num_epochs: int = 100,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
        enable_scheduler: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 0.0,
        enable_pos_weight: bool = False,
        pos_weight_clip: float = 20.0,
        loss_weight_bce: float = 1.0,
        loss_weight_mse: float = 0.0,
        loss_weight_l1: float = 0.0,
        loss_weight_tv: float = 0.0,
        loss_weight_transition: float = 0.0,
        loss_weight_binarize: float = 0.0,
        loss_weight_std_floor: float = 0.0,
        std_floor_scale: float = 0.5,
        loss_weight_tv_floor: float = 0.0,
        tv_floor_scale: float = 0.8,
    ) -> float | None:
        if not TORCH_AVAILABLE:
            print("[UnitPredictor] PyTorch 不可用，跳过训练", flush=True)
            return None
        self._ensure_network(unit_id)
        g_int = int(unit_id)

        resolved_batch_strategy = normalize_nn_batch_strategy(
            self.batch_strategy if batch_strategy is None else batch_strategy
        )
        resolved_shuffle = self.shuffle if shuffle is None else bool(shuffle)
        if resolved_batch_strategy == 'full-batch':
            resolved_batch_size = max(1, self.n_samples)
        else:
            base_batch = self.batch_size if batch_size is None else int(batch_size)
            resolved_batch_size = max(1, base_batch)
        resolved_lr = self.learning_rate if learning_rate is None else max(float(learning_rate), 1e-8)

        X = np.array([self._extract_features(i) for i in range(self.n_samples)])
        Y = self.x_true[:, g_int, :]
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=resolved_shuffle and self.n_samples > 1,
        )

        net = self.networks[g_int]
        if self.net_variant == "tcn_shared_film":
            # 警告：共享模型逐机组 train_unit 会导致遗忘；仅用于小规模微调/诊断
            if self.shared_optimizer is None:
                raise RuntimeError("shared_optimizer is not initialized for tcn_shared_film")
            optimizer = self.shared_optimizer
            for pg in optimizer.param_groups:
                pg["lr"] = resolved_lr
        else:
            optimizer = optim.Adam(net.parameters(), lr=resolved_lr, weight_decay=self.weight_decay)
            self.optimizers[g_int] = optimizer
        scheduler = None
        if enable_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=max(0, int(scheduler_patience)),
                factor=float(scheduler_factor),
                min_lr=max(0.0, float(scheduler_min_lr)),
            )
        criterion = nn.BCEWithLogitsLoss()
        if enable_pos_weight:
            # 针对每个 time slot 估计 pos_weight（防止极端不平衡导致梯度被负类淹没）
            y_mean = np.asarray(Y, dtype=float).mean(axis=0)  # (T,)
            eps = 1e-6
            pos_w = (1.0 - y_mean + eps) / (y_mean + eps)
            pos_w = np.clip(pos_w, 1.0, float(pos_weight_clip))
            pos_w_tensor = torch.tensor(pos_w, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
        net.train()

        print_interval = max(1, num_epochs // 10)
        print(
            f"[UnitPredictor-{g_int}] 开始预训练 (epochs={num_epochs}, "
            f"batch_size={resolved_batch_size}, lr={resolved_lr:.1e}, "
            f"pos_rate={float(Y.mean()):.3f})",
            flush=True,
        )
        epoch_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                logits = net(batch_X)
                # 主目标：缩小 x_hat 与真值 x 的距离；同时保留 BCE 以稳定概率学习
                loss = 0.0
                if float(loss_weight_bce) != 0.0:
                    loss = loss + float(loss_weight_bce) * criterion(logits, batch_Y)
                probs = torch.sigmoid(logits)
                if float(loss_weight_mse) != 0.0:
                    loss = loss + float(loss_weight_mse) * torch.mean((probs - batch_Y) ** 2)
                if float(loss_weight_l1) != 0.0:
                    loss = loss + float(loss_weight_l1) * torch.mean(torch.abs(probs - batch_Y))
                if float(loss_weight_tv) != 0.0:
                    # 时间平滑：惩罚相邻时段输出抖动（更贴近启停序列形态）
                    loss = loss + float(loss_weight_tv) * torch.mean(torch.abs(probs[:, 1:] - probs[:, :-1]))
                if float(loss_weight_tv_floor) != 0.0:
                    # 反“均值塌陷”：要求预测的 TV >= scale * 真值 TV（仅在真值 TV>0 时起作用）
                    eps = 1e-6
                    dy_true = torch.abs(batch_Y[:, 1:] - batch_Y[:, :-1])
                    dy_pred = torch.abs(probs[:, 1:] - probs[:, :-1])
                    tv_true = torch.mean(dy_true, dim=1)  # (B,)
                    tv_pred = torch.mean(dy_pred, dim=1)  # (B,)
                    target = float(tv_floor_scale) * tv_true
                    loss = loss + float(loss_weight_tv_floor) * torch.mean(torch.relu(target - tv_pred + eps))
                if float(loss_weight_transition) != 0.0:
                    # 启停变化一致性：让预测的变化幅度 |Δx_hat| 贴近真值 |Δx|
                    dy_true = torch.abs(batch_Y[:, 1:] - batch_Y[:, :-1])
                    dy_pred = torch.abs(probs[:, 1:] - probs[:, :-1])
                    loss = loss + float(loss_weight_transition) * torch.mean(torch.abs(dy_pred - dy_true))
                if float(loss_weight_binarize) != 0.0:
                    # 远离“均值输出”：惩罚 p*(1-p)，推动输出更接近 0/1
                    loss = loss + float(loss_weight_binarize) * torch.mean(probs * (1.0 - probs))
                if float(loss_weight_std_floor) != 0.0:
                    # 防“均值塌陷”：要求预测在时间维的标准差 >= scale * 真值标准差
                    # （真值是 0/1 序列，std 反映启停变化强度；scale<1 允许更保守）
                    eps = 1e-6
                    y_std = torch.std(batch_Y, dim=1)  # (B,)
                    p_std = torch.std(probs, dim=1)    # (B,)
                    target = float(std_floor_scale) * y_std
                    loss = loss + float(loss_weight_std_floor) * torch.mean(torch.relu(target - p_std + eps))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= max(len(dataset), 1)
            if scheduler is not None:
                scheduler.step(epoch_loss)
            if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"  [UnitPredictor-{g_int}] Epoch {epoch+1}/{num_epochs}, "
                    f"Loss: {epoch_loss:.6f}, LR: {current_lr:.1e}",
                    flush=True,
                )
        net.eval()
        return epoch_loss

    def train_all_shared_joint(
        self,
        *,
        num_epochs: int = 200,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
        enable_scheduler: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 0.0,
        enable_pos_weight: bool = False,
        pos_weight_clip: float = 20.0,
        loss_weight_bce: float = 1.0,
        loss_weight_mse: float = 0.0,
        loss_weight_l1: float = 0.0,
        loss_weight_tv: float = 0.0,
        loss_weight_transition: float = 0.0,
        loss_weight_binarize: float = 0.0,
        loss_weight_std_floor: float = 0.0,
        std_floor_scale: float = 0.5,
        loss_weight_tv_floor: float = 0.0,
        tv_floor_scale: float = 0.8,
        unit_loss_weights: List[float] | None = None,
    ) -> None:
        if self.net_variant != "tcn_shared_film":
            raise RuntimeError("train_all_shared_joint is only for tcn_shared_film")
        if self.shared_model is None or self.shared_optimizer is None:
            raise RuntimeError("shared_model/shared_optimizer is not initialized")

        resolved_batch_strategy = normalize_nn_batch_strategy(self.batch_strategy if batch_strategy is None else batch_strategy)
        resolved_shuffle = self.shuffle if shuffle is None else bool(shuffle)
        if resolved_batch_strategy == 'full-batch':
            resolved_batch_size = max(1, self.n_samples)
        else:
            base_batch = self.batch_size if batch_size is None else int(batch_size)
            resolved_batch_size = max(1, base_batch)
        resolved_lr = self.learning_rate if learning_rate is None else max(float(learning_rate), 1e-8)

        X = np.array([self._extract_features(i) for i in range(self.n_samples)])
        Y = self.x_true[:, :, :]  # (N, ng, T)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=resolved_batch_size, shuffle=resolved_shuffle and self.n_samples > 1)

        optimizer = self.shared_optimizer
        for pg in optimizer.param_groups:
            pg["lr"] = resolved_lr
        scheduler = None
        if enable_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=max(0, int(scheduler_patience)),
                factor=float(scheduler_factor),
                min_lr=max(0.0, float(scheduler_min_lr)),
            )

        criterion = nn.BCEWithLogitsLoss()
        if enable_pos_weight:
            y_mean = np.asarray(Y, dtype=float).mean(axis=0)  # (ng, T)
            eps = 1e-6
            pos_w = (1.0 - y_mean + eps) / (y_mean + eps)
            pos_w = np.clip(pos_w, 1.0, float(pos_weight_clip))
            pos_w_tensor = torch.tensor(pos_w, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        self.shared_model.train()
        w_unit = None
        if unit_loss_weights is not None:
            w_np = np.asarray(unit_loss_weights, dtype=float).reshape(-1)
            if w_np.size != int(self.ng):
                raise ValueError(f"unit_loss_weights size mismatch: got {w_np.size}, expected {self.ng}")
            # normalize to mean=1 for stability
            w_np = w_np / max(float(w_np.mean()), 1e-12)
            w_unit = torch.tensor(w_np, dtype=torch.float32, device=self.device).view(1, self.ng, 1)
        print_interval = max(1, int(num_epochs) // 10)
        for epoch in range(int(num_epochs)):
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                logits = self.shared_model(batch_X)  # (B, ng, T)
                probs = torch.sigmoid(logits)
                loss = 0.0
                if float(loss_weight_bce) != 0.0:
                    if w_unit is None:
                        loss = loss + float(loss_weight_bce) * criterion(logits, batch_Y)
                    else:
                        bce_el = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, batch_Y, reduction="none"
                        )  # (B,ng,T)
                        loss = loss + float(loss_weight_bce) * torch.mean(bce_el * w_unit)
                if float(loss_weight_mse) != 0.0:
                    mse_el = (probs - batch_Y) ** 2
                    loss = loss + float(loss_weight_mse) * (torch.mean(mse_el) if w_unit is None else torch.mean(mse_el * w_unit))
                if float(loss_weight_l1) != 0.0:
                    l1_el = torch.abs(probs - batch_Y)
                    loss = loss + float(loss_weight_l1) * (torch.mean(l1_el) if w_unit is None else torch.mean(l1_el * w_unit))
                if float(loss_weight_tv) != 0.0:
                    tv_el = torch.abs(probs[:, :, 1:] - probs[:, :, :-1])  # (B,ng,T-1)
                    loss = loss + float(loss_weight_tv) * (torch.mean(tv_el) if w_unit is None else torch.mean(tv_el * w_unit))
                if float(loss_weight_tv_floor) != 0.0:
                    eps = 1e-6
                    dy_true = torch.abs(batch_Y[:, :, 1:] - batch_Y[:, :, :-1])
                    dy_pred = torch.abs(probs[:, :, 1:] - probs[:, :, :-1])
                    tv_true = torch.mean(dy_true, dim=2)  # (B,ng)
                    tv_pred = torch.mean(dy_pred, dim=2)  # (B,ng)
                    target = float(tv_floor_scale) * tv_true
                    floor_el = torch.relu(target - tv_pred + eps).unsqueeze(-1)  # (B,ng,1)
                    loss = loss + float(loss_weight_tv_floor) * (torch.mean(floor_el) if w_unit is None else torch.mean(floor_el * w_unit))
                if float(loss_weight_transition) != 0.0:
                    dy_true = torch.abs(batch_Y[:, :, 1:] - batch_Y[:, :, :-1])
                    dy_pred = torch.abs(probs[:, :, 1:] - probs[:, :, :-1])
                    tr_el = torch.abs(dy_pred - dy_true)  # (B,ng,T-1)
                    loss = loss + float(loss_weight_transition) * (torch.mean(tr_el) if w_unit is None else torch.mean(tr_el * w_unit))
                if float(loss_weight_binarize) != 0.0:
                    bin_el = probs * (1.0 - probs)
                    loss = loss + float(loss_weight_binarize) * (torch.mean(bin_el) if w_unit is None else torch.mean(bin_el * w_unit))
                if float(loss_weight_std_floor) != 0.0:
                    eps = 1e-6
                    y_std = torch.std(batch_Y, dim=2)  # (B,ng)
                    p_std = torch.std(probs, dim=2)    # (B,ng)
                    target = float(std_floor_scale) * y_std
                    std_el = torch.relu(target - p_std + eps).unsqueeze(-1)  # (B,ng,1)
                    loss = loss + float(loss_weight_std_floor) * (torch.mean(std_el) if w_unit is None else torch.mean(std_el * w_unit))

                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_X.size(0)
            epoch_loss /= max(len(dataset), 1)
            if scheduler is not None:
                scheduler.step(epoch_loss)
            if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == int(num_epochs) - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  [UnitPredictor-shared] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, LR: {current_lr:.1e}", flush=True)
        self.shared_model.eval()

    def train_all(
        self,
        num_epochs: int = 100,
        batch_size: int | None = None,
        batch_strategy: str | None = None,
        shuffle: bool | None = None,
        learning_rate: float | None = None,
        enable_scheduler: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 0.0,
        enable_pos_weight: bool = False,
        pos_weight_clip: float = 20.0,
        loss_weight_bce: float = 1.0,
        loss_weight_mse: float = 0.0,
        loss_weight_l1: float = 0.0,
        loss_weight_tv: float = 0.0,
        loss_weight_transition: float = 0.0,
        loss_weight_binarize: float = 0.0,
        loss_weight_std_floor: float = 0.0,
        std_floor_scale: float = 0.5,
        loss_weight_tv_floor: float = 0.0,
        tv_floor_scale: float = 0.8,
        unit_loss_weights: List[float] | None = None,
    ) -> None:
        if self.net_variant == "tcn_shared_film":
            self.train_all_shared_joint(
                num_epochs=num_epochs,
                batch_size=batch_size,
                batch_strategy=batch_strategy,
                shuffle=shuffle,
                learning_rate=learning_rate,
                enable_scheduler=enable_scheduler,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                scheduler_min_lr=scheduler_min_lr,
                enable_pos_weight=enable_pos_weight,
                pos_weight_clip=pos_weight_clip,
                loss_weight_bce=loss_weight_bce,
                loss_weight_mse=loss_weight_mse,
                loss_weight_l1=loss_weight_l1,
                loss_weight_tv=loss_weight_tv,
                loss_weight_transition=loss_weight_transition,
                loss_weight_binarize=loss_weight_binarize,
                loss_weight_std_floor=loss_weight_std_floor,
                std_floor_scale=std_floor_scale,
                loss_weight_tv_floor=loss_weight_tv_floor,
                tv_floor_scale=tv_floor_scale,
                unit_loss_weights=unit_loss_weights,
            )
            return
        for g in self.unit_ids:
            self.train_unit(
                unit_id=g,
                num_epochs=num_epochs,
                batch_size=batch_size,
                batch_strategy=batch_strategy,
                shuffle=shuffle,
                learning_rate=learning_rate,
                enable_scheduler=enable_scheduler,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                scheduler_min_lr=scheduler_min_lr,
                enable_pos_weight=enable_pos_weight,
                pos_weight_clip=pos_weight_clip,
                loss_weight_bce=loss_weight_bce,
                loss_weight_mse=loss_weight_mse,
                loss_weight_l1=loss_weight_l1,
                loss_weight_tv=loss_weight_tv,
                loss_weight_transition=loss_weight_transition,
                loss_weight_binarize=loss_weight_binarize,
                loss_weight_std_floor=loss_weight_std_floor,
                std_floor_scale=std_floor_scale,
                loss_weight_tv_floor=loss_weight_tv_floor,
                tv_floor_scale=tv_floor_scale,
            )

    def forward_logits(self, unit_id: int, features_tensor: torch.Tensor) -> torch.Tensor:
        """带梯度的 logits 输出 (B, T)。供 BCD 内部微调使用。"""
        self._ensure_network(unit_id)
        return self.networks[int(unit_id)](features_tensor)

    def get_network(self, unit_id: int) -> nn.Module:
        self._ensure_network(unit_id)
        return self.networks[int(unit_id)]

    def predict_probs(self, unit_id: int, sample_or_features) -> np.ndarray:
        """no_grad 下返回 (T,) sigmoid 概率。"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")
        self._ensure_network(unit_id)
        if isinstance(sample_or_features, dict):
            features = get_feature_vector_from_sample(dict(sample_or_features))
        else:
            features = np.asarray(sample_or_features, dtype=float).reshape(-1)
        features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        net = self.networks[int(unit_id)]
        was_training = net.training
        net.eval()
        with torch.no_grad():
            probs = torch.sigmoid(net(features_tensor)).squeeze(0).cpu().numpy()
        if was_training:
            net.train()
        return probs

    def save(self, filepath: str) -> None:
        if not TORCH_AVAILABLE:
            return
        dirpath = os.path.dirname(os.path.abspath(filepath))
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        payload = {
            'metadata': {
                'ng': self.ng,
                'T': self.T,
                'input_dim': self.input_dim,
                'hidden_dims': list(self.hidden_dims),
                'unit_ids': list(self.unit_ids),
                'learning_rate': self.learning_rate,
                'net_variant': self.net_variant,
                'resmlp_width': self.resmlp_width,
                'resmlp_depth': self.resmlp_depth,
                'tconv_channels': self.tconv_channels,
                'tconv_depth': self.tconv_depth,
                'tcn_channels': self.tcn_channels,
                'tcn_depth': self.tcn_depth,
                'dropout': self.dropout,
            },
        }
        if self.net_variant == "tcn_shared_film":
            payload['shared'] = {
                'shared_model': None if self.shared_model is None else self.shared_model.state_dict(),
            }
        else:
            payload['state_dicts'] = {int(g): self.networks[g].state_dict() for g in self.networks}
        torch.save(payload, filepath)
        print(f"✓ 单机组 0/1 预测器已保存: {filepath}", flush=True)

    def load(self, filepath: str) -> bool:
        if not TORCH_AVAILABLE:
            return False
        state = torch.load(filepath, map_location=self.device)
        meta = state.get('metadata', {}) or {}
        loaded_variant = str(meta.get('net_variant', self.net_variant) or self.net_variant).strip().lower()
        want_variant = str(self.net_variant).strip().lower()
        if loaded_variant != want_variant:
            print(
                f"! [UnitPredictor] checkpoint net_variant={loaded_variant!r} 与当前 {want_variant!r} 不一致，跳过加载: {filepath}",
                flush=True,
            )
            return False
        # 允许从 checkpoint 恢复共享网络（仅当当前也配置为共享网络）
        if loaded_variant == "tcn_shared_film" and self.net_variant == "tcn_shared_film":
            shared = state.get('shared', {}) or {}
            loaded_any = False
            # 确保 shared 组件存在
            for g in (meta.get('unit_ids') or self.unit_ids):
                self._ensure_network(int(g))
            if self.shared_model is None:
                raise RuntimeError("shared_model is not initialized for tcn_shared_film")
            if shared.get('shared_model') is not None:
                loaded_sd = shared.get('shared_model') or {}
                if not isinstance(loaded_sd, dict) or len(loaded_sd) == 0:
                    print(f"! [UnitPredictor] checkpoint shared_model 为空或格式异常，跳过加载: {filepath}", flush=True)
                else:
                    current_sd = self.shared_model.state_dict()
                    filtered_sd = {}
                    skipped = []
                    for k, v in loaded_sd.items():
                        if k not in current_sd:
                            skipped.append((k, "missing_key"))
                            continue
                        try:
                            if tuple(current_sd[k].shape) != tuple(v.shape):
                                skipped.append((k, f"shape {tuple(v.shape)} -> {tuple(current_sd[k].shape)}"))
                                continue
                        except Exception:
                            skipped.append((k, "shape_check_failed"))
                            continue
                        filtered_sd[k] = v

                    # 只加载完全匹配的参数，避免 shape mismatch 直接报错。
                    # 部分加载会使骨干大量随机初始化，效果比「完全从头训」更差，故低匹配率时整网放弃加载。
                    _min_shared_key_ratio = 0.85
                    n_total = len(current_sd)
                    n_loaded = len(filtered_sd)
                    n_skipped = len(skipped)
                    if n_loaded == 0 or (n_total > 0 and n_loaded < _min_shared_key_ratio * n_total):
                        print(
                            "! [UnitPredictor] checkpoint 与当前 SharedTCNFiLMNet 不匹配或匹配键过少 "
                            f"({n_loaded}/{n_total} < {_min_shared_key_ratio:.0%})，跳过加载并从头训练。"
                            f" 请使用同 net_variant / 同 input_dim·T·channels·depth 训练得到的 unit_predictor.pth，"
                            f"或设 UNIT_PREDICTOR_LOAD_PATH=None。ckpt={filepath}",
                            flush=True,
                        )
                    else:
                        missing, unexpected = self.shared_model.load_state_dict(filtered_sd, strict=False)
                        print(
                            f"✓ 单机组 0/1 预测器已加载(共享骨干, 兼容模式): {filepath} "
                            f"(loaded={n_loaded}/{n_total}, skipped={n_skipped}, missing={len(missing)}, unexpected={len(unexpected)})",
                            flush=True,
                        )
                        loaded_any = True
                        if n_skipped > 0:
                            preview = ", ".join([f"{k}({reason})" for k, reason in skipped[:6]])
                            more = "" if n_skipped <= 6 else f", ...(+{n_skipped-6})"
                            print(f"  [UnitPredictor] skipped params: {preview}{more}", flush=True)
            self.shared_optimizer = optim.Adam(
                self.shared_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            # 注：上面兼容加载已经打印过一次更详细的统计；这里保留兜底提示以兼容旧日志习惯
            if shared.get('shared_model') is None:
                print(f"✓ 单机组 0/1 预测器已加载(共享骨干): {filepath}", flush=True)
            self.shared_model.eval()
            return loaded_any

        state_dicts = state.get('state_dicts', {}) or {}
        if not state_dicts:
            print(f"! [UnitPredictor] checkpoint has no state_dicts; no weights loaded: {filepath}", flush=True)
            return False
        loaded_count = 0
        for g, sd in state_dicts.items():
            g_int = int(g)
            if g_int not in self.unit_ids:
                self.unit_ids.append(g_int)
            self._ensure_network(g_int)
            self.networks[g_int].load_state_dict(sd)
            loaded_count += 1
        print(
            f"✓ 单机组 0/1 预测器已加载: {filepath} "
            f"（已恢复机组: {sorted(self.networks.keys())}）",
            flush=True,
        )
        for net in self.networks.values():
            net.eval()
        return loaded_count > 0


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
        pg_cost_hidden_dims: List[int] = None,
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
        if pg_cost_hidden_dims is None:
            pg_cost_hidden_dims = list(hidden_dims)

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

        # c_pg 使用独立网络，不共享主 surrogate backbone
        self.pg_input_proj = nn.Linear(input_dim, pg_cost_hidden_dims[0])
        pg_res_blocks = []
        prev_pg_dim = pg_cost_hidden_dims[0]
        for hd in pg_cost_hidden_dims:
            pg_res_blocks.append(ResBlock(prev_pg_dim, hd))
            prev_pg_dim = hd
        self.pg_res_blocks = nn.ModuleList(pg_res_blocks)

        pg_feat_dim = prev_pg_dim
        pg_head_hidden = max(pg_feat_dim // 2, 64)
        pg_head_mid = max(pg_head_hidden // 2, 32)

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
        # pg 调整项头：输出 T 个无界标量；幅度由训练器侧 softbound（对 |c_pg| 超出 pg_cost_scale 的平滑惩罚）约束
        self.pg_cost_net = nn.Sequential(
            nn.Linear(pg_feat_dim, pg_head_hidden), nn.LayerNorm(pg_head_hidden), nn.LeakyReLU(0.01),
            nn.Linear(pg_head_hidden, pg_head_mid), nn.LayerNorm(pg_head_mid), nn.LeakyReLU(0.01),
            nn.Linear(pg_head_mid, self.T)
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

    def encode_pg_features(self, x):
        features = self.pg_input_proj(x)
        for block in self.pg_res_blocks:
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
        """c_pg 单独前向：线性头直接输出 T 维修正量（不再经 tanh 硬饱和）。"""
        features = self.encode_pg_features(x)
        return self.pg_cost_net(features)

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
    if strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
        T = int(getattr(trainer, 'T', 0) or 0)
        group_size = max(int(getattr(trainer, 'all_mode_group_size', 4) or 4), 1)
        head = group_size * max(T - 2, 0)
        default_offsets = [
            SURROGATE_TRIPLE_WINDOW_OFFSETS if k < head else SURROGATE_SINGLE_TIME_OFFSETS
            for k in range(n_constraints)
        ]
        if 'resolved' in locals() and resolved:
            return (resolved + default_offsets[len(resolved):])[:n_constraints]
        return default_offsets
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
                 lp_backend: str = LP_BACKEND_GUROBI,
                 constraint_generation_strategy: str = "sensitive",
                 rho_primal_init: float = 1e-3,
                 rho_dual_init: float = 1e-3,
                 rho_dual_pg_init: float | None = None,
                 rho_dual_x_init: float | None = None,
                 rho_dual_coc_init: float | None = None,
                 rho_binary_init: float = 1.0,
                 rho_binary_max: float = 1e4,
                 rho_opt_init: float = 1e-3,
                 gamma_base: float = 1e-3,
                 mu_lower_bound_init: float = 0.1,
                 mu_individual_lower_bound_round: int = 3,
                 mu_group_lower_bound_round: int = 50,
                 mu_signed_round_interval: int | None = None,
                 mu_sign_hysteresis_rounds: int = 2,
                 mu_sign_flip_min_share: float = 0.67,
                 x_bound_dual_zero_rounds: int = 0,
                 pg_cost_start_round: int = 3,
                 pg_cost_scale_multiplier: float = 1.2,
                 nn_hidden_dims: List[int] = None,
                 pg_cost_hidden_dims: List[int] = None,
                 nn_learning_rate: float = 1e-4,
                 cost_learning_rate: float = 1e-5,
                 pg_cost_lr: float = 2e-5,
                 pg_cost_surr_lr: float = 5e-5,
                 nn_batch_strategy: str = "full-batch",
                 nn_batch_size: int = 4,
                 nn_shuffle: bool = True,
                 pg_cost_nn_epochs: int | None = None,
                 pg_cost_reg_deadband: float = 0.5,
                 pg_cost_softbound_weight: float = 1.0,
                 nn_smooth_abs_eps: float = 1e-6,
                 pg_cost_smooth_abs_eps: float = 1e-6,
                 pg_cost_batch_strategy: str | None = None,
                 pg_cost_batch_size: int | None = None,
                 pg_cost_shuffle: bool | None = None,
                 pg_cost_use_sample_weights: bool = True,
                 pg_cost_sample_weight_power: float = 1.0,
                 pg_cost_sample_weight_clip: float = 10.0,
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
                 single_mu_cap_penalty_weight: float = 0.0,
                 single_mu_cap_initial_weight: float | None = None,
                 single_mu_cap_final_weight: float | None = None,
                 single_mu_cap_initial: float | None = None,
                 single_mu_cap_final: float | None = None,
                 single_mu_cap_start_round: int = 0,
                 single_mu_cap_end_round: int = 0,
                 ignore_startup_shutdown_costs: bool = False,
                 unit_predictor: 'SingleUnitBinaryPredictorTrainer | None' = None,
                 use_unit_predictor: bool = False,
                 predictor_warmup_rounds: int = 0,
                 sign4_curriculum_rounds: int = 0,
                 sign4_initial_scale: float = 1.0,
                 sign4_final_scale: float = 1.0,
                 sign4_delay_rounds: int = 0,
                 unit_predictor_finetune_lr: float = 1e-5,
                 unit_predictor_weight_decay: float = 1e-4,
                 pg_cost_single_sample_reg_scale: float | None = None,
                 pg_cost_c_pg_adam_weight_decay: float | None = None,
                 main_direct_train_config: dict | None = None,
                 c_pg_direct_train_config: dict | None = None,
                 nn_main_eta_min_ratio: float = 0.08,
                 nn_main_lr_late_scale: float = 0.42,
                 nn_main_adam_weight_decay: float = 1e-4,
                 nn_main_grad_clip: float = 0.85,
                 nn_main_kkt_lr_scale: float = 1.0,
                 case_name: str | None = None,
                 enable_surrogate_delta_reference_lift: bool | None = None,
                 surrogate_delta_reference_eps: float = 1e-6,
                 surrogate_delta_reference_scope: str = "sign4_only",
                 surrogate_delta_reference_min_abs_factor: float = 1e-9,
                 skip_initial_solve: bool = False,
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
            case_name: 算例名；``case118`` 时最小开停机与 MTI 数据对齐，否则沿用 ``min(4,T)``。
            device: 计算设备
        """
        self.ppc = ppc
        self.ppc_raw = ppc
        ppc_int = ext2int(ppc)
        self.baseMVA = ppc_int['baseMVA']
        self.bus = ppc_int['bus']
        self.gen = ppc_int['gen']
        self.branch = ppc_int['branch']
        self.nl = int(self.branch.shape[0])
        self.gencost = ppc_int['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        self.unit_id = unit_id
        self._lp_backend = normalize_lp_backend(lp_backend)
        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            assert_lp_backend_available(self._lp_backend)
        else:
            _require_gurobi_available("SubproblemSurrogateTrainer with lp_backend='gurobi'")
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
        self.pg_cost_softbound_weight = max(float(pg_cost_softbound_weight), 0.0)
        self.nn_smooth_abs_eps = max(float(nn_smooth_abs_eps), 0.0)
        self.pg_cost_smooth_abs_eps = max(float(pg_cost_smooth_abs_eps), 0.0)
        self.pg_cost_batch_strategy = (
            normalize_nn_batch_strategy(pg_cost_batch_strategy)
            if pg_cost_batch_strategy is not None
            else None
        )
        self.pg_cost_batch_size = None if pg_cost_batch_size is None else max(int(pg_cost_batch_size), 1)
        self.pg_cost_shuffle = None if pg_cost_shuffle is None else bool(pg_cost_shuffle)
        self.pg_cost_use_sample_weights = bool(pg_cost_use_sample_weights)
        self.pg_cost_sample_weight_power = max(float(pg_cost_sample_weight_power), 0.0)
        self.pg_cost_sample_weight_clip = max(float(pg_cost_sample_weight_clip), 1.0)
        self.requested_max_constraints = max_constraints
        self.max_constraints = max_constraints  # V3新增
        
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]

        self.case_name = case_name
        self.subproblem_Ton, self.subproblem_Toff = _compute_subproblem_min_up_down_tau_max(
            case_name,
            self.ppc_raw,
            self.gen,
            self.T,
            float(self.T_delta),
            int(unit_id),
        )

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
        self.rho_binary_max = max(float(rho_binary_max), self.rho_binary)
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
        self.mu_sign_hysteresis_rounds = max(int(mu_sign_hysteresis_rounds), 1)
        self.mu_sign_flip_min_share = min(max(float(mu_sign_flip_min_share), 0.5), 1.0)
        self.x_bound_dual_zero_rounds = max(int(x_bound_dual_zero_rounds), 0)
        self.pg_cost_start_round = max(int(pg_cost_start_round), 0)
        self.pg_cost_scale_multiplier = max(float(pg_cost_scale_multiplier), 1e-6)
        self.nn_hidden_dims = normalize_nn_hidden_dims(nn_hidden_dims, [256, 256])
        self.pg_cost_hidden_dims = normalize_nn_hidden_dims(pg_cost_hidden_dims, self.nn_hidden_dims)
        self.nn_learning_rate = max(float(nn_learning_rate), 1e-8)
        self.cost_learning_rate = max(float(cost_learning_rate), 1e-8)
        self.pg_cost_lr = max(float(pg_cost_lr), 1e-8)
        self.pg_cost_surr_lr = max(float(pg_cost_surr_lr), 1e-8)
        self.nn_batch_strategy = normalize_nn_batch_strategy(nn_batch_strategy)
        self.nn_batch_size = max(int(nn_batch_size), 1)
        self.nn_shuffle = bool(nn_shuffle)
        # 0 表示 BCD 内跳过 NN-c_pg（可微 KKT loss），仅依赖 direct-c_pg 与缓存刷新
        self.pg_cost_nn_epochs = None if pg_cost_nn_epochs is None else max(int(pg_cost_nn_epochs), 0)
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
        single_mu_cap_weight_fallback = max(float(single_mu_cap_penalty_weight), 0.0)
        self.single_mu_cap_initial_weight = (
            0.0 if single_mu_cap_initial_weight is None
            else max(float(single_mu_cap_initial_weight), 0.0)
        )
        self.single_mu_cap_final_weight = (
            single_mu_cap_weight_fallback if single_mu_cap_final_weight is None
            else max(float(single_mu_cap_final_weight), 0.0)
        )
        self.single_mu_cap_penalty_weight = self.single_mu_cap_final_weight
        self.single_mu_cap_initial = (
            None if single_mu_cap_initial is None else max(float(single_mu_cap_initial), 0.0)
        )
        self.single_mu_cap_final = (
            None if single_mu_cap_final is None else max(float(single_mu_cap_final), 0.0)
        )
        self.single_mu_cap_start_round = max(int(single_mu_cap_start_round), 0)
        self.single_mu_cap_end_round = max(
            int(single_mu_cap_end_round),
            self.single_mu_cap_start_round,
        )
        self.ignore_startup_shutdown_costs = bool(ignore_startup_shutdown_costs)
        self.iter_number = 0
        self.x_cost_scale = 2.0
        self.pg_cost_scale = 1.0
        self._sync_rho_dual_summary()

        # 单机组 0/1 预测器（可选代理约束生成方式）
        self.unit_predictor = unit_predictor
        self.use_unit_predictor = bool(use_unit_predictor)
        self.predictor_warmup_rounds = max(int(predictor_warmup_rounds), 0)
        self.sign4_curriculum_rounds = max(int(sign4_curriculum_rounds), 0)
        self.sign4_initial_scale = max(float(sign4_initial_scale), 0.0)
        self.sign4_final_scale = max(float(sign4_final_scale), 0.0)
        self.sign4_delay_rounds = max(int(sign4_delay_rounds), 0)
        if enable_surrogate_delta_reference_lift is None:
            self.enable_surrogate_delta_reference_lift = self.constraint_generation_strategy in (
                CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
                CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
            )
        else:
            self.enable_surrogate_delta_reference_lift = bool(enable_surrogate_delta_reference_lift)
        self.surrogate_delta_reference_eps = max(float(surrogate_delta_reference_eps), 0.0)
        self.surrogate_delta_reference_scope = self._normalize_surrogate_delta_reference_scope(
            surrogate_delta_reference_scope
        )
        self.surrogate_delta_reference_min_abs_factor = max(
            float(surrogate_delta_reference_min_abs_factor), 1e-15
        )
        self.unit_predictor_finetune_lr = max(float(unit_predictor_finetune_lr), 1e-10)
        self.unit_predictor_weight_decay = max(float(unit_predictor_weight_decay), 0.0)
        self._unit_predictor_optimizer = None  # 在 iter_with_surrogate_nn 中按需构造
        self._unit_predictor_scheduler = None

        # 单样本时 c_pg 正则与驻点项冲突更明显：默认可压 reg、并去掉 c_pg Adam 的 weight decay，便于将 obj_dual_pg 压到 0 附近
        if pg_cost_single_sample_reg_scale is not None:
            self._c_pg_reg_loss_scale = max(float(pg_cost_single_sample_reg_scale), 0.0)
        else:
            self._c_pg_reg_loss_scale = 0.1 if self.n_samples == 1 else 1.0
        if pg_cost_c_pg_adam_weight_decay is not None:
            self.pg_cost_c_pg_adam_weight_decay = max(float(pg_cost_c_pg_adam_weight_decay), 0.0)
        else:
            self.pg_cost_c_pg_adam_weight_decay = 0.0 if self.n_samples == 1 else 1e-4
        self.main_direct_train_config = dict(main_direct_train_config or {})
        self.c_pg_direct_train_config = dict(c_pg_direct_train_config or {})
        # NN-main（可微 KKT）：与 direct-NN-main 一致用每轮 BCD 内 CosineAnnealingLR；
        # nn_main_lr_late_scale 随外循环减小基准步长，类比 subproblem_lp_solver 中 rho 增大后的精细收敛。
        self.nn_main_eta_min_ratio = max(float(nn_main_eta_min_ratio), 0.0)
        self.nn_main_lr_late_scale = min(max(float(nn_main_lr_late_scale), 1e-3), 1.0)
        self.nn_main_adam_weight_decay = max(float(nn_main_adam_weight_decay), 0.0)
        self.nn_main_grad_clip = max(float(nn_main_grad_clip), 1e-6)
        # 仅作用于 iter_with_surrogate_nn（KKT），不作用于 direct-NN-main；<1 可抑制大 rho 下 loss 尖峰
        self.nn_main_kkt_lr_scale = max(float(nn_main_kkt_lr_scale), 1e-6)
        self._surrogate_bcd_max_iter = 1
        
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
        self.surrogate_direction_pending_signs = np.zeros(self.num_coupling_constraints, dtype=float)
        self.surrogate_direction_pending_counts = np.zeros(self.num_coupling_constraints, dtype=int)
        self.mu = np.ones((self.n_samples, self.num_coupling_constraints), dtype=float) * self.mu_lower_bound

        # 固有约束的对偶变量（由dual block更新，用于NN loss的完整KKT驻点条件）
        self.lambda_inherent = [None] * self.n_samples

        # 存储每个样本的敏感时段索引
        self.sensitive_timesteps = [[] for _ in range(self.n_samples)]
        self.surrogate_constraint_offsets = [[] for _ in range(self.n_samples)]

        # 场景特征维度（与 _extract_features 中 pd_flat 前缀一致）；unit_predictor 仅消费此段
        self._scenario_feature_dim = len(
            get_feature_vector_from_sample(dict(self.active_set_data[0]))
        )
        
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
        self._skip_initial_solve = bool(skip_initial_solve)
        if self._skip_initial_solve:
            print(
                f"  [Unit-{self.unit_id}] skip initial solve for checkpoint loading/test-only evaluation",
                flush=True,
            )
        else:
            self._initialize_solve()
        if self.enable_surrogate_delta_reference_lift:
            self._sync_surrogate_direction_strategy_state()
            self._lift_surrogate_delta_for_reference_x()

        # ----- Persistent Gurobi model cache -----
        # 每个样本只建一次模型，后续迭代通过 setObjective / chgCoeff 更新
        self._primal_models: dict = {}           # sample_id -> gp.Model
        self._primal_vars: dict = {}             # sample_id -> vars_dict
        self._primal_model_n_coupling: dict = {} # sample_id -> num_coupling at build time

        self._dual_sub_models: dict = {}         # sample_id -> gp.Model
        self._dual_sub_vars: dict = {}           # sample_id -> vars_dict
        self._dual_sub_model_state: dict = {}    # sample_id -> (lb, x_bound_dual_ub)

        # 迭代间输出差异正则：用于抑制 NN 输出在相邻 BCD 迭代间剧烈跳变
        self._prev_alpha_values = None
        self._prev_beta_values = None
        self._prev_gamma_values = None
        self._prev_delta_values = None
        self._prev_cost_values = None
        self._prev_pg_cost_values = None
        self._loss_tensor_cache = None
        self._loss_tensor_cache_signature = None
        self._loss_tensor_cache_dirty = True

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
            pg_cost_hidden_dims=self.pg_cost_hidden_dims,
            x_cost_scale=self.x_cost_scale,
            pg_cost_scale=self.pg_cost_scale,
            delta_base=delta_base,
            delta_scale=delta_scale,
        ).to(self.device)

        # 分离参数组：主优化器管理 backbone + alpha/beta/gamma/delta 头
        main_params = self._main_network_parameters()
        self.optimizer = optim.Adam(main_params, lr=self.nn_learning_rate)
        # x / pg 调整项头拆分优化，便于仅对 c_pg 提高学习率并延迟启用
        self.cost_optimizer = optim.Adam(
            self._cost_network_parameters(),
            lr=self.cost_learning_rate,
        )
        self.pg_cost_optimizer = optim.Adam(
            self._pg_network_parameters(),
            lr=self.pg_cost_lr,
        )

        print(f"  - 代理约束网络输入维度: {input_dim}", flush=True)
        print(f"  - main隐藏层: {self.nn_hidden_dims}", flush=True)
        print(f"  - c_pg隐藏层: {self.pg_cost_hidden_dims}", flush=True)
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
            f"c_pg_softbound_w={self.pg_cost_softbound_weight:.4g}, "
            f"c_pg_reg_scale={self._c_pg_reg_loss_scale:.3g} (L2/iter/soft reg), "
            f"c_pg_wd={self.pg_cost_c_pg_adam_weight_decay:.1e}, "
            f"nn_smooth_eps={self.nn_smooth_abs_eps:.1e}, "
            f"c_pg_smooth_eps={self.pg_cost_smooth_abs_eps:.1e}",
            flush=True,
        )
        print(f"  - nn_dual_term_interval={self.nn_dual_term_interval}", flush=True)
        print(
            f"  - nn_main(refine): eta_min_ratio={self.nn_main_eta_min_ratio:.3g}, "
            f"lr_late_scale={self.nn_main_lr_late_scale:.3g}, "
            f"adam_wd={self.nn_main_adam_weight_decay:.1e}, grad_clip={self.nn_main_grad_clip:.3g}, "
            f"kkt_lr_scale={self.nn_main_kkt_lr_scale:.3g}",
            flush=True,
        )

    def _nn_main_bcd_lr_scale(self) -> float:
        """BCD 外循环越靠后，NN-main 基准学习率越小（与 LP 块 penalty 随轮次加重后更偏局部细化一致）。"""
        late = float(self.nn_main_lr_late_scale)
        T = max(int(getattr(self, "_surrogate_bcd_max_iter", 1)), 1)
        t = min(max(int(self.iter_number), 0), T - 1)
        if T <= 1:
            return 1.0
        w = t / (T - 1)
        return (1.0 - w) + late * w

    def _cost_network_parameters(self) -> list:
        return list(self.surrogate_net.cost_net.parameters())

    def _pg_network_parameters(self) -> list:
        return (
            list(self.surrogate_net.pg_input_proj.parameters())
            + list(self.surrogate_net.pg_res_blocks.parameters())
            + list(self.surrogate_net.pg_cost_net.parameters())
        )

    def _main_network_parameters(self) -> list:
        aux_param_ids = {
            id(param)
            for param in (self._cost_network_parameters() + self._pg_network_parameters())
        }
        return [param for param in self.surrogate_net.parameters() if id(param) not in aux_param_ids]

    def _pg_costs_active(self) -> bool:
        return self.iter_number >= self.pg_cost_start_round

    def subproblem_generation_no_load_coeff(self, g: int | None = None) -> float:
        """无负荷（开机固定）发电成本系数 b，满足 cpower = a*pg + b*x。

        ``ignore_startup_shutdown_costs=True`` 时，除启停费外 **b 也置零**，与子问题 LP 块一致。
        """
        gi = int(self.unit_id if g is None else g)
        if self.ignore_startup_shutdown_costs:
            return 0.0
        return float(self.gencost[gi, -1]) / float(self.T_delta)

    def _invalidate_loss_tensor_cache(self) -> None:
        """Mark cached loss tensors stale after BCD state or restored state changes."""
        self._loss_tensor_cache_dirty = True

    def _loss_tensor_cache_key(self, device) -> tuple:
        lambda_ids = tuple(id(x) for x in self.lambda_inherent)
        sensitive_key = tuple(tuple(int(v) for v in ts) for ts in self.sensitive_timesteps)
        offsets_key = tuple(
            tuple(tuple(int(o) for o in offsets) for offsets in sample_offsets)
            for sample_offsets in self.surrogate_constraint_offsets
        )
        _dir_signs_arr = np.asarray(self._get_surrogate_direction_signs(), dtype=float)
        _curr_factors_arr = self._sign4_curriculum_factors(self.num_coupling_constraints)
        direction_key = tuple((_dir_signs_arr * _curr_factors_arr).reshape(-1).tolist())
        return (
            str(device),
            id(self.x),
            id(self.mu),
            id(self.lambda_vals),
            lambda_ids,
            sensitive_key,
            offsets_key,
            direction_key,
            id(self._prev_alpha_values),
            id(self._prev_beta_values),
            id(self._prev_gamma_values),
            id(self._prev_delta_values),
            id(self._prev_cost_values),
            id(self._prev_pg_cost_values),
            bool(self.ignore_startup_shutdown_costs),
        )

    def _lambda_inherent_tensor_dict(self, sample_id: int, device) -> dict | None:
        lam_inh = self.lambda_inherent[sample_id]
        if lam_inh is None:
            return None
        out = {}
        for key, value in lam_inh.items():
            try:
                out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
            except Exception:
                pass
        return out

    def _x_stationarity_inherent_np(self, sample_id: int) -> np.ndarray:
        """Constant part of x-stationarity before surrogate alpha/beta/gamma and c_x terms."""
        g = self.unit_id
        lam_inh = self.lambda_inherent[sample_id]
        out = np.zeros(self.T, dtype=np.float32)
        b_val = float(self.subproblem_generation_no_load_coeff(g))
        pmin_v = float(self.gen[g, PMIN])
        pmax_v = float(self.gen[g, PMAX])
        ru_v = float(self.Ru_all[g])
        rd_v = float(self.Rd_all[g])
        ru_co_v = float(self.Ru_co_all[g])
        rd_co_v = float(self.Rd_co_all[g])
        start_c = 0.0 if self.ignore_startup_shutdown_costs else float(self.gencost[g, 1])
        shut_c = 0.0 if self.ignore_startup_shutdown_costs else float(self.gencost[g, 2])
        ton_l = int(self.subproblem_Ton)
        toff_l = int(self.subproblem_Toff)
        for t in range(self.T):
            val = b_val
            if lam_inh is not None:
                val += pmin_v * float(lam_inh['lambda_pg_lower'][t])
                val -= pmax_v * float(lam_inh['lambda_pg_upper'][t])
                lam_ru = lam_inh['lambda_ramp_up']
                lam_rd = lam_inh['lambda_ramp_down']
                if t < self.T - 1:
                    val += (ru_co_v - ru_v) * float(lam_ru[t])
                if t > 0:
                    val += (rd_co_v - rd_v) * float(lam_rd[t - 1])

                lam_mon = lam_inh['lambda_min_on']
                lam_moff = lam_inh['lambda_min_off']
                for tau in range(1, ton_l + 1):
                    tau_row = lam_mon[tau - 1]
                    for t1 in range(self.T - tau):
                        k = float(tau_row[t1])
                        if t == t1 + 1:
                            val += k
                        if t == t1:
                            val -= k
                        if t == t1 + tau:
                            val -= k
                for tau in range(1, toff_l + 1):
                    tau_row = lam_moff[tau - 1]
                    for t1 in range(self.T - tau):
                        k = float(tau_row[t1])
                        if t == t1 + 1:
                            val -= k
                        if t == t1:
                            val += k
                        if t == t1 + tau:
                            val += k

                lam_sc = lam_inh['lambda_start_cost']
                lam_shc = lam_inh['lambda_shut_cost']
                if t > 0:
                    val += start_c * float(lam_sc[t - 1])
                    val -= shut_c * float(lam_shc[t - 1])
                if t < self.T - 1:
                    val -= start_c * float(lam_sc[t])
                    val += shut_c * float(lam_shc[t])
                val += float(lam_inh['lambda_x_upper'][t]) - float(lam_inh['lambda_x_lower'][t])
            out[t] = val
        return out

    def _pg_stationarity_const_np(self, sample_id: int) -> np.ndarray:
        """Constant part of pg-stationarity before c_pg[t]."""
        g = self.unit_id
        lam_inh = self.lambda_inherent[sample_id]
        out = np.zeros(self.T, dtype=np.float32)
        a_val = float(self.gencost[g, -2] / self.T_delta)
        lambda_val = np.asarray(self.lambda_vals[sample_id], dtype=float).reshape(-1)
        if lam_inh is None:
            return out
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
            out[t] = pg_const
        return out

    def _build_loss_constraint_metadata(self, sample_id: int, device) -> dict:
        sensitive_t = list(self.sensitive_timesteps[sample_id])
        offsets_by_k = self._constraint_offsets_for_sample(sample_id)
        x_np = np.asarray(self.x[sample_id], dtype=np.float32).reshape(-1)
        n_constraints = len(sensitive_t)
        x_alpha = np.zeros(n_constraints, dtype=np.float32)
        x_beta = np.zeros(n_constraints, dtype=np.float32)
        x_gamma = np.zeros(n_constraints, dtype=np.float32)
        term_times: list[int] = []
        term_k: list[int] = []
        term_kind: list[int] = []
        for k, ts in enumerate(sensitive_t):
            for time_idx, kind in iterate_surrogate_constraint_terms(
                int(ts),
                offsets_by_k[k],
                0,
                1,
                2,
                self.T,
            ):
                if kind == 0:
                    x_alpha[k] = x_np[int(time_idx)]
                elif kind == 1:
                    x_beta[k] = x_np[int(time_idx)]
                else:
                    x_gamma[k] = x_np[int(time_idx)]
                term_times.append(int(time_idx))
                term_k.append(int(k))
                term_kind.append(int(kind))
        return {
            'n_constraints': n_constraints,
            'x_alpha': torch.as_tensor(x_alpha, dtype=torch.float32, device=device),
            'x_beta': torch.as_tensor(x_beta, dtype=torch.float32, device=device),
            'x_gamma': torch.as_tensor(x_gamma, dtype=torch.float32, device=device),
            'term_times': torch.as_tensor(term_times, dtype=torch.long, device=device),
            'term_k': torch.as_tensor(term_k, dtype=torch.long, device=device),
            'term_kind': torch.as_tensor(term_kind, dtype=torch.long, device=device),
        }

    def _refresh_loss_tensor_cache(self, device=None, force: bool = False) -> dict:
        """Build tensors and sparse metadata reused by all differentiable loss calls."""
        if device is None:
            device = self.device
        signature = self._loss_tensor_cache_key(device)
        if (
            not force
            and not self._loss_tensor_cache_dirty
            and self._loss_tensor_cache is not None
            and self._loss_tensor_cache_signature == signature
        ):
            return self._loss_tensor_cache

        cache = {
            'x': torch.as_tensor(self.x, dtype=torch.float32, device=device),
            'mu': torch.as_tensor(self.mu, dtype=torch.float32, device=device),
            'mu_abs': torch.abs(torch.as_tensor(self.mu, dtype=torch.float32, device=device)),
            'lambda_vals': torch.as_tensor(self.lambda_vals, dtype=torch.float32, device=device),
            'direction_signs': torch.as_tensor(
                self._get_surrogate_direction_signs() * self._sign4_curriculum_factors(
                    self.num_coupling_constraints
                ),
                dtype=torch.float32,
                device=device,
            ),
            'template_rhs_base': torch.as_tensor(
                self.template_rhs_base_vector[: self.num_coupling_constraints],
                dtype=torch.float32,
                device=device,
            ),
            'lambda_inherent': [
                self._lambda_inherent_tensor_dict(sample_id, device)
                for sample_id in range(self.n_samples)
            ],
            'x_stationarity_const': torch.as_tensor(
                np.asarray([
                    self._x_stationarity_inherent_np(sample_id)
                    for sample_id in range(self.n_samples)
                ], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            'pg_stationarity_const': torch.as_tensor(
                np.asarray([
                    self._pg_stationarity_const_np(sample_id)
                    for sample_id in range(self.n_samples)
                ], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            'metadata': [
                self._build_loss_constraint_metadata(sample_id, device)
                for sample_id in range(self.n_samples)
            ],
            'prev': {},
        }
        prev_specs = [
            ('alpha', self._prev_alpha_values),
            ('beta', self._prev_beta_values),
            ('gamma', self._prev_gamma_values),
            ('delta', self._prev_delta_values),
            ('cost', self._prev_cost_values),
            ('pg_cost', self._prev_pg_cost_values),
        ]
        for name, value in prev_specs:
            if value is not None:
                cache['prev'][name] = torch.as_tensor(value, dtype=torch.float32, device=device)

        self._loss_tensor_cache = cache
        self._loss_tensor_cache_signature = signature
        self._loss_tensor_cache_dirty = False
        return cache

    def _loss_cached_sample(self, sample_id: int, device=None) -> tuple[dict, dict]:
        cache = self._refresh_loss_tensor_cache(device=device)
        return cache, cache['metadata'][sample_id]

    # --------- 单机组 0/1 预测器（用于 single-time 切片的 alpha/delta 生成） ---------

    def _unit_predictor_active(self) -> bool:
        """当开关启用、predictor 存在、且策略包含 single-time 段时返回 True。"""
        if not getattr(self, 'use_unit_predictor', False):
            return False
        if getattr(self, 'unit_predictor', None) is None:
            return False
        if not TORCH_AVAILABLE:
            return False
        return self.constraint_generation_strategy in (
            CONSTRAINT_STRATEGY_ALL_SINGLE_TIME,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        )

    def _unit_predictor_finetune_active(self) -> bool:
        """Warmup gates predictor finetuning, not predictor-defined single constraints."""
        if not self._unit_predictor_active():
            return False
        return int(getattr(self, 'iter_number', 0)) >= int(getattr(self, 'predictor_warmup_rounds', 0))

    def _unit_predictor_slice(self) -> tuple[int, int]:
        """返回 single-time 段在约束向量中的 [start, end) 范围；不启用时返回 (0, 0)。"""
        if not self._unit_predictor_active():
            return (0, 0)
        nc = int(self.num_coupling_constraints)
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME:
            end = min(self.T, nc)
            return (0, end)
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
            head = self.all_mode_group_size * max(self.T - 2, 0)
            end = min(head + self.T, nc)
            start = min(head, end)
            return (start, end)
        return (0, 0)

    def _single_time_coupling_slice(self) -> tuple[int, int]:
        """single-time 尾段在 μ 向量中的 [start, end)；仅由 constraint_generation_strategy 决定（不依赖 predictor）。"""
        nc = int(self.num_coupling_constraints)
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME:
            end = min(self.T, nc)
            return (0, end)
        if self.constraint_generation_strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
            head = self.all_mode_group_size * max(self.T - 2, 0)
            end = min(head + self.T, nc)
            start = min(head, end)
            return (start, end)
        return (0, 0)

    def _current_single_mu_cap(self) -> tuple[float | None, float]:
        """Return the active soft cap for single-time surrogate duals."""
        final_weight = max(
            float(getattr(
                self,
                'single_mu_cap_final_weight',
                getattr(self, 'single_mu_cap_penalty_weight', 0.0),
            ) or 0.0),
            0.0,
        )
        initial_weight = max(
            float(getattr(self, 'single_mu_cap_initial_weight', 0.0) or 0.0),
            0.0,
        )
        if initial_weight <= 0.0 and final_weight <= 0.0:
            return None, 0.0
        st, en = self._single_time_coupling_slice()
        if st >= en:
            return None, 0.0
        initial = getattr(self, 'single_mu_cap_initial', None)
        final = getattr(self, 'single_mu_cap_final', None)
        if initial is None and final is None:
            return None, 0.0
        if initial is None:
            initial = final
        if final is None:
            final = initial
        start = max(int(getattr(self, 'single_mu_cap_start_round', 0) or 0), 0)
        end = max(int(getattr(self, 'single_mu_cap_end_round', start) or start), start)
        it = int(getattr(self, 'iter_number', 0) or 0)
        if it < start:
            return None, 0.0
        frac = 1.0 if end <= start else min(max((it - start) / float(end - start), 0.0), 1.0)
        cap = float(initial) + frac * (float(final) - float(initial))
        weight = initial_weight + frac * (final_weight - initial_weight)
        return max(cap, 0.0), weight

    def _log_single_mu_cap_status(
        self,
        prefix: str,
        raw_mu_values: np.ndarray | None = None,
        final_mu_values: np.ndarray | None = None,
    ) -> None:
        """Print per-round diagnostics for the single-time mu soft-cap schedule."""
        st, en = self._single_time_coupling_slice()
        cap, weight = self._current_single_mu_cap()
        if st >= en:
            print(f"{prefix}[dual-mu] single_cap=unavailable slice={st}:{en}", flush=True)
            return

        def _stats(values: np.ndarray | None) -> tuple[float, float, int]:
            if values is None:
                return 0.0, 0.0, 0
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.size == 0 or arr.shape[-1] < en:
                return 0.0, 0.0, 0
            vals = np.abs(arr[:, st:en].reshape(-1))
            if vals.size == 0:
                return 0.0, 0.0, 0
            max_abs = float(np.max(vals))
            mean_abs = float(np.mean(vals))
            if cap is None:
                return max_abs, mean_abs, 0
            return max_abs, mean_abs, int(np.sum(vals > float(cap) + 1e-7))

        raw_max, raw_mean, raw_exceed = _stats(raw_mu_values)
        final_source = self.mu if final_mu_values is None else final_mu_values
        final_max, _final_mean, final_exceed = _stats(final_source)
        state = "active" if cap is not None and weight > 0.0 else "inactive"
        cap_text = "None" if cap is None else f"{cap:.6f}"
        print(
            f"{prefix}[dual-mu] single_cap={state} cap={cap_text} weight={weight:.6f} "
            f"slice={st}:{en} raw_max={raw_max:.6f} raw_mean={raw_mean:.6f} "
            f"raw_exceed={raw_exceed} final_max={final_max:.6f} final_exceed={final_exceed}",
            flush=True,
        )

    def _build_single_mu_cap_penalty_obj(self, model, mu_abs, vars_dict=None):
        """Build/update Gurobi soft-cap penalty for the single-time mu tail."""
        cap, weight = self._current_single_mu_cap()
        if cap is None or weight <= 0.0:
            return gp.LinExpr()
        st, en = self._single_time_coupling_slice()
        if st >= en:
            return gp.LinExpr()

        if vars_dict is None:
            excess = model.addVars(range(st, en), lb=0.0, name='single_mu_cap_excess')
            for k in range(st, en):
                model.addConstr(excess[k] >= mu_abs[k] - cap, name=f'single_mu_cap_{k}')
            return weight * gp.quicksum(excess[k] for k in range(st, en))

        excess = vars_dict.get('single_mu_cap_excess')
        constrs = vars_dict.get('single_mu_cap_constrs')
        if excess is None or constrs is None:
            excess = model.addVars(range(st, en), lb=0.0, name='single_mu_cap_excess')
            constrs = {}
            for k in range(st, en):
                constrs[k] = model.addConstr(
                    excess[k] >= mu_abs[k] - cap,
                    name=f'single_mu_cap_{k}',
                )
            vars_dict['single_mu_cap_excess'] = excess
            vars_dict['single_mu_cap_constrs'] = constrs
            vars_dict['single_mu_cap_slice'] = (st, en)
            model.update()
        else:
            for constr in constrs.values():
                constr.RHS = -cap
        return weight * gp.quicksum(excess[k] for k in range(st, en))

    def _single_time_k_for_t(self, slice_start: int, t: int) -> int:
        """single-time 段内第 t 个槽位对应的全局耦合约束下标 k（与 μ 向量切片对齐）。"""
        return int(slice_start) + int(t)

    def _apply_mu2_coupling_dual_init(
        self, sample_id: int, mu2_per_t: np.ndarray | None
    ) -> None:
        """将按时段的真值边界 LP 对偶写入 μ₂ 段；μ₁ 不动。mu2_per_t 长度应为 T。"""
        if mu2_per_t is None:
            return
        st, en = self._single_time_coupling_slice()
        if st >= en:
            return
        vec = np.asarray(mu2_per_t, dtype=float).reshape(-1)
        width = en - st
        if vec.size == 0:
            return
        take = min(width, vec.size)
        seg = np.abs(vec[:take])
        out = np.full(width, float(self.mu_lower_bound), dtype=float)
        out[:take] = np.maximum(seg, float(self.mu_lower_bound))
        self.mu[sample_id, st:en] = out

    def _apply_x_fix_dual_init(
        self,
        sample_id: int,
        x_ref: np.ndarray,
        x_fix_dual_contrib: np.ndarray | None,
    ) -> None:
        """Split signed x-fix duals into natural x bounds or single-time surrogate duals.

        The signed value is interpreted in the same stationarity convention used by
        cal_viol_components: positive contributes like x<=., negative like x>=.
        """
        if x_fix_dual_contrib is None:
            return
        if not (0 <= int(sample_id) < len(self.lambda_inherent)):
            return
        lam_inh = self.lambda_inherent[sample_id]
        if lam_inh is None:
            return

        contrib = np.asarray(x_fix_dual_contrib, dtype=float).reshape(-1)
        x_arr = np.asarray(x_ref, dtype=float).reshape(-1)
        if contrib.size == 0 or x_arr.size == 0:
            return

        st, en = self._single_time_coupling_slice()
        single_width = max(int(en) - int(st), 0)
        single_mu = np.zeros(single_width, dtype=float)

        n = min(self.T, contrib.size, x_arr.size)
        for t in range(n):
            d = float(contrib[t])
            if abs(d) <= 1e-10:
                continue
            is_on = float(x_arr[t]) >= 0.5
            if not is_on:
                if d < 0.0:
                    lam_inh['lambda_x_lower'][t] += -d
                elif t < single_width:
                    single_mu[t] = max(single_mu[t], d)
            else:
                if d > 0.0:
                    lam_inh['lambda_x_upper'][t] += d
                elif t < single_width:
                    single_mu[t] = max(single_mu[t], -d)

        if single_width > 0:
            self.mu[sample_id, st:en] = single_mu

    def _unit_predictor_parameters(self) -> list:
        if not self._unit_predictor_active():
            return []
        return list(self.unit_predictor.get_network(self.unit_id).parameters())

    def _compute_single_time_alpha_delta_tensor(
        self, features_tensor: torch.Tensor, train_predictor: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用 predictor 可微前向得到 (alpha, delta)，形状与 forward_logits 输出一致。

        仅传入 ``get_feature_vector_from_sample`` 维度的场景向量（与训练一致），
        不得使用主 surrogate 的 ``[场景, λ, unit_params]`` 全向量。

        mapping:
            x_hat   = sigmoid(logits)
            alpha_t = 1 - 2 * x_hat
            delta_t = x_hat * (1 - 2 * x_hat)        # 等价 alpha_t * x_hat
        """
        dim = int(getattr(self, "_scenario_feature_dim", 0) or 0)
        if dim <= 0:
            dim = len(get_feature_vector_from_sample(dict(self.active_set_data[0])))
            self._scenario_feature_dim = dim
        scenario = features_tensor[..., :dim]
        if getattr(self, "unit_predictor", None) is not None:
            exp_in = int(self.unit_predictor.input_dim)
            if int(scenario.shape[-1]) != exp_in:
                raise ValueError(
                    f"UnitPredictor expects input_dim={exp_in}, got scenario width={int(scenario.shape[-1])}"
                )
        if train_predictor:
            logits = self.unit_predictor.forward_logits(self.unit_id, scenario)
        else:
            net = self.unit_predictor.get_network(self.unit_id)
            was_training = bool(net.training)
            net.eval()
            with torch.no_grad():
                logits = self.unit_predictor.forward_logits(self.unit_id, scenario)
            if was_training:
                net.train()
        x_hat = torch.sigmoid(logits)
        alpha = 1.0 - 2.0 * x_hat
        delta = x_hat * alpha
        return alpha, delta

    def _apply_unit_predictor_override(
        self,
        alphas_tensor: torch.Tensor,
        betas_tensor: torch.Tensor,
        gammas_tensor: torch.Tensor,
        deltas_tensor: torch.Tensor,
        features_tensor: torch.Tensor,
        train_predictor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 single-time 段的 (alpha, beta, gamma, delta) 替换为 predictor 派生值。

        输入/输出均假设 (..., nc) 的一维或二维张量（最后一维为约束数）。
        beta/gamma 在 single-time 段会被置零；sign4 段保持原张量不动。
        """
        if not self._unit_predictor_active():
            return alphas_tensor, betas_tensor, gammas_tensor, deltas_tensor
        start, end = self._unit_predictor_slice()
        if end <= start:
            return alphas_tensor, betas_tensor, gammas_tensor, deltas_tensor

        alpha_pred, delta_pred = self._compute_single_time_alpha_delta_tensor(
            features_tensor, train_predictor=train_predictor,
        )
        # alpha_pred / delta_pred 形状: (B, T) 或 (T,)；需要按张量 ndim 对齐
        target_len = end - start
        if alphas_tensor.ndim == 1:
            # 1D 约束张量对应 batch=1 或 squeeze 过的情形；把 predictor (B, T) 压成 (T,)
            alpha_flat = alpha_pred.reshape(-1)
            delta_flat = delta_pred.reshape(-1)
            if alpha_flat.numel() > self.T:
                alpha_flat = alpha_flat[: self.T]
                delta_flat = delta_flat[: self.T]
            segment_alpha = alpha_flat[:target_len]
            segment_delta = delta_flat[:target_len]
            zeros_seg = torch.zeros_like(segment_alpha)
            new_alpha = torch.cat([alphas_tensor[:start], segment_alpha, alphas_tensor[end:]])
            new_beta = torch.cat([betas_tensor[:start], zeros_seg, betas_tensor[end:]])
            new_gamma = torch.cat([gammas_tensor[:start], zeros_seg, gammas_tensor[end:]])
            new_delta = torch.cat([deltas_tensor[:start], segment_delta, deltas_tensor[end:]])
            return new_alpha, new_beta, new_gamma, new_delta
        # 2D 情形 (B, nc)
        if alpha_pred.ndim == 1:
            alpha_pred = alpha_pred.unsqueeze(0).expand(alphas_tensor.shape[0], -1)
            delta_pred = delta_pred.unsqueeze(0).expand(alphas_tensor.shape[0], -1)
        segment_alpha = alpha_pred[:, :target_len]
        segment_delta = delta_pred[:, :target_len]
        zeros_seg = torch.zeros_like(segment_alpha)
        new_alpha = torch.cat(
            [alphas_tensor[:, :start], segment_alpha, alphas_tensor[:, end:]], dim=1,
        )
        new_beta = torch.cat(
            [betas_tensor[:, :start], zeros_seg, betas_tensor[:, end:]], dim=1,
        )
        new_gamma = torch.cat(
            [gammas_tensor[:, :start], zeros_seg, gammas_tensor[:, end:]], dim=1,
        )
        new_delta = torch.cat(
            [deltas_tensor[:, :start], segment_delta, deltas_tensor[:, end:]], dim=1,
        )
        return new_alpha, new_beta, new_gamma, new_delta

    def _ensure_unit_predictor_optimizer(self, learning_rate: float | None = None) -> None:
        """按 finetune LR 为 predictor 对应机组的网络构造优化器（重复调用安全）。"""
        if not self._unit_predictor_finetune_active():
            self._unit_predictor_optimizer = None
            self._unit_predictor_scheduler = None
            return
        resolved_lr = (
            self.unit_predictor_finetune_lr if learning_rate is None
            else max(float(learning_rate), 1e-10)
        )
        params = self._unit_predictor_parameters()
        current_lr = None
        if self._unit_predictor_optimizer is not None:
            current_lr = self._unit_predictor_optimizer.param_groups[0].get('lr')
        if self._unit_predictor_optimizer is None or current_lr != resolved_lr:
            self._unit_predictor_optimizer = optim.Adam(
                params,
                lr=resolved_lr,
                weight_decay=self.unit_predictor_weight_decay,
            )

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
            for param in self._pg_network_parameters():
                param.requires_grad_(True)

    def _set_main_training_mode(self, enabled: bool) -> None:
        """冻结 c_pg 分支，仅训练 alpha/beta/gamma/delta 与 x-cost 分支。"""
        for param in self.surrogate_net.parameters():
            param.requires_grad_(not enabled)
        if enabled:
            for param in self._main_network_parameters() + self._cost_network_parameters():
                param.requires_grad_(True)

    def _main_direct_target_bounds(self, nc: int) -> tuple[np.ndarray, np.ndarray]:
        coeff_scale = float(getattr(self.surrogate_net, "coupling_coeff_scale", 2.0))
        cost_scale = float(getattr(self.surrogate_net, "x_cost_scale", getattr(self, "x_cost_scale", 2.0)))
        delta_base = float(getattr(self.surrogate_net, "delta_base", 3.0))
        delta_scale = float(getattr(self.surrogate_net, "delta_scale", 2.0))
        if self._uses_template_rhs_bases():
            base = np.asarray(self.template_rhs_base_vector[:nc], dtype=np.float64)
        else:
            base = np.zeros(nc, dtype=np.float64)
        lower = np.concatenate([
            np.full(3 * nc, -coeff_scale, dtype=np.float64),
            base + delta_base - delta_scale,
            np.full(self.T, -cost_scale, dtype=np.float64),
        ])
        upper = np.concatenate([
            np.full(3 * nc, coeff_scale, dtype=np.float64),
            base + delta_base + delta_scale,
            np.full(self.T, cost_scale, dtype=np.float64),
        ])
        return lower, upper

    def _build_main_direct_targets(self, config: dict) -> dict[str, np.ndarray | float]:
        """构造 NN-main 直接拟合标签：用当前 x/mu/固有对偶解一个带锚点的 KKT 代理最小二乘。"""
        n = int(self.n_samples)
        nc = int(self.num_coupling_constraints)
        n_vars = 4 * nc + self.T
        target_blend = min(max(float(config.get("target_blend", 0.75)), 0.0), 1.0)
        mu_threshold = float(config.get("mu_active_threshold", 1e-7))
        dual_w = float(config.get("dual_eq_weight", 1.0))
        active_w = float(config.get("active_opt_weight", 0.8))
        inactive_w = float(config.get("inactive_margin_weight", 0.0))
        inactive_margin = float(config.get("inactive_margin", 0.0))
        anchor = float(config.get("anchor_weight", 0.15))
        coeff_anchor = float(config.get("coeff_anchor_weight", anchor))
        delta_anchor = float(config.get("delta_anchor_weight", anchor))
        cost_anchor = float(config.get("cost_anchor_weight", anchor))

        lower, upper = self._main_direct_target_bounds(nc)
        alphas = np.zeros((n, nc), dtype=np.float32)
        betas = np.zeros((n, nc), dtype=np.float32)
        gammas = np.zeros((n, nc), dtype=np.float32)
        deltas = np.zeros((n, nc), dtype=np.float32)
        costs = np.zeros((n, self.T), dtype=np.float32)
        solved_residuals: list[float] = []

        for sample_id in range(n):
            y0 = np.concatenate([
                np.asarray(self.alpha_values[sample_id][:nc], dtype=np.float64),
                np.asarray(self.beta_values[sample_id][:nc], dtype=np.float64),
                np.asarray(self.gamma_values[sample_id][:nc], dtype=np.float64),
                np.asarray(self.delta_values[sample_id][:nc], dtype=np.float64),
                np.asarray(self.cost_values[sample_id][:self.T], dtype=np.float64),
            ])
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            weights: list[float] = []
            x_val = np.asarray(self.x[sample_id], dtype=np.float64).reshape(-1)
            mu_val = np.asarray(self.mu[sample_id], dtype=np.float64).reshape(-1)[:nc]
            inherent = self._x_stationarity_inherent_np(sample_id)
            offsets_by_k = self._constraint_offsets_for_sample(sample_id)
            signs = self._get_surrogate_direction_signs(nc)
            curriculum_factors = self._sign4_curriculum_factors(nc)

            for t in range(self.T):
                row = np.zeros(n_vars, dtype=np.float64)
                row[4 * nc + t] = 1.0
                for k, ts in enumerate(self.sensitive_timesteps[sample_id][:nc]):
                    for time_idx, kind in iterate_surrogate_constraint_terms(
                        int(ts), offsets_by_k[k], 0, 1, 2, self.T,
                    ):
                        if int(time_idx) != t:
                            continue
                        row[int(kind) * nc + k] += float(signs[k]) * float(curriculum_factors[k]) * float(mu_val[k])
                rows.append(row)
                rhs.append(-float(inherent[t]))
                weights.append(max(dual_w, 0.0))

            for k, ts in enumerate(self.sensitive_timesteps[sample_id][:nc]):
                row = np.zeros(n_vars, dtype=np.float64)
                for time_idx, kind in iterate_surrogate_constraint_terms(
                    int(ts), offsets_by_k[k], 0, 1, 2, self.T,
                ):
                    row[int(kind) * nc + k] += float(signs[k]) * float(x_val[int(time_idx)])
                row[3 * nc + k] = -float(signs[k])
                if abs(float(mu_val[k])) > mu_threshold and active_w > 0:
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
                anchor_diag = np.concatenate([
                    np.full(3 * nc, max(coeff_anchor, 1e-8), dtype=np.float64),
                    np.full(nc, max(delta_anchor, 1e-8), dtype=np.float64),
                    np.full(self.T, max(cost_anchor, 1e-8), dtype=np.float64),
                ])
                lhs = mat_w.T @ mat_w + np.diag(anchor_diag)
                rhs_vec = mat_w.T @ vec_w + anchor_diag * y0
                try:
                    y = np.linalg.solve(lhs, rhs_vec)
                except np.linalg.LinAlgError:
                    y = np.linalg.lstsq(lhs, rhs_vec, rcond=None)[0]
                solved_residuals.append(float(np.mean(np.abs(mat @ y - vec))) if vec.size else 0.0)
            else:
                y = y0.copy()
                solved_residuals.append(0.0)
            y = np.clip(y, lower, upper)
            y = (1.0 - target_blend) * y0 + target_blend * y
            alphas[sample_id] = y[:nc]
            betas[sample_id] = y[nc:2 * nc]
            gammas[sample_id] = y[2 * nc:3 * nc]
            deltas[sample_id] = y[3 * nc:4 * nc]
            costs[sample_id] = y[4 * nc:4 * nc + self.T]

        return {
            "alphas": alphas,
            "betas": betas,
            "gammas": gammas,
            "deltas": deltas,
            "costs": costs,
            "proxy_residual_mean": float(np.mean(solved_residuals)) if solved_residuals else 0.0,
        }

    def _main_direct_fast_metrics(self, sample_features, targets, target_scales) -> dict[str, float]:
        nc = int(self.num_coupling_constraints)
        totals = {"direct_mae": 0.0, "obj_primal": 0.0, "obj_dual_x": 0.0, "obj_opt": 0.0}
        with torch.no_grad():
            for sample_id, features in enumerate(sample_features):
                out = self.surrogate_net.forward_main(features.unsqueeze(0))
                a = out[0].squeeze(0)[:nc]
                b = out[1].squeeze(0)[:nc]
                g = out[2].squeeze(0)[:nc]
                d = self._postprocess_delta_tensor(out[3].squeeze(0)[:nc])
                c = out[4].squeeze(0)[:self.T]
                direct_err = (
                    torch.mean(torch.abs((a - targets["alphas"][sample_id]) / target_scales["coeff"]))
                    + torch.mean(torch.abs((b - targets["betas"][sample_id]) / target_scales["coeff"]))
                    + torch.mean(torch.abs((g - targets["gammas"][sample_id]) / target_scales["coeff"]))
                    + torch.mean(torch.abs((d - targets["deltas"][sample_id]) / target_scales["delta"]))
                    + torch.mean(torch.abs((c - targets["costs"][sample_id]) / target_scales["cost"]))
                ) / 5.0
                totals["direct_mae"] += float(direct_err.detach().cpu().item())
                _, comps = self.loss_function_differentiable(
                    sample_id, a, b, g, d, c, self.device, return_components=True,
                )
                totals["obj_primal"] += float(comps["obj_primal"].detach().cpu().item())
                totals["obj_dual_x"] += float(comps["obj_dual_x"].detach().cpu().item())
                totals["obj_opt"] += float(comps["obj_opt"].detach().cpu().item())
        denom = max(self.n_samples, 1)
        return {k: float(v / denom) for k, v in totals.items()}

    def iter_with_main_direct_targets(self, config: dict | None = None):
        """按 snapshot 脚本中的 NN-main direct-target 策略预训练主代理分支。"""
        config = dict(self.main_direct_train_config if config is None else config)
        epochs = int(config.get("direct_epochs", 0) or 0)
        if not TORCH_AVAILABLE or epochs <= 0:
            return None
        _, batch_size, shuffle = self._resolve_nn_batch_config(
            batch_size=config.get("direct_batch_size"),
            batch_strategy=config.get("direct_batch_strategy", "full-batch"),
            shuffle=config.get("direct_shuffle", False),
        )
        nc = int(self.num_coupling_constraints)
        lr = float(config.get("direct_lr", 2e-3))
        cost_lr = float(config.get("direct_cost_lr", lr * 0.25))
        eta_min_ratio = float(config.get("direct_eta_min_ratio", 0.1))
        grad_clip = float(config.get("direct_grad_clip", 3.0))
        wd = float(config.get("adam_weight_decay", 0.0) or 0.0)
        loss_kind = str(config.get("direct_loss", "mse")).strip().lower()
        beta = float(config.get("direct_beta", 1.0))
        log_interval = int(config.get("direct_log_interval", 50) or 0)
        target_check_interval = int(config.get("direct_target_check_interval", log_interval) or 0)
        direct_mae_target = config.get("direct_mae_target")
        direct_mae_target = None if direct_mae_target is None else float(direct_mae_target)
        feature_noise_std = max(float(config.get("feature_noise_std", 0.0) or 0.0), 0.0)
        coeff_w = float(config.get("coeff_loss_weight", 1.0))
        delta_w = float(config.get("delta_loss_weight", 0.6))
        cost_w = float(config.get("cost_loss_weight", 0.8))
        proxy_kkt_w = float(config.get("proxy_kkt_loss_weight", 0.0))

        np_targets = self._build_main_direct_targets(config)
        sample_features = [
            torch.tensor(self._extract_features(i), dtype=torch.float32, device=self.device)
            for i in range(self.n_samples)
        ]
        targets = {
            key: torch.tensor(np_targets[key], dtype=torch.float32, device=self.device)
            for key in ("alphas", "betas", "gammas", "deltas", "costs")
        }
        target_scales = {
            "coeff": torch.clamp(
                torch.mean(torch.abs(torch.cat([targets["alphas"], targets["betas"], targets["gammas"]], dim=1))),
                min=0.25,
            ),
            "delta": torch.clamp(torch.mean(torch.abs(targets["deltas"])), min=1.0),
            "cost": torch.clamp(torch.mean(torch.abs(targets["costs"])), min=0.25),
        }
        print(
            f"[Unit-{self.unit_id}][direct-NN-main] direct_loss={loss_kind}, "
            f"proxy_residual_mean={np_targets['proxy_residual_mean']:.6f}",
            flush=True,
        )

        def group_loss(pred, target, scale, weight):
            err = (pred - target) / scale
            if loss_kind in ("mse", "l2"):
                return weight * torch.sum(err * err)
            return weight * F.smooth_l1_loss(err, torch.zeros_like(err), beta=beta, reduction="sum")

        self.surrogate_net.train()
        self._set_main_training_mode(True)
        optimizer_main = optim.AdamW(self._main_network_parameters(), lr=lr, weight_decay=wd)
        optimizer_cost = optim.AdamW(self._cost_network_parameters(), lr=cost_lr, weight_decay=wd)
        scheduler_main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main, T_max=max(epochs, 1), eta_min=lr * max(eta_min_ratio, 0.0),
        )
        scheduler_cost = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_cost, T_max=max(epochs, 1), eta_min=cost_lr * max(eta_min_ratio, 0.0),
        )
        last_avg = None
        last_fast = None
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                optimizer_main.zero_grad()
                optimizer_cost.zero_grad()
                sample_indices = np.arange(self.n_samples, dtype=int)
                if shuffle and self.n_samples > 1:
                    np.random.shuffle(sample_indices)
                for sample_pos, _sample_id in enumerate(sample_indices):
                    batch_start = (sample_pos // batch_size) * batch_size
                    is_batch_end = ((sample_pos + 1) % batch_size == 0) or sample_pos == self.n_samples - 1
                    if not is_batch_end:
                        continue
                    batch_ids = sample_indices[batch_start: sample_pos + 1]
                    features_tensor = torch.stack([sample_features[int(i)] for i in batch_ids], dim=0)
                    if feature_noise_std > 0:
                        features_tensor = features_tensor + torch.randn_like(features_tensor) * feature_noise_std
                    out = self.surrogate_net.forward_main(features_tensor)
                    pred_a = out[0][:, :nc]
                    pred_b = out[1][:, :nc]
                    pred_g = out[2][:, :nc]
                    pred_d = self._postprocess_delta_tensor(out[3][:, :nc])
                    pred_c = out[4][:, :self.T]
                    batch_index = torch.as_tensor(batch_ids, dtype=torch.long, device=self.device)
                    loss = (
                        group_loss(pred_a, targets["alphas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                        + group_loss(pred_b, targets["betas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                        + group_loss(pred_g, targets["gammas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                        + group_loss(pred_d, targets["deltas"].index_select(0, batch_index), target_scales["delta"], delta_w)
                        + group_loss(pred_c, targets["costs"].index_select(0, batch_index), target_scales["cost"], cost_w)
                    )
                    if proxy_kkt_w > 0:
                        for local_pos, sid in enumerate(batch_ids):
                            loss = loss + proxy_kkt_w * self.loss_function_differentiable(
                                int(sid),
                                pred_a[local_pos],
                                pred_b[local_pos],
                                pred_g[local_pos],
                                pred_d[local_pos],
                                pred_c[local_pos],
                                self.device,
                            )
                    (loss / max(1, len(batch_ids))).backward()
                    epoch_loss += float(loss.detach().cpu().item())
                    params = self._main_network_parameters() + self._cost_network_parameters()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                    optimizer_main.step()
                    optimizer_cost.step()
                    optimizer_main.zero_grad()
                    optimizer_cost.zero_grad()
                scheduler_main.step()
                scheduler_cost.step()
                last_avg = epoch_loss / max(self.n_samples, 1)
                do_print = epoch == 0 or epoch == epochs - 1
                do_target_check = (
                    direct_mae_target is not None
                    and (
                        epoch == epochs - 1
                        or (
                            target_check_interval > 0
                            and (epoch + 1) % target_check_interval == 0
                        )
                    )
                )
                if do_print or do_target_check:
                    last_fast = self._main_direct_fast_metrics(sample_features, targets, target_scales)
                if do_print:
                    print(
                        f"  [Unit-{self.unit_id}][direct-NN-main] epoch {epoch+1:>4}/{epochs}, "
                        f"avg_target_loss={last_avg:.6f}, lr={optimizer_main.param_groups[0]['lr']:.2e}, "
                        f"cost_lr={optimizer_cost.param_groups[0]['lr']:.2e}, "
                        f"direct_mae={last_fast['direct_mae']:.6f}",
                        flush=True,
                    )
        finally:
            self._set_main_training_mode(False)
        self._refresh_cached_surrogate_outputs()
        return {
            "avg_target_loss": None if last_avg is None else float(last_avg),
            "fast": last_fast,
        }

    def _c_pg_direct_target_for_sample(self, sample_id: int) -> np.ndarray:
        """c_pg[t] = -pg stationarity constant."""
        return -self._pg_stationarity_const_np(sample_id).astype(np.float32)

    def iter_with_c_pg_direct_targets(self, config: dict | None = None):
        """按 snapshot 脚本中的 c_pg direct-target 策略预训练 c_pg 分支。"""
        config = dict(self.c_pg_direct_train_config if config is None else config)
        epochs = int(config.get("direct_epochs", 0) or 0)
        if not TORCH_AVAILABLE or epochs <= 0 or not self._pg_costs_active():
            return None
        _, batch_size, shuffle = self._resolve_nn_batch_config(
            batch_size=config.get("direct_batch_size"),
            batch_strategy=config.get("direct_batch_strategy", "full-batch"),
            shuffle=config.get("direct_shuffle", False),
        )
        lr = float(config.get("direct_lr", 2e-3))
        eta_min_ratio = float(config.get("direct_eta_min_ratio", 0.05))
        beta = float(config.get("direct_beta", 1.0))
        loss_kind = str(config.get("direct_loss", "huber")).strip().lower()
        grad_clip = float(config.get("direct_grad_clip", 2.0))
        feature_noise_std = max(float(config.get("feature_noise_std", 0.0) or 0.0), 0.0)
        log_interval = int(config.get("direct_log_interval", 50) or 0)
        target_check_interval = int(config.get("direct_target_check_interval", log_interval) or 0)
        obj_dual_pg_target = config.get("direct_obj_dual_pg_target")
        obj_dual_pg_target = None if obj_dual_pg_target is None else float(obj_dual_pg_target)
        wd = config.get("adam_weight_decay", self.pg_cost_c_pg_adam_weight_decay)
        wd = 0.0 if wd is None else float(wd)

        sample_features = [
            torch.tensor(self._extract_features(i), dtype=torch.float32, device=self.device)
            for i in range(self.n_samples)
        ]
        targets = [
            torch.tensor(self._c_pg_direct_target_for_sample(i), dtype=torch.float32, device=self.device)
            for i in range(self.n_samples)
        ]
        full_features_tensor = torch.stack(sample_features, dim=0)
        full_target_tensor = torch.stack(targets, dim=0)
        target_scale = config.get("direct_target_scale")
        if target_scale is None:
            target_scale_tensor = torch.clamp(torch.mean(torch.abs(full_target_tensor)), min=1.0)
        else:
            target_scale_tensor = torch.as_tensor(
                max(float(target_scale), 1e-6), dtype=torch.float32, device=self.device,
            )
        print(
            f"[Unit-{self.unit_id}][direct-c_pg] direct_loss={loss_kind}, "
            f"target_scale={float(target_scale_tensor.detach().cpu().item()):.6f}",
            flush=True,
        )

        def direct_target_obj_dual_pg() -> float:
            with torch.no_grad():
                out = self._gate_pg_cost_tensor(
                    self.surrogate_net.forward_pg_cost(full_features_tensor)[:, : self.T]
                )
                return float(torch.sum(torch.abs(out - full_target_tensor)).detach().cpu().item())

        optimizer = optim.AdamW(self._pg_network_parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=lr * max(eta_min_ratio, 0.0),
        )
        self.surrogate_net.train()
        self._set_c_pg_training_mode(True)
        last_avg = None
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                optimizer.zero_grad()
                sample_indices = np.arange(self.n_samples, dtype=int)
                if shuffle and self.n_samples > 1:
                    np.random.shuffle(sample_indices)
                for sample_pos, _sample_id in enumerate(sample_indices):
                    batch_start = (sample_pos // batch_size) * batch_size
                    is_batch_end = ((sample_pos + 1) % batch_size == 0) or sample_pos == self.n_samples - 1
                    if not is_batch_end:
                        continue
                    batch_ids = sample_indices[batch_start: sample_pos + 1]
                    features_tensor = torch.stack([sample_features[int(i)] for i in batch_ids], dim=0)
                    if feature_noise_std > 0:
                        features_tensor = features_tensor + torch.randn_like(features_tensor) * feature_noise_std
                    target = torch.stack([targets[int(i)] for i in batch_ids], dim=0)
                    out = self._gate_pg_cost_tensor(
                        self.surrogate_net.forward_pg_cost(features_tensor)[:, : self.T]
                    )
                    if loss_kind in ("mse", "l2"):
                        err = (out - target) / target_scale_tensor
                        loss = torch.sum(err * err)
                    else:
                        loss = F.smooth_l1_loss(out, target, beta=beta, reduction="sum")
                    if self.pg_cost_softbound_weight > 0 and self.pg_cost_scale > 0:
                        scale = torch.as_tensor(float(self.pg_cost_scale), dtype=out.dtype, device=out.device)
                        excess = self._smooth_relu(torch.abs(out) - scale, eps=self.pg_cost_smooth_abs_eps)
                        loss = loss + float(self.pg_cost_softbound_weight) * torch.sum(excess * excess)
                    (loss / max(1, len(batch_ids))).backward()
                    epoch_loss += float(loss.detach().cpu().item())
                    torch.nn.utils.clip_grad_norm_(self._pg_network_parameters(), max_norm=grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                last_avg = epoch_loss / max(self.n_samples, 1)
                do_print = epoch == 0 or epoch == epochs - 1
                do_target_check = (
                    obj_dual_pg_target is not None
                    and (
                        epoch == epochs - 1
                        or (
                            target_check_interval > 0
                            and (epoch + 1) % target_check_interval == 0
                        )
                    )
                )
                metric_obj_dual_pg = (
                    direct_target_obj_dual_pg() if (do_print or do_target_check) else None
                )
                if do_print:
                    print(
                        f"  [Unit-{self.unit_id}][direct-c_pg] epoch {epoch+1:>4}/{epochs}, "
                        f"avg_target_loss={last_avg:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
                        f"fast_obj_dual_pg={metric_obj_dual_pg:.6f}",
                        flush=True,
                    )
        finally:
            self._set_c_pg_training_mode(False)
        self._refresh_cached_surrogate_outputs()
        return {
            "avg_target_loss": None if last_avg is None else float(last_avg),
            "fast_obj_dual_pg": direct_target_obj_dual_pg(),
        }

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

            alphas, betas, gammas, deltas = self._apply_unit_predictor_override(
                alphas, betas, gammas, deltas, feat_tensor,
            )

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

    def _generate_initial_values_from_nn(self):
        """用未训练的 SubproblemSurrogateNet forward pass 生成 alpha/beta/gamma/delta 初值。"""
        print(
            f"  - 正在 NN forward 生成代理约束初值（n_samples={self.n_samples}, device={self.device}）…",
            flush=True,
        )
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

    def _mu_floor_schedule_iter(self) -> int:
        """μ 下界阶段切换用的有效外循环轮次。

        在 Sign4 延期（``sign4_delay_rounds > 0`` 且策略含 sign4 模板）时：
        延期内 ``iter_number < delay`` 返回 ``-1``（视为 Sign4 加入前，保持 individual 相）；
        自 ``iter_number == delay`` 起记 ``t = 0, 1, …``（与 Sign4 权重可非零的起始轮对齐）。
        无延期或非 sign4 模板策略时等价于 ``iter_number``。
        """
        delay = max(int(getattr(self, "sign4_delay_rounds", 0) or 0), 0)
        if delay <= 0:
            return int(self.iter_number)
        if self.constraint_generation_strategy not in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        ):
            return int(self.iter_number)
        i = int(self.iter_number)
        if i < delay:
            return -1
        return i - delay

    def _configure_surrogate_bcd_run(self, max_iter: int) -> None:
        """与外循环 ``max_iter`` 对齐的 BCD 辅助量（μ 符号松弛截止轮等）。"""
        max_i = max(int(max_iter), 1)
        self._surrogate_bcd_max_iter = max_i
        delay_mu = max(int(getattr(self, "sign4_delay_rounds", 0) or 0), 0)
        if (
            delay_mu > 0
            and self.constraint_generation_strategy
            in (
                CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
                CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
            )
        ):
            active_span = max(max_i - delay_mu, 1)
            self._mu_sign_relaxation_last_iter = delay_mu + max(active_span // 2, 0)
        else:
            self._mu_sign_relaxation_last_iter = max(max_i // 2, 0)

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
        # 仅在前半段 BCD 启用：对偶块符号松弛 + 由 μ 更新 surrogate_direction_signs
        cutoff = int(getattr(self, '_mu_sign_relaxation_last_iter', 10**9))
        if int(getattr(self, 'iter_number', 0)) >= cutoff:
            return False
        return (
            interval > 0
            and lb > 0.0
            and ((self.iter_number + 1) % interval == 0)
        )

    def _ensure_surrogate_direction_hysteresis_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        n_constraints = int(self.num_coupling_constraints)
        pending_signs = getattr(self, 'surrogate_direction_pending_signs', None)
        pending_counts = getattr(self, 'surrogate_direction_pending_counts', None)
        if pending_signs is None or np.asarray(pending_signs).shape != (n_constraints,):
            pending_signs = np.zeros(n_constraints, dtype=float)
        else:
            pending_signs = np.asarray(pending_signs, dtype=float).copy()
        if pending_counts is None or np.asarray(pending_counts).shape != (n_constraints,):
            pending_counts = np.zeros(n_constraints, dtype=int)
        else:
            pending_counts = np.asarray(pending_counts, dtype=int).copy()
        self.surrogate_direction_pending_signs = pending_signs
        self.surrogate_direction_pending_counts = pending_counts
        return pending_signs, pending_counts

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
        pending_signs, pending_counts = self._ensure_surrogate_direction_hysteresis_buffers()
        hysteresis_rounds = max(int(getattr(self, 'mu_sign_hysteresis_rounds', 1) or 1), 1)
        flip_min_share = min(max(float(getattr(self, 'mu_sign_flip_min_share', 0.5) or 0.5), 0.5), 1.0)
        desired = base_signs.copy()
        desired[pos_count > neg_count] = 1.0
        desired[neg_count > pos_count] = -1.0

        for k in range(self.num_coupling_constraints):
            current_sign = 1.0 if float(base_signs[k]) >= 0.0 else -1.0
            desired_sign = 1.0 if float(desired[k]) >= 0.0 else -1.0
            if desired_sign == current_sign:
                pending_signs[k] = 0.0
                pending_counts[k] = 0
                resolved[k] = current_sign
                continue

            decisive_votes = int(pos_count[k] + neg_count[k])
            desired_votes = int(pos_count[k] if desired_sign > 0.0 else neg_count[k])
            desired_share = (float(desired_votes) / float(decisive_votes)) if decisive_votes > 0 else 0.0
            if desired_share < flip_min_share:
                pending_signs[k] = 0.0
                pending_counts[k] = 0
                resolved[k] = current_sign
                continue

            if hysteresis_rounds <= 1:
                resolved[k] = desired_sign
                pending_signs[k] = 0.0
                pending_counts[k] = 0
                continue

            if float(pending_signs[k]) == desired_sign:
                pending_counts[k] += 1
            else:
                pending_signs[k] = desired_sign
                pending_counts[k] = 1

            if int(pending_counts[k]) >= hysteresis_rounds:
                resolved[k] = desired_sign
                pending_signs[k] = 0.0
                pending_counts[k] = 0
            else:
                resolved[k] = current_sign

        self.surrogate_direction_pending_signs = pending_signs
        self.surrogate_direction_pending_counts = pending_counts
        return resolved

    def _current_sign4_curriculum_scale(self) -> float:
        rounds = max(int(getattr(self, 'sign4_curriculum_rounds', 0) or 0), 0)
        initial = max(float(getattr(self, 'sign4_initial_scale', 1.0) or 0.0), 0.0)
        final = max(float(getattr(self, 'sign4_final_scale', 1.0) or 0.0), 0.0)
        delay = max(int(getattr(self, 'sign4_delay_rounds', 0) or 0), 0)
        iter_n = float(getattr(self, 'iter_number', 0))
        if delay > 0 and iter_n < float(delay):
            return 0.0
        effective_iter = max(iter_n - float(delay), 0.0)
        if rounds <= 0:
            return final
        progress = min(max(effective_iter, 0.0) / float(rounds), 1.0)
        return initial + (final - initial) * progress

    def _sign4_curriculum_factors(self, size: int) -> np.ndarray:
        factors = np.ones(int(size), dtype=float)
        if self.constraint_generation_strategy not in (
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        ):
            return factors
        n_sign4 = min(int(size), self.all_mode_group_size * max(int(self.T) - 2, 0))
        if n_sign4 > 0:
            factors[:n_sign4] = self._current_sign4_curriculum_scale()
        return factors

    def _apply_surrogate_direction_to_params(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        deltas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        signs = self._get_surrogate_direction_signs(len(alphas))
        factors = signs * self._sign4_curriculum_factors(len(alphas))
        return (
            np.asarray(alphas, dtype=float) * factors,
            np.asarray(betas, dtype=float) * factors,
            np.asarray(gammas, dtype=float) * factors,
            np.asarray(deltas, dtype=float) * factors,
        )

    @staticmethod
    def _normalize_surrogate_delta_reference_scope(scope: str | None) -> str:
        s = (scope or "sign4_only").strip().lower().replace("-", "_")
        if s in ("all", "all_coupling", "full"):
            return "all_coupling"
        return "sign4_only"

    def _lift_surrogate_delta_for_reference_x(self) -> None:
        """初始化一次性：按各样本 ``x_true`` 调整原始 ``δ``，使有效不等式在参考点上满足 ``lhs_eff ≤ δ_eff − ε``（消除代理耦合的 primal 违反项）。

        使用与 primal block 相同的 ``f_k = sign_k × curriculum_k`` 缩放；仅改 ``delta_values``，不改动网络权重。
        """
        if not self.enable_surrogate_delta_reference_lift:
            return
        eps = float(self.surrogate_delta_reference_eps)
        min_f = float(self.surrogate_delta_reference_min_abs_factor)
        scope = self.surrogate_delta_reference_scope
        signs = np.asarray(self._get_surrogate_direction_signs(), dtype=float)
        factors_full = np.asarray(
            self._sign4_curriculum_factors(self.num_coupling_constraints), dtype=float
        )
        f_arr = signs * factors_full
        n_adj = 0
        max_viol = 0.0
        for sample_id in range(self.n_samples):
            x_true = self.active_set_data[sample_id].get("x_true")
            if x_true is None:
                x_ref = np.asarray(self.x[sample_id], dtype=float)
            else:
                x_ref = np.asarray(x_true, dtype=float)
            sensitive = self.sensitive_timesteps[sample_id]
            offsets_list = self._constraint_offsets_for_sample(sample_id)
            n_active = min(len(sensitive), len(offsets_list))
            if n_active <= 0:
                continue
            if scope == "all_coupling":
                ks = range(n_active)
            else:
                if self.constraint_generation_strategy not in (
                    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
                    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
                ):
                    ks = range(0)
                else:
                    n_sign4 = min(
                        self.all_mode_group_size * max(int(self.T) - 2, 0),
                        n_active,
                    )
                    ks = range(n_sign4)
            dv = np.asarray(self.delta_values[sample_id], dtype=float).copy()
            av = np.asarray(self.alpha_values[sample_id], dtype=float)
            bv = np.asarray(self.beta_values[sample_id], dtype=float)
            gv = np.asarray(self.gamma_values[sample_id], dtype=float)
            for k in ks:
                if k < 0 or k >= int(self.num_coupling_constraints):
                    continue
                f_k = float(f_arr[k])
                if abs(f_k) < min_f:
                    continue
                lhs_raw = float(
                    build_surrogate_constraint_expression(
                        x_ref,
                        int(sensitive[k]),
                        offsets_list[k],
                        float(av[k]),
                        float(bv[k]),
                        float(gv[k]),
                        int(self.T),
                    )
                )
                lhs_eff = f_k * lhs_raw
                delta_eff = f_k * float(dv[k])
                viol = lhs_eff - delta_eff
                if viol <= 1e-14:
                    continue
                max_viol = max(max_viol, viol)
                delta_eff_new = lhs_eff + eps
                dv[k] = delta_eff_new / f_k
                n_adj += 1
            self.delta_values[sample_id] = dv
        if n_adj > 0:
            print(
                f"  [Unit-{self.unit_id}] surrogate δ 参考锚定: 调整 {n_adj} 项, "
                f"max(lhs_eff−δ_eff)≈{max_viol:.6g}（scope={scope}, ε={eps:g}）",
                flush=True,
            )

    def _constraint_offsets_for_sample(self, sample_id: int) -> list[tuple[int, ...]]:
        return resolve_constraint_offsets_from_trainer(
            self,
            sample_id,
            len(self.sensitive_timesteps[sample_id]),
        )

    def _get_mu_lower_bound_phase(self) -> str:
        t = self._mu_floor_schedule_iter()
        if t < 0:
            return "individual"
        if t < self.mu_individual_lower_bound_round:
            return "individual"
        if t < self.mu_group_lower_bound_round:
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
        b     = self.subproblem_generation_no_load_coeff(g)
        sc    = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shc   = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton   = int(self.subproblem_Ton)
        Toff  = int(self.subproblem_Toff)

        print(
            f"  [Unit-{self.unit_id}] 开始初始化求解（{self.n_samples} 个样本，backend={self._lp_backend}）…",
            flush=True,
        )

        def _get_pi(m, name):
            """安全获取约束对偶变量（负值截断为0）"""
            try:
                return max(0.0, m.getConstrByName(name).Pi)
            except Exception:
                return 0.0

        def _get_x_fix_stationarity_contribution(m, name):
            try:
                return -float(m.getConstrByName(name).Pi)
            except Exception:
                return 0.0

        def _gurobi_status_name(st: int) -> str:
            m = {
                int(GRB.LOADED): "LOADED",
                int(GRB.OPTIMAL): "OPTIMAL",
                int(GRB.INFEASIBLE): "INFEASIBLE",
                int(GRB.INF_OR_UNBD): "INF_OR_UNBD",
                int(GRB.UNBOUNDED): "UNBOUNDED",
            }
            return m.get(int(st), f"status_{st}")

        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            assert_lp_backend_available(self._lp_backend)
            progress_every = max(1, min(50, self.n_samples // 6))
            t_init = time.perf_counter()
            for sample_id in range(self.n_samples):
                if (
                    sample_id == 0
                    or (sample_id + 1) % progress_every == 0
                    or sample_id == self.n_samples - 1
                ):
                    print(
                        f"  [Unit-{self.unit_id}] init_solve 样本 {sample_id + 1}/{self.n_samples} "
                        f"（已用 {time.perf_counter() - t_init:.1f}s）",
                        flush=True,
                    )
                # init LP 非最优时 solve_init_lp 直接 raise；不在此用假对偶继续 BCD。
                result = solve_init_lp_backend(self, sample_id)
                x_lp = result['x_sol']
                self.pg[sample_id] = result['pg_sol']
                self.x[sample_id] = x_lp
                self.active_set_data[sample_id]['x_true'] = result['x_true']
                (
                    self.sensitive_timesteps[sample_id],
                    self.surrogate_constraint_offsets[sample_id],
                ) = select_constraint_layout(
                    x_lp,
                    strategy=self.constraint_generation_strategy,
                    threshold_low=0.1,
                    threshold_high=0.9,
                    max_constraints=self.max_constraints,
                )
                self.coc[sample_id] = result['coc_sol']
                self.cpower[sample_id] = result['cpower_sol']
                self.lambda_inherent[sample_id] = result['lambda_inherent']
                self._apply_x_fix_dual_init(
                    sample_id,
                    result.get("x_true", x_lp),
                    result.get("x_fix_dual_contrib"),
                )
            print(
                f"  [Unit-{self.unit_id}] 初始化求解完成（{self.n_samples} 个样本，"
                f"共 {time.perf_counter() - t_init:.1f}s）",
                flush=True,
            )
            return

        progress_every = max(1, min(50, self.n_samples // 6))
        t_init = time.perf_counter()
        for sample_id in range(self.n_samples):
            if (
                sample_id == 0
                or (sample_id + 1) % progress_every == 0
                or sample_id == self.n_samples - 1
            ):
                print(
                    f"  [Unit-{self.unit_id}] init_solve 样本 {sample_id + 1}/{self.n_samples} "
                    f"（已用 {time.perf_counter() - t_init:.1f}s）",
                    flush=True,
                )
            lambda_val = self.lambda_vals[sample_id]

            # 与 solve_init_lp / _recover_unit_x_from_sample 一致：先 UCM 行，再 active_set 覆盖
            x_init = _recover_unit_x_from_sample(self, sample_id)
            s0 = self.active_set_data[sample_id]
            if 'unit_commitment_matrix' in s0 and 'active_set' in s0:
                _x_source = 'ucm+active_set'
            elif 'unit_commitment_matrix' in s0:
                _x_source = 'unit_commitment_matrix'
            elif 'active_set' in s0:
                _x_source = 'active_set'
            else:
                _x_source = 'none'

            if np.all(x_init == 0) and _x_source != 'none':
                print(f"  ⚠ 样本 {sample_id} 机组 {g}: x_init 全零（来源={_x_source}），"
                      f"该机组在所有时段均关机或数据缺失", flush=True)
            elif _x_source == 'none':
                print(f"  ⚠ 样本 {sample_id} 机组 {g}: 数据中既无 active_set 也无 "
                      f"unit_commitment_matrix，x_init 使用全零默认值", flush=True)

            # 求解单机组 LP 松弛，目标: cost - λᵀpg。若「全约束」不可行（真值 x 与最小开/停等冲突），
            # 则自动重试：省略最小开/停线性化，λ_min_on/off 取 0（与 subproblem_lp_solver 文档语义一致）。
            x_target = np.asarray(x_init >= 0.5, dtype=float)
            init_solved = False
            last_status: int | None = None
            for include_mud in (True, False):
                model = gp.Model('init_subproblem_LP' + ('' if include_mud else '_no_mud'))
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

                if include_mud:
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

                # 真值边界（μ₂ 初值影子价格）；x_init 与 active_set / unit_commitment_matrix 一致
                for t in range(self.T):
                    model.addConstr(x[t] == float(x_target[t]), name=f'x_fix_{t}')

                # 目标: min cost - λᵀpg（用于提取有意义的对偶变量）
                obj = (gp.quicksum(cpower[t] for t in range(self.T)) +
                       gp.quicksum(coc[t]    for t in range(self.T-1)) -
                       gp.quicksum(lambda_val[t] * pg[t] for t in range(self.T)))
                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                last_status = int(model.status)
                if last_status != int(GRB.OPTIMAL):
                    model.Params.NumericFocus = 2
                    model.Params.DualReductions = 0
                    model.optimize()
                    last_status = int(model.status)
                if last_status == int(GRB.OPTIMAL):
                    if not include_mud:
                        print(
                            f"  ⚠ 样本 {sample_id} 机组 {g}: 全约束 init LP 不可行/未证最优，"
                            f"已使用**省略最小开/停**的 init LP 继续（min_on/min_off 对偶初值按 0）",
                            flush=True,
                        )
                    x_lp = np.array([x[t].X for t in range(self.T)])    # LP 松弛解（连续）
                    self.pg[sample_id]     = np.array([pg[t].X     for t in range(self.T)])
                    self.x[sample_id]      = x_lp                        # ← 连续松弛解
                    # 将 JSON 整数解存入 sample，作为 iter_with_primal_block 的持久锚点
                    # （不随 BCD 迭代更新，确保 x_true 始终指向原始 MILP 最优解）
                    self.active_set_data[sample_id]['x_true'] = x_target.copy()

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
                    x_fix_dual_contrib = np.zeros(self.T, dtype=float)
                    for t in range(self.T):
                        x_fix_dual_contrib[t] = _get_x_fix_stationarity_contribution(
                            model, f'x_fix_{t}'
                        )
                    self._apply_x_fix_dual_init(sample_id, x_target, x_fix_dual_contrib)
                    init_solved = True
                try:
                    model.dispose()
                except Exception:
                    pass
                if init_solved:
                    break

            if not init_solved:
                raise RuntimeError(
                    f"init LP（Gurobi）未最优: unit_id={g}, sample_id={sample_id}, "
                    f"status={last_status} ({_gurobi_status_name(last_status) if last_status is not None else '?'})"
                )

        print(
            f"  [Unit-{self.unit_id}] 初始化求解完成（Gurobi init_lp，{self.n_samples} 个样本，"
            f"共 {time.perf_counter() - t_init:.1f}s）",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Persistent primal-block model helpers
    # ------------------------------------------------------------------

    def _build_primal_model(self, sample_id: int, alphas, betas, gammas, deltas):
        """一次性建立 primal 块 Gurobi 模型（变量 + 约束结构）。
        耦合约束以初始 alpha/beta/gamma/delta 值建立，后续通过 chgCoeff 更新。
        返回 (model, vars_dict, coupling_constr_refs)。
        """
        g    = self.unit_id
        Pmin = self.gen[g, PMIN]
        Pmax = self.gen[g, PMAX]
        a    = self.gencost[g, -2] / self.T_delta
        b    = self.subproblem_generation_no_load_coeff(g)
        Ru   = float(self.Ru_all[g]); Rd   = float(self.Rd_all[g])
        Ru_co= float(self.Ru_co_all[g]); Rd_co= float(self.Rd_co_all[g])
        sc   = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shc  = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton  = int(self.subproblem_Ton)
        Toff = int(self.subproblem_Toff)

        model = gp.Model('primal_block')
        model.Params.OutputFlag = 0

        pg     = model.addVars(self.T,   lb=0,      name='pg')
        x      = model.addVars(self.T,   lb=0, ub=1, name='x')
        coc    = model.addVars(self.T-1, lb=0,      name='coc')
        cpower = model.addVars(self.T,   lb=0,      name='cpower')

        x_true = self.active_set_data[sample_id].get('x_true', None)
        if x_true is None:
            x_true = self.x[sample_id]
        x_binary_dev    = model.addVars(self.T,   lb=0, name='x_binary_dev')
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
        surrogate_viols    = model.addVars(self.num_coupling_constraints, lb=0, name='surr_viol')
        surrogate_abs_vals = model.addVars(self.num_coupling_constraints, lb=0, name='surr_abs')

        obj_primal = gp.LinExpr()
        obj_binary = gp.LinExpr()

        # x 偏差
        for t in range(self.T):
            model.addConstr(x_binary_dev[t] >= x[t] - float(x_true[t]),  name=f'xdev_pos_{t}')
            model.addConstr(x_binary_dev[t] >= float(x_true[t]) - x[t],  name=f'xdev_neg_{t}')
            obj_binary += x_binary_dev[t]

        # 发电上下限
        for t in range(self.T):
            lo_e = Pmin * x[t] - pg[t]
            up_e = pg[t] - Pmax * x[t]
            model.addConstr(pg_lower_viol[t] >= lo_e,  name=f'pg_lo_v_{t}')
            model.addConstr(pg_lower_abs[t]  >= lo_e,  name=f'pg_lo_a1_{t}')
            model.addConstr(pg_lower_abs[t]  >= -lo_e, name=f'pg_lo_a2_{t}')
            model.addConstr(pg_upper_viol[t] >= up_e,  name=f'pg_up_v_{t}')
            model.addConstr(pg_upper_abs[t]  >= up_e,  name=f'pg_up_a1_{t}')
            model.addConstr(pg_upper_abs[t]  >= -up_e, name=f'pg_up_a2_{t}')
            obj_primal += pg_lower_viol[t] + pg_upper_viol[t]

        # 爬坡约束
        for t in range(1, self.T):
            ru_e = pg[t] - pg[t-1] - Ru*x[t-1] - Ru_co*(1-x[t-1])
            rd_e = pg[t-1] - pg[t] - Rd*x[t] - Rd_co*(1-x[t])
            model.addConstr(ramp_up_viol[t-1]  >= ru_e,  name=f'ru_v_{t}')
            model.addConstr(ramp_up_abs[t-1]   >= ru_e,  name=f'ru_a1_{t}')
            model.addConstr(ramp_up_abs[t-1]   >= -ru_e, name=f'ru_a2_{t}')
            model.addConstr(ramp_down_viol[t-1] >= rd_e,  name=f'rd_v_{t}')
            model.addConstr(ramp_down_abs[t-1]  >= rd_e,  name=f'rd_a1_{t}')
            model.addConstr(ramp_down_abs[t-1]  >= -rd_e, name=f'rd_a2_{t}')
            obj_primal += ramp_up_viol[t-1] + ramp_down_viol[t-1]

        # 最小开关机时间
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                e = x[t1+1] - x[t1] - x[t1+tau]
                model.addConstr(min_on_viol[tau-1,t1] >= e,  name=f'mon_v_{tau}_{t1}')
                model.addConstr(min_on_abs[tau-1,t1]  >= e,  name=f'mon_a1_{tau}_{t1}')
                model.addConstr(min_on_abs[tau-1,t1]  >= -e, name=f'mon_a2_{tau}_{t1}')
                obj_primal += min_on_viol[tau-1, t1]
        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                e = -x[t1+1] + x[t1] - (1 - x[t1+tau])
                model.addConstr(min_off_viol[tau-1,t1] >= e,  name=f'moff_v_{tau}_{t1}')
                model.addConstr(min_off_abs[tau-1,t1]  >= e,  name=f'moff_a1_{tau}_{t1}')
                model.addConstr(min_off_abs[tau-1,t1]  >= -e, name=f'moff_a2_{tau}_{t1}')
                obj_primal += min_off_viol[tau-1, t1]

        # 启停成本
        for t in range(1, self.T):
            sc_e  = sc  * (x[t] - x[t-1]) - coc[t-1]
            shc_e = shc * (x[t-1] - x[t]) - coc[t-1]
            model.addConstr(start_cost_viol[t-1] >= sc_e,   name=f'sc_v_{t}')
            model.addConstr(start_cost_abs[t-1]  >= sc_e,   name=f'sc_a1_{t}')
            model.addConstr(start_cost_abs[t-1]  >= -sc_e,  name=f'sc_a2_{t}')
            model.addConstr(shut_cost_viol[t-1]  >= shc_e,  name=f'shc_v_{t}')
            model.addConstr(shut_cost_abs[t-1]   >= shc_e,  name=f'shc_a1_{t}')
            model.addConstr(shut_cost_abs[t-1]   >= -shc_e, name=f'shc_a2_{t}')
            obj_primal += start_cost_viol[t-1] + shut_cost_viol[t-1]

        # 发电成本定义
        for t in range(self.T):
            model.addConstr(cpower[t] == a*pg[t] + b*x[t], name=f'cpower_{t}')

        # 代理耦合约束（以初始 alpha/beta/gamma/delta 建立，后续 chgCoeff 更新）
        sensitive_t         = self.sensitive_timesteps[sample_id]
        constraint_offsets  = self._constraint_offsets_for_sample(sample_id)
        coupling_constr_refs = []  # list of {'c_viol', 'c_abs_pos', 'c_abs_neg', 'x_vars', 'offsets'}
        for k, t_k in enumerate(sensitive_t):
            off = constraint_offsets[k]
            # 用初始参数值建立约束
            lhs_init = build_surrogate_constraint_expression(
                x, t_k, off, float(alphas[k]), float(betas[k]), float(gammas[k]), self.T)
            rhs_init = float(deltas[k])
            c_viol    = model.addConstr(surrogate_viols[k]    >= lhs_init - rhs_init, name=f'surr_viol_{k}')
            c_abs_pos = model.addConstr(surrogate_abs_vals[k] >= lhs_init - rhs_init, name=f'surr_abs_pos_{k}')
            c_abs_neg = model.addConstr(surrogate_abs_vals[k] >= rhs_init - lhs_init, name=f'surr_abs_neg_{k}')
            # 记录约束中涉及的 x 变量及偏移，供 chgCoeff 使用
            x_vars_with_offset = []
            for dx, xv_var in [(0, x[t_k]), (1, x[t_k+1] if t_k+1 < self.T else None),
                                (2, x[t_k+2] if t_k+2 < self.T else None)]:
                coeff_name = ['alphas', 'betas', 'gammas'][dx]
                if xv_var is not None:
                    x_vars_with_offset.append((xv_var, dx))  # dx: 0=alpha, 1=beta, 2=gamma
            coupling_constr_refs.append({
                'c_viol': c_viol, 'c_abs_pos': c_abs_pos, 'c_abs_neg': c_abs_neg,
                'x_vars': x_vars_with_offset,
                't_k': t_k,
            })
            obj_primal += surrogate_viols[k]

        model.update()

        vars_dict = {
            'pg': pg, 'x': x, 'coc': coc, 'cpower': cpower,
            'x_binary_dev': x_binary_dev,
            'pg_lower_abs': pg_lower_abs, 'pg_upper_abs': pg_upper_abs,
            'ramp_up_abs': ramp_up_abs,   'ramp_down_abs': ramp_down_abs,
            'min_on_abs': min_on_abs,     'min_off_abs': min_off_abs,
            'start_cost_abs': start_cost_abs, 'shut_cost_abs': shut_cost_abs,
            'surrogate_abs_vals': surrogate_abs_vals,
            'obj_primal': obj_primal,
            'obj_binary': obj_binary,
            'Ton': Ton, 'Toff': Toff,
            'coupling_constr_refs': coupling_constr_refs,
            'sensitive_t': sensitive_t,
        }
        return model, vars_dict

    def _update_primal_coupling_coefficients(self, model, vars_dict,
                                              alphas, betas, gammas, deltas):
        """通过 chgCoeff + RHS 更新耦合约束系数（每次 NN 更新后调用）。"""
        refs = vars_dict['coupling_constr_refs']
        for k, info in enumerate(refs):
            coeffs = [float(alphas[k]), float(betas[k]), float(gammas[k])]
            rhs    = float(deltas[k])
            for x_var, dx in info['x_vars']:
                c = coeffs[dx]
                # lhs >= alpha*x_tk + beta*x_tk+1 + gamma*x_tk+2 - delta
                # Ax >= b form: viol - alpha*x - beta*x1 - gamma*x2 >= -delta
                model.chgCoeff(info['c_viol'],    x_var, -c)
                model.chgCoeff(info['c_abs_pos'], x_var, -c)
                model.chgCoeff(info['c_abs_neg'], x_var,  c)
            info['c_viol'].RHS    = -rhs
            info['c_abs_pos'].RHS = -rhs
            info['c_abs_neg'].RHS =  rhs

    def _update_primal_model_objective(self, sample_id: int, model, vars_dict,
                                        alphas, betas, gammas, deltas):
        """每次迭代：更新耦合约束系数，重建目标函数，调用 setObjective。"""
        # 1. 更新耦合约束系数（chgCoeff）
        self._update_primal_coupling_coefficients(model, vars_dict, alphas, betas, gammas, deltas)

        pg  = vars_dict['pg']
        x   = vars_dict['x']
        coc = vars_dict['coc']
        pg_lower_abs    = vars_dict['pg_lower_abs']
        pg_upper_abs    = vars_dict['pg_upper_abs']
        ramp_up_abs     = vars_dict['ramp_up_abs']
        ramp_down_abs   = vars_dict['ramp_down_abs']
        min_on_abs      = vars_dict['min_on_abs']
        min_off_abs     = vars_dict['min_off_abs']
        start_cost_abs  = vars_dict['start_cost_abs']
        shut_cost_abs   = vars_dict['shut_cost_abs']
        surr_abs        = vars_dict['surrogate_abs_vals']
        Ton  = vars_dict['Ton']
        Toff = vars_dict['Toff']
        sensitive_t = vars_dict['sensitive_t']
        lam_inh = self.lambda_inherent[sample_id]
        mu_vals = np.abs(self.mu[sample_id])

        # 2. 重建 obj_opt（动态对偶系数）
        obj_opt = gp.LinExpr()
        for t in range(self.T):
            obj_opt += pg_lower_abs[t] * abs(float(lam_inh['lambda_pg_lower'][t]))
            obj_opt += pg_upper_abs[t] * abs(float(lam_inh['lambda_pg_upper'][t]))
            obj_opt += x[t]        * abs(float(lam_inh['lambda_x_lower'][t]))
            obj_opt += (1 - x[t]) * abs(float(lam_inh['lambda_x_upper'][t]))
        for t in range(1, self.T):
            obj_opt += ramp_up_abs[t-1]   * abs(float(lam_inh['lambda_ramp_up'][t-1]))
            obj_opt += ramp_down_abs[t-1] * abs(float(lam_inh['lambda_ramp_down'][t-1]))
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                obj_opt += min_on_abs[tau-1,t1] * abs(float(lam_inh['lambda_min_on'][tau-1][t1]))
        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                obj_opt += min_off_abs[tau-1,t1] * abs(float(lam_inh['lambda_min_off'][tau-1][t1]))
        for t in range(1, self.T):
            obj_opt += start_cost_abs[t-1] * abs(float(lam_inh['lambda_start_cost'][t-1]))
            obj_opt += shut_cost_abs[t-1]  * abs(float(lam_inh['lambda_shut_cost'][t-1]))
            obj_opt += coc[t-1]            * abs(float(lam_inh['lambda_coc_nonneg'][t-1]))
        # surrogate coupling obj_opt
        for k in range(len(sensitive_t)):
            obj_opt += surr_abs[k] * float(mu_vals[k])

        # 3. Prox
        obj_prox = self._build_primal_block_prox_obj(model, sample_id, pg, x, coc)

        vars_dict['obj_opt'] = obj_opt
        vars_dict['obj_prox'] = obj_prox

        model.setObjective(
            self.rho_primal  * vars_dict['obj_primal']
            + self.rho_opt   * obj_opt
            + self.rho_binary* vars_dict['obj_binary']
            + self.pg_block_prox_weight * obj_prox,
            GRB.MINIMIZE,
        )

    # ------------------------------------------------------------------

    def _emit_subproblem_block_log(self, sample_id: int, line: str) -> None:
        """Gurobi 块日志：默认直接打印；样本级线程并行时先入队，由主线程按 sample_id 排序后输出。"""
        if getattr(self, '_defer_subproblem_block_log', False):
            lock = getattr(self, '_pending_block_logs_lock', None)
            bucket = getattr(self, '_pending_block_logs', None)
            if lock is not None and bucket is not None:
                with lock:
                    bucket.append((int(sample_id), line))
                return
        print(line, flush=True)

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
        BCD迭代：原始块 - V3三时段耦合约束版本（persistent model）
        固定代理约束参数(alphas, betas, gammas, deltas)和对偶变量(mu)，更新原始变量(pg, x)
        """
        g = self.unit_id
        alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
            alphas, betas, gammas, deltas)

        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            assert_lp_backend_available(self._lp_backend)
            return solve_primal_block_backend(
                self, sample_id, alphas, betas, gammas, deltas,
                costs=costs, pg_costs=pg_costs)

        # 决定是否需要重建（仅首次，或敏感时段集合变化时）
        n_coupling = len(self.sensitive_timesteps[sample_id])
        need_rebuild = (
            sample_id not in self._primal_models
            or self._primal_model_n_coupling.get(sample_id) != n_coupling
        )
        if need_rebuild:
            if sample_id in self._primal_models:
                try:
                    self._primal_models[sample_id].dispose()
                except Exception:
                    pass
            model, vars_dict = self._build_primal_model(sample_id, alphas, betas, gammas, deltas)
            self._primal_models[sample_id]         = model
            self._primal_vars[sample_id]           = vars_dict
            self._primal_model_n_coupling[sample_id] = n_coupling
        else:
            model     = self._primal_models[sample_id]
            vars_dict = self._primal_vars[sample_id]

        self._update_primal_model_objective(sample_id, model, vars_dict,
                                            alphas, betas, gammas, deltas)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            pg  = vars_dict['pg']
            x   = vars_dict['x']
            coc = vars_dict['coc']
            cp  = vars_dict['cpower']
            if sample_id <= 2:
                obj_o = vars_dict.get('obj_opt')
                obj_px = vars_dict.get('obj_prox')
                obj_o_s = f"{obj_o.getValue():.4f}" if obj_o is not None else "n/a"
                obj_px_s = f"{obj_px.getValue():.4f}" if obj_px is not None else "n/a"
                self._emit_subproblem_block_log(
                    sample_id,
                    f"[Unit-{self.unit_id}] primal_block, sample_id: {sample_id}, "
                    f"obj_primal: {vars_dict['obj_primal'].getValue():.4f}, "
                    f"obj_opt: {obj_o_s}, "
                    f"obj_binary: {vars_dict['obj_binary'].getValue():.4f}, "
                    f"obj_prox: {obj_px_s}",
                )
            pg_sol     = np.array([pg[t].X  for t in range(self.T)])
            x_sol      = np.array([x[t].X   for t in range(self.T)])
            coc_sol    = np.array([coc[t].X for t in range(self.T-1)])
            cpower_sol = np.array([cp[t].X  for t in range(self.T)])
            return pg_sol, x_sol, coc_sol, cpower_sol
        else:
            print(
                f"[Unit-{self.unit_id}] 警告: 原始块求解失败，状态: {model.status}",
                flush=True,
            )
            return None, None, None, None
    
    # ------------------------------------------------------------------
    # Persistent dual-subproblem model helpers
    # ------------------------------------------------------------------

    def _build_dual_sub_model(self, sample_id: int,
                               alphas, betas, gammas, deltas,
                               costs, pg_costs,
                               lb, sign_relax_round, x_bound_dual_ub, phase):
        """一次性建立 dual-sub 块 Gurobi 模型。
        x 驻点约束中 mu 的 alpha/beta/gamma 系数以初始值建立，后续通过 chgCoeff 更新。
        """
        g = self.unit_id
        a    = self.gencost[g, -2] / self.T_delta
        b    = self.subproblem_generation_no_load_coeff(g)
        Pmin = self.gen[g, PMIN]; Pmax = self.gen[g, PMAX]
        Ru   = float(self.Ru_all[g]); Rd   = float(self.Rd_all[g])
        Ru_co= float(self.Ru_co_all[g]); Rd_co= float(self.Rd_co_all[g])
        start_cost = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shut_cost  = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton  = int(self.subproblem_Ton)
        Toff = int(self.subproblem_Toff)
        lambda_val = self.lambda_vals[sample_id]

        model = gp.Model('dual_sub_block')
        model.Params.OutputFlag = 0
        model.Params.NumericFocus = 2
        model.Params.MIPGap = 1e-6

        lam_pg_lower   = model.addVars(self.T,    lb=0, name='lam_pgl')
        lam_pg_upper   = model.addVars(self.T,    lb=0, name='lam_pgu')
        lam_ramp_up    = model.addVars(self.T-1,  lb=0, name='lam_ru')
        lam_ramp_down  = model.addVars(self.T-1,  lb=0, name='lam_rd')
        lam_start_cost = model.addVars(self.T-1,  lb=0, name='lam_sc')
        lam_shut_cost  = model.addVars(self.T-1,  lb=0, name='lam_shc')
        lam_coc_nonneg = model.addVars(self.T-1,  lb=0, name='lam_coc')
        lam_x_upper    = model.addVars(self.T,    lb=0, ub=x_bound_dual_ub, name='lam_xu')
        lam_x_lower    = model.addVars(self.T,    lb=0, ub=x_bound_dual_ub, name='lam_xl')
        lam_min_on     = {}
        lam_min_off    = {}
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                lam_min_on[tau-1, t1]  = model.addVar(lb=0, name=f'lam_mon_{tau-1}_{t1}')
        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                lam_min_off[tau-1, t1] = model.addVar(lb=0, name=f'lam_moff_{tau-1}_{t1}')

        mu_lb_val = -GRB.INFINITY if (phase != "none" and sign_relax_round) else 0.0
        mu     = model.addVars(self.num_coupling_constraints, lb=mu_lb_val, name='mu')
        mu_abs = model.addVars(self.num_coupling_constraints, lb=0,         name='mu_abs')
        for k in range(self.num_coupling_constraints):
            model.addConstr(mu_abs[k] >= mu[k],  name=f'mu_abs_pos_{k}')
            model.addConstr(mu_abs[k] >= -mu[k], name=f'mu_abs_neg_{k}')
            if phase == "individual" and lb > 0:
                if sign_relax_round:
                    model.addConstr(mu_abs[k] >= lb, name=f'mu_abs_lb_{k}')
                else:
                    model.addConstr(mu[k] >= lb, name=f'mu_lb_{k}')
        if phase == "group" and lb > 0 and self._uses_group_mu_lower_bound():
            for gi in range(self.num_coupling_constraints // self.all_mode_group_size):
                gstart = gi * self.all_mode_group_size
                gstop  = gstart + self.all_mode_group_size
                model.addConstr(
                    gp.quicksum((mu_abs[k] if sign_relax_round else mu[k])
                                for k in range(gstart, gstop)) >= lb,
                    name=f'mu_group_lb_{gi}',
                )

        obj_dual_pg  = gp.LinExpr()
        obj_dual_x   = gp.LinExpr()
        obj_dual_coc = gp.LinExpr()

        # pg 驻点（完全静态系数）
        dual_pg_abs_constrs = []  # list of (c_pos, c_neg) per time step t
        for t in range(self.T):
            expr = a + (float(pg_costs[t]) if pg_costs is not None else 0.0) - float(lambda_val[t])
            expr -= lam_pg_lower[t]
            expr += lam_pg_upper[t]
            if t > 0:
                expr += lam_ramp_up[t-1]
                expr -= lam_ramp_down[t-1]
            if t < self.T - 1:
                expr -= lam_ramp_up[t]
                expr += lam_ramp_down[t]
            abs_v = model.addVar(lb=0, name=f'abs_pg_{t}')
            c_pg_pos = model.addConstr(abs_v >= expr,  name=f'abs_pg_pos_{t}')
            c_pg_neg = model.addConstr(abs_v >= -expr, name=f'abs_pg_neg_{t}')
            dual_pg_abs_constrs.append((c_pg_pos, c_pg_neg))
            obj_dual_pg += abs_v

        # x 驻点（mu 的 alpha/beta/gamma 系数以初始值建立）
        sensitive_t        = self.sensitive_timesteps[sample_id]
        constraint_offsets = self._constraint_offsets_for_sample(sample_id)
        dual_x_mu_terms    = {}  # t -> [(c_pos, c_neg, mu[k], k, coeff_key)]
        dual_x_abs_constrs = []  # list of (c_pos, c_neg) per time step t, for RHS updates
        for t in range(self.T):
            expr = b + (float(costs[t]) if costs is not None else 0.0)
            expr += Pmin * lam_pg_lower[t]
            expr -= Pmax * lam_pg_upper[t]
            if t < self.T - 1:
                expr += (Ru_co - Ru) * lam_ramp_up[t]
            if t > 0:
                expr += (Rd_co - Rd) * lam_ramp_down[t-1]
            for tau in range(1, Ton+1):
                for t1 in range(self.T - tau):
                    k = lam_min_on[tau-1, t1]
                    if t == t1+1:   expr += k
                    if t == t1:     expr -= k
                    if t == t1+tau: expr -= k
            for tau in range(1, Toff+1):
                for t1 in range(self.T - tau):
                    k = lam_min_off[tau-1, t1]
                    if t == t1+1:   expr -= k
                    if t == t1:     expr += k
                    if t == t1+tau: expr += k
            if t > 0:
                expr += start_cost * lam_start_cost[t-1]
                expr -= shut_cost  * lam_shut_cost[t-1]
            if t < self.T - 1:
                expr -= start_cost * lam_start_cost[t]
                expr += shut_cost  * lam_shut_cost[t]

            # 代理约束 mu 贡献（以初始值建立，后续 chgCoeff 更新）
            mu_terms_t = []
            for k_idx, ts in enumerate(sensitive_t):
                for time_idx, coeff in iterate_surrogate_constraint_terms(
                        ts, constraint_offsets[k_idx],
                        float(alphas[k_idx]), float(betas[k_idx]), float(gammas[k_idx]), self.T):
                    if time_idx == t:
                        expr += coeff * mu[k_idx]

            expr += lam_x_upper[t] - lam_x_lower[t]

            abs_v = model.addVar(lb=0, name=f'abs_x_{t}')
            c_pos = model.addConstr(abs_v >= expr,  name=f'abs_x_pos_{t}')
            c_neg = model.addConstr(abs_v >= -expr, name=f'abs_x_neg_{t}')
            dual_x_abs_constrs.append((c_pos, c_neg))
            obj_dual_x += abs_v

            # 记录各 mu 项（用于后续 chgCoeff）
            for k_idx, ts in enumerate(sensitive_t):
                for time_idx, coeff in iterate_surrogate_constraint_terms(
                        ts, constraint_offsets[k_idx],
                        float(alphas[k_idx]), float(betas[k_idx]), float(gammas[k_idx]), self.T):
                    if time_idx == t:
                        mu_terms_t.append((c_pos, c_neg, mu[k_idx], k_idx,
                                           ts, constraint_offsets[k_idx]))
            if mu_terms_t:
                dual_x_mu_terms[t] = mu_terms_t

        # coc 驻点（完全静态）
        for t in range(self.T-1):
            expr = 1 - lam_start_cost[t] - lam_shut_cost[t] - lam_coc_nonneg[t]
            abs_v = model.addVar(lb=0, name=f'abs_coc_{t}')
            model.addConstr(abs_v >= expr,  name=f'abs_coc_pos_{t}')
            model.addConstr(abs_v >= -expr, name=f'abs_coc_neg_{t}')
            obj_dual_coc += abs_v

        model.update()

        vars_dict = {
            'lam_pg_lower': lam_pg_lower, 'lam_pg_upper': lam_pg_upper,
            'lam_ramp_up': lam_ramp_up,   'lam_ramp_down': lam_ramp_down,
            'lam_start_cost': lam_start_cost, 'lam_shut_cost': lam_shut_cost,
            'lam_coc_nonneg': lam_coc_nonneg,
            'lam_x_upper': lam_x_upper, 'lam_x_lower': lam_x_lower,
            'lam_min_on': lam_min_on, 'lam_min_off': lam_min_off,
            'mu': mu, 'mu_abs': mu_abs,
            'obj_dual_pg': obj_dual_pg, 'obj_dual_x': obj_dual_x, 'obj_dual_coc': obj_dual_coc,
            'Ton': Ton, 'Toff': Toff,
            'dual_x_mu_terms': dual_x_mu_terms,
            'sensitive_t': sensitive_t,
            'constraint_offsets': constraint_offsets,
            # RHS update support: constraint refs + last-used constant terms
            'dual_x_abs_constrs': dual_x_abs_constrs,
            'last_costs': np.array(costs, dtype=float) if costs is not None else None,
            'dual_pg_abs_constrs': dual_pg_abs_constrs,
            'last_pg_costs': np.array(pg_costs, dtype=float) if pg_costs is not None else None,
        }
        return model, vars_dict

    def _apply_dual_sub_mu_chgcoeff(self, model, vars_dict, alphas, betas, gammas):
        """通过 chgCoeff 更新 x 驻点约束中 mu 的 alpha/beta/gamma 系数。"""
        sensitive_t        = vars_dict['sensitive_t']
        constraint_offsets = vars_dict['constraint_offsets']
        mu                 = vars_dict['mu']
        dual_x_mu_terms    = vars_dict['dual_x_mu_terms']

        for t, mu_terms_t in dual_x_mu_terms.items():
            for c_pos, c_neg, mu_var, k_idx, ts, off in mu_terms_t:
                # Recompute the coefficient for this (t, k) using new alphas/betas/gammas
                new_coeff = 0.0
                for time_idx, coeff in iterate_surrogate_constraint_terms(
                        ts, off,
                        float(alphas[k_idx]), float(betas[k_idx]), float(gammas[k_idx]), self.T):
                    if time_idx == t:
                        new_coeff = coeff
                        break
                # abs_v >= expr + new_coeff*mu  →  abs_v - expr - new_coeff*mu >= 0
                model.chgCoeff(c_pos, mu_var, -new_coeff)
                model.chgCoeff(c_neg, mu_var,  new_coeff)

    def _update_dual_sub_model_objective(self, sample_id: int, model, vars_dict,
                                          alphas, betas, gammas, deltas,
                                          costs=None, pg_costs=None):
        """每次迭代：更新 mu 驻点系数，重建 obj_opt + prox，调用 setObjective。"""
        # 1. 更新 x 驻点约束中 mu 的系数
        self._apply_dual_sub_mu_chgcoeff(model, vars_dict, alphas, betas, gammas)

        # 2. 更新 x 驻点约束的常数项 RHS（c_x 随 NN 更新而变化）
        #    约束形式: abs_v >= b + c_x[t] + lambda_terms  →  c_pos.RHS = b + c_x[t]
        #              abs_v >= -(b + c_x[t] + lambda_terms) →  c_neg.RHS = -(b + c_x[t])
        if costs is not None and 'dual_x_abs_constrs' in vars_dict:
            old_costs = vars_dict.get('last_costs')
            if old_costs is not None:
                dual_x_abs_constrs = vars_dict['dual_x_abs_constrs']
                for t in range(self.T):
                    delta = float(costs[t]) - float(old_costs[t])
                    if abs(delta) > 1e-12:
                        c_pos, c_neg = dual_x_abs_constrs[t]
                        c_pos.RHS += delta
                        c_neg.RHS -= delta
            vars_dict['last_costs'] = np.array(costs, dtype=float)

        # 3. 更新 pg 驻点约束的常数项 RHS（c_pg 随 NN 更新而变化）
        #    约束形式: abs_v >= a + c_pg[t] - lambda[t] + lambda_terms
        #              →  c_pg_pos.RHS = a + c_pg[t] - lambda[t]
        if pg_costs is not None and 'dual_pg_abs_constrs' in vars_dict:
            old_pg_costs = vars_dict.get('last_pg_costs')
            if old_pg_costs is not None:
                dual_pg_abs_constrs = vars_dict['dual_pg_abs_constrs']
                for t in range(self.T):
                    delta = float(pg_costs[t]) - float(old_pg_costs[t])
                    if abs(delta) > 1e-12:
                        c_pg_pos, c_pg_neg = dual_pg_abs_constrs[t]
                        c_pg_pos.RHS += delta
                        c_pg_neg.RHS -= delta
            vars_dict['last_pg_costs'] = np.array(pg_costs, dtype=float)

        g = self.unit_id
        pg_val  = self.pg[sample_id]
        x_val   = self.x[sample_id]
        coc_val = self.coc[sample_id]
        Pmin = self.gen[g, PMIN]; Pmax = self.gen[g, PMAX]
        Ru   = float(self.Ru_all[g]); Rd   = float(self.Rd_all[g])
        Ru_co= float(self.Ru_co_all[g]); Rd_co= float(self.Rd_co_all[g])
        start_cost = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shut_cost  = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton  = vars_dict['Ton']; Toff = vars_dict['Toff']
        sensitive_t        = vars_dict['sensitive_t']
        constraint_offsets = vars_dict['constraint_offsets']
        lam_pgl   = vars_dict['lam_pg_lower']; lam_pgu = vars_dict['lam_pg_upper']
        lam_ru    = vars_dict['lam_ramp_up'];  lam_rd  = vars_dict['lam_ramp_down']
        lam_sc    = vars_dict['lam_start_cost']; lam_shc = vars_dict['lam_shut_cost']
        lam_coc   = vars_dict['lam_coc_nonneg']
        lam_xu    = vars_dict['lam_x_upper'];  lam_xl  = vars_dict['lam_x_lower']
        lam_mon   = vars_dict['lam_min_on'];   lam_moff= vars_dict['lam_min_off']
        mu        = vars_dict['mu']; mu_abs = vars_dict['mu_abs']

        # 4. 重建 obj_opt（动态标量系数）
        obj_opt = gp.LinExpr()
        for t in range(self.T):
            pgl_v = abs(pg_val[t] - Pmin * x_val[t])
            if pgl_v > 1e-10: obj_opt += pgl_v * lam_pgl[t]
            pgu_v = abs(Pmax * x_val[t] - pg_val[t])
            if pgu_v > 1e-10: obj_opt += pgu_v * lam_pgu[t]
        for t in range(1, self.T):
            limit_u = Ru*x_val[t-1] + Ru_co*(1-x_val[t-1])
            rv_u = abs(pg_val[t] - pg_val[t-1] - limit_u)
            if rv_u > 1e-10: obj_opt += rv_u * lam_ru[t-1]
            limit_d = Rd*x_val[t] + Rd_co*(1-x_val[t])
            rv_d = abs(pg_val[t-1] - pg_val[t] - limit_d)
            if rv_d > 1e-10: obj_opt += rv_d * lam_rd[t-1]
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                v = abs(x_val[t1+1] - x_val[t1] - x_val[t1+tau])
                if v > 1e-10: obj_opt += v * lam_mon[tau-1, t1]
        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                v = abs(-x_val[t1+1] + x_val[t1] - 1 + x_val[t1+tau])
                if v > 1e-10: obj_opt += v * lam_moff[tau-1, t1]
        for t in range(self.T-1):
            sc_v  = abs(coc_val[t] - start_cost*(x_val[t+1] - x_val[t]))
            shc_v = abs(coc_val[t] - shut_cost*(x_val[t] - x_val[t+1]))
            coc_v = abs(coc_val[t])
            if sc_v  > 1e-10: obj_opt += sc_v  * lam_sc[t]
            if shc_v > 1e-10: obj_opt += shc_v * lam_shc[t]
            if coc_v > 1e-10: obj_opt += coc_v * lam_coc[t]
        for t in range(self.T):
            xl_v = abs(x_val[t])
            xu_v = abs(x_val[t] - 1)
            if xl_v > 1e-10: obj_opt += xl_v * lam_xl[t]
            if xu_v > 1e-10: obj_opt += xu_v * lam_xu[t]
        for k_idx, ts in enumerate(sensitive_t):
            lhs = build_surrogate_constraint_expression(
                x_val, ts, constraint_offsets[k_idx],
                float(alphas[k_idx]), float(betas[k_idx]), float(gammas[k_idx]), self.T)
            viol = abs(lhs - float(deltas[k_idx]))
            if viol > 1e-10:
                obj_opt += viol * mu_abs[k_idx]

        # 5. Prox
        obj_dual_prox = self._build_dual_block_prox_obj(
            model, sample_id,
            lam_pgl, lam_pgu, lam_ru, lam_rd, lam_sc, lam_shc, lam_coc,
            lam_xu, lam_xl, lam_mon, lam_moff, mu, Ton, Toff,
        )
        obj_single_mu_cap = self._build_single_mu_cap_penalty_obj(model, mu_abs, vars_dict)

        vars_dict['obj_opt'] = obj_opt
        vars_dict['obj_dual_prox'] = obj_dual_prox
        vars_dict['obj_single_mu_cap'] = obj_single_mu_cap

        model.setObjective(
            self.rho_dual_pg  * vars_dict['obj_dual_pg']
            + self.rho_dual_x  * vars_dict['obj_dual_x']
            + self.rho_dual_coc* vars_dict['obj_dual_coc']
            + self.rho_opt     * obj_opt
            + self.dual_block_prox_weight * obj_dual_prox
            + obj_single_mu_cap,
            GRB.MINIMIZE,
        )

    # ------------------------------------------------------------------

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
        BCD迭代：对偶块 - V3三时段耦合约束完整版本（persistent model）
        固定原始变量(pg, x, coc)和代理约束参数，联合更新所有对偶变量。
        """
        g = self.unit_id

        phase            = self._get_mu_lower_bound_phase()
        lb               = self._current_mu_lower_bound_value()
        sign_relax_round = self._is_mu_sign_relaxation_round()
        x_bound_dual_ub  = 0.0 if self._force_zero_x_bound_duals() else GRB.INFINITY
        alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
            alphas, betas, gammas, deltas)

        if self._lp_backend == LP_BACKEND_CVXPY_HIGHS:
            assert_lp_backend_available(self._lp_backend)
            return solve_dual_block_backend(
                self, sample_id, alphas, betas, gammas, deltas,
                costs=costs, pg_costs=pg_costs)

        current_state = (lb, sign_relax_round, x_bound_dual_ub,
                         len(self.sensitive_timesteps[sample_id]))
        need_rebuild = (
            sample_id not in self._dual_sub_models
            or self._dual_sub_model_state.get(sample_id) != current_state
        )
        if need_rebuild:
            if sample_id in self._dual_sub_models:
                try:
                    self._dual_sub_models[sample_id].dispose()
                except Exception:
                    pass
            model, vars_dict = self._build_dual_sub_model(
                sample_id, alphas, betas, gammas, deltas,
                costs, pg_costs, lb, sign_relax_round, x_bound_dual_ub, phase)
            self._dual_sub_models[sample_id]      = model
            self._dual_sub_vars[sample_id]        = vars_dict
            self._dual_sub_model_state[sample_id] = current_state
        else:
            model     = self._dual_sub_models[sample_id]
            vars_dict = self._dual_sub_vars[sample_id]

        self._update_dual_sub_model_objective(
            sample_id, model, vars_dict, alphas, betas, gammas, deltas,
            costs=costs, pg_costs=pg_costs)
        model.optimize()

        Ton  = vars_dict['Ton']
        Toff = vars_dict['Toff']
        lam_pgl  = vars_dict['lam_pg_lower']; lam_pgu = vars_dict['lam_pg_upper']
        lam_ru   = vars_dict['lam_ramp_up'];  lam_rd  = vars_dict['lam_ramp_down']
        lam_sc   = vars_dict['lam_start_cost']; lam_shc = vars_dict['lam_shut_cost']
        lam_coc  = vars_dict['lam_coc_nonneg']
        lam_xu   = vars_dict['lam_x_upper'];  lam_xl  = vars_dict['lam_x_lower']
        lam_mon  = vars_dict['lam_min_on'];   lam_moff= vars_dict['lam_min_off']
        mu       = vars_dict['mu']

        if model.status == GRB.OPTIMAL:
            lambda_inherent_sol = {
                'lambda_pg_lower':   np.array([lam_pgl[t].X  for t in range(self.T)]),
                'lambda_pg_upper':   np.array([lam_pgu[t].X  for t in range(self.T)]),
                'lambda_ramp_up':    np.array([lam_ru[t].X   for t in range(self.T-1)]),
                'lambda_ramp_down':  np.array([lam_rd[t].X   for t in range(self.T-1)]),
                'lambda_min_on':     np.array([[lam_mon[tau-1, t1].X
                                                for t1 in range(self.T - tau)]
                                               for tau in range(1, Ton+1)], dtype=object),
                'lambda_min_off':    np.array([[lam_moff[tau-1, t1].X
                                                for t1 in range(self.T - tau)]
                                               for tau in range(1, Toff+1)], dtype=object),
                'lambda_start_cost': np.array([lam_sc[t].X   for t in range(self.T-1)]),
                'lambda_shut_cost':  np.array([lam_shc[t].X  for t in range(self.T-1)]),
                'lambda_coc_nonneg': np.array([lam_coc[t].X  for t in range(self.T-1)]),
                'lambda_x_upper':    np.array([lam_xu[t].X   for t in range(self.T)]),
                'lambda_x_lower':    np.array([lam_xl[t].X   for t in range(self.T)]),
            }
            mu_sol = np.array([mu[k].X for k in range(self.num_coupling_constraints)])

            # 与 primal_block / cvxpy solve_dual_block 一致：每个机组仅打印前 3 个样本
            if sample_id <= 2:
                obj_dp = float(vars_dict['obj_dual_pg'].getValue())
                obj_dx = float(vars_dict['obj_dual_x'].getValue())
                obj_dc = float(vars_dict['obj_dual_coc'].getValue())
                obj_d = obj_dp + obj_dx + obj_dc
                obj_o = float(vars_dict['obj_opt'].getValue())
                obj_px = float(vars_dict['obj_dual_prox'].getValue())
                _single_cap, single_cap_weight = self._current_single_mu_cap()
                self._emit_subproblem_block_log(
                    sample_id,
                    f"[Unit-{self.unit_id}] dual_block, sample_id: {sample_id}, "
                    f"status: optimal, "
                    f"obj_dual_pg: {obj_dp:.6f}, "
                    f"obj_dual_x: {obj_dx:.6f}, "
                    f"obj_dual_coc: {obj_dc:.6f}, "
                    f"obj_dual: {obj_d:.6f}, "
                    f"obj_opt: {obj_o:.6f}, "
                    f"obj_dual_prox: {obj_px:.6f}, "
                    f"single_mu_cap_weight: {single_cap_weight:.6f}",
                )
            return lambda_inherent_sol, mu_sol
        else:
            print(
                f"[Unit-{self.unit_id}] 警告: 对偶块求解失败 sample={sample_id}，状态: {model.status}",
                flush=True,
            )
            return None, None

    def _iter_with_dual_block_original(self, sample_id: int,
                                        alphas, betas, gammas, deltas,
                                        costs=None, pg_costs=None):
        """保留原始实现以备参考（不再被主流程调用）。"""
        g = self.unit_id
        pg_val  = self.pg[sample_id]
        x_val   = self.x[sample_id]
        coc_val = self.coc[sample_id]
        lambda_val = self.lambda_vals[sample_id]
        a    = self.gencost[g, -2] / self.T_delta
        b    = self.subproblem_generation_no_load_coeff(g)
        Pmin = self.gen[g, PMIN]; Pmax = self.gen[g, PMAX]
        Ru   = float(self.Ru_all[g]); Rd   = float(self.Rd_all[g])
        Ru_co= float(self.Ru_co_all[g]); Rd_co= float(self.Rd_co_all[g])
        start_cost = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
        shut_cost  = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
        Ton  = int(self.subproblem_Ton)
        Toff = int(self.subproblem_Toff)
        phase            = self._get_mu_lower_bound_phase()
        lb               = self._current_mu_lower_bound_value()
        sign_relax_round = self._is_mu_sign_relaxation_round()
        alphas, betas, gammas, deltas = self._apply_surrogate_direction_to_params(
            alphas, betas, gammas, deltas)
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
        obj_single_mu_cap = self._build_single_mu_cap_penalty_obj(model, mu_abs)
        model.setObjective(
            self.rho_dual_pg * obj_dual_pg
            + self.rho_dual_x * obj_dual_x
            + self.rho_dual_coc * obj_dual_coc
            + self.rho_opt * obj_opt
            + self.dual_block_prox_weight * obj_dual_prox
            + obj_single_mu_cap,
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

            # 与 primal_block / cvxpy solve_dual_block 一致：每个机组仅打印前 3 个样本
            if sample_id <= 2:
                _single_cap, single_cap_weight = self._current_single_mu_cap()
                print(
                    f"[Unit-{self.unit_id}] dual_block, sample_id: {sample_id}, "
                    f"status: optimal, "
                    f"obj_dual_pg: {obj_dual_pg.getValue():.6f}, "
                    f"obj_dual_x: {obj_dual_x.getValue():.6f}, "
                    f"obj_dual_coc: {obj_dual_coc.getValue():.6f}, "
                    f"obj_dual: {obj_dual.getValue():.6f}, "
                    f"obj_opt: {(obj_opt.getValue() if hasattr(obj_opt, 'getValue') else obj_opt):.6f}, "
                    f"obj_dual_prox: {(obj_dual_prox.getValue() if hasattr(obj_dual_prox, 'getValue') else 0.0):.6f}, "
                    f"single_mu_cap_weight: {single_cap_weight:.6f}",
                    flush=True,
                )

            return lambda_inherent_sol, mu_sol
        else:
            print(
                f"[Unit-{self.unit_id}] 警告: 对偶块求解失败 sample={sample_id}，状态: {model.status}",
                flush=True,
            )
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
            self.subproblem_generation_no_load_coeff(g) / (Pmax + 1e-8),   # 归一化无负荷成本 b
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
        cache, metadata = self._loss_cached_sample(sample_id, device=device)
        n_constraints = int(metadata['n_constraints'])
        mu_vals = cache['mu'][sample_id][:n_constraints]
        mu_abs_vals = cache['mu_abs'][sample_id][:n_constraints]
        direction_signs = cache['direction_signs'][:len(alphas_tensor)]
        signed_alphas = alphas_tensor * direction_signs
        signed_betas = betas_tensor * direction_signs
        signed_gammas = gammas_tensor * direction_signs
        signed_deltas = deltas_tensor * direction_signs
        
        # ========== 计算obj_primal ==========
        # V3三时段约束违反量（按 sensitive_timesteps 索引）
        x_alpha = metadata['x_alpha']
        x_beta = metadata['x_beta']
        x_gamma = metadata['x_gamma']
        if n_constraints > 0:
            coupling_lhs = (
                signed_alphas[:n_constraints] * x_alpha
                + signed_betas[:n_constraints] * x_beta
                + signed_gammas[:n_constraints] * x_gamma
            )
            coupling_residual = coupling_lhs - signed_deltas[:n_constraints]
            obj_primal = torch.sum(self._smooth_relu(coupling_residual))
            obj_opt = torch.sum(
                self._smooth_abs(coupling_residual, self.nn_smooth_abs_eps)
                * mu_abs_vals
            )
        else:
            obj_primal = torch.zeros((), dtype=alphas_tensor.dtype, device=device)
            obj_opt = torch.zeros((), dtype=alphas_tensor.dtype, device=device)

        # ========== 计算obj_opt ==========
        # V3互补松弛（按 sensitive_timesteps 索引）
        
        # ========== 计算obj_dual_x ==========
        # x[t]驻点：b + Pmin*lam_pg_lower[t] - Pmax*lam_pg_upper[t]
        #           + ramp_co_terms + min_on/off_terms + start/shut_terms
        #           + coupling_terms(alpha,beta,gamma,mu) + lam_x_upper[t] - lam_x_lower[t] = 0
        #
        # 固有项（常数，来自dual block存储的lambda_inherent）
        # 代理耦合项（含alpha,beta,gamma张量，提供NN梯度）
        inherent_const = cache['x_stationarity_const'][sample_id]
        dual_coupling = torch.zeros(self.T, dtype=costs_tensor.dtype, device=device)
        term_times = metadata['term_times']
        if term_times.numel() > 0:
            term_k = metadata['term_k']
            term_kind = metadata['term_kind']
            alpha_terms = signed_alphas[term_k]
            beta_terms = signed_betas[term_k]
            gamma_terms = signed_gammas[term_k]
            coeff_terms = torch.where(
                term_kind == 0,
                alpha_terms,
                torch.where(term_kind == 1, beta_terms, gamma_terms),
            )
            dual_coupling = dual_coupling.scatter_add(
                0,
                term_times,
                coeff_terms * mu_vals[term_k],
            )
        dual_expr = inherent_const + costs_tensor[:self.T] + dual_coupling
        obj_dual_x = torch.sum(self._smooth_abs(dual_expr, self.nn_smooth_abs_eps))

        # 死区正则：限制幅值失控，但不给模板附近/小范围波动施加默认回拉。
        reg_loss = self.reg_weight * (
            self._deadband_quadratic(alphas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(betas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(gammas_tensor, self.coeff_reg_deadband)
            + self._deadband_quadratic(costs_tensor, self.aux_cost_reg_deadband)
        )
        if self._uses_template_rhs_bases():
            delta_base_tensor = cache['template_rhs_base'][:deltas_tensor.shape[0]].to(dtype=deltas_tensor.dtype)
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
        prev_cache = cache.get('prev', {})
        if self.iter_delta_reg_weight > 0 and 'alpha' in prev_cache:
            nc = int(self.num_coupling_constraints)
            prev_alphas = prev_cache['alpha'][sample_id][:nc].to(dtype=alphas_tensor.dtype)
            prev_betas = prev_cache['beta'][sample_id][:nc].to(dtype=betas_tensor.dtype)
            prev_gammas = prev_cache['gamma'][sample_id][:nc].to(dtype=gammas_tensor.dtype)
            prev_deltas = prev_cache['delta'][sample_id][:nc].to(dtype=deltas_tensor.dtype)
            prev_costs = prev_cache['cost'][sample_id][: self.T].to(dtype=costs_tensor.dtype)
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
        cache, _ = self._loss_cached_sample(sample_id, device=device)
        if cache['lambda_inherent'][sample_id] is None:
            obj_dual_pg = torch.zeros((), dtype=pg_costs_tensor.dtype, device=device)
        else:
            residual = cache['pg_stationarity_const'][sample_id] + pg_costs_tensor[:self.T]
            obj_dual_pg = torch.sum(self._smooth_abs(residual, self.pg_cost_smooth_abs_eps))

        reg_loss = self.reg_weight * self._deadband_quadratic(
            pg_costs_tensor,
            self.pg_cost_reg_deadband,
        )
        prev_cache = cache.get('prev', {})
        if self.iter_delta_reg_weight > 0 and 'pg_cost' in prev_cache:
            prev_pg_costs = prev_cache['pg_cost'][sample_id][: self.T].to(dtype=pg_costs_tensor.dtype)
            reg_loss = reg_loss + self.iter_delta_reg_weight * self._iter_delta_regularization(
                pg_costs_tensor,
                prev_pg_costs,
                self.iter_delta_reg_deadband,
            )
        if self.pg_cost_softbound_weight > 0 and self.pg_cost_scale > 0:
            s = torch.as_tensor(
                float(self.pg_cost_scale),
                dtype=pg_costs_tensor.dtype,
                device=pg_costs_tensor.device,
            )
            excess = self._smooth_relu(
                torch.abs(pg_costs_tensor) - s,
                eps=self.pg_cost_smooth_abs_eps,
            )
            reg_loss = reg_loss + self.pg_cost_softbound_weight * torch.sum(excess * excess)
        reg_loss = reg_loss * self._c_pg_reg_loss_scale
        dual_pg_term = self.loss_ratio_dual_pg * obj_dual_pg
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
        BCD 迭代：主代理网络的可微 KKT 微调（NN-main）。

        与 ``iter_with_main_direct_targets`` 分工：direct 阶段用大步长拟合解析/锚定目标；
        本阶段用小步长在当前 BCD 状态下细化驻点残差。调度上对齐 ``direct-NN-main`` 的
        ``CosineAnnealingLR``（每轮外循环内余弦降到 ``eta_min``），并随外循环推进减小
        基准学习率（参见 ``_nn_main_bcd_lr_scale``，类比 ``subproblem_lp_solver`` 中
        penalty 权重随迭代累积后的局部精炼）。
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

        bcd_lr_scale = self._nn_main_bcd_lr_scale()
        kkt_lr_scale = float(self.nn_main_kkt_lr_scale)
        lr0 = float(resolved_learning_rate) * bcd_lr_scale * kkt_lr_scale
        cost_lr0 = float(resolved_cost_learning_rate) * bcd_lr_scale * kkt_lr_scale
        eta_min_ratio = max(float(self.nn_main_eta_min_ratio), 0.0)
        nn_wd = float(self.nn_main_adam_weight_decay)
        grad_clip_nn = float(self.nn_main_grad_clip)
        T_ep = max(int(num_epochs), 1)

        # 前 N 轮 BCD 重建优化器（适应剧烈变化），之后保持动量；是否重建仅看名义超参是否变化，
        # 不用当前 param_group['lr']（scheduler 会把它降到 eta_min，否则会每轮误判重建）。
        optimizer_persist_after = 5
        rebuild = (self.iter_number < optimizer_persist_after)
        stored_nominal_main = getattr(self, '_surr_nn_nominal_main_lr', None)
        stored_nominal_cost = getattr(self, '_surr_nn_nominal_cost_lr', None)
        stored_wd = getattr(self, '_surr_nn_adam_wd', None)
        if (
            rebuild
            or not hasattr(self, '_surr_optimizer')
            or self._surr_optimizer is None
            or not hasattr(self, '_surr_cost_optimizer')
            or self._surr_cost_optimizer is None
            or stored_nominal_main != resolved_learning_rate
            or stored_nominal_cost != resolved_cost_learning_rate
            or stored_wd != nn_wd
        ):
            main_params = self._main_network_parameters()
            self._surr_optimizer = optim.Adam(
                main_params, lr=lr0, weight_decay=nn_wd,
            )
            self._surr_cost_optimizer = optim.Adam(
                self._cost_network_parameters(),
                lr=cost_lr0,
                weight_decay=nn_wd,
            )
            self._surr_nn_nominal_main_lr = resolved_learning_rate
            self._surr_nn_nominal_cost_lr = resolved_cost_learning_rate
            self._surr_nn_adam_wd = nn_wd
        else:
            for pg in self._surr_optimizer.param_groups:
                pg['lr'] = lr0
            for pg in self._surr_cost_optimizer.param_groups:
                pg['lr'] = cost_lr0

        # 每轮 BCD 内独立余弦周期（无 WarmRestarts 跨外循环拉回高峰学习率）
        self._surr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self._surr_optimizer, T_max=T_ep, eta_min=lr0 * eta_min_ratio,
        )
        self._surr_cost_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self._surr_cost_optimizer, T_max=T_ep, eta_min=cost_lr0 * eta_min_ratio,
        )

        # 单机组预测器微调优化器（仅在 _unit_predictor_active 时生效）
        predictor_override_active = self._unit_predictor_active()
        predictor_train_active = self._unit_predictor_finetune_active()
        if predictor_train_active:
            self._ensure_unit_predictor_optimizer()
            self.unit_predictor.get_network(self.unit_id).train()

        self.surrogate_net.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_p = epoch_dx = epoch_o = epoch_r = 0.0
            self._surr_optimizer.zero_grad()
            self._surr_cost_optimizer.zero_grad()
            if predictor_train_active and self._unit_predictor_optimizer is not None:
                self._unit_predictor_optimizer.zero_grad()
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

                # 可选：用 0/1 预测器覆盖 single-time 段的 (alpha, beta, gamma, delta)
                if predictor_override_active:
                    alphas_tensor, betas_tensor, gammas_tensor, deltas_tensor = (
                        self._apply_unit_predictor_override(
                            alphas_tensor,
                            betas_tensor,
                            gammas_tensor,
                            deltas_tensor,
                            features_tensor,
                            train_predictor=predictor_train_active,
                        )
                    )

                # 主 loss 不再包含 c_pg 对应项（标量已为 ρ 加权后的 KKT 目标，与 direct MSE 不可比）
                loss, _lc = self.loss_function_differentiable(
                    sample_id, alphas_tensor, betas_tensor, gammas_tensor, deltas_tensor,
                    costs_tensor, self.device, return_components=True,
                )
                (loss / actual_batch_size).backward()
                epoch_loss += float(loss.detach().cpu().item())
                epoch_p += float(_lc["primal_term"].detach().cpu().item())
                epoch_dx += float(_lc["dual_x_term"].detach().cpu().item())
                epoch_o += float(_lc["opt_term"].detach().cpu().item())
                epoch_r += float(_lc["reg_term"].detach().cpu().item())
                batch_count += 1

                # batch 满或 epoch 结束：clip + step
                if batch_count == resolved_batch_size or sample_pos == self.n_samples - 1:
                    torch.nn.utils.clip_grad_norm_(
                        self.surrogate_net.parameters(), max_norm=grad_clip_nn,
                    )
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._unit_predictor_parameters(), max_norm=1.0,
                        )
                    self._surr_optimizer.step()
                    self._surr_cost_optimizer.step()
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        self._unit_predictor_optimizer.step()
                    self._surr_optimizer.zero_grad()
                    self._surr_cost_optimizer.zero_grad()
                    if predictor_train_active and self._unit_predictor_optimizer is not None:
                        self._unit_predictor_optimizer.zero_grad()
                    batch_count = 0

            self._surr_scheduler.step()
            self._surr_cost_scheduler.step()

            if epoch == 0 or epoch == num_epochs - 1:
                inv_n = 1.0 / max(self.n_samples, 1)
                print(
                    f"  [Unit-{self.unit_id}][NN-main] epoch {epoch+1}/{num_epochs}, "
                    f"avg_kkt_loss={epoch_loss * inv_n:.6f} "
                    f"(ρ加权: primal={epoch_p * inv_n:.4g}, dual_x={epoch_dx * inv_n:.4g}, "
                    f"opt={epoch_o * inv_n:.4g}, reg={epoch_r * inv_n:.4g})",
                    flush=True,
                )

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
        log_interval: int | None = None,
        log_metrics: bool = False,
    ):
        """
        BCD迭代：c_pg 单独训练器。
        仅更新 pg_cost_net，并使用独立 loss。

        Args:
            log_interval: 保留兼容；**不再影响** stdout，仅第 1 轮与最后一轮打印。
            log_metrics: 为 True 时在每次打印时额外 ``cal_nn_logging_components`` 输出
            ``obj_dual_pg`` / ``reg_pg``（略增开销）。
        """
        if not TORCH_AVAILABLE or not self._pg_costs_active():
            self._last_pg_cost_nn_loss = None
            return

        resolved_epochs = 10 if num_epochs is None else int(num_epochs)
        if resolved_epochs <= 0:
            self._last_pg_cost_nn_loss = None
            return
        num_epochs = resolved_epochs

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
                self._pg_network_parameters(),
                lr=resolved_learning_rate,
                weight_decay=self.pg_cost_c_pg_adam_weight_decay,
            )
            # 单样本时余弦退火会无谓扰动学习率，固定 lr 更利于把驻点残差压到 0
            if self.n_samples > 1:
                self._surr_pg_cost_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self._surr_pg_cost_optimizer, T_0=max(num_epochs, 1), T_mult=1,
                )
            else:
                self._surr_pg_cost_scheduler = None

        self.surrogate_net.train()
        self._set_c_pg_training_mode(True)
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                self._surr_pg_cost_optimizer.zero_grad()
                sample_indices = np.arange(self.n_samples, dtype=int)
                if resolved_shuffle and self.n_samples > 1:
                    np.random.shuffle(sample_indices)

                for batch_start in range(0, self.n_samples, resolved_batch_size):
                    batch_ids = sample_indices[batch_start: batch_start + resolved_batch_size]
                    batch_losses = []

                    for sample_id in batch_ids:
                        features = self._extract_features(int(sample_id))
                        features_tensor = torch.tensor(
                            features,
                            dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0)

                        pg_costs_out = self.surrogate_net.forward_pg_cost(features_tensor)
                        pg_costs_tensor = self._gate_pg_cost_tensor(
                            pg_costs_out.squeeze(0)[:self.T]
                        )

                        batch_losses.append(
                            self.loss_function_c_pg_differentiable(
                                int(sample_id),
                                pg_costs_tensor,
                                self.device,
                            )
                        )
                    # 难样本加权：按当前样本 loss 大小提升其梯度权重（mean≈1，clip 防止爆炸）
                    # Relative hard-sample weighting keeps mean batch scale stable.
                    if self.pg_cost_use_sample_weights and self.pg_cost_sample_weight_power > 0:
                        with torch.no_grad():
                            detached = torch.stack([
                                torch.clamp(loss.detach(), min=0.0)
                                for loss in batch_losses
                            ])
                            mean_loss = torch.clamp(torch.mean(detached), min=1e-12)
                            weights = torch.pow(
                                detached / mean_loss,
                                float(self.pg_cost_sample_weight_power),
                            )
                            weights = torch.clamp(
                                weights,
                                min=1e-6,
                                max=float(self.pg_cost_sample_weight_clip),
                            )
                        batch_loss = torch.stack([
                            loss * weights[idx]
                            for idx, loss in enumerate(batch_losses)
                        ]).sum()
                    else:
                        batch_loss = torch.stack(batch_losses).sum()

                    raw_batch_loss = sum(float(loss.detach().cpu().item()) for loss in batch_losses)
                    (batch_loss / max(1, len(batch_ids))).backward()
                    epoch_loss += raw_batch_loss

                    torch.nn.utils.clip_grad_norm_(
                        self._pg_network_parameters(),
                        max_norm=1.0,
                    )
                    self._surr_pg_cost_optimizer.step()
                    self._surr_pg_cost_optimizer.zero_grad()

                if self._surr_pg_cost_scheduler is not None:
                    self._surr_pg_cost_scheduler.step()

                avg = epoch_loss / max(self.n_samples, 1)
                lr_cur = float(self._surr_pg_cost_optimizer.param_groups[0].get("lr", 0.0))
                do_log = epoch == 0 or epoch == num_epochs - 1
                if do_log:
                    line = (
                        f"  [Unit-{self.unit_id}][NN-c_pg] epoch {epoch+1:>4}/{num_epochs}, "
                        f"avg_loss={avg:.6f}, sum_loss={epoch_loss:.6f}, lr={lr_cur:.2e}"
                    )
                    if log_metrics:
                        try:
                            # cal_nn_logging_components 用 self.pg_cost_values 等缓存；
                            # 仅 backward 时网络在变，不刷新则指标一直为进循环前的数。
                            self._refresh_cached_surrogate_outputs()
                            m = self.cal_nn_logging_components()
                            line += (
                                f", obj_dual_pg={m['obj_dual_pg']:.6f}, reg_pg={m['reg_pg']:.6f}"
                            )
                        except Exception as exc:
                            line += f", metrics_err={exc!r}"
                    print(line, flush=True)

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
            b_v     = self.subproblem_generation_no_load_coeff(g)   # 无负荷成本系数
            Pmin_v  = float(self.gen[g, PMIN])
            Pmax_v  = float(self.gen[g, PMAX])
            Ru_v    = float(self.Ru_all[g])
            Rd_v    = float(self.Rd_all[g])
            Ru_co_v = float(self.Ru_co_all[g])
            Rd_co_v = float(self.Rd_co_all[g])
            sc_v    = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 1]
            shc_v   = 0.0 if self.ignore_startup_shutdown_costs else self.gencost[g, 2]
            Ton_v   = int(self.subproblem_Ton)
            Toff_v  = int(self.subproblem_Toff)

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
            Ton_v = int(self.subproblem_Ton)
            Toff_v = int(self.subproblem_Toff)

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

    def cal_obj_binary_gap(self) -> float:
        """与 primal block 中 ``obj_binary`` 一致：各样本 ``x`` 相对参考解 ``x_true`` 的 L1 距离之和。"""
        total = 0.0
        for sample_id in range(self.n_samples):
            x_true = self.active_set_data[sample_id].get("x_true")
            if x_true is None:
                continue
            xv = np.asarray(self.x[sample_id], dtype=float)
            xt = np.asarray(x_true, dtype=float)
            total += float(np.sum(np.abs(xv - xt)))
        return total

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
        self._configure_surrogate_bcd_run(max_iter)
        gamma = self.gamma_base / (self.n_samples * max(max_iter, 1))
        gamma_dual = gamma * self.gamma_dual_component_scale
        self.gamma = gamma
        if pg_cost_nn_epochs is not None:
            resolved_pg_cost_nn_epochs = max(int(pg_cost_nn_epochs), 0)
        elif self.pg_cost_nn_epochs is not None:
            resolved_pg_cost_nn_epochs = max(int(self.pg_cost_nn_epochs), 0)
        else:
            resolved_pg_cost_nn_epochs = 10
        log_u = f"[Unit-{self.unit_id}]"

        for i in range(max_iter):
            print(f"{log_u} 🔄 迭代 {i+1}/{max_iter}", flush=True)
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
            self._invalidate_loss_tensor_cache()
            
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
            self._invalidate_loss_tensor_cache()
            self._log_single_mu_cap_status(log_u, raw_mu_solutions[solved_mask], self.mu[solved_mask])
            
            _z = lambda v: v if abs(v) >= 1e-12 else 0.0
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
            # 3. 神经网络更新代理约束参数
            #    先缓存上一代输出，供“迭代间差异正则”使用（默认权重为0时不影响训练）
            self._prev_alpha_values = self.alpha_values.copy()
            self._prev_beta_values = self.beta_values.copy()
            self._prev_gamma_values = self.gamma_values.copy()
            self._prev_delta_values = self.delta_values.copy()
            self._prev_cost_values = self.cost_values.copy()
            self._prev_pg_cost_values = self.pg_cost_values.copy()
            self._invalidate_loss_tensor_cache()
            self.iter_with_main_direct_targets()
            self.iter_with_surrogate_nn(
                num_epochs=nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
                cost_learning_rate=cost_learning_rate,
            )
            self.iter_with_c_pg_direct_targets()
            self.iter_with_c_pg_nn(
                num_epochs=resolved_pg_cost_nn_epochs,
                batch_size=(self.pg_cost_batch_size if self.pg_cost_batch_size is not None else nn_batch_size),
                batch_strategy=(self.pg_cost_batch_strategy if self.pg_cost_batch_strategy is not None else nn_batch_strategy),
                shuffle=(self.pg_cost_shuffle if self.pg_cost_shuffle is not None else nn_shuffle),
                learning_rate=pg_cost_surr_learning_rate,
            )
            self._refresh_cached_surrogate_outputs()

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

            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = (
                _z(obj_primal), _z(obj_dual_pg), _z(obj_dual_x), _z(obj_dual_coc), _z(obj_dual), _z(obj_opt)
            )
            obj_binary = self.cal_obj_binary_gap()
            # 前3次迭代冻结rho，之后再按累加式更新
            if i >= 3:
                self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
                self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
                self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
                self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
                self._sync_rho_dual_summary()
                self.rho_binary = min(self.rho_binary + gamma * obj_binary, self.rho_binary_max)
                self.rho_opt    = min(self.rho_opt    + gamma * obj_opt,    self.rho_max)

            print(
                f"{log_u}   ρ_primal={self.rho_primal:.4f}, ρ_dual_pg={self.rho_dual_pg:.4f}, "
                f"ρ_dual_x={self.rho_dual_x:.4f}, ρ_dual_coc={self.rho_dual_coc:.4f}, "
                f"ρ_dual={self.rho_dual:.4f}, ρ_binary={self.rho_binary:.4f} "
                f"(≤{self.rho_binary_max:.4g}), ρ_opt={self.rho_opt:.4f}",
                flush=True,
            )
            print(f"{log_u} " + "-" * 40, flush=True)

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
            self.subproblem_generation_no_load_coeff(g) / (Pmax + 1e-8),
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

        nc = int(self.num_coupling_constraints)
        with torch.no_grad():
            alphas, betas, gammas, deltas, costs = self.surrogate_net.forward_main(features_tensor)
            pg_costs = self.surrogate_net.forward_pg_cost(features_tensor)
            a_full = alphas.squeeze(0)
            b_full = betas.squeeze(0)
            c_full = gammas.squeeze(0)
            d_proc = self._postprocess_delta_tensor(deltas.squeeze(0))
            a_nc = a_full[:nc]
            b_nc = b_full[:nc]
            c_nc = c_full[:nc]
            d_nc = d_proc[:nc]
            a_nc, b_nc, c_nc, d_nc = self._apply_unit_predictor_override(
                a_nc, b_nc, c_nc, d_nc, features_tensor,
            )
            if a_full.numel() > nc:
                alphas_t = torch.cat([a_nc, a_full[nc:]], dim=0)
                betas_t = torch.cat([b_nc, b_full[nc:]], dim=0)
                gammas_t = torch.cat([c_nc, c_full[nc:]], dim=0)
                deltas_t = torch.cat([d_nc, d_proc[nc:]], dim=0)
            else:
                alphas_t, betas_t, gammas_t, deltas_t = a_nc, b_nc, c_nc, d_nc

        alphas_np = alphas_t.cpu().numpy()
        betas_np = betas_t.cpu().numpy()
        gammas_np = gammas_t.cpu().numpy()
        deltas_np = deltas_t.cpu().numpy()
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
                'surrogate_direction_pending_signs': self.surrogate_direction_pending_signs,
                'surrogate_direction_pending_counts': self.surrogate_direction_pending_counts,
                'rho_primal': self.rho_primal,
                'rho_dual': self.rho_dual,
                'rho_dual_pg': self.rho_dual_pg,
                'rho_dual_x': self.rho_dual_x,
                'rho_dual_coc': self.rho_dual_coc,
                'rho_binary': self.rho_binary,
                'rho_binary_max': self.rho_binary_max,
                'rho_opt': self.rho_opt,
                'gamma_dual_component_scale': self.gamma_dual_component_scale,
                'iter_number': self.iter_number,
                'num_coupling_constraints': self.num_coupling_constraints,
                'max_constraints': self.max_constraints,
                'requested_max_constraints': self.requested_max_constraints,
                'constraint_generation_strategy': self.constraint_generation_strategy,
                'lp_backend': self._lp_backend,
                'ignore_startup_shutdown_costs': self.ignore_startup_shutdown_costs,
                'predictor_warmup_rounds': self.predictor_warmup_rounds,
                'sign4_curriculum_rounds': self.sign4_curriculum_rounds,
                'sign4_initial_scale': self.sign4_initial_scale,
                'sign4_final_scale': self.sign4_final_scale,
                'sign4_delay_rounds': self.sign4_delay_rounds,
                'mu_individual_lower_bound_round': self.mu_individual_lower_bound_round,
                'mu_group_lower_bound_round': self.mu_group_lower_bound_round,
                'mu_signed_round_interval': self.mu_signed_round_interval,
                'mu_sign_hysteresis_rounds': self.mu_sign_hysteresis_rounds,
                'mu_sign_flip_min_share': self.mu_sign_flip_min_share,
                'x_bound_dual_zero_rounds': self.x_bound_dual_zero_rounds,
                'pg_cost_start_round': self.pg_cost_start_round,
                'pg_cost_scale_multiplier': self.pg_cost_scale_multiplier,
                'nn_learning_rate': self.nn_learning_rate,
                'nn_main_eta_min_ratio': self.nn_main_eta_min_ratio,
                'nn_main_lr_late_scale': self.nn_main_lr_late_scale,
                'nn_main_adam_weight_decay': self.nn_main_adam_weight_decay,
                'nn_main_grad_clip': self.nn_main_grad_clip,
                'nn_main_kkt_lr_scale': self.nn_main_kkt_lr_scale,
                'nn_hidden_dims': self.nn_hidden_dims,
                'pg_cost_hidden_dims': self.pg_cost_hidden_dims,
                'cost_learning_rate': self.cost_learning_rate,
                'pg_cost_lr': self.pg_cost_lr,
                'pg_cost_surr_lr': self.pg_cost_surr_lr,
                'pg_cost_nn_epochs': self.pg_cost_nn_epochs,
                'nn_batch_strategy': self.nn_batch_strategy,
                'nn_batch_size': self.nn_batch_size,
                'nn_shuffle': self.nn_shuffle,
                'pg_cost_reg_deadband': self.pg_cost_reg_deadband,
                'pg_cost_softbound_weight': self.pg_cost_softbound_weight,
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
                'single_mu_cap_penalty_weight': self.single_mu_cap_penalty_weight,
                'single_mu_cap_initial_weight': self.single_mu_cap_initial_weight,
                'single_mu_cap_final_weight': self.single_mu_cap_final_weight,
                'single_mu_cap_initial': self.single_mu_cap_initial,
                'single_mu_cap_final': self.single_mu_cap_final,
                'single_mu_cap_start_round': self.single_mu_cap_start_round,
                'single_mu_cap_end_round': self.single_mu_cap_end_round,
                'template_rhs_jitter_scale': self.template_rhs_jitter_scale,
                'template_rhs_reg_deadband': self.template_rhs_reg_deadband,
                'coeff_reg_deadband': self.coeff_reg_deadband,
                'aux_cost_reg_deadband': self.aux_cost_reg_deadband,
                'lambda_inherent': self.lambda_inherent,
                'scenario_feature_dim': int(getattr(self, '_scenario_feature_dim', 0) or 0),
            }
            
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            
            torch.save(state, filepath)
            print(f"✓ V3三时段耦合代理约束模型已保存: {filepath}", flush=True)

    def collect_c_pg_loss_snapshot(self) -> dict:
        """收集当前 trainer 状态下 c_pg 可微损失相关的原始量（便于落盘与离线复现/针对性测试）。

        在多样本训练任意 BCD 轮次、刚更新完对偶/缓存后调用，可得到与
        ``loss_function_c_pg_differentiable`` 一致分解的 per-sample 数据
        （含 ``pg_const``、当前 ``forward_pg_cost`` 与 loss 各项）。

        Returns:
            可 JSON 序列化的 dict（``schema_version``=1），键 ``samples`` 为每样本一条记录。
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("collect_c_pg_loss_snapshot requires PyTorch")
        g = self.unit_id
        a_val = float(self.gencost[g, -2] / self.T_delta)
        prev_list = None
        if self._prev_pg_cost_values is not None:
            prev_list = np.asarray(self._prev_pg_cost_values, dtype=float).tolist()

        out: dict = {
            "schema_version": 1,
            "unit_id": int(g),
            "T": int(self.T),
            "n_samples": int(self.n_samples),
            "iter_number": int(self.iter_number),
            "a_linear_per_step": a_val,
            "pg_cost_scale": float(self.pg_cost_scale),
            "rho_dual_pg": float(self.rho_dual_pg),
            "loss_ratio_dual_pg": float(self.loss_ratio_dual_pg),
            "loss_ratio_reg": float(self.loss_ratio_reg),
            "_c_pg_reg_loss_scale": float(self._c_pg_reg_loss_scale),
            "reg_weight": float(self.reg_weight),
            "pg_cost_reg_deadband": float(self.pg_cost_reg_deadband),
            "iter_delta_reg_weight": float(self.iter_delta_reg_weight),
            "iter_delta_reg_deadband": float(self.iter_delta_reg_deadband),
            "pg_cost_softbound_weight": float(self.pg_cost_softbound_weight),
            "pg_cost_smooth_abs_eps": float(self.pg_cost_smooth_abs_eps),
            "prev_pg_cost_values": prev_list,
            "samples": [],
        }
        was_training = self.surrogate_net.training
        self.surrogate_net.eval()
        try:
            for sample_id in range(self.n_samples):
                features = np.asarray(self._extract_features(sample_id), dtype=np.float64).reshape(-1)
                li = self.lambda_inherent[sample_id]
                li_cpg = None
                pg_const_t = self._pg_stationarity_const_np(sample_id).astype(float).tolist()
                if li is not None:
                    li_cpg = {
                        "lambda_pg_lower": np.asarray(
                            li["lambda_pg_lower"], dtype=float
                        ).reshape(-1).tolist(),
                        "lambda_pg_upper": np.asarray(
                            li["lambda_pg_upper"], dtype=float
                        ).reshape(-1).tolist(),
                        "lambda_ramp_up": np.asarray(
                            li["lambda_ramp_up"], dtype=float
                        ).reshape(-1).tolist(),
                        "lambda_ramp_down": np.asarray(
                            li["lambda_ramp_down"], dtype=float
                        ).reshape(-1).tolist(),
                    }
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    pg_out = self.surrogate_net.forward_pg_cost(features_tensor)
                pg_tensor = self._gate_pg_cost_tensor(
                    pg_out.squeeze(0)[: self.T]
                )
                loss, comp = self.loss_function_c_pg_differentiable(
                    sample_id, pg_tensor, self.device, return_components=True
                )
                out["samples"].append(
                    {
                        "sample_id": int(sample_id),
                        "features": features.astype(float).tolist(),
                        "lambda_vals": lam.astype(float).tolist(),
                        "lambda_inherent_is_none": li is None,
                        "lambda_inherent_c_pg": li_cpg,
                        "pg_const_per_t": pg_const_t,
                        "implied_c_pg_zero_residual": [
                            -float(x) for x in pg_const_t
                        ] if pg_const_t else [],
                        "pg_cost_cached": np.asarray(
                            self.pg_cost_values[sample_id], dtype=float
                        ).tolist(),
                        "pg_cost_forward": pg_tensor.detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                        .tolist(),
                        "loss_total": float(loss.detach().cpu().item()),
                        "obj_dual_pg": float(comp["obj_dual_pg"].cpu().item()),
                        "dual_pg_term": float(comp["dual_pg_term"].cpu().item()),
                        "reg_term": float(comp["reg_term"].cpu().item()),
                    }
                )
        finally:
            if was_training:
                self.surrogate_net.train()
        return out

    def load(self, filepath: str):
        """加载V3模型"""
        if TORCH_AVAILABLE:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
            rebuild_network = False
            saved_hidden_dims = state.get('nn_hidden_dims')
            if saved_hidden_dims is not None:
                resolved_hidden_dims = normalize_nn_hidden_dims(saved_hidden_dims, self.nn_hidden_dims)
                if resolved_hidden_dims != self.nn_hidden_dims:
                    self.nn_hidden_dims = resolved_hidden_dims
                    rebuild_network = True
            saved_pg_cost_hidden_dims = state.get('pg_cost_hidden_dims')
            if saved_pg_cost_hidden_dims is not None:
                resolved_pg_cost_hidden_dims = normalize_nn_hidden_dims(
                    saved_pg_cost_hidden_dims,
                    self.pg_cost_hidden_dims,
                )
                if resolved_pg_cost_hidden_dims != self.pg_cost_hidden_dims:
                    self.pg_cost_hidden_dims = resolved_pg_cost_hidden_dims
                    rebuild_network = True
            if rebuild_network:
                self._init_neural_network()
            surrogate_state_dict = dict(state['surrogate_net_state_dict'])
            if 'pg_input_proj.weight' not in surrogate_state_dict and 'input_proj.weight' in surrogate_state_dict:
                surrogate_state_dict['pg_input_proj.weight'] = surrogate_state_dict['input_proj.weight'].clone()
                surrogate_state_dict['pg_input_proj.bias'] = surrogate_state_dict['input_proj.bias'].clone()
                for block_idx in range(len(self.surrogate_net.pg_res_blocks)):
                    for suffix in ('fc1.weight', 'fc1.bias', 'ln1.weight', 'ln1.bias', 'fc2.weight', 'fc2.bias', 'ln2.weight', 'ln2.bias'):
                        main_key = f'res_blocks.{block_idx}.{suffix}'
                        pg_key = f'pg_res_blocks.{block_idx}.{suffix}'
                        if main_key in surrogate_state_dict and pg_key not in surrogate_state_dict:
                            surrogate_state_dict[pg_key] = surrogate_state_dict[main_key].clone()
            self.surrogate_net.load_state_dict(surrogate_state_dict, strict=False)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alpha_values = state['alpha_values']
            self.beta_values = state['beta_values']
            self.gamma_values = state['gamma_values']
            self.delta_values = state.get('delta_values', np.full_like(self.gamma_values, 3.0))
            self.cost_values = state.get('cost_values', np.zeros((self.n_samples, self.T)))
            self.pg_cost_values = state.get('pg_cost_values', np.zeros((self.n_samples, self.T)))
            self.x_cost_scale = state.get('x_cost_scale', self.x_cost_scale)
            self.pg_cost_scale = state.get('pg_cost_scale', self.pg_cost_scale)
            _sfd = state.get('scenario_feature_dim')
            if _sfd is not None and int(_sfd) > 0:
                self._scenario_feature_dim = int(_sfd)
            self.mu = state['mu']
            loaded_direction_signs = state.get('surrogate_direction_signs')
            if loaded_direction_signs is None:
                self.surrogate_direction_signs = np.ones(self.num_coupling_constraints, dtype=float)
            else:
                self.surrogate_direction_signs = np.asarray(loaded_direction_signs, dtype=float).reshape(-1)
            loaded_pending_signs = state.get('surrogate_direction_pending_signs')
            if loaded_pending_signs is None:
                self.surrogate_direction_pending_signs = np.zeros(self.num_coupling_constraints, dtype=float)
            else:
                self.surrogate_direction_pending_signs = np.asarray(loaded_pending_signs, dtype=float).reshape(-1)
            loaded_pending_counts = state.get('surrogate_direction_pending_counts')
            if loaded_pending_counts is None:
                self.surrogate_direction_pending_counts = np.zeros(self.num_coupling_constraints, dtype=int)
            else:
                self.surrogate_direction_pending_counts = np.asarray(loaded_pending_counts, dtype=int).reshape(-1)
            self.rho_primal = state['rho_primal']
            self.rho_dual_pg = state.get('rho_dual_pg', state.get('rho_dual', self.rho_dual_pg))
            self.rho_dual_x = state.get('rho_dual_x', state.get('rho_dual', self.rho_dual_x))
            self.rho_dual_coc = state.get('rho_dual_coc', state.get('rho_dual', self.rho_dual_coc))
            self._sync_rho_dual_summary()
            self.rho_binary = state.get('rho_binary', self.rho_binary)
            if 'rho_binary_max' in state:
                self.rho_binary_max = max(float(state['rho_binary_max']), self.rho_binary)
            else:
                self.rho_binary_max = max(self.rho_binary_max, self.rho_binary)
            self.rho_binary = min(self.rho_binary, self.rho_binary_max)
            self.rho_opt = state['rho_opt']
            self.gamma_dual_component_scale = state.get(
                'gamma_dual_component_scale',
                self.gamma_dual_component_scale,
            )
            saved_strategy = state.get('constraint_generation_strategy', 'sensitive')
            self.constraint_generation_strategy = normalize_constraint_generation_strategy(saved_strategy)
            self._lp_backend = normalize_lp_backend(state.get('lp_backend', self._lp_backend))
            self.ignore_startup_shutdown_costs = bool(
                state.get('ignore_startup_shutdown_costs', self.ignore_startup_shutdown_costs)
            )
            self.predictor_warmup_rounds = max(
                int(state.get('predictor_warmup_rounds', self.predictor_warmup_rounds)),
                0,
            )
            self.sign4_curriculum_rounds = max(
                int(state.get('sign4_curriculum_rounds', self.sign4_curriculum_rounds)),
                0,
            )
            self.sign4_initial_scale = max(
                float(state.get('sign4_initial_scale', self.sign4_initial_scale)),
                0.0,
            )
            self.sign4_final_scale = max(
                float(state.get('sign4_final_scale', self.sign4_final_scale)),
                0.0,
            )
            self.sign4_delay_rounds = max(
                int(state.get('sign4_delay_rounds', getattr(self, 'sign4_delay_rounds', 0))),
                0,
            )
            self.iter_number = int(state.get(
                'iter_number',
                max(
                    int(getattr(self, 'iter_number', 0)),
                    self.predictor_warmup_rounds,
                    self.sign4_curriculum_rounds,
                ),
            ))
            self.all_mode_group_size = (
                4
                if self.constraint_generation_strategy in (
                    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
                    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
                )
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
            self.mu_sign_hysteresis_rounds = max(
                int(state.get('mu_sign_hysteresis_rounds', self.mu_sign_hysteresis_rounds)),
                1,
            )
            self.mu_sign_flip_min_share = min(
                max(float(state.get('mu_sign_flip_min_share', self.mu_sign_flip_min_share)), 0.5),
                1.0,
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
            self.nn_main_eta_min_ratio = max(
                float(state.get('nn_main_eta_min_ratio', self.nn_main_eta_min_ratio)),
                0.0,
            )
            self.nn_main_lr_late_scale = min(
                max(float(state.get('nn_main_lr_late_scale', self.nn_main_lr_late_scale)), 1e-3),
                1.0,
            )
            self.nn_main_adam_weight_decay = max(
                float(state.get('nn_main_adam_weight_decay', self.nn_main_adam_weight_decay)),
                0.0,
            )
            self.nn_main_grad_clip = max(
                float(state.get('nn_main_grad_clip', self.nn_main_grad_clip)),
                1e-6,
            )
            self.nn_main_kkt_lr_scale = max(
                float(state.get('nn_main_kkt_lr_scale', self.nn_main_kkt_lr_scale)),
                1e-6,
            )
            self.nn_hidden_dims = normalize_nn_hidden_dims(
                state.get('nn_hidden_dims'),
                self.nn_hidden_dims,
            )
            self.pg_cost_hidden_dims = normalize_nn_hidden_dims(
                state.get('pg_cost_hidden_dims'),
                self.pg_cost_hidden_dims,
            )
            self.cost_learning_rate = state.get('cost_learning_rate', self.cost_learning_rate)
            self.pg_cost_lr = state.get('pg_cost_lr', self.pg_cost_lr)
            self.pg_cost_surr_lr = state.get('pg_cost_surr_lr', self.pg_cost_surr_lr)
            loaded_pg_cost_nn_epochs = state.get('pg_cost_nn_epochs', self.pg_cost_nn_epochs)
            self.pg_cost_nn_epochs = (
                None if loaded_pg_cost_nn_epochs is None else max(int(loaded_pg_cost_nn_epochs), 0)
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
            self.single_mu_cap_penalty_weight = max(
                float(state.get(
                    'single_mu_cap_penalty_weight',
                    getattr(self, 'single_mu_cap_penalty_weight', 0.0),
                )),
                0.0,
            )
            self.single_mu_cap_initial_weight = max(
                float(state.get(
                    'single_mu_cap_initial_weight',
                    getattr(self, 'single_mu_cap_initial_weight', 0.0),
                )),
                0.0,
            )
            self.single_mu_cap_final_weight = max(
                float(state.get(
                    'single_mu_cap_final_weight',
                    self.single_mu_cap_penalty_weight,
                )),
                0.0,
            )
            self.single_mu_cap_penalty_weight = self.single_mu_cap_final_weight
            loaded_single_mu_cap_initial = state.get(
                'single_mu_cap_initial',
                getattr(self, 'single_mu_cap_initial', None),
            )
            loaded_single_mu_cap_final = state.get(
                'single_mu_cap_final',
                getattr(self, 'single_mu_cap_final', None),
            )
            self.single_mu_cap_initial = (
                None if loaded_single_mu_cap_initial is None
                else max(float(loaded_single_mu_cap_initial), 0.0)
            )
            self.single_mu_cap_final = (
                None if loaded_single_mu_cap_final is None
                else max(float(loaded_single_mu_cap_final), 0.0)
            )
            self.single_mu_cap_start_round = max(
                int(state.get(
                    'single_mu_cap_start_round',
                    getattr(self, 'single_mu_cap_start_round', 0),
                )),
                0,
            )
            self.single_mu_cap_end_round = max(
                int(state.get(
                    'single_mu_cap_end_round',
                    getattr(self, 'single_mu_cap_end_round', self.single_mu_cap_start_round),
                )),
                self.single_mu_cap_start_round,
            )
            self.pg_cost_reg_deadband = state.get(
                'pg_cost_reg_deadband',
                self.pg_cost_reg_deadband,
            )
            self.pg_cost_softbound_weight = max(
                float(state.get('pg_cost_softbound_weight', self.pg_cost_softbound_weight)),
                0.0,
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
            self._ensure_surrogate_direction_hysteresis_buffers()
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
        'lp_backend': state.get('lp_backend', LP_BACKEND_GUROBI),
        'ignore_startup_shutdown_costs': bool(state.get('ignore_startup_shutdown_costs', False)),
        'max_constraints': state.get('max_constraints'),
        'requested_max_constraints': state.get('requested_max_constraints'),
        'num_coupling_constraints': state.get('num_coupling_constraints'),
        'mu_individual_lower_bound_round': state.get('mu_individual_lower_bound_round'),
        'mu_group_lower_bound_round': state.get('mu_group_lower_bound_round'),
        'mu_signed_round_interval': state.get('mu_signed_round_interval'),
        'mu_sign_hysteresis_rounds': state.get('mu_sign_hysteresis_rounds'),
        'mu_sign_flip_min_share': state.get('mu_sign_flip_min_share'),
        'predictor_warmup_rounds': state.get('predictor_warmup_rounds'),
        'sign4_curriculum_rounds': state.get('sign4_curriculum_rounds'),
        'sign4_initial_scale': state.get('sign4_initial_scale'),
        'sign4_final_scale': state.get('sign4_final_scale'),
        'sign4_delay_rounds': state.get('sign4_delay_rounds'),
        'single_mu_cap_penalty_weight': state.get('single_mu_cap_penalty_weight'),
        'single_mu_cap_initial_weight': state.get('single_mu_cap_initial_weight'),
        'single_mu_cap_final_weight': state.get('single_mu_cap_final_weight'),
        'single_mu_cap_initial': state.get('single_mu_cap_initial'),
        'single_mu_cap_final': state.get('single_mu_cap_final'),
        'single_mu_cap_start_round': state.get('single_mu_cap_start_round'),
        'single_mu_cap_end_round': state.get('single_mu_cap_end_round'),
        'surrogate_direction_signs': state.get('surrogate_direction_signs'),
        'nn_hidden_dims': state.get('nn_hidden_dims'),
        'pg_cost_hidden_dims': state.get('pg_cost_hidden_dims', state.get('nn_hidden_dims')),
    }


def build_lambda_pg_electricity_price_targets(
    ppc,
    active_set_data: List[Dict],
    T_delta: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the same label matrix Y as DualVariablePredictorTrainer._solve_for_true_dual_variables:
    each row is ``lambda_pg_electricity_price`` for one sample, shape (N, ng*T).

    Returns:
        Y: float array (n_samples, ng * T)
        meta: counts and dimensions (n_from_cache, n_from_ed, ng, T, ...)
    """
    ppc_int = ext2int(ppc)
    ng = int(ppc_int["gen"].shape[0])
    nl = int(ppc_int["branch"].shape[0])
    n_samples = len(active_set_data)
    T = int(active_set_data[0]["pd_data"].shape[1])
    sens = _build_generator_injection_sensitivity(ppc)

    n_from_cache = 0
    n_from_ed = 0
    rows: List[np.ndarray] = []

    for sample_id in range(n_samples):
        sample = active_set_data[sample_id]
        effective = _get_effective_pg_prices_from_sample_or_dual_payload(
            sample, T, ng, nl, sens,
        )
        if effective is None:
            n_from_ed += 1
            x_sol = _recover_unit_commitment_matrix(sample, ng, T)
            payload = _solve_pg_electricity_price_from_ed(
                ppc,
                sample["pd_data"],
                T_delta,
                x_sol,
                renewable_data=sample.get("renewable_data"),
                verbose=False,
            )
            effective = np.asarray(payload["lambda_pg_electricity_price"], dtype=float)
            sample["lambda_pg_electricity_price"] = effective.copy()
        else:
            n_from_cache += 1
            effective = np.asarray(effective, dtype=float)

        rows.append(effective.reshape(-1))

    Y = np.asarray(rows, dtype=float)
    meta = {
        "n_samples": n_samples,
        "ng": ng,
        "T": T,
        "output_dim": ng * T,
        "n_from_cache": n_from_cache,
        "n_from_ed": n_from_ed,
    }
    return Y, meta


def train_dual_predictor_from_data(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                                    num_epochs: int = 100, batch_size: int = 8,
                                    batch_strategy: str = "full-batch",
                                    shuffle: bool = True,
                                    learning_rate: float = 1e-3,
                                    save_path: str = None, device=None,
                                    dual_net_variant: str = "temporal_conv",
                                    dual_normalize_targets: bool = True,
                                    dual_cosine_loss_weight: float = 0.12,
                                    dual_smooth_l1_beta: float = 2.0) -> DualVariablePredictorTrainer:
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
        dual_net_variant=dual_net_variant,
        dual_normalize_targets=dual_normalize_targets,
        dual_cosine_loss_weight=dual_cosine_loss_weight,
        dual_smooth_l1_beta=dual_smooth_l1_beta,
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


def train_unit_predictor_from_data(
    ppc,
    active_set_data: List[Dict],
    T_delta: float = 1.0,
    unit_ids: List[int] | None = None,
    num_epochs: int = 200,
    batch_size: int = 32,
    batch_strategy: str = 'full-batch',
    shuffle: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: List[int] | None = None,
    net_variant: str = "mlp",
    resmlp_width: int = 512,
    resmlp_depth: int = 4,
    tconv_channels: int = 64,
    tconv_depth: int = 4,
    tcn_channels: int = 64,
    tcn_depth: int = 6,
    dropout: float = 0.1,
    save_path: str | None = None,
    load_path: str | None = None,
    enable_scheduler: bool = True,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 0.0,
    enable_pos_weight: bool = False,
    pos_weight_clip: float = 20.0,
    loss_weight_bce: float = 1.0,
    loss_weight_mse: float = 0.0,
    loss_weight_l1: float = 0.0,
    loss_weight_tv: float = 0.0,
    loss_weight_transition: float = 0.0,
    loss_weight_binarize: float = 0.0,
    loss_weight_std_floor: float = 0.0,
    std_floor_scale: float = 0.5,
    loss_weight_tv_floor: float = 0.0,
    tv_floor_scale: float = 0.8,
    device=None,
) -> SingleUnitBinaryPredictorTrainer:
    """构造并预训练每机组独立的 0/1 变量预测器。

    - 首次训练：BCEWithLogitsLoss 拟合 ``unit_commitment_matrix`` 标签。
    - 继续训练：若 ``load_path`` 指向已有 checkpoint，则先 load 再 finetune。
    - ``save_path`` 非空则保存到该路径（供进程池 worker 加载）。
    """
    print("=" * 60, flush=True)
    print("训练单机组 0/1 变量预测器", flush=True)
    print("=" * 60, flush=True)

    predictor = SingleUnitBinaryPredictorTrainer(
        ppc,
        active_set_data,
        T_delta,
        unit_ids=unit_ids,
        hidden_dims=hidden_dims,
        net_variant=net_variant,
        resmlp_width=resmlp_width,
        resmlp_depth=resmlp_depth,
        tconv_channels=tconv_channels,
        tconv_depth=tconv_depth,
        tcn_channels=tcn_channels,
        tcn_depth=tcn_depth,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_strategy=batch_strategy,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
    )

    loaded_from_checkpoint = False
    if load_path is not None and os.path.exists(load_path):
        loaded_from_checkpoint = bool(predictor.load(load_path))
        if not loaded_from_checkpoint and int(num_epochs) <= 0:
            raise RuntimeError(
                f"unit_predictor checkpoint was requested but no compatible weights were loaded: {load_path}"
            )
        if not loaded_from_checkpoint:
            print(
                f"! [UnitPredictor] load failed or incompatible; training from scratch for {int(num_epochs)} epochs",
                flush=True,
            )

    if int(num_epochs) > 0:
        predictor.train_all(
            num_epochs=num_epochs,
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            shuffle=shuffle,
            learning_rate=learning_rate,
            enable_scheduler=enable_scheduler,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            scheduler_min_lr=scheduler_min_lr,
            enable_pos_weight=enable_pos_weight,
            pos_weight_clip=pos_weight_clip,
            loss_weight_bce=loss_weight_bce,
            loss_weight_mse=loss_weight_mse,
            loss_weight_l1=loss_weight_l1,
            loss_weight_tv=loss_weight_tv,
            loss_weight_transition=loss_weight_transition,
            loss_weight_binarize=loss_weight_binarize,
            loss_weight_std_floor=loss_weight_std_floor,
            std_floor_scale=std_floor_scale,
            loss_weight_tv_floor=loss_weight_tv_floor,
            tv_floor_scale=tv_floor_scale,
        )
    elif loaded_from_checkpoint:
        print("[UnitPredictor] checkpoint loaded; skip extra pretraining (epochs=0)", flush=True)

    if save_path:
        predictor.save(save_path)

    return predictor


def train_subproblem_surrogate_from_data(ppc, active_set_data: List[Dict], unit_id: int,
                                          T_delta: float = 1.0, lambda_predictor=None,
                                          max_iter: int = 20, nn_epochs: int = 10,
                                          lp_backend: str = LP_BACKEND_GUROBI,
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
                                          mu_sign_hysteresis_rounds: int = 2,
                                          mu_sign_flip_min_share: float = 0.67,
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
        lp_backend=lp_backend,
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
        mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
        mu_sign_flip_min_share=mu_sign_flip_min_share,
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
                                      lp_backend: str = LP_BACKEND_GUROBI,
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
                                      mu_sign_hysteresis_rounds: int = 2,
                                      mu_sign_flip_min_share: float = 0.67,
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
            lp_backend=lp_backend,
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
            mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
            mu_sign_flip_min_share=mu_sign_flip_min_share,
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
                         lp_backend: str | None = None,
                         constraint_generation_strategy: str | None = None,
                         ignore_startup_shutdown_costs: bool | None = None,
                         case_name: str | None = None,
                         unit_predictor_path: str | None = None,
                         use_unit_predictor: bool = False,
                         unit_predictor_hidden_dims: List[int] | None = None,
                         unit_predictor_net_variant: str = "mlp",
                         unit_predictor_resmlp_width: int = 512,
                         unit_predictor_resmlp_depth: int = 4,
                         unit_predictor_tcn_channels: int = 64,
                         unit_predictor_tcn_depth: int = 6,
                         unit_predictor_tconv_channels: int = 64,
                         unit_predictor_tconv_depth: int = 4,
                         unit_predictor_dropout: float = 0.1,
                         unit_predictor_finetune_lr: float = 1e-5,
                         unit_predictor_weight_decay: float = 1e-4,
                         skip_initial_solve: bool = True,
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
    
    # 加载对偶预测器（预读 checkpoint 以对齐网络结构与标准化参数）
    dual_path = os.path.join(load_dir, 'dual_predictor.pth')
    peek: dict = {}
    if os.path.exists(dual_path) and TORCH_AVAILABLE:
        try:
            try:
                peek = torch.load(dual_path, map_location='cpu', weights_only=False)
            except TypeError:
                peek = torch.load(dual_path, map_location='cpu')
        except Exception as exc:
            print(f"警告: 预读 dual_predictor.pth 失败: {exc}", flush=True)
    dual_predictor = DualVariablePredictorTrainer(
        ppc,
        active_set_data,
        T_delta,
        device,
        dual_net_variant=str(peek.get('dual_net_variant', 'mlp')),
        dual_normalize_targets=bool(peek.get('dual_normalize_targets', False)),
        dual_cosine_loss_weight=float(peek.get('dual_cosine_loss_weight', 0.0)),
        dual_smooth_l1_beta=float(peek.get('dual_smooth_l1_beta', 1.0)),
    )
    if os.path.exists(dual_path):
        dual_predictor.load(dual_path)
    else:
        print(f"警告: 未找到对偶预测器模型 {dual_path}", flush=True)
    
    # 加载代理约束模型
    trainers = {}
    unit_predictor_obj = None
    if bool(use_unit_predictor):
        resolved_unit_predictor_path = unit_predictor_path or os.path.join(load_dir, 'unit_predictor.pth')
        if os.path.exists(resolved_unit_predictor_path):
            unit_predictor_obj = SingleUnitBinaryPredictorTrainer(
                ppc,
                active_set_data,
                T_delta,
                unit_ids=[int(g) for g in unit_ids],
                hidden_dims=unit_predictor_hidden_dims,
                net_variant=unit_predictor_net_variant,
                resmlp_width=unit_predictor_resmlp_width,
                resmlp_depth=unit_predictor_resmlp_depth,
                tcn_channels=unit_predictor_tcn_channels,
                tcn_depth=unit_predictor_tcn_depth,
                tconv_channels=unit_predictor_tconv_channels,
                tconv_depth=unit_predictor_tconv_depth,
                dropout=unit_predictor_dropout,
                weight_decay=unit_predictor_weight_decay,
                device=device,
            )
            loaded_ok = bool(unit_predictor_obj.load(resolved_unit_predictor_path))
            if not loaded_ok:
                raise RuntimeError(
                    f"unit_predictor checkpoint incompatible or empty: {resolved_unit_predictor_path}"
                )
        else:
            print(
                f"Warning: use_unit_predictor=True but unit_predictor checkpoint is missing: "
                f"{resolved_unit_predictor_path}",
                flush=True,
            )

    missing_surrogate_units = []
    for g in unit_ids:
        surrogate_path = os.path.join(load_dir, f'surrogate_unit_{g}.pth')
        if not os.path.exists(surrogate_path):
            print(f"警告: 未找到机组{g}代理约束模型 {surrogate_path}", flush=True)
            missing_surrogate_units.append(int(g))
            continue

        metadata = _load_surrogate_model_metadata(surrogate_path, device=device)
        saved_strategy = normalize_constraint_generation_strategy(
            metadata.get('constraint_generation_strategy', 'sensitive')
        )
        saved_lp_backend = normalize_lp_backend(metadata.get('lp_backend', LP_BACKEND_GUROBI))
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
        requested_lp_backend = (
            saved_lp_backend if lp_backend is None else normalize_lp_backend(lp_backend)
        )
        if requested_lp_backend != saved_lp_backend:
            raise ValueError(
                f"Surrogate model lp_backend mismatch for unit {g}: "
                f"requested={requested_lp_backend}, saved={saved_lp_backend}"
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
            lp_backend=requested_lp_backend,
            constraint_generation_strategy=requested_strategy,
            ignore_startup_shutdown_costs=requested_ignore_startup_shutdown_costs,
            nn_hidden_dims=metadata.get('nn_hidden_dims'),
            pg_cost_hidden_dims=metadata.get('pg_cost_hidden_dims'),
            case_name=case_name,
            unit_predictor=unit_predictor_obj,
            use_unit_predictor=(unit_predictor_obj is not None),
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            skip_initial_solve=skip_initial_solve,
            device=device,
        )
        trainer.load(surrogate_path)
        trainers[g] = trainer

    if missing_surrogate_units:
        print(
            f"Warning: missing surrogate model files for requested units: {missing_surrogate_units}",
            flush=True,
        )
    
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
    Ton = int(getattr(trainer, "subproblem_Ton", min(4, T)))
    Toff = int(getattr(trainer, "subproblem_Toff", min(4, T)))
    for tau in range(1, Ton+1):
        for t1 in range(T - tau):
            model.addConstr(x[t1+1] - x[t1] <= x[t1+tau])
    for tau in range(1, Toff+1):
        for t1 in range(T - tau):
            model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau])
    
    # 发电成本
    b_nl = trainer.subproblem_generation_no_load_coeff(g)
    for t in range(T):
        model.addConstr(
            cpower[t] >= trainer.gencost[g, -2] / trainer.T_delta * pg[t] + b_nl * x[t]
        )
    
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
    dual_net_variant: str = "mlp",
    dual_normalize_targets: bool = False,
    dual_cosine_loss_weight: float = 0.0,
    dual_smooth_l1_beta: float = 1.0,
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
    self.dual_net_variant = str(dual_net_variant or "mlp").strip().lower()
    self.dual_normalize_targets = bool(dual_normalize_targets)
    self.dual_cosine_loss_weight = max(float(dual_cosine_loss_weight), 0.0)
    self.dual_smooth_l1_beta = max(float(dual_smooth_l1_beta), 1e-6)
    self._dual_y_mean_np: Optional[np.ndarray] = None
    self._dual_y_std_np: Optional[np.ndarray] = None

    # 先求监督标签，再做目标标准化与网络构造（标准化仅用负荷侧可得的统计量）
    self.lambda_targets = self._solve_for_true_dual_variables()
    self.lambda_true = self.lambda_targets
    if self.dual_normalize_targets and self.lambda_targets.size:
        y = np.asarray(self.lambda_targets, dtype=np.float64)
        self._dual_y_mean_np = y.mean(axis=0)
        self._dual_y_std_np = np.maximum(y.std(axis=0), 1e-3)

    if TORCH_AVAILABLE:
        use_temporal = (
            self.dual_net_variant == "temporal_conv"
            and self.input_dim == 2 * self.nb * self.T
        )
        if self.dual_net_variant == "temporal_conv" and not use_temporal:
            print(
                f"[DualPredictor] temporal_conv 需要 input_dim==2*nb*T "
                f"（当前 input_dim={self.input_dim}, 2*nb*T={2 * self.nb * self.T}），回退到 MLP",
                flush=True,
            )
        if use_temporal:
            self.network = DualVariablePredictorNetTemporalConv(
                self.nb, self.T, self.output_dim
            ).to(self.device)
        else:
            self.network = DualVariablePredictorNet(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    if self.lambda_targets.size:
        y_flat = self.lambda_targets.reshape(-1)
        print(
            f"[DualPredictor] lambda_targets: shape={self.lambda_targets.shape}, "
            f"min={float(np.min(y_flat)):.4g}, max={float(np.max(y_flat)):.4g}, "
            f"mean_abs={float(np.mean(np.abs(y_flat))):.4g}, std={float(np.std(y_flat)):.4g}",
            flush=True,
        )
    print(
        f"[DualPredictor] 训练配置: net={self.dual_net_variant}, "
        f"normalize_targets={self.dual_normalize_targets}, "
        f"cosine_w={self.dual_cosine_loss_weight}, smooth_l1_beta={self.dual_smooth_l1_beta}",
        flush=True,
    )


def _dual_predictor_trainer_solve_true(self) -> np.ndarray:
    lambda_targets = {}
    lambda_payloads = {}
    for sample_id in range(self.n_samples):
        sample = self.active_set_data[sample_id]
        effective = _get_effective_pg_prices_from_sample_or_dual_payload(
            sample,
            self.T,
            self.ng,
            self.nl,
            self.generator_injection_sensitivity,
        )
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
    self.network.train()

    y_mean_t: torch.Tensor | None = None
    y_std_t: torch.Tensor | None = None
    if self.dual_normalize_targets and self._dual_y_mean_np is not None:
        y_mean_t = torch.as_tensor(self._dual_y_mean_np, dtype=torch.float32, device=self.device)
        y_std_t = torch.as_tensor(self._dual_y_std_np, dtype=torch.float32, device=self.device)

    beta = float(self.dual_smooth_l1_beta)
    cos_w = float(self.dual_cosine_loss_weight)

    print_interval = max(1, num_epochs // 10)
    print(
        f"[DualPredictor] 开始训练 (epochs={num_epochs}, batch_strategy={resolved_batch_strategy}, "
        f"batch_size={resolved_batch_size}, lr={resolved_learning_rate:.1e}, device={self.device}, "
        f"loss=SmoothL1(beta={beta})"
        + (f"+{cos_w}*cosine" if cos_w > 0 else "")
        + ")",
        flush=True,
    )

    for _epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            self.optimizer.zero_grad()
            pred = self.network(batch_X)
            if y_mean_t is not None and y_std_t is not None:
                y_norm = (batch_Y - y_mean_t) / y_std_t
                loss_fit = torch.nn.functional.smooth_l1_loss(pred, y_norm, beta=beta)
                pred_phys = pred * y_std_t + y_mean_t
            else:
                loss_fit = torch.nn.functional.smooth_l1_loss(pred, batch_Y, beta=beta)
                pred_phys = pred
            if cos_w > 0:
                loss = loss_fit + cos_w * _dual_predictor_batch_cosine_loss(pred_phys, batch_Y)
            else:
                loss = loss_fit
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
        flat = np.asarray(packed, dtype=np.float64).reshape(-1)
        if getattr(self, "dual_normalize_targets", False) and getattr(self, "_dual_y_mean_np", None) is not None:
            flat = flat * np.asarray(self._dual_y_std_np, dtype=np.float64) + np.asarray(
                self._dual_y_mean_np, dtype=np.float64
            )
        payload = flat.reshape(self.ng, self.T)
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
        state['dual_net_variant'] = getattr(self, 'dual_net_variant', 'mlp')
        state['dual_normalize_targets'] = bool(getattr(self, 'dual_normalize_targets', False))
        state['dual_cosine_loss_weight'] = float(getattr(self, 'dual_cosine_loss_weight', 0.0))
        state['dual_smooth_l1_beta'] = float(getattr(self, 'dual_smooth_l1_beta', 1.0))
        if getattr(self, '_dual_y_mean_np', None) is not None:
            state['dual_y_mean'] = np.asarray(self._dual_y_mean_np, dtype=np.float64)
            state['dual_y_std'] = np.asarray(self._dual_y_std_np, dtype=np.float64)
        torch.save(state, filepath)


def _dual_predictor_trainer_load(self, filepath: str):
    if TORCH_AVAILABLE:
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
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
            self.dual_net_variant = str(state.get('dual_net_variant', 'mlp')).strip().lower()
            self.dual_normalize_targets = bool(state.get('dual_normalize_targets', False))
            self.dual_cosine_loss_weight = float(state.get('dual_cosine_loss_weight', 0.0))
            self.dual_smooth_l1_beta = float(state.get('dual_smooth_l1_beta', 1.0))
            if state.get('dual_y_mean') is not None and state.get('dual_y_std') is not None:
                self._dual_y_mean_np = np.asarray(state['dual_y_mean'], dtype=np.float64)
                self._dual_y_std_np = np.asarray(state['dual_y_std'], dtype=np.float64)
            legacy_dim = state.get('lambda_output_dim')
            if legacy_dim == self.output_dim:
                self._legacy_mode = None
                use_temporal = (
                    self.dual_net_variant == 'temporal_conv'
                    and self.input_dim == 2 * self.nb * self.T
                )
                if use_temporal:
                    self.network = DualVariablePredictorNetTemporalConv(
                        self.nb, self.T, self.output_dim
                    ).to(self.device)
                else:
                    self.network = DualVariablePredictorNet(self.input_dim, self.output_dim).to(self.device)
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
                self.network.load_state_dict(state['network_state_dict'])
                if 'optimizer_state_dict' in state:
                    try:
                        self.optimizer.load_state_dict(state['optimizer_state_dict'])
                    except Exception:
                        pass
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
        effective = _get_effective_pg_prices_from_sample_or_dual_payload(
            sample,
            self.T,
            self.ng,
            self.nl,
            self.generator_injection_sensitivity,
        )
        if effective is None:
            x_sol = _recover_unit_commitment_matrix(sample, self.ng, self.T)
            payload = _solve_pg_electricity_price_from_ed(
                self.ppc,
                sample['pd_data'],
                self.T_delta,
                x_sol,
                renewable_data=sample.get('renewable_data'),
                verbose=False,
                ignore_fixed_generation_cost=bool(self.ignore_startup_shutdown_costs),
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
