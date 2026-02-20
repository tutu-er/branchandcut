"""
UC-NN: 结合BCD迭代和神经网络更新的混合方法
================================================================================
重构自 uc_NN_BCD.py，整合代码结构，保持功能不变

主要组件：
- ActiveSetReader: 读取和解析活动集JSON文件
- Agent_NN: 混合方法的主类，结合BCD迭代和神经网络

功能说明：
- x和对偶变量采用BCD方法迭代
- theta和zeta变量采用神经网络更新
- 约束构建采用直接优化系数的形式
================================================================================
"""

# ==============================================================================
# 导入模块
# ==============================================================================
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import io
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap

# PyTorch导入（可选）
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将无法使用神经网络功能", flush=True)

# PyPower导入
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A
from pypower.makePTDF import makePTDF

# ED求解器导入（可选）
try:
    from ed_gurobipy import EconomicDispatchGurobi
    ED_GUROBI_AVAILABLE = True
except ImportError:
    ED_GUROBI_AVAILABLE = False
    print("警告: ed_gurobipy未安装，将无法使用ED问题求解功能", flush=True)

# PyPower测试用例（可选）
try:
    import pypower
    import pypower.case39
    import pypower.case14
    import pypower.case30
    PYPOWER_AVAILABLE = True
except ImportError:
    PYPOWER_AVAILABLE = False
    print("警告: pypower未安装，测试代码可能无法运行", flush=True)

# 设置输出缓冲
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)


# ==============================================================================
# 工具类：ActiveSetReader
# ==============================================================================
class ActiveSetReader:
    """读取和解析活动集JSON文件的工具类"""
    
    def __init__(self, json_filepath: str):
        """初始化活动集读取器"""
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
        """获取指定样本的完整数据"""
        samples = self.data.get('all_samples', [])
        if 0 <= sample_id < len(samples):
            return samples[sample_id]
        print(f"样本ID {sample_id} 超出范围 [0, {len(samples)-1}]", flush=True)
        return None
    
    def get_total_samples_count(self) -> int:
        """获取总样本数量"""
        return len(self.data.get('all_samples', []))
    
    def load_all_samples(self) -> List[Dict]:
        """加载所有样本的活动集数据"""
        all_samples_data = []
        total_samples = self.get_total_samples_count()
        print(f"开始加载 {total_samples} 个样本的数据...", flush=True)
        
        for sample_id in range(total_samples):
            try:
                active_constraints, active_variables, pd_data = self.extract_active_constraints_and_variables(sample_id)
                unit_commitment = self.get_unit_commitment_matrix(sample_id)
                
                sample_data = {
                    'sample_id': sample_id,
                    'active_constraints': active_constraints,
                    'active_variables': active_variables,
                    'pd_data': pd_data,
                    'unit_commitment_matrix': unit_commitment
                }
                
                sample = self.get_sample_data(sample_id)
                if sample and 'lambda' in sample:
                    sample_data['lambda'] = sample['lambda']
                
                all_samples_data.append(sample_data)
                
                if (sample_id + 1) % 10 == 0:
                    print(f"已加载 {sample_id + 1}/{total_samples} 个样本", flush=True)
                    
            except Exception as e:
                print(f"加载样本 {sample_id} 时出错: {e}", flush=True)
                all_samples_data.append({
                    'sample_id': sample_id,
                    'active_constraints': [],
                    'active_variables': [],
                    'pd_data': np.array([]),
                    'unit_commitment_matrix': np.array([]),
                    'error': str(e)
                })
        
        print(f"✓ 完成加载所有样本数据", flush=True)
        return all_samples_data
    
    def extract_active_constraints_and_variables(self, sample_id: int) -> Tuple[List, List, np.ndarray]:
        """提取指定样本的起作用约束、变量和对应的Pd数据"""
        sample = self.get_sample_data(sample_id)
        if sample is None:
            return [], [], np.array([])
        
        active_set = sample['active_set']
        pd_data = np.array(sample['pd_data'])
        
        active_constraints = []
        active_variables = []
        
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
                active_constraints.append({
                    'type': 'constraint',
                    'constraint_id': item,
                    'dual_value': None,
                    'constraint_name': f'constraint_{item}'
                })
        
        return active_constraints, active_variables, pd_data
    
    def get_unit_commitment_matrix(self, sample_id: int) -> np.ndarray:
        """获取机组启停状态矩阵"""
        _, active_variables, _ = self.extract_active_constraints_and_variables(sample_id)
        
        if not active_variables:
            return np.array([])
        
        binary_vars = [v for v in active_variables if v['type'] == 'binary_variable']
        if not binary_vars:
            return np.array([])
            
        max_unit = max(v['unit_id'] for v in binary_vars) + 1
        max_time = max(v['time_slot'] for v in binary_vars) + 1
        
        unit_commitment = np.zeros((max_unit, max_time), dtype=int)
        for var in binary_vars:
            unit_commitment[var['unit_id'], var['time_slot']] = var['value']
        
        return unit_commitment


def load_active_set_from_json(json_filepath: str, sample_id: Optional[int] = None):
    """从JSON文件加载活动集数据"""
    reader = ActiveSetReader(json_filepath)
    
    if sample_id is not None:
        active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
        unit_commitment = reader.get_unit_commitment_matrix(sample_id)
        
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
        
        sample = reader.get_sample_data(sample_id)
        if sample and 'lambda' in sample:
            sample_data['lambda'] = sample['lambda']
        
        return sample_data
    else:
        all_samples_data = reader.load_all_samples()
        print(f"=== 加载所有活动集数据 ===", flush=True)
        print(f"总样本数量: {len(all_samples_data)}", flush=True)
        return all_samples_data


# ==============================================================================
# 主类：Agent_NN
# ==============================================================================
class Agent_NN:
    """
    结合BCD迭代和神经网络更新的混合方法
    
    - x和对偶变量：使用BCD方法迭代
    - theta和zeta：使用神经网络更新
    - 约束构建：直接优化系数
    """
    
    # --------------------------------------------------------------------------
    # 初始化方法
    # --------------------------------------------------------------------------
    def __init__(self, ppc, active_set_data, T_delta, union_analysis=None):
        """初始化Agent_NN"""
        self.ppc = ppc
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        
        # 迭代参数
        self.iter_number = 0
        self.penalty_factor = 1e2
        self.dual_para_bound = 0.1
        self.dual_para_bound_quit_iteration = 50
        
        # BCD迭代参数
        self.rho_primal = 1e-2
        self.rho_dual = 1e-2
        self.rho_opt = 1e-2
        self.gamma = 1e-1
        
        # 约束参数
        self.constraint_violation_weight = 0
        self.constraint_violation_epsilon = 1e-3
        self.enable_theta_constraints = True
        self.enable_zeta_constraints = True
        self.use_per_variable_zeta_constraints = True
        self.use_fischer_burmeister_for_loss = False
        
        # 问题维度
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
        self.ng = self.gen.shape[0]
        self.nl = self.branch.shape[0]
        
        self.active_set_data = active_set_data
        
        # 初始化变量
        self.theta_vars = {}
        self.zeta_vars = {}
        
        # 初始化求解
        self.pg, self.x, self.coc, self.cpower, self.lambda_ = self._initialize_solve()
        
        # 创建union_analysis
        if union_analysis is None:
            self._current_union_analysis = self._create_union_analysis_from_x_init(self.x, self.lambda_)
        else:
            self._current_union_analysis = union_analysis
        
        # 创建theta和zeta变量
        self._add_theta_variables_for_branches(self._current_union_analysis)
        self._add_zeta_variables_for_units(self._current_union_analysis)
        
        # 初始化theta和zeta值
        self.theta_values, self.mu = self._initialize_theta_values(self._current_union_analysis)
        self.zeta_values, self.ita = self._initialize_zeta_values(self._current_union_analysis)
        
        # 初始化神经网络
        if TORCH_AVAILABLE:
            self._init_neural_network()
        else:
            self.theta_net = None
            self.zeta_net = None
            self.device = None
    
    def _initialize_solve(self):
        """初始化求解，获得初始x和lambda"""
        pg_sol, x_sol, coc_sol, cpower_sol, lambda_sol = [], [], [], [], []
        
        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']
            model = gp.Model('initial_solve')
            model.Params.OutputFlag = 0
            
            # 添加变量
            pg = model.addVars(self.ng, self.T, lb=0, name='pg')
            x = model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
            coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
            cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
            
            # 添加约束
            self._add_uc_constraints(model, pg, x, coc, cpower, Pd)
            
            # 设置目标函数
            primal_obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                         gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
            model.setObjective(primal_obj, GRB.MINIMIZE)
            model.setParam("Presolve", 2)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                pg_sol.append(np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)]))
                x_sol.append(np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)]))
                coc_sol.append(np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)]))
                cpower_sol.append(np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)]))
                lambda_sol.append(self._extract_dual_variables_as_arrays(model))
            else:
                print(f"警告: 样本 {sample_id} 的模型求解失败，状态: {model.status}", flush=True)
                pg_sol.append(np.zeros((self.ng, self.T)))
                x_sol.append(np.zeros((self.ng, self.T)))
                coc_sol.append(np.zeros((self.ng, self.T-1)))
                cpower_sol.append(np.zeros((self.ng, self.T)))
                lambda_sol.append(self._create_empty_lambda_dict())
        
        return np.array(pg_sol), np.array(x_sol), np.array(coc_sol), np.array(cpower_sol), lambda_sol
    
    def _add_uc_constraints(self, model, pg, x, coc, cpower, Pd):
        """添加UC问题的基本约束"""
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
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        for g in range(self.ng):
            for t in range(1, self.T):
                model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + self.gen[g, PMAX] * (1 - x[g, t-1]), 
                              name=f'ramp_up_{g}_{t}')
                model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + self.gen[g, PMAX] * (1 - x[g, t]), 
                              name=f'ramp_down_{g}_{t}')
        
        # 最小开关机时间约束
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+t], name=f'min_on_{g}_{t}_{t1}')
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+t], name=f'min_off_{g}_{t}_{t1}')
        
        # 启停成本约束
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]), name=f'start_cost_{g}_{t}')
                model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]), name=f'shut_cost_{g}_{t}')
                model.addConstr(coc[g, t-1] >= 0, name=f'coc_nonneg_{g}_{t}')
        
        # 发电成本约束
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] >= self.gencost[g, -2]/self.T_delta * pg[g, t] + 
                              self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')
    
    # --------------------------------------------------------------------------
    # 对偶变量提取方法
    # --------------------------------------------------------------------------
    def _extract_dual_variables(self, model) -> Dict:
        """通过约束名称提取对偶变量"""
        implicit_duals = {}
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        try:
            # 功率平衡约束
            implicit_duals['power_balance'] = {}
            for t in range(self.T):
                constr = model.getConstrByName(f'power_balance_{t}')
                if constr:
                    try:
                        implicit_duals['power_balance'][t] = constr.Pi
                    except:
                        implicit_duals['power_balance'][t] = 0.0
            
            # 发电上下限约束
            for prefix in ['pg_lower', 'pg_upper']:
                implicit_duals[prefix] = {}
                for g in range(self.ng):
                    implicit_duals[prefix][g] = {}
                    for t in range(self.T):
                        constr = model.getConstrByName(f'{prefix}_{g}_{t}')
                        if constr:
                            try:
                                implicit_duals[prefix][g][t] = constr.Pi
                            except:
                                implicit_duals[prefix][g][t] = 0.0
            
            # 爬坡约束
            for prefix in ['ramp_up', 'ramp_down']:
                implicit_duals[prefix] = {}
                for g in range(self.ng):
                    implicit_duals[prefix][g] = {}
                    for t in range(1, self.T):
                        constr = model.getConstrByName(f'{prefix}_{g}_{t}')
                        if constr:
                            try:
                                implicit_duals[prefix][g][t-1] = constr.Pi
                            except:
                                implicit_duals[prefix][g][t-1] = 0.0
            
            # 最小开关机时间约束
            for prefix, T_limit in [('min_on', Ton), ('min_off', Toff)]:
                implicit_duals[prefix] = {}
                for g in range(self.ng):
                    implicit_duals[prefix][g] = {}
                    for tau in range(1, T_limit+1):
                        for t1 in range(self.T - tau):
                            cname = f'{prefix}_{g}_{tau}_{t1}'
                            constr = model.getConstrByName(cname)
                            if constr:
                                if tau not in implicit_duals[prefix][g]:
                                    implicit_duals[prefix][g][tau] = {}
                                try:
                                    implicit_duals[prefix][g][tau][t1] = constr.Pi
                                except:
                                    implicit_duals[prefix][g][tau][t1] = 0.0
            
            # 启停成本约束
            for prefix in ['start_cost', 'shut_cost', 'coc_nonneg']:
                implicit_duals[prefix] = {}
                for g in range(self.ng):
                    implicit_duals[prefix][g] = {}
                    for t in range(1, self.T):
                        constr = model.getConstrByName(f'{prefix}_{g}_{t}')
                        if constr:
                            try:
                                implicit_duals[prefix][g][t-1] = constr.Pi
                            except:
                                implicit_duals[prefix][g][t-1] = 0.0
            
            # 发电成本约束
            implicit_duals['cpower'] = {}
            for g in range(self.ng):
                implicit_duals['cpower'][g] = {}
                for t in range(self.T):
                    constr = model.getConstrByName(f'cpower_{g}_{t}')
                    if constr:
                        try:
                            implicit_duals['cpower'][g][t] = constr.Pi
                        except:
                            implicit_duals['cpower'][g][t] = 0.0
            
            # DCPF约束
            for prefix in ['dcpf_upper', 'dcpf_lower']:
                implicit_duals[prefix] = {}
                for l in range(self.nl):
                    implicit_duals[prefix][l] = {}
                    for t in range(self.T):
                        constr = model.getConstrByName(f'flow_{prefix.split("_")[1]}_{l}_{t}')
                        if constr:
                            try:
                                implicit_duals[prefix][l][t] = constr.Pi
                            except:
                                implicit_duals[prefix][l][t] = 0.0
            
            # x变量上下界约束
            for prefix in ['x_upper', 'x_lower']:
                implicit_duals[prefix] = {}
                for g in range(self.ng):
                    implicit_duals[prefix][g] = {}
                    for t in range(self.T):
                        constr = model.getConstrByName(f'{prefix}_{g}_{t}')
                        if constr:
                            try:
                                implicit_duals[prefix][g][t] = constr.Pi
                            except:
                                implicit_duals[prefix][g][t] = 0.0

            return implicit_duals
        except Exception as e:
            print(f"❌ 对偶变量提取过程中出错: {e}", flush=True)
            return {}
    
    def _extract_dual_variables_as_arrays(self, model) -> Dict:
        """提取对偶变量并转换为numpy数组格式"""
        try:
            implicit_duals_dict = self._extract_dual_variables(model)
            lambda_sol = {}
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            
            # 功率平衡约束: shape (T,)
            lambda_sol['lambda_power_balance'] = np.array([
                implicit_duals_dict.get('power_balance', {}).get(t, 0) for t in range(self.T)
            ])
            
            # 发电上下限约束: shape (ng, T)
            for prefix in ['pg_lower', 'pg_upper']:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.array([
                        [abs(implicit_duals_dict[prefix].get(g, {}).get(t, 0)) for t in range(self.T)] 
                        for g in range(self.ng)
                    ])
                else:
                    lambda_sol[key] = np.zeros((self.ng, self.T))
            
            # 爬坡约束: shape (ng, T-1)
            for prefix in ['ramp_up', 'ramp_down']:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.array([
                        [abs(implicit_duals_dict[prefix].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                        for g in range(self.ng)
                    ])
                else:
                    lambda_sol[key] = np.zeros((self.ng, self.T-1))
            
            # 最小开关机时间约束: shape (ng, T_limit, T)
            for prefix, T_limit in [('min_on', Ton), ('min_off', Toff)]:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.zeros((self.ng, T_limit, self.T))
                    for g in range(self.ng):
                        for tau in range(T_limit):
                            for t in range(self.T):
                                val = implicit_duals_dict[prefix].get(g, {}).get(tau+1, {}).get(t, 0)
                                lambda_sol[key][g, tau, t] = abs(val)
                else:
                    lambda_sol[key] = np.zeros((self.ng, T_limit, self.T))
            
            # 启停成本约束: shape (ng, T-1)
            for prefix in ['start_cost', 'shut_cost', 'coc_nonneg']:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.array([
                        [abs(implicit_duals_dict[prefix].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                        for g in range(self.ng)
                    ])
                else:
                    lambda_sol[key] = np.zeros((self.ng, self.T-1))
            
            # 发电成本约束: shape (ng, T)
            if 'cpower' in implicit_duals_dict:
                lambda_sol['lambda_cpower'] = np.array([
                    [abs(implicit_duals_dict['cpower'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            else:
                lambda_sol['lambda_cpower'] = np.zeros((self.ng, self.T))
            
            # DCPF约束: shape (nl, T)
            for prefix in ['dcpf_upper', 'dcpf_lower']:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.array([
                        [abs(implicit_duals_dict[prefix].get(l, {}).get(t, 0)) for t in range(self.T)] 
                        for l in range(self.nl)
                    ])
                else:
                    lambda_sol[key] = np.zeros((self.nl, self.T))
            
            # x变量上下界约束: shape (ng, T)
            for prefix in ['x_upper', 'x_lower']:
                key = f'lambda_{prefix}'
                if prefix in implicit_duals_dict:
                    lambda_sol[key] = np.array([
                        [abs(implicit_duals_dict[prefix].get(g, {}).get(t, 0)) for t in range(self.T)] 
                        for g in range(self.ng)
                    ])
                else:
                    lambda_sol[key] = np.zeros((self.ng, self.T))
            
            return lambda_sol
            
        except Exception as e:
            print(f"❌ 对偶变量数组转换失败: {e}", flush=True)
            return self._create_empty_lambda_dict()
    
    def _create_empty_lambda_dict(self) -> Dict:
        """创建空的对偶变量字典"""
        Ton = min(4, self.T)
        Toff = min(4, self.T)
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
    
    # --------------------------------------------------------------------------
    # Union Analysis和约束创建方法
    # --------------------------------------------------------------------------
    def _create_union_analysis_from_x_init(self, x_init, lambda_init) -> Dict:
        """创建union_analysis"""
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
        manual_constraints_count = 0
        
        if self.enable_theta_constraints:
            union_constraints = self._compute_dcpf_constraints_for_fractional_times(fractional_variables, lambda_init)
            M = 4
            manual_constraints = self._add_manual_constraints_all_units(M)
            union_constraints.extend(manual_constraints)
            manual_constraints_count = len(manual_constraints)
        
        if self.enable_zeta_constraints:
            if self.use_per_variable_zeta_constraints:
                union_zeta_constraints = self._create_per_variable_zeta_constraints()
            else:
                union_zeta_constraints = self._compute_specialized_constraints_of_balance_node(fractional_variables)
        
        print(f"生成 {len(union_constraints)} 个theta约束 (包含 {manual_constraints_count} 个手动添加的约束), "
              f"生成 {len(union_zeta_constraints)} 个zeta约束")
        
        return {
            'union_constraints': union_constraints,
            'union_zeta_constraints': union_zeta_constraints,
        }
    
    def _compute_dcpf_constraints_for_fractional_times(self, fractional_variables, lambda_init) -> List:
        """计算DCPF约束"""
        union_constraints = []
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        
        for var in fractional_variables:
            unit_id = var['unit_id']
            time_slot = var['time_slot']
            bus_idx = int(self.gen[unit_id, GEN_BUS])
            
            if 0 <= bus_idx < nb:
                nonzero_coefficients = []
                for branch_id in range(self.nl):
                    coeff = PTDF[branch_id, bus_idx]
                    if abs(coeff) > 1e-6:
                        nonzero_coefficients.append({
                            'unit_id': unit_id,
                            'branch_id': branch_id,
                            'coefficient': coeff
                        })
                
                if nonzero_coefficients:
                    union_constraints.append({
                        'branch_id': branch_id,
                        'time_slot': time_slot,
                        'constraint_type': 'dcpf_upper',
                        'nonzero_pg_coefficients': nonzero_coefficients,
                        'constraint_name': f'dcpf_upper_{branch_id}_{time_slot}'
                    })
        
        return union_constraints
    
    def _add_manual_constraints_all_units(self, M=4) -> List:
        """手动添加包含所有机组的约束"""
        manual_constraints = []
        max_branch_id = self.nl - 1
        
        for t in range(self.T):
            for m in range(M):
                virtual_branch_id = max_branch_id + 1 + m
                nonzero_coefficients = [{
                    'unit_id': g,
                    'branch_id': virtual_branch_id,
                    'coefficient': 1.0 / self.ng
                } for g in range(self.ng)]
                
                manual_constraints.append({
                    'branch_id': virtual_branch_id,
                    'time_slot': t,
                    'constraint_type': 'manual_all_units',
                    'nonzero_pg_coefficients': nonzero_coefficients,
                    'constraint_name': f'manual_all_units_{virtual_branch_id}_{t}'
                })
        
        return manual_constraints
    
    def _create_per_variable_zeta_constraints(self) -> List:
        """创建每个变量一一对应的zeta约束"""
        constraints = [{
            'unit_id': g,
            'time_slot': t,
            'constraint_type': 'per_variable_zeta',
            'constraint_name': f'zeta_per_variable_{g}_{t}',
            'is_per_variable': True
        } for g in range(self.ng) for t in range(self.T)]
        
        print(f"✓ 创建了 {len(constraints)} 个每个变量一一对应的zeta约束")
        return constraints
    
    def _compute_specialized_constraints_of_balance_node(self, fractional_variables) -> List:
        """计算平衡节点约束"""
        return [{
            'time_slot': var['time_slot'],
            'unit_id': var['unit_id'],
            'constraint_type': 'balance_node_power',
            'constraint_name': f"balance_node_power_{var['unit_id']}_{var['time_slot']}"
        } for var in fractional_variables]
    
    # --------------------------------------------------------------------------
    # Theta和Zeta变量初始化方法
    # --------------------------------------------------------------------------
    def _initialize_theta_values(self, union_analysis=None) -> Tuple[Dict, np.ndarray]:
        """初始化theta值"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化theta值")
            return {}, np.zeros((self.n_samples, self.nl, self.T))
        
        theta_values = {}
        np.random.seed(42)
        
        for constraint in union_analysis['union_constraints']:
            branch_id = constraint['branch_id']
            time_slot = constraint['time_slot']
            
            for coeff_info in constraint.get('nonzero_pg_coefficients', []):
                unit_id = coeff_info['unit_id']
                var_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                theta_values[var_name] = np.random.normal(0.0, 0.01)
            
            theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
            theta_values[theta_rhs_name] = np.random.normal(1.0, 0.01)
        
        mu_init = np.zeros((self.n_samples, self.nl, self.T), dtype=float)
        return theta_values, mu_init
    
    def _initialize_zeta_values(self, union_analysis=None) -> Tuple[Dict, np.ndarray]:
        """初始化zeta值"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化zeta值")
            return {}, np.zeros((self.n_samples, self.ng, self.T))
        
        zeta_values = {}
        np.random.seed(43)
        
        for constraint in union_analysis['union_zeta_constraints']:
            unit_id = constraint["unit_id"]
            time_slot = constraint["time_slot"]
            
            var_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            zeta_values[var_name] = np.random.normal(0.0, 0.01)
            
            zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            zeta_values[zeta_rhs_name] = np.random.normal(1.0, 0.01)
        
        ita_init = np.zeros((self.n_samples, self.ng, self.T), dtype=float)
        return zeta_values, ita_init
    
    def _add_theta_variables_for_branches(self, union_analysis=None):
        """为参数化约束添加theta变量"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建theta变量")
            return
        
        self.theta_vars = {}
        for constraint in union_analysis['union_constraints']:
            branch_id = constraint['branch_id']
            time_slot = constraint['time_slot']
            
            for coeff_info in constraint.get('nonzero_pg_coefficients', []):
                unit_id = coeff_info['unit_id']
                var_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                self.theta_vars[var_name] = {
                    'branch_id': branch_id, 'unit_id': unit_id, 
                    'time_slot': time_slot, 'var_name': var_name, 'value': 0.0
                }
            
            theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
            self.theta_vars[theta_rhs_name] = {
                'branch_id': branch_id, 'time_slot': time_slot,
                'var_name': theta_rhs_name, 'value': 1.0
            }
        
        print(f"✓ 创建了 {len(self.theta_vars)} 个theta变量")
    
    def _add_zeta_variables_for_units(self, union_analysis=None):
        """为参数化约束添加zeta变量"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建zeta变量")
            return
        
        self.zeta_vars = {}
        for constraint in union_analysis['union_zeta_constraints']:
            unit_id = constraint["unit_id"]
            time_slot = constraint["time_slot"]
            
            var_name = f'zeta_unit_{unit_id}_time_{time_slot}'
            self.zeta_vars[var_name] = {
                'unit_id': unit_id, 'time_slot': time_slot,
                'var_name': var_name, 'value': 0.0
            }
            
            zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
            self.zeta_vars[zeta_rhs_name] = {
                'unit_id': unit_id, 'time_slot': time_slot,
                'var_name': zeta_rhs_name, 'value': 1.0
            }
        
        print(f"✓ 创建了 {len(self.zeta_vars)} 个zeta变量")
    
    # --------------------------------------------------------------------------
    # 神经网络初始化和特征提取
    # --------------------------------------------------------------------------
    def _init_neural_network(self):
        """初始化神经网络模型"""
        if not TORCH_AVAILABLE:
            return
        
        # 检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("⚠ 未检测到GPU，将使用CPU", flush=True)
        
        # 计算维度
        sample_features = self._extract_features(0)
        input_dim = len(sample_features)
        self.theta_output_dim = len(self.theta_values)
        self.zeta_output_dim = len(self.zeta_values)
        
        # 创建网络
        def create_net(output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        
        self.theta_net = create_net(self.theta_output_dim).to(self.device)
        self.zeta_net = create_net(self.zeta_output_dim).to(self.device)
        
        # 创建优化器
        all_params = list(self.theta_net.parameters()) + list(self.zeta_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=1e-4)
        
        # 保存变量名列表
        self.theta_var_names = list(self.theta_values.keys())
        self.zeta_var_names = list(self.zeta_values.keys())
        
        print(f"✓ 初始化神经网络: 设备={self.device}, 输入维度={input_dim}, "
              f"theta输出={self.theta_output_dim}, zeta输出={self.zeta_output_dim}", flush=True)
    
    def _extract_features(self, sample_id) -> np.ndarray:
        """从样本中提取特征"""
        pd_data = self.active_set_data[sample_id]['pd_data']
        return pd_data.flatten()
    
    def _tensor_to_theta_dict(self, theta_tensor) -> Dict:
        """将theta张量转换为字典"""
        if not TORCH_AVAILABLE or theta_tensor is None:
            return self.theta_values.copy()
        values = theta_tensor.detach().cpu().numpy()
        return {name: float(val) for name, val in zip(self.theta_var_names, values)}
    
    def _tensor_to_zeta_dict(self, zeta_tensor) -> Dict:
        """将zeta张量转换为字典"""
        if not TORCH_AVAILABLE or zeta_tensor is None:
            return self.zeta_values.copy()
        values = zeta_tensor.detach().cpu().numpy()
        return {name: float(val) for name, val in zip(self.zeta_var_names, values)}
    
    # --------------------------------------------------------------------------
    # BCD迭代方法
    # --------------------------------------------------------------------------
    def iter_with_pg_block(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """迭代PG块：更新x, pg等原始变量"""
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('iter_with_pg_block')
        model.Params.OutputFlag = 0
        
        # 添加变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 辅助变量
        Ton, Toff = min(4, self.T), min(4, self.T)
        power_balance_viol = model.addVars(self.T, lb=0, name='power_balance_viol')
        pg_lower_viol = model.addVars(self.ng, self.T, lb=0, name='pg_lower_viol')
        pg_upper_viol = model.addVars(self.ng, self.T, lb=0, name='pg_upper_viol')
        pg_lower_abs = model.addVars(self.ng, self.T, lb=0, name='pg_lower_abs')
        pg_upper_abs = model.addVars(self.ng, self.T, lb=0, name='pg_upper_abs')
        ramp_up_viol = model.addVars(self.ng, self.T-1, lb=0, name='ramp_up_viol')
        ramp_down_viol = model.addVars(self.ng, self.T-1, lb=0, name='ramp_down_viol')
        ramp_up_abs = model.addVars(self.ng, self.T-1, lb=0, name='ramp_up_abs')
        ramp_down_abs = model.addVars(self.ng, self.T-1, lb=0, name='ramp_down_abs')
        min_on_viol = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_viol')
        min_off_viol = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_viol')
        min_on_abs = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_abs')
        min_off_abs = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_abs')
        start_cost_viol = model.addVars(self.ng, self.T, lb=0, name='start_cost_viol')
        shut_cost_viol = model.addVars(self.ng, self.T, lb=0, name='shut_cost_viol')
        start_cost_abs = model.addVars(self.ng, self.T, lb=0, name='start_cost_abs')
        shut_cost_abs = model.addVars(self.ng, self.T, lb=0, name='shut_cost_abs')
        dcpf_upper_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_viol')
        dcpf_upper_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_abs')
        dcpf_lower_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_viol')
        dcpf_lower_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_abs')
        x_binary_dev = model.addVars(self.ng, self.T, lb=0, name='x_binary_dev')

        obj_primal, obj_opt, obj_binary = 0, 0, 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_viol[t] >= power_balance_expr, name=f'power_balance_pos_{t}')
            model.addConstr(power_balance_viol[t] >= -power_balance_expr, name=f'power_balance_neg_{t}')
            obj_primal += power_balance_viol[t]
            obj_opt += power_balance_viol[t] * abs(self.lambda_[sample_id]['lambda_power_balance'][t])
            
            for g in range(self.ng):
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_viol[g, t] >= pg_lower_expr, name=f'pg_lower_viol_{g}_{t}')
                obj_primal += pg_lower_viol[g, t]
                
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_viol[g, t] >= pg_upper_expr, name=f'pg_upper_viol_{g}_{t}')
                obj_primal += pg_upper_viol[g, t]
                
                model.addConstr(pg_lower_abs[g, t] >= pg_lower_expr, name=f'pg_lower_abs1_{g}_{t}')
                model.addConstr(pg_lower_abs[g, t] >= -pg_lower_expr, name=f'pg_lower_abs2_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t] >= pg_upper_expr, name=f'pg_upper_abs1_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t] >= -pg_upper_expr, name=f'pg_upper_abs2_{g}_{t}')

                obj_opt += pg_lower_abs[g, t] * abs(self.lambda_[sample_id]['lambda_pg_lower'][g, t])
                obj_opt += pg_upper_abs[g, t] * abs(self.lambda_[sample_id]['lambda_pg_upper'][g, t])
                obj_opt += x[g, t] * abs(self.lambda_[sample_id]['lambda_x_lower'][g, t])
                obj_opt += (1 - x[g, t]) * abs(self.lambda_[sample_id]['lambda_x_upper'][g, t])

        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_viol[g, t-1] >= ramp_up_expr, name=f'ramp_up_viol_{g}_{t}')
                obj_primal += ramp_up_viol[g, t-1]
                model.addConstr(ramp_up_abs[g, t-1] >= ramp_up_expr, name=f'ramp_up_abs1_{g}_{t}')
                model.addConstr(ramp_up_abs[g, t-1] >= -ramp_up_expr, name=f'ramp_up_abs2_{g}_{t}')

                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_viol[g, t-1] >= ramp_down_expr, name=f'ramp_down_viol_{g}_{t}')
                obj_primal += ramp_down_viol[g, t-1]
                model.addConstr(ramp_down_abs[g, t-1] >= ramp_down_expr, name=f'ramp_down_abs1_{g}_{t}')
                model.addConstr(ramp_down_abs[g, t-1] >= -ramp_down_expr, name=f'ramp_down_abs2_{g}_{t}')

                obj_opt += ramp_up_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_ramp_up'][g, t-1])
                obj_opt += ramp_down_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_ramp_down'][g, t-1])

        # 最小开关机时间约束
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_viol[g, t-1, t1] >= min_on_expr, name=f'min_on_viol_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g, t-1, t1] >= min_on_expr, name=f'min_on_abs1_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g, t-1, t1] >= -min_on_expr, name=f'min_on_abs2_{g}_{t}_{t1}')
                    obj_primal += min_on_viol[g, t-1, t1]
                    obj_opt += min_on_abs[g, t-1, t1] * abs(self.lambda_[sample_id]['lambda_min_on'][g, t-1, t1])

        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                    model.addConstr(min_off_viol[g, t-1, t1] >= min_off_expr, name=f'min_off_viol_{g}_{t}_{t1}')
                    model.addConstr(min_off_abs[g, t-1, t1] >= min_off_expr, name=f'min_off_abs1_{g}_{t}_{t1}')
                    model.addConstr(min_off_abs[g, t-1, t1] >= -min_off_expr, name=f'min_off_abs2_{g}_{t}_{t1}')
                    obj_primal += min_off_viol[g, t-1, t1]
                    obj_opt += min_off_abs[g, t-1, t1] * abs(self.lambda_[sample_id]['lambda_min_off'][g, t-1, t1])

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(start_cost_viol[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1])
                model.addConstr(start_cost_abs[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1])
                model.addConstr(start_cost_abs[g, t-1] >= -start_cost[g] * (x[g, t] - x[g, t-1]) + coc[g, t-1])
                obj_primal += start_cost_viol[g, t-1]
                obj_opt += start_cost_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_start_cost'][g, t-1])

                model.addConstr(shut_cost_viol[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1])
                model.addConstr(shut_cost_abs[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1])
                model.addConstr(shut_cost_abs[g, t-1] >= -shut_cost[g] * (x[g, t-1] - x[g, t]) + coc[g, t-1])
                obj_primal += shut_cost_viol[g, t-1]
                obj_opt += shut_cost_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_shut_cost'][g, t-1])
                obj_opt += coc[g, t-1] * abs(self.lambda_[sample_id]['lambda_coc_nonneg'][g, t-1])

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + 
                              self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')

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
                model.addConstr(dcpf_upper_viol[l, t] >= flow[l] - branch_limit[l])
                model.addConstr(dcpf_lower_viol[l, t] >= -flow[l] - branch_limit[l])
                model.addConstr(dcpf_upper_abs[l, t] >= flow[l] - branch_limit[l])
                model.addConstr(dcpf_upper_abs[l, t] >= -flow[l] + branch_limit[l])
                model.addConstr(dcpf_lower_abs[l, t] >= -flow[l] - branch_limit[l])
                model.addConstr(dcpf_lower_abs[l, t] >= flow[l] + branch_limit[l])
                obj_primal += dcpf_upper_viol[l, t] + dcpf_lower_viol[l, t]
                obj_opt += dcpf_upper_abs[l, t] * abs(self.lambda_[sample_id]['lambda_dcpf_upper'][l, t])
                obj_opt += dcpf_lower_abs[l, t] * abs(self.lambda_[sample_id]['lambda_dcpf_lower'][l, t])

        # 二进制变量偏差
        unit_commitment_matrix = self.active_set_data[sample_id].get('unit_commitment_matrix', None)
        for g in range(self.ng):
            for t in range(self.T):
                target_value = unit_commitment_matrix[g, t]
                x_dev_expr = x[g, t] - target_value
                model.addConstr(x_binary_dev[g, t] >= x_dev_expr, name=f'x_dev_pos_{g}_{t}')
                model.addConstr(x_binary_dev[g, t] >= -x_dev_expr, name=f'x_dev_neg_{g}_{t}')
                obj_binary += x_binary_dev[g, t]

        # 添加参数化约束的罚项
        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis and theta_values:
            model, para_primal, para_opt = self._add_parametric_penalties_pg_block(
                model, x, sample_id, theta_values, union_analysis, PTDF=PTDF, branch_limit=branch_limit
            )
            obj_primal += para_primal
            obj_opt += para_opt
        
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis and zeta_values:
            model, para_primal, para_opt = self._add_parametric_balance_power_penalties_pg_block(
                model, x, sample_id, zeta_values, union_analysis
            )
            obj_primal += para_primal
            obj_opt += para_opt
        
        # 设置目标函数
        total_objective = obj_binary + self.rho_primal * obj_primal + self.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)
        model.Params.MIPGap = 1e-6
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])
            
            if sample_id <= 2:
                print(f"pg_block, sample_id: {sample_id}, obj_primal: {obj_primal.getValue()}, "
                      f"obj_opt: {obj_opt.getValue()}, obj_binary: {obj_binary.getValue()}")
            
            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ PG块模型求解失败，状态: {model.status}", flush=True)
            return None, None, None, None
    
    def iter_with_dual_block(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """迭代对偶块：更新对偶变量lambda, mu, ita"""
        model = gp.Model('iter_with_dual_block')
        model.Params.OutputFlag = 0
        Pd = self.active_set_data[sample_id]['pd_data']
        
        Ton, Toff = min(4, self.T), min(4, self.T)
        
        # 对偶变量
        lambda_power_balance = model.addVars(self.T, lb=-GRB.INFINITY, name='lambda_power_balance')
        lambda_pg_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_lower')
        lambda_pg_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_upper')
        lambda_ramp_up = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_up')
        lambda_ramp_down = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_down')
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
        
        # mu和ita变量
        lb = self.dual_para_bound if self.iter_number < self.dual_para_bound_quit_iteration else 0
        mu = model.addVars(self.nl, self.T, lb=lb, name='mu')
        ita = model.addVars(self.ng, self.T, lb=lb, name='ita')

        # 预计算
        nb = Pd.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]

        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        obj_dual, obj_opt = 0, 0
                
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
                
                ptdfg_col = (PTDF @ G[:, g]).T
                for l in range(self.branch.shape[0]):
                    pg_coeff = ptdfg_col[l]
                    dual_expr += pg_coeff * (lambda_dcpf_upper[l, t] - lambda_dcpf_lower[l, t])
                
                dual_expr_pg_abs = model.addVar(lb=0, name=f'dual_expr_abs_pg_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= dual_expr)
                model.addConstr(dual_expr_pg_abs >= -dual_expr)
                obj_dual += dual_expr_pg_abs
        
        # x变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self.gencost[g, -1] / self.T_delta
                dual_expr += lambda_x_upper[g, t] - lambda_x_lower[g, t]
                dual_expr += self.gen[g, PMIN] * lambda_pg_lower[g, t]
                dual_expr -= self.gen[g, PMAX] * lambda_pg_upper[g, t]
                
                if t > 0:
                    dual_expr += (Rd_co[g] - Rd[g]) * lambda_ramp_down[g, t-1]
                if t < self.T - 1:
                    dual_expr += (Ru_co[g] - Ru[g]) * lambda_ramp_up[g, t]

                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr += lambda_min_on[g, tau-1, t1]
                        if t == t1:
                            dual_expr -= lambda_min_on[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr -= lambda_min_on[g, tau-1, t1]
                            
                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr -= lambda_min_off[g, tau-1, t1]
                        if t == t1:
                            dual_expr += lambda_min_off[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr += lambda_min_off[g, tau-1, t1]

                if t > 0:
                    dual_expr += start_cost[g] * lambda_start_cost[g, t-1]
                    dual_expr -= shut_cost[g] * lambda_shut_cost[g, t-1]
                if t < self.T - 1:
                    dual_expr -= start_cost[g] * lambda_start_cost[g, t]
                    dual_expr += shut_cost[g] * lambda_shut_cost[g, t]

                # 参数化约束的对偶贡献
                if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
                    dual_expr_para = self._add_parametric_constraints_dual_block_const_to_model(
                        model, g, t, mu, sample_id, theta_values, union_analysis
                    )
                    dual_expr += dual_expr_para

                if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
                    dual_expr_para = self._add_parametric_balance_power_constraints_dual_block_const_to_model(
                        model, g, t, ita, sample_id, zeta_values, union_analysis
                    )
                    dual_expr += dual_expr_para
                                
                dual_expr_x_abs = model.addVar(lb=0, name=f'dual_expr_abs_x_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= dual_expr)
                model.addConstr(dual_expr_x_abs >= -dual_expr)
                obj_dual += dual_expr_x_abs

        # coc变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T-1):
                dual_expr = 1 - lambda_start_cost[g, t] - lambda_shut_cost[g, t] - lambda_coc_nonneg[g, t]
                dual_expr_coc_abs = model.addVar(lb=0, name=f'dual_expr_abs_coc_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= dual_expr)
                model.addConstr(dual_expr_coc_abs >= -dual_expr)
                obj_dual += dual_expr_coc_abs

        # cpower变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(lambda_cpower[g, t] == 1)

        # 原问题约束违反量
        for t in range(self.T):
            lambda_power_balance_abs = model.addVar(lb=0)
            model.addConstr(lambda_power_balance_abs >= lambda_power_balance[t])
            model.addConstr(lambda_power_balance_abs >= -lambda_power_balance[t])
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

        # 添加参数化约束的obj_opt项
        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis and theta_values:
            model, obj_opt_para = self._add_parametric_obj_dual_block(
                model, self.x[sample_id, :, :], mu, sample_id, theta_values, union_analysis, PTDF=PTDF, branch_limit=branch_limit
            )
            obj_opt += obj_opt_para
        
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis and zeta_values:
            model, obj_opt_para = self._add_parametric_balance_power_obj_dual_block(
                model, self.x[sample_id, :, :], ita, sample_id, zeta_values, union_analysis
            )
            obj_opt += obj_opt_para
   
        # 设置目标函数
        total_objective = self.rho_dual * obj_dual + self.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)

        model.Params.MIPGap = 1e-6
        model.Params.Presolve = 0
        model.Params.NumericFocus = 2
        model.Params.ScaleFlag = 2
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            lambda_sol = {
                'lambda_power_balance': np.array([lambda_power_balance[t].X for t in range(self.T)]),
                'lambda_pg_lower': np.array([[lambda_pg_lower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_pg_upper': np.array([[lambda_pg_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_ramp_up': np.array([[lambda_ramp_up[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_ramp_down': np.array([[lambda_ramp_down[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_min_on': np.array([[[lambda_min_on[g, tau, t].X for t in range(self.T)] for tau in range(Ton)] for g in range(self.ng)]),
                'lambda_min_off': np.array([[[lambda_min_off[g, tau, t].X for t in range(self.T)] for tau in range(Toff)] for g in range(self.ng)]),
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
                print(f"dual_block, sample_id: {sample_id}, obj_dual: {obj_dual.getValue()}, obj_opt: {obj_opt.getValue()}")
            
            return lambda_sol, mu_sol, ita_sol
        else:
            print(f"❌ 对偶块模型求解失败，状态: {model.status}", flush=True)
            return None, None, None
    
    def iter_with_theta_zeta_neural_network(self, union_analysis=None, num_epochs=1):
        """使用神经网络更新theta和zeta"""
        if not TORCH_AVAILABLE or self.theta_net is None or self.zeta_net is None:
            print("警告: 神经网络不可用，跳过theta/zeta更新", flush=True)
            return self.theta_values, self.zeta_values
        
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        self.theta_net.train()
        self.zeta_net.train()
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            
            for sample_id in range(self.n_samples):
                features = self._extract_features(sample_id)
                features_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)
                if self.device:
                    features_tensor = features_tensor.to(self.device)
                
                theta_output = self.theta_net(features_tensor)
                zeta_output = self.zeta_net(features_tensor)
                
                self.optimizer.zero_grad()
                
                differentiable_loss = self.loss_function_differentiable(
                    sample_id, theta_output[0], zeta_output[0], union_analysis, device=self.device
                )
                
                differentiable_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.zeta_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_total_loss += differentiable_loss.detach().cpu().item()
            
            if epoch == 0 or epoch == num_epochs - 1:
                avg_loss = epoch_total_loss / self.n_samples
                print(f"[NN-theta/zeta] epoch {epoch+1}/{num_epochs}, avg_loss = {avg_loss:.6f}", flush=True)
        
        # 更新theta和zeta值
        sample_id = 0
        features = self._extract_features(sample_id)
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)
        if self.device:
            features_tensor = features_tensor.to(self.device)
        
        self.theta_net.eval()
        self.zeta_net.eval()
        with torch.no_grad():
            theta_output = self.theta_net(features_tensor)
            zeta_output = self.zeta_net(features_tensor)
        
        return self._tensor_to_theta_dict(theta_output[0]), self._tensor_to_zeta_dict(zeta_output[0])
    
    def iter(self, max_iter=20, union_analysis=None):
        """主迭代循环"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        EPS = 1e-10
        
        for i in range(max_iter):
            print(f"🔄 迭代 {i+1}/{max_iter} 开始", flush=True)
            self.iter_number = i
            
            # 1. 迭代PG块
            for sample_id in range(self.n_samples):
                pg_sol, x_sol, cpower_sol, coc_sol = self.iter_with_pg_block(
                    sample_id=sample_id, theta_values=self.theta_values,
                    zeta_values=self.zeta_values, union_analysis=union_analysis
                )
                if pg_sol is None:
                    print("❌ PG块迭代失败，终止迭代", flush=True)
                    break
                
                pg_sol = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                x_sol = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                x_sol = np.where(np.abs(x_sol - 1) < EPS, 1, x_sol)
                
                self.pg[sample_id, :, :] = pg_sol
                self.x[sample_id, :, :] = x_sol
                self.cpower[sample_id, :, :] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
                self.coc[sample_id, :, :] = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)
            
            # 2. 迭代对偶块
            for sample_id in range(self.n_samples):
                lambda_sol, mu_sol, ita_sol = self.iter_with_dual_block(
                    sample_id=sample_id, theta_values=self.theta_values,
                    zeta_values=self.zeta_values, union_analysis=union_analysis
                )
                if lambda_sol is None or mu_sol is None:
                    print("❌ 对偶块迭代失败，终止迭代", flush=True)
                    break
                
                self.lambda_[sample_id] = lambda_sol
                self.mu[sample_id, :, :] = np.where(np.abs(mu_sol) < EPS, 0, mu_sol)
                self.ita[sample_id, :, :] = np.where(np.abs(ita_sol) < EPS, 0, ita_sol)
            
            # 3. 使用神经网络更新theta和zeta
            theta_values_new, zeta_values_new = self.iter_with_theta_zeta_neural_network(
                union_analysis=union_analysis, num_epochs=10
            )
            if theta_values_new is None or zeta_values_new is None:
                print("❌ Theta/Zeta神经网络更新失败，终止迭代", flush=True)
                break
            self.theta_values = theta_values_new
            self.zeta_values = zeta_values_new
            
            print(f"✅ 迭代 {i+1}/{max_iter} 成功", flush=True)
            
            # 计算违反量
            obj_primal, obj_dual, obj_opt = self.cal_viol(union_analysis=union_analysis)
            obj_primal = obj_primal if abs(obj_primal) >= 1e-12 else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= 1e-12 else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= 1e-12 else 0.0
            
            print(f'obj_primal:{obj_primal}, obj_dual:{obj_dual}, obj_opt:{obj_opt}', flush=True)
            self.rho_primal += self.gamma * obj_primal
            self.rho_dual += self.gamma * obj_dual
            self.rho_opt += self.gamma * obj_opt
            print(f"当前惩罚参数: ρ_primal={self.rho_primal}, ρ_dual={self.rho_dual}, ρ_opt={self.rho_opt}", flush=True)
            print("--------------------------------", flush=True)
            time.sleep(1)
        
        return self.theta_values, self.zeta_values
    
    # --------------------------------------------------------------------------
    # 约束违反量计算
    # --------------------------------------------------------------------------
    def cal_viol(self, union_analysis=None):
        """计算约束违反量"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        obj_primal, obj_opt, obj_dual = 0, 0, 0
        
        Ton, Toff = min(4, self.T), min(4, self.T)
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]
        PTDF_G = PTDF @ G
        
        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']
            pg = self.pg[sample_id, :, :]
            x = self.x[sample_id, :, :]
            
            # 功率平衡约束
            for t in range(self.T):
                power_balance_expr = np.sum(pg[:, t]) - np.sum(Pd[:, t])
                obj_primal += abs(power_balance_expr)
                if sample_id < len(self.lambda_) and 'lambda_power_balance' in self.lambda_[sample_id]:
                    obj_opt += abs(power_balance_expr) * abs(self.lambda_[sample_id]['lambda_power_balance'][t])
                
                for g in range(self.ng):
                    pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                    pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                    obj_primal += max(0, pg_lower_expr) + max(0, pg_upper_expr)
            
            # 参数化约束违反量
            if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis and self.theta_values:
                for constraint_info in union_analysis['union_constraints']:
                    branch_id = constraint_info['branch_id']
                    time_slot = constraint_info['time_slot']
                    
                    lhs_expr = 0.0
                    for coeff_info in constraint_info['nonzero_pg_coefficients']:
                        unit_id = coeff_info['unit_id']
                        theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                        theta = self.theta_values.get(theta_name, 0.0)
                        lhs_expr += theta * x[unit_id, time_slot]
                    
                    theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
                    theta_rhs = self.theta_values.get(theta_rhs_name, 1.0)
                    violation = max(0, lhs_expr - theta_rhs)
                    obj_primal += violation
            
            if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis and self.zeta_values:
                for constraint in union_analysis['union_zeta_constraints']:
                    unit_id = constraint['unit_id']
                    time_slot = constraint['time_slot']
                    
                    zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                    zeta = self.zeta_values.get(zeta_name, 0.0)
                    lhs_expr = zeta * x[unit_id, time_slot]
                    
                    zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                    zeta_rhs = self.zeta_values.get(zeta_rhs_name, 1.0)
                    violation = max(0, lhs_expr - zeta_rhs)
                    obj_primal += violation
            
            # 对偶约束的违反量
            if sample_id < len(self.lambda_):
                for g in range(self.ng):
                    for t in range(self.T):
                        dual_expr = self.gencost[g, -2] / self.T_delta * self.lambda_[sample_id]['lambda_cpower'][g, t]
                        dual_expr -= self.lambda_[sample_id]['lambda_power_balance'][t]
                        dual_expr -= self.lambda_[sample_id]['lambda_pg_lower'][g, t]
                        dual_expr += self.lambda_[sample_id]['lambda_pg_upper'][g, t]
                        
                        if t > 0:
                            dual_expr += self.lambda_[sample_id]['lambda_ramp_up'][g, t-1]
                            dual_expr -= self.lambda_[sample_id]['lambda_ramp_down'][g, t-1]
                        if t < self.T - 1:
                            dual_expr -= self.lambda_[sample_id]['lambda_ramp_up'][g, t]
                            dual_expr += self.lambda_[sample_id]['lambda_ramp_down'][g, t]

                        ptdfg_col = PTDF_G[:, g]
                        for l in range(self.nl):
                            pg_coeff = ptdfg_col[l]
                            dual_expr += pg_coeff * (
                                self.lambda_[sample_id]['lambda_dcpf_upper'][l, t]
                                - self.lambda_[sample_id]['lambda_dcpf_lower'][l, t]
                            )
                        obj_dual += abs(dual_expr)
        
        return obj_primal, obj_dual, obj_opt
    
    # --------------------------------------------------------------------------
    # 可微分Loss函数
    # --------------------------------------------------------------------------
    def loss_function_differentiable(self, sample_id, theta_tensor, zeta_tensor, union_analysis=None, device=None):
        """可微分的loss函数"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装，无法使用可微分loss函数")
        
        if device is None:
            device = self.device if hasattr(self, 'device') else torch.device('cpu')
        
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        x = torch.tensor(self.x[sample_id], dtype=torch.float32, device=device)
        lambda_dict = self.lambda_[sample_id]
        
        gen_PMIN = torch.tensor(self.gen[:, PMIN], dtype=torch.float32, device=device)
        gen_PMAX = torch.tensor(self.gen[:, PMAX], dtype=torch.float32, device=device)
        gencost_fixed = torch.tensor(self.gencost[:, -1] / self.T_delta, dtype=torch.float32, device=device)
        
        Ton, Toff = min(4, self.T), min(4, self.T)
        Ru_co = torch.tensor(0.3 * self.gen[:, PMAX], dtype=torch.float32, device=device)
        Rd_co = torch.tensor(0.3 * self.gen[:, PMAX], dtype=torch.float32, device=device)
        Ru = torch.tensor(0.4 * self.gen[:, PMAX] / self.T_delta, dtype=torch.float32, device=device)
        Rd = torch.tensor(0.4 * self.gen[:, PMAX] / self.T_delta, dtype=torch.float32, device=device)
        start_cost = torch.tensor(self.gencost[:, 1], dtype=torch.float32, device=device)
        shut_cost = torch.tensor(self.gencost[:, 2], dtype=torch.float32, device=device)
        
        lambda_cpower = torch.tensor(np.array(lambda_dict.get('lambda_cpower', np.zeros((self.ng, self.T)))), dtype=torch.float32, device=device)
        lambda_pg_lower = torch.tensor(np.array(lambda_dict.get('lambda_pg_lower', np.zeros((self.ng, self.T)))), dtype=torch.float32, device=device)
        lambda_pg_upper = torch.tensor(np.array(lambda_dict.get('lambda_pg_upper', np.zeros((self.ng, self.T)))), dtype=torch.float32, device=device)
        lambda_ramp_down = torch.tensor(np.array(lambda_dict.get('lambda_ramp_down', np.zeros((self.ng, self.T-1)))), dtype=torch.float32, device=device)
        lambda_ramp_up = torch.tensor(np.array(lambda_dict.get('lambda_ramp_up', np.zeros((self.ng, self.T-1)))), dtype=torch.float32, device=device)
        lambda_min_on = torch.tensor(np.array(lambda_dict.get('lambda_min_on', np.zeros((self.ng, Ton, self.T)))), dtype=torch.float32, device=device)
        lambda_min_off = torch.tensor(np.array(lambda_dict.get('lambda_min_off', np.zeros((self.ng, Toff, self.T)))), dtype=torch.float32, device=device)
        lambda_start_cost = torch.tensor(np.array(lambda_dict.get('lambda_start_cost', np.zeros((self.ng, self.T-1)))), dtype=torch.float32, device=device)
        lambda_shut_cost = torch.tensor(np.array(lambda_dict.get('lambda_shut_cost', np.zeros((self.ng, self.T-1)))), dtype=torch.float32, device=device)
        lambda_x_upper = torch.tensor(np.array(lambda_dict.get('lambda_x_upper', np.zeros((self.ng, self.T)))), dtype=torch.float32, device=device)
        lambda_x_lower = torch.tensor(np.array(lambda_dict.get('lambda_x_lower', np.zeros((self.ng, self.T)))), dtype=torch.float32, device=device)
        
        if not hasattr(self, '_theta_name_to_idx'):
            self._theta_name_to_idx = {name: idx for idx, name in enumerate(self.theta_var_names)}
        if not hasattr(self, '_zeta_name_to_idx'):
            self._zeta_name_to_idx = {name: idx for idx, name in enumerate(self.zeta_var_names)}
        
        mu = torch.tensor(self.mu[sample_id], dtype=torch.float32, device=device)
        ita = torch.tensor(self.ita[sample_id], dtype=torch.float32, device=device)
        
        # x_LP: 用于约束违反惩罚项计算
        x_LP = torch.tensor(self.x[sample_id], dtype=torch.float32, device=device)
        lambda_weight = getattr(self, 'constraint_violation_weight', 0)
        epsilon = getattr(self, 'constraint_violation_epsilon', 1e-3)
        
        obj_primal = torch.tensor(0.0, device=device, requires_grad=True)
        obj_opt = torch.tensor(0.0, device=device, requires_grad=True)
        obj_dual = torch.tensor(0.0, device=device, requires_grad=True)
        obj_constraint_violation = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 计算theta相关的参数化约束损失
        if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                
                lhs_expr = torch.tensor(0.0, device=device)
                for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                    unit_id = coeff_info['unit_id']
                    theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                    theta_idx = self._theta_name_to_idx.get(theta_name, -1)
                    if theta_idx >= 0:
                        theta = theta_tensor[theta_idx]
                        lhs_expr = lhs_expr + theta * x[unit_id, time_slot]
                
                theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
                theta_rhs_idx = self._theta_name_to_idx.get(theta_rhs_name, -1)
                theta_rhs = theta_tensor[theta_rhs_idx] if theta_rhs_idx >= 0 else torch.tensor(1.0, device=device)
                
                violation = lhs_expr - theta_rhs
                obj_primal = obj_primal + torch.relu(violation)
                
                if branch_id < self.nl:
                    obj_opt = obj_opt + torch.abs(violation) * mu[branch_id, time_slot]
                else:
                    default_mu = torch.tensor(self.dual_para_bound, device=device)
                    obj_opt = obj_opt + torch.abs(violation) * default_mu
                
                # 约束违反惩罚项
                lhs_expr_x_LP = torch.tensor(0.0, device=device)
                for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                    unit_id = coeff_info['unit_id']
                    theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                    theta_idx = self._theta_name_to_idx.get(theta_name, -1)
                    if theta_idx >= 0:
                        theta = theta_tensor[theta_idx]
                        lhs_expr_x_LP = lhs_expr_x_LP + theta * x_LP[unit_id, time_slot]
                
                d_value = theta_tensor[theta_rhs_idx] if theta_rhs_idx >= 0 else torch.tensor(1.0, device=device)
                constraint_violation_val = torch.relu(d_value + epsilon - lhs_expr_x_LP)
                obj_constraint_violation = obj_constraint_violation + lambda_weight * constraint_violation_val
        
        # 计算zeta相关的参数化约束损失
        if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
            for constraint in union_analysis['union_zeta_constraints']:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                
                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta_idx = self._zeta_name_to_idx.get(zeta_name, -1)
                if zeta_idx >= 0:
                    zeta = zeta_tensor[zeta_idx]
                    lhs_expr = zeta * x[unit_id, time_slot]
                    
                    zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                    zeta_rhs_idx = self._zeta_name_to_idx.get(zeta_rhs_name, -1)
                    zeta_rhs = zeta_tensor[zeta_rhs_idx] if zeta_rhs_idx >= 0 else torch.tensor(1.0, device=device)
                    
                    violation = lhs_expr - zeta_rhs
                    obj_primal = obj_primal + torch.relu(violation)
                    obj_opt = obj_opt + torch.abs(violation) * ita[unit_id, time_slot]
                    
                    # 约束违反惩罚项
                    lhs_expr_x_LP = zeta * x_LP[unit_id, time_slot]
                    d_value = zeta_tensor[zeta_rhs_idx] if zeta_rhs_idx >= 0 else torch.tensor(1.0, device=device)
                    constraint_violation_val = torch.relu(d_value + epsilon - lhs_expr_x_LP)
                    obj_constraint_violation = obj_constraint_violation + lambda_weight * constraint_violation_val
        
        # 计算对偶约束损失
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = gencost_fixed[g] * lambda_cpower[g, t]
                dual_expr = dual_expr + lambda_x_upper[g, t] - lambda_x_lower[g, t]
                dual_expr = dual_expr + gen_PMIN[g] * lambda_pg_lower[g, t]
                dual_expr = dual_expr - gen_PMAX[g] * lambda_pg_upper[g, t]
                                
                if t > 0:
                    dual_expr = dual_expr + (Rd_co[g] - Rd[g]) * lambda_ramp_down[g, t-1]
                if t < self.T - 1:
                    dual_expr = dual_expr + (Ru_co[g] - Ru[g]) * lambda_ramp_up[g, t]
                
                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr = dual_expr + lambda_min_on[g, tau-1, t1]
                        if t == t1:
                            dual_expr = dual_expr - lambda_min_on[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr = dual_expr - lambda_min_on[g, tau-1, t1]
                
                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr = dual_expr - lambda_min_off[g, tau-1, t1]
                        if t == t1:
                            dual_expr = dual_expr + lambda_min_off[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr = dual_expr + lambda_min_off[g, tau-1, t1]
                
                if t > 0:
                    dual_expr = dual_expr + start_cost[g] * lambda_start_cost[g, t-1]
                    dual_expr = dual_expr - shut_cost[g] * lambda_shut_cost[g, t-1]
                if t < self.T - 1:
                    dual_expr = dual_expr - start_cost[g] * lambda_start_cost[g, t]
                    dual_expr = dual_expr + shut_cost[g] * lambda_shut_cost[g, t]
                
                # 参数化约束的对偶贡献
                if self.enable_theta_constraints and union_analysis and 'union_constraints' in union_analysis:
                    for constraint_info in union_analysis['union_constraints']:
                        branch_id = constraint_info['branch_id']
                        time_slot = constraint_info['time_slot']
                        if time_slot != t:
                            continue
                        
                        for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                            unit_id = coeff_info['unit_id']
                            if unit_id != g:
                                continue
                            
                            theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                            theta_idx = self._theta_name_to_idx.get(theta_name, -1)
                            if theta_idx >= 0:
                                theta = theta_tensor[theta_idx]
                                if branch_id < self.nl:
                                    dual_expr = dual_expr + theta * mu[branch_id, time_slot]
                                else:
                                    default_mu = torch.tensor(self.dual_para_bound, device=device)
                                    dual_expr = dual_expr + theta * default_mu
                
                if self.enable_zeta_constraints and union_analysis and 'union_zeta_constraints' in union_analysis:
                    for constraint in union_analysis['union_zeta_constraints']:
                        constraint_unit_id = constraint['unit_id']
                        time_slot_c = constraint['time_slot']
                        
                        if time_slot_c != t or constraint_unit_id != g:
                            continue
                        
                        zeta_name = f'zeta_unit_{constraint_unit_id}_time_{time_slot_c}'
                        zeta_idx = self._zeta_name_to_idx.get(zeta_name, -1)
                        if zeta_idx >= 0:
                            zeta = zeta_tensor[zeta_idx]
                            dual_expr = dual_expr + zeta * ita[constraint_unit_id, time_slot_c]
                
                obj_dual = obj_dual + torch.abs(dual_expr)
        
        # Fischer-Burmeister函数处理互补松弛条件
        obj_complementary = torch.tensor(0.0, device=device, requires_grad=True)
        if self.use_fischer_burmeister_for_loss:
            eps_fb = 1e-10
            
            def fischer_burmeister(a, b):
                return a + b - torch.sqrt(a**2 + b**2 + eps_fb)
            
            for g in range(self.ng):
                for t in range(self.T):
                    x_val = x[g, t]
                    g_upper = x_val - 1.0
                    fb_upper = fischer_burmeister(lambda_x_upper[g, t], -g_upper)
                    obj_complementary = obj_complementary + fb_upper ** 2
                    
                    g_lower = -x_val
                    fb_lower = fischer_burmeister(lambda_x_lower[g, t], -g_lower)
                    obj_complementary = obj_complementary + fb_lower ** 2
            
            for g in range(self.ng):
                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        g_min_on = x[g, t1+1] - x[g, t1] - x[g, t1+tau]
                        fb_min_on = fischer_burmeister(lambda_min_on[g, tau-1, t1], -g_min_on)
                        obj_complementary = obj_complementary + fb_min_on ** 2
            
            for g in range(self.ng):
                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        g_min_off = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+tau])
                        fb_min_off = fischer_burmeister(lambda_min_off[g, tau-1, t1], -g_min_off)
                        obj_complementary = obj_complementary + fb_min_off ** 2
            
            for g in range(self.ng):
                for t in range(1, self.T):
                    g_start = (x[g, t] - x[g, t-1]) - 1.0
                    fb_start = fischer_burmeister(lambda_start_cost[g, t-1], -g_start)
                    obj_complementary = obj_complementary + fb_start ** 2
                    
                    g_shut = (x[g, t-1] - x[g, t]) - 1.0
                    fb_shut = fischer_burmeister(lambda_shut_cost[g, t-1], -g_shut)
                    obj_complementary = obj_complementary + fb_shut ** 2
        
        loss = obj_primal * self.rho_primal + obj_dual * self.rho_dual + obj_opt * self.rho_opt + obj_constraint_violation
        return loss
    
    # --------------------------------------------------------------------------
    # 参数化约束辅助方法
    # --------------------------------------------------------------------------
    def _add_parametric_penalties_pg_block(self, model, x, sample_id, theta_values=None, union_analysis=None, PTDF=None, branch_limit=None):
        """添加包含theta参数的DCPF罚项"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return model, gp.LinExpr(), gp.LinExpr()
        
        if theta_values is None:
            theta_values = {}
        
        try:
            obj_primal, obj_opt = gp.LinExpr(), gp.LinExpr()
            
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                
                lhs_expr = 0
                for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                    unit_id = coeff_info['unit_id']
                    theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                    theta = theta_values.get(theta_name, 0.0)
                    lhs_expr += theta * x[unit_id, time_slot]
                
                theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
                parametric_rhs = theta_values.get(theta_rhs_name, 1.0)

                parametric_rhs_viol = model.addVar(lb=0, name=f'parametric_rhs_viol_{branch_id}_{time_slot}')
                parametric_rhs_abs = model.addVar(lb=0, name=f'parametric_rhs_abs_{branch_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol >= lhs_expr - parametric_rhs)
                model.addConstr(parametric_rhs_abs >= lhs_expr - parametric_rhs)
                model.addConstr(parametric_rhs_abs >= -lhs_expr + parametric_rhs)

                obj_primal += parametric_rhs_viol

                if hasattr(self, 'mu') and sample_id < len(self.mu) and branch_id < self.nl:
                    mu_val = self.mu[sample_id, branch_id, time_slot]
                    obj_opt += parametric_rhs_abs * abs(mu_val)
                else:
                    obj_opt += parametric_rhs_abs * self.dual_para_bound

            model.update()
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return model, gp.LinExpr(), gp.LinExpr()
    
    def _add_parametric_balance_power_penalties_pg_block(self, model, x, sample_id, zeta_values=None, union_analysis=None):
        """添加包含zeta参数的平衡功率罚项"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return model, gp.LinExpr(), gp.LinExpr()
        
        if zeta_values is None:
            zeta_values = {}
        
        try:
            obj_primal, obj_opt = gp.LinExpr(), gp.LinExpr()
            
            for constraint in union_analysis['union_zeta_constraints']:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']

                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)
                lhs_expr = zeta * x[unit_id, time_slot]

                zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                parametric_rhs = zeta_values.get(zeta_rhs_name, 1.0)

                parametric_rhs_viol = model.addVar(lb=0, name=f'parametric_balance_power_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVar(lb=0, name=f'parametric_balance_power_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol >= lhs_expr - parametric_rhs)
                model.addConstr(parametric_rhs_abs >= lhs_expr - parametric_rhs)
                model.addConstr(parametric_rhs_abs >= -lhs_expr + parametric_rhs)
                
                obj_primal += parametric_rhs_viol

                if hasattr(self, 'ita') and sample_id < len(self.ita):
                    ita_val = self.ita[sample_id, unit_id, time_slot]
                    obj_opt += parametric_rhs_abs * abs(ita_val)
                else:
                    obj_opt += parametric_rhs_abs * self.dual_para_bound

            model.update()
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return model, gp.LinExpr(), gp.LinExpr()
    
    def _add_parametric_constraints_dual_block_const_to_model(self, model, g_id, t_id, mu, sample_id, theta_values, union_analysis):
        """添加theta参数化约束的对偶贡献"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return 0.0
        
        if theta_values is None:
            theta_values = {}
        
        try:
            dual_expr = 0.0
            
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                if time_slot != t_id:
                    continue

                for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                    unit_id = coeff_info['unit_id']
                    if unit_id != g_id:
                        continue
                    
                    theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                    theta = theta_values.get(theta_name, 0.0)
                    
                    if branch_id < self.nl:
                        dual_expr += theta * mu[branch_id, time_slot]
                    else:
                        dual_expr += theta * self.dual_para_bound

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return 0.0
    
    def _add_parametric_balance_power_constraints_dual_block_const_to_model(self, model, g_id, t_id, ita, sample_id, zeta_values, union_analysis):
        """添加zeta参数化约束的对偶贡献"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return 0.0

        if zeta_values is None:
            zeta_values = {}
        
        try:
            dual_expr = 0
            
            for constraint in union_analysis['union_zeta_constraints']:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                
                if time_slot != t_id or unit_id != g_id:
                    continue

                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)
                dual_expr += zeta * ita[unit_id, time_slot]

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return 0.0
    
    def _add_parametric_obj_dual_block(self, model, x, mu, sample_id, theta_values=None, union_analysis=None, PTDF=None, branch_limit=None):
        """添加包含theta参数的obj_opt项"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not self.enable_theta_constraints or not union_analysis or 'union_constraints' not in union_analysis:
            return model, gp.LinExpr()
        
        if theta_values is None:
            theta_values = {}
        
        try:
            obj_opt = gp.LinExpr()
            
            for constraint_info in union_analysis['union_constraints']:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                
                lhs_expr = 0.0
                for coeff_info in constraint_info.get('nonzero_pg_coefficients', []):
                    unit_id = coeff_info['unit_id']
                    theta_name = f'theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}'
                    theta = theta_values.get(theta_name, 0.0)
                    lhs_expr += theta * x[unit_id, time_slot]
                
                theta_rhs_name = f'theta_rhs_branch_{branch_id}_time_{time_slot}'
                parametric_rhs = theta_values.get(theta_rhs_name, 1.0)

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                if parametric_rhs_abs > 1e-10:
                    if branch_id < self.nl:
                        obj_opt += parametric_rhs_abs * mu[branch_id, time_slot]
                    else:
                        obj_opt += parametric_rhs_abs * self.dual_para_bound

            model.update()
            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return model, gp.LinExpr()
    
    def _add_parametric_balance_power_obj_dual_block(self, model, x, ita, sample_id, zeta_values=None, union_analysis=None):
        """添加包含zeta参数的obj_opt项"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not self.enable_zeta_constraints or not union_analysis or 'union_zeta_constraints' not in union_analysis:
            return model, gp.LinExpr()
        
        if zeta_values is None:
            zeta_values = {}

        try:
            obj_opt = gp.LinExpr()
            
            for constraint in union_analysis['union_zeta_constraints']:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']

                zeta_name = f'zeta_unit_{unit_id}_time_{time_slot}'
                zeta = zeta_values.get(zeta_name, 0.0)
                lhs_expr = zeta * x[unit_id, time_slot]

                zeta_rhs_name = f'zeta_rhs_unit_{unit_id}_time_{time_slot}'
                parametric_rhs = zeta_values.get(zeta_rhs_name, 1.0)

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                if parametric_rhs_abs > 1e-10:
                    obj_opt += parametric_rhs_abs * ita[unit_id, time_slot]

            model.update()
            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}", flush=True)
            return model, gp.LinExpr()
    
    # --------------------------------------------------------------------------
    # LP求解方法
    # --------------------------------------------------------------------------
    def solve_LP_without_theta_constraints(self, sample_id, union_analysis=None):
        """求解不含theta约束的LP问题"""
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('LP_without_theta')
        model.Params.OutputFlag = 0
        
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        self._add_uc_constraints(model, pg, x, coc, cpower, Pd)
        
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
        return None
    
    def solve_LP_with_theta_constraints(self, sample_id, union_analysis=None):
        """求解含theta约束的LP问题"""
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('LP_with_theta')
        model.Params.OutputFlag = 0
        
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        self._add_uc_constraints(model, pg, x, coc, cpower, Pd)
        
        obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
              gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
        
        # 获取神经网络预测的theta和zeta
        obj_primal = None
        features = self._extract_features(sample_id)
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)
        if self.device:
            features_tensor = features_tensor.to(self.device)

        self.theta_net.eval()
        self.zeta_net.eval()

        with torch.no_grad():
            theta_out = self.theta_net(features_tensor)
            zeta_out = self.zeta_net(features_tensor)

        theta_arr = theta_out.detach().cpu().numpy().flatten()
        theta = {name: float(val) for name, val in zip(self.theta_var_names, theta_arr)}

        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                G[bus_idx, g] = 1
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]

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
        if obj_primal is not None:
            obj_primal += parametric_obj_primal
        
        if obj_primal is not None:
            penalty_weight = self.penalty_factor
            obj_model = obj + penalty_weight * obj_primal
        else:
            obj_model = obj
        
        model.setObjective(obj_model, GRB.MINIMIZE)
        model.Params.MIPGap = 1e-10
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
    
    def heuristic_sol_x_spec(self, sample_id, x_LP, x_LP_refined):
        """启发式解恢复"""
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
        """分析代理模型的总体性能"""
        differ_LP, differ_LP_refined = 0, 0
        differ_LP_heu, differ_LP_heu_refined = 0, 0
        differ_spec_heu = 0
        
        for sample_id in range(self.n_samples):
            sol_LP = self.solve_LP_without_theta_constraints(sample_id)
            if sol_LP is None:
                continue
            x_LP = sol_LP[1]
            
            sol_LP_refined = self.solve_LP_with_theta_constraints(sample_id, self._current_union_analysis)
            if sol_LP_refined is None:
                continue
            x_LP_refined = sol_LP_refined[1]
            
            unit_commitment_matrix = self.active_set_data[sample_id]['unit_commitment_matrix']
            if isinstance(unit_commitment_matrix, np.ndarray):
                if unit_commitment_matrix.ndim == 1:
                    if unit_commitment_matrix.size == self.ng * self.T:
                        x_int = unit_commitment_matrix.reshape(self.ng, self.T)
                    else:
                        continue
                elif unit_commitment_matrix.ndim == 2:
                    if unit_commitment_matrix.shape == (self.ng, self.T):
                        x_int = unit_commitment_matrix
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
            differ_LP += np.sum(np.abs(x_LP - x_int))
            differ_LP_refined += np.sum(np.abs(x_LP_refined - x_int))
            differ_LP_heu += np.sum(np.abs(np.round(x_LP).astype(int) - x_int))
            differ_LP_heu_refined += np.sum(np.abs(np.round(x_LP_refined).astype(int) - x_int))
            differ_spec_heu += np.sum(np.abs(self.heuristic_sol_x_spec(sample_id, x_LP, x_LP_refined) - x_int))
            
            if sample_id == 1:
                x_true = x_int
                x_round1 = np.round(x_LP).astype(int)
                x_round2 = np.round(x_LP_refined).astype(int)
                x_round3 = np.round(self.heuristic_sol_x_spec(sample_id, x_LP, x_LP_refined)).astype(int)
                self.plot_sample0_binary_comparison(x_true, x_round1, x_round2,
                                                    labels=("True optimum", "LP Optimized", "BCD-Neural Optimized"),
                                                    save_path="result/fig/sample0_binary_comparison.png",
                                                    show=True)
                
                # 求解ED问题并比较最优值
                if ED_GUROBI_AVAILABLE:
                    print("\n" + "="*60, flush=True)
                    print("Sample 1: ED问题求解结果比较", flush=True)
                    print("="*60, flush=True)
                    
                    Pd_sample = self.active_set_data[sample_id]['pd_data']
                    
                    # 求解x_true对应的ED问题
                    try:
                        ed_true = EconomicDispatchGurobi(self.ppc, Pd_sample, self.T_delta, x_true)
                        ed_true.model.Params.OutputFlag = 0
                        ed_true.model.optimize()
                        if ed_true.model.status == GRB.OPTIMAL:
                            obj_ed_true = ed_true.model.ObjVal
                            print(f"✓ x_true (真实最优解) 的ED最优值: {obj_ed_true:.6f}", flush=True)
                        else:
                            print(f"✗ x_true 的ED问题求解失败，状态: {ed_true.model.status}", flush=True)
                    except Exception as e:
                        print(f"✗ x_true 的ED问题求解出错: {e}", flush=True)
                    
                    # 求解x_round1对应的ED问题
                    try:
                        ed_round1 = EconomicDispatchGurobi(self.ppc, Pd_sample, self.T_delta, x_round1)
                        ed_round1.model.Params.OutputFlag = 0
                        ed_round1.model.optimize()
                        if ed_round1.model.status == GRB.OPTIMAL:
                            obj_ed_round1 = ed_round1.model.ObjVal
                            print(f"✓ x_round1 (LP Optimized) 的ED最优值: {obj_ed_round1:.6f}", flush=True)
                        else:
                            print(f"✗ x_round1 的ED问题求解失败，状态: {ed_round1.model.status}", flush=True)
                    except Exception as e:
                        print(f"✗ x_round1 的ED问题求解出错: {e}", flush=True)
                    
                    # 求解x_round2对应的ED问题
                    try:
                        ed_round2 = EconomicDispatchGurobi(self.ppc, Pd_sample, self.T_delta, x_round2)
                        ed_round2.model.Params.OutputFlag = 0
                        ed_round2.model.optimize()
                        if ed_round2.model.status == GRB.OPTIMAL:
                            obj_ed_round2 = ed_round2.model.ObjVal
                            print(f"✓ x_round2 (BCD-Neural Optimized) 的ED最优值: {obj_ed_round2:.6f}", flush=True)
                        else:
                            print(f"✗ x_round2 的ED问题求解失败，状态: {ed_round2.model.status}", flush=True)
                    except Exception as e:
                        print(f"✗ x_round2 的ED问题求解出错: {e}", flush=True)
                    
                    # 求解x_round3对应的ED问题
                    try:
                        ed_round3 = EconomicDispatchGurobi(self.ppc, Pd_sample, self.T_delta, x_round3)
                        ed_round3.model.Params.OutputFlag = 0
                        ed_round3.model.optimize()
                        if ed_round3.model.status == GRB.OPTIMAL:
                            obj_ed_round3 = ed_round3.model.ObjVal
                            print(f"✓ x_round3 (Heuristic Optimized) 的ED最优值: {obj_ed_round3:.6f}", flush=True)
                        else:
                            print(f"✗ x_round3 的ED问题求解失败，状态: {ed_round3.model.status}", flush=True)
                    except Exception as e:
                        print(f"✗ x_round3 的ED问题求解出错: {e}", flush=True)
        
        print(f"最优间隙（不含theta约束）: {differ_LP}", flush=True)
        print(f"最优间隙（含theta约束）: {differ_LP_refined}", flush=True)
        print(f"最优间隙（四舍五入不含theta约束）: {differ_LP_heu}", flush=True)
        print(f"最优间隙（四舍五入含theta约束）: {differ_LP_heu_refined}", flush=True)
        print(f"特殊解恢复恢复间隙: {differ_spec_heu}", flush=True)
    
    # --------------------------------------------------------------------------
    # 保存和加载方法
    # --------------------------------------------------------------------------
    def save_theta_values(self, filepath: str, ensure_dir: bool = True) -> None:
        """保存theta和zeta值到JSON文件"""
        if not hasattr(self, 'theta_values') or self.theta_values is None:
            raise RuntimeError("theta_values 未初始化，无法保存。")
        
        if ensure_dir:
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
        
        theta_serializable = {str(k): float(v) for k, v in self.theta_values.items()}
        zeta_serializable = {str(k): float(v) for k, v in self.zeta_values.items()} if hasattr(self, 'zeta_values') and self.zeta_values else {}
        
        data = {'theta_values': theta_serializable, 'zeta_values': zeta_serializable}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ theta_values 和 zeta_values 已保存到: {filepath}", flush=True)
    
    def load_theta_values(self, filepath: str) -> dict:
        """从JSON文件加载theta和zeta值"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"theta 文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tv = data.get('theta_values', {})
        zv = data.get('zeta_values', {})
        
        self.theta_values = {str(k): float(v) for k, v in tv.items()}
        if zv:
            self.zeta_values = {str(k): float(v) for k, v in zv.items()}
        
        print(f"✓ theta_values 和 zeta_values 已从文件加载: {filepath}", flush=True)
        return self.theta_values
    
    def save_model_parameters(self, filepath: str, ensure_dir: bool = True) -> None:
        """保存神经网络模型参数"""
        if not TORCH_AVAILABLE or self.theta_net is None or self.zeta_net is None:
            raise RuntimeError("神经网络未初始化，无法保存模型参数。")
        
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
            "device": str(self.device) if hasattr(self, "device") and self.device is not None else "cpu",
        }
        
        torch.save(state, filepath)
        print(f"✓ 模型参数已保存到: {filepath}", flush=True)
    
    def load_model_parameters(self, filepath: str, map_location: str = "cpu") -> None:
        """从文件加载神经网络模型参数"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装，无法加载模型参数。")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型参数文件不存在: {filepath}")
        
        state = torch.load(filepath, map_location=map_location)
        
        if self.theta_net is None or self.zeta_net is None:
            self._init_neural_network()
        
        self.theta_net.load_state_dict(state["theta_net_state_dict"])
        self.zeta_net.load_state_dict(state["zeta_net_state_dict"])
        
        if state.get("optimizer_state_dict") is not None and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        
        if state.get("theta_var_names") is not None:
            self.theta_var_names = state["theta_var_names"]
        if state.get("zeta_var_names") is not None:
            self.zeta_var_names = state["zeta_var_names"]
        
        print(f"✓ 模型参数已从文件加载: {filepath}", flush=True)

    # --------------------------------------------------------------------------
    # 可视化方法
    # --------------------------------------------------------------------------
    def plot_sample0_binary_comparison(self, x_true: np.ndarray, x_round1: np.ndarray, x_round2: np.ndarray,
                                        labels: tuple = ("Ground Truth", "Standard Rounding", "BCD-Neural Optimized"),
                                        save_path: str = "result/fig/binary_comparison_pro.pdf", show: bool = True) -> None:
        """绘制二进制变量比较图"""
        ng, T = self.ng, self.T
        m_true = np.array(x_true).reshape(ng, T)
        m_r1 = np.array(x_round1).reshape(ng, T)
        m_r2 = np.array(x_round2).reshape(ng, T)

        diff1 = np.abs(m_true - m_r1)
        diff2 = np.abs(m_true - m_r2)
        
        rate1 = (m_true == m_r1).mean() * 100
        rate2 = (m_true == m_r2).mean() * 100

        on_color = "#2c3e50"
        off_color = "#ecf0f1"
        cmap_main = ListedColormap([off_color, on_color])

        err_color = "#b33939"
        corr_color = "#f7f1e3"
        cmap_diff = ListedColormap([corr_color, err_color])

        plt.rcParams['font.family'] = 'serif'
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), 
                                gridspec_kw={'width_ratios': [1, 1], 'hspace': 0.4, 'wspace': 0.12})
        
        data_list = [m_true, m_r1, m_r2]
        diff_list = [None, diff1, diff2]
        titles = [f"A. {labels[0]}", f"B. {labels[1]}", f"C. {labels[2]}"]

        for i in range(3):
            ax_m = axes[i, 0]
            sns.heatmap(data_list[i], ax=ax_m, cmap=cmap_main, cbar=False, 
                        linewidths=0.1, linecolor='white', xticklabels=5, yticklabels=5)
            ax_m.set_title(titles[i], loc='left', fontsize=12, fontweight='bold')
            ax_m.set_ylabel("Unit Index")
            
            ax_e = axes[i, 1]
            if i == 0:
                ax_e.axis('off')
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
                acc_text = f"Accuracy: {rate1 if i==1 else rate2:.2f}%"
                ax_e.set_title(acc_text, loc='right', fontsize=10, 
                            color=err_color if i==1 else 'green', fontweight='bold')

        axes[2, 0].set_xlabel("Time Slots (h)")
        axes[2, 1].set_xlabel("Time Slots (h)")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 已保存图像: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    if not PYPOWER_AVAILABLE:
        print("错误: pypower未安装，无法运行测试代码", flush=True)
        exit(1)
    
    # 加载active_set数据
    json_file = "result/active_sets_20251221_161355.json"
    active_set_data = load_active_set_from_json(json_file)
    
    # 准备ppc数据
    case = 'case39'
    ppc = pypower.case39.case39()
    ppc['branch'][:, pypower.idx_brch.RATE_A] = ppc['branch'][:, pypower.idx_brch.RATE_A]
    ppc['gencost'][:, -2] = np.array([[0.3, 0.28, 0.33, 0.35, 0.2, 0.34, 0.22, 0.28, 0.32, 0.36]])
    T_delta = 1
    
    # 创建模型对象
    print("=" * 60, flush=True)
    print("初始化 Agent_NN 模型", flush=True)
    print("=" * 60, flush=True)
    agent = Agent_NN(ppc, active_set_data=active_set_data, T_delta=T_delta)
    
    # 可选：运行BCD迭代
    # print("\n" + "=" * 60, flush=True)
    # print("开始BCD迭代（结合神经网络更新）", flush=True)
    # print("=" * 60, flush=True)
    # agent.iter(max_iter=20)
    
    # 可选：加载已保存的模型参数
    # agent.load_model_parameters(f'result/net/model_parameters_final_20251222_193649_case39.pth')
    
    # 分析代理模型性能
    print("\n" + "=" * 60, flush=True)
    print("分析代理模型性能", flush=True)
    print("=" * 60, flush=True)
    agent.analyse_surrogate_model_totle()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'result/theta_zeta_values_final_{timestamp}.json'
    agent.save_theta_values(save_path)
    
    save_path = f'result/net/model_parameters_final_{timestamp}_{case}.pth'
    agent.save_model_parameters(save_path)
    
    print("\n" + "=" * 60, flush=True)
    print("测试完成！", flush=True)
    print("=" * 60, flush=True)
