from datetime import datetime
import json
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pypower
import pypower.case14
import pypower.case9
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
from pathlib import Path
import io
import sys
import re

from typing import Dict, List, Tuple, Optional

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_active_set_from_json(json_filepath: str, sample_id: Optional[int] = None) -> Dict:
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
        
        print(f"=== 加载活动集数据 (样本 {sample_id}) ===")
        print(f"活动约束数量: {len(active_constraints)}")
        print(f"活动变量数量: {len(active_variables)}")
        print(f"Pd数据形状: {pd_data.shape}")

        return {
            'sample_id': sample_id,
            'active_constraints': active_constraints,
            'active_variables': active_variables,
            'pd_data': pd_data,
            'unit_commitment_matrix': unit_commitment,
            'single_sample': True
        }
    else:
        # 加载所有样本
        all_samples_data = reader.load_all_samples()
        
        print(f"=== 加载所有活动集数据 ===")
        print(f"总样本数量: {len(all_samples_data)}")
        
        return all_samples_data

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
            print(f"样本ID {sample_id} 超出范围 [0, {len(samples)-1}]")
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
        
        print(f"开始加载 {total_samples} 个样本的数据...")
        
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
                
                all_samples_data.append(sample_data)
                
                if (sample_id + 1) % 10 == 0:
                    print(f"已加载 {sample_id + 1}/{total_samples} 个样本")
                    
            except Exception as e:
                print(f"加载样本 {sample_id} 时出错: {e}")
                # 添加空数据以保持索引一致性
                all_samples_data.append({
                    'sample_id': sample_id,
                    'active_constraints': [],
                    'active_variables': [],
                    'pd_data': np.array([]),
                    'unit_commitment_matrix': np.array([]),
                    'error': str(e)
                })
        
        print(f"✓ 完成加载所有样本数据")
        return all_samples_data
    
    def get_samples_summary(self) -> Dict:
        """
        获取所有样本的统计摘要
        
        Returns:
            包含统计信息的字典
        """
        all_samples = self.load_all_samples()
        
        summary = {
            'total_samples': len(all_samples),
            'valid_samples': 0,
            'error_samples': 0,
            'constraint_stats': {'min': float('inf'), 'max': 0, 'avg': 0},
            'variable_stats': {'min': float('inf'), 'max': 0, 'avg': 0},
            'pd_data_shapes': set()
        }
        
        constraint_counts = []
        variable_counts = []
        
        for sample in all_samples:
            if 'error' in sample:
                summary['error_samples'] += 1
                continue
                
            summary['valid_samples'] += 1
            
            constraint_count = len(sample['active_constraints'])
            variable_count = len(sample['active_variables'])
            
            constraint_counts.append(constraint_count)
            variable_counts.append(variable_count)
            
            if hasattr(sample['pd_data'], 'shape'):
                summary['pd_data_shapes'].add(sample['pd_data'].shape)
        
        if constraint_counts:
            summary['constraint_stats'] = {
                'min': min(constraint_counts),
                'max': max(constraint_counts),
                'avg': sum(constraint_counts) / len(constraint_counts)
            }
        
        if variable_counts:
            summary['variable_stats'] = {
                'min': min(variable_counts),
                'max': max(variable_counts),
                'avg': sum(variable_counts) / len(variable_counts)
            }
        
        summary['pd_data_shapes'] = list(summary['pd_data_shapes'])
        
        return summary

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
        pd_data = np.array(sample['pd_data'])
        
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

def load_specific_samples(json_filepath: str, sample_ids: List[int]) -> List[Dict]:
    """
    加载指定的多个样本
    
    Args:
        json_filepath: JSON文件路径
        sample_ids: 要加载的样本ID列表
        
    Returns:
        包含指定样本数据的列表
    """
    reader = ActiveSetReader(json_filepath)
    samples_data = []
    
    print(f"开始加载指定的 {len(sample_ids)} 个样本...")
    
    for sample_id in sample_ids:
        try:
            active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
            unit_commitment = reader.get_unit_commitment_matrix(sample_id)
            
            sample_data = {
                'sample_id': sample_id,
                'active_constraints': active_constraints,
                'active_variables': active_variables,
                'pd_data': pd_data,
                'unit_commitment_matrix': unit_commitment
            }
            
            samples_data.append(sample_data)
            
        except Exception as e:
            print(f"加载样本 {sample_id} 时出错: {e}")
            samples_data.append({
                'sample_id': sample_id,
                'error': str(e)
            })
    
    print(f"✓ 完成加载指定样本数据")
    return samples_data

class Iter_BCD:
    """
    迭代BCD类，用于处理参数化约束和模型更新。
    """
    
    def __init__(self, ppc, active_set_data, T_delta, union_analysis=None):
        self.ppc = ppc
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        
        self.rho_primal = 1e-3
        self.rho_dual = 1e-1
        self.rho_opt = 1e-3
        
        self.gamma = 1e-7
        
        # 处理单个样本或多个样本的情况
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
            
        self.ng = self.gen.shape[0]
        self.nl = self.branch.shape[0]
        
        self.active_set_data = active_set_data
        
        # 初始化theta变量字典
        self.theta_vars = {}
        self.zeta_vars = {}

        # 初始化求解
        self.pg, self.x, self.coc, self.cpower, self.lambda_ = self.initialize_solve()
        
        # 如果没有提供union_analysis，则基于x_init创建
        if union_analysis is None:
            self._current_union_analysis = self._create_union_analysis_from_x_init(self.x, self.lambda_)
            # 创建theta变量
            self.add_theta_variables_for_branches(self._current_union_analysis)
        elif union_analysis:
            self.add_theta_variables_for_branches(union_analysis)

        self.theta_values, self.mu = self.initialize_theta_values(self._current_union_analysis)

        self.zeta_values, self.ita_lower, self.ita_upper = self.initialize_zeta_values(self._current_union_analysis)

        # result = self.iter_with_pg_block(sample_id=0, theta_values=self.theta_values, union_analysis=self._current_union_analysis)
        
        # result = self.iter_with_dual_block(sample_id=0, theta_values=self.theta_values, union_analysis=self._current_union_analysis)
        
        # result = self.iter_with_theta_block(union_analysis=self._current_union_analysis)

        pass
    
    def initialize_solve(self):
        pg_sol = []
        x_sol = []
        coc_sol = []
        cpower_sol = []
        lambda_sol = []
        for sample_id in range(self.n_samples):
            model = gp.Model('UnitCommitment')
            model.Params.OutputFlag = 0
            pg = model.addVars(self.ng, self.T, lb=0, name='pg')
            x = model.addVars(self.ng, self.T, lb=0, name='x')        
            coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
            cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            Pd = self.active_set_data[sample_id]['pd_data']
            
            for t in range(self.T):
                model.addConstr(
                    gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]), 
                    name=f'power_balance_{t}'
                )
                for g in range(self.ng):
                    model.addConstr(
                        pg[g, t] >= self.gen[g, PMIN] * x[g, t],
                        name=f'pg_lower_{g}_{t}'
                    )
                    model.addConstr(
                        pg[g, t] <= self.gen[g, PMAX] * x[g, t],
                        name=f'pg_upper_{g}_{t}'
                    )
            
            # 爬坡约束
            Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
            Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
            Ru_co = 0.3 * self.gen[:, PMAX]
            Rd_co = 0.3 * self.gen[:, PMAX]
            for t in range(1, self.T):
                for g in range(self.ng):
                    model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1]), name=f'ramp_up_{g}_{t}')
                    model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t]), name=f'ramp_down_{g}_{t}')
            # 最小开机时间和最小关机时间约束
            # 最小开机时间约束（与matlab一致）
            for g in range(self.ng):
                for t in range(1, Ton+1):
                    for t1 in range(self.T - t):
                        model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+t],
                        name=f'min_on_{g}_{t}_{t1}')
            # 最小关机时间约束（与matlab一致）
            for g in range(self.ng):
                for t in range(1, Toff+1):
                    for t1 in range(self.T - t):
                        model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+t],
                        name=f'min_off_{g}_{t}_{t1}')
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
            # 潮流约束
            # G: 机组-节点映射矩阵，需用户根据数据准备
            # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            # 计算PTDF
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]  # 线路容量
            for t in range(self.T):
                flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
                for l in range(self.branch.shape[0]):
                    model.addConstr(flow[l] <= branch_limit[l],
                    name=f'flow_upper_{l}_{t}')
                    model.addConstr(flow[l] >= -branch_limit[l],
                    name=f'flow_lower_{l}_{t}')
            
            for t in range(self.T):
                for g in range(self.ng):
                    model.addConstr(x[g, t] <= 1, name=f'x_upper_{g}_{t}')
                    model.addConstr(x[g, t] >= 0, name=f'x_lower_{g}_{t}')
                    
            primal_obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                    gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
           
            model.setObjective(primal_obj, GRB.MINIMIZE)
            
            model.setParam("Presolve", 2)

            model.Params.OutputFlag = 0  # 禁用日志输出
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                pg_sol_sample = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
                x_sol_sample = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
                lambda_sol_sample = self.extract_dual_variables_as_arrays(model)
                coc_sol_sample = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])
                cpower_sol_sample = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])

                pg_sol.append(pg_sol_sample)
                x_sol.append(x_sol_sample)
                lambda_sol.append(lambda_sol_sample)
                coc_sol.append(coc_sol_sample)
                cpower_sol.append(cpower_sol_sample)

        pg_sol = np.array(pg_sol)
        x_sol = np.array(x_sol)
        
        lambda_sol = np.array(lambda_sol)
        coc_sol = np.array(coc_sol)
        cpower_sol = np.array(cpower_sol)

        return pg_sol, x_sol, coc_sol, cpower_sol, lambda_sol

    def extract_dual_variables(self, model):
        """
        通过约束名称提取对偶变量并进行比较
        
        Args:
            model: 求解后的Gurobi模型
            explicit_duals: 显式对偶变量字典
            
        Returns:
            Dict: 包含比较结果的字典
        """
        # print("\n=== 基于约束名称的对偶变量提取 ===")
        
        implicit_duals = {}
        
        try:
            # 1. 功率平衡约束的对偶变量
            # print("1. 提取功率平衡约束的对偶变量...")
            implicit_duals['power_balance'] = {}
            for t in range(self.T):
                constr = model.getConstrByName(f'power_balance_{t}')
                if constr is not None:
                    implicit_duals['power_balance'][t] = constr.Pi
            
            # 2. 发电上下限约束的对偶变量
            # print("2. 提取发电上下限约束的对偶变量...")
            implicit_duals['pg_lower'] = {}
            implicit_duals['pg_upper'] = {}
            for g in range(self.ng):
                implicit_duals['pg_lower'][g] = {}
                implicit_duals['pg_upper'][g] = {}
                for t in range(self.T):
                    # 下限约束
                    constr_lower = model.getConstrByName(f'pg_lower_{g}_{t}')
                    if constr_lower is not None:
                        implicit_duals['pg_lower'][g][t] = constr_lower.Pi
                    
                    # 上限约束
                    constr_upper = model.getConstrByName(f'pg_upper_{g}_{t}')
                    if constr_upper is not None:
                        implicit_duals['pg_upper'][g][t] = constr_upper.Pi
            
            # 3. 爬坡约束的对偶变量
            # print("3. 提取爬坡约束的对偶变量...")
            implicit_duals['ramp_up'] = {}
            implicit_duals['ramp_down'] = {}
            for g in range(self.ng):
                implicit_duals['ramp_up'][g] = {}
                implicit_duals['ramp_down'][g] = {}
                for t in range(1, self.T):
                    # 上爬坡约束
                    constr_ramp_up = model.getConstrByName(f'ramp_up_{g}_{t}')
                    if constr_ramp_up is not None:
                        implicit_duals['ramp_up'][g][t-1] = constr_ramp_up.Pi
                    
                    # 下爬坡约束
                    constr_ramp_down = model.getConstrByName(f'ramp_down_{g}_{t}')
                    if constr_ramp_down is not None:
                        implicit_duals['ramp_down'][g][t-1] = constr_ramp_down.Pi

            # 4. 最小开机/关机时间约束的对偶变量
            # print("4. 提取最小开机/关机时间约束的对偶变量...")
            implicit_duals['min_on'] = {}
            implicit_duals['min_off'] = {}
            
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            
            for g in range(self.ng):
                implicit_duals['min_on'][g] = {}
                implicit_duals['min_off'][g] = {}
                
                # 最小开机时间约束
                for tau in range(1, Ton+1):
                    for t1 in range(self.T - tau):
                        cname_on = f'min_on_{g}_{tau}_{t1}'
                        constr_on = model.getConstrByName(cname_on)
                        if constr_on is not None:
                            if tau not in implicit_duals['min_on'][g]:
                                implicit_duals['min_on'][g][tau] = {}
                            implicit_duals['min_on'][g][tau][t1] = constr_on.Pi
                
                # 最小关机时间约束
                for tau in range(1, Toff+1):
                    for t1 in range(self.T - tau):
                        cname_off = f'min_off_{g}_{tau}_{t1}'
                        constr_off = model.getConstrByName(cname_off)
                        if constr_off is not None:
                            if tau not in implicit_duals['min_off'][g]:
                                implicit_duals['min_off'][g][tau] = {}
                            implicit_duals['min_off'][g][tau][t1] = constr_off.Pi
            
            # 5. 启停成本约束的对偶变量
            # print("5. 提取启停成本约束的对偶变量...")
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
                        implicit_duals['start_cost'][g][t-1] = constr_start.Pi
                    
                    # 关机成本约束
                    constr_shut = model.getConstrByName(f'shut_cost_{g}_{t}')
                    if constr_shut is not None:
                        implicit_duals['shut_cost'][g][t-1] = constr_shut.Pi
                    
                    # 非负约束
                    constr_nonneg = model.getConstrByName(f'coc_nonneg_{g}_{t}')
                    if constr_nonneg is not None:
                        implicit_duals['coc_nonneg'][g][t-1] = constr_nonneg.Pi
            
            # 6. 发电成本约束的对偶变量（原来缺失的）
            # print("6. 提取发电成本约束的对偶变量...")
            implicit_duals['cpower'] = {}
            
            for g in range(self.ng):
                implicit_duals['cpower'][g] = {}
                for t in range(self.T):
                    constr_cpower = model.getConstrByName(f'cpower_{g}_{t}')
                    if constr_cpower is not None:
                        implicit_duals['cpower'][g][t] = constr_cpower.Pi
            
            # 7. DCPF潮流约束的对偶变量（修正名称）
            # print("7. 提取DCPF潮流约束的对偶变量...")
            implicit_duals['dcpf_upper'] = {}
            implicit_duals['dcpf_lower'] = {}
            
            for l in range(self.nl):
                implicit_duals['dcpf_upper'][l] = {}
                implicit_duals['dcpf_lower'][l] = {}
                for t in range(self.T):
                    # 潮流上限约束（修正名称为flow_upper）
                    constr_flow_upper = model.getConstrByName(f'flow_upper_{l}_{t}')
                    if constr_flow_upper is not None:
                        implicit_duals['dcpf_upper'][l][t] = constr_flow_upper.Pi
                    
                    # 潮流下限约束（修正名称为flow_lower）
                    constr_flow_lower = model.getConstrByName(f'flow_lower_{l}_{t}')
                    if constr_flow_lower is not None:
                        implicit_duals['dcpf_lower'][l][t] = constr_flow_lower.Pi
            
            # 8. x变量上下界约束的对偶变量（原来缺失的）
            # print("8. 提取x变量上下界约束的对偶变量...")
            implicit_duals['x_upper'] = {}
            implicit_duals['x_lower'] = {}
            
            for g in range(self.ng):
                implicit_duals['x_upper'][g] = {}
                implicit_duals['x_lower'][g] = {}
                for t in range(self.T):
                    # x上界约束
                    constr_x_upper = model.getConstrByName(f'x_upper_{g}_{t}')
                    if constr_x_upper is not None:
                        implicit_duals['x_upper'][g][t] = constr_x_upper.Pi
                    
                    # x下界约束
                    constr_x_lower = model.getConstrByName(f'x_lower_{g}_{t}')
                    if constr_x_lower is not None:
                        implicit_duals['x_lower'][g][t] = constr_x_lower.Pi
            
            # 统计提取结果
            # print("\n=== 对偶变量提取统计 ===")
            total_extracted = 0
            
            constraint_types = [
                ('功率平衡', implicit_duals.get('power_balance', {})),
                ('发电下限', implicit_duals.get('pg_lower', {})),
                ('发电上限', implicit_duals.get('pg_upper', {})),
                ('上爬坡', implicit_duals.get('ramp_up', {})),
                ('下爬坡', implicit_duals.get('ramp_down', {})),
                ('最小开机', implicit_duals.get('min_on', {})),
                ('最小关机', implicit_duals.get('min_off', {})),
                ('启动成本', implicit_duals.get('start_cost', {})),
                ('关机成本', implicit_duals.get('shut_cost', {})),
                ('非负约束', implicit_duals.get('coc_nonneg', {})),
                ('发电成本', implicit_duals.get('cpower', {})),
                ('潮流上限', implicit_duals.get('dcpf_upper', {})),
                ('潮流下限', implicit_duals.get('dcpf_lower', {})),
                ('x上界', implicit_duals.get('x_upper', {})),
                ('x下界', implicit_duals.get('x_lower', {})),
            ]
            
            for constraint_name, constraint_data in constraint_types:
                if isinstance(constraint_data, dict):
                    # 计算嵌套字典中的总元素数
                    count = 0
                    for key, value in constraint_data.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    count += len(sub_value)
                                else:
                                    count += 1
                        else:
                            count += 1
                    total_extracted += count
            
            # print(f"总计提取对偶变量: {total_extracted} 个")
            # print("✓ 成功通过约束名称提取所有隐式对偶变量")

            return implicit_duals

        except Exception as e:
            print(f"❌ 对偶变量提取过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def extract_dual_variables_as_arrays(self, model):
        """
        提取对偶变量并转换为与solve_with_dual输出格式一致的numpy数组
        
        Args:
            model: 求解后的Gurobi模型
            
        Returns:
            Dict: 包含numpy数组格式的对偶变量
        """
        # print("\n=== 提取对偶变量并转换为数组格式 ===")
        
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
            
            # 4. 最小开关机时间约束对偶变量: shape (ng, Ton/Toff, T)
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            
            if 'min_on' in implicit_duals_dict:
                lambda_sol_implicit['lambda_min_on'] = np.zeros((self.ng, Ton, self.T))
                for g in range(self.ng):
                    for tau in range(Ton):
                        for t in range(self.T):
                            val = implicit_duals_dict['min_on'].get(g, {}).get(tau+1, {}).get(t, 0)
                            lambda_sol_implicit['lambda_min_on'][g, tau, t] = abs(val)
            
            if 'min_off' in implicit_duals_dict:
                lambda_sol_implicit['lambda_min_off'] = np.zeros((self.ng, Toff, self.T))
                for g in range(self.ng):
                    for tau in range(Toff):
                        for t in range(self.T):
                            val = implicit_duals_dict['min_off'].get(g, {}).get(tau+1, {}).get(t, 0)
                            lambda_sol_implicit['lambda_min_off'][g, tau, t] = abs(val)
            
            # 5. 启停成本约束对偶变量: shape (ng, T-1)
            constraint_names = ['start_cost', 'shut_cost', 'coc_nonneg']
            lambda_names = ['lambda_start_cost', 'lambda_shut_cost', 'lambda_coc_nonneg']
            
            for constraint_name, lambda_name in zip(constraint_names, lambda_names):
                if constraint_name in implicit_duals_dict:
                    lambda_sol_implicit[lambda_name] = np.array([
                        [abs(implicit_duals_dict[constraint_name].get(g, {}).get(t, 0)) for t in range(self.T-1)] 
                        for g in range(self.ng)
                    ])
            
            # 6. 发电成本约束对偶变量: shape (ng, T)
            if 'cpower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_cpower'] = np.array([
                    [abs(implicit_duals_dict['cpower'].get(g, {}).get(t, 0)) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            
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
            
            print("✓ 成功将对偶变量转换为数组格式")
            return lambda_sol_implicit
            
        except Exception as e:
            print(f"❌ 对偶变量数组转换失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _create_union_analysis_from_x_init(self, x_init, lambda_init):
        """
        基于初始化求解的x_init创建union_analysis
        参考dfsm_training中的逻辑，识别起作用的DCPF约束
        
        Args:
            x_init: 初始化求解得到的x解，shape (n_samples, ng, T)
            
        Returns:
            Dict: union_analysis字典
        """
        print("\n=== 基于x_init创建union_analysis ===")
        
        try:
            x_sol = x_init
            
            # 找到非整数变量
            fractional_variables = []
            tolerance = 0.1
            
            for g in range(self.ng):
                for t in range(self.T):
                    for sample_id in range(self.n_samples):
                        x_val = x_sol[sample_id, g, t]
                        x_true = self.active_set_data[sample_id]['unit_commitment_matrix'][g, t]
                        if tolerance < x_val < (1 - tolerance):
                            fractional_variables.append({
                                'unit_id': g,
                                'time_slot': t,
                            'variable_name': f'x[{g},{t}]'
                            })
                            break
                        elif abs(x_val - x_true) > 0.4:
                            fractional_variables.append({
                                'unit_id': g,
                                'time_slot': t,
                            'variable_name': f'x[{g},{t}]'
                            })
                            break
            
            print(f"发现 {len(fractional_variables)} 个非整数/非正确变量")
            
            # 计算DCPF约束系数
            union_constraints = self._compute_dcpf_constraints_for_fractional_times(
                fractional_variables, lambda_init
            )
            union_zeta_constraints = self._compute_specialized_constraints_of_balance_node(
                fractional_variables
            )

            print(f"生成 {len(union_constraints)} 个union约束, 生成 {len(union_zeta_constraints)} 个zeta约束")
            
            return {
                'union_constraints': union_constraints,
                'union_zeta_constraints': union_zeta_constraints,
            }
            
            
        except Exception as e:
            print(f"❌ 创建union_analysis失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _compute_dcpf_constraints_for_fractional_times(self, fractional_variables, lambda_init):
        """
        计算涉及非整数变量时段的DCPF约束
        
        Args:
            fractional_variables: 非整数变量列表
            x_sol: x变量解，shape (ng, T)
            
        Returns:
            List: DCPF约束列表
        """
        # 提取非整数变量的时段
        fractional_time_slots = set()
        for frac_var in fractional_variables:
            fractional_time_slots.add(frac_var['time_slot'])
        
        print(f"非整数变量涉及时段: {sorted(fractional_time_slots)}")
        
        union_constraints = []
        
        try:
            # 构建机组-节点映射矩阵G
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            # 只为涉及非整数变量的时段和所有支路生成约束
            for time_slot in fractional_time_slots:
                for branch_id in range(self.branch.shape[0]):
                    
                    # 计算该支路在该时段的约束系数
                    ptdf_row = PTDF[branch_id, :]
                    pg_coefficients = ptdf_row @ G
                    
                    # 过滤非零系数
                    coeff_threshold = 1e-8
                    nonzero_coefficients = []
                    
                    for g in range(self.ng):
                        coeff = pg_coefficients[g]
                        if abs(coeff) > coeff_threshold:
                            nonzero_coefficients.append({
                                'unit_id': g,
                                'coefficient': 0
                            })
                    
                    if not nonzero_coefficients:
                        continue
                    
                    lambda_upper_flag = False
                    lambda_lower_flag = False
                    lambda_coeff = 1e-8
                    for sample_id in range(self.n_samples):
                        if lambda_init[sample_id]['lambda_dcpf_upper'][branch_id, time_slot] > lambda_coeff:
                            lambda_upper_flag = True
                        if lambda_init[sample_id]['lambda_dcpf_lower'][branch_id, time_slot] > lambda_coeff:
                            lambda_lower_flag = True

                    if lambda_upper_flag or lambda_lower_flag:                    
                        # 检查该约束是否涉及非整数变量
                        involves_fractional = any(
                            frac_var['time_slot'] == time_slot and 
                            any(coeff['unit_id'] == frac_var['unit_id'] for coeff in nonzero_coefficients)
                            for frac_var in fractional_variables
                        )
                        
                        if involves_fractional:
                            # 找到涉及的非整数机组
                            fractional_units = []
                            for frac_var in fractional_variables:
                                if frac_var['time_slot'] == time_slot:
                                    unit_id = frac_var['unit_id']
                                    if any(coeff['unit_id'] == unit_id for coeff in nonzero_coefficients):
                                        pg_coeff = next((c['coefficient'] for c in nonzero_coefficients if c['unit_id'] == unit_id), 0)
                                        fractional_units.append({
                                            'unit_id': unit_id,
                                            'pg_coefficient': pg_coeff
                                        })
                            if lambda_upper_flag:            
                                
                                # 添加上限约束
                                union_constraints.append({
                                    'branch_id': branch_id,
                                    'time_slot': time_slot,
                                    'constraint_type': 'dcpf_upper',
                                    'constraint_name': f'dcpf_upper_{branch_id}_{time_slot}',
                                    'nonzero_pg_coefficients': nonzero_coefficients,
                                    'total_nonzero_variables': len(nonzero_coefficients),
                                    'branch_limit': float(branch_limit[branch_id]),
                                    'involves_fractional_variables': True,
                                    'fractional_units': fractional_units,
                                    'in_json': False,  # 这是基于当前解生成的
                                    'in_current': True
                                })
                            if lambda_lower_flag:
                                
                                # 下限约束的系数需要取负号
                                nonzero_coefficients_lower = [
                                    {'unit_id': coeff['unit_id'], 'coefficient': -coeff['coefficient']}
                                    for coeff in nonzero_coefficients
                                ]
                                
                                union_constraints.append({
                                    'branch_id': branch_id,
                                    'time_slot': time_slot,
                                    'constraint_type': 'dcpf_lower',
                                    'constraint_name': f'dcpf_lower_{branch_id}_{time_slot}',
                                    'nonzero_pg_coefficients': nonzero_coefficients_lower,
                                    'total_nonzero_variables': len(nonzero_coefficients_lower),
                                    'branch_limit': float(branch_limit[branch_id]),
                                    'involves_fractional_variables': True,
                                    'fractional_units': fractional_units,
                                    'in_json': False,
                                    'in_current': True
                                })
            
            print(f"✓ 生成了 {len(union_constraints)} 个DCPF约束")
            
            # 按支路和时段排序
            union_constraints.sort(key=lambda x: (x['branch_id'], x['time_slot'], x['constraint_type']))
            
            return union_constraints
            
        except Exception as e:
            print(f"❌ 计算DCPF约束失败: {e}")
            return []

    def _compute_specialized_constraints_of_balance_node(self, fractional_variables):
        """
        计算涉及非整数变量时段的平衡节点功率约束
        
        Args:
            fractional_variables: 非整数变量列表

        Returns:
            List: 平衡节点功率约束列表
        """
        fractional_time_slots = set()
        balance_fractional_units = []
        for frac_var in fractional_variables:
            if self.bus[self.gen[frac_var['unit_id'], GEN_BUS], BUS_TYPE] == 3:  # 平衡节点类型为3
                fractional_time_slots.add(frac_var['time_slot'])
                balance_fractional_units.append(frac_var)

        print(f"平衡节点非整数变量涉及时段: {sorted(fractional_time_slots)}")
        
        union_constraints = []
        
        # TODO: 实现平衡节点功率约束的计算逻辑
        
        for var in balance_fractional_units:
            
            print(f"平衡节点非整数变量: 机组 {var['unit_id']} 时段 {var['time_slot']}")
            
            union_constraints.append({
                'time_slot': var['time_slot'],
                'unit_id': var['unit_id'],
                'constraint_type': 'balance_node_power',
                'constraint_name': f"balance_node_power_{var['unit_id']}_{var['time_slot']}"
            })            
       
        return union_constraints

    def initialize_theta_values(self, union_analysis=None):
        """
        根据union_analysis信息初始化全零的theta数值和对偶变量初始值
        
        Args:
            union_analysis: 并集约束分析结果，如果为None则使用self._current_union_analysis
            
        Returns:
            Dict: 包含theta变量值和对偶变量初始值的字典
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化theta值和对偶变量")
            return {}
        
        union_constraints = union_analysis['union_constraints']

        # 获取所有相关的支路ID
        branch_ids = set()
        for constraint in union_constraints:
            branch_ids.add(constraint['branch_id'])
        
        # 初始化结果字典
        initialization_values = {
            'theta_values': {},
            'dual_values': {},
            'constraint_mapping': {}
        }
        
        # 初始化theta数值
        for branch_id in branch_ids:
            # 为每个机组创建3个theta变量的零值 (theta_0, theta_1, theta_2)
            for unit_id in range(self.ng):
                for order in range(3):
                    var_name = f'theta_branch_{branch_id}_unit_{unit_id}_{order}'
                    initialization_values['theta_values'][var_name] = 1
            
            # 为右端项创建theta变量的零值
            for order in range(3):
                var_name = f'theta_branch_{branch_id}_rhs_{order}'
                initialization_values['theta_values'][var_name] = 1
        
        #初始化mu
        mu_init = np.zeros((self.n_samples, self.branch.shape[0], self.T), dtype=float)
        for sample in range(self.n_samples):
            for b in range(mu_init.shape[1]):
                for t in range(mu_init.shape[2]):
                    mu_init[sample,b,t] = 1e-3

        return initialization_values['theta_values'], mu_init

    def initialize_zeta_values(self, union_analysis=None):
        """
        根据union_analysis信息初始化全零的zeta数值和对偶变量初始值

        Args:
            union_analysis: 并集约束分析结果，如果为None则使用self._current_union_analysis
            
        Returns:
            Dict: 包含zeta变量值和对偶变量初始值的字典
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法初始化zeta值和对偶变量")
            return {}
        
        union_constraints = union_analysis['union_zeta_constraints']

        # 初始化结果字典
        initialization_values = {
            'zeta_values': {},
            'dual_values': {},
            'constraint_mapping': {}
        }

        # 初始化zeta数值
        for constraint in union_constraints:
            # 为每个机组创建3个zeta变量的零值 (zeta_0, zeta_1, zeta_2)
            for order in range(3):
                var_name = f'zeta_lower_unit_{constraint["unit_id"]}_time_{constraint["time_slot"]}_{order}'
                initialization_values['zeta_values'][var_name] = 1

            # 为右端项创建zeta变量的零值
            for order in range(3):
                var_name = f'zeta_lower_unit_{constraint["unit_id"]}_time_{constraint["time_slot"]}_rhs_{order}'
                initialization_values['zeta_values'][var_name] = 1

            for order in range(3):
                var_name = f'zeta_upper_unit_{constraint["unit_id"]}_time_{constraint["time_slot"]}_{order}'
                initialization_values['zeta_values'][var_name] = 1

            # 为右端项创建zeta变量的零值
            for order in range(3):
                var_name = f'zeta_upper_unit_{constraint["unit_id"]}_time_{constraint["time_slot"]}_rhs_{order}'
                initialization_values['zeta_values'][var_name] = 1
                
        #初始化zeta
        ita_lower_init = np.zeros((self.n_samples, 1, self.T), dtype=float)
        for sample in range(self.n_samples):
            for i in range(ita_lower_init.shape[1]):
                for t in range(ita_lower_init.shape[2]):
                    ita_lower_init[sample,i,t] = 1e-3

        ita_upper_init = np.zeros((self.n_samples, 1, self.T), dtype=float)
        for sample in range(self.n_samples):
            for i in range(ita_upper_init.shape[1]):
                for t in range(ita_upper_init.shape[2]):
                    ita_upper_init[sample,i,t] = 1e-3

        return initialization_values['zeta_values'], ita_lower_init, ita_upper_init

    def add_theta_variables_for_branches(self, union_analysis=None):
        """
        为参数化约束添加theta变量
        
        Args:
            union_analysis: 并集约束分析结果，如果为None则使用self._current_union_analysis
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建theta变量")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        # 获取所有相关的支路ID
        branch_ids = set()
        for constraint in union_constraints:
            branch_ids.add(constraint['branch_id'])
        
        print(f"为 {len(branch_ids)} 条支路创建theta变量")
        
        # 初始化theta变量字典
        self.theta_vars = {}
        
        # 为每条支路创建theta变量
        for branch_id in branch_ids:
            # 为每个机组创建3个theta变量 (theta_0, theta_1, theta_2)
            for unit_id in range(self.ng):
                for order in range(3):
                    var_name = f'theta_branch_{branch_id}_unit_{unit_id}_{order}'
                    self.theta_vars[var_name] = {
                        'branch_id': branch_id,
                        'unit_id': unit_id,
                        'order': order,
                        'var_name': var_name,
                        'value': 0.0  # 默认值
                    }
            
            # 为右端项创建theta变量
            for order in range(3):
                var_name = f'theta_branch_{branch_id}_rhs_{order}'
                self.theta_vars[var_name] = {
                    'branch_id': branch_id,
                    'unit_id': None,
                    'order': order,
                    'var_name': var_name,
                    'value': 0.0  # 默认值
                }
        
        print(f"✓ 创建了 {len(self.theta_vars)} 个theta变量")

    def add_zeta_variables_for_units(self, union_analysis=None):
        """
        为参数化约束添加zeta变量

        Args:
            union_analysis: 并集约束分析结果，如果为None则使用self._current_union_analysis
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("警告: 没有union_analysis数据，无法创建zeta变量")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']

        # 初始化zeta变量字典
        self.zeta_vars = {}
                
        # 初始化zeta数值
        for constraint in union_constraints:
            # 为每个机组创建3个zeta变量的零值 (zeta_0, zeta_1, zeta_2)
            for order in range(3):
                var_name = f'zeta_lower_unit_{constraint["unit_id"]}_{order}'
                self.zeta_vars[var_name] = {
                    'unit_id': constraint["unit_id"],
                    'order': order,
                    'var_name': var_name,
                    'value': 0.0  # 默认值
                }

            # 为右端项创建zeta变量的零值
            for order in range(3):
                var_name = f'zeta_lower_unit_{constraint["unit_id"]}_rhs_{order}'
                self.zeta_vars[var_name] = {
                    'unit_id': constraint["unit_id"],
                    'order': order,
                    'var_name': var_name,
                    'value': 0.0  # 默认值
                }

            for order in range(3):
                var_name = f'zeta_upper_unit_{constraint["unit_id"]}_{order}'
                self.zeta_vars[var_name] = {
                    'unit_id': constraint["unit_id"],
                    'order': order,
                    'var_name': var_name,
                    'value': 0.0  # 默认值
                }

            # 为右端项创建zeta变量的零值
            for order in range(3):
                var_name = f'zeta_upper_unit_{constraint["unit_id"]}_rhs_{order}'
                self.zeta_vars[var_name] = {
                    'unit_id': constraint["unit_id"],
                    'order': order,
                    'var_name': var_name,
                    'value': 0.0  # 默认值
                }
                
        print(f"✓ 创建了 {len(self.zeta_vars)} 个zeta变量")

    def _add_parametric_dcpf_constraints_with_theta(self, model, pg, sample_id, theta_values=None, union_analysis=None):
        """
        添加包含theta参数的DCPF约束
        
        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            print("未提供theta值，使用默认值0")
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                
                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * pg[unit_id, time_slot]
                
                # 构建右端项: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
                theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                
                theta_rhs_0 = theta_values.get(theta_rhs_0_name, 0.0)
                theta_rhs_1 = theta_values.get(theta_rhs_1_name, 0.0)
                theta_rhs_2 = theta_values.get(theta_rhs_2_name, 0.0)
                
                parametric_rhs = u + theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u
                
                # 添加约束
                if constraint_type == 'upper' or constraint_type == 'le':
                    model.addConstr(lhs_expr <= parametric_rhs, 
                                  name=f'parametric_dcpf_upper_branch_{branch_id}_t_{time_slot}')
                elif constraint_type == 'lower' or constraint_type == 'ge':
                    model.addConstr(lhs_expr >= parametric_rhs, 
                                  name=f'parametric_dcpf_lower_branch_{branch_id}_t_{time_slot}')
                else:
                    # 默认为上限约束
                    model.addConstr(lhs_expr <= parametric_rhs, 
                                  name=f'parametric_dcpf_branch_{branch_id}_t_{time_slot}')
                
                constraint_count += 1
            
            print(f"✓ 成功添加 {constraint_count} 个参数化约束")
            
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_balance_power_constraints_with_zeta(self, model, pg, sample_id, zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的DCPF约束

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            print("未提供zeta值，使用默认值0")
            zeta_values = {}
        
        try:           
            constraint_count = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 1

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * pg[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                model.addConstr(lhs_expr >= parametric_rhs, 
                                name=f'parametric_balance_power_lower_unit_{unit_id}_t_{time_slot}')
                
                constraint_count += 1

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * pg[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                model.addConstr(lhs_expr <= parametric_rhs, 
                                name=f'parametric_balance_power_upper_unit_{unit_id}_t_{time_slot}')

                constraint_count += 1
                            
            print(f"✓ 成功添加 {constraint_count} 个参数化约束")
            
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_penalties_pg_block(self, model, x, sample_id,  theta_values=None, union_analysis=None):
        """
        添加包含theta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            print("未提供theta值，使用默认值0")
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            obj_primal = 0
            obj_opt = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, time_slot]
                
                # 构建右端项: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
                theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                
                theta_rhs_0 = theta_values.get(theta_rhs_0_name, 0.0)
                theta_rhs_1 = theta_values.get(theta_rhs_1_name, 0.0)
                theta_rhs_2 = theta_values.get(theta_rhs_2_name, 0.0)
                
                parametric_rhs = theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u

                parametric_rhs_viol = model.addVars(1, 1, lb=0, name=f'parametric_rhs_viol_{branch_id}_{time_slot}')
                parametric_rhs_abs = model.addVars(1, 1, lb=0, name=f'parametric_rhs_abs_{branch_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_rhs_viol_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_rhs_abs1_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= -lhs_expr + parametric_rhs, name=f'parametric_rhs_abs2_{branch_id}_{time_slot}')

                obj_primal += parametric_rhs_viol[0, 0]

                obj_opt += parametric_rhs_abs[0, 0] * self.mu[sample_id, branch_id, time_slot]

            model.update()
            
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_constraints_dual_block(self, model, g_id, t_id, mu, sample_id, theta_values=None, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            
            dual_expr = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                if time_slot != t_id:
                    continue
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']
                
                model.addConstr(mu[branch_id, time_slot] >= 1e-3, name=f'parametric_mu_least_{branch_id}_{time_slot}')

                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                for coeff_info in nonzero_coefficients:
                
                    unit_id = coeff_info['unit_id']
                    if unit_id != g_id:
                        continue
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    dual_expr += parametric_coeff * mu[branch_id, time_slot]

            model.update()
            
            return model, dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()      

    def _add_parametric_obj_dual_block(self, model, x, mu, sample_id, theta_values=None, union_analysis=None):
        """
        添加包含theta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            print("未提供theta值，使用默认值0")
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            obj_opt = 0
            constraint_count = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']                
                
                model.addConstr(mu[branch_id, time_slot] >= 1e-6, name=f'parametric_mu_least_{branch_id}_{time_slot}')
                
                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                
                # 构建左端项表达式
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, time_slot]
                
                # 构建右端项: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
                theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                
                theta_rhs_0 = theta_values.get(theta_rhs_0_name, 0.0)
                theta_rhs_1 = theta_values.get(theta_rhs_1_name, 0.0)
                theta_rhs_2 = theta_values.get(theta_rhs_2_name, 0.0)
                
                parametric_rhs = theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                obj_opt += parametric_rhs_abs * mu[branch_id, time_slot]

            model.update()

            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_variational_primal_theta_block(self, model, x, sample_id, theta, union_analysis=None):
        """
        添加包含theta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            obj_primal = 0
            obj_opt = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                if constraint_type == 'dcpf_upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'dcpf_lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'

                    theta_0 = theta[theta_0_name]
                    theta_1 = theta[theta_1_name]
                    theta_2 = theta[theta_2_name]

                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, time_slot]
                
                # 构建右端项: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
                theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                
                theta_rhs_0 = theta[theta_rhs_0_name]
                theta_rhs_1 = theta[theta_rhs_1_name]
                theta_rhs_2 = theta[theta_rhs_2_name]

                parametric_rhs = theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u

                parametric_rhs_viol = model.addVar(lb=0, name=f'sample_{sample_id}_parametric_rhs_viol_{branch_id}_{time_slot}')
                parametric_rhs_abs = model.addVar(lb=0, name=f'sample_{sample_id}_parametric_rhs_abs_{branch_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol >= lhs_expr - parametric_rhs, name=f'sample_{sample_id}_parametric_rhs_viol_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= lhs_expr - parametric_rhs, name=f'sample_{sample_id}_parametric_rhs_abs1_{branch_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs >= -lhs_expr + parametric_rhs, name=f'sample_{sample_id}_parametric_rhs_abs2_{branch_id}_{time_slot}')

                obj_primal += parametric_rhs_abs

                obj_opt += parametric_rhs_abs * self.mu[sample_id, branch_id, time_slot]

            model.update()
            
            return model, obj_primal, obj_opt
        
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_variational_dual_theta_block(self, model, g_id, t_id, mu, sample_id, theta, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            
            dual_expr = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                if time_slot != t_id:
                    continue
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                for coeff_info in nonzero_coefficients:
                
                    unit_id = coeff_info['unit_id']
                    if unit_id != g_id:
                        continue
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta[theta_0_name]
                    theta_1 = theta[theta_1_name]
                    theta_2 = theta[theta_2_name]

                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u

                    dual_expr += parametric_coeff * mu[branch_id, time_slot]

            model.update()

            return model, dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()    

    def _add_parametric_penalties_pg_block_const(self, x, sample_id, theta_values=None, union_analysis=None):
        """
        添加包含theta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            print("未提供theta值，使用默认值0")
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            constraint_count = 0
            obj_primal = 0
            obj_opt = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    # 添加到左端项
                    lhs_expr += parametric_coeff * x[unit_id, time_slot]
                
                # 构建右端项: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
                theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                
                theta_rhs_0 = theta_values.get(theta_rhs_0_name, 0.0)
                theta_rhs_1 = theta_values.get(theta_rhs_1_name, 0.0)
                theta_rhs_2 = theta_values.get(theta_rhs_2_name, 0.0)
                
                parametric_rhs = theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u

                obj_primal += max(0, lhs_expr - parametric_rhs)

                obj_opt += abs(lhs_expr - parametric_rhs) * self.mu[sample_id, branch_id, time_slot]

            return obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_constraints_dual_block_const(self, g_id, t_id, mu, sample_id, theta_values=None, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if theta_values is None:
            theta_values = {}
        
        try:
            # 构建机组-节点映射矩阵
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            # 计算PTDF矩阵
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            dual_expr = 0
            
            for constraint_info in union_constraints:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                if time_slot != t_id:
                    continue
                constraint_type = constraint_info.get('constraint_type', 'upper')
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']

                if constraint_type == 'upper':
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                elif constraint_type == 'lower':
                    u = branch_limit[branch_id] - PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]
                else:
                    u = branch_limit[branch_id] + PTDF[branch_id, :] @ self.active_set_data[sample_id]['pd_data'][:, time_slot]

                for coeff_info in nonzero_coefficients:
                
                    unit_id = coeff_info['unit_id']
                    if unit_id != g_id:
                        continue
                    original_coeff = 1
                    
                    # 获取theta变量值
                    theta_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                    theta_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                    theta_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                    
                    theta_0 = theta_values.get(theta_0_name, 0.0)
                    theta_1 = theta_values.get(theta_1_name, 0.0)
                    theta_2 = theta_values.get(theta_2_name, 0.0)
                    
                    # 计算参数化系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    parametric_coeff = original_coeff + theta_0 + theta_1 * u + theta_2 * u * u
                    
                    dual_expr += parametric_coeff * mu[branch_id, time_slot]

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()   

    def _add_parametric_balance_power_penalties_pg_block(self, model, x, sample_id,  zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            zeta_values: zeta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_constraints']
        
        if zeta_values is None:
            print("未提供zeta值，使用默认值0")
            zeta_values = {}
        
        try:
            obj_primal = 0
            obj_opt = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 1

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_viol = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_lower_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_lower_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_lower_rhs_viol_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_lower_rhs_abs1_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= -lhs_expr + parametric_rhs, name=f'parametric_balance_power_lower_rhs_abs2_{unit_id}_{time_slot}')

                obj_primal += parametric_rhs_viol[0, 0]

                obj_opt += parametric_rhs_abs[0, 0] * self.ita_lower[sample_id, 0, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_viol = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_upper_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_upper_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_upper_rhs_viol_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_upper_rhs_abs1_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= -lhs_expr + parametric_rhs, name=f'parametric_balance_power_upper_rhs_abs2_{unit_id}_{time_slot}')

                obj_primal += parametric_rhs_viol[0, 0]

                obj_opt += parametric_rhs_abs[0, 0] * self.ita_upper[sample_id, 0, time_slot]

            model.update()
            
            return model, obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_balance_power_constraints_dual_block(self, model, g_id, t_id, ita_lower, ita_upper, sample_id, zeta_values=None, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return

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

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                model.addConstr(ita_lower[0, time_slot] + ita_upper[0, time_slot] >= 1e-3, name=f'parametric_ita_least_{time_slot}')

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                dual_expr += - parametric_rhs * ita_lower[0, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u
                    
                dual_expr += parametric_rhs * ita_upper[0, time_slot]
            
            return model, dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()      

    def _add_parametric_balance_power_obj_dual_block(self, model, x, ita_lower, ita_upper, sample_id, zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            theta_values: theta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            print("未提供zeta值，使用默认值0")
            zeta_values = {}

        try:
            obj_opt = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 1

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                obj_opt += parametric_rhs_abs * ita_lower[0, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_abs = abs(lhs_expr - parametric_rhs)

                obj_opt += parametric_rhs_abs * ita_upper[0, time_slot]

            model.update()

            return model, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_variational_balance_power_primal_zeta_block(self, model, x, sample_id, zeta, union_analysis=None):
        """
        添加包含zeta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            zeta_values: zeta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        try:
            obj_primal = 0
            obj_opt = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 1

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta[zeta_0_name]
                zeta_1 = zeta[zeta_1_name]
                zeta_2 = zeta[zeta_2_name]

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta[zeta_rhs_0_name]
                zeta_rhs_1 = zeta[zeta_rhs_1_name]
                zeta_rhs_2 = zeta[zeta_rhs_2_name]

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_viol = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_lower_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_lower_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_lower_rhs_viol_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_lower_rhs_abs1_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= -lhs_expr + parametric_rhs, name=f'parametric_balance_power_lower_rhs_abs2_{unit_id}_{time_slot}')

                obj_primal += parametric_rhs_viol[0, 0]

                obj_opt += parametric_rhs_abs[0, 0] * self.ita_lower[sample_id, 0, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta[zeta_0_name]
                zeta_1 = zeta[zeta_1_name]
                zeta_2 = zeta[zeta_2_name]

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta[zeta_rhs_0_name]
                zeta_rhs_1 = zeta[zeta_rhs_1_name]
                zeta_rhs_2 = zeta[zeta_rhs_2_name]

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                parametric_rhs_viol = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_upper_rhs_viol_{unit_id}_{time_slot}')
                parametric_rhs_abs = model.addVars(1, 1, lb=0, name=f'parametric_balance_power_upper_rhs_abs_{unit_id}_{time_slot}')

                model.addConstr(parametric_rhs_viol[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_upper_rhs_viol_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= lhs_expr - parametric_rhs, name=f'parametric_balance_power_upper_rhs_abs1_{unit_id}_{time_slot}')
                model.addConstr(parametric_rhs_abs[0, 0] >= -lhs_expr + parametric_rhs, name=f'parametric_balance_power_upper_rhs_abs2_{unit_id}_{time_slot}')

                obj_primal += parametric_rhs_viol[0, 0]

                obj_opt += parametric_rhs_abs[0, 0] * self.ita_upper[sample_id, 0, time_slot]

            model.update()
            
            return model, obj_primal, obj_opt
        
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_variational_balance_power_dual_zeta_block(self, model, g_id, t_id, ita_lower, ita_upper, sample_id, zeta, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        try:            
            dual_expr = 0
            
            for constraint in union_constraints:
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                
                if time_slot != t_id or unit_id != g_id:
                    continue

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta[zeta_rhs_0_name]
                zeta_rhs_1 = zeta[zeta_rhs_1_name]
                zeta_rhs_2 = zeta[zeta_rhs_2_name]

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                dual_expr += - parametric_rhs * ita_lower[0, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta[zeta_rhs_0_name]
                zeta_rhs_1 = zeta[zeta_rhs_1_name]
                zeta_rhs_2 = zeta[zeta_rhs_2_name]

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u
                    
                dual_expr += parametric_rhs * ita_upper[0, time_slot]

            model.update()

            return model, dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()    

    def _add_parametric_balance_power_penalties_pg_block_const(self, x, sample_id, zeta_values=None, union_analysis=None):
        """
        添加包含zeta参数的DCPF罚项

        Args:
            model: Gurobi模型
            pg: 功率变量
            zeta_values: zeta参数值字典，如果为None则使用默认值0
            union_analysis: 并集约束分析结果
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return
        
        union_constraints = union_analysis['union_zeta_constraints']
        
        if zeta_values is None:
            print("未提供zeta值，使用默认值0")
            zeta_values = {}
        
        try:
            obj_primal = 0
            obj_opt = 0
            
            for constraint in union_constraints:            
                unit_id = constraint['unit_id']
                time_slot = constraint['time_slot']
                original_coeff = 1

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                obj_primal += max(0, lhs_expr - parametric_rhs)

                obj_opt += abs(lhs_expr - parametric_rhs) * self.ita_lower[sample_id, 0, time_slot]

                # 构建左端项表达式
                lhs_expr = 0
                zeta_0_name = f'zeta_upper_lower_unit_{unit_id}_time_{time_slot}_0'
                zeta_1_name = f'zeta_upper_lower_unit_{unit_id}_time_{time_slot}_1'
                zeta_2_name = f'zeta_upper_lower_unit_{unit_id}_time_{time_slot}_2'

                zeta_0 = zeta_values.get(zeta_0_name, 0.0)
                zeta_1 = zeta_values.get(zeta_1_name, 0.0)
                zeta_2 = zeta_values.get(zeta_2_name, 0.0)

                # 计算参数化系数: original_coeff + zeta_0 + zeta_1*u + zeta_2*u^2
                parametric_coeff = original_coeff + zeta_0 + zeta_1 * u + zeta_2 * u * u

                # 添加到左端项
                lhs_expr += parametric_coeff * x[unit_id, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                obj_primal += max(0, lhs_expr - parametric_rhs)

                obj_opt += abs(lhs_expr - parametric_rhs) * self.ita_upper[sample_id, 0, time_slot]

            return obj_primal, obj_opt
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()

    def _add_parametric_balance_power_constraints_dual_block_const(self, g_id, t_id, ita_lower, ita_upper, sample_id, zeta_values=None, union_analysis=None): 
        """

        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        
        if not union_analysis or 'union_zeta_constraints' not in union_analysis:
            print("⚠ 未提供union_analysis，跳过参数化约束")
            return

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

                u = np.sum(self.active_set_data[sample_id]['pd_data'][:, time_slot])

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_lower_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u

                dual_expr += - parametric_rhs * ita_lower[0, time_slot]

                # 构建右端项: u + zeta_rhs_0 + zeta_rhs_1*u + zeta_rhs_2*u^2
                zeta_rhs_0_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_0'
                zeta_rhs_1_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_1'
                zeta_rhs_2_name = f'zeta_upper_unit_{unit_id}_time_{time_slot}_rhs_2'

                zeta_rhs_0 = zeta_values.get(zeta_rhs_0_name, 0.0)
                zeta_rhs_1 = zeta_values.get(zeta_rhs_1_name, 0.0)
                zeta_rhs_2 = zeta_values.get(zeta_rhs_2_name, 0.0)

                parametric_rhs = zeta_rhs_0 + zeta_rhs_1 * u + zeta_rhs_2 * u * u
                    
                dual_expr += parametric_rhs * ita_upper[0, time_slot]

            return dual_expr
        except Exception as e:
            print(f"❌ 添加参数化约束时出错: {e}")
            import traceback
            traceback.print_exc()   
                               
    def iter_with_pg_block(self, sample_id=0, theta_values=None, zeta_values=None, union_analysis=None):
        """
        迭代PG块，处理包含PG变量的A矩阵
        """
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('iter_with_pg_block')
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')        
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')

        # 辅助变量用于处理绝对值和最大值
        # 功率平衡约束违反量
        power_balance_viol = model.addVars(self.T, lb=0, name='power_balance_viol')
        
        # 发电限制约束违反量
        pg_lower_viol = model.addVars(self.ng, self.T, lb=0, name='pg_lower_viol')
        pg_upper_viol = model.addVars(self.ng, self.T, lb=0, name='pg_upper_viol')

        pg_lower_abs = model.addVars(self.ng, self.T, lb=0, name='pg_lower_abs')
        pg_upper_abs = model.addVars(self.ng, self.T, lb=0, name='pg_upper_abs')
        
        # 爬坡约束违反量
        ramp_up_viol = model.addVars(self.ng, self.T-1, lb=0, name='ramp_up_viol')
        ramp_down_viol = model.addVars(self.ng, self.T-1, lb=0, name='ramp_down_viol')

        ramp_up_abs = model.addVars(self.ng, self.T-1, lb=0, name='ramp_up_abs')
        ramp_down_abs = model.addVars(self.ng, self.T-1, lb=0, name='ramp_down_abs')
        
        # 最小开关机时间
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        # 最小开关机时间违反量
        min_on_viol = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_viol')
        min_off_viol = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_viol')
        
        min_on_abs = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_abs')
        min_off_abs = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_abs')

        # 启停成本违反量
        start_cost_viol = model.addVars(self.ng, self.T, lb=0, name='start_cost_viol')
        shut_cost_viol = model.addVars(self.ng, self.T, lb=0, name='shut_cost_viol')

        # 启停成本绝对值
        start_cost_abs = model.addVars(self.ng, self.T, lb=0, name='start_cost_abs')
        shut_cost_abs = model.addVars(self.ng, self.T, lb=0, name='shut_cost_abs')

        # 发电成本
        cpower_viol = model.addVars(self.ng, self.T, lb=0, name='cpower_viol')
        cpower_abs = model.addVars(self.ng, self.T, lb=0, name='cpower_abs')
        
        # 潮流约束
        dcpf_upper_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_viol')
        dcpf_upper_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_abs')
        dcpf_lower_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_viol')
        dcpf_lower_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_abs')

        # x变量约束违反量
        x_lower_viol = model.addVars(self.ng, self.T, lb=0, name='x_lower_viol')
        x_upper_viol = model.addVars(self.ng, self.T, lb=0, name='x_upper_viol')

        x_lower_abs = model.addVars(self.ng, self.T, lb=0, name='x_lower_abs')
        x_upper_abs = model.addVars(self.ng, self.T, lb=0, name='x_upper_abs')

        # 二进制变量偏差
        x_binary_dev = model.addVars(self.ng, self.T, lb=0, name='x_binary_dev')

        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        # 构建约束和目标函数项
        obj_primal = 0
        obj_opt = 0
        obj_binary = 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            
            # 添加绝对值约束
            model.addConstr(power_balance_viol[t] >= power_balance_expr, name=f'power_balance_pos_{t}')
            model.addConstr(power_balance_viol[t] >= -power_balance_expr, name=f'power_balance_neg_{t}')
            
            obj_primal += power_balance_viol[t]
            
            obj_opt += power_balance_viol[t] * abs(self.lambda_[sample_id]['lambda_power_balance'][t])
            
            # 发电上下限约束
            for g in range(self.ng):
                # 下限约束违反
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_viol[g, t] >= pg_lower_expr, name=f'pg_lower_viol_{g}_{t}')
                obj_primal += pg_lower_viol[g, t]
                
                # 上限约束违反
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_viol[g, t] >= pg_upper_expr, name=f'pg_upper_viol_{g}_{t}')
                obj_primal += pg_upper_viol[g, t]
                
                model.addConstr(pg_lower_abs[g, t] >= pg_lower_expr, name=f'pg_lower_abs1_{g}_{t}')
                model.addConstr(pg_lower_abs[g, t] >=-pg_lower_expr, name=f'pg_lower_abs2_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t] >= pg_upper_expr, name=f'pg_upper_abs1_{g}_{t}')
                model.addConstr(pg_upper_abs[g, t] >=-pg_upper_expr, name=f'pg_upper_abs2_{g}_{t}')

                obj_opt += pg_lower_abs[g, t] * abs(self.lambda_[sample_id]['lambda_pg_lower'][g, t])
                obj_opt += pg_upper_abs[g, t] * abs(self.lambda_[sample_id]['lambda_pg_upper'][g, t])

                # x变量约束违反
                # model.addConstr(x_lower_viol[g, t] >= -x[g, t], name=f'x_lower_viol_{g}_{t}')
                # model.addConstr(x_upper_viol[g, t] >= x[g, t] - 1, name=f'x_upper_viol_{g}_{t}')
                # obj_primal += x_lower_viol[g, t] + x_upper_viol[g, t]

                obj_opt += x[g, t] * abs(self.lambda_[sample_id]['lambda_x_lower'][g, t])
                obj_opt += (1 - x[g, t]) * abs(self.lambda_[sample_id]['lambda_x_upper'][g, t])

        # 爬坡约束（类似处理）
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                # 上爬坡约束违反
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_viol[g, t-1] >= ramp_up_expr, name=f'ramp_up_viol_{g}_{t}')
                obj_primal += ramp_up_viol[g, t-1]
                
                model.addConstr(ramp_up_abs[g, t-1] >= ramp_up_expr, name=f'ramp_up_abs1_{g}_{t}')
                model.addConstr(ramp_up_abs[g, t-1] >= -ramp_up_expr, name=f'ramp_up_abs2_{g}_{t}')

                # 下爬坡约束违反
                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_viol[g, t-1] >= ramp_down_expr, name=f'ramp_down_viol_{g}_{t}')
                obj_primal += ramp_down_viol[g, t-1]
                
                model.addConstr(ramp_down_abs[g, t-1] >= ramp_down_expr, name=f'ramp_down_abs1_{g}_{t}')
                model.addConstr(ramp_down_abs[g, t-1] >= -ramp_down_expr, name=f'ramp_down_abs2_{g}_{t}')

                obj_opt += ramp_up_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_ramp_up'][g, t-1])
                obj_opt += ramp_down_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_ramp_down'][g, t-1])

        # 最小开机时间和最小关机时间约束
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_viol[g, t-1, t1] >= min_on_expr, name=f'min_on_viol_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g, t-1, t1] >= min_on_expr, name=f'min_on_abs1_{g}_{t}_{t1}')
                    model.addConstr(min_on_abs[g, t-1, t1] >= -min_on_expr, name=f'min_on_abs2_{g}_{t}_{t1}')
                    
                    obj_primal += min_on_viol[g, t-1, t1]
                    obj_opt += min_on_abs[g, t-1, t1] * abs(self.lambda_[sample_id]['lambda_min_on'][g, t-1, t1])

        # 最小关机时间约束（与matlab一致）
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
                model.addConstr(start_cost_viol[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1], name=f'start_cost_viol_{g}_{t}')
                model.addConstr(start_cost_abs[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1], name=f'start_cost_abs1_{g}_{t}')
                model.addConstr(start_cost_abs[g, t-1] >= -start_cost[g] * (x[g, t] - x[g, t-1]) + coc[g, t-1], name=f'start_cost_abs2_{g}_{t}')

                obj_primal += start_cost_viol[g, t-1]
                obj_opt += start_cost_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_start_cost'][g, t-1])

                model.addConstr(shut_cost_viol[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1], name=f'shut_cost_viol_{g}_{t}')
                model.addConstr(shut_cost_abs[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1], name=f'shut_cost_abs1_{g}_{t}')
                model.addConstr(shut_cost_abs[g, t-1] >= -shut_cost[g] * (x[g, t-1] - x[g, t]) + coc[g, t-1], name=f'shut_cost_abs2_{g}_{t}')

                obj_primal += shut_cost_viol[g, t-1]
                obj_opt += shut_cost_abs[g, t-1] * abs(self.lambda_[sample_id]['lambda_shut_cost'][g, t-1])

                obj_opt += coc[g, t-1] * abs(self.lambda_[sample_id]['lambda_coc_nonneg'][g, t-1])
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')

        # 潮流约束
        # G: 机组-节点映射矩阵，需用户根据数据准备
        # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr(dcpf_upper_viol[l, t] >= flow[l] - branch_limit[l], name=f'dcpf_upper_viol_{l}_{t}')
                model.addConstr(dcpf_lower_viol[l, t] >= -flow[l] - branch_limit[l], name=f'dcpf_lower_viol_{l}_{t}')

                model.addConstr(dcpf_upper_abs[l, t] >= flow[l] - branch_limit[l], name=f'dcpf_upper_abs1_{l}_{t}')
                model.addConstr(dcpf_upper_abs[l, t] >= -flow[l] + branch_limit[l], name=f'dcpf_upper_abs2_{l}_{t}')
                model.addConstr(dcpf_lower_abs[l, t] >= -flow[l] - branch_limit[l], name=f'dcpf_lower_abs1_{l}_{t}')
                model.addConstr(dcpf_lower_abs[l, t] >= flow[l] + branch_limit[l], name=f'dcpf_lower_abs2_{l}_{t}')

                obj_primal += dcpf_upper_viol[l, t] + dcpf_lower_viol[l, t]
                obj_opt += dcpf_upper_abs[l, t] * abs(self.lambda_[sample_id]['lambda_dcpf_upper'][l, t])
                obj_opt += dcpf_lower_abs[l, t] * abs(self.lambda_[sample_id]['lambda_dcpf_lower'][l, t])

        # 二进制变量偏差
        for g in range(self.ng):
            for t in range(self.T):
                target_value = self.active_set_data[sample_id]['unit_commitment_matrix'][g, t]
                x_dev_expr = x[g, t] - target_value
                model.addConstr(x_binary_dev[g, t] >= x_dev_expr, name=f'x_dev_pos_{g}_{t}')
                model.addConstr(x_binary_dev[g, t] >= -x_dev_expr, name=f'x_dev_neg_{g}_{t}')
                obj_binary += x_binary_dev[g, t]

        # 添加参数化约束的罚项
        if union_analysis is not None and theta_values is not None:
            model, parametric_obj_primal, parametric_obj_opt = self._add_parametric_penalties_pg_block(
                model, x, sample_id, theta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt

        if union_analysis is not None and zeta_values is not None:
            model, parametric_obj_primal, parametric_obj_opt = self._add_parametric_balance_power_penalties_pg_block(
                model, x, sample_id, zeta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt
            
        # 设置目标函数
        total_objective = obj_binary + self.rho_primal * obj_primal + self.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])

            if sample_id <= 1:
                print(f"pg_block, sample_id: {sample_id}, obj_primal: {obj_primal.getValue()}, obj_opt: {obj_opt.getValue()}, obj_binary: {obj_binary.getValue()}")
          
            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None

    def iter_with_dual_block(self, sample_id=0, theta_values=None, union_analysis=None):
        model = gp.Model('iter_with_dual_block')
        model.Params.OutputFlag = 0 
        Pd = self.active_set_data[sample_id]['pd_data']       
        # 功率平衡约束的对偶变量（无符号限制）
        lambda_power_balance = model.addVars(self.T, lb=-GRB.INFINITY, name='lambda_power_balance')
        
        # 发电上下限约束的对偶变量（≥0）
        lambda_pg_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_lower')
        lambda_pg_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_upper')
        
        # 爬坡约束的对偶变量（≥0）
        lambda_ramp_up = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_up')
        lambda_ramp_down = model.addVars(self.ng, self.T-1, lb=0, name='lambda_ramp_down')
        
        # 最小开机/关机时间约束的对偶变量
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        lambda_min_on = model.addVars(self.ng, Ton, self.T, lb=0, name='lambda_min_on')
        lambda_min_off = model.addVars(self.ng, Toff, self.T, lb=0, name='lambda_min_off')
        
        # 启停成本约束的对偶变量
        lambda_start_cost = model.addVars(self.ng, self.T-1, lb=0, name='lambda_start_cost')
        lambda_shut_cost = model.addVars(self.ng, self.T-1, lb=0, name='lambda_shut_cost')
        lambda_coc_nonneg = model.addVars(self.ng, self.T-1, lb=0, name='lambda_coc_nonneg')
        
        # 发电成本约束的对偶变量
        lambda_cpower = model.addVars(self.ng, self.T, lb=0, name='lambda_cpower')
        
        # DCPF潮流约束的对偶变量
        lambda_dcpf_upper = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_upper')
        lambda_dcpf_lower = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_lower')
        
        # x上界的对偶变量
        lambda_x_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_x_upper')
        lambda_x_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_x_lower') 
        
        mu = model.addVars(self.nl, self.T, lb=0, name='mu')  # DCPF约束的对偶变量
        ita_lower = model.addVars(1, self.T, lb=0, name='ita_lower')
        ita_upper = model.addVars(1, self.T, lb=0, name='ita_upper')


        nb = Pd.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量

        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        obj_dual = 0
        obj_opt = 0
                
        # pg变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # 基础项：发电成本系数
                dual_expr = self.gencost[g, -2] / self.T_delta
                
                # 功率平衡约束贡献
                dual_expr -= lambda_power_balance[t]
                
                # 发电上下限约束贡献
                dual_expr -= lambda_pg_lower[g, t]
                dual_expr += lambda_pg_upper[g, t]
                
                # 爬坡约束贡献
                if t > 0:  # 当前时段的爬坡约束
                    dual_expr += lambda_ramp_up[g, t-1]
                    dual_expr -= lambda_ramp_down[g, t-1]
                if t < self.T - 1:  # 下一时段的爬坡约束
                    dual_expr -= lambda_ramp_up[g, t]
                    dual_expr += lambda_ramp_down[g, t]
                
                # DCPF约束贡献
                ptdfg_col = (PTDF @ G[:, g]).T
                for l in range(self.branch.shape[0]):
                    pg_coeff = ptdfg_col[l]
                    dual_expr += pg_coeff * (lambda_dcpf_upper[l, t] - lambda_dcpf_lower[l, t])
                
                dual_expr_pg_abs = model.addVar(lb=0, name=f'dual_expr_abs_pg_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= dual_expr, name=f'dual_expr_abs_pg_pos_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= -dual_expr, name=f'dual_expr_abs_pg_neg_{g}_{t}')
                # 对偶约束：梯度 = 0
                obj_dual += dual_expr_pg_abs

        # x变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # 基础项：固定成本
                dual_expr = self.gencost[g, -1] / self.T_delta
                # x上下界约束贡献
                dual_expr += lambda_x_upper[g, t] - lambda_x_lower[g, t]

                # 发电上下限约束贡献
                dual_expr += self.gen[g, PMIN] * lambda_pg_lower[g, t]
                dual_expr -= self.gen[g, PMAX] * lambda_pg_upper[g, t]
                
                # 爬坡约束贡献
                if t > 0:
                    dual_expr += (Rd_co[g] - Rd[g]) * lambda_ramp_down[g, t-1]
                if t < self.T - 1:
                    dual_expr += (Ru_co[g] - Ru[g]) * lambda_ramp_up[g, t]

                # 最小开机时间约束贡献
                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr += lambda_min_on[g, tau-1, t1]
                        if t == t1:
                            dual_expr -= lambda_min_on[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr -= lambda_min_on[g, tau-1, t1]
                            
                # 最小关机时间约束贡献
                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr -= lambda_min_off[g, tau-1, t1]
                        if t == t1:
                            dual_expr += lambda_min_off[g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr += lambda_min_off[g, tau-1, t1]

                # 启停成本约束贡献
                if t > 0:
                    dual_expr += start_cost[g] * lambda_start_cost[g, t-1]
                    dual_expr -= shut_cost[g] * lambda_shut_cost[g, t-1]
                if t < self.T- 1:
                    dual_expr -= start_cost[g] * lambda_start_cost[g, t]
                    dual_expr += shut_cost[g] * lambda_shut_cost[g, t]

                model, dual_expr_para = self._add_parametric_constraints_dual_block(model, g, t, mu, sample_id, theta_values, union_analysis)

                dual_expr += dual_expr_para

                model, dual_expr_para = self._add_parametric_balance_power_constraints_dual_block(model, g, t, ita_lower, ita_upper, sample_id, theta_values, union_analysis)

                dual_expr += dual_expr_para
                                
                dual_expr_x_abs = model.addVar(lb=0, name=f'dual_expr_abs_x_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= dual_expr, name=f'dual_expr_abs_x_pos_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= -dual_expr, name=f'dual_expr_abs_x_neg_{g}_{t}')
                # 对偶约束：梯度 = 0
                obj_dual += dual_expr_x_abs

        # coc变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T-1):
                dual_expr = 1  # 目标函数中的系数
                dual_expr -= lambda_start_cost[g, t]
                dual_expr -= lambda_shut_cost[g, t]
                dual_expr -= lambda_coc_nonneg[g, t]
                
                dual_expr_coc_abs = model.addVar(lb=0, name=f'dual_expr_abs_coc_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= dual_expr, name=f'dual_expr_abs_coc_pos_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= -dual_expr, name=f'dual_expr_abs_coc_neg_{g}_{t}')
                # 对偶约束：梯度 = 0
                obj_dual += dual_expr_coc_abs

        # cpower变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # dual_expr = 1  # 目标函数中的系数
                # dual_expr -= lambda_cpower[g, t]
                
                # dual_expr_abs = model.addVar(lb=0, name=f'dual_expr_abs_cpower_{g}_{t}')
                # model.addConstr(dual_expr_abs >= dual_expr, name=f'dual_expr_abs_cpower_pos_{g}_{t}')
                # model.addConstr(dual_expr_abs >= -dual_expr, name=f'dual_expr_abs_cpower_neg_{g}_{t}')
                # # 对偶约束：梯度 = 0
                # obj_dual += dual_expr_abs

                model.addConstr(lambda_cpower[g, t] == 1, name=f'dual_expr_cpower_solid_{g}_{t}')

        # 原问题约束 - 所有约束都添加明确的名称
        for t in range(self.T):
            lambda_power_balance_abs = model.addVar(lb=0, name=f'lambda_power_balance_{t}')
            model.addConstr(lambda_power_balance_abs >=  lambda_power_balance[t], name=f'lambda_power_balance_pos_{t}')
            model.addConstr(lambda_power_balance_abs >= -lambda_power_balance[t], name=f'lambda_power_balance_neg_{t}')
            obj_opt += abs(sum(self.pg[sample_id, g, t] for g in range(self.ng)) - np.sum(Pd[:, t])) * lambda_power_balance_abs

            for g in range(self.ng):
                obj_opt += abs(self.pg[sample_id, g, t] - self.gen[g, PMIN] * self.x[sample_id, g, t]) * lambda_pg_lower[g, t]
                obj_opt += abs(self.gen[g, PMAX] * self.x[sample_id, g, t] - self.pg[sample_id, g, t]) * lambda_pg_upper[g, t]

        # 爬坡约束
        for t in range(1, self.T):
            for g in range(self.ng):
                obj_opt += abs(self.pg[sample_id, g, t] - self.pg[sample_id, g, t-1] - (Ru[g] * self.x[sample_id, g, t-1] + Ru_co[g] * (1 - self.x[sample_id, g, t-1]))) * lambda_ramp_up[g, t-1]
                obj_opt += abs(self.pg[sample_id, g, t-1] - self.pg[sample_id, g, t] - (Rd[g] * self.x[sample_id, g, t] + Rd_co[g] * (1 - self.x[sample_id, g, t]))) * lambda_ramp_down[g, t-1]
        # 最小开机时间和最小关机时间约束
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    obj_opt += abs(self.x[sample_id, g, t1+1] - self.x[sample_id, g, t1] - self.x[sample_id, g, t1+t]) * lambda_min_on[g, t-1, t1]
        # 最小关机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    obj_opt += abs(-self.x[sample_id, g, t1+1] + self.x[sample_id, g, t1] - 1 + self.x[sample_id, g, t1+t]) * lambda_min_off[g, t-1, t1]
        # 启停成本
        for t in range(1, self.T):
            for g in range(self.ng):
                obj_opt += abs(self.coc[sample_id, g, t-1]) * lambda_coc_nonneg[g, t-1]
                obj_opt += abs(self.coc[sample_id, g, t-1] - start_cost[g] * (self.x[sample_id, g, t] - self.x[sample_id, g, t-1])) * lambda_start_cost[g, t-1]
                obj_opt += abs(self.coc[sample_id, g, t-1] - shut_cost[g] * (self.x[sample_id, g, t-1] - self.x[sample_id, g, t])) * lambda_shut_cost[g, t-1]
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                obj_opt += abs(self.cpower[sample_id, g, t] - (self.gencost[g, -2]/self.T_delta * self.pg[sample_id, g, t] + self.gencost[g, -1]/self.T_delta * self.x[sample_id, g, t]))
        # 潮流约束
        # G: 机组-节点映射矩阵，需用户根据数据准备
        # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([self.pg[sample_id, g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                obj_opt += abs(flow[l] - branch_limit[l]) * lambda_dcpf_upper[l, t] + abs(flow[l] + branch_limit[l]) * lambda_dcpf_lower[l, t]
        
        for t in range(self.T):
            for g in range(self.ng):
                obj_opt += abs(self.x[sample_id, g, t]) * lambda_x_lower[g, t]
                obj_opt += abs(self.x[sample_id, g, t] - 1) * lambda_x_upper[g, t]    

        model, obj_opt_para = self._add_parametric_obj_dual_block(model, self.x[sample_id, :, :], mu, sample_id, theta_values, union_analysis)
        obj_opt += obj_opt_para
        
        model, obj_opt_para = self._add_parametric_balance_power_obj_dual_block(model, self.x[sample_id, :, :], ita_lower, ita_upper, sample_id, theta_values, union_analysis)
        obj_opt += obj_opt_para
   
        # 设置目标函数
        total_objective = self.rho_dual * obj_dual + self.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.Params.Presolve = 0
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            lambda_sol = {
                # 功率平衡约束对偶变量: shape (T,)
                'lambda_power_balance': np.array([lambda_power_balance[t].X for t in range(self.T)]),
                
                # 发电上下限约束对偶变量: shape (ng, T)
                'lambda_pg_lower': np.array([[lambda_pg_lower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_pg_upper': np.array([[lambda_pg_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                
                # 爬坡约束对偶变量: shape (ng, T-1)
                'lambda_ramp_up': np.array([[lambda_ramp_up[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_ramp_down': np.array([[lambda_ramp_down[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                
                # 最小开关机时间约束对偶变量: shape (ng, Ton, T) 和 (ng, Toff, T)
                'lambda_min_on': np.array([[[lambda_min_on[g, tau, t].X for t in range(self.T)] for tau in range(Ton)] for g in range(self.ng)]),
                'lambda_min_off': np.array([[[lambda_min_off[g, tau, t].X for t in range(self.T)] for tau in range(Toff)] for g in range(self.ng)]),
                
                # 启停成本约束对偶变量: shape (ng, T-1)
                'lambda_start_cost': np.array([[lambda_start_cost[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_shut_cost': np.array([[lambda_shut_cost[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                'lambda_coc_nonneg': np.array([[lambda_coc_nonneg[g, t].X for t in range(self.T-1)] for g in range(self.ng)]),
                
                # 发电成本约束对偶变量: shape (ng, T)
                'lambda_cpower': np.array([[lambda_cpower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                
                # DCPF约束对偶变量: shape (nl, T) - 修正维度
                'lambda_dcpf_upper': np.array([[lambda_dcpf_upper[l, t].X for t in range(self.T)] for l in range(self.nl)]),
                'lambda_dcpf_lower': np.array([[lambda_dcpf_lower[l, t].X for t in range(self.T)] for l in range(self.nl)]),

                
                # x变量上下界约束对偶变量: shape (ng, T) (原来缺失的)
                'lambda_x_upper': np.array([[lambda_x_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_x_lower': np.array([[lambda_x_lower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            }
            
            mu_sol = np.array([[mu[l, t].X for t in range(self.T)] for l in range(self.nl)])  # DCPF约束对偶变量
            
            if sample_id <= 1:            
                print(f"dual_block, sample_id: {sample_id}, obj_dual: {obj_dual.getValue()}, obj_opt: {obj_opt.getValue()}")
            
            return lambda_sol, mu_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None        
                                    
    def iter_with_theta_block(self, union_analysis=None):

        model = gp.Model("iter_with_theta_block")
        model_theta_vars = {}
        model_theta_vars_abs = {}
        for theta_var in self.theta_values:
            var = model.addVar(lb=-1e1, ub=1e1, name=theta_var)
            model_theta_vars[theta_var] = var
            model_theta_vars_abs[theta_var] = model.addVar(lb=0, name=f'{theta_var}_abs')
            model.addConstr(model_theta_vars_abs[theta_var] >= var, name=f'{theta_var}_abs_pos')
            model.addConstr(model_theta_vars_abs[theta_var] >= -var, name=f'{theta_var}_abs_neg')

        model_zeta_vars = {}
        model_zeta_vars_abs = {}
        for zeta_var in self.zeta_values:
            var = model.addVar(lb=-1e1, ub=1e1, name=zeta_var)
            model_zeta_vars[zeta_var] = var
            model_zeta_vars_abs[zeta_var] = model.addVar(lb=0, name=f'{zeta_var}_abs')
            model.addConstr(model_zeta_vars_abs[zeta_var] >= var, name=f'{zeta_var}_abs_pos')
            model.addConstr(model_zeta_vars_abs[zeta_var] >= -var, name=f'{zeta_var}_abs_neg')

        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量

        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        
        obj_primal = 0
        obj_opt = 0
        
        model.update()
        
        for sample_id in range(self.n_samples):
            model, obj_primal_para, obj_opt_para = self._add_variational_primal_theta_block(model, self.x[sample_id, :, :], sample_id, model_theta_vars, union_analysis)
            obj_primal += obj_primal_para
            obj_opt += obj_opt_para

            model, obj_primal_para, obj_opt_para = self._add_variational_balance_power_primal_zeta_block(model, self.x[sample_id, :, :], sample_id, model_zeta_vars, union_analysis)
            obj_primal += obj_primal_para
            obj_opt += obj_opt_para

            model.update()
        
        obj_dual = 0
        
        penalty_factor = 1e-4  # 惩罚因子
        penal_theta = penalty_factor * gp.quicksum(model_theta_vars_abs[theta_var] for theta_var in self.theta_values)

                # x变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # 基础项：固定成本
                dual_expr = self.gencost[g, -1] / self.T_delta * self.lambda_[sample_id]['lambda_cpower'][g, t]

                # x上下界约束贡献
                dual_expr += self.lambda_[sample_id]['lambda_x_upper'][g, t] - self.lambda_[sample_id]['lambda_x_lower'][g, t]

                # 发电上下限约束贡献
                dual_expr += self.gen[g, PMIN] * self.lambda_[sample_id]['lambda_pg_lower'][g, t]
                dual_expr -= self.gen[g, PMAX] * self.lambda_[sample_id]['lambda_pg_upper'][g, t]

                # 爬坡约束贡献
                if t > 0:
                    dual_expr += (Rd_co[g] - Rd[g]) * self.lambda_[sample_id]['lambda_ramp_down'][g, t-1]
                if t < self.T - 1:
                    dual_expr += (Ru_co[g] - Ru[g]) * self.lambda_[sample_id]['lambda_ramp_up'][g, t]

                # 最小开机时间约束贡献
                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr += self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]
                        if t == t1:
                            dual_expr -= self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr -= self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]

                # 最小关机时间约束贡献
                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr -= self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]
                        if t == t1:
                            dual_expr += self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]
                        if t == t1 + tau:
                            dual_expr += self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]

                # 启停成本约束贡献
                if t > 0:
                    dual_expr += start_cost[g] * self.lambda_[sample_id]['lambda_start_cost'][g, t-1]
                    dual_expr -= shut_cost[g] * self.lambda_[sample_id]['lambda_shut_cost'][g, t-1]
                if t < self.T- 1:
                    dual_expr -= start_cost[g] * self.lambda_[sample_id]['lambda_start_cost'][g, t]
                    dual_expr += shut_cost[g] * self.lambda_[sample_id]['lambda_shut_cost'][g, t]

                model, dual_expr_para = self._add_variational_dual_theta_block(model, g, t, self.mu[sample_id, :, :], sample_id, model_theta_vars, union_analysis)
                dual_expr += dual_expr_para

                model, dual_expr_para = self._add_variational_balance_power_dual_zeta_block(model, g, t, self.ita_lower[sample_id, :, :], self.ita_upper[sample_id, :, :],sample_id, model_zeta_vars, union_analysis)
                dual_expr += dual_expr_para
                
                dual_expr_abs = model.addVar(lb=0, name=f'dual_expr_abs_x_{g}_{t}')
                model.addConstr(dual_expr_abs >= dual_expr, name=f'dual_expr_abs_x_pos_{g}_{t}')
                model.addConstr(dual_expr_abs >= -dual_expr, name=f'dual_expr_abs_x_neg_{g}_{t}')
                # 对偶约束：梯度 = 0
                obj_dual += dual_expr_abs
                
        # 设置目标函数
        total_objective = self.rho_primal * obj_primal + self.rho_dual * obj_dual + self.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # 处理最优解
            theta_value_sol = {}
            for theta_var in model_theta_vars:
                theta_value_sol[theta_var] = model_theta_vars[theta_var].X
                
            zeta_value_sol = {}
            for zeta_var in model_zeta_vars:
                zeta_value_sol[zeta_var] = model_zeta_vars[zeta_var].X

            print(f"theta_block, status: {model.status}, obj_primal_theta_part: {obj_primal.getValue()}, obj_dual_theta_part: {obj_dual.getValue()}, obj_opt_theta_part: {obj_opt.getValue()}")
            print(f"zeta_block, status: {model.status}, obj_primal_zeta_part: {obj_primal.getValue()}, obj_dual_zeta_part: {obj_dual.getValue()}, obj_opt_zeta_part: {obj_opt.getValue()}")
            
            return theta_value_sol, zeta_value_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None

    def cal_viol(self, union_analysis=None):        
        obj_primal = 0
        obj_opt = 0
        obj_dual = 0
                
        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']
            pg = self.pg[sample_id, :, :]
            x = self.x[sample_id, :, :]
            coc = self.coc[sample_id, :, :]
            cpower = self.cpower[sample_id, :, :]

            Ton = min(4, self.T)
            Toff = min(4, self.T)

            # 构建约束和目标函数项
     
            # 功率平衡约束
            for t in range(self.T):
                power_balance_expr = sum([pg[g, t] for g in range(self.ng)]) - np.sum(Pd[:, t])
                
                obj_primal += max(0, power_balance_expr)
                
                obj_opt += abs(power_balance_expr) * abs(self.lambda_[sample_id]['lambda_power_balance'][t])
                
                # 发电上下限约束
                for g in range(self.ng):
                    # 下限约束违反
                    pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]                   
                    # 上限约束违反
                    pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                    
                    obj_primal += max(0, pg_lower_expr) + max(0, pg_upper_expr)                    
                    obj_opt += abs(pg_lower_expr) * abs(self.lambda_[sample_id]['lambda_pg_lower'][g, t])
                    obj_opt += abs(pg_upper_expr) * abs(self.lambda_[sample_id]['lambda_pg_upper'][g, t])

                    # x变量约束违反
                    # model.addConstr(x_lower_viol[g, t] >= -x[g, t], name=f'x_lower_viol_{g}_{t}')
                    # model.addConstr(x_upper_viol[g, t] >= x[g, t] - 1, name=f'x_upper_viol_{g}_{t}')
                    # obj_primal += x_lower_viol[g, t] + x_upper_viol[g, t]

                    obj_opt += x[g, t] * abs(self.lambda_[sample_id]['lambda_x_lower'][g, t])
                    obj_opt += (1 - x[g, t]) * abs(self.lambda_[sample_id]['lambda_x_upper'][g, t])

            # 爬坡约束（类似处理）
            Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
            Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
            Ru_co = 0.3 * self.gen[:, PMAX]
            Rd_co = 0.3 * self.gen[:, PMAX]

            for t in range(1, self.T):
                for g in range(self.ng):
                    # 上爬坡约束违反
                    ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                    obj_primal += max(0, ramp_up_expr)

                    # 下爬坡约束违反
                    ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                    obj_primal += max(0, ramp_down_expr)

                    obj_opt += abs(ramp_up_expr) * abs(self.lambda_[sample_id]['lambda_ramp_up'][g, t-1])
                    obj_opt += abs(ramp_down_expr) * abs(self.lambda_[sample_id]['lambda_ramp_down'][g, t-1])

            # 最小开机时间和最小关机时间约束
            # 最小开机时间约束（与matlab一致）
            for g in range(self.ng):
                for t in range(1, Ton+1):
                    for t1 in range(self.T - t):
                        min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]

                        obj_primal += max(0, min_on_expr)
                        obj_opt += abs(min_on_expr) * abs(self.lambda_[sample_id]['lambda_min_on'][g, t-1, t1])

            # 最小关机时间约束（与matlab一致）
            for g in range(self.ng):
                for t in range(1, Toff+1):
                    for t1 in range(self.T - t):
                        min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])

                        obj_primal += max(0, min_off_expr)
                        obj_opt += abs(min_off_expr) * abs(self.lambda_[sample_id]['lambda_min_off'][g, t-1, t1])

            # 启停成本
            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    start_cost_expr = start_cost[g] * (x[g, t] - x[g, t-1]) - coc[g, t-1]

                    obj_primal += max(0, start_cost_expr)
                    obj_opt += abs(start_cost_expr) * abs(self.lambda_[sample_id]['lambda_start_cost'][g, t-1])

                    shut_cost_expr = shut_cost[g] * (x[g, t-1] - x[g, t]) - coc[g, t-1]

                    obj_primal += max(0, shut_cost_expr)
                    obj_opt += abs(shut_cost_expr) * abs(self.lambda_[sample_id]['lambda_shut_cost'][g, t-1])

                    obj_opt += coc[g, t-1] * abs(self.lambda_[sample_id]['lambda_coc_nonneg'][g, t-1])
            # 发电成本
            for t in range(self.T):
                for g in range(self.ng):
                    cpower_expr = self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t] - cpower[g, t]

                    obj_primal += max(0, cpower_expr)
                    obj_opt += abs(cpower_expr) * abs(self.lambda_[sample_id]['lambda_cpower'][g, t])

            # 潮流约束
            # G: 机组-节点映射矩阵，需用户根据数据准备
            # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
            nb = self.bus.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            # 计算PTDF
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]  # 线路容量
            for t in range(self.T):
                flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
                for l in range(self.branch.shape[0]):
                    dcpf_upper_expr = flow[l] - branch_limit[l]
                    dcpf_lower_expr = -flow[l] - branch_limit[l]

                    obj_primal += max(0, dcpf_upper_expr) + max(0, dcpf_lower_expr)
                    obj_opt += abs(dcpf_upper_expr) * abs(self.lambda_[sample_id]['lambda_dcpf_upper'][l, t])
                    obj_opt += abs(dcpf_lower_expr) * abs(self.lambda_[sample_id]['lambda_dcpf_lower'][l, t])

            parametric_obj_primal, parametric_obj_opt = self._add_parametric_penalties_pg_block_const(
                x, sample_id, self.theta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt

            parametric_obj_primal, parametric_obj_opt = self._add_parametric_balance_power_penalties_pg_block_const(
                x, sample_id, self.zeta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt
                    
            # pg变量的对偶约束
            for g in range(self.ng):
                for t in range(self.T):
                    # 基础项：发电成本系数
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

                    # DCPF约束贡献
                    ptdfg_col = (PTDF @ G[:, g]).T
                    for l in range(self.branch.shape[0]):
                        pg_coeff = ptdfg_col[l]
                        dual_expr += pg_coeff * (self.lambda_[sample_id]['lambda_dcpf_upper'][l, t] - self.lambda_[sample_id]['lambda_dcpf_lower'][l, t])
                    
                    # 对偶约束：梯度 = 0
                    obj_dual += abs(dual_expr)

            # x变量的对偶约束
            for g in range(self.ng):
                for t in range(self.T):
                    # 基础项：固定成本
                    dual_expr = self.gencost[g, -1] / self.T_delta * self.lambda_[sample_id]['lambda_cpower'][g, t]

                    # x上下界约束贡献
                    dual_expr += self.lambda_[sample_id]['lambda_x_upper'][g, t] - self.lambda_[sample_id]['lambda_x_lower'][g, t]

                    # 发电上下限约束贡献
                    dual_expr += self.gen[g, PMIN] * self.lambda_[sample_id]['lambda_pg_lower'][g, t]
                    dual_expr -= self.gen[g, PMAX] * self.lambda_[sample_id]['lambda_pg_upper'][g, t]

                    # 爬坡约束贡献
                    if t > 0:
                        dual_expr += (Rd_co[g] - Rd[g]) * self.lambda_[sample_id]['lambda_ramp_down'][g, t-1]
                    if t < self.T - 1:
                        dual_expr += (Ru_co[g] - Ru[g]) * self.lambda_[sample_id]['lambda_ramp_up'][g, t]

                    # 最小开机时间约束贡献
                    for tau in range(1, Ton + 1):
                        for t1 in range(self.T - tau):
                            if t == t1 + 1:
                                dual_expr += self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]
                            if t == t1:
                                dual_expr -= self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]
                            if t == t1 + tau:
                                dual_expr -= self.lambda_[sample_id]['lambda_min_on'][g, tau-1, t1]

                    # 最小关机时间约束贡献
                    for tau in range(1, Toff + 1):
                        for t1 in range(self.T - tau):
                            if t == t1 + 1:
                                dual_expr -= self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]
                            if t == t1:
                                dual_expr += self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]
                            if t == t1 + tau:
                                dual_expr += self.lambda_[sample_id]['lambda_min_off'][g, tau-1, t1]

                    # 启停成本约束贡献
                    if t > 0:
                        dual_expr += start_cost[g] * self.lambda_[sample_id]['lambda_start_cost'][g, t-1]
                        dual_expr -= shut_cost[g] * self.lambda_[sample_id]['lambda_shut_cost'][g, t-1]
                    if t < self.T- 1:
                        dual_expr -= start_cost[g] * self.lambda_[sample_id]['lambda_start_cost'][g, t]
                        dual_expr += shut_cost[g] * self.lambda_[sample_id]['lambda_shut_cost'][g, t]

                    dual_expr_para = self._add_parametric_constraints_dual_block_const(g, t, self.mu[sample_id, :, :], sample_id, self.theta_values, union_analysis)
                    dual_expr += dual_expr_para

                    dual_expr_para = self._add_parametric_balance_power_constraints_dual_block_const(g, t, self.ita_lower[sample_id, :, :], self.ita_upper[sample_id, :, :], sample_id, self.zeta_values, union_analysis)
                    dual_expr += dual_expr_para

                    # 对偶约束：梯度 = 0
                    obj_dual += abs(dual_expr)
        
            # coc变量的对偶约束
            for g in range(self.ng):
                for t in range(self.T-1):
                    dual_expr = 1  # 目标函数中的系数
                    dual_expr -= self.lambda_[sample_id]['lambda_start_cost'][g, t]
                    dual_expr -= self.lambda_[sample_id]['lambda_shut_cost'][g, t]
                    dual_expr -= self.lambda_[sample_id]['lambda_coc_nonneg'][g, t]

                    obj_dual += abs(dual_expr)

            # cpower变量的对偶约束
            for g in range(self.ng):
                for t in range(self.T):
                    dual_expr = 1  # 目标函数中的系数
                    dual_expr -= self.lambda_[sample_id]['lambda_cpower'][g, t]
                    
                    # 对偶约束：梯度 = 0
                    obj_dual += abs(dual_expr)
                    
        return obj_primal, obj_dual, obj_opt

    def iter(self, max_iter=20, union_analysis=None):
        if union_analysis is None:
            union_analysis = self._current_union_analysis
        for i in range(max_iter):
            print(f"🔄 迭代 {i+1}/{max_iter} 开始")
            # 迭代PG块
            for sample_id in range(self.n_samples):
                pg_sol, x_sol, cpower_sol, coc_sol = self.iter_with_pg_block(sample_id=sample_id, theta_values=self.theta_values, union_analysis=union_analysis)
                if pg_sol is None:
                    print("❌ PG块迭代失败，终止迭代")
                    break
                self.pg[sample_id, :, :] = pg_sol
                self.x[sample_id, :, :] = x_sol
                self.cpower[sample_id, :, :] = cpower_sol
                self.coc[sample_id, :, :] = coc_sol

            # 迭代对偶块
            for sample_id in range(self.n_samples):
                lambda_sol, mu_sol = self.iter_with_dual_block(sample_id=sample_id, theta_values=self.theta_values, union_analysis=union_analysis)
                if lambda_sol is None or mu_sol is None:
                    print("❌ 对偶块迭代失败，终止迭代")
                    break
                self.lambda_[sample_id] = lambda_sol
                self.mu[sample_id, :, :] = mu_sol

            # 迭代theta块
            theta_value_sol, zeta_value_sol = self.iter_with_theta_block(union_analysis=union_analysis)
            if theta_value_sol is None or zeta_value_sol is None:
                print("❌ Theta块迭代失败，终止迭代")
                break
            self.theta_values = theta_value_sol
            self.zeta_values = zeta_value_sol
            # print(self.theta_values)
            # print(self.zeta_values)
            # print(self.mu)
            print(f"✅ 迭代 {i+1}/{max_iter} 成功")
            
            # 递增\rho
            obj_primal, obj_dual, obj_opt = self.cal_viol(union_analysis=union_analysis)
            print(f'obj_primal:{obj_primal}, obj_dual:{obj_dual}, obj_opt:{obj_opt}')
            self.rho_primal += self.gamma * obj_primal
            self.rho_dual += self.gamma * obj_dual
            self.rho_opt += self.gamma * obj_opt
            print(f"当前惩罚参数: ρ_primal={self.rho_primal}, ρ_dual={self.rho_dual}, ρ_opt={self.rho_opt}")

            time.sleep(1)
            pass
        self.save_theta_values(f'result/theta_values_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        return self.theta_values

    def solve_LP_with_theta_constraints(self, sample_id, theta_values, zeta_values, union_analysis=None):
        # 此函数用于在给定theta约束下求解LP问题
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('iter_with_pg_block')
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')        
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 最小开关机时间
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        # 构建约束和目标函数项
        obj = 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_expr == 0, name=f'power_balance_{t}')
            
            # 发电上下限约束
            for g in range(self.ng):
                # 下限约束违反
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_expr <= 0, name=f'pg_lower_viol_{g}_{t}')
                
                # 上限约束违反
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_expr <= 0, name=f'pg_upper_viol_{g}_{t}')

        # 爬坡约束（类似处理）
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                # 上爬坡约束违反
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_expr <= 0, name=f'ramp_up_viol_{g}_{t}')

                # 下爬坡约束违反
                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_expr <= 0, name=f'ramp_down_viol_{g}_{t}')

        # 最小开机时间和最小关机时间约束
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_expr <= 0, name=f'min_on_viol_{g}_{t}_{t1}')

        # 最小关机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                    model.addConstr(min_off_expr <= 0, name=f'min_off_viol_{g}_{t}_{t1}')

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(start_cost[g] * (x[g, t] - x[g, t-1]) <= coc[g, t-1], name=f'start_cost_viol_{g}_{t}')
                model.addConstr(shut_cost[g] * (x[g, t-1] - x[g, t]) <= coc[g, t-1], name=f'shut_cost_viol_{g}_{t}')
                
                obj += coc[g, t-1]

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')
                
                obj += cpower[g, t]

        # 潮流约束
        # G: 机组-节点映射矩阵，需用户根据数据准备
        # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr( flow[l] - branch_limit[l] <= 0, name=f'dcpf_upper_viol_{l}_{t}')
                model.addConstr(-flow[l] - branch_limit[l] <= 0, name=f'dcpf_lower_viol_{l}_{t}')

        # 添加参数化约束的罚项
        if union_analysis is not None and theta_values is not None:
            model, parametric_obj_primal, parametric_obj_opt = self._add_parametric_penalties_pg_block(
                model, x, sample_id, theta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt

        if union_analysis is not None and zeta_values is not None:
            model, parametric_obj_primal, parametric_obj_opt = self._add_parametric_balance_power_penalties_pg_block(
                model, x, sample_id, zeta_values, union_analysis
            )
            obj_primal += parametric_obj_primal
            obj_opt += parametric_obj_opt

        # 设置目标函数
        model.setObjective(obj, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])

            if sample_id <= 1:
                print(f"pg_block, sample_id: {sample_id}, obj: {model.ObjVal}")

            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None        

    def solve_MILP(self, sample_id, union_analysis=None):
        # 此函数用于在给定theta约束下求解MILP问题
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('solve_MILP')
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.BINARY, lb=0, ub=1, name='x')        
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 最小开关机时间
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        # 构建约束和目标函数项
        obj = 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_expr == 0, name=f'power_balance_{t}')
            
            # 发电上下限约束
            for g in range(self.ng):
                # 下限约束违反
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_expr <= 0, name=f'pg_lower_viol_{g}_{t}')
                
                # 上限约束违反
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_expr <= 0, name=f'pg_upper_viol_{g}_{t}')

        # 爬坡约束（类似处理）
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                # 上爬坡约束违反
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_expr <= 0, name=f'ramp_up_viol_{g}_{t}')

                # 下爬坡约束违反
                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_expr <= 0, name=f'ramp_down_viol_{g}_{t}')

        # 最小开机时间和最小关机时间约束
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_expr <= 0, name=f'min_on_viol_{g}_{t}_{t1}')

        # 最小关机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                    model.addConstr(min_off_expr <= 0, name=f'min_off_viol_{g}_{t}_{t1}')

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(start_cost[g] * (x[g, t] - x[g, t-1]) <= coc[g, t-1], name=f'start_cost_viol_{g}_{t}')
                model.addConstr(shut_cost[g] * (x[g, t-1] - x[g, t]) <= coc[g, t-1], name=f'shut_cost_viol_{g}_{t}')
                
                obj += coc[g, t-1]

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')
                
                obj += cpower[g, t]

        # 潮流约束
        # G: 机组-节点映射矩阵，需用户根据数据准备
        # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr( flow[l] - branch_limit[l] <= 0, name=f'dcpf_upper_viol_{l}_{t}')
                model.addConstr(-flow[l] - branch_limit[l] <= 0, name=f'dcpf_lower_viol_{l}_{t}')

        # 设置目标函数
        model.setObjective(obj, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])

            if sample_id <= 1:
                print(f"pg_block, sample_id: {sample_id}, obj: {model.ObjVal}")

            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None   

    def solve_LP_without_theta_constraints(self, sample_id, union_analysis=None):
        # 此函数用于在给定theta约束下求解LP问题
        Pd = self.active_set_data[sample_id]['pd_data']
        model = gp.Model('iter_with_pg_block')
        model.Params.OutputFlag = 0
        
        # 主要变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')        
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 最小开关机时间
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        
        # 构建约束和目标函数项
        obj = 0
        
        # 功率平衡约束
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_expr == 0, name=f'power_balance_{t}')
            
            # 发电上下限约束
            for g in range(self.ng):
                # 下限约束违反
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_expr <= 0, name=f'pg_lower_viol_{g}_{t}')
                
                # 上限约束违反
                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_expr <= 0, name=f'pg_upper_viol_{g}_{t}')

        # 爬坡约束（类似处理）
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                # 上爬坡约束违反
                ramp_up_expr = pg[g, t] - pg[g, t-1] - Ru[g] * x[g, t-1] - Ru_co[g] * (1 - x[g, t-1])
                model.addConstr(ramp_up_expr <= 0, name=f'ramp_up_viol_{g}_{t}')

                # 下爬坡约束违反
                ramp_down_expr = pg[g, t-1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_expr <= 0, name=f'ramp_down_viol_{g}_{t}')

        # 最小开机时间和最小关机时间约束
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1+1] - x[g, t1] - x[g, t1+t]
                    model.addConstr(min_on_expr <= 0, name=f'min_on_viol_{g}_{t}_{t1}')

        # 最小关机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1+1] + x[g, t1] - (1 - x[g, t1+t])
                    model.addConstr(min_off_expr <= 0, name=f'min_off_viol_{g}_{t}_{t1}')

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                model.addConstr(start_cost[g] * (x[g, t] - x[g, t-1]) <= coc[g, t-1], name=f'start_cost_viol_{g}_{t}')
                model.addConstr(shut_cost[g] * (x[g, t-1] - x[g, t]) <= coc[g, t-1], name=f'shut_cost_viol_{g}_{t}')
                
                obj += coc[g, t-1]

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(cpower[g, t] == self.gencost[g, -2]/self.T_delta * pg[g, t] + self.gencost[g, -1]/self.T_delta * x[g, t], name=f'cpower_{g}_{t}')
                
                obj += cpower[g, t]

        # 潮流约束
        # G: 机组-节点映射矩阵，需用户根据数据准备
        # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
        nb = self.bus.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr( flow[l] - branch_limit[l] <= 0, name=f'dcpf_upper_viol_{l}_{t}')
                model.addConstr(-flow[l] - branch_limit[l] <= 0, name=f'dcpf_lower_viol_{l}_{t}')

        # 设置目标函数
        model.setObjective(obj, GRB.MINIMIZE)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T-1)] for g in range(self.ng)])

            if sample_id <= 1:
                print(f"pg_block, sample_id: {sample_id}, obj: {model.ObjVal}")

            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"❌ 模型求解失败，状态: {model.status}")
            return None   
                
    def heuristic_sol_x_rounding(self, sample_id, x_LP):
        # 对x进行四舍五入
        x_rounded = np.round(x_LP)
        return x_rounded
    
    def compare_x_heu_and_x_int(self, sample_id, x_heu):
        x_int = self.active_set_data[sample_id]['unit_commitment_matrix']
        
        # print(f'x_int for sample_id {sample_id}:\n', x_int)
        
        x_int = np.round(x_int).astype(int)
        x_heu = np.round(x_heu).astype(int)

        # 使用 numpy 的逻辑运算（并注意括号，避免运算符优先级问题）
        x_is_int_not_heu = np.logical_and(x_heu == 0, x_int == 1).astype(int)
        x_is_heu_not_int = np.logical_and(x_heu == 1, x_int == 0).astype(int)
        x_commen = np.logical_and(x_heu == 1, x_int == 1).astype(int)
        
        ### heatmap ###
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sns.heatmap(x_is_int_not_heu, cmap="Reds", cbar=False, annot=False, ax=ax, alpha=0.8)
        sns.heatmap(x_is_heu_not_int, cmap="Blues", cbar=False, annot=False, ax=ax, alpha=0.6)
        sns.heatmap(x_commen, cmap="Greens", cbar=False, annot=False, ax=ax, alpha=0.6)
        plt.title(f"Comparison of Heuristic and Integer x for sample_id: {sample_id}")
        
        # 生成图例（简单处理）
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Integer Only')
        blue_patch = mpatches.Patch(color='blue', label='Heuristic Only')
        green_patch = mpatches.Patch(color='green', label='Common')
        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right')
        plt.show() 

    def analyse_surrogate_model(self):
        typical_sample_id = [0]
        for sample_id in typical_sample_id:
            sol_LP = self.solve_LP_without_theta_constraints(sample_id)
            x_LP = sol_LP[1]
            sol_LP_refined = self.solve_LP_with_theta_constraints(sample_id, self.theta_values, self.zeta_values)
            x_LP_refined = sol_LP_refined[1]
            
            # sol_MILP = self.solve_MILP(sample_id)
            # x_MILP = sol_MILP[1]
            
            print(f"x_LP (without theta constraints) for sample_id {sample_id}:\n", x_LP)
            print("-"*50)
            print(f"x_LP_refined (with theta constraints) for sample_id {sample_id}:\n", x_LP_refined)
            
            
            # print("-"*50)
            # print(f"x_MILP for sample_id {sample_id}:\n", x_MILP.astype(int))

                  
            x_heu = self.heuristic_sol_x_rounding(sample_id, x_LP)
            self.compare_x_heu_and_x_int(sample_id, x_heu)

    def analyse_surrogate_model_totle(self):
        differ_LP = 0
        differ_LP_refined = 0
        for sample_id in range(self.n_samples):
            sol_LP = self.solve_LP_without_theta_constraints(sample_id)
            x_LP = sol_LP[1]
            sol_LP_refined = self.solve_LP_with_theta_constraints(sample_id, self.theta_values, self.zeta_values)
            x_LP_refined = sol_LP_refined[1]
            
            differ_LP += np.sum(np.abs(x_LP - self.active_set_data[sample_id]['unit_commitment_matrix']))
            differ_LP_refined += np.sum(np.abs(x_LP_refined - self.active_set_data[sample_id]['unit_commitment_matrix']))
            
        print(f"最优间隙（不含theta约束）: {differ_LP}")
        print(f"最优间隙（含theta约束）: {differ_LP_refined}")      
            


    def save_theta_values(self, filepath: str, ensure_dir: bool = True) -> None:
        """
        将 self.theta_values 保存为 JSON 文件。

        Args:
            filepath: 要保存的文件路径（例如 'result/theta_values.json'）
            ensure_dir: 如果为 True，则在保存前创建目标目录（默认 True）
        """
        import json, os

        if not hasattr(self, 'theta_values') or self.theta_values is None:
            raise RuntimeError("theta_values 未初始化，无法保存。")

        if ensure_dir:
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        # 将可能的 numpy 类型转换为 Python 原生类型
        serializable = {str(k): float(v) for k, v in self.theta_values.items()}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'theta_values': serializable}, f, indent=2, ensure_ascii=False)

        print(f"✓ theta_values 已保存到: {filepath}")

    def load_theta_values(self, filepath: str) -> dict:
        """
        从 JSON 文件加载 theta_values 并赋值给 self.theta_values。

        Args:
            filepath: 包含 theta_values 的 JSON 文件路径

        Returns:
            加载并设置后的 theta_values 字典
        """
        import json, os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"theta 文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tv = data.get('theta_values', {})
        # 保证值为 float
        self.theta_values = {str(k): float(v) for k, v in tv.items()}

        print(f"✓ theta_values 已从文件加载: {filepath}，变量数量: {len(self.theta_values)}")
        return self.theta_values           
            
        
if __name__ == "__main__":
    json_file = "result/active_sets_20251101_013925.json"
    
    active_set_data = load_active_set_from_json(json_file)
    
    active_set_data = active_set_data[:3]

    ppc = pypower.case9.case9()
    ppc['branch'][:, pypower.idx_brch.RATE_A] = ppc['branch'][:, pypower.idx_brch.RATE_A]
    T_delta = 1
    # 创建模型对象

    iter_bcd = Iter_BCD(ppc, active_set_data=active_set_data, T_delta=T_delta)
    iter_bcd.iter(max_iter=100)  # 运行迭代
    # iter_bcd.load_theta_values('result/theta_values_final.json')
    iter_bcd.analyse_surrogate_model_totle()