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
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, \
ANGMIN, ANGMAX
from pathlib import Path
import io
import sys
import re
import json
from typing import Dict, List, Tuple, Optional

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_active_set_from_json(json_filepath: str, sample_id: int = 0) -> Dict:
    """
    从JSON文件加载活动集数据
    
    Args:
        json_filepath: JSON文件路径
        sample_id: 要加载的样本ID
        
    Returns:
        包含活动约束、变量和Pd数据的字典
    """
    reader = ActiveSetReader(json_filepath)
    active_constraints, active_variables, pd_data = reader.extract_active_constraints_and_variables(sample_id)
    unit_commitment = reader.get_unit_commitment_matrix(sample_id)
    
    print(f"=== 加载活动集数据 (样本 {sample_id}) ===")

    return {
        'active_constraints': active_constraints,
        'active_variables': active_variables,
        'pd_data': pd_data,
        'unit_commitment_matrix': unit_commitment,
        'sample_id': sample_id
    }
    
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

class UnitCommitmentModel:
    def __init__(self, ppc, Pd, T_delta):
        self.ppc = ppc
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.Pd = Pd
        self.T_delta = T_delta
        self.T = Pd.shape[1]
        self.ng = self.gen.shape[0]
        self.nb = self.branch.shape[0]
        
        # 建立约束编号映射
        self._build_constraint_mapping()
        
        self.model = gp.Model('UnitCommitment')
        self.model.Params.OutputFlag = 0
        self.pg = self.model.addVars(self.ng, self.T, lb=0, name='pg')
        self.x = self.model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
        self.coc = self.model.addVars(self.ng, self.T-1, lb=0, name='coc')
        self.cpower = self.model.addVars(self.ng, self.T, lb=0, name='cpower')
        self._build_model()
    
    def _build_constraint_mapping(self):
        """建立约束编号到约束类型的映射"""
        self.constraint_map = {}
        constraint_id = 0
        
        # 有功平衡约束
        for t in range(self.T):
            self.constraint_map[constraint_id] = {
                'type': 'power_balance',
                'time_slot': t,
                'constraint_name': f'power_balance_{t}'
            }
            constraint_id += 1
        
        # 发电约束 (下限和上限)
        for t in range(self.T):
            for g in range(self.ng):
                # 下限约束
                self.constraint_map[constraint_id] = {
                    'type': 'gen_min',
                    'generator': g,
                    'time_slot': t,
                    'constraint_name': f'gen_min_{g}_{t}'
                }
                constraint_id += 1
                
                # 上限约束
                self.constraint_map[constraint_id] = {
                    'type': 'gen_max',
                    'generator': g,
                    'time_slot': t,
                    'constraint_name': f'gen_max_{g}_{t}'
                }
                constraint_id += 1
        
        # 爬坡约束
        for t in range(1, self.T):
            for g in range(self.ng):
                # 上爬坡约束
                self.constraint_map[constraint_id] = {
                    'type': 'ramp_up',
                    'generator': g,
                    'time_slot': t,
                    'constraint_name': f'ramp_up_{g}_{t}'
                }
                constraint_id += 1
                
                # 下爬坡约束
                self.constraint_map[constraint_id] = {
                    'type': 'ramp_down',
                    'generator': g,
                    'time_slot': t,
                    'constraint_name': f'ramp_down_{g}_{t}'
                }
                constraint_id += 1
        
        # 发电成本约束
        for t in range(self.T):
            for g in range(self.ng):
                self.constraint_map[constraint_id] = {
                    'type': 'power_cost',
                    'generator': g,
                    'time_slot': t,
                    'constraint_name': f'power_cost_{g}_{t}'
                }
                constraint_id += 1
        
        # DCPF潮流约束 - 这是重点！
        try:
            self.dcpf_constraint_start_id = constraint_id
            for t in range(self.T):
                for l in range(self.branch.shape[0]):
                    # 上限约束
                    self.constraint_map[constraint_id] = {
                        'type': 'dcpf_upper',
                        'branch': l,
                        'time_slot': t,
                        'constraint_name': f'dcpf_upper_{l}_{t}'
                    }
                    constraint_id += 1
                    
                    # 下限约束
                    self.constraint_map[constraint_id] = {
                        'type': 'dcpf_lower',
                        'branch': l,
                        'time_slot': t,
                        'constraint_name': f'dcpf_lower_{l}_{t}'
                    }
                    constraint_id += 1
            
            self.dcpf_constraint_end_id = constraint_id - 1
            print(f"DCPF约束ID范围: {self.dcpf_constraint_start_id} - {self.dcpf_constraint_end_id}")
        except:
            self.dcpf_constraint_start_id = None
            self.dcpf_constraint_end_id = None
    
    def identify_active_dcpf_constraints(self, active_constraints: List[Dict]) -> List[Dict]:
        """
        识别起作用的DCPF约束
        
        Args:
            active_constraints: 活动约束列表
            
        Returns:
            起作用的DCPF约束列表
        """
        active_dcpf_constraints = []
        
        for constraint in active_constraints:
            constraint_id = constraint.get('constraint_id')
            
            if constraint_id is not None and isinstance(constraint_id, int):
                # 检查是否在DCPF约束范围内
                if (self.dcpf_constraint_start_id is not None and 
                    self.dcpf_constraint_start_id <= constraint_id <= self.dcpf_constraint_end_id):
                    
                    # 从映射中获取DCPF约束信息
                    constraint_info = self.constraint_map.get(constraint_id, {})
                    
                    if constraint_info.get('type') in ['dcpf_upper', 'dcpf_lower']:
                        active_dcpf_constraints.append({
                            'constraint_id': constraint_id,
                            'constraint_type': constraint_info['type'],
                            'branch_id': constraint_info['branch'],
                            'time_slot': constraint_info['time_slot'],
                            'dual_value': constraint.get('dual_value'),
                            'constraint_name': constraint_info['constraint_name'],
                            'original_constraint': constraint
                        })
        
        return active_dcpf_constraints
    
    def get_active_dcpf_constraints_interface(self, json_filepath: str, sample_id: int = 0) -> List[Dict]:
        """
        从JSON文件识别起作用的DCPF约束的接口
        
        Args:
            json_filepath: JSON文件路径
            sample_id: 样本ID
            
        Returns:
            起作用的DCPF约束列表
        """
        # 加载活动集数据
        active_set_data = load_active_set_from_json(json_filepath, sample_id)
        active_constraints = active_set_data['active_constraints']
        
        # 识别DCPF约束
        active_dcpf_constraints = self.identify_active_dcpf_constraints(active_constraints)
        
        print(f"\n=== 起作用的DCPF约束分析 ===")
        print(f"总活动约束数: {len(active_constraints)}")
        print(f"起作用DCPF约束数: {len(active_dcpf_constraints)}")
        
        if active_dcpf_constraints:
            print("\n起作用的DCPF约束详情:")
            for i, dcpf_constraint in enumerate(active_dcpf_constraints):
                print(f"  {i+1}. {dcpf_constraint['constraint_name']}")
                print(f"     支路: {dcpf_constraint['branch_id']}, 时段: {dcpf_constraint['time_slot']}")
                print(f"     类型: {dcpf_constraint['constraint_type']}")
                if dcpf_constraint['dual_value'] is not None:
                    print(f"     对偶值: {dcpf_constraint['dual_value']}")
                print()
        else:
            print("未找到起作用的DCPF约束")
        
        return active_dcpf_constraints

    def _build_model(self):
        # 有功平衡
        for t in range(self.T):
            self.model.addConstr(gp.quicksum(self.pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]), name=f'power_balance_{t}')
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.model.addConstr(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])
        
        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.model.addConstr(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        
        # 最小开机时间和最小关机时间约束
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    self.model.addConstr(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t])
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    self.model.addConstr(-self.x[g, t1+1] + self.x[g, t1] <= 1 - self.x[g, t1+t])
        
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.coc[g, t-1] >= start_cost[g] * (self.x[g, t] - self.x[g, t-1]))
                self.model.addConstr(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]))
                self.model.addConstr(self.coc[g, t-1] >= 0)
        
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addConstr(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
        
        # 潮流约束
        try:
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            for t in range(self.T):
                flow = PTDF @ (G @ np.array([self.pg[g, t] for g in range(self.ng)]) - self.Pd[:, t])
                for l in range(self.branch.shape[0]):
                    self.model.addConstr(flow[l] <= branch_limit[l])
                    self.model.addConstr(flow[l] >= -branch_limit[l])
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')

        # 目标函数
        obj = gp.quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T)) \
            + gp.quicksum(self.coc[g, t] for g in range(self.ng) for t in range(self.T-1))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        self.model.setParam("Presolve", 2)

    def solve(self):
        self.model.Params.OutputFlag = 0
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[self.x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {self.model.objVal}")
            return pg_sol, x_sol, self.model.objVal
        else:
            print("未找到最优解")
            return None, None, None

if __name__ == "__main__":
    # 方法1: 直接读取JSON文件分析
    json_file = "result/active_sets_20250803_025349.json"  # 替换为您的JSON文件路径
    
    try:
        sample_id = 12
        active_set_data = load_active_set_from_json(json_file, sample_id=sample_id)

        Pd = active_set_data['pd_data']
        ppc = pypower.case9.case9()
        ppc['branch'][:, pypower.idx_brch.RATE_A] = ppc['branch'][:, pypower.idx_brch.RATE_A] * 0.45
        T_delta = 1
        # 创建模型对象
        uc = UnitCommitmentModel(ppc, Pd, T_delta)
        
        # 使用接口识别起作用的DCPF约束
        active_dcpf_constraints = uc.get_active_dcpf_constraints_interface(json_file, sample_id=sample_id)

        print(f"\n=== 活动集数据摘要 ===")
        print(f"样本ID: {active_set_data['sample_id']}")
        print(f"起作用约束数: {len(active_set_data['active_constraints'])}")
        print(f"活动变量数: {len(active_set_data['active_variables'])}")
        print(f"起作用DCPF约束数: {len(active_dcpf_constraints)}")

        # 方法3: 普通求解用于对比
        pg_sol, x_sol, total_cost = uc.solve()
        
        if pg_sol is not None:
            print(f"\n求解成功，总成本: {total_cost}")
        else:
            print("未找到可行解")
            
    except FileNotFoundError:
        print(f"JSON文件未找到: {json_file}")
        print("请确保已运行ActiveSetLearner并生成了JSON文件")
        
        # 如果没有JSON文件，执行普通求解
        load_df = pd.read_csv('src/load.csv', header=None)
        Pd = load_df.values
        ppc = pypower.case9.case9()
        T_delta = 1
        uc = UnitCommitmentModel(ppc, Pd, T_delta)
        
        pg_sol, x_sol, total_cost = uc.solve()
        if pg_sol is not None:
            print("普通求解成功")
        else:
            print("未找到可行解")


