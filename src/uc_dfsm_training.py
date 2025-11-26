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
        # self.x = self.model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
        self.x = self.model.addVars(self.ng, self.T, lb=0, ub=1, name='x')
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
        
        # if active_dcpf_constraints:
        #     print("\n起作用的DCPF约束详情:")
        #     for i, dcpf_constraint in enumerate(active_dcpf_constraints):
        #         print(f"  {i+1}. {dcpf_constraint['constraint_name']}")
        #         print(f"     支路: {dcpf_constraint['branch_id']}, 时段: {dcpf_constraint['time_slot']}")
        #         print(f"     类型: {dcpf_constraint['constraint_type']}")
        #         if dcpf_constraint['dual_value'] is not None:
        #             print(f"     对偶值: {dcpf_constraint['dual_value']}")
        #         print()
        # else:
        #     print("未找到起作用的DCPF约束")
        
        return active_dcpf_constraints

    def get_active_dcpf_constraints_from_solution(self) -> List[Dict]:
        """
        从当前求解结果中提取起作用的DCPF约束
        
        Returns:
            起作用的DCPF约束列表
        """
        if self.model.status != GRB.OPTIMAL:
            print("模型未求解到最优解，无法分析约束")
            return []
        
        # 获取所有约束
        constraints = self.model.getConstrs()
        active_dcpf_constraints = []
        total_active_constraints = 0  # 统计总的起作用约束
        
        # 分析每个约束
        for constr in constraints:
            try:
                dual_value = constr.Pi  # Gurobi中的对偶值
                constraint_name = constr.ConstrName  # 约束名称
                
                is_active = abs(dual_value) > 1e-6
                
                if is_active:
                    total_active_constraints += 1  # 统计所有起作用的约束
                    
                    # 通过约束名称识别DCPF约束
                    if constraint_name.startswith('dcpf_'):
                        # 解析约束名称获取信息
                        parts = constraint_name.split('_')
                        if len(parts) >= 4:
                            constraint_type = f"dcpf_{parts[1]}"
                            branch_id = int(parts[2])
                            time_slot = int(parts[3])
                            
                            active_dcpf_constraints.append({
                                'constraint_name': constraint_name,
                                'constraint_type': constraint_type,
                                'branch_id': branch_id,
                                'time_slot': time_slot,
                                'dual_value': dual_value,
                                'constraint_object': constr
                            })
            except Exception as e:
                print(f"分析约束 {constr.ConstrName} 时出错: {e}")
        
        # 按支路和时段排序
        active_dcpf_constraints.sort(key=lambda x: (x['branch_id'], x['time_slot'], x['constraint_type']))
        
        print(f"\n=== 当前求解中起作用的DCPF约束 ===")
        print(f"总约束数: {len(constraints)}")
        print(f"起作用约束总数: {total_active_constraints}")
        print(f"总计: {len(active_dcpf_constraints)} 个DCPF约束起作用")
        
        return active_dcpf_constraints

    def get_union_constraints_and_fractional_info(self, json_filepath: str, sample_id: int = 0) -> Dict:
        """
        获取JSON和当前求解结果中起作用约束的并集，
        并找到与x_sol非整数变量组成的约束的系数和右端项

        Args:
            json_filepath: JSON文件路径
            sample_id: 样本ID

        Returns:
            包含并集约束和非整数变量约束信息的字典
        """
        # 获取JSON中的起作用约束
        json_dcpf_constraints = self.get_active_dcpf_constraints_interface(json_filepath, sample_id)

        # 获取当前求解结果中的起作用约束
        current_dcpf_constraints = self.get_active_dcpf_constraints_from_solution()

        # 创建约束标识集合 (branch_id, time_slot, constraint_type)
        json_constraint_set = set()
        for constraint in json_dcpf_constraints:
            branch_id = constraint.get('branch_id')
            time_slot = constraint.get('time_slot')
            constraint_type = constraint.get('constraint_type')
            if all([branch_id is not None, time_slot is not None, constraint_type is not None]):
                json_constraint_set.add((branch_id, time_slot, constraint_type))

        current_constraint_set = set()
        for constraint in current_dcpf_constraints:
            branch_id = constraint.get('branch_id')
            time_slot = constraint.get('time_slot')
            constraint_type = constraint.get('constraint_type')
            if all([branch_id is not None, time_slot is not None, constraint_type is not None]):
                current_constraint_set.add((branch_id, time_slot, constraint_type))

        # 计算并集
        union_constraint_set = json_constraint_set.union(current_constraint_set)

        # 获取当前求解的x_sol并找到非整数变量
        if self.model.status != GRB.OPTIMAL:
            print("模型未求解到最优解")
            return {}

        pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
        x_sol = np.array([[self.x[g, t].X for t in range(self.T)] for g in range(self.ng)])
        fractional_variables = []

        # 找到非整数变量 (接近0.5的变量)
        tolerance = 1e-6
        for g in range(self.ng):
            for t in range(self.T):
                x_val = x_sol[g, t]
                if tolerance < x_val < (1 - tolerance):  # 非整数变量
                    fractional_variables.append({
                        'unit_id': g,
                        'time_slot': t,
                        'value': x_val,
                        'variable_name': f'x[{g},{t}]'
                    })

        print(f"\n=== 并集约束和非整数变量分析 ===")
        print(f"JSON中起作用DCPF约束数: {len(json_constraint_set)}")
        print(f"当前求解起作用DCPF约束数: {len(current_constraint_set)}")
        print(f"并集DCPF约束数: {len(union_constraint_set)}")
        print(f"非整数变量数: {len(fractional_variables)}")

        # 提取并集约束的系数和右端项
        constraint_coefficients = []

        try:
            # 重建PTDF矩阵用于计算约束系数
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1

            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]

            for branch_id, time_slot, constraint_type in union_constraint_set:
                # 计算约束系数
                ptdf_row = PTDF[branch_id, :]  # 该支路的PTDF系数
                pg_coefficients = ptdf_row @ G  # 对pg变量的系数
                
                # 设置系数阈值，去掉接近0的系数
                coeff_threshold = 1e-8
                
                # 过滤掉系数为0的变量，并构建非零系数字典
                nonzero_pg_coefficients = {}
                nonzero_coefficients_list = []
                
                for g in range(self.ng):
                    coeff = pg_coefficients[g] * pg_sol[g, time_slot] / x_sol[g, time_slot] if x_sol[g, time_slot] != 0 else 0
                    if abs(coeff) > coeff_threshold:
                        nonzero_pg_coefficients[g] = coeff
                        nonzero_coefficients_list.append({
                            'unit_id': g,
                            'coefficient': float(coeff)
                        })
                
                # 右端项
                if constraint_type == 'dcpf_upper':
                    rhs = branch_limit[branch_id] + ptdf_row @ self.Pd[:, time_slot]
                else:  # dcpf_lower
                    rhs = branch_limit[branch_id] - ptdf_row @ self.Pd[:, time_slot]

                # 检查该约束是否涉及非整数变量
                involves_fractional = False
                fractional_units_in_constraint = []

                for frac_var in fractional_variables:
                    if frac_var['time_slot'] == time_slot:
                        unit_id = frac_var['unit_id']
                        # 只有当该机组在当前约束中有非零系数时才考虑
                        if unit_id in nonzero_pg_coefficients:
                            involves_fractional = True
                            fractional_units_in_constraint.append({
                                'unit_id': unit_id,
                                'x_value': frac_var['value'],
                                'pg_coefficient': nonzero_pg_coefficients[unit_id]
                            })

                constraint_info = {
                    'branch_id': branch_id,
                    'time_slot': time_slot,
                    'constraint_type': constraint_type,
                    'constraint_name': f'{constraint_type}_{branch_id}_{time_slot}',
                    'nonzero_pg_coefficients': nonzero_coefficients_list,  # 只保存非零系数
                    'total_nonzero_variables': len(nonzero_coefficients_list),  # 非零系数变量数量
                    'rhs': float(rhs),
                    'branch_limit': float(branch_limit[branch_id]),
                    'load_term': float(ptdf_row @ self.Pd[:, time_slot]),
                    'involves_fractional_variables': involves_fractional,
                    'fractional_units': fractional_units_in_constraint,
                    'in_json': (branch_id, time_slot, constraint_type) in json_constraint_set,
                    'in_current': (branch_id, time_slot, constraint_type) in current_constraint_set
                }

                constraint_coefficients.append(constraint_info)

        except Exception as e:
            print(f"计算约束系数时出错: {e}")
            return {}

        # 按是否涉及非整数变量分类
        constraints_with_fractional = [c for c in constraint_coefficients if c['involves_fractional_variables']]
        constraints_without_fractional = [c for c in constraint_coefficients if not c['involves_fractional_variables']]

        print(f"\n涉及非整数变量的并集约束数: {len(constraints_with_fractional)}")
        print(f"不涉及非整数变量的并集约束数: {len(constraints_without_fractional)}")

        # 显示涉及非整数变量的约束详情
        if constraints_with_fractional:
            print(f"\n=== 涉及非整数变量的约束详情 ===")
            for i, constraint in enumerate(constraints_with_fractional[:5]):  # 只显示前5个
                print(f"\n{i+1}. {constraint['constraint_name']}")
                print(f"   支路: {constraint['branch_id']}, 时段: {constraint['time_slot']}")
                print(f"   约束类型: {constraint['constraint_type']}")
                print(f"   右端项: {constraint['rhs']:.6f}")
                print(f"   支路容量: {constraint['branch_limit']:.2f}")
                print(f"   负荷项: {constraint['load_term']:.6f}")
                print(f"   来源: JSON={constraint['in_json']}, 当前={constraint['in_current']}")
                print(f"   涉及的非整数变量:")
                for frac_unit in constraint['fractional_units']:
                    print(f"     机组{frac_unit['unit_id']}: x={frac_unit['x_value']:.6f}, "
                        f"系数={frac_unit['pg_coefficient']:.6f}")

            if len(constraints_with_fractional) > 5:
                print(f"   ... 还有 {len(constraints_with_fractional) - 5} 个约束")

        return {
            'union_constraints': constraint_coefficients,
            'constraints_with_fractional': constraints_with_fractional,
            'constraints_without_fractional': constraints_without_fractional,
            'fractional_variables': fractional_variables,
            'statistics': {
                'json_dcpf_count': len(json_constraint_set),
                'current_dcpf_count': len(current_constraint_set),
                'union_dcpf_count': len(union_constraint_set),
                'fractional_variables_count': len(fractional_variables),
                'constraints_with_fractional_count': len(constraints_with_fractional),
                'constraints_without_fractional_count': len(constraints_without_fractional)
            }
        }
    
    def add_theta_variables_for_branches(self, union_analysis: Dict):
        """
        为每条支路添加theta相关变量，同一支路在不同时段共享theta参数
        
        Args:
            union_analysis: 并集约束分析结果
        """
        self.theta_vars = {}
        
        # 获取所有涉及的支路ID
        union_constraints = union_analysis.get('union_constraints', [])
        branch_ids = set()
        for constraint in union_constraints:
            branch_ids.add(constraint['branch_id'])
        
        print(f"为 {len(branch_ids)} 条支路添加theta变量")
        
        # 为每条支路添加右端项theta参数（同一支路在不同时段共享）
        for branch_id in sorted(branch_ids):
            # 每条支路的右端项theta参数
            self.theta_vars[f'theta_branch_{branch_id}_rhs_0'] = self.model.addVar(
                lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_rhs_0'
            )
            self.theta_vars[f'theta_branch_{branch_id}_rhs_1'] = self.model.addVar(
                lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_rhs_1'
            )
            self.theta_vars[f'theta_branch_{branch_id}_rhs_2'] = self.model.addVar(
                lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_rhs_2'
            )
            
            # 为每条支路的每个机组添加左端项系数theta参数
            for g in range(self.ng):
                self.theta_vars[f'theta_branch_{branch_id}_unit_{g}_0'] = self.model.addVar(
                    lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_unit_{g}_0'
                )
                self.theta_vars[f'theta_branch_{branch_id}_unit_{g}_1'] = self.model.addVar(
                    lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_unit_{g}_1'
                )
                self.theta_vars[f'theta_branch_{branch_id}_unit_{g}_2'] = self.model.addVar(
                    lb=-GRB.INFINITY, name=f'theta_branch_{branch_id}_unit_{g}_2'
                )
        
        self.model.update()
        print(f"共添加 {len(self.theta_vars)} 个theta变量")

    def add_parametric_constraints_for_all_dcpf(self, union_analysis: Dict) -> None:
        """
        为所有union中的DCPF约束添加参数化约束
        同一支路的theta在不同时段保持一致，但不同支路间theta不同
        
        约束形式：
        对于支路l在时段t：
        sum_g[(a_{l,g} + theta_branch_l_unit_g_0 + theta_branch_l_unit_g_1*u + theta_branch_l_unit_g_2*u^2) * pg_{g,t}]
        <= u + theta_branch_l_rhs_0 + theta_branch_l_rhs_1*u + theta_branch_l_rhs_2*u^2
        
        Args:
            union_analysis: 并集约束分析结果
        """
        print(f"\n=== 为所有DCPF约束添加参数化约束（支路theta一致，线路间不同） ===")
        
        union_constraints = union_analysis.get('union_constraints', [])
        if not union_constraints:
            print("没有union约束，跳过添加参数化约束")
            return
        
        # 首先添加theta变量
        self.add_theta_variables_for_branches(union_analysis)
        
        added_constraints = 0
        
        for constraint_info in union_constraints:
            try:
                branch_id = constraint_info['branch_id']
                time_slot = constraint_info['time_slot']
                constraint_type = constraint_info['constraint_type']
                original_rhs = constraint_info['rhs']  # 作为输入u
                nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
                constraint_name = constraint_info['constraint_name']
                
                u = original_rhs
                
                # 构建新约束的左侧
                lhs_expr = 0
                
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = coeff_info['coefficient']  # 原系数
                    
                    # 使用支路相关的theta变量（同一支路在不同时段共享）
                    theta_unit_0 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_0']
                    theta_unit_1 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_1']
                    theta_unit_2 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_2']
                    
                    # 新系数: original_coeff + theta_unit_0 + theta_unit_1*u + theta_unit_2*u^2
                    new_coeff = (original_coeff + theta_unit_0 + 
                            theta_unit_1 * u + theta_unit_2 * u * u)
                    
                    # 添加到左侧表达式
                    lhs_expr += new_coeff * self.pg[unit_id, time_slot]
                
                # 构建新约束的右侧 - 使用支路特定的右端项theta
                theta_rhs_0 = self.theta_vars[f'theta_branch_{branch_id}_rhs_0']
                theta_rhs_1 = self.theta_vars[f'theta_branch_{branch_id}_rhs_1']
                theta_rhs_2 = self.theta_vars[f'theta_branch_{branch_id}_rhs_2']
                
                # 新右端项: u + theta_branch_rhs_0 + theta_branch_rhs_1*u + theta_branch_rhs_2*u^2
                new_rhs = (u + theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u)
                
                # 添加新约束
                new_constraint_name = f'parametric_{constraint_name}'
                self.model.addConstr(lhs_expr <= new_rhs, name=new_constraint_name)
                added_constraints += 1
                
            except Exception as e:
                print(f"添加约束 {constraint_info['constraint_name']} 时出错: {e}")
        
        print(f"\n成功添加 {added_constraints} 个参数化约束")
        
        # 更新模型
        self.model.update()

    def _build_model(self):
        # 有功平衡
        for t in range(self.T):
            self.model.addConstr(gp.quicksum(self.pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]), name=f'power_balance_{t}')
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t], name=f'gen_min_{g}_{t}')
                self.model.addConstr(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t], name=f'gen_max_{g}_{t}')
        
        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]), name=f'ramp_up_{g}_{t}')
                self.model.addConstr(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]), name=f'ramp_down_{g}_{t}')
        
        # 最小开机时间和最小关机时间约束
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    self.model.addConstr(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t], name=f'min_up_time_{g}_{t1}_{t1+t}')
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    self.model.addConstr(-self.x[g, t1+1] + self.x[g, t1] <= 1 - self.x[g, t1+t], name=f'min_down_time_{g}_{t1}_{t1+t}')
        
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.coc[g, t-1] >= start_cost[g] * (self.x[g, t] - self.x[g, t-1]), name=f'start_cost_{g}_{t}')
                self.model.addConstr(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]), name=f'shut_cost_{g}_{t}')
                self.model.addConstr(self.coc[g, t-1] >= 0, name=f'cost_nonneg_{g}_{t}')
        
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addConstr(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t], name=f'power_cost_{g}_{t}')
        
        # 潮流约束 - 添加约束名称
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
                    # 添加约束名称，与mapping保持一致
                    self.model.addConstr(flow[l] <= branch_limit[l], name=f'dcpf_upper_{l}_{t}')
                    self.model.addConstr(flow[l] >= -branch_limit[l], name=f'dcpf_lower_{l}_{t}')
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

    def extract_constraint_matrices(self, union_analysis: Dict = None, 
                                theta_values: Optional[Dict] = None) -> Dict:
        """
        将所有约束整理成 Ax <= b 的矩阵形式
        
        Args:
            union_analysis: 并集约束分析结果（用于参数化约束）
            theta_values: theta参数值字典，如果为None则theta作为变量保持在A矩阵中
            
        Returns:
            包含各种约束矩阵的字典
        """
        print(f"\n=== 提取约束矩阵 Ax <= b 形式 ===")
        print(f"Theta模式: {'给定值' if theta_values else '保持为变量'}")
        
        if union_analysis is not None:
            self._current_union_analysis = union_analysis
        
        # 变量顺序: [pg[0,0], pg[0,1], ..., pg[ng-1,T-1], x[0,0], x[0,1], ..., x[ng-1,T-1], 
        #           coc[0,0], ..., coc[ng-1,T-2], cpower[0,0], ..., cpower[ng-1,T-1]]
        
        n_pg = self.ng * self.T
        n_x = self.ng * self.T
        n_coc = self.ng * (self.T - 1)
        n_cpower = self.ng * self.T
        total_vars = n_pg + n_x + n_coc + n_cpower
        
        print(f"变量数量: pg={n_pg}, x={n_x}, coc={n_coc}, cpower={n_cpower}")
        print(f"总变量数: {total_vars}")
        
        # 1. 功率平衡约束、DCPF约束和参数化约束的联合矩阵 A
        A_power_dcpf, b_power_dcpf = self._extract_power_balance_and_dcpf_constraints(
            total_vars, n_pg, n_x, union_analysis, theta_values
        )
        
        # 2. 机组运行相关约束（每个机组独立）A_g
        A_g_list, b_g_list = self._extract_unit_constraints(total_vars, n_pg, n_x, n_coc, n_cpower)
        
        # 3. 目标函数系数向量 c
        c_vector = self._extract_objective_coefficients(total_vars, n_pg, n_x, n_coc, n_cpower)
        
        return {
            'A_power_dcpf': A_power_dcpf,
            'b_power_dcpf': b_power_dcpf,
            'A_g_list': A_g_list,  # 每个机组的约束矩阵列表
            'b_g_list': b_g_list,  # 每个机组的右端项列表
            'c_vector': c_vector,
            'variable_info': {
                'total_vars': total_vars,
                'n_pg': n_pg,
                'n_x': n_x,
                'n_coc': n_coc,
                'n_cpower': n_cpower,
                'theta_mode': 'given_values' if theta_values else 'variables',
                'variable_order': self._get_variable_order()
            }
        }

    def _extract_power_balance_and_dcpf_constraints(self, total_vars: int, n_pg: int, n_x: int,
                                                union_analysis: Dict = None,
                                                theta_values: Optional[Dict] = None) -> Tuple:
        """提取功率平衡约束、DCPF约束和参数化约束"""
        
        constraints = []
        rhs_values = []
        
        # 1. 功率平衡约束: sum(pg[g,t]) = sum(Pd[:,t])
        # 转换为: sum(pg[g,t]) - sum(Pd[:,t]) <= 0 和 -sum(pg[g,t]) + sum(Pd[:,t]) <= 0
        for t in range(self.T):
            # sum(pg[g,t]) <= sum(Pd[:,t])
            constraint_row = np.zeros(total_vars)
            for g in range(self.ng):
                pg_idx = g * self.T + t
                constraint_row[pg_idx] = 1.0
            constraints.append(constraint_row)
            rhs_values.append(np.sum(self.Pd[:, t]))
            
            # -sum(pg[g,t]) <= -sum(Pd[:,t])
            constraint_row = np.zeros(total_vars)
            for g in range(self.ng):
                pg_idx = g * self.T + t
                constraint_row[pg_idx] = -1.0
            constraints.append(constraint_row)
            rhs_values.append(-np.sum(self.Pd[:, t]))
        
        # 2. 原始DCPF约束
        try:
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            
            for t in range(self.T):
                for l in range(self.branch.shape[0]):
                    ptdf_row = PTDF[l, :]
                    pg_coeffs = ptdf_row @ G
                    
                    # 上限约束: PTDF @ (G @ pg - Pd) <= limit
                    constraint_row = np.zeros(total_vars)
                    for g in range(self.ng):
                        pg_idx = g * self.T + t
                        constraint_row[pg_idx] = pg_coeffs[g]
                    constraints.append(constraint_row)
                    rhs_values.append(branch_limit[l] + ptdf_row @ self.Pd[:, t])
                    
                    # 下限约束: PTDF @ (G @ pg - Pd) >= -limit
                    # 转换为: -(PTDF @ G) @ pg <= limit - PTDF @ Pd
                    constraint_row = np.zeros(total_vars)
                    for g in range(self.ng):
                        pg_idx = g * self.T + t
                        constraint_row[pg_idx] = -pg_coeffs[g]
                    constraints.append(constraint_row)
                    rhs_values.append(branch_limit[l] - ptdf_row @ self.Pd[:, t])
        
        except ImportError:
            print("未安装pypower，跳过DCPF约束")
        
        # 3. 参数化约束（如果存在）
        if union_analysis is not None:
            parametric_constraints, parametric_rhs = self._extract_parametric_constraints(
                total_vars, n_pg, union_analysis, theta_values
            )
            if parametric_constraints:
                constraints.extend(parametric_constraints)
                rhs_values.extend(parametric_rhs)
        
        if not constraints:
            return np.array([]).reshape(0, total_vars), np.array([])
        
        A_matrix = np.vstack(constraints)
        b_vector = np.array(rhs_values)
        
        print(f"功率平衡+DCPF+参数化约束矩阵: {A_matrix.shape}")
        
        return A_matrix, b_vector

    def _extract_parametric_constraints(self, total_vars: int, n_pg: int, 
                                    union_analysis: Dict, 
                                    theta_values: Optional[Dict] = None) -> Tuple[List, List]:
        """
        提取参数化约束
        
        Args:
            total_vars: 总变量数
            n_pg: pg变量数
            union_analysis: 并集约束分析
            theta_values: theta参数值字典，如果为None则theta作为变量保持在A矩阵中
            
        Returns:
            约束矩阵行列表和右端项列表
        """
        union_constraints = union_analysis.get('union_constraints', [])
        if not union_constraints:
            return [], []
        
        # 如果theta_values为None，需要先添加theta变量
        if theta_values is None and not hasattr(self, 'theta_vars'):
            self.add_theta_variables_for_branches(union_analysis)
        
        constraints = []
        rhs_values = []
        
        print(f"提取参数化约束，theta模式: {'给定值' if theta_values else '保持为变量'}")
        
        for constraint_info in union_constraints:
            branch_id = constraint_info['branch_id']
            time_slot = constraint_info['time_slot']
            original_rhs = constraint_info['rhs']
            nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
            
            u = original_rhs
            
            if theta_values is None:
                # theta作为变量保持在A矩阵中
                constraint_row = self._build_parametric_constraint_with_theta_vars(
                    total_vars, n_pg, branch_id, time_slot, nonzero_coefficients, u
                )
                # 提取右端项
                rhs_expr = self._build_parametric_rhs_with_theta_vars(branch_id, u)
                rhs = rhs_expr
            else:
                # theta给定具体值，计算数值矩阵
                constraint_row = np.zeros(total_vars)
                
                # 处理左端项系数
                for coeff_info in nonzero_coefficients:
                    unit_id = coeff_info['unit_id']
                    original_coeff = coeff_info['coefficient']
                    
                    # pg变量索引
                    pg_idx = unit_id * self.T + time_slot
                    
                    # 获取theta值
                    theta_0_val = theta_values.get(f'theta_branch_{branch_id}_unit_{unit_id}_0', 0.0)
                    theta_1_val = theta_values.get(f'theta_branch_{branch_id}_unit_{unit_id}_1', 0.0)
                    theta_2_val = theta_values.get(f'theta_branch_{branch_id}_unit_{unit_id}_2', 0.0)
                    
                    # 计算新系数: original_coeff + theta_0 + theta_1*u + theta_2*u^2
                    new_coeff = (original_coeff + theta_0_val + 
                            theta_1_val * u + theta_2_val * u * u)
                    
                    constraint_row[pg_idx] = new_coeff
                
                # 计算右端项
                theta_rhs_0_val = theta_values.get(f'theta_branch_{branch_id}_rhs_0', 0.0)
                theta_rhs_1_val = theta_values.get(f'theta_branch_{branch_id}_rhs_1', 0.0)
                theta_rhs_2_val = theta_values.get(f'theta_branch_{branch_id}_rhs_2', 0.0)
                
                rhs = (u + theta_rhs_0_val + theta_rhs_1_val * u + theta_rhs_2_val * u * u)
            
            constraints.append(constraint_row)
            rhs_values.append(rhs)
        
        print(f"提取了 {len(constraints)} 个参数化约束")
        return constraints, rhs_values

    def _build_parametric_constraint_with_theta_vars(self, total_vars: int, n_pg: int,
                                                    branch_id: int, time_slot: int,
                                                    nonzero_coefficients: List, u: float):
        """
        构建包含theta变量的参数化约束行（返回变量表达式）
        
        这里返回的constraint_row包含theta变量，用于A矩阵是theta函数的情况
        """
        # 创建表达式约束行（每个元素可能是常数或Gurobi表达式）
        constraint_row = np.zeros(total_vars, dtype=object)
        
        # 处理pg变量的系数
        for coeff_info in nonzero_coefficients:
            unit_id = coeff_info['unit_id']
            original_coeff = coeff_info['coefficient']
            
            # pg变量索引
            pg_idx = unit_id * self.T + time_slot
            
            # 获取theta变量
            theta_unit_0 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_0']
            theta_unit_1 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_1']
            theta_unit_2 = self.theta_vars[f'theta_branch_{branch_id}_unit_{unit_id}_2']
            
            # 构建系数表达式: original_coeff + theta_0 + theta_1*u + theta_2*u^2
            coeff_expr = (original_coeff + theta_unit_0 + 
                        theta_unit_1 * u + theta_unit_2 * u * u)
            
            constraint_row[pg_idx] = coeff_expr
        
        return constraint_row

    def _build_parametric_rhs_with_theta_vars(self, branch_id: int, u: float):
        """
        构建包含theta变量的右端项表达式
        
        Args:
            branch_id: 支路ID
            u: 原始右端项值
            
        Returns:
            包含theta变量的右端项表达式
        """
        # 获取右端项的theta变量
        theta_rhs_0 = self.theta_vars[f'theta_branch_{branch_id}_rhs_0']
        theta_rhs_1 = self.theta_vars[f'theta_branch_{branch_id}_rhs_1']
        theta_rhs_2 = self.theta_vars[f'theta_branch_{branch_id}_rhs_2']
        
        # 构建右端项表达式: u + theta_rhs_0 + theta_rhs_1*u + theta_rhs_2*u^2
        rhs_expr = u + theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u
        
        return rhs_expr
    
    def _extract_unit_constraints(self, total_vars: int, n_pg: int, n_x: int, 
                                n_coc: int, n_cpower: int) -> Tuple[List, List]:
        """提取每个机组的独立约束 A_g"""
        A_g_list = []
        b_g_list = []
        
        for g in range(self.ng):
            constraints_g = []
            rhs_g = []
            
            # 每个机组的变量索引偏移
            pg_offset = g * self.T
            x_offset = n_pg + g * self.T
            coc_offset = n_pg + n_x + g * (self.T - 1)
            cpower_offset = n_pg + n_x + n_coc + g * self.T
            
            # 1. 发电约束: PMIN * x[g,t] <= pg[g,t] <= PMAX * x[g,t]
            for t in range(self.T):
                # pg[g,t] - PMAX * x[g,t] <= 0
                constraint_row = np.zeros(total_vars)
                constraint_row[pg_offset + t] = 1.0
                constraint_row[x_offset + t] = -self.gen[g, PMAX]
                constraints_g.append(constraint_row)
                rhs_g.append(0.0)
                
                # -pg[g,t] + PMIN * x[g,t] <= 0
                constraint_row = np.zeros(total_vars)
                constraint_row[pg_offset + t] = -1.0
                constraint_row[x_offset + t] = self.gen[g, PMIN]
                constraints_g.append(constraint_row)
                rhs_g.append(0.0)
            
            # 2. 爬坡约束
            Ru = 0.4 * self.gen[g, PMAX] / self.T_delta
            Rd = 0.4 * self.gen[g, PMAX] / self.T_delta
            Ru_co = 0.3 * self.gen[g, PMAX]
            Rd_co = 0.3 * self.gen[g, PMAX]
            
            for t in range(1, self.T):
                # 上爬坡: pg[g,t] - pg[g,t-1] - Ru * x[g,t-1] + Ru_co * x[g,t-1] <= Ru_co
                constraint_row = np.zeros(total_vars)
                constraint_row[pg_offset + t] = 1.0
                constraint_row[pg_offset + t - 1] = -1.0
                constraint_row[x_offset + t - 1] = -Ru + Ru_co
                constraints_g.append(constraint_row)
                rhs_g.append(Ru_co)
                
                # 下爬坡: pg[g,t-1] - pg[g,t] - Rd * x[g,t] + Rd_co * x[g,t] <= Rd_co
                constraint_row = np.zeros(total_vars)
                constraint_row[pg_offset + t - 1] = 1.0
                constraint_row[pg_offset + t] = -1.0
                constraint_row[x_offset + t] = -Rd + Rd_co
                constraints_g.append(constraint_row)
                rhs_g.append(Rd_co)
            
            # 3. 最小开机/关机时间约束
            Ton = int(4 * self.T_delta)
            Toff = int(4 * self.T_delta)
            
            # 最小开机时间: x[g,t1+1] - x[g,t1] - x[g,t1+t] <= 0
            for t in range(1, min(Ton+1, self.T)):
                for t1 in range(self.T - t):
                    if t1 + t < self.T:
                        constraint_row = np.zeros(total_vars)
                        constraint_row[x_offset + t1 + 1] = 1.0
                        constraint_row[x_offset + t1] = -1.0
                        constraint_row[x_offset + t1 + t] = -1.0
                        constraints_g.append(constraint_row)
                        rhs_g.append(0.0)
            
            # 最小关机时间: -x[g,t1+1] + x[g,t1] + x[g,t1+t] <= 1
            for t in range(1, min(Toff+1, self.T)):
                for t1 in range(self.T - t):
                    if t1 + t < self.T:
                        constraint_row = np.zeros(total_vars)
                        constraint_row[x_offset + t1 + 1] = -1.0
                        constraint_row[x_offset + t1] = 1.0
                        constraint_row[x_offset + t1 + t] = 1.0
                        constraints_g.append(constraint_row)
                        rhs_g.append(1.0)
            
            # 4. 启停成本约束
            start_cost = self.gencost[g, 1]
            shut_cost = self.gencost[g, 2]
            
            for t in range(1, self.T):
                # coc[g,t-1] >= start_cost * (x[g,t] - x[g,t-1])
                # 即: -coc[g,t-1] + start_cost * x[g,t] - start_cost * x[g,t-1] <= 0
                constraint_row = np.zeros(total_vars)
                constraint_row[coc_offset + t - 1] = -1.0
                constraint_row[x_offset + t] = start_cost
                constraint_row[x_offset + t - 1] = -start_cost
                constraints_g.append(constraint_row)
                rhs_g.append(0.0)
                
                # coc[g,t-1] >= shut_cost * (x[g,t-1] - x[g,t])
                # 即: -coc[g,t-1] + shut_cost * x[g,t-1] - shut_cost * x[g,t] <= 0
                constraint_row = np.zeros(total_vars)
                constraint_row[coc_offset + t - 1] = -1.0
                constraint_row[x_offset + t - 1] = shut_cost
                constraint_row[x_offset + t] = -shut_cost
                constraints_g.append(constraint_row)
                rhs_g.append(0.0)
                
                # coc[g,t-1] >= 0 (已经在变量定义中处理)
            
            # 5. 发电成本约束
            for t in range(self.T):
                # cpower[g,t] >= cost_coeff * pg[g,t] + fixed_cost * x[g,t]
                # 即: -cpower[g,t] + cost_coeff * pg[g,t] + fixed_cost * x[g,t] <= 0
                constraint_row = np.zeros(total_vars)
                constraint_row[cpower_offset + t] = -1.0
                constraint_row[pg_offset + t] = self.gencost[g, -2] / self.T_delta
                constraint_row[x_offset + t] = self.gencost[g, -1] / self.T_delta
                constraints_g.append(constraint_row)
                rhs_g.append(0.0)
            
            # 转换为矩阵形式
            if constraints_g:
                A_g = np.vstack(constraints_g)
                b_g = np.array(rhs_g)
            else:
                A_g = np.array([]).reshape(0, total_vars)
                b_g = np.array([])
            
            A_g_list.append(A_g)
            b_g_list.append(b_g)
            
            print(f"机组{g}约束矩阵: {A_g.shape}")
        
        return A_g_list, b_g_list

    def _extract_objective_coefficients(self, total_vars: int, n_pg: int, n_x: int, 
                                    n_coc: int, n_cpower: int) -> np.ndarray:
        """提取目标函数系数向量 c，使得目标函数为 c^T * x"""
        
        c_vector = np.zeros(total_vars)
        
        # 目标函数: min sum(cpower[g,t]) + sum(coc[g,t])
        
        # cpower变量的系数为1
        cpower_start = n_pg + n_x + n_coc
        for g in range(self.ng):
            for t in range(self.T):
                cpower_idx = cpower_start + g * self.T + t
                c_vector[cpower_idx] = 1.0
        
        # coc变量的系数为1
        coc_start = n_pg + n_x
        for g in range(self.ng):
            for t in range(self.T - 1):
                coc_idx = coc_start + g * (self.T - 1) + t
                c_vector[coc_idx] = 1.0
        
        print(f"目标函数系数向量维度: {c_vector.shape}")
        print(f"非零系数数量: {np.count_nonzero(c_vector)}")
        
        return c_vector

    def _get_variable_order(self) -> List[str]:
        """获取变量顺序说明"""
        order = []
        
        # pg变量
        for g in range(self.ng):
            for t in range(self.T):
                order.append(f'pg[{g},{t}]')
        
        # x变量
        for g in range(self.ng):
            for t in range(self.T):
                order.append(f'x[{g},{t}]')
        
        # coc变量
        for g in range(self.ng):
            for t in range(self.T-1):
                order.append(f'coc[{g},{t}]')
        
        # cpower变量
        for g in range(self.ng):
            for t in range(self.T):
                order.append(f'cpower[{g},{t}]')
        
        return order

    def evaluate_parametric_matrix_with_theta_values(self, A_matrix_with_theta_vars, 
                                                     b_vector_with_theta_vars,
                                                    theta_values: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        当A矩阵包含theta变量时，用给定的theta值计算数值矩阵
        
        Args:
            A_matrix_with_theta_vars: 包含theta变量的A矩阵
            theta_values: theta参数值字典
            
        Returns:
            计算后的数值矩阵
        """
        if A_matrix_with_theta_vars.size == 0:
            return A_matrix_with_theta_vars
        
        A_numeric = np.zeros(A_matrix_with_theta_vars.shape)
        
        for i in range(A_matrix_with_theta_vars.shape[0]):
            for j in range(A_matrix_with_theta_vars.shape[1]):
                element = A_matrix_with_theta_vars[i, j]
                
                if isinstance(element, (int, float)):
                    A_numeric[i, j] = element
                elif hasattr(element, 'getValue'):
                    # Gurobi表达式，需要用theta值计算
                    try:
                        # 设置theta变量值并计算表达式
                        value = self._evaluate_gurobi_expression(element, theta_values)
                        A_numeric[i, j] = value
                    except:
                        A_numeric[i, j] = 0.0
                else:
                    A_numeric[i, j] = 0.0
                    
        if hasattr(b_vector_with_theta_vars, '__len__') and len(b_vector_with_theta_vars) > 0:
            b_numeric = np.zeros(len(b_vector_with_theta_vars))
            
            for i, b_element in enumerate(b_vector_with_theta_vars):
                if isinstance(b_element, (int, float)):
                    b_numeric[i] = b_element
                elif hasattr(b_element, 'getValue'):
                    # Gurobi表达式，需要用theta值计算
                    try:
                        value = self._evaluate_gurobi_expression(b_element, theta_values)
                        b_numeric[i] = value
                    except:
                        b_numeric[i] = 0.0
                else:
                    b_numeric[i] = 0.0
        else:
            b_numeric = np.array([])
        
        return A_numeric, b_numeric

    def _evaluate_gurobi_expression(self, expr, theta_values: Dict) -> float:
        """计算Gurobi表达式在给定theta值下的数值"""
        try:
            # 获取表达式的常数项
            constant = expr.getConstant() if hasattr(expr, 'getConstant') else 0.0
            
            # 获取线性项
            linear_value = 0.0
            if hasattr(expr, 'size') and expr.size() > 0:
                for i in range(expr.size()):
                    var = expr.getVar(i)
                    coeff = expr.getCoeff(i)
                    var_name = var.VarName
                    
                    if var_name in theta_values:
                        linear_value += coeff * theta_values[var_name]
            
            return constant + linear_value
        except:
            return 0.0

    def print_matrix_summary(self, constraint_matrices: Dict):
        """打印约束矩阵摘要信息"""
        print(f"\n=== 约束矩阵摘要 ===")
        
        # 功率平衡+DCPF约束
        A_power_dcpf = constraint_matrices['A_power_dcpf']
        b_power_dcpf = constraint_matrices['b_power_dcpf']
        print(f"功率平衡+DCPF+参数化约束:")
        print(f"  矩阵维度: {A_power_dcpf.shape}")
        print(f"  右端项维度: {b_power_dcpf.shape}")
        if A_power_dcpf.size > 0:
            if A_power_dcpf.dtype == object:
                print(f"  矩阵类型: 包含theta变量表达式")
            else:
                print(f"  非零元素: {np.count_nonzero(A_power_dcpf)}")
                print(f"  稀疏度: {1 - np.count_nonzero(A_power_dcpf) / A_power_dcpf.size:.4f}")
        
        # 机组约束
        A_g_list = constraint_matrices['A_g_list']
        b_g_list = constraint_matrices['b_g_list']
        print(f"\n机组独立约束:")
        total_unit_constraints = 0
        total_unit_nonzeros = 0
        for g in range(len(A_g_list)):
            A_g = A_g_list[g]
            b_g = b_g_list[g]
            total_unit_constraints += A_g.shape[0]
            if A_g.size > 0:
                total_unit_nonzeros += np.count_nonzero(A_g)
            print(f"  机组{g}: {A_g.shape[0]}约束, {np.count_nonzero(A_g) if A_g.size > 0 else 0}非零元素")
        
        print(f"  总机组约束数: {total_unit_constraints}")
        print(f"  总非零元素: {total_unit_nonzeros}")
        
        # 目标函数
        c_vector = constraint_matrices['c_vector']
        print(f"\n目标函数:")
        print(f"  系数向量维度: {c_vector.shape}")
        print(f"  非零系数: {np.count_nonzero(c_vector)}")
        
        # 变量信息
        var_info = constraint_matrices['variable_info']
        print(f"\n变量信息:")
        print(f"  总变量数: {var_info['total_vars']}")
        print(f"  pg变量: {var_info['n_pg']}")
        print(f"  x变量: {var_info['n_x']}")
        print(f"  coc变量: {var_info['n_coc']}")
        print(f"  cpower变量: {var_info['n_cpower']}")
        print(f"  theta模式: {var_info['theta_mode']}")

    def demonstrate_matrix_extraction(self, union_analysis: Dict) -> None:
        """演示矩阵提取的两种模式"""
        print(f"\n=== 演示矩阵提取 ===")
        
        # 模式1: theta保持为变量
        print(f"\n--- 模式1: Theta保持为变量 ---")
        matrices_with_theta_vars = self.extract_constraint_matrices(union_analysis)
        self.print_matrix_summary(matrices_with_theta_vars)
        
        # 模式2: 给定theta值
        print(f"\n--- 模式2: 给定Theta值 ---")
        # 构造一些测试theta值
        test_theta_values = {}
        if hasattr(self, 'theta_vars'):
            for var_name in list(self.theta_vars.keys())[:10]:  # 只设置前10个
                if 'rhs' in var_name:
                    test_theta_values[var_name] = 0.1
                else:
                    test_theta_values[var_name] = 0.05
        
        matrices_with_theta_values = self.extract_constraint_matrices(union_analysis, test_theta_values)
        self.print_matrix_summary(matrices_with_theta_values)
        
        # 比较两种模式
        print(f"\n--- 模式比较 ---")
        A1 = matrices_with_theta_vars['A_power_dcpf']
        A2 = matrices_with_theta_values['A_power_dcpf']
        
        print(f"Theta变量模式矩阵: {A1.shape}, dtype={A1.dtype}")
        print(f"Theta数值模式矩阵: {A2.shape}, dtype={A2.dtype}")
        
        if A1.dtype == object:
            print(f"Theta变量模式包含表达式，可用evaluate_parametric_matrix_with_theta_values计算数值")

    def _add_theta_vars_to_new_model(self, new_model) -> Dict:
        """将theta变量添加到新模型中，并更新所有相关表达式"""
        new_theta_vars = {}
        
        print(f"正在将 {len(self.theta_vars)} 个theta变量迁移到新模型...")
        
        # 1. 在新模型中创建对应的theta变量
        for var_name, original_var in self.theta_vars.items():
            new_var = new_model.addVar(
                lb=original_var.LB,
                ub=original_var.UB,
                name=original_var.VarName
            )
            new_theta_vars[var_name] = new_var
        
        # 2. 更新模型以便变量可用
        new_model.update()
        
        print(f"✓ 已在新模型中创建 {len(new_theta_vars)} 个theta变量")
        
        return new_theta_vars

    def _rebuild_parametric_matrices_for_new_model(self, A_with_theta, b_with_theta, 
                                                new_theta_vars: Dict, union_analysis: Dict):
        """为新模型重建参数化矩阵，使用新模型中的theta变量"""
        
        print(f"为新模型重建参数化矩阵...")
        
        if A_with_theta.dtype != object or A_with_theta.size == 0:
            print("A矩阵不包含theta表达式，无需重建")
            return A_with_theta, b_with_theta
        
        # 获取矩阵信息
        n_rows, n_cols = A_with_theta.shape
        
        # 创建新的矩阵
        A_new = np.empty((n_rows, n_cols), dtype=object)
        
        # 初始化为数值0
        for i in range(n_rows):
            for j in range(n_cols):
                A_new[i, j] = 0.0
        
        # 重建参数化约束
        union_constraints = union_analysis.get('union_constraints', [])
        
        # 计算变量维度
        n_pg = self.ng * self.T
        n_x = self.ng * self.T
        n_coc = self.ng * (self.T - 1)
        n_cpower = self.ng * self.T
        total_vars = n_pg + n_x + n_coc + n_cpower
        
        # 计算有多少行是参数化约束
        num_power_balance = self.T * 2  # 功率平衡约束（等式转换为两个不等式）
        num_dcpf_original = self.T * self.branch.shape[0] * 2  # 原始DCPF约束
        num_parametric = len(union_constraints)  # 参数化约束
        
        print(f"矩阵结构: 功率平衡={num_power_balance}, 原始DCPF={num_dcpf_original}, 参数化={num_parametric}")
        
        # 前面的约束保持不变（功率平衡和原始DCPF）
        A_new[:num_power_balance + num_dcpf_original, :] = A_with_theta[:num_power_balance + num_dcpf_original, :]
        
        # 跳过功率平衡和原始DCPF约束
        parametric_start_idx = num_power_balance + num_dcpf_original
        
        # 重建参数化约束部分
        for param_idx, constraint_info in enumerate(union_constraints):
            row_idx = parametric_start_idx + param_idx
            
            if row_idx >= n_rows:
                print(f"警告: 行索引 {row_idx} 超出矩阵范围 {n_rows}")
                break
            
            branch_id = constraint_info['branch_id']
            time_slot = constraint_info['time_slot']
            original_rhs = constraint_info['rhs']
            nonzero_coefficients = constraint_info['nonzero_pg_coefficients']
            
            u = original_rhs
            
            # 重建这一行的系数
            for coeff_info in nonzero_coefficients:
                unit_id = coeff_info['unit_id']
                original_coeff = coeff_info['coefficient']
                
                # pg变量索引
                pg_idx = unit_id * self.T + time_slot
                
                if pg_idx >= n_cols:
                    continue
                
                # 使用新模型中的theta变量构建表达式
                theta_unit_0_name = f'theta_branch_{branch_id}_unit_{unit_id}_0'
                theta_unit_1_name = f'theta_branch_{branch_id}_unit_{unit_id}_1'
                theta_unit_2_name = f'theta_branch_{branch_id}_unit_{unit_id}_2'
                
                if all(name in new_theta_vars for name in [theta_unit_0_name, theta_unit_1_name, theta_unit_2_name]):
                    # 获取新模型中的theta变量
                    theta_unit_0 = new_theta_vars[theta_unit_0_name]
                    theta_unit_1 = new_theta_vars[theta_unit_1_name]
                    theta_unit_2 = new_theta_vars[theta_unit_2_name]
                    
                    # 构建新的系数表达式
                    coeff_expr = (original_coeff + theta_unit_0 + 
                                theta_unit_1 * u + theta_unit_2 * u * u)
                    
                    A_new[row_idx, pg_idx] = coeff_expr
                else:
                    # 如果theta变量不存在，使用原系数
                    A_new[row_idx, pg_idx] = original_coeff
        
        # 重建右端项向量
        b_new = []
        
        if hasattr(b_with_theta, '__len__') and len(b_with_theta) > 0:
            for i, b_element in enumerate(b_with_theta):
                if i < parametric_start_idx:
                    # 非参数化部分保持不变
                    b_new.append(b_element)
                else:
                    # 参数化部分需要重建
                    param_idx = i - parametric_start_idx
                    if param_idx < len(union_constraints):
                        constraint_info = union_constraints[param_idx]
                        branch_id = constraint_info['branch_id']
                        u = constraint_info['rhs']
                        
                        # 使用新模型中的theta变量重建右端项
                        theta_rhs_0_name = f'theta_branch_{branch_id}_rhs_0'
                        theta_rhs_1_name = f'theta_branch_{branch_id}_rhs_1'
                        theta_rhs_2_name = f'theta_branch_{branch_id}_rhs_2'
                        
                        if all(name in new_theta_vars for name in [theta_rhs_0_name, theta_rhs_1_name, theta_rhs_2_name]):
                            theta_rhs_0 = new_theta_vars[theta_rhs_0_name]
                            theta_rhs_1 = new_theta_vars[theta_rhs_1_name]
                            theta_rhs_2 = new_theta_vars[theta_rhs_2_name]
                            
                            rhs_expr = u + theta_rhs_0 + theta_rhs_1 * u + theta_rhs_2 * u * u
                            b_new.append(rhs_expr)
                        else:
                            b_new.append(u)  # 使用原右端项
                    else:
                        b_new.append(b_element)
        else:
            b_new = b_with_theta
        
        print(f"✓ 重建完成，新矩阵形状: {A_new.shape}")
        
        return A_new, np.array(b_new) if b_new else np.array([])

    def solve_theta_optimization_problem(self, A_with_theta, b_with_theta, 
                                    c_vector, A_g_list, b_g_list,
                                    objective_type='min_sum_A') -> Dict:
        """以theta为变量求解优化问题（修复版本）"""
        
        print(f"\n=== 以Theta为变量求解优化问题 ===")
        
        # 步骤1：检查theta变量是否已存在
        if not hasattr(self, 'theta_vars') or not self.theta_vars:
            print("错误：theta变量未定义！")
            return {'status': 'ERROR', 'error': 'theta_vars_not_defined'}
        
        print(f"发现 {len(self.theta_vars)} 个theta变量")
        
        # 步骤2：检查A_with_theta是否包含theta表达式
        if A_with_theta.dtype != object:
            print("警告：A_with_theta不包含theta表达式")
            return {'status': 'ERROR', 'error': 'no_theta_expressions_in_A'}
        
        # 步骤3：创建新的优化模型
        theta_model = gp.Model('ThetaOptimization')
        theta_model.Params.OutputFlag = 1
        
        # 步骤4：将theta变量添加到新模型中
        new_theta_vars = self._add_theta_vars_to_new_model(theta_model)
        theta_model.update()
        
        # 步骤5：为新模型重建参数化矩阵
        union_analysis = getattr(self, '_current_union_analysis', {})
        A_new, b_new = self._rebuild_parametric_matrices_for_new_model(
            A_with_theta, b_with_theta, new_theta_vars, union_analysis
        )
        
        # 步骤6：创建新的决策变量
        var_info = self._get_variable_info()
        pg_vars = theta_model.addVars(self.ng, self.T, lb=0, name='pg')
        x_vars = theta_model.addVars(self.ng, self.T, lb=0, ub=1, name='x')
        coc_vars = theta_model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower_vars = theta_model.addVars(self.ng, self.T, lb=0, name='cpower')
        
        # 步骤7：构建变量向量
        var_vector = self._build_variable_vector(pg_vars, x_vars, coc_vars, cpower_vars)
        
        # 步骤8：更新模型以确保所有变量可用
        theta_model.update()
        
        # 步骤9：测试添加约束
        # print(f"测试添加约束...")
        # try:
        #     # 测试能否访问A_new中的表达式
        #     test_row = 498
        #     if test_row < A_new.shape[0]:
        #         lhs_expr = 0
        #         expr_count = 0
                
        #         for j in range(A_new.shape[1]):
        #             element = A_new[test_row, j]
                    
        #             if isinstance(element, (int, float)) and element != 0:
        #                 if j < len(var_vector):
        #                     lhs_expr += element * var_vector[j]
        #                     expr_count += 1
        #             elif hasattr(element, 'getValue'):
        #                 # 检查这是否是新模型中的表达式
        #                 try:
        #                     if j < len(var_vector):
        #                         lhs_expr += element * var_vector[j]
        #                         expr_count += 1
        #                 except Exception as e:
        #                     print(f"列 {j} 表达式处理失败: {e}")
                
        #         if expr_count > 0:
        #             # 添加测试约束
        #             theta_model.addConstr(lhs_expr <= 100, name=f'test_constraint_{test_row}')
        #             print(f"✓ 成功添加测试约束，包含 {expr_count} 项")
        #         else:
        #             print(f"⚠ 测试行 {test_row} 没有有效表达式")
        #     else:
        #         print(f"⚠ 测试行 {test_row} 超出矩阵范围 {A_new.shape[0]}")
        
        # except Exception as e:
        #     print(f"❌ 添加测试约束失败: {e}")
        #     return {'status': 'ERROR', 'error': f'constraint_addition_failed: {e}'}

        theta_model.addConstr(gp.quicksum(A_new[498, i] for i in range(A_new.shape[1])) <= 100, name=f'constraint')

        # 步骤10：设置简单目标函数并求解
        theta_model.setObjective(0, GRB.MINIMIZE)
        
        print(f"开始求解...")
        theta_model.optimize()
        
        # 步骤11：返回结果
        if theta_model.status == GRB.OPTIMAL:
            print(f"✓ 求解成功")
            return {
                'status': 'OPTIMAL',
                'objective_value': theta_model.objVal,
                'solve_time': theta_model.Runtime
            }
        else:
            print(f"❌ 求解失败，状态: {theta_model.status}")
            return {
                'status': f'FAILED_{theta_model.status}',
                'gurobi_status': theta_model.status
            }

    def _get_variable_info(self) -> Dict:
        """获取变量信息"""
        n_pg = self.ng * self.T
        n_x = self.ng * self.T
        n_coc = self.ng * (self.T - 1)
        n_cpower = self.ng * self.T
        total_vars = n_pg + n_x + n_coc + n_cpower
        
        return {
            'n_pg': n_pg,
            'n_x': n_x,
            'n_coc': n_coc,
            'n_cpower': n_cpower,
            'total_vars': total_vars
        }

    def _build_variable_vector(self, pg_vars, x_vars, coc_vars, cpower_vars) -> List:
        """构建变量向量，保持与矩阵提取时相同的顺序"""
        var_vector = []
        
        # pg变量
        for g in range(self.ng):
            for t in range(self.T):
                var_vector.append(pg_vars[g, t])
        
        # x变量
        for g in range(self.ng):
            for t in range(self.T):
                var_vector.append(x_vars[g, t])
        
        # coc变量
        for g in range(self.ng):
            for t in range(self.T-1):
                var_vector.append(coc_vars[g, t])
        
        # cpower变量
        for g in range(self.ng):
            for t in range(self.T):
                var_vector.append(cpower_vars[g, t])
        
        return var_vector

    def _verify_theta_vars_in_matrix(self, A_matrix) -> set:
        """验证A矩阵中包含的theta变量"""
        theta_vars_found = set()
        
        if A_matrix.dtype == object and A_matrix.size > 0:
            for i in range(A_matrix.shape[0]):
                for j in range(A_matrix.shape[1]):
                    element = A_matrix[i, j]
                    if hasattr(element, 'getValue'):
                        # 这是一个Gurobi表达式，提取其中的变量
                        try:
                            # 获取表达式中的所有变量
                            expr_vars = self._extract_vars_from_expression(element)
                            theta_vars_found.update(expr_vars)
                        except:
                            pass
        
        return theta_vars_found

    def _extract_vars_from_expression(self, expr) -> set:
        """从Gurobi表达式中提取变量名"""
        var_names = set()
        
        try:
            # 获取表达式中的所有变量
            if hasattr(expr, 'getVars'):
                for var in expr.getVars():
                    var_names.add(var.VarName)
            elif hasattr(expr, 'VarName'):
                # 单个变量
                var_names.add(expr.VarName)
        except:
            pass
        
        return var_names

    def _add_theta_vars_to_new_model(self, new_model) -> Dict:
        """将theta变量添加到新模型中"""
        new_theta_vars = {}
        
        for var_name, original_var in self.theta_vars.items():
            # 在新模型中创建对应的theta变量
            new_var = new_model.addVar(
                lb=original_var.LB,
                ub=original_var.UB,
                name=original_var.VarName
            )
            new_theta_vars[var_name] = new_var
            
            # 重要：更新表达式中的变量引用
            # 这里需要确保A_with_theta中的表达式引用新模型中的变量
        
        return new_theta_vars
    
    def check_theta_readiness_for_optimization(self, A_with_theta, b_with_theta) -> bool:
        """检查theta优化的准备状态"""
        
        print(f"\n=== 检查Theta优化准备状态 ===")
        
        # 1. 检查theta变量是否存在
        if not hasattr(self, 'theta_vars'):
            print("❌ theta_vars属性不存在")
            return False
        
        if not self.theta_vars:
            print("❌ theta_vars为空")
            return False
        
        print(f"✓ 找到 {len(self.theta_vars)} 个theta变量")
        
        # 2. 检查A矩阵是否包含表达式
        if A_with_theta.dtype != object:
            print("❌ A_with_theta不包含Gurobi表达式")
            return False
        
        print(f"✓ A矩阵包含表达式对象")
        
        # 3. 测试表达式访问
        try:
            expr_count = 0
            for i in range(min(3, A_with_theta.shape[0])):
                for j in range(min(5, A_with_theta.shape[1])):
                    element = A_with_theta[i, j]
                    if hasattr(element, 'getValue'):
                        expr_count += 1
                        # 测试能否访问表达式
                        _ = str(element)
            
            print(f"✓ 找到 {expr_count} 个可访问的表达式")
            
        except Exception as e:
            print(f"❌ 表达式访问测试失败: {e}")
            return False
        
        # 4. 检查b向量
        if hasattr(b_with_theta, '__len__'):
            b_expr_count = 0
            for element in b_with_theta:
                if hasattr(element, 'getValue'):
                    b_expr_count += 1
            print(f"✓ b向量中有 {b_expr_count} 个表达式")
        
        print(f"✓ 所有检查通过，可以进行theta优化")
        return True

    def safe_solve_theta_optimization(self, A_with_theta, b_with_theta, 
                                    c_vector, A_g_list, b_g_list,
                                    objective_type='min_sum_A') -> Dict:
        """安全的theta优化求解（包含完整检查）"""
        
        # 执行实际的优化求解
        return self.solve_theta_optimization_problem(
            A_with_theta, b_with_theta, c_vector, A_g_list, b_g_list, objective_type
        )
        
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
        
        # 求解原始模型
        pg_sol, x_sol, total_cost = uc.solve()
        
        if pg_sol is not None:
            print(f"\n原始模型求解成功，总成本: {total_cost}")
            
            # 获取并集约束分析
            union_analysis = uc.get_union_constraints_and_fractional_info(json_file, sample_id)
            
            # 演示矩阵提取
            uc.demonstrate_matrix_extraction(union_analysis)
            
            # 详细示例：两种theta处理方式
            print(f"\n=== 详细示例：约束矩阵形式 ===")
            
            # 情况1: theta保持为变量（A矩阵包含theta表达式）
            print(f"\n情况1: Theta保持为变量")
            matrices_theta_vars = uc.extract_constraint_matrices(union_analysis, theta_values=None)
            
            A_with_theta = matrices_theta_vars['A_power_dcpf']
            A_g_list = matrices_theta_vars['A_g_list']
            c_vector = matrices_theta_vars['c_vector']
            
            print(f"A矩阵(包含theta): {A_with_theta.shape}")
            print(f"机组约束矩阵数量: {len(A_g_list)}")
            print(f"目标函数向量: {c_vector.shape}")
            
            # 情况2: theta给定具体值（A矩阵为数值矩阵）
            print(f"\n情况2: 给定Theta具体值")
            
            # 定义测试theta值
            test_theta_values = {}
            branch_ids = set()
            for constraint in union_analysis.get('union_constraints', []):
                branch_ids.add(constraint['branch_id'])
            
            for branch_id in list(branch_ids)[:3]:  # 只设置前3条支路
                # 右端项theta
                test_theta_values[f'theta_branch_{branch_id}_rhs_0'] = 0.1
                test_theta_values[f'theta_branch_{branch_id}_rhs_1'] = 0.01
                test_theta_values[f'theta_branch_{branch_id}_rhs_2'] = 0.001
                
                # 系数theta
                for g in range(min(3, uc.ng)):  # 只设置前3个机组
                    test_theta_values[f'theta_branch_{branch_id}_unit_{g}_0'] = 0.05
                    test_theta_values[f'theta_branch_{branch_id}_unit_{g}_1'] = 0.01
                    test_theta_values[f'theta_branch_{branch_id}_unit_{g}_2'] = 0.001
            
            matrices_theta_numeric = uc.extract_constraint_matrices(union_analysis, test_theta_values)
            
            A_numeric = matrices_theta_numeric['A_power_dcpf']
            b_numeric = matrices_theta_numeric['b_power_dcpf']
            print(f"A矩阵(数值): {A_numeric.shape}")
            print(f"设置的theta参数数量: {len(test_theta_values)}")
            
            if A_numeric.size > 0:
                print(f"A矩阵元素范围: [{np.min(A_numeric):.6f}, {np.max(A_numeric):.6f}]")
                print(f"A矩阵非零元素: {np.count_nonzero(A_numeric)}")
            
            # 如果A矩阵包含theta变量，也可以后续计算数值
            # if A_with_theta.dtype == object and A_with_theta.size > 0:
            #     print(f"\n将theta变量矩阵转换为数值矩阵:")
            #     A_converted, b_converted = self.evaluate_parametric_constraints_with_theta_values(
            #         A_numeric, b_numeric, test_theta_values
            #     )
            #     print(f"转换后矩阵: {A_converted.shape}")
            #     if A_converted.size > 0:
            #         print(f"转换后元素范围: [{np.min(A_converted):.6f}, {np.max(A_converted):.6f}]")
            
            # 保存结果
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存矩阵信息（不保存大矩阵本身）
            matrix_info = {
                'sample_id': sample_id,
                'timestamp': timestamp,
                'original_cost': total_cost,
                'theta_variable_mode': {
                    'A_shape': A_with_theta.shape,
                    'A_dtype': str(A_with_theta.dtype),
                    'has_theta_expressions': A_with_theta.dtype == object
                },
                'theta_numeric_mode': {
                    'A_shape': A_numeric.shape,
                    'A_dtype': str(A_numeric.dtype),
                    'theta_values_used': test_theta_values
                },
                'unit_constraints': {
                    'num_units': len(A_g_list),
                    'A_g_shapes': [A_g.shape for A_g in A_g_list]
                },
                'objective': {
                    'c_vector_shape': c_vector.shape,
                    'nonzero_coefficients': int(np.count_nonzero(c_vector))
                }
            }
            
            info_file = f"result/constraint_matrices_info_{timestamp}.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(matrix_info, f, indent=2, ensure_ascii=False)
            
            print(f"\n约束矩阵信息已保存到: {info_file}")
            
        else:
            print("原始模型求解失败")
            
    except FileNotFoundError:
        print(f"JSON文件未找到: {json_file}")
        print("使用默认数据进行矩阵提取演示")
        
        # 使用默认数据
        load_df = pd.read_csv('src/load.csv', header=None)
        Pd = load_df.values
        ppc = pypower.case9.case9()
        T_delta = 1
        uc = UnitCommitmentModel(ppc, Pd, T_delta)
        
        # 提取约束矩阵（无参数化约束）
        constraint_matrices = uc.extract_constraint_matrices()
        uc.print_matrix_summary(constraint_matrices)
        
    matrices_with_theta = uc.extract_constraint_matrices(union_analysis, theta_values=None)
    A_with_theta = matrices_with_theta['A_power_dcpf']
    b_with_theta = matrices_with_theta['b_power_dcpf']
    
    # 步骤5: 检查准备状态
    if uc.check_theta_readiness_for_optimization(A_with_theta, b_with_theta):
        # 步骤6: 执行theta优化
        theta_results = uc.safe_solve_theta_optimization(
            A_with_theta, b_with_theta, 
            matrices_with_theta['c_vector'],
            matrices_with_theta['A_g_list'], 
            matrices_with_theta['b_g_list'],
            'min_sum_A'
        )
        
        print(f"Theta优化结果: {theta_results['status']}")


