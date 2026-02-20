import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import io
import time
import json
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
from pypower.idx_gen import GEN_BUS, PMIN, PMAX

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

# 设置输出缓冲
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)


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
                    
            except Exception as e:
                print(f"加载样本 {sample_id} 时出错: {e}", flush=True)
                all_samples_data.append({
                    'sample_id': sample_id,
                    'error': str(e)
                })
        
        return all_samples_data
    
    def extract_active_constraints_and_variables(self, sample_id: int) -> Tuple[List, List, np.ndarray]:
        sample = self.get_sample_data(sample_id)
        if sample is None:
            return [], [], np.array([])
        
        active_set = sample.get('active_set', [])
        pd_data = np.array(sample.get('pd_data', []))
        
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
            hidden_dims = [256, 512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
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
        self.active_set_data = active_set_data
        
        # 设置设备
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # 输入输出维度
        self.input_dim = self.nb * self.T
        self.output_dim = self.T
        
        # 初始化网络
        if TORCH_AVAILABLE:
            self.network = DualVariablePredictorNet(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            ).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        
        # 求解原始问题获取对偶变量真值
        self.lambda_true = self._solve_for_true_dual_variables()
        
        print(f"✓ 对偶变量预测训练器初始化完成", flush=True)
        print(f"  - 输入维度: {self.input_dim}, 输出维度: {self.output_dim}", flush=True)
    
    def _solve_for_true_dual_variables(self) -> np.ndarray:
        """求解UC问题获取功率平衡约束的对偶变量真值"""
        lambda_true = []
        
        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']
            model = gp.Model('uc_for_dual')
            model.Params.OutputFlag = 0
            
            # 变量
            pg = model.addVars(self.ng, self.T, lb=0, name='pg')
            x = model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
            coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
            cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
            
            # 功率平衡约束
            for t in range(self.T):
                model.addConstr(
                    gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]),
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
                    model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + self.gen[g, PMAX] * (1 - x[g, t-1]))
                    model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + self.gen[g, PMAX] * (1 - x[g, t]))
            
            # 最小开关机时间约束
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            for g in range(self.ng):
                for tau in range(1, Ton+1):
                    for t1 in range(self.T - tau):
                        model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
                for tau in range(1, Toff+1):
                    for t1 in range(self.T - tau):
                        model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])
            
            # 启停成本
            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]))
                    model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]))
            
            # 发电成本
            for t in range(self.T):
                for g in range(self.ng):
                    model.addConstr(cpower[g, t] >= self.gencost[g, -2]/self.T_delta * pg[g, t] + 
                                  self.gencost[g, -1]/self.T_delta * x[g, t])
            
            # 目标函数
            obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                   gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                lambda_sample = np.zeros(self.T)
                for t in range(self.T):
                    constr = model.getConstrByName(f'power_balance_{t}')
                    if constr is not None:
                        try:
                            lambda_sample[t] = constr.Pi
                        except:
                            lambda_sample[t] = 0.0
                lambda_true.append(lambda_sample)
            else:
                lambda_true.append(np.zeros(self.T))
        
        return np.array(lambda_true)
    
    def _extract_features(self, sample_id: int) -> np.ndarray:
        """提取Pd数据作为特征"""
        pd_data = self.active_set_data[sample_id]['pd_data']
        return pd_data.flatten()
    
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
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}", flush=True)
        
        print(f"✓ 对偶变量预测网络训练完成", flush=True)
    
    def predict(self, pd_data: np.ndarray) -> np.ndarray:
        """预测对偶变量"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")
        
        self.network.eval()
        pd_flat = pd_data.flatten()
        pd_tensor = torch.tensor(pd_flat, dtype=torch.float32, device=self.device)
        
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

class SubproblemSurrogateNet(nn.Module):
    """
    单机组子问题的代理约束网络
    
    输入: Pd数据 + 对偶变量λ (pd_dim + T)
    输出: 代理约束参数 (alpha, beta)
          约束形式: sum_t(alpha_t * x_t) <= beta
    """
    
    def __init__(self, input_dim: int, T: int, hidden_dims: List[int] = None):
        super(SubproblemSurrogateNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        # alpha网络：预测约束系数
        alpha_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            alpha_layers.append(nn.Linear(prev_dim, hidden_dim))
            alpha_layers.append(nn.ReLU())
            alpha_layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        alpha_layers.append(nn.Linear(prev_dim, T))
        self.alpha_net = nn.Sequential(*alpha_layers)
        
        # beta网络：预测右端项（标量）
        beta_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            beta_layers.append(nn.Linear(prev_dim, hidden_dim))
            beta_layers.append(nn.ReLU())
            beta_layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        beta_layers.append(nn.Linear(prev_dim, 1))
        beta_layers.append(nn.Softplus())  # 确保beta非负
        self.beta_net = nn.Sequential(*beta_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        alpha = self.alpha_net(x)
        beta = self.beta_net(x)
        return alpha, beta


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
                 lambda_predictor: DualVariablePredictorTrainer = None, device=None):
        """
        初始化单机组子问题代理约束训练器
        
        Args:
            ppc: PyPower案例数据
            active_set_data: 活动集数据
            T_delta: 时间间隔
            unit_id: 机组索引
            lambda_predictor: 已训练的对偶变量预测器（可选）
            device: 计算设备
        """
        self.ppc = ppc
        ppc_int = ext2int(ppc)
        self.baseMVA = ppc_int['baseMVA']
        self.bus = ppc_int['bus']
        self.gen = ppc_int['gen']
        self.branch = ppc_int['branch']
        self.gencost = ppc_int['gencost']
        self.n_samples = len(active_set_data)
        self.T_delta = T_delta
        self.unit_id = unit_id
        
        if isinstance(active_set_data, list):
            self.T = active_set_data[0]['pd_data'].shape[1]
        else:
            self.T = active_set_data['pd_data'].shape[1]
            
        self.ng = self.gen.shape[0]
        self.nb = self.bus.shape[0]
        self.active_set_data = active_set_data
        
        # 对偶变量预测器
        self.lambda_predictor = lambda_predictor
        
        # 设备
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # BCD迭代参数
        self.rho_primal = 1e-2
        self.rho_dual = 1e-2
        self.rho_opt = 1e-2
        self.gamma = 1e-1
        self.mu_lower_bound = 0.1
        self.iter_number = 0
        
        # 初始化原始变量和对偶变量存储
        self.pg = np.zeros((self.n_samples, self.T))
        self.x = np.zeros((self.n_samples, self.T))
        self.coc = np.zeros((self.n_samples, self.T-1))
        self.cpower = np.zeros((self.n_samples, self.T))
        self.mu = np.ones((self.n_samples,)) * self.mu_lower_bound  # 代理约束的对偶变量（标量）
        
        # 获取对偶变量λ
        self.lambda_vals = self._get_lambda_values()
        
        # 初始化代理约束参数
        self.alpha_values = np.zeros((self.n_samples, self.T))
        self.beta_values = np.ones((self.n_samples,))
        
        # 初始化神经网络
        if TORCH_AVAILABLE:
            self._init_neural_network()
        
        # 初始化求解
        self._initialize_solve()
        
        print(f"✓ 机组{unit_id}子问题代理约束训练器初始化完成", flush=True)
    
    def _get_lambda_values(self) -> np.ndarray:
        """获取对偶变量λ"""
        if self.lambda_predictor is not None:
            # 使用预测器
            lambda_vals = []
            for sample_id in range(self.n_samples):
                pd_data = self.active_set_data[sample_id]['pd_data']
                lambda_pred = self.lambda_predictor.predict(pd_data)
                lambda_vals.append(lambda_pred)
            return np.array(lambda_vals)
        else:
            # 使用真值（需要先求解原问题）
            return self._solve_for_lambda()
    
    def _solve_for_lambda(self) -> np.ndarray:
        """求解原问题获取λ"""
        lambda_vals = []
        for sample_id in range(self.n_samples):
            Pd = self.active_set_data[sample_id]['pd_data']
            model = gp.Model('uc_for_lambda')
            model.Params.OutputFlag = 0
            
            pg = model.addVars(self.ng, self.T, lb=0, name='pg')
            x = model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
            coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
            cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')
            
            # 功率平衡约束
            for t in range(self.T):
                model.addConstr(gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(Pd[:, t]),
                              name=f'power_balance_{t}')
            
            # 其他约束（简化）
            for g in range(self.ng):
                for t in range(self.T):
                    model.addConstr(pg[g, t] >= self.gen[g, PMIN] * x[g, t])
                    model.addConstr(pg[g, t] <= self.gen[g, PMAX] * x[g, t])
            
            Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
            Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
            for g in range(self.ng):
                for t in range(1, self.T):
                    model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + self.gen[g, PMAX] * (1 - x[g, t-1]))
                    model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + self.gen[g, PMAX] * (1 - x[g, t]))
            
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            for g in range(self.ng):
                for tau in range(1, Ton+1):
                    for t1 in range(self.T - tau):
                        model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
                for tau in range(1, Toff+1):
                    for t1 in range(self.T - tau):
                        model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])
            
            start_cost = self.gencost[:, 1]
            shut_cost = self.gencost[:, 2]
            for t in range(1, self.T):
                for g in range(self.ng):
                    model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]))
                    model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]))
            
            for t in range(self.T):
                for g in range(self.ng):
                    model.addConstr(cpower[g, t] >= self.gencost[g, -2]/self.T_delta * pg[g, t] + 
                                  self.gencost[g, -1]/self.T_delta * x[g, t])
            
            obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                   gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                lambda_sample = np.zeros(self.T)
                for t in range(self.T):
                    constr = model.getConstrByName(f'power_balance_{t}')
                    if constr is not None:
                        try:
                            lambda_sample[t] = constr.Pi
                        except:
                            pass
                lambda_vals.append(lambda_sample)
            else:
                lambda_vals.append(np.zeros(self.T))
        
        return np.array(lambda_vals)
    
    def _init_neural_network(self):
        """初始化代理约束神经网络"""
        input_dim = self.nb * self.T + self.T  # Pd + λ
        
        self.surrogate_net = SubproblemSurrogateNet(
            input_dim=input_dim,
            T=self.T,
            hidden_dims=[128, 256, 128]
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.surrogate_net.parameters(), lr=1e-4)
        
        print(f"  - 代理约束网络输入维度: {input_dim}", flush=True)
    
    def _initialize_solve(self):
        """初始化求解，获取初始的pg, x, coc, cpower"""
        g = self.unit_id
        
        for sample_id in range(self.n_samples):
            lambda_val = self.lambda_vals[sample_id]
            
            model = gp.Model('init_subproblem')
            model.Params.OutputFlag = 0
            
            pg = model.addVars(self.T, lb=0, name='pg')
            x = model.addVars(self.T, vtype=GRB.BINARY, name='x')
            coc = model.addVars(self.T-1, lb=0, name='coc')
            cpower = model.addVars(self.T, lb=0, name='cpower')
            
            # 发电上下限约束
            for t in range(self.T):
                model.addConstr(pg[t] >= self.gen[g, PMIN] * x[t], name=f'pg_lower_{t}')
                model.addConstr(pg[t] <= self.gen[g, PMAX] * x[t], name=f'pg_upper_{t}')
            
            # 爬坡约束
            Ru = 0.4 * self.gen[g, PMAX] / self.T_delta
            Rd = 0.4 * self.gen[g, PMAX] / self.T_delta
            for t in range(1, self.T):
                model.addConstr(pg[t] - pg[t-1] <= Ru * x[t-1] + self.gen[g, PMAX] * (1 - x[t-1]), name=f'ramp_up_{t}')
                model.addConstr(pg[t-1] - pg[t] <= Rd * x[t] + self.gen[g, PMAX] * (1 - x[t]), name=f'ramp_down_{t}')
            
            # 最小开关机时间约束
            Ton = min(4, self.T)
            Toff = min(4, self.T)
            for tau in range(1, Ton+1):
                for t1 in range(self.T - tau):
                    model.addConstr(x[t1+1] - x[t1] <= x[t1+tau], name=f'min_on_{tau}_{t1}')
            for tau in range(1, Toff+1):
                for t1 in range(self.T - tau):
                    model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau], name=f'min_off_{tau}_{t1}')
            
            # 启停成本
            start_cost = self.gencost[g, 1]
            shut_cost = self.gencost[g, 2]
            for t in range(1, self.T):
                model.addConstr(coc[t-1] >= start_cost * (x[t] - x[t-1]), name=f'start_cost_{t}')
                model.addConstr(coc[t-1] >= shut_cost * (x[t-1] - x[t]), name=f'shut_cost_{t}')
            
            # 发电成本
            for t in range(self.T):
                model.addConstr(cpower[t] >= self.gencost[g, -2]/self.T_delta * pg[t] + 
                              self.gencost[g, -1]/self.T_delta * x[t], name=f'cpower_{t}')
            
            # 目标函数: cost - λᵀ × pg
            obj = gp.quicksum(cpower[t] for t in range(self.T))
            obj += gp.quicksum(coc[t] for t in range(self.T-1))
            obj -= gp.quicksum(lambda_val[t] * pg[t] for t in range(self.T))
            
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                self.pg[sample_id] = np.array([pg[t].X for t in range(self.T)])
                self.x[sample_id] = np.array([x[t].X for t in range(self.T)])
                self.coc[sample_id] = np.array([coc[t].X for t in range(self.T-1)])
                self.cpower[sample_id] = np.array([cpower[t].X for t in range(self.T)])
    
    def iter_with_primal_block(self, sample_id: int, alpha: np.ndarray, beta: float):
        """
        BCD迭代：原始块
        固定代理约束参数(alpha, beta)和对偶变量(mu)，更新原始变量(pg, x)
        
        目标函数:
            min  cost - λᵀ × pg 
                 + rho_primal * max(0, sum_t(alpha_t * x_t) - beta)
                 + rho_opt * |sum_t(alpha_t * x_t) - beta| * mu
        """
        g = self.unit_id
        lambda_val = self.lambda_vals[sample_id]
        mu_val = self.mu[sample_id]
        
        model = gp.Model('primal_block')
        model.Params.OutputFlag = 0
        
        # 变量（x为连续，LP松弛）
        pg = model.addVars(self.T, lb=0, name='pg')
        x = model.addVars(self.T, lb=0, ub=1, name='x')
        coc = model.addVars(self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.T, lb=0, name='cpower')
        
        # 代理约束违反量
        surrogate_viol = model.addVar(lb=0, name='surrogate_viol')
        surrogate_abs = model.addVar(lb=0, name='surrogate_abs')
        
        # 发电上下限约束
        for t in range(self.T):
            model.addConstr(pg[t] >= self.gen[g, PMIN] * x[t], name=f'pg_lower_{t}')
            model.addConstr(pg[t] <= self.gen[g, PMAX] * x[t], name=f'pg_upper_{t}')
        
        # 爬坡约束
        Ru = 0.4 * self.gen[g, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[g, PMAX] / self.T_delta
        for t in range(1, self.T):
            model.addConstr(pg[t] - pg[t-1] <= Ru * x[t-1] + self.gen[g, PMAX] * (1 - x[t-1]), name=f'ramp_up_{t}')
            model.addConstr(pg[t-1] - pg[t] <= Rd * x[t] + self.gen[g, PMAX] * (1 - x[t]), name=f'ramp_down_{t}')
        
        # 最小开关机时间约束
        Ton = min(4, self.T)
        Toff = min(4, self.T)
        for tau in range(1, Ton+1):
            for t1 in range(self.T - tau):
                model.addConstr(x[t1+1] - x[t1] <= x[t1+tau], name=f'min_on_{tau}_{t1}')
        for tau in range(1, Toff+1):
            for t1 in range(self.T - tau):
                model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau], name=f'min_off_{tau}_{t1}')
        
        # 启停成本
        start_cost = self.gencost[g, 1]
        shut_cost = self.gencost[g, 2]
        for t in range(1, self.T):
            model.addConstr(coc[t-1] >= start_cost * (x[t] - x[t-1]), name=f'start_cost_{t}')
            model.addConstr(coc[t-1] >= shut_cost * (x[t-1] - x[t]), name=f'shut_cost_{t}')
        
        # 发电成本
        for t in range(self.T):
            model.addConstr(cpower[t] >= self.gencost[g, -2]/self.T_delta * pg[t] + 
                          self.gencost[g, -1]/self.T_delta * x[t], name=f'cpower_{t}')
        
        # 代理约束: sum_t(alpha_t * x_t) <= beta
        surrogate_lhs = gp.quicksum(alpha[t] * x[t] for t in range(self.T))
        model.addConstr(surrogate_viol >= surrogate_lhs - beta, name='surrogate_viol_constr')
        model.addConstr(surrogate_abs >= surrogate_lhs - beta, name='surrogate_abs_pos')
        model.addConstr(surrogate_abs >= beta - surrogate_lhs, name='surrogate_abs_neg')
        
        # 目标函数
        obj_cost = gp.quicksum(cpower[t] for t in range(self.T))
        obj_cost += gp.quicksum(coc[t] for t in range(self.T-1))
        obj_lambda = -gp.quicksum(lambda_val[t] * pg[t] for t in range(self.T))
        obj_primal = self.rho_primal * surrogate_viol
        obj_opt = self.rho_opt * surrogate_abs * mu_val
        
        model.setObjective(obj_cost + obj_lambda + obj_primal + obj_opt, GRB.MINIMIZE)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([pg[t].X for t in range(self.T)])
            x_sol = np.array([x[t].X for t in range(self.T)])
            coc_sol = np.array([coc[t].X for t in range(self.T-1)])
            cpower_sol = np.array([cpower[t].X for t in range(self.T)])
            return pg_sol, x_sol, coc_sol, cpower_sol
        else:
            print(f"警告: 原始块求解失败，状态: {model.status}", flush=True)
            return None, None, None, None
    
    def iter_with_dual_block(self, sample_id: int, alpha: np.ndarray, beta: float):
        """
        BCD迭代：对偶块
        固定原始变量(pg, x)和代理约束参数(alpha, beta)，更新对偶变量(mu)
        
        对偶问题:
            min  rho_dual * |dual_feasibility| + rho_opt * |constraint_violation| * mu
            s.t. mu >= mu_lower_bound
        """
        g = self.unit_id
        x_val = self.x[sample_id]
        
        model = gp.Model('dual_block')
        model.Params.OutputFlag = 0
        
        # 对偶变量
        if self.iter_number < 50:
            mu = model.addVar(lb=self.mu_lower_bound, name='mu')
        else:
            mu = model.addVar(lb=0, name='mu')
        
        # 代理约束违反量
        surrogate_expr = np.sum(alpha * x_val) - beta
        surrogate_viol = abs(surrogate_expr)
        
        # 对偶可行性约束
        # 对于代理约束 sum_t(alpha_t * x_t) <= beta，
        # x变量的对偶约束中需要加上 alpha_t * mu
        # 简化处理：最小化 mu 的大小同时满足约束
        
        obj_dual = 0
        obj_opt = self.rho_opt * surrogate_viol * mu
        
        model.setObjective(obj_dual + obj_opt, GRB.MINIMIZE)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return mu.X
        else:
            return self.mu_lower_bound
    
    def _extract_features(self, sample_id: int) -> np.ndarray:
        """提取特征: [Pd, λ]"""
        pd_data = self.active_set_data[sample_id]['pd_data']
        pd_flat = pd_data.flatten()
        lambda_val = self.lambda_vals[sample_id]
        return np.concatenate([pd_flat, lambda_val])
    
    def loss_function_differentiable(self, sample_id: int, alpha_tensor: torch.Tensor, 
                                     beta_tensor: torch.Tensor, device) -> torch.Tensor:
        """
        可微分的loss函数（与uc_NN_BCD.py的loss_function_differentiable一致）
        
        使用BCD迭代得到的变量值(x, mu)计算loss
        
        Loss = rho_primal * obj_primal + rho_dual * obj_dual + rho_opt * obj_opt
        
        其中:
        - obj_primal: 代理约束违反量 max(0, sum_t(alpha_t * x_t) - beta)
        - obj_dual: 对偶约束违反量 |alpha_t * mu - gradient_t|
        - obj_opt: 互补松弛条件 |sum_t(alpha_t * x_t) - beta| * mu
        """
        g = self.unit_id
        
        # 从BCD迭代得到的变量
        x_val = torch.tensor(self.x[sample_id], dtype=torch.float32, device=device)
        mu_val = torch.tensor(self.mu[sample_id], dtype=torch.float32, device=device)
        lambda_val = torch.tensor(self.lambda_vals[sample_id], dtype=torch.float32, device=device)
        
        # 机组参数
        gen_PMIN = torch.tensor(self.gen[g, PMIN], dtype=torch.float32, device=device)
        gen_PMAX = torch.tensor(self.gen[g, PMAX], dtype=torch.float32, device=device)
        gencost_var = torch.tensor(self.gencost[g, -2] / self.T_delta, dtype=torch.float32, device=device)
        gencost_fixed = torch.tensor(self.gencost[g, -1] / self.T_delta, dtype=torch.float32, device=device)
        
        # 目标x（如果有的话）
        unit_commitment = self.active_set_data[sample_id].get('unit_commitment_matrix', None)
        if unit_commitment is not None and g < unit_commitment.shape[0]:
            x_target = torch.tensor(unit_commitment[g], dtype=torch.float32, device=device)
        else:
            x_target = None
        
        # ========== 计算obj_primal ==========
        # 代理约束违反量: max(0, sum_t(alpha_t * x_t) - beta)
        surrogate_lhs = torch.sum(alpha_tensor * x_val)
        surrogate_violation = torch.relu(surrogate_lhs - beta_tensor.squeeze())
        obj_primal = surrogate_violation
        
        # ========== 计算obj_opt ==========
        # 互补松弛: |sum_t(alpha_t * x_t) - beta| * mu
        surrogate_abs = torch.abs(surrogate_lhs - beta_tensor.squeeze())
        obj_opt = surrogate_abs * mu_val
        
        # ========== 计算obj_dual ==========
        # x变量的对偶约束: gencost_fixed - lambda + ... + alpha * mu = 0
        # 简化版本：只考虑代理约束的贡献
        obj_dual = torch.tensor(0.0, device=device)
        for t in range(self.T):
            # 基础对偶表达式（简化）
            dual_expr = gencost_fixed - lambda_val[t]
            # 代理约束的对偶贡献
            dual_expr = dual_expr + alpha_tensor[t] * mu_val
            obj_dual = obj_dual + torch.abs(dual_expr)
        
        # ========== 附加损失：确保代理约束有效 ==========
        # 1. 代理约束应该切掉LP松弛解但保留整数解
        if x_target is not None:
            # 对于真实整数解，代理约束应该被满足
            target_lhs = torch.sum(alpha_tensor * x_target)
            target_violation = torch.relu(target_lhs - beta_tensor.squeeze())
            loss_target_feasibility = target_violation * 10.0  # 大权重
        else:
            loss_target_feasibility = torch.tensor(0.0, device=device)
        
        # 2. 整数性损失：鼓励x接近0或1
        loss_integrality = torch.sum(x_val * (1 - x_val))
        
        # 3. 与目标x的偏差
        if x_target is not None:
            loss_deviation = torch.sum((x_val - x_target) ** 2)
        else:
            loss_deviation = torch.tensor(0.0, device=device)
        
        # 总损失
        loss = (self.rho_primal * obj_primal + 
                self.rho_dual * obj_dual + 
                self.rho_opt * obj_opt +
                loss_target_feasibility +
                0.1 * loss_integrality +
                loss_deviation)
        
        return loss
    
    def iter_with_surrogate_nn(self, num_epochs: int = 10):
        """
        BCD迭代：神经网络更新代理约束参数
        使用loss_function_differentiable进行训练
        """
        if not TORCH_AVAILABLE:
            return
        
        self.surrogate_net.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for sample_id in range(self.n_samples):
                # 提取特征
                features = self._extract_features(sample_id)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # 前向传播
                alpha_out, beta_out = self.surrogate_net(features_tensor)
                alpha_tensor = alpha_out.squeeze(0)
                beta_tensor = beta_out.squeeze(0)
                
                # 计算loss
                self.optimizer.zero_grad()
                loss = self.loss_function_differentiable(
                    sample_id, alpha_tensor, beta_tensor, self.device
                )
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.surrogate_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 更新alpha和beta值
                self.alpha_values[sample_id] = alpha_tensor.detach().cpu().numpy()
                self.beta_values[sample_id] = beta_tensor.detach().cpu().numpy().item()
            
            if epoch == 0 or epoch == num_epochs - 1:
                print(f"  [NN] epoch {epoch+1}/{num_epochs}, avg_loss = {epoch_loss/self.n_samples:.6f}", flush=True)
    
    def cal_viol(self) -> Tuple[float, float, float]:
        """计算约束违反量"""
        obj_primal = 0.0
        obj_dual = 0.0
        obj_opt = 0.0
        
        g = self.unit_id
        
        for sample_id in range(self.n_samples):
            x_val = self.x[sample_id]
            alpha = self.alpha_values[sample_id]
            beta = self.beta_values[sample_id]
            mu_val = self.mu[sample_id]
            lambda_val = self.lambda_vals[sample_id]
            
            # 代理约束违反
            surrogate_lhs = np.sum(alpha * x_val)
            surrogate_viol = max(0, surrogate_lhs - beta)
            obj_primal += surrogate_viol
            
            # 互补松弛
            obj_opt += abs(surrogate_lhs - beta) * mu_val
            
            # 对偶约束（简化）
            gencost_fixed = self.gencost[g, -1] / self.T_delta
            for t in range(self.T):
                dual_expr = gencost_fixed - lambda_val[t] + alpha[t] * mu_val
                obj_dual += abs(dual_expr)
        
        return obj_primal, obj_dual, obj_opt
    
    def iter(self, max_iter: int = 20, nn_epochs: int = 10):
        """
        主BCD迭代循环
        """
        print(f"开始BCD迭代训练 (机组{self.unit_id})...", flush=True)
        
        for i in range(max_iter):
            print(f"🔄 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i
            
            EPS = 1e-10
            
            # 1. 原始块迭代
            for sample_id in range(self.n_samples):
                alpha = self.alpha_values[sample_id]
                beta = self.beta_values[sample_id]
                
                pg_sol, x_sol, coc_sol, cpower_sol = self.iter_with_primal_block(
                    sample_id, alpha, beta
                )
                
                if pg_sol is not None:
                    self.pg[sample_id] = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                    self.x[sample_id] = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                    self.x[sample_id] = np.where(np.abs(self.x[sample_id] - 1) < EPS, 1, self.x[sample_id])
                    self.coc[sample_id] = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)
                    self.cpower[sample_id] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
            
            # 2. 对偶块迭代
            for sample_id in range(self.n_samples):
                alpha = self.alpha_values[sample_id]
                beta = self.beta_values[sample_id]
                
                mu_sol = self.iter_with_dual_block(sample_id, alpha, beta)
                self.mu[sample_id] = max(mu_sol, 0) if self.iter_number >= 50 else max(mu_sol, self.mu_lower_bound)
            
            # 3. 神经网络更新代理约束参数
            self.iter_with_surrogate_nn(num_epochs=nn_epochs)
            
            # 计算违反量
            obj_primal, obj_dual, obj_opt = self.cal_viol()
            obj_primal = obj_primal if abs(obj_primal) >= 1e-12 else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= 1e-12 else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= 1e-12 else 0.0
            
            print(f"  obj_primal: {obj_primal:.6f}, obj_dual: {obj_dual:.6f}, obj_opt: {obj_opt:.6f}", flush=True)
            
            # 更新惩罚参数
            self.rho_primal += self.gamma * obj_primal
            self.rho_dual += self.gamma * obj_dual
            self.rho_opt += self.gamma * obj_opt
            
            print(f"  ρ_primal={self.rho_primal:.4f}, ρ_dual={self.rho_dual:.4f}, ρ_opt={self.rho_opt:.4f}", flush=True)
            print("  " + "-" * 40, flush=True)
        
        print(f"✓ 机组{self.unit_id}代理约束训练完成", flush=True)
    
    def get_surrogate_params(self, pd_data: np.ndarray, lambda_val: np.ndarray) -> Tuple[np.ndarray, float]:
        """获取代理约束参数"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用")
        
        self.surrogate_net.eval()
        
        pd_flat = pd_data.flatten()
        features = np.concatenate([pd_flat, lambda_val])
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            alpha, beta = self.surrogate_net(features_tensor)
        
        return alpha.squeeze(0).cpu().numpy(), beta.squeeze(0).cpu().numpy().item()
    
    def save(self, filepath: str):
        """保存模型"""
        if TORCH_AVAILABLE:
            state = {
                'surrogate_net_state_dict': self.surrogate_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'alpha_values': self.alpha_values,
                'beta_values': self.beta_values,
                'mu': self.mu,
                'rho_primal': self.rho_primal,
                'rho_dual': self.rho_dual,
                'rho_opt': self.rho_opt
            }
            
            dirpath = os.path.dirname(os.path.abspath(filepath))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            
            torch.save(state, filepath)
            print(f"✓ 代理约束模型已保存: {filepath}", flush=True)
    
    def load(self, filepath: str):
        """加载模型"""
        if TORCH_AVAILABLE:
            state = torch.load(filepath, map_location=self.device)
            self.surrogate_net.load_state_dict(state['surrogate_net_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alpha_values = state['alpha_values']
            self.beta_values = state['beta_values']
            self.mu = state['mu']
            self.rho_primal = state['rho_primal']
            self.rho_dual = state['rho_dual']
            self.rho_opt = state['rho_opt']
            print(f"✓ 代理约束模型已加载: {filepath}", flush=True)


# ========================== 训练代码 ==========================

def train_dual_predictor_from_data(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                                    num_epochs: int = 100, batch_size: int = 8,
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
    predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_delta, device)
    
    # 训练
    predictor.train(num_epochs=num_epochs, batch_size=batch_size)
    
    # 保存模型
    if save_path:
        predictor.save(save_path)
    
    return predictor


def train_subproblem_surrogate_from_data(ppc, active_set_data: List[Dict], unit_id: int,
                                          T_delta: float = 1.0, lambda_predictor=None,
                                          max_iter: int = 20, nn_epochs: int = 10,
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
        lambda_predictor=lambda_predictor, device=device
    )
    
    # BCD迭代训练
    trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)
    
    # 保存模型
    if save_path:
        trainer.save(save_path)
    
    return trainer


def train_all_subproblem_surrogates(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                                     lambda_predictor=None, unit_ids: List[int] = None,
                                     max_iter: int = 20, nn_epochs: int = 10,
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
            lambda_predictor=lambda_predictor, device=device
        )
        
        trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)
        trainers[g] = trainer
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{g}.pth'))
    
    print(f"\n✓ 所有机组代理约束训练完成", flush=True)
    return trainers


def train_complete_model(ppc, active_set_data: List[Dict], T_delta: float = 1.0,
                          unit_ids: List[int] = None,
                          dual_epochs: int = 100, dual_batch_size: int = 8,
                          surrogate_max_iter: int = 20, surrogate_nn_epochs: int = 10,
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
        num_epochs=dual_epochs, batch_size=dual_batch_size,
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
        save_dir=save_dir, device=device
    )
    
    print("\n" + "=" * 60, flush=True)
    print("完整模型训练完成!", flush=True)
    print("=" * 60, flush=True)
    
    return dual_predictor, trainers


def load_trained_models(ppc, active_set_data: List[Dict], T_delta: float,
                        load_dir: str, unit_ids: List[int] = None, device=None):
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
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, g,
            lambda_predictor=dual_predictor, device=device
        )
        
        surrogate_path = os.path.join(load_dir, f'surrogate_unit_{g}.pth')
        if os.path.exists(surrogate_path):
            trainer.load(surrogate_path)
            trainers[g] = trainer
        else:
            print(f"警告: 未找到机组{g}代理约束模型 {surrogate_path}", flush=True)
    
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
        pd_data = active_set_data[sample_id]['pd_data']
        lambda_pred = dual_predictor.predict(pd_data)
        lambda_true = dual_predictor.lambda_true[sample_id]
        
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
            
            # 无代理约束
            x_without = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, None, None)
            gap_without = np.sum(x_without * (1 - x_without))
            total_gap_without += gap_without
            
            # 有代理约束
            x_with = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, alpha, beta)
            gap_with = np.sum(x_with * (1 - x_with))
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
        feasibility_rate = feasible_count / n_eval * 100
        
        print(f"\n  机组 {g}:", flush=True)
        print(f"    整数性间隙 (无代理): {avg_gap_without:.4f}", flush=True)
        print(f"    整数性间隙 (有代理): {avg_gap_with:.4f}", flush=True)
        print(f"    间隙减少: {gap_reduction:.2f}%", flush=True)
        print(f"    真实解可行率: {feasibility_rate:.1f}%", flush=True)


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
            # 随机选择开机时段
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
    
    T = 8
    if active_set_data is None:
        active_set_data = generate_test_data(ppc, T=T, n_samples=20)
    
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
        lambda_pred = predictor.predict(test_pd)
        lambda_true = predictor.lambda_true[sample_id]
        
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
        
        # 1. 无代理约束的LP松弛
        x_without = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, None, None)
        integrality_gap_without = np.sum(x_without * (1 - x_without))
        total_integrality_gap_without += integrality_gap_without
        
        # 2. 有代理约束的LP松弛
        x_with = solve_subproblem_LP_simple(trainer, sample_id, lambda_val, alpha, beta)
        integrality_gap_with = np.sum(x_with * (1 - x_with))
        total_integrality_gap_with += integrality_gap_with
        
        # 3. 代理约束违反量
        constraint_viol = max(0, np.sum(alpha * x_with) - beta)
        total_constraint_violation += constraint_viol
        
        # 4. 真实解的可行性（代理约束是否保留真实解）
        if x_target is not None:
            target_lhs = np.sum(alpha * x_target)
            if target_lhs <= beta + 1e-6:
                target_feasibility_rate += 1.0
        
        if sample_id < 3:
            print(f"\n  样本 {sample_id}:", flush=True)
            print(f"    无代理约束整数性间隙: {integrality_gap_without:.4f}", flush=True)
            print(f"    有代理约束整数性间隙: {integrality_gap_with:.4f}", flush=True)
            print(f"    代理约束违反量: {constraint_viol:.6f}", flush=True)
            if x_target is not None:
                print(f"    真实解可行: {target_lhs <= beta + 1e-6}", flush=True)
    
    avg_gap_without = total_integrality_gap_without / n_test
    avg_gap_with = total_integrality_gap_with / n_test
    avg_violation = total_constraint_violation / n_test
    feasibility_rate = target_feasibility_rate / n_test * 100
    
    print(f"\n  === 总体评估 ===", flush=True)
    print(f"  平均整数性间隙 (无代理约束): {avg_gap_without:.4f}", flush=True)
    print(f"  平均整数性间隙 (有代理约束): {avg_gap_with:.4f}", flush=True)
    print(f"  间隙减少: {(avg_gap_without - avg_gap_with) / max(avg_gap_without, 1e-6) * 100:.2f}%", flush=True)
    print(f"  平均代理约束违反量: {avg_violation:.6f}", flush=True)
    print(f"  真实解可行率: {feasibility_rate:.1f}%", flush=True)


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
    if active_set_data is None:
        active_set_data = generate_test_data(ppc, T=T, n_samples=15)
    
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
        alpha1, beta1 = trainer.get_surrogate_params(test_pd, trainer.lambda_vals[0])
        alpha2, beta2 = trainer2.get_surrogate_params(test_pd, trainer2.lambda_vals[0])
        diff_alpha = np.max(np.abs(alpha1 - alpha2))
        diff_beta = abs(beta1 - beta2)
        print(f"  代理约束加载验证: alpha差异 = {diff_alpha:.8f}, beta差异 = {diff_beta:.8f}", flush=True)
        assert diff_alpha < 1e-5 and diff_beta < 1e-5, "代理约束加载失败"
        
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
        T = 8
        T_delta = 1.0
        unit_ids = None  # None表示所有机组，或指定如 [0, 1, 2]
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
        
        ppc = pypower.case30.case30()
        active_set_data = generate_test_data(ppc, T=8, n_samples=20)
        
        predictor = train_dual_predictor_from_data(
            ppc, active_set_data, T_delta=1.0,
            num_epochs=100, batch_size=8,
            save_path=os.path.join(result_dir, 'dual_predictor.pth')
        )
        
    elif mode == 3:
        # 仅训练指定机组代理约束
        print("\n>>> 仅训练指定机组代理约束 <<<\n")
        
        ppc = pypower.case30.case30()
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
        ppc = pypower.case30.case30()
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
                 surrogate_nn_epochs: int = 10):
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
        save_dir=save_dir
    )
    
    # 评估模型
    evaluate_trained_models(dual_predictor, trainers, active_set_data)
    
    return dual_predictor, trainers


if __name__ == "__main__":
    main()
