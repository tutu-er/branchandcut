import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pypower
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
from pathlib import Path
import io
import sys
import re

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# TODO: 数据读取部分请根据你的数据格式自行实现
# 这里假设你已将 case39_UC、load3996.mat 等数据转为 numpy 数组或 csv 并加载为变量
# 例如：gen, branch, gencost, Pd, T_delta, T, nb, ng, nl, baseMVA
# 你需要自行实现数据加载部分

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
        self.nl = self.branch.shape[0]
        self.model = gp.Model('UnitCommitment')
        self.model.Params.OutputFlag = 0
        self.pg = self.model.addVars(self.ng, self.T, lb=0, name='pg')
        self.x = self.model.addVars(self.ng, self.T, vtype=GRB.BINARY, name='x')
        self.coc = self.model.addVars(self.ng, self.T-1, lb=0, name='coc')
        self.cpower = self.model.addVars(self.ng, self.T, lb=0, name='cpower')
        self._build_model()

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
        Ton = int(4 * self.T_delta)  # 最小开机时间
        Toff = int(4 * self.T_delta) # 最小关机时间
        # 最小开机时间约束（与matlab一致）
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    self.model.addConstr(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t])
        # 最小关机时间约束（与matlab一致）
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
            # G: 机组-节点映射矩阵，需用户根据数据准备
            # 这里假设 G 为 (nb, ng) 的0-1矩阵，gen[:,0]为机组母线编号（1-based）
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            # 计算PTDF
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]  # 线路容量
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
        
        # self.model.setParam("Presolve", 2)
        self.model.setParam('MIPGap', 1e-10)

    def solve(self):
        # self.cut_count_log = []  # 用于记录每次callback的割平面数

        # def cut_callback(model, where):
        #     if where == GRB.Callback.MIP:
        #         try:
        #             cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
        #             self.cut_count_log.append(cutcnt)
        #         except Exception:
        #             pass

        # self.model.Params.OutputFlag = 1  # 启用日志输出
        self.model.optimize()

        # 求解结束后输出割平面统计
        # if hasattr(self.model, 'CutCount'):
        #     print(f"Gurobi求解过程中总割平面数: {self.model.CutCount}")
        # if self.cut_count_log:
        #     print(f"Callback记录的割平面数变化: {self.cut_count_log}")

        if self.model.status == GRB.OPTIMAL:
            pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[self.x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {self.model.objVal}")
            return pg_sol, x_sol, self.model.objVal
        else:
            print("未找到最优解")
            return None, None, None

    def solve_with_manual_cuts(self, add_cut_func=None):
        """
        求解模型，并在回调中手动添加割平面。
        add_cut_func: 用户自定义的割平面添加函数，签名为 add_cut_func(model, where)
        """
        self.cut_count_log = []

        def cut_callback(model, where):
            if where == GRB.Callback.MIP:
                # 记录割平面数
                try:
                    cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
                    self.cut_count_log.append(cutcnt)
                except Exception:
                    pass
                # 手动添加割平面
            elif where == GRB.Callback.MIPNODE:
                if add_cut_func is not None:
                    add_cut_func(model, where)

        self.model.Params.OutputFlag = 1
        self.model.Params.MIPGap = 1e-10
        self.model.optimize(cut_callback)

        # 求解结束后输出割平面统计
        if hasattr(self.model, 'CutCount'):
            print(f"Gurobi求解过程中总割平面数: {self.model.CutCount}")
        if self.cut_count_log:
            print(f"Callback记录的割平面数变化: {self.cut_count_log}")

        if self.model.status == GRB.OPTIMAL:
            pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[self.x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {self.model.objVal}")
            return pg_sol, x_sol, self.model.objVal
        else:
            print("未找到最优解")
            return None, None, None

    def solve_with_dual(self, type=0):
        model = gp.Model('UnitCommitment')
        model.Params.OutputFlag = 0
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, lb=0, name='x')        
        coc = model.addVars(self.ng, self.T-1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')

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

        # 原问题约束 - 所有约束都添加明确的名称
        for t in range(self.T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]), 
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
        nb = self.Pd.shape[0]
        G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            G[bus_idx, g] = 1
        # 计算PTDF
        PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        branch_limit = self.branch[:, RATE_A]  # 线路容量
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - self.Pd[:, t])
            for l in range(self.branch.shape[0]):
                model.addConstr(flow[l] <= branch_limit[l],
                name=f'flow_upper_{l}_{t}')
                model.addConstr(flow[l] >= -branch_limit[l],
                name=f'flow_lower_{l}_{t}')
        
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(x[g, t] <= 1, name=f'x_upper_{g}_{t}')
                model.addConstr(x[g, t] >= 0, name=f'x_lower_{g}_{t}')

        # pg变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # 基础项：发电成本系数
                dual_expr = self.gencost[g, -2] / self.T_delta * lambda_cpower[g, t]
                
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
                
                # 对偶约束：梯度 = 0
                model.addConstr(dual_expr == 0, name=f'dual_pg_{g}_{t}')

        # x变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                # 基础项：固定成本
                dual_expr = self.gencost[g, -1] / self.T_delta * lambda_cpower[g, t]

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
                
                model.addConstr(dual_expr == 0, name=f'dual_x_{g}_{t}')
    
        # coc变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T-1):
                dual_expr = 1  # 目标函数中的系数
                dual_expr -= lambda_start_cost[g, t]
                dual_expr -= lambda_shut_cost[g, t]
                dual_expr -= lambda_coc_nonneg[g, t]
                
                model.addConstr(dual_expr == 0, name=f'dual_coc_{g}_{t}')
            
        # cpower变量的对偶约束
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = 1  # 目标函数中的系数
                dual_expr -= lambda_cpower[g, t]
                
                model.addConstr(dual_expr == 0, name=f'dual_cpower_{g}_{t}')
        
        # 强对偶条件：原问题目标值 = 对偶问题目标值
        primal_obj = (gp.quicksum(cpower[g, t] for g in range(self.ng) for t in range(self.T)) +
                    gp.quicksum(coc[g, t] for g in range(self.ng) for t in range(self.T-1)))
        
        # 对偶目标函数计算
        dual_obj = 0

        # x上界约束对对偶目标的贡献
        for g in range(self.ng):
            for t in range(self.T):
                dual_obj += lambda_x_upper[g, t]

        # 功率平衡约束对对偶目标的贡献
        for t in range(self.T):
            dual_obj -= lambda_power_balance[t] * np.sum(Pd[:, t])
        
        # 爬坡约束对对偶目标的贡献
        for g in range(self.ng):
            for t in range(self.T-1):
                dual_obj += lambda_ramp_up[g, t] * (Ru_co[g])
                dual_obj += lambda_ramp_down[g, t] * (Rd_co[g])
        
        for g in range(self.ng):
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    dual_obj += lambda_min_off[g, t-1, t1]
        
        # DCPF约束对对偶目标的贡献
        for l in range(self.nl):
            ptdf_row = PTDF[l, :]
            for t in range(self.T):
                flow_constant = ptdf_row @ Pd[:, t]
                dual_obj += lambda_dcpf_upper[l, t] * (branch_limit[l] + flow_constant)
                dual_obj += lambda_dcpf_lower[l, t] * (branch_limit[l] - flow_constant)
        
        # model.addConstr(dual_obj >= -1e6)
        
        # 强对偶约束：原问题目标 = 对偶问题目标
        if type == 2:
            model.addConstr(primal_obj == -dual_obj, name='strong_duality')
            model.setObjective(primal_obj, GRB.MINIMIZE)

        if type == 0:
            model.addConstr(primal_obj == -dual_obj, name='strong_duality')
            model.setObjective(primal_obj, GRB.MINIMIZE)
        if type == 1:
            model.addConstr(primal_obj == -dual_obj, name='strong_duality')
            model.setObjective(dual_obj, GRB.MINIMIZE)
        
        model.setParam("Presolve", 2)
        
        model.Params.OutputFlag = 1  # 启用日志输出
        model.Params.MIPGap = 1e-10
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {model.objVal}")
            lambda_ = {
                'lambda_power_balance': lambda_power_balance,
                'lambda_pg_lower': lambda_pg_lower,
                'lambda_pg_upper': lambda_pg_upper,
                'lambda_ramp_up': lambda_ramp_up,
                'lambda_ramp_down': lambda_ramp_down,
                'lambda_min_on': lambda_min_on,
                'lambda_min_off': lambda_min_off,
                'lambda_start_cost': lambda_start_cost,
                'lambda_shut_cost': lambda_shut_cost,
                'lambda_coc_nonneg': lambda_coc_nonneg,
                'lambda_cpower': lambda_cpower,
                'lambda_dcpf_upper': lambda_dcpf_upper,
                'lambda_dcpf_lower': lambda_dcpf_lower
            }
            
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
                
                # DCPF约束对偶变量: shape (T, nl)
                'lambda_dcpf_upper': np.array([[lambda_dcpf_upper[l, t].X for l in range(self.nl)] for t in range(self.T)]),
                'lambda_dcpf_lower': np.array([[lambda_dcpf_lower[l, t].X for l in range(self.nl)] for t in range(self.T)]),
                
                # x变量上下界约束对偶变量: shape (ng, T) (原来缺失的)
                'lambda_x_upper': np.array([[lambda_x_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_x_lower': np.array([[lambda_x_lower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            }
                        
            # dual_comparison = self.extract_and_compare_dual_variables(model, lambda_)
            
            return pg_sol, x_sol, model.objVal, lambda_sol
        
    def extract_and_compare_dual_variables(self, model, explicit_duals):
        """
        通过约束名称提取对偶变量并进行比较
        
        Args:
            model: 求解后的Gurobi模型
            explicit_duals: 显式对偶变量字典
            
        Returns:
            Dict: 包含比较结果的字典
        """
        print("\n=== 基于约束名称的对偶变量比较 ===")
        
        implicit_duals = {}
        
        try:
            # 1. 功率平衡约束的对偶变量
            implicit_duals['power_balance'] = {}
            for t in range(self.T):
                constr = model.getConstrByName(f'power_balance_{t}')
                if constr is not None:
                    implicit_duals['power_balance'][t] = constr.Pi
            
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
                        implicit_duals['pg_lower'][g][t] = constr_lower.Pi
                    
                    # 上限约束
                    constr_upper = model.getConstrByName(f'pg_upper_{g}_{t}')
                    if constr_upper is not None:
                        implicit_duals['pg_upper'][g][t] = constr_upper.Pi
            
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
                        implicit_duals['ramp_up'][g][t-1] = constr_ramp_up.Pi
                    
                    # 下爬坡约束
                    constr_ramp_down = model.getConstrByName(f'ramp_down_{g}_{t}')
                    if constr_ramp_down is not None:
                        implicit_duals['ramp_down'][g][t-1] = constr_ramp_down.Pi

            
            # 4. 最小开机/关机时间约束的对偶变量（通过约束名直接提取）
            implicit_duals['min_on'] = {}
            implicit_duals['min_off'] = {}
            for g in range(self.ng):
                implicit_duals['min_on'][g] = {}
                implicit_duals['min_off'][g] = {}
                for tau in range(1, min(4, self.T)+1):
                    for t1 in range(self.T - tau):
                        # 最小开机时间约束
                        cname_on = f'min_on_{g}_{tau}_{t1}'
                        constr_on = model.getConstrByName(cname_on)
                        if constr_on is not None:
                            if tau not in implicit_duals['min_on'][g]:
                                implicit_duals['min_on'][g][tau] = {}
                            implicit_duals['min_on'][g][tau][t1] = constr_on.Pi
                        # 最小关机时间约束
                        cname_off = f'min_off_{g}_{tau}_{t1}'
                        constr_off = model.getConstrByName(cname_off)
                        if constr_off is not None:
                            if tau not in implicit_duals['min_off'][g]:
                                implicit_duals['min_off'][g][tau] = {}
                            implicit_duals['min_off'][g][tau][t1] = constr_off.Pi
            
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
                        implicit_duals['start_cost'][g][t-1] = constr_start.Pi
                    
                    # 关机成本约束
                    constr_shut = model.getConstrByName(f'shut_cost_{g}_{t}')
                    if constr_shut is not None:
                        implicit_duals['shut_cost'][g][t-1] = constr_shut.Pi
                    
                    # 非负约束
                    constr_nonneg = model.getConstrByName(f'coc_nonneg_{g}_{t}')
                    if constr_nonneg is not None:
                        implicit_duals['coc_nonneg'][g][t-1] = constr_nonneg.Pi
    
            
            # 6. DCPF约束的对偶变量
            implicit_duals['dcpf_upper'] = {}
            implicit_duals['dcpf_lower'] = {}
            for l in range(self.nl):
                implicit_duals['dcpf_upper'][l] = {}
                implicit_duals['dcpf_lower'][l] = {}
                for t in range(self.T):
                    # 上限约束
                    constr_dcpf_upper = model.getConstrByName(f'dcpf_upper_{l}_{t}')
                    if constr_dcpf_upper is not None:
                        implicit_duals['dcpf_upper'][l][t] = constr_dcpf_upper.Pi
                    
                    # 下限约束
                    constr_dcpf_lower = model.getConstrByName(f'dcpf_lower_{l}_{t}')
                    if constr_dcpf_lower is not None:
                        implicit_duals['dcpf_lower'][l][t] = constr_dcpf_lower.Pi
            
            print("✓ 成功通过约束名称提取所有隐式对偶变量")
            
            comparison_results = self.compare_dual_values(explicit_duals, implicit_duals)
        
            return {
                'implicit_duals': implicit_duals,
                'comparison_results': comparison_results,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 对偶变量比较过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'implicit_duals': {},
                'comparison_results': {},
                'success': False,
                'error': str(e)
            }

    def extract_dual_variables(self, model):
        """
        通过约束名称提取对偶变量并进行比较
        
        Args:
            model: 求解后的Gurobi模型
            explicit_duals: 显式对偶变量字典
            
        Returns:
            Dict: 包含比较结果的字典
        """
        print("\n=== 基于约束名称的对偶变量提取 ===")
        
        implicit_duals = {}
        
        try:
            # 1. 功率平衡约束的对偶变量
            print("1. 提取功率平衡约束的对偶变量...")
            implicit_duals['power_balance'] = {}
            for t in range(self.T):
                constr = model.getConstrByName(f'power_balance_{t}')
                if constr is not None:
                    implicit_duals['power_balance'][t] = constr.Pi
            
            # 2. 发电上下限约束的对偶变量
            print("2. 提取发电上下限约束的对偶变量...")
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
            print("3. 提取爬坡约束的对偶变量...")
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
            print("4. 提取最小开机/关机时间约束的对偶变量...")
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
            print("5. 提取启停成本约束的对偶变量...")
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
            print("6. 提取发电成本约束的对偶变量...")
            implicit_duals['cpower'] = {}
            
            for g in range(self.ng):
                implicit_duals['cpower'][g] = {}
                for t in range(self.T):
                    constr_cpower = model.getConstrByName(f'cpower_{g}_{t}')
                    if constr_cpower is not None:
                        implicit_duals['cpower'][g][t] = constr_cpower.Pi
            
            # 7. DCPF潮流约束的对偶变量（修正名称）
            print("7. 提取DCPF潮流约束的对偶变量...")
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
            print("8. 提取x变量上下界约束的对偶变量...")
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
            print("\n=== 对偶变量提取统计 ===")
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
            
            print(f"总计提取对偶变量: {total_extracted} 个")
            print("✓ 成功通过约束名称提取所有隐式对偶变量")

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
        print("\n=== 提取对偶变量并转换为数组格式 ===")
        
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
                    [implicit_duals_dict['pg_lower'].get(g, {}).get(t, 0) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_pg_upper'] = np.array([
                    [implicit_duals_dict['pg_upper'].get(g, {}).get(t, 0) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            
            # 3. 爬坡约束对偶变量: shape (ng, T-1)
            if 'ramp_up' in implicit_duals_dict and 'ramp_down' in implicit_duals_dict:
                lambda_sol_implicit['lambda_ramp_up'] = np.array([
                    [implicit_duals_dict['ramp_up'].get(g, {}).get(t, 0) for t in range(self.T-1)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_ramp_down'] = np.array([
                    [implicit_duals_dict['ramp_down'].get(g, {}).get(t, 0) for t in range(self.T-1)] 
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
                            lambda_sol_implicit['lambda_min_on'][g, tau, t] = val
            
            if 'min_off' in implicit_duals_dict:
                lambda_sol_implicit['lambda_min_off'] = np.zeros((self.ng, Toff, self.T))
                for g in range(self.ng):
                    for tau in range(Toff):
                        for t in range(self.T):
                            val = implicit_duals_dict['min_off'].get(g, {}).get(tau+1, {}).get(t, 0)
                            lambda_sol_implicit['lambda_min_off'][g, tau, t] = val
            
            # 5. 启停成本约束对偶变量: shape (ng, T-1)
            constraint_names = ['start_cost', 'shut_cost', 'coc_nonneg']
            lambda_names = ['lambda_start_cost', 'lambda_shut_cost', 'lambda_coc_nonneg']
            
            for constraint_name, lambda_name in zip(constraint_names, lambda_names):
                if constraint_name in implicit_duals_dict:
                    lambda_sol_implicit[lambda_name] = np.array([
                        [implicit_duals_dict[constraint_name].get(g, {}).get(t, 0) for t in range(self.T-1)] 
                        for g in range(self.ng)
                    ])
            
            # 6. 发电成本约束对偶变量: shape (ng, T)
            if 'cpower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_cpower'] = np.array([
                    [implicit_duals_dict['cpower'].get(g, {}).get(t, 0) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            
            # 7. DCPF约束对偶变量: shape (nl, T) -> 转置为 (T, nl)
            if 'dcpf_upper' in implicit_duals_dict and 'dcpf_lower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_dcpf_upper'] = np.array([
                    [implicit_duals_dict['dcpf_upper'].get(l, {}).get(t, 0) for l in range(self.nl)] 
                    for t in range(self.T)
                ])
                lambda_sol_implicit['lambda_dcpf_lower'] = np.array([
                    [implicit_duals_dict['dcpf_lower'].get(l, {}).get(t, 0) for l in range(self.nl)] 
                    for t in range(self.T)
                ])
            
            if 'x_upper' in implicit_duals_dict and 'x_lower' in implicit_duals_dict:
                lambda_sol_implicit['lambda_x_upper'] = np.array([
                    [implicit_duals_dict['x_upper'].get(g, {}).get(t, 0) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
                lambda_sol_implicit['lambda_x_lower'] = np.array([
                    [implicit_duals_dict['x_lower'].get(g, {}).get(t, 0) for t in range(self.T)] 
                    for g in range(self.ng)
                ])
            
            print("✓ 成功将对偶变量转换为数组格式")
            return lambda_sol_implicit
            
        except Exception as e:
            print(f"❌ 对偶变量数组转换失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
                
    def compare_dual_values(self, explicit_duals, implicit_duals):
        print("\n=== 对偶变量值比较 ===")

        comparison_results = {}
        # 比较功率平衡约束
        power_balance_diff = []
        if 'lambda_power_balance' in explicit_duals:
            for t in range(self.T):
                explicit_val = explicit_duals['lambda_power_balance'][t].X
                implicit_val = implicit_duals['power_balance'].get(t, 0)
                diff = abs(explicit_val - implicit_val)
                power_balance_diff.append(diff)
                
                if diff > 1e-6:
                    print(f"功率平衡约束 t={t}: 显式={explicit_val:.6f}, 隐式={implicit_val:.6f}, 差值={diff:.6f}")
        
        comparison_results['power_balance'] = {
            'max_diff': max(power_balance_diff) if power_balance_diff else 0,
            'avg_diff': np.mean(power_balance_diff) if power_balance_diff else 0,
            'count': len(power_balance_diff)
        }
        
        # 比较发电上下限约束
        pg_lower_diff = []
        pg_upper_diff = []
        
        if 'lambda_pg_lower' in explicit_duals and 'lambda_pg_upper' in explicit_duals:
            for g in range(self.ng):
                for t in range(self.T):
                    # 下限约束比较
                    explicit_lower = explicit_duals['lambda_pg_lower'][g, t].X
                    implicit_lower = implicit_duals['pg_lower'].get(g, {}).get(t, 0)
                    diff_lower = abs(explicit_lower - implicit_lower)
                    pg_lower_diff.append(diff_lower)
                    
                    # 上限约束比较
                    explicit_upper = explicit_duals['lambda_pg_upper'][g, t].X
                    implicit_upper = implicit_duals['pg_upper'].get(g, {}).get(t, 0)
                    diff_upper = abs(explicit_upper - implicit_upper)
                    pg_upper_diff.append(diff_upper)
                    
                    if diff_lower > 1e-6 or diff_upper > 1e-6:
                        print(f"发电约束 g={g}, t={t}: 下限隐式={implicit_lower:.6f},  下限显式={explicit_lower:.6f},上限隐式={implicit_upper:.6f}, 上限显式={explicit_upper:.6f}")

        comparison_results['pg_constraints'] = {
            'lower_max_diff': max(pg_lower_diff) if pg_lower_diff else 0,
            'upper_max_diff': max(pg_upper_diff) if pg_upper_diff else 0,
            'lower_avg_diff': np.mean(pg_lower_diff) if pg_lower_diff else 0,
            'upper_avg_diff': np.mean(pg_upper_diff) if pg_upper_diff else 0
        }
        
        # 比较爬坡约束
        ramp_up_diff = []
        ramp_down_diff = []
        
        if 'lambda_ramp_up' in explicit_duals and 'lambda_ramp_down' in explicit_duals:
            for g in range(self.ng):
                for t in range(self.T-1):
                    # 上爬坡约束比较
                    explicit_up = explicit_duals['lambda_ramp_up'][g, t].X
                    implicit_up = implicit_duals['ramp_up'].get(g, {}).get(t, 0)
                    diff_up = abs(explicit_up - implicit_up)
                    ramp_up_diff.append(diff_up)
                    
                    # 下爬坡约束比较
                    explicit_down = explicit_duals['lambda_ramp_down'][g, t].X
                    implicit_down = implicit_duals['ramp_down'].get(g, {}).get(t, 0)
                    diff_down = abs(explicit_down - implicit_down)
                    ramp_down_diff.append(diff_down)
                    
                    if diff_up > 1e-6 or diff_down > 1e-6:
                        print(f"爬坡约束 g={g}, t={t}: 上爬坡差值={diff_up:.6f}, 下爬坡差值={diff_down:.6f}")
        
        comparison_results['ramp_constraints'] = {
            'up_max_diff': max(ramp_up_diff) if ramp_up_diff else 0,
            'down_max_diff': max(ramp_down_diff) if ramp_down_diff else 0,
            'up_avg_diff': np.mean(ramp_up_diff) if ramp_up_diff else 0,
            'down_avg_diff': np.mean(ramp_down_diff) if ramp_down_diff else 0
        }
        
        # 统计总体比较结果
        all_diffs = power_balance_diff + pg_lower_diff + pg_upper_diff + ramp_up_diff + ramp_down_diff
        
        comparison_results['overall'] = {
            'total_comparisons': len(all_diffs),
            'max_difference': max(all_diffs) if all_diffs else 0,
            'avg_difference': np.mean(all_diffs) if all_diffs else 0,
            'significant_differences': sum(1 for d in all_diffs if d > 1e-6),
            'tolerance_1e6': sum(1 for d in all_diffs if d <= 1e-6),
            'tolerance_1e8': sum(1 for d in all_diffs if d <= 1e-8)
        }
        
        print(f"\n=== 总体比较结果 ===")
        print(f"总比较次数: {comparison_results['overall']['total_comparisons']}")
        print(f"最大差值: {comparison_results['overall']['max_difference']:.8f}")
        print(f"平均差值: {comparison_results['overall']['avg_difference']:.8f}")
        print(f"显著差异数量 (>1e-6): {comparison_results['overall']['significant_differences']}")
        print(f"高精度匹配数量 (≤1e-8): {comparison_results['overall']['tolerance_1e8']}")
        
        # 一致性分析
        if comparison_results['overall']['max_difference'] < 1e-6:
            print("✓ 显式对偶变量与隐式对偶变量高度一致")
        elif comparison_results['overall']['max_difference'] < 1e-3:
            print("⚠ 显式对偶变量与隐式对偶变量基本一致，但存在小幅差异")
        else:
            print("❌ 显式对偶变量与隐式对偶变量存在显著差异")
            
        return comparison_results

    def outer_compare_dual_values(self, explicit_duals, implicit_duals):
        print("\n=== 对偶变量值比较 ===")

        comparison_results = {}
        # 比较功率平衡约束
        power_balance_diff = []
        if 'lambda_power_balance' in explicit_duals:
            for t in range(self.T):
                explicit_val = explicit_duals['lambda_power_balance'][t]
                implicit_val = implicit_duals['lambda_power_balance'][t]
                diff = abs(explicit_val - implicit_val)
                power_balance_diff.append(diff)
                
                if diff > 1e-6:
                    print(f"功率平衡约束 t={t}: 显式={explicit_val:.6f}, 隐式={implicit_val:.6f}, 差值={diff:.6f}")
        
        comparison_results['power_balance'] = {
            'max_diff': max(power_balance_diff) if power_balance_diff else 0,
            'avg_diff': np.mean(power_balance_diff) if power_balance_diff else 0,
            'count': len(power_balance_diff)
        }
        
        # 比较发电上下限约束
        pg_lower_diff = []
        pg_upper_diff = []
        
        if 'lambda_pg_lower' in explicit_duals and 'lambda_pg_upper' in explicit_duals:
            for g in range(self.ng):
                for t in range(self.T):
                    # 下限约束比较
                    explicit_lower = explicit_duals['lambda_pg_lower'][g, t]
                    implicit_lower = implicit_duals['lambda_pg_lower'][g, t]
                    diff_lower = abs(explicit_lower - implicit_lower)
                    pg_lower_diff.append(diff_lower)
                    
                    # 上限约束比较
                    explicit_upper = explicit_duals['lambda_pg_upper'][g, t]
                    implicit_upper = implicit_duals['lambda_pg_upper'][g, t]
                    diff_upper = abs(explicit_upper - implicit_upper)
                    pg_upper_diff.append(diff_upper)
                    
                    if diff_lower > 1e-6 or diff_upper > 1e-6:
                        print(f"发电约束 g={g}, t={t}: 下限隐式={implicit_lower:.6f},  下限显式={explicit_lower:.6f},上限隐式={implicit_upper:.6f}, 上限显式={explicit_upper:.6f}")

        comparison_results['pg_constraints'] = {
            'lower_max_diff': max(pg_lower_diff) if pg_lower_diff else 0,
            'upper_max_diff': max(pg_upper_diff) if pg_upper_diff else 0,
            'lower_avg_diff': np.mean(pg_lower_diff) if pg_lower_diff else 0,
            'upper_avg_diff': np.mean(pg_upper_diff) if pg_upper_diff else 0
        }
        
        # 比较爬坡约束
        ramp_up_diff = []
        ramp_down_diff = []
        
        if 'lambda_ramp_up' in explicit_duals and 'lambda_ramp_down' in explicit_duals:
            for g in range(self.ng):
                for t in range(self.T-1):
                    # 上爬坡约束比较
                    explicit_up = explicit_duals['lambda_ramp_up'][g, t]
                    implicit_up = implicit_duals['lambda_ramp_up'][g, t]
                    diff_up = abs(explicit_up - implicit_up)
                    ramp_up_diff.append(diff_up)
                    
                    # 下爬坡约束比较
                    explicit_down = explicit_duals['lambda_ramp_down'][g, t]
                    implicit_down = implicit_duals['lambda_ramp_down'][g, t]
                    diff_down = abs(explicit_down - implicit_down)
                    ramp_down_diff.append(diff_down)
                    
                    if diff_up > 1e-6 or diff_down > 1e-6:
                        print(f"爬坡约束 g={g}, t={t}: 上爬坡差值={diff_up:.6f}, 下爬坡差值={diff_down:.6f}")
        
        comparison_results['ramp_constraints'] = {
            'up_max_diff': max(ramp_up_diff) if ramp_up_diff else 0,
            'down_max_diff': max(ramp_down_diff) if ramp_down_diff else 0,
            'up_avg_diff': np.mean(ramp_up_diff) if ramp_up_diff else 0,
            'down_avg_diff': np.mean(ramp_down_diff) if ramp_down_diff else 0
        }
        
        # 统计总体比较结果
        all_diffs = power_balance_diff + pg_lower_diff + pg_upper_diff + ramp_up_diff + ramp_down_diff
        
        comparison_results['overall'] = {
            'total_comparisons': len(all_diffs),
            'max_difference': max(all_diffs) if all_diffs else 0,
            'avg_difference': np.mean(all_diffs) if all_diffs else 0,
            'significant_differences': sum(1 for d in all_diffs if d > 1e-6),
            'tolerance_1e6': sum(1 for d in all_diffs if d <= 1e-6),
            'tolerance_1e8': sum(1 for d in all_diffs if d <= 1e-8)
        }
        
        print(f"\n=== 总体比较结果 ===")
        print(f"总比较次数: {comparison_results['overall']['total_comparisons']}")
        print(f"最大差值: {comparison_results['overall']['max_difference']:.8f}")
        print(f"平均差值: {comparison_results['overall']['avg_difference']:.8f}")
        print(f"显著差异数量 (>1e-6): {comparison_results['overall']['significant_differences']}")
        print(f"高精度匹配数量 (≤1e-8): {comparison_results['overall']['tolerance_1e8']}")
        
        # 一致性分析
        if comparison_results['overall']['max_difference'] < 1e-6:
            print("✓ 显式对偶变量与隐式对偶变量高度一致")
        elif comparison_results['overall']['max_difference'] < 1e-3:
            print("⚠ 显式对偶变量与隐式对偶变量基本一致，但存在小幅差异")
        else:
            print("❌ 显式对偶变量与隐式对偶变量存在显著差异")
            
        return comparison_results        

# 示例调用：
# uc = UnitCommitmentModel(gen, branch, gencost, Pd, T_delta)
# pg_sol, x_sol, total_cost = uc.solve()
if __name__ == "__main__":
    load_df = pd.read_csv('src/load.csv', header=None)
    Pd = load_df.values  # shape: (nb, T)
    T = Pd.shape[1]
    T_delta = 4  # 如有需要可根据数据调整

    ppc = get_case39_pypower()

    # 创建模型对象
    uc = UnitCommitmentModel(ppc, Pd, T_delta)
    pg_sol, x_sol, total_cost, lambda_sol = uc.solve_with_dual()
    _, _, _, lambda_sol_dual = uc.solve_with_dual(type=1)
    uc.outer_compare_dual_values(lambda_sol, lambda_sol_dual)

    if pg_sol is not None:
        pass
        # print("机组出力方案：", pg_sol)
        # print("机组启停方案：", x_sol)
        # print("总成本：", total_cost)

        # ====== 结果绘图 ======
        # # 机组出力折线图
        # plt.figure(figsize=(12, 6))
        # for g in range(pg_sol.shape[0]):
        #     if np.sum(x_sol[g, :]) > 0:
        #         plt.plot(range(1, pg_sol.shape[1]+1), pg_sol[g, :], label=f'机组{g+1}')
        # plt.xlabel('时段')
        # plt.ylabel('出力 (MW)')
        # plt.title('机组出力折线图')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # 启停状态热力图
        # plt.figure(figsize=(12, 4))
        # sns.heatmap(x_sol, cmap='Blues', cbar=False)
        # plt.xlabel('时段')
        # plt.ylabel('机组编号')
        # plt.title('机组启停状态热力图 (蓝色=运行, 白色=停机)')
        # plt.tight_layout()
        # plt.show()
    else:
        print("未找到可行解")


