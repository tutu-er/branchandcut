import numpy as np
import gurobipy as gp
from gurobipy import GRB

import io
import sys
import re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# TODO: 数据读取部分请根据你的数据格式自行实现
# 这里假设你已将 case39_UC、load3996.mat 等数据转为 numpy 数组或 csv 并加载为变量
# 例如：gen, branch, gencost, Pd, T_delta, T, nb, ng, nl, baseMVA
# 你需要自行实现数据加载部分

class UnitCommitmentModel:
    def __init__(self, gen, branch, gencost, Pd, T_delta):
        self.gen = gen
        self.branch = branch
        self.gencost = gencost
        self.Pd = Pd
        self.T_delta = T_delta
        self.T = Pd.shape[1]
        self.ng = gen.shape[0]
        self.nb = branch.shape[0]
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
                self.model.addConstr(self.pg[g, t] >= self.gen[g, 9] * self.x[g, t])
                self.model.addConstr(self.pg[g, t] <= self.gen[g, 8] * self.x[g, t])
        # 爬坡约束
        Ru = 0.4 * self.gen[:, 8] / self.T_delta
        Rd = 0.4 * self.gen[:, 8] / self.T_delta
        Ru_co = 0.3 * self.gen[:, 8]
        Rd_co = 0.3 * self.gen[:, 8]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.model.addConstr(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.coc[g, t-1] >= -start_cost[g] * (self.x[g, t] - self.x[g, t-1]))
                self.model.addConstr(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]))
                self.model.addConstr(self.coc[g, t-1] >= 0)
        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addConstr(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
        # 目标函数
        obj = gp.quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T)) \
            + gp.quicksum(self.coc[g, t] for g in range(self.ng) for t in range(self.T-1))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        self.model.Params.OutputFlag = 0  # 禁用输出

    def solve(self):
        self.cut_count_log = []  # 用于记录每次callback的割平面数

        def cut_callback(model, where):
            if where == GRB.Callback.MIP:
                try:
                    cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
                    self.cut_count_log.append(cutcnt)
                except Exception:
                    pass

        # 日志重定向到内存
        log_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_buffer
        self.model.Params.OutputFlag = 1  # 启用日志输出
        self.model.optimize(cut_callback)
        sys.stdout = old_stdout
        log_content = log_buffer.getvalue()

        # 求解结束后输出割平面统计
        if hasattr(self.model, 'CutCount'):
            print(f"Gurobi求解过程中总割平面数: {self.model.CutCount}")
        if self.cut_count_log:
            print(f"Callback记录的割平面数变化: {self.cut_count_log}")

        # 日志分析割类型统计
        cut_stats = re.findall(r'([A-Za-z]+):\s+(\d+)', log_content)
        if cut_stats:
            print('Gurobi割类型统计:')
            for cut_type, count in cut_stats:
                print(f'  {cut_type}: {count}')

        if self.model.status == GRB.OPTIMAL:
            pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[self.x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {self.model.objVal}")
            return pg_sol, x_sol, self.model.objVal
        else:
            print("未找到最优解")
            return None, None, None

# 示例调用：
# uc = UnitCommitmentModel(gen, branch, gencost, Pd, T_delta)
# pg_sol, x_sol, total_cost = uc.solve()
