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
        self.nb = self.branch.shape[0]
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
        
        self.model.setParam("Presolve", 2)


    def solve(self):
        self.cut_count_log = []  # 用于记录每次callback的割平面数

        def cut_callback(model, where):
            if where == GRB.Callback.MIP:
                try:
                    cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
                    self.cut_count_log.append(cutcnt)
                except Exception:
                    pass

        self.model.Params.OutputFlag = 1  # 启用日志输出
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
    pg_sol, x_sol, total_cost = uc.solve_with_manual_cuts()

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


