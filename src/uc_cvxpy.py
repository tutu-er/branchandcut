import numpy as np
import cvxpy as cp
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

class UnitCommitmentModelCVXPY:
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
        self.constraints = None
        self.prob = None
        self._build_model()

    def _build_model(self):
        ng, T = self.ng, self.T
        # 定义变量
        self.pg = cp.Variable((ng, T))
        self.x = cp.Variable((ng, T), boolean=True)
        self.coc = cp.Variable((ng, T-1))
        self.cpower = cp.Variable((ng, T))
        self.constraints = []
        # 有功平衡
        for t in range(T):
            self.constraints.append(cp.sum(self.pg[:, t]) == np.sum(self.Pd[:, t]))
            for g in range(ng):
                self.constraints.append(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.constraints.append(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])
        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, T):
            for g in range(ng):
                self.constraints.append(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.constraints.append(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        # 最小开机/关机时间
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(ng):
            for t in range(1, Ton+1):
                for t1 in range(T - t):
                    self.constraints.append(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t])
        for g in range(ng):
            for t in range(1, Toff+1):
                for t1 in range(T - t):
                    self.constraints.append(-self.x[g, t1+1] + self.x[g, t1] <= 1 - self.x[g, t1+t])
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, T):
            for g in range(ng):
                self.constraints.append(self.coc[g, t-1] >= -start_cost[g] * (self.x[g, t] - self.x[g, t-1]))
                self.constraints.append(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]))
                self.constraints.append(self.coc[g, t-1] >= 0)
        # 发电成本
        for t in range(T):
            for g in range(ng):
                self.constraints.append(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
        # 潮流约束
        try:
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            for t in range(T):
                flow = PTDF @ (G @ self.pg[:, t] - self.Pd[:, t])
                for l in range(self.branch.shape[0]):
                    self.constraints.append(flow[l] <= branch_limit[l])
                    self.constraints.append(flow[l] >= -branch_limit[l])
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')
        # 目标函数
        obj = cp.sum(self.cpower) + cp.sum(self.coc)
        self.prob = cp.Problem(cp.Minimize(obj), self.constraints)

    def solve(self):
        result = self.prob.solve(solver=cp.GUROBI, verbose=False)
        if self.prob.status == cp.OPTIMAL or self.prob.status == cp.OPTIMAL_INACCURATE:
            pg_sol = self.pg.value
            x_sol = self.x.value
            # print(f"总运行成本: {self.prob.value}")
            return pg_sol, x_sol, self.prob.value
        else:
            print("未找到最优解")
            return None, None, None

# 示例调用：
# uc = UnitCommitmentModelCVXPY(gen, branch, gencost, Pd, T_delta)
# pg_sol, x_sol, total_cost = uc.solve()
