import numpy as np
import cvxpy as cp
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

class EconomicDispatchCVXPY:
    def __init__(self, ppc, Pd, T_delta, x):
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
        self.x = x  # 二值变量由外部给定，shape=(ng, T)
        self.constraints = None
        self.prob = None
        self._build_model()

    def _build_model(self):
        ng, T = self.ng, self.T
        self.pg = cp.Variable((ng, T))
        self.cpower = cp.Variable((ng, T))
        self.constraints = []
        for t in range(T):
            self.constraints.append(cp.sum(self.pg[:, t]) == np.sum(self.Pd[:, t]))
            for g in range(ng):
                self.constraints.append(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.constraints.append(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, T):
            for g in range(ng):
                self.constraints.append(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.constraints.append(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        for t in range(T):
            for g in range(ng):
                self.constraints.append(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
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
        obj = cp.sum(self.cpower)
        self.prob = cp.Problem(cp.Minimize(obj), self.constraints)

    def solve(self):
        result = self.prob.solve(solver=cp.GUROBI, verbose=False)
        if self.prob.status == cp.OPTIMAL or self.prob.status == cp.OPTIMAL_INACCURATE:
            pg_sol = self.pg.value
            # print(f"总运行成本: {self.prob.value}")
            return pg_sol, self.prob.value
        else:
            print("未找到最优解")
            return None, None

# 示例调用：
# ed = EconomicDispatchCVXPY(gen, branch, gencost, Pd, T_delta, x)
# pg_sol, total_cost = ed.solve()
