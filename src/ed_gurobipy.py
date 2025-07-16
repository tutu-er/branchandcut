import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

class EconomicDispatchGurobi:
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
        self.model = gp.Model('EconomicDispatch')
        self.model.Params.OutputFlag = 0
        self.pg = self.model.addVars(self.ng, self.T, lb=0, name='pg')
        self.cpower = self.model.addVars(self.ng, self.T, lb=0, name='cpower')
        self._build_model()

    def _build_model(self):
        for t in range(self.T):
            self.model.addConstr(gp.quicksum(self.pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]), name=f'power_balance_{t}')
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.model.addConstr(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.model.addConstr(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addConstr(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
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
        obj = gp.quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T))
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.setParam("Presolve", 2)

    def solve(self):
        self.model.Params.OutputFlag = 1
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            pg_sol = np.array([[self.pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            print(f"总运行成本: {self.model.objVal}")
            return pg_sol, self.model.objVal
        else:
            print("未找到最优解")
            return None, None

# 示例调用：
# ed = EconomicDispatchGurobi(gen, branch, gencost, Pd, T_delta, x)
# pg_sol, total_cost = ed.solve()
