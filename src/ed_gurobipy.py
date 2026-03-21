import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

class EconomicDispatchGurobi:
    def __init__(self, ppc, Pd, T_delta, x, renewable_data=None, verbose=False):
        self.ppc = ppc
        self.ppc_raw = ppc
        self.verbose = verbose
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.load_data = np.asarray(Pd, dtype=float)
        self.renewable_data = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
        if self.renewable_data is not None and self.renewable_data.shape != self.load_data.shape:
            raise ValueError(
                f"renewable_data shape {self.renewable_data.shape} does not match load shape {self.load_data.shape}"
            )
        self.Pd = self.load_data
        self.net_load = self.load_data if self.renewable_data is None else self.load_data - self.renewable_data
        self.T_delta = T_delta
        self.T = self.load_data.shape[1]
        self.ng = self.gen.shape[0]
        self.nb = self.load_data.shape[0]
        if self.renewable_data is not None:
            self.renewable_bus_ids = np.where(np.any(self.renewable_data > 1e-9, axis=1))[0]
        else:
            self.renewable_bus_ids = np.array([], dtype=int)
        self.nr = len(self.renewable_bus_ids)
        self.x = x  # 二值变量由外部给定，shape=(ng, T)
        self.model = gp.Model('EconomicDispatch')
        self.model.Params.OutputFlag = 1 if self.verbose else 0
        self.model.Params.LogToConsole = 1 if self.verbose else 0
        self.pg = self.model.addVars(self.ng, self.T, lb=0, name='pg')
        self.p_ren = self.model.addVars(self.nr, self.T, lb=0, name='p_ren') if self.nr > 0 else None
        self.cpower = self.model.addVars(self.ng, self.T, lb=0, name='cpower')
        self._build_model()

    def _get_ramp_limits(self):
        default_up = 0.4 * self.gen[:, PMAX] / self.T_delta
        default_down = 0.4 * self.gen[:, PMAX] / self.T_delta
        default_up_co = 0.3 * self.gen[:, PMAX]
        default_down_co = 0.3 * self.gen[:, PMAX]

        ramp_up_h = self._get_custom_generator_array('uc_ramp_up_mw_per_h')
        ramp_down_h = self._get_custom_generator_array('uc_ramp_down_mw_per_h')
        if ramp_up_h is None or ramp_down_h is None:
            return default_up, default_down, default_up_co, default_down_co

        Ru = np.asarray(ramp_up_h, dtype=float) * self.T_delta
        Rd = np.asarray(ramp_down_h, dtype=float) * self.T_delta
        Ru = np.maximum(Ru, default_up)
        Rd = np.maximum(Rd, default_down)
        Ru_co = np.maximum(Ru, self.gen[:, PMIN])
        Rd_co = np.maximum(Rd, self.gen[:, PMIN])
        return Ru, Rd, Ru_co, Rd_co

    def _get_custom_generator_array(self, key):
        values = self.ppc_raw.get(key)
        if values is None:
            return None
        values = np.asarray(values)
        if values.shape[0] != self.ng:
            return None
        raw_gen = np.asarray(self.ppc_raw.get('gen'))
        if raw_gen.shape[0] != self.ng:
            return values
        order = np.argsort(raw_gen[:, GEN_BUS], kind='stable')
        return values[order]

    def _build_model(self):
        for t in range(self.T):
            renewable_supply = (
                gp.quicksum(self.p_ren[r, t] for r in range(self.nr))
                if self.nr > 0 else 0
            )
            self.model.addConstr(
                gp.quicksum(self.pg[g, t] for g in range(self.ng)) + renewable_supply
                == np.sum(self.load_data[:, t]),
                name=f'power_balance_{t}'
            )
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.model.addConstr(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])
            for r, bus_idx in enumerate(self.renewable_bus_ids):
                self.model.addConstr(self.p_ren[r, t] <= self.renewable_data[bus_idx, t], name=f'ren_upper_{r}_{t}')
        Ru, Rd, Ru_co, Rd_co = self._get_ramp_limits()
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addConstr(self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1]))
                self.model.addConstr(self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t]))
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addConstr(self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t])
        try:
            G = np.zeros((self.nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            R = np.zeros((self.nb, self.nr))
            for r, bus_idx in enumerate(self.renewable_bus_ids):
                R[bus_idx, r] = 1
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            for t in range(self.T):
                thermal_injection = G @ np.array([self.pg[g, t] for g in range(self.ng)])
                renewable_injection = (
                    R @ np.array([self.p_ren[r, t] for r in range(self.nr)])
                    if self.nr > 0 else 0
                )
                flow = PTDF @ (thermal_injection + renewable_injection - self.load_data[:, t])
                for l in range(self.branch.shape[0]):
                    self.model.addConstr(flow[l] <= branch_limit[l], name=f'flow_upper_{l}_{t}')
                    self.model.addConstr(flow[l] >= -branch_limit[l], name=f'flow_lower_{l}_{t}')
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')
        obj = gp.quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T))
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.setParam("Presolve", 2)

    def solve(self):
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
