import numpy as np
import cplex
from cplex.exceptions import CplexError
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

class UnitCommitmentModelCplex:
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
        self.model = cplex.Cplex()
        self.model.objective.set_sense(self.model.objective.sense.minimize)
        self._build_model()

    def _build_model(self):
        ng, T = self.ng, self.T
        # 变量名
        pg_names = [f"pg_{g}_{t}" for g in range(ng) for t in range(T)]
        x_names = [f"x_{g}_{t}" for g in range(ng) for t in range(T)]
        coc_names = [f"coc_{g}_{t}" for g in range(ng) for t in range(T-1)]
        cpower_names = [f"cpower_{g}_{t}" for g in range(ng) for t in range(T)]
        # 连续变量
        self.model.variables.add(names=pg_names, lb=[0]*ng*T)
        self.model.variables.add(names=coc_names, lb=[0]*ng*(T-1))
        self.model.variables.add(names=cpower_names, lb=[0]*ng*T)
        # 二进制变量
        self.model.variables.add(names=x_names, types=[self.model.variables.type.binary]*ng*T)
        # 有功平衡
        for t in range(T):
            ind = [f"pg_{g}_{t}" for g in range(ng)]
            val = [1.0]*ng
            self.model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind, val)],
                senses=["E"],
                rhs=[float(np.sum(self.Pd[:, t]))],
                names=[f"power_balance_{t}"])
            for g in range(ng):
                # pmin
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([f"pg_{g}_{t}", f"x_{g}_{t}"], [1.0, -self.gen[g, PMIN]])],
                    senses=["G"], rhs=[0])
                # pmax
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([f"pg_{g}_{t}", f"x_{g}_{t}"], [1.0, -self.gen[g, PMAX]])],
                    senses=["L"], rhs=[0])
        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, T):
            for g in range(ng):
                # up
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"pg_{g}_{t}", f"pg_{g}_{t-1}", f"x_{g}_{t-1}", f"x_{g}_{t-1}"
                    ], [1, -1, -Ru[g], Ru_co[g]])],
                    senses=["L"], rhs=[Ru_co[g]])
                # down
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"pg_{g}_{t-1}", f"pg_{g}_{t}", f"x_{g}_{t}", f"x_{g}_{t}"
                    ], [1, -1, -Rd[g], Rd_co[g]])],
                    senses=["L"], rhs=[Rd_co[g]])
        # 最小开机/关机时间
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(ng):
            for t in range(1, Ton+1):
                for t1 in range(T - t):
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair([
                            f"x_{g}_{t1+1}", f"x_{g}_{t1}", f"x_{g}_{t1+t}"
                        ], [1, -1, -1])],
                        senses=["L"], rhs=[0])
        for g in range(ng):
            for t in range(1, Toff+1):
                for t1 in range(T - t):
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair([
                            f"x_{g}_{t1}", f"x_{g}_{t1+1}", f"x_{g}_{t1+t}"
                        ], [1, -1, 1])],
                        senses=["L"], rhs=[1])
        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, T):
            for g in range(ng):
                # 启动
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"coc_{g}_{t-1}", f"x_{g}_{t}", f"x_{g}_{t-1}"
                    ], [1, start_cost[g], -start_cost[g]])],
                    senses=["G"], rhs=[0])
                # 关机
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"coc_{g}_{t-1}", f"x_{g}_{t-1}", f"x_{g}_{t}"
                    ], [1, shut_cost[g], -shut_cost[g]])],
                    senses=["G"], rhs=[0])
                # 非负
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([f"coc_{g}_{t-1}"], [1])],
                    senses=["G"], rhs=[0])
        # 发电成本
        for t in range(T):
            for g in range(ng):
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"cpower_{g}_{t}", f"pg_{g}_{t}", f"x_{g}_{t}"
                    ], [1, -self.gencost[g, -2]/self.T_delta, -self.gencost[g, -1]/self.T_delta])],
                    senses=["G"], rhs=[0])
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
                # 这里直接用变量名构造表达式
                for l in range(self.branch.shape[0]):
                    expr = []
                    coeff = []
                    for g in range(self.ng):
                        expr.append(f"pg_{g}_{t}")
                        coeff.append(PTDF[l, :] @ G[:, g])
                    expr += [f"const_load_{i}_{t}" for i in range(nb)]
                    coeff += list(-PTDF[l, :])
                    # 上下限
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(expr, coeff)],
                        senses=["L"], rhs=[branch_limit[l]])
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(expr, [-c for c in coeff])],
                        senses=["L"], rhs=[branch_limit[l]])
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')
        # 目标函数
        obj = [0]*(self.model.variables.get_num())
        for t in range(self.T):
            for g in range(self.ng):
                idx = self.model.variables.get_indices(f"cpower_{g}_{t}")
                obj[idx] += 1
        for t in range(self.T-1):
            for g in range(self.ng):
                idx = self.model.variables.get_indices(f"coc_{g}_{t}")
                obj[idx] += 1
        self.model.objective.set_linear(list(enumerate(obj)))

    def solve(self):
        try:
            self.model.parameters.mip.display.set(2)
            self.model.solve()
        except CplexError as e:
            print(e)
            return None, None, None
        if self.model.solution.get_status() == 101:
            pg_sol = np.zeros((self.ng, self.T))
            x_sol = np.zeros((self.ng, self.T))
            for g in range(self.ng):
                for t in range(self.T):
                    pg_sol[g, t] = self.model.solution.get_values(f"pg_{g}_{t}")
                    x_sol[g, t] = self.model.solution.get_values(f"x_{g}_{t}")
            print(f"总运行成本: {self.model.solution.get_objective_value()}")
            return pg_sol, x_sol, self.model.solution.get_objective_value()
        else:
            print("未找到最优解")
            return None, None, None

# 示例调用：
# uc = UnitCommitmentModelCplex(gen, branch, gencost, Pd, T_delta)
# pg_sol, x_sol, total_cost = uc.solve()
