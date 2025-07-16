import numpy as np
import cplex
from cplex.exceptions import CplexError
import pandas as pd
import pypower
import pypower.case9
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, VG, MBASE, GEN_STATUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX

from src.case39_pypower import get_case39_pypower

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
        constraints = 0
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
                    lin_expr=[cplex.SparsePair([f"pg_{g}_{t}", f"x_{g}_{t}"], [1.0, float(-self.gen[g, PMIN])])],
                    senses=["G"], rhs=[0])
                # pmax
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([f"pg_{g}_{t}", f"x_{g}_{t}"], [1.0, float(-self.gen[g, PMAX])])],
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
                        f"pg_{g}_{t}", f"pg_{g}_{t-1}", f"x_{g}_{t-1}"
                    ], [1, -1, Ru_co[g]-Ru[g]])],
                    senses=["L"], rhs=[Ru_co[g]])
                # down
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([
                        f"pg_{g}_{t-1}", f"pg_{g}_{t}", f"x_{g}_{t}"
                    ], [1, -1, Rd_co[g]-Rd[g]])],
                    senses=["L"], rhs=[Rd_co[g]])
        # 最小开机/关机时间
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(ng):
            for t in range(2, Ton+1):
                for t1 in range(T - t):
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair([
                            f"x_{g}_{t1+1}", f"x_{g}_{t1}", f"x_{g}_{t1+t}"
                        ], [1, -1, -1])],
                        senses=["L"], rhs=[0])
        for g in range(ng):
            for t in range(2, Toff+1):
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
                    const_term = 0.0
                    for g in range(self.ng):
                        expr.append(f"pg_{g}_{t}")
                        coeff.append(PTDF[l, :] @ G[:, g])
                    # 负荷部分直接作为常数项
                    const_term = PTDF[l, :] @ self.Pd[:, t]
                    # 上限
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(expr, coeff)],
                        senses=["L"],
                        rhs=[branch_limit[l] + const_term]
                    )
                    # 下限
                    self.model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(expr, [-c for c in coeff])],
                        senses=["L"],
                        rhs=[branch_limit[l] - const_term]
                    )
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

    def solve_with_manual_cuts(self, add_cut_func=None):
        try:
            self.model.parameters.mip.display.set(2)
            # CPLEX Python API 没有像 Gurobi 那样的 callback 机制直接支持用户自定义割平面，
            # 但可以通过分支回调（branch callback）或循环求解+动态加约束的方式实现。
            # 这里采用“循环求解+动态加cut”的通用方式（伪代码，需用户实现add_cut_func逻辑）：

            max_iter = 20  # 最多迭代次数，防止死循环
            for iter_count in range(max_iter):
                self.model.solve()
                status = self.model.solution.get_status()
                if status != 101:
                    print("未找到最优解")
                    return None, None, None
                # 用户自定义割平面检测与添加
                if add_cut_func is not None:
                    cuts_added = add_cut_func(self.model)
                    print(f"第{iter_count+1}轮：已添加割平面 {cuts_added} 条")
                    if cuts_added == 0:
                        break  # 没有新cut，终止
                else:
                    break  # 没有cut函数，直接退出

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
        except CplexError as e:
            print(e)
            return None, None, None

if __name__ == "__main__":
    load_df = pd.read_csv('src/load.csv', header=None)
    Pd_base = load_df.values
    Pd_base = np.sum(Pd_base[:, ::4], axis=0)   # shape: (nb, T)
    
    ppc = pypower.case9.case9()
    
    Pd = ppc['bus'][:, pypower.idx_bus.PD]  # 假设Pd为bus数据中的Pd列
    Pd = Pd[:, None] * Pd_base[None, :] / np.max(Pd_base)
    
    T = Pd.shape[1]
    T_delta = 1 # 如有需要可根据数据调整

    # 创建模型对象
    uc = UnitCommitmentModelCplex(ppc, Pd, T_delta)
    pg_sol, x_sol, total_cost = uc.solve()

    if pg_sol is not None:
        pass
    else:
        print("未找到可行解")