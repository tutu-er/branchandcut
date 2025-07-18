import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A

from pathlib import Path
import io
import sys
import re

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower

class UnitCommitmentModelSCIP:
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
        self.model = Model("UnitCommitment")
        self.model.setParam('display/verblevel', 4)
        self._build_model()

    def _build_model(self):
        # 变量
        self.pg = {}
        self.x = {}
        self.coc = {}
        self.cpower = {}
        for g in range(self.ng):
            for t in range(self.T):
                self.pg[g, t] = self.model.addVar(lb=0, name=f"pg_{g}_{t}")
                self.x[g, t] = self.model.addVar(vtype="B", name=f"x_{g}_{t}")
                self.cpower[g, t] = self.model.addVar(lb=0, name=f"cpower_{g}_{t}")
            for t in range(self.T-1):
                self.coc[g, t] = self.model.addVar(lb=0, name=f"coc_{g}_{t}")

        # 有功平衡和出力上下界
        for t in range(self.T):
            self.model.addCons(
                quicksum(self.pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]),
                name=f'power_balance_{t}'
            )
            for g in range(self.ng):
                self.model.addCons(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.model.addCons(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])

        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addCons(
                    self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1])
                )
                self.model.addCons(
                    self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t])
                )

        # 最小开机/关机时间约束
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    self.model.addCons(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t])
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    self.model.addCons(-self.x[g, t1+1] + self.x[g, t1] <= 1 - self.x[g, t1+t])

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addCons(self.coc[g, t-1] >= -start_cost[g] * (self.x[g, t] - self.x[g, t-1]))
                self.model.addCons(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]))
                self.model.addCons(self.coc[g, t-1] >= 0)

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addCons(
                    self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t]
                )

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
                for l in range(self.branch.shape[0]):
                    expr = quicksum(
                        (PTDF[l, :] @ G[:, g]) * self.pg[g, t] for g in range(self.ng)
                    )
                    const_term = PTDF[l, :] @ self.Pd[:, t]
                    self.model.addCons(expr - const_term <= branch_limit[l])
                    self.model.addCons(-expr + const_term <= branch_limit[l])
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')

        # 目标函数
        obj = quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T)) \
            + quicksum(self.coc[g, t] for g in range(self.ng) for t in range(self.T-1))
        self.model.setObjective(obj, "minimize")

    def solve(self):
        self.model.optimize()
        status = self.model.getStatus()
        if status == "optimal":
            pg_sol = np.zeros((self.ng, self.T))
            x_sol = np.zeros((self.ng, self.T))
            for g in range(self.ng):
                for t in range(self.T):
                    pg_sol[g, t] = self.model.getVal(self.pg[g, t])
                    x_sol[g, t] = self.model.getVal(self.x[g, t])
            print(f"总运行成本: {self.model.getObjVal()}")
            return pg_sol, x_sol, self.model.getObjVal()
        else:
            print("未找到最优解")
            return None, None, None

if __name__ == "__main__":
    load_df = pd.read_csv('src/load.csv', header=None)
    Pd = load_df.values  # shape: (nb, T)
    T = Pd.shape[1]
    T_delta = 4  # 如有需要可根据数据调整

    ppc = get_case39_pypower()

    # 创建模型对象
    uc = UnitCommitmentModelSCIP(ppc, Pd, T_delta)
    pg_sol, x_sol, total_cost = uc.solve()

    if pg_sol is not None:
        pass
        # print("机组出力方案：", pg_sol)
        # print("机组启停方案：", x_sol)
        # print("总成本：", total_cost)

        # ====== 结果绘图 ======
        # import matplotlib.pyplot as plt
        # import seaborn as sns
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