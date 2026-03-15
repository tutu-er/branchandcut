"""简单测试 Gurobi license 是否可用。"""

import gurobipy as gp
from gurobipy import GRB


def test_gurobi_license():
    """创建一个最小 LP 验证 license 正常工作。"""
    m = gp.Model("license_test")
    m.Params.OutputFlag = 0
    x = m.addVar(name="x")
    m.setObjective(x, GRB.MINIMIZE)
    m.addConstr(x >= 1)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    assert abs(x.X - 1.0) < 1e-6
    print(f"Gurobi license OK — {m.Params.LicenseExpiration}", flush=True)


if __name__ == "__main__":
    test_gurobi_license()
