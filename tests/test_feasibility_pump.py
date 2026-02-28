"""
端到端测试：可行性泵在 case30 数据上的验证

测试内容：
  1. 单元测试 - check_uc_feasibility 对 MILP 最优解应返回 True
  2. 集成测试 - recover_integer_solution 完整 pipeline：
     * 可行率（是否找到满足 UC 约束的整数解）
     * FP 成功率（是否在 max_iter 内收敛）
     * 目标函数差距（与 Gurobi MILP 对比）
     * 耗时统计

运行：
    python tests/test_feasibility_pump.py
"""

import sys
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# 项目根目录
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

import pypower.case30
from pypower.ext2int import ext2int
from pypower.idx_gen import PMIN, PMAX

from src.uc_NN_subproblem_v3 import (
    ActiveSetReader,
    SubproblemSurrogateTrainer,
    DualVariablePredictorTrainer,
)
from src.feasibility_pump import (
    round_to_integer,
    check_uc_feasibility,
    recover_integer_solution,
)

# ── 路径常量 ────────────────────────────────────────────────────────────────
DATA_JSON  = os.path.join(ROOT, 'result', 'active_sets_case30_20251223_002959.json')
MODEL_DIR  = os.path.join(ROOT, 'result', 'subproblem_models')
T_DELTA    = 1.0


# ============================================================
# 辅助函数
# ============================================================

def load_models(ppc, active_set_data):
    """加载预训练的 lambda 预测器和各机组代理约束训练器。"""
    print("加载预训练模型 ...")

    lambda_predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_DELTA)
    pred_path = os.path.join(MODEL_DIR, 'dual_predictor.pth')
    lambda_predictor.load(pred_path)
    print(f"  lambda 预测器: 已加载 ({pred_path})")

    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    trainers = {}
    for g in range(ng):
        path = os.path.join(MODEL_DIR, f'surrogate_unit_{g}.pth')
        if os.path.exists(path):
            trainer = SubproblemSurrogateTrainer(
                ppc, active_set_data, T_DELTA, unit_id=g,
                lambda_predictor=lambda_predictor
            )
            trainer.load(path)
            trainers[g] = trainer
            print(f"  机组 {g}: 已加载 ({path})")
        else:
            print(f"  机组 {g}: 无模型文件，跳过")

    return lambda_predictor, trainers


def solve_milp_basic(ppc, pd_data, T_delta=1.0, time_limit=120.0):
    """
    用 Gurobi 求解基本 UC MILP（无潮流约束），返回 (x_opt, obj_val)。
    失败时返回 (None, None)。
    """
    ppc_int = ext2int(ppc)
    gen     = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng  = gen.shape[0]
    T   = pd_data.shape[1]
    Pd_sum = np.sum(pd_data, axis=0)

    m = gp.Model('uc_milp')
    m.Params.OutputFlag = 0
    m.Params.MIPGap    = 1e-4
    m.Params.TimeLimit = time_limit

    pg     = m.addVars(ng, T, lb=0,            name='pg')
    x      = m.addVars(ng, T, vtype=GRB.BINARY, name='x')
    coc    = m.addVars(ng, T-1, lb=0,           name='coc')
    cpower = m.addVars(ng, T,   lb=0,           name='cpower')

    Ru    = 0.4 * gen[:, PMAX] / T_delta
    Rd    = 0.4 * gen[:, PMAX] / T_delta
    Ru_co = 0.3 * gen[:, PMAX]
    Rd_co = 0.3 * gen[:, PMAX]
    Ton   = min(int(4 * T_delta), T - 1)
    Toff  = min(int(4 * T_delta), T - 1)
    sc    = gencost[:, 1]   # 启动成本
    hc    = gencost[:, 2]   # 停机成本

    # 功率平衡 + 上下限
    for t in range(T):
        m.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]))
        for g in range(ng):
            m.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            m.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

    for g in range(ng):
        # 爬坡
        for t in range(1, T):
            m.addConstr(pg[g,t] - pg[g,t-1] <= Ru[g]*x[g,t-1] + Ru_co[g]*(1-x[g,t-1]))
            m.addConstr(pg[g,t-1] - pg[g,t] <= Rd[g]*x[g,t]   + Rd_co[g]*(1-x[g,t]))
        # 最小开关机时间
        for tau in range(1, Ton+1):
            for t1 in range(T - tau):
                m.addConstr(x[g,t1+1] - x[g,t1] <= x[g,t1+tau])
        for tau in range(1, Toff+1):
            for t1 in range(T - tau):
                m.addConstr(-x[g,t1+1] + x[g,t1] <= 1 - x[g,t1+tau])
        # 启停成本
        for t in range(1, T):
            m.addConstr(coc[g,t-1] >= sc[g]*(x[g,t] - x[g,t-1]))
            m.addConstr(coc[g,t-1] >= hc[g]*(x[g,t-1] - x[g,t]))
        # 发电成本
        for t in range(T):
            m.addConstr(cpower[g,t] >= gencost[g,-2]/T_delta*pg[g,t]
                                      + gencost[g,-1]/T_delta*x[g,t])

    obj = (gp.quicksum(cpower[g,t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g,t]   for g in range(ng) for t in range(T-1)))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        x_sol = np.round(np.array([[x[g,t].X for t in range(T)]
                                    for g in range(ng)])).astype(int)
        return x_sol, m.ObjVal
    return None, None


def compute_obj_fixed_x(x_int, ppc, pd_data, T_delta=1.0):
    """
    给定整数解 x，固定 x 后求 LP（发电调度），计算并返回总成本。
    若不可行（功率平衡无法满足），返回 inf。
    """
    ppc_int = ext2int(ppc)
    gen     = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng, T   = x_int.shape
    Pd_sum  = np.sum(pd_data, axis=0)

    m = gp.Model('obj_lp')
    m.Params.OutputFlag = 0

    pg     = m.addVars(ng, T,   lb=0, name='pg')
    coc    = m.addVars(ng, T-1, lb=0, name='coc')
    cpower = m.addVars(ng, T,   lb=0, name='cpower')

    Ru    = 0.4 * gen[:, PMAX] / T_delta
    Rd    = 0.4 * gen[:, PMAX] / T_delta
    Ru_co = 0.3 * gen[:, PMAX]
    Rd_co = 0.3 * gen[:, PMAX]
    sc    = gencost[:, 1]
    hc    = gencost[:, 2]

    for t in range(T):
        m.addConstr(gp.quicksum(pg[g,t] for g in range(ng)) == float(Pd_sum[t]))
    for g in range(ng):
        for t in range(T):
            m.addConstr(pg[g,t] >= gen[g, PMIN] * x_int[g,t])
            m.addConstr(pg[g,t] <= gen[g, PMAX] * x_int[g,t])
        for t in range(1, T):
            m.addConstr(pg[g,t] - pg[g,t-1] <= Ru[g]*x_int[g,t-1] + Ru_co[g]*(1-x_int[g,t-1]))
            m.addConstr(pg[g,t-1] - pg[g,t] <= Rd[g]*x_int[g,t]   + Rd_co[g]*(1-x_int[g,t]))
            m.addConstr(coc[g,t-1] >= sc[g]*(x_int[g,t] - x_int[g,t-1]))
            m.addConstr(coc[g,t-1] >= hc[g]*(x_int[g,t-1] - x_int[g,t]))
        for t in range(T):
            m.addConstr(cpower[g,t] >= gencost[g,-2]/T_delta*pg[g,t]
                                      + gencost[g,-1]/T_delta*x_int[g,t])

    obj = (gp.quicksum(cpower[g,t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g,t]   for g in range(ng) for t in range(T-1)))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    return m.ObjVal if m.status == GRB.OPTIMAL else float('inf')


# ============================================================
# 测试 1：check_uc_feasibility 对 MILP 解应返回 True
# ============================================================

def test_feasibility_check_on_milp(ppc, active_set_data, n_test=3):
    """
    对前 n_test 个样本，分别求解 MILP 并验证 check_uc_feasibility 返回 True。
    """
    print("\n" + "=" * 60)
    print("测试 1：check_uc_feasibility 对 MILP 解的正确性")
    print("=" * 60)

    n_pass = 0
    for sid in range(min(n_test, len(active_set_data))):
        pd_data = np.array(active_set_data[sid]['pd_data'])
        t0 = time.time()
        x_milp, obj_milp = solve_milp_basic(ppc, pd_data, T_DELTA)
        t_milp = time.time() - t0

        if x_milp is None:
            print(f"  样本 {sid}: MILP 未求得最优解，跳过")
            continue

        feas, reason = check_uc_feasibility(x_milp, ppc, pd_data, T_DELTA)
        ok = "PASS" if feas else "FAIL"
        n_pass += feas
        print(f"  样本 {sid}: MILP obj={obj_milp:10.2f}, 可行={feas:5}, [{ok}]  "
              f"(MILP {t_milp:.1f}s)")
        if not feas:
            print(f"    原因: {reason}")

    total = min(n_test, len(active_set_data))
    print(f"\n  结果: {n_pass}/{total} 通过")
    return n_pass == total


# ============================================================
# 测试 2：recover_integer_solution 端到端
# ============================================================

def test_recover_integer_solution(ppc, active_set_data, trainers, lambda_predictor,
                                   n_test=None, fp_iter=50):
    """
    对每个样本运行完整的 recover_integer_solution pipeline，
    并与 Gurobi MILP 目标函数值对比。
    """
    print("\n" + "=" * 60)
    print("测试 2：recover_integer_solution 端到端 pipeline")
    print("=" * 60)

    n_samples = len(active_set_data) if n_test is None else min(n_test, len(active_set_data))
    results = []

    for sid in range(n_samples):
        pd_data = np.array(active_set_data[sid]['pd_data'])
        print(f"\n--- 样本 {sid} (Pd_total={np.sum(pd_data[:,0]):.1f} MW @ t=0) ---")

        # MILP 基准
        t0 = time.time()
        x_milp, obj_milp = solve_milp_basic(ppc, pd_data, T_DELTA)
        t_milp = time.time() - t0
        print(f"  MILP: obj={obj_milp:.2f}, t={t_milp:.1f}s")

        # 可行性泵
        t0 = time.time()
        x_fp, fp_success = recover_integer_solution(
            pd_data, trainers, lambda_predictor, ppc, T_DELTA,
            n_perturbations=5,
            conf_threshold=0.15,
            max_fp_iter=fp_iter,
            verbose=True,
            rng=np.random.default_rng(42)
        )
        t_fp = time.time() - t0

        # 验证可行性
        feas_fp, reason_fp = check_uc_feasibility(x_fp, ppc, pd_data, T_DELTA)

        # 计算目标
        if feas_fp:
            obj_fp = compute_obj_fixed_x(x_fp, ppc, pd_data, T_DELTA)
        else:
            obj_fp = float('inf')

        gap = ((obj_fp - obj_milp) / max(abs(obj_milp), 1.0) * 100
               if obj_milp is not None and obj_fp < float('inf') else float('nan'))

        status = "OK" if feas_fp else "INFEAS"
        print(f"  FP:   obj={obj_fp:.2f}, feas={feas_fp}, gap={gap:+.2f}%, "
              f"fp_success={fp_success}, t={t_fp:.1f}s [{status}]")
        if not feas_fp:
            print(f"    原因: {reason_fp}")

        results.append({
            'sid': sid, 'success': fp_success, 'feas': feas_fp,
            'obj_milp': obj_milp, 'obj_fp': obj_fp, 'gap': gap,
            't_milp': t_milp, 't_fp': t_fp,
        })

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    n_feas    = sum(1 for r in results if r['feas'])
    n_success = sum(1 for r in results if r['success'])
    gaps      = [r['gap'] for r in results if r['feas'] and not np.isnan(r['gap'])]

    print(f"  样本数:       {n_samples}")
    print(f"  可行率:       {n_feas}/{n_samples}  ({100*n_feas/n_samples:.1f}%)")
    print(f"  FP 收敛率:    {n_success}/{n_samples}  ({100*n_success/n_samples:.1f}%)")
    if gaps:
        print(f"  Gap (vs MILP): 平均={np.mean(gaps):+.2f}%, 最大={np.max(gaps):+.2f}%, 最小={np.min(gaps):+.2f}%")
    print(f"  FP 平均耗时:  {np.mean([r['t_fp'] for r in results]):.1f}s")
    print(f"  MILP 平均耗时:{np.mean([r['t_milp'] for r in results]):.1f}s")

    return results


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    overall_t0 = time.time()
    print("=" * 60)
    print("Case30 可行性泵 端到端测试")
    print("=" * 60)

    # 加载数据
    print("\n>>> 加载数据 ...")
    ppc    = pypower.case30.case30()
    reader = ActiveSetReader(DATA_JSON)
    active_set_data = reader.load_all_samples()
    T = active_set_data[0]['pd_data'].shape[1]
    ng = ext2int(ppc)['gen'].shape[0]
    print(f"  案例: case30, ng={ng}, T={T}, n_samples={len(active_set_data)}, T_delta={T_DELTA}")

    # 加载模型
    lambda_predictor, trainers = load_models(ppc, active_set_data)
    print(f"  已加载 {len(trainers)} 个机组的代理约束模型\n")

    # 测试 1
    test1_pass = test_feasibility_check_on_milp(ppc, active_set_data, n_test=3)

    # 测试 2（全部 12 样本）
    results = test_recover_integer_solution(
        ppc, active_set_data, trainers, lambda_predictor,
        n_test=None, fp_iter=50
    )

    print(f"\n总耗时: {time.time() - overall_t0:.1f}s")
    print("=" * 60)

    # 最终判断
    n_feas = sum(1 for r in results if r['feas'])
    n_total = len(results)
    if test1_pass and n_feas == n_total:
        print("全部测试通过")
        sys.exit(0)
    else:
        print(f"部分测试未通过 (check={test1_pass}, feas={n_feas}/{n_total})")
        sys.exit(1)
