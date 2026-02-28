"""
可行性泵（Feasibility Pump）实现
用于从LP松弛解恢复UC问题的整数可行解

Pipeline：
  1. 求解全局UC LP松弛（加入代理约束）→ x_LP
  2. 通过各机组子问题LP（含代理约束 + 参数扰动）收集多组整数解
  3. 以"LP整数性强 + 多来源一致"识别高可信度变量并固定
  4. 可行性泵：LP投影（最小化 L1 距离）+ 四舍五入，迭代至整数可行
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Optional, Tuple

from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX

from src.uc_NN_subproblem_v3 import SubproblemSurrogateTrainer


# ========================== Step 1：整数恢复启发式 ==========================

def round_to_integer(x_LP: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将LP松弛解四舍五入为整数解。

    Args:
        x_LP: LP松弛解，shape (ng, T) 或 (T,)，值域 [0, 1]
        threshold: 四舍五入阈值（默认0.5）

    Returns:
        整数解，shape 与输入相同，值为 0 或 1
    """
    return (x_LP >= threshold).astype(int)


# ========================== 内部辅助：代理约束添加 ==========================

def _add_surrogate_constraints(
    model: gp.Model,
    x_vars: dict,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    T: int,
    prefix: str = ''
) -> None:
    """
    向 Gurobi 模型添加 V3 三时段代理约束。

    约束形式：alphas[k]*x[t_k] + betas[k]*x[t_k+1] + gammas[k]*x[t_k+2] <= deltas[k]
    时段映射：t_k = k % (T-2)（循环分配，确保 t_k+2 不越界）

    Args:
        model: Gurobi 模型
        x_vars: {t: Gurobi Var} 时段到变量的映射
        alphas: (max_constraints,) 第一时段系数
        betas: (max_constraints,) 第二时段系数
        gammas: (max_constraints,) 第三时段系数
        deltas: (max_constraints,) 右端项
        T: 时段总数
        prefix: 约束命名前缀（用于区分不同机组）
    """
    T_triples = max(1, T - 2)
    for k in range(len(alphas)):
        t_k  = k % T_triples
        t_k1 = min(t_k + 1, T - 1)
        t_k2 = min(t_k + 2, T - 1)
        a, b, c, r = float(alphas[k]), float(betas[k]), float(gammas[k]), float(deltas[k])
        if abs(a) > 1e-10 or abs(b) > 1e-10 or abs(c) > 1e-10:
            model.addConstr(
                a * x_vars[t_k] + b * x_vars[t_k1] + c * x_vars[t_k2] <= r,
                name=f'{prefix}surr_{k}'
            )


# ========================== 内部辅助：单机组 LP（多代理约束） ==========================

def _solve_unit_LP_with_surrogate(
    trainer: SubproblemSurrogateTrainer,
    lambda_val: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray
) -> np.ndarray:
    """
    求解单机组子问题 LP（使用 V3 三时段代理约束）。

    Args:
        trainer: 该机组的 SubproblemSurrogateTrainer
        lambda_val: (T,) 功率平衡对偶变量
        alphas, betas, gammas, deltas: V3 代理约束参数，各 shape (max_constraints,)

    Returns:
        x_LP: (T,) LP 松弛解；若不可行返回零向量
    """
    g = trainer.unit_id
    T = trainer.T

    model = gp.Model('unit_lp_surrogate')
    model.Params.OutputFlag = 0

    pg = model.addVars(T, lb=0, name='pg')
    x = model.addVars(T, lb=0, ub=1, name='x')
    cpower = model.addVars(T, lb=0, name='cpower')

    # 发电上下限
    for t in range(T):
        model.addConstr(pg[t] >= trainer.gen[g, PMIN] * x[t])
        model.addConstr(pg[t] <= trainer.gen[g, PMAX] * x[t])

    # 爬坡约束（与 UnitCommitmentModel 一致）
    Ru = 0.4 * trainer.gen[g, PMAX] / trainer.T_delta
    Rd = 0.4 * trainer.gen[g, PMAX] / trainer.T_delta
    Ru_co = 0.3 * trainer.gen[g, PMAX]
    Rd_co = 0.3 * trainer.gen[g, PMAX]
    for t in range(1, T):
        model.addConstr(pg[t] - pg[t-1] <= Ru * x[t-1] + Ru_co * (1 - x[t-1]))
        model.addConstr(pg[t-1] - pg[t] <= Rd * x[t] + Rd_co * (1 - x[t]))

    # 最小开关机时间
    Ton = min(int(4 * trainer.T_delta), T - 1)
    Toff = min(int(4 * trainer.T_delta), T - 1)
    for tau in range(1, Ton + 1):
        for t1 in range(T - tau):
            model.addConstr(x[t1+1] - x[t1] <= x[t1+tau])
    for tau in range(1, Toff + 1):
        for t1 in range(T - tau):
            model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau])

    # 发电成本（线性化）
    for t in range(T):
        model.addConstr(
            cpower[t] >= trainer.gencost[g, -2] / trainer.T_delta * pg[t]
                       + trainer.gencost[g, -1] / trainer.T_delta * x[t]
        )

    # V3 三时段代理约束
    x_dict = {t: x[t] for t in range(T)}
    _add_surrogate_constraints(model, x_dict, alphas, betas, gammas, deltas, T)

    # 目标：最小化成本 - 拉格朗日对偶项
    obj = gp.quicksum(cpower[t] for t in range(T))
    obj -= gp.quicksum(float(lambda_val[t]) * pg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return np.array([x[t].X for t in range(T)])
    return np.zeros(T)


# ========================== Step 2：收集多组整数解 ==========================

def collect_integer_solutions(
    pd_data: np.ndarray,
    lambda_val: np.ndarray,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    n_perturbations: int = 5,
    perturb_std: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    收集多组整数解（来自子问题 LP + 代理参数扰动）。

    Args:
        pd_data: (nb_load, T) 负荷数据
        lambda_val: (T,) 功率平衡对偶变量
        trainers: {unit_id: SubproblemSurrogateTrainer}
        n_perturbations: 扰动次数
        perturb_std: 扰动标准差（相对值）
        rng: 随机数生成器

    Returns:
        x_init_k:   (ng, T) 各机组子问题 LP 整数解
        x_init_k_m: (ng, n_perturbations, T) 扰动参数后的多组整数解
    """
    if rng is None:
        rng = np.random.default_rng()

    unit_ids = sorted(trainers.keys())
    ng = max(unit_ids) + 1
    T = pd_data.shape[1]

    x_init_k = np.zeros((ng, T), dtype=int)
    x_init_k_m = np.zeros((ng, n_perturbations, T), dtype=int)

    for g in unit_ids:
        trainer = trainers[g]
        alphas, betas, gammas, deltas = trainer.get_surrogate_params(pd_data, lambda_val)

        # 原始子问题 LP
        x_LP_k = _solve_unit_LP_with_surrogate(trainer, lambda_val, alphas, betas, gammas, deltas)
        x_init_k[g] = round_to_integer(x_LP_k)

        # 扰动代理参数，生成多组解
        for m in range(n_perturbations):
            noise_a = 1.0 + perturb_std * rng.standard_normal(len(alphas))
            noise_b = 1.0 + perturb_std * rng.standard_normal(len(betas))
            noise_c = 1.0 + perturb_std * rng.standard_normal(len(gammas))
            noise_r = 1.0 + perturb_std * rng.standard_normal(len(deltas))
            x_LP_m = _solve_unit_LP_with_surrogate(
                trainer, lambda_val,
                alphas * noise_a, betas * noise_b, gammas * noise_c, deltas * noise_r
            )
            x_init_k_m[g, m] = round_to_integer(x_LP_m)

    return x_init_k, x_init_k_m


# ========================== Step 3：识别高可信度变量 ==========================

def identify_trusted_mask(
    x_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    conf_threshold: float = 0.15
) -> np.ndarray:
    """
    识别高可信度（整数性强且多来源一致）的变量。

    条件1：LP 值远离 0.5（x_LP < conf_threshold 或 x_LP > 1 - conf_threshold）
    条件2：多数投票结果与 round(x_LP) 一致

    Args:
        x_LP:       (ng, T) 全局 LP 松弛解
        x_init_k:   (ng, T) 子问题 LP 整数解
        x_init_k_m: (ng, n_perturbations, T) 扰动整数解
        conf_threshold: LP 整数性置信阈值

    Returns:
        trusted_mask: (ng, T) bool 数组，True 表示高可信度（固定）
    """
    # 条件1：整数性强
    near_zero = x_LP < conf_threshold
    near_one = x_LP > 1.0 - conf_threshold
    integrality_confident = near_zero | near_one

    # 条件2：多数投票一致性
    n_pert = x_init_k_m.shape[1]
    x_ref = np.round(x_LP).astype(int)

    # 汇总所有投票（x_init_k + 所有扰动解）
    vote_sum = x_init_k.astype(float) + np.sum(x_init_k_m.astype(float), axis=1)
    n_votes = 1 + n_pert
    majority = (vote_sum / n_votes >= 0.5).astype(int)
    consistent = (majority == x_ref)

    return integrality_confident & consistent


# ========================== Step 6 辅助：全局 LP 松弛 ==========================

def solve_global_LP_relaxation(
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_val: np.ndarray
) -> np.ndarray:
    """
    构建完整 UC LP 松弛（x ∈ [0,1]），加入各机组代理约束，求解得 x_LP。

    Args:
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔（小时）
        trainers: {unit_id: trainer}
        lambda_val: (T,) 功率平衡对偶变量（用于查询代理约束参数）

    Returns:
        x_LP: (ng, T) LP 松弛解；若不可行返回零矩阵
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng = gen.shape[0]
    T = pd_data.shape[1]
    Pd_sum = np.sum(pd_data, axis=0)  # (T,)

    model = gp.Model('global_LP_relaxation')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    x = model.addVars(ng, T, lb=0, ub=1, name='x')
    cpower = model.addVars(ng, T, lb=0, name='cpower')

    # 功率平衡
    for t in range(T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
            name=f'pb_{t}'
        )

    # 爬坡参数（与 UnitCommitmentModel 一致）
    Ru = 0.4 * gen[:, PMAX] / T_delta
    Rd = 0.4 * gen[:, PMAX] / T_delta
    Ru_co = 0.3 * gen[:, PMAX]
    Rd_co = 0.3 * gen[:, PMAX]
    Ton = min(int(4 * T_delta), T - 1)
    Toff = min(int(4 * T_delta), T - 1)

    for g in range(ng):
        # 发电上下限
        for t in range(T):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

        # 爬坡约束
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1])
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
            )

        # 最小开关机时间
        for tau in range(1, Ton + 1):
            for t1 in range(T - tau):
                model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
        for tau in range(1, Toff + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

        # 发电成本
        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                               + gencost[g, -1] / T_delta * x[g, t]
            )

        # V3 三时段代理约束（仅对已训练机组）
        if g in trainers:
            alphas, betas, gammas, deltas = trainers[g].get_surrogate_params(pd_data, lambda_val)
            x_g = {t: x[g, t] for t in range(T)}
            _add_surrogate_constraints(model, x_g, alphas, betas, gammas, deltas, T, prefix=f'g{g}_')

    # 目标：最小化总发电成本
    obj = gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return np.array([[x[g, t].X for t in range(T)] for g in range(ng)])

    print(f"  警告: 全局 LP 松弛求解失败 (status={model.status})，返回零矩阵", flush=True)
    return np.zeros((ng, T))


# ========================== Step 5：可行性验证 ==========================

def check_uc_feasibility(
    x_int: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    tol: float = 1e-6
) -> Tuple[bool, str]:
    """
    验证给定整数解是否满足 UC 约束。

    检查顺序：
    1. 最小开关机时间约束（代数检查）
    2. 功率平衡 + 爬坡约束（固定 x，求解 LP 验证 pg 可行性）

    Args:
        x_int: (ng, T) 整数解（值为 0 或 1）
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔
        tol: 约束违反容忍度

    Returns:
        (is_feasible, reason): True 表示可行，否则附带违反原因
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    ng, T = x_int.shape
    Pd_sum = np.sum(pd_data, axis=0)

    Ton = min(int(4 * T_delta), T - 1)
    Toff = min(int(4 * T_delta), T - 1)

    # 检查1：最小开关机时间（代数）
    for g in range(ng):
        for tau in range(1, Ton + 1):
            for t1 in range(T - tau):
                if x_int[g, t1+1] - x_int[g, t1] > x_int[g, t1+tau] + tol:
                    return False, f"最小开机时间违反: 机组{g}, t1={t1}, tau={tau}"
        for tau in range(1, Toff + 1):
            for t1 in range(T - tau):
                if -x_int[g, t1+1] + x_int[g, t1] > 1 - x_int[g, t1+tau] + tol:
                    return False, f"最小关机时间违反: 机组{g}, t1={t1}, tau={tau}"

    # 检查2：功率平衡 + 爬坡（LP 软可行性）
    model = gp.Model('uc_feasibility_check')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    slack_pos = model.addVars(T, lb=0, name='sp')
    slack_neg = model.addVars(T, lb=0, name='sn')

    Ru = 0.4 * gen[:, PMAX] / T_delta
    Rd = 0.4 * gen[:, PMAX] / T_delta
    Ru_co = 0.3 * gen[:, PMAX]
    Rd_co = 0.3 * gen[:, PMAX]

    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x_int[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x_int[g, t])
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1] <= Ru[g] * x_int[g, t-1] + Ru_co[g] * (1 - x_int[g, t-1])
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t] <= Rd[g] * x_int[g, t] + Rd_co[g] * (1 - x_int[g, t])
            )

    for t in range(T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + slack_pos[t] - slack_neg[t]
            == float(Pd_sum[t])
        )

    obj = gp.quicksum(slack_pos[t] + slack_neg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        total_slack = model.ObjVal
        if total_slack > tol:
            return False, f"功率平衡不可行: 总松弛量={total_slack:.4f} MW"
        return True, "可行"

    return False, f"可行性 LP 求解失败: status={model.status}"


# ========================== Step 4：可行性泵主循环 ==========================

def run_feasibility_pump(
    x_curr: np.ndarray,
    trusted_mask: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    max_iter: int = 50,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, bool]:
    """
    可行性泵主循环。

    在固定高可信度变量的条件下，通过"LP投影 → 四舍五入"循环寻找整数可行解。

    Args:
        x_curr: (ng, T) 初始整数点（0/1 矩阵）
        trusted_mask: (ng, T) bool，True 表示固定不变
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔
        max_iter: 最大迭代次数
        rng: 随机数生成器（用于振荡扰动）
        verbose: 是否打印迭代信息

    Returns:
        (x_result, success): 最终整数点，是否找到可行解
    """
    if rng is None:
        rng = np.random.default_rng()

    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    ng, T = x_curr.shape
    Pd_sum = np.sum(pd_data, axis=0)

    Ru = 0.4 * gen[:, PMAX] / T_delta
    Rd = 0.4 * gen[:, PMAX] / T_delta
    Ru_co = 0.3 * gen[:, PMAX]
    Rd_co = 0.3 * gen[:, PMAX]
    Ton = min(int(4 * T_delta), T - 1)
    Toff = min(int(4 * T_delta), T - 1)

    history: List[tuple] = []
    no_improve_count = 0

    x_curr = x_curr.copy()

    for iteration in range(max_iter):
        # 检验当前点是否已可行
        is_feas, reason = check_uc_feasibility(x_curr, ppc, pd_data, T_delta)
        if is_feas:
            if verbose:
                print(f"  FP: 第{iteration}轮找到可行解", flush=True)
            return x_curr, True

        # LP Projection：最小化 L1(x, x_curr)，满足 UC 连续约束
        model = gp.Model('fp_projection')
        model.Params.OutputFlag = 0

        pg = model.addVars(ng, T, lb=0, name='pg')
        x = model.addVars(ng, T, lb=0, ub=1, name='x')
        d = model.addVars(ng, T, lb=0, name='d')   # |x - x_curr| 辅助变量

        # 功率平衡
        for t in range(T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
                name=f'pb_{t}'
            )

        for g in range(ng):
            # 发电上下限
            for t in range(T):
                model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

            # 爬坡约束
            for t in range(1, T):
                model.addConstr(
                    pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1])
                )
                model.addConstr(
                    pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
                )

            # 最小开关机时间
            for tau in range(1, Ton + 1):
                for t1 in range(T - tau):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
            for tau in range(1, Toff + 1):
                for t1 in range(T - tau):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

            # 每个变量的 L1 距离约束 or 固定可信变量
            for t in range(T):
                x_ref = float(x_curr[g, t])
                if trusted_mask[g, t]:
                    model.addConstr(x[g, t] == x_ref)
                    model.addConstr(d[g, t] == 0.0)
                else:
                    model.addConstr(d[g, t] >= x[g, t] - x_ref)
                    model.addConstr(d[g, t] >= x_ref - x[g, t])

        # 目标：最小化 L1 距离（仅对非可信变量）
        obj = gp.quicksum(
            d[g, t]
            for g in range(ng)
            for t in range(T)
            if not trusted_mask[g, t]
        )
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            if verbose:
                print(f"  FP: 第{iteration}轮 LP 投影失败 (status={model.status})", flush=True)
            break

        x_LP_proj = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])

        # 四舍五入得到新整数点，强制保持可信变量
        x_next = round_to_integer(x_LP_proj)
        x_next[trusted_mask] = x_curr[trusted_mask]

        if verbose:
            l1_dist = float(model.ObjVal)
            changed = int(np.sum(x_next != x_curr))
            print(f"  FP iter {iteration}: L1投影={l1_dist:.3f}, 变化位数={changed}", flush=True)

        # 振荡检测（最近历史中出现过相同点）
        x_key = tuple(x_next.flatten())
        if x_key in history:
            no_improve_count += 1
            if no_improve_count >= 3:
                if verbose:
                    print(f"  FP: 检测到振荡，随机扰动低置信度变量", flush=True)
                # 随机翻转少量非可信变量
                free_idx = np.argwhere(~trusted_mask)
                n_flip = max(1, len(free_idx) // 10)
                chosen = rng.choice(len(free_idx), size=n_flip, replace=False)
                for idx in chosen:
                    g_f, t_f = free_idx[idx]
                    x_next[g_f, t_f] = 1 - x_next[g_f, t_f]
                no_improve_count = 0
        else:
            no_improve_count = 0

        history.append(x_key)
        if len(history) > 10:
            history.pop(0)

        x_curr = x_next

    if verbose:
        print(f"  FP: 达到最大迭代 {max_iter} 次，未找到可行解", flush=True)
    return x_curr, False


# ========================== Step 6：顶层接口 ==========================

def recover_integer_solution(
    pd_data: np.ndarray,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_predictor,
    ppc: dict,
    T_delta: float,
    n_perturbations: int = 5,
    conf_threshold: float = 0.15,
    max_fp_iter: int = 50,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Optional[np.ndarray], bool]:
    """
    顶层接口：从 LP 松弛解恢复 UC 整数可行解。

    Pipeline：
      1. 通过 lambda_predictor 获取对偶变量
      2. 求解全局 UC LP 松弛（含代理约束）→ x_LP，启发式四舍五入 → x_init
      3. 各机组子问题 LP（+ 参数扰动）收集多组整数解
      4. 识别高可信度变量（整数性强 + 多来源一致）
      5. 可行性泵：从 x_init 出发；失败则从 x_init_k 再试

    Args:
        pd_data: (nb_load, T) 负荷数据
        trainers: {unit_id: SubproblemSurrogateTrainer} 已训练的代理约束训练器
        lambda_predictor: 对偶变量预测器，需支持 `predict(pd_data) -> (T,)`
        ppc: PyPower 案例数据
        T_delta: 时间间隔（小时）
        n_perturbations: 参数扰动次数
        conf_threshold: LP 整数性置信阈值
        max_fp_iter: 可行性泵最大迭代次数
        verbose: 是否打印进度
        rng: 随机数生成器（传 None 则使用固定种子 42）

    Returns:
        (x_feasible, success): 整数解矩阵 (ng, T)，以及是否为可行解
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1：获取对偶变量
    if verbose:
        print("Step 1: 获取对偶变量 lambda ...", flush=True)
    lambda_val = lambda_predictor.predict(pd_data)  # (T,)

    # Step 2：全局 LP 松弛 + 启发式四舍五入
    if verbose:
        print("Step 2: 求解全局 UC LP 松弛 ...", flush=True)
    x_LP = solve_global_LP_relaxation(ppc, pd_data, T_delta, trainers, lambda_val)
    x_init = round_to_integer(x_LP)

    integrality_gap = float(np.mean(np.minimum(x_LP, 1 - x_LP)))  # 平均到0或1的距离
    if verbose:
        print(f"  整数性间隙（平均）: {integrality_gap:.4f}", flush=True)

    # Step 3：收集多组整数解
    if verbose:
        print("Step 3: 收集多组整数解（子问题 LP + 扰动）...", flush=True)
    x_init_k, x_init_k_m = collect_integer_solutions(
        pd_data, lambda_val, trainers, n_perturbations=n_perturbations, rng=rng
    )

    # Step 4：识别高可信度变量
    if verbose:
        print("Step 4: 识别高可信度变量 ...", flush=True)
    trusted_mask = identify_trusted_mask(
        x_LP, x_init_k, x_init_k_m, conf_threshold=conf_threshold
    )
    ng, T = x_LP.shape
    n_trusted = int(np.sum(trusted_mask))
    if verbose:
        print(f"  可信变量: {n_trusted}/{ng*T} ({100*n_trusted/(ng*T):.1f}%)", flush=True)

    # Step 5：可行性泵（第一轮：从全局 LP 整数化解出发）
    if verbose:
        print("Step 5: 运行可行性泵（初始点：全局 LP 四舍五入解）...", flush=True)
    x_result, success = run_feasibility_pump(
        x_init, trusted_mask, ppc, pd_data, T_delta,
        max_iter=max_fp_iter, rng=rng, verbose=verbose
    )

    if not success:
        # 第二轮：从子问题 LP 整数解出发重试
        if verbose:
            print("Step 5b: 可行性泵（初始点：子问题 LP 整数解）...", flush=True)
        x_result, success = run_feasibility_pump(
            x_init_k, trusted_mask, ppc, pd_data, T_delta,
            max_iter=max_fp_iter, rng=rng, verbose=verbose
        )

    if verbose:
        status_str = "✓ 找到可行解" if success else "✗ 未找到可行解（返回最近整数点）"
        print(status_str, flush=True)

    return x_result, success
