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
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A, BR_STATUS

try:
    from uc_NN_subproblem import SubproblemSurrogateTrainer
except ImportError:
    from src.uc_NN_subproblem import SubproblemSurrogateTrainer
try:
    from sparse_surrogate_mining import (
        SparseSurrogateLibrary,
        add_sparse_parameterized_constraints,
    )
except ImportError:
    from src.sparse_surrogate_mining import (
        SparseSurrogateLibrary,
        add_sparse_parameterized_constraints,
    )
try:
    from sparse_constraint_templates import (
        SparseConstraintTemplateLibrary,
        add_sparse_x_templates_to_model,
    )
except ImportError:
    from src.sparse_constraint_templates import (
        SparseConstraintTemplateLibrary,
        add_sparse_x_templates_to_model,
    )
try:
    from scenario_utils import get_sample_net_load, normalize_sample_arrays
except ImportError:
    from src.scenario_utils import get_sample_net_load, normalize_sample_arrays


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


def _repair_min_up_down_heuristic(x_int: np.ndarray, T_delta: float) -> np.ndarray:
    """对整数点做轻量最小开关机时间修复，提升热启动质量。"""
    x_repaired = np.asarray(x_int, dtype=int).copy()
    ng, T = x_repaired.shape
    Ton = min(int(4 * T_delta), T - 1)
    Toff = min(int(4 * T_delta), T - 1)

    if T <= 1:
        return x_repaired

    for g in range(ng):
        changed = True
        while changed:
            changed = False

            # 启动后必须至少持续 Ton 个时段开机
            for t in range(T - 1):
                if x_repaired[g, t] == 0 and x_repaired[g, t + 1] == 1:
                    end = min(T, t + 1 + Ton)
                    if np.any(x_repaired[g, t + 1:end] == 0):
                        x_repaired[g, t + 1:end] = 1
                        changed = True

            # 关机后必须至少持续 Toff 个时段停机
            for t in range(T - 1):
                if x_repaired[g, t] == 1 and x_repaired[g, t + 1] == 0:
                    end = min(T, t + 1 + Toff)
                    if np.any(x_repaired[g, t + 1:end] == 1):
                        x_repaired[g, t + 1:end] = 0
                        changed = True

    return x_repaired


def _temporal_neighbor_average(x: np.ndarray, radius: int = 1) -> np.ndarray:
    """按时间维计算邻域平均，用于方向性热启动与周边平均约束。"""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[np.newaxis, :]

    ng, T = x_arr.shape
    avg = np.zeros_like(x_arr, dtype=float)
    for t in range(T):
        lo = max(0, t - radius)
        hi = min(T, t + radius + 1)
        avg[:, t] = np.mean(x_arr[:, lo:hi], axis=1)
    return avg if x.ndim == 2 else avg[0]


def _build_directional_hot_start(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    trusted_mask: np.ndarray
) -> np.ndarray:
    """利用 surrogate 相对 LP 的方向性，生成非对称热启动点。"""
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)

    direction = x_surr_LP - x_LP
    lp_neighbor = _temporal_neighbor_average(x_LP, radius=1)
    surr_neighbor = _temporal_neighbor_average(x_surr_LP, radius=1)
    round_neighbor = 0.5 * _temporal_neighbor_average(lp_round, radius=1) \
                     + 0.5 * _temporal_neighbor_average(surr_round, radius=1)

    # surrogate 比全局 LP 更强调局部结构，因此给更高权重；
    # direction 与邻域平均共同决定是否做“反四舍五入”。
    directional_score = (
        0.35 * x_LP
        + 0.65 * x_surr_LP
        + 2.20 * direction
        + 0.60 * (0.5 * lp_neighbor + 0.5 * surr_neighbor - 0.5)
        + 0.35 * (round_neighbor - 0.5)
    )

    x_directional = (directional_score >= 0.5).astype(int)

    upward_mask = (
        (~trusted_mask)
        & (direction >= 0.05)
        & ((surr_neighbor >= 0.55) | (round_neighbor >= 0.75))
    )
    downward_mask = (
        (~trusted_mask)
        & (direction <= -0.05)
        & ((surr_neighbor <= 0.45) | (round_neighbor <= 0.25))
    )

    x_directional[upward_mask] = 1
    x_directional[downward_mask] = 0
    x_directional[trusted_mask] = lp_round[trusted_mask]
    return x_directional


def _neighbor_average_1d(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """对 surrogate 约束参数做邻域平均平滑。"""
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.mean(arr[lo:hi])
    return out


def _perturb_surrogate_outputs(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    rng: np.random.Generator,
    perturb_std: float = 0.10,
    neighborhood_weight: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对 surrogate 网络输出参数做扰动，并引入周边平均约束。"""
    def _perturb_one(arr: np.ndarray, clip_nonnegative: bool = False) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        scale = np.maximum(np.abs(arr), 1.0)
        noisy = arr + perturb_std * scale * rng.standard_normal(arr.shape)
        neighbor_avg = _neighbor_average_1d(noisy, radius=1)
        mixed = (1.0 - neighborhood_weight) * noisy + neighborhood_weight * neighbor_avg
        if clip_nonnegative:
            mixed = np.maximum(mixed, 0.0)
        return mixed

    return (
        _perturb_one(alphas, clip_nonnegative=False),
        _perturb_one(betas, clip_nonnegative=False),
        _perturb_one(gammas, clip_nonnegative=False),
        _perturb_one(deltas, clip_nonnegative=True),
    )


def _build_hot_start_candidates(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    T_delta: float
) -> List[Tuple[str, np.ndarray]]:
    """根据全局 LP 与 surrogate LP 解构造多组启发式热启动点。"""
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)
    directional = _build_directional_hot_start(x_LP, x_surr_LP, trusted_mask)

    lp_margin = np.abs(x_LP - 0.5)
    surr_margin = np.abs(x_surr_LP - 0.5)
    choose_lp = lp_margin >= surr_margin

    blended = np.where(choose_lp, lp_round, surr_round)
    agreement = np.where(lp_round == surr_round, lp_round, blended)

    vote_sum = (
        lp_round.astype(float)
        + surr_round.astype(float)
        + x_init_k.astype(float)
        + np.sum(x_init_k_m.astype(float), axis=1)
    )
    n_votes = 3 + x_init_k_m.shape[1]
    majority = (vote_sum / n_votes >= 0.5).astype(int)

    trusted_ref = np.where(trusted_mask, lp_round, majority)
    confidence_mix = np.where(trusted_mask, lp_round, agreement)

    candidate_specs = [
        ("lp_round", lp_round),
        ("surrogate_lp_round", surr_round),
        ("directional_surrogate_start", directional),
        ("lp_surrogate_confidence_mix", blended),
        ("lp_surrogate_agreement", agreement),
        ("vote_majority", majority),
        ("trusted_vote_mix", trusted_ref),
        ("trusted_confidence_mix", confidence_mix),
    ]

    unique_candidates: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for name, x_candidate in candidate_specs:
        repaired = _repair_min_up_down_heuristic(x_candidate, T_delta)
        key = tuple(repaired.flatten().tolist())
        if key not in seen:
            seen.add(key)
            unique_candidates.append((name, repaired))

    return unique_candidates


def _rank_hot_start_candidates(
    candidates: List[Tuple[str, np.ndarray]],
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray
) -> List[Tuple[str, np.ndarray, float]]:
    """按与 LP / surrogate LP / 投票参考的一致性对热启动候选排序。"""
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)
    vote_sum = x_init_k.astype(float) + np.sum(x_init_k_m.astype(float), axis=1)
    vote_majority = (vote_sum / (1 + x_init_k_m.shape[1]) >= 0.5).astype(int)

    ranked: List[Tuple[str, np.ndarray, float]] = []
    for name, x_candidate in candidates:
        trusted_match = float(np.sum(x_candidate[trusted_mask] == lp_round[trusted_mask]))
        vote_match = float(np.sum(x_candidate == vote_majority))
        lp_dist = float(np.sum(np.abs(x_candidate - x_LP)))
        surr_dist = float(np.sum(np.abs(x_candidate - x_surr_LP)))
        surrogate_match = float(np.sum(x_candidate == surr_round))
        score = (
            3.0 * trusted_match
            + 1.5 * vote_match
            + 0.5 * surrogate_match
            - 0.75 * lp_dist
            - 0.50 * surr_dist
        )
        ranked.append((name, x_candidate, score))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


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


# ========================== 内部辅助：DC 潮流数据预计算 ==========================

def _build_ptdf_data(ppc_int: dict) -> tuple:
    """计算 DC 潮流约束所需的 PTDF 矩阵与发电机-总线关联矩阵。

    Args:
        ppc_int: ext2int 处理后的 PyPower 案例字典（总线已 0-indexed）。

    Returns:
        PTDF:         (nl, nb) 功率转移分布因子矩阵（无量纲）。
        ptdf_g:       (nl, ng) PTDF @ G_bus，即 pg 变量的线路潮流系数（MW/MW）。
        branch_limit: (nl,)   线路热容量上限，RATE_A（MW）。
        active_lines: 需施加约束的线路索引列表（RATE_A > 0 且线路在线）。
    """
    gen    = ppc_int['gen']
    branch = ppc_int['branch']
    ng     = gen.shape[0]
    nb     = ppc_int['bus'].shape[0]
    nl     = branch.shape[0]

    PTDF = makePTDF(ppc_int['baseMVA'], ppc_int['bus'], branch)  # (nl, nb)

    # 发电机-总线关联矩阵（ext2int 后总线 0-indexed）
    G_bus = np.zeros((nb, ng))
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        G_bus[bus_idx, g] = 1.0

    ptdf_g       = PTDF @ G_bus            # (nl, ng)
    branch_limit = branch[:, RATE_A]       # (nl,) MW

    # 仅约束有热容限制（RATE_A > 0）且在线（BR_STATUS = 1）的线路
    active_lines = [
        l for l in range(nl)
        if branch_limit[l] > 1e-6 and branch[l, BR_STATUS] > 0
    ]
    return PTDF, ptdf_g, branch_limit, active_lines


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
    pd_data: np.ndarray | dict,
    lambda_val: np.ndarray,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    n_perturbations: int = 5,
    perturb_std: float = 0.1,
    neighborhood_weight: float = 0.35,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    收集多组整数解（来自子问题 LP + 代理参数扰动）。

    Args:
        pd_data: (nb_load, T) 负荷数据
        lambda_val: (T,) 功率平衡对偶变量
        trainers: {unit_id: SubproblemSurrogateTrainer}
        n_perturbations: 扰动次数
        perturb_std: surrogate 网络输出参数扰动标准差（相对值）
        neighborhood_weight: 周边平均约束权重，越大表示越强调相邻约束平滑
        rng: 随机数生成器

    Returns:
        x_surr_lp:  (ng, T) surrogate 子问题 LP 连续解
        x_init_k:   (ng, T) 各机组子问题 LP 整数解
        x_init_k_m: (ng, n_perturbations, T) 扰动参数后的多组整数解
    """
    if rng is None:
        rng = np.random.default_rng()

    sample = None
    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        pd_matrix = get_sample_net_load(sample)
    else:
        pd_matrix = pd_data

    unit_ids = sorted(trainers.keys())
    ng = max(unit_ids) + 1
    T = pd_matrix.shape[1]

    x_surr_lp = np.zeros((ng, T))
    x_init_k = np.zeros((ng, T), dtype=int)
    x_init_k_m = np.zeros((ng, n_perturbations, T), dtype=int)

    for g in unit_ids:
        trainer = trainers[g]
        alphas, betas, gammas, deltas, *_ = trainer.get_surrogate_params(sample if sample is not None else pd_matrix, lambda_val)

        # 原始子问题 LP
        x_LP_k = _solve_unit_LP_with_surrogate(trainer, lambda_val, alphas, betas, gammas, deltas)
        x_surr_lp[g] = x_LP_k
        x_init_k[g] = round_to_integer(x_LP_k)

        # 扰动 surrogate 网络输出参数，并用周边平均约束平滑，生成多组解
        for m in range(n_perturbations):
            alphas_m, betas_m, gammas_m, deltas_m = _perturb_surrogate_outputs(
                alphas, betas, gammas, deltas,
                rng=rng,
                perturb_std=perturb_std,
                neighborhood_weight=neighborhood_weight,
            )
            x_LP_m = _solve_unit_LP_with_surrogate(
                trainer, lambda_val, alphas_m, betas_m, gammas_m, deltas_m
            )
            x_init_k_m[g, m] = round_to_integer(x_LP_m)

    return x_surr_lp, x_init_k, x_init_k_m


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
    pd_data: np.ndarray | dict,
    T_delta: float,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_val: np.ndarray,
    agent=None,
    sparse_library: Optional[SparseSurrogateLibrary] = None,
    sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
) -> np.ndarray:
    """
    构建完整 UC LP 松弛（x ∈ [0,1]），加入 V3 代理约束 + BCD theta/zeta 代理约束，求解得 x_LP。

    Args:
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔（小时）
        trainers: {unit_id: trainer}
        lambda_val: (T,) 功率平衡对偶变量（用于查询代理约束参数）
        agent: （可选）已训练的 Agent_NN_BCD 实例；若提供则额外加入其
            theta/zeta 参数化代理约束（M-penalty 软约束形式）
        sparse_library: （可选）离线筛选出的稀疏参数化约束库。若提供则以
            `lhs(x, pg) - rhs(Pd) <= slack` 的软约束形式加入全局 LP。
        sparse_x_template_library: （可选）x[g,t] 稀疏支持集模板库，以
            `sum x[g_k, t_k] <= rhs + slack` 的形式软注入。

    Returns:
        x_LP: (ng, T) LP 松弛解；若不可行返回零矩阵
    """
    sample = None
    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        pd_matrix = get_sample_net_load(sample)
    else:
        pd_matrix = pd_data

    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng = gen.shape[0]
    T = pd_matrix.shape[1]
    Pd_sum = np.sum(pd_matrix, axis=0)  # (T,)

    model = gp.Model('global_LP_relaxation')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    x = model.addVars(ng, T, lb=0, ub=1, name='x')
    cpower = model.addVars(ng, T, lb=0, name='cpower')
    coc = model.addVars(ng, T - 1, lb=0, name='coc')   # 启停成本

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
    start_cost = gencost[:, 1]   # gencost 第 1 列：启动成本
    shut_cost  = gencost[:, 2]   # gencost 第 2 列：关机成本

    # 代理约束惩罚系数（大 M）：保证 LP 始终可行，违反代理约束时以高代价惩罚
    # 取值参考 BCD 中 penalty_factor 量级，比发电成本大一个数量级
    M_SURR = 1e5
    surr_slacks: list = []          # 收集所有代理约束松弛变量，用于构造惩罚项

    T_triples = max(1, T - 2)       # 预计算三时段约束索引范围

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

        # 启停成本（参考 BCD / uc_cplex.py）
        for t in range(1, T):
            model.addConstr(
                coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]),
                name=f'start_cost_{g}_{t}'
            )
            model.addConstr(
                coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]),
                name=f'shut_cost_{g}_{t}'
            )

        # 发电成本
        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                               + gencost[g, -1] / T_delta * x[g, t]
            )

        # V3 全局代理约束（所有已训练机组）——软约束形式（M-penalty）
        # 参考 BCD 中对原始约束的处理：slack_k >= lhs - delta; obj += M * slack_k
        # 这样即使代理约束紧张时 LP 仍可行，违背量以大 M 惩罚
        if g in trainers:
            alphas, betas, gammas, deltas, *_ = trainers[g].get_surrogate_params(
                sample if sample is not None else pd_matrix, lambda_val
            )
            for k in range(len(alphas)):
                t_k  = k % T_triples
                t_k1 = min(t_k + 1, T - 1)
                t_k2 = min(t_k + 2, T - 1)
                a = float(alphas[k])
                b = float(betas[k])
                c = float(gammas[k])
                r = float(deltas[k])
                if abs(a) > 1e-10 or abs(b) > 1e-10 or abs(c) > 1e-10:
                    slack_k = model.addVar(lb=0, name=f'g{g}_surr_slack_{k}')
                    model.addConstr(
                        a * x[g, t_k] + b * x[g, t_k1] + c * x[g, t_k2] - r <= slack_k,
                        name=f'g{g}_surr_{k}'
                    )
                    surr_slacks.append(slack_k)

    # DC 线路潮流约束（PTDF 方法，硬约束）
    # 假设 pd_data 形状为 (nb, T)，行顺序与 ext2int 后的总线顺序一致
    try:
        _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = _PTDF @ pd_matrix   # (nl, T)：负荷对各线路潮流的贡献
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit,  name=f'flow_upper_{l}_{t}')
                model.addConstr(flow_expr >= -limit, name=f'flow_lower_{l}_{t}')
    except Exception as _dc_err:
        print(f"  警告: DC 潮流约束构建失败（{_dc_err}），跳过线路约束", flush=True)

    # BCD theta/zeta 代理约束（M-penalty 软约束）
    # theta 约束：DCPF 参数化约束，形式 Σ theta_{branch,unit,t}*x[unit,t] <= theta_rhs_{branch,t}
    # zeta  约束：按变量参数化约束，形式 zeta_{unit,t}*x[unit,t] <= zeta_rhs_{unit,t}
    if agent is not None:
        _ua = getattr(agent, '_current_union_analysis', None)
        _tv = getattr(agent, 'theta_values', {}) or {}
        _zv = getattr(agent, 'zeta_values', {}) or {}

        # --- theta 约束 ---
        _uc = (_ua or {}).get('union_constraints', [])
        for _ci in _uc:
            _bid  = _ci['branch_id']
            _ts   = _ci['time_slot']
            _nzc  = _ci.get('nonzero_pg_coefficients', [])
            _lhs  = gp.LinExpr()
            for _ci2 in _nzc:
                _uid   = _ci2['unit_id']
                _tname = f'theta_branch_{_bid}_unit_{_uid}_time_{_ts}'
                _coeff = float(_tv.get(_tname, 0.0))
                if abs(_coeff) > 1e-10 and _uid < ng and _ts < T:
                    _lhs += _coeff * x[_uid, _ts]
            _rhs_name = f'theta_rhs_branch_{_bid}_time_{_ts}'
            _rhs = float(_tv.get(_rhs_name, 1.0))
            _slack = model.addVar(lb=0, name=f'theta_slack_{_bid}_{_ts}')
            model.addConstr(_lhs - _rhs <= _slack, name=f'theta_surr_{_bid}_{_ts}')
            surr_slacks.append(_slack)

        # --- zeta 约束 ---
        _zuc = (_ua or {}).get('union_zeta_constraints', [])
        for _zc in _zuc:
            _uid  = _zc['unit_id']
            _ts   = _zc['time_slot']
            _zname = f'zeta_unit_{_uid}_time_{_ts}'
            _coeff = float(_zv.get(_zname, 0.0))
            _rhs_name = f'zeta_rhs_unit_{_uid}_time_{_ts}'
            _rhs = float(_zv.get(_rhs_name, 1.0))
            if abs(_coeff) > 1e-10 and _uid < ng and _ts < T:
                _slack = model.addVar(lb=0, name=f'zeta_slack_{_uid}_{_ts}')
                model.addConstr(
                    _coeff * x[_uid, _ts] - _rhs <= _slack,
                    name=f'zeta_surr_{_uid}_{_ts}'
                )
                surr_slacks.append(_slack)

    # 可选的稀疏参数化约束库（M-penalty 软约束）
    add_sparse_parameterized_constraints(
        model,
        x,
        pg,
        pd_data,
        sparse_library=sparse_library,
        surr_slacks=surr_slacks,
        name_prefix="sparse",
    )
    add_sparse_x_templates_to_model(
        model,
        x,
        template_library=sparse_x_template_library,
        surr_slacks=surr_slacks,
        name_prefix="sparse_x",
    )

    # 目标：发电成本 + 启停成本 + M × Σ 代理约束违背量（软约束惩罚）
    obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
    if surr_slacks:
        obj += M_SURR * gp.quicksum(surr_slacks)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return np.array([[x[g, t].X for t in range(T)] for g in range(ng)])

    print(f"  警告: 全局 LP 松弛求解失败 (status={model.status})，返回零矩阵", flush=True)
    return np.zeros((ng, T))

def solve_global_LP_relaxation_without_surrogate(
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float
) -> np.ndarray:
    """
    构建完整 UC LP 松弛（x ∈ [0,1]），不含任何代理约束，仅含 UC 基础约束和 DC 潮流约束。

    Args:
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔（小时）

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
    coc = model.addVars(ng, T - 1, lb=0, name='coc')   # 启停成本

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
    start_cost = gencost[:, 1]   # gencost 第 1 列：启动成本
    shut_cost  = gencost[:, 2]   # gencost 第 2 列：关机成本

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

        # 启停成本（参考 BCD / uc_cplex.py）
        for t in range(1, T):
            model.addConstr(
                coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]),
                name=f'start_cost_{g}_{t}'
            )
            model.addConstr(
                coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]),
                name=f'shut_cost_{g}_{t}'
            )

        # 发电成本
        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                               + gencost[g, -1] / T_delta * x[g, t]
            )

    # DC 线路潮流约束（PTDF 方法，硬约束）
    # 假设 pd_data 形状为 (nb, T)，行顺序与 ext2int 后的总线顺序一致
    try:
        _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = _PTDF @ pd_data   # (nl, T)：负荷对各线路潮流的贡献
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit,  name=f'flow_upper_{l}_{t}')
                model.addConstr(flow_expr >= -limit, name=f'flow_lower_{l}_{t}')
    except Exception as _dc_err:
        print(f"  警告: DC 潮流约束构建失败（{_dc_err}），跳过线路约束", flush=True)

    # 目标：发电成本 + 启停成本
    obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
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

    # 检查3：DC 线路潮流约束（作为硬约束加入 LP）
    # 功率平衡含松弛变量；若 LP 因线路约束不可行则说明网络拥塞
    _dc_lines_added = False
    try:
        _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = _PTDF @ pd_data   # (nl, T)
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
        _dc_lines_added = True
    except Exception:
        pass   # DC 约束不可用时退化为仅检查功率平衡

    obj = gp.quicksum(slack_pos[t] + slack_neg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        total_slack = model.ObjVal
        if total_slack > tol:
            return False, f"功率平衡不可行: 总松弛量={total_slack:.4f} MW"
        return True, "可行"

    if model.status == GRB.INFEASIBLE and _dc_lines_added:
        return False, "DC 线路潮流约束不可行（网络拥塞）"
    return False, f"可行性 LP 求解失败: status={model.status}"


# ========================== Step 4：可行性泵主循环 ==========================

def run_feasibility_pump(
    x_curr: np.ndarray,
    trusted_mask: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    x_pool: Optional[np.ndarray] = None,
    max_iter: int = 50,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, bool]:
    """
    可行性泵主循环。

    在固定高可信度变量的条件下，通过"LP投影 → 四舍五入"循环寻找整数可行解。
    从第二次迭代（iteration >= 1）起，若提供 x_pool，则将 x 约束为已知整数解的凸组合。

    Args:
        x_curr: (ng, T) 初始整数点（0/1 矩阵）
        trusted_mask: (ng, T) bool，True 表示固定不变
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔
        x_pool: (n_pool, ng, T) 整数解池，iteration >= 1 时用于凸组合约束（可选）
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

    # 预计算 DC 潮流数据（循环外一次性完成，避免重复构建 PTDF）
    _dc_available = False
    _ptdf_g = None
    _ptdf_Pd = None
    _branch_limit = None
    _active_lines = []
    try:
        _PTDF, _ptdf_g, _branch_limit, _active_lines = _build_ptdf_data(ppc_int)
        _ptdf_Pd = _PTDF @ pd_data   # (nl, T)
        _dc_available = True
        if verbose:
            print(f"  FP: DC 潮流约束已启用（{len(_active_lines)} 条有效线路）", flush=True)
    except Exception as _e:
        if verbose:
            print(f"  FP 警告: DC 潮流约束不可用（{_e}）", flush=True)

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

        # iteration >= 1 且提供了 x_pool：将 x 约束为整数解池的凸组合
        if x_pool is not None and iteration >= 1:
            n_pool = x_pool.shape[0]
            omega = model.addVars(ng, n_pool, lb=0, name='omega')
            for g in range(ng):
                model.addConstr(gp.quicksum(omega[g, j] for j in range(n_pool)) == 1.0)
                for t in range(T):
                    model.addConstr(
                        x[g, t] == gp.quicksum(
                            float(x_pool[j, g, t]) * omega[g, j] for j in range(n_pool)
                        )
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

        # DC 线路潮流约束（使用预计算的 PTDF 数据）
        if _dc_available:
            for l in _active_lines:
                limit = float(_branch_limit[l])
                for t in range(T):
                    flow_expr = (
                        gp.quicksum(float(_ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                        - float(_ptdf_Pd[l, t])
                    )
                    model.addConstr(flow_expr <= limit)
                    model.addConstr(flow_expr >= -limit)

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
    pd_data: np.ndarray | dict,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_predictor,
    ppc: dict,
    T_delta: float,
    manager=None,
    sparse_library: Optional[SparseSurrogateLibrary] = None,
    sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    n_perturbations: int = 5,
    conf_threshold: float = 0.15,
    max_fp_iter: int = 50,
    perturb_std: float = 0.1,
    neighborhood_weight: float = 0.35,
    use_hot_start: bool = True,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Optional[np.ndarray], bool]:
    """
    顶层接口：从 LP 松弛解恢复 UC 整数可行解。

    Pipeline：
      1. 通过 lambda_predictor 获取对偶变量
      2. 求解全局 UC LP 松弛（含代理约束）→ x_LP，启发式四舍五入 → x_init
         （若提供 manager，则使用 manager.solve_global 替代默认 LP 求解器）
      3. 各机组子问题 LP（+ 参数扰动）收集 surrogate LP 连续解与多组整数解
      4. 识别高可信度变量（整数性强 + 多来源一致）
      5. 基于全局 LP / surrogate LP 构造多组热启动候选
      6. 可行性泵：按热启动候选优先级依次尝试

    Args:
        pd_data: (nb_load, T) 负荷数据
        trainers: {unit_id: SubproblemSurrogateTrainer} 已训练的代理约束训练器
        lambda_predictor: 对偶变量预测器，需支持 `predict(pd_data) -> (T,)`
        ppc: PyPower 案例数据
        T_delta: 时间间隔（小时）
        manager: （可选）UnifiedSurrogateManager 实例；若提供则使用其
            solve_global 方法求解全局 LP 松弛（同时包含 theta/zeta 和 V3 代理约束）
        sparse_library: （可选）离线筛选出的稀疏参数化约束库，仅在显式提供时启用
        sparse_x_template_library: （可选）稀疏 x 支持集模板库，仅在显式提供时启用
        n_perturbations: 参数扰动次数
        conf_threshold: LP 整数性置信阈值
        max_fp_iter: 可行性泵最大迭代次数
        perturb_std: surrogate 网络输出参数扰动标准差
        neighborhood_weight: 周边平均约束权重
        use_hot_start: 是否启用基于 LP 与 surrogate LP 的热启动候选
        verbose: 是否打印进度
        rng: 随机数生成器（传 None 则使用固定种子 42）

    Returns:
        (x_feasible, success): 整数解矩阵 (ng, T)，以及是否为可行解
    """
    sample = None
    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        scenario_input = sample
        pd_data = get_sample_net_load(sample)
    else:
        scenario_input = pd_data

    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1：获取对偶变量
    if verbose:
        print("Step 1: 获取对偶变量 lambda ...", flush=True)
    lambda_val = lambda_predictor.predict(scenario_input)  # (T,)

    # Step 2：全局 LP 松弛 + 启发式四舍五入
    if verbose:
        print("Step 2: 求解全局 UC LP 松弛 ...", flush=True)
    if manager is not None:
        x_LP = manager.solve_global(
            scenario_input,
            lambda_val,
            sparse_library=sparse_library,
            sparse_x_template_library=sparse_x_template_library,
        )
    else:
        x_LP = solve_global_LP_relaxation(
            ppc,
            scenario_input,
            T_delta,
            trainers,
            lambda_val,
            sparse_library=sparse_library,
            sparse_x_template_library=sparse_x_template_library,
        )
    x_init = round_to_integer(x_LP)

    integrality_gap = float(np.mean(np.minimum(x_LP, 1 - x_LP)))  # 平均到0或1的距离
    if verbose:
        print(f"  整数性间隙（平均）: {integrality_gap:.4f}", flush=True)

    # Step 3：收集多组整数解
    if verbose:
        print("Step 3: 收集多组整数解（子问题 LP + 扰动）...", flush=True)
    x_surr_lp, x_init_k, x_init_k_m = collect_integer_solutions(
        scenario_input, lambda_val, trainers,
        n_perturbations=n_perturbations,
        perturb_std=perturb_std,
        neighborhood_weight=neighborhood_weight,
        rng=rng
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

    # Step 5：基于全局 LP 与 surrogate LP 的启发式热启动候选
    hot_start_candidates: List[Tuple[str, np.ndarray, float]] = []
    if use_hot_start:
        hot_start_candidates = _rank_hot_start_candidates(
            _build_hot_start_candidates(x_LP, x_surr_lp, x_init_k, x_init_k_m, trusted_mask, T_delta),
            x_LP, x_surr_lp, x_init_k, x_init_k_m, trusted_mask
        )
    else:
        hot_start_candidates = [
            ("lp_round", x_init, 0.0),
            ("surrogate_lp_round", _repair_min_up_down_heuristic(x_init_k, T_delta), -1.0),
        ]

    # 构建整数解池：子问题整数解 + 扰动解 + 热启动候选，供 FP 投影阶段使用
    x_pool_list = [x_init_k]
    x_pool_list.extend(list(x_init_k_m.transpose(1, 0, 2)))
    x_pool_list.extend([x_start for _, x_start, _ in hot_start_candidates])

    x_pool_unique: List[np.ndarray] = []
    x_pool_seen = set()
    for x_candidate in x_pool_list:
        key = tuple(np.asarray(x_candidate, dtype=int).flatten().tolist())
        if key not in x_pool_seen:
            x_pool_seen.add(key)
            x_pool_unique.append(np.asarray(x_candidate, dtype=int))
    x_pool = np.stack(x_pool_unique, axis=0)

    if verbose:
        print("Step 5: 运行可行性泵热启动 ...", flush=True)
        for idx, (name, _x_start, score) in enumerate(hot_start_candidates, start=1):
            print(f"  热启动候选 {idx}: {name}, score={score:.2f}", flush=True)

    x_result = hot_start_candidates[0][1] if hot_start_candidates else x_init
    success = False

    for idx, (name, x_start, _score) in enumerate(hot_start_candidates, start=1):
        start_feas, _reason = check_uc_feasibility(x_start, ppc, pd_data, T_delta)
        if start_feas:
            if verbose:
                print(f"  热启动候选 {idx}（{name}）已直接可行，无需进入 FP", flush=True)
            return x_start, True

        if verbose:
            print(f"  运行 FP 热启动 {idx}/{len(hot_start_candidates)}: {name}", flush=True)
        x_result, success = run_feasibility_pump(
            x_start, trusted_mask, ppc, pd_data, T_delta,
            x_pool=x_pool, max_iter=max_fp_iter, rng=rng, verbose=verbose
        )
        if success:
            break

    if verbose:
        status_str = "✓ 找到可行解" if success else "✗ 未找到可行解（返回最近整数点）"
        print(status_str, flush=True)

    return x_result, success
