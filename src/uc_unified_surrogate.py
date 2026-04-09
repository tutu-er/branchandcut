"""
统一代理约束管理器（UnifiedSurrogateManager）
================================================================================
接受已训练好的 Agent_NN（全局 theta/zeta 约束）和各机组 SubproblemSurrogateTrainer
（per-generator V3 代理约束），在同一全局 LP 中双路更新对偶变量，并为
feasibility_pump 提供统一的全局求解接口。

关键设计：
- 双路对偶更新：路径1（theta/zeta, 委托 agent.iter_with_dual_block）
                路径2（V3 per-generator, 委托 trainers[g].iter_with_dual_block）
- 推理接口：solve_global(pd_data, lambda_val) → x_LP (ng, T)
- feasibility_pump 集成：recover_integer_solution 增加 manager 可选参数
================================================================================
"""

import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Optional, Tuple

# 避免与 uc_NN.py 中直接替换 stdout 冲突：先 reconfigure
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A, BR_STATUS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from feasibility_pump import _add_surrogate_constraints, _build_ptdf_data
except ImportError:
    from src.feasibility_pump import _add_surrogate_constraints, _build_ptdf_data
try:
    from uc_NN_subproblem import build_surrogate_constraint_expression, resolve_constraint_offsets_from_trainer
except ImportError:
    from src.uc_NN_subproblem import build_surrogate_constraint_expression, resolve_constraint_offsets_from_trainer
try:
    from scenario_utils import get_feature_vector_from_sample, get_sample_net_load, normalize_sample_arrays
except ImportError:
    from src.scenario_utils import get_feature_vector_from_sample, get_sample_net_load, normalize_sample_arrays
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


class UnifiedSurrogateManager:
    """
    统一代理约束管理器：将全局 BCD（theta/zeta）与 per-generator V3 代理约束
    组合在同一全局 LP 中进行原始/对偶双路更新。

    Attributes:
        agent: 已训练好的 Agent_NN 实例
        trainers: {unit_id: SubproblemSurrogateTrainer}，已训练好的各机组 V3 模型
        n_samples: 样本数量
        ng: 机组数
        T: 时段数
        nl: 线路数
    """

    def __init__(
        self,
        agent,
        trainers: Dict,
        active_set_data: List[Dict],
        sparse_library: Optional[SparseSurrogateLibrary] = None,
        sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    ) -> None:
        """
        初始化统一代理约束管理器。

        Args:
            agent: 已训练好的 Agent_NN 实例（含 theta_net/zeta_net）
            trainers: {unit_id: SubproblemSurrogateTrainer} 各机组已训练 V3 模型
            active_set_data: 样本数据列表，每项含 'pd_data', 'lambda' 等字段
            sparse_library: （可选）离线筛选出的稀疏参数化约束库
            sparse_x_template_library: （可选）x 稀疏支持集模板库
        """
        self.agent = agent
        self.trainers = trainers
        self.active_set_data = active_set_data
        self.sparse_library = sparse_library
        self.sparse_x_template_library = sparse_x_template_library

        # 从 agent 读取基本维度和参数
        self.ppc = agent.ppc
        self.T = agent.T
        self.ng = agent.ng
        self.nl = agent.nl
        self.T_delta = agent.T_delta
        self.n_samples = len(active_set_data)

        # 惩罚参数（与 agent 保持同步）。
        # 兼容旧版 Agent_NN_BCD 仅暴露 gamma_base 而未显式存储 gamma 的情况。
        self.rho_primal = agent.rho_primal
        self.rho_dual = agent.rho_dual
        self.rho_opt = agent.rho_opt
        self.gamma = getattr(agent, 'gamma', getattr(agent, 'gamma_base', 0.0))

        # 从 agent 复制原始变量和对偶变量
        self.pg = agent.pg.copy()      # (n_samples, ng, T)
        self.x = agent.x.copy()       # (n_samples, ng, T)

        # 内部缓存：ext2int 处理后的 ppc
        self._ppc_int = ext2int(self.ppc)
        self._gen = self._ppc_int['gen']
        self._gencost = self._ppc_int['gencost']

    # =========================================================================
    # 核心方法：原始块（全局 LP 求解 pg/x）
    # =========================================================================

    def iter_with_primal_block(self, sample_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        在同一全局 LP 中施加两类约束，求解更新 pg/x。

        施加约束：
          1. UC 基础约束（功率平衡、爬坡、最小开停机、发电成本）
          2. theta/zeta 软约束（M-penalty，来自 agent）
          3. per-generator V3 代理软约束（M-penalty，来自 trainers[g]）

        Args:
            sample_id: 样本索引

        Returns:
            (pg_sol, x_sol): 各 shape (ng, T)；求解失败时返回当前缓存值
        """
        sample = normalize_sample_arrays(self.active_set_data[sample_id])
        Pd = get_sample_net_load(sample)
        Pd_sum = np.sum(Pd, axis=0)  # (T,)

        # 推断当前 theta/zeta 参数
        lambda_val = self._get_lambda_val(sample_id)
        theta_values, zeta_values = self.get_theta_zeta_params(sample, lambda_val)

        gen = self._gen
        gencost = self._gencost
        ng, T = self.ng, self.T
        T_delta = self.T_delta

        Ru = 0.4 * gen[:, PMAX] / T_delta
        Rd = 0.4 * gen[:, PMAX] / T_delta
        Ru_co = 0.3 * gen[:, PMAX]
        Rd_co = 0.3 * gen[:, PMAX]
        Ton = min(int(4 * T_delta), T - 1)
        Toff = min(int(4 * T_delta), T - 1)
        start_cost = gencost[:, 1]
        shut_cost = gencost[:, 2]
        M_SURR = 1e5
        model = gp.Model('unified_primal_block')
        model.Params.OutputFlag = 0

        pg = model.addVars(ng, T, lb=0, name='pg')
        x = model.addVars(ng, T, lb=0, ub=1, name='x')
        cpower = model.addVars(ng, T, lb=0, name='cpower')
        coc = model.addVars(ng, T - 1, lb=0, name='coc')

        surr_slacks: list = []

        # --- UC 基础约束 ---
        for t in range(T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
                name=f'pb_{t}'
            )

        for g in range(ng):
            for t in range(T):
                model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

            for t in range(1, T):
                model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1]))
                model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t]))

            for tau in range(1, Ton + 1):
                for t1 in range(T - tau):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
            for tau in range(1, Toff + 1):
                for t1 in range(T - tau):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

            for t in range(1, T):
                model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]))
                model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]))

            for t in range(T):
                model.addConstr(
                    cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                                   + gencost[g, -1] / T_delta * x[g, t]
                )

        # --- theta 软约束（M-penalty）---
        ua = self.agent._current_union_analysis or {}
        for ci in ua.get('union_constraints', []):
            bid = ci['branch_id']
            ts = ci['time_slot']
            lhs = gp.LinExpr()
            for ci2 in ci.get('nonzero_pg_coefficients', []):
                uid = ci2['unit_id']
                tname = f'theta_branch_{bid}_unit_{uid}_time_{ts}'
                coeff = float(theta_values.get(tname, 0.0))
                if abs(coeff) > 1e-10 and uid < ng and ts < T:
                    lhs += coeff * x[uid, ts]
            rhs_name = f'theta_rhs_branch_{bid}_time_{ts}'
            rhs = float(theta_values.get(rhs_name, 1.0))
            slack = model.addVar(lb=0, name=f'theta_slack_{bid}_{ts}')
            model.addConstr(lhs - rhs <= slack, name=f'theta_surr_{bid}_{ts}')
            surr_slacks.append(slack)

        # --- zeta 软约束（M-penalty）---
        for zc in ua.get('union_zeta_constraints', []):
            uid = zc['unit_id']
            ts = zc['time_slot']
            zname = f'zeta_unit_{uid}_time_{ts}'
            coeff = float(zeta_values.get(zname, 0.0))
            rhs_name = f'zeta_rhs_unit_{uid}_time_{ts}'
            rhs = float(zeta_values.get(rhs_name, 1.0))
            if abs(coeff) > 1e-10 and uid < ng and ts < T:
                slack = model.addVar(lb=0, name=f'zeta_slack_{uid}_{ts}')
                model.addConstr(coeff * x[uid, ts] - rhs <= slack, name=f'zeta_surr_{uid}_{ts}')
                surr_slacks.append(slack)

        # --- per-generator V3 代理软约束（M-penalty）---
        for g in range(ng):
            if g not in self.trainers:
                continue
            trainer = self.trainers[g]
            alphas = trainer.alpha_values[sample_id]
            betas = trainer.beta_values[sample_id]
            gammas = trainer.gamma_values[sample_id]
            deltas = trainer.delta_values[sample_id]
            alphas, betas, gammas, deltas = trainer._apply_surrogate_direction_to_params(
                alphas, betas, gammas, deltas
            )
            constraint_timesteps = trainer.sensitive_timesteps[sample_id]
            constraint_offsets = resolve_constraint_offsets_from_trainer(
                trainer,
                sample_id,
                len(constraint_timesteps),
            )
            x_vars = {t: x[g, t] for t in range(T)}
            for k in range(len(alphas)):
                if k >= len(constraint_timesteps):
                    break
                t_k = constraint_timesteps[k]
                a_v = float(alphas[k])
                b_v = float(betas[k])
                c_v = float(gammas[k])
                r_v = float(deltas[k])
                if abs(a_v) > 1e-10 or abs(b_v) > 1e-10 or abs(c_v) > 1e-10:
                    slack_k = model.addVar(lb=0, name=f'g{g}_surr_slack_{k}')
                    model.addConstr(
                        build_surrogate_constraint_expression(
                            x_vars,
                            t_k,
                            constraint_offsets[k],
                            a_v,
                            b_v,
                            c_v,
                            T,
                        ) - r_v <= slack_k,
                        name=f'g{g}_surr_{k}'
                    )
                    surr_slacks.append(slack_k)

        # --- 目标：发电成本 + 启停成本 + M × Σ 代理约束违背量 ---
        obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
               + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
        if surr_slacks:
            obj += M_SURR * gp.quicksum(surr_slacks)
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(T)] for g in range(ng)])
            x_sol = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
            self.pg[sample_id] = pg_sol
            self.x[sample_id] = x_sol
            # 同步到 agent 和各 trainer
            self.agent.pg[sample_id] = pg_sol
            self.agent.x[sample_id] = x_sol
            for g in range(ng):
                if g in self.trainers:
                    self.trainers[g].x[sample_id] = x_sol[g]
                    self.trainers[g].pg[sample_id] = pg_sol[g]
            return pg_sol, x_sol
        else:
            print(f"  [Unified] 原始块求解失败 sample={sample_id}, status={model.status}", flush=True)
            return self.pg[sample_id].copy(), self.x[sample_id].copy()

    # =========================================================================
    # 核心方法：双路对偶更新
    # =========================================================================

    def iter_with_dual_block(self, sample_id: int) -> None:
        """
        固定当前 pg/x，对两套对偶变量分别更新（双路）。

        路径1（全局 theta/zeta）：
            调用 agent.iter_with_dual_block → 更新 agent.mu / agent.ita
        路径2（per-generator V3）：
            对每个 g 同步 x，再调用 trainers[g].iter_with_dual_block
            → 更新 trainers[g].mu

        Args:
            sample_id: 样本索引
        """
        ua = self.agent._current_union_analysis

        # --- 路径 1：全局 theta/zeta 对偶 ---
        lambda_val = self._get_lambda_val(sample_id)
        theta_values, zeta_values = self.get_theta_zeta_params(
            normalize_sample_arrays(self.active_set_data[sample_id]), lambda_val
        )
        lambda_sol, mu_sol, ita_sol = self.agent.iter_with_dual_block(
            sample_id=sample_id,
            theta_values=theta_values,
            zeta_values=zeta_values,
            union_analysis=ua
        )
        if lambda_sol is not None:
            self.agent.lambda_[sample_id] = lambda_sol
        if mu_sol is not None:
            EPS = 1e-10
            self.agent.mu[sample_id] = np.where(np.abs(mu_sol) < EPS, 0, mu_sol)
        if ita_sol is not None:
            EPS = 1e-10
            self.agent.ita[sample_id] = np.where(np.abs(ita_sol) < EPS, 0, ita_sol)

        # --- 路径 2：per-generator V3 代理对偶 ---
        for g in range(self.ng):
            if g not in self.trainers:
                continue
            trainer = self.trainers[g]
            # 同步 x 到 trainer
            trainer.x[sample_id] = self.x[sample_id, g, :]
            trainer.pg[sample_id] = self.pg[sample_id, g, :]

            alphas = trainer.alpha_values[sample_id]
            betas = trainer.beta_values[sample_id]
            gammas = trainer.gamma_values[sample_id]
            deltas = trainer.delta_values[sample_id]
            costs = trainer.cost_values[sample_id]

            try:
                result = trainer.iter_with_dual_block(
                    sample_id, alphas, betas, gammas, deltas, costs
                )
                if result is not None:
                    _, mu_v3 = result
                    if mu_v3 is not None:
                        EPS = 1e-10
                        trainer.mu[sample_id] = np.where(np.abs(mu_v3) < EPS, 0, mu_v3)
            except Exception as e:
                print(f"  [Unified] 机组 {g} 对偶块失败 sample={sample_id}: {e}", flush=True)

    # =========================================================================
    # 主迭代循环（不含 NN 更新）
    # =========================================================================

    def run(self, max_iter: int = 20) -> None:
        """
        联合对偶迭代主循环（不包含 NN 参数更新）。

        每轮迭代：
          1. 对所有样本执行原始块（全局 LP，更新 pg/x）
          2. 对所有样本执行双路对偶块（更新 agent.mu/ita 和 trainers[g].mu）
          3. 更新惩罚参数 rho_primal/dual/opt

        Args:
            max_iter: 最大迭代轮数
        """
        for i in range(max_iter):
            print(f"[Unified] 迭代 {i+1}/{max_iter}", flush=True)
            self.agent.iter_number = i

            # 原始块
            for s in range(self.n_samples):
                self.iter_with_primal_block(s)

            # 双路对偶块
            for s in range(self.n_samples):
                self.iter_with_dual_block(s)

            # 更新惩罚参数
            self._update_penalty_params()
            print(
                f"  ρ_primal={self.rho_primal:.4f}, "
                f"ρ_dual={self.rho_dual:.4f}, "
                f"ρ_opt={self.rho_opt:.4f}",
                flush=True
            )

    def _update_penalty_params(self) -> None:
        """计算违反量并累加更新惩罚参数。"""
        try:
            obj_primal, obj_dual, obj_opt = self.agent.cal_viol(
                union_analysis=self.agent._current_union_analysis
            )
        except Exception as e:
            print(f"  [Unified] cal_viol 失败: {e}，跳过惩罚参数更新", flush=True)
            return
        obj_primal = obj_primal if abs(obj_primal) >= 1e-12 else 0.0
        obj_dual = obj_dual if abs(obj_dual) >= 1e-12 else 0.0
        obj_opt = obj_opt if abs(obj_opt) >= 1e-12 else 0.0

        self.rho_primal += self.gamma * obj_primal
        self.rho_dual += self.gamma * obj_dual
        self.rho_opt += self.gamma * obj_opt

        # 同步到 agent
        self.agent.rho_primal = self.rho_primal
        self.agent.rho_dual = self.rho_dual
        self.agent.rho_opt = self.rho_opt

    # =========================================================================
    # 推理接口（供 feasibility_pump 调用）
    # =========================================================================

    def solve_global(
        self,
        pd_data: np.ndarray,
        lambda_val: np.ndarray,
        sparse_library: Optional[SparseSurrogateLibrary] = None,
        sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    ) -> np.ndarray:
        """
        给定新负荷，构建包含两类约束的全局 LP 松弛并求解。

        Args:
            pd_data: (nb_load, T) 负荷数据
            lambda_val: (T,) 功率平衡对偶变量
            sparse_library: （可选）覆盖 manager 默认库的稀疏参数化约束库
            sparse_x_template_library: （可选）覆盖 manager 默认库的 x 稀疏支持集模板库

        Returns:
            x_LP: (ng, T) LP 松弛解；不可行时返回零矩阵
        """
        theta_values, zeta_values = self.get_theta_zeta_params(pd_data, lambda_val)
        surrogate_params = {
            g: self.get_surrogate_params(g, pd_data, lambda_val)
            for g in range(self.ng) if g in self.trainers
        }

        ppc_int = self._ppc_int
        gen = self._gen
        gencost = self._gencost
        ng, T = self.ng, self.T
        T_delta = self.T_delta
        Pd_sum = np.sum(pd_data, axis=0)

        Ru = 0.4 * gen[:, PMAX] / T_delta
        Rd = 0.4 * gen[:, PMAX] / T_delta
        Ru_co = 0.3 * gen[:, PMAX]
        Rd_co = 0.3 * gen[:, PMAX]
        Ton = min(int(4 * T_delta), T - 1)
        Toff = min(int(4 * T_delta), T - 1)
        start_cost = gencost[:, 1]
        shut_cost = gencost[:, 2]
        M_SURR = 1e5

        model = gp.Model('unified_global_LP')
        model.Params.OutputFlag = 0

        pg = model.addVars(ng, T, lb=0, name='pg')
        x = model.addVars(ng, T, lb=0, ub=1, name='x')
        cpower = model.addVars(ng, T, lb=0, name='cpower')
        coc = model.addVars(ng, T - 1, lb=0, name='coc')
        surr_slacks: list = []

        # 功率平衡
        for t in range(T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
                name=f'pb_{t}'
            )

        for g in range(ng):
            for t in range(T):
                model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

            for t in range(1, T):
                model.addConstr(pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1]))
                model.addConstr(pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t]))

            for tau in range(1, Ton + 1):
                for t1 in range(T - tau):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
            for tau in range(1, Toff + 1):
                for t1 in range(T - tau):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

            for t in range(1, T):
                model.addConstr(coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]))
                model.addConstr(coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]))

            for t in range(T):
                model.addConstr(
                    cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                                   + gencost[g, -1] / T_delta * x[g, t]
                )

        # theta 软约束
        ua = self.agent._current_union_analysis or {}
        for ci in ua.get('union_constraints', []):
            bid = ci['branch_id']
            ts = ci['time_slot']
            lhs = gp.LinExpr()
            for ci2 in ci.get('nonzero_pg_coefficients', []):
                uid = ci2['unit_id']
                tname = f'theta_branch_{bid}_unit_{uid}_time_{ts}'
                coeff = float(theta_values.get(tname, 0.0))
                if abs(coeff) > 1e-10 and uid < ng and ts < T:
                    lhs += coeff * x[uid, ts]
            rhs_name = f'theta_rhs_branch_{bid}_time_{ts}'
            rhs = float(theta_values.get(rhs_name, 1.0))
            slack = model.addVar(lb=0, name=f'theta_slack_{bid}_{ts}')
            model.addConstr(lhs - rhs <= slack)
            surr_slacks.append(slack)

        # zeta 软约束
        for zc in ua.get('union_zeta_constraints', []):
            uid = zc['unit_id']
            ts = zc['time_slot']
            zname = f'zeta_unit_{uid}_time_{ts}'
            coeff = float(zeta_values.get(zname, 0.0))
            rhs_name = f'zeta_rhs_unit_{uid}_time_{ts}'
            rhs = float(zeta_values.get(rhs_name, 1.0))
            if abs(coeff) > 1e-10 and uid < ng and ts < T:
                slack = model.addVar(lb=0, name=f'zeta_slack_{uid}_{ts}')
                model.addConstr(coeff * x[uid, ts] - rhs <= slack)
                surr_slacks.append(slack)

        # per-generator V3 代理软约束
        for g in range(ng):
            if g not in surrogate_params:
                continue
            alphas, betas, gammas, deltas, *_ = surrogate_params[g]
            trainer = self.trainers.get(g)
            constraint_timesteps = trainer.sensitive_timesteps[sample_id] if trainer is not None else []
            constraint_offsets = (
                resolve_constraint_offsets_from_trainer(
                    trainer,
                    sample_id,
                    len(constraint_timesteps),
                )
                if trainer is not None else []
            )
            x_vars = {t: x[g, t] for t in range(T)}
            for k in range(len(alphas)):
                if k >= len(constraint_timesteps):
                    break
                t_k = constraint_timesteps[k]
                a_v = float(alphas[k])
                b_v = float(betas[k])
                c_v = float(gammas[k])
                r_v = float(deltas[k])
                if abs(a_v) > 1e-10 or abs(b_v) > 1e-10 or abs(c_v) > 1e-10:
                    slack_k = model.addVar(lb=0, name=f'g{g}_surr_slack_{k}')
                    model.addConstr(
                        build_surrogate_constraint_expression(
                            x_vars,
                            t_k,
                            constraint_offsets[k],
                            a_v,
                            b_v,
                            c_v,
                            T,
                        ) - r_v <= slack_k,
                        name=f'g{g}_surr_{k}'
                    )
                    surr_slacks.append(slack_k)

        add_sparse_parameterized_constraints(
            model,
            x,
            pg,
            pd_data,
            sparse_library=sparse_library or self.sparse_library,
            surr_slacks=surr_slacks,
            name_prefix="sparse",
        )
        add_sparse_x_templates_to_model(
            model,
            x,
            template_library=sparse_x_template_library or self.sparse_x_template_library,
            surr_slacks=surr_slacks,
            name_prefix="sparse_x",
        )

        # DC 线路潮流约束（硬约束）
        try:
            _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
            ptdf_Pd = _PTDF @ pd_data
            for l in active_lines:
                limit = float(branch_limit[l])
                for t in range(T):
                    flow_expr = (
                        gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                        - float(ptdf_Pd[l, t])
                    )
                    model.addConstr(flow_expr <= limit)
                    model.addConstr(flow_expr >= -limit)
        except Exception as e:
            print(f"  [Unified] DC 潮流约束构建失败（{e}），跳过", flush=True)

        obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
               + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
        if surr_slacks:
            obj += M_SURR * gp.quicksum(surr_slacks)
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return np.array([[x[g, t].X for t in range(T)] for g in range(ng)])

        print(f"  [Unified] 全局 LP 求解失败 (status={model.status})，返回零矩阵", flush=True)
        return np.zeros((ng, T))

    def get_surrogate_params(
        self, g: int, pd_data: np.ndarray | dict, lambda_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取机组 g 的 V3 代理约束参数，委托给 trainers[g].get_surrogate_params。

        Args:
            g: 机组索引
            pd_data: (nb_load, T) 负荷数据
            lambda_val: (T,) 功率平衡对偶变量

        Returns:
            (alphas, betas, gammas, deltas, costs)，其中前四项 shape 为
            (max_constraints,)，costs shape 为 (T,)

        Raises:
            KeyError: 当 g 不在 trainers 中时
        """
        return self.trainers[g].get_surrogate_params(pd_data, lambda_val)

    def get_theta_zeta_params(
        self, pd_data: np.ndarray | dict, lambda_val: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        通过 agent.theta_net / agent.zeta_net 前向推断，返回 (theta_values, zeta_values)。

        当 PyTorch 不可用或 NN 未初始化时，返回 agent 当前存储的静态值。

        Args:
            pd_data: (nb_load, T) 负荷数据（用于提取特征 = pd_data.flatten()）
            lambda_val: (T,) 功率平衡对偶变量（当前实现不使用，保留接口一致性）

        Returns:
            theta_values: {name: float} theta/theta_rhs 参数字典
            zeta_values: {name: float} zeta/zeta_rhs 参数字典
        """
        agent = self.agent
        if not TORCH_AVAILABLE or agent.theta_net is None or agent.zeta_net is None:
            return agent.theta_values.copy(), agent.zeta_values.copy()

        try:
            import torch as _torch
            if isinstance(pd_data, dict):
                features = get_feature_vector_from_sample(pd_data)
            else:
                features = pd_data.flatten()
            features_tensor = _torch.tensor(features, dtype=_torch.float32).unsqueeze(0)
            if agent.device is not None:
                features_tensor = features_tensor.to(agent.device)

            agent.theta_net.eval()
            agent.zeta_net.eval()
            with _torch.no_grad():
                theta_out = agent.theta_net(features_tensor)
                zeta_out = agent.zeta_net(features_tensor)

            theta_values = agent._tensor_to_theta_dict(theta_out[0])
            zeta_values = agent._tensor_to_zeta_dict(zeta_out[0])
            return theta_values, zeta_values
        except Exception as e:
            print(f"  [Unified] NN 推断失败（{e}），使用静态 theta/zeta", flush=True)
            return agent.theta_values.copy(), agent.zeta_values.copy()

    # =========================================================================
    # 内部辅助
    # =========================================================================

    def _get_lambda_val(self, sample_id: int) -> np.ndarray:
        """获取样本的功率平衡对偶变量（lambda），优先从 active_set_data 读取。"""
        sample = self.active_set_data[sample_id]
        if 'lambda' in sample:
            lam = np.array(sample['lambda'])
            if lam.ndim == 0 or lam.size == 0:
                return np.zeros(self.T)
            return lam.flatten()[:self.T]
        # 从 agent 的已知对偶变量中取功率平衡分量
        try:
            return self.agent.lambda_[sample_id]['lambda_power_balance']
        except Exception:
            return np.zeros(self.T)
