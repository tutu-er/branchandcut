#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合BCD训练器：在Agent_NN_BCD的theta/zeta约束和SubproblemSurrogateTrainer的
surrogate约束同时存在的BCD迭代上求解。

BCD迭代流程：pg块→dual块→NN反传→cal_viol，所有约束用软约束形式。
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A
from pypower.makePTDF import makePTDF

from src.uc_NN_BCD import _get_uc_matrix_from_sample


class JointLPTrainer:
    """联合BCD训练器：Agent theta/zeta + Subproblem surrogate 约束。

    BCD循环：pg块→dual块→theta/zeta NN→surrogate NN→cal_viol→rho更新。

    Args:
        agent: 已初始化的Agent_NN_BCD实例（含theta_net, zeta_net）
        trainers: {unit_id: SubproblemSurrogateTrainer} 字典
    """

    def __init__(self, agent, trainers: Dict[int, object]):
        self.agent = agent
        self.trainers = trainers

        # 从 agent 复制系统参数引用
        self.gen = agent.gen
        self.bus = agent.bus
        self.branch = agent.branch
        self.gencost = agent.gencost
        self.baseMVA = agent.baseMVA
        self.T = agent.T
        self.T_delta = agent.T_delta
        self.ng = agent.ng
        self.nl = self.branch.shape[0]
        self.n_samples = agent.n_samples
        self.active_set_data = agent.active_set_data
        self.device = agent.device if hasattr(agent, 'device') else torch.device('cpu')

        # 预计算 PTDF 和 G 矩阵
        nb = self.bus.shape[0]
        self.G = np.zeros((nb, self.ng))
        for g in range(self.ng):
            bus_idx = int(self.gen[g, GEN_BUS])
            if 0 <= bus_idx < nb:
                self.G[bus_idx, g] = 1
        self.PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
        self.branch_limit = self.branch[:, RATE_A]

        # surrogate 对偶变量: {sample_id: {unit_id: ndarray(nc,)}}
        self.lambda_surr: Dict[int, Dict[int, np.ndarray]] = {}
        for s in range(self.n_samples):
            self.lambda_surr[s] = {
                g: np.ones(trainer.num_coupling_constraints) * 0.1
                for g, trainer in self.trainers.items()
            }

    # ------------------------------------------------------------------ #
    #  PG块：更新 x, pg, cpower, coc
    # ------------------------------------------------------------------ #
    def iter_with_pg_block(self, sample_id: int) -> Optional[Tuple]:
        """PG块迭代，复用agent模板 + surrogate罚项。

        Returns:
            (pg_sol, x_sol, cpower_sol, coc_sol) 或 None
        """
        agent = self.agent
        Pd = self.active_set_data[sample_id]['pd_data']
        union_analysis = getattr(agent, '_current_union_analysis', None)
        theta_values = agent.theta_values_list[sample_id]
        zeta_values = agent.zeta_values_list[sample_id]

        model = gp.Model('joint_pg_block')
        model.Params.OutputFlag = 0

        # 主变量
        pg = model.addVars(self.ng, self.T, lb=0, name='pg')
        x = model.addVars(self.ng, self.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        coc = model.addVars(self.ng, self.T - 1, lb=0, name='coc')
        cpower = model.addVars(self.ng, self.T, lb=0, name='cpower')

        # 违反量辅助变量
        power_balance_viol = model.addVars(self.T, lb=0, name='power_balance_viol')
        pg_lower_viol = model.addVars(self.ng, self.T, lb=0, name='pg_lower_viol')
        pg_upper_viol = model.addVars(self.ng, self.T, lb=0, name='pg_upper_viol')
        pg_lower_abs = model.addVars(self.ng, self.T, lb=0, name='pg_lower_abs')
        pg_upper_abs = model.addVars(self.ng, self.T, lb=0, name='pg_upper_abs')
        ramp_up_viol = model.addVars(self.ng, self.T - 1, lb=0, name='ramp_up_viol')
        ramp_down_viol = model.addVars(self.ng, self.T - 1, lb=0, name='ramp_down_viol')
        ramp_up_abs = model.addVars(self.ng, self.T - 1, lb=0, name='ramp_up_abs')
        ramp_down_abs = model.addVars(self.ng, self.T - 1, lb=0, name='ramp_down_abs')

        Ton = min(4, self.T)
        Toff = min(4, self.T)
        min_on_viol = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_viol')
        min_off_viol = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_viol')
        min_on_abs = model.addVars(self.ng, Ton, self.T, lb=0, name='min_on_abs')
        min_off_abs = model.addVars(self.ng, Toff, self.T, lb=0, name='min_off_abs')
        start_cost_viol = model.addVars(self.ng, self.T, lb=0, name='start_cost_viol')
        shut_cost_viol = model.addVars(self.ng, self.T, lb=0, name='shut_cost_viol')
        start_cost_abs = model.addVars(self.ng, self.T, lb=0, name='start_cost_abs')
        shut_cost_abs = model.addVars(self.ng, self.T, lb=0, name='shut_cost_abs')
        dcpf_upper_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_viol')
        dcpf_upper_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_upper_abs')
        dcpf_lower_viol = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_viol')
        dcpf_lower_abs = model.addVars(self.nl, self.T, lb=0, name='dcpf_lower_abs')
        x_binary_dev = model.addVars(self.ng, self.T, lb=0, name='x_binary_dev')

        obj_primal = 0
        obj_opt = 0
        obj_binary = 0

        # --- 功率平衡约束 ---
        for t in range(self.T):
            power_balance_expr = gp.quicksum(pg[g, t] for g in range(self.ng)) - np.sum(Pd[:, t])
            model.addConstr(power_balance_viol[t] >= power_balance_expr)
            model.addConstr(power_balance_viol[t] >= -power_balance_expr)
            obj_primal += power_balance_viol[t]
            obj_opt += power_balance_viol[t] * abs(agent.lambda_[sample_id]['lambda_power_balance'][t])

            for g in range(self.ng):
                pg_lower_expr = self.gen[g, PMIN] * x[g, t] - pg[g, t]
                model.addConstr(pg_lower_viol[g, t] >= pg_lower_expr)
                obj_primal += pg_lower_viol[g, t]

                pg_upper_expr = pg[g, t] - self.gen[g, PMAX] * x[g, t]
                model.addConstr(pg_upper_viol[g, t] >= pg_upper_expr)
                obj_primal += pg_upper_viol[g, t]

                model.addConstr(pg_lower_abs[g, t] >= pg_lower_expr)
                model.addConstr(pg_lower_abs[g, t] >= -pg_lower_expr)
                model.addConstr(pg_upper_abs[g, t] >= pg_upper_expr)
                model.addConstr(pg_upper_abs[g, t] >= -pg_upper_expr)

                obj_opt += pg_lower_abs[g, t] * abs(agent.lambda_[sample_id]['lambda_pg_lower'][g, t])
                obj_opt += pg_upper_abs[g, t] * abs(agent.lambda_[sample_id]['lambda_pg_upper'][g, t])
                obj_opt += x[g, t] * abs(agent.lambda_[sample_id]['lambda_x_lower'][g, t])
                obj_opt += (1 - x[g, t]) * abs(agent.lambda_[sample_id]['lambda_x_upper'][g, t])

        # --- 爬坡约束 ---
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]

        for t in range(1, self.T):
            for g in range(self.ng):
                ramp_up_expr = pg[g, t] - pg[g, t - 1] - Ru[g] * x[g, t - 1] - Ru_co[g] * (1 - x[g, t - 1])
                model.addConstr(ramp_up_viol[g, t - 1] >= ramp_up_expr)
                obj_primal += ramp_up_viol[g, t - 1]
                model.addConstr(ramp_up_abs[g, t - 1] >= ramp_up_expr)
                model.addConstr(ramp_up_abs[g, t - 1] >= -ramp_up_expr)

                ramp_down_expr = pg[g, t - 1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t])
                model.addConstr(ramp_down_viol[g, t - 1] >= ramp_down_expr)
                obj_primal += ramp_down_viol[g, t - 1]
                model.addConstr(ramp_down_abs[g, t - 1] >= ramp_down_expr)
                model.addConstr(ramp_down_abs[g, t - 1] >= -ramp_down_expr)

                obj_opt += ramp_up_abs[g, t - 1] * abs(agent.lambda_[sample_id]['lambda_ramp_up'][g, t - 1])
                obj_opt += ramp_down_abs[g, t - 1] * abs(agent.lambda_[sample_id]['lambda_ramp_down'][g, t - 1])

        # --- 最小开关机时间约束 ---
        for g in range(self.ng):
            for t in range(1, Ton + 1):
                for t1 in range(self.T - t):
                    min_on_expr = x[g, t1 + 1] - x[g, t1] - x[g, t1 + t]
                    model.addConstr(min_on_viol[g, t - 1, t1] >= min_on_expr)
                    model.addConstr(min_on_abs[g, t - 1, t1] >= min_on_expr)
                    model.addConstr(min_on_abs[g, t - 1, t1] >= -min_on_expr)
                    obj_primal += min_on_viol[g, t - 1, t1]
                    obj_opt += min_on_abs[g, t - 1, t1] * abs(agent.lambda_[sample_id]['lambda_min_on'][g, t - 1, t1])

        for g in range(self.ng):
            for t in range(1, Toff + 1):
                for t1 in range(self.T - t):
                    min_off_expr = -x[g, t1 + 1] + x[g, t1] - (1 - x[g, t1 + t])
                    model.addConstr(min_off_viol[g, t - 1, t1] >= min_off_expr)
                    model.addConstr(min_off_abs[g, t - 1, t1] >= min_off_expr)
                    model.addConstr(min_off_abs[g, t - 1, t1] >= -min_off_expr)
                    obj_primal += min_off_viol[g, t - 1, t1]
                    obj_opt += min_off_abs[g, t - 1, t1] * abs(agent.lambda_[sample_id]['lambda_min_off'][g, t - 1, t1])

        # --- 启停成本 ---
        start_cost_coeff = self.gencost[:, 1]
        shut_cost_coeff = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                sc_expr = start_cost_coeff[g] * (x[g, t] - x[g, t - 1]) - coc[g, t - 1]
                model.addConstr(start_cost_viol[g, t - 1] >= sc_expr)
                model.addConstr(start_cost_abs[g, t - 1] >= sc_expr)
                model.addConstr(start_cost_abs[g, t - 1] >= -sc_expr)
                obj_primal += start_cost_viol[g, t - 1]
                obj_opt += start_cost_abs[g, t - 1] * abs(agent.lambda_[sample_id]['lambda_start_cost'][g, t - 1])

                shc_expr = shut_cost_coeff[g] * (x[g, t - 1] - x[g, t]) - coc[g, t - 1]
                model.addConstr(shut_cost_viol[g, t - 1] >= shc_expr)
                model.addConstr(shut_cost_abs[g, t - 1] >= shc_expr)
                model.addConstr(shut_cost_abs[g, t - 1] >= -shc_expr)
                obj_primal += shut_cost_viol[g, t - 1]
                obj_opt += shut_cost_abs[g, t - 1] * abs(agent.lambda_[sample_id]['lambda_shut_cost'][g, t - 1])

                obj_opt += coc[g, t - 1] * abs(agent.lambda_[sample_id]['lambda_coc_nonneg'][g, t - 1])

        # --- 发电成本（等式约束） ---
        for t in range(self.T):
            for g in range(self.ng):
                model.addConstr(
                    cpower[g, t] == self.gencost[g, -2] / self.T_delta * pg[g, t]
                    + self.gencost[g, -1] / self.T_delta * x[g, t])

        # --- 潮流约束 ---
        PTDF = self.PTDF
        G = self.G
        branch_limit = self.branch_limit
        for t in range(self.T):
            flow = PTDF @ (G @ np.array([pg[g, t] for g in range(self.ng)]) - Pd[:, t])
            for l in range(self.nl):
                model.addConstr(dcpf_upper_viol[l, t] >= flow[l] - branch_limit[l])
                model.addConstr(dcpf_lower_viol[l, t] >= -flow[l] - branch_limit[l])
                model.addConstr(dcpf_upper_abs[l, t] >= flow[l] - branch_limit[l])
                model.addConstr(dcpf_upper_abs[l, t] >= -flow[l] + branch_limit[l])
                model.addConstr(dcpf_lower_abs[l, t] >= -flow[l] - branch_limit[l])
                model.addConstr(dcpf_lower_abs[l, t] >= flow[l] + branch_limit[l])
                obj_primal += dcpf_upper_viol[l, t] + dcpf_lower_viol[l, t]
                obj_opt += dcpf_upper_abs[l, t] * abs(agent.lambda_[sample_id]['lambda_dcpf_upper'][l, t])
                obj_opt += dcpf_lower_abs[l, t] * abs(agent.lambda_[sample_id]['lambda_dcpf_lower'][l, t])

        # --- 二进制变量偏差 ---
        unit_commitment_matrix = _get_uc_matrix_from_sample(
            self.active_set_data[sample_id], self.ng, self.T)
        if unit_commitment_matrix is None:
            unit_commitment_matrix = agent.x[sample_id]

        for g in range(self.ng):
            for t in range(self.T):
                target_value = unit_commitment_matrix[g, t]
                x_dev_expr = x[g, t] - target_value
                model.addConstr(x_binary_dev[g, t] >= x_dev_expr)
                model.addConstr(x_binary_dev[g, t] >= -x_dev_expr)
                obj_binary += x_binary_dev[g, t]

        # --- theta/zeta 罚项（委托agent方法） ---
        if (agent.enable_theta_constraints and union_analysis
                and 'union_constraints' in union_analysis and theta_values is not None):
            model, para_obj_primal, para_obj_opt = agent._add_parametric_penalties_pg_block(
                model, x, sample_id, theta_values, union_analysis,
                PTDF=PTDF, branch_limit=branch_limit)
            obj_primal += para_obj_primal
            obj_opt += para_obj_opt

        if (agent.enable_zeta_constraints and union_analysis
                and 'union_zeta_constraints' in union_analysis and zeta_values is not None):
            model, para_obj_primal, para_obj_opt = agent._add_parametric_balance_power_penalties_pg_block(
                model, x, sample_id, zeta_values, union_analysis)
            obj_primal += para_obj_primal
            obj_opt += para_obj_opt

        # --- surrogate 罚项（新增） ---
        surr_params = self._get_surrogate_params(sample_id)
        for g, trainer in self.trainers.items():
            a, b, c, d = surr_params[g]
            sensitive_t = trainer.sensitive_timesteps[sample_id]
            for k, t_k in enumerate(sensitive_t):
                if t_k + 2 >= self.T:
                    continue
                surr_expr = (float(a[k]) * x[g, t_k]
                             + float(b[k]) * x[g, t_k + 1]
                             + float(c[k]) * x[g, t_k + 2]
                             - float(d[k]))
                # primal: 单侧违反
                viol = model.addVar(lb=0, name=f'surr_viol_{g}_{k}')
                model.addConstr(viol >= surr_expr)
                obj_primal += viol
                # opt: 双侧 abs * |lambda_surr|
                abs_v = model.addVar(lb=0, name=f'surr_abs_{g}_{k}')
                model.addConstr(abs_v >= surr_expr)
                model.addConstr(abs_v >= -surr_expr)
                obj_opt += abs_v * abs(float(self.lambda_surr[sample_id][g][k]))

        # --- 目标函数 ---
        total_objective = obj_binary + agent.rho_primal * obj_primal + agent.rho_opt * obj_opt
        model.setObjective(total_objective, GRB.MINIMIZE)
        model.Params.MIPGap = 1e-6
        model.optimize()

        if model.status == GRB.OPTIMAL:
            pg_sol = np.array([[pg[g, t].X for t in range(self.T)] for g in range(self.ng)])
            x_sol = np.array([[x[g, t].X for t in range(self.T)] for g in range(self.ng)])
            cpower_sol = np.array([[cpower[g, t].X for t in range(self.T)] for g in range(self.ng)])
            coc_sol = np.array([[coc[g, t].X for t in range(self.T - 1)] for g in range(self.ng)])

            if sample_id <= 2:
                print(f"  pg_block s={sample_id}: "
                      f"obj_primal={obj_primal.getValue():.4f}, "
                      f"obj_opt={obj_opt.getValue():.4f}, "
                      f"obj_binary={obj_binary.getValue():.4f}", flush=True)
            return pg_sol, x_sol, cpower_sol, coc_sol
        else:
            print(f"  PG块求解失败 sample_id={sample_id}, status={model.status}", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  Dual块：更新 lambda, mu, ita, lambda_surr
    # ------------------------------------------------------------------ #
    def iter_with_dual_block(self, sample_id: int) -> Optional[Tuple]:
        """Dual块迭代，复用agent模板 + surrogate对偶变量。

        Returns:
            (lambda_sol, mu_sol, ita_sol, lambda_surr_sol) 或 None
        """
        agent = self.agent
        Pd = self.active_set_data[sample_id]['pd_data']
        union_analysis = getattr(agent, '_current_union_analysis', None)
        theta_values = agent.theta_values_list[sample_id]
        zeta_values = agent.zeta_values_list[sample_id]

        model = gp.Model('joint_dual_block')
        model.Params.OutputFlag = 0

        # --- 对偶变量 ---
        lambda_power_balance = model.addVars(self.T, lb=-GRB.INFINITY, name='lambda_power_balance')
        lambda_pg_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_lower')
        lambda_pg_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_pg_upper')
        lambda_ramp_up = model.addVars(self.ng, self.T - 1, lb=0, name='lambda_ramp_up')
        lambda_ramp_down = model.addVars(self.ng, self.T - 1, lb=0, name='lambda_ramp_down')

        Ton = min(4, self.T)
        Toff = min(4, self.T)
        lambda_min_on = model.addVars(self.ng, Ton, self.T, lb=0, name='lambda_min_on')
        lambda_min_off = model.addVars(self.ng, Toff, self.T, lb=0, name='lambda_min_off')
        lambda_start_cost = model.addVars(self.ng, self.T - 1, lb=0, name='lambda_start_cost')
        lambda_shut_cost = model.addVars(self.ng, self.T - 1, lb=0, name='lambda_shut_cost')
        lambda_coc_nonneg = model.addVars(self.ng, self.T - 1, lb=0, name='lambda_coc_nonneg')
        lambda_cpower = model.addVars(self.ng, self.T, lb=0, name='lambda_cpower')
        lambda_dcpf_upper = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_upper')
        lambda_dcpf_lower = model.addVars(self.nl, self.T, lb=0, name='lambda_dcpf_lower')
        lambda_x_upper = model.addVars(self.ng, self.T, lb=0, name='lambda_x_upper')
        lambda_x_lower = model.addVars(self.ng, self.T, lb=0, name='lambda_x_lower')

        # mu/ita 变量（dual_decay控制下界）
        dual_decay_round_ = getattr(agent, 'dual_decay_round', None)
        if dual_decay_round_ is None:
            dual_decay_round_ = getattr(agent, 'dual_para_bound_quit_iteration', 50)
        iter_number = getattr(agent, 'iter_number', 0)
        dual_para_bound = getattr(agent, 'dual_para_bound', 0.1)

        mu_lb = dual_para_bound if iter_number < dual_decay_round_ else 0
        ita_lb = dual_para_bound if iter_number < dual_decay_round_ else 0
        mu = model.addVars(self.nl, self.T, lb=mu_lb, name='mu')
        ita = model.addVars(self.ng, self.T, lb=ita_lb, name='ita')

        # surrogate 对偶变量
        lambda_surr_var: Dict[int, list] = {}
        surr_params = self._get_surrogate_params(sample_id)
        for g, trainer in self.trainers.items():
            sensitive_t = trainer.sensitive_timesteps[sample_id]
            nc = len(sensitive_t)
            lambda_surr_var[g] = [
                model.addVar(lb=0, name=f'lambda_surr_{g}_{k}')
                for k in range(nc)
            ]

        mu_max = model.addVar(lb=0, name='mu_max')
        ita_max = model.addVar(lb=0, name='ita_max')
        deadband = 100
        for l in range(self.nl):
            for t in range(self.T):
                model.addConstr(mu_max >= mu[l, t] - deadband)
                model.addConstr(mu_max >= -mu[l, t] - deadband)
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(ita_max >= ita[g, t] - deadband)
                model.addConstr(ita_max >= -ita[g, t] - deadband)
        penalty_factor = 0
        penal_mu = penalty_factor * mu_max
        penal_ita = penalty_factor * ita_max

        PTDF = self.PTDF
        G_mat = self.G
        branch_limit = self.branch_limit
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        start_cost_coeff = self.gencost[:, 1]
        shut_cost_coeff = self.gencost[:, 2]

        obj_dual = 0
        obj_opt = 0

        # --- pg变量的对偶约束 ---
        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self.gencost[g, -2] / self.T_delta
                dual_expr -= lambda_power_balance[t]
                dual_expr -= lambda_pg_lower[g, t]
                dual_expr += lambda_pg_upper[g, t]

                if t > 0:
                    dual_expr += lambda_ramp_up[g, t - 1]
                    dual_expr -= lambda_ramp_down[g, t - 1]
                if t < self.T - 1:
                    dual_expr -= lambda_ramp_up[g, t]
                    dual_expr += lambda_ramp_down[g, t]

                ptdfg_col = (PTDF @ G_mat[:, g]).T
                for l in range(self.nl):
                    pg_coeff = ptdfg_col[l]
                    dual_expr += pg_coeff * (lambda_dcpf_upper[l, t] - lambda_dcpf_lower[l, t])

                dual_expr_pg_abs = model.addVar(lb=0, name=f'dual_abs_pg_{g}_{t}')
                model.addConstr(dual_expr_pg_abs >= dual_expr)
                model.addConstr(dual_expr_pg_abs >= -dual_expr)
                obj_dual += dual_expr_pg_abs

        # --- x变量的对偶约束 ---
        # 预构建surrogate对x[g,t]的贡献查找表
        surr_x_contrib: Dict[Tuple[int, int], list] = {}
        for g_s, trainer in self.trainers.items():
            a, b, c, d = surr_params[g_s]
            sensitive_t = trainer.sensitive_timesteps[sample_id]
            for k, t_k in enumerate(sensitive_t):
                if t_k + 2 >= self.T:
                    continue
                # a[k]*x[g_s, t_k]
                surr_x_contrib.setdefault((g_s, t_k), []).append(
                    (float(a[k]), lambda_surr_var[g_s][k]))
                # b[k]*x[g_s, t_k+1]
                surr_x_contrib.setdefault((g_s, t_k + 1), []).append(
                    (float(b[k]), lambda_surr_var[g_s][k]))
                # c[k]*x[g_s, t_k+2]
                surr_x_contrib.setdefault((g_s, t_k + 2), []).append(
                    (float(c[k]), lambda_surr_var[g_s][k]))

        for g in range(self.ng):
            for t in range(self.T):
                dual_expr = self.gencost[g, -1] / self.T_delta
                dual_expr += lambda_x_upper[g, t] - lambda_x_lower[g, t]
                dual_expr += self.gen[g, PMIN] * lambda_pg_lower[g, t]
                dual_expr -= self.gen[g, PMAX] * lambda_pg_upper[g, t]

                if t > 0:
                    dual_expr += (Rd_co[g] - Rd[g]) * lambda_ramp_down[g, t - 1]
                if t < self.T - 1:
                    dual_expr += (Ru_co[g] - Ru[g]) * lambda_ramp_up[g, t]

                for tau in range(1, Ton + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr += lambda_min_on[g, tau - 1, t1]
                        if t == t1:
                            dual_expr -= lambda_min_on[g, tau - 1, t1]
                        if t == t1 + tau:
                            dual_expr -= lambda_min_on[g, tau - 1, t1]

                for tau in range(1, Toff + 1):
                    for t1 in range(self.T - tau):
                        if t == t1 + 1:
                            dual_expr -= lambda_min_off[g, tau - 1, t1]
                        if t == t1:
                            dual_expr += lambda_min_off[g, tau - 1, t1]
                        if t == t1 + tau:
                            dual_expr += lambda_min_off[g, tau - 1, t1]

                if t > 0:
                    dual_expr += start_cost_coeff[g] * lambda_start_cost[g, t - 1]
                    dual_expr -= shut_cost_coeff[g] * lambda_shut_cost[g, t - 1]
                if t < self.T - 1:
                    dual_expr -= start_cost_coeff[g] * lambda_start_cost[g, t]
                    dual_expr += shut_cost_coeff[g] * lambda_shut_cost[g, t]

                # theta 对偶贡献（委托）
                if (agent.enable_theta_constraints and union_analysis
                        and 'union_constraints' in union_analysis):
                    dual_expr_para = agent._add_parametric_constraints_dual_block_const_to_model(
                        model, g, t, mu, sample_id, theta_values, union_analysis)
                    dual_expr += dual_expr_para

                # zeta 对偶贡献（委托）
                if (agent.enable_zeta_constraints and union_analysis
                        and 'union_zeta_constraints' in union_analysis):
                    dual_expr_para = agent._add_parametric_balance_power_constraints_dual_block_const_to_model(
                        model, g, t, ita, sample_id, zeta_values, union_analysis)
                    dual_expr += dual_expr_para

                # surrogate 对偶贡献
                for coeff, lam_var in surr_x_contrib.get((g, t), []):
                    dual_expr += coeff * lam_var

                dual_expr_x_abs = model.addVar(lb=0, name=f'dual_abs_x_{g}_{t}')
                model.addConstr(dual_expr_x_abs >= dual_expr)
                model.addConstr(dual_expr_x_abs >= -dual_expr)
                obj_dual += dual_expr_x_abs

        # --- cpower 对偶约束 ---
        for g in range(self.ng):
            for t in range(self.T):
                model.addConstr(lambda_cpower[g, t] == 1)

        # --- coc 对偶约束 ---
        for g in range(self.ng):
            for t in range(self.T - 1):
                dual_expr = 1
                dual_expr -= lambda_start_cost[g, t]
                dual_expr -= lambda_shut_cost[g, t]
                dual_expr -= lambda_coc_nonneg[g, t]

                dual_expr_coc_abs = model.addVar(lb=0, name=f'dual_abs_coc_{g}_{t}')
                model.addConstr(dual_expr_coc_abs >= dual_expr)
                model.addConstr(dual_expr_coc_abs >= -dual_expr)
                obj_dual += dual_expr_coc_abs

        # --- obj_opt: 原问题约束违反量 * 对偶变量 ---
        pg_arr = agent.pg[sample_id]
        x_arr = agent.x[sample_id]
        coc_arr = agent.coc[sample_id]

        for t in range(self.T):
            lambda_pb_abs = model.addVar(lb=0, name=f'lpb_abs_{t}')
            model.addConstr(lambda_pb_abs >= lambda_power_balance[t])
            model.addConstr(lambda_pb_abs >= -lambda_power_balance[t])
            pb_viol = abs(sum(pg_arr[g, t] for g in range(self.ng)) - np.sum(Pd[:, t]))
            if pb_viol > 1e-10:
                obj_opt += pb_viol * lambda_pb_abs

            for g in range(self.ng):
                pgl_viol = abs(pg_arr[g, t] - self.gen[g, PMIN] * x_arr[g, t])
                if pgl_viol > 1e-10:
                    obj_opt += pgl_viol * lambda_pg_lower[g, t]
                pgu_viol = abs(self.gen[g, PMAX] * x_arr[g, t] - pg_arr[g, t])
                if pgu_viol > 1e-10:
                    obj_opt += pgu_viol * lambda_pg_upper[g, t]

        for t in range(1, self.T):
            for g in range(self.ng):
                ru_viol = abs(pg_arr[g, t] - pg_arr[g, t - 1] - (Ru[g] * x_arr[g, t - 1] + Ru_co[g] * (1 - x_arr[g, t - 1])))
                if ru_viol > 1e-10:
                    obj_opt += ru_viol * lambda_ramp_up[g, t - 1]
                rd_viol = abs(pg_arr[g, t - 1] - pg_arr[g, t] - (Rd[g] * x_arr[g, t] + Rd_co[g] * (1 - x_arr[g, t])))
                if rd_viol > 1e-10:
                    obj_opt += rd_viol * lambda_ramp_down[g, t - 1]

        for g in range(self.ng):
            for t in range(1, Ton + 1):
                for t1 in range(self.T - t):
                    mon_viol = abs(x_arr[g, t1 + 1] - x_arr[g, t1] - x_arr[g, t1 + t])
                    if mon_viol > 1e-10:
                        obj_opt += mon_viol * lambda_min_on[g, t - 1, t1]

        for g in range(self.ng):
            for t in range(1, Toff + 1):
                for t1 in range(self.T - t):
                    moff_viol = abs(-x_arr[g, t1 + 1] + x_arr[g, t1] - 1 + x_arr[g, t1 + t])
                    if moff_viol > 1e-10:
                        obj_opt += moff_viol * lambda_min_off[g, t - 1, t1]

        for t in range(1, self.T):
            for g in range(self.ng):
                coc_v = abs(coc_arr[g, t - 1])
                if coc_v > 1e-10:
                    obj_opt += coc_v * lambda_coc_nonneg[g, t - 1]
                sc_v = abs(coc_arr[g, t - 1] - start_cost_coeff[g] * (x_arr[g, t] - x_arr[g, t - 1]))
                if sc_v > 1e-10:
                    obj_opt += sc_v * lambda_start_cost[g, t - 1]
                shc_v = abs(coc_arr[g, t - 1] - shut_cost_coeff[g] * (x_arr[g, t - 1] - x_arr[g, t]))
                if shc_v > 1e-10:
                    obj_opt += shc_v * lambda_shut_cost[g, t - 1]

        for t in range(self.T):
            flow = PTDF @ (G_mat @ pg_arr[:, t] - Pd[:, t])
            for l in range(self.nl):
                du_viol = abs(flow[l] - branch_limit[l])
                dl_viol = abs(flow[l] + branch_limit[l])
                if du_viol > 1e-10:
                    obj_opt += du_viol * lambda_dcpf_upper[l, t]
                if dl_viol > 1e-10:
                    obj_opt += dl_viol * lambda_dcpf_lower[l, t]

        for t in range(self.T):
            for g in range(self.ng):
                xl_viol = abs(x_arr[g, t])
                if xl_viol > 1e-10:
                    obj_opt += xl_viol * lambda_x_lower[g, t]
                xu_viol = abs(x_arr[g, t] - 1)
                if xu_viol > 1e-10:
                    obj_opt += xu_viol * lambda_x_upper[g, t]

        # theta/zeta obj_opt（委托）
        if (agent.enable_theta_constraints and union_analysis
                and 'union_constraints' in union_analysis and theta_values is not None):
            model, obj_opt_para = agent._add_parametric_obj_dual_block(
                model, x_arr, mu, sample_id, theta_values, union_analysis,
                PTDF=PTDF, branch_limit=branch_limit)
            obj_opt += obj_opt_para

        if (agent.enable_zeta_constraints and union_analysis
                and 'union_zeta_constraints' in union_analysis and zeta_values is not None):
            model, obj_opt_para = agent._add_parametric_balance_power_obj_dual_block(
                model, x_arr, ita, sample_id, zeta_values, union_analysis)
            obj_opt += obj_opt_para

        # surrogate obj_opt
        for g_s, trainer in self.trainers.items():
            a, b, c, d = surr_params[g_s]
            sensitive_t = trainer.sensitive_timesteps[sample_id]
            for k, t_k in enumerate(sensitive_t):
                if t_k + 2 >= self.T:
                    continue
                surr_val = (float(a[k]) * x_arr[g_s, t_k]
                            + float(b[k]) * x_arr[g_s, t_k + 1]
                            + float(c[k]) * x_arr[g_s, t_k + 2]
                            - float(d[k]))
                abs_surr_val = abs(surr_val)
                if abs_surr_val > 1e-10:
                    obj_opt += abs_surr_val * lambda_surr_var[g_s][k]

        # --- 目标函数 ---
        total_objective = agent.rho_dual * obj_dual + agent.rho_opt * obj_opt + penal_mu + penal_ita
        model.setObjective(total_objective, GRB.MINIMIZE)
        model.Params.MIPGap = 1e-6
        model.Params.Presolve = 0
        model.Params.NumericFocus = 2
        model.Params.ScaleFlag = 2
        model.optimize()

        if model.status == GRB.OPTIMAL:
            lambda_sol = {
                'lambda_power_balance': np.array([lambda_power_balance[t].X for t in range(self.T)]),
                'lambda_pg_lower': np.array([[lambda_pg_lower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_pg_upper': np.array([[lambda_pg_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_ramp_up': np.array([[lambda_ramp_up[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_ramp_down': np.array([[lambda_ramp_down[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_min_on': np.array([[[lambda_min_on[g, tau, t].X for t in range(self.T)] for tau in range(Ton)] for g in range(self.ng)]),
                'lambda_min_off': np.array([[[lambda_min_off[g, tau, t].X for t in range(self.T)] for tau in range(Toff)] for g in range(self.ng)]),
                'lambda_start_cost': np.array([[lambda_start_cost[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_shut_cost': np.array([[lambda_shut_cost[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_coc_nonneg': np.array([[lambda_coc_nonneg[g, t].X for t in range(self.T - 1)] for g in range(self.ng)]),
                'lambda_cpower': np.array([[lambda_cpower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_dcpf_upper': np.array([[lambda_dcpf_upper[l, t].X for t in range(self.T)] for l in range(self.nl)]),
                'lambda_dcpf_lower': np.array([[lambda_dcpf_lower[l, t].X for t in range(self.T)] for l in range(self.nl)]),
                'lambda_x_upper': np.array([[lambda_x_upper[g, t].X for t in range(self.T)] for g in range(self.ng)]),
                'lambda_x_lower': np.array([[lambda_x_lower[g, t].X for t in range(self.T)] for g in range(self.ng)]),
            }
            mu_sol = np.array([[mu[l, t].X for t in range(self.T)] for l in range(self.nl)])
            ita_sol = np.array([[ita[g, t].X for t in range(self.T)] for g in range(self.ng)])

            # surrogate 对偶解
            lambda_surr_sol: Dict[int, np.ndarray] = {}
            for g_s, trainer in self.trainers.items():
                nc = len(trainer.sensitive_timesteps[sample_id])
                lambda_surr_sol[g_s] = np.array([
                    lambda_surr_var[g_s][k].X for k in range(nc)])

            if sample_id <= 2:
                print(f"  dual_block s={sample_id}: "
                      f"obj_dual={obj_dual.getValue():.4f}, "
                      f"obj_opt={obj_opt.getValue():.4f}", flush=True)

            return lambda_sol, mu_sol, ita_sol, lambda_surr_sol
        else:
            print(f"  Dual块求解失败 sample_id={sample_id}, status={model.status}", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  主迭代循环
    # ------------------------------------------------------------------ #
    def iter(self, max_iter: int = 20, dual_decay_round: int = 10,
             nn_epochs: int = 10, surr_nn_epochs: int = 10) -> None:
        """BCD主循环：pg块→dual块→theta/zeta NN→surrogate NN→cal_viol。

        Args:
            max_iter: 外层迭代次数
            dual_decay_round: dual_para_bound衰减轮次
            nn_epochs: theta/zeta NN训练epoch数
            surr_nn_epochs: surrogate NN训练epoch数
        """
        agent = self.agent
        union_analysis = getattr(agent, '_current_union_analysis', None)
        agent.dual_decay_round = dual_decay_round

        gamma = agent.gamma_base / (self.n_samples * max_iter)

        print("\n" + "=" * 70)
        print(f"联合BCD训练: max_iter={max_iter}, nn_epochs={nn_epochs}, "
              f"surr_nn_epochs={surr_nn_epochs}, n_samples={self.n_samples}")
        print("=" * 70, flush=True)

        for i in range(max_iter):
            print(f"\n迭代 {i + 1}/{max_iter}", flush=True)
            agent.iter_number = i
            EPS = 1e-10

            # 1. PG块 → agent.pg/x/cpower/coc, 同步 trainer.x
            for s in range(self.n_samples):
                result = self.iter_with_pg_block(s)
                if result is None:
                    print("  PG块失败，跳过该样本", flush=True)
                    continue
                pg_sol, x_sol, cpower_sol, coc_sol = result

                pg_sol = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                x_sol = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                x_sol = np.where(np.abs(x_sol - 1) < EPS, 1, x_sol)

                agent.pg[s, :, :] = pg_sol
                agent.x[s, :, :] = x_sol
                agent.cpower[s, :, :] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
                agent.coc[s, :, :] = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)

                # 同步 trainer.x
                for g, trainer in self.trainers.items():
                    trainer.x[s] = x_sol[g, :]

            # 2. Dual块 → agent.lambda_/mu/ita, self.lambda_surr
            for s in range(self.n_samples):
                result = self.iter_with_dual_block(s)
                if result is None:
                    print("  Dual块失败，跳过该样本", flush=True)
                    continue
                lambda_sol, mu_sol, ita_sol, lambda_surr_sol = result

                agent.lambda_[s] = lambda_sol
                agent.mu[s, :, :] = np.where(np.abs(mu_sol) < EPS, 0, mu_sol)
                agent.ita[s, :, :] = np.where(np.abs(ita_sol) < EPS, 0, ita_sol)
                self.lambda_surr[s] = lambda_surr_sol

                # 同步 trainer.mu
                for g, trainer in self.trainers.items():
                    if g in lambda_surr_sol:
                        trainer.mu[s] = lambda_surr_sol[g]

            # 3. 刷新迭代级张量缓存
            if TORCH_AVAILABLE and hasattr(agent, 'device'):
                agent._refresh_iter_tensor_cache()

            # 4a. theta/zeta NN 更新
            theta_values_new, zeta_values_new = agent.iter_with_theta_zeta_neural_network(
                union_analysis=union_analysis, num_epochs=nn_epochs)
            if theta_values_new is not None and zeta_values_new is not None:
                agent.theta_values_list = theta_values_new
                agent.zeta_values_list = zeta_values_new
                agent.theta_values = theta_values_new[0]
                agent.zeta_values = zeta_values_new[0]

            # 4b. surrogate NN 更新
            for g, trainer in self.trainers.items():
                trainer.iter_with_surrogate_nn(num_epochs=surr_nn_epochs)

            # 5. cal_viol → rho 更新
            obj_primal, obj_dual, obj_opt = self.cal_viol()

            EPS2 = 1e-12
            obj_primal = obj_primal if abs(obj_primal) >= EPS2 else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= EPS2 else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= EPS2 else 0.0

            integrality = self._compute_avg_integrality()
            print(f"  obj_primal={obj_primal:.4f}, obj_dual={obj_dual:.4f}, "
                  f"obj_opt={obj_opt:.4f}, integrality={integrality:.6f}", flush=True)

            agent.rho_primal = min(agent.rho_primal + gamma * obj_primal, agent.rho_max)
            agent.rho_dual = min(agent.rho_dual + gamma * obj_dual, agent.rho_max)
            agent.rho_opt = min(agent.rho_opt + gamma * obj_opt, agent.rho_max)
            print(f"  rho: primal={agent.rho_primal:.4f}, "
                  f"dual={agent.rho_dual:.4f}, opt={agent.rho_opt:.4f}", flush=True)
            print("  " + "-" * 40, flush=True)
            time.sleep(1)

        print("\n联合BCD训练完成", flush=True)

    # ------------------------------------------------------------------ #
    #  cal_viol: 基础 + surrogate 违反量
    # ------------------------------------------------------------------ #
    def cal_viol(self) -> Tuple[float, float, float]:
        """计算约束违反量：调用agent.cal_viol + 追加surrogate违反量。

        Returns:
            (obj_primal, obj_dual, obj_opt) 总违反量
        """
        agent = self.agent
        union_analysis = getattr(agent, '_current_union_analysis', None)
        obj_primal, obj_dual, obj_opt = agent.cal_viol(union_analysis=union_analysis)

        # 追加 surrogate 违反量
        for s in range(self.n_samples):
            surr_params = self._get_surrogate_params(s)
            x_arr = agent.x[s]

            for g, trainer in self.trainers.items():
                a, b, c, d = surr_params[g]
                sensitive_t = trainer.sensitive_timesteps[s]
                for k, t_k in enumerate(sensitive_t):
                    if t_k + 2 >= self.T:
                        continue
                    surr_val = (float(a[k]) * x_arr[g, t_k]
                                + float(b[k]) * x_arr[g, t_k + 1]
                                + float(c[k]) * x_arr[g, t_k + 2]
                                - float(d[k]))
                    # primal: 单侧违反
                    obj_primal += max(0, surr_val)
                    # opt: 双侧 * |lambda_surr|
                    abs_surr_val = abs(surr_val)
                    if g in self.lambda_surr[s] and k < len(self.lambda_surr[s][g]):
                        obj_opt += abs_surr_val * abs(float(self.lambda_surr[s][g][k]))

        return obj_primal, obj_dual, obj_opt

    # ------------------------------------------------------------------ #
    #  辅助方法
    # ------------------------------------------------------------------ #
    def _get_theta_zeta_dicts(self, sample_id: int):
        """从agent的NN获取theta/zeta字典值（detach，不保留梯度）。"""
        agent = self.agent
        features = agent._extract_features(sample_id)
        feat_t = torch.tensor(np.array(features), dtype=torch.float32,
                              device=self.device).unsqueeze(0)

        agent.theta_net.eval()
        agent.zeta_net.eval()
        with torch.no_grad():
            theta_out = agent.theta_net(feat_t)
            zeta_out = agent.zeta_net(feat_t)
        agent.theta_net.train()
        agent.zeta_net.train()

        theta_arr = theta_out.cpu().numpy().flatten()
        theta_dict = {name: float(val)
                      for name, val in zip(agent.theta_var_names, theta_arr)}

        zeta_arr = zeta_out.cpu().numpy().flatten()
        zeta_dict = {name: float(val)
                     for name, val in zip(agent.zeta_var_names, zeta_arr)}

        return theta_dict, zeta_dict

    def _get_surrogate_params(self, sample_id: int) -> Dict[int, Tuple]:
        """从各trainer的NN获取surrogate参数（detach）。

        Returns:
            dict[unit_id -> (alphas, betas, gammas, deltas)] 各为numpy数组
        """
        params = {}
        for g, trainer in self.trainers.items():
            features = trainer._extract_features(sample_id)
            feat_t = torch.tensor(features, dtype=torch.float32,
                                  device=trainer.device).unsqueeze(0)
            trainer.surrogate_net.eval()
            with torch.no_grad():
                a, b, c, d = trainer.surrogate_net(feat_t)
            trainer.surrogate_net.train()
            nc = trainer.num_coupling_constraints
            params[g] = (a.cpu().numpy().flatten()[:nc],
                         b.cpu().numpy().flatten()[:nc],
                         c.cpu().numpy().flatten()[:nc],
                         d.cpu().numpy().flatten()[:nc])
        return params

    def _compute_avg_integrality(self) -> float:
        """计算所有样本x的平均整数性指标 sum x*(1-x)。"""
        total = 0.0
        for s in range(self.n_samples):
            x_s = self.agent.x[s]
            total += float(np.sum(x_s * (1 - x_s)))
        return total / self.n_samples
