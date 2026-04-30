"""
uc_NN_subproblem_parallel.py
================================
并行版 subproblem_v3 训练，分两个并行层级：

Level 1（机组级，跨进程）
    train_all_surrogates_parallel() 用 ProcessPoolExecutor 并发训练各机组。
    受 Gurobi 许可证限制，默认最多 4 个并发进程（可通过 n_workers 调整）。

Level 2（样本级，跨线程）
    ParallelSubproblemSurrogateTrainer 继承 SubproblemSurrogateTrainer，
    重写 iter()，在 primal/dual block 内用 ThreadPoolExecutor 并发提交各样本。
    Gurobi 求解时释放 GIL，线程并发有效；结果收集后由主线程顺序写入状态。
    NN block 保持串行（批次化训练，不需改动）。

用法
----
    from uc_NN_subproblem_parallel import (
        ParallelSubproblemSurrogateTrainer,
        train_all_surrogates_parallel,
    )
"""

import copy
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from types import SimpleNamespace

import numpy as np
from scenario_utils import normalize_sample_arrays
from pypower.ext2int import ext2int

# ── 路径设置（worker 进程也需要能 import src.*）──────────────
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uc_NN_subproblem import (
    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
    CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
    LP_BACKEND_CVXPY_HIGHS,
    LP_BACKEND_GUROBI,
    SubproblemSurrogateTrainer,
    _extract_pg_electricity_price_matrix,
    _get_sample_pg_electricity_price_matrix,
    _recover_unit_commitment_matrix,
    _solve_pg_electricity_price_from_ed,
    generate_test_data,
    normalize_lp_backend,
    train_all_subproblem_surrogates,
)
try:
    from subproblem_lp_solver import (
        solve_dual_block as solve_dual_block_backend,
        solve_primal_block as solve_primal_block_backend,
    )
except ImportError:
    from src.subproblem_lp_solver import (
        solve_dual_block as solve_dual_block_backend,
        solve_primal_block as solve_primal_block_backend,
    )

def _strict_cvxpy_highs_diagnostics_enabled() -> bool:
    flag = os.environ.get("STRICT_CVXPY_HIGHS", "0")
    return str(flag).strip().lower() in ("1", "true", "yes", "on")


# ════════════════════════════════════════════════════════════════════
# Level 2：样本级并行（线程）
# ════════════════════════════════════════════════════════════════════

class ParallelSubproblemSurrogateTrainer(SubproblemSurrogateTrainer):
    """样本级线程并行的 BCD 训练器。

    继承自 SubproblemSurrogateTrainer，重写 iter() 在 primal/dual block
    内使用 ThreadPoolExecutor 并发求解各样本的 Gurobi 子问题。
    NN block 保持串行（批次化训练）。

    Args:
        ppc: PyPower 案例数据。
        active_set_data: 活动集数据列表。
        T_delta: 时间间隔。
        unit_id: 机组索引。
        lambda_predictor: 已训练的对偶变量预测器（可选）。
        max_constraints: 最大约束数量。
        device: 计算设备。
        n_workers: 并发线程数，默认 min(n_samples, 4)。
    """

    def __init__(
        self,
        ppc,
        active_set_data: List[Dict],
        T_delta: float,
        unit_id: int,
        lambda_predictor=None,
        max_constraints: int = 20,
        lp_backend: str = LP_BACKEND_GUROBI,
        constraint_generation_strategy: str = "sensitive",
        rho_primal_init: float = 1e-3,
        rho_dual_init: float = 1e-3,
        rho_dual_pg_init: float | None = None,
        rho_dual_x_init: float | None = None,
        rho_dual_coc_init: float | None = None,
        rho_binary_init: float = 1.0,
        rho_binary_max: float = 1e4,
        rho_opt_init: float = 1e-3,
        gamma_base: float = 1e-3,
        mu_lower_bound_init: float = 0.1,
        mu_individual_lower_bound_round: int = 3,
        mu_group_lower_bound_round: int = 50,
        mu_signed_round_interval: int | None = None,
        mu_sign_hysteresis_rounds: int = 2,
        mu_sign_flip_min_share: float = 0.67,
        x_bound_dual_zero_rounds: int = 0,
        pg_cost_start_round: int = 3,
        pg_cost_scale_multiplier: float = 1.2,
        nn_hidden_dims: list[int] | None = None,
        pg_cost_hidden_dims: list[int] | None = None,
        nn_learning_rate: float = 1e-4,
        cost_learning_rate: float = 1e-5,
        pg_cost_lr: float = 2e-5,
        pg_cost_surr_lr: float = 5e-5,
        nn_batch_strategy: str = "full-batch",
        nn_batch_size: int = 4,
        nn_shuffle: bool = True,
        pg_cost_nn_epochs: int | None = None,
        pg_cost_reg_deadband: float = 0.25,
        pg_cost_softbound_weight: float = 1.0,
        nn_smooth_abs_eps: float = 1e-6,
        pg_cost_smooth_abs_eps: float = 1e-6,
        pg_cost_batch_strategy: str | None = None,
        pg_cost_batch_size: int | None = None,
        pg_cost_shuffle: bool | None = None,
        pg_cost_use_sample_weights: bool = True,
        pg_cost_sample_weight_power: float = 1.0,
        pg_cost_sample_weight_clip: float = 10.0,
        iter_delta_reg_weight: float = 5e-5,
        iter_delta_reg_deadband: float = 0.10,
        loss_ratio_primal: float = 1.0,
        loss_ratio_dual_pg: float = 1.0,
        loss_ratio_dual_x: float = 1.0,
        nn_dual_term_interval: int | None = 1,
        loss_ratio_opt: float = 1.0,
        loss_ratio_reg: float = 1.0,
        pg_block_prox_weight: float = 2e-2,
        dual_block_prox_weight: float = 1e-2,
        ignore_startup_shutdown_costs: bool = False,
        unit_predictor=None,
        use_unit_predictor: bool = False,
        predictor_warmup_rounds: int = 0,
        sign4_curriculum_rounds: int = 0,
        sign4_initial_scale: float = 1.0,
        sign4_final_scale: float = 1.0,
        sign4_delay_rounds: int = 0,
        unit_predictor_finetune_lr: float = 1e-5,
        unit_predictor_weight_decay: float = 1e-4,
        pg_cost_single_sample_reg_scale: float | None = None,
        pg_cost_c_pg_adam_weight_decay: float | None = None,
        main_direct_train_config: dict | None = None,
        c_pg_direct_train_config: dict | None = None,
        nn_main_eta_min_ratio: float = 0.08,
        nn_main_lr_late_scale: float = 0.42,
        nn_main_adam_weight_decay: float = 1e-4,
        nn_main_grad_clip: float = 0.85,
        nn_main_kkt_lr_scale: float = 1.0,
        case_name: str | None = None,
        enable_surrogate_delta_reference_lift: bool | None = None,
        surrogate_delta_reference_eps: float = 1e-6,
        surrogate_delta_reference_scope: str = "sign4_only",
        surrogate_delta_reference_min_abs_factor: float = 1e-9,
        device=None,
        n_workers: int = 4,
    ):
        super().__init__(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=lambda_predictor,
            max_constraints=max_constraints,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_binary_max=rho_binary_max,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_lower_bound_init=mu_lower_bound_init,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            mu_signed_round_interval=mu_signed_round_interval,
            mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
            mu_sign_flip_min_share=mu_sign_flip_min_share,
            x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_scale_multiplier=pg_cost_scale_multiplier,
            nn_hidden_dims=nn_hidden_dims,
            pg_cost_hidden_dims=pg_cost_hidden_dims,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            pg_cost_nn_epochs=pg_cost_nn_epochs,
            pg_cost_reg_deadband=pg_cost_reg_deadband,
            pg_cost_softbound_weight=pg_cost_softbound_weight,
            nn_smooth_abs_eps=nn_smooth_abs_eps,
            pg_cost_smooth_abs_eps=pg_cost_smooth_abs_eps,
            pg_cost_batch_strategy=pg_cost_batch_strategy,
            pg_cost_batch_size=pg_cost_batch_size,
            pg_cost_shuffle=pg_cost_shuffle,
            pg_cost_use_sample_weights=pg_cost_use_sample_weights,
            pg_cost_sample_weight_power=pg_cost_sample_weight_power,
            pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
            iter_delta_reg_weight=iter_delta_reg_weight,
            iter_delta_reg_deadband=iter_delta_reg_deadband,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            nn_dual_term_interval=nn_dual_term_interval,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            unit_predictor=unit_predictor,
            use_unit_predictor=use_unit_predictor,
            predictor_warmup_rounds=predictor_warmup_rounds,
            sign4_curriculum_rounds=sign4_curriculum_rounds,
            sign4_initial_scale=sign4_initial_scale,
            sign4_final_scale=sign4_final_scale,
            sign4_delay_rounds=sign4_delay_rounds,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
            pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
            main_direct_train_config=main_direct_train_config,
            c_pg_direct_train_config=c_pg_direct_train_config,
            nn_main_eta_min_ratio=nn_main_eta_min_ratio,
            nn_main_lr_late_scale=nn_main_lr_late_scale,
            nn_main_adam_weight_decay=nn_main_adam_weight_decay,
            nn_main_grad_clip=nn_main_grad_clip,
            nn_main_kkt_lr_scale=nn_main_kkt_lr_scale,
            case_name=case_name,
            enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
            surrogate_delta_reference_eps=surrogate_delta_reference_eps,
            surrogate_delta_reference_scope=surrogate_delta_reference_scope,
            surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
            device=device,
        )
        self.n_workers = min(n_workers, self.n_samples)

    def _build_sample_worker_state(self, sample_id: int) -> dict:
        return {
            # For cvxpy_highs worker process: proxy only contains a single-sample view
            # (lists with length 1), so the solver must be called with local sample_id=0.
            # We keep the original/global sample id here for logging parity with the
            # non-parallel trainer.
            'display_sample_id': int(sample_id),
            'unit_id': self.unit_id,
            'T': self.T,
            'T_delta': self.T_delta,
            'gen': np.asarray(self.gen, dtype=float).copy(),
            'gencost': np.asarray(self.gencost, dtype=float).copy(),
            'Ru_all': np.asarray(self.Ru_all, dtype=float).copy(),
            'Rd_all': np.asarray(self.Rd_all, dtype=float).copy(),
            'Ru_co_all': np.asarray(self.Ru_co_all, dtype=float).copy(),
            'Rd_co_all': np.asarray(self.Rd_co_all, dtype=float).copy(),
            'ignore_startup_shutdown_costs': bool(self.ignore_startup_shutdown_costs),
            'rho_primal': float(self.rho_primal),
            'rho_opt': float(self.rho_opt),
            'rho_binary': float(self.rho_binary),
            'rho_dual_pg': float(self.rho_dual_pg),
            'rho_dual_x': float(self.rho_dual_x),
            'rho_dual_coc': float(self.rho_dual_coc),
            'pg_block_prox_weight': float(self.pg_block_prox_weight),
            'dual_block_prox_weight': float(self.dual_block_prox_weight),
            'num_coupling_constraints': int(self.num_coupling_constraints),
            'all_mode_group_size': int(self.all_mode_group_size),
            'constraint_generation_strategy': str(self.constraint_generation_strategy),
            'sign4_curriculum_rounds': int(self.sign4_curriculum_rounds),
            'sign4_initial_scale': float(self.sign4_initial_scale),
            'sign4_final_scale': float(self.sign4_final_scale),
            'sign4_delay_rounds': int(getattr(self, 'sign4_delay_rounds', 0)),
            'mu_lower_bound': float(self.mu_lower_bound),
            'mu_individual_lower_bound_round': int(self.mu_individual_lower_bound_round),
            'mu_group_lower_bound_round': int(self.mu_group_lower_bound_round),
            'mu_signed_round_interval': int(self.mu_signed_round_interval),
            'x_bound_dual_zero_rounds': int(self.x_bound_dual_zero_rounds),
            'iter_number': int(self.iter_number),
            '_mu_sign_relaxation_last_iter': int(
                getattr(self, '_mu_sign_relaxation_last_iter', 10**9)
            ),
            'use_group_mu_lower_bound': bool(self._uses_group_mu_lower_bound()),
            'surrogate_direction_signs': self._get_surrogate_direction_signs(),
            'pg': [np.asarray(self.pg[sample_id], dtype=float).copy()],
            'x': [np.asarray(self.x[sample_id], dtype=float).copy()],
            'coc': [np.asarray(self.coc[sample_id], dtype=float).copy()],
            'mu': [np.asarray(self.mu[sample_id], dtype=float).copy()],
            'lambda_vals': [np.asarray(self.lambda_vals[sample_id], dtype=float).copy()],
            'lambda_inherent': [copy.deepcopy(self.lambda_inherent[sample_id])],
            'sensitive_timesteps': [list(self.sensitive_timesteps[sample_id])],
            'surrogate_constraint_offsets': [copy.deepcopy(self.surrogate_constraint_offsets[sample_id])],
            'active_set_data': [{'x_true': copy.deepcopy(self.active_set_data[sample_id].get('x_true'))}],
            'subproblem_Ton': int(self.subproblem_Ton),
            'subproblem_Toff': int(self.subproblem_Toff),
        }

    def _run_cvxpy_highs_sample_pool(self, block: str, block_args: list[tuple], prefix: str) -> Dict[int, tuple]:
        results: Dict[int, tuple] = {}
        # Use threads for cvxpy_highs sample-level parallelism on Windows.
        # Process spawning re-imports the whole training entrypoint in each child,
        # which is expensive and can fail when optional heavy modules (for example
        # pandas via case118 loaders) are imported under tight virtual-memory limits.
        ordered_logs: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_sid = {}
            for s, alphas, betas, gammas, deltas, costs, pg_costs in block_args:
                task = {
                    'block': block,
                    'sample_id': s,
                    'trainer_state': self._build_sample_worker_state(s),
                    'alphas': np.asarray(alphas, dtype=float).copy(),
                    'betas': np.asarray(betas, dtype=float).copy(),
                    'gammas': np.asarray(gammas, dtype=float).copy(),
                    'deltas': np.asarray(deltas, dtype=float).copy(),
                    'costs': None if costs is None else np.asarray(costs, dtype=float).copy(),
                    'pg_costs': None if pg_costs is None else np.asarray(pg_costs, dtype=float).copy(),
                }
                future = executor.submit(_solve_sample_block_worker, task)
                future_to_sid[future] = s

            for future in as_completed(future_to_sid):
                s = future_to_sid[future]
                try:
                    payload = future.result()
                    results[s] = payload['result']
                    msg = payload.get('lp_log')
                    if msg:
                        ordered_logs[s] = msg
                except Exception as exc:
                    print(f"{prefix} {block}_block sample={s} 异常: {exc}", flush=True)
                    if _strict_cvxpy_highs_diagnostics_enabled():
                        raise
                    results[s] = (None, None, None, None) if block == 'primal' else (None, None)
        for sid in sorted(ordered_logs):
            print(ordered_logs[sid], flush=True)
        return results

    def iter(
        self,
        max_iter: int = 20,
        nn_epochs: int = 10,
        pg_cost_nn_epochs: int | None = None,
        nn_batch_strategy: str | None = None,
        nn_batch_size: int | None = None,
        nn_shuffle: bool | None = None,
        nn_learning_rate: float | None = None,
        cost_learning_rate: float | None = None,
        pg_cost_surr_learning_rate: float | None = None,
    ):
        """主 BCD 迭代循环（样本级线程并行版本）。

        primal/dual block 内并发提交各样本；结果收集后主线程顺序更新状态。
        NN block 保持串行。
        """
        prefix = f"[Unit-{self.unit_id}]"
        print(
            f"{prefix} 开始样本级并行BCD迭代 "
            f"(n_workers={self.n_workers}, max_iter={max_iter}, nn_epochs={nn_epochs})",
            flush=True,
        )

        self._configure_surrogate_bcd_run(max_iter)
        EPS = 1e-10
        gamma = self.gamma_base / (self.n_samples * max(max_iter, 1))
        gamma_dual = gamma * self.gamma_dual_component_scale
        self.gamma = gamma
        if pg_cost_nn_epochs is not None:
            resolved_pg_cost_nn_epochs = max(int(pg_cost_nn_epochs), 0)
        elif self.pg_cost_nn_epochs is not None:
            resolved_pg_cost_nn_epochs = max(int(self.pg_cost_nn_epochs), 0)
        else:
            resolved_pg_cost_nn_epochs = 10

        for i in range(max_iter):
            print(f"{prefix} 🔄 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i
            self._sync_surrogate_direction_strategy_state()

            # ── 1. Primal block（线程并行） ──────────────────────────
            # 提前复制参数，避免线程间共享可变数组
            primal_args = [
                (
                    s,
                    self.alpha_values[s].copy(),
                    self.beta_values[s].copy(),
                    self.gamma_values[s].copy(),
                    self.delta_values[s].copy(),
                    self.cost_values[s].copy(),
                    self.pg_cost_values[s].copy(),
                )
                for s in range(self.n_samples)
            ]

            primal_results: Dict[int, tuple] = {}
            if self._lp_backend == LP_BACKEND_CVXPY_HIGHS and self.n_workers > 1:
                primal_results = self._run_cvxpy_highs_sample_pool('primal', primal_args, prefix)
            else:
                # 非 CVXPY 池路径（Gurobi，或 CVXPY 且 n_workers==1）：Gurobi 块经 _emit_subproblem_block_log 入队后按 sample_id 输出；CVXPY 仍在 solve_* 内直接打印
                self._pending_block_logs = []
                self._pending_block_logs_lock = threading.Lock()
                self._defer_subproblem_block_log = True
                primal_results = {}
                try:
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        future_to_sid = {
                            executor.submit(
                                self.iter_with_primal_block,
                                s, alphas, betas, gammas, deltas, costs, pg_costs,
                            ): s
                            for s, alphas, betas, gammas, deltas, costs, pg_costs in primal_args
                        }
                        for future in as_completed(future_to_sid):
                            s = future_to_sid[future]
                            try:
                                primal_results[s] = future.result()
                            except Exception as exc:
                                print(f"{prefix} primal_block sample={s} 异常: {exc}", flush=True)
                                if _strict_cvxpy_highs_diagnostics_enabled():
                                    raise
                                primal_results[s] = (None, None, None, None)
                finally:
                    self._defer_subproblem_block_log = False
                for _, line in sorted(self._pending_block_logs, key=lambda x: x[0]):
                    print(line, flush=True)

            # 顺序写入状态（避免并发写）
            for s in range(self.n_samples):
                pg_sol, x_sol, coc_sol, cpower_sol = primal_results[s]
                if pg_sol is not None:
                    self.pg[s]     = np.where(np.abs(pg_sol) < EPS, 0, pg_sol)
                    self.x[s]      = np.where(np.abs(x_sol) < EPS, 0, x_sol)
                    self.x[s]      = np.where(np.abs(self.x[s] - 1) < EPS, 1, self.x[s])
                    self.coc[s]    = np.where(np.abs(coc_sol) < EPS, 0, coc_sol)
                    self.cpower[s] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)

            # ── 2. Dual block（线程并行） ────────────────────────────
            lb_mu = self._current_mu_lower_bound_value()

            dual_args = [
                (
                    s,
                    self.alpha_values[s].copy(),
                    self.beta_values[s].copy(),
                    self.gamma_values[s].copy(),
                    self.delta_values[s].copy(),
                    self.cost_values[s].copy(),
                    self.pg_cost_values[s].copy(),
                )
                for s in range(self.n_samples)
            ]

            dual_results: Dict[int, tuple] = {}
            if self._lp_backend == LP_BACKEND_CVXPY_HIGHS and self.n_workers > 1:
                dual_results = self._run_cvxpy_highs_sample_pool('dual', dual_args, prefix)
            else:
                self._pending_block_logs = []
                self._pending_block_logs_lock = threading.Lock()
                self._defer_subproblem_block_log = True
                dual_results = {}
                try:
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        future_to_sid = {
                            executor.submit(
                                self.iter_with_dual_block,
                                s, alphas, betas, gammas, deltas, costs, pg_costs,
                            ): s
                            for s, alphas, betas, gammas, deltas, costs, pg_costs in dual_args
                        }
                        for future in as_completed(future_to_sid):
                            s = future_to_sid[future]
                            try:
                                dual_results[s] = future.result()
                            except Exception as exc:
                                print(f"{prefix} dual_block sample={s} 异常: {exc}", flush=True)
                                if _strict_cvxpy_highs_diagnostics_enabled():
                                    raise
                                dual_results[s] = (None, None)
                finally:
                    self._defer_subproblem_block_log = False
                for _, line in sorted(self._pending_block_logs, key=lambda x: x[0]):
                    print(line, flush=True)

            # 顺序写入状态
            for s in range(self.n_samples):
                lambda_inherent_sol, mu_sol = dual_results[s]
                if lambda_inherent_sol is not None:
                    self.lambda_inherent[s] = lambda_inherent_sol
                    self.mu[s] = self._apply_mu_lower_bound_policy(mu_sol, lb_mu)

            _z = lambda v: v if abs(v) >= 1e-12 else 0.0
            nn_metrics_pre = self.cal_nn_logging_components()
            print(
                f"{prefix}[NN-metric][before] "
                f"obj_primal={nn_metrics_pre['obj_primal']:.6f}, "
                f"obj_dual_pg={nn_metrics_pre['obj_dual_pg']:.6f}, "
                f"obj_dual_x={nn_metrics_pre['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_pre['obj_opt']:.6f}, "
                f"reg_main={nn_metrics_pre['reg_main']:.6f}, "
                f"reg_pg={nn_metrics_pre['reg_pg']:.6f}",
                flush=True,
            )

            # ── 3. NN block（串行，批次化训练） ──────────────────────
            # 与非并行版一致：每轮 BCD 内先 direct-target 预训练，再可微 KKT 微调；
            # c_pg 同理（direct 在 pg_cost_start_round 之后才会真正更新，见 iter_with_c_pg_direct_targets）。
            self._prev_alpha_values = self.alpha_values.copy()
            self._prev_beta_values = self.beta_values.copy()
            self._prev_gamma_values = self.gamma_values.copy()
            self._prev_delta_values = self.delta_values.copy()
            self._prev_cost_values = self.cost_values.copy()
            self._prev_pg_cost_values = self.pg_cost_values.copy()
            self._invalidate_loss_tensor_cache()
            self.iter_with_main_direct_targets()
            self.iter_with_surrogate_nn(
                num_epochs=nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
                cost_learning_rate=cost_learning_rate,
            )
            self.iter_with_c_pg_direct_targets()
            self.iter_with_c_pg_nn(
                num_epochs=resolved_pg_cost_nn_epochs,
                batch_size=(
                    self.pg_cost_batch_size
                    if self.pg_cost_batch_size is not None
                    else nn_batch_size
                ),
                batch_strategy=(
                    self.pg_cost_batch_strategy
                    if self.pg_cost_batch_strategy is not None
                    else nn_batch_strategy
                ),
                shuffle=(
                    self.pg_cost_shuffle
                    if self.pg_cost_shuffle is not None
                    else nn_shuffle
                ),
                learning_rate=pg_cost_surr_learning_rate,
            )
            self._refresh_cached_surrogate_outputs()

            nn_metrics_after = self.cal_nn_logging_components()
            print(
                f"{prefix}[NN-metric][after] "
                f"obj_primal={nn_metrics_after['obj_primal']:.6f}, "
                f"obj_dual_pg={nn_metrics_after['obj_dual_pg']:.6f}, "
                f"obj_dual_x={nn_metrics_after['obj_dual_x']:.6f}, "
                f"obj_opt={nn_metrics_after['obj_opt']:.6f}, "
                f"reg_main={nn_metrics_after['reg_main']:.6f}, "
                f"reg_pg={nn_metrics_after['reg_pg']:.6f}",
                flush=True,
            )

            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = (
                _z(obj_primal), _z(obj_dual_pg), _z(obj_dual_x), _z(obj_dual_coc), _z(obj_dual), _z(obj_opt)
            )
            obj_binary = self.cal_obj_binary_gap()

            if i == max_iter - 1:
                obj_opt_breakdown = self.cal_obj_opt_breakdown()
                print(
                    f"{prefix}[full][final] obj_opt_breakdown: "
                    f"surrogate={obj_opt_breakdown['surrogate']:.6f}, "
                    f"pg_lower={obj_opt_breakdown['pg_lower']:.6f}, "
                    f"pg_upper={obj_opt_breakdown['pg_upper']:.6f}, "
                    f"x_lower={obj_opt_breakdown['x_lower']:.6f}, "
                    f"x_upper={obj_opt_breakdown['x_upper']:.6f}, "
                    f"ramp_up={obj_opt_breakdown['ramp_up']:.6f}, "
                    f"ramp_down={obj_opt_breakdown['ramp_down']:.6f}, "
                    f"min_on={obj_opt_breakdown['min_on']:.6f}, "
                    f"min_off={obj_opt_breakdown['min_off']:.6f}, "
                    f"start_cost={obj_opt_breakdown['start_cost']:.6f}, "
                    f"shut_cost={obj_opt_breakdown['shut_cost']:.6f}, "
                    f"coc_nonneg={obj_opt_breakdown['coc_nonneg']:.6f}, "
                    f"total={obj_opt_breakdown['total']:.6f}",
                    flush=True,
                )

            if i >= 3:
                self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
                self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
                self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
                self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
                self._sync_rho_dual_summary()
                self.rho_binary = min(self.rho_binary + gamma * obj_binary, self.rho_binary_max)
                self.rho_opt    = min(self.rho_opt    + gamma * obj_opt,    self.rho_max)

            print(
                f"{prefix}   ρ_primal={self.rho_primal:.4f}, ρ_dual_pg={self.rho_dual_pg:.4f}, "
                f"ρ_dual_x={self.rho_dual_x:.4f}, ρ_dual_coc={self.rho_dual_coc:.4f}, "
                f"ρ_dual={self.rho_dual:.4f}, ρ_binary={self.rho_binary:.4f} "
                f"(≤{self.rho_binary_max:.4g}), ρ_opt={self.rho_opt:.4f}",
                flush=True,
            )
            # 与非并行版一致；带机组前缀便于机组级多进程 stdout 交错时辨认
            print(f"{prefix} " + "-" * 40, flush=True)

        print(f"{prefix} ✓ 样本级并行训练完成", flush=True)


# ════════════════════════════════════════════════════════════════════
# Level 1：机组级并行（进程）
# ════════════════════════════════════════════════════════════════════

class _SampleWorkerTrainerProxy(SimpleNamespace):
    def _get_surrogate_direction_signs(self, size: int | None = None) -> np.ndarray:
        signs = np.asarray(self.surrogate_direction_signs, dtype=float)
        if size is None:
            return signs.copy()
        return signs[: int(size)].copy()

    def _current_sign4_curriculum_scale(self) -> float:
        rounds = max(int(getattr(self, 'sign4_curriculum_rounds', 0) or 0), 0)
        initial = max(float(getattr(self, 'sign4_initial_scale', 1.0) or 0.0), 0.0)
        final = max(float(getattr(self, 'sign4_final_scale', 1.0) or 0.0), 0.0)
        delay = max(int(getattr(self, 'sign4_delay_rounds', 0) or 0), 0)
        iter_n = float(getattr(self, 'iter_number', 0))
        if delay > 0 and iter_n < float(delay):
            return 0.0
        effective_iter = max(iter_n - float(delay), 0.0)
        if rounds <= 0:
            return final
        progress = min(max(effective_iter, 0.0) / float(rounds), 1.0)
        return initial + (final - initial) * progress

    def _sign4_curriculum_factors(self, size: int) -> np.ndarray:
        factors = np.ones(int(size), dtype=float)
        strategy = str(getattr(self, 'constraint_generation_strategy', '') or '')
        if strategy not in {'all_templates_sign4', 'all_templates_sign4_plus_single'}:
            return factors
        n_sign4 = min(int(size), int(self.all_mode_group_size) * max(int(self.T) - 2, 0))
        if n_sign4 > 0:
            factors[:n_sign4] = self._current_sign4_curriculum_scale()
        return factors

    def _apply_surrogate_direction_to_params(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        gammas: np.ndarray,
        deltas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        signs = self._get_surrogate_direction_signs(len(alphas))
        factors = signs * self._sign4_curriculum_factors(len(alphas))
        return (
            np.asarray(alphas, dtype=float) * factors,
            np.asarray(betas, dtype=float) * factors,
            np.asarray(gammas, dtype=float) * factors,
            np.asarray(deltas, dtype=float) * factors,
        )

    def _constraint_offsets_for_sample(self, sample_id: int) -> list[tuple[int, ...]]:
        return list(self.surrogate_constraint_offsets[sample_id])

    def _uses_group_mu_lower_bound(self) -> bool:
        return bool(self.use_group_mu_lower_bound)

    def _mu_floor_schedule_iter(self) -> int:
        """与 ``SubproblemSurrogateTrainer._mu_floor_schedule_iter`` 一致（worker 仅用标量快照）。"""
        delay = max(int(getattr(self, 'sign4_delay_rounds', 0) or 0), 0)
        if delay <= 0:
            return int(self.iter_number)
        strategy = str(getattr(self, 'constraint_generation_strategy', '') or '')
        if strategy not in {
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
            CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        }:
            return int(self.iter_number)
        i = int(self.iter_number)
        if i < delay:
            return -1
        return i - delay

    def _get_mu_lower_bound_phase(self) -> str:
        t = self._mu_floor_schedule_iter()
        if t < 0:
            return "individual"
        if t < self.mu_individual_lower_bound_round:
            return "individual"
        if t < self.mu_group_lower_bound_round:
            return "group" if self._uses_group_mu_lower_bound() else "individual"
        return "none"

    def _current_mu_lower_bound_value(self) -> float:
        return self.mu_lower_bound if self._get_mu_lower_bound_phase() != "none" else 0.0

    def _is_mu_sign_relaxation_round(self) -> bool:
        interval = int(getattr(self, 'mu_signed_round_interval', 0) or 0)
        lb = float(self._current_mu_lower_bound_value())
        cutoff = int(getattr(self, '_mu_sign_relaxation_last_iter', 10**9))
        if int(getattr(self, 'iter_number', 0)) >= cutoff:
            return False
        return (
            interval > 0
            and lb > 0.0
            and ((int(self.iter_number) + 1) % interval == 0)
        )

    def _force_zero_x_bound_duals(self) -> bool:
        return self.iter_number < self.x_bound_dual_zero_rounds


def _solve_sample_block_worker(task: dict) -> dict:
    proxy = _SampleWorkerTrainerProxy(**task['trainer_state'])
    # 延迟到主线程按 sample_id 排序后打印，避免线程完成顺序导致 primal/dual 行交错难读
    proxy._defer_lp_block_log = True
    sample_id = 0
    if task['block'] == 'primal':
        result = solve_primal_block_backend(
            proxy,
            sample_id,
            task['alphas'],
            task['betas'],
            task['gammas'],
            task['deltas'],
            costs=task.get('costs'),
            pg_costs=task.get('pg_costs'),
        )
    elif task['block'] == 'dual':
        result = solve_dual_block_backend(
            proxy,
            sample_id,
            task['alphas'],
            task['betas'],
            task['gammas'],
            task['deltas'],
            costs=task.get('costs'),
            pg_costs=task.get('pg_costs'),
        )
    else:
        raise ValueError(f"Unsupported sample worker block: {task['block']}")
    return {
        'sample_id': int(task['sample_id']),
        'result': result,
        'lp_log': getattr(proxy, '_deferred_lp_block_log', None),
    }


def _train_unit_worker(args: dict) -> dict:
    """顶层可 pickle 的 worker 函数（供 ProcessPoolExecutor 调用）。

    在子进程中构建 trainer，运行 iter()，返回序列化状态 dict。

    Args:
        args: 包含以下键的字典：
            ppc, active_set_data, lambda_vals (np.ndarray | None),
            unit_id, T_delta, max_iter, nn_epochs,
            sample_n_workers, use_sample_parallel, save_dir.

    Returns:
        包含 unit_id 和 alpha/beta/gamma/delta/mu/rho 的状态字典。
    """
    # 子进程路径修复（Windows spawn 不继承父进程 sys.path）
    _src = str(Path(__file__).resolve().parent)
    _root = str(Path(__file__).resolve().parent.parent)
    for _p in [_src, _root]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from uc_NN_subproblem import (
        SubproblemSurrogateTrainer,
        SingleUnitBinaryPredictorTrainer,
    )

    unit_id = args['unit_id']
    ppc = args['ppc']
    # deepcopy 全量样本可能很慢且不易占满多核；先打日志避免误判为“卡死无输出”
    print(
        f"[Unit-{unit_id}] worker pid={os.getpid()}：开始 deepcopy(active_set_data)…",
        flush=True,
    )
    active_set_data = copy.deepcopy(args['active_set_data'])
    print(f"[Unit-{unit_id}] 数据复制完成，开始构建 trainer", flush=True)
    lambda_vals         = args.get('lambda_vals')
    T_delta             = args['T_delta']
    max_iter            = args['max_iter']
    nn_epochs           = args['nn_epochs']
    rho_primal_init     = args.get('rho_primal_init', 1e-3)
    rho_dual_init       = args.get('rho_dual_init', 1e-3)
    rho_dual_pg_init    = args.get('rho_dual_pg_init')
    rho_dual_x_init     = args.get('rho_dual_x_init')
    rho_dual_coc_init   = args.get('rho_dual_coc_init')
    rho_binary_init     = args.get('rho_binary_init', 1.0)
    rho_binary_max      = float(args.get('rho_binary_max', 1e4))
    rho_opt_init        = args.get('rho_opt_init', 1e-3)
    gamma_base          = args.get('gamma_base', 1e-3)
    mu_lower_bound_init = args.get('mu_lower_bound_init', 0.1)
    mu_individual_lower_bound_round = args.get('mu_individual_lower_bound_round', 3)
    mu_group_lower_bound_round = args.get('mu_group_lower_bound_round', 50)
    mu_signed_round_interval = args.get('mu_signed_round_interval')
    mu_sign_hysteresis_rounds = args.get('mu_sign_hysteresis_rounds', 2)
    mu_sign_flip_min_share = args.get('mu_sign_flip_min_share', 0.67)
    x_bound_dual_zero_rounds = args.get('x_bound_dual_zero_rounds', 0)
    ignore_startup_shutdown_costs = args.get('ignore_startup_shutdown_costs', False)
    nn_learning_rate    = args.get('nn_learning_rate', 1e-4)
    cost_learning_rate  = args.get('cost_learning_rate', 1e-5)
    nn_batch_strategy   = args.get('nn_batch_strategy', 'full-batch')
    nn_batch_size       = args.get('nn_batch_size', 4)
    nn_shuffle          = args.get('nn_shuffle', True)
    nn_smooth_abs_eps   = args.get('nn_smooth_abs_eps', 1e-6)
    nn_main_eta_min_ratio = float(args.get('nn_main_eta_min_ratio', 0.08))
    nn_main_lr_late_scale = float(args.get('nn_main_lr_late_scale', 0.42))
    nn_main_adam_weight_decay = float(args.get('nn_main_adam_weight_decay', 1e-4))
    nn_main_grad_clip = float(args.get('nn_main_grad_clip', 0.85))
    nn_main_kkt_lr_scale = float(args.get('nn_main_kkt_lr_scale', 1.0))
    loss_ratio_primal   = args.get('loss_ratio_primal', 1.0)
    loss_ratio_dual_pg  = args.get('loss_ratio_dual_pg', 1.0)
    loss_ratio_dual_x   = args.get('loss_ratio_dual_x', 1.0)
    nn_dual_term_interval = args.get('nn_dual_term_interval', 1)
    loss_ratio_opt      = args.get('loss_ratio_opt', 1.0)
    loss_ratio_reg      = args.get('loss_ratio_reg', 1.0)
    pg_cost_nn_epochs   = args.get('pg_cost_nn_epochs')
    pg_cost_start_round = args.get('pg_cost_start_round', 3)
    pg_cost_lr          = args.get('pg_cost_lr', 2e-5)
    pg_cost_surr_lr     = args.get('pg_cost_surr_lr', 5e-5)
    pg_cost_batch_strategy = args.get('pg_cost_batch_strategy')
    pg_cost_batch_size  = args.get('pg_cost_batch_size')
    pg_cost_shuffle     = args.get('pg_cost_shuffle')
    pg_cost_use_sample_weights = args.get('pg_cost_use_sample_weights', True)
    pg_cost_sample_weight_power = args.get('pg_cost_sample_weight_power', 1.0)
    pg_cost_sample_weight_clip = args.get('pg_cost_sample_weight_clip', 10.0)
    pg_cost_single_sample_reg_scale = args.get('pg_cost_single_sample_reg_scale')
    pg_cost_c_pg_adam_weight_decay = args.get('pg_cost_c_pg_adam_weight_decay')
    main_direct_train_config = args.get('main_direct_train_config')
    c_pg_direct_train_config = args.get('c_pg_direct_train_config')
    pg_cost_reg_deadband = args.get('pg_cost_reg_deadband', 0.25)
    pg_cost_softbound_weight = args.get('pg_cost_softbound_weight', 1.0)
    iter_delta_reg_weight = args.get('iter_delta_reg_weight', 5e-5)
    iter_delta_reg_deadband = args.get('iter_delta_reg_deadband', 0.10)
    pg_block_prox_weight = args.get('pg_block_prox_weight', 2e-2)
    dual_block_prox_weight = args.get('dual_block_prox_weight', 1e-2)
    sample_n_workers    = args.get('sample_n_workers', 4)
    use_sample_parallel = args.get('use_sample_parallel', True)
    lp_backend          = args.get('lp_backend', LP_BACKEND_GUROBI)
    constraint_generation_strategy = args.get('constraint_generation_strategy', 'sensitive')
    save_dir            = args.get('save_dir')
    unit_predictor_path = args.get('unit_predictor_path')
    use_unit_predictor_flag = bool(args.get('use_unit_predictor', False))
    predictor_warmup_rounds = args.get('predictor_warmup_rounds', 0)
    sign4_curriculum_rounds = args.get('sign4_curriculum_rounds', 0)
    sign4_initial_scale = args.get('sign4_initial_scale', 1.0)
    sign4_final_scale = args.get('sign4_final_scale', 1.0)
    sign4_delay_rounds = args.get('sign4_delay_rounds', 0)
    enable_surrogate_delta_reference_lift = args.get('enable_surrogate_delta_reference_lift', None)
    surrogate_delta_reference_eps = args.get('surrogate_delta_reference_eps', 1e-6)
    surrogate_delta_reference_scope = args.get('surrogate_delta_reference_scope', "sign4_only")
    surrogate_delta_reference_min_abs_factor = args.get('surrogate_delta_reference_min_abs_factor', 1e-9)
    unit_predictor_finetune_lr = args.get('unit_predictor_finetune_lr', 1e-5)
    unit_predictor_weight_decay = args.get('unit_predictor_weight_decay', 1e-4)
    unit_predictor_hidden_dims = args.get('unit_predictor_hidden_dims')
    unit_predictor_net_variant = args.get('unit_predictor_net_variant', 'mlp')
    unit_predictor_tcn_channels = args.get('unit_predictor_tcn_channels', 64)
    unit_predictor_tcn_depth = args.get('unit_predictor_tcn_depth', 6)
    unit_predictor_tconv_channels = args.get('unit_predictor_tconv_channels', 64)
    unit_predictor_tconv_depth = args.get('unit_predictor_tconv_depth', 3)
    unit_predictor_dropout = args.get('unit_predictor_dropout', 0.1)
    case_name = args.get('case_name')
    # 机组级 ProcessPool 多进程时，各子进程若均默认选用 CUDA，常在 NN 首次 forward
    # （_refresh_cached_surrogate_outputs）处互斥/死锁或长时间无输出；默认强制 NN 用 CPU。
    force_nn_cpu_for_multiprocess = bool(args.get('force_nn_cpu_for_multiprocess', False))
    if force_nn_cpu_for_multiprocess and str(
        os.environ.get('SUBPROBLEM_UNIT_WORKER_ALLOW_CUDA', '0')
    ).strip().lower() in ('1', 'true', 'yes', 'on'):
        force_nn_cpu_for_multiprocess = False
    nn_device = None
    if force_nn_cpu_for_multiprocess:
        import torch
        torch.set_num_threads(
            max(1, int(os.environ.get('SUBPROBLEM_UNIT_WORKER_TORCH_THREADS', '1')))
        )
        nn_device = torch.device('cpu')
        print(
            f"[Unit-{unit_id}] 机组级多进程：代理约束 NN 使用 {nn_device} "
            f"（避免多进程 CUDA 初始化/forward 卡住；单机训练仍可用 GPU）",
            flush=True,
        )

    prefix = f"[Unit-{unit_id}]"

    # 将预计算的 lambda_vals 注入 active_set_data，避免子进程重复求解 LP
    if lambda_vals is not None:
        for i, sample in enumerate(active_set_data):
            sample['lambda_pg_electricity_price'] = copy.deepcopy(lambda_vals[i])

    # 单机组 0/1 预测器：通过磁盘路径重建（跨进程 pickle PyTorch 模型不可靠）
    unit_predictor_obj = None
    if use_unit_predictor_flag and unit_predictor_path and os.path.exists(unit_predictor_path):
        try:
            unit_predictor_obj = SingleUnitBinaryPredictorTrainer(
                ppc, active_set_data, T_delta,
                unit_ids=[unit_id],
                hidden_dims=unit_predictor_hidden_dims,
                net_variant=unit_predictor_net_variant,
                tcn_channels=unit_predictor_tcn_channels,
                tcn_depth=unit_predictor_tcn_depth,
                tconv_channels=unit_predictor_tconv_channels,
                tconv_depth=unit_predictor_tconv_depth,
                dropout=unit_predictor_dropout,
                weight_decay=unit_predictor_weight_decay,
                device=nn_device,
            )
            loaded_ok = bool(unit_predictor_obj.load(unit_predictor_path))
            if not loaded_ok:
                raise RuntimeError(
                    f"unit_predictor checkpoint incompatible or empty: {unit_predictor_path}"
                )
            print(
                f"{prefix} 已在子进程加载 unit_predictor: {unit_predictor_path}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{prefix} 加载 unit_predictor 失败，改为禁用覆盖: {exc}",
                flush=True,
            )
            unit_predictor_obj = None
            raise
    effective_use_unit_predictor = unit_predictor_obj is not None

    # 构建 trainer
    if use_sample_parallel:
        # 导入并行 trainer（子进程中重新 import，无问题）
        from uc_NN_subproblem_parallel import ParallelSubproblemSurrogateTrainer
        trainer = ParallelSubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_binary_max=rho_binary_max,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_lower_bound_init=mu_lower_bound_init,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            mu_signed_round_interval=mu_signed_round_interval,
            mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
            mu_sign_flip_min_share=mu_sign_flip_min_share,
            x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            nn_smooth_abs_eps=nn_smooth_abs_eps,
            nn_main_eta_min_ratio=nn_main_eta_min_ratio,
            nn_main_lr_late_scale=nn_main_lr_late_scale,
            nn_main_adam_weight_decay=nn_main_adam_weight_decay,
            nn_main_grad_clip=nn_main_grad_clip,
            nn_main_kkt_lr_scale=nn_main_kkt_lr_scale,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            nn_dual_term_interval=nn_dual_term_interval,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            pg_cost_nn_epochs=pg_cost_nn_epochs,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            pg_cost_batch_strategy=pg_cost_batch_strategy,
            pg_cost_batch_size=pg_cost_batch_size,
            pg_cost_shuffle=pg_cost_shuffle,
            pg_cost_use_sample_weights=pg_cost_use_sample_weights,
            pg_cost_sample_weight_power=pg_cost_sample_weight_power,
            pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
            pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
            pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
            main_direct_train_config=main_direct_train_config,
            c_pg_direct_train_config=c_pg_direct_train_config,
            pg_cost_reg_deadband=pg_cost_reg_deadband,
            pg_cost_softbound_weight=pg_cost_softbound_weight,
            iter_delta_reg_weight=iter_delta_reg_weight,
            iter_delta_reg_deadband=iter_delta_reg_deadband,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            unit_predictor=unit_predictor_obj,
            use_unit_predictor=effective_use_unit_predictor,
            predictor_warmup_rounds=predictor_warmup_rounds,
            sign4_curriculum_rounds=sign4_curriculum_rounds,
            sign4_initial_scale=sign4_initial_scale,
            sign4_final_scale=sign4_final_scale,
            sign4_delay_rounds=sign4_delay_rounds,
            enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
            surrogate_delta_reference_eps=surrogate_delta_reference_eps,
            surrogate_delta_reference_scope=surrogate_delta_reference_scope,
            surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            case_name=case_name,
            device=nn_device,
            n_workers=sample_n_workers,
        )
    else:
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
            lp_backend=lp_backend,
            constraint_generation_strategy=constraint_generation_strategy,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_binary_init=rho_binary_init,
            rho_binary_max=rho_binary_max,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_lower_bound_init=mu_lower_bound_init,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            mu_signed_round_interval=mu_signed_round_interval,
            mu_sign_hysteresis_rounds=mu_sign_hysteresis_rounds,
            mu_sign_flip_min_share=mu_sign_flip_min_share,
            x_bound_dual_zero_rounds=x_bound_dual_zero_rounds,
            ignore_startup_shutdown_costs=ignore_startup_shutdown_costs,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            nn_smooth_abs_eps=nn_smooth_abs_eps,
            nn_main_eta_min_ratio=nn_main_eta_min_ratio,
            nn_main_lr_late_scale=nn_main_lr_late_scale,
            nn_main_adam_weight_decay=nn_main_adam_weight_decay,
            nn_main_grad_clip=nn_main_grad_clip,
            nn_main_kkt_lr_scale=nn_main_kkt_lr_scale,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            nn_dual_term_interval=nn_dual_term_interval,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            pg_cost_nn_epochs=pg_cost_nn_epochs,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            pg_cost_batch_strategy=pg_cost_batch_strategy,
            pg_cost_batch_size=pg_cost_batch_size,
            pg_cost_shuffle=pg_cost_shuffle,
            pg_cost_use_sample_weights=pg_cost_use_sample_weights,
            pg_cost_sample_weight_power=pg_cost_sample_weight_power,
            pg_cost_sample_weight_clip=pg_cost_sample_weight_clip,
            pg_cost_single_sample_reg_scale=pg_cost_single_sample_reg_scale,
            pg_cost_c_pg_adam_weight_decay=pg_cost_c_pg_adam_weight_decay,
            main_direct_train_config=main_direct_train_config,
            c_pg_direct_train_config=c_pg_direct_train_config,
            pg_cost_reg_deadband=pg_cost_reg_deadband,
            pg_cost_softbound_weight=pg_cost_softbound_weight,
            iter_delta_reg_weight=iter_delta_reg_weight,
            iter_delta_reg_deadband=iter_delta_reg_deadband,
            pg_block_prox_weight=pg_block_prox_weight,
            dual_block_prox_weight=dual_block_prox_weight,
            unit_predictor=unit_predictor_obj,
            use_unit_predictor=effective_use_unit_predictor,
            predictor_warmup_rounds=predictor_warmup_rounds,
            sign4_curriculum_rounds=sign4_curriculum_rounds,
            sign4_initial_scale=sign4_initial_scale,
            sign4_final_scale=sign4_final_scale,
            sign4_delay_rounds=sign4_delay_rounds,
            enable_surrogate_delta_reference_lift=enable_surrogate_delta_reference_lift,
            surrogate_delta_reference_eps=surrogate_delta_reference_eps,
            surrogate_delta_reference_scope=surrogate_delta_reference_scope,
            surrogate_delta_reference_min_abs_factor=surrogate_delta_reference_min_abs_factor,
            unit_predictor_finetune_lr=unit_predictor_finetune_lr,
            unit_predictor_weight_decay=unit_predictor_weight_decay,
            case_name=case_name,
            device=nn_device,
        )

    trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        trainer.save(os.path.join(save_dir, f'surrogate_unit_{unit_id}.pth'))

    print(f"{prefix} worker 完成", flush=True)

    return {
        'unit_id':      unit_id,
        'alpha_values': trainer.alpha_values,
        'beta_values':  trainer.beta_values,
        'gamma_values': trainer.gamma_values,
        'delta_values': trainer.delta_values,
        'cost_values':  trainer.cost_values,
        'pg_cost_values': trainer.pg_cost_values,
        'mu':           trainer.mu,
        'rho_primal':   trainer.rho_primal,
        'rho_dual':     trainer.rho_dual,
        'rho_dual_pg':  trainer.rho_dual_pg,
        'rho_dual_x':   trainer.rho_dual_x,
        'rho_dual_coc': trainer.rho_dual_coc,
        'rho_opt':      trainer.rho_opt,
    }


def _precompute_global_lambda_payloads(
    ppc,
    active_set_data: List[Dict],
    T_delta: float,
) -> List[object]:
    """在主进程中预计算所有样本的 global lambda payloads。

    优先从 active_set_data 中读取已有 lambda 字段；
    若缺失则通过 ED 回退求解并重建完整全局对偶载荷。

    Args:
        ppc: PyPower 案例数据。
        active_set_data: 活动集数据列表。
        T_delta: 时间间隔。

    Returns:
        按样本顺序返回 lambda payload 列表。
    """
    n_samples = len(active_set_data)
    T = active_set_data[0]['pd_data'].shape[1]

    # 若所有样本都有 lambda，直接提取
    all_have_lambda = all(
        'lambda' in s and s['lambda'] is not None
        for s in active_set_data
    )
    if all_have_lambda:
        vals = [copy.deepcopy(s['lambda']) for s in active_set_data]
        print(f"✓ 从数据中读取 {n_samples} 个样本的 lambda_vals", flush=True)
        return vals

    print("预计算 global lambda payloads...", flush=True)
    tmp = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta, unit_id=0,
        lambda_predictor=None,
    )
    payloads: List[object] = []
    for sample in active_set_data:
        if (
            'lambda' in sample
            and sample['lambda'] is not None
            and _has_complete_effective_pg_dual(
                sample['lambda'],
                T,
                tmp.ng,
                tmp.branch.shape[0],
            )
        ):
            payloads.append(copy.deepcopy(sample['lambda']))
            continue

        x_sol = _recover_unit_commitment_matrix(sample, tmp.ng, T)
        payloads.append(
            _solve_global_dual_payload_from_ed(
                ppc,
                sample['pd_data'],
                T_delta,
                x_sol,
                tmp.generator_injection_sensitivity,
                renewable_data=sample.get('renewable_data'),
            )
        )
    return payloads


def _precompute_lambda_vals(
    ppc,
    active_set_data: List[Dict],
    T_delta: float,
    lambda_predictor=None,
    lp_backend: str = LP_BACKEND_GUROBI,
    ignore_fixed_generation_cost: bool = False,
) -> List[object]:
    """Precompute per-unit electricity-price matrices for all samples."""
    n_samples = len(active_set_data)
    if n_samples == 0:
        return []
    ppc_int = ext2int(ppc)
    ng = ppc_int["gen"].shape[0]
    T = int(np.asarray(active_set_data[0]["pd_data"]).shape[1])

    if lambda_predictor is not None:
        predicted_vals: List[np.ndarray] = []
        all_resolved = True
        for sample in active_set_data:
            predicted = lambda_predictor.predict(normalize_sample_arrays(sample))
            effective = _extract_pg_electricity_price_matrix(predicted, T, ng)
            if effective is None:
                all_resolved = False
                break
            predicted_vals.append(effective)
        if all_resolved and len(predicted_vals) == n_samples:
            print(f"✓ 从 lambda_predictor 提取 {n_samples} 个样本的 electricity prices", flush=True)
            return predicted_vals
        print("⚠ lambda_predictor 未提供可直接解析的 electricity prices，回退到缓存/ED 预计算", flush=True)

    cached_vals = [
        _get_sample_pg_electricity_price_matrix(sample, T, ng)
        for sample in active_set_data
    ]
    if all(val is not None for val in cached_vals):
        print(f"✓ 从样本缓存读取 {n_samples} 个 electricity prices", flush=True)
        return [np.asarray(val, dtype=float).copy() for val in cached_vals]

    print("预计算 electricity prices...", flush=True)
    price_vals: List[np.ndarray] = []
    for sample, cached_val in zip(active_set_data, cached_vals):
        if cached_val is not None:
            price_vals.append(np.asarray(cached_val, dtype=float).copy())
            continue
        x_sol = _recover_unit_commitment_matrix(sample, ng, T)
        payload = _solve_pg_electricity_price_from_ed(
            ppc,
            sample['pd_data'],
            T_delta,
            x_sol,
            renewable_data=sample.get('renewable_data'),
            verbose=False,
            lp_backend=lp_backend,
            ignore_fixed_generation_cost=ignore_fixed_generation_cost,
        )
        price_vals.append(payload['lambda_pg_electricity_price'])
    return price_vals


def train_all_surrogates_parallel(
    ppc,
    active_set_data: List[Dict],
    T_delta: float = 1.0,
    lambda_predictor=None,
    lp_backend: str = LP_BACKEND_GUROBI,
    unit_ids: Optional[List[int]] = None,
    max_iter: int = 20,
    nn_epochs: int = 10,
    rho_primal_init: float = 1e-3,
    rho_dual_init: float = 1e-3,
    rho_dual_pg_init: float | None = None,
    rho_dual_x_init: float | None = None,
    rho_dual_coc_init: float | None = None,
    rho_binary_init: float = 1.0,
    rho_binary_max: float = 1e4,
    rho_opt_init: float = 1e-3,
    gamma_base: float = 1e-3,
    mu_lower_bound_init: float = 0.1,
    mu_individual_lower_bound_round: int = 3,
    mu_group_lower_bound_round: int = 50,
    mu_signed_round_interval: int | None = None,
    mu_sign_hysteresis_rounds: int = 2,
    mu_sign_flip_min_share: float = 0.67,
    x_bound_dual_zero_rounds: int = 0,
    ignore_startup_shutdown_costs: bool = False,
    nn_learning_rate: float = 1e-4,
    cost_learning_rate: float = 1e-5,
    nn_batch_strategy: str = "full-batch",
    nn_batch_size: int = 4,
    nn_shuffle: bool = True,
    nn_smooth_abs_eps: float = 1e-6,
    nn_main_eta_min_ratio: float = 0.08,
    nn_main_lr_late_scale: float = 0.42,
    nn_main_adam_weight_decay: float = 1e-4,
    nn_main_grad_clip: float = 0.85,
    nn_main_kkt_lr_scale: float = 1.0,
    loss_ratio_primal: float = 1.0,
    loss_ratio_dual_pg: float = 1.0,
    loss_ratio_dual_x: float = 1.0,
    nn_dual_term_interval: int | None = 1,
    loss_ratio_opt: float = 1.0,
    loss_ratio_reg: float = 1.0,
    pg_cost_nn_epochs: int | None = None,
    pg_cost_start_round: int = 3,
    pg_cost_lr: float = 2e-5,
    pg_cost_surr_lr: float = 5e-5,
    pg_cost_batch_strategy: str | None = None,
    pg_cost_batch_size: int | None = None,
    pg_cost_shuffle: bool | None = None,
    pg_cost_use_sample_weights: bool = True,
    pg_cost_sample_weight_power: float = 1.0,
    pg_cost_sample_weight_clip: float = 10.0,
    pg_cost_single_sample_reg_scale: float | None = None,
    pg_cost_c_pg_adam_weight_decay: float | None = None,
    main_direct_train_config: dict | None = None,
    c_pg_direct_train_config: dict | None = None,
    pg_cost_reg_deadband: float = 0.25,
    pg_cost_softbound_weight: float = 1.0,
    iter_delta_reg_weight: float = 5e-5,
    iter_delta_reg_deadband: float = 0.10,
    pg_block_prox_weight: float = 2e-2,
    dual_block_prox_weight: float = 1e-2,
    constraint_generation_strategy: str = "sensitive",
    save_dir: Optional[str] = None,
    device=None,
    n_workers: Optional[int] = None,
    sample_n_workers: int = 4,
    use_sample_parallel: bool = True,
    unit_predictor_path: Optional[str] = None,
    use_unit_predictor: bool = False,
    predictor_warmup_rounds: int = 0,
    sign4_curriculum_rounds: int = 0,
    sign4_initial_scale: float = 1.0,
    sign4_final_scale: float = 1.0,
    sign4_delay_rounds: int = 0,
    enable_surrogate_delta_reference_lift: bool | None = None,
    surrogate_delta_reference_eps: float = 1e-6,
    surrogate_delta_reference_scope: str = "sign4_only",
    surrogate_delta_reference_min_abs_factor: float = 1e-9,
    unit_predictor_finetune_lr: float = 1e-5,
    unit_predictor_weight_decay: float = 1e-4,
    unit_predictor_hidden_dims: Optional[List[int]] = None,
    unit_predictor_net_variant: str = "mlp",
    unit_predictor_tcn_channels: int = 64,
    unit_predictor_tcn_depth: int = 6,
    unit_predictor_tconv_channels: int = 64,
    unit_predictor_tconv_depth: int = 3,
    unit_predictor_dropout: float = 0.1,
    case_name: str | None = None,
) -> Dict[int, dict]:
    """并行训练所有机组的子问题代理约束（Level 1：机组级进程并行）。

    注意：并发进程数受 Gurobi 许可证限制，默认上限为 4。
    若持有学术或完整许可证，可适当增大 n_workers。

    Args:
        ppc: PyPower 案例数据。
        active_set_data: 活动集数据列表。
        T_delta: 时间间隔。
        lambda_predictor: 已训练的对偶变量预测器；若传入，在主进程提取 lambda_vals。
        unit_ids: 要训练的机组 ID 列表，默认所有机组。
        max_iter: BCD 最大迭代次数。
        nn_epochs: 每次 BCD 迭代内 NN 训练轮数。
        save_dir: 模型保存目录（可选）。
        constraint_generation_strategy: 与 ``SubproblemSurrogateTrainer`` 一致；须与主进程
            ``run_surrogate`` / ``load_trained_models`` 所用策略相同，否则保存的 pth 元数据会不匹配。
        device: 计算设备（子进程重新初始化，此参数目前未传递至子进程）。
        n_workers: 并发进程数，默认 min(len(unit_ids), 4)（Gurobi 许可证限制）。
        sample_n_workers: 每个进程内的样本级线程数（Level 2）。
        use_sample_parallel: 是否在子进程内启用 Level 2 线程并行。

    Returns:
        {unit_id: state_dict} 字典，state_dict 含 alpha/beta/gamma/delta/mu/rho。
    """
    from pypower.ext2int import ext2int
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]
    lp_backend = normalize_lp_backend(lp_backend)

    if unit_ids is None:
        unit_ids = list(range(ng))

    # Gurobi 许可证限制：默认并发进程上限 4
    MAX_GUROBI_CONCURRENT = 4
    if n_workers is None:
        if lp_backend == LP_BACKEND_CVXPY_HIGHS:
            n_workers = min(len(unit_ids), os.cpu_count() or 4)
        else:
            n_workers = min(len(unit_ids), MAX_GUROBI_CONCURRENT)
    n_workers = max(1, n_workers)
    sample_n_workers = max(1, int(sample_n_workers))
    use_sample_parallel = bool(use_sample_parallel)

    if (
        lp_backend == LP_BACKEND_CVXPY_HIGHS
        and use_sample_parallel
        and n_workers > 1
        and sample_n_workers > 1
    ):
        print(
            "cvxpy_highs detected nested process parallelism; "
            "disabling sample-level parallelism for multi-unit execution.",
            flush=True,
        )
        use_sample_parallel = False
        sample_n_workers = 1

    print("=" * 60, flush=True)
    print(
        f"并行训练所有机组代理约束 "
        f"({len(unit_ids)} 个机组, n_workers={n_workers})",
        flush=True,
    )
    print("=" * 60, flush=True)

    # 构造每个 worker 的参数 dict
    lambda_vals = _precompute_lambda_vals(
        ppc,
        active_set_data,
        T_delta,
        lambda_predictor=lambda_predictor,
        lp_backend=lp_backend,
        ignore_fixed_generation_cost=bool(ignore_startup_shutdown_costs),
    )

    worker_args = [
        {
            'ppc':                ppc,
            'active_set_data':    active_set_data,
            'lambda_vals':        lambda_vals,
            'unit_id':            g,
            'T_delta':            T_delta,
            'max_iter':           max_iter,
            'nn_epochs':          nn_epochs,
            'rho_primal_init':    rho_primal_init,
            'rho_dual_init':      rho_dual_init,
            'rho_dual_pg_init':   rho_dual_pg_init,
            'rho_dual_x_init':    rho_dual_x_init,
            'rho_dual_coc_init':  rho_dual_coc_init,
            'rho_binary_init':    rho_binary_init,
            'rho_binary_max':     rho_binary_max,
            'rho_opt_init':       rho_opt_init,
            'gamma_base':         gamma_base,
            'mu_lower_bound_init': mu_lower_bound_init,
            'mu_individual_lower_bound_round': mu_individual_lower_bound_round,
            'mu_group_lower_bound_round': mu_group_lower_bound_round,
            'mu_signed_round_interval': mu_signed_round_interval,
            'mu_sign_hysteresis_rounds': mu_sign_hysteresis_rounds,
            'mu_sign_flip_min_share': mu_sign_flip_min_share,
            'x_bound_dual_zero_rounds': x_bound_dual_zero_rounds,
            'ignore_startup_shutdown_costs': ignore_startup_shutdown_costs,
            'nn_learning_rate':   nn_learning_rate,
            'cost_learning_rate': cost_learning_rate,
            'nn_batch_strategy':  nn_batch_strategy,
            'nn_batch_size':      nn_batch_size,
            'nn_shuffle':         nn_shuffle,
            'nn_smooth_abs_eps':  nn_smooth_abs_eps,
            'nn_main_eta_min_ratio': nn_main_eta_min_ratio,
            'nn_main_lr_late_scale': nn_main_lr_late_scale,
            'nn_main_adam_weight_decay': nn_main_adam_weight_decay,
            'nn_main_grad_clip': nn_main_grad_clip,
            'nn_main_kkt_lr_scale': nn_main_kkt_lr_scale,
            'loss_ratio_primal':  loss_ratio_primal,
            'loss_ratio_dual_pg': loss_ratio_dual_pg,
            'loss_ratio_dual_x':  loss_ratio_dual_x,
            'nn_dual_term_interval': nn_dual_term_interval,
            'loss_ratio_opt':     loss_ratio_opt,
            'loss_ratio_reg':     loss_ratio_reg,
            'pg_cost_nn_epochs':  pg_cost_nn_epochs,
            'pg_cost_start_round': pg_cost_start_round,
            'pg_cost_lr':         pg_cost_lr,
            'pg_cost_surr_lr':    pg_cost_surr_lr,
            'pg_cost_batch_strategy': pg_cost_batch_strategy,
            'pg_cost_batch_size': pg_cost_batch_size,
            'pg_cost_shuffle':    pg_cost_shuffle,
            'pg_cost_use_sample_weights': pg_cost_use_sample_weights,
            'pg_cost_sample_weight_power': pg_cost_sample_weight_power,
            'pg_cost_sample_weight_clip': pg_cost_sample_weight_clip,
            'pg_cost_single_sample_reg_scale': pg_cost_single_sample_reg_scale,
            'pg_cost_c_pg_adam_weight_decay': pg_cost_c_pg_adam_weight_decay,
            'main_direct_train_config': main_direct_train_config,
            'c_pg_direct_train_config': c_pg_direct_train_config,
            'pg_cost_reg_deadband': pg_cost_reg_deadband,
            'pg_cost_softbound_weight': pg_cost_softbound_weight,
            'iter_delta_reg_weight': iter_delta_reg_weight,
            'iter_delta_reg_deadband': iter_delta_reg_deadband,
            'pg_block_prox_weight': pg_block_prox_weight,
            'dual_block_prox_weight': dual_block_prox_weight,
            'constraint_generation_strategy': constraint_generation_strategy,
            'sample_n_workers':   sample_n_workers,
            'use_sample_parallel': use_sample_parallel,
            'lp_backend':         lp_backend,
            'save_dir':           save_dir,
            # n_workers>1 时子进程内强制 NN 用 CPU，见 _train_unit_worker
            'force_nn_cpu_for_multiprocess': (n_workers > 1),
            # 单机组 0/1 预测器（通过磁盘路径 lazy 加载，避免跨进程 pickle 模型）
            'unit_predictor_path': unit_predictor_path,
            'use_unit_predictor':  bool(use_unit_predictor and unit_predictor_path),
            'predictor_warmup_rounds': predictor_warmup_rounds,
            'sign4_curriculum_rounds': sign4_curriculum_rounds,
            'sign4_initial_scale': sign4_initial_scale,
            'sign4_final_scale': sign4_final_scale,
            'sign4_delay_rounds': sign4_delay_rounds,
            'enable_surrogate_delta_reference_lift': enable_surrogate_delta_reference_lift,
            'surrogate_delta_reference_eps': surrogate_delta_reference_eps,
            'surrogate_delta_reference_scope': surrogate_delta_reference_scope,
            'surrogate_delta_reference_min_abs_factor': surrogate_delta_reference_min_abs_factor,
            'unit_predictor_finetune_lr': unit_predictor_finetune_lr,
            'unit_predictor_weight_decay': unit_predictor_weight_decay,
            'unit_predictor_hidden_dims': unit_predictor_hidden_dims,
            'unit_predictor_net_variant': unit_predictor_net_variant,
            'unit_predictor_tcn_channels': unit_predictor_tcn_channels,
            'unit_predictor_tcn_depth': unit_predictor_tcn_depth,
            'unit_predictor_tconv_channels': unit_predictor_tconv_channels,
            'unit_predictor_tconv_depth': unit_predictor_tconv_depth,
            'unit_predictor_dropout': unit_predictor_dropout,
            'case_name': case_name,
        }
        for g in unit_ids
    ]

    results: Dict[int, dict] = {}
    failures: Dict[int, Exception] = {}

    executor = ProcessPoolExecutor(max_workers=n_workers)
    wait_on_close = True
    cancel_pending = False
    try:
        future_to_uid = {
            executor.submit(_train_unit_worker, args): args['unit_id']
            for args in worker_args
        }
        for future in as_completed(future_to_uid):
            uid = future_to_uid[future]
            try:
                state = future.result()
                results[uid] = state
                print(f"✓ 机组 {uid} 并行训练完成", flush=True)
            except Exception as exc:
                print(f"✗ 机组 {uid} 训练失败: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                failures[uid] = exc
    except KeyboardInterrupt:
        print(
            "\n收到 KeyboardInterrupt：正在以 wait=False 关闭进程池；"
            "已在求解器内核中运行的子任务可能仍短暂存活，必要时请对进程组使用 kill -9。",
            flush=True,
        )
        wait_on_close = False
        cancel_pending = True
        raise
    finally:
        # 避免仅用 with：中断时默认 shutdown(wait=True) 会长时间卡在 join 子进程上
        executor.shutdown(wait=wait_on_close, cancel_futures=cancel_pending)
    if failures or len(results) != len(unit_ids):
        missing_units = set(int(u) for u in unit_ids) - set(results.keys())
        failed_units = sorted(set(failures.keys()) | missing_units)
        raise RuntimeError(
            f"parallel subproblem training incomplete: {len(results)}/{len(unit_ids)} succeeded; "
            f"failed_units={failed_units}"
        )

    print(f"\n✓ 所有机组并行训练完成 ({len(results)}/{len(unit_ids)})", flush=True)
    return results


# ════════════════════════════════════════════════════════════════════
# __main__：功能验证与加速比对比
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from datetime import datetime

    # ── 路径设置 ─────────────────────────────────────────────────────
    _src = str(Path(__file__).resolve().parent)
    _root = str(Path(__file__).resolve().parent.parent)
    for _p in [_src, _root]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        import pypower.case39
        ppc = pypower.case39.case39()
    except ImportError:
        print("错误: pypower 未安装", flush=True)
        sys.exit(1)

    # ── 测试参数 ─────────────────────────────────────────────────────
    T         = 8
    N_SAMPLES = 5
    UNIT_IDS  = [0, 1, 2]
    MAX_ITER  = 7      # 保持迭代次数短，聚焦并行效果验证
    NN_EPOCHS = 5

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_base = _root + f'/result/subproblem_models/parallel_test_{ts}'
    os.makedirs(save_base, exist_ok=True)

    print("=" * 60, flush=True)
    print(
        f"并行训练验证  T={T}, n_samples={N_SAMPLES}, "
        f"units={UNIT_IDS}",
        flush=True,
    )
    print(f"max_iter={MAX_ITER}, nn_epochs={NN_EPOCHS}", flush=True)
    print("=" * 60, flush=True)

    # ── 生成测试数据 ─────────────────────────────────────────────────
    active_set_data = generate_test_data(ppc, T=T, n_samples=N_SAMPLES, seed=42)

    # 注入零值 lambda，避免子进程/线程调用 CVXPY 求 LP
    T_actual = active_set_data[0]['pd_data'].shape[1]
    for sample in active_set_data:
        if 'lambda' not in sample:
            sample['lambda'] = np.zeros(T_actual).tolist()

    timings: Dict[str, float] = {}

    # ── 1. 串行训练（baseline） ──────────────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【串行训练】", flush=True)
    data_serial = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    serial_trainers = train_all_subproblem_surrogates(
        ppc, data_serial, T_delta=1.0,
        lambda_predictor=None,
        unit_ids=UNIT_IDS,
        max_iter=MAX_ITER, nn_epochs=NN_EPOCHS,
        save_dir=os.path.join(save_base, 'serial'),
    )
    timings['serial'] = time.perf_counter() - t0
    print(f"串行完成，耗时 {timings['serial']:.2f}s", flush=True)

    # ── 2. 机组级进程并行（Level 1，不含 Level 2） ───────────────────
    print("\n" + "─" * 50, flush=True)
    print("【机组级进程并行（Level 1）】", flush=True)
    data_proc = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    results_l1 = train_all_surrogates_parallel(
        ppc, data_proc, T_delta=1.0,
        lambda_predictor=None,
        unit_ids=UNIT_IDS,
        max_iter=MAX_ITER, nn_epochs=NN_EPOCHS,
        save_dir=os.path.join(save_base, 'parallel_l1'),
        n_workers=min(len(UNIT_IDS), 4),
        use_sample_parallel=False,   # Level 1 only
    )
    timings['level1'] = time.perf_counter() - t0
    print(f"Level-1 并行完成，耗时 {timings['level1']:.2f}s", flush=True)

    # ── 3. 样本级线程并行（Level 2，单机组验证） ─────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【样本级线程并行（Level 2，单机组 unit=0）】", flush=True)
    data_thread = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    trainer_l2 = ParallelSubproblemSurrogateTrainer(
        ppc, data_thread, 1.0, unit_id=0,
        lambda_predictor=None,
        n_workers=min(N_SAMPLES, 4),
    )
    trainer_l2.iter(max_iter=MAX_ITER, nn_epochs=NN_EPOCHS)
    timings['level2_single'] = time.perf_counter() - t0
    print(f"Level-2 并行（单机组）完成，耗时 {timings['level2_single']:.2f}s", flush=True)

    # ── 4. 双层并行（Level 1 + Level 2） ─────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【双层并行（Level 1 + Level 2）】", flush=True)
    data_both = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    results_both = train_all_surrogates_parallel(
        ppc, data_both, T_delta=1.0,
        lambda_predictor=None,
        unit_ids=UNIT_IDS,
        max_iter=MAX_ITER, nn_epochs=NN_EPOCHS,
        save_dir=os.path.join(save_base, 'parallel_both'),
        n_workers=min(len(UNIT_IDS), 4),
        sample_n_workers=min(N_SAMPLES, 4),
        use_sample_parallel=True,    # Level 1 + Level 2
    )
    timings['level1_l2'] = time.perf_counter() - t0
    print(f"双层并行完成，耗时 {timings['level1_l2']:.2f}s", flush=True)

    # ── 结果验证：检查并行结果量级 ───────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("结果验证（alpha/delta 量级对比，串行 vs Level-1 并行）", flush=True)
    print("=" * 60, flush=True)
    for uid in UNIT_IDS:
        s_trainer = serial_trainers.get(uid)
        p_state   = results_l1.get(uid)
        if s_trainer is None or p_state is None:
            continue
        s_alpha = float(np.mean(np.abs(s_trainer.alpha_values)))
        p_alpha = float(np.mean(np.abs(p_state['alpha_values'])))
        s_delta = float(np.mean(s_trainer.delta_values))
        p_delta = float(np.mean(p_state['delta_values']))
        print(
            f"  Unit-{uid}: "
            f"串行 alpha={s_alpha:.4f} delta={s_delta:.4f} | "
            f"并行 alpha={p_alpha:.4f} delta={p_delta:.4f}",
            flush=True,
        )

    # ── 加速比汇总表 ─────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("加速比汇总表", flush=True)
    print("=" * 60, flush=True)
    ref = timings['serial']
    rows = [
        ("串行",               timings['serial'],        1.0),
        ("Level-1 进程并行",   timings['level1'],        ref / timings['level1']),
        ("Level-2 线程(单机)", timings['level2_single'], ref / timings['level2_single']),
        ("Level-1+2 双层",     timings['level1_l2'],     ref / timings['level1_l2']),
    ]
    print(f"{'模式':<20} {'耗时(s)':>10} {'加速比':>10}", flush=True)
    print("-" * 44, flush=True)
    for name, t, speedup in rows:
        print(f"{name:<20} {t:>10.2f} {speedup:>10.2f}x", flush=True)

    print(f"\n结果已保存至: {save_base}", flush=True)
    print("验证完成！", flush=True)
