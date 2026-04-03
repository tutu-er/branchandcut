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
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scenario_utils import normalize_sample_arrays

# ── 路径设置（worker 进程也需要能 import src.*）──────────────
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uc_NN_subproblem import (
    SubproblemSurrogateTrainer,
    _extract_pg_electricity_price_matrix,
    _get_sample_pg_electricity_price_matrix,
    _recover_unit_commitment_matrix,
    _solve_pg_electricity_price_from_ed,
    generate_test_data,
    train_all_subproblem_surrogates,
)


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
        constraint_generation_strategy: str = "sensitive",
        rho_primal_init: float = 1e-3,
        rho_dual_init: float = 1e-3,
        rho_dual_pg_init: float | None = None,
        rho_dual_x_init: float | None = None,
        rho_dual_coc_init: float | None = None,
        rho_opt_init: float = 1e-3,
        gamma_base: float = 1e-3,
        mu_lower_bound_init: float = 0.1,
        mu_individual_lower_bound_round: int = 3,
        mu_group_lower_bound_round: int = 50,
        pg_cost_start_round: int = 3,
        pg_cost_scale_multiplier: float = 1.2,
        nn_hidden_dims: list[int] | None = None,
        nn_learning_rate: float = 1e-4,
        cost_learning_rate: float = 1e-5,
        pg_cost_lr: float = 2e-5,
        pg_cost_surr_lr: float = 5e-5,
        nn_batch_strategy: str = "full-batch",
        nn_batch_size: int = 4,
        nn_shuffle: bool = True,
        pg_cost_reg_deadband: float = 0.25,
        loss_ratio_primal: float = 1.0,
        loss_ratio_dual_pg: float = 1.0,
        loss_ratio_dual_x: float = 1.0,
        loss_ratio_opt: float = 1.0,
        loss_ratio_reg: float = 1.0,
        device=None,
        n_workers: int = 4,
    ):
        super().__init__(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=lambda_predictor,
            max_constraints=max_constraints,
            constraint_generation_strategy=constraint_generation_strategy,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_lower_bound_init=mu_lower_bound_init,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            pg_cost_start_round=pg_cost_start_round,
            pg_cost_scale_multiplier=pg_cost_scale_multiplier,
            nn_hidden_dims=nn_hidden_dims,
            nn_learning_rate=nn_learning_rate,
            cost_learning_rate=cost_learning_rate,
            pg_cost_lr=pg_cost_lr,
            pg_cost_surr_lr=pg_cost_surr_lr,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
            pg_cost_reg_deadband=pg_cost_reg_deadband,
            loss_ratio_primal=loss_ratio_primal,
            loss_ratio_dual_pg=loss_ratio_dual_pg,
            loss_ratio_dual_x=loss_ratio_dual_x,
            loss_ratio_opt=loss_ratio_opt,
            loss_ratio_reg=loss_ratio_reg,
            device=device,
        )
        self.n_workers = min(n_workers, self.n_samples)

    def iter(
        self,
        max_iter: int = 20,
        nn_epochs: int = 10,
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

        EPS = 1e-10
        gamma = self.gamma_base / (self.n_samples * max_iter)
        gamma_dual = gamma * self.gamma_dual_component_scale
        self.gamma = gamma

        for i in range(max_iter):
            print(f"{prefix} 🔄 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i

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
                        primal_results[s] = (None, None, None, None)

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
                        dual_results[s] = (None, None)

            # 顺序写入状态
            for s in range(self.n_samples):
                lambda_inherent_sol, mu_sol = dual_results[s]
                if lambda_inherent_sol is not None:
                    self.lambda_inherent[s] = lambda_inherent_sol
                    self.mu[s] = self._apply_mu_lower_bound_policy(mu_sol, lb_mu)

            # ── 3. NN block（串行，批次化训练） ──────────────────────
            self.iter_with_surrogate_nn(
                num_epochs=nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
                cost_learning_rate=cost_learning_rate,
            )
            self.iter_with_c_pg_nn(
                num_epochs=nn_epochs,
                batch_size=nn_batch_size,
                batch_strategy=nn_batch_strategy,
                shuffle=nn_shuffle,
                learning_rate=pg_cost_surr_learning_rate,
            )

            # ── 计算并打印违反量 ─────────────────────────────────────
            obj_primal, obj_dual_pg, obj_dual_x, obj_dual_coc, obj_dual, obj_opt = self.cal_viol_components()
            obj_primal = obj_primal if abs(obj_primal) >= 1e-12 else 0.0
            obj_dual_pg = obj_dual_pg if abs(obj_dual_pg) >= 1e-12 else 0.0
            obj_dual_x = obj_dual_x if abs(obj_dual_x) >= 1e-12 else 0.0
            obj_dual_coc = obj_dual_coc if abs(obj_dual_coc) >= 1e-12 else 0.0
            obj_dual   = obj_dual   if abs(obj_dual)   >= 1e-12 else 0.0
            obj_opt    = obj_opt    if abs(obj_opt)    >= 1e-12 else 0.0

            print(
                f"{prefix}   obj_primal={obj_primal:.6f}, "
                f"obj_dual_pg={obj_dual_pg:.6f}, obj_dual_x={obj_dual_x:.6f}, "
                f"obj_dual_coc={obj_dual_coc:.6f}, obj_dual={obj_dual:.6f}, obj_opt={obj_opt:.6f}",
                flush=True,
            )

            if i >= 3:
                self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
                self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
                self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
                self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
                self._sync_rho_dual_summary()
                self.rho_opt    = min(self.rho_opt    + gamma * obj_opt,    self.rho_max)
            
            print(f"{prefix}   ρ_primal={self.rho_primal:.4f}, ρ_dual={self.rho_dual:.4f}, ρ_opt={self.rho_opt:.4f}", flush=True)

        print(f"{prefix} ✓ 样本级并行训练完成", flush=True)


# ════════════════════════════════════════════════════════════════════
# Level 1：机组级并行（进程）
# ════════════════════════════════════════════════════════════════════

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

    from uc_NN_subproblem import SubproblemSurrogateTrainer

    unit_id             = args['unit_id']
    ppc                 = args['ppc']
    active_set_data     = copy.deepcopy(args['active_set_data'])
    lambda_vals         = args.get('lambda_vals')
    T_delta             = args['T_delta']
    max_iter            = args['max_iter']
    nn_epochs           = args['nn_epochs']
    gamma_base          = args.get('gamma_base', 1e-3)
    mu_individual_lower_bound_round = args.get('mu_individual_lower_bound_round', 3)
    mu_group_lower_bound_round = args.get('mu_group_lower_bound_round', 50)
    sample_n_workers    = args.get('sample_n_workers', 4)
    use_sample_parallel = args.get('use_sample_parallel', True)
    save_dir            = args.get('save_dir')

    prefix = f"[Unit-{unit_id}]"
    print(f"{prefix} worker 启动 (pid={os.getpid()})", flush=True)

    # 将预计算的 lambda_vals 注入 active_set_data，避免子进程重复求解 LP
    if lambda_vals is not None:
        for i, sample in enumerate(active_set_data):
            sample['lambda_pg_electricity_price'] = copy.deepcopy(lambda_vals[i])

    # 构建 trainer
    if use_sample_parallel:
        # 导入并行 trainer（子进程中重新 import，无问题）
        from uc_NN_subproblem_parallel import ParallelSubproblemSurrogateTrainer
        trainer = ParallelSubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
            gamma_base=gamma_base,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
            n_workers=sample_n_workers,
        )
    else:
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
            gamma_base=gamma_base,
            mu_individual_lower_bound_round=mu_individual_lower_bound_round,
            mu_group_lower_bound_round=mu_group_lower_bound_round,
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


def _precompute_lambda_vals(
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
) -> List[object]:
    """Precompute per-unit electricity-price matrices for all samples."""
    n_samples = len(active_set_data)
    tmp = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta, unit_id=0,
        lambda_predictor=None,
    )
    T = tmp.T
    ng = tmp.ng

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
        print("⚠ lambda_predictor 未提供新的电价格式，回退到手动 ED 预计算", flush=True)

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
        )
        price_vals.append(payload['lambda_pg_electricity_price'])
    return price_vals


def train_all_surrogates_parallel(
    ppc,
    active_set_data: List[Dict],
    T_delta: float = 1.0,
    lambda_predictor=None,
    unit_ids: Optional[List[int]] = None,
    max_iter: int = 20,
    nn_epochs: int = 10,
    gamma_base: float = 1e-3,
    mu_individual_lower_bound_round: int = 3,
    mu_group_lower_bound_round: int = 50,
    save_dir: Optional[str] = None,
    device=None,
    n_workers: Optional[int] = None,
    sample_n_workers: int = 4,
    use_sample_parallel: bool = True,
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

    if unit_ids is None:
        unit_ids = list(range(ng))

    # Gurobi 许可证限制：默认并发进程上限 4
    MAX_GUROBI_CONCURRENT = 4
    if n_workers is None:
        n_workers = min(len(unit_ids), MAX_GUROBI_CONCURRENT)
    n_workers = max(1, n_workers)

    print("=" * 60, flush=True)
    print(
        f"并行训练所有机组代理约束 "
        f"({len(unit_ids)} 个机组, n_workers={n_workers})",
        flush=True,
    )
    print("=" * 60, flush=True)

    # 预计算 lambda_vals（主进程一次，避免每个 worker 重复计算）
    if lambda_predictor is not None:
        n_samples = len(active_set_data)
        lambda_vals = [
            lambda_predictor.predict(normalize_sample_arrays(active_set_data[i]))
            for i in range(n_samples)
        ]
        print(
            f"✓ 从 lambda_predictor 提取 {len(lambda_vals)} 个样本的 global lambda payloads",
            flush=True,
        )
    else:
        lambda_vals = _precompute_lambda_vals(ppc, active_set_data, T_delta)

    # 构造每个 worker 的参数 dict
    lambda_vals = _precompute_lambda_vals(
        ppc,
        active_set_data,
        T_delta,
        lambda_predictor=lambda_predictor,
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
            'gamma_base':         gamma_base,
            'mu_individual_lower_bound_round': mu_individual_lower_bound_round,
            'mu_group_lower_bound_round': mu_group_lower_bound_round,
            'sample_n_workers':   sample_n_workers,
            'use_sample_parallel': use_sample_parallel,
            'save_dir':           save_dir,
        }
        for g in unit_ids
    ]

    results: Dict[int, dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
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
