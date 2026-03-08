"""
uc_NN_subproblem_v3_parallel.py
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
    from uc_NN_subproblem_v3_parallel import (
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

# ── 路径设置（worker 进程也需要能 import src.*）──────────────
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uc_NN_subproblem_v3 import (
    SubproblemSurrogateTrainer,
    _extract_lambda_power_balance,
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
        device=None,
        n_workers: int = 4,
    ):
        super().__init__(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=lambda_predictor,
            max_constraints=max_constraints,
            device=device,
        )
        self.n_workers = min(n_workers, self.n_samples)

    def iter(self, max_iter: int = 20, nn_epochs: int = 10):
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
                )
                for s in range(self.n_samples)
            ]

            primal_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_primal_block,
                        s, alphas, betas, gammas, deltas,
                    ): s
                    for s, alphas, betas, gammas, deltas in primal_args
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
            lb_mu = 0.0 if self.iter_number >= 50 else self.mu_lower_bound

            dual_args = [
                (
                    s,
                    self.alpha_values[s].copy(),
                    self.beta_values[s].copy(),
                    self.gamma_values[s].copy(),
                    self.delta_values[s].copy(),
                )
                for s in range(self.n_samples)
            ]

            dual_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_dual_block,
                        s, alphas, betas, gammas, deltas,
                    ): s
                    for s, alphas, betas, gammas, deltas in dual_args
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
                    self.mu[s] = np.maximum(mu_sol, lb_mu)

            # ── 3. NN block（串行，批次化训练） ──────────────────────
            self.iter_with_surrogate_nn(num_epochs=nn_epochs)

            # ── 计算并打印违反量 ─────────────────────────────────────
            obj_primal, obj_dual, obj_opt = self.cal_viol()
            obj_primal = obj_primal if abs(obj_primal) >= 1e-12 else 0.0
            obj_dual   = obj_dual   if abs(obj_dual)   >= 1e-12 else 0.0
            obj_opt    = obj_opt    if abs(obj_opt)    >= 1e-12 else 0.0

            print(
                f"{prefix}   obj_primal={obj_primal:.6f}, "
                f"obj_dual={obj_dual:.6f}, obj_opt={obj_opt:.6f}",
                flush=True,
            )

            self.rho_primal += self.gamma * obj_primal
            self.rho_dual   += self.gamma * obj_dual
            self.rho_opt    += self.gamma * obj_opt

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

    from uc_NN_subproblem_v3 import SubproblemSurrogateTrainer

    unit_id             = args['unit_id']
    ppc                 = args['ppc']
    active_set_data     = copy.deepcopy(args['active_set_data'])
    lambda_vals         = args.get('lambda_vals')   # (n_samples, T) ndarray or None
    T_delta             = args['T_delta']
    max_iter            = args['max_iter']
    nn_epochs           = args['nn_epochs']
    sample_n_workers    = args.get('sample_n_workers', 4)
    use_sample_parallel = args.get('use_sample_parallel', True)
    save_dir            = args.get('save_dir')

    prefix = f"[Unit-{unit_id}]"
    print(f"{prefix} worker 启动 (pid={os.getpid()})", flush=True)

    # 将预计算的 lambda_vals 注入 active_set_data，避免子进程重复求解 LP
    if lambda_vals is not None:
        for i, sample in enumerate(active_set_data):
            sample['lambda'] = lambda_vals[i].tolist()

    # 构建 trainer
    if use_sample_parallel:
        # 导入并行 trainer（子进程中重新 import，无问题）
        from uc_NN_subproblem_v3_parallel import ParallelSubproblemSurrogateTrainer
        trainer = ParallelSubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
            n_workers=sample_n_workers,
        )
    else:
        trainer = SubproblemSurrogateTrainer(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=None,
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
        'mu':           trainer.mu,
        'rho_primal':   trainer.rho_primal,
        'rho_dual':     trainer.rho_dual,
        'rho_opt':      trainer.rho_opt,
    }


def _precompute_lambda_vals(
    ppc,
    active_set_data: List[Dict],
    T_delta: float,
) -> np.ndarray:
    """在主进程中预计算所有样本的 lambda_vals (n_samples, T)。

    优先从 active_set_data 中读取已有 lambda 字段；
    若缺失则通过创建临时 trainer（unit=0）触发 _solve_for_lambda()。

    Args:
        ppc: PyPower 案例数据。
        active_set_data: 活动集数据列表。
        T_delta: 时间间隔。

    Returns:
        lambda_vals 数组，shape (n_samples, T)。
    """
    n_samples = len(active_set_data)
    T = active_set_data[0]['pd_data'].shape[1]

    # 若所有样本都有 lambda，直接提取
    all_have_lambda = all(
        'lambda' in s and s['lambda'] is not None
        for s in active_set_data
    )
    if all_have_lambda:
        vals = np.array([
            _extract_lambda_power_balance(s['lambda'], T)
            for s in active_set_data
        ])
        print(f"✓ 从数据中读取 {n_samples} 个样本的 lambda_vals", flush=True)
        return vals

    # 否则创建 unit=0 的临时 trainer 触发 lambda 计算（各机组共享同一组 lambda）
    print("预计算 lambda_vals（unit=0 临时 trainer）...", flush=True)
    tmp = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta, unit_id=0,
        lambda_predictor=None,
    )
    return tmp.lambda_vals.copy()


def train_all_surrogates_parallel(
    ppc,
    active_set_data: List[Dict],
    T_delta: float = 1.0,
    lambda_predictor=None,
    unit_ids: Optional[List[int]] = None,
    max_iter: int = 20,
    nn_epochs: int = 10,
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
        lambda_vals = np.array([
            lambda_predictor.predict(active_set_data[i]['pd_data'])
            for i in range(n_samples)
        ])
        print(
            f"✓ 从 lambda_predictor 提取 lambda_vals shape={lambda_vals.shape}",
            flush=True,
        )
    else:
        lambda_vals = _precompute_lambda_vals(ppc, active_set_data, T_delta)

    # 构造每个 worker 的参数 dict
    worker_args = [
        {
            'ppc':                ppc,
            'active_set_data':    active_set_data,
            'lambda_vals':        lambda_vals,
            'unit_id':            g,
            'T_delta':            T_delta,
            'max_iter':           max_iter,
            'nn_epochs':          nn_epochs,
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
