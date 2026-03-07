"""
uc_NN_subproblem_v3_parallel.py

SubproblemSurrogateTrainer 并行训练版本（新建文件，不修改原始实现）。

并行层级设计：
  Level 1 — 机组级并行（ProcessPoolExecutor）：
    各机组完全独立，每个 worker 进程独立构建 trainer 并运行 iter()，
    返回可序列化的结果 dict 由主进程汇总。

  Level 2 — 样本级并行（ThreadPoolExecutor）：
    继承 SubproblemSurrogateTrainer，重写 iter()，
    primal/dual block 内并发提交各样本的 Gurobi 求解。
    Gurobi 在求解时释放 GIL，线程并发有效。
    NN block 保持串行（批次化训练）。

注意：并发 Gurobi 进程数受许可证限制，默认不超过 4。

用法：
    python src/uc_NN_subproblem_v3_parallel.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np

# 将项目根目录添加到 sys.path（src/ 下所有文件均需此步骤）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_SRC_DIR = str(Path(__file__).resolve().parent)
for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.uc_NN_subproblem_v3 import (
    SubproblemSurrogateTrainer,
    generate_test_data,
    load_active_set_from_json,
    train_all_subproblem_surrogates,
    _extract_lambda_power_balance,
    TORCH_AVAILABLE,
)
from pypower.ext2int import ext2int

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


# ========================== Level 2: 样本级并行（线程）==========================

class ParallelSubproblemSurrogateTrainer(SubproblemSurrogateTrainer):
    """
    样本级线程并行版的 SubproblemSurrogateTrainer。

    重写 iter()：primal/dual block 内用 ThreadPoolExecutor 并发各样本
    的 Gurobi 求解，收集结果后主线程顺序更新共享状态。
    NN block 保持串行（批次训练，无需改动）。

    Args:
        n_workers: 线程数，默认 min(n_samples, 4)。
        其余参数同 SubproblemSurrogateTrainer。
    """

    def __init__(
        self,
        ppc,
        active_set_data,
        T_delta: float,
        unit_id: int,
        lambda_predictor=None,
        max_constraints: int = 20,
        device=None,
        n_workers: int = 4,
    ):
        # n_workers 须在 super().__init__() 前赋值，因为父类 __init__ 会打印信息
        self.n_workers = n_workers
        super().__init__(
            ppc, active_set_data, T_delta, unit_id,
            lambda_predictor=lambda_predictor,
            max_constraints=max_constraints,
            device=device,
        )

    # ── 单样本 worker（供线程池调用）────────────────────────────────────────

    def _run_primal_sample(
        self, sample_id: int
    ) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray]]:
        """单样本原始块求解，复制参数数组保证线程安全。"""
        alphas = self.alpha_values[sample_id].copy()
        betas  = self.beta_values[sample_id].copy()
        gammas = self.gamma_values[sample_id].copy()
        deltas = self.delta_values[sample_id].copy()
        pg_sol, x_sol, coc_sol, cpower_sol = self.iter_with_primal_block(
            sample_id, alphas, betas, gammas, deltas
        )
        return sample_id, pg_sol, x_sol, coc_sol, cpower_sol

    def _run_dual_sample(
        self, sample_id: int
    ) -> Tuple[int, Optional[dict], Optional[np.ndarray]]:
        """单样本对偶块求解，复制参数数组保证线程安全。"""
        alphas = self.alpha_values[sample_id].copy()
        betas  = self.beta_values[sample_id].copy()
        gammas = self.gamma_values[sample_id].copy()
        deltas = self.delta_values[sample_id].copy()
        lambda_inherent_sol, mu_sol = self.iter_with_dual_block(
            sample_id, alphas, betas, gammas, deltas
        )
        return sample_id, lambda_inherent_sol, mu_sol

    # ── 主迭代循环（重写）────────────────────────────────────────────────────

    def iter(self, max_iter: int = 20, nn_epochs: int = 10):
        """
        主 BCD 迭代循环（样本级线程并行版本）。

        primal/dual block 内并发提交各样本 Gurobi 求解，
        等所有 future 完成后，主线程按 sample_id 顺序更新共享状态。
        NN block 保持串行。
        """
        n_workers = min(self.n_workers, self.n_samples)
        prefix = f"[Unit-{self.unit_id}]"
        print(
            f"{prefix} 开始并行BCD迭代 (V3, 样本线程数={n_workers})...",
            flush=True,
        )

        EPS = 1e-10

        for i in range(max_iter):
            print(f"{prefix} 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i

            # ── 1. 并行原始块 ──────────────────────────────────────────────
            primal_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self._run_primal_sample, s): s
                    for s in range(self.n_samples)
                }
                for fut in as_completed(futures):
                    s_id, pg_sol, x_sol, coc_sol, cpower_sol = fut.result()
                    if pg_sol is not None:
                        primal_results[s_id] = (pg_sol, x_sol, coc_sol, cpower_sol)

            for s_id, (pg_sol, x_sol, coc_sol, cpower_sol) in primal_results.items():
                self.pg[s_id] = np.where(np.abs(pg_sol) < EPS, 0.0, pg_sol)
                self.x[s_id]  = np.where(np.abs(x_sol)  < EPS, 0.0, x_sol)
                self.x[s_id]  = np.where(np.abs(self.x[s_id] - 1) < EPS, 1.0, self.x[s_id])
                self.coc[s_id]    = np.where(np.abs(coc_sol)    < EPS, 0.0, coc_sol)
                self.cpower[s_id] = np.where(np.abs(cpower_sol) < EPS, 0.0, cpower_sol)

            # ── 2. 并行对偶块 ──────────────────────────────────────────────
            lb_mu = 0.0 if self.iter_number >= 50 else self.mu_lower_bound
            dual_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self._run_dual_sample, s): s
                    for s in range(self.n_samples)
                }
                for fut in as_completed(futures):
                    s_id, lam_inh, mu_sol = fut.result()
                    if lam_inh is not None:
                        dual_results[s_id] = (lam_inh, mu_sol)

            for s_id, (lam_inh, mu_sol) in dual_results.items():
                self.lambda_inherent[s_id] = lam_inh
                self.mu[s_id] = np.maximum(mu_sol, lb_mu)

            # ── 3. NN block（串行）────────────────────────────────────────
            self.iter_with_surrogate_nn(num_epochs=nn_epochs)

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

            print(
                f"{prefix}   rho_p={self.rho_primal:.4f}, "
                f"rho_d={self.rho_dual:.4f}, rho_o={self.rho_opt:.4f}",
                flush=True,
            )

        print(f"{prefix} ✓ V3三时段耦合代理约束训练完成", flush=True)


# ========================== Level 1: 机组级并行（进程）==========================

def _embed_lambda_into_data(
    active_set_data: List[Dict], lambda_vals: np.ndarray
) -> List[Dict]:
    """
    将预计算的 lambda_vals 嵌入 active_set_data 的浅拷贝。

    嵌入后子进程读取到 'lambda' 字段，跳过 LP 求解（优化性能）。
    lambda_vals: (n_samples, T)
    """
    embedded = []
    for i, sample in enumerate(active_set_data):
        s = dict(sample)
        # 以 list 格式嵌入（_extract_lambda_power_balance 支持 list）
        s['lambda'] = lambda_vals[i].tolist()
        embedded.append(s)
    return embedded


def _train_unit_worker(args: dict) -> dict:
    """
    顶层可 pickle 的机组训练 worker 函数（供 ProcessPoolExecutor 调用）。

    每个 worker 进程独立构建 trainer、运行 iter()，返回序列化结果 dict。
    lambda_vals 已由主进程预嵌入 active_set_data，子进程无需重复求解 LP。

    Args:
        args: 包含以下键的字典
            unit_id           (int)  : 机组索引
            ppc               (dict) : PyPower 案例数据（含 numpy arrays，可 pickle）
            active_set_data   (list) : 已嵌入 lambda 的样本数据
            T_delta           (float): 时间间隔，默认 1.0
            max_iter          (int)  : BCD 最大迭代次数
            nn_epochs         (int)  : NN 训练轮数/迭代
            save_dir          (str|None): 可选保存目录
            use_sample_parallel (bool): 是否启用 Level 2 线程并行
            sample_n_workers  (int)  : Level 2 线程数
            max_constraints   (int)  : 最大约束数

    Returns:
        序列化友好的结果 dict，包含 numpy arrays + 标量；
        失败时返回 {'unit_id': ..., 'success': False, 'error': ...}。
    """
    # 子进程重新设置 sys.path
    import sys
    from pathlib import Path
    _root = str(Path(__file__).resolve().parent.parent)
    _src  = str(Path(__file__).resolve().parent)
    for _p in [_root, _src]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import os
    import numpy as np

    unit_id             = args['unit_id']
    ppc                 = args['ppc']
    active_set_data     = args['active_set_data']
    T_delta             = args.get('T_delta', 1.0)
    max_iter            = args.get('max_iter', 20)
    nn_epochs           = args.get('nn_epochs', 10)
    save_dir            = args.get('save_dir')
    use_sample_parallel = args.get('use_sample_parallel', True)
    sample_n_workers    = args.get('sample_n_workers', 4)
    max_constraints     = args.get('max_constraints', 20)

    prefix = f"[Unit-{unit_id}]"
    print(f"{prefix} 子进程启动，开始训练...", flush=True)

    try:
        if use_sample_parallel:
            from src.uc_NN_subproblem_v3_parallel import ParallelSubproblemSurrogateTrainer
            trainer = ParallelSubproblemSurrogateTrainer(
                ppc, active_set_data, T_delta, unit_id,
                lambda_predictor=None,
                max_constraints=max_constraints,
                n_workers=sample_n_workers,
            )
        else:
            from src.uc_NN_subproblem_v3 import SubproblemSurrogateTrainer
            trainer = SubproblemSurrogateTrainer(
                ppc, active_set_data, T_delta, unit_id,
                lambda_predictor=None,
                max_constraints=max_constraints,
            )

        trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            trainer.save(os.path.join(save_dir, f'surrogate_unit_{unit_id}.pth'))

        print(f"{prefix} 训练完成", flush=True)

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
            'success':      True,
        }

    except Exception as exc:
        import traceback
        print(f"{prefix} 训练失败: {exc}", flush=True)
        traceback.print_exc()
        return {
            'unit_id': unit_id,
            'success': False,
            'error':   str(exc),
        }


def train_all_surrogates_parallel(
    ppc,
    active_set_data: List[Dict],
    T_delta: float = 1.0,
    lambda_predictor=None,
    unit_ids: List[int] = None,
    max_iter: int = 20,
    nn_epochs: int = 10,
    save_dir: str = None,
    device=None,
    n_workers: int = None,
    sample_n_workers: int = 4,
    use_sample_parallel: bool = True,
) -> Dict[int, dict]:
    """
    机组级并行训练所有机组代理约束（Level 1：进程级并行）。

    流程：
      1. 若提供 lambda_predictor，主进程预计算 lambda_vals 并嵌入数据，
         避免子进程重复求解 LP。
      2. 用 ProcessPoolExecutor 并发各机组 _train_unit_worker。
      3. 收集各进程结果 dict，以 {unit_id: state_dict} 形式返回。

    注意：
      - 并发进程数受 Gurobi 许可证限制，建议 n_workers <= 4。
      - device 参数保留但不传给子进程（CUDA 在 spawn 模式下不安全）。
      - Windows 下 ProcessPoolExecutor 默认使用 spawn，需要
        主脚本有 `if __name__ == '__main__':` 保护。

    Args:
        ppc: PyPower 案例数据
        active_set_data: 活动集数据列表
        T_delta: 时间间隔
        lambda_predictor: 已训练的对偶变量预测器（可选）
        unit_ids: 要训练的机组 ID 列表（默认全部机组）
        max_iter: BCD 最大迭代次数
        nn_epochs: 每次 BCD 迭代中 NN 训练轮数
        save_dir: 模型保存目录（可选）
        device: 保留参数（子进程不使用）
        n_workers: 并行进程数（默认 min(len(unit_ids), cpu_count(), 4)）
        sample_n_workers: 每进程内样本并行线程数（Level 2）
        use_sample_parallel: 是否在子进程内启用 Level 2 线程并行

    Returns:
        {unit_id: state_dict}，成功的机组包含 alpha/beta/gamma/delta/mu 等 numpy arrays；
        失败的机组包含 {'success': False, 'error': ...}。
    """
    ppc_int = ext2int(ppc)
    ng = ppc_int['gen'].shape[0]

    if unit_ids is None:
        unit_ids = list(range(ng))

    # 默认进程数：受许可证限制不超过 4
    if n_workers is None:
        n_workers = min(len(unit_ids), cpu_count(), 4)

    print("=" * 60, flush=True)
    print(
        f"机组级并行训练 ({len(unit_ids)} 个机组, {n_workers} 进程, "
        f"Level2={'启用' if use_sample_parallel else '禁用'})",
        flush=True,
    )
    print("=" * 60, flush=True)

    # 若有 lambda_predictor，主进程预计算并嵌入数据（避免子进程重复 LP 求解）
    if lambda_predictor is not None:
        print("主进程预计算 lambda_vals（使用 predictor）...", flush=True)
        n_samples = len(active_set_data)
        T = active_set_data[0]['pd_data'].shape[1]
        lambda_vals = np.zeros((n_samples, T))
        for i, sample in enumerate(active_set_data):
            lambda_vals[i] = lambda_predictor.predict(sample['pd_data'])
        worker_data = _embed_lambda_into_data(active_set_data, lambda_vals)
    else:
        # 子进程自行从样本数据读取 lambda（或内部求解 LP）
        worker_data = active_set_data

    # 构建每个机组的 worker args
    worker_args = [
        {
            'unit_id':             unit_id,
            'ppc':                 ppc,
            'active_set_data':     worker_data,
            'T_delta':             T_delta,
            'max_iter':            max_iter,
            'nn_epochs':           nn_epochs,
            'save_dir':            save_dir,
            'use_sample_parallel': use_sample_parallel,
            'sample_n_workers':    sample_n_workers,
            'max_constraints':     20,
        }
        for unit_id in unit_ids
    ]

    results: Dict[int, dict] = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_unit = {
            executor.submit(_train_unit_worker, a): a['unit_id']
            for a in worker_args
        }
        for fut in as_completed(future_to_unit):
            uid = future_to_unit[fut]
            try:
                result = fut.result()
                results[uid] = result
                status = "完成" if result.get('success') else f"失败: {result.get('error')}"
                print(f"[主进程] 机组 {uid} {status}", flush=True)
            except Exception as exc:
                print(f"[主进程] 机组 {uid} 进程异常: {exc}", flush=True)
                results[uid] = {'unit_id': uid, 'success': False, 'error': str(exc)}

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r.get('success'))
    print(
        f"\n✓ 机组级并行训练完成，{n_ok}/{len(unit_ids)} 个机组成功，"
        f"耗时 {elapsed:.1f}s",
        flush=True,
    )
    return results


# ========================== 测试主函数 ==========================

def _time_serial_single(ppc, active_set_data: List[Dict],
                         unit_id: int, max_iter: int, nn_epochs: int) -> float:
    """计时：单机组串行训练（基准对比用）。"""
    t0 = time.time()
    trainer = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta=1.0, unit_id=unit_id,
        lambda_predictor=None, max_constraints=20,
    )
    trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)
    return time.time() - t0


def _time_parallel_single(ppc, active_set_data: List[Dict],
                           unit_id: int, max_iter: int, nn_epochs: int,
                           n_sample_workers: int) -> float:
    """计时：单机组样本级并行训练（Level 2 对比）。"""
    t0 = time.time()
    trainer = ParallelSubproblemSurrogateTrainer(
        ppc, active_set_data, T_delta=1.0, unit_id=unit_id,
        lambda_predictor=None, max_constraints=20,
        n_workers=n_sample_workers,
    )
    trainer.iter(max_iter=max_iter, nn_epochs=nn_epochs)
    return time.time() - t0


def main():
    """
    并行训练三种模式的速度对比测试。

    测试流程：
      1. 生成小规模测试数据（case39/case30, T=8, n_samples=5, unit_ids=[0,1,2]）
      2. 串行训练所有机组，记录耗时
      3. 机组级并行训练，记录耗时
      4. 样本级并行训练（机组0），对比单机组串行，记录耗时
      5. 打印加速比对比表
      6. 验证结果并保存计时 JSON

    验证：并行版结果（alpha 量级）与串行版数量级相同。
    """
    # ── 加载测试案例 ──────────────────────────────────────────────────────
    try:
        import pypower.case39
        ppc = pypower.case39.case39()
        case_name = "IEEE 39-bus"
    except ImportError:
        try:
            import pypower.case30
            ppc = pypower.case30.case30()
            case_name = "IEEE 30-bus"
        except ImportError:
            print("pypower 未安装，无法运行测试", flush=True)
            return

    print(f"使用 {case_name} 测试系统", flush=True)

    # ── 测试参数 ──────────────────────────────────────────────────────────
    T          = 8      # 时段数（较短，加快测试）
    n_samples  = 5      # 样本数
    unit_ids   = [0, 1, 2]
    max_iter   = 3      # 少迭代，聚焦速度对比而非收敛质量
    nn_epochs  = 3
    n_proc     = min(len(unit_ids), 3)   # 进程数
    n_thread   = min(n_samples, 4)       # 线程数

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_base = Path(_PROJECT_ROOT) / "result" / "subproblem_models" / f"parallel_test_{ts}"
    save_base.mkdir(parents=True, exist_ok=True)

    print(
        f"\n配置: T={T}, n_samples={n_samples}, unit_ids={unit_ids}, "
        f"max_iter={max_iter}, nn_epochs={nn_epochs}",
        flush=True,
    )

    print(f"\n生成测试数据...", flush=True)
    active_set_data = generate_test_data(ppc, T=T, n_samples=n_samples, seed=42)

    timings: Dict[str, float] = {}

    # ==================== 模式1：串行训练所有机组 ====================
    print("\n" + "=" * 60, flush=True)
    print("【模式1】串行训练（原始实现）", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    serial_trainers = train_all_subproblem_surrogates(
        ppc, active_set_data, T_delta=1.0,
        lambda_predictor=None, unit_ids=unit_ids,
        max_iter=max_iter, nn_epochs=nn_epochs,
        save_dir=str(save_base / "serial"),
    )
    timings['serial_all'] = time.time() - t0
    print(f"串行耗时: {timings['serial_all']:.1f}s", flush=True)

    # ==================== 模式2：机组级并行训练 ====================
    print("\n" + "=" * 60, flush=True)
    print("【模式2】机组级并行训练（ProcessPoolExecutor）", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    proc_results = train_all_surrogates_parallel(
        ppc, active_set_data, T_delta=1.0,
        lambda_predictor=None, unit_ids=unit_ids,
        max_iter=max_iter, nn_epochs=nn_epochs,
        save_dir=str(save_base / "process_parallel"),
        n_workers=n_proc,
        sample_n_workers=1,     # Level 2 禁用，单独对比
        use_sample_parallel=False,
    )
    timings['process_parallel_all'] = time.time() - t0
    print(f"机组级并行耗时: {timings['process_parallel_all']:.1f}s", flush=True)

    # ==================== 模式3：样本级并行（机组0）vs 串行（机组0）====================
    print("\n" + "=" * 60, flush=True)
    print("【模式3a】串行训练单机组（机组0，基准）", flush=True)
    print("=" * 60, flush=True)
    timings['serial_unit0'] = _time_serial_single(
        ppc, active_set_data, unit_ids[0], max_iter, nn_epochs
    )
    print(f"串行单机组耗时: {timings['serial_unit0']:.1f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("【模式3b】样本级并行训练（机组0，ThreadPoolExecutor）", flush=True)
    print("=" * 60, flush=True)
    timings['sample_parallel_unit0'] = _time_parallel_single(
        ppc, active_set_data, unit_ids[0], max_iter, nn_epochs, n_thread
    )
    print(f"样本级并行单机组耗时: {timings['sample_parallel_unit0']:.1f}s", flush=True)

    # ==================== 加速比对比表 ====================
    print("\n" + "=" * 60, flush=True)
    print("加速比对比", flush=True)
    print("=" * 60, flush=True)
    w = 38
    print(f"{'模式':<{w}} {'耗时(s)':>8} {'加速比':>8}", flush=True)
    print("-" * 60, flush=True)

    s_all  = timings['serial_all']
    p_all  = timings['process_parallel_all']
    s_u0   = timings['serial_unit0']
    sp_u0  = timings['sample_parallel_unit0']

    sp_proc   = s_all  / p_all  if p_all  > 0 else float('inf')
    sp_thread = s_u0   / sp_u0  if sp_u0  > 0 else float('inf')

    rows = [
        ("串行（所有机组）",           s_all,  "1.00x"),
        (f"机组级并行（{n_proc}进程）", p_all,  f"{sp_proc:.2f}x"),
        ("串行（机组0）",              s_u0,   "(基准)"),
        (f"样本级并行（{n_thread}线程，机组0）", sp_u0, f"{sp_thread:.2f}x"),
    ]
    for label, t, sp in rows:
        print(f"{label:<{w}} {t:>8.1f} {sp:>8}", flush=True)
    print("=" * 60, flush=True)

    # ==================== 结果验证 ====================
    print("\n结果验证（alpha_values 均值对比，量级应相近）:", flush=True)
    for uid in unit_ids:
        if uid in serial_trainers and uid in proc_results:
            r = proc_results[uid]
            if r.get('success'):
                s_alpha = serial_trainers[uid].alpha_values.mean()
                p_alpha = r['alpha_values'].mean()
                print(
                    f"  机组 {uid}: serial_alpha_mean={s_alpha:.4f}, "
                    f"process_alpha_mean={p_alpha:.4f}",
                    flush=True,
                )

    # ==================== 保存计时结果 ====================
    timing_file = save_base / "timings.json"
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
    print(f"\n计时结果已保存: {timing_file}", flush=True)
    print("✓ 并行训练测试完成", flush=True)


if __name__ == '__main__':
    # Windows 下 ProcessPoolExecutor 使用 spawn，必须在此保护块内调用 main()
    main()
