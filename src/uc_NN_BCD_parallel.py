"""
uc_NN_BCD_parallel.py
======================
并行版 Agent_NN_BCD 训练，在新文件中实现，不修改原始 uc_NN_BCD.py。

并行层级
--------
BCD 的 Agent_NN_BCD 是全机组联合求解（区别于 subproblem_v3 的 per-unit 分离），
因此只有 **样本级线程并行**（无进程级机组并行）：

- PG 块：用 ThreadPoolExecutor 并发提交各样本的 Gurobi 求解；
  结果收集后主线程顺序写回 self.pg / self.x / self.cpower / self.coc。
- Dual 块：同样用 ThreadPoolExecutor 并发；
  结果收集后主线程顺序写回 self.lambda_ / self.mu / self.ita。
- NN 块（theta/zeta 神经网络更新）：保持串行，批次化训练无需改动。

Gurobi 求解时自动释放 GIL，线程并发有效。
每个块调用（iter_with_pg_block / iter_with_dual_block）内部创建独立 Model 对象，
线程安全无共享状态写入冲突。

用法
----
    from uc_NN_BCD_parallel import ParallelAgent_NN_BCD

    agent = ParallelAgent_NN_BCD(ppc, active_set_data, T_delta, n_workers=4)
    agent.iter(max_iter=20)
"""

import copy
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── 路径设置 ────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uc_NN_BCD import Agent_NN_BCD


# ════════════════════════════════════════════════════════════════════
# ParallelAgent_NN_BCD：样本级线程并行
# ════════════════════════════════════════════════════════════════════

class ParallelAgent_NN_BCD(Agent_NN_BCD):
    """样本级线程并行的 BCD 训练器。

    继承自 Agent_NN_BCD，仅重写 iter()，在 PG 块和 Dual 块内部
    使用 ThreadPoolExecutor 并发求解各样本的 Gurobi 子问题。
    NN 块（theta/zeta 神经网络更新）保持串行。

    Args:
        ppc: PyPower 案例数据。
        active_set_data: 活动集数据列表。
        T_delta: 时间间隔。
        union_analysis: 预先构建的 union_analysis（可选）。
        n_workers: 并发线程数，默认 min(n_samples, 4)。
    """

    def __init__(
        self,
        ppc,
        active_set_data,
        T_delta: float,
        union_analysis=None,
        n_workers: int = 4,
    ):
        super().__init__(ppc, active_set_data, T_delta, union_analysis)
        self.n_workers = min(n_workers, self.n_samples)

    def iter(self, max_iter: int = 20, nn_epochs: int = 10, union_analysis=None):
        """主 BCD 迭代循环（样本级线程并行版本）。

        PG 块和 Dual 块使用 ThreadPoolExecutor 并发提交各样本；
        结果收集后主线程顺序写回共享状态，保证无并发写冲突。
        NN 块保持串行不变。

        Args:
            max_iter: 最大迭代次数。
            union_analysis: union_analysis 数据（可选，默认使用初始化时构建的）。

        Returns:
            (theta_values, zeta_values) 元组，与串行版本一致。
        """
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        EPS = 1e-10
        gamma = self.gamma_base / (self.n_samples * max_iter)

        print(
            f"[ParallelBCD] 开始并行BCD迭代 "
            f"(n_workers={self.n_workers}, max_iter={max_iter})",
            flush=True,
        )

        for i in range(max_iter):
            print(f"[ParallelBCD] 🔄 迭代 {i+1}/{max_iter}", flush=True)
            self.iter_number = i

            # 快照当前 theta/zeta（线程只读，不修改；提前取值避免迭代中途被 NN 块改变）
            theta_snap = self.theta_values
            zeta_snap  = self.zeta_values

            # ── 1. PG 块（线程并行） ─────────────────────────────────
            pg_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_pg_block,
                        sample_id,
                        theta_snap,
                        zeta_snap,
                        union_analysis,
                    ): sample_id
                    for sample_id in range(self.n_samples)
                }
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        pg_results[sid] = future.result()
                    except Exception as exc:
                        print(
                            f"[ParallelBCD] PG块 sample={sid} 异常: {exc}",
                            flush=True,
                        )
                        pg_results[sid] = (None, None, None, None)

            # 顺序写回状态（避免并发写入）
            pg_block_ok = True
            for sid in range(self.n_samples):
                pg_sol, x_sol, cpower_sol, coc_sol = pg_results[sid]
                if pg_sol is None:
                    print(f"[ParallelBCD] ❌ PG块 sample={sid} 失败，跳过本样本", flush=True)
                    pg_block_ok = False
                    continue
                self.pg[sid, :, :]     = np.where(np.abs(pg_sol)    < EPS, 0, pg_sol)
                self.x[sid, :, :]      = np.where(np.abs(x_sol)     < EPS, 0, x_sol)
                self.x[sid, :, :]      = np.where(np.abs(self.x[sid, :, :] - 1) < EPS,
                                                   1, self.x[sid, :, :])
                self.cpower[sid, :, :] = np.where(np.abs(cpower_sol) < EPS, 0, cpower_sol)
                self.coc[sid, :, :]    = np.where(np.abs(coc_sol)    < EPS, 0, coc_sol)

            if not pg_block_ok:
                print("[ParallelBCD] ⚠ 部分PG块失败，继续迭代", flush=True)

            # ── 2. Dual 块（线程并行） ───────────────────────────────
            dual_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_dual_block,
                        sample_id,
                        theta_snap,
                        zeta_snap,
                        union_analysis,
                    ): sample_id
                    for sample_id in range(self.n_samples)
                }
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        dual_results[sid] = future.result()
                    except Exception as exc:
                        print(
                            f"[ParallelBCD] Dual块 sample={sid} 异常: {exc}",
                            flush=True,
                        )
                        dual_results[sid] = (None, None, None)

            # 顺序写回状态
            dual_block_ok = True
            for sid in range(self.n_samples):
                lambda_sol, mu_sol, ita_sol = dual_results[sid]
                if lambda_sol is None or mu_sol is None:
                    print(f"[ParallelBCD] ❌ Dual块 sample={sid} 失败，跳过本样本", flush=True)
                    dual_block_ok = False
                    continue
                self.lambda_[sid] = lambda_sol
                self.mu[sid, :, :]  = np.where(np.abs(mu_sol)  < EPS, 0, mu_sol)
                self.ita[sid, :, :] = np.where(np.abs(ita_sol) < EPS, 0, ita_sol)

            if not dual_block_ok:
                print("[ParallelBCD] ⚠ 部分Dual块失败，继续迭代", flush=True)

            # ── 3. NN 块（串行，theta/zeta 神经网络更新） ────────────
            if TORCH_AVAILABLE and hasattr(self, 'device'):
                self._refresh_iter_tensor_cache()
     
            theta_new, zeta_new = self.iter_with_theta_zeta_neural_network(
                union_analysis=union_analysis, num_epochs=nn_epochs
            )
            if theta_new is None or zeta_new is None:
                print("[ParallelBCD] ❌ NN 块更新失败，终止迭代", flush=True)
                break
            self.theta_values = theta_new
            self.zeta_values  = zeta_new

            print(f"[ParallelBCD] ✅ 迭代 {i+1}/{max_iter} 完成", flush=True)

            # ── 计算并打印违反量 ─────────────────────────────────────
            obj_primal, obj_dual, obj_opt = self.cal_viol(union_analysis=union_analysis)
            EPS12 = 1e-12
            obj_primal = obj_primal if abs(obj_primal) >= EPS12 else 0.0
            obj_dual   = obj_dual   if abs(obj_dual)   >= EPS12 else 0.0
            obj_opt    = obj_opt    if abs(obj_opt)    >= EPS12 else 0.0

            print(
                f"[ParallelBCD] obj_primal={obj_primal:.6f}, "
                f"obj_dual={obj_dual:.6f}, obj_opt={obj_opt:.6f}",
                flush=True,
            )

            self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
            self.rho_dual   = min(self.rho_dual   + gamma * obj_dual,   self.rho_max)
            self.rho_opt    = min(self.rho_opt     + gamma * obj_opt,    self.rho_max)

            print(
                f"[ParallelBCD] ρ_primal={self.rho_primal:.4f}, "
                f"ρ_dual={self.rho_dual:.4f}, ρ_opt={self.rho_opt:.4f}",
                flush=True,
            )
            print("[ParallelBCD] " + "─" * 40, flush=True)

        return self.theta_values, self.zeta_values


# ════════════════════════════════════════════════════════════════════
# __main__：功能验证与加速比对比
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import copy
    from datetime import datetime

    # ── 路径设置 ─────────────────────────────────────────────────────
    _src  = str(Path(__file__).resolve().parent)
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

    # 导入测试数据生成器（复用 subproblem_v3 的实现）
    from uc_NN_subproblem_v3 import generate_test_data

    # ── 测试参数 ─────────────────────────────────────────────────────
    T         = 8
    N_SAMPLES = 5
    MAX_ITER  = 3      # 迭代次数短，聚焦并行效果验证
    NN_EPOCHS = 5      # iter_with_theta_zeta_neural_network 的 num_epochs

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_base = _root + f'/result/bcd_models/parallel_test_{ts}'
    os.makedirs(save_base, exist_ok=True)

    print("=" * 60, flush=True)
    print(
        f"并行BCD训练验证  T={T}, n_samples={N_SAMPLES}",
        flush=True,
    )
    print(f"max_iter={MAX_ITER}, nn_epochs={NN_EPOCHS}", flush=True)
    print("=" * 60, flush=True)

    # ── 生成测试数据 ─────────────────────────────────────────────────
    # generate_test_data 生成的样本含 pd_data (nb, T) 和 unit_commitment_matrix (ng, T)
    # Agent_NN_BCD.initialize_solve() 会读取 unit_commitment_matrix 直接用 LP 初始化，无需 MILP
    active_set_data = generate_test_data(ppc, T=T, n_samples=N_SAMPLES, seed=42)

    timings: Dict[str, float] = {}

    # ── 1. 串行训练（baseline） ──────────────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【串行训练 Agent_NN_BCD】", flush=True)
    data_serial = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    agent_serial = Agent_NN_BCD(ppc, data_serial, T_delta=1.0)
    agent_serial.iter(max_iter=MAX_ITER)
    timings['serial'] = time.perf_counter() - t0
    print(f"串行完成，耗时 {timings['serial']:.2f}s", flush=True)

    # 保存串行结果
    try:
        agent_serial.save_model_parameters(
            os.path.join(save_base, 'serial_bcd.pth')
        )
    except Exception as e:
        print(f"保存串行模型失败（非致命）: {e}", flush=True)

    # ── 2. 并行训练（n_workers=4） ───────────────────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【并行训练 ParallelAgent_NN_BCD (n_workers=4)】", flush=True)
    data_parallel = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    agent_parallel = ParallelAgent_NN_BCD(
        ppc, data_parallel, T_delta=1.0, n_workers=4
    )
    agent_parallel.iter(max_iter=MAX_ITER)
    timings['parallel_4'] = time.perf_counter() - t0
    print(f"并行(n_workers=4)完成，耗时 {timings['parallel_4']:.2f}s", flush=True)

    # 保存并行结果
    try:
        agent_parallel.save_model_parameters(
            os.path.join(save_base, 'parallel_bcd_4w.pth')
        )
    except Exception as e:
        print(f"保存并行模型失败（非致命）: {e}", flush=True)

    # ── 3. 并行训练（n_workers=2，对比不同线程数） ────────────────────
    print("\n" + "─" * 50, flush=True)
    print("【并行训练 ParallelAgent_NN_BCD (n_workers=2)】", flush=True)
    data_parallel2 = copy.deepcopy(active_set_data)
    t0 = time.perf_counter()
    agent_parallel2 = ParallelAgent_NN_BCD(
        ppc, data_parallel2, T_delta=1.0, n_workers=2
    )
    agent_parallel2.iter(max_iter=MAX_ITER)
    timings['parallel_2'] = time.perf_counter() - t0
    print(f"并行(n_workers=2)完成，耗时 {timings['parallel_2']:.2f}s", flush=True)

    # ── 结果验证：检查 lambda (power_balance) 量级 ───────────────────
    print("\n" + "=" * 60, flush=True)
    print("结果验证（lambda_power_balance 量级对比）", flush=True)
    print("=" * 60, flush=True)
    for sid in range(min(3, N_SAMPLES)):
        s_lam = float(np.mean(np.abs(
            agent_serial.lambda_[sid].get('lambda_power_balance',
                                           np.zeros(T))
        )))
        p_lam = float(np.mean(np.abs(
            agent_parallel.lambda_[sid].get('lambda_power_balance',
                                             np.zeros(T))
        )))
        print(
            f"  sample-{sid}: 串行 λ_pb={s_lam:.4f} | 并行 λ_pb={p_lam:.4f}",
            flush=True,
        )

    # ── mu/ita 范数对比 ──────────────────────────────────────────────
    s_mu  = float(np.mean(np.abs(agent_serial.mu)))
    p_mu  = float(np.mean(np.abs(agent_parallel.mu)))
    s_ita = float(np.mean(np.abs(agent_serial.ita)))
    p_ita = float(np.mean(np.abs(agent_parallel.ita)))
    print(f"  mu  均值绝对值: 串行={s_mu:.4f} | 并行={p_mu:.4f}", flush=True)
    print(f"  ita 均值绝对值: 串行={s_ita:.4f} | 并行={p_ita:.4f}", flush=True)

    # ── 加速比汇总表 ─────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("加速比汇总表", flush=True)
    print("=" * 60, flush=True)
    ref = timings['serial']
    rows = [
        ("串行",                 timings['serial'],     1.0),
        ("并行(n_workers=4)",    timings['parallel_4'], ref / timings['parallel_4']),
        ("并行(n_workers=2)",    timings['parallel_2'], ref / timings['parallel_2']),
    ]
    print(f"{'模式':<24} {'耗时(s)':>10} {'加速比':>10}", flush=True)
    print("-" * 48, flush=True)
    for name, t, speedup in rows:
        print(f"{name:<24} {t:>10.2f} {speedup:>10.2f}x", flush=True)

    print(f"\n结果已保存至: {save_base}", flush=True)
    print("验证完成！", flush=True)
