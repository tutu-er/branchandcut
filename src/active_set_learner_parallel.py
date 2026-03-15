"""并行化 ActiveSetLearner，使用 ProcessPoolExecutor 加速采样。"""

import os
import sys
import time
import io
import contextlib
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.ActiveSetLearner import ActiveSetLearner

from pypower.idx_brch import RATE_A


def _solve_single_sample(args: tuple) -> dict | None:
    """模块级 worker 函数，在子进程中求解单个样本。

    Args:
        args: (ppc, Pd, T_delta, gurobi_threads, sample_id) 元组。

    Returns:
        包含 pd_data, active_set, lambda 的字典，求解失败返回 None。
    """
    ppc, Pd, T_delta, gurobi_threads, sample_id = args

    # 延迟导入，避免主进程 fork 时的序列化问题
    from src.uc_gurobipy import UnitCommitmentModel
    from src.ed_gurobipy import EconomicDispatchGurobi

    try:
        # 静默求解器输出
        with contextlib.redirect_stdout(io.StringIO()):
            # Step 1: 求解 UC (MILP)
            uc = UnitCommitmentModel(ppc, Pd, T_delta)
            uc.model.Params.Threads = gurobi_threads
            pg_sol, x_sol, total_cost = uc.solve()

            # Step 2: 用 x 求解 ED (LP)
            ed = EconomicDispatchGurobi(ppc, Pd, T_delta, x_sol)
            ed.model.Params.Threads = gurobi_threads
            pg_sol, total_cost = ed.solve()

        # Step 3: 提取 lambda
        T = Pd.shape[1]
        lambda_vals = []
        for t in range(T):
            constr = ed.model.getConstrByName(f"power_balance_{t}")
            lambda_vals.append(float(constr.Pi) if constr is not None else 0.0)

        # Step 4: 构建活动集（只用二进制变量 x，不含 LP 活跃约束索引）
        x_sol_list = [
            [[i, j], int(x_sol[i, j])]
            for i in range(x_sol.shape[0])
            for j in range(x_sol.shape[1])
        ]
        active = list(x_sol_list)

        def make_hashable(item):
            if isinstance(item, list):
                return tuple(tuple(x) if isinstance(x, list) else x for x in item)
            return item

        active_set = frozenset(make_hashable(item) for item in active)

        return {"pd_data": Pd, "active_set": active_set, "lambda": lambda_vals}

    except Exception as e:
        # 在子进程中打印失败样本的标识，便于定位问题负荷
        print(f"  [Worker] 样本 sample_id={sample_id} 求解失败: {e}", flush=True)
        return None


class ParallelActiveSetLearner(ActiveSetLearner):
    """并行版 ActiveSetLearner，接口与父类完全一致。

    Args:
        n_workers: 并行 worker 数，默认自动计算 (cpu_count // gurobi_threads)。
        gurobi_threads: 每个 worker 内 Gurobi 使用的线程数。
        其余参数同 ActiveSetLearner。
    """

    def __init__(
        self,
        alpha: float = 0.05,
        delta: float = 0.01,
        epsilon: float = 0.04,
        ppc=None,
        T_delta: int = 4,
        Pd=None,
        case_name: str | None = None,
        n_workers: int | None = None,
        gurobi_threads: int = 4,
    ):
        super().__init__(
            alpha=alpha,
            delta=delta,
            epsilon=epsilon,
            ppc=ppc,
            T_delta=T_delta,
            Pd=Pd,
            case_name=case_name,
        )
        self.gurobi_threads = gurobi_threads
        if n_workers is None:
            cpu = os.cpu_count() or 4
            self.n_workers = max(1, cpu // gurobi_threads)
        else:
            self.n_workers = n_workers
        print(
            f"ParallelActiveSetLearner: n_workers={self.n_workers}, "
            f"gurobi_threads={self.gurobi_threads}",
            flush=True,
        )

    def run(self, max_samples: int = 22000) -> set:
        """并行版 DiscoverMass 算法。

        Args:
            max_samples: 最大采样数。

        Returns:
            发现的活动集集合。
        """
        epsilon = self.epsilon
        alpha = self.alpha
        M = 1
        O: set = set()
        samples: list = []
        iter_count = 0
        global_seed_counter = 0
        WM = self.W

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            while True:
                iter_count += 1
                print(
                    f"迭代{iter_count}: 当前窗口WM={WM}, 当前M={M}",
                    flush=True,
                )

                # 准备本窗口的所有 Pd 扰动
                actual_wm = min(WM, max_samples - len(samples))
                if actual_wm <= 0:
                    break

                pd_list = []
                seeds = []
                for i in range(actual_wm):
                    seed = global_seed_counter + i
                    seeds.append(seed)
                    pd_list.append(self._generate_random_Pd(rng=seed))
                global_seed_counter += actual_wm

                # 并行提交，附带 sample_id（这里用全局 seed 标识）
                args_list = [
                    (self.ppc, pd, self.T_delta, self.gurobi_threads, seed)
                    for pd, seed in zip(pd_list, seeds)
                ]
                futures = {
                    pool.submit(_solve_single_sample, a): i
                    for i, a in enumerate(args_list)
                }

                # 收集结果，显示进度
                results = [None] * actual_wm
                done_count = 0
                t_start = time.time()
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    results[idx] = result
                    done_count += 1

                    # 进度条
                    bar_len = 30
                    percent = done_count / actual_wm
                    filled_len = int(bar_len * percent)
                    bar = "█" * filled_len + "-" * (bar_len - filled_len)
                    elapsed = time.time() - t_start
                    if done_count > 0:
                        eta = elapsed / done_count * (actual_wm - done_count)
                    else:
                        eta = 0.0
                    print(
                        f"\r  采样进度: |{bar}| {done_count}/{actual_wm} "
                        f"({percent:.0%}) ETA: {eta:.0f}s",
                        end="",
                        flush=True,
                    )
                print(flush=True)

                # 按原始顺序收集有效结果，并在主进程打印失败样本
                new_active_sets = set()
                for i, r in enumerate(results):
                    if r is None:
                        print(
                            f"  [Main] 样本 i={i}, seed={seeds[i]} 求解失败（worker 返回 None），已跳过。",
                            flush=True,
                        )
                        continue
                    samples.append((r["pd_data"], r["active_set"], r["lambda"]))
                    if r["active_set"] not in O:
                        new_active_sets.add(r["active_set"])

                O.update(new_active_sets)
                RM_W = len(new_active_sets) / actual_wm
                print(
                    f"  发现率RM_W={RM_W:.4f}，目标发现率R={alpha - epsilon:.4f}，"
                    f"累计活动集数={len(O)}",
                    flush=True,
                )

                if RM_W < alpha - epsilon or len(samples) >= max_samples:
                    print("  停止条件触发，算法终止。", flush=True)
                    break
                M += 1

        self.samples = samples
        self.observed_active_sets = O
        self.M = M
        return O


if __name__ == "__main__":
    import pandas as pd
    import pypower.case118
    import pypower.idx_bus

    load_df = pd.read_csv("src/load.csv", header=None)
    Pd_base = load_df.values
    Pd_base = np.sum(Pd_base, axis=0)
    group_size = 4
    valid_steps = (Pd_base.size // group_size) * group_size
    Pd_base = Pd_base[:valid_steps].reshape(-1, group_size).sum(axis=1)

    ppc = pypower.case118.case118()

    Pd = ppc["bus"][:, pypower.idx_bus.PD]
    Pd = Pd[:, None] * Pd_base[None, :] / np.max(Pd_base) * 2.2
    
    ppc["branch"][:, RATE_A] = ppc["branch"][:, RATE_A] * 0.12

    learner = ParallelActiveSetLearner(
        alpha=0.75,
        delta=0.15,
        epsilon=0.10,
        ppc=ppc,
        T_delta=1,
        Pd=Pd,
        case_name="case118",
        n_workers=4,
        gurobi_threads=4,
    )
    active_sets = learner.run(max_samples=200)

    print(f"发现的活动集数量: {len(active_sets)}", flush=True)

    json_filename = learner.save_active_sets_json()
    print(f"完整JSON文件已保存: {json_filename}", flush=True)
