"""
Parallel BCD wrapper for Agent_NN_BCD.

This module keeps the neural-network block serial, while parallelizing the
per-sample PG and dual blocks with ThreadPoolExecutor.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uc_NN_BCD import Agent_NN_BCD


class ParallelAgent_NN_BCD(Agent_NN_BCD):
    """Sample-level threaded version of Agent_NN_BCD."""

    def __init__(
        self,
        ppc,
        active_set_data,
        T_delta: float,
        union_analysis=None,
        lambda_init_strategy: str = "lp_relaxation",
        max_theta_constraints_per_time_slot: int = 10,
        theta_hot_start_strategy: str = "dcpf_relative",
        zeta_hot_start_strategy: str = "zero",
        theta_gaussian_std: float = 0.01,
        zeta_gaussian_std: float = 0.01,
        enable_dropout_during_nn_training: bool = True,
        rho_primal_init: float = 1e-2,
        rho_dual_init: float = 1e-2,
        rho_dual_pg_init: float | None = None,
        rho_dual_x_init: float | None = None,
        rho_dual_coc_init: float | None = None,
        rho_opt_init: float = 1e-2,
        gamma_base: float = 1e-2,
        mu_dual_floor_init: float = 0.1,
        ita_dual_floor_init: float = 0.1,
        nn_learning_rate: float = 5e-5,
        nn_batch_strategy: str = "full-batch",
        nn_batch_size: int = 4,
        nn_shuffle: bool = True,
        n_workers: int = 4,
    ):
        super().__init__(
            ppc,
            active_set_data,
            T_delta,
            union_analysis,
            lambda_init_strategy=lambda_init_strategy,
            max_theta_constraints_per_time_slot=max_theta_constraints_per_time_slot,
            theta_hot_start_strategy=theta_hot_start_strategy,
            zeta_hot_start_strategy=zeta_hot_start_strategy,
            theta_gaussian_std=theta_gaussian_std,
            zeta_gaussian_std=zeta_gaussian_std,
            enable_dropout_during_nn_training=enable_dropout_during_nn_training,
            rho_primal_init=rho_primal_init,
            rho_dual_init=rho_dual_init,
            rho_dual_pg_init=rho_dual_pg_init,
            rho_dual_x_init=rho_dual_x_init,
            rho_dual_coc_init=rho_dual_coc_init,
            rho_opt_init=rho_opt_init,
            gamma_base=gamma_base,
            mu_dual_floor_init=mu_dual_floor_init,
            ita_dual_floor_init=ita_dual_floor_init,
            nn_learning_rate=nn_learning_rate,
            nn_batch_strategy=nn_batch_strategy,
            nn_batch_size=nn_batch_size,
            nn_shuffle=nn_shuffle,
        )
        self.n_workers = min(n_workers, self.n_samples)

    def iter(
        self,
        max_iter: int = 20,
        dual_decay_round: int = 10,
        nn_epochs: int = 10,
        union_analysis=None,
        nn_batch_strategy: str | None = None,
        nn_batch_size: int | None = None,
        nn_shuffle: bool | None = None,
        nn_learning_rate: float | None = None,
    ):
        if union_analysis is None:
            union_analysis = self._current_union_analysis

        eps = 1e-10
        gamma = self.gamma_base / (self.n_samples * max_iter)
        self.dual_decay_round = dual_decay_round

        print(
            f"[ParallelBCD] Start parallel BCD (n_workers={self.n_workers}, max_iter={max_iter})",
            flush=True,
        )

        for i in range(max_iter):
            print(f"[ParallelBCD] Iteration {i+1}/{max_iter}", flush=True)
            self.iter_number = i

            theta_snap_list = self.theta_values_list
            zeta_snap_list = self.zeta_values_list

            pg_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_pg_block,
                        sample_id,
                        theta_snap_list[sample_id],
                        zeta_snap_list[sample_id],
                        union_analysis,
                    ): sample_id
                    for sample_id in range(self.n_samples)
                }
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        pg_results[sid] = future.result()
                    except Exception as exc:
                        print(f"[ParallelBCD] PG block sample={sid} failed: {exc}", flush=True)
                        pg_results[sid] = (None, None, None, None)

            pg_block_ok = True
            for sid in range(self.n_samples):
                pg_sol, x_sol, cpower_sol, coc_sol = pg_results[sid]
                if pg_sol is None:
                    print(f"[ParallelBCD] Skip sample={sid} because PG block failed", flush=True)
                    pg_block_ok = False
                    continue
                self.pg[sid, :, :] = np.where(np.abs(pg_sol) < eps, 0, pg_sol)
                self.x[sid, :, :] = np.where(np.abs(x_sol) < eps, 0, x_sol)
                self.x[sid, :, :] = np.where(np.abs(self.x[sid, :, :] - 1) < eps, 1, self.x[sid, :, :])
                self.cpower[sid, :, :] = np.where(np.abs(cpower_sol) < eps, 0, cpower_sol)
                self.coc[sid, :, :] = np.where(np.abs(coc_sol) < eps, 0, coc_sol)

            if not pg_block_ok:
                print("[ParallelBCD] Some PG subproblems failed; continue", flush=True)

            dual_results: Dict[int, tuple] = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_sid = {
                    executor.submit(
                        self.iter_with_dual_block,
                        sample_id,
                        theta_snap_list[sample_id],
                        zeta_snap_list[sample_id],
                        union_analysis,
                    ): sample_id
                    for sample_id in range(self.n_samples)
                }
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        dual_results[sid] = future.result()
                    except Exception as exc:
                        print(f"[ParallelBCD] Dual block sample={sid} failed: {exc}", flush=True)
                        dual_results[sid] = (None, None, None)

            dual_block_ok = True
            for sid in range(self.n_samples):
                lambda_sol, mu_sol, ita_sol = dual_results[sid]
                if lambda_sol is None or mu_sol is None:
                    print(f"[ParallelBCD] Skip sample={sid} because dual block failed", flush=True)
                    dual_block_ok = False
                    continue
                self.lambda_[sid] = lambda_sol
                self.mu[sid, :, :] = np.where(np.abs(mu_sol) < eps, 0, mu_sol)
                self.ita[sid, :, :] = np.where(np.abs(ita_sol) < eps, 0, ita_sol)

            if not dual_block_ok:
                print("[ParallelBCD] Some dual subproblems failed; continue", flush=True)

            (
                obj_primal,
                obj_dual_pg,
                obj_dual_x,
                obj_dual_coc,
                obj_dual,
                obj_opt,
            ) = self.cal_viol_components(union_analysis=union_analysis)
            eps12 = 1e-12
            obj_primal = obj_primal if abs(obj_primal) >= eps12 else 0.0
            obj_dual_pg = obj_dual_pg if abs(obj_dual_pg) >= eps12 else 0.0
            obj_dual_x = obj_dual_x if abs(obj_dual_x) >= eps12 else 0.0
            obj_dual_coc = obj_dual_coc if abs(obj_dual_coc) >= eps12 else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= eps12 else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= eps12 else 0.0

            print(
                f"[ParallelBCD] obj_primal={obj_primal:.6f}, "
                f"obj_dual_pg={obj_dual_pg:.6f}, obj_dual_x={obj_dual_x:.6f}, "
                f"obj_dual_coc={obj_dual_coc:.6f}, obj_dual={obj_dual:.6f}, "
                f"obj_opt={obj_opt:.6f}",
                flush=True,
            )

            print(
                f"[ParallelBCD][NN-loss] obj_primal={obj_primal:.6f}, "
                f"obj_dual_x={obj_dual_x:.6f}, obj_opt={obj_opt:.6f}",
                flush=True,
            )
            if TORCH_AVAILABLE and hasattr(self, 'device'):
                self._refresh_iter_tensor_cache()

            theta_new, zeta_new = self.iter_with_theta_zeta_neural_network(
                union_analysis=union_analysis,
                num_epochs=nn_epochs,
                batch_strategy=nn_batch_strategy,
                batch_size=nn_batch_size,
                shuffle=nn_shuffle,
                learning_rate=nn_learning_rate,
            )
            if theta_new is None or zeta_new is None:
                print("[ParallelBCD] NN block failed; stop", flush=True)
                break

            self.theta_values_list = theta_new
            self.zeta_values_list = zeta_new
            self.theta_values = self.theta_values_list[0]
            self.zeta_values = self.zeta_values_list[0]

            print(f"[ParallelBCD] Iteration {i+1}/{max_iter} finished", flush=True)

            (
                obj_primal,
                obj_dual_pg,
                obj_dual_x,
                obj_dual_coc,
                obj_dual,
                obj_opt,
            ) = self.cal_viol_components(union_analysis=union_analysis)
            obj_primal = obj_primal if abs(obj_primal) >= eps12 else 0.0
            obj_dual_pg = obj_dual_pg if abs(obj_dual_pg) >= eps12 else 0.0
            obj_dual_x = obj_dual_x if abs(obj_dual_x) >= eps12 else 0.0
            obj_dual_coc = obj_dual_coc if abs(obj_dual_coc) >= eps12 else 0.0
            obj_dual = obj_dual if abs(obj_dual) >= eps12 else 0.0
            obj_opt = obj_opt if abs(obj_opt) >= eps12 else 0.0

            print(
                f"[ParallelBCD] obj_primal={obj_primal:.6f}, "
                f"obj_dual_pg={obj_dual_pg:.6f}, obj_dual_x={obj_dual_x:.6f}, "
                f"obj_dual_coc={obj_dual_coc:.6f}, obj_dual={obj_dual:.6f}, "
                f"obj_opt={obj_opt:.6f}",
                flush=True,
            )

            self.rho_primal = min(self.rho_primal + gamma * obj_primal, self.rho_max)
            gamma_dual = gamma * self.gamma_dual_component_scale
            self.rho_dual_pg = min(self.rho_dual_pg + gamma_dual * obj_dual_pg, self.rho_max)
            self.rho_dual_x = min(self.rho_dual_x + gamma_dual * obj_dual_x, self.rho_max)
            self.rho_dual_coc = min(self.rho_dual_coc + gamma_dual * obj_dual_coc, self.rho_max)
            self._sync_rho_dual_summary()
            self.rho_opt = min(self.rho_opt + gamma * obj_opt, self.rho_max)

            print(
                f"[ParallelBCD] rho_primal={self.rho_primal:.4f}, "
                f"rho_dual_pg={self.rho_dual_pg:.4f}, rho_dual_x={self.rho_dual_x:.4f}, "
                f"rho_dual_coc={self.rho_dual_coc:.4f}, rho_dual={self.rho_dual:.4f}, "
                f"rho_opt={self.rho_opt:.4f}",
                flush=True,
            )
            print("[ParallelBCD] " + "-" * 40, flush=True)

        return self.theta_values_list, self.zeta_values_list


if __name__ == '__main__':
    print("ParallelAgent_NN_BCD module loaded.", flush=True)
