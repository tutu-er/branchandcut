#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Micro-benchmark for cached subproblem differentiable losses.

The script loads the same synchronized bundle used by
``scripts/subproblem_loss_snapshot.py`` and checks that cold-cache and warm-cache
loss calls agree while reporting average call time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional research dependency.
    torch = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


N_REPEATS: int = 50
SAMPLE_ID: int = 0


def _load_trainer():
    from scripts import subproblem_loss_snapshot as snapshot
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    snapshot._bootstrap_case118_light()
    snapshot._apply_script_overrides()
    active_path = snapshot._resolve_active_set_path()
    samples = snapshot._load_active_set_samples(active_path)[: max(1, int(snapshot.MAX_SAMPLES))]
    ppc = get_case_ppc(snapshot.CASE)
    bundle = snapshot._abs(snapshot.TEST_BUNDLE_DIR)
    _dual, trainers = load_trained_models(
        ppc,
        samples,
        float(snapshot.T_DELTA),
        str(bundle),
        unit_ids=[int(snapshot.UNIT_ID)],
        lp_backend=str(snapshot.LP_BACKEND).strip().lower(),
        constraint_generation_strategy=snapshot.CONSTRAINT_STRATEGY,
        ignore_startup_shutdown_costs=bool(snapshot.IGNORE_STARTUP_SHUTDOWN),
    )
    return trainers[int(snapshot.UNIT_ID)]


def _current_tensors(trainer, sample_id: int):
    device = trainer.device
    alphas = torch.tensor(trainer.alpha_values[sample_id], dtype=torch.float32, device=device)
    betas = torch.tensor(trainer.beta_values[sample_id], dtype=torch.float32, device=device)
    gammas = torch.tensor(trainer.gamma_values[sample_id], dtype=torch.float32, device=device)
    deltas = torch.tensor(trainer.delta_values[sample_id], dtype=torch.float32, device=device)
    costs = torch.tensor(trainer.cost_values[sample_id], dtype=torch.float32, device=device)
    pg_costs = torch.tensor(trainer.pg_cost_values[sample_id], dtype=torch.float32, device=device)
    return alphas, betas, gammas, deltas, costs, pg_costs


def _sync_if_cuda(device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _time_call(fn, repeats: int, device) -> tuple[float, float]:
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    last = 0.0
    for _ in range(repeats):
        value = fn()
        last = float(value.detach().cpu().item())
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    return last, elapsed / max(repeats, 1)


def main() -> None:
    if torch is None:
        print("[subproblem-loss-cache] skipped: PyTorch is not installed in this environment", flush=True)
        return
    trainer = _load_trainer()
    sample_id = min(max(int(SAMPLE_ID), 0), trainer.n_samples - 1)
    alphas, betas, gammas, deltas, costs, pg_costs = _current_tensors(trainer, sample_id)

    trainer._invalidate_loss_tensor_cache()
    main_cold, _ = trainer.loss_function_differentiable(
        sample_id, alphas, betas, gammas, deltas, costs, trainer.device, return_components=True
    )
    cpg_cold, _ = trainer.loss_function_c_pg_differentiable(
        sample_id, pg_costs, trainer.device, return_components=True
    )
    trainer._invalidate_loss_tensor_cache()
    main_warm, main_avg = _time_call(
        lambda: trainer.loss_function_differentiable(
            sample_id, alphas, betas, gammas, deltas, costs, trainer.device
        ),
        int(N_REPEATS),
        trainer.device,
    )
    cpg_warm, cpg_avg = _time_call(
        lambda: trainer.loss_function_c_pg_differentiable(sample_id, pg_costs, trainer.device),
        int(N_REPEATS),
        trainer.device,
    )

    main_diff = abs(float(main_cold.detach().cpu().item()) - main_warm)
    cpg_diff = abs(float(cpg_cold.detach().cpu().item()) - cpg_warm)
    print(
        f"[subproblem-loss-cache] sample={sample_id} repeats={N_REPEATS} device={trainer.device}",
        flush=True,
    )
    print(
        f"  NN-main cold/warm diff={main_diff:.6e}, warm_avg={main_avg * 1000:.3f} ms",
        flush=True,
    )
    print(
        f"  c_pg    cold/warm diff={cpg_diff:.6e}, warm_avg={cpg_avg * 1000:.3f} ms",
        flush=True,
    )
    if main_diff > 1e-5 or cpg_diff > 1e-5:
        raise SystemExit("cached loss changed value between cold and warm calls")


if __name__ == "__main__":
    main()
