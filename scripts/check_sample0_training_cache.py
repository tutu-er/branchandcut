#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

if not hasattr(np, "in1d"):
    def _compat_in1d(ar1, ar2, assume_unique=False, invert=False):
        return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)

    np.in1d = _compat_in1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.case_registry import get_case_ppc
from src.feasibility_pump import (
    _extract_unit_lambda,
    _gurobi_status_name,
    _solve_unit_LP_with_surrogate,
    solve_global_LP_relaxation,
)
from src.scenario_utils import normalize_sample_arrays
from src.uc_NN_subproblem import load_trained_models


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json_data(data_file: Path) -> list[dict]:
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = data.get("all_samples", [])
    if not all_samples:
        raise ValueError(f"No all_samples found in {data_file}")
    for idx, sample in enumerate(all_samples):
        normalize_sample_arrays(sample)
        sample.setdefault("sample_id", idx)
    return all_samples


def extract_true_solution(sample: dict, ng: int, T: int) -> np.ndarray:
    x_true = np.zeros((ng, T), dtype=float)
    if "unit_commitment_matrix" in sample:
        uc = np.asarray(sample["unit_commitment_matrix"], dtype=float)
        rows = min(ng, uc.shape[0])
        cols = min(T, uc.shape[1])
        x_true[:rows, :cols] = uc[:rows, :cols]
        return x_true

    for item in sample.get("active_set", []):
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list) and len(item[0]) == 2:
            g, t = item[0]
            if 0 <= g < ng and 0 <= t < T:
                x_true[g, t] = float(item[1])
    return x_true


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = np.abs(arr_a - arr_b)
    return {
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "l2": float(np.linalg.norm(diff)),
    }


def commitment_stats(x: np.ndarray, x_true: np.ndarray | None = None) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    stats = {
        "integrality": float(np.mean(np.minimum(x_arr, 1.0 - x_arr))),
    }
    if x_true is not None:
        x_round = (x_arr >= 0.5).astype(int)
        x_true_int = np.asarray(x_true, dtype=int)
        stats["hamming"] = int(np.sum(x_round != x_true_int))
        stats["l1"] = float(np.sum(np.abs(x_arr - x_true)))
    return stats


def print_surrogate_diagnostics(label: str, details: dict, max_rows: int = 8) -> None:
    log(
        f"  {label} status: {details.get('status_name', details.get('status'))}, "
        f"hard_status={details.get('hard_status_name', details.get('hard_status'))}, "
        f"fallback_triggered={details.get('fallback_triggered', False)}, "
        f"soft={details.get('used_soft_surrogate', False)}"
    )
    if 'objective_value' in details:
        log(
            f"  {label} surrogate summary: "
            f"viol_sum={details.get('surrogate_violation_sum', 0.0):.6f}, "
            f"viol_max={details.get('surrogate_violation_max', 0.0):.6f}, "
            f"slack_sum={details.get('surrogate_slack_sum', 0.0):.6f}, "
            f"slack_max={details.get('surrogate_slack_max', 0.0):.6f}"
        )
    rows = list(details.get('surrogate_violations', []))
    if not rows:
        return
    rows.sort(key=lambda item: (float(item.get('slack', 0.0)), float(item.get('violation', 0.0))), reverse=True)
    log(f"  {label} top surrogate rows:")
    for row in rows[:max_rows]:
        t0, t1, t2 = row['indices']
        log(
            f"    k={row['k']}, t={row['timestep']}: "
            f"{row['alpha']:.3f}*x[{t0}] + {row['beta']:.3f}*x[{t1}] + {row['gamma']:.3f}*x[{t2}] "
            f"<= {row['rhs']:.3f} | lhs={row['lhs']:.3f}, "
            f"viol={row['violation']:.6f}, slack={row['slack']:.6f}"
        )


def cached_surrogate_tuple(trainer, sample_id: int) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(trainer.alpha_values[sample_id], dtype=float).copy(),
        np.asarray(trainer.beta_values[sample_id], dtype=float).copy(),
        np.asarray(trainer.gamma_values[sample_id], dtype=float).copy(),
        np.asarray(trainer.delta_values[sample_id], dtype=float).copy(),
        np.asarray(trainer.cost_values[sample_id], dtype=float).copy(),
        np.asarray(trainer.pg_cost_values[sample_id], dtype=float).copy(),
    )


@contextmanager
def patch_trainers_with_cached_sample(trainers: dict, sample_id: int):
    originals = {}
    try:
        for unit_id, trainer in trainers.items():
            originals[unit_id] = trainer.get_surrogate_params
            cached_tuple = cached_surrogate_tuple(trainer, sample_id)

            def _cached_getter(pd_data, lambda_val, renewable_data=None, _vals=cached_tuple):
                return tuple(np.asarray(val, dtype=float).copy() for val in _vals)

            trainer.get_surrogate_params = _cached_getter
        yield
    finally:
        for unit_id, trainer in trainers.items():
            trainer.get_surrogate_params = originals[unit_id]


def build_cached_lambda_matrix(trainers: dict, sample_id: int) -> np.ndarray:
    unit_ids = sorted(trainers.keys())
    return np.stack(
        [np.asarray(trainers[uid].lambda_vals[sample_id], dtype=float) for uid in unit_ids],
        axis=0,
    )


def main() -> None:
    case_name = "case3lite"
    t_delta = 1.0
    sample_id = 0
    objective_probe_unit_id = 0
    model_dir = ROOT / "result" / "surrogate_models" / "subproblem_models_case3lite_20260329_120709"
    data_file = ROOT / "result" / "active_set" / "active_sets_case3lite_T24_n200_20260328_102856.json"
    strategy = "all_templates_sign4"
    skip_global = False

    log(f"case={case_name}")
    log(f"model_dir={model_dir}")
    log(f"data_file={data_file}")
    log(f"sample_id={sample_id}")
    log(f"objective_probe_unit_id={objective_probe_unit_id}")
    log(f"strategy={strategy}")

    ppc = get_case_ppc(case_name)
    all_samples = load_json_data(data_file)
    if not (0 <= sample_id < len(all_samples)):
        raise IndexError(f"sample_id={sample_id} out of range for {len(all_samples)} samples")
    sample = all_samples[sample_id]

    dual_predictor, trainers = load_trained_models(
        ppc,
        all_samples,
        t_delta,
        load_dir=str(model_dir),
        unit_ids=None,
        constraint_generation_strategy=strategy,
    )

    ng = ppc["gen"].shape[0]
    T = np.asarray(sample["pd_data"], dtype=float).shape[1]
    x_true = extract_true_solution(sample, ng, T)

    log(
        f"dual_predictor: legacy_mode={getattr(dual_predictor, '_legacy_mode', None)}, "
        f"output_dim={getattr(dual_predictor, 'output_dim', 'n/a')}"
    )

    lambda_pred = np.asarray(dual_predictor.predict(sample), dtype=float)
    if lambda_pred.shape == (T, ng):
        lambda_pred = lambda_pred.T
    lambda_cache = build_cached_lambda_matrix(trainers, sample_id)
    log(
        f"lambda predictor shape={lambda_pred.shape}, cache shape={lambda_cache.shape}, "
        f"diff={diff_stats(lambda_pred, lambda_cache)}"
    )

    if objective_probe_unit_id in trainers:
        trainer = trainers[objective_probe_unit_id]
        lambda_probe = _extract_unit_lambda(
            lambda_pred,
            T,
            unit_id=objective_probe_unit_id,
            trainer=trainer,
        )
        (
            _alphas_probe,
            _betas_probe,
            _gammas_probe,
            _deltas_probe,
            _costs_probe,
            pg_costs_probe,
        ) = trainer.get_surrogate_params(
            sample,
            lambda_probe,
            renewable_data=sample.get("renewable_data"),
        )
        a_pg = float(trainer.gencost[objective_probe_unit_id, -2] / trainer.T_delta)

        print("\n" + "=" * 80)
        print(
            f"Sample {sample_id} Unit {objective_probe_unit_id} PG Objective Coefficients",
            flush=True,
        )
        print("=" * 80, flush=True)
        print(f"base linear pg cost a = {a_pg:.6f}", flush=True)
        print(
            "subproblem objective on pg[t]: cpower[t] + c_pg[t]*pg[t] - lambda[t]*pg[t]",
            flush=True,
        )
        print(
            "effective marginal coefficient (when cpower[t] = a*pg[t] + b*x[t]): "
            "a + c_pg[t] - lambda[t]",
            flush=True,
        )
        print("t | lambda[t] | c_pg[t] | a + c_pg[t] - lambda[t]", flush=True)
        for t in range(T):
            eff_coeff = a_pg + float(pg_costs_probe[t]) - float(lambda_probe[t])
            print(
                f"{t:2d} | {float(lambda_probe[t]):9.4f} | "
                f"{float(pg_costs_probe[t]):8.4f} | {eff_coeff:11.6f}",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("Per-Unit Comparison", flush=True)
    print("=" * 80, flush=True)

    x_sub_forward = np.full((ng, T), np.nan, dtype=float)
    x_sub_cached_pred = np.full((ng, T), np.nan, dtype=float)
    x_sub_cached_cache = np.full((ng, T), np.nan, dtype=float)

    for unit_id, trainer in sorted(trainers.items()):
        if sample_id >= trainer.alpha_values.shape[0]:
            raise IndexError(
                f"sample_id={sample_id} out of range for trainer {unit_id} cache "
                f"shape {trainer.alpha_values.shape}"
            )

        lambda_pred_unit = _extract_unit_lambda(lambda_pred, T, unit_id=unit_id, trainer=trainer)
        lambda_cache_unit = np.asarray(trainer.lambda_vals[sample_id], dtype=float)
        cached_tuple = cached_surrogate_tuple(trainer, sample_id)
        forward_tuple = trainer.get_surrogate_params(sample, lambda_pred_unit, renewable_data=sample.get("renewable_data"))

        param_names = ("alpha", "beta", "gamma", "delta", "c_x", "c_pg")
        log(f"\nunit={unit_id}")
        log(f"  lambda diff: {diff_stats(lambda_pred_unit, lambda_cache_unit)}")
        for name, cached_val, forward_val in zip(param_names, cached_tuple, forward_tuple):
            log(f"  {name} diff: {diff_stats(cached_val, forward_val)}")

        x_forward, status_forward, details_forward = _solve_unit_LP_with_surrogate(
            trainer,
            lambda_pred,
            forward_tuple[0],
            forward_tuple[1],
            forward_tuple[2],
            forward_tuple[3],
            costs=forward_tuple[4],
            pg_costs=forward_tuple[5],
            scenario_sample=sample,
        )
        x_cached_pred, status_cached_pred, details_cached_pred = _solve_unit_LP_with_surrogate(
            trainer,
            lambda_pred,
            cached_tuple[0],
            cached_tuple[1],
            cached_tuple[2],
            cached_tuple[3],
            costs=cached_tuple[4],
            pg_costs=cached_tuple[5],
            scenario_sample=sample,
        )
        x_cached_cache, status_cached_cache, details_cached_cache = _solve_unit_LP_with_surrogate(
            trainer,
            lambda_cache_unit,
            cached_tuple[0],
            cached_tuple[1],
            cached_tuple[2],
            cached_tuple[3],
            costs=cached_tuple[4],
            pg_costs=cached_tuple[5],
            scenario_sample=sample,
        )

        x_sub_forward[unit_id] = x_forward
        x_sub_cached_pred[unit_id] = x_cached_pred
        x_sub_cached_cache[unit_id] = x_cached_cache

        log(
            f"  statuses: forward={_gurobi_status_name(status_forward)}, "
            f"cached_pred_lambda={_gurobi_status_name(status_cached_pred)}, "
            f"cached_cache_lambda={_gurobi_status_name(status_cached_cache)}"
        )
        log(f"  x_forward stats: {commitment_stats(x_forward, x_true[unit_id])}")
        log(f"  x_cached_pred_lambda stats: {commitment_stats(x_cached_pred, x_true[unit_id])}")
        log(f"  x_cached_cache_lambda stats: {commitment_stats(x_cached_cache, x_true[unit_id])}")
        log(f"  x forward vs cached(pred lambda) diff: {diff_stats(x_forward, x_cached_pred)}")
        log(f"  x cached(pred lambda) vs cached(cache lambda) diff: {diff_stats(x_cached_pred, x_cached_cache)}")
        if unit_id == objective_probe_unit_id:
            print(
                "  x_true: "
                f"{np.array2string(np.asarray(x_true[unit_id], dtype=float), precision=3, max_line_width=200)}",
                flush=True,
            )
            print(
                "  x_forward: "
                f"{np.array2string(np.asarray(x_forward, dtype=float), precision=3, max_line_width=200)}",
                flush=True,
            )
            print(
                "  x_cached_pred_lambda: "
                f"{np.array2string(np.asarray(x_cached_pred, dtype=float), precision=3, max_line_width=200)}",
                flush=True,
            )
            print(
                "  x_cached_cache_lambda: "
                f"{np.array2string(np.asarray(x_cached_cache, dtype=float), precision=3, max_line_width=200)}",
                flush=True,
            )
            print_surrogate_diagnostics("forward", details_forward)
            print_surrogate_diagnostics("cached_pred_lambda", details_cached_pred)
            print_surrogate_diagnostics("cached_cache_lambda", details_cached_cache)

    if not skip_global:
        print("\n" + "=" * 80)
        print("Global LP Comparison", flush=True)
        print("=" * 80, flush=True)
        x_global_forward = solve_global_LP_relaxation(
            ppc,
            sample,
            t_delta,
            trainers,
            lambda_pred,
        )
        with patch_trainers_with_cached_sample(trainers, sample_id):
            x_global_cached_pred = solve_global_LP_relaxation(
                ppc,
                sample,
                t_delta,
                trainers,
                lambda_pred,
            )
            x_global_cached_cache = solve_global_LP_relaxation(
                ppc,
                sample,
                t_delta,
                trainers,
                lambda_cache,
            )

        log(f"x_global_forward stats: {commitment_stats(x_global_forward, x_true)}")
        log(f"x_global_cached_pred_lambda stats: {commitment_stats(x_global_cached_pred, x_true)}")
        log(f"x_global_cached_cache_lambda stats: {commitment_stats(x_global_cached_cache, x_true)}")
        log(f"global forward vs cached(pred lambda) diff: {diff_stats(x_global_forward, x_global_cached_pred)}")
        log(f"global cached(pred lambda) vs cached(cache lambda) diff: {diff_stats(x_global_cached_pred, x_global_cached_cache)}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
