#!/usr/bin/env python3
"""Inspect case3lite unit-0 surrogate subproblem on sample 0.

This script follows the current ``run_test_case3lite.py`` preset, loads the
same surrogate model directory, and prints:

- global plain LP x for unit 0
- global surrogate LP x for unit 0
- unit-0 surrogate subproblem LP/MILP x
- JSON optimal UC x for unit 0
- every unit-0 surrogate subproblem row evaluated at those x vectors
- whether the hard subproblem needed the soft-surrogate fallback
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_test as rt  # noqa: E402
import run_test_case3lite as case3lite_preset  # noqa: E402
from feasibility_pump import (  # noqa: E402
    _extract_unit_lambda,
    _resolve_surrogate_constraint_layout,
    _solve_unit_LP_with_surrogate,
    _solve_unit_MILP_with_surrogate,
    build_surrogate_constraint_expression,
    get_sample_net_load,
    solve_global_LP_relaxation,
    solve_global_LP_relaxation_without_surrogate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=case3lite_preset.MODEL_DIR)
    parser.add_argument("--active-sets", default=case3lite_preset.ACTIVE_SETS_FILE)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--strategy", default=case3lite_preset.SURROGATE_CONSTRAINT_STRATEGY)
    parser.add_argument("--precision", type=int, default=4)
    return parser.parse_args()


def _resolve_path(path_text: str | None, fallback_auto_latest: bool = False) -> Path:
    if path_text:
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()
    if fallback_auto_latest:
        return rt.pick_data_file(ROOT / "result" / "active_set", "case3lite").resolve()
    raise ValueError("path is required")


def _array_line(name: str, values: np.ndarray, precision: int) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    text = np.array2string(arr, precision=precision, suppress_small=False, max_line_width=240)
    print(f"{name}: {text}")


def _distance(name: str, values: np.ndarray, x_true: np.ndarray) -> None:
    vals = np.asarray(values, dtype=float).reshape(-1)
    ref = np.asarray(x_true, dtype=float).reshape(-1)
    if vals.shape != ref.shape or np.any(~np.isfinite(vals)):
        print(f"{name}: distance unavailable")
        return
    l1 = float(np.sum(np.abs(vals - ref)))
    ham = int(np.sum(np.rint(vals).astype(int) != ref.astype(int)))
    print(f"{name}: L1_to_opt={l1:.6g}, rounded_hamming_to_opt={ham}")


def _status_line(label: str, status: int, details: dict) -> None:
    if "error" in details:
        print(f"{label}: ERROR {details['error']}")
        return
    print(
        f"{label}: status={details.get('status_name', status)} "
        f"binary_x={details.get('binary_x')} "
        f"used_soft_surrogate={details.get('used_soft_surrogate')} "
        f"fallback_triggered={details.get('fallback_triggered')} "
        f"slack_sum={details.get('surrogate_slack_sum', 0.0):.6g} "
        f"slack_max={details.get('surrogate_slack_max', 0.0):.6g}"
    )
    if details.get("hard_status_name") is not None:
        print(
            f"  hard_status={details.get('hard_status_name')} "
            f"soft_status={details.get('soft_status_name')}"
        )


def _nan_solution(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=float)


def _safe_call(label: str, fn, fallback):
    try:
        return fn()
    except Exception as exc:
        print(f"{label}: ERROR {type(exc).__name__}: {exc}")
        return fallback(exc)


def _iter_constraint_rows(
    trainer,
    sample: dict,
    x_vectors: dict[str, np.ndarray],
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
) -> Iterable[dict]:
    timestep_map, offset_map = _resolve_surrogate_constraint_layout(
        trainer,
        sample,
        trainer.T,
        len(alphas),
    )
    a_eff, b_eff, g_eff, d_eff = trainer._apply_surrogate_direction_to_params(
        np.asarray(alphas, dtype=float),
        np.asarray(betas, dtype=float),
        np.asarray(gammas, dtype=float),
        np.asarray(deltas, dtype=float),
    )
    for k, timestep in enumerate(timestep_map):
        a = float(a_eff[k])
        b = float(b_eff[k])
        c = float(g_eff[k])
        rhs = float(d_eff[k])
        if abs(a) <= 1e-10 and abs(b) <= 1e-10 and abs(c) <= 1e-10:
            continue
        row = {
            "k": int(k),
            "t": int(timestep),
            "offsets": tuple(int(v) for v in offset_map[k]),
            "alpha": a,
            "beta": b,
            "gamma": c,
            "rhs": rhs,
        }
        for name, x_val in x_vectors.items():
            lhs = build_surrogate_constraint_expression(
                np.asarray(x_val, dtype=float).reshape(-1),
                int(timestep),
                offset_map[k],
                a,
                b,
                c,
                trainer.T,
            )
            row[f"{name}_lhs"] = float(lhs)
            row[f"{name}_viol"] = float(max(0.0, lhs - rhs))
        yield row


def main() -> None:
    args = _parse_args()
    unit_id = int(args.unit)
    sample_id = int(args.sample)

    active_sets = _resolve_path(args.active_sets, fallback_auto_latest=True)
    model_dir = _resolve_path(args.model_dir)
    if not active_sets.exists():
        raise FileNotFoundError(f"active-set file not found: {active_sets}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    os.environ.setdefault("RUN_TEST_DISABLE_PLOTS", "1")
    rt.CASE_NAME = "case3lite"
    rt.SURROGATE_CONSTRAINT_STRATEGY = args.strategy

    ppc = rt.get_case_ppc("case3lite")
    all_samples = rt.load_json_data(active_sets)
    if not (0 <= sample_id < len(all_samples)):
        raise IndexError(f"sample {sample_id} out of range; n={len(all_samples)}")
    sample = all_samples[sample_id]
    T_delta = rt.T_DELTA

    dual_predictor, trainers = rt.load_trained_models_for_test(
        ppc,
        all_samples,
        T_delta,
        model_dir=str(model_dir),
        unit_ids=[unit_id],
        requested_strategy=args.strategy,
        requested_ignore_startup_shutdown_costs=case3lite_preset.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS,
    )
    trainer = trainers[unit_id]

    pd_matrix = get_sample_net_load(sample)
    lambda_val = dual_predictor.predict(sample)
    lambda_unit = _extract_unit_lambda(lambda_val, trainer.T, unit_id=unit_id, trainer=trainer)
    renewable_data = sample.get("renewable_data") if isinstance(sample, dict) else None
    alphas, betas, gammas, deltas, costs, pg_costs = trainer.get_surrogate_params(
        sample,
        lambda_unit,
        renewable_data=renewable_data,
    )

    x_true_full = rt._extract_true_solution(sample, (ppc["gen"].shape[0], pd_matrix.shape[1]))
    x_true = np.asarray(x_true_full[unit_id], dtype=float)

    print("=" * 88)
    print("case3lite unit subproblem diagnostic")
    print(f"model_dir={model_dir}")
    print(f"active_sets={active_sets}")
    print(f"sample={sample_id} unit={unit_id} T={trainer.T}")
    print("=" * 88)

    x_plain_global = _safe_call(
        "global_plain_LP",
        lambda: solve_global_LP_relaxation_without_surrogate(ppc, pd_matrix, T_delta),
        lambda exc: np.full((ppc["gen"].shape[0], trainer.T), np.nan, dtype=float),
    )
    x_surr_global, global_stats = _safe_call(
        "global_surrogate_LP",
        lambda: solve_global_LP_relaxation(
            ppc,
            sample,
            T_delta,
            trainers,
            lambda_val,
            return_stats=True,
        ),
        lambda exc: (
            np.full((ppc["gen"].shape[0], trainer.T), np.nan, dtype=float),
            {"error": f"{type(exc).__name__}: {exc}", "all_stage_attempts": []},
        ),
    )
    x_unit_lp, status_lp, details_lp = _safe_call(
        "unit_surrogate_subproblem_LP",
        lambda: _solve_unit_LP_with_surrogate(
            trainer,
            lambda_val,
            alphas,
            betas,
            gammas,
            deltas,
            costs=costs,
            pg_costs=pg_costs,
            scenario_sample=sample,
        ),
        lambda exc: (_nan_solution(trainer.T), -1, {"error": f"{type(exc).__name__}: {exc}"}),
    )
    x_unit_milp, status_milp, details_milp = _safe_call(
        "unit_surrogate_subproblem_MILP",
        lambda: _solve_unit_MILP_with_surrogate(
            trainer,
            lambda_val,
            alphas,
            betas,
            gammas,
            deltas,
            costs=costs,
            pg_costs=pg_costs,
            scenario_sample=sample,
        ),
        lambda exc: (_nan_solution(trainer.T), -1, {"error": f"{type(exc).__name__}: {exc}"}),
    )

    print("\nSolutions")
    _array_line("global_plain_LP_unit0", x_plain_global[unit_id], args.precision)
    _array_line("global_surrogate_LP_unit0", x_surr_global[unit_id], args.precision)
    _array_line("unit_surrogate_subproblem_LP", x_unit_lp, args.precision)
    _array_line("unit_surrogate_subproblem_MILP", x_unit_milp, args.precision)
    _array_line("optimal_JSON_unit0", x_true, args.precision)

    print("\nDistances to optimal_JSON_unit0")
    _distance("global_plain_LP_unit0", x_plain_global[unit_id], x_true)
    _distance("global_surrogate_LP_unit0", x_surr_global[unit_id], x_true)
    _distance("unit_surrogate_subproblem_LP", x_unit_lp, x_true)
    _distance("unit_surrogate_subproblem_MILP", x_unit_milp, x_true)

    print("\nSoftening / solver status")
    if "error" in global_stats:
        print(f"global_surrogate_LP: ERROR {global_stats['error']}")
    else:
        print(
            "global_surrogate_LP: "
            f"stage={global_stats.get('stage_name')} "
            f"status={global_stats.get('status_name')} "
            f"used_soft_subproblem={global_stats.get('used_soft_subproblem')} "
            f"subproblem_slack_sum={global_stats.get('subproblem_slack_sum', 0.0):.6g} "
            f"subproblem_slack_max={global_stats.get('subproblem_slack_max', 0.0):.6g}"
        )
    for attempt in global_stats.get("all_stage_attempts", []):
        print(
            "  global_stage "
            f"{attempt.get('stage_index')}:{attempt.get('stage_name')} "
            f"status={attempt.get('status_name')} "
            f"used_soft_subproblem={attempt.get('used_soft_subproblem')} "
            f"subproblem_slack_sum={attempt.get('subproblem_slack_sum', 0.0):.6g} "
            f"subproblem_slack_max={attempt.get('subproblem_slack_max', 0.0):.6g}"
        )
    _status_line("unit_surrogate_subproblem_LP", status_lp, details_lp)
    _status_line("unit_surrogate_subproblem_MILP", status_milp, details_milp)

    print("\nAll unit surrogate constraints")
    x_vectors = {
        "opt": x_true,
        "unitLP": x_unit_lp,
        "unitMILP": x_unit_milp,
        "globalSurrLP": np.asarray(x_surr_global[unit_id], dtype=float),
        "globalPlainLP": np.asarray(x_plain_global[unit_id], dtype=float),
    }
    header = (
        "k  t  offsets      alpha       beta      gamma        rhs | "
        "opt_lhs opt_v  unitLP_lhs unitLP_v  unitMILP_lhs unitMILP_v  "
        "globalSurr_lhs globalSurr_v  plainLP_lhs plainLP_v"
    )
    print(header)
    print("-" * len(header))
    for row in _iter_constraint_rows(trainer, sample, x_vectors, alphas, betas, gammas, deltas):
        print(
            f"{row['k']:2d} {row['t']:2d} {str(row['offsets']):12s} "
            f"{row['alpha']:10.5g} {row['beta']:10.5g} {row['gamma']:10.5g} {row['rhs']:10.5g} | "
            f"{row['opt_lhs']:8.4f} {row['opt_viol']:6.2g} "
            f"{row['unitLP_lhs']:10.4f} {row['unitLP_viol']:8.2g} "
            f"{row['unitMILP_lhs']:12.4f} {row['unitMILP_viol']:10.2g} "
            f"{row['globalSurrLP_lhs']:14.4f} {row['globalSurrLP_viol']:12.2g} "
            f"{row['globalPlainLP_lhs']:11.4f} {row['globalPlainLP_viol']:9.2g}"
        )


if __name__ == "__main__":
    main()
