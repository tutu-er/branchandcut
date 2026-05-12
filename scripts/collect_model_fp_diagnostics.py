#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect forward-solve diagnostics for tailoring feasibility pump settings.

This script runs a trained surrogate bundle on selected active-set samples and
writes one JSON artifact containing the data needed to compare:

* true UC commitment vs. plain global LP
* true UC commitment vs. global LP with surrogate/proxy rows
* true UC commitment vs. independent per-unit surrogate LP/MILP solves
* optional current feasibility-pump recovery details

Example:
    python scripts/collect_model_fp_diagnostics.py ^
        --case case3lite ^
        --model-dir result/surrogate_models/subproblem_models_case3lite_20260509_190031 ^
        --samples 20 ^
        --run-fp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

get_case_ppc = None
load_v3_active_set_json = None
get_sample_net_load = None
normalize_sample_arrays = None
load_trained_models = None
_extract_unit_lambda = None
_gurobi_status_name = None
_solve_unit_LP_with_surrogate = None
_solve_unit_MILP_with_surrogate = None
recover_integer_solution = None
solve_global_LP_relaxation = None
solve_global_LP_relaxation_without_surrogate = None
GRB = None


def _ensure_runtime_imports() -> None:
    """Import solver-backed project modules lazily so ``--help`` works anywhere."""
    global get_case_ppc, load_v3_active_set_json, get_sample_net_load
    global normalize_sample_arrays, load_trained_models
    global _extract_unit_lambda, _gurobi_status_name
    global _solve_unit_LP_with_surrogate, _solve_unit_MILP_with_surrogate
    global recover_integer_solution, solve_global_LP_relaxation
    global solve_global_LP_relaxation_without_surrogate, GRB

    if recover_integer_solution is not None:
        return

    try:
        from gurobipy import GRB as _GRB
    except Exception as exc:  # pragma: no cover - depends on local solver env.
        raise RuntimeError(
            "gurobipy is required to collect diagnostics. "
            "Run from the project solver environment, e.g. "
            "`conda run -n poweropt python scripts/collect_model_fp_diagnostics.py ...`."
        ) from exc

    from src.case_registry import get_case_ppc as _get_case_ppc
    from src.dataset_json_utils import load_v3_active_set_json as _load_v3_active_set_json
    from src.feasibility_pump import (
        _extract_unit_lambda as __extract_unit_lambda,
        _gurobi_status_name as __gurobi_status_name,
        _solve_unit_LP_with_surrogate as __solve_unit_LP_with_surrogate,
        _solve_unit_MILP_with_surrogate as __solve_unit_MILP_with_surrogate,
        recover_integer_solution as _recover_integer_solution,
        solve_global_LP_relaxation as _solve_global_LP_relaxation,
        solve_global_LP_relaxation_without_surrogate as _solve_global_LP_relaxation_without_surrogate,
    )
    from src.scenario_utils import (
        get_sample_net_load as _get_sample_net_load,
        normalize_sample_arrays as _normalize_sample_arrays,
    )
    from src.uc_NN_subproblem import load_trained_models as _load_trained_models

    GRB = _GRB
    get_case_ppc = _get_case_ppc
    load_v3_active_set_json = _load_v3_active_set_json
    get_sample_net_load = _get_sample_net_load
    normalize_sample_arrays = _normalize_sample_arrays
    load_trained_models = _load_trained_models
    _extract_unit_lambda = __extract_unit_lambda
    _gurobi_status_name = __gurobi_status_name
    _solve_unit_LP_with_surrogate = __solve_unit_LP_with_surrogate
    _solve_unit_MILP_with_surrogate = __solve_unit_MILP_with_surrogate
    recover_integer_solution = _recover_integer_solution
    solve_global_LP_relaxation = _solve_global_LP_relaxation
    solve_global_LP_relaxation_without_surrogate = _solve_global_LP_relaxation_without_surrogate


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _latest_active_set(case_name: str) -> Path:
    candidates = sorted(
        (ROOT / "result" / "active_set").glob(f"active_sets_{case_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No active-set JSON found for case={case_name!r}")
    return candidates[0]


def _latest_model_dir(case_name: str) -> Path:
    candidates = sorted(
        (ROOT / "result" / "surrogate_models").glob(f"subproblem_models_{case_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if any(p.glob("surrogate_unit_*.pth"))]
    if not candidates:
        raise FileNotFoundError(f"No surrogate model dir found for case={case_name!r}")
    return candidates[0]


def _resolve_path(path_text: Optional[str], fallback: Path) -> Path:
    if path_text is None or not str(path_text).strip():
        return fallback.resolve()
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _parse_unit_ids(text: str | None) -> Optional[List[int]]:
    if text is None:
        return None
    stripped = str(text).strip().lower()
    if stripped in ("", "all", "none"):
        return None
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def _parse_sample_slice(samples: List[dict], start: int, count: int) -> List[dict]:
    lo = max(0, int(start))
    hi = len(samples) if count <= 0 else min(len(samples), lo + int(count))
    sliced = samples[lo:hi]
    for local_idx, sample in enumerate(sliced, start=lo):
        sample.setdefault("sample_id", local_idx)
        sample.setdefault("source_sample_id", local_idx)
        normalize_sample_arrays(sample)
    return sliced


def _extract_true_solution(sample: dict, shape: Tuple[int, int]) -> np.ndarray:
    ng, T = shape
    x_true = np.full((ng, T), np.nan, dtype=float)
    if "unit_commitment_matrix" in sample:
        raw = np.asarray(sample["unit_commitment_matrix"], dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        r = min(raw.shape[0], ng)
        c = min(raw.shape[1], T)
        x_true[:r, :c] = raw[:r, :c]
        return x_true

    if "active_set" in sample:
        x_true[:] = 0.0
        for item in sample["active_set"]:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                if 0 <= int(g) < ng and 0 <= int(t) < T:
                    x_true[int(g), int(t)] = float(item[1])
    return x_true


def _distance_metrics(x_candidate: Optional[np.ndarray], x_true: np.ndarray) -> dict:
    if x_candidate is None:
        return {"available": False}
    x_arr = np.asarray(x_candidate, dtype=float)
    x_ref = np.asarray(x_true, dtype=float)
    if x_arr.shape != x_ref.shape:
        return {
            "available": False,
            "reason": f"shape mismatch candidate={x_arr.shape} true={x_ref.shape}",
        }
    valid = np.isfinite(x_arr) & np.isfinite(x_ref)
    total = int(valid.size)
    covered = int(np.sum(valid))
    if covered == 0:
        return {
            "available": False,
            "covered": 0,
            "total": total,
            "coverage_ratio": 0.0,
        }
    x_v = x_arr[valid]
    y_v = x_ref[valid]
    rounded = np.rint(np.clip(x_v, 0.0, 1.0)).astype(int)
    y_int = np.rint(np.clip(y_v, 0.0, 1.0)).astype(int)
    false_on = int(np.sum((rounded == 1) & (y_int == 0)))
    false_off = int(np.sum((rounded == 0) & (y_int == 1)))
    return {
        "available": True,
        "covered": covered,
        "total": total,
        "coverage_ratio": float(covered / max(total, 1)),
        "l1": float(np.sum(np.abs(x_v - y_v))),
        "mean_abs": float(np.mean(np.abs(x_v - y_v))),
        "hamming": int(false_on + false_off),
        "false_on": false_on,
        "false_off": false_off,
        "integrality_gap": float(np.mean(np.minimum(x_v, 1.0 - x_v))),
    }


def _binary_error_matrix(x_candidate: Optional[np.ndarray], x_true: np.ndarray) -> Optional[np.ndarray]:
    if x_candidate is None:
        return None
    x_arr = np.asarray(x_candidate, dtype=float)
    if x_arr.shape != x_true.shape:
        return None
    valid = np.isfinite(x_arr) & np.isfinite(x_true)
    rounded = np.rint(np.clip(x_arr, 0.0, 1.0)).astype(int)
    truth = np.rint(np.clip(x_true, 0.0, 1.0)).astype(int)
    err = np.zeros_like(rounded, dtype=int)
    err[valid & (rounded == 1) & (truth == 0)] = 1
    err[valid & (rounded == 0) & (truth == 1)] = -1
    return err


def _solve_subproblem_matrix(
    sample: dict,
    lambda_val: Any,
    trainers: Dict[int, Any],
    shape: Tuple[int, int],
    solve_milp: bool,
    surrogate_constraint_scope: str,
) -> Tuple[np.ndarray, List[dict]]:
    ng, T = shape
    x_sub = np.full((ng, T), np.nan, dtype=float)
    rows: List[dict] = []
    solve_one = _solve_unit_MILP_with_surrogate if solve_milp else _solve_unit_LP_with_surrogate
    renewable_data = sample.get("renewable_data") if isinstance(sample, dict) else None

    for unit_id, trainer in sorted(trainers.items()):
        g = int(unit_id)
        if not (0 <= g < ng):
            continue
        try:
            lambda_unit = _extract_unit_lambda(lambda_val, T, unit_id=g, trainer=trainer)
            alphas, betas, gammas, deltas, costs, pg_costs = trainer.get_surrogate_params(
                sample,
                lambda_unit,
                renewable_data=renewable_data,
            )
            x_unit, status, details = solve_one(
                trainer,
                lambda_unit,
                alphas,
                betas,
                gammas,
                deltas,
                costs=costs,
                pg_costs=pg_costs,
                scenario_sample=sample,
                surrogate_constraint_scope=surrogate_constraint_scope,
            )
            if GRB is not None and status == GRB.OPTIMAL:
                x_sub[g, : min(T, len(x_unit))] = np.asarray(x_unit, dtype=float)[:T]
            rows.append(
                {
                    "unit_id": g,
                    "solve_type": "milp" if solve_milp else "lp",
                    "status": int(status) if status is not None else None,
                    "status_name": _gurobi_status_name(status),
                    "fallback_triggered": bool((details or {}).get("fallback_triggered", False)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "unit_id": g,
                    "solve_type": "milp" if solve_milp else "lp",
                    "status": None,
                    "status_name": "EXCEPTION",
                    "error": str(exc),
                }
            )
    return x_sub, rows


def _clean_solve_stats(stats: Optional[dict]) -> dict:
    if not stats:
        return {}
    keep = {
        "stage_index",
        "stage_name",
        "status",
        "status_name",
        "runtime_sec",
        "objective",
        "objective_problem",
        "objective_uc_cost",
        "objective_surrogate_aux",
        "num_constraints",
        "num_proxy_constraints",
        "num_subproblem_surrogate_constraints",
        "num_bcd_theta_constraints",
        "num_bcd_zeta_constraints",
        "subproblem_slack_sum",
        "subproblem_slack_max",
        "bcd_slack_sum",
        "bcd_slack_max",
        "used_fallback_stage",
        "surrogate_constraint_scope",
        "bcd_proxy_scope",
    }
    return {k: stats.get(k) for k in sorted(keep) if k in stats}


def collect_diagnostics(args: argparse.Namespace) -> dict:
    _ensure_runtime_imports()

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))
    output_path = _resolve_path(
        args.output,
        ROOT / "result" / "fp_diagnostics" / (
            f"fp_forward_diagnostics_{args.case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[diagnostics] case={args.case}", flush=True)
    print(f"[diagnostics] active_sets={active_set_path}", flush=True)
    print(f"[diagnostics] model_dir={model_dir}", flush=True)
    print(f"[diagnostics] output={output_path}", flush=True)

    ppc = get_case_ppc(args.case)
    all_samples = load_v3_active_set_json(active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True))
    selected_samples = _parse_sample_slice(all_samples, args.start, args.samples)
    if not selected_samples:
        raise ValueError("No samples selected")

    first_pd = get_sample_net_load(selected_samples[0])
    T_delta = float(args.t_delta)
    unit_ids = _parse_unit_ids(args.unit_ids)

    dual_predictor, trainers = load_trained_models(
        ppc,
        all_samples,
        T_delta,
        str(model_dir),
        unit_ids=unit_ids,
        lp_backend=args.lp_backend,
        constraint_generation_strategy=args.strategy,
        ignore_startup_shutdown_costs=(
            None if args.ignore_startup_shutdown_costs is None else bool(args.ignore_startup_shutdown_costs)
        ),
        case_name=args.case,
        use_unit_predictor=bool(args.use_unit_predictor),
        skip_initial_solve=True,
    )

    records: List[dict] = []
    ng = int(np.asarray(ppc["gen"]).shape[0])
    T = int(first_pd.shape[1])

    for local_idx, sample in enumerate(selected_samples, start=1):
        sample_id = sample.get("sample_id", sample.get("source_sample_id", args.start + local_idx - 1))
        print(f"[sample {local_idx}/{len(selected_samples)}] sample_id={sample_id}", flush=True)
        pd_data = get_sample_net_load(sample)
        shape = (ng, pd_data.shape[1])
        x_true = _extract_true_solution(sample, shape)

        record: dict = {
            "sample_index": int(local_idx - 1),
            "sample_id": int(sample_id) if str(sample_id).lstrip("-").isdigit() else sample_id,
            "pd_shape": list(pd_data.shape),
            "pd_sum": np.sum(pd_data, axis=0),
            "has_true_solution": bool(np.isfinite(x_true).any()),
            "solutions": {"x_true": x_true},
            "metrics": {},
            "unit_status": [],
            "solve_stats": {},
        }

        try:
            lambda_val = dual_predictor.predict(sample)
            record["lambda_summary"] = {
                "type": type(lambda_val).__name__,
                "shape": list(np.asarray(lambda_val).shape) if not isinstance(lambda_val, dict) else None,
                "keys": sorted(lambda_val.keys()) if isinstance(lambda_val, dict) else None,
            }
        except Exception as exc:
            record["error"] = f"dual_predictor failed: {exc}"
            records.append(record)
            continue

        x_lp_plain = None
        if not args.skip_plain_lp:
            try:
                x_lp_plain = solve_global_LP_relaxation_without_surrogate(ppc, pd_data, T_delta)
                record["solutions"]["x_lp_plain"] = x_lp_plain
                record["metrics"]["plain_lp_to_true"] = _distance_metrics(x_lp_plain, x_true)
                record["errors_plain_lp"] = _binary_error_matrix(x_lp_plain, x_true)
            except Exception as exc:
                record["plain_lp_error"] = str(exc)

        x_lp_proxy = None
        if not args.skip_proxy_lp:
            try:
                x_lp_proxy, proxy_stats = solve_global_LP_relaxation(
                    ppc,
                    sample,
                    T_delta,
                    trainers,
                    lambda_val,
                    surrogate_constraint_scope=args.surrogate_constraint_scope,
                    bcd_proxy_scope=args.bcd_proxy_scope,
                    return_stats=True,
                )
                record["solutions"]["x_lp_proxy"] = x_lp_proxy
                record["metrics"]["proxy_lp_to_true"] = _distance_metrics(x_lp_proxy, x_true)
                record["errors_proxy_lp"] = _binary_error_matrix(x_lp_proxy, x_true)
                record["solve_stats"]["proxy_lp"] = _clean_solve_stats(proxy_stats)
            except Exception as exc:
                record["proxy_lp_error"] = str(exc)

        if not args.skip_subproblem:
            x_sub_lp, rows = _solve_subproblem_matrix(
                sample,
                lambda_val,
                trainers,
                shape,
                solve_milp=False,
                surrogate_constraint_scope=args.surrogate_constraint_scope,
            )
            record["solutions"]["x_subproblem_lp"] = x_sub_lp
            record["metrics"]["subproblem_lp_to_true"] = _distance_metrics(x_sub_lp, x_true)
            record["errors_subproblem_lp"] = _binary_error_matrix(x_sub_lp, x_true)
            record["unit_status"].extend(rows)

        if args.subproblem_milp:
            x_sub_milp, rows = _solve_subproblem_matrix(
                sample,
                lambda_val,
                trainers,
                shape,
                solve_milp=True,
                surrogate_constraint_scope=args.surrogate_constraint_scope,
            )
            record["solutions"]["x_subproblem_milp"] = x_sub_milp
            record["metrics"]["subproblem_milp_to_true"] = _distance_metrics(x_sub_milp, x_true)
            record["errors_subproblem_milp"] = _binary_error_matrix(x_sub_milp, x_true)
            record["unit_status"].extend(rows)

        if args.run_fp:
            try:
                x_fp, success, fp_details = recover_integer_solution(
                    sample,
                    trainers,
                    dual_predictor,
                    ppc,
                    T_delta,
                    max_fp_iter=args.max_fp_iter,
                    conf_threshold=args.fp_conf_threshold,
                    max_perturbation_hot_starts=args.max_perturbation_hot_starts,
                    max_unit_options_per_generator=args.max_unit_options_per_generator,
                    max_unit_combination_candidates=args.max_unit_combination_candidates,
                    max_nearby_commitment_hot_starts=args.max_nearby_commitment_hot_starts,
                    nearby_commitment_pool_size=args.nearby_commitment_pool_size,
                    parallel_fp_starts=args.parallel_fp_starts,
                    scenario_bank=all_samples,
                    surrogate_screen_mode=args.surrogate_screen_mode,
                    surrogate_screen_max_constraints_per_unit=args.surrogate_screen_max_constraints_per_unit,
                    surrogate_screen_min_support_ratio=args.surrogate_screen_min_support_ratio,
                    surrogate_screen_max_normalized_violation=args.surrogate_screen_max_normalized_violation,
                    surrogate_screen_min_mean_margin=args.surrogate_screen_min_mean_margin,
                    surrogate_screen_candidate_violation_tol=args.surrogate_screen_candidate_violation_tol,
                    surrogate_screen_soft_penalty=args.surrogate_screen_soft_penalty,
                    projection_objective_tau=args.projection_objective_tau,
                    return_details=True,
                    verbose=bool(args.verbose_fp),
                )
                record["solutions"]["x_fp"] = x_fp
                record["metrics"]["fp_to_true"] = _distance_metrics(x_fp, x_true)
                record["fp_success"] = bool(success)
                record["fp_details"] = {
                    "selected_hot_start": fp_details.get("selected_hot_start"),
                    "projection_objective_tau": fp_details.get("projection_objective_tau"),
                    "surrogate_screen_summary": fp_details.get("surrogate_screen_summary"),
                }
            except Exception as exc:
                record["fp_error"] = str(exc)

        records.append(record)

    result = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": args.case,
            "active_set_path": str(active_set_path),
            "model_dir": str(model_dir),
            "t_delta": T_delta,
            "sample_start": int(args.start),
            "sample_count": int(len(selected_samples)),
            "unit_ids": unit_ids if unit_ids is not None else "all",
            "strategy": args.strategy,
            "lp_backend": args.lp_backend,
            "surrogate_constraint_scope": args.surrogate_constraint_scope,
            "bcd_proxy_scope": args.bcd_proxy_scope,
            "run_fp": bool(args.run_fp),
        },
        "records": records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[done] wrote {output_path}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="case3lite")
    p.add_argument("--active-sets", default=None, help="Active-set JSON; default = latest for case.")
    p.add_argument("--model-dir", default=None, help="Trained model directory; default = latest for case.")
    p.add_argument("--output", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--samples", type=int, default=8, help="<=0 means all samples from --start.")
    p.add_argument("--unit-ids", default="all")
    p.add_argument("--t-delta", type=float, default=1.0)
    p.add_argument("--strategy", default=None, help="Constraint generation strategy; default = checkpoint metadata.")
    p.add_argument("--lp-backend", default=None, help="Subproblem LP backend; default = checkpoint metadata.")
    p.add_argument("--ignore-startup-shutdown-costs", type=int, choices=(0, 1), default=None)
    p.add_argument("--use-unit-predictor", action="store_true")
    p.add_argument("--surrogate-constraint-scope", choices=("all", "sign4"), default="all")
    p.add_argument("--bcd-proxy-scope", choices=("both", "theta", "zeta", "none"), default="none")
    p.add_argument("--skip-plain-lp", action="store_true")
    p.add_argument("--skip-proxy-lp", action="store_true")
    p.add_argument("--skip-subproblem", action="store_true")
    p.add_argument("--subproblem-milp", action="store_true")
    p.add_argument("--run-fp", action="store_true")
    p.add_argument("--verbose-fp", action="store_true")
    p.add_argument("--max-fp-iter", type=int, default=50)
    p.add_argument("--fp-conf-threshold", type=float, default=0.15)
    p.add_argument("--max-perturbation-hot-starts", type=int, default=6)
    p.add_argument("--max-unit-options-per-generator", type=int, default=4)
    p.add_argument("--max-unit-combination-candidates", type=int, default=12)
    p.add_argument("--max-nearby-commitment-hot-starts", type=int, default=4)
    p.add_argument("--nearby-commitment-pool-size", type=int, default=12)
    p.add_argument("--parallel-fp-starts", type=int, default=1)
    p.add_argument("--surrogate-screen-mode", default="robust")
    p.add_argument("--surrogate-screen-max-constraints-per-unit", type=int, default=3)
    p.add_argument("--surrogate-screen-min-support-ratio", type=float, default=0.85)
    p.add_argument("--surrogate-screen-max-normalized-violation", type=float, default=0.05)
    p.add_argument("--surrogate-screen-min-mean-margin", type=float, default=0.02)
    p.add_argument("--surrogate-screen-candidate-violation-tol", type=float, default=0.02)
    p.add_argument("--surrogate-screen-soft-penalty", type=float, default=25.0)
    p.add_argument("--projection-objective-tau", default="adaptive")
    return p


def main() -> None:
    collect_diagnostics(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
