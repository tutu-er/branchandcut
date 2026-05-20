#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: tailored Algorithm-II FP vs traditional Algorithm-I FP.

This script loads a case's surrogate models and active-set samples, then
runs both the *vanilla* feasibility pump (``fp_strategy='vanilla'``) and
the *tailored* feasibility pump (``fp_strategy='tailored'``) on the same
sample set. It writes a per-sample CSV plus a JSON summary that lets you
verify the tailored variant outperforms the textbook FP on:

  * feasibility recovery success rate
  * Hamming distance to the true integer solution
  * average wall-clock time per sample

Example:
    python scripts/compare_tailored_vs_vanilla_fp.py ^
        --case case14 ^
        --active-sets result/active_set/active_sets_case14_T24_n600_20260503_222929.json ^
        --model-dir result/surrogate_models/subproblem_models_case14_20260510_ideal ^
        --bcd-model result/bcd_models/bcd_model_case14_20260504_222135.pth ^
        --samples 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers from the diagnostics collector.
from scripts.collect_model_fp_diagnostics import (
    _ensure_runtime_imports,
    _latest_active_set,
    _latest_model_dir,
    _parse_sample_slice,
    _parse_unit_ids,
    _resolve_path,
)


def _json_default(value: Any) -> Any:
    import math
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


def _hamming(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[int]:
    if a is None or b is None:
        return None
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return None
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if not bool(np.any(mask)):
        return None
    a_round = np.rint(a_arr[mask]).astype(int)
    b_round = np.rint(b_arr[mask]).astype(int)
    return int(np.sum(a_round != b_round))


def _extract_x_true(sample: dict, ng: int, T: int) -> Optional[np.ndarray]:
    """Reuse the diagnostics extractor so both pipelines agree on x_true."""
    from scripts.collect_model_fp_diagnostics import _extract_true_solution

    arr = _extract_true_solution(sample, (ng, T))
    if np.isfinite(arr).any():
        return arr
    return None


def _summarize_fp_history(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    histories = details.get("fp_histories") or []
    chi_random_used_any = False
    theta_added_total = 0
    perturbations = 0
    total_iters = 0
    delta_first: Optional[float] = None
    delta_last: Optional[float] = None
    for h in histories:
        for entry in (h.get("history") or []):
            total_iters += 1
            if entry.get("chi_random_used_for_stall"):
                chi_random_used_any = True
            theta_added_total += int(entry.get("theta_resample_added", 0) or 0)
            if entry.get("perturbation_applied"):
                perturbations += 1
            delta_value = entry.get("delta_k", entry.get("l1_projection"))
            if delta_value is not None:
                try:
                    dv = float(delta_value)
                except (TypeError, ValueError):
                    dv = None
                if dv is not None:
                    if delta_first is None:
                        delta_first = dv
                    delta_last = dv
    return {
        "selected_hot_start": details.get("selected_hot_start"),
        "chi_random_used": bool(chi_random_used_any),
        "theta_resample_added_total": int(theta_added_total),
        "total_iterations": int(total_iters),
        "perturbations_triggered": int(perturbations),
        "delta_first": delta_first,
        "delta_last": delta_last,
    }


def _extract_iteration_trace(details: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten ``fp_histories`` into a list of per-iteration rows for plotting.

    Each row carries: ``hot_start_index``, ``hot_start_name``, ``iteration``,
    ``delta_k``, ``delta_hat_k``, ``soft_penalty``, ``tau_cost``,
    ``primal_objective``, ``cycle_hit``, ``equipotential_cycle``,
    ``perturbation_applied``, ``perturbation_mode``, ``chi_random_used``,
    ``theta_resample_added``, ``post_feasible``.
    """
    if not isinstance(details, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for hist in details.get("fp_histories") or []:
        base = {
            "hot_start_index": hist.get("hot_start_index"),
            "hot_start_name": hist.get("hot_start_name"),
            "parallel": bool(hist.get("parallel", False)),
            "entered_fp_iterations": bool(hist.get("entered_fp_iterations", False)),
            "termination": hist.get("termination"),
        }
        for step in hist.get("history") or []:
            rows.append({
                **base,
                "iteration": step.get("iteration"),
                "delta_k": step.get("delta_k", step.get("l1_projection")),
                "delta_hat_k": step.get("delta_hat_k", step.get("phi_hat")),
                "soft_penalty": step.get("soft_penalty"),
                "tau_cost": step.get("tau_cost"),
                "primal_objective": step.get("primal_objective"),
                "changed_bits": step.get("changed_bits"),
                "projection_status": step.get("projection_status"),
                "projection_status_name": step.get("projection_status_name"),
                "step_termination": step.get("termination"),
                "cycle_hit": bool(step.get("cycle_hit")),
                "equipotential_cycle": bool(step.get("equipotential_cycle")),
                "perturbation_applied": bool(step.get("perturbation_applied")),
                "perturbation_mode": step.get("perturbation_mode"),
                "chi_random_used_for_stall": bool(step.get("chi_random_used_for_stall")),
                "theta_resample_added": int(step.get("theta_resample_added", 0) or 0),
                "post_feasible": bool(step.get("post_feasible")),
                "rounding_strategy": step.get("rounding_strategy"),
            })
    return rows


def _run_one_sample(
    label: str,
    sample: dict,
    lambda_val: Any,
    trainers: Dict[int, Any],
    ppc: dict,
    T_delta: float,
    agent: Any,
    *,
    fp_strategy: str,
    rounding_strategy: str,
    chi_alpha: float,
    chi_random_samples: int,
    chi_random_evaluator_weight: float,
    enable_stall_theta_resample: bool,
    max_fp_iter: int,
    verbose: bool,
    extra_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    from src.feasibility_pump import recover_integer_solution

    kwargs = dict(extra_kwargs or {})
    kwargs.update(
        {
            "agent": agent,
            "fp_strategy": fp_strategy,
            "rounding_strategy": rounding_strategy,
            "chi_alpha": chi_alpha,
            "chi_random_samples": chi_random_samples,
            "chi_random_evaluator_weight": chi_random_evaluator_weight,
            "enable_stall_theta_resample": enable_stall_theta_resample,
            "max_fp_iter": max_fp_iter,
            "verbose": verbose,
            "return_details": True,
        }
    )

    # Need a lambda_predictor object with a .predict method that returns the
    # already-computed lambda for the supplied sample (so we don't redo NN
    # work between vanilla and tailored). Wrap it in a tiny shim.
    class _CachedLambdaPredictor:
        def __init__(self, value: Any) -> None:
            self._value = value

        def predict(self, _sample: Any) -> Any:
            return self._value

    cached_predictor = _CachedLambdaPredictor(lambda_val)

    t0 = time.time()
    try:
        result, success, details = recover_integer_solution(
            sample,
            trainers,
            cached_predictor,
            ppc,
            T_delta,
            **kwargs,
        )
        elapsed = time.time() - t0
        summary = _summarize_fp_history(details)
        return {
            "label": label,
            "fp_success": bool(success),
            "x_result": np.asarray(result, dtype=int),
            "wallclock_sec": float(elapsed),
            "details_summary": summary,
            "details": details,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "label": label,
            "fp_success": False,
            "x_result": None,
            "wallclock_sec": float(elapsed),
            "error": str(exc),
            "details_summary": {},
            "details": {},
        }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    succ = [r["fp_success"] for r in results]
    hams = [r["hamming_to_true"] for r in results if r.get("hamming_to_true") is not None]
    times = [r["wallclock_sec"] for r in results]
    iters = [r["details_summary"].get("total_iterations", 0) for r in results]
    return {
        "success_rate": float(np.mean(succ)) if succ else None,
        "success_count": int(sum(succ)),
        "mean_hamming_to_true": float(np.mean(hams)) if hams else None,
        "max_hamming_to_true": int(np.max(hams)) if hams else None,
        "mean_wallclock_sec": float(np.mean(times)) if times else None,
        "mean_total_iterations": float(np.mean(iters)) if iters else None,
        "n_samples": int(len(results)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="case14")
    p.add_argument("--active-sets", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--bcd-model", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--unit-ids", default="all")
    p.add_argument("--t-delta", type=float, default=1.0)
    p.add_argument("--max-fp-iter", type=int, default=50)
    p.add_argument("--skip-feasible-hot-starts", action="store_true",
                   help="Force FP to enter the main loop even if a hot-start "
                        "is already feasible (useful when you want δ_k curves).")
    p.add_argument("--surrogate-constraint-scope", choices=("all", "sign4", "none"), default="all")
    p.add_argument("--bcd-proxy-scope", choices=("both", "theta", "zeta", "none"), default="theta")
    p.add_argument("--tailored-config", default=None,
                   help="Tailored FP config JSON; overlays its kwargs on tailored run only.")
    p.add_argument("--chi-alpha", type=float, default=3.0)
    p.add_argument("--chi-random-samples", type=int, default=8)
    p.add_argument("--chi-random-evaluator-weight", type=float, default=0.05)
    p.add_argument("--output", default=None, help="Summary JSON output path.")
    p.add_argument("--csv-output", default=None, help="Per-sample CSV output path.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-missing-surrogates", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    _ensure_runtime_imports()

    # After _ensure_runtime_imports() the module-level names in
    # collect_model_fp_diagnostics are populated; pull them back into our
    # local namespace so we don't depend on import paths that have moved.
    from scripts import collect_model_fp_diagnostics as cdiag
    from scripts.collect_model_fp_diagnostics import (
        _load_bcd_agent,
        _load_tailored_fp_kwargs,
    )

    get_case_ppc = cdiag.get_case_ppc
    load_v3_active_set_json = cdiag.load_v3_active_set_json
    load_trained_models = cdiag.load_trained_models
    get_sample_net_load = cdiag.get_sample_net_load

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))
    bcd_model_path = (
        _resolve_path(args.bcd_model, None)
        if args.bcd_model is not None and str(args.bcd_model).strip()
        else None
    )

    print(f"[compare] case={args.case}", flush=True)
    print(f"[compare] active_sets={active_set_path}", flush=True)
    print(f"[compare] model_dir={model_dir}", flush=True)
    if bcd_model_path is not None:
        print(f"[compare] bcd_model={bcd_model_path}", flush=True)

    ppc = get_case_ppc(args.case)
    all_samples = load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )
    selected_samples = _parse_sample_slice(all_samples, args.start, args.samples)
    if not selected_samples:
        raise ValueError("No samples selected")

    T_delta = float(args.t_delta)
    unit_ids = _parse_unit_ids(args.unit_ids)

    dual_predictor, trainers = load_trained_models(
        ppc,
        all_samples,
        T_delta,
        str(model_dir),
        unit_ids=unit_ids,
        case_name=args.case,
        skip_initial_solve=True,
    )

    agent = None
    if bcd_model_path is not None:
        agent = _load_bcd_agent(ppc, active_set_path, bcd_model_path, T_delta)

    tailored_overrides = _load_tailored_fp_kwargs(
        _resolve_path(args.tailored_config, None)
        if args.tailored_config else None
    )

    ng = int(np.asarray(ppc["gen"]).shape[0])

    per_sample_rows: List[Dict[str, Any]] = []

    for local_idx, sample in enumerate(selected_samples, start=1):
        sample_id = sample.get("sample_id", sample.get("source_sample_id", args.start + local_idx - 1))
        pd_data = get_sample_net_load(sample)
        T = int(pd_data.shape[1])
        x_true = _extract_x_true(sample, ng, T)

        print(f"[sample {local_idx}/{len(selected_samples)}] sample_id={sample_id}", flush=True)
        try:
            lambda_val = dual_predictor.predict(sample)
        except Exception as exc:
            print(f"  [skip] dual_predictor failed: {exc}", flush=True)
            continue

        vanilla_extra = {
            "surrogate_constraint_scope": args.surrogate_constraint_scope,
            "bcd_proxy_scope": args.bcd_proxy_scope,
        }
        if args.skip_feasible_hot_starts:
            vanilla_extra["skip_feasible_hot_starts"] = True
        vanilla = _run_one_sample(
            "vanilla",
            sample,
            lambda_val,
            trainers,
            ppc,
            T_delta,
            agent,
            fp_strategy="vanilla",
            rounding_strategy="x_round",
            chi_alpha=args.chi_alpha,
            chi_random_samples=args.chi_random_samples,
            chi_random_evaluator_weight=args.chi_random_evaluator_weight,
            enable_stall_theta_resample=False,
            max_fp_iter=args.max_fp_iter,
            verbose=args.verbose,
            extra_kwargs=vanilla_extra,
        )

        tailored_extra = {
            "surrogate_constraint_scope": args.surrogate_constraint_scope,
            "bcd_proxy_scope": args.bcd_proxy_scope,
        }
        if args.skip_feasible_hot_starts:
            tailored_extra["skip_feasible_hot_starts"] = True
        tailored_extra.update(tailored_overrides or {})
        tailored = _run_one_sample(
            "tailored",
            sample,
            lambda_val,
            trainers,
            ppc,
            T_delta,
            agent,
            fp_strategy="tailored",
            rounding_strategy=tailored_extra.pop("rounding_strategy", "chi_argmax"),
            chi_alpha=args.chi_alpha,
            chi_random_samples=args.chi_random_samples,
            chi_random_evaluator_weight=args.chi_random_evaluator_weight,
            enable_stall_theta_resample=True,
            max_fp_iter=args.max_fp_iter,
            verbose=args.verbose,
            extra_kwargs=tailored_extra,
        )

        for result in (vanilla, tailored):
            result["hamming_to_true"] = _hamming(result.get("x_result"), x_true)

        print(
            f"  vanilla : success={vanilla['fp_success']} "
            f"hamming={vanilla['hamming_to_true']} time={vanilla['wallclock_sec']:.2f}s",
            flush=True,
        )
        print(
            f"  tailored: success={tailored['fp_success']} "
            f"hamming={tailored['hamming_to_true']} time={tailored['wallclock_sec']:.2f}s",
            flush=True,
        )

        vanilla_trace = _extract_iteration_trace(vanilla.get("details"))
        tailored_trace = _extract_iteration_trace(tailored.get("details"))

        per_sample_rows.append({
            "sample_id": sample_id,
            "ng": ng,
            "T": T,
            "vanilla": {
                **{k: v for k, v in vanilla.items() if k not in ("x_result", "details")},
                "iteration_trace": vanilla_trace,
            },
            "tailored": {
                **{k: v for k, v in tailored.items() if k not in ("x_result", "details")},
                "iteration_trace": tailored_trace,
            },
        })

    vanilla_results = [
        {
            "fp_success": row["vanilla"]["fp_success"],
            "hamming_to_true": row["vanilla"]["hamming_to_true"],
            "wallclock_sec": row["vanilla"]["wallclock_sec"],
            "details_summary": row["vanilla"]["details_summary"],
        }
        for row in per_sample_rows
    ]
    tailored_results = [
        {
            "fp_success": row["tailored"]["fp_success"],
            "hamming_to_true": row["tailored"]["hamming_to_true"],
            "wallclock_sec": row["tailored"]["wallclock_sec"],
            "details_summary": row["tailored"]["details_summary"],
        }
        for row in per_sample_rows
    ]

    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": args.case,
            "active_set_path": str(active_set_path),
            "model_dir": str(model_dir),
            "bcd_model": None if bcd_model_path is None else str(bcd_model_path),
            "max_fp_iter": int(args.max_fp_iter),
            "n_samples": len(per_sample_rows),
            "tailored_config": (
                None if not args.tailored_config else str(args.tailored_config)
            ),
        },
        "vanilla_aggregate": _aggregate(vanilla_results),
        "tailored_aggregate": _aggregate(tailored_results),
        "per_sample": per_sample_rows,
    }

    output_path = _resolve_path(
        args.output,
        ROOT / "result" / "fp_diagnostics" / (
            f"fp_compare_{args.case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[done] wrote summary {output_path}", flush=True)

    if args.csv_output:
        csv_path = _resolve_path(args.csv_output, None)
        if csv_path is None:
            csv_path = output_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "vanilla_success", "vanilla_hamming", "vanilla_time",
                "tailored_success", "tailored_hamming", "tailored_time",
                "tailored_chi_random_used", "tailored_theta_resample_added",
            ])
            for row in per_sample_rows:
                v = row["vanilla"]
                t = row["tailored"]
                writer.writerow([
                    row["sample_id"],
                    v["fp_success"], v["hamming_to_true"], v["wallclock_sec"],
                    t["fp_success"], t["hamming_to_true"], t["wallclock_sec"],
                    t["details_summary"].get("chi_random_used", False),
                    t["details_summary"].get("theta_resample_added_total", 0),
                ])
        print(f"[done] wrote per-sample CSV {csv_path}", flush=True)

    print("--- aggregate ---", flush=True)
    print(json.dumps({
        "vanilla": summary["vanilla_aggregate"],
        "tailored": summary["tailored_aggregate"],
    }, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
