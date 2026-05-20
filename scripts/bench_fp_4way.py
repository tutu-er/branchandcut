#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4-way FP benchmark: vanilla / tailored-full / tailored-no_history /
tailored-no_history+engagement=K.

For every sample we run the four strategies and capture:
  * fp_success (bool)
  * hamming_to_true
  * uc_dispatch_obj (cost of the returned commitment)
  * total_fp_iterations (sum over hot-starts that entered the main loop)
  * wallclock_sec
  * selected_hot_start

A JSON summary + CSV table are written alongside.
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

# Re-use compare_tailored_vs_vanilla_fp helpers for loading.
from scripts.collect_model_fp_diagnostics import (  # noqa: E402
    _ensure_runtime_imports,
    _latest_active_set,
    _latest_model_dir,
    _parse_sample_slice,
    _resolve_path,
)
from scripts.compare_tailored_vs_vanilla_fp import (  # noqa: E402
    _extract_x_true,
    _hamming,
)


STRATEGIES = [
    # === DEFAULT STRATEGY (history-aware theta-flip + FP iterations) ===
    # Build a candidate pool from theta-only LP, full LP harvest, round-repair
    # and top-K historical error-bit flips, then run the FP projection loop.
    # Requires --error-bit-map at the CLI.
    # Listed first so plots and CSV rows put it in the leftmost slot.
    {
        "label": "theta_flip_case3lite",
        "fp_strategy": "n/a",
        "extras": {
            "mode": "theta_flip_case3lite",
            "flip_top_k": 10,
            "suspicious_error_rate_min": 0.15,
            "harvest_full_lp": True,
            "defer_full_lp_to_fp": True,
            "full_lp_inject_iter": 3,
            "run_fp_iterations": True,
            "fp_max_iter": 10,
            "fp_min_iter": 3,
            "max_global_combinations": 24,
        },
    },
    {
        "label": "theta_flip",
        "fp_strategy": "n/a",
        "extras": {
            "mode": "theta_flip",
            "flip_top_k": 10,
            "initial_bcd_proxy_scope": "theta",
            "harvest_full_lp": True,
            "defer_full_lp_to_fp": True,
            "full_lp_inject_iter": 3,
            "rescue_when_infeasible": True,
            "rescue_max_combinations": 30,
            "rescue_max_combo_size": 3,
            "run_fp_iterations": True,
            "fp_max_iter": 10,
            "fp_min_iter": 3,
        },
    },
    {
        "label": "vanilla",
        "fp_strategy": "vanilla",
        "extras": {
            # Textbook FP baseline should actually enter the projection loop.
            # Otherwise a feasible LP-round hot start is accepted at precheck
            # time and reports iters=0, which is not a fair iteration trace.
            "skip_feasible_hot_starts": True,
            # Classical FP terminates once the rounded integer state cycles
            # (or an equipotential plateau is detected).  Do not force extra
            # iterations after that.
            "min_fp_iter_before_feasible_accept": 0,
            "terminate_on_cycle": True,
        },
    },
    {
        "label": "tailored_full",
        "fp_strategy": "tailored",
        "extras": {"rounding_strategy": "chi_argmax"},
    },
    {
        "label": "tailored_no_history",
        "fp_strategy": "tailored",
        "extras": {
            "rounding_strategy": "chi_argmax",
            "disable_historical_hot_starts": True,
            # Skip 4 dedupe-redundant mix blends in hot-start build + cap
            # combinatorial enumeration to cut per-iter overhead ~30 %.
            "lean_hot_starts": True,
            "max_perturbation_hot_starts": 2,
            "max_unit_combination_candidates": 4,
        },
    },
    {
        "label": "tailored_no_history_engage3",
        "fp_strategy": "tailored",
        "extras": {
            "rounding_strategy": "chi_argmax",
            "disable_historical_hot_starts": True,
            "surrogate_engagement_iter": 3,
            "lean_hot_starts": True,
            "max_perturbation_hot_starts": 2,
            "max_unit_combination_candidates": 4,
        },
    },
    {
        "label": "tailored_no_history_bcd_refresh",
        "fp_strategy": "tailored",
        "extras": {
            "rounding_strategy": "chi_argmax",
            "disable_historical_hot_starts": True,
            "lean_hot_starts": True,
            "max_perturbation_hot_starts": 2,
            "max_unit_combination_candidates": 4,
            # Re-run global BCD-LP + per-unit MILP after stall_K cycle hits
            # OR every interval_M iterations, capped at refresh_max calls.
            "noh_milp_refresh_stall": 3,
            "noh_milp_refresh_interval": 8,
            "noh_milp_refresh_max": 3,
            # CRITICAL: keep σ small (≤0.01). σ=0.1 was found to push MILP
            # candidates to ham=44+ (see check_subproblem_milp_quality.log).
            "noh_milp_refresh_lambda_sigma": 0.01,
            "enable_pool_tabu_prune": True,
            "pool_tabu_drop_threshold": 3,
        },
    },
]


def _run_one(label: str,
             sample: dict,
             cached_predictor,
             trainers,
             ppc: dict,
             T_delta: float,
             agent,
             fp_strategy: str,
             extras: Dict[str, Any],
             max_fp_iter: int,
             x_true: Optional[np.ndarray]) -> Dict[str, Any]:
    from src.feasibility_pump import (
        _evaluate_commitment_dispatch_cost,
        recover_integer_solution,
        recover_via_theta_flip,
    )

    extras_local = dict(extras or {})
    mode = str(extras_local.pop('mode', 'fp')).lower()
    kwargs = {
        "agent": agent,
        "verbose": False,
        "return_details": True,
    }

    t0 = time.time()
    try:
        if mode == 'theta_flip':
            tf_kwargs = dict(extras_local)
            # Make sure error_bit_map is loaded once and reused
            x_rec, ok, details = recover_via_theta_flip(
                sample, trainers, cached_predictor, ppc, T_delta,
                agent=agent,
                verbose=False,
                return_details=True,
                **tf_kwargs,
            )
        elif mode == 'theta_flip_case3lite':
            from src.feasibility_pump_case3lite import recover_via_theta_flip_case3lite

            tf_kwargs = dict(extras_local)
            x_rec, ok, details = recover_via_theta_flip_case3lite(
                sample, trainers, cached_predictor, ppc, T_delta,
                agent=agent,
                verbose=False,
                return_details=True,
                **tf_kwargs,
            )
        else:
            kwargs["fp_strategy"] = fp_strategy
            kwargs["max_fp_iter"] = max_fp_iter
            kwargs.update(extras_local)
            x_rec, ok, details = recover_integer_solution(
                sample, trainers, cached_predictor, ppc, T_delta, **kwargs,
            )
        elapsed = time.time() - t0
    except Exception as exc:
        return {
            "label": label, "fp_success": False, "wallclock_sec": time.time() - t0,
            "error": str(exc), "hamming_to_true": None,
            "uc_dispatch_obj": None, "total_iterations": 0,
            "selected_hot_start": None,
        }

    x_int = (
        np.round(np.asarray(x_rec, dtype=float)).astype(int)
        if x_rec is not None else None
    )
    ham = _hamming(x_int, x_true) if x_true is not None else None
    uc_obj: Optional[float] = None
    if x_int is not None:
        try:
            uc_obj = _evaluate_commitment_dispatch_cost(
                x_int, ppc, sample.get('pd') if isinstance(sample, dict) else sample,
                T_delta,
            )
        except Exception:
            try:
                from scripts.collect_model_fp_diagnostics import get_sample_net_load
                pd_data = get_sample_net_load(sample)
                uc_obj = _evaluate_commitment_dispatch_cost(
                    x_int, ppc, pd_data, T_delta,
                )
            except Exception as exc2:
                print(f"    [{label}] uc_obj eval failed: {exc2}", flush=True)
                uc_obj = None

    total_iter = 0
    milp_refresh_invocations = 0
    pool_size_max = 0
    tabu_drops_max = 0
    iteration_trace: List[Dict[str, Any]] = []
    for h in details.get("fp_histories") or []:
        hist_rows = h.get("history") or []
        hot_name = h.get("name") or h.get("hot_start_name") or label
        hot_index = len(iteration_trace)
        total_iter += len(hist_rows)
        for r in hist_rows:
            rr = dict(r)
            rr.setdefault("hot_start_name", hot_name)
            rr.setdefault("hot_start_index", hot_index)
            rr.setdefault("event", "fp_iteration")
            iteration_trace.append(rr)
            milp_refresh_invocations = max(
                milp_refresh_invocations,
                int(r.get("noh_milp_refresh_used_so_far") or 0),
            )
            pool_size_max = max(pool_size_max, int(r.get("candidate_pool_size") or 0))
            tabu_drops_max = max(tabu_drops_max, int(r.get("pool_tabu_counts_max") or 0))

    selected_hot = details.get("selected_hot_start") or details.get("selected")
    return {
        "label": label,
        "fp_success": bool(ok),
        "wallclock_sec": float(elapsed),
        "hamming_to_true": ham,
        "uc_dispatch_obj": uc_obj,
        "total_iterations": int(total_iter),
        "selected_hot_start": selected_hot,
        "best_feasible_summary": details.get("best_feasible_summary"),
        "milp_refresh_invocations": int(milp_refresh_invocations),
        "pool_size_max": int(pool_size_max),
        "tabu_counts_max": int(tabu_drops_max),
        "noh_milp_refresh_callback_available": bool(
            details.get("noh_milp_refresh_callback_available")
        ),
        "theta_flip_n_candidates": int(details.get("n_candidates") or 0)
            if mode.startswith('theta_flip') else 0,
        "theta_flip_n_flipped": int(details.get("n_flipped") or 0)
            if mode.startswith('theta_flip') else 0,
        "theta_flip_n_suspicious_bits": int(details.get("n_suspicious_bits") or 0)
            if mode == 'theta_flip_case3lite' else None,
        "theta_flip_initial_bcd_proxy_scope": details.get("initial_bcd_proxy_scope")
            if mode.startswith('theta_flip') else None,
        "theta_flip_initial_lp_stats": details.get("initial_lp_stats")
            if mode == 'theta_flip' else None,
        "initial_lp_stats": details.get("initial_lp_stats"),
        "iteration_trace": iteration_trace,
    }


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    succ = [r.get("fp_success") for r in rows]
    hams = [r.get("hamming_to_true") for r in rows if r.get("hamming_to_true") is not None]
    times = [r.get("wallclock_sec", 0.0) for r in rows]
    iters = [r.get("total_iterations", 0) for r in rows]
    objs = [r.get("uc_dispatch_obj") for r in rows if r.get("uc_dispatch_obj") is not None]
    return {
        "n_samples": len(rows),
        "success_rate": float(np.mean(succ)) if succ else None,
        "success_count": int(sum(succ)),
        "mean_hamming": float(np.mean(hams)) if hams else None,
        "max_hamming": int(np.max(hams)) if hams else None,
        "mean_time_sec": float(np.mean(times)) if times else None,
        "mean_iterations": float(np.mean(iters)) if iters else None,
        "mean_uc_obj": float(np.mean(objs)) if objs else None,
        "uc_obj_coverage": int(len(objs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--active-sets", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--bcd-model", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-fp-iter", type=int, default=20)
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--engagement-iter", type=int, default=3,
                        help="K for the 'tailored_no_history_engageK' strategy")
    parser.add_argument("--error-bit-map", default=None,
                        help="JSON map from build_history_error_bit_map.py; "
                             "if provided, enables the 'theta_flip' strategy.")
    parser.add_argument("--flip-top-k", type=int, default=10,
                        help="Top-K error-prone bits to flip in theta_flip strategy.")
    parser.add_argument("--theta-flip-initial-bcd-scope",
                        choices=["both", "theta", "zeta", "none"],
                        default="theta",
                        help="BCD proxy rows used by theta_flip's initial LP. "
                             "Default 'theta' reproduces the theta-only variant; "
                             "'both' uses the full theta+zeta BCD model.")
    parser.add_argument("--strategies", default="all",
                        help="Comma-separated strategy labels to run, e.g. "
                             "'theta_flip' or 'theta_flip,vanilla'. Default: all.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()

    # Allow CLI override of the engagement-iter for the engageK strategy.
    if args.engagement_iter != 3:
        for s in STRATEGIES:
            if s.get("label", "").startswith("tailored_no_history_engage"):
                s["extras"]["surrogate_engagement_iter"] = int(args.engagement_iter)
                s["label"] = f"tailored_no_history_engage{int(args.engagement_iter)}"
                break

    # Filter / configure theta-flip strategies at runtime depending on whether
    # --error-bit-map was provided.
    for idx in reversed(range(len(STRATEGIES))):
        label = STRATEGIES[idx].get("label", "")
        if not label.startswith("theta_flip"):
            continue
        if args.error_bit_map:
            STRATEGIES[idx]["extras"]["error_bit_map"] = str(
                _resolve_path(args.error_bit_map, None)
            )
            STRATEGIES[idx]["extras"]["flip_top_k"] = int(args.flip_top_k)
            if label == "theta_flip":
                STRATEGIES[idx]["extras"]["initial_bcd_proxy_scope"] = str(
                    args.theta_flip_initial_bcd_scope
                )
        else:
            print(
                f"[bench] WARNING: --error-bit-map not provided; "
                f"skipping strategy `{label}`.",
                flush=True,
            )
            del STRATEGIES[idx]

    if str(args.strategies).strip().lower() not in {"", "all", "*"}:
        wanted = {
            s.strip()
            for s in str(args.strategies).split(",")
            if s.strip()
        }
        STRATEGIES[:] = [s for s in STRATEGIES if s.get("label") in wanted]
        missing = sorted(wanted - {s.get("label") for s in STRATEGIES})
        if missing:
            print(f"[bench] WARNING: requested strategies not found: {missing}",
                  flush=True)
        if not STRATEGIES:
            raise ValueError("No strategies selected after applying --strategies")

    _ensure_runtime_imports()
    from scripts import collect_model_fp_diagnostics as cdiag
    from scripts.collect_model_fp_diagnostics import _load_bcd_agent

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))
    bcd_model_path = (
        _resolve_path(args.bcd_model, None) if args.bcd_model else None
    )

    print(f"[bench] case={args.case}")
    print(f"[bench] active_sets={active_set_path}")
    print(f"[bench] model_dir={model_dir}")
    if bcd_model_path:
        print(f"[bench] bcd_model={bcd_model_path}")

    ppc = cdiag.get_case_ppc(args.case)
    all_samples = cdiag.load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )
    selected = _parse_sample_slice(all_samples, args.start, args.samples)
    if not selected:
        raise ValueError("No samples selected")

    T_delta = float(args.t_delta)
    dual_predictor, trainers = cdiag.load_trained_models(
        ppc, all_samples, T_delta, str(model_dir),
        unit_ids=None, case_name=args.case, skip_initial_solve=True,
    )
    agent = (
        _load_bcd_agent(ppc, active_set_path, bcd_model_path, T_delta)
        if bcd_model_path else None
    )

    ng = int(np.asarray(ppc["gen"]).shape[0])

    per_sample: List[Dict[str, Any]] = []

    for local_idx, sample in enumerate(selected, start=1):
        sid = sample.get("sample_id", sample.get("source_sample_id", args.start + local_idx - 1))
        pd_data = cdiag.get_sample_net_load(sample)
        T = int(pd_data.shape[1])
        x_true = _extract_x_true(sample, ng, T)
        try:
            lam_val = dual_predictor.predict(sample)
        except Exception as exc:
            print(f"  [skip] sample {sid}: predictor failed: {exc}")
            continue
        cached = type("Cd", (), {"predict": lambda self, _s, _v=lam_val: _v})()

        print(f"\n[sample {local_idx}/{len(selected)}] sample_id={sid}", flush=True)
        row = {"sample_id": sid, "ng": ng, "T": T, "results": {}}
        for cfg in STRATEGIES:
            label = cfg["label"]
            result = _run_one(
                label, sample, cached, trainers, ppc, T_delta, agent,
                cfg["fp_strategy"], cfg["extras"], int(args.max_fp_iter), x_true,
            )
            print(
                f"  {label:32s}: success={result['fp_success']}  "
                f"ham={result['hamming_to_true']}  uc_obj={result['uc_dispatch_obj']}  "
                f"iters={result['total_iterations']}  t={result['wallclock_sec']:.2f}s  "
                f"hot={result['selected_hot_start']}  "
                f"milp_refresh={result.get('milp_refresh_invocations', 0)}  "
                f"pool_max={result.get('pool_size_max', 0)}  "
                f"tabu_max={result.get('tabu_counts_max', 0)}",
                flush=True,
            )
            row["results"][label] = result
        per_sample.append(row)

    aggregates = {}
    for cfg in STRATEGIES:
        label = cfg["label"]
        rows = [r["results"].get(label, {}) for r in per_sample]
        aggregates[label] = _agg(rows)

    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": args.case,
            "active_set_path": str(active_set_path),
            "model_dir": str(model_dir),
            "bcd_model": None if bcd_model_path is None else str(bcd_model_path),
            "max_fp_iter": int(args.max_fp_iter),
            "n_samples": len(per_sample),
            "strategies": [{"label": c["label"], "fp_strategy": c["fp_strategy"],
                             "extras": c["extras"]} for c in STRATEGIES],
        },
        "aggregates": aggregates,
        "per_sample": per_sample,
    }

    out_path = _resolve_path(
        args.output,
        ROOT / "result" / "fp_diagnostics" / (
            f"bench_fp_4way_{args.case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_default, ensure_ascii=False)
    print(f"\n[done] summary -> {out_path}")

    csv_path = (
        _resolve_path(args.csv_output, None) if args.csv_output else out_path.with_suffix(".csv")
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id"]
        for cfg in STRATEGIES:
            lab = cfg["label"]
            header += [
                f"{lab}_success", f"{lab}_ham", f"{lab}_uc_obj",
                f"{lab}_iters", f"{lab}_time",
            ]
        w.writerow(header)
        for row in per_sample:
            line = [row["sample_id"]]
            for cfg in STRATEGIES:
                r = row["results"].get(cfg["label"], {})
                line += [
                    r.get("fp_success"), r.get("hamming_to_true"), r.get("uc_dispatch_obj"),
                    r.get("total_iterations"), r.get("wallclock_sec"),
                ]
            w.writerow(line)
    print(f"[done] csv     -> {csv_path}")

    print("\n=== aggregates ===")
    for label, agg in aggregates.items():
        print(f"  {label:32s}: succ={agg['success_rate']}  "
              f"mean_ham={agg['mean_hamming']}  "
              f"mean_uc_obj={agg['mean_uc_obj']}  "
              f"mean_iters={agg['mean_iterations']}  "
              f"mean_t={agg['mean_time_sec']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
