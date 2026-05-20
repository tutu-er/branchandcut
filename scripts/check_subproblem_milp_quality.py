#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare each hot-start candidate against x_true (Hamming distance).

Focus: how close is the **raw** sign4+single sub-proxy MILP output
(`subproblem_milp_base`) to the true optimal commitment? Also evaluate
the LP-relaxation-based candidates (`lp_round`, `surrogate_lp_round`,
`unit_lp_round`) for context.

If `subproblem_milp_base` is itself already 5+ bits away from `x_true`,
then no amount of FP iteration starting from it is likely to close the
gap -- the bottleneck is the sub-proxy MILP, not the FP scheme.

Output: a JSON + a stdout table summarising per-sample Hamming distances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_model_fp_diagnostics import (  # noqa: E402
    _ensure_runtime_imports,
    _latest_active_set,
    _latest_model_dir,
    _resolve_path,
)
from scripts.compare_tailored_vs_vanilla_fp import _extract_x_true  # noqa: E402


def _hamming(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int(np.sum(np.asarray(a, dtype=int) != np.asarray(b, dtype=int)))


def _round_int(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=float)), 0, 1).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--active-sets", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _ensure_runtime_imports()
    from scripts import collect_model_fp_diagnostics as cdiag
    from src.feasibility_pump import (
        _build_hot_start_candidates,
        _evaluate_commitment_dispatch_cost,
        collect_integer_solutions,
        _repair_min_up_down_heuristic,
    )

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))

    print(f"[check] case={args.case}")
    print(f"[check] active_sets={active_set_path}")
    print(f"[check] model_dir={model_dir}")

    ppc = cdiag.get_case_ppc(args.case)
    all_samples = cdiag.load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )
    samples = all_samples[args.start:args.start + args.samples]
    T_delta = float(args.t_delta)

    dual_predictor, trainers = cdiag.load_trained_models(
        ppc, all_samples, T_delta, str(model_dir),
        unit_ids=None, case_name=args.case, skip_initial_solve=True,
    )

    ng = int(np.asarray(ppc["gen"]).shape[0])

    rows: List[Dict[str, Any]] = []
    for local_idx, sample in enumerate(samples, start=1):
        sid = sample.get("sample_id", sample.get("source_sample_id", args.start + local_idx - 1))
        pd_data = cdiag.get_sample_net_load(sample)
        T = int(pd_data.shape[1])
        x_true = _extract_x_true(sample, ng, T)

        try:
            lam_val = dual_predictor.predict(sample)
        except Exception as exc:
            print(f"  [skip] sample {sid}: predictor failed: {exc}", flush=True)
            continue

        print(f"\n[sample {local_idx}/{len(samples)}] sample_id={sid}", flush=True)

        try:
            x_surr_lp, x_init_k, x_init_k_m, detail = collect_integer_solutions(
                sample, lam_val, trainers,
                n_perturbations=2,
                n_similar_scenarios=0,
                similar_scenario_pool_size=0,
                n_load_perturbations=0,
                load_perturbation_scale=0.05,
                use_milp_candidate=True,
                milp_for_perturbations=True,
                surrogate_constraint_scope="all",
                return_details=True,
                lambda_predictor=dual_predictor,
            )
        except Exception as exc:
            print(f"  [error] collect_integer_solutions: {exc}", flush=True)
            continue

        x_init_k_milp = detail.get("x_init_k_milp")
        x_init_k_m_milp = detail.get("x_init_k_m_milp")

        candidate_rows: Dict[str, np.ndarray] = {}
        candidate_rows["surrogate_lp_round"] = _repair_min_up_down_heuristic(
            _round_int(x_surr_lp), T_delta, ppc=ppc, unit_ids=None
        )
        candidate_rows["unit_lp_round"] = _round_int(x_init_k)
        if x_init_k_milp is not None:
            x_milp = _round_int(np.asarray(x_init_k_milp))
            candidate_rows["subproblem_milp_base"] = _repair_min_up_down_heuristic(
                x_milp, T_delta, ppc=ppc, unit_ids=None
            )
        if x_init_k_m_milp is not None and np.asarray(x_init_k_m_milp).ndim == 3:
            arr = np.asarray(x_init_k_m_milp, dtype=int)
            for m in range(arr.shape[1]):
                candidate_rows[f"subproblem_milp_perturb_{m + 1}"] = (
                    _repair_min_up_down_heuristic(
                        arr[:, m, :], T_delta, ppc=ppc, unit_ids=None
                    )
                )

        row = {"sample_id": sid, "T": T, "ng": ng,
               "hamming_total_bits": ng * T,
               "candidates": {}}
        x_true_obj = None
        x_true_uc_feas = None
        x_true_uc_reason = None
        if x_true is not None:
            try:
                x_true_obj = _evaluate_commitment_dispatch_cost(
                    np.asarray(x_true, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception:
                x_true_obj = None
            try:
                from src.feasibility_pump import check_uc_feasibility
                x_true_uc_feas, x_true_uc_reason = check_uc_feasibility(
                    np.asarray(x_true, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception as exc:
                x_true_uc_reason = f"error: {exc}"
        row["x_true_obj"] = x_true_obj
        row["x_true_uc_feasibility"] = bool(x_true_uc_feas) if x_true_uc_feas is not None else None
        row["x_true_uc_reason"] = str(x_true_uc_reason) if x_true_uc_reason else None

        print(f"  total binary bits = {ng * T}")
        print(f"  x_true dispatch obj = {x_true_obj}")
        print(f"  x_true uc_feas      = {x_true_uc_feas}  reason={x_true_uc_reason!r}")
        from src.feasibility_pump import check_uc_feasibility
        print(f"  {'candidate':30s} {'ham':>4s} {'uc_obj':>14s}  {'uc_feas':>7s}  reason")
        for name, cand in candidate_rows.items():
            ham = _hamming(cand, x_true) if x_true is not None else None
            try:
                uc_obj = _evaluate_commitment_dispatch_cost(
                    np.asarray(cand, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception:
                uc_obj = None
            try:
                feas, reason = check_uc_feasibility(
                    np.asarray(cand, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception as exc:
                feas, reason = False, f"check err: {exc}"
            uc_str = f"{uc_obj:.2f}" if uc_obj is not None else "DISP_INFEAS"
            print(f"  {name:30s} {str(ham):>4s} {uc_str:>14s}  {str(feas):>7s}  {reason}")
            row["candidates"][name] = {
                "hamming_to_true": ham,
                "uc_dispatch_obj": uc_obj,
                "uc_feasibility": bool(feas),
                "uc_reason": str(reason),
            }

        # Sanity: try x_true itself, and a "1-bit-from-x_true" mutation
        if x_true is not None:
            try:
                feas_true, reason_true = check_uc_feasibility(
                    np.asarray(x_true, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception as exc:
                feas_true, reason_true = False, f"check err: {exc}"
            uc_obj_true = x_true_obj if x_true_obj is not None else None
            uc_str_true = f"{uc_obj_true:.2f}" if uc_obj_true is not None else "DISP_INFEAS"
            print(f"  {'== x_true (reference) ==':30s} {0:>4d} {uc_str_true:>14s}  {str(feas_true):>7s}  {reason_true}")
            row["candidates"]["x_true"] = {
                "hamming_to_true": 0,
                "uc_dispatch_obj": uc_obj_true,
                "uc_feasibility": bool(feas_true),
                "uc_reason": str(reason_true),
            }
        rows.append(row)

    summary = {
        "case": args.case,
        "n_samples": len(rows),
        "ng_x_T": ng * (rows[0]["T"] if rows else 0),
        "per_sample": rows,
    }

    out_path = _resolve_path(
        args.output,
        ROOT / "result" / "fp_diagnostics" / f"subproblem_milp_quality_{args.case}.json",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_default, ensure_ascii=False)
    print(f"\n[done] wrote {out_path}")

    if rows:
        candidate_names = list(rows[0]["candidates"].keys())
        print("\n=== aggregate Hamming distances (mean ± std) over samples ===")
        for name in candidate_names:
            hams = [r["candidates"][name]["hamming_to_true"] for r in rows
                    if r["candidates"][name]["hamming_to_true"] is not None]
            uc_objs = [r["candidates"][name]["uc_dispatch_obj"] for r in rows
                       if r["candidates"][name]["uc_dispatch_obj"] is not None]
            mean_ham = float(np.mean(hams)) if hams else None
            std_ham = float(np.std(hams)) if hams else None
            min_ham = int(np.min(hams)) if hams else None
            max_ham = int(np.max(hams)) if hams else None
            print(
                f"  {name:30s}  ham mean={mean_ham:.2f}  std={std_ham:.2f}  "
                f"min={min_ham}  max={max_ham}  "
                f"uc_obj_coverage={len(uc_objs)}/{len(rows)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
