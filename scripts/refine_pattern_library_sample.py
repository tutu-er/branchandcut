"""Batch targeted secondary optimization for selected sample_ids under a pattern library.

This script is intentionally self-contained for the current case118 workflow:
- all input/output parameters are embedded in ``main()``
- it reads the original active-set JSON, the saved pattern-library JSON,
  and the derived active_set-like JSON
- it summarizes which samples have large restricted-vs-optimal cost gaps
- it runs targeted local refinement for a batch of sample_ids
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

try:
    from src.commitment_clustering import (
        _load_commitment_samples_from_json,
        _pattern_to_key,
        evaluate_commitment_cost,
        solve_pattern_restricted_uc,
    )
    from src.mti118_data_loader import load_case118_ppc_with_mti_limits
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def _load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_allowed_patterns_from_result(pattern_library_json: Dict) -> List[List[np.ndarray]]:
    allowed_patterns: List[List[np.ndarray]] = []
    for generator_entry in pattern_library_json["pattern_library"]:
        patterns_g: List[np.ndarray] = []
        for pattern_entry in generator_entry.get("patterns", []):
            pattern = np.asarray([int(ch) for ch in str(pattern_entry["pattern"])], dtype=int)
            patterns_g.append(pattern)
        allowed_patterns.append(patterns_g)
    return allowed_patterns


def _find_sample(samples: List[Dict], sample_id: int) -> Dict:
    for sample in samples:
        if int(sample["sample_id"]) == int(sample_id):
            return sample
    raise KeyError(f"sample_id={sample_id} not found")


def _find_active_set_like_sample(active_set_like_json: Dict, sample_id: int) -> Optional[Dict]:
    for sample in active_set_like_json.get("all_samples", []):
        if int(sample["sample_id"]) == int(sample_id):
            return sample
    return None


def _solve_once(
    ppc: dict,
    sample: Dict,
    allowed_patterns: List[List[np.ndarray]],
    t_delta: float,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
) -> Dict:
    x_sol, obj_val, status, selected_indices = solve_pattern_restricted_uc(
        ppc=ppc,
        scenario=sample,
        allowed_patterns=allowed_patterns,
        T_delta=t_delta,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    matched_units = None
    changed_units = None
    if x_sol is not None:
        x_opt = np.asarray(sample["unit_commitment_matrix"], dtype=int)
        matched_units = int(sum(
            1 for g in range(x_opt.shape[0])
            if np.array_equal(np.asarray(x_sol[g, :], dtype=int), x_opt[g, :])
        ))
        changed_units = int(x_opt.shape[0] - matched_units)

    return {
        "success": x_sol is not None,
        "status": status,
        "objective_value": obj_val,
        "x_sol": x_sol,
        "selected_pattern_indices": selected_indices,
        "matched_unit_patterns": matched_units,
        "changed_unit_patterns": changed_units,
    }


def _cost_gap_pct(restricted_cost: float, optimal_cost: float) -> Optional[float]:
    if optimal_cost is None or not np.isfinite(optimal_cost) or optimal_cost <= 0:
        return None
    return float((restricted_cost - optimal_cost) / optimal_cost * 100.0)


def _ensure_pattern(
    allowed_patterns: List[List[np.ndarray]],
    generator_id: int,
    pattern_key: Tuple[int, ...],
) -> bool:
    for existing in allowed_patterns[generator_id]:
        if _pattern_to_key(existing) == pattern_key:
            return False
    allowed_patterns[generator_id].append(np.asarray(pattern_key, dtype=int))
    return True


def _clone_patterns(allowed_patterns: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    return [
        [np.asarray(pattern, dtype=int).copy() for pattern in patterns_g]
        for patterns_g in allowed_patterns
    ]


def _safe_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _serialize_dual_payload(payload: Dict) -> Dict:
    return {
        key: np.asarray(value, dtype=float).tolist()
        for key, value in payload.items()
    }


def _dual_payload_for_x_matrix(
    ppc: dict,
    sample: Dict,
    t_delta: float,
    x_sol: np.ndarray,
    refresh_source: str,
) -> Dict[str, object]:
    """与 convert_pattern_library_to_active_set_like / build_refined_case118_active_set 一致：固定 x 后解 ED 提取全局对偶。"""
    from src.uc_NN_subproblem import (
        _build_generator_injection_sensitivity,
        _solve_global_dual_payload_from_ed,
    )

    pd_data = np.asarray(sample.get("pd_data", sample.get("load_data")), dtype=float)
    renewable_data = sample.get("renewable_data")
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    x_arr = np.asarray(x_sol, dtype=float)
    generator_injection_sensitivity = _build_generator_injection_sensitivity(ppc)
    payload = _solve_global_dual_payload_from_ed(
        ppc,
        pd_data,
        float(t_delta),
        x_arr,
        generator_injection_sensitivity,
        renewable_data=renewable_arr,
    )
    serialized = _serialize_dual_payload(payload)
    return {
        "lambda": serialized,
        "lambda_pg_electricity_price": serialized["lambda_pg_effective"],
        "lambda_refresh_source": refresh_source,
    }


def _try_attach_dual(
    ppc: dict,
    sample: Dict,
    t_delta: float,
    x_sol: np.ndarray | None,
    refresh_source: str,
    sample_id: int,
    label: str,
) -> Dict[str, object] | None:
    if x_sol is None:
        return None
    try:
        return _dual_payload_for_x_matrix(ppc, sample, t_delta, x_sol, refresh_source)
    except Exception as exc:
        print(
            f"  WARNING sample_id={sample_id} {label}: dual refresh failed ({refresh_source}): {exc}",
            flush=True,
        )
        return {
            "lambda_dual_refresh_failed": True,
            "lambda_dual_refresh_error": str(exc),
            "lambda_refresh_source": refresh_source,
        }


def _summarize_solution(tag: str, solve_info: Dict, optimal_cost: float) -> None:
    gap_pct = None
    if solve_info["success"]:
        gap_pct = _cost_gap_pct(solve_info["objective_value"], optimal_cost)
    print(f"\n[{tag}]")
    print(f"  success={solve_info['success']}")
    print(f"  status={solve_info['status']}")
    print(f"  objective={solve_info['objective_value']}")
    print(f"  gap_pct={gap_pct}")
    print(f"  matched_unit_patterns={solve_info.get('matched_unit_patterns')}")
    print(f"  changed_unit_patterns={solve_info.get('changed_unit_patterns')}")


def _compute_optimal_cost(ppc: dict, sample: Dict, t_delta: float) -> float:
    optimal_cost = sample.get("optimal_cost")
    if optimal_cost is not None and np.isfinite(optimal_cost):
        return float(optimal_cost)
    eval_result = evaluate_commitment_cost(
        ppc,
        sample["load_data"],
        sample["unit_commitment_matrix"],
        t_delta,
        renewable_data=sample.get("renewable_data"),
    )
    if eval_result["success"]:
        return float(eval_result["total_cost"])
    return float("inf")


def _top_gap_samples(pattern_library_data: Dict, top_k: int) -> List[Dict]:
    scenario_results = pattern_library_data.get("scenario_results") or pattern_library_data.get("scenarios") or []
    rows: List[Dict] = []
    for result in scenario_results:
        gap = result.get("cost_increase_pct")
        if result.get("success") and gap is not None:
            rows.append({
                "sample_id": int(result["sample_id"]),
                "gap_pct": float(gap),
                "changed_unit_patterns": int(result.get("changed_unit_patterns") or 0),
                "matched_unit_patterns": int(result.get("matched_unit_patterns") or 0),
                "solver_status": result.get("solver_status"),
            })
    rows.sort(key=lambda row: row["gap_pct"], reverse=True)
    return rows[:top_k]


def _refine_one_sample(
    ppc: dict,
    sample: Dict,
    sample_like: Optional[Dict],
    base_allowed_patterns: List[List[np.ndarray]],
    t_delta: float,
    base_time_limit: float,
    greedy_trial_time_limit: float,
    final_time_limit: float,
    mip_gap: float,
    verbose_solver: bool,
    max_greedy_steps: int,
) -> Dict:
    sample_id = int(sample["sample_id"])
    x_opt = np.asarray(sample["unit_commitment_matrix"], dtype=int)
    optimal_cost = _compute_optimal_cost(ppc, sample, t_delta)

    allowed_patterns = _clone_patterns(base_allowed_patterns)
    baseline = _solve_once(
        ppc=ppc,
        sample=sample,
        allowed_patterns=allowed_patterns,
        t_delta=t_delta,
        time_limit=base_time_limit,
        mip_gap=mip_gap,
        verbose=verbose_solver,
    )
    _summarize_solution(f"sample {sample_id} baseline", baseline, optimal_cost)

    baseline_matches_active_set_like = None
    if sample_like is not None and baseline["x_sol"] is not None:
        x_like = np.asarray(sample_like["unit_commitment_matrix"], dtype=int)
        baseline_matches_active_set_like = bool(np.array_equal(x_like, baseline["x_sol"]))
        print(f"  baseline_matches_active_set_like={baseline_matches_active_set_like}")

    mismatched_generators = [
        g for g in range(x_opt.shape[0])
        if baseline["x_sol"] is None
        or not np.array_equal(np.asarray(baseline["x_sol"][g, :], dtype=int), x_opt[g, :])
    ]
    print(f"  mismatched_generators={mismatched_generators}")

    greedy_patterns = _clone_patterns(allowed_patterns)
    greedy_history: List[Dict] = []
    remaining_generators = mismatched_generators[:]
    current_best = copy.deepcopy(baseline)

    for step in range(1, max_greedy_steps + 1):
        best_step_info = None
        best_generator = None

        for generator_id in remaining_generators:
            candidate_patterns = _clone_patterns(greedy_patterns)
            added = _ensure_pattern(
                candidate_patterns,
                generator_id=generator_id,
                pattern_key=_pattern_to_key(x_opt[generator_id, :]),
            )
            if not added:
                continue

            candidate = _solve_once(
                ppc=ppc,
                sample=sample,
                allowed_patterns=candidate_patterns,
                t_delta=t_delta,
                time_limit=greedy_trial_time_limit,
                mip_gap=mip_gap,
                verbose=verbose_solver,
            )
            if not candidate["success"]:
                continue
            if (
                not current_best["success"]
                or candidate["objective_value"] + 1e-6 < current_best["objective_value"]
            ):
                if best_step_info is None or (
                    candidate["objective_value"] + 1e-6 < best_step_info["objective_value"]
                ):
                    best_step_info = candidate
                    best_generator = generator_id

        if best_step_info is None or best_generator is None:
            print(f"  greedy step {step}: no improving single-generator augmentation found")
            break

        _ensure_pattern(
            greedy_patterns,
            generator_id=best_generator,
            pattern_key=_pattern_to_key(x_opt[best_generator, :]),
        )
        current_best = best_step_info
        remaining_generators = [g for g in remaining_generators if g != best_generator]

        greedy_entry = {
            "step": step,
            "chosen_generator": int(best_generator),
            "objective_value": float(current_best["objective_value"]),
            "gap_pct": _cost_gap_pct(current_best["objective_value"], optimal_cost),
            "matched_unit_patterns": current_best.get("matched_unit_patterns"),
            "changed_unit_patterns": current_best.get("changed_unit_patterns"),
            "status": current_best.get("status"),
        }
        greedy_history.append(greedy_entry)
        print(
            f"  greedy step {step}: add generator {best_generator} optimal pattern -> "
            f"objective={current_best['objective_value']}, gap_pct={greedy_entry['gap_pct']}"
        )

    greedy_best = current_best
    _summarize_solution(f"sample {sample_id} greedy_best", greedy_best, optimal_cost)

    full_repair_patterns = _clone_patterns(allowed_patterns)
    full_repair_added_generators: List[int] = []
    for generator_id in mismatched_generators:
        added = _ensure_pattern(
            full_repair_patterns,
            generator_id=generator_id,
            pattern_key=_pattern_to_key(x_opt[generator_id, :]),
        )
        if added:
            full_repair_added_generators.append(generator_id)

    full_repair = _solve_once(
        ppc=ppc,
        sample=sample,
        allowed_patterns=full_repair_patterns,
        t_delta=t_delta,
        time_limit=final_time_limit,
        mip_gap=mip_gap,
        verbose=verbose_solver,
    )
    _summarize_solution(f"sample {sample_id} full_repair", full_repair, optimal_cost)

    optimal_dual = _try_attach_dual(
        ppc,
        sample,
        t_delta,
        x_opt,
        "recomputed_from_ed_optimal_commitment",
        sample_id,
        "optimal_x",
    )
    baseline_dual = _try_attach_dual(
        ppc,
        sample,
        t_delta,
        baseline.get("x_sol"),
        "recomputed_from_ed_after_pattern_restricted_baseline",
        sample_id,
        "baseline_x",
    )
    greedy_best_dual = _try_attach_dual(
        ppc,
        sample,
        t_delta,
        greedy_best.get("x_sol"),
        "recomputed_from_ed_after_pattern_restricted_greedy_best",
        sample_id,
        "greedy_best_x",
    )
    full_repair_dual = _try_attach_dual(
        ppc,
        sample,
        t_delta,
        full_repair.get("x_sol"),
        "recomputed_from_ed_after_pattern_restricted_full_repair",
        sample_id,
        "full_repair_x",
    )

    def _merge_dual(block: Dict[str, object], dual: Dict[str, object] | None) -> Dict[str, object]:
        if dual:
            return {**block, **dual}
        return block

    return {
        "sample_id": sample_id,
        "optimal_cost": optimal_cost,
        "optimal_dual": optimal_dual,
        "baseline_matches_active_set_like": baseline_matches_active_set_like,
        "mismatched_generators": mismatched_generators,
        "baseline": _merge_dual(
            {
                "success": baseline["success"],
                "status": baseline["status"],
                "objective_value": baseline["objective_value"],
                "gap_pct": _cost_gap_pct(baseline["objective_value"], optimal_cost)
                if baseline["success"] else None,
                "matched_unit_patterns": baseline.get("matched_unit_patterns"),
                "changed_unit_patterns": baseline.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(baseline.get("selected_pattern_indices")),
            },
            baseline_dual,
        ),
        "greedy_history": greedy_history,
        "greedy_best": _merge_dual(
            {
                "success": greedy_best["success"],
                "status": greedy_best["status"],
                "objective_value": greedy_best["objective_value"],
                "gap_pct": _cost_gap_pct(greedy_best["objective_value"], optimal_cost)
                if greedy_best["success"] else None,
                "matched_unit_patterns": greedy_best.get("matched_unit_patterns"),
                "changed_unit_patterns": greedy_best.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(greedy_best.get("selected_pattern_indices")),
            },
            greedy_best_dual,
        ),
        "full_repair": _merge_dual(
            {
                "success": full_repair["success"],
                "status": full_repair["status"],
                "objective_value": full_repair["objective_value"],
                "gap_pct": _cost_gap_pct(full_repair["objective_value"], optimal_cost)
                if full_repair["success"] else None,
                "matched_unit_patterns": full_repair.get("matched_unit_patterns"),
                "changed_unit_patterns": full_repair.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(full_repair.get("selected_pattern_indices")),
                "added_generators": full_repair_added_generators,
            },
            full_repair_dual,
        ),
    }


def run_refinement_batch(
    ppc: dict,
    active_set_json: str,
    pattern_library_json: str,
    active_set_like_json: str,
    sample_ids: Optional[List[int]] = None,
    *,
    high_gap_top_k: int = 20,
    high_gap_threshold_pct: float = 0.7,
    t_delta: float = 1.0,
    base_time_limit: float = 600.0,
    greedy_trial_time_limit: float = 180.0,
    final_time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    max_greedy_steps: int = 8,
    verbose_solver: bool = False,
    strict_dual: bool = False,
    require_full_repair_success: bool = False,
) -> Dict:
    """Run batch refinement and return the result dict (caller writes to disk).

    Parameters
    ----------
    strict_dual:
        When True, dual recomputation failures raise immediately instead of
        being recorded as ``lambda_dual_refresh_failed``.
    require_full_repair_success:
        When True, raise RuntimeError for any sample whose full_repair did not
        succeed.
    """
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "[run_refinement_batch] Missing required dependency."
        ) from IMPORT_ERROR

    pattern_library_data = _load_json(pattern_library_json)
    global_top_gap = _top_gap_samples(pattern_library_data, high_gap_top_k)
    global_large_gap = [
        row for row in global_top_gap if row["gap_pct"] >= high_gap_threshold_pct
    ]

    if sample_ids is None:
        # Default: samples with cost gap above threshold, or top-k if none exceed threshold
        sample_ids = [row["sample_id"] for row in global_large_gap]
        if not sample_ids:
            sample_ids = [row["sample_id"] for row in global_top_gap]

    print("=" * 72, flush=True)
    print("  Batch Pattern-Library Refinement", flush=True)
    print("=" * 72, flush=True)
    print(f"sample_ids={sample_ids}", flush=True)
    print(f"pattern_library_json={pattern_library_json}", flush=True)
    print(f"active_set_like_json={active_set_like_json}", flush=True)

    samples = _load_commitment_samples_from_json(active_set_json)
    active_set_like_data = _load_json(active_set_like_json)
    base_allowed_patterns = _build_allowed_patterns_from_result(pattern_library_data)

    batch_results: List[Dict] = []
    for sid in sample_ids:
        print("\n" + "-" * 72, flush=True)
        print(f"Processing sample_id={sid}", flush=True)
        sample = _find_sample(samples, sid)
        sample_like = _find_active_set_like_sample(active_set_like_data, sid)

        if strict_dual:
            result = _refine_one_sample_strict_dual(
                ppc=ppc,
                sample=sample,
                sample_like=sample_like,
                base_allowed_patterns=base_allowed_patterns,
                t_delta=t_delta,
                base_time_limit=base_time_limit,
                greedy_trial_time_limit=greedy_trial_time_limit,
                final_time_limit=final_time_limit,
                mip_gap=mip_gap,
                verbose_solver=verbose_solver,
                max_greedy_steps=max_greedy_steps,
            )
        else:
            result = _refine_one_sample(
                ppc=ppc,
                sample=sample,
                sample_like=sample_like,
                base_allowed_patterns=base_allowed_patterns,
                t_delta=t_delta,
                base_time_limit=base_time_limit,
                greedy_trial_time_limit=greedy_trial_time_limit,
                final_time_limit=final_time_limit,
                mip_gap=mip_gap,
                verbose_solver=verbose_solver,
                max_greedy_steps=max_greedy_steps,
            )

        if require_full_repair_success and not result.get("full_repair", {}).get("success"):
            raise RuntimeError(
                f"[run_refinement_batch] full_repair failed for sample_id={sid} "
                f"(status={result.get('full_repair', {}).get('status')!r}). "
                "Use require_full_repair_success=False to skip instead."
            )

        batch_results.append(result)

    return {
        "metadata": {
            "case_name": "case118",
            "timestamp": datetime.now().isoformat(),
            "source_active_set_json": active_set_json,
            "source_pattern_library_json": pattern_library_json,
            "source_active_set_like_json": active_set_like_json,
        },
        "parameters": {
            "sample_ids": sample_ids,
            "high_gap_top_k": high_gap_top_k,
            "high_gap_threshold_pct": high_gap_threshold_pct,
            "t_delta": t_delta,
            "base_time_limit": base_time_limit,
            "greedy_trial_time_limit": greedy_trial_time_limit,
            "final_time_limit": final_time_limit,
            "mip_gap": mip_gap,
            "max_greedy_steps": max_greedy_steps,
            "strict_dual": strict_dual,
            "require_full_repair_success": require_full_repair_success,
        },
        "high_gap_samples": {
            "top_k": global_top_gap,
            "above_threshold": global_large_gap,
        },
        "batch_results": batch_results,
    }


def _refine_one_sample_strict_dual(
    ppc: dict,
    sample: Dict,
    sample_like: Optional[Dict],
    base_allowed_patterns: List[List[np.ndarray]],
    t_delta: float,
    base_time_limit: float,
    greedy_trial_time_limit: float,
    final_time_limit: float,
    mip_gap: float,
    verbose_solver: bool,
    max_greedy_steps: int,
) -> Dict:
    """Same as _refine_one_sample but dual failures raise immediately."""
    sample_id = int(sample["sample_id"])
    x_opt = np.asarray(sample["unit_commitment_matrix"], dtype=int)
    optimal_cost = _compute_optimal_cost(ppc, sample, t_delta)

    allowed_patterns = _clone_patterns(base_allowed_patterns)
    baseline = _solve_once(
        ppc=ppc, sample=sample, allowed_patterns=allowed_patterns,
        t_delta=t_delta, time_limit=base_time_limit, mip_gap=mip_gap,
        verbose=verbose_solver,
    )
    _summarize_solution(f"sample {sample_id} baseline", baseline, optimal_cost)

    baseline_matches_active_set_like = None
    if sample_like is not None and baseline["x_sol"] is not None:
        x_like = np.asarray(sample_like["unit_commitment_matrix"], dtype=int)
        baseline_matches_active_set_like = bool(np.array_equal(x_like, baseline["x_sol"]))
        print(f"  baseline_matches_active_set_like={baseline_matches_active_set_like}")

    mismatched_generators = [
        g for g in range(x_opt.shape[0])
        if baseline["x_sol"] is None
        or not np.array_equal(np.asarray(baseline["x_sol"][g, :], dtype=int), x_opt[g, :])
    ]
    print(f"  mismatched_generators={mismatched_generators}")

    greedy_patterns = _clone_patterns(allowed_patterns)
    greedy_history: List[Dict] = []
    remaining_generators = mismatched_generators[:]
    current_best = copy.deepcopy(baseline)

    for step in range(1, max_greedy_steps + 1):
        best_step_info = None
        best_generator = None
        for generator_id in remaining_generators:
            candidate_patterns = _clone_patterns(greedy_patterns)
            added = _ensure_pattern(
                candidate_patterns, generator_id=generator_id,
                pattern_key=_pattern_to_key(x_opt[generator_id, :]),
            )
            if not added:
                continue
            candidate = _solve_once(
                ppc=ppc, sample=sample, allowed_patterns=candidate_patterns,
                t_delta=t_delta, time_limit=greedy_trial_time_limit,
                mip_gap=mip_gap, verbose=verbose_solver,
            )
            if not candidate["success"]:
                continue
            if (
                not current_best["success"]
                or candidate["objective_value"] + 1e-6 < current_best["objective_value"]
            ):
                if best_step_info is None or (
                    candidate["objective_value"] + 1e-6 < best_step_info["objective_value"]
                ):
                    best_step_info = candidate
                    best_generator = generator_id

        if best_step_info is None or best_generator is None:
            print(f"  greedy step {step}: no improving single-generator augmentation found")
            break

        _ensure_pattern(
            greedy_patterns, generator_id=best_generator,
            pattern_key=_pattern_to_key(x_opt[best_generator, :]),
        )
        current_best = best_step_info
        remaining_generators = [g for g in remaining_generators if g != best_generator]
        greedy_entry = {
            "step": step,
            "chosen_generator": int(best_generator),
            "objective_value": float(current_best["objective_value"]),
            "gap_pct": _cost_gap_pct(current_best["objective_value"], optimal_cost),
            "matched_unit_patterns": current_best.get("matched_unit_patterns"),
            "changed_unit_patterns": current_best.get("changed_unit_patterns"),
            "status": current_best.get("status"),
        }
        greedy_history.append(greedy_entry)
        print(
            f"  greedy step {step}: add generator {best_generator} optimal pattern -> "
            f"objective={current_best['objective_value']}, gap_pct={greedy_entry['gap_pct']}"
        )

    greedy_best = current_best
    _summarize_solution(f"sample {sample_id} greedy_best", greedy_best, optimal_cost)

    full_repair_patterns = _clone_patterns(allowed_patterns)
    full_repair_added_generators: List[int] = []
    for generator_id in mismatched_generators:
        added = _ensure_pattern(
            full_repair_patterns, generator_id=generator_id,
            pattern_key=_pattern_to_key(x_opt[generator_id, :]),
        )
        if added:
            full_repair_added_generators.append(generator_id)

    full_repair = _solve_once(
        ppc=ppc, sample=sample, allowed_patterns=full_repair_patterns,
        t_delta=t_delta, time_limit=final_time_limit, mip_gap=mip_gap,
        verbose=verbose_solver,
    )
    _summarize_solution(f"sample {sample_id} full_repair", full_repair, optimal_cost)

    # Strict dual: raise immediately on failure
    optimal_dual = _dual_payload_for_x_matrix(
        ppc, sample, t_delta, x_opt,
        "recomputed_from_ed_optimal_commitment",
    )
    baseline_dual = (
        _dual_payload_for_x_matrix(
            ppc, sample, t_delta, baseline["x_sol"],
            "recomputed_from_ed_after_pattern_restricted_baseline",
        ) if baseline["x_sol"] is not None else None
    )
    greedy_best_dual = (
        _dual_payload_for_x_matrix(
            ppc, sample, t_delta, greedy_best["x_sol"],
            "recomputed_from_ed_after_pattern_restricted_greedy_best",
        ) if greedy_best["x_sol"] is not None else None
    )
    full_repair_dual = (
        _dual_payload_for_x_matrix(
            ppc, sample, t_delta, full_repair["x_sol"],
            "recomputed_from_ed_after_pattern_restricted_full_repair",
        ) if full_repair["x_sol"] is not None else None
    )

    def _merge_dual(block: Dict, dual: Optional[Dict]) -> Dict:
        if dual:
            return {**block, **dual}
        return block

    return {
        "sample_id": sample_id,
        "optimal_cost": optimal_cost,
        "optimal_dual": optimal_dual,
        "baseline_matches_active_set_like": baseline_matches_active_set_like,
        "mismatched_generators": mismatched_generators,
        "baseline": _merge_dual(
            {
                "success": baseline["success"],
                "status": baseline["status"],
                "objective_value": baseline["objective_value"],
                "gap_pct": _cost_gap_pct(baseline["objective_value"], optimal_cost)
                if baseline["success"] else None,
                "matched_unit_patterns": baseline.get("matched_unit_patterns"),
                "changed_unit_patterns": baseline.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(baseline.get("selected_pattern_indices")),
            },
            baseline_dual,
        ),
        "greedy_history": greedy_history,
        "greedy_best": _merge_dual(
            {
                "success": greedy_best["success"],
                "status": greedy_best["status"],
                "objective_value": greedy_best["objective_value"],
                "gap_pct": _cost_gap_pct(greedy_best["objective_value"], optimal_cost)
                if greedy_best["success"] else None,
                "matched_unit_patterns": greedy_best.get("matched_unit_patterns"),
                "changed_unit_patterns": greedy_best.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(greedy_best.get("selected_pattern_indices")),
            },
            greedy_best_dual,
        ),
        "full_repair": _merge_dual(
            {
                "success": full_repair["success"],
                "status": full_repair["status"],
                "objective_value": full_repair["objective_value"],
                "gap_pct": _cost_gap_pct(full_repair["objective_value"], optimal_cost)
                if full_repair["success"] else None,
                "matched_unit_patterns": full_repair.get("matched_unit_patterns"),
                "changed_unit_patterns": full_repair.get("changed_unit_patterns"),
                "selected_pattern_indices": _safe_json_value(full_repair.get("selected_pattern_indices")),
                "added_generators": full_repair_added_generators,
            },
            full_repair_dual,
        ),
    }


def main() -> None:
    ACTIVE_SET_JSON = (
        r"result/active_set/active_sets_case118_T0_n366_20260322_063917.json"
    )
    PATTERN_LIBRARY_JSON = (
        r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json"
    )
    ACTIVE_SET_LIKE_JSON = (
        r"result/commitment_clustering/"
        r"pattern_library_case118_K10_20260408_132932_active_set_like.json"
    )
    OUTPUT_JSON = (
        r"result/commitment_clustering/"
        r"sample_refinement_case118_batch.json"
    )

    SAMPLE_IDS = [143, 132, 257, 313, 305]
    HIGH_GAP_TOP_K = 20
    HIGH_GAP_THRESHOLD_PCT = 0.7
    AGGREGATE_THERMAL_BY_BUS = True
    T_DELTA = 1.0
    BASE_TIME_LIMIT = 600.0
    GREEDY_TRIAL_TIME_LIMIT = 180.0
    FINAL_TIME_LIMIT = 600.0
    MIP_GAP = 1e-4
    VERBOSE_SOLVER = False
    MAX_GREEDY_STEPS = 8

    if IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Missing required dependency: {IMPORT_ERROR}. "
            "This script requires a Python environment with gurobipy and commitment_clustering available."
        ) from IMPORT_ERROR

    ppc = load_case118_ppc_with_mti_limits(
        aggregate_thermal_by_bus=AGGREGATE_THERMAL_BY_BUS,
    )

    output = run_refinement_batch(
        ppc=ppc,
        active_set_json=ACTIVE_SET_JSON,
        pattern_library_json=PATTERN_LIBRARY_JSON,
        active_set_like_json=ACTIVE_SET_LIKE_JSON,
        sample_ids=SAMPLE_IDS,
        high_gap_top_k=HIGH_GAP_TOP_K,
        high_gap_threshold_pct=HIGH_GAP_THRESHOLD_PCT,
        t_delta=T_DELTA,
        base_time_limit=BASE_TIME_LIMIT,
        greedy_trial_time_limit=GREEDY_TRIAL_TIME_LIMIT,
        final_time_limit=FINAL_TIME_LIMIT,
        mip_gap=MIP_GAP,
        max_greedy_steps=MAX_GREEDY_STEPS,
        verbose_solver=VERBOSE_SOLVER,
        strict_dual=False,
        require_full_repair_success=False,
    )

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nSaved batch refinement report to: {output_path}")


if __name__ == "__main__":
    main()
