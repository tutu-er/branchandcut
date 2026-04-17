"""Core pipeline stages for the case118 pattern-library workflow.

This module provides one function per pipeline stage:

    run_phase_active_set(...)       -> active_set JSON path
    run_phase_pattern_library(...)  -> pattern_library JSON path
    run_phase_convert(...)          -> active_set_like JSON path
    run_phase_refine(...)           -> refinement batch dict
    run_phase_build_refined(...)    -> refined active_set_like JSON path

Each stage raises on any failure; nothing is silently skipped.
Call the stages in order from run_case118_pattern_pipeline.py.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared dual-refresh helper (single authoritative implementation)
# ---------------------------------------------------------------------------

def refresh_sample_dual_from_ed(
    sample: Dict,
    ppc: dict,
    t_delta: float,
    refresh_source: str = "recomputed_from_ed_pipeline_final",
) -> None:
    """Fix sample['unit_commitment_matrix'], solve ED, write lambda fields in-place.

    Raises RuntimeError if ED is infeasible.
    """
    from src.uc_NN_subproblem import (
        _build_generator_injection_sensitivity,
        _solve_global_dual_payload_from_ed,
    )

    x_arr = np.asarray(sample["unit_commitment_matrix"], dtype=float)
    pd_data = np.asarray(sample.get("pd_data", sample.get("load_data")), dtype=float)
    renewable_data = sample.get("renewable_data")
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    generator_injection_sensitivity = _build_generator_injection_sensitivity(ppc)
    payload = _solve_global_dual_payload_from_ed(
        ppc,
        pd_data,
        float(t_delta),
        x_arr,
        generator_injection_sensitivity,
        renewable_data=renewable_arr,
    )
    serialized = {key: np.asarray(v, dtype=float).tolist() for key, v in payload.items()}
    sample["lambda"] = serialized
    sample["lambda_pg_electricity_price"] = serialized["lambda_pg_effective"]
    sample["lambda_refresh_source"] = refresh_source


def refresh_dual_all_samples(
    all_samples: List[Dict],
    ppc: dict,
    t_delta: float,
    refresh_source: str = "recomputed_from_ed_pipeline_final",
) -> None:
    """Recompute ED dual for every sample. Raises immediately if any ED fails."""
    for sample in all_samples:
        sample_id = sample.get("sample_id", "?")
        try:
            refresh_sample_dual_from_ed(sample, ppc, t_delta, refresh_source)
        except Exception as exc:
            raise RuntimeError(
                f"[refresh_dual_all_samples] ED dual failed for sample_id={sample_id}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def assert_pattern_library_all_success(scenario_results: List[Dict]) -> None:
    """Raise if any scenario failed in the pattern-library evaluation pass."""
    failed = [r for r in scenario_results if not r.get("success")]
    if failed:
        ids = [r.get("sample_id") for r in failed]
        statuses = [r.get("solver_status") for r in failed]
        raise RuntimeError(
            f"[assert_pattern_library_all_success] {len(failed)} scenario(s) failed: "
            f"sample_ids={ids}, statuses={statuses}"
        )


def validate_final_commitments_ed_feasible(
    ppc: dict,
    all_samples: List[Dict],
    t_delta: float,
) -> None:
    """Fix every sample's unit_commitment_matrix and verify ED is feasible.

    Raises RuntimeError listing all infeasible sample_ids.
    """
    from src.commitment_clustering import evaluate_commitment_cost

    infeasible = []
    for sample in all_samples:
        sample_id = sample.get("sample_id", "?")
        x = sample.get("unit_commitment_matrix")
        if x is None:
            infeasible.append((sample_id, "missing unit_commitment_matrix"))
            continue
        load = np.asarray(sample.get("load_data", sample.get("pd_data")), dtype=float)
        renewable = sample.get("renewable_data")
        result = evaluate_commitment_cost(
            ppc, load, np.asarray(x, dtype=float), t_delta,
            renewable_data=None if renewable is None else np.asarray(renewable, dtype=float),
        )
        if not result.get("success"):
            infeasible.append((sample_id, result.get("reason", "ED_infeasible")))

    if infeasible:
        msg = "; ".join(f"sample_id={sid}: {reason}" for sid, reason in infeasible)
        raise RuntimeError(
            f"[validate_final_commitments_ed_feasible] {len(infeasible)} sample(s) "
            f"have infeasible final commitment: {msg}"
        )


def _assert_converted_samples_have_lambda(converted_samples: List[Dict]) -> None:
    _need_lambda_statuses = frozenset({
        "converted_from_pattern_library",
        "converted_pattern_then_rescue_uc",
        "converted_pattern_then_fallback_original_ed",
    })
    bad = [
        s.get("sample_id")
        for s in converted_samples
        if s.get("conversion_status") in _need_lambda_statuses and not s.get("lambda")
    ]
    if bad:
        raise RuntimeError(
            f"[run_phase_convert] {len(bad)} converted sample(s) are missing lambda: {bad}"
        )


# ---------------------------------------------------------------------------
# Phase 1 – active set (serial, strict)
# ---------------------------------------------------------------------------

def run_phase_active_set(
    ppc: dict,
    scenarios: List[Dict],
    *,
    alpha: float = 0.75,
    delta: float = 0.15,
    epsilon: float = 0.15,
    t_delta: float = 1.0,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose_solver: bool = False,
) -> str:
    """Solve each scenario with the serial ActiveSetLearner and save JSON.

    Raises RuntimeError if any scenario solve fails (UC or ED).
    Returns the path to the saved active_set JSON.
    """
    from src.ActiveSetLearner import ActiveSetLearner

    learner = ActiveSetLearner(
        alpha=alpha,
        delta=delta,
        epsilon=epsilon,
        ppc=ppc,
        T_delta=t_delta,
        Pd=None,
        case_name="case118",
        verbose_solver=verbose_solver,
    )
    # run_on_precomputed_scenarios in the base class raises on individual solve
    # failures (via _solve_optimization raising RuntimeError).
    learner.run_on_precomputed_scenarios(scenarios, max_samples=max_samples)

    if len(learner.samples) == 0:
        raise RuntimeError("[run_phase_active_set] No samples were solved successfully.")

    n_expected = len(scenarios) if max_samples is None else min(max_samples, len(scenarios))
    if len(learner.samples) < n_expected:
        raise RuntimeError(
            f"[run_phase_active_set] Expected {n_expected} solved scenarios but got "
            f"{len(learner.samples)}. Some scenarios failed silently."
        )

    path = learner.save_active_sets_json(filename=output_path)
    print(f"[Phase1] Active set JSON: {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Phase 2 – pattern library
# ---------------------------------------------------------------------------

def run_phase_pattern_library(
    ppc: dict,
    active_set_path: str,
    *,
    t_delta: float = 1.0,
    initial_patterns_per_unit: int = 10,
    max_patterns_per_unit: Optional[int] = None,
    gurobi_time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    max_samples: Optional[int] = None,
    verbose_solver: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Build per-generator pattern library and verify all scenarios succeed.

    Raises RuntimeError if any scenario is infeasible after pattern expansion.
    Returns the path to the saved pattern_library JSON.
    """
    from src.commitment_clustering import CommitmentPatternLibrary

    library = CommitmentPatternLibrary(
        ppc=ppc,
        T_delta=t_delta,
        case_name="case118",
        initial_patterns_per_unit=initial_patterns_per_unit,
        max_patterns_per_unit=max_patterns_per_unit,
        gurobi_time_limit=gurobi_time_limit,
        mip_gap=mip_gap,
        max_samples=max_samples,
        verbose=verbose_solver,
    )
    library.run(active_set_path)
    assert_pattern_library_all_success(library.scenario_results)

    path = library.save_results(output_path)
    print(f"[Phase2] Pattern library JSON: {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Phase 3 – convert to active_set_like
# ---------------------------------------------------------------------------

def run_phase_convert(
    active_set_path: str,
    pattern_library_path: str,
    output_path: str,
    *,
    ed_infeasibility_heal: bool = True,
    heal_mip_time_limit: float = 180.0,
) -> str:
    """Convert pattern-library result to active_set_like JSON.

    Uses fallback_to_original=False so any missing/failed pattern result raises.
    When ``ed_infeasibility_heal`` is True, ED infeasibility on the pattern x
    triggers an augmented pattern-restricted UC re-solve, then optional
    fallback to the original optimal commitment (see convert script).
    Verifies that all converted samples have a non-empty lambda.
    Returns output_path.
    """
    import scripts.convert_pattern_library_to_active_set_like as _conv

    result_path = _conv.convert_pattern_library_to_active_set_like(
        active_set_json_path=active_set_path,
        pattern_library_json_path=pattern_library_path,
        output_path=output_path,
        fallback_to_original=False,
        ed_infeasibility_heal=ed_infeasibility_heal,
        heal_mip_time_limit=heal_mip_time_limit,
    )

    # Verify lambda presence on converted samples
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    _assert_converted_samples_have_lambda(data.get("all_samples", []))

    print(f"[Phase3] active_set_like JSON: {result_path}", flush=True)
    return result_path


# ---------------------------------------------------------------------------
# Phase 4 – refine (strict)
# ---------------------------------------------------------------------------

def run_phase_refine(
    ppc: dict,
    active_set_path: str,
    pattern_library_path: str,
    active_set_like_path: str,
    output_path: str,
    *,
    sample_ids: Optional[List[int]] = None,
    high_gap_top_k: int = 20,
    high_gap_threshold_pct: float = 0.7,
    t_delta: float = 1.0,
    base_time_limit: float = 600.0,
    greedy_trial_time_limit: float = 180.0,
    final_time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    max_greedy_steps: int = 8,
    verbose_solver: bool = False,
) -> str:
    """Run batch refinement and write refinement JSON.

    Raises RuntimeError if any import dependency is missing, if any
    targeted sample's full_repair fails, or if any dual recomputation fails.
    Returns the refinement JSON path.
    """
    import scripts.refine_pattern_library_sample as _refine

    if _refine.IMPORT_ERROR is not None:
        raise RuntimeError(
            "[run_phase_refine] Missing required dependency; cannot run refinement."
        ) from _refine.IMPORT_ERROR

    result_dict = _refine.run_refinement_batch(
        ppc=ppc,
        active_set_json=active_set_path,
        pattern_library_json=pattern_library_path,
        active_set_like_json=active_set_like_path,
        sample_ids=sample_ids,
        high_gap_top_k=high_gap_top_k,
        high_gap_threshold_pct=high_gap_threshold_pct,
        t_delta=t_delta,
        base_time_limit=base_time_limit,
        greedy_trial_time_limit=greedy_trial_time_limit,
        final_time_limit=final_time_limit,
        mip_gap=mip_gap,
        max_greedy_steps=max_greedy_steps,
        verbose_solver=verbose_solver,
        strict_dual=True,
        require_full_repair_success=True,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[Phase4] Refinement JSON: {output_path}", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Phase 5 – build refined active_set_like
# ---------------------------------------------------------------------------

def run_phase_build_refined(
    ppc: dict,
    active_set_like_path: str,
    pattern_library_path: str,
    refinement_path: str,
    output_path: str,
    t_delta: float = 1.0,
) -> str:
    """Build refined active_set_like JSON, then refresh dual for ALL samples.

    strict mode: require_full_repair_success_for_rows=True.
    After build, every sample's lambda is refreshed unconditionally so
    lambda is always consistent with the final unit_commitment_matrix.
    Raises on any failure.
    Returns output_path.
    """
    import scripts.build_refined_case118_active_set as _build

    result_path = _build.build_refined_active_set(
        active_set_like_json=active_set_like_path,
        pattern_library_json=pattern_library_path,
        refinement_json=refinement_path,
        output_json=output_path,
        require_full_repair_success_for_rows=True,
    )

    # Unconditionally refresh dual for all samples in the output file.
    print("[Phase5] Refreshing dual payload for all samples...", flush=True)
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    all_samples = data["all_samples"]
    refresh_dual_all_samples(
        all_samples, ppc, t_delta,
        refresh_source="recomputed_from_ed_pipeline_final",
    )

    # Persist refreshed data
    data["all_samples"] = all_samples
    data.setdefault("metadata", {})["dual_final_refresh_timestamp"] = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    Path(result_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Phase5] Refined active_set_like JSON: {result_path}", flush=True)
    return result_path
