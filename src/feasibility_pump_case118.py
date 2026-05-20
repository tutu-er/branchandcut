from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pypower.ext2int import ext2int
from pypower.idx_gen import PMAX

try:
    from feasibility_pump import (
        _build_hot_start_candidates,
        _candidate_key,
        _coerce_scenario_sample,
        _compute_hot_start_support_reference,
        _compute_vote_majority,
        _estimate_commitment_primal_objective,
        _extract_commitment_from_sample,
        _feature_distance,
        _filter_named_commitment_candidates_by_surrogate_screen,
        _get_scenario_bank,
        _rank_hot_start_candidates,
        _repair_commitment_logic_heuristic,
        _sanitize_named_commitment_candidates,
        _score_hot_start_candidate,
        _select_stable_surrogate_screen_constraints,
        check_uc_feasibility,
        collect_integer_solutions,
        identify_trusted_mask,
        round_to_integer,
        run_feasibility_pump,
        solve_global_LP_relaxation,
    )
    from scenario_utils import get_feature_vector_from_sample, get_sample_net_load, normalize_sample_arrays
except ImportError:
    from src.feasibility_pump import (
        _build_hot_start_candidates,
        _candidate_key,
        _coerce_scenario_sample,
        _compute_hot_start_support_reference,
        _compute_vote_majority,
        _estimate_commitment_primal_objective,
        _extract_commitment_from_sample,
        _feature_distance,
        _filter_named_commitment_candidates_by_surrogate_screen,
        _get_scenario_bank,
        _rank_hot_start_candidates,
        _repair_commitment_logic_heuristic,
        _sanitize_named_commitment_candidates,
        _score_hot_start_candidate,
        _select_stable_surrogate_screen_constraints,
        check_uc_feasibility,
        collect_integer_solutions,
        identify_trusted_mask,
        round_to_integer,
        run_feasibility_pump,
        solve_global_LP_relaxation,
    )
    from src.scenario_utils import get_feature_vector_from_sample, get_sample_net_load, normalize_sample_arrays


def _load_default_case118_ppc() -> dict:
    try:
        from mti118_data_loader import load_case118_ppc_with_mti_limits
    except ImportError:
        from src.mti118_data_loader import load_case118_ppc_with_mti_limits

    return load_case118_ppc_with_mti_limits(aggregate_thermal_by_bus=True)


def _as_candidate_pool(candidates: List[Tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
    if not candidates:
        return None
    return np.stack([np.asarray(candidate, dtype=int) for _name, candidate in candidates], axis=0)


def _mean_abs_distance(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_arr = np.asarray(candidate, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    valid = np.isfinite(reference_arr)
    if candidate_arr.shape != reference_arr.shape or not np.any(valid):
        return float("inf")
    return float(np.mean(np.abs(candidate_arr[valid] - reference_arr[valid])))


def _subproblem_distance(candidate: np.ndarray, x_init_k: np.ndarray, x_init_k_m: np.ndarray) -> float:
    distances = [_mean_abs_distance(candidate, x_init_k)]
    if x_init_k_m.ndim == 3 and x_init_k_m.shape[1] > 0:
        distances.extend(_mean_abs_distance(candidate, x_init_k_m[:, m, :]) for m in range(x_init_k_m.shape[1]))
        distances.append(_mean_abs_distance(candidate, _compute_vote_majority(x_init_k, x_init_k_m)))
    return float(np.min(distances))


def _candidate_metrics(
    candidate: np.ndarray,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    scenario_distance: Optional[float] = None,
) -> dict:
    return {
        "lp_distance": _mean_abs_distance(candidate, x_lp),
        "surrogate_lp_distance": _mean_abs_distance(candidate, x_surr_lp),
        "subproblem_distance": _subproblem_distance(candidate, x_init_k, x_init_k_m),
        "scenario_distance": None if scenario_distance is None else float(scenario_distance),
    }


def _rank_case118_candidates(
    candidate_specs: List[Tuple[str, np.ndarray]],
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    scenario_distances: Optional[Dict[str, float]] = None,
    nearby_commitment_pool: Optional[np.ndarray] = None,
    max_candidates: int = 20,
) -> Tuple[List[Tuple[str, np.ndarray, float]], List[dict]]:
    """Rank candidates with explicit LP, surrogate LP, and subproblem distances."""
    vote_reference = _compute_vote_majority(x_init_k, x_init_k_m)
    support_reference = _compute_hot_start_support_reference(
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        nearby_commitment_pool=nearby_commitment_pool,
    )
    scenario_distances = scenario_distances or {}

    ranked: List[Tuple[str, np.ndarray, float]] = []
    records: List[dict] = []
    seen = set()
    for name, candidate in candidate_specs:
        candidate_int = np.asarray(candidate, dtype=int)
        key = _candidate_key(candidate_int)
        if key in seen:
            continue
        seen.add(key)

        metrics = _candidate_metrics(
            candidate_int,
            x_lp,
            x_surr_lp,
            x_init_k,
            x_init_k_m,
            scenario_distance=scenario_distances.get(str(name)),
        )
        base_score = _score_hot_start_candidate(
            candidate_int,
            x_lp,
            x_surr_lp,
            vote_reference,
            trusted_mask,
            support_reference=support_reference,
            nearby_commitment_pool=nearby_commitment_pool,
        )
        n_bits = max(int(candidate_int.size), 1)
        score = (
            float(base_score)
            - 0.70 * metrics["subproblem_distance"] * n_bits
            - 0.20 * metrics["lp_distance"] * n_bits
            - 0.15 * metrics["surrogate_lp_distance"] * n_bits
        )
        if metrics["scenario_distance"] is not None:
            score -= 4.0 * float(metrics["scenario_distance"])

        ranked.append((str(name), candidate_int, float(score)))
        records.append({"name": str(name), "score": float(score), **metrics})

    ranked.sort(key=lambda item: item[2], reverse=True)
    records.sort(key=lambda item: item["score"], reverse=True)
    max_candidates = max(1, int(max_candidates))
    return ranked[:max_candidates], records[:max_candidates]


def build_case118_historical_candidates(
    target_sample: dict,
    trainers: Dict[int, object],
    scenario_bank: Optional[List[dict]],
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    ppc: dict,
    T_delta: float,
    max_candidates: int = 10,
    candidate_pool_size: int = 40,
) -> Tuple[List[Tuple[str, np.ndarray]], List[dict]]:
    """Retrieve similar historical case118 scenarios and keep distance-consistent commitments."""
    if scenario_bank is None:
        scenario_bank = _get_scenario_bank(trainers)
    if not scenario_bank:
        return [], []

    target = normalize_sample_arrays(dict(target_sample))
    target_features = get_feature_vector_from_sample(target)
    ng, T = x_lp.shape
    scored: List[Tuple[float, dict]] = []
    for sample in scenario_bank:
        if not isinstance(sample, dict):
            continue
        try:
            sample_norm = normalize_sample_arrays(dict(sample))
            feature_vec = get_feature_vector_from_sample(sample_norm)
        except Exception:
            continue
        if feature_vec.shape != target_features.shape:
            continue
        distance = _feature_distance(target_features, feature_vec)
        if distance <= 1e-12:
            continue
        scored.append((float(distance), sample_norm))

    if not scored:
        return [], []

    scored.sort(key=lambda item: item[0])
    raw_specs: List[Tuple[str, np.ndarray]] = []
    scenario_distances: Dict[str, float] = {}
    for distance, sample in scored[: max(int(candidate_pool_size), int(max_candidates))]:
        commitment = _extract_commitment_from_sample(sample, ng, T)
        if commitment is None:
            continue
        name = f"case118_history_{len(raw_specs) + 1}"
        raw_specs.append((name, commitment))
        scenario_distances[name] = distance

    sanitized, _rejected = _sanitize_named_commitment_candidates(raw_specs, ppc, T_delta)
    ranked, records = _rank_case118_candidates(
        sanitized,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        trusted_mask=np.zeros_like(x_lp, dtype=bool),
        scenario_distances=scenario_distances,
        nearby_commitment_pool=None,
        max_candidates=max_candidates,
    )
    return [(name, candidate) for name, candidate, _score in ranked], records


def _smooth_commitment(candidate: np.ndarray) -> np.ndarray:
    x = np.asarray(candidate, dtype=int).copy()
    if x.ndim != 2 or x.shape[1] < 3:
        return x
    for g in range(x.shape[0]):
        for t in range(1, x.shape[1] - 1):
            if x[g, t - 1] == x[g, t + 1] and x[g, t] != x[g, t - 1]:
                x[g, t] = x[g, t - 1]
    return x


def _repair_capacity_shortfall(candidate: np.ndarray, ppc: dict, pd_data: np.ndarray, reserve_margin: float) -> np.ndarray:
    ppc_int = ext2int(ppc)
    gen = np.asarray(ppc_int["gen"], dtype=float)
    gencost = np.asarray(ppc_int.get("gencost", np.zeros((gen.shape[0], 7))), dtype=float)
    pmax = np.asarray(gen[:, PMAX], dtype=float)
    ng, T = np.asarray(candidate).shape
    x = np.asarray(candidate, dtype=int).copy()
    load = np.sum(np.asarray(pd_data, dtype=float), axis=0)

    if gencost.shape[0] == ng and gencost.shape[1] >= 2:
        cost_proxy = gencost[:, -2] / np.maximum(pmax, 1.0)
    else:
        cost_proxy = 1.0 / np.maximum(pmax, 1.0)
    merit_order = np.argsort(cost_proxy, kind="stable")

    for t in range(T):
        required = float(load[t]) * (1.0 + float(reserve_margin))
        online_capacity = float(np.sum(pmax * x[:, t]))
        if online_capacity >= required:
            continue
        for g in merit_order:
            if x[g, t] == 1:
                continue
            x[g, t] = 1
            online_capacity += float(pmax[g])
            if online_capacity >= required:
                break
    return x


def build_case118_heuristic_candidates(
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    reserve_margin: float = 0.03,
) -> List[Tuple[str, np.ndarray]]:
    """Blend LP, subproblem and capacity heuristics into extra case118 starts."""
    lp_round = round_to_integer(x_lp)
    surr_round = round_to_integer(x_surr_lp)
    vote = _compute_vote_majority(x_init_k, x_init_k_m)
    support = _compute_hot_start_support_reference(x_lp, x_surr_lp, x_init_k, x_init_k_m)
    support_round = round_to_integer(support)

    raw_specs = [
        ("case118_vote_majority", vote),
        ("case118_support_round", support_round),
        ("case118_lp_surr_or", np.maximum(lp_round, surr_round)),
        ("case118_subproblem_or", np.maximum(vote, x_init_k)),
        ("case118_smoothed_vote", _smooth_commitment(vote)),
        ("case118_capacity_support", _repair_capacity_shortfall(support_round, ppc, pd_data, reserve_margin)),
        ("case118_capacity_vote", _repair_capacity_shortfall(vote, ppc, pd_data, reserve_margin)),
    ]

    repaired_specs: List[Tuple[str, np.ndarray]] = []
    for name, candidate in raw_specs:
        repaired = _repair_commitment_logic_heuristic(candidate, T_delta, ppc=ppc)
        repaired_specs.append((name, repaired))

    sanitized, _rejected = _sanitize_named_commitment_candidates(repaired_specs, ppc, T_delta)
    return sanitized


def recover_integer_solution_case118(
    pd_data: np.ndarray | dict,
    trainers: Dict[int, object],
    lambda_predictor,
    ppc: Optional[dict] = None,
    T_delta: float = 1.0,
    agent=None,
    manager=None,
    n_perturbations: int = 8,
    n_similar_scenarios: int = 4,
    similar_scenario_pool_size: int = 24,
    n_load_perturbations: int = 2,
    load_perturbation_scale: float = 0.025,
    conf_threshold: float = 0.15,
    max_fp_iter: int = 50,
    perturb_std: float = 0.10,
    neighborhood_weight: float = 0.35,
    max_history_hot_starts: int = 10,
    history_candidate_pool_size: int = 40,
    max_heuristic_hot_starts: int = 8,
    max_generic_hot_starts: int = 12,
    max_pool_candidates: int = 48,
    stall_perturbation_mode: str = "pool_then_flip",
    stall_flip_fraction: float = 0.10,
    scenario_bank: Optional[List[dict]] = None,
    surrogate_screen_mode: str = "robust",
    surrogate_screen_max_constraints_per_unit: int = 3,
    surrogate_screen_min_support_ratio: float = 0.85,
    surrogate_screen_max_normalized_violation: float = 0.05,
    surrogate_screen_min_mean_margin: float = 0.02,
    surrogate_screen_candidate_violation_tol: float = 0.02,
    surrogate_screen_soft_penalty: float = 25.0,
    projection_objective_tau="adaptive",
    use_subproblem_milp_candidate: bool = True,
    subproblem_milp_for_perturbations: bool = False,
    return_details: bool = False,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """Case118-specific FP with history retrieval, distance filtering and heuristic fusion."""
    if rng is None:
        rng = np.random.default_rng(42)
    if ppc is None:
        ppc = _load_default_case118_ppc()

    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        scenario_input = sample
    else:
        sample = _coerce_scenario_sample(pd_data)
        scenario_input = sample
    pd_matrix = get_sample_net_load(sample)

    if verbose:
        print("Case118 FP Step 1/6: predict duals and solve surrogate-warm LP", flush=True)
    lambda_val = lambda_predictor.predict(scenario_input)
    if manager is not None:
        x_lp = manager.solve_global(scenario_input, lambda_val)
    else:
        x_lp = solve_global_LP_relaxation(ppc, scenario_input, T_delta, trainers, lambda_val, agent=agent)
    x_init = round_to_integer(x_lp)

    if verbose:
        gap = float(np.mean(np.minimum(x_lp, 1.0 - x_lp)))
        print(f"  Case118 FP: global LP integrality gap={gap:.4f}", flush=True)
        print("Case118 FP Step 2/6: solve base and perturbed surrogate subproblems", flush=True)

    x_surr_lp, x_init_k, x_init_k_m, sub_details = collect_integer_solutions(
        scenario_input,
        lambda_val,
        trainers,
        n_perturbations=n_perturbations,
        n_similar_scenarios=n_similar_scenarios,
        similar_scenario_pool_size=similar_scenario_pool_size,
        n_load_perturbations=n_load_perturbations,
        load_perturbation_scale=load_perturbation_scale,
        perturb_std=perturb_std,
        neighborhood_weight=neighborhood_weight,
        lambda_predictor=lambda_predictor,
        rng=rng,
        use_milp_candidate=use_subproblem_milp_candidate,
        milp_for_perturbations=subproblem_milp_for_perturbations,
        return_details=True,
    )
    x_init_k_milp = sub_details.get("x_init_k_milp")
    x_init_k_m_milp = sub_details.get("x_init_k_m_milp")

    trusted_mask = identify_trusted_mask(x_lp, x_init_k, x_init_k_m, conf_threshold=conf_threshold)
    if verbose:
        print(
            f"Case118 FP Step 3/6: trusted bits={int(np.sum(trusted_mask))}/{trusted_mask.size}",
            flush=True,
        )

    history_candidates, history_records = build_case118_historical_candidates(
        sample,
        trainers,
        scenario_bank,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        ppc,
        T_delta,
        max_candidates=max_history_hot_starts,
        candidate_pool_size=history_candidate_pool_size,
    )
    history_pool = _as_candidate_pool(history_candidates)

    heuristic_candidates = build_case118_heuristic_candidates(
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        ppc,
        pd_matrix,
        T_delta,
    )

    generic_candidates = _build_hot_start_candidates(
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        trusted_mask,
        T_delta,
        nearby_commitment_candidates=history_candidates,
        max_perturbation_hot_starts=max(2, max_generic_hot_starts // 3),
        max_unit_options_per_generator=3,
        max_unit_combination_candidates=max_generic_hot_starts,
        ppc=ppc,
        unit_ids=None,
    )
    generic_ranked = _rank_hot_start_candidates(
        generic_candidates,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        trusted_mask,
        nearby_commitment_candidates=history_candidates,
    )
    generic_specs = [(name, candidate) for name, candidate, _score in generic_ranked[:max_generic_hot_starts]]

    if verbose:
        print(
            "Case118 FP Step 4/6: candidate fusion "
            f"history={len(history_candidates)}, heuristic={len(heuristic_candidates)}, generic={len(generic_specs)}",
            flush=True,
        )

    all_candidate_specs = (
        history_candidates
        + heuristic_candidates[:max_heuristic_hot_starts]
        + generic_specs
    )
    if x_init_k_milp is not None:
        all_candidate_specs.append(("case118_subproblem_milp_base", np.asarray(x_init_k_milp, dtype=int)))
    if x_init_k_m_milp is not None and x_init_k_m_milp.ndim == 3:
        all_candidate_specs.extend(
            (f"case118_subproblem_milp_perturb_{m + 1}", np.asarray(x_init_k_m_milp[:, m, :], dtype=int))
            for m in range(x_init_k_m_milp.shape[1])
        )

    sanitized_specs, rejected_structural = _sanitize_named_commitment_candidates(all_candidate_specs, ppc, T_delta)
    ranked, ranked_records = _rank_case118_candidates(
        sanitized_specs,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        trusted_mask,
        nearby_commitment_pool=history_pool,
        max_candidates=max_pool_candidates,
    )

    surrogate_screen_constraints: List[dict] = []
    surrogate_screen_summary = {
        "mode": str(surrogate_screen_mode),
        "hot_starts_before": int(len(ranked)),
        "hot_starts_after": int(len(ranked)),
        "x_pool_before": 0,
        "x_pool_after": 0,
        "n_constraints": 0,
        "constraints_per_unit": {},
        "hot_start_rejected": [],
        "x_pool_rejected": [],
    }

    if str(surrogate_screen_mode).strip().lower() not in ("none", "off", "false", "0"):
        surrogate_screen_constraints = _select_stable_surrogate_screen_constraints(
            scenario_input,
            trainers,
            lambda_val,
            x_lp,
            x_surr_lp,
            x_init_k,
            x_init_k_m,
            nearby_commitment_candidates=history_candidates,
            max_constraints_per_unit=surrogate_screen_max_constraints_per_unit,
            min_support_ratio=surrogate_screen_min_support_ratio,
            max_normalized_violation=surrogate_screen_max_normalized_violation,
            min_mean_margin=surrogate_screen_min_mean_margin,
        )
        per_unit_counts: Dict[int, int] = {}
        for row in surrogate_screen_constraints:
            unit_id = int(row["unit_id"])
            per_unit_counts[unit_id] = per_unit_counts.get(unit_id, 0) + 1
        surrogate_screen_summary["n_constraints"] = int(len(surrogate_screen_constraints))
        surrogate_screen_summary["constraints_per_unit"] = per_unit_counts

    hot_start_specs = [(name, candidate) for name, candidate, _score in ranked]
    if surrogate_screen_constraints and hot_start_specs:
        hot_start_specs, rejected_screen = _filter_named_commitment_candidates_by_surrogate_screen(
            hot_start_specs,
            surrogate_screen_constraints,
            normalized_violation_tol=surrogate_screen_candidate_violation_tol,
        )
        surrogate_screen_summary["hot_start_rejected"] = list(rejected_screen)
    surrogate_screen_summary["hot_starts_after"] = int(len(hot_start_specs))

    if not hot_start_specs:
        fallback, _rejected = _sanitize_named_commitment_candidates(
            [("case118_lp_round_fallback", x_init), ("case118_subproblem_fallback", x_init_k)],
            ppc,
            T_delta,
        )
        hot_start_specs = fallback

    pool_specs: List[Tuple[str, np.ndarray]] = [
        ("case118_subproblem_base", x_init_k),
        ("case118_lp_round", x_init),
    ]
    pool_specs.extend((f"case118_subproblem_perturb_{m + 1}", x_init_k_m[:, m, :]) for m in range(x_init_k_m.shape[1]))
    pool_specs.extend(hot_start_specs)
    pool_specs, rejected_pool_structural = _sanitize_named_commitment_candidates(pool_specs, ppc, T_delta)
    surrogate_screen_summary["x_pool_before"] = int(len(pool_specs))
    if surrogate_screen_constraints and pool_specs:
        pool_specs, rejected_pool_screen = _filter_named_commitment_candidates_by_surrogate_screen(
            pool_specs,
            surrogate_screen_constraints,
            normalized_violation_tol=surrogate_screen_candidate_violation_tol,
        )
        surrogate_screen_summary["x_pool_rejected"] = list(rejected_pool_screen)

    x_pool = _as_candidate_pool(pool_specs[:max_pool_candidates])
    surrogate_screen_summary["x_pool_after"] = int(0 if x_pool is None else x_pool.shape[0])

    score_lookup = {name: score for name, _candidate, score in ranked}
    hot_start_candidates = [
        (name, candidate, float(score_lookup.get(name, -1.0e6 - idx)))
        for idx, (name, candidate) in enumerate(hot_start_specs[:max_pool_candidates])
    ]

    if verbose:
        print("Case118 FP Step 5/6: run feasibility pump with screened starts", flush=True)
        for idx, (name, _candidate, score) in enumerate(hot_start_candidates[:12], start=1):
            print(f"  start {idx}: {name}, score={score:.2f}", flush=True)

    x_result = hot_start_candidates[0][1] if hot_start_candidates else x_init
    success = False
    selected_name = None
    for idx, (name, x_start, _score) in enumerate(hot_start_candidates, start=1):
        is_feasible, _reason = check_uc_feasibility(x_start, ppc, pd_matrix, T_delta)
        if is_feasible:
            x_result = np.asarray(x_start, dtype=int)
            success = True
            selected_name = name
            break
        if verbose:
            print(f"  Case118 FP hot start {idx}/{len(hot_start_candidates)}: {name}", flush=True)
        x_result, success = run_feasibility_pump(
            x_start,
            trusted_mask,
            ppc,
            pd_matrix,
            T_delta,
            x_pool=x_pool,
            surrogate_screen_constraints=surrogate_screen_constraints,
            surrogate_screen_soft_penalty=surrogate_screen_soft_penalty,
            projection_objective_tau=projection_objective_tau,
            max_iter=max_fp_iter,
            stall_perturbation_mode=stall_perturbation_mode,
            stall_flip_fraction=stall_flip_fraction,
            rng=rng,
            verbose=verbose,
        )
        selected_name = name
        if success:
            break

    if verbose:
        print(
            "Case118 FP Step 6/6: "
            + ("feasible solution found" if success else "no feasible solution"),
            flush=True,
        )

    details = {
        "x_lp": np.asarray(x_lp, dtype=float),
        "x_surr_lp": np.asarray(x_surr_lp, dtype=float),
        "x_init": np.asarray(x_init, dtype=int),
        "x_init_k": np.asarray(x_init_k, dtype=int),
        "x_init_k_m": np.asarray(x_init_k_m, dtype=int),
        "trusted_mask": np.asarray(trusted_mask, dtype=bool),
        "history_candidates": history_candidates,
        "history_candidate_records": history_records,
        "heuristic_candidates": heuristic_candidates,
        "ranked_candidate_records": ranked_records,
        "surrogate_screen_constraints": surrogate_screen_constraints,
        "surrogate_screen_summary": surrogate_screen_summary,
        "selected_hot_start": selected_name,
        "x_result": None if x_result is None else np.asarray(x_result, dtype=int),
        "rejected_structural_candidates": rejected_structural,
        "rejected_pool_structural_candidates": rejected_pool_structural,
        "objective_estimate": (
            None
            if x_result is None
            else _estimate_commitment_primal_objective(np.asarray(x_result, dtype=int), ppc, pd_matrix, T_delta)
        ),
    }
    if return_details:
        return x_result, success, details
    return x_result, success


__all__ = [
    "build_case118_historical_candidates",
    "build_case118_heuristic_candidates",
    "recover_integer_solution_case118",
]
