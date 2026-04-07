from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from feasibility_pump import (
        _build_nearby_commitment_candidates,
        _candidate_key,
        _coerce_scenario_sample,
        _compute_hot_start_support_reference,
        _estimate_commitment_primal_objective,
        _compute_vote_majority,
        _extract_commitment_from_sample,
        _repair_commitment_logic_heuristic,
        _score_hot_start_candidate,
        _sanitize_named_commitment_candidates,
        check_commitment_logic_feasibility,
        check_uc_feasibility,
        collect_integer_solutions,
        identify_trusted_mask,
        round_to_integer,
        run_feasibility_pump,
        solve_global_LP_relaxation,
    )
    from scenario_utils import get_sample_net_load, normalize_sample_arrays
except ImportError:
    from src.feasibility_pump import (
        _build_nearby_commitment_candidates,
        _candidate_key,
        _coerce_scenario_sample,
        _compute_hot_start_support_reference,
        _estimate_commitment_primal_objective,
        _compute_vote_majority,
        _extract_commitment_from_sample,
        _repair_commitment_logic_heuristic,
        _score_hot_start_candidate,
        _sanitize_named_commitment_candidates,
        check_commitment_logic_feasibility,
        check_uc_feasibility,
        collect_integer_solutions,
        identify_trusted_mask,
        round_to_integer,
        run_feasibility_pump,
        solve_global_LP_relaxation,
    )
    from src.scenario_utils import get_sample_net_load, normalize_sample_arrays


def _softmax_weights(scores: List[float], temperature: float = 1.0) -> np.ndarray:
    if not scores:
        return np.asarray([], dtype=float)
    arr = np.asarray(scores, dtype=float) / max(float(temperature), 1e-6)
    arr = arr - np.max(arr)
    weights = np.exp(arr)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _unique_binary_rows(rows: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    unique: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for name, row in rows:
        row_int = np.asarray(row, dtype=int)
        key = tuple(row_int.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, row_int))
    return unique


def _single_switch_pattern(T: int, tau: int, direction: str) -> np.ndarray:
    tau = int(np.clip(tau, 0, T))
    if direction == "up":
        return np.concatenate([np.zeros(tau, dtype=int), np.ones(T - tau, dtype=int)])
    return np.concatenate([np.ones(tau, dtype=int), np.zeros(T - tau, dtype=int)])


def _detect_single_switch_tau(row: np.ndarray) -> Optional[int]:
    row_int = np.asarray(row, dtype=int).flatten()
    if row_int.size <= 1:
        return None
    diff_idx = np.where(np.diff(row_int) != 0)[0]
    if diff_idx.size != 1:
        return None
    return int(diff_idx[0] + 1)


def _g1_switch_timing_bonus(
    row: np.ndarray,
    direction: str,
    reference_tau: Optional[int],
    horizon: int,
) -> float:
    """Bias G1 single-switch candidates toward earlier startup when the signal trends upward."""
    tau = _detect_single_switch_tau(row)
    if tau is None or reference_tau is None or horizon <= 1:
        return 0.0
    if direction != "up":
        return 0.0

    tau = int(np.clip(tau, 0, horizon))
    reference_tau = int(np.clip(reference_tau, 0, horizon))
    normalized_shift = (reference_tau - tau) / max(horizon - 1, 1)
    early_reward = max(normalized_shift, 0.0)
    late_penalty = max(-normalized_shift, 0.0)
    return float(2.4 * early_reward - 0.6 * late_penalty)


def _score_unit_option(
    row: np.ndarray,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    vote_majority: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: np.ndarray,
    nearby_rows: Optional[np.ndarray] = None,
    bonus: float = 0.0,
) -> float:
    score = _score_hot_start_candidate(
        np.asarray(row, dtype=int),
        np.asarray(x_lp, dtype=float),
        np.asarray(x_surr_lp, dtype=float),
        np.asarray(vote_majority, dtype=int),
        np.asarray(trusted_mask, dtype=bool),
        support_reference=np.asarray(support_reference, dtype=float),
        nearby_commitment_pool=None if nearby_rows is None else np.asarray(nearby_rows, dtype=int),
    )
    return float(score + bonus)


def _append_scored_option(
    options: List[dict],
    name: str,
    row: np.ndarray,
    score: float,
) -> None:
    options.append(
        {
            "name": str(name),
            "row": np.asarray(row, dtype=int),
            "score": float(score),
        }
    )


def _finalize_unit_options(
    options: List[dict],
    max_options: int,
) -> List[dict]:
    deduped: List[dict] = []
    seen = set()
    for option in sorted(options, key=lambda item: item["score"], reverse=True):
        key = tuple(np.asarray(option["row"], dtype=int).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
        if len(deduped) >= max(1, int(max_options)):
            break

    weights = _softmax_weights([item["score"] for item in deduped], temperature=3.0)
    for option, weight in zip(deduped, weights):
        option["weight"] = float(weight)
    return deduped


def _sanitize_case3lite_unit_options(
    options: List[dict],
    ppc: dict,
    T_delta: float,
    unit_id: int,
    fallback_rows: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> List[dict]:
    best_by_key: Dict[tuple, dict] = {}
    unit_ids = np.asarray([unit_id], dtype=int)

    def _consume(option_items: List[dict]) -> None:
        for option in sorted(option_items, key=lambda item: item["score"], reverse=True):
            repaired_row = _repair_commitment_logic_heuristic(
                np.asarray(option["row"], dtype=int),
                T_delta,
                ppc=ppc,
                unit_ids=unit_ids,
            )
            is_valid, _reason = check_commitment_logic_feasibility(
                repaired_row,
                ppc,
                T_delta,
                unit_ids=unit_ids,
            )
            if not is_valid:
                continue

            key = tuple(np.asarray(repaired_row, dtype=int).tolist())
            if key in best_by_key:
                continue
            sanitized = dict(option)
            sanitized["row"] = np.asarray(repaired_row, dtype=int)
            best_by_key[key] = sanitized

    _consume(list(options))
    if best_by_key or not fallback_rows:
        return list(best_by_key.values())

    fallback_options = [
        {
            "name": str(name),
            "row": np.asarray(row, dtype=int),
            "score": -1e6 - idx,
        }
        for idx, (name, row) in enumerate(fallback_rows)
    ]
    _consume(fallback_options)
    return list(best_by_key.values())


def _repair_and_validate_case3lite_candidate(
    candidate: np.ndarray,
    ppc: dict,
    T_delta: float,
) -> Optional[np.ndarray]:
    repaired = _repair_commitment_logic_heuristic(
        np.asarray(candidate, dtype=int),
        T_delta,
        ppc=ppc,
    )
    is_valid, _reason = check_commitment_logic_feasibility(
        repaired,
        ppc,
        T_delta,
    )
    if not is_valid:
        return None
    return np.asarray(repaired, dtype=int)


def _build_g0_options(
    g: int,
    T: int,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    vote_majority: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: np.ndarray,
    nearby_rows: Optional[np.ndarray],
) -> Tuple[List[dict], np.ndarray]:
    options: List[dict] = []
    lp_round = round_to_integer(x_lp)
    surr_round = round_to_integer(x_surr_lp)
    unit_lp = np.asarray(x_init_k, dtype=int)
    all_on = np.ones(T, dtype=int)

    support_mean = float(np.mean(support_reference))
    support_floor = float(np.min(support_reference))
    prefer_all_on = support_mean >= 0.82 and support_floor >= 0.50

    candidates = [
        ("always_on", all_on, 2.0 if prefer_all_on else 0.5),
        ("lp_round", lp_round, 0.2),
        ("surrogate_round", surr_round, 0.0),
        ("unit_lp", unit_lp, 0.0),
        ("vote_majority", vote_majority, 0.0),
    ]
    if x_init_k_m.ndim == 2 and x_init_k_m.size > 0:
        for idx in range(x_init_k_m.shape[0]):
            candidates.append((f"perturb_{idx + 1}", x_init_k_m[idx], 0.0))
    if nearby_rows is not None and nearby_rows.size > 0:
        for idx in range(nearby_rows.shape[0]):
            candidates.append((f"nearby_{idx + 1}", nearby_rows[idx], 0.0))

    for name, row in _unique_binary_rows([(n, r) for n, r, _b in candidates]):
        row_bonus = next(b for n, _r, b in candidates if n == name)
        score = _score_unit_option(
            row,
            x_lp,
            x_surr_lp,
            vote_majority,
            trusted_mask,
            support_reference,
            nearby_rows=nearby_rows,
            bonus=row_bonus,
        )
        _append_scored_option(options, name, row, score)

    trusted_override = np.asarray(trusted_mask, dtype=bool).copy()
    if prefer_all_on:
        trusted_override[:] = True
    else:
        trusted_override[:] = trusted_override & (support_reference >= 0.88)

    return _finalize_unit_options(options, max_options=4), trusted_override


def _build_g1_options(
    g: int,
    T: int,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    vote_majority: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: np.ndarray,
    nearby_rows: Optional[np.ndarray],
) -> Tuple[List[dict], np.ndarray]:
    options: List[dict] = []
    signal = np.clip(
        0.45 * x_lp + 0.30 * x_surr_lp + 0.25 * support_reference,
        0.0,
        1.0,
    )
    direction = "up" if float(signal[-1] - signal[0]) >= 0.0 else "down"
    round_switch = round_to_integer(signal)

    tau_candidates = set()
    crossing = np.where(np.diff(signal >= 0.5) != 0)[0]
    if crossing.size > 0:
        for idx in crossing:
            if direction == "up":
                tau_candidates.update([int(idx - 2), int(idx - 1), int(idx), int(idx + 1), int(idx + 2)])
            else:
                tau_candidates.update([int(idx), int(idx + 1), int(idx + 2)])
    else:
        pivot = int(np.argmin(np.abs(signal - 0.5)))
        if direction == "up":
            tau_candidates.update([pivot - 2, pivot - 1, pivot, pivot + 1])
        else:
            tau_candidates.update([pivot, pivot + 1])

    direct_rows = [round_to_integer(x_lp), round_to_integer(x_surr_lp), vote_majority, x_init_k]
    if x_init_k_m.ndim == 2 and x_init_k_m.size > 0:
        direct_rows.extend(list(np.asarray(x_init_k_m, dtype=int)))
    if nearby_rows is not None and nearby_rows.size > 0:
        direct_rows.extend(list(np.asarray(nearby_rows, dtype=int)))
    reference_tau = _detect_single_switch_tau(round_switch)
    for row in direct_rows:
        tau = _detect_single_switch_tau(row)
        if tau is not None:
            if direction == "up":
                tau_candidates.update([tau - 2, tau - 1, tau, tau + 1])
            else:
                tau_candidates.update([tau - 1, tau, tau + 1])
            if reference_tau is None:
                reference_tau = tau

    tau_candidates = {int(np.clip(tau, 0, T)) for tau in tau_candidates}
    if reference_tau is None and tau_candidates:
        reference_tau = min(tau_candidates) if direction == "up" else max(tau_candidates)

    base_rows = [
        ("round_switch", round_switch, 0.6),
        ("lp_round", round_to_integer(x_lp), 0.0),
        ("surrogate_round", round_to_integer(x_surr_lp), 0.0),
        ("vote_majority", vote_majority, 0.0),
        ("unit_lp", x_init_k, 0.0),
    ]
    for name, row, bonus in base_rows:
        score = _score_unit_option(
            row,
            x_lp,
            x_surr_lp,
            vote_majority,
            trusted_mask,
            support_reference,
            nearby_rows=nearby_rows,
            bonus=bonus + _g1_switch_timing_bonus(row, direction, reference_tau, T),
        )
        _append_scored_option(options, name, row, score)

    for tau in sorted(tau_candidates):
        row = _single_switch_pattern(T, tau, direction=direction)
        score = _score_unit_option(
            row,
            x_lp,
            x_surr_lp,
            vote_majority,
            trusted_mask,
            support_reference,
            nearby_rows=nearby_rows,
            bonus=1.0 + _g1_switch_timing_bonus(row, direction, reference_tau, T),
        )
        _append_scored_option(options, f"corrected_single_switch_t{tau}", row, score)

    trusted_override = np.asarray(trusted_mask, dtype=bool).copy()
    trusted_override[:] = trusted_override & (np.abs(signal - 0.5) >= 0.34)
    best_switch = None
    for option in sorted(options, key=lambda item: item["score"], reverse=True):
        if option["name"].startswith("corrected_single_switch_"):
            best_switch = option["row"]
            break
    if best_switch is not None:
        tau = _detect_single_switch_tau(best_switch)
        if tau is not None:
            lo = max(0, tau - 2)
            hi = min(T, tau + 2)
            trusted_override[lo:hi] = False

    return _finalize_unit_options(options, max_options=6), trusted_override


def _build_g2_block_rows(
    T: int,
    support_reference: np.ndarray,
    load_profile: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    block_rows: List[Tuple[str, np.ndarray]] = []
    if T <= 0:
        return block_rows

    load_norm = np.asarray(load_profile, dtype=float)
    if load_norm.size != T:
        load_norm = np.resize(load_norm, T)
    if np.max(load_norm) > 1e-9:
        load_norm = load_norm / np.max(load_norm)

    activity = np.clip(0.60 * support_reference + 0.40 * load_norm, 0.0, 1.0)
    peak_order = np.argsort(-activity)
    centers = list(dict.fromkeys(int(idx) for idx in peak_order[: min(4, T)]))
    lengths = [2, 3, 4, 5]
    for center in centers:
        for length in lengths:
            start = max(0, center - length // 2)
            end = min(T, start + length)
            start = max(0, end - length)
            row = np.zeros(T, dtype=int)
            row[start:end] = 1
            block_rows.append((f"block_{start}_{end}", row))
    return _unique_binary_rows(block_rows)


def _build_g2_options(
    g: int,
    T: int,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    vote_majority: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: np.ndarray,
    nearby_rows: Optional[np.ndarray],
    load_profile: np.ndarray,
) -> Tuple[List[dict], np.ndarray]:
    options: List[dict] = []
    base_rows = [
        ("all_off", np.zeros(T, dtype=int), 1.2 if float(np.mean(support_reference)) <= 0.20 else 0.1),
        ("lp_round", round_to_integer(x_lp), 0.0),
        ("surrogate_round", round_to_integer(x_surr_lp), 0.0),
        ("vote_majority", vote_majority, 0.0),
        ("unit_lp", x_init_k, 0.0),
    ]
    if x_init_k_m.ndim == 2 and x_init_k_m.size > 0:
        for idx in range(x_init_k_m.shape[0]):
            base_rows.append((f"perturb_{idx + 1}", x_init_k_m[idx], 0.0))
    if nearby_rows is not None and nearby_rows.size > 0:
        for idx in range(nearby_rows.shape[0]):
            base_rows.append((f"nearby_{idx + 1}", nearby_rows[idx], 0.0))
    base_rows.extend((name, row, 0.4) for name, row in _build_g2_block_rows(T, support_reference, load_profile))

    for name, row, bonus in base_rows:
        score = _score_unit_option(
            row,
            x_lp,
            x_surr_lp,
            vote_majority,
            trusted_mask,
            support_reference,
            nearby_rows=nearby_rows,
            bonus=bonus,
        )
        _append_scored_option(options, name, row, score)

    trusted_override = np.asarray(trusted_mask, dtype=bool).copy()
    trusted_override[:] = trusted_override & (np.abs(support_reference - 0.5) >= 0.47)
    trusted_override[:] = trusted_override & ((support_reference <= 0.05) | (support_reference >= 0.95))

    return _finalize_unit_options(options, max_options=6), trusted_override


def _build_case3lite_unit_options(
    sample: dict,
    ppc: dict,
    T_delta: float,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    base_trusted_mask: np.ndarray,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]],
) -> Tuple[List[List[dict]], np.ndarray, np.ndarray]:
    ng, T = x_lp.shape
    vote_majority = _compute_vote_majority(x_init_k, x_init_k_m)
    nearby_pool = None
    if nearby_commitment_candidates:
        nearby_pool = np.stack(
            [np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates],
            axis=0,
        )
    support_reference = _compute_hot_start_support_reference(
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        nearby_commitment_pool=nearby_pool,
    )

    load_profile = np.sum(get_sample_net_load(sample), axis=0)
    trusted_mask = np.asarray(base_trusted_mask, dtype=bool).copy()
    unit_options: List[List[dict]] = []
    for g in range(ng):
        nearby_rows = None if nearby_pool is None else nearby_pool[:, g, :]
        if g == 0:
            options, trusted_row = _build_g0_options(
                g,
                T,
                x_lp[g],
                x_surr_lp[g],
                x_init_k[g],
                x_init_k_m[g],
                vote_majority[g],
                trusted_mask[g],
                support_reference[g],
                nearby_rows,
            )
        elif g == 1:
            options, trusted_row = _build_g1_options(
                g,
                T,
                x_lp[g],
                x_surr_lp[g],
                x_init_k[g],
                x_init_k_m[g],
                vote_majority[g],
                trusted_mask[g],
                support_reference[g],
                nearby_rows,
            )
        else:
            options, trusted_row = _build_g2_options(
                g,
                T,
                x_lp[g],
                x_surr_lp[g],
                x_init_k[g],
                x_init_k_m[g],
                vote_majority[g],
                trusted_mask[g],
                support_reference[g],
                nearby_rows,
                load_profile,
            )
        fallback_rows = [
            ("unit_lp_fallback", x_init_k[g]),
            ("lp_round_fallback", round_to_integer(x_lp[g])),
            ("surrogate_round_fallback", round_to_integer(x_surr_lp[g])),
            ("vote_fallback", vote_majority[g]),
            ("all_off_fallback", np.zeros(T, dtype=int)),
            ("all_on_fallback", np.ones(T, dtype=int)),
        ]
        options = _sanitize_case3lite_unit_options(
            options,
            ppc,
            T_delta,
            g,
            fallback_rows=fallback_rows,
        )
        options = _finalize_unit_options(options, max_options=max(1, len(options)))
        unit_options.append(options)
        trusted_mask[g] = trusted_row

    return unit_options, trusted_mask, support_reference


def _build_case3lite_global_combinations(
    unit_options: List[List[dict]],
    ppc: dict,
    pd_data: np.ndarray,
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: np.ndarray,
    T_delta: float,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]],
    max_global_combinations: int,
) -> List[dict]:
    vote_majority = _compute_vote_majority(x_init_k, x_init_k_m)
    nearby_pool = None
    if nearby_commitment_candidates:
        nearby_pool = np.stack(
            [np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates],
            axis=0,
        )
    g1_direction = None
    g1_reference_tau = None
    if x_lp.shape[0] >= 2:
        g1_signal = np.clip(
            0.45 * np.asarray(x_lp[1], dtype=float)
            + 0.30 * np.asarray(x_surr_lp[1], dtype=float)
            + 0.25 * np.asarray(support_reference[1], dtype=float),
            0.0,
            1.0,
        )
        g1_direction = "up" if float(g1_signal[-1] - g1_signal[0]) >= 0.0 else "down"
        g1_reference_tau = _detect_single_switch_tau(round_to_integer(g1_signal))

    combos: List[dict] = []
    seen = set()
    for combo_index, option_tuple in enumerate(itertools.product(*unit_options), start=1):
        rows = [np.asarray(item["row"], dtype=int) for item in option_tuple]
        candidate = np.stack(rows, axis=0)
        repaired = _repair_and_validate_case3lite_candidate(candidate, ppc, T_delta)
        if repaired is None:
            continue
        key = _candidate_key(repaired)
        if key in seen:
            continue
        seen.add(key)

        local_score = float(sum(item["score"] for item in option_tuple))
        global_match = _score_hot_start_candidate(
            repaired,
            x_lp,
            x_surr_lp,
            vote_majority,
            trusted_mask,
            support_reference=support_reference,
            nearby_commitment_pool=nearby_pool,
        )
        bonus = 0.0
        if repaired.shape[0] >= 1 and np.all(repaired[0] == 1):
            bonus += 1.5
        if repaired.shape[0] >= 2 and _detect_single_switch_tau(repaired[1]) is not None:
            bonus += 0.8
            bonus += 0.8 * _g1_switch_timing_bonus(
                repaired[1],
                g1_direction or "up",
                g1_reference_tau,
                repaired.shape[1],
            )
        combo_score = float(global_match + 0.40 * local_score + bonus)
        objective_estimate = _estimate_commitment_primal_objective(
            repaired,
            ppc,
            pd_data,
            T_delta,
        )

        combos.append(
            {
                "id": combo_index,
                "candidate": repaired,
                "score": combo_score,
                "objective_estimate": float(objective_estimate),
                "unit_choice_names": [item["name"] for item in option_tuple],
                "unit_choice_weights": [float(item.get("weight", 0.0)) for item in option_tuple],
                "unit_option_scores": [float(item["score"]) for item in option_tuple],
            }
        )

    if not combos:
        fallback_specs = [
            ("unit_lp_fallback_combo", np.asarray(x_init_k, dtype=int)),
            ("joint_lp_fallback_combo", round_to_integer(x_lp)),
            ("surrogate_lp_fallback_combo", round_to_integer(x_surr_lp)),
        ]
        fallback_candidates, _rejected_fallbacks = _sanitize_named_commitment_candidates(
            fallback_specs,
            ppc,
            T_delta,
        )
        for combo_index, (name, candidate) in enumerate(fallback_candidates, start=1):
            combos.append(
                {
                    "id": combo_index,
                    "candidate": np.asarray(candidate, dtype=int),
                    "score": -1e6 - combo_index,
                    "objective_estimate": float(
                        _estimate_commitment_primal_objective(
                            candidate,
                            ppc,
                            pd_data,
                            T_delta,
                        )
                    ),
                    "weight": 0.0,
                    "unit_choice_names": [name] * candidate.shape[0],
                    "unit_choice_weights": [0.0] * candidate.shape[0],
                    "unit_option_scores": [0.0] * candidate.shape[0],
                }
            )

    combos.sort(key=lambda item: item["score"], reverse=True)
    combos = combos[: max(1, int(max_global_combinations))]
    weights = _softmax_weights([item["score"] for item in combos], temperature=4.0)
    for combo, weight in zip(combos, weights):
        combo["weight"] = float(weight)
    return combos


def _build_case3lite_x_pool(
    ppc: dict,
    T_delta: float,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    global_combinations: List[dict],
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]],
) -> Optional[np.ndarray]:
    x_pool_list: List[np.ndarray] = [np.asarray(x_init_k, dtype=int)]
    if x_init_k_m.ndim == 3 and x_init_k_m.shape[1] > 0:
        x_pool_list.extend(list(np.asarray(x_init_k_m, dtype=int).transpose(1, 0, 2)))
    if nearby_commitment_candidates:
        x_pool_list.extend(np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates)
    x_pool_list.extend(np.asarray(combo["candidate"], dtype=int) for combo in global_combinations)

    unique: List[np.ndarray] = []
    seen = set()
    for candidate in x_pool_list:
        candidate_int = _repair_and_validate_case3lite_candidate(candidate, ppc, T_delta)
        if candidate_int is None:
            continue
        key = _candidate_key(candidate_int)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate_int)

    if not unique:
        base_candidate = _repair_and_validate_case3lite_candidate(x_init_k, ppc, T_delta)
        if base_candidate is not None:
            unique.append(base_candidate)

    return np.stack(unique, axis=0) if unique else None


def _print_unit_option_weights(unit_options: List[List[dict]]) -> None:
    print("Case3lite custom FP unit option weights:", flush=True)
    for g, options in enumerate(unit_options):
        print(f"  G{g}:", flush=True)
        for option in options:
            print(
                f"    {option['name']:<28} weight={option.get('weight', 0.0):.3f} "
                f"score={option['score']:.2f}",
                flush=True,
            )


def _print_global_combo_weights(global_combinations: List[dict], top_n: int = 8) -> None:
    print("Case3lite custom FP global combination weights:", flush=True)
    for rank, combo in enumerate(global_combinations[:top_n], start=1):
        pieces = [
            f"G{g}:{name}@{weight:.3f}"
            for g, (name, weight) in enumerate(
                zip(combo["unit_choice_names"], combo["unit_choice_weights"])
            )
        ]
        objective_piece = ""
        if combo.get("objective_estimate") is not None:
            objective_piece = f" obj≈{float(combo['objective_estimate']):.2f}"
        print(
            f"  #{rank:<2d} combo_weight={combo.get('weight', 0.0):.3f} "
            f"score={combo['score']:.2f}{objective_piece}  {' | '.join(pieces)}",
            flush=True,
        )


def _print_final_combo_summary(combo: dict) -> None:
    print("Case3lite custom FP final feasible combination:", flush=True)
    objective_piece = ""
    if combo.get("objective_estimate") is not None:
        objective_piece = f" obj≈{float(combo['objective_estimate']):.2f}"
    print(
        f"  combo_weight={combo.get('weight', 0.0):.3f} score={combo['score']:.2f}{objective_piece}",
        flush=True,
    )
    for g, (name, weight, score) in enumerate(
        zip(
            combo["unit_choice_names"],
            combo["unit_choice_weights"],
            combo["unit_option_scores"],
        )
    ):
        print(
            f"  G{g}: {name:<28} option_weight={weight:.3f} option_score={score:.2f}",
            flush=True,
        )


def _save_fig(fig: "plt.Figure", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix(".png")))
    fig.savefig(str(path.with_suffix(".pdf")))
    plt.close(fig)


def _plot_unit_options(
    unit_options: List[List[dict]],
    plot_dir: Path,
    sample_tag: str,
) -> None:
    if not MPL_AVAILABLE:
        return

    n_units = len(unit_options)
    fig, axes = plt.subplots(n_units, 1, figsize=(12, 2.8 * n_units), squeeze=False)
    fig.suptitle(f"Case3lite custom FP unit candidates [{sample_tag}]", fontsize=12, fontweight="bold")
    for g, options in enumerate(unit_options):
        ax = axes[g, 0]
        data = np.stack([np.asarray(item["row"], dtype=int) for item in options], axis=0)
        im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        labels = [
            f"{item['name']}  w={item.get('weight', 0.0):.2f}"
            for item in options
        ]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel(f"G{g}")
        if g == n_units - 1:
            ax.set_xlabel("Time period")
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)

    fig.tight_layout()
    _save_fig(fig, plot_dir / f"{sample_tag}_unit_candidates")


def _plot_final_feasible_unit_combination(
    unit_options: List[List[dict]],
    final_solution: Optional[np.ndarray],
    plot_dir: Path,
    sample_tag: str,
) -> None:
    if not MPL_AVAILABLE or final_solution is None:
        return

    final_arr = np.asarray(final_solution, dtype=int)
    n_units = final_arr.shape[0]
    fig, axes = plt.subplots(n_units, 1, figsize=(12, 2.2 * n_units), squeeze=False)
    fig.suptitle(
        f"Case3lite custom FP final feasible unit combination [{sample_tag}]",
        fontsize=12,
        fontweight="bold",
    )

    for g in range(n_units):
        ax = axes[g, 0]
        row = final_arr[g][np.newaxis, :]
        im = ax.imshow(row, aspect="auto", cmap="Blues", vmin=0, vmax=1)

        matched_name = "final_feasible"
        for option in unit_options[g]:
            option_row = np.asarray(option["row"], dtype=int)
            if option_row.shape == final_arr[g].shape and np.array_equal(option_row, final_arr[g]):
                matched_name = f"{option['name']}  w={option.get('weight', 0.0):.2f}"
                break
        else:
            matched_name = "fp_adjusted_final"

        ax.set_yticks([0])
        ax.set_yticklabels([matched_name], fontsize=8)
        ax.set_ylabel(f"G{g}")
        if g == n_units - 1:
            ax.set_xlabel("Time period")
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)

    fig.tight_layout()
    _save_fig(fig, plot_dir / f"{sample_tag}_final_feasible_unit_combination")


def _plot_global_combinations(
    global_combinations: List[dict],
    final_solution: Optional[np.ndarray],
    reference_solution: Optional[np.ndarray],
    plot_dir: Path,
    sample_tag: str,
) -> None:
    if not MPL_AVAILABLE or not global_combinations:
        return

    top_k = min(8, len(global_combinations))
    panels: List[Tuple[str, np.ndarray]] = []
    for rank, combo in enumerate(global_combinations[:top_k], start=1):
        title = f"combo{rank} w={combo.get('weight', 0.0):.2f}"
        panels.append((title, np.asarray(combo["candidate"], dtype=int)))
    if final_solution is not None:
        panels.append(("final_feasible", np.asarray(final_solution, dtype=int)))
    if reference_solution is not None:
        panels.append(("reference", np.asarray(reference_solution, dtype=int)))

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.8 * len(panels)), squeeze=False)
    fig.suptitle(f"Case3lite custom FP combinations [{sample_tag}]", fontsize=12, fontweight="bold")
    for ax, (title, data) in zip(axes[:, 0], panels):
        im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"G{g}" for g in range(data.shape[0])], fontsize=8)
        ax.set_xlabel("Time period")
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)

    fig.tight_layout()
    _save_fig(fig, plot_dir / f"{sample_tag}_global_combinations")


def _plot_lp_surrogate_comparison(
    x_lp: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    final_solution: Optional[np.ndarray],
    reference_solution: Optional[np.ndarray],
    plot_dir: Path,
    sample_tag: str,
) -> None:
    if not MPL_AVAILABLE:
        return

    panels: List[Tuple[str, np.ndarray, str]] = [
        ("joint_lp", np.asarray(x_lp, dtype=float), "viridis"),
        ("surrogate_lp", np.asarray(x_surr_lp, dtype=float), "viridis"),
        ("surrogate_round", np.asarray(x_init_k, dtype=int), "Blues"),
    ]
    if final_solution is not None:
        panels.append(("final_feasible", np.asarray(final_solution, dtype=int), "Blues"))
    if reference_solution is not None:
        panels.append(("reference", np.asarray(reference_solution, dtype=int), "Blues"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.8 * len(panels)), squeeze=False)
    fig.suptitle(f"Case3lite custom FP LP vs surrogate [{sample_tag}]", fontsize=12, fontweight="bold")
    for ax, (title, data, cmap) in zip(axes[:, 0], panels):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"G{g}" for g in range(data.shape[0])], fontsize=8)
        ax.set_xlabel("Time period")
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)

    fig.tight_layout()
    _save_fig(fig, plot_dir / f"{sample_tag}_lp_surrogate_comparison")


def recover_integer_solution_case3lite(
    pd_data: np.ndarray | dict,
    trainers: Dict[int, object],
    lambda_predictor,
    ppc: dict,
    T_delta: float,
    agent=None,
    n_perturbations: int = 5,
    n_similar_scenarios: int = 0,
    similar_scenario_pool_size: int = 10,
    n_load_perturbations: int = 0,
    load_perturbation_scale: float = 0.03,
    conf_threshold: float = 0.15,
    max_fp_iter: int = 50,
    perturb_std: float = 0.1,
    neighborhood_weight: float = 0.35,
    stall_perturbation_mode: str = "pool_then_flip",
    stall_flip_fraction: float = 0.10,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None,
    scenario_bank: Optional[List[dict]] = None,
    plot_dir: Optional[str | Path] = None,
    sample_tag: str = "sample",
    max_global_combinations: int = 24,
) -> Tuple[Optional[np.ndarray], bool, dict]:
    sample = normalize_sample_arrays(dict(_coerce_scenario_sample(pd_data)))
    pd_matrix = get_sample_net_load(sample)
    if rng is None:
        rng = np.random.default_rng(42)

    expected_ng = int(ppc['gen'].shape[0])
    loaded_unit_ids = sorted(
        int(unit_id)
        for unit_id in trainers.keys()
        if 0 <= int(unit_id) < expected_ng
    )
    expected_unit_ids = list(range(expected_ng))
    if loaded_unit_ids != expected_unit_ids:
        missing_unit_ids = [unit_id for unit_id in expected_unit_ids if unit_id not in loaded_unit_ids]
        extra_unit_ids = [unit_id for unit_id in loaded_unit_ids if unit_id not in expected_unit_ids]
        raise ValueError(
            "recover_integer_solution_case3lite() requires surrogate trainers for all generators "
            f"because it builds a global FP trusted mask. "
            f"loaded_unit_ids={loaded_unit_ids}, expected_unit_ids={expected_unit_ids}, "
            f"missing_unit_ids={missing_unit_ids}, extra_unit_ids={extra_unit_ids}. "
            "Do not use partial UNIT_IDS when running the global FP pipeline."
        )

    if verbose:
        print("Case3lite custom FP: Step 1/6 predict lambda", flush=True)
    lambda_val = lambda_predictor.predict(sample)

    if verbose:
        print("Case3lite custom FP: Step 2/6 solve joint LP", flush=True)
    x_lp = solve_global_LP_relaxation(
        ppc,
        sample,
        T_delta,
        trainers,
        lambda_val,
        agent=agent,
    )

    if verbose:
        print("Case3lite custom FP: Step 3/6 collect surrogate candidates", flush=True)
    x_surr_lp, x_init_k, x_init_k_m = collect_integer_solutions(
        sample,
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
    )

    base_trusted_mask = identify_trusted_mask(
        x_lp,
        x_init_k,
        x_init_k_m,
        conf_threshold=conf_threshold,
    )

    nearby_commitment_candidates = _build_nearby_commitment_candidates(
        sample,
        trainers,
        scenario_bank,
        x_lp.shape[0],
        x_lp.shape[1],
        n_candidates=4,
        candidate_pool_size=12,
        rng=rng,
    )
    nearby_commitment_candidates, _rejected_nearby_candidates = _sanitize_named_commitment_candidates(
        nearby_commitment_candidates,
        ppc,
        T_delta,
    )
    if verbose and _rejected_nearby_candidates:
        print(
            f"Case3lite custom FP: filtered {len(_rejected_nearby_candidates)} "
            f"structurally infeasible nearby commitments",
            flush=True,
        )

    if verbose:
        print("Case3lite custom FP: Step 4/6 build role-based unit options", flush=True)
    unit_options, trusted_mask, support_reference = _build_case3lite_unit_options(
        sample,
        ppc,
        T_delta,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        base_trusted_mask,
        nearby_commitment_candidates,
    )
    _print_unit_option_weights(unit_options)

    if verbose:
        print("Case3lite custom FP: Step 5/6 compose global combinations", flush=True)
    global_combinations = _build_case3lite_global_combinations(
        unit_options,
        ppc,
        pd_matrix,
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        trusted_mask,
        support_reference,
        T_delta,
        nearby_commitment_candidates,
        max_global_combinations=max_global_combinations,
    )
    _print_global_combo_weights(global_combinations)

    x_pool = _build_case3lite_x_pool(
        ppc,
        T_delta,
        x_init_k,
        x_init_k_m,
        global_combinations,
        nearby_commitment_candidates,
    )

    final_solution = None
    final_success = False
    final_combo = None
    final_objective_estimate = None
    for rank, combo in enumerate(global_combinations, start=1):
        x_start = np.asarray(combo["candidate"], dtype=int)
        start_feasible, _reason = check_uc_feasibility(x_start, ppc, pd_matrix, T_delta)
        if verbose:
            print(
                f"Case3lite custom FP: try combo {rank}/{len(global_combinations)} "
                f"(weight={combo.get('weight', 0.0):.3f}, score={combo['score']:.2f})",
                flush=True,
            )
        candidate_solution = None
        candidate_success = False
        if start_feasible:
            candidate_solution = x_start
            candidate_success = True
        else:
            x_result, success = run_feasibility_pump(
                x_start,
                trusted_mask,
                ppc,
                pd_matrix,
                T_delta,
                x_pool=x_pool,
                max_iter=max_fp_iter,
                stall_perturbation_mode=stall_perturbation_mode,
                stall_flip_fraction=stall_flip_fraction,
                rng=rng,
                verbose=False,
            )
            if success:
                candidate_solution = np.asarray(x_result, dtype=int)
                candidate_success = True

        if not candidate_success or candidate_solution is None:
            continue

        candidate_objective_estimate = _estimate_commitment_primal_objective(
            candidate_solution,
            ppc,
            pd_matrix,
            T_delta,
        )
        if verbose:
            print(
                f"    feasible candidate objective≈{candidate_objective_estimate:.2f}",
                flush=True,
            )

        combo["final_objective_estimate"] = float(candidate_objective_estimate)
        if (
            final_solution is None
            or final_objective_estimate is None
            or candidate_objective_estimate < final_objective_estimate - 1e-6
            or (
                abs(candidate_objective_estimate - final_objective_estimate) <= 1e-6
                and combo["score"] > final_combo["score"]
            )
        ):
            final_solution = np.asarray(candidate_solution, dtype=int)
            final_success = True
            final_combo = combo
            final_objective_estimate = float(candidate_objective_estimate)
            if verbose:
                print(
                    f"    incumbent updated by combo {rank}: obj≈{final_objective_estimate:.2f}",
                    flush=True,
                )

    if final_combo is not None:
        final_combo["objective_estimate"] = float(final_objective_estimate)
        _print_final_combo_summary(final_combo)

    reference_solution = _extract_commitment_from_sample(sample, x_lp.shape[0], x_lp.shape[1])
    if plot_dir is not None:
        plot_dir_path = Path(plot_dir)
        _plot_unit_options(unit_options, plot_dir_path, sample_tag)
        _plot_final_feasible_unit_combination(
            unit_options,
            final_solution,
            plot_dir_path,
            sample_tag,
        )
        _plot_global_combinations(
            global_combinations,
            final_solution,
            reference_solution,
            plot_dir_path,
            sample_tag,
        )
        _plot_lp_surrogate_comparison(
            x_lp,
            x_surr_lp,
            x_init_k,
            final_solution,
            reference_solution,
            plot_dir_path,
            sample_tag,
        )

    details = {
        "x_lp": x_lp,
        "x_surr_lp": x_surr_lp,
        "x_init_k": x_init_k,
        "x_init_k_m": x_init_k_m,
        "unit_options": unit_options,
        "global_combinations": global_combinations,
        "trusted_mask": trusted_mask,
        "support_reference": support_reference,
        "reference_solution": reference_solution,
        "selected_combo": final_combo,
        "final_objective_estimate": final_objective_estimate,
    }
    return final_solution, final_success, details
