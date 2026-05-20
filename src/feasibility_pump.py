"""
可行性泵（Feasibility Pump）实�?
用于从LP松弛解恢复UC问题的整数可行解

Pipeline�?
  1. 求解全局UC LP松弛（加入代理约束）�?x_LP
  2. 通过各机组子问题LP（含代理约束 + 参数扰动）收集多组整数解
  3. �?LP整数性强 + 多来源一�?识别高可信度变量并固�?
  4. 可行性泵：LP投影（最小化 L1 距离�? 四舍五入，迭代至整数可行
"""

import json
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A, BR_STATUS

try:
    from uc_NN_subproblem import (
        SubproblemSurrogateTrainer,
        CONSTRAINT_STRATEGY_ALL,
        CONSTRAINT_STRATEGY_ALL_SINGLE_TIME,
        CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
        CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        SURROGATE_SINGLE_TIME_OFFSETS,
        SURROGATE_TRIPLE_WINDOW_OFFSETS,
        build_surrogate_constraint_expression,
        normalize_constraint_generation_strategy,
        resolve_constraint_offsets_from_trainer,
        select_constraint_layout,
    )
except ImportError:
    from src.uc_NN_subproblem import (
        SubproblemSurrogateTrainer,
        CONSTRAINT_STRATEGY_ALL,
        CONSTRAINT_STRATEGY_ALL_SINGLE_TIME,
        CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4,
        CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE,
        SURROGATE_SINGLE_TIME_OFFSETS,
        SURROGATE_TRIPLE_WINDOW_OFFSETS,
        build_surrogate_constraint_expression,
        normalize_constraint_generation_strategy,
        resolve_constraint_offsets_from_trainer,
        select_constraint_layout,
    )
try:
    from sparse_surrogate_mining import (
        SparseSurrogateLibrary,
        add_sparse_parameterized_constraints,
    )
except ImportError:
    from src.sparse_surrogate_mining import (
        SparseSurrogateLibrary,
        add_sparse_parameterized_constraints,
    )
try:
    from sparse_constraint_templates import (
        SparseConstraintTemplateLibrary,
        add_sparse_x_templates_to_model,
    )
except ImportError:
    from src.sparse_constraint_templates import (
        SparseConstraintTemplateLibrary,
        add_sparse_x_templates_to_model,
    )
try:
    from scenario_utils import (
        get_feature_vector_from_sample,
        get_sample_load_data,
        get_sample_net_load,
        get_sample_renewable_data,
        normalize_sample_arrays,
    )
except ImportError:
    from src.scenario_utils import (
        get_feature_vector_from_sample,
        get_sample_load_data,
        get_sample_net_load,
        get_sample_renewable_data,
        normalize_sample_arrays,
    )


# ========================== Step 1：整数恢复启发式 ==========================

def round_to_integer(x_LP: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将LP松弛解四舍五入为整数解�?

    Args:
        x_LP: LP松弛解，shape (ng, T) �?(T,)，值域 [0, 1]
        threshold: 四舍五入阈值（默认0.5�?

    Returns:
        整数解，shape 与输入相同，值为 0 �?1
    """
    return (x_LP >= threshold).astype(int)


def _extract_unit_lambda(
    lambda_val,
    T: int,
    unit_id: Optional[int] = None,
    trainer: Optional['SubproblemSurrogateTrainer'] = None,
) -> np.ndarray:
    """Normalize lambda payloads to a single unit/time vector of shape (T,)."""
    if isinstance(lambda_val, dict):
        if 'lambda_pg_electricity_price' in lambda_val:
            arr = np.asarray(lambda_val['lambda_pg_electricity_price'], dtype=float)
        elif 'lambda_pg_effective' in lambda_val:
            arr = np.asarray(lambda_val['lambda_pg_effective'], dtype=float)
        else:
            arr = np.asarray(
                lambda_val.get('lambda_power_balance', np.zeros(T)),
                dtype=float,
            )
            lambda_du = np.asarray(
                lambda_val.get('lambda_dcpf_upper', np.zeros((0, T))),
                dtype=float,
            )
            lambda_dl = np.asarray(
                lambda_val.get('lambda_dcpf_lower', np.zeros((0, T))),
                dtype=float,
            )
            sensitivity = getattr(trainer, 'generator_injection_sensitivity', None)
            if (
                sensitivity is not None
                and arr.shape == (T,)
                and lambda_du.ndim == 2
                and lambda_dl.shape == lambda_du.shape
                and lambda_du.shape[1] == T
                and lambda_du.shape[0] == sensitivity.shape[0]
            ):
                arr = arr[np.newaxis, :] - sensitivity.T @ (lambda_du - lambda_dl)
    else:
        arr = np.asarray(lambda_val, dtype=float)

    if arr.ndim == 0:
        return np.full(T, float(arr), dtype=float)

    if arr.ndim == 1:
        if arr.size == T:
            return arr.astype(float, copy=True)
        if arr.size == 1:
            return np.full(T, float(arr[0]), dtype=float)
        if arr.size % T == 0:
            arr = arr.reshape(-1, T)
        else:
            raise ValueError(f"lambda_val has incompatible shape {arr.shape}, expected (*, {T})")

    if arr.ndim >= 2:
        if arr.shape[-1] != T:
            if arr.size % T != 0:
                raise ValueError(f"lambda_val has incompatible shape {arr.shape}, expected last dim {T}")
            arr = arr.reshape(-1, T)
        else:
            arr = arr.reshape(-1, T)

        idx = 0 if unit_id is None else int(unit_id)
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(
                f"unit_id={unit_id} out of range for lambda_val with shape {arr.shape}"
            )
        return arr[idx].astype(float, copy=True)

    raise ValueError(f"Unsupported lambda_val shape: {arr.shape}")


def _resolve_surrogate_sample_id(
    trainer: Optional['SubproblemSurrogateTrainer'],
    sample: Optional[dict],
) -> Optional[int]:
    """Resolve the dataset sample id used by sensitive surrogate constraints."""
    if trainer is None or not isinstance(sample, dict):
        return None

    candidate = sample.get('sample_id', sample.get('source_sample_id'))
    if candidate is None:
        return None

    try:
        sample_id = int(candidate)
    except (TypeError, ValueError):
        return None

    sensitive_timesteps = getattr(trainer, 'sensitive_timesteps', None)
    if not isinstance(sensitive_timesteps, list):
        return None
    if 0 <= sample_id < len(sensitive_timesteps):
        return sample_id
    return None


def _resolve_surrogate_constraint_timesteps(
    trainer: Optional['SubproblemSurrogateTrainer'],
    sample: Optional[dict],
    T: int,
    n_constraints: int,
) -> List[int]:
    """Map surrogate parameter indices to the intended 3-period windows."""
    if n_constraints <= 0:
        return []

    strategy = normalize_constraint_generation_strategy(
        getattr(trainer, 'constraint_generation_strategy', 'sensitive') or 'sensitive'
    )

    if strategy != 'sensitive':
        layout_timesteps, _ = select_constraint_layout(
            np.zeros(max(int(T), 0), dtype=float),
            strategy=strategy,
            max_constraints=n_constraints,
        )
        if len(layout_timesteps) < n_constraints:
            raise ValueError(
                f"Strategy {strategy} produced {len(layout_timesteps)} timesteps, "
                f"but {n_constraints} surrogate constraints were requested."
            )
        return list(layout_timesteps[:n_constraints])

    if T < 3:
        return []

    sample_id = _resolve_surrogate_sample_id(trainer, sample)
    if sample_id is None:
        raise ValueError(
            "Sensitive surrogate constraints require a resolvable sample_id/source_sample_id "
            "to preserve the training-time timestep mapping."
        )

    sensitive_timesteps = getattr(trainer, 'sensitive_timesteps', [])
    resolved = list(sensitive_timesteps[sample_id])
    if len(resolved) < n_constraints:
        raise ValueError(
            f"Sample {sample_id} only has {len(resolved)} sensitive timesteps, "
            f"but {n_constraints} surrogate constraints were requested."
        )
    return resolved[:n_constraints]


def _resolve_surrogate_constraint_layout(
    trainer: Optional['SubproblemSurrogateTrainer'],
    sample: Optional[dict],
    T: int,
    n_constraints: int,
) -> Tuple[List[int], List[tuple[int, ...]]]:
    timesteps = _resolve_surrogate_constraint_timesteps(trainer, sample, T, n_constraints)
    sample_id = _resolve_surrogate_sample_id(trainer, sample)
    try:
        offsets = resolve_constraint_offsets_from_trainer(trainer, sample_id, len(timesteps))
    except TypeError:
        offsets = resolve_constraint_offsets_from_trainer(trainer, sample_id)
    offsets = list(offsets)
    if len(offsets) < len(timesteps):
        default_offsets = (
            SURROGATE_SINGLE_TIME_OFFSETS
            if normalize_constraint_generation_strategy(
                getattr(trainer, 'constraint_generation_strategy', 'sensitive') or 'sensitive'
            ) == CONSTRAINT_STRATEGY_ALL_SINGLE_TIME
            else SURROGATE_TRIPLE_WINDOW_OFFSETS
        )
        offsets = offsets + [default_offsets] * (len(timesteps) - len(offsets))
    return timesteps, offsets[:len(timesteps)]


def _evaluate_surrogate_constraint_row(
    x_row: np.ndarray,
    timestep: int,
    offsets: tuple[int, ...],
    alpha: float,
    beta: float,
    gamma: float,
    rhs: float,
    horizon: int,
) -> dict:
    """Evaluate one surrogate row on a unit commitment vector."""
    x_arr = np.asarray(x_row, dtype=float).reshape(-1)
    lhs = float(
        build_surrogate_constraint_expression(
            x_arr,
            int(timestep),
            tuple(offsets),
            float(alpha),
            float(beta),
            float(gamma),
            int(horizon),
        )
    )
    violation = max(0.0, lhs - float(rhs))
    coef_scale = max(abs(float(alpha)) + abs(float(beta)) + abs(float(gamma)), 1.0)
    return {
        'lhs': lhs,
        'rhs': float(rhs),
        'violation': float(violation),
        'margin': float(rhs - lhs),
        'normalized_violation': float(violation / coef_scale),
        'normalized_margin': float((rhs - lhs) / coef_scale),
        'coef_scale': float(coef_scale),
    }


def _build_surrogate_screen_reference_rows(
    x_LP: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    """Collect diverse per-unit reference rows used to select stable surrogate rows."""
    x_lp_arr = np.asarray(x_LP, dtype=float)
    x_surr_arr = np.asarray(x_surr_lp, dtype=float)
    x_init_arr = np.asarray(x_init_k, dtype=int)
    ng = x_init_arr.shape[0]

    references: Dict[int, List[Tuple[str, np.ndarray]]] = {g: [] for g in range(ng)}
    for g in range(ng):
        references[g].extend([
            ('joint_lp', np.asarray(x_lp_arr[g], dtype=float)),
            ('joint_lp_round', np.asarray(round_to_integer(x_lp_arr[g]), dtype=int)),
            ('surrogate_lp', np.asarray(x_surr_arr[g], dtype=float)),
            ('surrogate_lp_round', np.asarray(round_to_integer(x_surr_arr[g]), dtype=int)),
            ('subproblem_round', np.asarray(x_init_arr[g], dtype=int)),
        ])

    x_init_k_m_arr = np.asarray(x_init_k_m)
    if x_init_k_m_arr.ndim == 3 and x_init_k_m_arr.shape[1] > 0:
        for m in range(x_init_k_m_arr.shape[1]):
            for g in range(ng):
                references[g].append(
                    (f'perturb_{m + 1}', np.asarray(x_init_k_m_arr[g, m, :], dtype=int))
                )

    if nearby_commitment_candidates:
        for idx, (_name, candidate) in enumerate(nearby_commitment_candidates, start=1):
            candidate_arr = np.asarray(candidate, dtype=int)
            if candidate_arr.ndim != 2 or candidate_arr.shape[0] != ng:
                continue
            for g in range(ng):
                references[g].append((f'nearby_{idx}', candidate_arr[g]))

    deduped: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    for g, rows in references.items():
        seen = set()
        kept: List[Tuple[str, np.ndarray]] = []
        for name, row in rows:
            row_arr = np.asarray(row, dtype=float).reshape(-1)
            key = tuple(np.round(row_arr, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            kept.append((str(name), row_arr))
        deduped[g] = kept
    return deduped


def _select_stable_surrogate_screen_constraints(
    scenario_input: np.ndarray | dict,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_val,
    x_LP: np.ndarray,
    x_surr_lp: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]] = None,
    max_constraints_per_unit: int = 3,
    min_support_ratio: float = 0.85,
    max_normalized_violation: float = 0.05,
    min_mean_margin: float = 0.02,
) -> List[dict]:
    """Pick surrogate rows that remain satisfied across diverse local references."""
    sample = _coerce_scenario_sample(scenario_input)
    pd_matrix = get_sample_net_load(sample)
    ng, T = np.asarray(x_init_k).shape
    reference_rows = _build_surrogate_screen_reference_rows(
        x_LP,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        nearby_commitment_candidates=nearby_commitment_candidates,
    )

    selected: List[dict] = []
    for g in range(ng):
        trainer = trainers.get(g)
        if trainer is None:
            continue

        lambda_unit = _extract_unit_lambda(
            lambda_val,
            T,
            unit_id=g,
            trainer=trainer,
        )
        alphas, betas, gammas, deltas, _costs, _pg_costs = trainer.get_surrogate_params(
            sample if isinstance(sample, dict) else pd_matrix,
            lambda_unit,
        )
        timestep_map, offset_map = _resolve_surrogate_constraint_layout(
            trainer,
            sample if isinstance(sample, dict) else None,
            T,
            len(alphas),
        )

        unit_candidates: List[dict] = []
        refs = reference_rows.get(g, [])
        if not refs:
            continue

        for k, t_k in enumerate(timestep_map):
            a = float(alphas[k])
            b = float(betas[k])
            c = float(gammas[k])
            rhs = float(deltas[k])
            if abs(a) <= 1e-10 and abs(b) <= 1e-10 and abs(c) <= 1e-10:
                continue

            metrics = [
                _evaluate_surrogate_constraint_row(
                    row,
                    t_k,
                    tuple(offset_map[k]),
                    a,
                    b,
                    c,
                    rhs,
                    T,
                )
                for _name, row in refs
            ]
            support_ratio = float(np.mean([
                metric['normalized_violation'] <= max_normalized_violation
                for metric in metrics
            ]))
            mean_margin = float(np.mean([metric['normalized_margin'] for metric in metrics]))
            min_margin = float(np.min([metric['normalized_margin'] for metric in metrics]))
            max_violation = float(np.max([metric['normalized_violation'] for metric in metrics]))
            if (
                support_ratio < float(min_support_ratio)
                or max_violation > float(max_normalized_violation)
                or mean_margin < float(min_mean_margin)
            ):
                continue

            score = float(
                2.5 * support_ratio
                + 0.8 * mean_margin
                + 0.4 * min_margin
                - 1.5 * max_violation
            )
            unit_candidates.append(
                {
                    'unit_id': int(g),
                    'constraint_index': int(k),
                    'timestep': int(t_k),
                    'offsets': tuple(offset_map[k]),
                    'alpha': a,
                    'beta': b,
                    'gamma': c,
                    'delta': rhs,
                    'coef_scale': float(max(abs(a) + abs(b) + abs(c), 1.0)),
                    'support_ratio': support_ratio,
                    'mean_normalized_margin': mean_margin,
                    'min_normalized_margin': min_margin,
                    'max_normalized_violation': max_violation,
                    'reference_count': int(len(metrics)),
                    'score': score,
                }
            )

        unit_candidates.sort(key=lambda row: row['score'], reverse=True)
        selected.extend(unit_candidates[: max(0, int(max_constraints_per_unit))])

    selected.sort(key=lambda row: (row['unit_id'], -row['score'], row['constraint_index']))
    return selected


def _filter_named_commitment_candidates_by_surrogate_screen(
    candidate_specs: List[Tuple[str, np.ndarray]],
    surrogate_screen_constraints: Optional[List[dict]],
    normalized_violation_tol: float = 0.02,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, str]]]:
    """Filter full commitment candidates by a selected stable surrogate-row subset."""
    if not surrogate_screen_constraints:
        return list(candidate_specs), []

    kept: List[Tuple[str, np.ndarray]] = []
    rejected: List[Tuple[str, str]] = []
    seen = set()
    for name, candidate in candidate_specs:
        candidate_arr = np.asarray(candidate, dtype=int)
        if candidate_arr.ndim != 2:
            rejected.append((str(name), 'candidate is not a 2D commitment matrix'))
            continue

        violations = []
        for row in surrogate_screen_constraints:
            g = int(row['unit_id'])
            if g < 0 or g >= candidate_arr.shape[0]:
                continue
            metric = _evaluate_surrogate_constraint_row(
                candidate_arr[g],
                int(row['timestep']),
                tuple(row['offsets']),
                float(row['alpha']),
                float(row['beta']),
                float(row['gamma']),
                float(row['delta']),
                candidate_arr.shape[1],
            )
            if metric['normalized_violation'] > float(normalized_violation_tol):
                violations.append(
                    f"G{g}/k{int(row['constraint_index'])}:nv={metric['normalized_violation']:.3f}"
                )

        if violations:
            rejected.append((str(name), ', '.join(violations[:3])))
            continue

        key = _candidate_key(candidate_arr)
        if key in seen:
            continue
        seen.add(key)
        kept.append((str(name), candidate_arr))

    return kept, rejected


def _finalize_recover_integer_solution_result(
    x_result: Optional[np.ndarray],
    success: bool,
    details: dict,
    return_details: bool,
):
    if return_details:
        return x_result, success, details
    return x_result, success


def _estimate_commitment_primal_objective(
    x_commitment: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
) -> float:
    """Estimate the original UC objective scale for adaptive FP tau selection."""
    ppc_int = ext2int(ppc)
    gen = np.asarray(ppc_int['gen'], dtype=float)
    gencost = np.asarray(ppc_int['gencost'], dtype=float)
    x_arr = np.asarray(x_commitment, dtype=float)
    ng, T = x_arr.shape
    load_sum = np.sum(np.asarray(pd_data, dtype=float), axis=0)

    pmin = gen[:, PMIN]
    pmax = gen[:, PMAX]
    linear_pg_cost = gencost[:, -2] / T_delta
    linear_x_cost = gencost[:, -1] / T_delta
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    total_cost = 0.0
    for t in range(T):
        x_t = np.clip(x_arr[:, t], 0.0, 1.0)
        online_mask = x_t > 1e-6
        if not np.any(online_mask):
            online_mask = np.ones(ng, dtype=bool)
            x_t = np.ones(ng, dtype=float)

        pg_guess = pmin * x_t
        residual = float(load_sum[t] - np.sum(pg_guess))
        headroom = np.maximum(pmax * x_t - pg_guess, 0.0)
        total_headroom = float(np.sum(headroom[online_mask]))
        if residual > 0.0 and total_headroom > 1e-9:
            pg_guess += residual * headroom / total_headroom
        pg_guess = np.clip(pg_guess, 0.0, pmax * x_t)

        supply = float(np.sum(pg_guess))
        if supply < float(load_sum[t]):
            deficit = float(load_sum[t] - supply)
            support = np.maximum(pmax - pg_guess, 0.0)
            support_sum = float(np.sum(support))
            if support_sum > 1e-9:
                pg_guess += deficit * support / support_sum
                pg_guess = np.clip(pg_guess, 0.0, pmax)

        total_cost += float(np.sum(linear_pg_cost * pg_guess + linear_x_cost * x_t))

    if T >= 2:
        x_prev = np.clip(x_arr[:, :-1], 0.0, 1.0)
        x_next = np.clip(x_arr[:, 1:], 0.0, 1.0)
        total_cost += float(np.sum(start_cost[:, None] * np.maximum(x_next - x_prev, 0.0)))
        total_cost += float(np.sum(shut_cost[:, None] * np.maximum(x_prev - x_next, 0.0)))

    return max(float(total_cost), 1.0)


def _repair_commitment_logic_heuristic(
    x_int: np.ndarray,
    T_delta: float,
    ppc: Optional[dict] = None,
    unit_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a light repair so a binary commitment better respects min up/down logic."""
    x_arr = np.asarray(x_int, dtype=int)
    was_vector = x_arr.ndim == 1
    if was_vector:
        x_arr = x_arr[np.newaxis, :]

    x_repaired = x_arr.copy()
    ng, T = x_repaired.shape
    if T <= 1:
        return x_repaired[0] if was_vector else x_repaired

    if ppc is not None:
        full_ng = ext2int(ppc)['gen'].shape[0]
        if unit_ids is None:
            if ng != full_ng:
                raise ValueError("unit_ids must be provided when repairing a subset of units")
            local_unit_ids = np.arange(ng, dtype=int)
        else:
            local_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
            if local_unit_ids.size != ng:
                raise ValueError("unit_ids length must match the number of repaired commitment rows")
        min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, full_ng, T_delta, T)
        ton_steps = min_up_steps[local_unit_ids]
        toff_steps = min_down_steps[local_unit_ids]
    else:
        default_steps = min(max(int(4 * T_delta), 1), max(T - 1, 0))
        ton_steps = np.full(ng, default_steps, dtype=int)
        toff_steps = np.full(ng, default_steps, dtype=int)

    for local_g in range(ng):
        Ton = int(ton_steps[local_g])
        Toff = int(toff_steps[local_g])
        changed = True
        while changed:
            changed = False

            for t in range(T - 1):
                if x_repaired[local_g, t] == 0 and x_repaired[local_g, t + 1] == 1:
                    end = min(T, t + 1 + Ton)
                    if np.any(x_repaired[local_g, t + 1:end] == 0):
                        x_repaired[local_g, t + 1:end] = 1
                        changed = True

            for t in range(T - 1):
                if x_repaired[local_g, t] == 1 and x_repaired[local_g, t + 1] == 0:
                    end = min(T, t + 1 + Toff)
                    if np.any(x_repaired[local_g, t + 1:end] == 1):
                        x_repaired[local_g, t + 1:end] = 0
                        changed = True

    return x_repaired[0] if was_vector else x_repaired


def _repair_min_up_down_heuristic(
    x_int: np.ndarray,
    T_delta: float,
    ppc: Optional[dict] = None,
    unit_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Repair commitment with per-unit min up/down when ppc is provided.

    When ``ppc`` is supplied, the actual per-unit minimum up/down times from
    ``ppc`` are used so the repair matches the single-unit on/off constraints
    that will later gate the candidate during validation. Without ``ppc``, the
    legacy default of ``4 * T_delta`` is used (kept only for backwards
    compatibility with callers that have no ppc context).
    """
    return _repair_commitment_logic_heuristic(
        x_int,
        T_delta,
        ppc=ppc,
        unit_ids=unit_ids,
    )


def _temporal_neighbor_average(x: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute a temporal neighborhood average along the time axis."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[np.newaxis, :]

    ng, T = x_arr.shape
    avg = np.zeros_like(x_arr, dtype=float)
    for t in range(T):
        lo = max(0, t - radius)
        hi = min(T, t + radius + 1)
        avg[:, t] = np.mean(x_arr[:, lo:hi], axis=1)
    return avg if x.ndim == 2 else avg[0]


def _build_directional_hot_start(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    trusted_mask: np.ndarray
) -> np.ndarray:
    """Build a directional hot start by comparing surrogate LP and global LP."""
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)

    direction = x_surr_LP - x_LP
    lp_neighbor = _temporal_neighbor_average(x_LP, radius=1)
    surr_neighbor = _temporal_neighbor_average(x_surr_LP, radius=1)
    round_neighbor = 0.5 * _temporal_neighbor_average(lp_round, radius=1) \
                     + 0.5 * _temporal_neighbor_average(surr_round, radius=1)

    # surrogate 比全局 LP 更强调局部结构，因此给更高权重；
    # direction 与邻域平均共同决定是否做“反四舍五入”�?
    directional_score = (
        0.35 * x_LP
        + 0.65 * x_surr_LP
        + 2.20 * direction
        + 0.60 * (0.5 * lp_neighbor + 0.5 * surr_neighbor - 0.5)
        + 0.35 * (round_neighbor - 0.5)
    )

    x_directional = (directional_score >= 0.5).astype(int)

    upward_mask = (
        (~trusted_mask)
        & (direction >= 0.05)
        & ((surr_neighbor >= 0.55) | (round_neighbor >= 0.75))
    )
    downward_mask = (
        (~trusted_mask)
        & (direction <= -0.05)
        & ((surr_neighbor <= 0.45) | (round_neighbor <= 0.25))
    )

    x_directional[upward_mask] = 1
    x_directional[downward_mask] = 0
    x_directional[trusted_mask] = lp_round[trusted_mask]
    return x_directional


def _neighbor_average_1d(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute a 1D neighborhood average for surrogate parameters."""
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.mean(arr[lo:hi])
    return out


def _perturb_surrogate_outputs(
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    rng: np.random.Generator,
    perturb_std: float = 0.10,
    neighborhood_weight: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perturb surrogate outputs and blend them with neighborhood averages."""
    def _perturb_one(arr: np.ndarray, clip_nonnegative: bool = False) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        scale = np.maximum(np.abs(arr), 1.0)
        noisy = arr + perturb_std * scale * rng.standard_normal(arr.shape)
        neighbor_avg = _neighbor_average_1d(noisy, radius=1)
        mixed = (1.0 - neighborhood_weight) * noisy + neighborhood_weight * neighbor_avg
        if clip_nonnegative:
            mixed = np.maximum(mixed, 0.0)
        return mixed

    return (
        _perturb_one(alphas, clip_nonnegative=False),
        _perturb_one(betas, clip_nonnegative=False),
        _perturb_one(gammas, clip_nonnegative=False),
        _perturb_one(deltas, clip_nonnegative=True),
    )


def _coerce_scenario_sample(pd_data: np.ndarray | dict) -> dict:
    """Convert array/dict input into a normalized scenario sample dict."""
    if isinstance(pd_data, dict):
        return normalize_sample_arrays(dict(pd_data))
    pd_matrix = np.asarray(pd_data, dtype=float)
    return normalize_sample_arrays({'pd_data': pd_matrix})


def _predict_lambda_for_scenario(
    lambda_predictor,
    scenario_sample: dict,
    fallback_lambda,
):
    """Predict lambda for a scenario when possible, otherwise reuse the current lambda."""
    if lambda_predictor is None:
        return fallback_lambda
    try:
        lambda_pred = lambda_predictor.predict(scenario_sample)
        if isinstance(lambda_pred, dict):
            return lambda_pred
        lambda_pred = np.asarray(lambda_pred, dtype=float)
        fallback_arr = np.asarray(fallback_lambda, dtype=float)
        if lambda_pred.shape == fallback_arr.shape:
            return lambda_pred
    except Exception:
        pass
    return fallback_lambda


def _get_scenario_bank(
    trainers: Dict[int, 'SubproblemSurrogateTrainer']
) -> List[dict]:
    """Reuse the trainer dataset as a retrieval bank for scenario-based perturbations."""
    for trainer in trainers.values():
        active_set_data = getattr(trainer, 'active_set_data', None)
        if isinstance(active_set_data, list) and active_set_data:
            return [
                normalize_sample_arrays(dict(sample))
                for sample in active_set_data
                if isinstance(sample, dict)
            ]
    return []


def _feature_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Scaled L2 distance to reduce the effect of feature magnitude."""
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return float(np.linalg.norm((lhs - rhs) / scale))


def _find_similar_scenarios(
    target_sample: dict,
    scenario_bank: List[dict],
    n_candidates: int,
    top_k: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Retrieve similar historical scenarios based on load/renewable NN features."""
    if n_candidates <= 0 or not scenario_bank:
        return []

    target_features = get_feature_vector_from_sample(dict(target_sample))
    scored_candidates: List[Tuple[float, dict]] = []
    for sample in scenario_bank:
        try:
            feature_vec = get_feature_vector_from_sample(dict(sample))
        except Exception:
            continue
        if feature_vec.shape != target_features.shape:
            continue
        distance = _feature_distance(target_features, feature_vec)
        if distance <= 1e-12:
            continue
        scored_candidates.append((distance, sample))

    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda item: item[0])
    shortlist = [sample for _, sample in scored_candidates[:max(1, top_k)]]
    if len(shortlist) <= n_candidates:
        return [normalize_sample_arrays(dict(sample)) for sample in shortlist]

    chosen_idx = rng.choice(len(shortlist), size=n_candidates, replace=False)
    return [normalize_sample_arrays(dict(shortlist[idx])) for idx in chosen_idx]


def _extract_commitment_from_sample(
    sample: dict,
    ng: int,
    T: int,
) -> Optional[np.ndarray]:
    """Extract a binary commitment matrix from a historical sample when available."""
    if not isinstance(sample, dict):
        return None

    if 'x_true' in sample and sample['x_true'] is not None:
        x_true = np.asarray(sample['x_true'], dtype=float)
        if x_true.ndim == 2:
            x_sol = np.zeros((ng, T), dtype=float)
            rows = min(ng, x_true.shape[0])
            cols = min(T, x_true.shape[1])
            x_sol[:rows, :cols] = x_true[:rows, :cols]
            return round_to_integer(np.clip(x_sol, 0.0, 1.0))

    if 'unit_commitment_matrix' in sample and sample['unit_commitment_matrix'] is not None:
        uc = np.asarray(sample['unit_commitment_matrix'], dtype=float)
        if uc.ndim == 2:
            x_sol = np.zeros((ng, T), dtype=float)
            rows = min(ng, uc.shape[0])
            cols = min(T, uc.shape[1])
            x_sol[:rows, :cols] = uc[:rows, :cols]
            return round_to_integer(np.clip(x_sol, 0.0, 1.0))

    active_set = sample.get('active_set', None)
    if not isinstance(active_set, list):
        return None

    x_sol = np.zeros((ng, T), dtype=float)
    found_commitment = False
    for item in active_set:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list) and len(item[0]) == 2:
            g, t = item[0]
            value = item[1]
        elif isinstance(item, dict):
            g = item.get('unit_id', item.get('generator_id', item.get('g')))
            t = item.get('time_slot', item.get('time', item.get('t')))
            value = item.get('value', item.get('x'))
        else:
            continue

        if g is None or t is None or value is None:
            continue
        g = int(g)
        t = int(t)
        if 0 <= g < ng and 0 <= t < T:
            x_sol[g, t] = float(value)
            found_commitment = True

    if not found_commitment:
        return None
    return round_to_integer(np.clip(x_sol, 0.0, 1.0))


def _build_nearby_commitment_candidates(
    target_sample: dict,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    scenario_bank: Optional[List[dict]],
    ng: int,
    T: int,
    n_candidates: int,
    candidate_pool_size: int,
    rng: np.random.Generator,
) -> List[Tuple[str, np.ndarray]]:
    """Build hot starts from nearby scenarios' historical optimal commitments."""
    if n_candidates <= 0:
        return []

    if scenario_bank is None:
        scenario_bank = _get_scenario_bank(trainers)
    else:
        scenario_bank = [
            normalize_sample_arrays(dict(sample))
            for sample in scenario_bank
            if isinstance(sample, dict)
        ]
    if not scenario_bank:
        return []

    target_features = get_feature_vector_from_sample(dict(target_sample))
    scored_candidates: List[Tuple[float, dict]] = []
    for sample in scenario_bank:
        try:
            feature_vec = get_feature_vector_from_sample(dict(sample))
        except Exception:
            continue
        if feature_vec.shape != target_features.shape:
            continue
        distance = _feature_distance(target_features, feature_vec)
        if distance <= 1e-12:
            continue
        scored_candidates.append((distance, sample))

    if not scored_candidates:
        return []

    scored_candidates.sort(key=lambda item: item[0])
    preferred_pool = scored_candidates[:max(candidate_pool_size, n_candidates)]
    fallback_pool = scored_candidates[max(candidate_pool_size, n_candidates):]

    nearby_candidates: List[Tuple[str, np.ndarray]] = []
    seen = set()
    ordered_pool = preferred_pool + fallback_pool
    for _distance, scenario in ordered_pool:
        commitment = _extract_commitment_from_sample(scenario, ng, T)
        if commitment is None:
            continue
        key = _candidate_key(commitment)
        if key in seen:
            continue
        seen.add(key)
        nearby_candidates.append(
            (f"nearby_opt_commitment_{len(nearby_candidates) + 1}", commitment)
        )
        if len(nearby_candidates) >= n_candidates:
            break
    return nearby_candidates


def _generate_load_perturbed_scenarios(
    base_sample: dict,
    n_candidates: int,
    perturb_scale: float,
    rng: np.random.Generator,
) -> List[dict]:
    """Generate nearby scenarios by mildly perturbing load while preserving renewables."""
    if n_candidates <= 0:
        return []

    base_sample = normalize_sample_arrays(dict(base_sample))
    load_data = np.asarray(get_sample_load_data(base_sample), dtype=float)
    renewable_data = np.asarray(get_sample_renewable_data(base_sample), dtype=float)
    candidates: List[dict] = []

    for _ in range(n_candidates):
        global_scale = 1.0 + 0.35 * perturb_scale * rng.standard_normal()
        time_scale = 1.0 + 0.50 * perturb_scale * rng.standard_normal((1, load_data.shape[1]))
        bus_scale = 1.0 + 0.50 * perturb_scale * rng.standard_normal(load_data.shape)
        perturbed_load = load_data * global_scale * time_scale * bus_scale
        perturbed_load = np.maximum(perturbed_load, 0.0)

        candidate = {
            'load_data': perturbed_load,
            'renewable_data': renewable_data.copy(),
        }
        if 'sample_id' in base_sample:
            candidate['source_sample_id'] = base_sample['sample_id']
        candidates.append(normalize_sample_arrays(candidate))

    return candidates


def _build_surrogate_parameter_candidates(
    base_sample: dict,
    base_lambda: np.ndarray,
    trainer: 'SubproblemSurrogateTrainer',
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    rng: np.random.Generator,
    lambda_predictor=None,
    n_param_perturbations: int = 5,
    perturb_std: float = 0.10,
    neighborhood_weight: float = 0.35,
    n_similar_scenarios: int = 0,
    similar_scenario_pool_size: int = 10,
    n_load_perturbations: int = 0,
    load_perturbation_scale: float = 0.03,
) -> List[Tuple[str, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build multiple surrogate parameter sets from direct/randomized and scenario-based strategies."""
    parameter_sets: List[
        Tuple[str, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    base_lambda_unit = _extract_unit_lambda(
        base_lambda,
        trainer.T,
        unit_id=trainer.unit_id,
        trainer=trainer,
    )
    alphas, betas, gammas, deltas, costs, pg_costs = trainer.get_surrogate_params(
        base_sample,
        base_lambda_unit,
    )
    parameter_sets.append(("base", base_sample, alphas, betas, gammas, deltas, costs, pg_costs))

    for idx in range(n_param_perturbations):
        alphas_m, betas_m, gammas_m, deltas_m = _perturb_surrogate_outputs(
            alphas,
            betas,
            gammas,
            deltas,
            rng=rng,
            perturb_std=perturb_std,
            neighborhood_weight=neighborhood_weight,
        )
        parameter_sets.append(
            (f"direct_randomized_{idx}", base_sample, alphas_m, betas_m, gammas_m, deltas_m, costs, pg_costs)
        )

    scenario_bank = _get_scenario_bank(trainers)
    similar_scenarios = _find_similar_scenarios(
        target_sample=base_sample,
        scenario_bank=scenario_bank,
        n_candidates=n_similar_scenarios,
        top_k=similar_scenario_pool_size,
        rng=rng,
    )
    for idx, scenario in enumerate(similar_scenarios):
        lambda_sim = _predict_lambda_for_scenario(lambda_predictor, scenario, base_lambda)
        lambda_sim_unit = _extract_unit_lambda(
            lambda_sim,
            trainer.T,
            unit_id=trainer.unit_id,
            trainer=trainer,
        )
        alpha_s, beta_s, gamma_s, delta_s, costs_s, pg_costs_s = trainer.get_surrogate_params(
            scenario,
            lambda_sim_unit,
        )
        parameter_sets.append(
            (f"similar_scenario_{idx}", scenario, alpha_s, beta_s, gamma_s, delta_s, costs_s, pg_costs_s)
        )

    perturbed_scenarios = _generate_load_perturbed_scenarios(
        base_sample=base_sample,
        n_candidates=n_load_perturbations,
        perturb_scale=load_perturbation_scale,
        rng=rng,
    )
    for idx, scenario in enumerate(perturbed_scenarios):
        lambda_pert = _predict_lambda_for_scenario(lambda_predictor, scenario, base_lambda)
        lambda_pert_unit = _extract_unit_lambda(
            lambda_pert,
            trainer.T,
            unit_id=trainer.unit_id,
            trainer=trainer,
        )
        alpha_p, beta_p, gamma_p, delta_p, costs_p, pg_costs_p = trainer.get_surrogate_params(
            scenario,
            lambda_pert_unit,
        )
        parameter_sets.append(
            (f"load_perturbed_{idx}", scenario, alpha_p, beta_p, gamma_p, delta_p, costs_p, pg_costs_p)
        )

    return parameter_sets


def _select_pool_restart_candidate(
    x_reference: np.ndarray,
    x_pool: Optional[np.ndarray],
    trusted_mask: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Prefer a pool candidate that differs on free variables when FP stalls."""
    if x_pool is None or len(x_pool) == 0:
        return None

    free_mask = ~trusted_mask
    if not np.any(free_mask):
        return None

    scored: List[Tuple[float, np.ndarray]] = []
    for candidate in x_pool:
        candidate_int = np.asarray(candidate, dtype=int)
        distance = float(np.sum(candidate_int[free_mask] != x_reference[free_mask]))
        if distance > 0:
            scored.append((distance, candidate_int))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_count = min(3, len(scored))
    chosen = scored[int(rng.integers(0, top_count))][1].copy()
    chosen[trusted_mask] = x_reference[trusted_mask]
    return chosen


def _candidate_key(x_candidate: np.ndarray) -> tuple:
    """Return a hashable key for binary-candidate deduplication."""
    return tuple(np.asarray(x_candidate, dtype=int).flatten().tolist())


def _compute_vote_majority(
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray
) -> np.ndarray:
    """Aggregate surrogate-derived integer candidates into a majority reference."""
    vote_sum = x_init_k.astype(float)
    n_votes = 1
    if x_init_k_m.size > 0:
        vote_sum = vote_sum + np.sum(x_init_k_m.astype(float), axis=1)
        n_votes += x_init_k_m.shape[1]
    return (vote_sum / max(n_votes, 1) >= 0.5).astype(int)


def _compute_hot_start_support_reference(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    nearby_commitment_pool: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a soft support map from LP, surrogate, unit LP, perturbation, and nearby pools."""
    support = (
        0.45 * np.asarray(x_LP, dtype=float)
        + 0.35 * np.asarray(x_surr_LP, dtype=float)
        + 0.55 * np.asarray(x_init_k, dtype=float)
    )
    total_weight = 1.35

    if x_init_k_m.size > 0:
        support += 0.25 * np.mean(np.asarray(x_init_k_m, dtype=float), axis=1)
        total_weight += 0.25

    if nearby_commitment_pool is not None and nearby_commitment_pool.size > 0:
        support += 0.18 * np.mean(np.asarray(nearby_commitment_pool, dtype=float), axis=0)
        total_weight += 0.18

    return np.clip(support / max(total_weight, 1e-8), 0.0, 1.0)


def _score_hot_start_candidate(
    x_candidate: np.ndarray,
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    vote_reference: np.ndarray,
    trusted_mask: np.ndarray,
    support_reference: Optional[np.ndarray] = None,
    nearby_commitment_pool: Optional[np.ndarray] = None,
) -> float:
    """Score a binary hot start against LP, surrogate LP, and vote references."""
    x_candidate = np.asarray(x_candidate, dtype=int)
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)

    trusted_match = float(np.sum(x_candidate[trusted_mask] == lp_round[trusted_mask]))
    vote_match = float(np.sum(x_candidate == vote_reference))
    surrogate_match = float(np.sum(x_candidate == surr_round))
    lp_dist = float(np.sum(np.abs(x_candidate - x_LP)))
    surr_dist = float(np.sum(np.abs(x_candidate - x_surr_LP)))
    support_match = 0.0
    nearby_agreement = 0.0

    if support_reference is not None:
        support_reference = np.clip(np.asarray(support_reference, dtype=float), 0.0, 1.0)
        support_match = float(np.sum(
            x_candidate * support_reference
            + (1 - x_candidate) * (1.0 - support_reference)
        ))

    if nearby_commitment_pool is not None and nearby_commitment_pool.size > 0:
        nearby_commitment_pool = np.asarray(nearby_commitment_pool, dtype=int)
        nearby_agreement = float(np.mean(np.sum(
            nearby_commitment_pool == x_candidate,
            axis=tuple(range(1, nearby_commitment_pool.ndim)),
        )))

    return (
        3.0 * trusted_match
        + 1.8 * vote_match
        + 0.6 * surrogate_match
        + 0.35 * support_match
        + 0.08 * nearby_agreement
        - 0.45 * lp_dist
        - 0.30 * surr_dist
    )


def _build_unit_combination_hot_starts(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    T_delta: float,
    support_reference: Optional[np.ndarray] = None,
    nearby_commitment_pool: Optional[np.ndarray] = None,
    max_unit_options_per_generator: int = 4,
    max_combination_candidates: int = 12,
    ppc: Optional[dict] = None,
    unit_ids: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Assemble mixed global starts from top per-unit candidates using beam search.

    When ``ppc`` is supplied, every per-unit option is first repaired and then
    validated against the unit-specific minimum up/down constraints. Options
    that still violate the single-unit logic after repair are discarded before
    they enter the beam search so the resulting full-matrix candidates already
    satisfy each unit's on/off constraints by construction.
    """
    ng, _T = x_LP.shape
    if ng == 0:
        return []

    if ppc is not None:
        if unit_ids is None:
            local_unit_ids = np.arange(ng, dtype=int)
        else:
            local_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
            if local_unit_ids.size != ng:
                raise ValueError(
                    "unit_ids length must match the commitment rows passed to "
                    "_build_unit_combination_hot_starts"
                )
    else:
        local_unit_ids = np.arange(ng, dtype=int)

    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)
    directional = _build_directional_hot_start(x_LP, x_surr_LP, trusted_mask)
    vote_majority = _compute_vote_majority(x_init_k, x_init_k_m)
    n_perturb = x_init_k_m.shape[1] if x_init_k_m.ndim >= 3 else 0

    unit_option_lists: List[List[Tuple[str, np.ndarray, float]]] = []
    for g in range(ng):
        option_specs: List[Tuple[str, np.ndarray]] = [
            ("lp", lp_round[g]),
            ("surr", surr_round[g]),
            ("directional", directional[g]),
            ("unit_lp", x_init_k[g]),
            ("vote", vote_majority[g]),
        ]
        if nearby_commitment_pool is not None and nearby_commitment_pool.size > 0:
            for m in range(nearby_commitment_pool.shape[0]):
                option_specs.append((f"nearby{m + 1}", nearby_commitment_pool[m, g]))
        for m in range(n_perturb):
            option_specs.append((f"pert{m + 1}", x_init_k_m[g, m]))

        unit_id_for_repair = int(local_unit_ids[g]) if ppc is not None else None
        unique_options: List[Tuple[str, np.ndarray, float]] = []
        seen = set()
        for name, row in option_specs:
            row_int = np.asarray(row, dtype=int).reshape(-1)

            # Step A: per-unit repair using the real min up/down from ppc when
            # available; falls back to the legacy default otherwise.
            if ppc is not None:
                repaired_row = _repair_commitment_logic_heuristic(
                    row_int[np.newaxis, :],
                    T_delta,
                    ppc=ppc,
                    unit_ids=np.array([unit_id_for_repair], dtype=int),
                )[0]
                # Step B: hard-check the per-unit on/off constraints. Skip
                # options that still violate single-unit logic after repair so
                # the beam never produces a structurally infeasible matrix.
                is_valid, _reason = check_commitment_logic_feasibility(
                    repaired_row[np.newaxis, :],
                    ppc,
                    T_delta,
                    unit_ids=np.array([unit_id_for_repair], dtype=int),
                )
                if not is_valid:
                    continue
                row_int = np.asarray(repaired_row, dtype=int)

            key = tuple(row_int.tolist())
            if key in seen:
                continue
            seen.add(key)
            score = _score_hot_start_candidate(
                row_int,
                x_LP[g],
                x_surr_LP[g],
                vote_majority[g],
                trusted_mask[g],
                support_reference=None if support_reference is None else support_reference[g],
                nearby_commitment_pool=(
                    None if nearby_commitment_pool is None or nearby_commitment_pool.size == 0
                    else nearby_commitment_pool[:, g, :]
                ),
            )
            unique_options.append((name, row_int, score))

        # Always keep at least one option per unit so beam search can proceed.
        if not unique_options:
            fallback_row = np.asarray(lp_round[g], dtype=int)
            if ppc is not None:
                fallback_row = _repair_commitment_logic_heuristic(
                    fallback_row[np.newaxis, :],
                    T_delta,
                    ppc=ppc,
                    unit_ids=np.array([unit_id_for_repair], dtype=int),
                )[0].astype(int)
            unique_options.append(("lp_fallback", fallback_row, 0.0))

        unique_options.sort(key=lambda item: item[2], reverse=True)
        unit_option_lists.append(unique_options[:max(1, max_unit_options_per_generator)])

    beam: List[Tuple[float, List[np.ndarray]]] = [(0.0, [])]
    beam_width = max(1, max_combination_candidates)
    for g in range(ng):
        next_beam: List[Tuple[float, List[np.ndarray]]] = []
        for partial_score, partial_rows in beam:
            for _name, row, row_score in unit_option_lists[g]:
                next_beam.append((partial_score + row_score, partial_rows + [row]))
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_width]

    mixed_candidates: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for idx, (_score, rows) in enumerate(beam, start=1):
        candidate = np.stack(rows, axis=0).astype(int, copy=False)
        repaired = _repair_min_up_down_heuristic(
            candidate, T_delta, ppc=ppc, unit_ids=unit_ids
        )
        if ppc is not None:
            is_valid, _reason = check_commitment_logic_feasibility(
                repaired, ppc, T_delta, unit_ids=unit_ids
            )
            if not is_valid:
                continue
        key = _candidate_key(repaired)
        if key in seen:
            continue
        seen.add(key)
        mixed_candidates.append((f"unit_combo_{idx}", repaired))

    return mixed_candidates


def _build_hot_start_candidates(
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    T_delta: float,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]] = None,
    max_perturbation_hot_starts: int = 6,
    max_unit_options_per_generator: int = 4,
    max_unit_combination_candidates: int = 12,
    ppc: Optional[dict] = None,
    unit_ids: Optional[np.ndarray] = None,
    lean_hot_starts: bool = False,
) -> List[Tuple[str, np.ndarray]]:
    """Build multiple global hot-start candidates from LP and surrogate references.

    Args:
        lean_hot_starts: When True, drops the redundant
            ``lp_surrogate_confidence_mix`` / ``lp_surrogate_agreement`` /
            ``trusted_vote_mix`` / ``trusted_confidence_mix`` blends that
            almost always dedupe back onto ``lp_round`` (per profiling on
            case14, these 4 mix entries contributed 0 unique bytes in 80 %
            of samples). Combined with smaller perturbation/combination caps
            this gives a measurable per-iter speed-up.
    """
    lp_round = round_to_integer(x_LP)
    surr_round = round_to_integer(x_surr_LP)
    directional = _build_directional_hot_start(x_LP, x_surr_LP, trusted_mask)

    lp_margin = np.abs(x_LP - 0.5)
    surr_margin = np.abs(x_surr_LP - 0.5)
    choose_lp = lp_margin >= surr_margin

    blended = np.where(choose_lp, lp_round, surr_round)
    agreement = np.where(lp_round == surr_round, lp_round, blended)

    vote_sum = (
        lp_round.astype(float)
        + surr_round.astype(float)
        + x_init_k.astype(float)
        + np.sum(x_init_k_m.astype(float), axis=1)
    )
    n_votes = 3 + x_init_k_m.shape[1]
    majority = (vote_sum / n_votes >= 0.5).astype(int)

    trusted_ref = np.where(trusted_mask, lp_round, majority)
    confidence_mix = np.where(trusted_mask, lp_round, agreement)

    candidate_specs = [
        ("lp_round", lp_round),
        ("surrogate_lp_round", surr_round),
        ("unit_lp_round", x_init_k),
        ("directional_surrogate_start", directional),
    ]
    nearby_commitment_pool = None
    if nearby_commitment_candidates:
        candidate_specs.extend(list(nearby_commitment_candidates))
        nearby_commitment_pool = np.stack(
            [np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates],
            axis=0,
        )
    if lean_hot_starts:
        # Only keep the consensus vote_majority -- the 4 mix blends dedupe
        # against lp_round/surrogate_lp_round in the vast majority of cases.
        candidate_specs.append(("vote_majority", majority))
    else:
        candidate_specs.extend([
            ("lp_surrogate_confidence_mix", blended),
            ("lp_surrogate_agreement", agreement),
            ("vote_majority", majority),
            ("trusted_vote_mix", trusted_ref),
            ("trusted_confidence_mix", confidence_mix),
        ])
    support_reference = _compute_hot_start_support_reference(
        x_LP,
        x_surr_LP,
        x_init_k,
        x_init_k_m,
        nearby_commitment_pool=nearby_commitment_pool,
    )
    if x_init_k_m.ndim >= 3 and x_init_k_m.shape[1] > 0:
        vote_reference = _compute_vote_majority(x_init_k, x_init_k_m)
        perturbation_specs: List[Tuple[str, np.ndarray, float]] = []
        for m in range(x_init_k_m.shape[1]):
            candidate = x_init_k_m[:, m, :]
            score = _score_hot_start_candidate(
                candidate,
                x_LP,
                x_surr_LP,
                vote_reference,
                trusted_mask,
                support_reference=support_reference,
                nearby_commitment_pool=nearby_commitment_pool,
            )
            perturbation_specs.append((f"perturbed_unit_pool_{m + 1}", candidate, score))
        perturbation_specs.sort(key=lambda item: item[2], reverse=True)
        candidate_specs.extend(
            (name, candidate)
            for name, candidate, _score in perturbation_specs[:max(0, max_perturbation_hot_starts)]
        )

    candidate_specs.extend(
        _build_unit_combination_hot_starts(
            x_LP,
            x_surr_LP,
            x_init_k,
            x_init_k_m,
            trusted_mask,
            T_delta,
            support_reference=support_reference,
            nearby_commitment_pool=nearby_commitment_pool,
            max_unit_options_per_generator=max_unit_options_per_generator,
            max_combination_candidates=max_unit_combination_candidates,
            ppc=ppc,
            unit_ids=unit_ids,
        )
    )

    unique_candidates: List[Tuple[str, np.ndarray]] = []
    key_to_index: Dict[tuple, int] = {}
    for name, x_candidate in candidate_specs:
        repaired = _repair_min_up_down_heuristic(
            x_candidate, T_delta, ppc=ppc, unit_ids=unit_ids
        )
        if ppc is not None:
            is_valid, _reason = check_commitment_logic_feasibility(
                repaired, ppc, T_delta, unit_ids=unit_ids
            )
            if not is_valid:
                # Drop candidates that still violate per-unit min up/down
                # logic after the ppc-aware repair. The downstream sanitization
                # would have rejected them anyway, but pre-filtering here
                # ensures every entry that lands in the pool already respects
                # the single-unit on/off constraints.
                continue
        key = _candidate_key(repaired)
        if key not in key_to_index:
            key_to_index[key] = len(unique_candidates)
            unique_candidates.append((name, repaired))
            continue

        existing_idx = key_to_index[key]
        existing_name, _existing_candidate = unique_candidates[existing_idx]
        if name.startswith("nearby_opt_commitment_") and not existing_name.startswith("nearby_opt_commitment_"):
            unique_candidates[existing_idx] = (name, repaired)

    return unique_candidates


def _rank_hot_start_candidates(
    candidates: List[Tuple[str, np.ndarray]],
    x_LP: np.ndarray,
    x_surr_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    trusted_mask: np.ndarray,
    nearby_commitment_candidates: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> List[Tuple[str, np.ndarray, float]]:
    """Rank hot starts by agreement with LP, surrogate LP, and vote references."""
    vote_majority = _compute_vote_majority(x_init_k, x_init_k_m)
    nearby_commitment_pool = None
    if nearby_commitment_candidates:
        nearby_commitment_pool = np.stack(
            [np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates],
            axis=0,
        )
    support_reference = _compute_hot_start_support_reference(
        x_LP,
        x_surr_LP,
        x_init_k,
        x_init_k_m,
        nearby_commitment_pool=nearby_commitment_pool,
    )

    ranked: List[Tuple[str, np.ndarray, float]] = []
    for name, x_candidate in candidates:
        score = _score_hot_start_candidate(
            x_candidate,
            x_LP,
            x_surr_LP,
            vote_majority,
            trusted_mask,
            support_reference=support_reference,
            nearby_commitment_pool=nearby_commitment_pool,
        )
        ranked.append((name, x_candidate, score))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


# ========================== 内部辅助：代理约束添�?==========================

def _run_fp_hot_start_task(
    task_idx: int,
    name: str,
    x_start: np.ndarray,
    trusted_mask: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    x_pool: Optional[np.ndarray],
    surrogate_screen_constraints: Optional[List[dict]],
    surrogate_screen_soft_penalty: float,
    projection_objective_tau,
    max_iter: int,
    stall_perturbation_mode: str,
    stall_flip_fraction: float,
    rng_seed: int,
    rounding_strategy: str = 'x_round',
    chi_alpha: float = 3.0,
    chi_random_samples: int = 8,
    chi_random_evaluator_weight: float = 0.05,
    stall_theta_resample_callback: Optional[Any] = None,
    stall_theta_resample_after_chi_random: int = 2,
    surrogate_engagement_iter: int = 0,
    noh_milp_refresh_callback: Optional[Any] = None,
    noh_milp_refresh_stall: int = 0,
    noh_milp_refresh_interval: int = 0,
    noh_milp_refresh_max: int = 0,
    enable_pool_tabu_prune: bool = False,
    pool_tabu_drop_threshold: int = 3,
) -> Tuple[int, str, np.ndarray, bool, Dict[str, Any]]:
    """Run one FP task with an isolated RNG for optional parallel execution."""
    task_rng = np.random.default_rng(int(rng_seed))
    x_result, success, fp_details = run_feasibility_pump(
        x_start,
        trusted_mask,
        ppc,
        pd_data,
        T_delta,
        x_pool=x_pool,
        surrogate_screen_constraints=surrogate_screen_constraints,
        surrogate_screen_soft_penalty=surrogate_screen_soft_penalty,
        projection_objective_tau=projection_objective_tau,
        max_iter=max_iter,
        stall_perturbation_mode=stall_perturbation_mode,
        stall_flip_fraction=stall_flip_fraction,
        rng=task_rng,
        verbose=False,
        return_history=True,
        rounding_strategy=rounding_strategy,
        chi_alpha=chi_alpha,
        chi_random_samples=chi_random_samples,
        chi_random_evaluator_weight=chi_random_evaluator_weight,
        stall_theta_resample_callback=stall_theta_resample_callback,
        stall_theta_resample_after_chi_random=stall_theta_resample_after_chi_random,
        surrogate_engagement_iter=surrogate_engagement_iter,
        noh_milp_refresh_callback=noh_milp_refresh_callback,
        noh_milp_refresh_stall=noh_milp_refresh_stall,
        noh_milp_refresh_interval=noh_milp_refresh_interval,
        noh_milp_refresh_max=noh_milp_refresh_max,
        enable_pool_tabu_prune=enable_pool_tabu_prune,
        pool_tabu_drop_threshold=pool_tabu_drop_threshold,
    )
    return task_idx, name, x_result, success, fp_details


def _add_surrogate_constraints(
    model: gp.Model,
    x_vars: dict,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    T: int,
    prefix: str = '',
    trainer: Optional['SubproblemSurrogateTrainer'] = None,
    sample: Optional[dict] = None,
    surrogate_constraint_scope: str = "all",
) -> None:
    """
    �?Gurobi 模型添加 V3 三时段代理约束�?

    约束形式：alphas[k]*x[t_k] + betas[k]*x[t_k+1] + gammas[k]*x[t_k+2] <= deltas[k]
    时段映射：t_k = k % (T-2)（循环分配，确保 t_k+2 不越界）

    Args:
        model: Gurobi 模型
        x_vars: {t: Gurobi Var} 时段到变量的映射
        alphas: (max_constraints,) 第一时段系数
        betas: (max_constraints,) 第二时段系数
        gammas: (max_constraints,) 第三时段系数
        deltas: (max_constraints,) 右端�?
        T: 时段总数
        prefix: 约束命名前缀（用于区分不同机组）
    """
    timestep_map, offset_map = _resolve_surrogate_constraint_layout(
        trainer,
        sample,
        T,
        len(alphas),
    )
    scope_norm = _normalize_surrogate_constraint_scope(surrogate_constraint_scope)
    for k, t_k in enumerate(timestep_map):
        if not _allow_surrogate_constraint_by_scope(trainer, k, scope_norm):
            continue
        a, b, c, r = float(alphas[k]), float(betas[k]), float(gammas[k]), float(deltas[k])
        if abs(a) > 1e-10 or abs(b) > 1e-10 or abs(c) > 1e-10:
            model.addConstr(
                build_surrogate_constraint_expression(
                    x_vars,
                    t_k,
                    offset_map[k],
                    a,
                    b,
                    c,
                    T,
                ) <= r,
                name=f'{prefix}surr_{k}'
            )


# ========================== 内部辅助：DC 潮流数据预计�?==========================

def _build_ptdf_data(ppc_int: dict) -> tuple:
    """计算 DC 潮流约束所需�?PTDF 矩阵与发电机-总线关联矩阵�?

    Args:
        ppc_int: ext2int 处理后的 PyPower 案例字典（总线�?0-indexed）�?

    Returns:
        PTDF:         (nl, nb) 功率转移分布因子矩阵（无量纲）�?
        ptdf_g:       (nl, ng) PTDF @ G_bus，即 pg 变量的线路潮流系数（MW/MW）�?
        branch_limit: (nl,)   线路热容量上限，RATE_A（MW）�?
        active_lines: 需施加约束的线路索引列表（RATE_A > 0 且线路在线）�?
    """
    gen    = ppc_int['gen']
    branch = ppc_int['branch']
    ng     = gen.shape[0]
    nb     = ppc_int['bus'].shape[0]
    nl     = branch.shape[0]

    PTDF = makePTDF(ppc_int['baseMVA'], ppc_int['bus'], branch)  # (nl, nb)

    # 发电�?总线关联矩阵（ext2int 后总线 0-indexed�?
    G_bus = np.zeros((nb, ng))
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        G_bus[bus_idx, g] = 1.0

    ptdf_g       = PTDF @ G_bus            # (nl, ng)
    branch_limit = branch[:, RATE_A]       # (nl,) MW

    # 仅约束有热容限制（RATE_A > 0）且在线（BR_STATUS = 1）的线路
    active_lines = [
        l for l in range(nl)
        if branch_limit[l] > 1e-6 and branch[l, BR_STATUS] > 0
    ]
    return PTDF, ptdf_g, branch_limit, active_lines


# ========================== 内部辅助：单机组 LP（多代理约束�?==========================

def _solve_unit_surrogate_model(
    trainer: SubproblemSurrogateTrainer,
    lambda_val: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: Optional[np.ndarray] = None,
    pg_costs: Optional[np.ndarray] = None,
    scenario_sample: Optional[dict] = None,
    surrogate_soft_penalty: float = 1e8,
    binary_x: bool = False,
    surrogate_constraint_scope: str = "all",
) -> Tuple[np.ndarray, int, dict]:
    """
    求解单机组子问题 LP（使�?V3 三时段代理约束）�?

    Args:
        trainer: 该机组的 SubproblemSurrogateTrainer
        lambda_val: (T,) 功率平衡对偶变量
        alphas, betas, gammas, deltas: V3 代理约束参数，各 shape (max_constraints,)

    Returns:
        x_LP: (T,) LP 松弛解；若求解失败返回 NaN 向量
        status: Gurobi 状态码
        details: 求解诊断信息
    """
    g = trainer.unit_id
    T = trainer.T
    lambda_unit = _extract_unit_lambda(lambda_val, T, unit_id=g, trainer=trainer)
    timestep_map, offset_map = _resolve_surrogate_constraint_layout(
        trainer,
        scenario_sample,
        T,
        len(alphas),
    )
    scope_norm = _normalize_surrogate_constraint_scope(surrogate_constraint_scope)

    def _solve_once(use_soft_surrogate: bool) -> Tuple[np.ndarray, int, dict]:
        model_name = 'unit_milp_surrogate' if binary_x else 'unit_lp_surrogate'
        model = gp.Model(model_name)
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0

        pg = model.addVars(T, lb=0, name='pg')
        x = model.addVars(
            T,
            lb=0,
            ub=1,
            vtype=GRB.BINARY if binary_x else GRB.CONTINUOUS,
            name='x',
        )
        cpower = model.addVars(T, lb=0, name='cpower')

        for t in range(T):
            model.addConstr(pg[t] >= trainer.gen[g, PMIN] * x[t])
            model.addConstr(pg[t] <= trainer.gen[g, PMAX] * x[t])

        ppc_ref = getattr(trainer, 'ppc_raw', getattr(trainer, 'ppc', {'gen': trainer.gen}))
        Ru_all, Rd_all, Ru_co_all, Rd_co_all = _get_ramp_limits_from_ppc(
            ppc_ref,
            trainer.gen,
            trainer.T_delta,
        )
        Ru = float(Ru_all[g])
        Rd = float(Rd_all[g])
        Ru_co = float(Ru_co_all[g])
        Rd_co = float(Rd_co_all[g])
        for t in range(1, T):
            model.addConstr(pg[t] - pg[t-1] <= Ru * x[t-1] + Ru_co * (1 - x[t-1]))
            model.addConstr(pg[t-1] - pg[t] <= Rd * x[t] + Rd_co * (1 - x[t]))

        min_up_steps, min_down_steps = _get_min_up_down_time_steps(
            ppc_ref, trainer.ng, trainer.T_delta, T,
        )
        Ton = int(min_up_steps[g])
        Toff = int(min_down_steps[g])
        for tau in range(1, Ton + 1):
            for t1 in range(T - tau):
                model.addConstr(x[t1+1] - x[t1] <= x[t1+tau])
        for tau in range(1, Toff + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[t1+1] + x[t1] <= 1 - x[t1+tau])

        b_nl = trainer.subproblem_generation_no_load_coeff(g)
        for t in range(T):
            model.addConstr(
                cpower[t] >= trainer.gencost[g, -2] / trainer.T_delta * pg[t] + b_nl * x[t]
            )

        surrogate_slacks = []
        weighted_surrogate_slacks: List[Tuple[Any, float]] = []
        surrogate_rows = []
        for k, t_k in enumerate(timestep_map):
            if not _allow_surrogate_constraint_by_scope(trainer, k, scope_norm):
                continue
            a = float(alphas[k])
            b = float(betas[k])
            c = float(gammas[k])
            r = float(deltas[k])
            if abs(a) <= 1e-10 and abs(b) <= 1e-10 and abs(c) <= 1e-10:
                continue
            expr = build_surrogate_constraint_expression(
                x,
                t_k,
                offset_map[k],
                a,
                b,
                c,
                T,
            ) - r
            slack_var = None
            if use_soft_surrogate:
                slack_var = model.addVar(lb=0, name=f'surr_slack_{k}')
                model.addConstr(expr <= slack_var, name=f'surr_{k}')
                surrogate_slacks.append(slack_var)
                penalty_factor = (
                    SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR
                    if _is_sign4_surrogate_constraint(trainer, k)
                    else SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR
                )
                weighted_surrogate_slacks.append((slack_var, penalty_factor))
            else:
                model.addConstr(expr <= 0.0, name=f'surr_{k}')
            surrogate_rows.append({
                'k': int(k),
                'timestep': int(t_k),
                'offsets': tuple(offset_map[k]),
                'alpha': a,
                'beta': b,
                'gamma': c,
                'delta': r,
                'slack_var': slack_var,
            })

        obj = gp.quicksum(cpower[t] for t in range(T))
        if costs is not None:
            obj += gp.quicksum(float(costs[t]) * x[t] for t in range(min(T, len(costs))))
        if pg_costs is not None:
            obj += gp.quicksum(float(pg_costs[t]) * pg[t] for t in range(min(T, len(pg_costs))))
        obj -= gp.quicksum(float(lambda_unit[t]) * pg[t] for t in range(T))
        obj += _surrogate_relaxation_penalty_expr(
            surrogate_soft_penalty,
            weighted_surrogate_slacks,
        )
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        details = {
            'status': int(model.status),
            'status_name': _gurobi_status_name(model.status),
            'binary_x': bool(binary_x),
            'used_soft_surrogate': bool(use_soft_surrogate),
            'surrogate_soft_penalty': float(surrogate_soft_penalty) if use_soft_surrogate else None,
            'surrogate_soft_preferred_penalty_factor': (
                float(SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR)
                if use_soft_surrogate
                else None
            ),
            'surrogate_soft_strict_penalty_factor': (
                float(SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR)
                if use_soft_surrogate
                else None
            ),
            'n_surrogate_constraints': int(len(surrogate_rows)),
        }

        if model.status == GRB.OPTIMAL:
            x_sol = np.array([x[t].X for t in range(T)], dtype=float)
            pg_sol = np.array([pg[t].X for t in range(T)], dtype=float)
            details['objective_value'] = float(model.ObjVal)
            details['x_solution'] = x_sol.copy()
            details['pg_solution'] = pg_sol.copy()

            surrogate_violations = []
            violation_sum = 0.0
            violation_max = 0.0
            slack_sum = 0.0
            slack_max = 0.0
            for row in surrogate_rows:
                lhs = build_surrogate_constraint_expression(
                    x_sol,
                    row['timestep'],
                    row['offsets'],
                    row['alpha'],
                    row['beta'],
                    row['gamma'],
                    T,
                )
                violation = max(0.0, lhs - row['delta'])
                slack_val = float(row['slack_var'].X) if row['slack_var'] is not None else 0.0
                violation_sum += violation
                violation_max = max(violation_max, violation)
                slack_sum += slack_val
                slack_max = max(slack_max, slack_val)
                surrogate_violations.append({
                    'k': row['k'],
                    'timestep': row['timestep'],
                    'offsets': row['offsets'],
                    'lhs': float(lhs),
                    'rhs': float(row['delta']),
                    'violation': float(violation),
                    'slack': float(slack_val),
                    'alpha': float(row['alpha']),
                    'beta': float(row['beta']),
                    'gamma': float(row['gamma']),
                })
            details['surrogate_violation_sum'] = float(violation_sum)
            details['surrogate_violation_max'] = float(violation_max)
            details['surrogate_slack_sum'] = float(slack_sum)
            details['surrogate_slack_max'] = float(slack_max)
            details['surrogate_violations'] = surrogate_violations
            return x_sol, int(model.status), details

        return np.full(T, np.nan, dtype=float), int(model.status), details

    x_hard, status_hard, details_hard = _solve_once(use_soft_surrogate=False)
    details_hard['hard_status'] = int(status_hard)
    details_hard['hard_status_name'] = _gurobi_status_name(status_hard)
    details_hard['fallback_triggered'] = False
    if status_hard == GRB.OPTIMAL:
        return x_hard, status_hard, details_hard

    if status_hard in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        x_soft, status_soft, details_soft = _solve_once(use_soft_surrogate=True)
        details_soft['hard_status'] = int(status_hard)
        details_soft['hard_status_name'] = _gurobi_status_name(status_hard)
        details_soft['soft_status'] = int(status_soft)
        details_soft['soft_status_name'] = _gurobi_status_name(status_soft)
        details_soft['fallback_triggered'] = True
        return x_soft, status_soft, details_soft

    return x_hard, status_hard, details_hard


def _solve_unit_LP_with_surrogate(
    trainer: SubproblemSurrogateTrainer,
    lambda_val: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: Optional[np.ndarray] = None,
    pg_costs: Optional[np.ndarray] = None,
    scenario_sample: Optional[dict] = None,
    surrogate_soft_penalty: float = 1e8,
    surrogate_constraint_scope: str = "all",
) -> Tuple[np.ndarray, int, dict]:
    """Solve the surrogate subproblem with relaxed commitment variables."""
    return _solve_unit_surrogate_model(
        trainer,
        lambda_val,
        alphas,
        betas,
        gammas,
        deltas,
        costs=costs,
        pg_costs=pg_costs,
        scenario_sample=scenario_sample,
        surrogate_soft_penalty=surrogate_soft_penalty,
        binary_x=False,
        surrogate_constraint_scope=surrogate_constraint_scope,
    )


def _solve_unit_MILP_with_surrogate(
    trainer: SubproblemSurrogateTrainer,
    lambda_val: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: Optional[np.ndarray] = None,
    pg_costs: Optional[np.ndarray] = None,
    scenario_sample: Optional[dict] = None,
    surrogate_soft_penalty: float = 1e8,
    surrogate_constraint_scope: str = "all",
) -> Tuple[np.ndarray, int, dict]:
    """Solve the surrogate subproblem with binary commitment variables."""
    return _solve_unit_surrogate_model(
        trainer,
        lambda_val,
        alphas,
        betas,
        gammas,
        deltas,
        costs=costs,
        pg_costs=pg_costs,
        scenario_sample=scenario_sample,
        surrogate_soft_penalty=surrogate_soft_penalty,
        binary_x=True,
        surrogate_constraint_scope=surrogate_constraint_scope,
    )


# ========================== Step 2：收集多组整数解 ==========================

def collect_integer_solutions(
    pd_data: np.ndarray | dict,
    lambda_val: np.ndarray,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    n_perturbations: int = 5,
    n_similar_scenarios: int = 0,
    similar_scenario_pool_size: int = 10,
    n_load_perturbations: int = 0,
    load_perturbation_scale: float = 0.03,
    perturb_std: float = 0.1,
    neighborhood_weight: float = 0.35,
    lambda_predictor=None,
    rng: Optional[np.random.Generator] = None,
    use_milp_candidate: bool = False,
    milp_for_perturbations: bool = False,
    surrogate_constraint_scope: str = "all",
    return_details: bool = False,
):
    """
    收集多组整数解（来自子问�?LP + 代理参数扰动）�?

    Args:
        pd_data: (nb_load, T) 负荷数据
        lambda_val: 全局对偶变量载荷；可为功率平衡向量、每机组有效系数矩阵或全局 dual dict
        trainers: {unit_id: SubproblemSurrogateTrainer}
        n_perturbations: 直接随机�?surrogate 参数的扰动次�?        n_similar_scenarios: 从历史场景库检索相似负�?新能源样本的次数
        similar_scenario_pool_size: 相似场景候选池大小
        n_load_perturbations: 对当前负荷做小扰动后重新�?NN 的次�?        load_perturbation_scale: 负荷小扰动幅�?        perturb_std: surrogate 网络输出参数扰动标准差（相对值）
        neighborhood_weight: 周边平均约束权重，越大表示越强调相邻约束平滑
        lambda_predictor: 可选，对场景扰动后重新预测 lambda
        rng: 随机数生成器

    Returns:
        x_surr_lp:  (ng, T) surrogate 子问�?LP 连续�?
        x_init_k:   (ng, T) 各机组子问题 LP 整数�?
        x_init_k_m: (ng, n_candidates, T) 多策略扰动后的多组整数解
    """
    if rng is None:
        rng = np.random.default_rng()
    surrogate_constraint_scope_norm = _normalize_surrogate_constraint_scope(surrogate_constraint_scope)

    sample = _coerce_scenario_sample(pd_data)
    pd_matrix = get_sample_net_load(sample)

    unit_ids = sorted(trainers.keys())
    ng = max(unit_ids) + 1
    T = pd_matrix.shape[1]
    n_candidates = n_perturbations + n_similar_scenarios + n_load_perturbations

    x_surr_lp = np.full((ng, T), np.nan, dtype=float)
    x_init_k = np.zeros((ng, T), dtype=int)
    x_init_k_m = np.zeros((ng, n_candidates, T), dtype=int)
    x_init_k_milp = np.zeros((ng, T), dtype=int) if use_milp_candidate else None
    x_init_k_m_milp = (
        np.zeros((ng, n_candidates, T), dtype=int)
        if (use_milp_candidate and milp_for_perturbations and n_candidates > 0)
        else None
    )

    for g in unit_ids:
        trainer = trainers[g]
        param_candidates = _build_surrogate_parameter_candidates(
            base_sample=sample,
            base_lambda=lambda_val,
            trainer=trainer,
            trainers=trainers,
            rng=rng,
            lambda_predictor=lambda_predictor,
            n_param_perturbations=n_perturbations,
            perturb_std=perturb_std,
            neighborhood_weight=neighborhood_weight,
            n_similar_scenarios=n_similar_scenarios,
            similar_scenario_pool_size=similar_scenario_pool_size,
            n_load_perturbations=n_load_perturbations,
            load_perturbation_scale=load_perturbation_scale,
        )
        _base_name, base_scenario, alphas, betas, gammas, deltas, costs, pg_costs = param_candidates[0]

        # 原始子问�?LP
        solve_kwargs = {
            'costs': costs,
            'pg_costs': pg_costs,
            'scenario_sample': base_scenario,
        }
        if surrogate_constraint_scope_norm != "all":
            solve_kwargs['surrogate_constraint_scope'] = surrogate_constraint_scope_norm

        x_LP_k, status_k, details_k = _solve_unit_LP_with_surrogate(
            trainer,
            lambda_val,
            alphas,
            betas,
            gammas,
            deltas,
            **solve_kwargs,
        )
        if status_k != GRB.OPTIMAL:
            raise RuntimeError(
                f"unit {g} surrogate LP failed: status={_gurobi_status_name(status_k)}, "
                f"fallback_triggered={details_k.get('fallback_triggered', False)}"
            )
        x_surr_lp[g] = x_LP_k
        x_init_k[g] = round_to_integer(x_LP_k)
        if x_init_k_milp is not None:
            x_MILP_k, status_milp_k, _details_milp_k = _solve_unit_MILP_with_surrogate(
                trainer,
                lambda_val,
                alphas,
                betas,
                gammas,
                deltas,
                **solve_kwargs,
            )
            if status_milp_k == GRB.OPTIMAL:
                x_init_k_milp[g] = round_to_integer(x_MILP_k)
            else:
                print(
                    f"  Warning: unit {g} base surrogate MILP failed "
                    f"(status={_gurobi_status_name(status_milp_k)}); reuse LP candidate",
                    flush=True,
                )
                x_init_k_milp[g] = x_init_k[g]

        # Solve additional unit LPs from perturbed / retrieved surrogate parameters.
        for m, (
            _name,
            scenario_m,
            alphas_m,
            betas_m,
            gammas_m,
            deltas_m,
            costs_m,
            pg_costs_m,
        ) in enumerate(param_candidates[1:]):
            solve_kwargs_m = {
                'costs': costs_m,
                'pg_costs': pg_costs_m,
                'scenario_sample': scenario_m,
            }
            if surrogate_constraint_scope_norm != "all":
                solve_kwargs_m['surrogate_constraint_scope'] = surrogate_constraint_scope_norm

            x_LP_m, status_m, details_m = _solve_unit_LP_with_surrogate(
                trainer,
                lambda_val,
                alphas_m,
                betas_m,
                gammas_m,
                deltas_m,
                **solve_kwargs_m,
            )
            if status_m == GRB.OPTIMAL:
                x_init_k_m[g, m] = round_to_integer(x_LP_m)
            else:
                print(
                    f"  Warning: unit {g} perturbed surrogate LP failed "
                    f"(status={_gurobi_status_name(status_m)}); reuse base candidate",
                    flush=True,
                )
                x_init_k_m[g, m] = x_init_k[g]
            if x_init_k_m_milp is not None:
                x_MILP_m, status_milp_m, _details_milp_m = _solve_unit_MILP_with_surrogate(
                    trainer,
                    lambda_val,
                    alphas_m,
                    betas_m,
                    gammas_m,
                    deltas_m,
                    **solve_kwargs_m,
                )
                if status_milp_m == GRB.OPTIMAL:
                    x_init_k_m_milp[g, m] = round_to_integer(x_MILP_m)
                else:
                    x_init_k_m_milp[g, m] = x_init_k_m[g, m]

    if return_details:
        details = {
            'use_milp_candidate': bool(use_milp_candidate),
            'milp_for_perturbations': bool(milp_for_perturbations),
            'surrogate_constraint_scope': surrogate_constraint_scope_norm,
            'x_init_k_milp': None if x_init_k_milp is None else np.asarray(x_init_k_milp, dtype=int),
            'x_init_k_m_milp': (
                None if x_init_k_m_milp is None else np.asarray(x_init_k_m_milp, dtype=int)
            ),
        }
        return x_surr_lp, x_init_k, x_init_k_m, details

    return x_surr_lp, x_init_k, x_init_k_m


# ========================== Step 3：识别高可信度变�?==========================

def identify_trusted_mask(
    x_LP: np.ndarray,
    x_init_k: np.ndarray,
    x_init_k_m: np.ndarray,
    conf_threshold: float = 0.15
) -> np.ndarray:
    """
    识别高可信度（整数性强且多来源一致）的变量�?

    条件1：LP 值远�?0.5（x_LP < conf_threshold �?x_LP > 1 - conf_threshold�?
    条件2：多数投票结果与 round(x_LP) 一�?

    Args:
        x_LP:       (ng, T) 全局 LP 松弛�?
        x_init_k:   (ng, T) 子问�?LP 整数�?
        x_init_k_m: (ng, n_perturbations, T) 扰动整数�?
        conf_threshold: LP 整数性置信阈�?

    Returns:
        trusted_mask: (ng, T) bool 数组，True 表示高可信度（固定）
    """
    x_lp_arr = np.asarray(x_LP)
    x_init_arr = np.asarray(x_init_k)
    x_init_pert_arr = np.asarray(x_init_k_m)
    if x_lp_arr.shape != x_init_arr.shape:
        raise ValueError(
            "identify_trusted_mask() requires x_LP and x_init_k to have the same shape. "
            f"Got x_LP.shape={x_lp_arr.shape}, x_init_k.shape={x_init_arr.shape}. "
            "This usually means the global FP pipeline was called with only a partial set "
            "of surrogate trainers."
        )
    if x_init_pert_arr.ndim != 3:
        raise ValueError(
            "identify_trusted_mask() requires x_init_k_m to be a 3D array with shape "
            "(ng, n_perturbations, T). "
            f"Got x_init_k_m.shape={x_init_pert_arr.shape}."
        )
    if x_init_pert_arr.shape[0] != x_lp_arr.shape[0] or x_init_pert_arr.shape[2] != x_lp_arr.shape[1]:
        raise ValueError(
            "identify_trusted_mask() requires x_init_k_m to align with x_LP on generator/time axes. "
            f"Got x_LP.shape={x_lp_arr.shape}, x_init_k_m.shape={x_init_pert_arr.shape}. "
            "This usually means the global FP pipeline was called with only a partial set "
            "of surrogate trainers."
        )

    # 条件1：整数性强
    near_zero = x_lp_arr < conf_threshold
    near_one = x_lp_arr > 1.0 - conf_threshold
    integrality_confident = near_zero | near_one

    # 条件2：多数投票一致�?
    n_pert = x_init_pert_arr.shape[1]
    x_ref = np.round(x_lp_arr).astype(int)

    # 汇总所有投票（x_init_k + 所有扰动解�?
    vote_sum = x_init_arr.astype(float) + np.sum(x_init_pert_arr.astype(float), axis=1)
    n_votes = 1 + n_pert
    majority = (vote_sum / n_votes >= 0.5).astype(int)
    consistent = (majority == x_ref)

    return integrality_confident & consistent


# ========================== Step 6 辅助：全局 LP 松弛 ==========================

DEFAULT_SURROGATE_PENALTY = 1e8
DEFAULT_UNIT_SURROGATE_SOFT_PENALTY = 1e8
SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR = 1e-1
SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR = 1.0


def _is_sign4_surrogate_constraint(
    trainer: Optional['SubproblemSurrogateTrainer'],
    constraint_index: int,
) -> bool:
    """Return True for the sign4 head block in all-template surrogate layouts."""
    strategy = normalize_constraint_generation_strategy(
        getattr(trainer, 'constraint_generation_strategy', 'sensitive') or 'sensitive'
    )
    if strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4:
        return True
    if strategy == CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4_PLUS_SINGLE:
        T = int(getattr(trainer, 'T', 0) or 0)
        group_size = max(int(getattr(trainer, 'all_mode_group_size', 4) or 4), 1)
        sign4_count = group_size * max(T - 2, 0)
        return int(constraint_index) < sign4_count
    return False


def _normalize_surrogate_constraint_scope(scope: str | None) -> str:
    scope_norm = str(scope or "all").strip().lower().replace("-", "_")
    aliases = {
        "": "all",
        "both": "all",
        "full": "all",
        "all_constraints": "all",
        "sign4_only": "sign4",
        "sign_4": "sign4",
        "no_subproblem": "none",
        "no_subproblem_surrogate": "none",
        "subproblem_none": "none",
        "none": "none",
        "off": "none",
        "false": "none",
    }
    scope_norm = aliases.get(scope_norm, scope_norm)
    if scope_norm not in {"all", "sign4", "none"}:
        raise ValueError(f"Unsupported surrogate_constraint_scope={scope!r}; expected 'all', 'sign4', or 'none'")
    return scope_norm


def _allow_surrogate_constraint_by_scope(
    trainer: Optional['SubproblemSurrogateTrainer'],
    constraint_index: int,
    scope: str | None,
) -> bool:
    scope_norm = _normalize_surrogate_constraint_scope(scope)
    if scope_norm == "all":
        return True
    if scope_norm == "none":
        return False
    return _is_sign4_surrogate_constraint(trainer, constraint_index)


def _surrogate_relaxation_penalty_expr(base_penalty: float, weighted_slacks: List[Tuple[Any, float]]):
    if not weighted_slacks:
        return 0.0
    return float(base_penalty) * gp.quicksum(
        float(factor) * slack for slack, factor in weighted_slacks
    )


def _gurobi_status_name(status: Optional[int]) -> str:
    status_map = {
        GRB.LOADED: 'LOADED',
        GRB.OPTIMAL: 'OPTIMAL',
        GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.INF_OR_UNBD: 'INF_OR_UNBD',
        GRB.UNBOUNDED: 'UNBOUNDED',
        GRB.CUTOFF: 'CUTOFF',
        GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',
        GRB.NODE_LIMIT: 'NODE_LIMIT',
        GRB.TIME_LIMIT: 'TIME_LIMIT',
        GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
        GRB.INTERRUPTED: 'INTERRUPTED',
        GRB.NUMERIC: 'NUMERIC',
        GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT',
    }
    if status is None:
        return 'UNKNOWN'
    return status_map.get(int(status), f'STATUS_{int(status)}')


def _build_surrogate_relaxation_stages() -> List[dict]:
    """Hard -> soften subproblem -> soften both."""
    return [
        {
            'name': 'hard_bcd_hard_subproblem',
            'hard_subproblem': True,
            'hard_bcd': True,
            'penalty_subproblem': None,
            'penalty_bcd': None,
        },
        {
            'name': 'hard_bcd_soft_subproblem',
            'hard_subproblem': False,
            'hard_bcd': True,
            'penalty_subproblem': DEFAULT_SURROGATE_PENALTY,
            'penalty_bcd': None,
        },
        {
            'name': 'soft_bcd_soft_subproblem',
            'hard_subproblem': False,
            'hard_bcd': False,
            'penalty_subproblem': DEFAULT_SURROGATE_PENALTY,
            'penalty_bcd': DEFAULT_SURROGATE_PENALTY,
        },
    ]

def solve_global_LP_relaxation(
    ppc: dict,
    pd_data: np.ndarray | dict,
    T_delta: float,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_val: np.ndarray,
    agent=None,
    sparse_library: Optional[SparseSurrogateLibrary] = None,
    sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    surrogate_constraint_scope: str = "all",
    bcd_proxy_scope: str = "both",
    bcd_theta_constraint_filter: Optional[Iterable[Tuple[int, int]]] = None,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    构建完整 UC LP 松弛（x �?[0,1]），按以下顺序分级尝试代理约束：
      1. BCD + subproblem 全部硬约�?      2. 保持 BCD 为硬约束，仅�?subproblem 约束软化到高罚项
      3. �?BCD �?subproblem 都软化到高罚�?
    Args:
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔（小时）
        trainers: {unit_id: trainer}
        lambda_val: 全局对偶变量载荷（用于查询每机组代理约束参数）
        agent: （可选）已训练的 Agent_NN_BCD 实例；若提供则额外加入其
            theta/zeta 参数化代理约�?        sparse_library: （可选）离线筛选出的稀疏参数化约束库。若提供则以
            `lhs(x, pg) - rhs(Pd) <= slack` 的软约束形式加入全局 LP�?
        sparse_x_template_library: （可选）x[g,t] 稀疏支持集模板库，�?
            `sum x[g_k, t_k] <= rhs + slack` 的形式软注入�?

    Returns:
        x_LP: (ng, T) LP 松弛解；若不可行返回零矩�?
    """
    sample = None
    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        pd_matrix = get_sample_net_load(sample)
    else:
        pd_matrix = pd_data
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng = gen.shape[0]
    T = pd_matrix.shape[1]
    Pd_sum = np.sum(pd_matrix, axis=0)  # (T,)
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, T_delta, T)
    start_cost = gencost[:, 1]   # gencost �?1 列：启动成本
    shut_cost  = gencost[:, 2]   # gencost �?2 列：关机成本

    last_status = None
    stage_stats: List[Dict[str, Any]] = []
    stages = _build_surrogate_relaxation_stages()
    bcd_proxy_scope_norm = str(bcd_proxy_scope or "both").strip().lower()
    if bcd_proxy_scope_norm not in {"both", "theta", "zeta", "none"}:
        raise ValueError(f"Unsupported bcd_proxy_scope={bcd_proxy_scope!r}")
    surrogate_constraint_scope_norm = _normalize_surrogate_constraint_scope(surrogate_constraint_scope)
    add_bcd_theta = bcd_proxy_scope_norm in {"both", "theta"}
    add_bcd_zeta = bcd_proxy_scope_norm in {"both", "zeta"}
    bcd_theta_constraint_filter_set = (
        {
            (int(branch_id), int(time_slot))
            for branch_id, time_slot in bcd_theta_constraint_filter
        }
        if bcd_theta_constraint_filter is not None
        else None
    )

    def _safe_attr(_model, _name: str, _default=None):
        try:
            return getattr(_model, _name)
        except Exception:
            return _default

    def _collect_model_stats(_model, _stage: dict, _stage_index: int) -> Dict[str, Any]:
        try:
            constr_names = [c.ConstrName for c in _model.getConstrs()]
        except Exception:
            constr_names = []
        n_subproblem = sum(1 for name in constr_names if name.startswith("g") and "_surr_" in name)
        n_theta = sum(1 for name in constr_names if name.startswith("theta_surr_"))
        n_zeta = sum(1 for name in constr_names if name.startswith("zeta_surr_"))
        n_sparse = sum(1 for name in constr_names if name.startswith("sparse"))
        n_total = int(_safe_attr(_model, "NumConstrs", len(constr_names)) or 0)
        n_proxy = int(n_subproblem + n_theta + n_zeta + n_sparse)
        sol_count = int(_safe_attr(_model, "SolCount", 0) or 0)
        stats = {
            "stage_index": int(_stage_index),
            "stage_name": str(_stage["name"]),
            "status": int(_safe_attr(_model, "Status", _model.status) or _model.status),
            "status_name": _gurobi_status_name(_safe_attr(_model, "Status", _model.status)),
            "runtime_sec": float(_safe_attr(_model, "Runtime", float("nan"))),
            "work": float(_safe_attr(_model, "Work", float("nan"))),
            "iter_count": float(_safe_attr(_model, "IterCount", float("nan"))),
            "bar_iter_count": float(_safe_attr(_model, "BarIterCount", float("nan"))),
            "objective": float(_safe_attr(_model, "ObjVal", float("nan"))) if sol_count > 0 else float("nan"),
            "objective_bound": float(_safe_attr(_model, "ObjBound", float("nan"))) if sol_count > 0 else float("nan"),
            "num_vars": int(_safe_attr(_model, "NumVars", 0) or 0),
            "num_constraints": n_total,
            "num_nonzeros": int(_safe_attr(_model, "NumNZs", 0) or 0),
            "num_proxy_constraints": n_proxy,
            "surrogate_constraint_scope": surrogate_constraint_scope_norm,
            "num_subproblem_surrogate_constraints": int(n_subproblem),
            "num_bcd_theta_constraints": int(n_theta),
            "num_bcd_zeta_constraints": int(n_zeta),
            "bcd_theta_constraint_filter_count": (
                None
                if bcd_theta_constraint_filter_set is None
                else int(len(bcd_theta_constraint_filter_set))
            ),
            "num_sparse_constraints": int(n_sparse),
            "num_base_constraints": int(max(0, n_total - n_proxy)),
            "num_subproblem_slacks": int(len(subproblem_slacks)),
            "num_bcd_slacks": int(len(bcd_slacks)),
            "num_subproblem_sign4_slacks": int(len(subproblem_sign4_slacks)),
            "num_subproblem_strict_slacks": int(len(subproblem_strict_slacks)),
            "num_bcd_theta_slacks": int(len(bcd_theta_slacks)),
            "num_bcd_zeta_slacks": int(len(bcd_zeta_slacks)),
            "bcd_proxy_scope": bcd_proxy_scope_norm,
            "surrogate_soft_preferred_penalty_factor": float(
                SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR
            ),
            "surrogate_soft_strict_penalty_factor": float(
                SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR
            ),
            "hard_subproblem": bool(_stage["hard_subproblem"]),
            "hard_bcd": bool(_stage["hard_bcd"]),
            "used_soft_subproblem": not bool(_stage["hard_subproblem"]),
            "used_soft_bcd": not bool(_stage["hard_bcd"]),
            "used_fallback_stage": int(_stage_index) > 1,
        }
        if subproblem_slacks and sol_count > 0:
            vals = [float(v.X) for v in subproblem_slacks]
            stats["subproblem_slack_sum"] = float(np.sum(vals))
            stats["subproblem_slack_max"] = float(np.max(vals))
        else:
            stats["subproblem_slack_sum"] = 0.0
            stats["subproblem_slack_max"] = 0.0
        if bcd_slacks and sol_count > 0:
            vals = [float(v.X) for v in bcd_slacks]
            stats["bcd_slack_sum"] = float(np.sum(vals))
            stats["bcd_slack_max"] = float(np.max(vals))
        else:
            stats["bcd_slack_sum"] = 0.0
            stats["bcd_slack_max"] = 0.0
        return stats

    def _collect_main_constraint_activity_rows(_model, _stage: dict, _stage_index: int) -> List[Dict[str, Any]]:
        var_values = {}
        try:
            var_values = {v.VarName: float(v.X) for v in _model.getVars()}
        except Exception:
            var_values = {}

        rows: List[Dict[str, Any]] = []
        try:
            constrs = list(_model.getConstrs())
        except Exception:
            return rows

        for c in constrs:
            name = str(getattr(c, "ConstrName", ""))
            kind = None
            unit_id = None
            constraint_index = None
            branch_id = None
            time_slot = None
            relaxation_var = None

            m = re.match(r"^g(\d+)_surr_(\d+)$", name)
            if m:
                kind = "subproblem_surrogate"
                unit_id = int(m.group(1))
                constraint_index = int(m.group(2))
                relaxation_var = f"g{unit_id}_surr_slack_{constraint_index}"
            else:
                m = re.match(r"^theta_surr_(\d+)_(\d+)$", name)
                if m:
                    kind = "bcd_theta"
                    branch_id = int(m.group(1))
                    time_slot = int(m.group(2))
                    relaxation_var = f"theta_slack_{branch_id}_{time_slot}"
                else:
                    m = re.match(r"^zeta_surr_(\d+)_(\d+)$", name)
                    if m:
                        kind = "bcd_zeta"
                        unit_id = int(m.group(1))
                        time_slot = int(m.group(2))
                        relaxation_var = f"zeta_slack_{unit_id}_{time_slot}"
                    elif name.startswith("sparse"):
                        kind = "sparse_surrogate"

            if kind is None:
                continue

            try:
                slack = float(c.Slack)
            except Exception:
                slack = float("nan")
            try:
                dual = float(c.Pi)
            except Exception:
                dual = float("nan")
            relax_value = (
                float(var_values[relaxation_var])
                if relaxation_var is not None and relaxation_var in var_values
                else 0.0
            )
            rows.append({
                "stage_index": int(_stage_index),
                "stage_name": str(_stage["name"]),
                "constraint_name": name,
                "kind": kind,
                "unit_id": unit_id,
                "constraint_index": constraint_index,
                "branch_id": branch_id,
                "time_slot": time_slot,
                "row_slack": slack,
                "abs_row_slack": abs(slack) if np.isfinite(slack) else float("nan"),
                "dual": dual,
                "abs_dual": abs(dual) if np.isfinite(dual) else float("nan"),
                "is_row_active_1e_6": bool(np.isfinite(slack) and abs(slack) <= 1e-6),
                "is_dual_active_1e_7": bool(np.isfinite(dual) and abs(dual) > 1e-7),
                "relaxation_var": relaxation_var,
                "relaxation_value": relax_value,
                "is_relaxed": bool(relax_value > 1e-8),
                "hard_subproblem": bool(_stage["hard_subproblem"]),
                "hard_bcd": bool(_stage["hard_bcd"]),
            })
        return rows

    def _infer_agent_theta_zeta(_agent, _sample, _pd_matrix):
        """Infer sample-specific theta/zeta for the current scenario.

        Fallback order:
        1. NN forward pass on current sample features
        2. agent.theta_values / agent.zeta_values static values
        """
        theta_fallback = dict(getattr(_agent, 'theta_values', {}) or {})
        zeta_fallback = dict(getattr(_agent, 'zeta_values', {}) or {})

        if _agent is None:
            return theta_fallback, zeta_fallback

        theta_net = getattr(_agent, 'theta_net', None)
        zeta_net = getattr(_agent, 'zeta_net', None)
        device = getattr(_agent, 'device', None)
        theta_to_dict = getattr(_agent, '_tensor_to_theta_dict', None)
        zeta_to_dict = getattr(_agent, '_tensor_to_zeta_dict', None)
        expected_dim_fn = getattr(_agent, '_get_expected_feature_dim', None)

        if theta_net is None or zeta_net is None or theta_to_dict is None or zeta_to_dict is None:
            return theta_fallback, zeta_fallback

        try:
            import torch

            sample_for_features = _sample
            if sample_for_features is None:
                sample_for_features = normalize_sample_arrays({'pd_data': _pd_matrix})

            features = np.asarray(get_feature_vector_from_sample(sample_for_features), dtype=np.float32)
            expected_dim = expected_dim_fn() if callable(expected_dim_fn) else None
            if expected_dim is not None and features.size != expected_dim:
                net_load_features = np.asarray(get_sample_net_load(sample_for_features), dtype=np.float32).flatten()
                if net_load_features.size == expected_dim:
                    features = net_load_features

            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            if device is not None:
                features_tensor = features_tensor.to(device)

            theta_net.eval()
            zeta_net.eval()
            with torch.no_grad():
                theta_out = theta_net(features_tensor)
                zeta_out = zeta_net(features_tensor)
            theta_net.train()
            zeta_net.train()

            return theta_to_dict(theta_out[0]), zeta_to_dict(zeta_out[0])
        except Exception:
            return theta_fallback, zeta_fallback

    for stage_index, stage in enumerate(stages, start=1):
        model = gp.Model(f"global_LP_relaxation_{stage['name']}")
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0

        pg = model.addVars(ng, T, lb=0, name='pg')
        x = model.addVars(ng, T, lb=0, ub=1, name='x')
        cpower = model.addVars(ng, T, lb=0, name='cpower')
        coc = model.addVars(ng, T - 1, lb=0, name='coc')
        subproblem_slacks: list = []
        subproblem_sign4_slacks: list = []
        subproblem_strict_slacks: list = []
        bcd_slacks: list = []
        bcd_theta_slacks: list = []
        bcd_zeta_slacks: list = []
        weighted_subproblem_slacks: List[Tuple[Any, float]] = []
        weighted_bcd_slacks: List[Tuple[Any, float]] = []
        aux_obj = gp.LinExpr()

        for t in range(T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
                name=f'pb_{t}'
            )

        for g in range(ng):
            for t in range(T):
                model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

            for t in range(1, T):
                model.addConstr(
                    pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1])
                )
                model.addConstr(
                    pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
                )

            Ton_g = int(min_up_steps[g])
            Toff_g = int(min_down_steps[g])
            for tau in range(1, Ton_g + 1):
                for t1 in range(T - tau):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
            for tau in range(1, Toff_g + 1):
                for t1 in range(T - tau):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

            for t in range(1, T):
                model.addConstr(
                    coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]),
                    name=f'start_cost_{g}_{t}'
                )
                model.addConstr(
                    coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]),
                    name=f'shut_cost_{g}_{t}'
                )

            for t in range(T):
                b_nl = (
                    trainers[g].subproblem_generation_no_load_coeff(g)
                    if g in trainers
                    else float(gencost[g, -1]) / T_delta
                )
                model.addConstr(
                    cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t] + b_nl * x[g, t]
                )

            if g in trainers:
                lambda_unit = _extract_unit_lambda(
                    lambda_val,
                    T,
                    unit_id=g,
                    trainer=trainers[g],
                )
                alphas, betas, gammas, deltas, costs, pg_costs = trainers[g].get_surrogate_params(
                    sample if sample is not None else pd_matrix, lambda_unit
                )
                aux_obj += gp.quicksum(float(costs[t]) * x[g, t] for t in range(min(T, len(costs))))
                aux_obj += gp.quicksum(float(pg_costs[t]) * pg[g, t] for t in range(min(T, len(pg_costs))))
                timestep_map, offset_map = _resolve_surrogate_constraint_layout(
                    trainers[g],
                    sample,
                    T,
                    len(alphas),
                )
                for k, t_k in enumerate(timestep_map):
                    if not _allow_surrogate_constraint_by_scope(
                        trainers[g],
                        k,
                        surrogate_constraint_scope_norm,
                    ):
                        continue
                    a = float(alphas[k])
                    b = float(betas[k])
                    c = float(gammas[k])
                    r = float(deltas[k])
                    if abs(a) <= 1e-10 and abs(b) <= 1e-10 and abs(c) <= 1e-10:
                        continue
                    expr = build_surrogate_constraint_expression(
                        {t: x[g, t] for t in range(T)},
                        t_k,
                        offset_map[k],
                        a,
                        b,
                        c,
                        T,
                    ) - r
                    if stage['hard_subproblem']:
                        model.addConstr(expr <= 0.0, name=f'g{g}_surr_{k}')
                    else:
                        slack_k = model.addVar(lb=0, name=f'g{g}_surr_slack_{k}')
                        model.addConstr(expr <= slack_k, name=f'g{g}_surr_{k}')
                        subproblem_slacks.append(slack_k)
                        if _is_sign4_surrogate_constraint(trainers[g], k):
                            subproblem_sign4_slacks.append(slack_k)
                            penalty_factor = SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR
                        else:
                            subproblem_strict_slacks.append(slack_k)
                            penalty_factor = SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR
                        weighted_subproblem_slacks.append((slack_k, penalty_factor))

        try:
            _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
            ptdf_Pd = _PTDF @ pd_matrix
            for l in active_lines:
                limit = float(branch_limit[l])
                for t in range(T):
                    flow_expr = (
                        gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                        - float(ptdf_Pd[l, t])
                    )
                    model.addConstr(flow_expr <= limit, name=f'flow_upper_{l}_{t}')
                    model.addConstr(flow_expr >= -limit, name=f'flow_lower_{l}_{t}')
        except Exception as _dc_err:
            print(f"  Warning: failed to build DC flow constraints ({_dc_err}); skip them", flush=True)

        if agent is not None:
            _ua = getattr(agent, '_current_union_analysis', None)
            _tv, _zv = _infer_agent_theta_zeta(agent, sample, pd_matrix)

            if add_bcd_theta:
                for _ci in (_ua or {}).get('union_constraints', []):
                    _bid = _ci['branch_id']
                    _ts = _ci['time_slot']
                    if (
                        bcd_theta_constraint_filter_set is not None
                        and (int(_bid), int(_ts)) not in bcd_theta_constraint_filter_set
                    ):
                        continue
                    _lhs = gp.LinExpr()
                    for _ci2 in _ci.get('nonzero_pg_coefficients', []):
                        _uid = _ci2['unit_id']
                        if hasattr(agent, '_theta_member_time_index'):
                            _member_time = int(agent._theta_member_time_index(_ci, _ci2))
                        else:
                            _member_time = int(_ci2.get('time_index', _ts))
                        if hasattr(agent, '_theta_var_name'):
                            _tname = agent._theta_var_name(_bid, _uid, _member_time)
                        else:
                            _tname = f'theta_branch_{_bid}_unit_{_uid}_time_{_member_time}'
                        _coeff = float(_tv.get(_tname, 0.0))
                        if abs(_coeff) > 1e-10 and _uid < ng and 0 <= _member_time < T:
                            _lhs += _coeff * x[_uid, _member_time]
                    if hasattr(agent, '_theta_rhs_name'):
                        _rhs_name = agent._theta_rhs_name(_bid, _ts)
                    else:
                        _rhs_name = f'theta_rhs_branch_{_bid}_time_{_ts}'
                    _rhs = float(_tv.get(_rhs_name, 1.0))
                    _direction = (
                        float(agent._get_theta_constraint_direction(_bid, _ts))
                        if hasattr(agent, '_get_theta_constraint_direction')
                        else 1.0
                    )
                    expr = _direction * (_lhs - _rhs)
                    if stage['hard_bcd']:
                        model.addConstr(expr <= 0.0, name=f'theta_surr_{_bid}_{_ts}')
                    else:
                        _slack = model.addVar(lb=0, name=f'theta_slack_{_bid}_{_ts}')
                        model.addConstr(expr <= _slack, name=f'theta_surr_{_bid}_{_ts}')
                        bcd_slacks.append(_slack)
                        bcd_theta_slacks.append(_slack)
                        weighted_bcd_slacks.append(
                            (_slack, SURROGATE_RELAXATION_PREFERRED_PENALTY_FACTOR)
                        )

            if add_bcd_zeta:
                for _zc in (_ua or {}).get('union_zeta_constraints', []):
                    _uid = _zc['unit_id']
                    _ts = _zc['time_slot']
                    if hasattr(agent, '_zeta_var_name'):
                        _zname = agent._zeta_var_name(_uid, _ts)
                    else:
                        _zname = f'zeta_unit_{_uid}_time_{_ts}'
                    _coeff = float(_zv.get(_zname, 0.0))
                    if hasattr(agent, '_zeta_rhs_name'):
                        _rhs_name = agent._zeta_rhs_name(_uid, _ts)
                    else:
                        _rhs_name = f'zeta_rhs_unit_{_uid}_time_{_ts}'
                    _rhs = float(_zv.get(_rhs_name, 1.0))
                    if abs(_coeff) <= 1e-10 or _uid >= ng or _ts >= T:
                        continue
                    _direction = (
                        float(agent._get_zeta_constraint_direction(_uid, _ts))
                        if hasattr(agent, '_get_zeta_constraint_direction')
                        else 1.0
                    )
                    expr = _direction * (_coeff * x[_uid, _ts] - _rhs)
                    if stage['hard_bcd']:
                        model.addConstr(expr <= 0.0, name=f'zeta_surr_{_uid}_{_ts}')
                    else:
                        _slack = model.addVar(lb=0, name=f'zeta_slack_{_uid}_{_ts}')
                        model.addConstr(expr <= _slack, name=f'zeta_surr_{_uid}_{_ts}')
                        bcd_slacks.append(_slack)
                        bcd_zeta_slacks.append(_slack)
                        weighted_bcd_slacks.append(
                            (_slack, SURROGATE_RELAXATION_STRICT_PENALTY_FACTOR)
                        )

        soft_slacks = subproblem_slacks + bcd_slacks
        add_sparse_parameterized_constraints(
            model,
            x,
            pg,
            pd_data,
            sparse_library=sparse_library,
            surr_slacks=soft_slacks,
            name_prefix="sparse",
        )
        add_sparse_x_templates_to_model(
            model,
            x,
            template_library=sparse_x_template_library,
            surr_slacks=soft_slacks,
            name_prefix="sparse_x",
        )

        uc_obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
                  + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
        problem_obj = uc_obj + aux_obj
        full_obj = problem_obj
        if subproblem_slacks:
            full_obj += _surrogate_relaxation_penalty_expr(
                stage['penalty_subproblem'],
                weighted_subproblem_slacks,
            )
        if bcd_slacks:
            full_obj += _surrogate_relaxation_penalty_expr(
                stage['penalty_bcd'],
                weighted_bcd_slacks,
            )
        model.setObjective(full_obj, GRB.MINIMIZE)
        model.optimize()

        last_status = model.status
        current_stats = _collect_model_stats(model, stage, stage_index)
        if int(current_stats.get("status", -1)) == int(GRB.OPTIMAL):
            current_stats["objective_solver_full"] = current_stats.get("objective", float("nan"))
            current_stats["objective"] = float(problem_obj.getValue())
            current_stats["objective_problem"] = float(problem_obj.getValue())
            current_stats["objective_uc_cost"] = float(uc_obj.getValue())
            current_stats["objective_surrogate_aux"] = float(aux_obj.getValue())
        stage_stats.append(dict(current_stats))
        if model.status == GRB.OPTIMAL:
            if stage_index > 1:
                print(f"  全局 LP 使用回退阶段求解成功: {stage['name']}", flush=True)
            x_sol = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
            if return_stats:
                current_stats["main_constraint_activity_rows"] = _collect_main_constraint_activity_rows(
                    model,
                    stage,
                    stage_index,
                )
                current_stats["all_stage_attempts"] = stage_stats
                return x_sol, current_stats
            return x_sol

        if stage_index < len(stages):
            # ASCII-only: avoids mojibake on Windows consoles not using UTF-8
            print(
                f"  [global_LP] stage={stage['name']} infeasible (status={model.status}); try next stage",
                flush=True,
            )

    print(f"  警告: 全局 LP 松弛求解失败 (status={last_status})，返回零矩阵", flush=True)
    x_zero = np.zeros((ng, T))
    if return_stats:
        failed_stats = dict(stage_stats[-1]) if stage_stats else {}
        failed_stats.update({
            "stage_name": failed_stats.get("stage_name", "failed"),
            "status": int(last_status) if last_status is not None else -1,
            "status_name": _gurobi_status_name(last_status),
            "all_stage_attempts": stage_stats,
        })
        return x_zero, failed_stats
    return x_zero

def solve_global_LP_relaxation_without_surrogate(
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    构建完整 UC LP 松弛（x �?[0,1]），不含任何代理约束，仅�?UC 基础约束�?DC 潮流约束�?

    Args:
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔（小时）

    Returns:
        x_LP: (ng, T) LP 松弛解；若不可行返回零矩�?
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng = gen.shape[0]
    T = pd_data.shape[1]
    Pd_sum = np.sum(pd_data, axis=0)  # (T,)

    model = gp.Model('global_LP_relaxation')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    x = model.addVars(ng, T, lb=0, ub=1, name='x')
    cpower = model.addVars(ng, T, lb=0, name='cpower')
    coc = model.addVars(ng, T - 1, lb=0, name='coc')   # 启停成本

    # 功率平衡
    for t in range(T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
            name=f'pb_{t}'
        )

    # 爬坡参数（与 UnitCommitmentModel 一致）
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, T_delta, T)
    start_cost = gencost[:, 1]   # gencost �?1 列：启动成本
    shut_cost  = gencost[:, 2]   # gencost �?2 列：关机成本

    for g in range(ng):
        # 发电上下�?
        for t in range(T):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

        # 爬坡约束
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1])
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
            )

        # 最小开关机时间
        Ton_g = int(min_up_steps[g])
        Toff_g = int(min_down_steps[g])
        for tau in range(1, Ton_g + 1):
            for t1 in range(T - tau):
                model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
        for tau in range(1, Toff_g + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

        # 启停成本（参�?BCD / uc_cplex.py�?
        for t in range(1, T):
            model.addConstr(
                coc[g, t-1] >= start_cost[g] * (x[g, t] - x[g, t-1]),
                name=f'start_cost_{g}_{t}'
            )
            model.addConstr(
                coc[g, t-1] >= shut_cost[g] * (x[g, t-1] - x[g, t]),
                name=f'shut_cost_{g}_{t}'
            )

        # 发电成本
        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                               + gencost[g, -1] / T_delta * x[g, t]
            )

    # DC 线路潮流约束（PTDF 方法，硬约束�?
    # 假设 pd_data 形状�?(nb, T)，行顺序�?ext2int 后的总线顺序一�?
    try:
        _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = _PTDF @ pd_data   # (nl, T)：负荷对各线路潮流的贡献
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit,  name=f'flow_upper_{l}_{t}')
                model.addConstr(flow_expr >= -limit, name=f'flow_lower_{l}_{t}')
    except Exception as _dc_err:
        print(f"  Warning: failed to build DC flow constraints ({_dc_err}); skip them", flush=True)

    # 目标：发电成�?+ 启停成本
    obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1)))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    stats = {
        "status": int(getattr(model, "Status", model.status)),
        "status_name": _gurobi_status_name(getattr(model, "Status", model.status)),
        "runtime_sec": float(getattr(model, "Runtime", float("nan"))),
        "work": float(getattr(model, "Work", float("nan"))),
        "iter_count": float(getattr(model, "IterCount", float("nan"))),
        "bar_iter_count": float(getattr(model, "BarIterCount", float("nan"))),
        "objective": (
            float(getattr(model, "ObjVal", float("nan")))
            if int(getattr(model, "SolCount", 0) or 0) > 0
            else float("nan")
        ),
        "num_vars": int(getattr(model, "NumVars", 0) or 0),
        "num_constraints": int(getattr(model, "NumConstrs", 0) or 0),
        "num_nonzeros": int(getattr(model, "NumNZs", 0) or 0),
    }

    if model.status == GRB.OPTIMAL:
        x_lp = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
        if return_stats:
            return x_lp, stats
        return x_lp

    print(f"  警告: 全局 LP 松弛求解失败 (status={model.status})，返回零矩阵", flush=True)
    x_lp = np.zeros((ng, T))
    if return_stats:
        return x_lp, stats
    return x_lp


# ========================== Step 5：可行性验�?==========================

def _get_ordered_ppc_generator_array(ppc: dict, key: str, ng: int) -> Optional[np.ndarray]:
    values = ppc.get(key)
    if values is None:
        return None

    values = np.asarray(values)
    if values.shape[0] != ng:
        return None

    raw_gen = np.asarray(ppc.get('gen'))
    if raw_gen.shape[0] != ng:
        return values

    order = np.argsort(raw_gen[:, GEN_BUS], kind='stable')
    return values[order]


def _get_min_up_down_time_steps(ppc: dict, ng: int, T_delta: float, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    default_steps = min(max(int(4 * T_delta), 1), max(horizon - 1, 0))

    min_up_h = _get_ordered_ppc_generator_array(ppc, 'uc_min_up_time_h', ng)
    min_down_h = _get_ordered_ppc_generator_array(ppc, 'uc_min_down_time_h', ng)
    if min_up_h is None or min_down_h is None:
        return (
            np.full(ng, default_steps, dtype=int),
            np.full(ng, default_steps, dtype=int),
        )

    min_up = np.maximum(np.ceil(np.asarray(min_up_h, dtype=float) / T_delta).astype(int), 1)
    min_down = np.maximum(np.ceil(np.asarray(min_down_h, dtype=float) / T_delta).astype(int), 1)
    min_up = np.minimum(min_up, max(horizon - 1, 0))
    min_down = np.minimum(min_down, max(horizon - 1, 0))
    return min_up, min_down


def _get_ramp_limits_from_ppc(ppc: dict, gen: np.ndarray, T_delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ramp limits aligned with UC training models."""
    default_up = 0.4 * gen[:, PMAX] / T_delta
    default_down = 0.4 * gen[:, PMAX] / T_delta
    default_up_co = 0.3 * gen[:, PMAX]
    default_down_co = 0.3 * gen[:, PMAX]

    ramp_up_h = _get_ordered_ppc_generator_array(ppc, 'uc_ramp_up_mw_per_h', gen.shape[0])
    ramp_down_h = _get_ordered_ppc_generator_array(ppc, 'uc_ramp_down_mw_per_h', gen.shape[0])
    if ramp_up_h is None or ramp_down_h is None:
        return default_up, default_down, default_up_co, default_down_co

    Ru = np.asarray(ramp_up_h, dtype=float) * T_delta
    Rd = np.asarray(ramp_down_h, dtype=float) * T_delta
    Ru = np.maximum(Ru, default_up)
    Rd = np.maximum(Rd, default_down)
    Ru_co = np.maximum(Ru, gen[:, PMIN])
    Rd_co = np.maximum(Rd, gen[:, PMIN])
    return Ru, Rd, Ru_co, Rd_co


def check_commitment_logic_feasibility(
    x_int: np.ndarray,
    ppc: dict,
    T_delta: float,
    unit_ids: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> Tuple[bool, str]:
    """Check binary on/off logic only: binary domain and minimum up/down time."""
    x_arr = np.asarray(x_int, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[np.newaxis, :]

    ng_local, T = x_arr.shape
    if T == 0:
        return False, "commitment horizon is empty"

    rounded = np.rint(x_arr)
    if np.any(np.abs(x_arr - rounded) > tol):
        return False, "commitment contains non-binary values"

    x_bin = rounded.astype(int)
    if np.any((x_bin != 0) & (x_bin != 1)):
        return False, "commitment contains values outside {0, 1}"

    full_ng = ext2int(ppc)['gen'].shape[0]
    if unit_ids is None:
        if ng_local != full_ng:
            return False, "unit_ids are required when checking a subset of units"
        local_unit_ids = np.arange(ng_local, dtype=int)
    else:
        local_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
        if local_unit_ids.size != ng_local:
            return False, "unit_ids length does not match the checked commitment rows"
        if np.any(local_unit_ids < 0) or np.any(local_unit_ids >= full_ng):
            return False, "unit_ids are out of range"

    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, full_ng, T_delta, T)

    for local_idx, unit_id in enumerate(local_unit_ids):
        row = x_bin[local_idx]
        for tau in range(1, int(min_up_steps[unit_id]) + 1):
            for t1 in range(T - tau):
                if row[t1 + 1] - row[t1] > row[t1 + tau] + tol:
                    return False, f"最小开机时间违反: 机组{unit_id}, t1={t1}, tau={tau}"
        for tau in range(1, int(min_down_steps[unit_id]) + 1):
            for t1 in range(T - tau):
                if -row[t1 + 1] + row[t1] > 1 - row[t1 + tau] + tol:
                    return False, f"最小关机时间违反: 机组{unit_id}, t1={t1}, tau={tau}"

    return True, "逻辑可行"


def _sanitize_named_commitment_candidates(
    candidate_specs: List[Tuple[str, np.ndarray]],
    ppc: dict,
    T_delta: float,
    unit_ids: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, str]]]:
    """Repair, validate, and deduplicate commitment candidates before they enter search pools."""
    sanitized: List[Tuple[str, np.ndarray]] = []
    rejected: List[Tuple[str, str]] = []
    seen = set()

    for name, candidate in candidate_specs:
        repaired = _repair_commitment_logic_heuristic(
            candidate,
            T_delta,
            ppc=ppc,
            unit_ids=unit_ids,
        )
        is_valid, reason = check_commitment_logic_feasibility(
            repaired,
            ppc,
            T_delta,
            unit_ids=unit_ids,
        )
        if not is_valid:
            rejected.append((str(name), reason))
            continue

        key = _candidate_key(repaired)
        if key in seen:
            continue
        seen.add(key)
        sanitized.append((str(name), np.asarray(repaired, dtype=int)))

    return sanitized, rejected


def check_uc_feasibility(
    x_int: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    tol: float = 1e-6
) -> Tuple[bool, str]:
    """
    验证给定整数解是否满�?UC 约束�?

    检查顺序：
    1. 最小开关机时间约束（代数检查）
    2. 功率平衡 + 爬坡约束（固�?x，求�?LP 验证 pg 可行性）

    Args:
        x_int: (ng, T) 整数解（值为 0 �?1�?
        ppc: PyPower 案例数据
        pd_data: (nb_load, T) 负荷数据
        T_delta: 时间间隔
        tol: 约束违反容忍�?

    Returns:
        (is_feasible, reason): True 表示可行，否则附带违反原�?
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    ng, T = x_int.shape
    Pd_sum = np.sum(pd_data, axis=0)

    logic_feasible, logic_reason = check_commitment_logic_feasibility(
        x_int,
        ppc,
        T_delta,
        tol=tol,
    )
    if not logic_feasible:
        return False, logic_reason

    # 检�?：功率平�?+ 爬坡（LP 软可行性）
    model = gp.Model('uc_feasibility_check')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    slack_pos = model.addVars(T, lb=0, name='sp')
    slack_neg = model.addVars(T, lb=0, name='sn')

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)

    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x_int[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x_int[g, t])
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1] <= Ru[g] * x_int[g, t-1] + Ru_co[g] * (1 - x_int[g, t-1])
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t] <= Rd[g] * x_int[g, t] + Rd_co[g] * (1 - x_int[g, t])
            )

    for t in range(T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + slack_pos[t] - slack_neg[t]
            == float(Pd_sum[t])
        )

    # 检�?：DC 线路潮流约束（作为硬约束加入 LP�?
    # 功率平衡含松弛变量；�?LP 因线路约束不可行则说明网络拥�?
    _dc_lines_added = False
    try:
        _PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = _PTDF @ pd_data   # (nl, T)
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
        _dc_lines_added = True
    except Exception:
        pass   # DC 约束不可用时退化为仅检查功率平�?

    obj = gp.quicksum(slack_pos[t] + slack_neg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        total_slack = model.ObjVal
        if total_slack > tol:
            return False, f"功率平衡不可行：总松弛量={total_slack:.4f} MW"
        return True, "可行"

    if model.status == GRB.INFEASIBLE and _dc_lines_added:
        return False, "DC flow constraints are infeasible"
    return False, f"功率平衡与爬坡 LP 求解失败: status={model.status}"


def _evaluate_commitment_dispatch_cost(
    x_int: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
) -> Optional[float]:
    """Given an integer commitment, solve the dispatch LP and return UC cost.

    Used by ``recover_integer_solution`` to decide whether a precheck-feasible
    hot-start is "good enough" to accept (relative to the LP lower bound).
    Returns ``None`` if dispatch is itself infeasible (which means the
    commitment is structurally bad even though check_uc_feasibility allowed it
    -- can happen because check_uc_feasibility uses a slack-augmented LP).
    """
    from pypower.idx_gen import PMAX, PMIN

    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng, T = x_int.shape
    Pd_sum = np.sum(pd_data, axis=0)
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    model = gp.Model('uc_dispatch_eval')
    model.Params.OutputFlag = 0
    pg = model.addVars(ng, T, lb=0, name='pg')
    cpower = model.addVars(ng, T, lb=0, name='cpower')
    coc = model.addVars(ng, max(T - 1, 0), lb=0, name='coc')

    for t in range(T):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]))
    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= float(gen[g, PMIN]) * float(x_int[g, t]))
            model.addConstr(pg[g, t] <= float(gen[g, PMAX]) * float(x_int[g, t]))
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                              + gencost[g, -1] / T_delta * float(x_int[g, t])
            )
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1]
                <= Ru[g] * float(x_int[g, t-1]) + Ru_co[g] * (1 - float(x_int[g, t-1]))
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t]
                <= Rd[g] * float(x_int[g, t]) + Rd_co[g] * (1 - float(x_int[g, t]))
            )
            model.addConstr(
                coc[g, t-1] >= float(start_cost[g])
                              * (float(x_int[g, t]) - float(x_int[g, t-1]))
            )
            model.addConstr(
                coc[g, t-1] >= float(shut_cost[g])
                              * (float(x_int[g, t-1]) - float(x_int[g, t]))
            )

    try:
        PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = PTDF @ pd_data
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
    except Exception:
        pass

    obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g, t] for g in range(ng) for t in range(max(T - 1, 0))))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        return None
    return float(model.ObjVal)


# ========================== Step 4：可行性泵主循�?==========================

def run_feasibility_pump(
    x_curr: np.ndarray,
    trusted_mask: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    x_pool: Optional[np.ndarray] = None,
    surrogate_screen_constraints: Optional[List[dict]] = None,
    surrogate_screen_soft_penalty: float = 25.0,
    projection_objective_tau = 'adaptive',
    max_iter: int = 50,
    stall_perturbation_mode: str = 'pool_then_flip',
    stall_flip_fraction: float = 0.10,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    return_history: bool = False,
    rounding_strategy: str = 'x_round',
    chi_alpha: float = 3.0,
    chi_random_samples: int = 8,
    chi_random_evaluator_weight: float = 0.05,
    delta_no_improve_eps: float = 1e-4,
    tabu_list_size: int = 12,
    stall_theta_resample_callback: Optional[Any] = None,
    stall_theta_resample_after_chi_random: int = 2,
    pool_extension_limit: int = 32,
    dump_iis_on_projection_failure: bool = False,
    surrogate_engagement_iter: int = 0,
    noh_milp_refresh_callback: Optional[Any] = None,
    noh_milp_refresh_stall: int = 3,
    noh_milp_refresh_interval: int = 0,
    noh_milp_refresh_max: int = 3,
    enable_pool_tabu_prune: bool = False,
    pool_tabu_drop_threshold: int = 3,
    min_iter_before_feasible_accept: int = 0,
    terminate_on_cycle: bool = False,
    pool_activation_iter: int = 1,
) -> Tuple[np.ndarray, bool] | Tuple[np.ndarray, bool, Dict[str, Any]]:
    """可行性泵主循环（Algorithm II 的核心实现）。

    主要环节与论文公式的对应关系：
      - 公式 (4-12)：LP 投影子问题 S_LP^{(k)}，在 ``x_pool`` 提供时启用候选凸组合
        约束（变量名 ``omega`` 对应论文中的 χ）。
      - 公式 (4-13)：当 ``rounding_strategy='chi_argmax'`` 时，按
        ``q*_g = argmax_j ω[g,j]`` 选择整数候选模式。
      - 公式 (4-25/4-26)：投影目标含 ``projection_objective_tau`` 加权的原始
        成本项（势函数 Φ̂、δ̂_k）。
      - 公式 (4-29/4-30)：``rounding_strategy='chi_random'`` 或停滞后切换到
        ``chi_random`` 时，按 ``ω^α`` 重新抽样并用 Ψ(y) = Δ(y_LP,y)+ω·c_y^T y
        评价以跳出等势循环。
      - 公式 (4-31/4-32)：``stall_theta_resample_callback`` 在持续停滞时被调用
        重新生成候选启停结构并扩展 ``x_pool``。

    Args:
        x_curr: (ng, T) 初始整数点（0/1 矩阵）。
        trusted_mask: (ng, T) bool，True 表示固定不变（传统 FP 应传零矩阵）。
        ppc: PyPower 案例数据。
        pd_data: (nb_load, T) 负荷数据。
        T_delta: 时间间隔。
        x_pool: (n_pool, ng, T) 整数解池，iteration ≥ 1 时启用凸组合（χ）约束。
        max_iter: 最大迭代次数。
        stall_perturbation_mode: 停滞扰动策略，``flip`` / ``pool_restart`` /
            ``pool_then_flip`` / ``chi_random`` / ``chi_random_then_theta``。
        rounding_strategy: 舍入策略，``x_round`` (默认/传统) /
            ``chi_argmax`` (eq 4-13) / ``chi_random`` (eq 4-29 + Ψ 评分)。
        chi_alpha: eq 4-29 中概率调节因子 α。
        chi_random_samples: 每次随机化舍入时的采样数量。
        chi_random_evaluator_weight: Ψ(y) 中 ω·c_y^T y 的权重。
        stall_theta_resample_callback: ``Callable[[np.ndarray], Optional[
            np.ndarray]]``，输入当前 x_curr，返回形如 (n_extra, ng, T) 的整数
            解集合用于扩展 ``x_pool``。
        stall_theta_resample_after_chi_random: 连续随机舍入失败多少次后触发
            θ-扰动重新生成候选。
        pool_extension_limit: ``x_pool`` 允许扩展到的最大行数。

    Returns:
        (x_result, success [, details])。
    """
    if rng is None:
        rng = np.random.default_rng()

    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng, T = x_curr.shape
    Pd_sum = np.sum(pd_data, axis=0)

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, T_delta, T)

    # 预计�?DC 潮流数据（循环外一次性完成，避免重复构建 PTDF�?
    _dc_available = False
    _ptdf_g = None
    _ptdf_Pd = None
    _branch_limit = None
    _active_lines = []
    try:
        _PTDF, _ptdf_g, _branch_limit, _active_lines = _build_ptdf_data(ppc_int)
        _ptdf_Pd = _PTDF @ pd_data   # (nl, T)
        _dc_available = True
        if verbose:
            print(f"  FP: DC flow constraints enabled ({len(_active_lines)} active lines)", flush=True)
    except Exception as _e:
        if verbose:
            print(f"  FP warning: DC flow constraints unavailable ({_e})", flush=True)

    history: List[tuple] = []
    trace: List[Dict[str, Any]] = []
    no_improve_count = 0
    adaptive_tau_reference_obj = _estimate_commitment_primal_objective(
        x_curr,
        ppc,
        pd_data,
        T_delta,
    )

    # Tracking structures for cycle / equipotential detection and the
    # tabu-list-based state revisit avoidance (论文 §4.4 中的 y_tilde^(k) 状态回访).
    tabu_states: List[tuple] = []
    delta_history: List[float] = []
    chi_random_attempts = 0
    pool_extensions_used = 0
    # Per-pool-slot tabu visit counter (key = bytes(x_pool[j])). Incremented
    # whenever a chi-argmax / chi-random round produces a candidate whose key
    # matches an x_pool row and the iteration is flagged as cycle_hit. When
    # the count reaches ``pool_tabu_drop_threshold`` we delete that slot.
    pool_tabu_counts: Dict[bytes, int] = {}
    pool_tabu_drops_applied: int = 0
    noh_milp_refresh_used: int = 0

    rounding_strategy_norm = str(rounding_strategy or 'x_round').strip().lower()
    if rounding_strategy_norm not in (
        'x_round',
        'chi_argmax',
        'chi_random',
    ):
        raise ValueError(f"Unsupported rounding_strategy={rounding_strategy!r}")

    stall_perturbation_norm = str(stall_perturbation_mode or 'pool_then_flip').strip().lower()
    if stall_perturbation_norm not in (
        'flip',
        'pool_restart',
        'pool_then_flip',
        'chi_random',
        'chi_random_then_theta',
        'none',
    ):
        raise ValueError(f"Unsupported stall_perturbation_mode={stall_perturbation_mode!r}")

    chi_alpha = float(max(chi_alpha, 1e-3))
    chi_random_samples = int(max(chi_random_samples, 1))
    chi_random_evaluator_weight = float(max(chi_random_evaluator_weight, 0.0))
    tabu_list_size = int(max(tabu_list_size, 1))

    # Build per-unit cost coefficients used by Ψ(y) (eq 4-30). Gencost columns
    # store the on-cost in column -1 (no-load) and ramp/start in -2; we use the
    # constant per-time on-cost which is the dominant cost contribution of y.
    on_cost_per_t = np.asarray(gencost[:, -1], dtype=float) / float(max(T_delta, 1e-9))

    x_curr = x_curr.copy()

    def _evaluate_psi(y_round: np.ndarray, y_LP_ref: np.ndarray) -> float:
        """Ψ(y) = Δ(y_LP, y) + ω · c_y^T y from eq (4-30).

        Δ is the L1 distance restricted to free variables (matches the
        projection distance used in the FP loop). The objective term is the
        on-cost contribution per generator-hour.
        """
        if y_round.ndim == 1:
            y_round = y_round.reshape(1, -1)
        diff = np.abs(y_LP_ref - y_round.astype(float))
        if trusted_mask is not None:
            diff = diff[~trusted_mask]
        delta_term = float(np.sum(diff))
        cost_term = 0.0
        if chi_random_evaluator_weight > 0.0:
            cost_term = float(np.sum(on_cost_per_t[:, None] * y_round.astype(float)))
        return delta_term + chi_random_evaluator_weight * cost_term

    def _chi_argmax_round(omega_arr: np.ndarray) -> np.ndarray:
        """eq (4-13): q_i* = argmax_q χ_{i,q}, ỹ_i = y_i^{(q_i*)}."""
        argmax_idx = np.argmax(omega_arr, axis=1)
        rounded = np.stack(
            [np.asarray(x_pool[int(j), g, :], dtype=int) for g, j in enumerate(argmax_idx)],
            axis=0,
        )
        return rounded

    def _chi_random_round(omega_arr: np.ndarray, y_LP_ref: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """eq (4-29) + Ψ-评分: 多次 ω^α 抽样并选择 Ψ 最优者。"""
        n_pool_local = omega_arr.shape[1]
        weights = np.clip(omega_arr, 1e-6, None) ** chi_alpha
        weights = weights / weights.sum(axis=1, keepdims=True)
        best_psi = float('inf')
        best_candidate = None
        sample_count = 0
        seen_keys: set = set()
        for _ in range(chi_random_samples):
            sampled_idx = np.array(
                [rng.choice(n_pool_local, p=weights[g]) for g in range(omega_arr.shape[0])],
                dtype=int,
            )
            candidate = np.stack(
                [np.asarray(x_pool[int(j), g, :], dtype=int) for g, j in enumerate(sampled_idx)],
                axis=0,
            )
            key = tuple(candidate.flatten())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            sample_count += 1
            psi_value = _evaluate_psi(candidate, y_LP_ref)
            if psi_value < best_psi:
                best_psi = psi_value
                best_candidate = candidate
        if best_candidate is None:
            best_candidate = _chi_argmax_round(omega_arr)
        return best_candidate, float(best_psi), int(sample_count)

    def _trigger_theta_resample(x_now: np.ndarray) -> int:
        nonlocal x_pool, pool_extensions_used
        if stall_theta_resample_callback is None:
            return 0
        if pool_extensions_used >= pool_extension_limit:
            return 0
        extra = stall_theta_resample_callback(x_now)
        if extra is None:
            return 0
        extra_arr = np.asarray(extra, dtype=int)
        if extra_arr.size == 0:
            return 0
        if extra_arr.ndim == 2:
            extra_arr = extra_arr[None, :, :]
        if x_pool is None:
            x_pool = extra_arr
        else:
            x_pool = np.concatenate([x_pool, extra_arr], axis=0)
        added = int(extra_arr.shape[0])
        pool_extensions_used += added
        if verbose:
            print(
                f"  FP: theta-perturbed candidate regen added {added} entries "
                f"(pool now n={x_pool.shape[0]})",
                flush=True,
            )
        return added

    def _dedupe_against_pool(extra_arr: np.ndarray) -> np.ndarray:
        """Drop rows already present in ``x_pool`` (deduplication helper)."""
        if x_pool is None or extra_arr.size == 0:
            return extra_arr
        existing = {bytes(x_pool[j].astype(np.int8).tobytes()) for j in range(x_pool.shape[0])}
        keep = []
        for j in range(extra_arr.shape[0]):
            key = bytes(extra_arr[j].astype(np.int8).tobytes())
            if key in existing:
                continue
            existing.add(key)
            keep.append(extra_arr[j])
        if not keep:
            return np.zeros((0,) + extra_arr.shape[1:], dtype=int)
        return np.stack(keep, axis=0).astype(int)

    def _trigger_noh_milp_refresh(iteration_idx: int,
                                  stall_counter_val: int,
                                  reason: str) -> int:
        """Re-run sign4+single sub-proxy MILP with perturbed parameters.

        Invoked when the FP gets stuck (stall_counter >= K) OR every M
        iterations. The callback is expected to return a ``(ng, n_new, T)``
        integer array (or a single ``(ng, T)`` slice) representing freshly
        solved unit commitments. They are deduped against ``x_pool`` and
        appended.
        """
        nonlocal x_pool, pool_extensions_used, noh_milp_refresh_used
        if noh_milp_refresh_callback is None:
            return 0
        if noh_milp_refresh_used >= max(0, int(noh_milp_refresh_max)):
            return 0
        if pool_extensions_used >= pool_extension_limit:
            return 0
        try:
            extra = noh_milp_refresh_callback(iteration_idx, stall_counter_val, reason)
        except Exception as exc:
            if verbose:
                print(f"  FP: noh_milp_refresh_callback raised: {exc}", flush=True)
            return 0
        if extra is None:
            return 0
        extra_arr = np.asarray(extra, dtype=int)
        if extra_arr.size == 0:
            return 0
        if extra_arr.ndim == 2:
            extra_arr = extra_arr[None, :, :]
        # ``extra_arr`` may come in as (ng, m, T) from a per-unit MILP solve;
        # detect and transpose so the leading axis is the candidate index.
        if extra_arr.shape[0] == ng and (extra_arr.shape[-1] == T):
            if extra_arr.ndim == 3 and extra_arr.shape[1] != T and extra_arr.shape[0] == ng:
                extra_arr = np.transpose(extra_arr, (1, 0, 2))
        deduped = _dedupe_against_pool(extra_arr)
        if deduped.shape[0] == 0:
            if verbose:
                print("  FP: noh_milp_refresh produced only duplicates", flush=True)
            return 0
        if x_pool is None:
            x_pool = deduped
        else:
            x_pool = np.concatenate([x_pool, deduped], axis=0)
        added = int(deduped.shape[0])
        pool_extensions_used += added
        noh_milp_refresh_used += 1
        if verbose:
            print(
                f"  FP: noH MILP refresh #{noh_milp_refresh_used} added {added} "
                f"candidate(s) (pool now n={x_pool.shape[0]}, reason='{reason}')",
                flush=True,
            )
        return added

    def _drop_pool_tabu_slots() -> int:
        """Remove rows from ``x_pool`` whose tabu count >= threshold."""
        nonlocal x_pool
        if not enable_pool_tabu_prune or x_pool is None:
            return 0
        if int(pool_tabu_drop_threshold) <= 0:
            return 0
        if x_pool.shape[0] <= 1:
            # never strip the last surviving slot -- omega needs at least 1.
            return 0
        keep_mask = np.ones(x_pool.shape[0], dtype=bool)
        for j in range(x_pool.shape[0]):
            key = bytes(x_pool[j].astype(np.int8).tobytes())
            if pool_tabu_counts.get(key, 0) >= int(pool_tabu_drop_threshold):
                keep_mask[j] = False
        if keep_mask.all():
            return 0
        if not keep_mask.any():
            return 0
        dropped = int(np.sum(~keep_mask))
        x_pool = x_pool[keep_mask]
        if verbose:
            print(
                f"  FP: tabu prune dropped {dropped} pool slot(s) "
                f"(pool now n={x_pool.shape[0]})",
                flush=True,
            )
        return dropped

    for iteration in range(max_iter):
        # 检验当前点是否已可�?
        is_feas, reason = check_uc_feasibility(x_curr, ppc, pd_data, T_delta)
        if is_feas and iteration >= max(0, int(min_iter_before_feasible_accept)):
            if verbose:
                print(f"  FP: found feasible solution at iteration {iteration}", flush=True)
            if return_history:
                return x_curr, True, {
                    'history': trace,
                    'iterations': int(iteration),
                    'termination': 'feasible',
                    'final_reason': str(reason),
                }
            return x_curr, True

        # LP Projection：最小化 L1(x, x_curr)，满�?UC 连续约束
        model = gp.Model('fp_projection')
        model.Params.OutputFlag = 0

        pg = model.addVars(ng, T, lb=0, name='pg')
        x = model.addVars(ng, T, lb=0, ub=1, name='x')
        d = model.addVars(ng, T, lb=0, name='d')   # |x - x_curr| 辅助变量
        cpower = model.addVars(ng, T, lb=0, name='cpower')
        coc = model.addVars(ng, max(T - 1, 0), lb=0, name='coc')

        # 功率平衡
        for t in range(T):
            model.addConstr(
                gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
                name=f'pb_{t}'
            )

        # Activate candidate-convex-hull projection after pool_activation_iter.
        # Vanilla uses x_pool=None; theta_flip uses pool_activation_iter=0 so
        # the first projection is already based on proxy-generated candidates.
        omega = None
        n_pool = 0
        pool_active_now = (
            x_pool is not None
            and iteration >= max(0, int(pool_activation_iter))
        )
        if pool_active_now:
            n_pool = x_pool.shape[0]
            omega = model.addVars(ng, n_pool, lb=0, name='omega')
            for g in range(ng):
                model.addConstr(
                    gp.quicksum(omega[g, j] for j in range(n_pool)) == 1.0,
                    name=f'chull_sumo_{g}',
                )
                for t in range(T):
                    model.addConstr(
                        x[g, t] == gp.quicksum(
                            float(x_pool[j, g, t]) * omega[g, j] for j in range(n_pool)
                        ),
                        name=f'chull_x_{g}_{t}',
                    )

        surrogate_screen_slacks = []
        # ``surrogate_engagement_iter > 0`` lets the FP run a few plain
        # iterations before the (potentially noisy) learned surrogate rows
        # are layered in -- a complexity ramp the user can dial up.
        surrogate_screen_active_now = (
            bool(surrogate_screen_constraints)
            and iteration >= max(0, int(surrogate_engagement_iter))
        )
        if surrogate_screen_active_now:
            for row in surrogate_screen_constraints:
                g = int(row['unit_id'])
                if g < 0 or g >= ng:
                    continue
                slack_var = model.addVar(lb=0, name=(
                    f"fp_screen_surr_slack_g{g}_k{int(row['constraint_index'])}_iter{iteration}"
                ))
                expr = (
                    build_surrogate_constraint_expression(
                        {t: x[g, t] for t in range(T)},
                        int(row['timestep']),
                        tuple(row['offsets']),
                        float(row['alpha']),
                        float(row['beta']),
                        float(row['gamma']),
                        T,
                    ) - float(row['delta'])
                )
                model.addConstr(
                    expr <= slack_var,
                    name=(
                        f"fp_screen_surr_g{g}_k{int(row['constraint_index'])}_"
                        f"iter{iteration}"
                    ),
                )
                surrogate_screen_slacks.append(
                    (
                        slack_var,
                        float(max(row.get('coef_scale', 1.0), 1e-6)),
                    )
                )

        for g in range(ng):
            # 发电上下�?
            for t in range(T):
                model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

            # 爬坡约束
            for t in range(1, T):
                model.addConstr(
                    pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1])
                )
                model.addConstr(
                    pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
                )
                model.addConstr(
                    coc[g, t-1] >= float(start_cost[g]) * (x[g, t] - x[g, t-1])
                )
                model.addConstr(
                    coc[g, t-1] >= float(shut_cost[g]) * (x[g, t-1] - x[g, t])
                )

            # 最小开关机时间
            Ton_g = int(min_up_steps[g])
            Toff_g = int(min_down_steps[g])
            for tau in range(1, Ton_g + 1):
                for t1 in range(T - tau):
                    model.addConstr(x[g, t1+1] - x[g, t1] <= x[g, t1+tau])
            for tau in range(1, Toff_g + 1):
                for t1 in range(T - tau):
                    model.addConstr(-x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau])

            # 每个变量�?L1 距离约束 or 固定可信变量
            for t in range(T):
                model.addConstr(
                    cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                                   + gencost[g, -1] / T_delta * x[g, t]
                )
                x_ref = float(x_curr[g, t])
                if trusted_mask[g, t]:
                    model.addConstr(x[g, t] == x_ref)
                    model.addConstr(d[g, t] == 0.0)
                else:
                    model.addConstr(d[g, t] >= x[g, t] - x_ref)
                    model.addConstr(d[g, t] >= x_ref - x[g, t])

        # DC 线路潮流约束（使用预计算�?PTDF 数据�?
        if _dc_available:
            for l in _active_lines:
                limit = float(_branch_limit[l])
                for t in range(T):
                    flow_expr = (
                        gp.quicksum(float(_ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                        - float(_ptdf_Pd[l, t])
                    )
                    model.addConstr(flow_expr <= limit)
                    model.addConstr(flow_expr >= -limit)

        tau_setting = projection_objective_tau
        if isinstance(tau_setting, str):
            tau_mode = tau_setting.strip().lower()
            if tau_mode in ('adaptive', 'auto', ''):
                tau_used = 1.0 / max(float(adaptive_tau_reference_obj), 1.0)
            elif tau_mode in ('none', 'off', 'false', '0'):
                tau_used = 0.0
            else:
                tau_used = float(tau_setting)
        elif tau_setting is None:
            tau_used = 1.0 / max(float(adaptive_tau_reference_obj), 1.0)
        else:
            tau_used = float(tau_setting)

        # 目标：最小化 L1 距离（仅对非可信变量�?
        obj = gp.quicksum(
            d[g, t]
            for g in range(ng)
            for t in range(T)
            if not trusted_mask[g, t]
        )
        if surrogate_screen_slacks:
            obj += float(surrogate_screen_soft_penalty) * gp.quicksum(
                slack_var / scale for slack_var, scale in surrogate_screen_slacks
            )
        if abs(float(tau_used)) > 1e-12:
            obj += float(tau_used) * (
                gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
                + gp.quicksum(coc[g, t] for g in range(ng) for t in range(max(T - 1, 0)))
            )
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        # ------------------------------------------------------------------
        # Omega-fallback: at iter >= 1 the model is augmented with the
        # χ-convex-hull constraint ``x = sum_j omega_j x_pool[j]``. When the
        # available x_pool rows together cannot satisfy DC-line + ramp +
        # min-up/down jointly, the projection becomes infeasible (status=3)
        # even though the un-restricted UC LP relaxation is non-empty (as the
        # probe shows). We relax the convex-hull equalities into inequalities
        # bracketed by an L1 drift variable that is penalised in the
        # objective; this keeps x preferentially inside conv(x_pool) while
        # letting it deviate when DC / ramp would otherwise prevent any
        # solution.
        omega_fallback_used = False
        if (
            model.status != GRB.OPTIMAL
            and omega is not None
            and n_pool > 0
        ):
            if verbose:
                print(
                    f"  FP: projection infeasible at iter {iteration} with "
                    f"omega convex-hull active (n_pool={n_pool}); retrying "
                    "with omega relaxed",
                    flush=True,
                )
            chull_x_constrs = []
            chull_sumo_constrs = []
            for c in list(model.getConstrs()):
                cname = c.ConstrName
                if cname.startswith('chull_x_'):
                    chull_x_constrs.append(c)
                elif cname.startswith('chull_sumo_'):
                    chull_sumo_constrs.append(c)
            for c in chull_x_constrs:
                model.remove(c)
            for c in chull_sumo_constrs:
                model.remove(c)
            drift_terms = []
            for g in range(ng):
                model.addConstr(
                    gp.quicksum(omega[g, j] for j in range(n_pool)) <= 1.0 + 1e-3,
                    name=f'chull_sumo_relax_{g}',
                )
            for g in range(ng):
                for t in range(T):
                    drift_var = model.addVar(lb=0, name=f'chull_drift_{g}_{t}')
                    drift_terms.append(drift_var)
                    hull_expr = gp.quicksum(
                        float(x_pool[j, g, t]) * omega[g, j] for j in range(n_pool)
                    )
                    model.addConstr(
                        x[g, t] - hull_expr <= drift_var,
                        name=f'chull_x_relax_pos_{g}_{t}',
                    )
                    model.addConstr(
                        -x[g, t] + hull_expr <= drift_var,
                        name=f'chull_x_relax_neg_{g}_{t}',
                    )
            penalty = 100.0
            new_obj = model.getObjective() + penalty * gp.quicksum(drift_terms)
            model.setObjective(new_obj, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                omega_fallback_used = True
                if verbose:
                    drift_used = sum(v.X for v in drift_terms)
                    print(
                        f"    omega-relaxed projection OPTIMAL "
                        f"(total drift={drift_used:.3f})",
                        flush=True,
                    )

        if model.status != GRB.OPTIMAL:
            if verbose:
                print(f"  FP: LP projection failed at iteration {iteration} (status={model.status})", flush=True)
            iis_records: List[Dict[str, Any]] = []
            iis_var_records: List[Dict[str, Any]] = []
            if dump_iis_on_projection_failure and model.status == GRB.INFEASIBLE:
                try:
                    model.computeIIS()
                    for c in model.getConstrs():
                        if c.IISConstr:
                            iis_records.append({
                                'name': c.ConstrName,
                                'sense': c.Sense,
                                'rhs': float(c.RHS),
                            })
                    for v in model.getVars():
                        if v.IISLB or v.IISUB:
                            iis_var_records.append({
                                'name': v.VarName,
                                'lb_in_iis': bool(v.IISLB),
                                'ub_in_iis': bool(v.IISUB),
                                'lb': float(v.LB),
                                'ub': float(v.UB),
                            })
                    if verbose:
                        print(
                            f"  FP IIS: {len(iis_records)} constraints, "
                            f"{len(iis_var_records)} variable bounds",
                            flush=True,
                        )
                        for rec in iis_records[:25]:
                            print(f"    IIS-C {rec['name']} {rec['sense']} {rec['rhs']}", flush=True)
                        for rec in iis_var_records[:10]:
                            print(
                                f"    IIS-V {rec['name']} "
                                f"[LB={rec['lb']:.4g} in_iis={rec['lb_in_iis']}, "
                                f"UB={rec['ub']:.4g} in_iis={rec['ub_in_iis']}]",
                                flush=True,
                            )
                except gp.GurobiError as exc:
                    if verbose:
                        print(f"  FP: IIS computation failed: {exc}", flush=True)
            if return_history:
                trace.append({
                    'iteration': int(iteration),
                    'pre_feasible': False,
                    'pre_reason': str(reason),
                    'projection_status': int(model.status),
                    'projection_status_name': _gurobi_status_name(model.status),
                    'termination': 'projection_failed',
                    'iis_constraints': iis_records,
                    'iis_var_bounds': iis_var_records,
                })
            break

        x_LP_proj = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
        primal_obj_value = (
            float(np.sum([[cpower[g, t].X for t in range(T)] for g in range(ng)]))
            + float(np.sum([[coc[g, t].X for t in range(max(T - 1, 0))] for g in range(ng)]))
        )
        adaptive_tau_reference_obj = max(primal_obj_value, 1.0)

        # Capture omega (χ) values for χ-rounding when convex combination is on.
        omega_arr: Optional[np.ndarray] = None
        if omega is not None and n_pool > 0:
            omega_arr = np.array(
                [[omega[g, j].X for j in range(n_pool)] for g in range(ng)],
                dtype=float,
            )
        omega_round_delta = None
        if omega_arr is not None:
            omega_round_delta = float(np.sum(np.abs(omega_arr - np.round(omega_arr))))

        rounding_strategy_used = 'x_round'
        psi_score = None
        chi_random_samples_used = 0
        if rounding_strategy_norm == 'chi_argmax' and omega_arr is not None:
            x_next = _chi_argmax_round(omega_arr)
            rounding_strategy_used = 'chi_argmax'
        elif rounding_strategy_norm == 'chi_random' and omega_arr is not None:
            x_next, psi_score, chi_random_samples_used = _chi_random_round(omega_arr, x_LP_proj)
            rounding_strategy_used = 'chi_random'
        else:
            # 默认/回退：传统逐分量四舍五入
            x_next = round_to_integer(x_LP_proj)
        x_next[trusted_mask] = x_curr[trusted_mask]
        x_rounded_pre_heuristic = np.asarray(x_next, dtype=int).copy()

        # Distance to the *input* integer point used in the projection
        # objective.  This is useful for debugging projection stability, but
        # it is not the standard FP fractional-potential after rounding.
        l1_to_current = float(np.sum(np.abs(
            x_LP_proj[~trusted_mask] - x_curr[~trusted_mask]
        )))
        # Standard FP potential after rounding:
        #   δ_k = Δ(y_LP^{(k)}, \tilde y^{(k)})
        # where \tilde y^{(k)} is the integer commitment obtained by the
        # rounding/χ-selection step before any stall-escape perturbation.
        # This quantity is the one we should plot as "delta".
        delta_to_rounded = float(np.sum(np.abs(
            x_LP_proj[~trusted_mask] - x_next[~trusted_mask]
        )))
        delta_history.append(float(delta_to_rounded))
        soft_penalty = 0.0
        if surrogate_screen_slacks:
            soft_penalty = float(surrogate_screen_soft_penalty) * float(np.sum([
                float(slack_var.X) / scale for slack_var, scale in surrogate_screen_slacks
            ]))
        primal_cost_term = 0.0
        if abs(float(tau_used)) > 1e-12:
            primal_cost_term = float(tau_used) * primal_obj_value
        changed = int(np.sum(x_next != x_curr))

        if verbose:
            print(
                f"  FP iter {iteration}: delta={delta_to_rounded:.3f}, "
                f"L1_to_current={l1_to_current:.3f}, "
                f"soft_penalty={soft_penalty:.3f}, "
                f"tau={tau_used:.4g}, tau_cost={primal_cost_term:.3f}, "
                f"obj_ref={adaptive_tau_reference_obj:.3f}, changed_bits={changed}",
                flush=True,
            )

        # 循环检测：(1) 舍入态在 tabu/history 中重现 → cycle_hit；
        # (2) 最近若干轮 δ_k 平台 → equipotential_cycle（仅用于停滞计数/终止判定，
        #     不再作为触发 pool_restart/flip 等策略的前置条件）。
        rounded_key = tuple(x_next.flatten())
        cycle_hit = rounded_key in tabu_states or rounded_key in history
        equipotential_cycle = False
        if len(delta_history) >= 3:
            recent_delta = delta_history[-3:]
            equipotential_cycle = (
                max(recent_delta) - min(recent_delta) <= delta_no_improve_eps
            )

        stall_detected = bool(cycle_hit or equipotential_cycle)
        terminate_for_cycle = bool(terminate_on_cycle and stall_detected)

        perturbation_applied = False
        perturbation_mode_used = None
        pool_restart_applied = False
        flipped_bits = 0
        chi_random_used_for_stall = False
        theta_resample_added = 0
        coverage_insufficient = False

        # Tabu-prune accounting: only when the rounded integer state repeats.
        if enable_pool_tabu_prune and cycle_hit:
            x_next_key = bytes(np.asarray(x_next, dtype=np.int8).tobytes())
            pool_tabu_counts[x_next_key] = pool_tabu_counts.get(x_next_key, 0) + 1

        # Independent of the stall branches below, give noH MILP refresh a
        # chance at fixed iteration intervals so the surrogate-MILP candidate
        # set keeps growing even without explicit stall detection.
        noh_milp_refresh_added = 0
        if (
            noh_milp_refresh_callback is not None
            and noh_milp_refresh_interval > 0
            and iteration > 0
            and (iteration % int(max(1, noh_milp_refresh_interval)) == 0)
            and noh_milp_refresh_used < int(max(0, noh_milp_refresh_max))
        ):
            noh_milp_refresh_added = _trigger_noh_milp_refresh(
                iteration, no_improve_count, 'interval'
            )

        if stall_detected:
            no_improve_count += 1
            # Trigger sign4+single MILP candidate refresh when stalled enough.
            if (
                noh_milp_refresh_callback is not None
                and no_improve_count >= int(max(1, noh_milp_refresh_stall))
                and noh_milp_refresh_used < int(max(0, noh_milp_refresh_max))
            ):
                added = _trigger_noh_milp_refresh(iteration, no_improve_count, 'stall')
                noh_milp_refresh_added += added

        # Apply stall-escape heuristics as soon as a rounded-state cycle is
        # detected.  Do not wait for equipotential plateau or a stall counter.
        if cycle_hit:
            if enable_pool_tabu_prune:
                _drop_pool_tabu_slots()
            stall_branch = stall_perturbation_norm
            if (
                stall_branch in ('chi_random', 'chi_random_then_theta')
                and omega_arr is not None
            ):
                candidate, psi_value, sample_count = _chi_random_round(omega_arr, x_LP_proj)
                candidate[trusted_mask] = x_curr[trusted_mask]
                if tuple(candidate.flatten()) not in tabu_states:
                    x_next = candidate
                    chi_random_used_for_stall = True
                    chi_random_attempts += 1
                    psi_score = float(psi_value)
                    chi_random_samples_used = sample_count
                    perturbation_applied = True
                    perturbation_mode_used = 'chi_random'
                else:
                    coverage_insufficient = True

            if (
                stall_branch == 'chi_random_then_theta'
                and chi_random_attempts >= stall_theta_resample_after_chi_random
                and stall_theta_resample_callback is not None
            ):
                theta_resample_added = _trigger_theta_resample(x_curr)
                if theta_resample_added > 0:
                    chi_random_attempts = 0
                    perturbation_applied = True
                    perturbation_mode_used = (
                        'theta_resample'
                        if not chi_random_used_for_stall
                        else 'chi_random+theta_resample'
                    )

            if (
                stall_branch in ('pool_restart', 'pool_then_flip')
                and not chi_random_used_for_stall
            ):
                pool_candidate = _select_pool_restart_candidate(x_next, x_pool, trusted_mask, rng)
                if pool_candidate is not None:
                    x_next = pool_candidate
                    pool_restart_applied = True
                    perturbation_applied = True
                    perturbation_mode_used = (
                        'pool_restart' if perturbation_mode_used is None
                        else perturbation_mode_used + '+pool_restart'
                    )

            if (
                stall_branch in ('flip', 'pool_then_flip')
                and not chi_random_used_for_stall
            ):
                free_idx = np.argwhere(~trusted_mask)
                if len(free_idx) > 0:
                    n_flip = max(1, int(np.ceil(len(free_idx) * stall_flip_fraction)))
                    chosen = rng.choice(len(free_idx), size=min(n_flip, len(free_idx)), replace=False)
                    for idx in chosen:
                        g_f, t_f = free_idx[idx]
                        x_next[g_f, t_f] = 1 - x_next[g_f, t_f]
                    flipped_bits = int(len(chosen))
                    perturbation_applied = True
                    perturbation_mode_used = (
                        'flip' if perturbation_mode_used is None
                        else perturbation_mode_used + '+flip'
                    )

            no_improve_count = 0
        elif not stall_detected:
            no_improve_count = 0
            chi_random_attempts = 0

        final_key = tuple(x_next.flatten())
        history.append(final_key)
        if len(history) > 10:
            history.pop(0)
        tabu_states.append(final_key)
        if len(tabu_states) > tabu_list_size:
            tabu_states.pop(0)

        if return_history:
            next_feasible, next_reason = check_uc_feasibility(x_next, ppc, pd_data, T_delta)
            changed_after_heuristic = int(np.sum(x_next != x_curr))
            candidate_pool_size = int(0 if x_pool is None else x_pool.shape[0])
            surrogate_screen_count = int(len(surrogate_screen_constraints or []))
            delta_after_heuristic = float(np.sum(np.abs(
                x_LP_proj[~trusted_mask] - x_next[~trusted_mask]
            )))
            phi_hat = float(delta_to_rounded + soft_penalty + primal_cost_term)
            include_trace_arrays = bool(
                ng * T <= 200
                and (omega_arr is None or omega_arr.size <= 256)
            )
            trace_arrays = {}
            if include_trace_arrays:
                trace_arrays = {
                    'x_curr_matrix': np.asarray(x_curr, dtype=int).tolist(),
                    'x_lp_proj_matrix': np.asarray(x_LP_proj, dtype=float).tolist(),
                    'x_rounded_pre_heuristic_matrix': (
                        np.asarray(x_rounded_pre_heuristic, dtype=int).tolist()
                    ),
                    'x_next_matrix': np.asarray(x_next, dtype=int).tolist(),
                    'omega_matrix': (
                        None if omega_arr is None
                        else np.asarray(omega_arr, dtype=float).tolist()
                    ),
                }
            trace.append({
                'iteration': int(iteration),
                'pre_feasible': False,
                'pre_reason': str(reason),
                'projection_status': int(model.status),
                'projection_status_name': _gurobi_status_name(model.status),
                # Backward-compatible field names now refer to the standard
                # rounded-distance potential.  ``delta_to_current_k`` keeps
                # the old projection-objective distance for diagnostics.
                'l1_projection': float(delta_to_rounded),
                'phi_project': float(delta_to_rounded),
                'phi_hat': phi_hat,
                'delta_k': float(delta_to_rounded),
                'delta_to_rounded_k': float(delta_to_rounded),
                'delta_to_current_k': float(l1_to_current),
                'delta_after_heuristic_k': float(delta_after_heuristic),
                'omega_round_delta_k': (
                    None if omega_round_delta is None else float(omega_round_delta)
                ),
                'delta_hat_k': phi_hat,
                'soft_penalty': float(soft_penalty),
                'surrogate_screen_soft_penalty_weight': float(surrogate_screen_soft_penalty),
                'tau': float(tau_used),
                'projection_objective_tau': str(projection_objective_tau),
                'tau_cost': float(primal_cost_term),
                'primal_objective': float(primal_obj_value),
                'changed_bits': int(changed),
                'changed_bits_after_heuristic': int(changed_after_heuristic),
                'n_trusted': int(np.sum(trusted_mask)),
                'n_free': int(np.size(trusted_mask) - np.sum(trusted_mask)),
                'trusted_bits': int(np.sum(trusted_mask)),
                'free_bits': int(np.size(trusted_mask) - np.sum(trusted_mask)),
                'candidate_convex_hull_active': bool(pool_active_now),
                'pool_activation_iter': int(max(0, pool_activation_iter)),
                'candidate_pool_size': candidate_pool_size,
                'surrogate_screen_active': bool(surrogate_screen_active_now),
                'surrogate_screen_available_count': surrogate_screen_count,
                'surrogate_screen_constraints': (
                    surrogate_screen_count if surrogate_screen_active_now else 0
                ),
                'surrogate_engagement_iter': int(max(0, surrogate_engagement_iter)),
                'noh_milp_refresh_added': int(noh_milp_refresh_added),
                'noh_milp_refresh_used_so_far': int(noh_milp_refresh_used),
                'pool_tabu_prune_active': bool(enable_pool_tabu_prune),
                'pool_tabu_counts_max': (
                    int(max(pool_tabu_counts.values())) if pool_tabu_counts else 0
                ),
                'omega_fallback_used': bool(omega_fallback_used),
                'cycle_hit': bool(cycle_hit),
                'equipotential_cycle': bool(equipotential_cycle),
                'rounded_state_revisited': bool(cycle_hit),
                'final_state_revisited': bool(final_key in history[:-1]),
                'perturbation_applied': bool(perturbation_applied),
                'perturbation_mode': perturbation_mode_used,
                'pool_restart_applied': bool(pool_restart_applied),
                'flipped_bits': int(flipped_bits),
                'chi_random_used_for_stall': bool(chi_random_used_for_stall),
                'chi_random_attempts': int(chi_random_attempts),
                'theta_resample_added': int(theta_resample_added),
                'coverage_insufficient': bool(coverage_insufficient),
                'rounding_strategy': str(rounding_strategy_used),
                'rounding_psi_score': None if psi_score is None else float(psi_score),
                'chi_random_samples_used': int(chi_random_samples_used),
                'post_feasible': bool(next_feasible),
                'post_reason': str(next_reason),
                **trace_arrays,
            })

        if terminate_for_cycle:
            if verbose:
                why = 'cycle_hit' if cycle_hit else 'equipotential_cycle'
                print(f"  FP: terminate on {why} at iteration {iteration}", flush=True)
            if return_history:
                trace[-1]['termination'] = (
                    'cycle_detected' if cycle_hit else 'equipotential_cycle'
                )
                return x_next, False, {
                    'history': trace,
                    'iterations': int(len(trace)),
                    'termination': trace[-1]['termination'],
                    'final_reason': trace[-1].get('post_reason', ''),
                }
            return x_next, False

        x_curr = x_next

    if verbose:
        print(f"  FP: reached max_iter={max_iter} without feasibility", flush=True)
    if return_history:
        return x_curr, False, {
            'history': trace,
            'iterations': int(len(trace)),
            'termination': 'max_iter_or_failed_projection',
            'final_reason': trace[-1].get('post_reason', '') if trace else '',
        }
    return x_curr, False


# ========================== Step 6b：History-aware theta-flip recovery ==========================

def _load_error_bit_map(
    source: Union[str, Path, Dict[str, Any]],
) -> Dict[str, Any]:
    """Load an error-bit map dict from a JSON path or pass through if dict."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def recover_via_theta_flip(
    pd_data: Union[np.ndarray, dict],
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_predictor,
    ppc: dict,
    T_delta: float,
    agent,
    error_bit_map: Union[str, Path, Dict[str, Any]],
    *,
    flip_top_k: int = 10,
    harvest_full_lp: bool = True,
    initial_bcd_proxy_scope: str = 'both',
    surrogate_constraint_scope_full: str = 'all',
    bcd_proxy_scope_full: str = 'both',
    defer_full_lp_to_fp: bool = True,
    full_lp_inject_iter: int = 3,
    rescue_when_infeasible: bool = True,
    rescue_max_combinations: int = 30,
    rescue_max_combo_size: int = 3,
    rescue_combo_pool_size: Optional[int] = None,
    run_fp_iterations: bool = False,
    fp_max_iter: int = 10,
    fp_min_iter: int = 3,
    manager=None,
    sparse_library: Optional[SparseSurrogateLibrary] = None,
    sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    verbose: bool = False,
    return_details: bool = False,
) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, Dict[str, Any]]]:
    """History-aware, one-shot recovery without FP iteration.

    Core flow (per user spec)::

        1. theta-only proxy LP                  →  round + repair    → main candidate
        2. full BCD+subproblem proxy LP (opt.)  →  round + repair    → harvested candidate
        3. for each historically-error-prone (g, t) in ``error_bit_map``,
           flip ``x_round_theta[g, t]`` and repair                   → flipped candidates
        4. UC-feasibility check on all candidates; pick cheapest by dispatch cost

    Args:
        pd_data: scenario dict (with ``Pd``/``pd_data``) or load matrix ``(nb, T)``.
        trainers: per-unit surrogate trainers (needed even for the initial LP to
            know the unit count; constraints disabled via scope='none').
        lambda_predictor: object with ``predict(sample) -> lambda_vec``.
        agent: trained BCD agent (theta/zeta proxy rows source). REQUIRED.
        error_bit_map: dict (already loaded) or path to JSON produced by
            ``scripts/build_history_error_bit_map.py``.
        flip_top_k: take the top-K most error-prone bits from the map.
        initial_bcd_proxy_scope: BCD proxy rows used for the initial LP;
            ``'both'`` uses the full theta+zeta BCD model, while ``'theta'``
            reproduces the older theta-only variant.
        harvest_full_lp: also solve one full BCD+subproblem LP as harvest.

    Returns:
        ``(x_rec, ok)`` or ``(x_rec, ok, details)`` if ``return_details``.
    """
    if agent is None:
        raise ValueError("recover_via_theta_flip requires a non-None BCD `agent` "
                         "for the initial BCD proxy LP")

    map_data = _load_error_bit_map(error_bit_map)
    top_bits = list(map_data.get('top_k_bits') or [])

    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        pd_matrix = get_sample_net_load(sample)
        scenario_input = sample
    else:
        pd_matrix = np.asarray(pd_data, dtype=float)
        sample = pd_matrix
        scenario_input = pd_matrix

    try:
        lambda_val = lambda_predictor.predict(sample)
    except Exception as exc:
        raise RuntimeError(f"lambda_predictor.predict failed: {exc}") from exc

    candidates: List[Tuple[str, np.ndarray]] = []

    def _solve_lp(scope_surr: str,
                  scope_bcd: str,
                  *,
                  return_stats: bool = False):
        if manager is not None:
            x_lp = manager.solve_global(
                scenario_input, lambda_val,
                sparse_library=sparse_library,
                sparse_x_template_library=sparse_x_template_library,
                surrogate_constraint_scope=scope_surr,
                bcd_proxy_scope=scope_bcd,
            )
            if return_stats:
                return x_lp, {
                    'surrogate_constraint_scope': str(scope_surr),
                    'bcd_proxy_scope': str(scope_bcd),
                    'stats_unavailable_reason': (
                        'manager.solve_global does not expose solve stats'
                    ),
                }
            return x_lp
        return solve_global_LP_relaxation(
            ppc, scenario_input, T_delta, trainers, lambda_val,
            agent=agent,
            sparse_library=sparse_library,
            sparse_x_template_library=sparse_x_template_library,
            surrogate_constraint_scope=scope_surr,
            bcd_proxy_scope=scope_bcd,
            return_stats=return_stats,
        )

    initial_bcd_proxy_scope_norm = str(initial_bcd_proxy_scope or 'both').strip().lower()
    if initial_bcd_proxy_scope_norm not in {'both', 'theta', 'zeta', 'none'}:
        raise ValueError(
            f"Unsupported initial_bcd_proxy_scope={initial_bcd_proxy_scope!r}"
        )

    # ---- 1) Initial LP: no subproblem surrogate, configurable BCD proxy rows ----
    if verbose:
        print(
            "[theta_flip] Step 1: initial BCD proxy LP relaxation "
            f"(bcd_scope={initial_bcd_proxy_scope_norm}) ...",
            flush=True,
        )
    x_LP_theta, initial_lp_stats = _solve_lp(
        scope_surr='none',
        scope_bcd=initial_bcd_proxy_scope_norm,
        return_stats=True,
    )
    x_round_theta = round_to_integer(np.asarray(x_LP_theta, dtype=float))
    x_round_theta = _repair_min_up_down_heuristic(
        np.asarray(x_round_theta, dtype=int), T_delta, ppc=ppc, unit_ids=None,
    )
    candidates.append(("theta_lp_round", x_round_theta))

    # ---- 2) Full surrogate LP harvest (optional) ----
    # In the iterative theta-flip strategy, the full global proxy LP should
    # not be a starting hot-start. It is instead delayed and injected into
    # the FP pool after a few projection/rounding iterations as a perturbation
    # candidate. This matches the intended flow:
    #   initial BCD proxy -> round/repair -> FP iterations -> full-LP injection.
    deferred_full_lp_candidate: Optional[np.ndarray] = None
    if harvest_full_lp:
        if verbose:
            print("[theta_flip] Step 2: full BCD+subproblem LP harvest ...", flush=True)
        try:
            x_LP_full = _solve_lp(
                scope_surr=surrogate_constraint_scope_full,
                scope_bcd=bcd_proxy_scope_full,
            )
            x_round_full = round_to_integer(np.asarray(x_LP_full, dtype=float))
            x_round_full = _repair_min_up_down_heuristic(
                np.asarray(x_round_full, dtype=int), T_delta, ppc=ppc, unit_ids=None,
            )
            if (
                run_fp_iterations
                and defer_full_lp_to_fp
                and int(full_lp_inject_iter) > 0
            ):
                deferred_full_lp_candidate = np.asarray(x_round_full, dtype=int)
            else:
                candidates.append(("full_lp_round", x_round_full))
        except Exception as exc:
            if verbose:
                print(f"[theta_flip] Step 2 full LP failed: {exc}", flush=True)

    # ---- 3) Flip top-K historical error-prone bits on x_round_theta ----
    n_flipped = 0
    if verbose:
        print(f"[theta_flip] Step 3: flip top-{flip_top_k} historical error bits ...",
              flush=True)
    ng, T = x_round_theta.shape
    for r in top_bits[: max(0, int(flip_top_k))]:
        try:
            g = int(r.get('g', -1))
            t = int(r.get('t', -1))
            rate = float(r.get('error_rate', 0.0))
        except Exception:
            continue
        if not (0 <= g < ng and 0 <= t < T):
            continue
        if rate <= 0.0:
            continue
        c = x_round_theta.copy()
        c[g, t] = 1 - int(c[g, t])
        c_rep = _repair_min_up_down_heuristic(
            np.asarray(c, dtype=int), T_delta, ppc=ppc, unit_ids=None,
        )
        candidates.append((f"flip_g{g}_t{t}_p{rate:.2f}", c_rep))
        n_flipped += 1

    # ---- 4) Evaluate all candidates: UC-feasibility + dispatch cost ----
    if verbose:
        print(f"[theta_flip] Step 4: evaluating {len(candidates)} candidates ...",
              flush=True)
    eval_log: List[Dict[str, Any]] = []
    best_name: Optional[str] = None
    best_x: Optional[np.ndarray] = None
    best_cost: float = float('inf')
    best_reason: str = ''
    for name, x_cand in candidates:
        ok, reason = check_uc_feasibility(x_cand, ppc, pd_matrix, T_delta)
        cost: Optional[float] = None
        if ok:
            cost = _evaluate_commitment_dispatch_cost(
                x_cand, ppc, pd_matrix, T_delta,
            )
            if cost is not None and float(cost) < best_cost:
                best_name = name
                best_x = np.asarray(x_cand, dtype=int)
                best_cost = float(cost)
                best_reason = reason
        eval_log.append({
            'name': name,
            'feasible': bool(ok),
            'reason': reason,
            'uc_cost': None if cost is None else float(cost),
        })
        if verbose:
            print(f"  candidate [{name:30s}] feasible={ok}  uc_cost={cost}",
                  flush=True)

    success = best_x is not None

    # Optional iterative variant requested for paper diagnostics:
    # use theta-only LP, full-proxy harvest, round-repair and historical
    # flip candidates only as the initial commitment pool, then run the
    # actual FP projection/rounding loop for a controlled number of steps.
    # This keeps the "progress over iterations" signal visible in plots,
    # instead of short-circuiting at the first feasible candidate.
    if run_fp_iterations:
        # Add multi-bit sensitive flips to the candidate pool as well.  These
        # are not merely a last-resort feasibility rescue: in practice the
        # optimal commitment is often a 2-3 bit correction of theta/full LP
        # rounding, so the FP should be allowed to project against these rows.
        import itertools  # local import keeps top of module clean
        pool_n = (
            rescue_combo_pool_size
            if rescue_combo_pool_size is not None
            else max(int(flip_top_k), 6)
        )
        bit_pool: List[Tuple[int, int, float]] = []
        for r in top_bits[: pool_n]:
            try:
                g_b = int(r.get('g', -1))
                t_b = int(r.get('t', -1))
                rate_b = float(r.get('error_rate', 0.0))
            except Exception:
                continue
            if 0 <= g_b < ng and 0 <= t_b < T and rate_b > 0.0:
                bit_pool.append((g_b, t_b, rate_b))

        combo_added = 0
        combo_budget = max(0, int(rescue_max_combinations))
        for combo_size in range(2, max(2, int(rescue_max_combo_size)) + 1):
            if combo_added >= combo_budget:
                break
            for combo in itertools.combinations(bit_pool, combo_size):
                if combo_added >= combo_budget:
                    break
                c = x_round_theta.copy()
                tag_parts = []
                for (g_b, t_b, _rate_b) in combo:
                    c[g_b, t_b] = 1 - int(c[g_b, t_b])
                    tag_parts.append(f"g{g_b}t{t_b}")
                c_rep = _repair_min_up_down_heuristic(
                    np.asarray(c, dtype=int), T_delta, ppc=ppc, unit_ids=None,
                )
                cand_name = (
                    "combo_k{ks}_{tg}".format(
                        ks=combo_size, tg="_".join(tag_parts)
                    )
                )
                candidates.append((cand_name, c_rep))
                combo_added += 1

                ok_c, reason_c = check_uc_feasibility(
                    c_rep, ppc, pd_matrix, T_delta,
                )
                cost_c: Optional[float] = None
                if ok_c:
                    cost_c = _evaluate_commitment_dispatch_cost(
                        c_rep, ppc, pd_matrix, T_delta,
                    )
                    if cost_c is not None and float(cost_c) < best_cost:
                        best_name = cand_name
                        best_x = np.asarray(c_rep, dtype=int)
                        best_cost = float(cost_c)
                        best_reason = reason_c
                        success = True
                eval_log.append({
                    'name': cand_name,
                    'feasible': bool(ok_c),
                    'reason': reason_c,
                    'uc_cost': None if cost_c is None else float(cost_c),
                })

        pool_items: List[np.ndarray] = []
        seen_pool = set()
        for _name, cand in candidates:
            arr = np.asarray(cand, dtype=int)
            key = bytes(arr.astype(np.int8).tobytes())
            if key in seen_pool:
                continue
            seen_pool.add(key)
            pool_items.append(arr)
        x_pool_arr = (
            np.stack(pool_items, axis=0)
            if pool_items else x_round_theta[None, :, :]
        )
        if verbose:
            print(
                f"[theta_flip] Step 5: run FP iterations "
                f"(pool={x_pool_arr.shape[0]}, max_iter={fp_max_iter}, "
                f"min_iter={fp_min_iter}, "
                f"full_lp_inject_iter={full_lp_inject_iter if deferred_full_lp_candidate is not None else 'off'})",
                flush=True,
            )

        full_lp_injected = {'done': False}

        def _deferred_full_lp_refresh(
            iteration_idx: int,
            stall_counter_val: int,
            reason: str,
        ) -> Optional[np.ndarray]:
            if deferred_full_lp_candidate is None:
                return None
            if full_lp_injected['done']:
                return None
            if int(iteration_idx) < int(full_lp_inject_iter):
                return None
            full_lp_injected['done'] = True
            if verbose:
                print(
                    f"[theta_flip] inject full_lp_round at FP iter {iteration_idx} "
                    f"(reason={reason})",
                    flush=True,
                )
            return deferred_full_lp_candidate[None, :, :]

        x_fp, ok_fp, fp_details = run_feasibility_pump(
            np.asarray(x_round_theta, dtype=int),
            np.zeros_like(x_round_theta, dtype=bool),
            ppc,
            pd_matrix,
            T_delta,
            x_pool=x_pool_arr,
            max_iter=max(1, int(fp_max_iter)),
            min_iter_before_feasible_accept=max(0, int(fp_min_iter)),
            stall_perturbation_mode='pool_then_flip',
            stall_flip_fraction=0.10,
            rng=np.random.default_rng(42),
            verbose=verbose,
            return_history=True,
            rounding_strategy='chi_argmax',
            enable_pool_tabu_prune=True,
            pool_tabu_drop_threshold=3,
            pool_activation_iter=0,
            noh_milp_refresh_callback=(
                _deferred_full_lp_refresh
                if deferred_full_lp_candidate is not None else None
            ),
            noh_milp_refresh_interval=max(1, int(full_lp_inject_iter))
                if deferred_full_lp_candidate is not None else 0,
            noh_milp_refresh_stall=999999,
            noh_milp_refresh_max=1,
        )
        fp_cost = None
        if ok_fp:
            fp_cost = _evaluate_commitment_dispatch_cost(
                x_fp, ppc, pd_matrix, T_delta,
            )
        # Preserve the best feasible incumbent found before/during the
        # iterative FP.  Without this guard, forcing a few FP iterations for
        # plotting can move away from an already optimal injected candidate.
        selected_name = 'theta_flip_fp_iter'
        selected_x = np.asarray(x_fp, dtype=int)
        selected_ok = bool(ok_fp)
        selected_cost = fp_cost
        if best_x is not None and (
            selected_cost is None or float(best_cost) <= float(selected_cost)
        ):
            selected_name = str(best_name or 'theta_flip_incumbent')
            selected_x = np.asarray(best_x, dtype=int)
            selected_ok = True
            selected_cost = float(best_cost)
        details = {
            'selected': selected_name,
            'selected_uc_cost': None if selected_cost is None else float(selected_cost),
            'candidates': eval_log,
            'n_candidates': len(candidates),
            'n_flipped': n_flipped,
            'harvest_full_lp_enabled': bool(harvest_full_lp),
            'flip_top_k': int(flip_top_k),
            'error_map_top_k_used': int(min(flip_top_k, len(top_bits))),
            'error_map_metadata': map_data.get('metadata', {}),
            'initial_bcd_proxy_scope': initial_bcd_proxy_scope_norm,
            'initial_lp_stats': initial_lp_stats,
            'run_fp_iterations': True,
            'fp_histories': [{
                'name': 'theta_flip_pool',
                'history': fp_details.get('history', []),
                'iterations': fp_details.get('iterations', 0),
                'termination': fp_details.get('termination'),
                'final_reason': fp_details.get('final_reason'),
            }],
            'theta_flip_pool_size': int(x_pool_arr.shape[0]),
            'theta_flip_combo_added': int(combo_added),
            'full_lp_deferred_to_fp': bool(deferred_full_lp_candidate is not None),
            'full_lp_inject_iter': int(full_lp_inject_iter),
            'full_lp_injected': bool(full_lp_injected['done']),
            'fp_returned_feasible': bool(ok_fp),
            'fp_returned_uc_cost': None if fp_cost is None else float(fp_cost),
            'incumbent_preserved': bool(selected_name != 'theta_flip_fp_iter'),
        }
        if return_details:
            return selected_x, selected_ok, details
        return selected_x, selected_ok

    # ---- 5) Multi-bit sensitivity rescue when all candidates infeasible ----
    # When no single-flip or LP-rounded candidate passes UC-feasibility, the
    # current x_pool's "convex hull" cannot generate a feasible commitment.
    # Iteratively flip *combinations* of historically error-prone bits on top
    # of x_round_theta until either (a) a UC-feasible commitment is found,
    # or (b) the combinatorial budget is exhausted. This expands the search
    # by relaxing the most-sensitive on/off variables back to free 0-1
    # variables and exploring their alternate assignment.
    rescue_attempted = 0
    rescue_log: List[Dict[str, Any]] = []
    if (not success) and rescue_when_infeasible and len(top_bits) >= 2:
        import itertools  # local import keeps top of module clean
        pool_n = (rescue_combo_pool_size
                  if rescue_combo_pool_size is not None
                  else max(int(flip_top_k), 6))
        bit_pool: List[Tuple[int, int, float]] = []
        for r in top_bits[: pool_n]:
            try:
                g_b = int(r.get('g', -1))
                t_b = int(r.get('t', -1))
                rate_b = float(r.get('error_rate', 0.0))
            except Exception:
                continue
            if not (0 <= g_b < ng and 0 <= t_b < T):
                continue
            if rate_b <= 0.0:
                continue
            bit_pool.append((g_b, t_b, rate_b))
        if verbose:
            print(f"[theta_flip] Step 5: rescue loop "
                  f"(combo_size 2..{rescue_max_combo_size}, "
                  f"bit_pool={len(bit_pool)}, budget={rescue_max_combinations})",
                  flush=True)

        budget_left = max(1, int(rescue_max_combinations))
        for combo_size in range(2, max(2, int(rescue_max_combo_size)) + 1):
            if success or budget_left <= 0:
                break
            for combo in itertools.combinations(bit_pool, combo_size):
                if budget_left <= 0:
                    break
                c = x_round_theta.copy()
                tag_parts = []
                for (g_b, t_b, rate_b) in combo:
                    c[g_b, t_b] = 1 - int(c[g_b, t_b])
                    tag_parts.append(f"g{g_b}t{t_b}")
                c_rep = _repair_min_up_down_heuristic(
                    np.asarray(c, dtype=int), T_delta, ppc=ppc, unit_ids=None,
                )
                ok_r, reason_r = check_uc_feasibility(
                    c_rep, ppc, pd_matrix, T_delta,
                )
                cost_r: Optional[float] = None
                if ok_r:
                    cost_r = _evaluate_commitment_dispatch_cost(
                        c_rep, ppc, pd_matrix, T_delta,
                    )
                cand_name = (
                    "rescue_k{ks}_{tg}".format(
                        ks=combo_size, tg="_".join(tag_parts)
                    )
                )
                rescue_log.append({
                    'name': cand_name,
                    'feasible': bool(ok_r),
                    'reason': reason_r,
                    'uc_cost': None if cost_r is None else float(cost_r),
                    'combo_size': int(combo_size),
                })
                rescue_attempted += 1
                budget_left -= 1
                if verbose and (rescue_attempted % 5 == 0 or ok_r):
                    print(f"  rescue [{cand_name:40s}] feasible={ok_r}  "
                          f"uc_cost={cost_r}", flush=True)
                if ok_r and cost_r is not None and float(cost_r) < best_cost:
                    best_name = cand_name
                    best_x = np.asarray(c_rep, dtype=int)
                    best_cost = float(cost_r)
                    best_reason = reason_r
                    success = True
                    break  # stop at first feasible rescue (smallest combo)
            if success:
                break

    if success:
        x_rec = best_x  # type: ignore[assignment]
    else:
        x_rec = np.asarray(x_round_theta, dtype=int)

    details = {
        'selected': best_name,
        'selected_uc_cost': float(best_cost) if success else None,
        'selected_reason': best_reason,
        'candidates': eval_log,
        'n_candidates': len(candidates),
        'n_flipped': n_flipped,
        'harvest_full_lp_enabled': bool(harvest_full_lp),
        'flip_top_k': int(flip_top_k),
        'error_map_top_k_used': int(min(flip_top_k, len(top_bits))),
        'error_map_metadata': map_data.get('metadata', {}),
        'initial_bcd_proxy_scope': initial_bcd_proxy_scope_norm,
        'initial_lp_stats': initial_lp_stats,
        # Step-5 rescue diagnostics
        'rescue_enabled': bool(rescue_when_infeasible),
        'rescue_attempted': int(rescue_attempted),
        'rescue_max_combo_size': int(rescue_max_combo_size),
        'rescue_log': rescue_log,
        'success_via_rescue': bool(
            success and isinstance(best_name, str) and best_name.startswith('rescue_')
        ),
    }
    if verbose:
        print(f"[theta_flip] DONE: success={success}  "
              f"selected={best_name}  uc_cost={best_cost if success else None}  "
              f"rescue_attempted={rescue_attempted}",
              flush=True)

    if return_details:
        return x_rec, success, details
    return x_rec, success


# ========================== Step 6：顶层接�?==========================

def recover_integer_solution(
    pd_data: np.ndarray | dict,
    trainers: Dict[int, 'SubproblemSurrogateTrainer'],
    lambda_predictor,
    ppc: dict,
    T_delta: float,
    agent=None,
    manager=None,
    sparse_library: Optional[SparseSurrogateLibrary] = None,
    sparse_x_template_library: Optional[SparseConstraintTemplateLibrary] = None,
    n_perturbations: int = 5,
    n_similar_scenarios: int = 0,
    similar_scenario_pool_size: int = 10,
    n_load_perturbations: int = 0,
    load_perturbation_scale: float = 0.03,
    conf_threshold: float = 0.15,
    max_fp_iter: int = 50,
    perturb_std: float = 0.1,
    neighborhood_weight: float = 0.35,
    use_hot_start: bool = True,
    stall_perturbation_mode: str = 'pool_then_flip',
    stall_flip_fraction: float = 0.10,
    verbose: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_perturbation_hot_starts: int = 6,
    max_unit_options_per_generator: int = 4,
    max_unit_combination_candidates: int = 12,
    max_nearby_commitment_hot_starts: int = 4,
    nearby_commitment_pool_size: int = 12,
    lean_hot_starts: bool = False,
    parallel_fp_starts: int = 1,
    scenario_bank: Optional[List[dict]] = None,
    surrogate_screen_mode: str = 'robust',
    surrogate_screen_max_constraints_per_unit: int = 3,
    surrogate_screen_min_support_ratio: float = 0.85,
    surrogate_screen_max_normalized_violation: float = 0.05,
    surrogate_screen_min_mean_margin: float = 0.02,
    surrogate_screen_candidate_violation_tol: float = 0.02,
    surrogate_screen_soft_penalty: float = 25.0,
    projection_objective_tau = 'adaptive',
    use_subproblem_milp_candidate: bool = True,
    subproblem_milp_for_perturbations: bool = False,
    skip_feasible_hot_starts: bool = False,
    surrogate_constraint_scope: str = "all",
    bcd_proxy_scope: str = "both",
    return_details: bool = False,
    fp_strategy: str = 'tailored',
    rounding_strategy: str = 'chi_argmax',
    chi_alpha: float = 3.0,
    chi_random_samples: int = 8,
    chi_random_evaluator_weight: float = 0.05,
    enable_stall_theta_resample: bool = True,
    stall_theta_resample_after_chi_random: int = 2,
    stall_theta_resample_load_perturb_scale: float = 0.05,
    stall_theta_resample_max_units: int = 2,
    precheck_objective_gap_threshold: Optional[float] = None,
    lp_lower_bound_obj: Optional[float] = None,
    disable_historical_hot_starts: bool = False,
    surrogate_engagement_iter: int = 0,
    noh_milp_refresh_stall: int = 0,
    noh_milp_refresh_interval: int = 0,
    noh_milp_refresh_max: int = 0,
    noh_milp_refresh_lambda_sigma: float = 0.01,
    enable_pool_tabu_prune: bool = False,
    pool_tabu_drop_threshold: int = 3,
    min_fp_iter_before_feasible_accept: int = 0,
    terminate_on_cycle: bool = False,
):
    """
    顶层接口：从 LP 松弛解恢�?UC 整数可行解�?

    Pipeline�?
      1. 通过 lambda_predictor 获取对偶变量
      2. 求解全局 UC LP 松弛（含代理约束）→ x_LP，启发式四舍五入 �?x_init
         （若提供 manager，则使用 manager.solve_global 替代默认 LP 求解器）
      3. 各机组子问题 LP�? 参数扰动）收�?surrogate LP 连续解与多组整数�?
      4. 识别高可信度变量（整数性强 + 多来源一致）
      5. 基于全局 LP / surrogate LP 构造多组热启动候�?
      6. 可行性泵：按热启动候选优先级依次尝试

    Args:
        pd_data: (nb_load, T) 负荷数据
        trainers: {unit_id: SubproblemSurrogateTrainer} 已训练的代理约束训练�?
        lambda_predictor: 对偶变量预测器，需支持返回全局 dual dict 或可兼容的旧格式
        ppc: PyPower 案例数据
        T_delta: 时间间隔（小时）
        manager: （可选）UnifiedSurrogateManager 实例；若提供则使用其
            solve_global 方法求解全局 LP 松弛（同时包�?theta/zeta �?V3 代理约束�?
        sparse_library: （可选）离线筛选出的稀疏参数化约束库，仅在显式提供时启�?
        sparse_x_template_library: （可选）稀�?x 支持集模板库，仅在显式提供时启用
        n_perturbations: 直接随机�?surrogate 参数的次�?        n_similar_scenarios: 检索相似负�?新能源场景并重过 NN 的次�?        similar_scenario_pool_size: 相似场景检索的候选池大小
        n_load_perturbations: 小负荷扰动后重过 NN 的次�?        load_perturbation_scale: 小负荷扰动幅�?        conf_threshold: LP 整数性置信阈�?        max_fp_iter: 可行性泵最大迭代次�?        perturb_std: surrogate 网络输出参数扰动标准�?        neighborhood_weight: 周边平均约束权重
        use_hot_start: 是否启用基于 LP �?surrogate LP 的热启动候�?        stall_perturbation_mode: FP 停滞时的扰动策略
        stall_flip_fraction: FP 停滞时随机翻转比�?        verbose: 是否打印进度
        rng: 随机数生成器（传 None 则使用固定种�?42�?
        scenario_bank: 可选历史样本库。若提供，则优先从这里检索邻近样本最优开机解；
            否则回退到 trainers[*].active_set_data
    Returns:
        (x_feasible, success): 整数解矩�?(ng, T)，以及是否为可行�?
    """
    sample = None
    if isinstance(pd_data, dict):
        sample = normalize_sample_arrays(dict(pd_data))
        scenario_input = sample
        pd_data = get_sample_net_load(sample)
    else:
        scenario_input = pd_data

    if rng is None:
        rng = np.random.default_rng(42)
    surrogate_constraint_scope_norm = _normalize_surrogate_constraint_scope(surrogate_constraint_scope)
    bcd_proxy_scope_norm = str(bcd_proxy_scope or "both").strip().lower()
    if bcd_proxy_scope_norm not in {"both", "theta", "zeta", "none"}:
        raise ValueError(f"Unsupported bcd_proxy_scope={bcd_proxy_scope!r}")

    # ----- FP strategy normalization -----
    # ``fp_strategy='vanilla'`` short-circuits all tailored enhancements so the
    # caller can benchmark against the textbook feasibility pump (Algorithm I).
    # ``'tailored'`` keeps every enhancement; ``'tailored_no_theta'`` disables
    # only the in-stall θ-perturbation regen (eq 4-31/4-32).
    fp_strategy_norm = str(fp_strategy or 'tailored').strip().lower()
    if fp_strategy_norm not in ('tailored', 'tailored_no_theta', 'vanilla'):
        raise ValueError(f"Unsupported fp_strategy={fp_strategy!r}")
    rounding_strategy_norm = str(rounding_strategy or 'chi_argmax').strip().lower()
    if fp_strategy_norm == 'vanilla':
        # Force vanilla settings regardless of caller's input.
        conf_threshold = 0.0  # nothing is "trusted"
        use_hot_start = False
        use_subproblem_milp_candidate = False
        n_perturbations = 0
        n_similar_scenarios = 0
        n_load_perturbations = 0
        max_nearby_commitment_hot_starts = 0
        nearby_commitment_pool_size = 0
        parallel_fp_starts = 1
        surrogate_screen_mode = 'none'
        surrogate_screen_soft_penalty = 0.0
        projection_objective_tau = 'none'
        rounding_strategy_norm = 'x_round'
        stall_perturbation_mode = 'flip'
        enable_stall_theta_resample = False
        # A textbook vanilla FP must not use learned proxy rows for its
        # initial LP relaxation.  Otherwise the "vanilla" trace can fail in
        # the proxy-constrained LP before it ever produces a meaningful
        # projection/rounding delta curve.
        surrogate_constraint_scope_norm = 'none'
        bcd_proxy_scope_norm = 'none'
        agent = None
        manager = None

    # If the caller asked us not to use historical neighbour commitments
    # (so the FP can't "cheat" by retrieving a stored optimum), shut every
    # related candidate generator off regardless of the strategy.
    if disable_historical_hot_starts:
        max_nearby_commitment_hot_starts = 0
        nearby_commitment_pool_size = 0

    # Step 1：获取对偶变�?
    if verbose:
        print("Step 1: 获取对偶变量 lambda ...", flush=True)
    lambda_val = lambda_predictor.predict(scenario_input)

    # Step 2：全局 LP 松弛 + 启发式四舍五�?
    if verbose:
        print("Step 2: 求解全局 UC LP 松弛 ...", flush=True)
    initial_lp_stats = None
    if fp_strategy_norm == 'vanilla':
        x_LP, initial_lp_stats = solve_global_LP_relaxation_without_surrogate(
            ppc,
            pd_data,
            T_delta,
            return_stats=True,
        )
    elif manager is not None:
        x_LP = manager.solve_global(
            scenario_input,
            lambda_val,
            sparse_library=sparse_library,
            sparse_x_template_library=sparse_x_template_library,
            surrogate_constraint_scope=surrogate_constraint_scope_norm,
            bcd_proxy_scope=bcd_proxy_scope_norm,
        )
    else:
        x_LP = solve_global_LP_relaxation(
            ppc,
            scenario_input,
            T_delta,
            trainers,
            lambda_val,
            agent=agent,
            sparse_library=sparse_library,
            sparse_x_template_library=sparse_x_template_library,
            surrogate_constraint_scope=surrogate_constraint_scope_norm,
            bcd_proxy_scope=bcd_proxy_scope_norm,
        )
    x_init = round_to_integer(x_LP)

    integrality_gap = float(np.mean(np.minimum(x_LP, 1 - x_LP)))  # 平均�?�?的距�?
    if verbose:
        print(f"  整数性间隙（平均�? {integrality_gap:.4f}", flush=True)

    # Step 3：收集多组整数解
    if verbose:
        print("Step 3: 收集多组整数解（子问�?LP + 扰动�?..", flush=True)
    x_surr_lp, x_init_k, x_init_k_m, candidate_details = collect_integer_solutions(
        scenario_input, lambda_val, trainers,
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
        surrogate_constraint_scope=surrogate_constraint_scope_norm,
        return_details=True,
    )
    x_init_k_milp = candidate_details.get('x_init_k_milp')
    x_init_k_m_milp = candidate_details.get('x_init_k_m_milp')

    # Step 4：识别高可信度变�?
    if verbose:
        print("Step 4: 识别高可信度变量 ...", flush=True)
    trusted_mask = identify_trusted_mask(
        x_LP, x_init_k, x_init_k_m, conf_threshold=conf_threshold
    )
    if fp_strategy_norm == 'vanilla':
        # Vanilla FP should never fix variables by the tailored "trusted bit"
        # rule.  With conf_threshold=inf this previously fixed everything,
        # making the LP projection infeasible whenever the rounded point was
        # dispatch-infeasible.  Keep the baseline fully free.
        trusted_mask = np.zeros_like(trusted_mask, dtype=bool)
    ng, T = x_LP.shape
    n_trusted = int(np.sum(trusted_mask))
    if verbose:
        print(f"  可信变量: {n_trusted}/{ng*T} ({100*n_trusted/(ng*T):.1f}%)", flush=True)

    # Step 5：基于全局 LP �?surrogate LP 的启发式热启动候�?
    hot_start_candidates: List[Tuple[str, np.ndarray, float]] = []
    nearby_commitment_candidates = _build_nearby_commitment_candidates(
        _coerce_scenario_sample(scenario_input),
        trainers,
        scenario_bank,
        ng,
        T,
        n_candidates=max_nearby_commitment_hot_starts,
        candidate_pool_size=nearby_commitment_pool_size,
        rng=rng,
    )
    nearby_commitment_candidates, rejected_nearby_candidates = _sanitize_named_commitment_candidates(
        nearby_commitment_candidates,
        ppc,
        T_delta,
    )
    if verbose:
        print(
            f"  Nearby historical commitment hot starts: {len(nearby_commitment_candidates)}",
            flush=True,
        )
        if rejected_nearby_candidates:
            print(
                f"  Filtered {len(rejected_nearby_candidates)} nearby commitments before pooling",
                flush=True,
            )

    surrogate_screen_constraints: List[dict] = []
    surrogate_screen_summary = {
        'mode': str(surrogate_screen_mode),
        'hot_starts_before': 0,
        'hot_starts_after': 0,
        'x_pool_before': 0,
        'x_pool_after': 0,
        'n_constraints': 0,
        'constraints_per_unit': {},
        'hot_start_retained_names': [],
        'hot_start_rejected': [],
        'x_pool_rejected': [],
    }
    if str(surrogate_screen_mode).strip().lower() not in ('none', 'off', 'false', '0'):
        surrogate_screen_constraints = _select_stable_surrogate_screen_constraints(
            scenario_input,
            trainers,
            lambda_val,
            x_LP,
            x_surr_lp,
            x_init_k,
            x_init_k_m,
            nearby_commitment_candidates=nearby_commitment_candidates,
            max_constraints_per_unit=surrogate_screen_max_constraints_per_unit,
            min_support_ratio=surrogate_screen_min_support_ratio,
            max_normalized_violation=surrogate_screen_max_normalized_violation,
            min_mean_margin=surrogate_screen_min_mean_margin,
        )
        per_unit_counts: Dict[int, int] = {}
        for row in surrogate_screen_constraints:
            per_unit_counts[int(row['unit_id'])] = per_unit_counts.get(int(row['unit_id']), 0) + 1
        surrogate_screen_summary['n_constraints'] = int(len(surrogate_screen_constraints))
        surrogate_screen_summary['constraints_per_unit'] = per_unit_counts
        if verbose:
            print(
                "Step 4.5: stable surrogate screen "
                f"selected {len(surrogate_screen_constraints)} rows "
                f"({per_unit_counts if per_unit_counts else 'no rows'})",
                flush=True,
            )
    if use_hot_start:
        hot_start_candidates = _rank_hot_start_candidates(
            _build_hot_start_candidates(
                x_LP,
                x_surr_lp,
                x_init_k,
                x_init_k_m,
                trusted_mask,
                T_delta,
                nearby_commitment_candidates=nearby_commitment_candidates,
                max_perturbation_hot_starts=max_perturbation_hot_starts,
                max_unit_options_per_generator=max_unit_options_per_generator,
                max_unit_combination_candidates=max_unit_combination_candidates,
                lean_hot_starts=bool(lean_hot_starts),
                ppc=ppc,
                unit_ids=None,
            ),
            x_LP,
            x_surr_lp,
            x_init_k,
            x_init_k_m,
            trusted_mask,
            nearby_commitment_candidates=nearby_commitment_candidates,
        )
        if x_init_k_milp is not None:
            vote_majority = _compute_vote_majority(x_init_k, x_init_k_m)
            nearby_commitment_pool = None
            if nearby_commitment_candidates:
                nearby_commitment_pool = np.stack(
                    [np.asarray(candidate, dtype=int) for _name, candidate in nearby_commitment_candidates],
                    axis=0,
                )
            support_reference = _compute_hot_start_support_reference(
                x_LP,
                x_surr_lp,
                x_init_k,
                x_init_k_m,
                nearby_commitment_pool=nearby_commitment_pool,
            )
            milp_hot_start_specs = [("subproblem_milp_base", np.asarray(x_init_k_milp, dtype=int))]
            if x_init_k_m_milp is not None and x_init_k_m_milp.ndim == 3:
                milp_hot_start_specs.extend(
                    (f"subproblem_milp_perturb_{m + 1}", np.asarray(x_init_k_m_milp[:, m, :], dtype=int))
                    for m in range(x_init_k_m_milp.shape[1])
                )
            for name, candidate in milp_hot_start_specs:
                repaired = _repair_min_up_down_heuristic(
                    candidate, T_delta, ppc=ppc, unit_ids=None
                )
                # Defer rejection to the downstream sanitizer (which is the
                # single source of truth for "this candidate enters the pool"),
                # but apply the ppc-aware repair here so the scoring already
                # reflects a candidate that respects each unit's on/off logic.
                score = _score_hot_start_candidate(
                    repaired,
                    x_LP,
                    x_surr_lp,
                    vote_majority,
                    trusted_mask,
                    support_reference=support_reference,
                    nearby_commitment_pool=nearby_commitment_pool,
                )
                hot_start_candidates.append((name, repaired, float(score) + 0.25))
            hot_start_candidates.sort(key=lambda item: item[2], reverse=True)
    else:
        if fp_strategy_norm == 'vanilla':
            hot_start_candidates = [("lp_round", x_init, 0.0)]
        else:
            hot_start_candidates = [
                ("lp_round", x_init, 0.0),
                (
                    "surrogate_lp_round",
                    _repair_min_up_down_heuristic(x_init_k, T_delta, ppc=ppc, unit_ids=None),
                    -1.0,
                ),
            ]
            if x_init_k_milp is not None:
                hot_start_candidates.insert(
                    1,
                    (
                        "subproblem_milp_base",
                        _repair_min_up_down_heuristic(
                            np.asarray(x_init_k_milp, dtype=int),
                            T_delta,
                            ppc=ppc,
                            unit_ids=None,
                        ),
                        -0.5,
                    ),
                )

    sanitized_hot_start_specs, rejected_hot_start_specs = _sanitize_named_commitment_candidates(
        [(name, x_start) for name, x_start, _score in hot_start_candidates],
        ppc,
        T_delta,
    )
    score_by_name = {str(name): float(score) for name, _x_start, score in hot_start_candidates}
    hot_start_candidates = [
        (name, x_start, score_by_name.get(name, 0.0))
        for name, x_start in sanitized_hot_start_specs
    ]
    if verbose and rejected_hot_start_specs:
        print(
            f"  Filtered {len(rejected_hot_start_specs)} structurally infeasible hot starts before FP",
            flush=True,
        )
    surrogate_screen_summary['hot_starts_before'] = int(len(hot_start_candidates))
    if surrogate_screen_constraints and hot_start_candidates:
        filtered_hot_start_specs, rejected_surrogate_hot_starts = _filter_named_commitment_candidates_by_surrogate_screen(
            [(name, x_start) for name, x_start, _score in hot_start_candidates],
            surrogate_screen_constraints,
            normalized_violation_tol=surrogate_screen_candidate_violation_tol,
        )
        score_by_name = {str(name): float(score) for name, _x_start, score in hot_start_candidates}
        hot_start_candidates = [
            (name, x_start, score_by_name.get(name, 0.0))
            for name, x_start in filtered_hot_start_specs
        ]
        surrogate_screen_summary['hot_start_retained_names'] = [name for name, _x in filtered_hot_start_specs]
        surrogate_screen_summary['hot_start_rejected'] = list(rejected_surrogate_hot_starts)
        if verbose and rejected_surrogate_hot_starts:
            print(
                f"  Stable surrogate screen removed {len(rejected_surrogate_hot_starts)} hot starts",
                flush=True,
            )
    if not hot_start_candidates:
        fallback_hot_start_specs, _fallback_hot_start_rejections = _sanitize_named_commitment_candidates(
            [
                ("lp_round_fallback", x_init),
                ("surrogate_round_fallback", x_init_k),
            ],
            ppc,
            T_delta,
        )
        hot_start_candidates = [
            (name, x_start, -1e6 - idx)
            for idx, (name, x_start) in enumerate(fallback_hot_start_specs)
        ]
    surrogate_screen_summary['hot_starts_after'] = int(len(hot_start_candidates))

    # 构建整数解池：子问题整数�?+ 扰动�?+ 热启动候选，�?FP 投影阶段使用
    if fp_strategy_norm == 'vanilla':
        x_pool_specs: List[Tuple[str, np.ndarray]] = []
    else:
        x_pool_specs = [("subproblem_base", x_init_k)]
        if x_init_k_milp is not None:
            x_pool_specs.append(("subproblem_milp_base", np.asarray(x_init_k_milp, dtype=int)))
        x_pool_specs.extend(
            (f"subproblem_perturb_{m + 1}", x_init_k_m[:, m, :])
            for m in range(x_init_k_m.shape[1])
        )
        if x_init_k_m_milp is not None and x_init_k_m_milp.ndim == 3:
            x_pool_specs.extend(
                (f"subproblem_milp_perturb_{m + 1}", x_init_k_m_milp[:, m, :])
                for m in range(x_init_k_m_milp.shape[1])
            )
    if fp_strategy_norm != 'vanilla':
        x_pool_specs.extend((name, x_start) for name, x_start, _score in hot_start_candidates)
    sanitized_x_pool_specs, rejected_x_pool_specs = _sanitize_named_commitment_candidates(
        x_pool_specs,
        ppc,
        T_delta,
    )
    surrogate_screen_summary['x_pool_before'] = int(len(sanitized_x_pool_specs))
    if surrogate_screen_constraints and sanitized_x_pool_specs:
        sanitized_x_pool_specs, rejected_surrogate_x_pool_specs = _filter_named_commitment_candidates_by_surrogate_screen(
            sanitized_x_pool_specs,
            surrogate_screen_constraints,
            normalized_violation_tol=surrogate_screen_candidate_violation_tol,
        )
        surrogate_screen_summary['x_pool_rejected'] = list(rejected_surrogate_x_pool_specs)
        if verbose and rejected_surrogate_x_pool_specs:
            print(
                f"  Stable surrogate screen removed {len(rejected_surrogate_x_pool_specs)} pool candidates",
                flush=True,
            )
    x_pool = (
        np.stack([x_candidate for _name, x_candidate in sanitized_x_pool_specs], axis=0)
        if sanitized_x_pool_specs
        else None
    )
    if not hot_start_candidates and x_pool is not None and x_pool.shape[0] > 0:
        hot_start_candidates = [("pool_fallback", np.asarray(x_pool[0], dtype=int), -1e6)]
    if verbose and rejected_x_pool_specs:
        print(
            f"  Filtered {len(rejected_x_pool_specs)} structurally infeasible pool candidates",
            flush=True,
        )
    surrogate_screen_summary['x_pool_after'] = int(
        0 if x_pool is None else int(x_pool.shape[0])
    )

    recovery_details = {
        'x_lp': np.asarray(x_LP, dtype=float),
        'x_surr_lp': np.asarray(x_surr_lp, dtype=float),
        'x_init': np.asarray(x_init, dtype=int),
        'x_init_k': np.asarray(x_init_k, dtype=int),
        'x_init_k_m': np.asarray(x_init_k_m, dtype=int),
        'x_init_k_milp': None if x_init_k_milp is None else np.asarray(x_init_k_milp, dtype=int),
        'x_init_k_m_milp': None if x_init_k_m_milp is None else np.asarray(x_init_k_m_milp, dtype=int),
        'trusted_mask': np.asarray(trusted_mask, dtype=bool),
        'nearby_commitment_candidates': list(nearby_commitment_candidates),
        'surrogate_screen_constraints': surrogate_screen_constraints,
        'surrogate_screen_summary': surrogate_screen_summary,
        'surrogate_screen_soft_penalty': float(surrogate_screen_soft_penalty),
        'projection_objective_tau': projection_objective_tau,
        'surrogate_constraint_scope': surrogate_constraint_scope_norm,
        'bcd_proxy_scope': bcd_proxy_scope_norm,
        'initial_lp_stats': initial_lp_stats,
        'skip_feasible_hot_starts': bool(skip_feasible_hot_starts),
        'fp_strategy': fp_strategy_norm,
        'rounding_strategy': rounding_strategy_norm,
        'chi_alpha': float(chi_alpha),
        'chi_random_samples': int(chi_random_samples),
        'chi_random_evaluator_weight': float(chi_random_evaluator_weight),
        'enable_stall_theta_resample': bool(enable_stall_theta_resample),
        'hot_start_prechecks': [],
        'fp_histories': [],
    }

    if verbose:
        print("Step 5: 运行可行性泵热启动...", flush=True)
        for idx, (name, _x_start, score) in enumerate(hot_start_candidates, start=1):
            print(f"  热启动候选{idx}: {name}, score={score:.2f}", flush=True)

    # ---------------------------------------------------------------------
    # Build the θ-perturbation candidate-regeneration callback (eq 4-31/4-32).
    # When the FP loop detects sustained stagnation that cannot be cleared by
    # in-pool random rounding (eq 4-29), this callback is invoked to enlarge
    # ``x_pool`` by re-running the subproblem surrogates on a perturbed input
    # scenario. The callback returns a (n_extra, ng, T) array of candidate
    # commitments; ``None`` means "no useful extension".
    # ---------------------------------------------------------------------
    stall_theta_resample_callback = None
    if (
        enable_stall_theta_resample
        and fp_strategy_norm != 'vanilla'
        and trainers
    ):
        recover_rng_state = {'rng': rng}  # mutable holder so closure can be reseeded
        def _theta_resample_callback(_x_curr_local: np.ndarray) -> Optional[np.ndarray]:
            local_rng = recover_rng_state['rng']
            # Build a perturbed scenario sample: add Gaussian noise to net load.
            try:
                base_sample = _coerce_scenario_sample(scenario_input)
            except Exception:
                base_sample = None
            perturbed_pd = np.asarray(pd_data, dtype=float).copy()
            noise = local_rng.normal(
                0.0,
                stall_theta_resample_load_perturb_scale * float(np.maximum(np.abs(perturbed_pd).mean(), 1e-3)),
                size=perturbed_pd.shape,
            )
            perturbed_pd = np.clip(perturbed_pd + noise, 0.0, None)
            if base_sample is not None:
                base_sample = dict(base_sample)
                base_sample['pd'] = perturbed_pd
                perturbed_input: Any = base_sample
            else:
                perturbed_input = perturbed_pd
            try:
                lambda_perturbed = lambda_predictor.predict(perturbed_input)
            except Exception:
                return None
            try:
                x_surr_extra, x_extra_k, x_extra_k_m, _detail = collect_integer_solutions(
                    perturbed_input,
                    lambda_perturbed,
                    trainers,
                    n_perturbations=max(1, min(3, n_perturbations)),
                    n_similar_scenarios=0,
                    similar_scenario_pool_size=0,
                    n_load_perturbations=0,
                    load_perturbation_scale=stall_theta_resample_load_perturb_scale,
                    perturb_std=perturb_std,
                    neighborhood_weight=neighborhood_weight,
                    lambda_predictor=lambda_predictor,
                    rng=local_rng,
                    use_milp_candidate=False,
                    milp_for_perturbations=False,
                    surrogate_constraint_scope=surrogate_constraint_scope_norm,
                    return_details=True,
                )
            except Exception:
                return None
            extras = []
            if x_extra_k is not None and np.asarray(x_extra_k).size > 0:
                extras.append(np.asarray(x_extra_k, dtype=int))
            if x_extra_k_m is not None and np.asarray(x_extra_k_m).ndim == 3:
                for m in range(int(np.asarray(x_extra_k_m).shape[1])):
                    extras.append(np.asarray(x_extra_k_m[:, m, :], dtype=int))
            if not extras:
                return None
            stacked = np.stack(extras, axis=0)
            # Per-unit min-up/down repair (论文 §4.3 候选启停必须满足单机组约束)。
            cleaned = []
            for row in stacked:
                repaired = _repair_min_up_down_heuristic(row, T_delta, ppc=ppc, unit_ids=None)
                ok, _ = check_commitment_logic_feasibility(repaired, ppc, T_delta)
                if ok:
                    cleaned.append(repaired)
            if not cleaned:
                return None
            limit_units = max(1, int(stall_theta_resample_max_units))
            return np.stack(cleaned[:limit_units], axis=0)
        stall_theta_resample_callback = _theta_resample_callback
    recovery_details['stall_theta_resample_callback_available'] = bool(
        stall_theta_resample_callback is not None
    )

    # ---------------------------------------------------------------------
    # noH-mode BCD+MILP refresh callback (eq 4-31/4-32 enhanced):
    # When the FP is stuck (stall counter ≥ K) OR every M iterations, rerun
    # the FULL "global BCD-aware LP + per-unit surrogate MILP" pipeline with
    # a (small) Gaussian-perturbed lambda. Two candidates per invocation:
    #   c1 = round_to_integer(x_LP_new)   # from solve_global_LP_relaxation
    #   c2 = x_init_k_milp_new            # from collect_integer_solutions
    # Both are repaired against min-up/down and deduped against the
    # existing ``x_pool`` before being appended.
    #
    # Rationale (see check_subproblem_milp_quality results): per-unit MILP
    # alone with high σ produces ham=40+ junk. Routing through BCD first
    # gives the surrogate constraints a chance to be reconciled across units
    # via theta/zeta coupling, yielding more globally-consistent candidates.
    # ---------------------------------------------------------------------
    noh_milp_refresh_callback = None
    enable_noh_milp_refresh = (
        bool(disable_historical_hot_starts)
        and fp_strategy_norm != 'vanilla'
        and trainers
        and (int(noh_milp_refresh_max) > 0)
        and (int(noh_milp_refresh_stall) > 0 or int(noh_milp_refresh_interval) > 0)
    )
    if enable_noh_milp_refresh:
        # ``sigma_state`` tracks the running λ-perturbation magnitude. When
        # consecutive refresh calls produce only duplicates we ramp it up by
        # 2x (capped at 16× base) so the BCD/MILP solver sees a meaningfully
        # different input on the next call. ``duplicate_streak`` is reset
        # whenever at least one non-duplicate candidate is produced.
        sigma_state = {
            'sigma': float(max(noh_milp_refresh_lambda_sigma, 0.0)),
            'duplicate_streak': 0,
            'base_sigma': float(max(noh_milp_refresh_lambda_sigma, 0.0)),
            'previous_keys': set(),
        }
        recover_rng_state_milp = {'rng': rng}
        def _noh_milp_refresh_callback(
            iteration_idx: int,
            stall_counter_val: int,
            reason: str,
        ) -> Optional[np.ndarray]:
            local_rng = recover_rng_state_milp['rng']
            sigma = float(sigma_state['sigma'])
            lambda_base_arr = np.asarray(lambda_val, dtype=float)
            if sigma > 0.0:
                lambda_new = (
                    lambda_base_arr
                    + local_rng.normal(0.0, sigma, size=lambda_base_arr.shape)
                )
            else:
                lambda_new = lambda_base_arr.copy()

            collected: List[np.ndarray] = []

            # ---- (1) Global BCD-aware LP relaxation with perturbed lambda ----
            try:
                if manager is not None:
                    x_LP_new = manager.solve_global(
                        scenario_input,
                        lambda_new,
                        sparse_library=sparse_library,
                        sparse_x_template_library=sparse_x_template_library,
                        surrogate_constraint_scope=surrogate_constraint_scope_norm,
                        bcd_proxy_scope=bcd_proxy_scope_norm,
                    )
                else:
                    x_LP_new = solve_global_LP_relaxation(
                        ppc,
                        scenario_input,
                        T_delta,
                        trainers,
                        lambda_new,
                        agent=agent,
                        sparse_library=sparse_library,
                        sparse_x_template_library=sparse_x_template_library,
                        surrogate_constraint_scope=surrogate_constraint_scope_norm,
                        bcd_proxy_scope=bcd_proxy_scope_norm,
                    )
                x_lp_round_new = round_to_integer(np.asarray(x_LP_new, dtype=float))
                lp_repaired = _repair_min_up_down_heuristic(
                    x_lp_round_new, T_delta, ppc=ppc, unit_ids=None
                )
                ok_lp, _ = check_commitment_logic_feasibility(lp_repaired, ppc, T_delta)
                if ok_lp:
                    collected.append(np.asarray(lp_repaired, dtype=int))
            except Exception as exc:
                if verbose:
                    print(f"  noH refresh: global BCD-LP failed: {exc}", flush=True)

            # ---- (2) Per-unit subproblem MILP with the same perturbed lambda ----
            try:
                _x_surr_lp_new, _x_init_k_new, _x_init_k_m_new, _det_new = (
                    collect_integer_solutions(
                        scenario_input,
                        lambda_new,
                        trainers,
                        n_perturbations=0,
                        n_similar_scenarios=0,
                        similar_scenario_pool_size=0,
                        n_load_perturbations=0,
                        load_perturbation_scale=0.0,
                        perturb_std=perturb_std,
                        neighborhood_weight=neighborhood_weight,
                        lambda_predictor=lambda_predictor,
                        rng=local_rng,
                        use_milp_candidate=True,
                        milp_for_perturbations=False,
                        surrogate_constraint_scope=surrogate_constraint_scope_norm,
                        return_details=True,
                    )
                )
                x_init_k_milp_new = _det_new.get('x_init_k_milp')
                if x_init_k_milp_new is not None:
                    milp_repaired = _repair_min_up_down_heuristic(
                        np.asarray(x_init_k_milp_new, dtype=int),
                        T_delta, ppc=ppc, unit_ids=None,
                    )
                    ok_milp, _ = check_commitment_logic_feasibility(
                        milp_repaired, ppc, T_delta,
                    )
                    if ok_milp:
                        collected.append(np.asarray(milp_repaired, dtype=int))
            except Exception as exc:
                if verbose:
                    print(f"  noH refresh: per-unit MILP failed: {exc}", flush=True)

            if not collected:
                # Treat empty output as a duplicate-equivalent for σ ramping.
                sigma_state['duplicate_streak'] += 1
                sigma_state['sigma'] = min(
                    sigma_state['base_sigma'] * (2 ** sigma_state['duplicate_streak']),
                    max(sigma_state['base_sigma'] * 16.0, 1e-6),
                )
                if verbose:
                    print(
                        f"  noH refresh: no fresh candidate "
                        f"(streak={sigma_state['duplicate_streak']}, "
                        f"next σ → {sigma_state['sigma']:.4g})",
                        flush=True,
                    )
                return None
            # Detect whether any returned candidate is genuinely new wrt
            # candidates this callback has previously emitted -- if not, we
            # raise σ next call so BCD/MILP actually move.
            prev_keys: set = sigma_state['previous_keys']
            new_keys = []
            for cand in collected:
                k = bytes(np.asarray(cand, dtype=np.int8).tobytes())
                new_keys.append(k)
            any_new = any(k not in prev_keys for k in new_keys)
            if any_new:
                sigma_state['duplicate_streak'] = 0
                sigma_state['sigma'] = sigma_state['base_sigma']
            else:
                sigma_state['duplicate_streak'] += 1
                sigma_state['sigma'] = min(
                    sigma_state['base_sigma'] * (2 ** sigma_state['duplicate_streak']),
                    max(sigma_state['base_sigma'] * 16.0, 1e-6),
                )
            prev_keys.update(new_keys)
            stacked = np.stack(collected, axis=0)
            if verbose:
                src_str = "BCD-LP" if len(collected) >= 1 else ""
                if len(collected) >= 2:
                    src_str += "+MILP"
                print(
                    f"  noH refresh: produced {stacked.shape[0]} candidate(s) "
                    f"via {src_str} (λ-σ={sigma:.4g}; "
                    f"new={any_new}; streak={sigma_state['duplicate_streak']}; "
                    f"next σ={sigma_state['sigma']:.4g})",
                    flush=True,
                )
            return stacked
        noh_milp_refresh_callback = _noh_milp_refresh_callback
    recovery_details['noh_milp_refresh_callback_available'] = bool(
        noh_milp_refresh_callback is not None
    )
    recovery_details['noh_milp_refresh_uses_global_bcd'] = bool(
        noh_milp_refresh_callback is not None
    )

    parallel_warm_result = None
    selected_hot_start_name = None
    skipped_feasible_fallback: Optional[Tuple[int, str, np.ndarray, float, str]] = None
    best_feasible_so_far: Optional[Tuple[float, str, np.ndarray]] = None  # (uc_obj, name, x)
    gap_thresh = (
        float(precheck_objective_gap_threshold)
        if precheck_objective_gap_threshold is not None
        else None
    )
    lp_lb_obj = (
        float(lp_lower_bound_obj)
        if lp_lower_bound_obj is not None
        else None
    )
    recovery_details['precheck_objective_gap_threshold'] = gap_thresh
    recovery_details['lp_lower_bound_obj'] = lp_lb_obj
    recovery_details['disable_historical_hot_starts'] = bool(disable_historical_hot_starts)
    recovery_details['surrogate_engagement_iter'] = int(max(0, surrogate_engagement_iter))
    recovery_details['noh_milp_refresh_max'] = int(max(0, noh_milp_refresh_max))
    recovery_details['noh_milp_refresh_stall'] = int(max(0, noh_milp_refresh_stall))
    recovery_details['noh_milp_refresh_interval'] = int(max(0, noh_milp_refresh_interval))
    recovery_details['noh_milp_refresh_lambda_sigma'] = float(noh_milp_refresh_lambda_sigma)
    recovery_details['enable_pool_tabu_prune'] = bool(enable_pool_tabu_prune)
    recovery_details['pool_tabu_drop_threshold'] = int(max(1, pool_tabu_drop_threshold))

    pending_fp_starts: List[Tuple[int, str, np.ndarray, float]] = []
    for idx, (name, x_start, score) in enumerate(hot_start_candidates, start=1):
        start_feas, _reason = check_uc_feasibility(x_start, ppc, pd_data, T_delta)
        precheck_record = {
            'hot_start_index': int(idx),
            'hot_start_name': str(name),
            'score': float(score),
            'precheck_feasible': bool(start_feas),
            'precheck_reason': str(_reason),
        }

        uc_obj_pre: Optional[float] = None
        precheck_gap: Optional[float] = None
        precheck_gap_accepted: Optional[bool] = None
        if start_feas:
            x_int_pre = np.asarray(x_start, dtype=int)
            uc_obj_pre = _evaluate_commitment_dispatch_cost(
                x_int_pre, ppc, pd_data, T_delta,
            )
            precheck_record['uc_dispatch_obj'] = uc_obj_pre
            if uc_obj_pre is not None:
                if best_feasible_so_far is None or uc_obj_pre < best_feasible_so_far[0]:
                    best_feasible_so_far = (uc_obj_pre, str(name), x_int_pre)
                if (
                    gap_thresh is not None
                    and lp_lb_obj is not None
                    and lp_lb_obj > 1e-9
                ):
                    precheck_gap = (uc_obj_pre - lp_lb_obj) / abs(lp_lb_obj)
                    precheck_record['precheck_gap_vs_lp_lb'] = precheck_gap
                    precheck_gap_accepted = bool(precheck_gap <= gap_thresh)
                    precheck_record['precheck_gap_accepted'] = precheck_gap_accepted
                else:
                    precheck_record['precheck_gap_vs_lp_lb'] = None
                    precheck_record['precheck_gap_accepted'] = True
            else:
                precheck_record['precheck_gap_vs_lp_lb'] = None
                precheck_record['precheck_gap_accepted'] = False
        recovery_details['hot_start_prechecks'].append(precheck_record)

        if start_feas:
            recovery_details['fp_histories'].append({
                **precheck_record,
                'parallel': False,
                'entered_fp_iterations': False,
                'iterations': 0,
                'termination': (
                    'feasible_hot_start_skipped_for_fp_test'
                    if skip_feasible_hot_starts
                    else 'hot_start_already_feasible'
                ),
                'final_reason': str(_reason),
                'history': [],
            })
            # Decide whether to accept immediately or treat as candidate-only.
            accept_immediately = (not skip_feasible_hot_starts) and (
                precheck_gap_accepted is None or precheck_gap_accepted
            )
            if skip_feasible_hot_starts:
                if skipped_feasible_fallback is None:
                    skipped_feasible_fallback = (
                        int(idx),
                        str(name),
                        np.asarray(x_start, dtype=int),
                        float(score),
                        str(_reason),
                    )
                if verbose:
                    print(
                        f"  Hot start {idx}: {name} is feasible; skip it to test FP iterations",
                        flush=True,
                    )
                continue
            if not accept_immediately:
                # Feasible but gap too large -- continue iterating to look for
                # a better solution. Keep the candidate as fallback via
                # ``best_feasible_so_far``.
                if verbose:
                    print(
                        f"  Hot start {idx}: {name} is feasible (uc_obj={uc_obj_pre}, "
                        f"gap={precheck_gap}) but exceeds gap threshold {gap_thresh}; "
                        "continue iterating FP",
                        flush=True,
                    )
                pending_fp_starts.append((idx, name, x_start, score))
                continue
            if verbose:
                print(
                    f"  Hot start {idx}: {name} is already feasible "
                    f"(uc_obj={uc_obj_pre}); skip FP",
                    flush=True,
                )
            recovery_details['selected_hot_start'] = name
            recovery_details['x_result'] = np.asarray(x_start, dtype=int)
            return _finalize_recover_integer_solution_result(
                np.asarray(x_start, dtype=int),
                True,
                recovery_details,
                return_details,
            )
        pending_fp_starts.append((idx, name, x_start, score))

    requested_parallel_count = min(max(1, int(parallel_fp_starts)), len(pending_fp_starts))
    executed_parallel_count = requested_parallel_count if requested_parallel_count > 1 else 0
    if executed_parallel_count > 1:
        if verbose:
            print(f"  Launching {executed_parallel_count} FP hot starts in parallel", flush=True)

        seeded_tasks = []
        task_scores: Dict[int, float] = {}
        for idx, name, x_start, score in pending_fp_starts[:executed_parallel_count]:
            task_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            seeded_tasks.append((idx, name, x_start, task_seed))
            task_scores[int(idx)] = float(score)

        with ThreadPoolExecutor(max_workers=executed_parallel_count) as executor:
            future_to_task = {
                executor.submit(
                    _run_fp_hot_start_task,
                    idx,
                    name,
                    x_start,
                    trusted_mask,
                    ppc,
                    pd_data,
                    T_delta,
                    x_pool,
                    surrogate_screen_constraints,
                    surrogate_screen_soft_penalty,
                    projection_objective_tau,
                    max_fp_iter,
                    stall_perturbation_mode,
                    stall_flip_fraction,
                    task_seed,
                    rounding_strategy_norm,
                    chi_alpha,
                    chi_random_samples,
                    chi_random_evaluator_weight,
                    stall_theta_resample_callback,
                    stall_theta_resample_after_chi_random,
                    surrogate_engagement_iter,
                    noh_milp_refresh_callback,
                    noh_milp_refresh_stall,
                    noh_milp_refresh_interval,
                    noh_milp_refresh_max,
                    enable_pool_tabu_prune,
                    pool_tabu_drop_threshold,
                ): (idx, name)
                for idx, name, x_start, task_seed in seeded_tasks
            }
            parallel_results: Dict[int, np.ndarray] = {}
            for future in as_completed(future_to_task):
                idx, name = future_to_task[future]
                try:
                    task_idx, _task_name, task_result, task_success, task_details = future.result()
                except Exception as exc:
                    if verbose:
                        print(f"  Parallel FP hot start {idx}: {name} failed with {exc}", flush=True)
                    continue
                parallel_results[task_idx] = task_result
                recovery_details['fp_histories'].append({
                    'hot_start_index': int(task_idx),
                    'hot_start_name': str(name),
                    'score': float(task_scores.get(int(task_idx), 0.0)),
                    'precheck_feasible': False,
                    'parallel': True,
                    'entered_fp_iterations': True,
                    **dict(task_details or {}),
                })
                parallel_warm_result = task_result
                if verbose:
                    status = "success" if task_success else "no_feasible_solution"
                    print(f"  Parallel FP hot start {task_idx}: {name} -> {status}", flush=True)
                if task_success:
                    recovery_details['selected_hot_start'] = name
                    recovery_details['x_result'] = np.asarray(task_result, dtype=int)
                    return _finalize_recover_integer_solution_result(
                        np.asarray(task_result, dtype=int),
                        True,
                        recovery_details,
                        return_details,
                    )

            if pending_fp_starts:
                first_parallel_idx = pending_fp_starts[0][0]
                parallel_warm_result = parallel_results.get(first_parallel_idx, parallel_warm_result)

    remaining_fp_starts = pending_fp_starts[executed_parallel_count:]
    if not remaining_fp_starts and skipped_feasible_fallback is not None:
        _idx, _name, _x_start, _score, _reason = skipped_feasible_fallback
        recovery_details['selected_hot_start'] = _name
        recovery_details['x_result'] = np.asarray(_x_start, dtype=int)
        return _finalize_recover_integer_solution_result(
            np.asarray(_x_start, dtype=int),
            True,
            recovery_details,
            return_details,
        )

    x_result = parallel_warm_result if parallel_warm_result is not None else (
        remaining_fp_starts[0][2] if remaining_fp_starts else x_init
    )
    success = False

    for idx, name, x_start, score in remaining_fp_starts:
        selected_hot_start_name = name
        start_feas, _reason = check_uc_feasibility(x_start, ppc, pd_data, T_delta)
        if start_feas:
            uc_obj_pre2: Optional[float] = _evaluate_commitment_dispatch_cost(
                np.asarray(x_start, dtype=int), ppc, pd_data, T_delta,
            )
            if uc_obj_pre2 is not None:
                if best_feasible_so_far is None or uc_obj_pre2 < best_feasible_so_far[0]:
                    best_feasible_so_far = (
                        uc_obj_pre2, str(name), np.asarray(x_start, dtype=int),
                    )
            gap2: Optional[float] = None
            gap_ok2: Optional[bool] = None
            if (
                gap_thresh is not None
                and lp_lb_obj is not None
                and lp_lb_obj > 1e-9
                and uc_obj_pre2 is not None
            ):
                gap2 = (uc_obj_pre2 - lp_lb_obj) / abs(lp_lb_obj)
                gap_ok2 = bool(gap2 <= gap_thresh)
            elif gap_thresh is None:
                gap_ok2 = True

            recovery_details['fp_histories'].append({
                'hot_start_index': int(idx),
                'hot_start_name': str(name),
                'score': float(score),
                'precheck_feasible': True,
                'precheck_reason': str(_reason),
                'uc_dispatch_obj': uc_obj_pre2,
                'precheck_gap_vs_lp_lb': gap2,
                'precheck_gap_accepted': gap_ok2,
                'parallel': False,
                'entered_fp_iterations': False,
                'iterations': 0,
                'termination': (
                    'feasible_hot_start_skipped_for_fp_test'
                    if skip_feasible_hot_starts
                    else 'hot_start_already_feasible'
                ),
                'final_reason': str(_reason),
                'history': [],
            })
            if skip_feasible_hot_starts:
                if skipped_feasible_fallback is None:
                    skipped_feasible_fallback = (
                        int(idx),
                        str(name),
                        np.asarray(x_start, dtype=int),
                        float(score),
                        str(_reason),
                    )
                if verbose:
                    print(
                        f"  Hot start {idx}: {name} is feasible; skip it to test FP iterations",
                        flush=True,
                    )
                continue
            if gap_ok2 is False:
                if verbose:
                    print(
                        f"  Hot start {idx}: {name} feasible (uc_obj={uc_obj_pre2}, "
                        f"gap={gap2}) but gap > {gap_thresh}; perturb and continue",
                        flush=True,
                    )
                # Append the original feasible x_start to x_pool so FP can pick
                # it up as a candidate via the convex hull / χ rounding paths,
                # then apply a small random bit flip to x_start itself so the
                # FP iter-0 check_uc_feasibility does NOT short-circuit.
                x_orig = np.asarray(x_start, dtype=int)
                if x_pool is None:
                    x_pool = x_orig[None, :, :]
                else:
                    x_pool = np.concatenate(
                        [x_pool, x_orig[None, :, :]], axis=0,
                    )
                ng_perturb, T_perturb = x_orig.shape
                n_flip = max(1, int(0.05 * ng_perturb * T_perturb))
                flat_idx = rng.choice(
                    ng_perturb * T_perturb, size=n_flip, replace=False,
                )
                x_perturb = x_orig.copy()
                for fi in flat_idx:
                    g_ = fi // T_perturb
                    t_ = fi % T_perturb
                    x_perturb[g_, t_] = 1 - x_perturb[g_, t_]
                x_start = x_perturb
                # fall through to FP iteration below with perturbed x_start
            else:
                if verbose:
                    print(
                        f"  Hot start {idx}: {name} is already feasible "
                        f"(uc_obj={uc_obj_pre2}); skip FP",
                        flush=True,
                    )
                recovery_details['selected_hot_start'] = name
                recovery_details['x_result'] = np.asarray(x_start, dtype=int)
                return _finalize_recover_integer_solution_result(
                    np.asarray(x_start, dtype=int),
                    True,
                    recovery_details,
                    return_details,
                )

        if verbose:
            print(f"  Run FP hot start {idx}/{len(pending_fp_starts)}: {name}", flush=True)
        x_result, success, fp_details = run_feasibility_pump(
            x_start, trusted_mask, ppc, pd_data, T_delta,
            x_pool=x_pool,
            surrogate_screen_constraints=surrogate_screen_constraints,
            surrogate_screen_soft_penalty=surrogate_screen_soft_penalty,
            projection_objective_tau=projection_objective_tau,
            max_iter=max_fp_iter,
            stall_perturbation_mode=stall_perturbation_mode,
            stall_flip_fraction=stall_flip_fraction,
            rng=rng,
            verbose=verbose,
            return_history=True,
            rounding_strategy=rounding_strategy_norm,
            surrogate_engagement_iter=surrogate_engagement_iter,
            noh_milp_refresh_callback=noh_milp_refresh_callback,
            noh_milp_refresh_stall=noh_milp_refresh_stall,
            noh_milp_refresh_interval=noh_milp_refresh_interval,
            noh_milp_refresh_max=noh_milp_refresh_max,
            enable_pool_tabu_prune=enable_pool_tabu_prune,
            pool_tabu_drop_threshold=pool_tabu_drop_threshold,
            chi_alpha=chi_alpha,
            chi_random_samples=chi_random_samples,
            chi_random_evaluator_weight=chi_random_evaluator_weight,
            stall_theta_resample_callback=stall_theta_resample_callback,
            stall_theta_resample_after_chi_random=stall_theta_resample_after_chi_random,
            min_iter_before_feasible_accept=min_fp_iter_before_feasible_accept,
            terminate_on_cycle=terminate_on_cycle,
            pool_activation_iter=1,
        )
        recovery_details['fp_histories'].append({
            'hot_start_index': int(idx),
            'hot_start_name': str(name),
            'score': float(score),
            'precheck_feasible': False,
            'precheck_reason': str(_reason),
            'parallel': False,
            'entered_fp_iterations': True,
            **dict(fp_details or {}),
        })
        if success:
            # Track this solution's UC dispatch cost too so we can pick the
            # best across (precheck-feasibles, FP-found).
            try:
                _fp_obj = _evaluate_commitment_dispatch_cost(
                    np.asarray(x_result, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception:
                _fp_obj = None
            if _fp_obj is not None and (
                best_feasible_so_far is None or _fp_obj < best_feasible_so_far[0]
            ):
                best_feasible_so_far = (
                    _fp_obj, str(name), np.asarray(x_result, dtype=int),
                )
            break

    # If FP did not improve over a precheck-feasible candidate (or didn't
    # succeed), fall back to ``best_feasible_so_far``.
    if not success and best_feasible_so_far is not None:
        if verbose:
            print(
                f"  Falling back to best precheck-feasible candidate "
                f"({best_feasible_so_far[1]}, uc_obj={best_feasible_so_far[0]})",
                flush=True,
            )
        x_result = best_feasible_so_far[2]
        success = True
        selected_hot_start_name = best_feasible_so_far[1] + '_precheck_fallback'
    elif success and best_feasible_so_far is not None:
        fp_obj = (
            best_feasible_so_far[0]
            if isinstance(x_result, np.ndarray)
            and best_feasible_so_far[2].shape == np.asarray(x_result).shape
            and np.array_equal(np.asarray(x_result, dtype=int), best_feasible_so_far[2])
            else None
        )
        if fp_obj is None and best_feasible_so_far is not None:
            try:
                fp_obj = _evaluate_commitment_dispatch_cost(
                    np.asarray(x_result, dtype=int), ppc, pd_data, T_delta,
                )
            except Exception:
                fp_obj = None
        if (
            fp_obj is not None
            and best_feasible_so_far[0] < fp_obj
        ):
            if verbose:
                print(
                    f"  FP solution worse than precheck candidate "
                    f"(fp_obj={fp_obj}, precheck_obj={best_feasible_so_far[0]}); "
                    "swapping to precheck candidate",
                    flush=True,
                )
            x_result = best_feasible_so_far[2]
            selected_hot_start_name = best_feasible_so_far[1] + '_precheck_better'

    if verbose:
        status_str = "FP finished: feasible solution found" if success else "FP finished: no feasible solution"
        print(status_str, flush=True)

    recovery_details['best_feasible_summary'] = (
        None if best_feasible_so_far is None
        else {
            'uc_obj': float(best_feasible_so_far[0]),
            'source_hot_start': str(best_feasible_so_far[1]),
        }
    )
    recovery_details['selected_hot_start'] = selected_hot_start_name
    recovery_details['x_result'] = None if x_result is None else np.asarray(x_result, dtype=int)
    return _finalize_recover_integer_solution_result(
        x_result,
        success,
        recovery_details,
        return_details,
    )
