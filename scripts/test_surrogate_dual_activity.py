#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze trained subproblem surrogate duals and test-time constraint activity.

This script answers two diagnostic questions for an already trained model bundle:

1. During training, what values did the surrogate-constraint dual variables ``mu`` take?
   Use ``--refresh-train-dual`` to run one extra in-memory BCD generation after
   the loaded checkpoint iteration before collecting ``mu``. ``--refresh-mode dual-only`` refreshes only primal/dual
   blocks, while ``--refresh-mode full`` also runs the main surrogate NN and
   c_pg NN update stages. Neither mode saves or overwrites the model checkpoint.
2. During test-time surrogate subproblem solves, which corresponding constraints are active?
   If a solve falls back to soft surrogate constraints, violated RHS values are lifted to
   the observed LHS so those rows are exactly feasible/active, then the subproblem is
   solved again and row activity is measured on the relaxed solve.

Example:
    python scripts/test_surrogate_dual_activity.py ^
        --case case118 ^
        --model-dir result/surrogate_models/subproblem_models_case118_20260420_175002 ^
        --units 0,1,2 ^
        --train-samples 32 ^
        --test-samples 16 ^
        --refresh-train-dual ^
        --refresh-mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    from src.case_registry import get_case_ppc
    from src.dataset_json_utils import load_v3_active_set_json
    from src.feasibility_pump import (
        _extract_unit_lambda,
        _gurobi_status_name,
        _resolve_surrogate_constraint_layout,
        _solve_unit_LP_with_surrogate,
        solve_global_LP_relaxation,
    )
    from src.scenario_utils import normalize_sample_arrays
    from src.uc_NN_subproblem import load_trained_models
except ImportError:
    from case_registry import get_case_ppc
    from dataset_json_utils import load_v3_active_set_json
    from feasibility_pump import (
        _extract_unit_lambda,
        _gurobi_status_name,
        _resolve_surrogate_constraint_layout,
        _solve_unit_LP_with_surrogate,
        solve_global_LP_relaxation,
    )
    from scenario_utils import normalize_sample_arrays
    from uc_NN_subproblem import load_trained_models


def _parse_unit_ids(text: Optional[str]) -> Optional[List[int]]:
    if text is None or not str(text).strip():
        return None
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _latest_model_dir(case_name: str) -> Path:
    root = ROOT / "result" / "surrogate_models"
    candidates = sorted(
        root.glob(f"subproblem_models_{case_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if list(p.glob("surrogate_unit_*.pth"))]
    if not candidates:
        raise FileNotFoundError(f"No subproblem model dir with surrogate_unit_*.pth found under {root} for {case_name}")
    return candidates[0]


def _latest_bcd_model_path(case_name: str) -> Path:
    root = ROOT / "result" / "bcd_models"
    candidates = sorted(
        root.glob(f"bcd_model_{case_name}_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No BCD model found under {root} for {case_name}")
    return candidates[0]


def _default_active_set_json(case_name: str) -> Path:
    if case_name == "case118":
        try:
            import run_training_case118 as c118

            return ROOT / c118.CASE118_ACTIVE_SET_JSON
        except Exception:
            pass
    candidates = sorted(
        (ROOT / "result" / "active_set").glob(f"active_sets_{case_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No active-set JSON found for {case_name}")
    return candidates[0]


def _get_ppc(case_name: str) -> dict:
    if case_name == "case118":
        try:
            from src.mti118_data_loader import load_case118_ppc_with_mti_limits
        except ImportError:
            from mti118_data_loader import load_case118_ppc_with_mti_limits

        return load_case118_ppc_with_mti_limits(aggregate_thermal_by_bus=True)
    return get_case_ppc(case_name)


def _case118_training_default(name: str, fallback=None):
    if name is None:
        return fallback
    try:
        import run_training_case118 as c118

        return getattr(c118, name, fallback)
    except Exception:
        return fallback


def _slice_samples(samples: List[dict], start: int, count: Optional[int]) -> List[dict]:
    start = max(0, int(start))
    if count is None:
        sliced = samples[start:]
    else:
        sliced = samples[start : start + max(0, int(count))]
    return [normalize_sample_arrays(dict(sample)) for sample in sliced]


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _summarize_array(values: np.ndarray, active_tol: float) -> dict:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "nonzero_rate": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "max": float(np.max(finite)),
        "nonzero_rate": float(np.mean(np.abs(finite) > active_tol)),
    }


def _resolve_pg_cost_nn_epochs(trainer, pg_cost_nn_epochs: Optional[int]) -> int:
    if pg_cost_nn_epochs is not None:
        return max(int(pg_cost_nn_epochs), 0)
    configured = getattr(trainer, "pg_cost_nn_epochs", None)
    if configured is not None:
        return max(int(configured), 0)
    return 10


def _prepare_continuation_generation(trainer, refresh_total_max_iter: Optional[int] = None) -> int:
    """Advance trainer.iter_number as one extra generation after the loaded checkpoint."""
    loaded_iter = int(getattr(trainer, "iter_number", -1))
    continuation_iter = loaded_iter + 1
    total_max_iter = (
        max(int(refresh_total_max_iter), continuation_iter + 1)
        if refresh_total_max_iter is not None
        else continuation_iter + 1
    )
    if hasattr(trainer, "_configure_surrogate_bcd_run"):
        trainer._configure_surrogate_bcd_run(total_max_iter)
    trainer.iter_number = continuation_iter
    trainer._sync_surrogate_direction_strategy_state()
    return continuation_iter


def _run_primal_dual_only_generation(
    trainer,
    sample_limit: Optional[int] = None,
    refresh_total_max_iter: Optional[int] = None,
) -> np.ndarray:
    """Refresh primal/dual blocks once and return raw mu before sign/floor finalization."""
    original_n = int(trainer.n_samples)
    if sample_limit is None:
        n = original_n
    else:
        n = min(original_n, max(0, int(sample_limit)))
    if n <= 0:
        return np.zeros((0, int(trainer.num_coupling_constraints)), dtype=float)

    old_n = trainer.n_samples
    trainer.n_samples = n
    try:
        _prepare_continuation_generation(trainer, refresh_total_max_iter=refresh_total_max_iter)
        eps = 1e-10
        for sample_id in range(n):
            pg_sol, x_sol, coc_sol, cpower_sol = trainer.iter_with_primal_block(
                sample_id,
                trainer.alpha_values[sample_id],
                trainer.beta_values[sample_id],
                trainer.gamma_values[sample_id],
                trainer.delta_values[sample_id],
                trainer.cost_values[sample_id],
                trainer.pg_cost_values[sample_id],
            )
            if pg_sol is None:
                continue
            trainer.pg[sample_id] = np.where(np.abs(pg_sol) < eps, 0, pg_sol)
            trainer.x[sample_id] = np.where(np.abs(x_sol) < eps, 0, x_sol)
            trainer.x[sample_id] = np.where(np.abs(trainer.x[sample_id] - 1) < eps, 1, trainer.x[sample_id])
            trainer.coc[sample_id] = np.where(np.abs(coc_sol) < eps, 0, coc_sol)
            trainer.cpower[sample_id] = np.where(np.abs(cpower_sol) < eps, 0, cpower_sol)

        lb_mu = trainer._current_mu_lower_bound_value()
        sign_relax_round = trainer._is_mu_sign_relaxation_round()
        raw_mu = np.full((n, int(trainer.num_coupling_constraints)), np.nan, dtype=float)
        solved = np.zeros(n, dtype=bool)
        for sample_id in range(n):
            lambda_inherent, mu_sol = trainer.iter_with_dual_block(
                sample_id,
                trainer.alpha_values[sample_id],
                trainer.beta_values[sample_id],
                trainer.gamma_values[sample_id],
                trainer.delta_values[sample_id],
                trainer.cost_values[sample_id],
                trainer.pg_cost_values[sample_id],
            )
            if lambda_inherent is None:
                continue
            trainer.lambda_inherent[sample_id] = lambda_inherent
            raw_mu[sample_id, : len(mu_sol)] = np.asarray(mu_sol, dtype=float).reshape(-1)
            solved[sample_id] = True

        if sign_relax_round and np.any(solved):
            trainer.surrogate_direction_signs = trainer._resolve_surrogate_direction_signs_from_mu(raw_mu[solved])
        if np.any(solved):
            signs = trainer._get_surrogate_direction_signs()
            for sample_id in np.where(solved)[0]:
                trainer.mu[sample_id] = trainer._finalize_mu_values(raw_mu[sample_id], lb_mu, direction_signs=signs)

        return raw_mu
    finally:
        trainer.n_samples = old_n


def _run_full_training_generation(
    trainer,
    sample_limit: Optional[int] = None,
    nn_epochs: int = 1,
    pg_cost_nn_epochs: Optional[int] = None,
    refresh_total_max_iter: Optional[int] = None,
) -> np.ndarray:
    """Run one in-memory training generation matching the normal subproblem BCD flow."""
    raw_mu = _run_primal_dual_only_generation(
        trainer,
        sample_limit,
        refresh_total_max_iter=refresh_total_max_iter,
    )

    trainer._invalidate_loss_tensor_cache()
    trainer._prev_alpha_values = trainer.alpha_values.copy()
    trainer._prev_beta_values = trainer.beta_values.copy()
    trainer._prev_gamma_values = trainer.gamma_values.copy()
    trainer._prev_delta_values = trainer.delta_values.copy()
    trainer._prev_cost_values = trainer.cost_values.copy()
    trainer._prev_pg_cost_values = trainer.pg_cost_values.copy()
    trainer._invalidate_loss_tensor_cache()

    trainer.iter_with_main_direct_targets()
    trainer.iter_with_surrogate_nn(
        num_epochs=max(int(nn_epochs), 0),
        batch_size=getattr(trainer, "nn_batch_size", None),
        batch_strategy=getattr(trainer, "nn_batch_strategy", None),
        shuffle=getattr(trainer, "nn_shuffle", None),
        learning_rate=None,
        cost_learning_rate=None,
    )
    trainer.iter_with_c_pg_direct_targets()
    trainer.iter_with_c_pg_nn(
        num_epochs=_resolve_pg_cost_nn_epochs(trainer, pg_cost_nn_epochs),
        batch_size=(
            trainer.pg_cost_batch_size
            if getattr(trainer, "pg_cost_batch_size", None) is not None
            else getattr(trainer, "nn_batch_size", None)
        ),
        batch_strategy=(
            trainer.pg_cost_batch_strategy
            if getattr(trainer, "pg_cost_batch_strategy", None) is not None
            else getattr(trainer, "nn_batch_strategy", None)
        ),
        shuffle=(
            trainer.pg_cost_shuffle
            if getattr(trainer, "pg_cost_shuffle", None) is not None
            else getattr(trainer, "nn_shuffle", None)
        ),
        learning_rate=None,
    )
    trainer._refresh_cached_surrogate_outputs()
    return raw_mu


def collect_training_mu_records(
    trainers: Dict[int, object],
    train_sample_limit: Optional[int],
    refresh_train_dual: bool,
    refresh_mode: str,
    refresh_nn_epochs: int,
    refresh_pg_cost_nn_epochs: Optional[int],
    refresh_total_max_iter: Optional[int],
    mu_active_tol: float,
) -> Tuple[List[dict], Dict[Tuple[int, int], dict]]:
    rows: List[dict] = []
    by_key: Dict[Tuple[int, int], dict] = {}
    for unit_id, trainer in sorted(trainers.items()):
        if refresh_train_dual:
            if refresh_mode == "full":
                print(
                    f"[train-mu] unit {unit_id}: refresh one full training generation "
                    f"(nn_epochs={refresh_nn_epochs}, pg_cost_nn_epochs={refresh_pg_cost_nn_epochs})",
                    flush=True,
                )
                raw_mu = _run_full_training_generation(
                    trainer,
                    train_sample_limit,
                    nn_epochs=refresh_nn_epochs,
                    pg_cost_nn_epochs=refresh_pg_cost_nn_epochs,
                    refresh_total_max_iter=refresh_total_max_iter,
                )
            else:
                print(f"[train-mu] unit {unit_id}: refresh one primal/dual-only generation", flush=True)
                raw_mu = _run_primal_dual_only_generation(
                    trainer,
                    train_sample_limit,
                    refresh_total_max_iter=refresh_total_max_iter,
                )
            mu_source = raw_mu
            source_name = f"refreshed_{refresh_mode}_raw_mu"
        else:
            n = trainer.mu.shape[0] if train_sample_limit is None else min(trainer.mu.shape[0], int(train_sample_limit))
            mu_source = np.asarray(trainer.mu[:n], dtype=float)
            source_name = "checkpoint_mu"

        if mu_source.ndim != 2:
            continue
        for k in range(mu_source.shape[1]):
            vals = mu_source[:, k]
            stats = _summarize_array(vals, mu_active_tol)
            row = {
                "unit_id": int(unit_id),
                "constraint_index": int(k),
                "mu_source": source_name,
                **{f"train_mu_{name}": value for name, value in stats.items()},
            }
            rows.append(row)
            by_key[(int(unit_id), int(k))] = row
    return rows, by_key


def _solve_test_unit_with_rhs_relaxation(
    trainer,
    sample: dict,
    lambda_val,
    active_tol: float,
    violation_tol: float,
) -> Tuple[List[dict], dict]:
    unit_id = int(trainer.unit_id)
    T = int(trainer.T)
    lambda_unit = _extract_unit_lambda(lambda_val, T, unit_id=unit_id, trainer=trainer)
    alphas, betas, gammas, deltas, costs, pg_costs = trainer.get_surrogate_params(sample, lambda_unit)

    x_first, status_first, details_first = _solve_unit_LP_with_surrogate(
        trainer,
        lambda_val,
        alphas,
        betas,
        gammas,
        deltas,
        costs=costs,
        pg_costs=pg_costs,
        scenario_sample=sample,
    )

    deltas_relaxed = np.asarray(deltas, dtype=float).copy()
    relaxed_mask = np.zeros_like(deltas_relaxed, dtype=bool)
    first_rows = details_first.get("surrogate_violations", []) or []
    for row in first_rows:
        k = int(row["k"])
        violation = _safe_float(row.get("violation", 0.0))
        lhs = _safe_float(row.get("lhs", float("nan")))
        if 0 <= k < deltas_relaxed.size and math.isfinite(lhs) and violation > violation_tol:
            deltas_relaxed[k] = lhs
            relaxed_mask[k] = True

    x_second, status_second, details_second = _solve_unit_LP_with_surrogate(
        trainer,
        lambda_val,
        alphas,
        betas,
        gammas,
        deltas_relaxed,
        costs=costs,
        pg_costs=pg_costs,
        scenario_sample=sample,
    )

    first_by_k = {int(row["k"]): row for row in first_rows}
    second_rows = details_second.get("surrogate_violations", []) or []
    records: List[dict] = []
    for row in second_rows:
        k = int(row["k"])
        first = first_by_k.get(k, {})
        lhs_second = _safe_float(row.get("lhs"))
        rhs_second = _safe_float(row.get("rhs"))
        gap_second = lhs_second - rhs_second if math.isfinite(lhs_second) and math.isfinite(rhs_second) else float("nan")
        records.append(
            {
                "unit_id": unit_id,
                "constraint_index": k,
                "timestep": int(row.get("timestep", -1)),
                "offsets": "|".join(str(x) for x in row.get("offsets", ())),
                "alpha": _safe_float(row.get("alpha")),
                "beta": _safe_float(row.get("beta")),
                "gamma": _safe_float(row.get("gamma")),
                "original_rhs": _safe_float(first.get("rhs", row.get("rhs"))),
                "relaxed_rhs": rhs_second,
                "rhs_relaxed": bool(0 <= k < relaxed_mask.size and relaxed_mask[k]),
                "first_lhs": _safe_float(first.get("lhs")),
                "first_violation": _safe_float(first.get("violation", 0.0)),
                "first_slack": _safe_float(first.get("slack", 0.0)),
                "second_lhs": lhs_second,
                "second_gap": gap_second,
                "second_violation": _safe_float(row.get("violation", 0.0)),
                "second_slack": _safe_float(row.get("slack", 0.0)),
                "test_active_after_relax": bool(math.isfinite(gap_second) and abs(gap_second) <= active_tol),
            }
        )

    summary = {
        "unit_id": unit_id,
        "first_status": int(status_first),
        "first_status_name": _gurobi_status_name(status_first),
        "first_used_soft_surrogate": bool(details_first.get("used_soft_surrogate", False)),
        "first_fallback_triggered": bool(details_first.get("fallback_triggered", False)),
        "first_violation_sum": _safe_float(details_first.get("surrogate_violation_sum", 0.0)),
        "first_slack_sum": _safe_float(details_first.get("surrogate_slack_sum", 0.0)),
        "second_status": int(status_second),
        "second_status_name": _gurobi_status_name(status_second),
        "second_used_soft_surrogate": bool(details_second.get("used_soft_surrogate", False)),
        "second_fallback_triggered": bool(details_second.get("fallback_triggered", False)),
        "n_constraints": int(len(records)),
        "n_rhs_relaxed": int(np.sum(relaxed_mask)),
        "n_active_after_relax": int(sum(1 for row in records if row["test_active_after_relax"])),
        "x_first_integrality_gap": float(np.mean(np.minimum(x_first, 1.0 - x_first))) if np.size(x_first) else float("nan"),
        "x_second_integrality_gap": float(np.mean(np.minimum(x_second, 1.0 - x_second))) if np.size(x_second) else float("nan"),
    }
    return records, summary


def collect_test_activity_records(
    trainers: Dict[int, object],
    dual_predictor,
    test_samples: List[dict],
    train_mu_by_key: Dict[Tuple[int, int], dict],
    active_tol: float,
    violation_tol: float,
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    summaries: List[dict] = []
    for sample_pos, sample in enumerate(test_samples):
        sample = normalize_sample_arrays(dict(sample))
        sample_id = int(sample.get("sample_id", sample_pos))
        print(f"[test] sample_pos={sample_pos}, sample_id={sample_id}", flush=True)
        lambda_val = dual_predictor.predict(sample)
        for unit_id, trainer in sorted(trainers.items()):
            try:
                unit_rows, summary = _solve_test_unit_with_rhs_relaxation(
                    trainer,
                    sample,
                    lambda_val,
                    active_tol=active_tol,
                    violation_tol=violation_tol,
                )
            except Exception as exc:
                summaries.append(
                    {
                        "sample_pos": int(sample_pos),
                        "sample_id": int(sample_id),
                        "unit_id": int(unit_id),
                        "error": str(exc),
                    }
                )
                print(f"  unit {unit_id}: failed: {exc}", flush=True)
                continue

            summary.update({"sample_pos": int(sample_pos), "sample_id": int(sample_id)})
            summaries.append(summary)
            for row in unit_rows:
                key = (int(row["unit_id"]), int(row["constraint_index"]))
                train_row = train_mu_by_key.get(key, {})
                merged = {
                    "sample_pos": int(sample_pos),
                    "sample_id": int(sample_id),
                    **row,
                    "train_mu_mean": train_row.get("train_mu_mean", float("nan")),
                    "train_mu_median": train_row.get("train_mu_median", float("nan")),
                    "train_mu_max": train_row.get("train_mu_max", float("nan")),
                    "train_mu_nonzero_rate": train_row.get("train_mu_nonzero_rate", float("nan")),
                }
                rows.append(merged)
    return rows, summaries


def aggregate_activity(rows: List[dict], train_mu_by_key: Dict[Tuple[int, int], dict]) -> List[dict]:
    grouped: Dict[Tuple[int, int], List[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["unit_id"]), int(row["constraint_index"])), []).append(row)

    out: List[dict] = []
    for key in sorted(set(grouped) | set(train_mu_by_key)):
        unit_id, k = key
        items = grouped.get(key, [])
        train = train_mu_by_key.get(key, {})
        if items:
            active = np.asarray([bool(item["test_active_after_relax"]) for item in items], dtype=float)
            relaxed = np.asarray([bool(item["rhs_relaxed"]) for item in items], dtype=float)
            first_violation = np.asarray([_safe_float(item["first_violation"]) for item in items], dtype=float)
            second_abs_gap = np.asarray([abs(_safe_float(item["second_gap"])) for item in items], dtype=float)
            n_test = len(items)
        else:
            active = relaxed = first_violation = second_abs_gap = np.asarray([], dtype=float)
            n_test = 0
        out.append(
            {
                "unit_id": unit_id,
                "constraint_index": k,
                "n_test_rows": int(n_test),
                "test_active_rate_after_relax": float(np.mean(active)) if active.size else float("nan"),
                "test_rhs_relaxed_rate": float(np.mean(relaxed)) if relaxed.size else float("nan"),
                "test_first_violation_mean": float(np.nanmean(first_violation)) if first_violation.size else float("nan"),
                "test_second_abs_gap_mean": float(np.nanmean(second_abs_gap)) if second_abs_gap.size else float("nan"),
                "train_mu_mean": train.get("train_mu_mean", float("nan")),
                "train_mu_median": train.get("train_mu_median", float("nan")),
                "train_mu_max": train.get("train_mu_max", float("nan")),
                "train_mu_nonzero_rate": train.get("train_mu_nonzero_rate", float("nan")),
                "train_mu_count": train.get("train_mu_count", 0),
            }
        )
    return out


def _load_main_bcd_agent(ppc: dict, active_path: Path, bcd_model_path: Path, t_delta: float):
    try:
        from src.uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json
    except ImportError:
        from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json

    active_set_data = load_active_set_from_json(str(active_path))
    agent = Agent_NN_BCD(ppc, active_set_data=active_set_data, T_delta=float(t_delta))
    agent.load_model_parameters(str(bcd_model_path), restore_rho_state=False)
    return agent


def collect_main_model_activity_records(
    ppc: dict,
    trainers: Dict[int, object],
    dual_predictor,
    test_samples: List[dict],
    agent,
    t_delta: float,
    include_subproblem_rows: bool = False,
    bcd_proxy_scope: str = "both",
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    summaries: List[dict] = []
    scope = str(bcd_proxy_scope or "both").strip().lower()
    if scope not in {"both", "theta", "zeta", "none"}:
        raise ValueError(f"Unsupported bcd_proxy_scope={bcd_proxy_scope!r}")
    allowed_kinds = set()
    if scope in {"both", "theta"}:
        allowed_kinds.add("bcd_theta")
    if scope in {"both", "zeta"}:
        allowed_kinds.add("bcd_zeta")
    if include_subproblem_rows:
        allowed_kinds.add("subproblem_surrogate")

    for sample_pos, sample in enumerate(test_samples):
        sample = normalize_sample_arrays(dict(sample))
        sample_id = int(sample.get("sample_id", sample_pos))
        print(f"[main-test] sample_pos={sample_pos}, sample_id={sample_id}", flush=True)
        try:
            if trainers:
                if dual_predictor is None:
                    raise RuntimeError("Subproblem trainers require a dual predictor for main activity")
                lambda_val = dual_predictor.predict(sample)
            else:
                lambda_val = np.zeros((0,), dtype=float)
            _, stats = solve_global_LP_relaxation(
                ppc,
                sample,
                float(t_delta),
                trainers,
                lambda_val,
                agent=agent,
                bcd_proxy_scope=scope,
                return_stats=True,
            )
        except Exception as exc:
            summaries.append(
                {
                    "sample_pos": int(sample_pos),
                    "sample_id": int(sample_id),
                    "error": str(exc),
                }
            )
            print(f"  main model activity failed: {exc}", flush=True)
            continue

        activity_rows = []
        for row in stats.get("main_constraint_activity_rows", []) or []:
            if str(row.get("kind")) not in allowed_kinds:
                continue
            out = {
                "sample_pos": int(sample_pos),
                "sample_id": int(sample_id),
                **dict(row),
            }
            rows.append(out)
            activity_rows.append(out)

        summaries.append(
            {
                "sample_pos": int(sample_pos),
                "sample_id": int(sample_id),
                "stage_index": stats.get("stage_index"),
                "stage_name": stats.get("stage_name"),
                "status": stats.get("status"),
                "status_name": stats.get("status_name"),
                "runtime_sec": stats.get("runtime_sec"),
                "objective": stats.get("objective"),
                "num_main_activity_rows": int(len(activity_rows)),
                "num_bcd_theta_rows": int(sum(1 for r in activity_rows if r.get("kind") == "bcd_theta")),
                "num_bcd_zeta_rows": int(sum(1 for r in activity_rows if r.get("kind") == "bcd_zeta")),
                "num_subproblem_rows": int(sum(1 for r in activity_rows if r.get("kind") == "subproblem_surrogate")),
                "bcd_proxy_scope": scope,
                "num_bcd_theta_constraints": stats.get("num_bcd_theta_constraints"),
                "num_bcd_zeta_constraints": stats.get("num_bcd_zeta_constraints"),
                "bcd_slack_sum": stats.get("bcd_slack_sum", 0.0),
                "bcd_slack_max": stats.get("bcd_slack_max", 0.0),
                "used_soft_bcd": stats.get("used_soft_bcd", False),
                "used_fallback_stage": stats.get("used_fallback_stage", False),
            }
        )
    return rows, summaries


def aggregate_main_model_activity(rows: List[dict], active_tol: float, dual_tol: float) -> List[dict]:
    grouped: Dict[Tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]], List[dict]] = {}
    for row in rows:
        key = (
            str(row.get("kind", "unknown")),
            row.get("branch_id"),
            row.get("unit_id"),
            row.get("constraint_index"),
            row.get("time_slot"),
        )
        grouped.setdefault(key, []).append(row)

    out: List[dict] = []
    for (kind, branch_id, unit_id, constraint_index, time_slot), items in sorted(
        grouped.items(),
        key=lambda kv: (
            kv[0][0],
            -1 if kv[0][1] is None else int(kv[0][1]),
            -1 if kv[0][2] is None else int(kv[0][2]),
            -1 if kv[0][3] is None else int(kv[0][3]),
            -1 if kv[0][4] is None else int(kv[0][4]),
        ),
    ):
        slacks = np.asarray([_safe_float(r.get("abs_row_slack")) for r in items], dtype=float)
        duals = np.asarray([_safe_float(r.get("abs_dual")) for r in items], dtype=float)
        relax = np.asarray([bool(r.get("is_relaxed", False)) for r in items], dtype=float)
        out.append(
            {
                "kind": kind,
                "branch_id": branch_id,
                "unit_id": unit_id,
                "constraint_index": constraint_index,
                "time_slot": time_slot,
                "n_rows": int(len(items)),
                "row_active_rate": float(np.mean(slacks <= float(active_tol))) if slacks.size else float("nan"),
                "dual_active_rate": float(np.mean(duals > float(dual_tol))) if duals.size else float("nan"),
                "relaxed_rate": float(np.mean(relax)) if relax.size else float("nan"),
                "abs_slack_mean": float(np.nanmean(slacks)) if slacks.size else float("nan"),
                "abs_slack_median": float(np.nanmedian(slacks)) if slacks.size else float("nan"),
                "abs_slack_max": float(np.nanmax(slacks)) if slacks.size else float("nan"),
                "abs_dual_mean": float(np.nanmean(duals)) if duals.size else float("nan"),
                "abs_dual_median": float(np.nanmedian(duals)) if duals.size else float("nan"),
                "abs_dual_max": float(np.nanmax(duals)) if duals.size else float("nan"),
            }
        )
    theta_rows = [
        row for row in out
        if str(row.get("kind")) == "bcd_theta"
        and row.get("branch_id") is not None
        and row.get("time_slot") is not None
    ]
    theta_keys = sorted({(int(row["branch_id"]), int(row["time_slot"])) for row in theta_rows})
    theta_id_by_key = {key: idx for idx, key in enumerate(theta_keys)}
    zeta_rows = [
        row for row in out
        if str(row.get("kind")) == "bcd_zeta"
        and row.get("unit_id") is not None
        and row.get("time_slot") is not None
    ]
    zeta_keys = sorted({(int(row["unit_id"]), int(row["time_slot"])) for row in zeta_rows})
    zeta_id_by_key = {key: idx for idx, key in enumerate(zeta_keys)}
    for row in out:
        if str(row.get("kind")) == "bcd_theta" and row.get("branch_id") is not None and row.get("time_slot") is not None:
            row["theta_constraint_id"] = theta_id_by_key[(int(row["branch_id"]), int(row["time_slot"]))]
        if str(row.get("kind")) == "bcd_zeta" and row.get("unit_id") is not None and row.get("time_slot") is not None:
            row["zeta_constraint_id"] = zeta_id_by_key[(int(row["unit_id"]), int(row["time_slot"]))]
    return out


def theta_activity_by_id_rows(main_aggregate_rows: List[dict]) -> List[dict]:
    rows = [
        row for row in main_aggregate_rows
        if str(row.get("kind")) == "bcd_theta"
        and row.get("theta_constraint_id") is not None
    ]
    return sorted(rows, key=lambda row: int(row["theta_constraint_id"]))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_heatmap(matrix: np.ndarray, title: str, path: Path, cbar_label: str) -> None:
    if not MPL_AVAILABLE or matrix.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.25), max(4, matrix.shape[0] * 0.28)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("constraint index")
    ax.set_ylabel("unit id")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_labeled_heatmap(
    matrix: np.ndarray,
    row_labels: List[int],
    col_labels: List[int] | List[str],
    title: str,
    path: Path,
    cbar_label: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "YlGnBu",
) -> None:
    if not MPL_AVAILABLE or matrix.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(8, min(24, len(col_labels) * 0.42 + 2.5))
    fig_h = max(4, min(18, len(row_labels) * 0.34 + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(col_labels) <= 60:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels([str(v) for v in col_labels], rotation=90 if len(col_labels) > 24 else 0)
    if len(row_labels) <= 80:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels([str(v) for v in row_labels])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _main_activity_matrix(
    aggregate_rows: List[dict],
    kind: str,
    field: str,
    row_field: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    selected = [
        row for row in aggregate_rows
        if str(row.get("kind")) == kind
        and row.get(row_field) is not None
        and row.get("time_slot") is not None
    ]
    if not selected:
        return np.zeros((0, 0), dtype=float), [], []
    row_labels = sorted({int(row[row_field]) for row in selected})
    col_labels = sorted({int(row["time_slot"]) for row in selected})
    row_to_idx = {value: idx for idx, value in enumerate(row_labels)}
    col_to_idx = {value: idx for idx, value in enumerate(col_labels)}
    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    for row in selected:
        matrix[row_to_idx[int(row[row_field])], col_to_idx[int(row["time_slot"])]] = _safe_float(row.get(field))
    return matrix, row_labels, col_labels


def _theta_activity_id_matrix(
    aggregate_rows: List[dict],
    field: str,
) -> Tuple[np.ndarray, List[int], List[str]]:
    selected = [
        row for row in aggregate_rows
        if str(row.get("kind")) == "bcd_theta"
        and row.get("theta_constraint_id") is not None
    ]
    if not selected:
        return np.zeros((0, 0), dtype=float), [], []
    selected = sorted(selected, key=lambda row: int(row["theta_constraint_id"]))
    row_labels = [int(row["theta_constraint_id"]) for row in selected]
    matrix = np.asarray([[_safe_float(row.get(field))] for row in selected], dtype=float)
    return matrix, row_labels, [field]


def make_main_activity_plots(output_dir: Path, aggregate_rows: List[dict], sample_rows: List[dict]) -> None:
    if not MPL_AVAILABLE or not aggregate_rows:
        return
    plot_dir = output_dir / "main_model_activity"
    theta_by_id = theta_activity_by_id_rows(aggregate_rows)
    if theta_by_id:
        x = np.asarray([int(row["theta_constraint_id"]) for row in theta_by_id], dtype=int)
        y = np.asarray([_safe_float(row.get("row_active_rate")) for row in theta_by_id], dtype=float)
        labels = [f"b{int(row['branch_id'])},t{int(row['time_slot'])}" for row in theta_by_id]
        fig_w = max(10, min(36, len(theta_by_id) * 0.22 + 3.0))
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        ax.bar(x, y, width=0.86, color="#3182bd", alpha=0.86)
        ax.set_title(f"Theta constraint activity rate by id (n={len(theta_by_id)})")
        ax.set_xlabel("theta constraint id")
        ax.set_ylabel("row active rate")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
        if len(theta_by_id) <= 80:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_xlabel("theta constraint id (branch,time)")
        fig.tight_layout()
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / "bcd_theta_activity_rate_by_constraint_id.png", dpi=180)
        plt.close(fig)

    plot_specs = [
        ("row_active_rate", "row active rate", "active rate"),
        ("dual_active_rate", "dual active rate", "dual active rate"),
        ("relaxed_rate", "relaxation rate", "relaxed rate"),
        ("abs_slack_mean", "mean absolute slack", "mean |slack|"),
    ]
    kind_specs = [
        ("bcd_theta", "branch_id", "Theta main proxy", "branch id"),
        ("bcd_zeta", "unit_id", "Zeta main proxy", "unit id"),
    ]
    for kind, row_field, title_prefix, ylabel in kind_specs:
        for field, label, cbar in plot_specs:
            if kind == "bcd_theta":
                matrix, row_labels, col_labels = _theta_activity_id_matrix(aggregate_rows, field)
                xlabel = ""
                ylabel_for_plot = "theta constraint id"
            else:
                matrix, row_labels, col_labels = _main_activity_matrix(aggregate_rows, kind, field, row_field)
                xlabel = "time slot"
                ylabel_for_plot = ylabel
            if matrix.size == 0:
                continue
            _plot_labeled_heatmap(
                matrix,
                row_labels,
                col_labels,
                f"{title_prefix} {label} by constraint id",
                plot_dir / f"{kind}_{field}_by_id_heatmap.png",
                cbar,
                xlabel,
                ylabel_for_plot,
            )

    if sample_rows:
        for kind in ("bcd_theta", "bcd_zeta"):
            selected = [row for row in sample_rows if str(row.get("kind")) == kind]
            if not selected:
                continue
            samples = sorted({int(row.get("sample_pos", row.get("sample_id", -1))) for row in selected})
            constraints = sorted({str(row.get("constraint_name", "")) for row in selected})
            if not samples or not constraints:
                continue
            # Keep very large cases readable by plotting the first 160 named constraints.
            constraints = constraints[:160]
            sample_to_idx = {value: idx for idx, value in enumerate(samples)}
            con_to_idx = {value: idx for idx, value in enumerate(constraints)}
            matrix = np.full((len(samples), len(constraints)), np.nan, dtype=float)
            for row in selected:
                name = str(row.get("constraint_name", ""))
                if name not in con_to_idx:
                    continue
                matrix[sample_to_idx[int(row.get("sample_pos", row.get("sample_id", -1)))], con_to_idx[name]] = (
                    1.0 if bool(row.get("is_row_active_1e_6", False)) else 0.0
                )
            fig_w = max(9, min(30, len(constraints) * 0.22 + 2.5))
            fig_h = max(4, min(14, len(samples) * 0.32 + 2.0))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"{kind} active rows by sample and explicit constraint name")
            ax.set_xlabel("constraint name")
            ax.set_ylabel("sample position")
            if len(constraints) <= 80:
                ax.set_xticks(np.arange(len(constraints)))
                ax.set_xticklabels(constraints, rotation=90, fontsize=7)
            if len(samples) <= 80:
                ax.set_yticks(np.arange(len(samples)))
                ax.set_yticklabels([str(v) for v in samples])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("active row (1=yes)")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{kind}_active_by_sample_constraint_name.png", dpi=180)
            plt.close(fig)


def make_plots(output_dir: Path, aggregate_rows: List[dict], test_rows: List[dict]) -> None:
    if not MPL_AVAILABLE:
        return
    if not aggregate_rows:
        return

    units = sorted({int(row["unit_id"]) for row in aggregate_rows})
    max_k = max(int(row["constraint_index"]) for row in aggregate_rows)
    unit_to_idx = {u: i for i, u in enumerate(units)}

    def matrix_for(field: str) -> np.ndarray:
        mat = np.full((len(units), max_k + 1), np.nan, dtype=float)
        for row in aggregate_rows:
            mat[unit_to_idx[int(row["unit_id"])], int(row["constraint_index"])] = _safe_float(row.get(field))
        return mat

    _plot_heatmap(
        matrix_for("train_mu_mean"),
        "Training surrogate dual mean by unit/constraint",
        output_dir / "train_mu_mean_heatmap.png",
        "mean(mu)",
    )
    _plot_heatmap(
        matrix_for("test_active_rate_after_relax"),
        "Test activity rate after RHS relaxation",
        output_dir / "test_activity_rate_heatmap.png",
        "active rate",
    )
    _plot_heatmap(
        matrix_for("test_rhs_relaxed_rate"),
        "Test RHS relaxation rate",
        output_dir / "test_rhs_relaxed_rate_heatmap.png",
        "relaxed rate",
    )

    x = np.asarray([_safe_float(row.get("train_mu_mean")) for row in aggregate_rows], dtype=float)
    y = np.asarray([_safe_float(row.get("test_active_rate_after_relax")) for row in aggregate_rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.any(mask):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x[mask], y[mask], s=22, alpha=0.75)
        ax.set_xlabel("training mean mu")
        ax.set_ylabel("test active rate after relaxation")
        ax.set_title("Training dual vs test activity")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "train_mu_vs_test_activity.png", dpi=180)
        plt.close(fig)

    if test_rows:
        values = np.asarray([_safe_float(row.get("first_violation")) for row in test_rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(values, bins=50, color="#3182bd", alpha=0.8)
            ax.set_xlabel("soft-solve surrogate violation")
            ax.set_ylabel("count")
            ax.set_title("Test-time surrogate violation distribution")
            fig.tight_layout()
            fig.savefig(output_dir / "test_violation_hist.png", dpi=180)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trained surrogate duals and test-time row activity.")
    parser.add_argument("--case", default="case118", help="case name, default case118")
    parser.add_argument("--active-set-json", default=None, help="active-set JSON path; default auto")
    parser.add_argument("--model-dir", default=None, help="trained subproblem model directory; default latest")
    parser.add_argument("--bcd-model", default=None, help="trained main BCD model path; default latest when --main-activity")
    parser.add_argument("--units", default=None, help="comma-separated unit ids; default all model units")
    parser.add_argument("--train-start", type=int, default=0, help="first sample used to initialize/load trainers")
    parser.add_argument("--train-samples", type=int, default=32, help="number of training samples to analyze")
    parser.add_argument("--test-start", type=int, default=0, help="first sample for test-time solves")
    parser.add_argument("--test-samples", type=int, default=16, help="number of test samples")
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--refresh-train-dual", action="store_true", help="run one extra in-memory generation for mu")
    parser.add_argument(
        "--refresh-mode",
        choices=("dual-only", "full"),
        default="full",
        help="dual-only refreshes primal/dual blocks; full also runs main surrogate NN and c_pg NN stages",
    )
    parser.add_argument(
        "--refresh-nn-epochs",
        type=int,
        default=None,
        help="main surrogate NN epochs used by --refresh-mode full; default uses case118 preset when available",
    )
    parser.add_argument(
        "--refresh-pg-cost-nn-epochs",
        type=int,
        default=None,
        help="c_pg NN epochs used by --refresh-mode full; default uses checkpoint setting",
    )
    parser.add_argument(
        "--refresh-total-max-iter",
        type=int,
        default=None,
        help=(
            "schedule horizon for the continued generation; default is loaded iter_number + 2, "
            "which appends one round after a completed checkpoint"
        ),
    )
    parser.add_argument("--mu-active-tol", type=float, default=1e-7)
    parser.add_argument("--active-tol", type=float, default=1e-5)
    parser.add_argument("--violation-tol", type=float, default=1e-7)
    parser.add_argument("--lp-backend", default=None, help="override subproblem LP backend")
    parser.add_argument("--strategy", default=None, help="override constraint generation strategy")
    parser.add_argument("--ignore-startup-shutdown-costs", action="store_true")
    parser.add_argument("--main-activity", action="store_true", help="also test main-model theta/zeta proxy constraint activity")
    parser.add_argument("--main-only", action="store_true", help="skip subproblem activity and run only main-model activity")
    parser.add_argument(
        "--main-bcd-proxy-scope",
        choices=("both", "theta", "zeta", "none"),
        default="both",
        help="which BCD main proxy constraints to add during main activity solves",
    )
    parser.add_argument(
        "--main-include-subproblem",
        action="store_true",
        help=(
            "load subproblem models and include their rows in main-problem activity CSVs; "
            "without this flag, --main-only loads only the BCD model"
        ),
    )
    parser.add_argument("--main-dual-active-tol", type=float, default=1e-7)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_name = str(args.case)
    run_main_activity = bool(args.main_activity or args.main_only)
    run_subproblem_activity = not bool(args.main_only)
    main_needs_subproblem = bool(run_main_activity and args.main_include_subproblem)
    need_subproblem_models = bool(run_subproblem_activity or main_needs_subproblem)
    resolved_refresh_nn_epochs = args.refresh_nn_epochs
    if resolved_refresh_nn_epochs is None:
        if case_name == "case118":
            resolved_refresh_nn_epochs = int(
                _case118_training_default("CASE118_SUBPROBLEM_NN_EPOCHS_PER_BCD", 10)
            )
        else:
            resolved_refresh_nn_epochs = 10
    resolved_refresh_pg_cost_nn_epochs = args.refresh_pg_cost_nn_epochs
    if resolved_refresh_pg_cost_nn_epochs is None and case_name == "case118":
        resolved_refresh_pg_cost_nn_epochs = int(
            _case118_training_default("CASE118_SUBPROBLEM_PG_COST_NN_EPOCHS", 0)
        )
    active_path = Path(args.active_set_json) if args.active_set_json else _default_active_set_json(case_name)
    if not active_path.is_absolute():
        active_path = ROOT / active_path
    model_dir = None
    if need_subproblem_models:
        model_dir = Path(args.model_dir) if args.model_dir else _latest_model_dir(case_name)
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
    bcd_model_path = None
    if run_main_activity:
        bcd_model_path = Path(args.bcd_model) if args.bcd_model else _latest_bcd_model_path(case_name)
        if not bcd_model_path.is_absolute():
            bcd_model_path = ROOT / bcd_model_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "result" / "model_tests" / f"{case_name}_dual_activity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("surrogate dual/activity model test", flush=True)
    print(f"case={case_name}", flush=True)
    print(f"active_set={active_path}", flush=True)
    print(f"activity_modes: subproblem={run_subproblem_activity}, main={run_main_activity}, main_include_subproblem={bool(args.main_include_subproblem)}", flush=True)
    print(f"main_bcd_proxy_scope={args.main_bcd_proxy_scope}", flush=True)
    if model_dir is not None:
        print(f"model_dir={model_dir}", flush=True)
    else:
        print("model_dir=(not loaded; pure BCD main activity)", flush=True)
    if bcd_model_path is not None:
        print(f"bcd_model={bcd_model_path}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(
        "refresh="
        f"{bool(args.refresh_train_dual)} mode={args.refresh_mode} "
        f"nn_epochs={resolved_refresh_nn_epochs} "
        f"pg_cost_nn_epochs={resolved_refresh_pg_cost_nn_epochs} "
        f"total_max_iter={args.refresh_total_max_iter}",
        flush=True,
    )
    print("=" * 80, flush=True)

    ppc = _get_ppc(case_name)
    all_samples = load_v3_active_set_json(active_path, announce=lambda msg: print(msg, flush=True))
    train_samples = _slice_samples(all_samples, args.train_start, args.train_samples)
    test_samples = _slice_samples(all_samples, args.test_start, args.test_samples)
    if not train_samples:
        raise ValueError("No training samples selected")
    if not test_samples:
        raise ValueError("No test samples selected")

    unit_ids = _parse_unit_ids(args.units)
    dual_predictor = None
    trainers: Dict[int, object] = {}
    if need_subproblem_models:
        if model_dir is None:
            raise RuntimeError("Subproblem activity requires --model-dir or an auto-discoverable model directory")
        dual_predictor, trainers = load_trained_models(
            ppc,
            train_samples,
            args.t_delta,
            str(model_dir),
            unit_ids=unit_ids,
            lp_backend=args.lp_backend,
            constraint_generation_strategy=args.strategy,
            ignore_startup_shutdown_costs=(True if args.ignore_startup_shutdown_costs else None),
            case_name=case_name,
        )
        if not trainers:
            requested = "all available units" if unit_ids is None else ",".join(str(u) for u in unit_ids)
            available = sorted(p.name for p in model_dir.glob("surrogate_unit_*.pth"))
            raise RuntimeError(
                "No surrogate trainers were loaded. "
                f"requested units={requested}; model_dir={model_dir}; "
                f"available files={available[:20]}"
            )

    train_rows: List[dict] = []
    train_by_key: Dict[Tuple[int, int], dict] = {}
    test_rows: List[dict] = []
    solve_summaries: List[dict] = []
    aggregate_rows: List[dict] = []
    if run_subproblem_activity:
        train_rows, train_by_key = collect_training_mu_records(
            trainers,
            train_sample_limit=args.train_samples,
            refresh_train_dual=bool(args.refresh_train_dual),
            refresh_mode=str(args.refresh_mode),
            refresh_nn_epochs=int(resolved_refresh_nn_epochs),
            refresh_pg_cost_nn_epochs=resolved_refresh_pg_cost_nn_epochs,
            refresh_total_max_iter=args.refresh_total_max_iter,
            mu_active_tol=float(args.mu_active_tol),
        )
        test_rows, solve_summaries = collect_test_activity_records(
            trainers,
            dual_predictor,
            test_samples,
            train_by_key,
            active_tol=float(args.active_tol),
            violation_tol=float(args.violation_tol),
        )
        aggregate_rows = aggregate_activity(test_rows, train_by_key)

    main_rows: List[dict] = []
    main_summaries: List[dict] = []
    main_aggregate_rows: List[dict] = []
    if run_main_activity:
        if bcd_model_path is None:
            raise RuntimeError("--main-activity requires a BCD model")
        print(f"[main] loading BCD agent from {bcd_model_path}", flush=True)
        main_agent = _load_main_bcd_agent(ppc, active_path, bcd_model_path, args.t_delta)
        main_rows, main_summaries = collect_main_model_activity_records(
            ppc,
            trainers,
            dual_predictor,
            test_samples,
            main_agent,
            t_delta=float(args.t_delta),
            include_subproblem_rows=bool(args.main_include_subproblem),
            bcd_proxy_scope=str(args.main_bcd_proxy_scope),
        )
        main_aggregate_rows = aggregate_main_model_activity(
            main_rows,
            active_tol=float(args.active_tol),
            dual_tol=float(args.main_dual_active_tol),
        )

    write_csv(output_dir / "training_mu_by_constraint.csv", train_rows)
    write_csv(output_dir / "test_activity_by_sample_constraint.csv", test_rows)
    write_csv(output_dir / "test_solve_summary.csv", solve_summaries)
    write_csv(output_dir / "aggregate_constraint_activity.csv", aggregate_rows)
    write_csv(output_dir / "main_model_activity_by_sample_row.csv", main_rows)
    write_csv(output_dir / "main_model_solve_summary.csv", main_summaries)
    write_csv(output_dir / "main_model_activity_summary.csv", main_aggregate_rows)
    main_theta_by_id_rows = theta_activity_by_id_rows(main_aggregate_rows)
    write_csv(output_dir / "main_model_theta_activity_by_constraint_id.csv", main_theta_by_id_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "case": case_name,
                "active_set_json": str(active_path),
                "model_dir": str(model_dir) if model_dir is not None else None,
                "bcd_model": str(bcd_model_path) if bcd_model_path is not None else None,
                "run_subproblem_activity": bool(run_subproblem_activity),
                "run_main_activity": bool(run_main_activity),
                "main_include_subproblem": bool(args.main_include_subproblem),
                "main_bcd_proxy_scope": str(args.main_bcd_proxy_scope),
                "n_train_samples": len(train_samples),
                "n_test_samples": len(test_samples),
                "unit_ids": sorted(int(u) for u in trainers.keys()),
                "refresh_train_dual": bool(args.refresh_train_dual),
                "refresh_mode": str(args.refresh_mode),
                "refresh_nn_epochs": int(resolved_refresh_nn_epochs),
                "refresh_pg_cost_nn_epochs": resolved_refresh_pg_cost_nn_epochs,
                "refresh_total_max_iter": args.refresh_total_max_iter,
                "n_training_rows": len(train_rows),
                "n_test_rows": len(test_rows),
                "n_aggregate_rows": len(aggregate_rows),
                "main_activity": bool(args.main_activity or args.main_only),
                "main_only": bool(args.main_only),
                "n_main_rows": len(main_rows),
                "n_main_summary_rows": len(main_summaries),
                "n_main_aggregate_rows": len(main_aggregate_rows),
                "n_main_theta_constraints": len(main_theta_by_id_rows),
                "main_row_active_rate_mean": (
                    float(np.nanmean([row["row_active_rate"] for row in main_aggregate_rows]))
                    if main_aggregate_rows
                    else float("nan")
                ),
                "test_active_rate_after_relax_mean": (
                    float(np.nanmean([row["test_active_rate_after_relax"] for row in aggregate_rows]))
                    if aggregate_rows
                    else float("nan")
                ),
                "test_rhs_relaxed_rate_mean": (
                    float(np.nanmean([row["test_rhs_relaxed_rate"] for row in aggregate_rows]))
                    if aggregate_rows
                    else float("nan")
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.no_plots:
        make_plots(output_dir, aggregate_rows, test_rows)
        make_main_activity_plots(output_dir, main_aggregate_rows, main_rows)

    print("\nDone.", flush=True)
    print(f"  training rows: {len(train_rows)}", flush=True)
    print(f"  test rows: {len(test_rows)}", flush=True)
    print(f"  aggregate rows: {len(aggregate_rows)}", flush=True)
    print(f"  main rows: {len(main_rows)}", flush=True)
    print(f"  main aggregate rows: {len(main_aggregate_rows)}", flush=True)
    print(f"  main theta constraints: {len(main_theta_by_id_rows)}", flush=True)
    print(f"  output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
