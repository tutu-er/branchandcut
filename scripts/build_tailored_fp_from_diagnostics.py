#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build tailored feasibility-pump settings from forward diagnostics.

Input is the JSON written by ``scripts/collect_model_fp_diagnostics.py``.
Output is a compact config JSON with:

* selected FP profile
* ``recover_integer_solution`` keyword arguments
* per-unit/per-time error guidance for later targeted repairs

Example:
    python scripts/build_tailored_fp_from_diagnostics.py ^
        --diagnostics result/fp_diagnostics/fp_forward_diagnostics_case3lite_YYYYMMDD_HHMMSS.json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _resolve_path(text: str | None, fallback: Path | None = None) -> Path:
    if text is None or not str(text).strip():
        if fallback is None:
            raise ValueError("Path is required")
        return fallback.resolve()
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _metric_value(record: dict, metric_name: str, key: str, default: float = float("nan")) -> float:
    metric = (record.get("metrics") or {}).get(metric_name) or {}
    if not metric.get("available", False):
        return default
    return _safe_float(metric.get(key), default)


def _mean_finite(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def _sum_metric(records: List[dict], metric_name: str, key: str) -> int:
    total = 0
    for record in records:
        metric = (record.get("metrics") or {}).get(metric_name) or {}
        if metric.get("available", False):
            total += int(metric.get(key, 0) or 0)
    return int(total)


def _stack_error_records(records: List[dict], error_key: str) -> Optional[np.ndarray]:
    mats = []
    for record in records:
        raw = record.get(error_key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            mats.append(arr)
    if not mats:
        return None
    shape = mats[0].shape
    mats = [m for m in mats if m.shape == shape]
    if not mats:
        return None
    return np.stack(mats, axis=0)


def _error_profile(records: List[dict], error_key: str) -> dict:
    stack = _stack_error_records(records, error_key)
    if stack is None:
        return {
            "available": False,
            "unit_error_rate": [],
            "time_error_rate": [],
            "false_on_rate_by_unit": [],
            "false_off_rate_by_unit": [],
            "top_units": [],
            "top_time_slots": [],
        }

    err = np.asarray(stack, dtype=float)
    nonzero = err != 0
    false_on = err > 0
    false_off = err < 0
    unit_error_rate = np.mean(nonzero, axis=(0, 2))
    time_error_rate = np.mean(nonzero, axis=(0, 1))
    false_on_unit = np.mean(false_on, axis=(0, 2))
    false_off_unit = np.mean(false_off, axis=(0, 2))

    top_units = [
        {
            "unit_id": int(idx),
            "error_rate": float(unit_error_rate[idx]),
            "false_on_rate": float(false_on_unit[idx]),
            "false_off_rate": float(false_off_unit[idx]),
        }
        for idx in np.argsort(-unit_error_rate)[: min(10, unit_error_rate.size)]
    ]
    top_time_slots = [
        {
            "time_slot": int(idx),
            "error_rate": float(time_error_rate[idx]),
        }
        for idx in np.argsort(-time_error_rate)[: min(12, time_error_rate.size)]
    ]
    return {
        "available": True,
        "unit_error_rate": unit_error_rate.tolist(),
        "time_error_rate": time_error_rate.tolist(),
        "false_on_rate_by_unit": false_on_unit.tolist(),
        "false_off_rate_by_unit": false_off_unit.tolist(),
        "top_units": top_units,
        "top_time_slots": top_time_slots,
    }


def _transition_bias_profile(records: List[dict], solution_key: str) -> dict:
    unit_delay_values: Dict[int, List[int]] = {}
    unit_start_error_count: Dict[int, int] = {}
    unit_stop_error_count: Dict[int, int] = {}

    for record in records:
        sols = record.get("solutions") or {}
        x_true = sols.get("x_true")
        x_pred = sols.get(solution_key)
        if x_true is None or x_pred is None:
            continue
        truth = np.rint(np.asarray(x_true, dtype=float)).astype(int)
        pred = np.rint(np.clip(np.asarray(x_pred, dtype=float), 0.0, 1.0)).astype(int)
        if truth.shape != pred.shape or truth.ndim != 2:
            continue
        ng, T = truth.shape
        for g in range(ng):
            true_starts = set(np.where((truth[g, :-1] == 0) & (truth[g, 1:] == 1))[0] + 1)
            pred_starts = set(np.where((pred[g, :-1] == 0) & (pred[g, 1:] == 1))[0] + 1)
            true_stops = set(np.where((truth[g, :-1] == 1) & (truth[g, 1:] == 0))[0] + 1)
            pred_stops = set(np.where((pred[g, :-1] == 1) & (pred[g, 1:] == 0))[0] + 1)
            if not true_starts and not pred_starts and not true_stops and not pred_stops:
                continue

            for t_true in true_starts:
                if not pred_starts:
                    unit_start_error_count[g] = unit_start_error_count.get(g, 0) + 1
                    continue
                nearest = min(pred_starts, key=lambda t: abs(t - t_true))
                delay = int(nearest - t_true)
                if delay != 0:
                    unit_delay_values.setdefault(g, []).append(delay)
                    unit_start_error_count[g] = unit_start_error_count.get(g, 0) + 1

            unmatched_stops = true_stops.symmetric_difference(pred_stops)
            if unmatched_stops:
                unit_stop_error_count[g] = unit_stop_error_count.get(g, 0) + len(unmatched_stops)

    rows = []
    for g in sorted(set(unit_delay_values) | set(unit_start_error_count) | set(unit_stop_error_count)):
        delays = np.asarray(unit_delay_values.get(g, []), dtype=float)
        avg_delay = float(np.mean(delays)) if delays.size else 0.0
        if avg_delay > 0.25:
            pattern = "delayed_start"
        elif avg_delay < -0.25:
            pattern = "premature_start"
        else:
            pattern = "mixed_or_no_delay"
        rows.append(
            {
                "unit_id": int(g),
                "pattern": pattern,
                "avg_start_delay": avg_delay,
                "start_error_count": int(unit_start_error_count.get(g, 0)),
                "stop_error_count": int(unit_stop_error_count.get(g, 0)),
            }
        )
    rows.sort(key=lambda r: (r["start_error_count"] + r["stop_error_count"], abs(r["avg_start_delay"])), reverse=True)
    return {
        "available": bool(rows),
        "top_transition_units": rows[:10],
    }


def _metrics_summary(records: List[dict]) -> dict:
    metric_names = [
        "plain_lp_to_true",
        "proxy_lp_to_true",
        "subproblem_lp_to_true",
        "subproblem_milp_to_true",
        "fp_to_true",
    ]
    summary = {}
    for name in metric_names:
        summary[name] = {
            "mean_hamming": _mean_finite(_metric_value(r, name, "hamming") for r in records),
            "mean_l1": _mean_finite(_metric_value(r, name, "l1") for r in records),
            "mean_integrality_gap": _mean_finite(_metric_value(r, name, "integrality_gap") for r in records),
            "false_on": _sum_metric(records, name, "false_on"),
            "false_off": _sum_metric(records, name, "false_off"),
            "available_count": int(sum(1 for r in records if ((r.get("metrics") or {}).get(name) or {}).get("available", False))),
        }
    fp_success = [bool(r.get("fp_success", False)) for r in records if "fp_success" in r]
    summary["fp_success_rate"] = float(np.mean(fp_success)) if fp_success else None
    return summary


def _choose_profile(summary: dict, args: argparse.Namespace) -> Tuple[str, dict]:
    plain_h = _safe_float(summary["plain_lp_to_true"]["mean_hamming"])
    proxy_h = _safe_float(summary["proxy_lp_to_true"]["mean_hamming"])
    sub_h = _safe_float(summary["subproblem_lp_to_true"]["mean_hamming"])
    sub_milp_h = _safe_float(summary["subproblem_milp_to_true"]["mean_hamming"])

    baseline = min(v for v in [plain_h, proxy_h, sub_h, sub_milp_h] if np.isfinite(v)) \
        if any(np.isfinite(v) for v in [plain_h, proxy_h, sub_h, sub_milp_h]) else float("nan")

    surrogate_best = np.isfinite(sub_h) and (
        (not np.isfinite(plain_h) or sub_h <= 0.90 * plain_h)
        and (not np.isfinite(proxy_h) or sub_h <= 1.05 * proxy_h)
    )
    proxy_best = np.isfinite(proxy_h) and (
        (not np.isfinite(plain_h) or proxy_h <= 0.95 * plain_h)
        and (not np.isfinite(sub_h) or proxy_h <= 0.95 * sub_h)
    )
    lp_best = np.isfinite(plain_h) and (
        (not np.isfinite(proxy_h) or plain_h <= 0.90 * proxy_h)
        and (not np.isfinite(sub_h) or plain_h <= 0.90 * sub_h)
    )
    milp_helpful = np.isfinite(sub_milp_h) and (
        (not np.isfinite(sub_h) or sub_milp_h <= 0.95 * sub_h)
    )

    false_off = int(summary["subproblem_lp_to_true"]["false_off"])
    false_on = int(summary["subproblem_lp_to_true"]["false_on"])
    total_dir = max(false_off + false_on, 1)
    false_off_ratio = false_off / total_dir
    false_on_ratio = false_on / total_dir

    if surrogate_best:
        profile = "surrogate_guided"
        kwargs = {
            "conf_threshold": 0.12,
            "max_perturbation_hot_starts": 10,
            "max_unit_options_per_generator": 5,
            "max_unit_combination_candidates": 20,
            "max_nearby_commitment_hot_starts": 6,
            "nearby_commitment_pool_size": 18,
            "parallel_fp_starts": 2,
            "surrogate_screen_mode": "robust",
            "surrogate_screen_max_constraints_per_unit": 4,
            "surrogate_screen_min_support_ratio": 0.78,
            "surrogate_screen_max_normalized_violation": 0.06,
            "surrogate_screen_min_mean_margin": 0.01,
            "surrogate_screen_candidate_violation_tol": 0.03,
            "surrogate_screen_soft_penalty": 35.0,
            "projection_objective_tau": "adaptive",
            "use_subproblem_milp_candidate": bool(milp_helpful or args.prefer_milp_candidates),
        }
    elif proxy_best:
        profile = "proxy_lp_guided"
        kwargs = {
            "conf_threshold": 0.18,
            "max_perturbation_hot_starts": 6,
            "max_unit_options_per_generator": 4,
            "max_unit_combination_candidates": 12,
            "max_nearby_commitment_hot_starts": 4,
            "nearby_commitment_pool_size": 12,
            "parallel_fp_starts": 1,
            "surrogate_screen_mode": "robust",
            "surrogate_screen_max_constraints_per_unit": 3,
            "surrogate_screen_min_support_ratio": 0.88,
            "surrogate_screen_max_normalized_violation": 0.04,
            "surrogate_screen_min_mean_margin": 0.02,
            "surrogate_screen_candidate_violation_tol": 0.02,
            "surrogate_screen_soft_penalty": 25.0,
            "projection_objective_tau": "adaptive",
            "use_subproblem_milp_candidate": bool(milp_helpful or args.prefer_milp_candidates),
        }
    elif lp_best:
        profile = "lp_anchor"
        kwargs = {
            "conf_threshold": 0.22,
            "max_perturbation_hot_starts": 4,
            "max_unit_options_per_generator": 3,
            "max_unit_combination_candidates": 8,
            "max_nearby_commitment_hot_starts": 3,
            "nearby_commitment_pool_size": 10,
            "parallel_fp_starts": 1,
            "surrogate_screen_mode": "none",
            "surrogate_screen_max_constraints_per_unit": 0,
            "surrogate_screen_min_support_ratio": 0.95,
            "surrogate_screen_max_normalized_violation": 0.02,
            "surrogate_screen_min_mean_margin": 0.03,
            "surrogate_screen_candidate_violation_tol": 0.01,
            "surrogate_screen_soft_penalty": 10.0,
            "projection_objective_tau": "adaptive",
            "use_subproblem_milp_candidate": bool(args.prefer_milp_candidates),
        }
    else:
        profile = "diversified_repair"
        kwargs = {
            "conf_threshold": 0.10,
            "max_perturbation_hot_starts": 12,
            "max_unit_options_per_generator": 5,
            "max_unit_combination_candidates": 24,
            "max_nearby_commitment_hot_starts": 8,
            "nearby_commitment_pool_size": 24,
            "parallel_fp_starts": 2,
            "surrogate_screen_mode": "robust",
            "surrogate_screen_max_constraints_per_unit": 2,
            "surrogate_screen_min_support_ratio": 0.75,
            "surrogate_screen_max_normalized_violation": 0.08,
            "surrogate_screen_min_mean_margin": 0.0,
            "surrogate_screen_candidate_violation_tol": 0.04,
            "surrogate_screen_soft_penalty": 20.0,
            "projection_objective_tau": "adaptive",
            "use_subproblem_milp_candidate": bool(milp_helpful or args.prefer_milp_candidates),
        }

    if false_off_ratio >= 0.65:
        kwargs["stall_flip_fraction"] = 0.16
        direction_hint = "surrogate_false_off_dominant_open_more"
    elif false_on_ratio >= 0.65:
        kwargs["stall_flip_fraction"] = 0.08
        direction_hint = "surrogate_false_on_dominant_close_more"
    else:
        kwargs["stall_flip_fraction"] = 0.10
        direction_hint = "mixed_error_direction"
    kwargs["stall_perturbation_mode"] = "pool_then_flip"

    if args.parallel_fp_starts is not None:
        kwargs["parallel_fp_starts"] = int(args.parallel_fp_starts)

    return profile, {
        "kwargs": kwargs,
        "rationale": {
            "baseline_mean_hamming": baseline,
            "plain_lp_mean_hamming": plain_h,
            "proxy_lp_mean_hamming": proxy_h,
            "subproblem_lp_mean_hamming": sub_h,
            "subproblem_milp_mean_hamming": sub_milp_h,
            "surrogate_best": bool(surrogate_best),
            "proxy_best": bool(proxy_best),
            "lp_best": bool(lp_best),
            "milp_helpful": bool(milp_helpful),
            "false_off_ratio": float(false_off_ratio),
            "false_on_ratio": float(false_on_ratio),
            "direction_hint": direction_hint,
        },
    }


def build_tailored_config(diagnostics: dict, args: argparse.Namespace) -> dict:
    records = list(diagnostics.get("records") or [])
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        raise ValueError("diagnostics JSON has no records")

    summary = _metrics_summary(records)
    profile_name, profile = _choose_profile(summary, args)

    guidance = {
        "plain_lp_error_profile": _error_profile(records, "errors_plain_lp"),
        "proxy_lp_error_profile": _error_profile(records, "errors_proxy_lp"),
        "subproblem_lp_error_profile": _error_profile(records, "errors_subproblem_lp"),
        "subproblem_milp_error_profile": _error_profile(records, "errors_subproblem_milp"),
        "subproblem_transition_bias": _transition_bias_profile(records, "x_subproblem_lp"),
        "proxy_transition_bias": _transition_bias_profile(records, "x_lp_proxy"),
    }

    metadata = diagnostics.get("metadata") or {}
    return {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_diagnostics": str(args.diagnostics),
            "case": metadata.get("case"),
            "active_set_path": metadata.get("active_set_path"),
            "model_dir": metadata.get("model_dir"),
            "sample_count": len(records),
        },
        "selected_profile": profile_name,
        "recover_integer_solution_kwargs": profile["kwargs"],
        "selection_rationale": profile["rationale"],
        "metric_summary": summary,
        "guidance": guidance,
    }


def build_recover_kwargs(config: dict, overrides: Optional[dict] = None) -> dict:
    """Return kwargs that can be passed into ``recover_integer_solution``."""
    kwargs = dict(config.get("recover_integer_solution_kwargs") or {})
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _emit_runner(config_path: Path, runner_path: Path) -> None:
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    rel_config = config_path
    text = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small import helper for the tailored FP config generated from diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(r"{rel_config}")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_recover_kwargs(overrides: dict | None = None) -> dict:
    config = load_config()
    kwargs = dict(config.get("recover_integer_solution_kwargs") or {{}})
    if overrides:
        kwargs.update(overrides)
    return kwargs


if __name__ == "__main__":
    cfg = load_config()
    print("selected_profile=", cfg.get("selected_profile"))
    print(json.dumps(cfg.get("recover_integer_solution_kwargs", {{}}), indent=2, ensure_ascii=False))
'''
    runner_path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", required=True, help="JSON from collect_model_fp_diagnostics.py")
    p.add_argument("--output", default=None, help="Tailored FP config JSON path.")
    p.add_argument("--emit-runner", default=None, help="Optional helper .py file that loads the config.")
    p.add_argument("--prefer-milp-candidates", action="store_true")
    p.add_argument("--parallel-fp-starts", type=int, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    diagnostics_path = _resolve_path(args.diagnostics)
    with diagnostics_path.open("r", encoding="utf-8") as f:
        diagnostics = json.load(f)

    case_name = (diagnostics.get("metadata") or {}).get("case") or "case"
    output_path = _resolve_path(
        args.output,
        ROOT / "result" / "fp_diagnostics" / (
            f"tailored_fp_config_{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_tailored_config(diagnostics, args)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[done] selected_profile={config['selected_profile']}", flush=True)
    print(f"[done] wrote {output_path}", flush=True)
    print("[kwargs]", json.dumps(config["recover_integer_solution_kwargs"], ensure_ascii=False, indent=2), flush=True)

    if args.emit_runner:
        runner_path = _resolve_path(args.emit_runner)
        _emit_runner(output_path, runner_path)
        print(f"[done] wrote runner {runner_path}", flush=True)


if __name__ == "__main__":
    main()
