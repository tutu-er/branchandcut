"""Convert a pattern-library result JSON into an active_set-like JSON.

This script merges:
1. an original active-set JSON containing ``all_samples``
2. a pattern-library result JSON containing ``pattern_library`` and per-sample results

The output keeps the original sample payloads (load, renewables, etc.) while
replacing ``unit_commitment_matrix`` / ``active_set`` with the restricted
commitment reconstructed from the saved pattern selections.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Embedded defaults for direct execution without CLI args.
DEFAULT_ACTIVE_SET_JSON = (
    r"result/active_set/active_sets_case118_T0_n366_20260322_063917.json"
)
DEFAULT_PATTERN_LIBRARY_JSON = (
    r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json"
)
DEFAULT_OUTPUT_JSON = (
    r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932_active_set_like.json"
)
DEFAULT_FALLBACK_TO_ORIGINAL = True


def _load_ppc_for_case(case_name: str):
    case_name_norm = str(case_name or "").strip().lower()
    if case_name_norm == "case118":
        from src.mti118_data_loader import load_case118_ppc_with_mti_limits

        return load_case118_ppc_with_mti_limits(aggregate_thermal_by_bus=True)
    if case_name_norm == "case30":
        from src.case30_uc_data import get_case30_uc_ppc

        return get_case30_uc_ppc()
    if case_name_norm == "case3":
        from src.case3_uc_data import get_case3_uc_ppc

        return get_case3_uc_ppc()
    if case_name_norm == "case3lite":
        from src.case3_uc_data import get_case3lite_uc_ppc

        return get_case3lite_uc_ppc()
    if case_name_norm == "case39":
        from src.case39_pypower import get_case39_pypower

        return get_case39_pypower()
    raise ValueError(f"Unsupported case_name for dual refresh: {case_name!r}")


def _serialize_dual_payload(payload: Dict) -> Dict:
    return {
        key: np.asarray(value, dtype=float).tolist()
        for key, value in payload.items()
    }


def _refresh_sample_dual_payload(sample: Dict, x_matrix: List[List[int]], ppc, t_delta: float) -> None:
    from src.uc_NN_subproblem import (
        _build_generator_injection_sensitivity,
        _solve_global_dual_payload_from_ed,
    )

    pd_data = np.asarray(sample.get("pd_data", sample.get("load_data")), dtype=float)
    renewable_data = sample.get("renewable_data")
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    x_arr = np.asarray(x_matrix, dtype=float)
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
    sample["lambda"] = serialized
    sample["lambda_pg_electricity_price"] = serialized["lambda_pg_effective"]
    sample["lambda_refresh_source"] = "recomputed_from_ed_after_pattern_conversion"


def _load_json(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_pattern_lookup(pattern_library: List[Dict]) -> Dict[int, Dict[int, List[int]]]:
    lookup: Dict[int, Dict[int, List[int]]] = {}
    for generator_entry in pattern_library:
        generator_id = int(generator_entry["generator_id"])
        generator_patterns: Dict[int, List[int]] = {}
        for pattern_entry in generator_entry.get("patterns", []):
            pattern_index = int(pattern_entry["pattern_index"])
            pattern_bits = [int(ch) for ch in str(pattern_entry["pattern"]).strip()]
            generator_patterns[pattern_index] = pattern_bits
        lookup[generator_id] = generator_patterns
    return lookup


def _reconstruct_commitment_matrix(
    selected_pattern_indices: List[int],
    pattern_lookup: Dict[int, Dict[int, List[int]]],
) -> List[List[int]]:
    matrix: List[List[int]] = []
    for generator_id, selected_index in enumerate(selected_pattern_indices):
        patterns_g = pattern_lookup.get(generator_id)
        if patterns_g is None:
            raise KeyError(f"generator_id={generator_id} not found in pattern_library")
        pattern = patterns_g.get(int(selected_index))
        if pattern is None:
            raise KeyError(
                f"pattern_index={selected_index} missing for generator_id={generator_id}"
            )
        matrix.append(pattern[:])
    return matrix


def _matrix_to_active_set(x_matrix: List[List[int]]) -> List[List[object]]:
    active_set: List[List[object]] = []
    for g, row in enumerate(x_matrix):
        for t, value in enumerate(row):
            active_set.append([[g, t], int(value)])
    return active_set


def _active_set_key(active_set: List[List[object]]) -> str:
    return json.dumps(active_set, ensure_ascii=False, separators=(",", ":"))


def _compute_size_statistics(active_sets: List[List[List[object]]]) -> Dict[str, float | None]:
    sizes = [len(active_set) for active_set in active_sets]
    if not sizes:
        return {
            "min_size": None,
            "max_size": None,
            "avg_size": None,
            "std_size": None,
        }

    avg_size = sum(sizes) / len(sizes)
    variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)
    return {
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "avg_size": float(avg_size),
        "std_size": float(variance ** 0.5),
    }


def convert_pattern_library_to_active_set_like(
    active_set_json_path: str,
    pattern_library_json_path: str,
    output_path: str,
    fallback_to_original: bool = True,
) -> str:
    active_set_data = _load_json(active_set_json_path)
    pattern_data = _load_json(pattern_library_json_path)

    original_samples = active_set_data.get("all_samples")
    if not isinstance(original_samples, list):
        raise ValueError("Input active-set JSON must contain a top-level all_samples list")

    scenario_results = pattern_data.get("scenario_results") or pattern_data.get("scenarios")
    if not isinstance(scenario_results, list):
        raise ValueError(
            "Pattern-library JSON must contain scenario_results or scenarios"
        )

    pattern_library = pattern_data.get("pattern_library")
    if not isinstance(pattern_library, list):
        raise ValueError("Pattern-library JSON must contain pattern_library")

    case_name = active_set_data.get("metadata", {}).get("case_name")
    t_delta = float(active_set_data.get("parameters", {}).get("T_delta", 1.0))
    ppc = _load_ppc_for_case(case_name)

    pattern_lookup = _build_pattern_lookup(pattern_library)
    scenario_by_sample_id = {
        int(scenario["sample_id"]): scenario
        for scenario in scenario_results
        if "sample_id" in scenario
    }

    converted_samples: List[Dict] = []
    converted_active_sets: List[List[List[object]]] = []
    unique_active_sets: List[List[List[object]]] = []
    seen_active_sets = set()
    dual_refreshed_sample_ids: List[int] = []

    for sample in original_samples:
        sample_id = int(sample["sample_id"])
        scenario = scenario_by_sample_id.get(sample_id)
        converted = copy.deepcopy(sample)
        original_x = copy.deepcopy(sample.get("unit_commitment_matrix"))
        conversion_status = "converted"

        if scenario is None:
            if not fallback_to_original:
                raise KeyError(f"sample_id={sample_id} missing from pattern-library results")
            x_matrix = original_x
            active_set = copy.deepcopy(sample.get("active_set"))
            conversion_status = "missing_pattern_result_fallback_to_original"
        elif not scenario.get("success", False):
            if not fallback_to_original:
                raise ValueError(
                    f"sample_id={sample_id} has no feasible restricted solution in pattern results"
                )
            x_matrix = original_x
            active_set = copy.deepcopy(sample.get("active_set"))
            conversion_status = "failed_pattern_result_fallback_to_original"
        else:
            selected_pattern_indices = scenario.get("selected_pattern_indices")
            if not isinstance(selected_pattern_indices, list):
                raise ValueError(
                    f"sample_id={sample_id} missing selected_pattern_indices in pattern results"
                )
            x_matrix = _reconstruct_commitment_matrix(selected_pattern_indices, pattern_lookup)
            active_set = _matrix_to_active_set(x_matrix)
            conversion_status = "converted_from_pattern_library"

        converted["original_unit_commitment_matrix"] = original_x
        converted["unit_commitment_matrix"] = x_matrix
        converted["active_set"] = active_set
        converted["conversion_status"] = conversion_status
        if conversion_status == "converted_from_pattern_library":
            _refresh_sample_dual_payload(converted, x_matrix, ppc, t_delta)
            dual_refreshed_sample_ids.append(sample_id)

        if scenario is not None:
            converted["source_solver_status"] = scenario.get("solver_status")
            converted["restricted_cost"] = scenario.get("restricted_cost")
            converted["cost_increase_pct"] = scenario.get("cost_increase_pct")
            converted["selected_pattern_indices"] = scenario.get("selected_pattern_indices")
            converted["matched_unit_patterns"] = scenario.get("matched_unit_patterns")
            converted["changed_unit_patterns"] = scenario.get("changed_unit_patterns")
            converted["solve_time_s"] = scenario.get("solve_time_s")
            if scenario.get("optimal_cost") is not None:
                converted["optimal_cost"] = scenario.get("optimal_cost")

        converted_samples.append(converted)
        converted_active_sets.append(active_set)

        active_set_key = _active_set_key(active_set)
        if active_set_key not in seen_active_sets:
            seen_active_sets.add(active_set_key)
            unique_active_sets.append(active_set)

    t_horizon = None
    if converted_samples:
        x0 = converted_samples[0].get("unit_commitment_matrix") or []
        if x0 and isinstance(x0, list) and isinstance(x0[0], list):
            t_horizon = len(x0[0])

    output = {
        "metadata": {
            "case_name": active_set_data.get("metadata", {}).get("case_name"),
            "total_active_sets": len(unique_active_sets),
            "total_samples": len(converted_samples),
            "T": t_horizon,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "size_statistics": _compute_size_statistics(converted_active_sets),
            "source_active_set_json": str(Path(active_set_json_path)),
            "source_pattern_library_json": str(Path(pattern_library_json_path)),
            "conversion_type": "pattern_library_to_active_set_like",
            "dual_payload_refreshed_count": len(dual_refreshed_sample_ids),
            "dual_payload_refreshed_sample_ids": dual_refreshed_sample_ids,
        },
        "parameters": {
            **active_set_data.get("parameters", {}),
            "fallback_to_original": bool(fallback_to_original),
        },
        "unique_active_sets": unique_active_sets,
        "all_samples": converted_samples,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-set-json", default=None, type=str)
    parser.add_argument("--pattern-library-json", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument(
        "--no-fallback-to-original",
        action="store_true",
        help="Fail if a sample is missing or unsuccessful in the pattern-library result",
    )
    args = parser.parse_args()

    active_set_json = args.active_set_json or DEFAULT_ACTIVE_SET_JSON
    pattern_library_json = args.pattern_library_json or DEFAULT_PATTERN_LIBRARY_JSON
    output_json = args.output or DEFAULT_OUTPUT_JSON
    fallback_to_original = (
        DEFAULT_FALLBACK_TO_ORIGINAL if not args.no_fallback_to_original else False
    )

    output_path = convert_pattern_library_to_active_set_like(
        active_set_json_path=active_set_json,
        pattern_library_json_path=pattern_library_json,
        output_path=output_json,
        fallback_to_original=fallback_to_original,
    )
    print(f"Converted JSON written to: {output_path}")


if __name__ == "__main__":
    main()
