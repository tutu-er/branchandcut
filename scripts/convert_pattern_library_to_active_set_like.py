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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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
    parser.add_argument("--active-set-json", required=True, type=str)
    parser.add_argument("--pattern-library-json", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--no-fallback-to-original",
        action="store_true",
        help="Fail if a sample is missing or unsuccessful in the pattern-library result",
    )
    args = parser.parse_args()

    output_path = convert_pattern_library_to_active_set_like(
        active_set_json_path=args.active_set_json,
        pattern_library_json_path=args.pattern_library_json,
        output_path=args.output,
        fallback_to_original=not args.no_fallback_to_original,
    )
    print(f"Converted JSON written to: {output_path}")


if __name__ == "__main__":
    main()
