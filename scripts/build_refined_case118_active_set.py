"""Build a refined case118 active-set-like JSON from batch refinement results.

This script starts from an existing active_set_like JSON produced from the
pattern-library workflow, then overwrites selected samples using the
``full_repair`` solutions stored in a refinement report.

The output remains compatible with ``src.uc_NN_subproblem.ActiveSetReader``:
- top-level ``unique_active_sets`` and ``all_samples``
- per-sample ``active_set`` containing binary-variable entries
- per-sample ``unit_commitment_matrix`` / ``pd_data`` / ``load_data``
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


def _load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pattern_library_lookup(pattern_library_data: Dict) -> List[List[List[int]]]:
    lookup: List[List[List[int]]] = []
    for generator_entry in pattern_library_data["pattern_library"]:
        patterns_g: List[List[int]] = []
        for pattern_entry in sorted(
            generator_entry.get("patterns", []),
            key=lambda item: int(item["pattern_index"]),
        ):
            pattern_bits = [int(ch) for ch in str(pattern_entry["pattern"]).strip()]
            patterns_g.append(pattern_bits)
        lookup.append(patterns_g)
    return lookup


def _matrix_to_active_set(x_matrix: Sequence[Sequence[int]]) -> List[List[object]]:
    active_set: List[List[object]] = []
    for g, row in enumerate(x_matrix):
        for t, value in enumerate(row):
            active_set.append([[int(g), int(t)], int(value)])
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


def _normalize_refinement_rows(refinement_data: Dict) -> List[Dict]:
    if "batch_results" in refinement_data:
        return list(refinement_data["batch_results"])

    if "full_repair" in refinement_data:
        sample_id = refinement_data.get("sample_id")
        if sample_id is None:
            metadata = refinement_data.get("metadata", {})
            sample_id = metadata.get("sample_id")
        if sample_id is None:
            raise ValueError("Single-sample refinement JSON is missing sample_id")

        row = copy.deepcopy(refinement_data)
        row["sample_id"] = int(sample_id)
        row.pop("metadata", None)
        row.pop("parameters", None)
        return [row]

    raise ValueError("Unsupported refinement JSON: expected batch_results or full_repair")


def _sample_index(samples: Sequence[Dict]) -> Dict[int, Dict]:
    return {int(sample["sample_id"]): sample for sample in samples}


def _reconstruct_refined_matrix(
    base_patterns: List[List[List[int]]],
    original_sample: Dict,
    selected_pattern_indices: Sequence[int],
    added_generators: Sequence[int],
) -> List[List[int]]:
    original_x = original_sample.get("original_unit_commitment_matrix")
    if original_x is None:
        original_x = original_sample.get("unit_commitment_matrix")
    if original_x is None:
        raise ValueError(f"sample_id={original_sample.get('sample_id')} missing original commitment matrix")

    added_set = {int(g) for g in added_generators}
    original_x = [[int(v) for v in row] for row in original_x]
    refined_x: List[List[int]] = []

    if len(selected_pattern_indices) != len(base_patterns):
        raise ValueError(
            f"selected_pattern_indices length {len(selected_pattern_indices)} "
            f"!= number of generators {len(base_patterns)}"
        )

    for g, selected_idx_raw in enumerate(selected_pattern_indices):
        selected_idx = int(selected_idx_raw)
        patterns_g = base_patterns[g]
        base_count = len(patterns_g)

        if 0 <= selected_idx < base_count:
            refined_x.append([int(v) for v in patterns_g[selected_idx]])
            continue

        if selected_idx == base_count and g in added_set:
            refined_x.append([int(v) for v in original_x[g]])
            continue

        raise ValueError(
            f"generator_id={g} has invalid selected_pattern_index={selected_idx}; "
            f"base_count={base_count}, added={g in added_set}"
        )

    return refined_x


def build_refined_active_set(
    active_set_like_json: str,
    pattern_library_json: str,
    refinement_json: str,
    output_json: str,
) -> str:
    active_set_like = _load_json(active_set_like_json)
    pattern_library_data = _load_json(pattern_library_json)
    refinement_data = _load_json(refinement_json)

    all_samples = copy.deepcopy(active_set_like.get("all_samples", []))
    if not isinstance(all_samples, list) or not all_samples:
        raise ValueError("active_set_like JSON must contain a non-empty all_samples list")

    samples_by_id = _sample_index(all_samples)
    base_patterns = _pattern_library_lookup(pattern_library_data)
    refinement_rows = _normalize_refinement_rows(refinement_data)

    refined_sample_ids: List[int] = []
    for row in refinement_rows:
        sample_id = int(row["sample_id"])
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"sample_id={sample_id} missing from active_set_like JSON")

        full_repair = row.get("full_repair") or {}
        if not full_repair.get("success", False):
            continue

        refined_x = _reconstruct_refined_matrix(
            base_patterns=base_patterns,
            original_sample=sample,
            selected_pattern_indices=full_repair["selected_pattern_indices"],
            added_generators=full_repair.get("added_generators", []),
        )
        refined_active_set = _matrix_to_active_set(refined_x)

        sample["unit_commitment_matrix"] = refined_x
        sample["active_set"] = refined_active_set
        sample["conversion_status"] = "refined_from_pattern_library_full_repair"
        sample["selected_pattern_indices"] = list(full_repair["selected_pattern_indices"])
        sample["matched_unit_patterns"] = full_repair.get("matched_unit_patterns")
        sample["changed_unit_patterns"] = full_repair.get("changed_unit_patterns")
        sample["restricted_cost"] = full_repair.get("objective_value")
        sample["cost_increase_pct"] = full_repair.get("gap_pct")
        sample["source_solver_status"] = full_repair.get("status")
        sample["refinement_source"] = "full_repair"
        sample["refinement_added_generators"] = list(full_repair.get("added_generators", []))

        refined_sample_ids.append(sample_id)

    unique_active_sets: List[List[List[object]]] = []
    seen_active_sets = set()
    converted_active_sets: List[List[List[object]]] = []
    for sample in all_samples:
        active_set = sample.get("active_set", [])
        converted_active_sets.append(active_set)
        key = _active_set_key(active_set)
        if key not in seen_active_sets:
            seen_active_sets.add(key)
            unique_active_sets.append(active_set)

    output = copy.deepcopy(active_set_like)
    output["metadata"] = {
        **active_set_like.get("metadata", {}),
        "total_active_sets": len(unique_active_sets),
        "total_samples": len(all_samples),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "size_statistics": _compute_size_statistics(converted_active_sets),
        "source_pattern_library_json": str(Path(pattern_library_json)),
        "source_refinement_json": str(Path(refinement_json)),
        "conversion_type": "pattern_library_active_set_like_refined",
        "refined_sample_count": len(refined_sample_ids),
        "refined_sample_ids": sorted(refined_sample_ids),
    }
    output["unique_active_sets"] = unique_active_sets
    output["all_samples"] = all_samples
    output["parameters"] = {
        **active_set_like.get("parameters", {}),
        "refinement_applied": True,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-set-like-json", required=True, type=str)
    parser.add_argument("--pattern-library-json", required=True, type=str)
    parser.add_argument("--refinement-json", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    output = build_refined_active_set(
        active_set_like_json=args.active_set_like_json,
        pattern_library_json=args.pattern_library_json,
        refinement_json=args.refinement_json,
        output_json=args.output,
    )
    print(f"Refined active-set JSON written to: {output}")


if __name__ == "__main__":
    main()
