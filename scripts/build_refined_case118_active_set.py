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
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Embedded defaults for direct execution without CLI args.
DEFAULT_ACTIVE_SET_LIKE_JSON = (
    r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932_active_set_like.json"
)
DEFAULT_PATTERN_LIBRARY_JSON = (
    r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json"
)
DEFAULT_REFINEMENT_JSON = (
    r"result/commitment_clustering/sample_refinement_case118_batch.json"
)
DEFAULT_OUTPUT_JSON = (
    r"result/commitment_clustering/pattern_library_case118_K10_20260408_132932_active_set_like_refined.json"
)


def _load_case118_ppc():
    try:
        from src.mti118_data_loader import load_case118_ppc_with_mti_limits

        return load_case118_ppc_with_mti_limits(aggregate_thermal_by_bus=True)
    except ModuleNotFoundError as exc:
        if exc.name != "pandas":
            raise

    from pypower.api import case118
    from pypower.idx_brch import F_BUS, RATE_A, T_BUS
    from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG

    def safe_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def bus_token_to_number(token: str) -> int | None:
        match = re.fullmatch(r"bus0*([0-9]+)", str(token).strip().lower())
        return None if match is None else int(match.group(1))

    def infer_fuel_key(generator_name: str) -> str | None:
        name = str(generator_name).strip()
        if name.startswith(("Solar ", "Wind ", "Hydro ", "Geo ")):
            return None
        if "Biomass" in name:
            return "biomass"
        if "Oil" in name:
            return "oil"
        if "Coal" in name:
            return "coal"
        if "NG" in name or "Natural Gas" in name:
            return "natural gas"
        return "natural gas"

    data_root = ROOT / "data"
    addl_dir = data_root / "additional-files-mti-118" / "Additional Files MTI 118"
    generators_path = addl_dir / "Generators.csv"
    lines_path = addl_dir / "Lines.csv"
    fuels_path = addl_dir / "Fuels and emission rates.csv"

    ppc = case118()

    limits_by_edge: dict[tuple[int, int], list[float]] = {}
    with lines_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bus_from = bus_token_to_number(row.get("Bus from "))
            bus_to = bus_token_to_number(row.get("Bus to"))
            limit = safe_float(row.get("Max Flow (MW)"))
            if bus_from is None or bus_to is None or limit <= 0:
                continue
            limits_by_edge.setdefault(tuple(sorted((bus_from, bus_to))), []).append(limit)

    for branch_idx in range(ppc["branch"].shape[0]):
        bus_from = int(ppc["branch"][branch_idx, F_BUS])
        bus_to = int(ppc["branch"][branch_idx, T_BUS])
        limits = limits_by_edge.get(tuple(sorted((bus_from, bus_to))))
        if limits:
            ppc["branch"][branch_idx, RATE_A] = limits.pop(0)

    fuels: dict[str, float] = {}
    with fuels_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        name_col = reader.fieldnames[0] if reader.fieldnames else ""
        for row in reader:
            fuel_name = str(row.get(name_col, "")).strip().lower()
            if fuel_name:
                fuels[fuel_name] = safe_float(row.get("Fue price ($/MMBTU)"))

    grouped: dict[str, dict] = {}
    with generators_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            generator_name = str(row["Generator Name"]).strip()
            fuel_key = infer_fuel_key(generator_name)
            if fuel_key is None:
                continue

            bus_name = str(row["bus of connection"]).strip().lower()
            bus_number = bus_token_to_number(bus_name)
            if bus_number is None:
                continue

            pmax = safe_float(row["Max Capacity (MW)"])
            pmin = min(max(safe_float(row["Min Stable Level (MW)"]), 0.0), pmax)
            startup_cost = safe_float(row["Start Cost ($)"])
            shutdown_cost = 0.1 * startup_cost
            vom = safe_float(row["VO&M Charge ($/MWh)"])
            fuel_price = fuels.get(fuel_key, 0.0)
            heat_rate_base = safe_float(row["Heat Rate Base (MMBTU/hr)"])
            heat_rate_inc = safe_float(row["Heat Rate Inc Band 1 (BTU/kWh)"])
            variable_fuel_cost = fuel_price * (heat_rate_inc / 1000.0)
            linear_cost = max(vom + variable_fuel_cost, 0.01)
            no_load_cost = max(heat_rate_base * fuel_price + vom * pmin, 0.0)

            item = grouped.setdefault(
                bus_name,
                {
                    "bus_number": bus_number,
                    "bus_name": bus_name,
                    "pmax": 0.0,
                    "pmin": 0.0,
                    "startup_cost": 0.0,
                    "shutdown_cost": 0.0,
                    "no_load_cost": 0.0,
                    "weighted_linear_cost": 0.0,
                    "ramp_up": 0.0,
                    "ramp_down": 0.0,
                    "min_up": 1,
                    "min_down": 1,
                    "generator_names": [],
                },
            )
            item["pmax"] += pmax
            item["pmin"] += pmin
            item["startup_cost"] += startup_cost
            item["shutdown_cost"] += shutdown_cost
            item["no_load_cost"] += no_load_cost
            item["weighted_linear_cost"] += linear_cost * max(pmax, 0.0)
            item["ramp_up"] += max(safe_float(row["Max Ramp Up (MW/min)"]) * 60.0, 0.0)
            item["ramp_down"] += max(safe_float(row["Max Ramp Down (MW/min)"]) * 60.0, 0.0)
            item["min_up"] = max(item["min_up"], max(int(round(safe_float(row["Min Up Time (h)"], 1.0))), 1))
            item["min_down"] = max(item["min_down"], max(int(round(safe_float(row["Min Down Time (h)"], 1.0))), 1))
            item["generator_names"].append(generator_name)

    thermal_rows = []
    gencost_rows = []
    ramp_up_mw_per_h: list[float] = []
    ramp_down_mw_per_h: list[float] = []
    min_up_time_h: list[int] = []
    min_down_time_h: list[int] = []
    generator_names: list[str] = []
    aggregated_unit_counts: list[int] = []

    for group_key in sorted(grouped, key=lambda name: (grouped[name]["bus_number"], name)):
        item = grouped[group_key]
        pmax = item["pmax"]
        pmin = min(item["pmin"], pmax)
        linear_cost = item["weighted_linear_cost"] / max(pmax, 1e-9)

        gen_row = np.zeros(21, dtype=float)
        gen_row[GEN_BUS] = item["bus_number"]
        gen_row[PG] = 0.0
        gen_row[QG] = 0.0
        gen_row[QMAX] = 0.0
        gen_row[QMIN] = 0.0
        gen_row[VG] = 1.0
        gen_row[MBASE] = float(ppc["baseMVA"])
        gen_row[GEN_STATUS] = 1.0
        gen_row[PMAX] = pmax
        gen_row[PMIN] = pmin
        thermal_rows.append(gen_row)
        gencost_rows.append(np.array([
            2.0,
            item["startup_cost"],
            item["shutdown_cost"],
            3.0,
            0.0,
            linear_cost,
            item["no_load_cost"],
        ], dtype=float))
        ramp_up_mw_per_h.append(item["ramp_up"])
        ramp_down_mw_per_h.append(item["ramp_down"])
        min_up_time_h.append(item["min_up"])
        min_down_time_h.append(item["min_down"])
        aggregated_unit_counts.append(len(item["generator_names"]))
        generator_names.append(f"{item['bus_name']}_thermal_agg[{len(item['generator_names'])}]")

    ppc["gen"] = np.vstack(thermal_rows) if thermal_rows else np.zeros((0, 21), dtype=float)
    ppc["gencost"] = np.vstack(gencost_rows) if gencost_rows else np.zeros((0, 7), dtype=float)
    ppc["uc_ramp_up_mw_per_h"] = np.asarray(ramp_up_mw_per_h, dtype=float)
    ppc["uc_ramp_down_mw_per_h"] = np.asarray(ramp_down_mw_per_h, dtype=float)
    ppc["uc_min_up_time_h"] = np.asarray(min_up_time_h, dtype=int)
    ppc["uc_min_down_time_h"] = np.asarray(min_down_time_h, dtype=int)
    ppc["uc_generator_names"] = generator_names
    ppc["uc_aggregated_unit_counts"] = np.asarray(aggregated_unit_counts, dtype=int)
    ppc["uc_aggregate_by_bus"] = True
    return ppc


def _serialize_dual_payload(payload: Dict) -> Dict:
    return {
        key: np.asarray(value, dtype=float).tolist()
        for key, value in payload.items()
    }


def _refresh_sample_dual_payload(sample: Dict, ppc, t_delta: float) -> None:
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
    serialized = _serialize_dual_payload(payload)
    sample["lambda"] = serialized
    sample["lambda_pg_electricity_price"] = serialized["lambda_pg_effective"]
    sample["lambda_refresh_source"] = "recomputed_from_ed_after_refinement"


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
    require_full_repair_success_for_rows: bool = False,
) -> str:
    """Build a refined active_set_like JSON.

    Parameters
    ----------
    require_full_repair_success_for_rows:
        When True, any refinement row that has a ``full_repair`` block with
        ``success=False`` raises RuntimeError instead of being silently skipped.
        Also requires that every sample_id appearing in batch_results has a
        successful full_repair.  Default False preserves the original behaviour.
    """
    active_set_like = _load_json(active_set_like_json)
    pattern_library_data = _load_json(pattern_library_json)
    refinement_data = _load_json(refinement_json)

    all_samples = copy.deepcopy(active_set_like.get("all_samples", []))
    if not isinstance(all_samples, list) or not all_samples:
        raise ValueError("active_set_like JSON must contain a non-empty all_samples list")

    t_delta = float(active_set_like.get("parameters", {}).get("T_delta", 1.0))
    ppc = _load_case118_ppc()

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
            if require_full_repair_success_for_rows:
                raise RuntimeError(
                    f"[build_refined_active_set] full_repair failed for sample_id={sample_id} "
                    f"(status={full_repair.get('status')!r}). "
                    "Set require_full_repair_success_for_rows=False to skip instead."
                )
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

    dual_refreshed_sample_ids: List[int] = []
    for sample in all_samples:
        current_x = sample.get("unit_commitment_matrix")
        original_x = sample.get("original_unit_commitment_matrix")
        conversion_status = str(sample.get("conversion_status", ""))
        needs_refresh = False
        if current_x is not None and original_x is not None:
            needs_refresh = current_x != original_x
        if not needs_refresh and conversion_status in {
            "converted_from_pattern_library",
            "refined_from_pattern_library_full_repair",
        }:
            needs_refresh = True
        if not needs_refresh:
            continue
        _refresh_sample_dual_payload(sample, ppc, t_delta)
        dual_refreshed_sample_ids.append(int(sample["sample_id"]))

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
        "dual_payload_refreshed_count": len(dual_refreshed_sample_ids),
        "dual_payload_refreshed_sample_ids": sorted(dual_refreshed_sample_ids),
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
    parser.add_argument("--active-set-like-json", default=None, type=str)
    parser.add_argument("--pattern-library-json", default=None, type=str)
    parser.add_argument("--refinement-json", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise instead of skipping samples whose full_repair failed.",
    )
    args = parser.parse_args()

    active_set_like_json = args.active_set_like_json or DEFAULT_ACTIVE_SET_LIKE_JSON
    pattern_library_json = args.pattern_library_json or DEFAULT_PATTERN_LIBRARY_JSON
    refinement_json = args.refinement_json or DEFAULT_REFINEMENT_JSON
    output_json = args.output or DEFAULT_OUTPUT_JSON

    output = build_refined_active_set(
        active_set_like_json=active_set_like_json,
        pattern_library_json=pattern_library_json,
        refinement_json=refinement_json,
        output_json=output_json,
        require_full_repair_success_for_rows=args.strict,
    )
    print(f"Refined active-set JSON written to: {output}")


if __name__ == "__main__":
    main()
