"""Rank case118 generators by commitment-pattern variability.

By default this analyzes the three case118 K10 active-set-like files produced
from the 20260418 pattern-library run:

* active_set_like
* active_set_like_refined
* active_set_like_refined_price_only_clipped

Examples
--------
    python scripts/analyze_case118_commitment_variability.py
    python scripts/analyze_case118_commitment_variability.py --top 15
    python scripts/analyze_case118_commitment_variability.py --rank-by entropy
    python scripts/analyze_case118_commitment_variability.py --input result/active_set/active_sets_case118_T0_n366_20260322_063917.json
    python scripts/analyze_case118_commitment_variability.py --csv result/commitment_clustering/case118_K10_unit_variability.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "active_sets_case118_*.json"
DEFAULT_CASE118_K10_INPUTS = [
    ROOT
    / "result"
    / "commitment_clustering"
    / "pattern_library_case118_K10_20260418_032025_active_set_like_20260418_032025.json",
    ROOT
    / "result"
    / "commitment_clustering"
    / "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025.json",
    ROOT
    / "result"
    / "commitment_clustering"
    / (
        "pattern_library_case118_K10_20260418_032025_active_set_like_refined_"
        "20260418_032025_price_only_clipped.json"
    ),
]


def find_latest_case118_active_set() -> Path:
    active_set_dir = ROOT / "result" / "active_set"
    candidates = sorted(
        active_set_dir.glob(DEFAULT_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No case118 active-set JSON found under {active_set_dir}"
        )
    return candidates[0]


def _iter_binary_entries(active_set: Iterable[Any]):
    for item in active_set:
        if not (isinstance(item, list) and len(item) == 2):
            continue
        index, value = item
        if isinstance(index, list) and len(index) == 2:
            yield int(index[0]), int(index[1]), int(round(float(value)))


def _infer_shape(samples: list[dict[str, Any]]) -> tuple[int, int]:
    max_g = -1
    max_t = -1
    for sample in samples:
        matrix = sample.get("unit_commitment_matrix")
        if matrix is not None:
            arr = np.asarray(matrix)
            if arr.ndim == 2 and arr.size:
                max_g = max(max_g, arr.shape[0] - 1)
                max_t = max(max_t, arr.shape[1] - 1)
                continue
        for g, t, _ in _iter_binary_entries(sample.get("active_set", [])):
            max_g = max(max_g, g)
            max_t = max(max_t, t)

    if max_g < 0 or max_t < 0:
        raise ValueError("Could not infer commitment matrix shape from samples")
    return max_g + 1, max_t + 1


def sample_to_commitment(sample: dict[str, Any], ng: int, horizon: int) -> np.ndarray:
    matrix = sample.get("unit_commitment_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=int)
        if arr.ndim == 2 and arr.size:
            out = np.zeros((ng, horizon), dtype=int)
            rows = min(ng, arr.shape[0])
            cols = min(horizon, arr.shape[1])
            out[:rows, :cols] = arr[:rows, :cols]
            return out

    out = np.zeros((ng, horizon), dtype=int)
    for g, t, value in _iter_binary_entries(sample.get("active_set", [])):
        if 0 <= g < ng and 0 <= t < horizon:
            out[g, t] = 1 if value >= 1 else 0
    return out


def _summarize_source(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata") or {}
    parameters = data.get("parameters") or {}
    samples = data.get("all_samples") or []
    status_counts = Counter(str(sample.get("conversion_status", "missing")) for sample in samples)
    refined_ids = metadata.get("refined_sample_ids")
    if refined_ids is None:
        refined_ids = [
            int(sample.get("sample_id", idx))
            for idx, sample in enumerate(samples)
            if str(sample.get("conversion_status", "")).startswith("refined_")
        ]

    return {
        "conversion_type": metadata.get("conversion_type", "unknown"),
        "metadata_samples": metadata.get("total_samples"),
        "metadata_active_sets": metadata.get("total_active_sets"),
        "timestamp": metadata.get("timestamp"),
        "refinement_applied": parameters.get("refinement_applied", False),
        "refined_sample_count": len(refined_ids or []),
        "refined_sample_ids": list(refined_ids or []),
        "status_counts": dict(status_counts),
        "has_metadata": bool(metadata),
        "has_parameters": bool(parameters),
    }


def load_commitment_stack(
    path: Path,
    max_samples: int | None,
) -> tuple[np.ndarray, int, dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("all_samples", [])
    source_summary = _summarize_source(data)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No all_samples found in {path}")

    ng, horizon = _infer_shape(samples)
    matrices = [
        sample_to_commitment(sample, ng, horizon)
        for sample in samples
        if "error" not in sample
    ]
    if not matrices:
        raise ValueError(f"No valid commitment samples found in {path}")
    valid_samples = [sample for sample in samples if "error" not in sample]
    return np.stack(matrices, axis=0), len(samples), source_summary, valid_samples


def pattern_string(pattern: tuple[int, ...]) -> str:
    return "".join(str(int(v)) for v in pattern)


def analyze_units(stack: np.ndarray) -> list[dict[str, Any]]:
    n_samples, ng, horizon = stack.shape
    rows: list[dict[str, Any]] = []
    denom = max(n_samples * max(horizon - 1, 1), 1)

    for g in range(ng):
        unit = stack[:, g, :].astype(int)
        transitions = np.abs(np.diff(unit, axis=1))
        startups = np.logical_and(unit[:, 1:] == 1, unit[:, :-1] == 0)
        shutdowns = np.logical_and(unit[:, 1:] == 0, unit[:, :-1] == 1)
        patterns = [tuple(int(v) for v in row) for row in unit]
        counts = Counter(patterns)
        most_common_pattern, most_common_count = counts.most_common(1)[0]
        entropy = -sum(
            (count / n_samples) * math.log2(count / n_samples)
            for count in counts.values()
        )
        max_entropy = math.log2(n_samples) if n_samples > 1 else 0.0

        hourly_on_rate = unit.mean(axis=0)
        hamming_variability = float(np.mean(2.0 * hourly_on_rate * (1.0 - hourly_on_rate)))
        rows.append(
            {
                "unit": g,
                "unique_patterns": len(counts),
                "total_transitions": int(transitions.sum()),
                "avg_transitions": float(transitions.sum() / n_samples),
                "transition_rate": float(transitions.sum() / denom),
                "startups": int(startups.sum()),
                "shutdowns": int(shutdowns.sum()),
                "on_rate": float(unit.mean()),
                "hamming_variability": hamming_variability,
                "entropy": float(entropy),
                "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
                "most_common_count": int(most_common_count),
                "most_common_frac": float(most_common_count / n_samples),
                "most_common_pattern": pattern_string(most_common_pattern),
            }
        )
    return rows


def add_original_diff_metrics(
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    ng: int,
    horizon: int,
) -> None:
    diff_totals = np.zeros(ng, dtype=int)
    comparable = 0
    for sample in samples:
        original = sample.get("original_unit_commitment_matrix")
        current = sample.get("unit_commitment_matrix")
        if original is None or current is None:
            continue
        original_arr = np.asarray(original, dtype=int)
        current_arr = np.asarray(current, dtype=int)
        if original_arr.ndim != 2 or current_arr.ndim != 2:
            continue
        rows_g = min(ng, original_arr.shape[0], current_arr.shape[0])
        cols_t = min(horizon, original_arr.shape[1], current_arr.shape[1])
        if rows_g <= 0 or cols_t <= 0:
            continue
        diff_totals[:rows_g] += np.abs(
            current_arr[:rows_g, :cols_t] - original_arr[:rows_g, :cols_t]
        ).sum(axis=1)
        comparable += 1

    denom = max(comparable * horizon, 1)
    for row in rows:
        unit = int(row["unit"])
        row["diff_from_original"] = int(diff_totals[unit])
        row["diff_from_original_rate"] = float(diff_totals[unit] / denom)
    if comparable == 0:
        for row in rows:
            row["diff_from_original"] = 0
            row["diff_from_original_rate"] = 0.0


def sort_rows(rows: list[dict[str, Any]], rank_by: str) -> list[dict[str, Any]]:
    secondary = {
        "unique_patterns": "avg_transitions",
        "avg_transitions": "unique_patterns",
        "total_transitions": "unique_patterns",
        "transition_rate": "unique_patterns",
        "hamming_variability": "unique_patterns",
        "entropy": "unique_patterns",
        "normalized_entropy": "unique_patterns",
        "diff_from_original": "unique_patterns",
        "diff_from_original_rate": "unique_patterns",
    }[rank_by]
    return sorted(
        rows,
        key=lambda r: (r[rank_by], r[secondary], -r["most_common_frac"]),
        reverse=True,
    )


def print_table(rows: list[dict[str, Any]], top: int, rank_by: str) -> None:
    shown = rows[:top]
    print(f"\nTop {len(shown)} units by {rank_by}", flush=True)
    print(
        "unit | uniq | entropy | total_sw | avg_sw | sw_rate | hamming_var | "
        "orig_diff | on_rate | mode_frac | most_common_pattern",
        flush=True,
    )
    print("-" * 139, flush=True)
    for r in shown:
        print(
            f"{r['unit']:4d} | "
            f"{r['unique_patterns']:4d} | "
            f"{r['entropy']:7.3f} | "
            f"{r['total_transitions']:8d} | "
            f"{r['avg_transitions']:6.3f} | "
            f"{r['transition_rate']:7.4f} | "
            f"{r['hamming_variability']:11.4f} | "
            f"{r.get('diff_from_original', 0):9d} | "
            f"{r['on_rate']:7.3f} | "
            f"{r['most_common_frac']:9.3f} | "
            f"{r['most_common_pattern']}",
            flush=True,
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "rank_by",
        "rank",
        "unit",
        "unique_patterns",
        "entropy",
        "normalized_entropy",
        "total_transitions",
        "avg_transitions",
        "transition_rate",
        "startups",
        "shutdowns",
        "on_rate",
        "hamming_variability",
        "diff_from_original",
        "diff_from_original_rate",
        "most_common_count",
        "most_common_frac",
        "most_common_pattern",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help=(
            "Active-set-like JSON. Can be passed more than once. Defaults to the "
            "three case118 K10 files listed in this script."
        ),
    )
    parser.add_argument(
        "--latest-active-set",
        action="store_true",
        help="Analyze the latest result/active_set/active_sets_case118_*.json instead of the K10 defaults.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of units to print.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit.")
    parser.add_argument(
        "--rank-by",
        choices=[
            "unique_patterns",
            "avg_transitions",
            "total_transitions",
            "transition_rate",
            "hamming_variability",
            "entropy",
            "normalized_entropy",
            "diff_from_original",
            "diff_from_original_rate",
        ],
        default="unique_patterns",
        help="Metric used for ranking units.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.latest_active_set:
        input_paths = [find_latest_case118_active_set()]
    elif args.input:
        input_paths = args.input
    else:
        input_paths = DEFAULT_CASE118_K10_INPUTS

    all_csv_rows: list[dict[str, Any]] = []
    for raw_path in input_paths:
        input_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        stack, requested_samples, summary, samples = load_commitment_stack(
            input_path,
            args.max_samples,
        )
        n_samples, ng, horizon = stack.shape
        print(f"\nInput: {input_path}", flush=True)
        print(
            f"Loaded commitment stack: samples={n_samples}/{requested_samples}, "
            f"units={ng}, horizon={horizon}",
            flush=True,
        )
        print(
            "Source: "
            f"conversion_type={summary['conversion_type']}, "
            f"refinement_applied={summary['refinement_applied']}, "
            f"refined_sample_count={summary['refined_sample_count']}, "
            f"has_metadata={summary['has_metadata']}, "
            f"status_counts={summary['status_counts']}",
            flush=True,
        )
        if summary["refined_sample_ids"]:
            print(f"Refined sample ids: {summary['refined_sample_ids']}", flush=True)

        rows = analyze_units(stack)
        add_original_diff_metrics(rows, samples, ng, horizon)
        rows = sort_rows(rows, args.rank_by)
        for rank, row in enumerate(rows, start=1):
            row["source_file"] = input_path.name
            row["rank_by"] = args.rank_by
            row["rank"] = rank
        all_csv_rows.extend(rows)
        print_table(rows, max(args.top, 0), args.rank_by)

    if args.csv is not None:
        csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
        write_csv(csv_path, all_csv_rows)
        print(f"\nCSV written: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
