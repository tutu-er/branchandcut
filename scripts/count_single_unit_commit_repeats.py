#!/usr/bin/env python3
"""Count repeated per-unit commitment patterns in an active-set JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _infer_shape_from_active_set(active_set: list) -> tuple[int, int] | None:
    max_g = -1
    max_t = -1
    for item in active_set:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], list)
            and len(item[0]) == 2
        ):
            g, t = item[0]
        elif isinstance(item, dict):
            g = item.get("unit_id", item.get("generator_id", item.get("g")))
            t = item.get("time_slot", item.get("time", item.get("t")))
        else:
            continue
        if g is None or t is None:
            continue
        max_g = max(max_g, int(g))
        max_t = max(max_t, int(t))
    if max_g < 0 or max_t < 0:
        return None
    return max_g + 1, max_t + 1


def extract_commitment_matrix(sample: dict) -> list[list[int]] | None:
    x_true = sample.get("x_true")
    if isinstance(x_true, list) and x_true and isinstance(x_true[0], list):
        return [[int(round(value)) for value in row] for row in x_true]

    uc = sample.get("unit_commitment_matrix")
    if isinstance(uc, list) and uc and isinstance(uc[0], list):
        return [[int(round(value)) for value in row] for row in uc]

    active_set = sample.get("active_set")
    if not isinstance(active_set, list):
        return None

    shape = _infer_shape_from_active_set(active_set)
    if shape is None:
        return None

    ng, horizon = shape
    matrix = [[0 for _ in range(horizon)] for _ in range(ng)]
    found = False
    for item in active_set:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], list)
            and len(item[0]) == 2
        ):
            g, t = item[0]
            value = item[1]
        elif isinstance(item, dict):
            g = item.get("unit_id", item.get("generator_id", item.get("g")))
            t = item.get("time_slot", item.get("time", item.get("t")))
            value = item.get("value", item.get("x"))
        else:
            continue
        if g is None or t is None or value is None:
            continue
        g = int(g)
        t = int(t)
        if 0 <= g < ng and 0 <= t < horizon:
            matrix[g][t] = int(round(float(value)))
            found = True
    if not found:
        return None
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count repeated single-unit commitment patterns in an active-set JSON file.",
    )
    parser.add_argument("json_path", type=Path, help="path to active-set JSON file")
    parser.add_argument("--top", type=int, default=3, help="top repeated patterns to print per unit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("all_samples", [])
    if not samples:
        raise ValueError(f"no samples found in {args.json_path}")

    matrices: list[list[list[int]]] = []
    skipped = 0
    for sample in samples:
        matrix = extract_commitment_matrix(sample)
        if matrix is None:
            skipped += 1
            continue
        matrices.append(matrix)

    if not matrices:
        raise ValueError(f"failed to decode any commitment matrix from {args.json_path}")

    ng = len(matrices[0])
    horizon = len(matrices[0][0]) if matrices[0] else 0
    unit_counters = [Counter() for _ in range(ng)]

    for matrix in matrices:
        if len(matrix) != ng:
            raise ValueError("inconsistent generator dimension across samples")
        for g in range(ng):
            pattern = tuple(int(v) for v in matrix[g])
            unit_counters[g][pattern] += 1

    print(f"json_path: {args.json_path}")
    print(f"decoded_samples: {len(matrices)}")
    print(f"skipped_samples: {skipped}")
    print(f"ng: {ng}")
    print(f"horizon: {horizon}")
    print()

    unique_counts = [len(counter) for counter in unit_counters]
    max_freqs = [max(counter.values()) for counter in unit_counters]
    repeat_pattern_counts = [sum(1 for freq in counter.values() if freq > 1) for counter in unit_counters]

    print("Summary")
    print(f"  mean_unique_patterns_per_unit: {sum(unique_counts) / max(ng, 1):.2f}")
    print(f"  min_unique_patterns_per_unit:  {min(unique_counts)}")
    print(f"  max_unique_patterns_per_unit:  {max(unique_counts)}")
    print(f"  mean_max_frequency_per_unit:   {sum(max_freqs) / max(ng, 1):.2f}")
    print()

    for g, counter in enumerate(unit_counters):
        unique_patterns = len(counter)
        repeated_patterns = sum(1 for freq in counter.values() if freq > 1)
        repeated_samples = sum(freq for freq in counter.values() if freq > 1)
        max_freq = max(counter.values())
        print(
            f"Unit {g:02d}: unique={unique_patterns:3d}  "
            f"repeated_patterns={repeated_patterns:3d}  "
            f"repeated_samples={repeated_samples:3d}  "
            f"max_freq={max_freq:3d}"
        )
        for rank, (pattern, freq) in enumerate(counter.most_common(args.top), start=1):
            on_hours = sum(pattern)
            print(f"  top{rank}: freq={freq:3d}, on_hours={on_hours:2d}, pattern={''.join(str(v) for v in pattern)}")


if __name__ == "__main__":
    main()
