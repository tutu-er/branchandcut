#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect commitment classes and build a balanced active-set subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def extract_commitment_matrix(sample: dict) -> list[list[int]]:
    if "unit_commitment_matrix" in sample:
        return [[int(round(float(v))) for v in row] for row in sample["unit_commitment_matrix"]]

    active_set = sample.get("active_set", [])
    max_g = max(int(item[0][0]) for item in active_set)
    max_t = max(int(item[0][1]) for item in active_set)
    matrix = [[0 for _ in range(max_t + 1)] for _ in range(max_g + 1)]
    for item in active_set:
        g, t = item[0]
        matrix[int(g)][int(t)] = int(round(float(item[1])))
    return matrix


def commitment_key(sample: dict) -> tuple[int, ...]:
    matrix = extract_commitment_matrix(sample)
    return tuple(v for row in matrix for v in row)


def key_to_matrix(key: tuple[int, ...], ng: int, horizon: int) -> list[list[int]]:
    return [list(key[g * horizon : (g + 1) * horizon]) for g in range(ng)]


def print_distribution(samples: list[dict], top: int) -> Counter:
    counter: Counter = Counter(commitment_key(sample) for sample in samples)
    first_idx: dict[tuple[int, ...], int] = {}
    for idx, sample in enumerate(samples):
        first_idx.setdefault(commitment_key(sample), idx)

    first_matrix = extract_commitment_matrix(samples[0])
    ng = len(first_matrix)
    horizon = len(first_matrix[0])

    print(f"total_samples: {len(samples)}")
    print(f"unique_commitments: {len(counter)}")
    print("distribution:")
    for rank, (key, freq) in enumerate(counter.most_common(top), start=1):
        matrix = key_to_matrix(key, ng, horizon)
        idx = first_idx[key]
        sample_id = samples[idx].get("sample_id", idx)
        print(
            f"  pattern {rank}: freq={freq}, first_index={idx}, "
            f"sample_id={sample_id}, on_hours={[sum(row) for row in matrix]}"
        )
        for g, row in enumerate(matrix):
            print(f"    unit{g}: {''.join(str(v) for v in row)}")
    return counter


def build_balanced_subset(samples: list[dict], n_classes: int, per_class: int | None) -> tuple[list[dict], list[int]]:
    grouped: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[commitment_key(sample)].append(sample)

    ranked = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    selected_groups = ranked[: int(n_classes)]
    if not selected_groups:
        raise ValueError("no commitment classes found")
    target = min(len(group) for _, group in selected_groups)
    if per_class is not None:
        target = min(target, int(per_class))
    if target <= 0:
        raise ValueError("balanced target per class must be positive")

    subset: list[dict] = []
    counts: list[int] = []
    chosen_groups: list[list[dict]] = []
    for _key, group in selected_groups:
        chosen = [dict(sample) for sample in group[:target]]
        counts.append(len(chosen))
        chosen_groups.append(chosen)

    for i in range(target):
        for group in chosen_groups:
            subset.append(group[i])

    for idx, sample in enumerate(subset):
        sample["sample_id"] = idx
    return subset, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("all_samples", [])
    if not samples:
        raise ValueError(f"no all_samples in {args.input}")

    counter = print_distribution(samples, args.top)
    if args.inspect_only:
        return

    subset, counts = build_balanced_subset(samples, args.classes, args.per_class)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output
    if output is None:
        case_name = data.get("metadata", {}).get("case_name", "unknown")
        horizon = data.get("metadata", {}).get("T", 0)
        output = (
            Path("result")
            / "active_set"
            / f"active_sets_{case_name}_top{args.classes}_balanced_T{horizon}_n{len(subset)}_{timestamp}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    balanced = dict(data)
    balanced["all_samples"] = subset
    balanced["unique_active_sets"] = [list(key) for key, _freq in counter.most_common(args.classes)]
    balanced["metadata"] = dict(data.get("metadata", {}))
    balanced["metadata"].update(
        {
            "total_active_sets": int(args.classes),
            "total_samples": len(subset),
            "timestamp": timestamp,
            "source_file": str(args.input).replace("\\", "/"),
            "balanced_commitment_subset": {
                "classes": int(args.classes),
                "per_class_counts": counts,
                "source_top_frequencies": [
                    int(freq) for _key, freq in counter.most_common(args.classes)
                ],
            },
        }
    )
    with output.open("w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, separators=(",", ":"))
    print(f"balanced_output: {output}")
    print(f"balanced_counts: {counts}")


if __name__ == "__main__":
    main()
