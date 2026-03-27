#!/usr/bin/env python3
"""Count unique unit-commitment patterns in an active-set JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _extract_commitment_matrix(sample: dict) -> list[list[int]] | None:
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
    ng, t_horizon = shape
    matrix = [[0 for _ in range(t_horizon)] for _ in range(ng)]
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
        if 0 <= g < ng and 0 <= t < t_horizon:
            matrix[g][t] = int(round(float(value)))
            found = True
    if not found:
        return None
    return matrix


def _extract_commitment_key(sample: dict) -> tuple[int, ...] | None:
    matrix = _extract_commitment_matrix(sample)
    if matrix is None:
        return None
    return tuple(value for row in matrix for value in row)


def count_unique_commitments(samples: Iterable[dict]) -> tuple[Counter, int]:
    counter: Counter = Counter()
    skipped = 0
    for sample in samples:
        key = _extract_commitment_key(sample)
        if key is None:
            skipped += 1
            continue
        counter[key] += 1
    return counter, skipped


def collect_representative_commitments(
    samples: list[dict],
) -> list[tuple[int, int, list[list[int]]]]:
    representatives: list[tuple[int, int, list[list[int]]]] = []
    seen: dict[tuple[int, ...], int] = {}

    for idx, sample in enumerate(samples):
        matrix = _extract_commitment_matrix(sample)
        if matrix is None:
            continue
        key = tuple(value for row in matrix for value in row)
        if key in seen:
            continue
        sample_id = int(sample.get("sample_id", idx))
        representatives.append((sample_id, idx, matrix))
        seen[key] = len(representatives) - 1

    return representatives


def plot_commitment_heatmaps(
    samples: list[dict],
    output_path: Path,
    sample_count: int,
) -> int:
    matrices: list[tuple[int, list[list[int]]]] = []
    for idx, sample in enumerate(samples):
        matrix = _extract_commitment_matrix(sample)
        if matrix is None:
            continue
        sample_id = sample.get("sample_id", idx)
        matrices.append((int(sample_id), matrix))
        if len(matrices) >= sample_count:
            break

    if not matrices:
        return 0

    n_plots = len(matrices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle("Unit Commitment Heatmaps", fontsize=14, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    for ax, (sample_id, matrix) in zip(axes.flat, matrices):
        ax.axis("on")
        image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Sample {sample_id}", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Unit")

    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.subplots_adjust(top=0.90, wspace=0.35, hspace=0.45)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return n_plots


def plot_representative_commitments(
    samples: list[dict],
    counter: Counter,
    output_path: Path,
) -> int:
    representatives = collect_representative_commitments(samples)
    if not representatives:
        return 0

    n_plots = len(representatives)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (sample_id, _sample_idx, matrix) in zip(axes.flat, representatives):
        ax.axis("on")
        ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Unit")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.98, wspace=0.35, hspace=0.45)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return n_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count unique unit-commitment patterns in an active-set JSON file."
    )
    parser.add_argument("json_path", type=Path, help="Path to active_sets_*.json")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the top-N most frequent commitment patterns.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=10,
        help="Number of samples to visualize as commitment heatmaps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/figures"),
        help="Directory for heatmap outputs.",
    )
    args = parser.parse_args()

    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("all_samples", [])
    counter, skipped = count_unique_commitments(samples)

    print(f"file: {args.json_path}")
    print(f"total_samples: {len(samples)}")
    print(f"decoded_samples: {sum(counter.values())}")
    print(f"skipped_samples: {skipped}")
    print(f"unique_unit_commitments: {len(counter)}")

    if counter:
        print("top_patterns:")
        for idx, (_key, freq) in enumerate(counter.most_common(max(args.top, 0)), start=1):
            print(f"  {idx}. freq={freq}")

    if args.plot_samples > 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.json_path.stem}_unit_commit_heatmap"
        plotted = plot_commitment_heatmaps(samples, output_path, args.plot_samples)
        print(f"heatmap_samples_plotted: {plotted}")
        print(f"heatmap_output: {output_path.with_suffix('.png')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representative_path = args.output_dir / f"{args.json_path.stem}_unit_commit_representatives"
    representative_count = plot_representative_commitments(samples, counter, representative_path)
    print(f"representative_patterns_plotted: {representative_count}")
    print(f"representative_output: {representative_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
