#!/usr/bin/env python3
"""Plot training metrics and snapshots from a training_metrics_*.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_metric_sections(metrics: dict) -> dict[str, list[dict]]:
    """顶层 list[dict] 与一层嵌套（如 surrogate/机组编号）均可识别。"""
    sections: dict[str, list[dict]] = {}
    for name, value in metrics.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            sections[name] = value
        elif isinstance(value, dict):
            for subname, subvalue in value.items():
                if (
                    isinstance(subvalue, list)
                    and subvalue
                    and isinstance(subvalue[0], dict)
                ):
                    sections[f"{name}/{subname}"] = subvalue
    return sections


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(data: dict, output_base: Path) -> int:
    sections = _pick_metric_sections(data.get("metrics", {}))
    if not sections:
        return 0

    metric_groups = [
        ("Objectives", ["obj_primal", "obj_dual", "obj_opt"]),
        ("Penalty Weights", ["rho_primal", "rho_dual", "rho_opt"]),
        ("Losses", ["nn_loss", "surr_nn_loss"]),
        ("Integrality", ["integrality"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Training Metrics: {output_base.stem}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (title, keys) in zip(axes.flat, metric_groups):
        plotted = False
        for section_name, records in sections.items():
            x_values = [record.get("iter", idx) for idx, record in enumerate(records)]
            for key in keys:
                y_values = [record.get(key) for record in records]
                if all(value is None for value in y_values):
                    continue
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    label=f"{section_name}:{key}",
                )
                plotted = True
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3, linestyle="--")
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#888888",
            )

    fig.tight_layout()
    _save_figure(fig, output_base)
    return 1


def _snapshot_matrix(snapshot: dict) -> np.ndarray:
    matrix = np.asarray(snapshot.get("data", []), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    return matrix


def plot_snapshot_heatmaps(data: dict, output_base: Path) -> int:
    snapshots = data.get("snapshots", {})
    available = {
        name: value
        for name, value in snapshots.items()
        if isinstance(value, list) and value
    }
    if not available:
        return 0

    row_names = [name for name in ("x", "pg", "lambda") if name in available]
    if not row_names:
        row_names = list(available.keys())
    n_rows = len(row_names)
    n_cols = max(len(available[name]) for name in row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Training Snapshots: {output_base.stem}",
        fontsize=14,
        fontweight="bold",
    )

    image_handles: dict[str, any] = {}
    cmap_by_row = {"x": "Blues", "pg": "viridis", "lambda": "magma"}

    for r, row_name in enumerate(row_names):
        row_snapshots = available[row_name]
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(row_snapshots):
                ax.axis("off")
                continue

            snapshot = row_snapshots[c]
            matrix = _snapshot_matrix(snapshot)
            image = ax.imshow(
                matrix,
                aspect="auto",
                cmap=cmap_by_row.get(row_name, "viridis"),
            )
            image_handles[row_name] = image
            stage = snapshot.get("stage", "?")
            iter_id = snapshot.get("iter", c)
            ax.set_title(f"{row_name} | {stage} iter {iter_id}", fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Index")

    for r, row_name in enumerate(row_names):
        if row_name in image_handles:
            fig.colorbar(
                image_handles[row_name],
                ax=axes[r, :].tolist(),
                fraction=0.02,
                pad=0.02,
            )

    fig.subplots_adjust(top=0.90, wspace=0.35, hspace=0.45)
    _save_figure(fig, output_base)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves and heatmaps from training_metrics JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to training_metrics_*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/figures"),
        help="Directory for plot outputs.",
    )
    args = parser.parse_args()

    data = _load_json(args.json_path)
    stem = args.json_path.stem

    curves_base = args.output_dir / f"{stem}_curves"
    snapshots_base = args.output_dir / f"{stem}_snapshots"

    curves_written = plot_metric_curves(data, curves_base)
    snapshots_written = plot_snapshot_heatmaps(data, snapshots_base)

    print(f"json: {args.json_path}")
    print(f"curves_written: {curves_written}")
    if curves_written:
        print(f"curves_png: {curves_base.with_suffix('.png')}")
    print(f"snapshots_written: {snapshots_written}")
    if snapshots_written:
        print(f"snapshots_png: {snapshots_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
