#!/usr/bin/env python3
"""Plot main-proxy constraint activity heatmaps from activity CSV.

The expected input is ``main_model_activity_by_sample_row.csv`` produced by
``scripts/test_surrogate_dual_activity.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "result"
    / "paper_eval"
    / "20260512_183843"
    / "raw"
    / "runs"
    / "case3lite"
    / "S01_bcd_theta"
    / "raw"
    / "figures"
    / "case3lite_global_surrogate_solve_stats"
    / "activity"
    / "main_model_activity_by_sample_row.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "result" / "model_tests" / "case3lite_main_theta_activity_heatmap"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _as_int(value, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _as_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_constraint_heatmap(rows: list[dict], output_dir: Path, title_prefix: str) -> Path:
    samples = sorted({_as_int(row.get("sample_pos", row.get("sample_id"))) for row in rows})
    constraints = sorted(
        {
            (
                _as_int(row.get("branch_id")),
                _as_int(row.get("time_slot")),
                str(row.get("constraint_name", "")),
            )
            for row in rows
        }
    )
    sample_to_idx = {sample: idx for idx, sample in enumerate(samples)}
    constraint_to_idx = {item: idx for idx, item in enumerate(constraints)}
    matrix = np.full((len(samples), len(constraints)), np.nan, dtype=float)
    slack = np.full_like(matrix, np.nan, dtype=float)

    for row in rows:
        sample = _as_int(row.get("sample_pos", row.get("sample_id")))
        item = (
            _as_int(row.get("branch_id")),
            _as_int(row.get("time_slot")),
            str(row.get("constraint_name", "")),
        )
        if sample not in sample_to_idx or item not in constraint_to_idx:
            continue
        i = sample_to_idx[sample]
        j = constraint_to_idx[item]
        matrix[i, j] = 1.0 if _as_bool(row.get("is_row_active_1e_6")) else 0.0
        slack[i, j] = _as_float(row.get("abs_row_slack"))

    fig_w = max(10.0, min(24.0, len(constraints) * 0.18 + 3.0))
    fig_h = max(4.8, min(12.0, len(samples) * 0.23 + 2.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    active_rate = float(np.nanmean(matrix)) if matrix.size else float("nan")
    mean_slack = float(np.nanmean(slack)) if slack.size else float("nan")
    ax.set_title(
        f"{title_prefix}: sample-constraint activity heatmap\n"
        f"active rate={active_rate:.3f}, mean |slack|={mean_slack:.4g}",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("theta constraint ordered by constraint id and time")
    ax.set_ylabel("test sample")
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels([str(s) for s in samples], fontsize=7)

    if len(constraints) <= 100:
        labels = [f"id{b},t{t}" for b, t, _name in constraints]
        ax.set_xticks(np.arange(len(constraints)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ticks = np.linspace(0, len(constraints) - 1, 12, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"id{constraints[i][0]},t{constraints[i][1]}" for i in ticks], rotation=45, ha="right")

    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("active row (1=yes)")
    fig.tight_layout()
    out = output_dir / "case3_main_theta_sample_constraint_activity_heatmap"
    _save(fig, out)
    return out.with_suffix(".png")


def plot_branch_time_heatmap(rows: list[dict], output_dir: Path, title_prefix: str) -> Path:
    constraint_ids = sorted({_as_int(row.get("branch_id")) for row in rows})
    times = sorted({_as_int(row.get("time_slot")) for row in rows})
    matrix = np.full((len(constraint_ids), len(times)), np.nan, dtype=float)
    constraint_to_idx = {constraint_id: idx for idx, constraint_id in enumerate(constraint_ids)}
    time_to_idx = {time: idx for idx, time in enumerate(times)}

    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        constraint_id = _as_int(row.get("branch_id"))
        time = _as_int(row.get("time_slot"))
        grouped.setdefault((constraint_id, time), []).append(1.0 if _as_bool(row.get("is_row_active_1e_6")) else 0.0)

    for (constraint_id, time), vals in grouped.items():
        matrix[constraint_to_idx[constraint_id], time_to_idx[time]] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10.0, max(2.8, len(constraint_ids) * 0.55 + 1.8)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{title_prefix}: theta activity rate by constraint id and time", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("time slot")
    ax.set_ylabel("constraint id")
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels([str(t) for t in times], fontsize=7)
    ax.set_yticks(np.arange(len(constraint_ids)))
    ax.set_yticklabels([str(c) for c in constraint_ids])
    for i, constraint_id in enumerate(constraint_ids):
        for j, time in enumerate(times):
            value = matrix[i, j]
            if np.isfinite(value) and value > 0:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("activity rate")
    fig.tight_layout()
    out = output_dir / "case3_main_theta_branch_time_activity_rate_heatmap"
    _save(fig, out)
    return out.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kind", default="bcd_theta")
    parser.add_argument("--title-prefix", default="case3(c3) main proxy")
    args = parser.parse_args()

    input_csv = args.input_csv if args.input_csv.is_absolute() else ROOT / args.input_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = [row for row in _read_rows(input_csv) if str(row.get("kind", "")) == args.kind]
    if not rows:
        raise SystemExit(f"No rows with kind={args.kind!r} in {input_csv}")

    written = [
        plot_sample_constraint_heatmap(rows, output_dir, args.title_prefix),
        plot_branch_time_heatmap(rows, output_dir, args.title_prefix),
    ]
    print(f"input: {input_csv}")
    print(f"rows: {len(rows)}")
    for path in written:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
