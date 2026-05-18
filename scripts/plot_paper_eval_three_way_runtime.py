#!/usr/bin/env python3
"""Paper-style plots from a paper_eval run directory (subset of ablations).

Uses ``data/sample_metrics.csv`` produced by ``run_paper_evaluation_suite.py``.

Default methods retained:
  - ``bcd_theta`` (BCD \\theta-only global surrogate LP family)
  - ``surrogate_sign4`` (subproblem surrogate, sign4 scope)
  - ``full_joint`` (full surrogate scope + BCD \\theta+\\zeta LP path)

Surrogate solves record per-sample wall time in ``runtime_sec``. True UC MILP
timings are **not** in that CSV; supply them explicitly for runtime comparison::

  MILP summary (one row per case)::

    case,median_runtime_sec
    case14,12.3
    case30lite,45.6

  Or per-sample (aligned with ``sample_index``)::

    case,sample_index,runtime_sec
    case14,0,11.0
    case14,1,13.5

Outputs PNG/PDF next to figures/ under ``--paper-eval-dir`` (override with
``--output-dir``).

Other figures worth drawing (not all implemented here): Hamming/int gap vs
baseline LP from ``global_*`` columns; BCD slack / subproblem slack bar charts;
scatter runtime vs integrality_gap; empirical CDF of speedup vs MILP; cost-gap
histograms vs ``objective_uc_cost`` if MILP objectives are exported separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METHODS = ("bcd_theta", "surrogate_sign4", "full_joint")

METHOD_DISPLAY = {
    "bcd_theta": r"BCD $\theta$",
    "surrogate_sign4": "Surr. sign4",
    "full_joint": "Full (all)",
    "milp": "MILP (ref.)",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _sf(x: str | None) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _si(x: str | None) -> int | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def _load_milp_summary(path: Path) -> dict[str, float]:
    """case -> median runtime (seconds)."""
    rows = _read_csv(path)
    if not rows:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        case = str(row.get("case", "")).strip()
        med = _sf(row.get("median_runtime_sec") or row.get("median_sec"))
        if case and med is not None:
            out[case] = med
    return out


def _load_milp_per_sample(path: Path) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for row in _read_csv(path):
        case = str(row.get("case", "")).strip()
        sid = _si(row.get("sample_index"))
        rt = _sf(row.get("runtime_sec"))
        if case and sid is not None and rt is not None:
            out[case, sid] = rt
    return out


def _collect_by_case_method(
    rows: list[dict],
    *,
    methods: Iterable[str],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    methods_set = set(methods)
    structured: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        if str(row.get("level", "")).lower() != "sample":
            continue
        case = str(row.get("case", "")).strip()
        method = str(row.get("method", "")).strip()
        if method not in methods_set or not case:
            continue
        bucket = structured.setdefault(case, {}).setdefault(
            method,
            {"integrality_gap": [], "normalized_hamming": [], "runtime_sec": []},
        )
        for key in ("integrality_gap", "normalized_hamming", "runtime_sec"):
            v = _sf(row.get(key))
            if v is not None:
                bucket[key].append(v)
    return structured


def _boxplot_compatible(ax, data: list, tick_labels: list[str]) -> None:
    kwargs = dict(showfliers=False, patch_artist=True)
    try:
        ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    except TypeError:
        ax.boxplot(data, labels=tick_labels, **kwargs)


def _plot_case_panels(
    case: str,
    data: dict[str, dict[str, list[float]]],
    methods_order: tuple[str, ...],
    milp_medians: dict[str, float],
    milp_per_sample: dict[tuple[str, int], float] | None,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    titles_metrics = (
        ("runtime_sec", "Runtime (s)"),
        ("integrality_gap", "Integrality gap"),
        ("normalized_hamming", "Normalized Hamming"),
    )

    for ax, (field, ylab) in zip(axes.flat, titles_metrics):
        plot_data = []
        labels = []
        for m in methods_order:
            vals = data.get(m, {}).get(field, [])
            if vals:
                plot_data.append(vals)
                labels.append(METHOD_DISPLAY[m])

        milp_vals: list[float] = []
        if milp_per_sample:
            for sid in sorted({k[1] for k in milp_per_sample if k[0] == case}):
                key = (case, sid)
                if key in milp_per_sample:
                    milp_vals.append(milp_per_sample[key])

        if milp_vals:
            plot_data.append(milp_vals)
            labels.append(METHOD_DISPLAY["milp"])

        if plot_data:
            _boxplot_compatible(ax, plot_data, labels)
            for idx, vals in enumerate(plot_data, start=1):
                x = np.full(len(vals), idx, dtype=float)
                jitter = np.linspace(-0.07, 0.07, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(x + jitter, vals, s=9, alpha=0.4, color="#215b7a", zorder=3)
        elif case in milp_medians and field == "runtime_sec":
            ax.axhline(milp_medians[case], color="#c0392b", linestyle="--", linewidth=1.6, label="MILP median")
            ax.set_ylabel(ylab)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(case, loc="left", fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=18)
            continue

        # Horizontal MILP median when only summary provided (runtime panel could still show surrogate boxes)
        if field == "runtime_sec" and case in milp_medians and plot_data:
            ax.axhline(
                milp_medians[case],
                color="#c0392b",
                linestyle="--",
                linewidth=1.3,
                label=f"MILP median ({milp_medians[case]:.3g}s)",
                zorder=2,
            )
            ax.legend(loc="upper right", fontsize=8)

        ax.set_ylabel(ylab)
        ax.set_title(case, loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=18)

    fig.suptitle(
        rf"{case}: BCD $\theta$, sign4, full (all) — quality & runtime vs MILP",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{case}_theta_sign4_full_runtime_milp_compare"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-eval-dir",
        type=Path,
        default=ROOT / "result" / "paper_eval" / "20260512_183843",
        help="Directory containing data/sample_metrics.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Figures subdirectory (default: paper-eval-dir/figures_three_way)")
    parser.add_argument(
        "--milp-summary-csv",
        type=Path,
        default=None,
        help="CSV with columns case,median_runtime_sec (optional mean_runtime_sec)",
    )
    parser.add_argument(
        "--milp-per-sample-csv",
        type=Path,
        default=None,
        help="CSV with columns case,sample_index,runtime_sec for MILP baseline boxplot",
    )
    args = parser.parse_args()

    eval_dir: Path = args.paper_eval_dir.resolve()
    csv_path = eval_dir / "data" / "sample_metrics.csv"
    rows = _read_csv(csv_path)

    methods_order = DEFAULT_METHODS
    by_case_struct = _collect_by_case_method(rows, methods=methods_order)

    milp_medians = _load_milp_summary(args.milp_summary_csv) if args.milp_summary_csv else {}
    milp_ps = _load_milp_per_sample(args.milp_per_sample_csv) if args.milp_per_sample_csv else None

    out_dir = (args.output_dir or (eval_dir / "figures_three_way")).resolve()

    cases_sorted = sorted(by_case_struct.keys())
    if not cases_sorted:
        raise SystemExit("No sample rows matching methods bcd_theta / surrogate_sign4 / full_joint.")

    for case in cases_sorted:
        _plot_case_panels(case, by_case_struct[case], methods_order, milp_medians, milp_ps, out_dir)

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
