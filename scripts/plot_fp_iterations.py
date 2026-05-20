#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Feasibility Pump iteration metrics (δ_k / δ̂_k / perturbations).

The script accepts either:

  * a ``fp_compare_*.json`` produced by ``scripts/compare_tailored_vs_vanilla_fp.py``
    (recommended -- it carries both ``vanilla`` and ``tailored`` iteration traces),
  * or a ``fp_forward_diagnostics_*.json`` produced by
    ``scripts/collect_model_fp_diagnostics.py`` (single-strategy traces, plotted
    on a single curve per sample).

It writes one figure per sample showing the iteration evolution of:

  * δ_k  -- the L1 distance between the LP projection y* and the rounded y
            (i.e. the FP potential function, eq 4-18/4-30 in the paper),
  * δ̂_k -- the cost-augmented potential δ_k + ω·c_y^T y,
  * perturbation events (cycle hit / equipotential cycle / χ-random /
    θ-resample / pool restart / bit flip).

A summary figure ``summary_delta_decay.png`` overlays the per-sample δ_k
trajectories for vanilla vs tailored, and reports the mean curves.

Example::

    python scripts/plot_fp_iterations.py \
        --input result/fp_diagnostics/fp_compare_case14_20260519_200357.json \
        --output-dir result/fp_diagnostics/plots_case14 \
        --max-samples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_COMPARE_LABELS = ("theta_flip", "vanilla")
DISPLAY_LABELS = {
    "theta_flip": "改进可行性泵",
    "theta_flip_case3lite": "改进可行性泵(case3)",
    "vanilla": "传统可行性泵",
    "tailored": "改进可行性泵",
}


def _normalize_plot_label(label: str) -> str:
    """Map bench strategy labels onto shared styling keys."""
    if label in {"vanilla", "tailored", "theta_flip"}:
        return label
    if label.startswith("theta_flip"):
        return "theta_flip"
    return label


def _resolve_bench_compare_labels(result_labels: List[str]) -> List[str]:
    """Pick the tailored + vanilla pair from a bench_fp_4way result block."""
    labels = list(result_labels)
    picked: List[str] = []
    for pref in PREFERRED_COMPARE_LABELS:
        if pref in labels:
            picked.append(pref)
            continue
        if pref == "theta_flip":
            variants = sorted(lb for lb in labels if lb.startswith("theta_flip"))
            if variants:
                picked.append(variants[0])
    if "vanilla" in labels and "vanilla" not in picked:
        picked.append("vanilla")
    return picked or labels


def _setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(fv):
        return None
    return fv


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, DISPLAY_LABELS.get(_normalize_plot_label(label), label))


def _trace_from_compare_row(label: str, sample_row: Dict[str, Any]) -> Dict[str, Any]:
    info = sample_row.get(label) or {}
    trace = info.get("iteration_trace") or []
    return {
        "label": label,
        "sample_id": sample_row.get("sample_id"),
        "fp_success": bool(info.get("fp_success")),
        "hamming_to_true": info.get("hamming_to_true"),
        "wallclock_sec": info.get("wallclock_sec"),
        "details_summary": info.get("details_summary") or {},
        "trace": trace,
    }


def _trace_from_bench_result(label: str, sample_row: Dict[str, Any]) -> Dict[str, Any]:
    info = (sample_row.get("results") or {}).get(label) or {}
    return {
        "label": label,
        "sample_id": sample_row.get("sample_id"),
        "fp_success": bool(info.get("fp_success")),
        "hamming_to_true": info.get("hamming_to_true"),
        "wallclock_sec": info.get("wallclock_sec"),
        "details_summary": {},
        "trace": info.get("iteration_trace") or [],
    }


def _traces_from_diagnostics(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Return one list-of-rows per sample from the legacy diagnostics file."""
    samples: List[List[Dict[str, Any]]] = []
    records = data.get("records") or []
    for rec in records:
        rows = rec.get("fp_iteration_plot_rows") or []
        if rows:
            samples.append(rows)
    return samples


def _filter_iter_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("event") in (None, "fp_iteration")
            and (r.get("iteration") is not None)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _split_by_hot_start(trace: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for row in trace:
        if row.get("iteration") is None:
            continue
        hs = row.get("hot_start_name") or f"hs_{row.get('hot_start_index')}"
        if hs not in groups:
            groups[hs] = []
            order.append(hs)
        groups[hs].append(row)
    return [(hs, groups[hs]) for hs in order]


def _pick_best_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick the hot-start whose trace either reached feasibility or got
    closest (lowest final δ_k). Returns rows sorted by iteration."""
    groups = _split_by_hot_start(trace)
    if not groups:
        return []

    def _score(rows: List[Dict[str, Any]]) -> Tuple[int, float, int]:
        if not rows:
            return (1, float("inf"), 0)
        feasible = any(r.get("post_feasible") for r in rows)
        last_delta = _coerce_float(rows[-1].get("delta_k"))
        if last_delta is None:
            last_delta = float("inf")
        return (0 if feasible else 1, last_delta, -len(rows))

    best_name, best_rows = min(groups, key=lambda kv: _score(kv[1]))
    return sorted(best_rows, key=lambda r: (r.get("iteration") or 0))


def _annotate_perturbations(ax: plt.Axes, rows: List[Dict[str, Any]]) -> None:
    handles: Dict[str, plt.Line2D] = {}
    for row in rows:
        x = row.get("iteration")
        if x is None:
            continue
        if row.get("equipotential_cycle"):
            line = ax.axvline(x, color="tab:purple", alpha=0.25, linestyle=":", linewidth=1)
            handles.setdefault("equipotential_cycle", line)
        elif row.get("cycle_hit"):
            line = ax.axvline(x, color="tab:red", alpha=0.20, linestyle=":", linewidth=1)
            handles.setdefault("cycle_hit", line)
        if row.get("chi_random_used_for_stall"):
            line = ax.axvline(x, color="tab:orange", alpha=0.55, linestyle="--", linewidth=1)
            handles.setdefault("chi_random", line)
        if (row.get("theta_resample_added") or 0) > 0:
            line = ax.axvline(x, color="tab:green", alpha=0.55, linestyle="-.", linewidth=1.2)
            handles.setdefault("theta_resample", line)
        mode = row.get("perturbation_mode")
        if mode == "pool_restart":
            line = ax.axvline(x, color="tab:cyan", alpha=0.55, linestyle="-", linewidth=1)
            handles.setdefault("pool_restart", line)
        elif mode == "flip" and row.get("perturbation_applied"):
            line = ax.axvline(x, color="tab:brown", alpha=0.4, linestyle="-", linewidth=0.8)
            handles.setdefault("bit_flip", line)
    if handles:
        ax.legend(
            list(handles.values()),
            list(handles.keys()),
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )


def _event_label_for_row(row: Dict[str, Any]) -> Optional[str]:
    if row.get("post_feasible"):
        return "可行"
    if row.get("perturbation_applied"):
        return str(row.get("perturbation_mode") or "扰动").replace("_", "+")
    if row.get("noh_milp_refresh_added"):
        return "pool注入"
    if row.get("cycle_hit") and row.get("equipotential_cycle"):
        return "等势循环"
    if row.get("cycle_hit"):
        return "循环"
    return None


def _annotate_key_events(ax: plt.Axes, rows: List[Dict[str, Any]], *, color: str) -> None:
    for row in rows:
        k = row.get("iteration")
        if k is None:
            continue
        label = _event_label_for_row(row)
        if not label:
            continue
        y = _coerce_float(row.get("delta_k"))
        if y is None:
            continue
        ax.annotate(
            label,
            xy=(float(k), y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
            alpha=0.85,
        )


def _plot_delta_overlay(
    sample_id: Any,
    traces: List[Dict[str, Any]],
    out_path: Path,
    *,
    show_omega: bool = False,
    show_objective: bool = True,
    title: Optional[str] = None,
    include_title_stats: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax_obj = ax.twinx() if show_objective else None

    colors = {
        "vanilla": "tab:blue",
        "tailored": "tab:red",
        "theta_flip": "tab:green",
    }
    delta_markers = {
        "vanilla": "s",
        "tailored": "o",
        "theta_flip": "o",
    }
    obj_markers = {
        "vanilla": "D",
        "tailored": "v",
        "theta_flip": "^",
    }
    offsets = {
        "vanilla": -0.04,
        "tailored": 0.0,
        "theta_flip": 0.04,
    }

    title_bits: List[str] = [] if title else [f"sample {sample_id}"]
    delta_handles: List[Any] = []
    delta_labels: List[str] = []
    obj_handles: List[Any] = []
    obj_labels: List[str] = []

    for trace_info in traces:
        rows = _pick_best_trace(trace_info["trace"])
        if not rows:
            continue
        label = trace_info["label"]
        style_key = _normalize_plot_label(label)
        x_offset = offsets.get(style_key, 0.0)
        iterations = [float(r.get("iteration")) + x_offset for r in rows]
        delta_k = [_coerce_float(r.get("delta_k")) for r in rows]
        lp_objective = [_coerce_float(r.get("primal_objective")) for r in rows]
        color = colors.get(style_key, "gray")
        short = _display_label(label)
        if include_title_stats and title is None:
            title_bits.append(
                f"{short}: ok={trace_info['fp_success']}, "
                f"h={trace_info.get('hamming_to_true')}, iters={len(rows)}"
            )
        line_delta, = ax.plot(
            iterations,
            delta_k,
            color=color,
            marker=delta_markers.get(style_key, "o"),
            markersize=6,
            linewidth=2.4,
            linestyle="-",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        delta_handles.append(line_delta)
        delta_labels.append(f"{short} $\\delta_k$ (实线)")
        _annotate_key_events(ax, rows, color=color)

        if show_objective and ax_obj is not None and any(v is not None for v in lp_objective):
            line_obj, = ax_obj.plot(
                iterations,
                lp_objective,
                color=color,
                marker=obj_markers.get(style_key, "^"),
                markersize=5,
                linewidth=2.0,
                linestyle="--",
                dashes=(6, 3),
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                alpha=0.95,
                zorder=2,
            )
            obj_handles.append(line_obj)
            obj_labels.append(f"{short} LP目标 (虚线)")

        if show_omega and style_key == "theta_flip":
            omega_delta = [_coerce_float(r.get("omega_round_delta_k")) for r in rows]
            if any(v is not None for v in omega_delta):
                line_om, = ax.plot(
                    iterations,
                    omega_delta,
                    color=color,
                    marker="x",
                    markersize=5,
                    linewidth=1.4,
                    linestyle="-.",
                    dashes=(4, 2, 1, 2),
                    alpha=0.7,
                    zorder=1,
                )
                delta_handles.append(line_om)
                delta_labels.append(f"{short} |ω-round| (点划线)")

    ax.set_ylabel(r"$\delta_k$")
    if ax_obj is not None:
        ax_obj.set_ylabel("LP投影目标值")
    ax.set_xlabel("FP 迭代 $k$")
    plot_title = title if title is not None else " / ".join(title_bits)
    ax.set_title(plot_title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(
        delta_handles + obj_handles,
        delta_labels + obj_labels,
        loc="best",
        fontsize=9,
        framealpha=0.85,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_single_trace(
    sample_id: Any,
    label: str,
    rows: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    rows = _pick_best_trace(rows)
    if not rows:
        return
    iterations = [r.get("iteration") for r in rows]
    delta_k = [_coerce_float(r.get("delta_k")) for r in rows]
    delta_hat = [_coerce_float(r.get("delta_hat_k")) for r in rows]
    soft_pen = [_coerce_float(r.get("soft_penalty")) for r in rows]
    changed = [_coerce_float(r.get("changed_bits")) for r in rows]
    omega_delta = [_coerce_float(r.get("omega_round_delta_k")) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    ax.plot(iterations, delta_k, color="tab:blue", marker="o", markersize=4)
    ax.set_title(r"$\delta_k$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\delta_k$")
    ax.grid(True, alpha=0.3)
    _annotate_key_events(ax, rows, color="tab:blue")

    ax = axes[1]
    if any(v is not None for v in omega_delta):
        ax.plot(iterations, omega_delta, color="tab:orange", marker="x", markersize=4)
        ax.set_title(r"$|\omega-\mathrm{round}(\omega)|_1$")
        ax.set_ylabel("omega_round_delta")
    elif any(v is not None for v in changed):
        ax.plot(iterations, changed, color="tab:purple", marker="d", markersize=4)
        ax.set_title("changed_bits")
        ax.set_ylabel("changed_bits")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"FP trace — sample {sample_id} [{_display_label(label)}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _aggregate_curves(
    per_sample_traces: List[Tuple[Any, Dict[str, List[Dict[str, Any]]]]],
    max_len: int,
    field: str = "delta_k",
    fallback_field: Optional[str] = "delta_hat_k",
) -> Dict[str, np.ndarray]:
    """Compute mean curve per label, padding shorter traces with last value."""
    by_label: Dict[str, List[np.ndarray]] = {}
    for _, by_lbl in per_sample_traces:
        for label, rows in by_lbl.items():
            rows = _pick_best_trace(rows)
            if not rows:
                continue
            values = [_coerce_float(r.get(field)) for r in rows]
            if not any(v is not None for v in values) and fallback_field:
                values = [_coerce_float(r.get(fallback_field)) for r in rows]
            values = [v for v in values if v is not None]
            if not values:
                continue
            arr = np.full(max_len, values[-1], dtype=float)
            arr[: len(values)] = values[:max_len]
            by_label.setdefault(label, []).append(arr)
    return {label: np.mean(np.vstack(arrs), axis=0) for label, arrs in by_label.items()}


def _max_trace_len(
    per_sample_traces: List[Tuple[Any, Dict[str, List[Dict[str, Any]]]]],
) -> int:
    max_len = 1
    for _, by_lbl in per_sample_traces:
        for rows in by_lbl.values():
            rows = _pick_best_trace(rows)
            max_len = max(max_len, len(rows))
    return max_len


def _plot_summary_decay(
    per_sample_traces: List[Tuple[Any, Dict[str, List[Dict[str, Any]]]]],
    out_path: Path,
    *,
    title: Optional[str] = None,
    show_objective: bool = True,
) -> None:
    if not per_sample_traces:
        return
    max_len = _max_trace_len(per_sample_traces)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax_obj = ax.twinx() if show_objective else None
    colors = {
        "vanilla": "tab:blue",
        "tailored": "tab:red",
        "theta_flip": "tab:green",
    }
    delta_markers = {
        "vanilla": "s",
        "tailored": "o",
        "theta_flip": "o",
    }
    obj_markers = {
        "vanilla": "D",
        "tailored": "v",
        "theta_flip": "^",
    }
    offsets = {
        "vanilla": -0.04,
        "tailored": 0.0,
        "theta_flip": 0.04,
    }

    for _, by_lbl in per_sample_traces:
        for label, rows in by_lbl.items():
            rows = _pick_best_trace(rows)
            if not rows:
                continue
            style_key = _normalize_plot_label(label)
            x_offset = offsets.get(style_key, 0.0)
            iterations = [float(r.get("iteration")) + x_offset for r in rows]
            deltas = [_coerce_float(r.get("delta_k")) for r in rows]
            if not any(v is not None for v in deltas):
                deltas = [_coerce_float(r.get("delta_hat_k")) for r in rows]
            ax.plot(
                iterations,
                deltas,
                color=colors.get(style_key, "gray"),
                alpha=0.16,
                linewidth=1.0,
                linestyle="-",
            )
            if show_objective and ax_obj is not None:
                lp_objective = [_coerce_float(r.get("primal_objective")) for r in rows]
                if any(v is not None for v in lp_objective):
                    ax_obj.plot(
                        iterations,
                        lp_objective,
                        color=colors.get(style_key, "gray"),
                        alpha=0.10,
                        linewidth=0.9,
                        linestyle="--",
                    )

    delta_handles: List[Any] = []
    delta_labels: List[str] = []
    obj_handles: List[Any] = []
    obj_labels: List[str] = []
    x_axis = np.arange(0, max_len, dtype=float)

    mean_delta = _aggregate_curves(per_sample_traces, max_len, field="delta_k")
    mean_objective = (
        _aggregate_curves(per_sample_traces, max_len, field="primal_objective", fallback_field=None)
        if show_objective
        else {}
    )

    for label, curve in mean_delta.items():
        style_key = _normalize_plot_label(label)
        x_offset = offsets.get(style_key, 0.0)
        short = _display_label(label)
        color = colors.get(style_key, "black")
        xs = x_axis[: len(curve)] + x_offset
        line_delta, = ax.plot(
            xs,
            curve[: len(curve)],
            color=color,
            marker=delta_markers.get(style_key, "o"),
            markersize=6,
            linewidth=2.4,
            linestyle="-",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        delta_handles.append(line_delta)
        delta_labels.append(f"{short} mean $\\delta_k$ (实线)")

        if show_objective and ax_obj is not None and label in mean_objective:
            obj_curve = mean_objective[label]
            line_obj, = ax_obj.plot(
                xs,
                obj_curve[: len(obj_curve)],
                color=color,
                marker=obj_markers.get(style_key, "^"),
                markersize=5,
                linewidth=2.0,
                linestyle="--",
                dashes=(6, 3),
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                alpha=0.95,
                zorder=2,
            )
            obj_handles.append(line_obj)
            obj_labels.append(f"{short} mean LP目标 (虚线)")

    n_samples = len(per_sample_traces)
    plot_title = title if title is not None else f"跨样本 mean $\\delta_k$ (n={n_samples})"
    ax.set_xlabel("FP 迭代 $k$")
    ax.set_ylabel(r"mean $\delta_k$")
    if ax_obj is not None:
        ax_obj.set_ylabel("mean LP投影目标值")
    ax.set_title(plot_title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(
        delta_handles + obj_handles,
        delta_labels + obj_labels,
        loc="best",
        fontsize=9,
        framealpha=0.85,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = ROOT / pp
    return pp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Path to fp_compare_*.json or fp_forward_diagnostics_*.json")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write PNG plots")
    parser.add_argument("--max-samples", type=int, default=8,
                        help="Cap the number of per-sample plots (default: 8). "
                             "Set <= 0 to disable.")
    parser.add_argument("--only-succeeded", action="store_true",
                        help="Only plot samples where at least one strategy succeeded")
    parser.add_argument("--show-omega", action="store_true",
                        help="Overlay |omega-round| on tailored/theta_flip overlay plots.")
    parser.add_argument("--only-sample", type=int, default=None,
                        help="Only plot the given sample_id (overlay / single traces).")
    parser.add_argument("--overlay-title", type=str, default=None,
                        help="Custom title for delta overlay plots (suppresses default title).")
    parser.add_argument("--no-title-stats", action="store_true",
                        help="Omit ok/h/iters from overlay titles (default title only).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only write the cross-sample summary figure.")
    parser.add_argument("--summary-title", type=str, default=None,
                        help="Custom title for the cross-sample summary plot.")
    args = parser.parse_args()
    _setup_chinese_font()

    in_path = _resolve_path(args.input)
    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(in_path)

    per_sample_traces: List[Tuple[Any, Dict[str, List[Dict[str, Any]]]]] = []

    if "per_sample" in data:
        per_sample = data.get("per_sample") or []
        for row in per_sample:
            sample_id = row.get("sample_id")
            label_traces: Dict[str, List[Dict[str, Any]]] = {}
            if isinstance(row.get("results"), dict):
                # bench_fp_4way.py schema: per_sample[*].results[label].iteration_trace
                result_labels = list((row.get("results") or {}).keys())
                compare_labels = _resolve_bench_compare_labels(result_labels)
                for label in compare_labels:
                    info = _trace_from_bench_result(label, row)
                    if info["trace"]:
                        label_traces[label] = info["trace"]
                succeeded = any(
                    bool((row.get("results") or {}).get(label, {}).get("fp_success"))
                    for label in label_traces
                )
            else:
                # compare_tailored_vs_vanilla_fp.py schema
                v_info = _trace_from_compare_row("vanilla", row)
                t_info = _trace_from_compare_row("tailored", row)
                if v_info["trace"]:
                    label_traces["vanilla"] = v_info["trace"]
                if t_info["trace"]:
                    label_traces["tailored"] = t_info["trace"]
                succeeded = bool(v_info["fp_success"] or t_info["fp_success"])
            if not label_traces:
                continue
            if args.only_succeeded and not succeeded:
                continue
            per_sample_traces.append((sample_id, label_traces))
    elif "records" in data:
        for rec in data.get("records") or []:
            rows = rec.get("fp_iteration_plot_rows") or []
            rows = _filter_iter_rows(rows)
            if not rows:
                continue
            sample_id = rec.get("sample_id")
            per_sample_traces.append((sample_id, {"diagnostics": rows}))
    else:
        print(f"[error] unrecognised input schema in {in_path}", file=sys.stderr)
        return 2

    if args.only_sample is not None:
        per_sample_traces = [
            (sid, by_lbl)
            for sid, by_lbl in per_sample_traces
            if sid == args.only_sample
        ]
        if not per_sample_traces:
            print(
                f"[error] sample_id={args.only_sample} not found in input",
                file=sys.stderr,
            )
            return 2
        summary_traces = list(per_sample_traces)
        plot_traces = list(per_sample_traces)
    else:
        summary_traces = list(per_sample_traces)
        plot_traces = (
            per_sample_traces
            if args.max_samples <= 0
            else per_sample_traces[: args.max_samples]
        )

    if not summary_traces:
        print("[warn] no iteration traces found in input", file=sys.stderr)
        return 1

    if not args.summary_only:
        for sample_id, by_lbl in plot_traces:
            if len(by_lbl) >= 2:
                overlay_path = out_dir / f"sample_{sample_id}_delta_overlay.png"
                traces_list = []
                for row in (data.get("per_sample") or []):
                    if row.get("sample_id") == sample_id:
                        if isinstance(row.get("results"), dict):
                            result_labels = list((row.get("results") or {}).keys())
                            for label in _resolve_bench_compare_labels(result_labels):
                                if label not in by_lbl:
                                    continue
                                traces_list.append(_trace_from_bench_result(label, row))
                        else:
                            traces_list.append(_trace_from_compare_row("vanilla", row))
                            traces_list.append(_trace_from_compare_row("tailored", row))
                        break
                _plot_delta_overlay(
                    sample_id,
                    traces_list,
                    overlay_path,
                    show_omega=bool(args.show_omega),
                    title=args.overlay_title,
                    include_title_stats=not bool(args.no_title_stats),
                )
                print(f"[done] wrote {overlay_path}", flush=True)
            else:
                for label, rows in by_lbl.items():
                    single_path = out_dir / f"sample_{sample_id}_{label}_trace.png"
                    _plot_single_trace(sample_id, label, rows, single_path)
                    print(f"[done] wrote {single_path}", flush=True)

    if args.only_sample is None:
        summary_path = out_dir / "summary_30samples_iteration.png"
        _plot_summary_decay(
            summary_traces,
            summary_path,
            title=args.summary_title,
        )
        print(f"[done] wrote {summary_path}", flush=True)
        legacy_summary_path = out_dir / "summary_delta_decay.png"
        if legacy_summary_path != summary_path:
            _plot_summary_decay(
                summary_traces,
                legacy_summary_path,
                title=args.summary_title,
            )
            print(f"[done] wrote {legacy_summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
