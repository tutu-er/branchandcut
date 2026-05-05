#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot selected subproblem ablation metrics with linear axes.

For each case in an ablation directory, this script writes:

* total objective residual comparison
* three objective components in one figure
* mean mu comparison
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = (
    ROOT
    / "result"
    / "subproblem_strategy_ablation"
    / "ablation_20260504_204211"
)
VARIANT_ORDER = ("method", "no_strategy")
VARIANT_LABELS = {
    "method": "Method",
    "no_strategy": "No strategy",
}
VARIANT_COLORS = {
    "method": "#2166ac",
    "no_strategy": "#d6604d",
}
COMPONENTS = (
    ("obj_primal", "Primal obj"),
    ("obj_dual", "Dual obj"),
    ("obj_opt", "Opt obj"),
)
FINAL_BAR_METRICS = (
    ("obj_total", "Total obj"),
    ("obj_primal", "Primal obj"),
    ("obj_dual", "Dual obj"),
    ("obj_opt", "Opt obj"),
    ("mu_mean", "Mean mu"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_surrogate_metrics(data: dict[str, Any]) -> list[dict[str, float]]:
    """Average surrogate records over units for each BCD iteration."""
    surrogate = data.get("metrics", {}).get("surrogate", {})
    if not isinstance(surrogate, dict):
        return []

    by_iter: dict[int, dict[str, list[float]]] = {}
    for records in surrogate.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or "iter" not in record:
                continue
            iteration = int(record["iter"])
            bucket = by_iter.setdefault(iteration, {})
            for key, value in record.items():
                if key == "iter":
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    bucket.setdefault(key, []).append(float(value))

    rows: list[dict[str, float]] = []
    for iteration in sorted(by_iter):
        row: dict[str, float] = {"iter": float(iteration)}
        for key, values in by_iter[iteration].items():
            if values:
                row[key] = float(mean(values))
        if all(key in row for key, _ in COMPONENTS):
            row["obj_total"] = sum(row[key] for key, _ in COMPONENTS)
        rows.append(row)
    return rows


def _discover_series(input_dir: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    series: dict[tuple[str, str], list[dict[str, float]]] = {}
    for path in sorted(input_dir.glob("metrics_*_*.json")):
        stem = path.stem
        prefix = "metrics_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :]
        variant = None
        for candidate in sorted(VARIANT_ORDER, key=len, reverse=True):
            suffix = f"_{candidate}"
            if rest.endswith(suffix):
                variant = candidate
                case = rest[: -len(suffix)]
                break
        if variant is None:
            continue
        rows = _aggregate_surrogate_metrics(_load_json(path))
        if rows:
            series[(case, variant)] = rows
    return series


def _savefig(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_line_set(ax: plt.Axes, rows: list[dict[str, float]], metric: str, variant: str) -> bool:
    rows = [row for row in rows if metric in row]
    if not rows:
        return False
    x = [row["iter"] + 1 for row in rows]
    y = [row[metric] for row in rows]
    ax.plot(
        x,
        y,
        color=VARIANT_COLORS.get(variant, None),
        label=VARIANT_LABELS.get(variant, variant),
        linewidth=1.8,
    )
    return True


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("BCD iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_log_y_if_obj(ax: plt.Axes, metric: str | None = None) -> None:
    if metric is None or metric.startswith("obj_"):
        ax.set_yscale("log")


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_range(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _log_values(values: list[float]) -> list[float]:
    return [math.log10(max(float(v), 1e-15)) for v in values]


def _polyline_points(
    xs: list[float],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    points = []
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / max(x_max - x_min, 1e-12) * width
        py = top + height - (y - y_min) / max(y_max - y_min, 1e-12) * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _svg_single_plot(
    title: str,
    ylabel: str,
    series: list[tuple[str, str, list[dict[str, float]], str]],
    metric: str,
    output_path: Path,
    log_y: bool = False,
    width: int = 760,
    height: int = 470,
) -> Path:
    plot_left, plot_top, plot_width, plot_height = 78, 58, width - 125, height - 130
    x_values: list[float] = []
    y_values: list[float] = []
    prepared = []
    for variant, label, rows, color in series:
        rows = [row for row in rows if metric in row]
        if not rows:
            continue
        xs = [row["iter"] + 1 for row in rows]
        ys_raw = [row[metric] for row in rows]
        ys = _log_values(ys_raw) if log_y else ys_raw
        x_values.extend(xs)
        y_values.extend(ys)
        prepared.append((variant, label, xs, ys, color))
    x_min, x_max = _nice_range(x_values)
    y_min, y_max = _nice_range(y_values)
    x_min = min(1.0, x_min)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{_svg_escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]
    for i in range(6):
        frac = i / 5
        x = plot_left + frac * plot_width
        y = plot_top + plot_height - frac * plot_height
        xv = x_min + frac * (x_max - x_min)
        yv = y_min + frac * (y_max - y_min)
        lines.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_top + plot_height}" stroke="#ddd" stroke-width="0.7"/>')
        lines.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" stroke="#ddd" stroke-width="0.7"/>')
        lines.append(f'<text x="{x:.2f}" y="{plot_top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.0f}</text>')
        tick_text = f"1e{yv:.0f}" if log_y else f"{yv:.3g}"
        lines.append(f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{tick_text}</text>')
    lines.append(f'<text x="{plot_left + plot_width/2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="13">BCD iteration</text>')
    y_label = f"{ylabel} (log scale)" if log_y else ylabel
    lines.append(f'<text transform="translate(20,{plot_top + plot_height/2:.2f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(y_label)}</text>')

    for _variant, label, xs, ys, color in prepared:
        pts = _polyline_points(xs, ys, x_min, x_max, y_min, y_max, plot_left, plot_top, plot_width, plot_height)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    legend_x, legend_y = plot_left + plot_width - 140, plot_top + 12
    for idx, (_variant, label, _xs, _ys, color) in enumerate(prepared):
        y = legend_y + idx * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="12">{_svg_escape(label)}</text>')
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _svg_components_plot(
    case: str,
    series: dict[tuple[str, str], list[dict[str, float]]],
    output_path: Path,
) -> Path:
    width, height = 1260, 430
    panel_w, panel_h = 350, 260
    starts = [(70, 78), (465, 78), (860, 78)]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{_svg_escape(case)}: objective components</text>',
    ]
    for (metric, title), (left, top) in zip(COMPONENTS, starts):
        prepared = []
        x_values: list[float] = []
        y_values: list[float] = []
        for variant in VARIANT_ORDER:
            rows = [row for row in series.get((case, variant), []) if metric in row]
            if not rows:
                continue
            xs = [row["iter"] + 1 for row in rows]
            ys = _log_values([row[metric] for row in rows])
            x_values.extend(xs)
            y_values.extend(ys)
            prepared.append((VARIANT_LABELS.get(variant, variant), xs, ys, VARIANT_COLORS.get(variant, "#333")))
        x_min, x_max = _nice_range(x_values)
        y_min, y_max = _nice_range(y_values)
        x_min = min(1.0, x_min)
        lines.append(f'<text x="{left + panel_w/2:.1f}" y="{top - 20}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{_svg_escape(title)}</text>')
        lines.append(f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" stroke="#333"/>')
        lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#333"/>')
        for i in range(5):
            frac = i / 4
            x = left + frac * panel_w
            y = top + panel_h - frac * panel_h
            xv = x_min + frac * (x_max - x_min)
            yv = y_min + frac * (y_max - y_min)
            lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_h}" stroke="#ddd" stroke-width="0.7"/>')
            lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="0.7"/>')
            lines.append(f'<text x="{x:.2f}" y="{top + panel_h + 20}" text-anchor="middle" font-family="Arial" font-size="10">{xv:.0f}</text>')
            lines.append(f'<text x="{left - 7}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="10">1e{yv:.0f}</text>')
        for label, xs, ys, color in prepared:
            pts = _polyline_points(xs, ys, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for idx, (label, _xs, _ys, color) in enumerate(prepared):
            lx, ly = left + panel_w - 120, top + 15 + idx * 20
            lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{lx + 28}" y="{ly + 4}" font-family="Arial" font-size="11">{_svg_escape(label)}</text>')
        lines.append(f'<text x="{left + panel_w/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="12">BCD iteration</text>')
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _final_metric_values(case: str, series: dict[tuple[str, str], list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        rows = series.get((case, variant), [])
        if not rows:
            continue
        final = rows[-1]
        values[variant] = {
            metric: float(final[metric])
            for metric, _label in FINAL_BAR_METRICS
            if metric in final
        }
    return values


def _svg_final_bar_plot(
    case: str,
    series: dict[tuple[str, str], list[dict[str, float]]],
    output_path: Path,
) -> Path:
    values = _final_metric_values(case, series)
    width, height = 900, 470
    left, top, plot_w, plot_h = 82, 58, 720, 310
    all_vals = [
        values.get(variant, {}).get(metric)
        for metric, _label in FINAL_BAR_METRICS
        for variant in VARIANT_ORDER
        if metric in values.get(variant, {})
    ]
    y_min = 0.0
    log_vals = _log_values([float(v) for v in all_vals if v is not None])
    y_min, y_max = _nice_range(log_vals)
    group_w = plot_w / len(FINAL_BAR_METRICS)
    bar_w = min(34.0, group_w / 3.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{_svg_escape(case)}: final metrics</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        yv = y_min + frac * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="0.7"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">1e{yv:.0f}</text>')
    for idx, (metric, label) in enumerate(FINAL_BAR_METRICS):
        center = left + group_w * (idx + 0.5)
        offsets = [-bar_w * 0.62, bar_w * 0.62]
        for offset, variant in zip(offsets, VARIANT_ORDER):
            if metric not in values.get(variant, {}):
                continue
            val = values[variant][metric]
            log_val = math.log10(max(val, 1e-15))
            h = (log_val - y_min) / max(y_max - y_min, 1e-12) * plot_h
            x = center + offset - bar_w / 2.0
            y = top + plot_h - h
            color = VARIANT_COLORS.get(variant, "#333")
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" opacity="0.88"/>')
            lines.append(f'<text x="{x + bar_w/2:.2f}" y="{max(y - 5, top + 10):.2f}" text-anchor="middle" font-family="Arial" font-size="9">{val:.2g}</text>')
        lines.append(f'<text x="{center:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="11">{_svg_escape(label)}</text>')
    legend_x, legend_y = left + plot_w + 22, top + 18
    for idx, variant in enumerate(VARIANT_ORDER):
        y = legend_y + idx * 24
        color = VARIANT_COLORS.get(variant, "#333")
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="16" height="16" fill="{color}" opacity="0.88"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{y + 3}" font-family="Arial" font-size="12">{_svg_escape(VARIANT_LABELS.get(variant, variant))}</text>')
    lines.append(f'<text transform="translate(20,{top + plot_h/2:.2f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">final value (log scale)</text>')
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_case(case: str, series: dict[tuple[str, str], list[dict[str, float]]], output_dir: Path) -> list[Path]:
    written: list[Path] = []

    if not MATPLOTLIB_AVAILABLE:
        variant_series = [
            (
                variant,
                VARIANT_LABELS.get(variant, variant),
                series.get((case, variant), []),
                VARIANT_COLORS.get(variant, "#333333"),
            )
            for variant in VARIANT_ORDER
        ]
        out = output_dir / f"{case}_total_obj.svg"
        written.append(_svg_single_plot(f"{case}: total objective residual", "total obj", variant_series, "obj_total", out, log_y=True))
        out = output_dir / f"{case}_obj_components.svg"
        written.append(_svg_components_plot(case, series, out))
        out = output_dir / f"{case}_mean_mu.svg"
        written.append(_svg_single_plot(f"{case}: mean mu", "mean mu", variant_series, "mu_mean", out))
        out = output_dir / f"{case}_final_bar.svg"
        written.append(_svg_final_bar_plot(case, series, out))
        return written

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    plotted = False
    for variant in VARIANT_ORDER:
        plotted = _plot_line_set(ax, series.get((case, variant), []), "obj_total", variant) or plotted
    ax.set_title(f"{case}: total objective residual")
    _style_axis(ax, "total obj")
    ax.set_yscale("log")
    if plotted:
        ax.legend(frameon=False)
    out = output_dir / f"{case}_total_obj"
    _savefig(fig, out)
    written.extend([out.with_suffix(".png"), out.with_suffix(".pdf")])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharex=True)
    for ax, (metric, title) in zip(axes, COMPONENTS):
        plotted = False
        for variant in VARIANT_ORDER:
            plotted = _plot_line_set(ax, series.get((case, variant), []), metric, variant) or plotted
        ax.set_title(title)
        _style_axis(ax, metric)
        ax.set_yscale("log")
        if plotted:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case}: objective components", y=1.03)
    fig.tight_layout()
    out = output_dir / f"{case}_obj_components"
    _savefig(fig, out)
    written.extend([out.with_suffix(".png"), out.with_suffix(".pdf")])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    plotted = False
    for variant in VARIANT_ORDER:
        plotted = _plot_line_set(ax, series.get((case, variant), []), "mu_mean", variant) or plotted
    ax.set_title(f"{case}: mean mu")
    _style_axis(ax, "mean mu")
    if plotted:
        ax.legend(frameon=False)
    out = output_dir / f"{case}_mean_mu"
    _savefig(fig, out)
    written.extend([out.with_suffix(".png"), out.with_suffix(".pdf")])

    values = _final_metric_values(case, series)
    labels = [label for _metric, label in FINAL_BAR_METRICS]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.34
    for idx, variant in enumerate(VARIANT_ORDER):
        vals = [
            values.get(variant, {}).get(metric, float("nan"))
            for metric, _label in FINAL_BAR_METRICS
        ]
        x_pos = [pos + (idx - 0.5) * width for pos in x]
        ax.bar(
            x_pos,
            vals,
            width=width,
            color=VARIANT_COLORS.get(variant, None),
            label=VARIANT_LABELS.get(variant, variant),
            alpha=0.88,
        )
    ax.set_title(f"{case}: final metrics")
    ax.set_ylabel("final value")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = output_dir / f"{case}_final_bar"
    _savefig(fig, out)
    written.extend([out.with_suffix(".png"), out.with_suffix(".pdf")])

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Ablation directory containing metrics_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Default: <input-dir>/requested_plots.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else (ROOT / args.input_dir)
    output_dir = args.output_dir or (input_dir / "requested_plots")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    series = _discover_series(input_dir)
    cases = sorted({case for case, _variant in series})
    if not cases:
        raise RuntimeError(f"no non-empty metrics_*.json series found in {input_dir}")

    written: list[Path] = []
    for case in cases:
        written.extend(plot_case(case, series, output_dir))

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    for path in written:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
