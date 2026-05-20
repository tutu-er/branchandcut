# -*- coding: utf-8 -*-
"""绘制 bench_fp_4way 的 FP 结果统计图（成功率 / Hamming / 迭代 / 耗时）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PREFERRED_COMPARE_LABELS = ("theta_flip", "vanilla")
DISPLAY_LABELS = {
    "theta_flip": "改进可行性泵",
    "theta_flip_case3lite": "改进可行性泵(case3)",
    "vanilla": "传统可行性泵",
}
CASE_DISPLAY = {
    "case3lite": "case3",
}
COLORS = {
    "theta_flip": "#2a9d8f",
    "vanilla": "#e76f51",
}


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


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


def _normalize_plot_label(label: str) -> str:
    if label in {"vanilla", "theta_flip"}:
        return label
    if label.startswith("theta_flip"):
        return "theta_flip"
    return label


def _resolve_bench_compare_labels(result_labels: Sequence[str]) -> List[str]:
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
    return picked or list(labels)


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, DISPLAY_LABELS.get(_normalize_plot_label(label), label))


def _display_case(case: str) -> str:
    return CASE_DISPLAY.get(str(case), str(case))


def _finite(values: Sequence[Any]) -> List[float]:
    arr: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            arr.append(fv)
    return arr


def _per_sample_metric(summary: dict, label: str, key: str) -> List[float]:
    return _finite(
        ((row.get("results") or {}).get(label) or {}).get(key)
        for row in summary.get("per_sample") or []
    )


def _per_sample_success(summary: dict, label: str) -> List[float]:
    vals: List[float] = []
    for row in summary.get("per_sample") or []:
        ok = ((row.get("results") or {}).get(label) or {}).get("fp_success")
        if ok is None:
            continue
        vals.append(1.0 if bool(ok) else 0.0)
    return vals


def _pick_compare_labels(summary: dict) -> List[str]:
    if summary.get("per_sample"):
        first = summary["per_sample"][0]
        if isinstance(first.get("results"), dict):
            return _resolve_bench_compare_labels(list(first["results"].keys()))
    return _resolve_bench_compare_labels(list((summary.get("aggregates") or {}).keys()))


def _bar_panel(
    ax,
    labels: Sequence[str],
    values: Sequence[Optional[float]],
    *,
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
    fmt: str = ".2f",
    as_percent: bool = False,
) -> None:
    x = np.arange(len(labels))
    colors = [COLORS.get(_normalize_plot_label(lb), "#888888") for lb in labels]
    plot_vals = [0.0 if v is None else float(v) for v in values]
    bars = ax.bar(x, plot_vals, width=0.58, color=colors, edgecolor="#444444", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_label(lb) for lb in labels], rotation=12, ha="right")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ymax = ax.get_ylim()[1]
    for bar, val in zip(bars, plot_vals):
        text = f"{val * 100:.0f}%" if as_percent else format(val, fmt)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03 * max(ymax, 1e-9),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _boxplot_panel(
    ax,
    data_by_label: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    colors = [COLORS.get(_normalize_plot_label(lb), "#888888") for lb in labels]
    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot(
        [list(vals) for vals in data_by_label],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": ".",
            "markerfacecolor": "#333333",
            "markeredgecolor": "#333333",
            "markersize": 5,
        },
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#777777",
            "markeredgecolor": "none",
            "alpha": 0.35,
            "markersize": 3,
        },
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels([_display_label(lb) for lb in labels], rotation=12, ha="right")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def _plot_stats_bars(summary: dict, labels: List[str], out_path: Path, *, title: Optional[str]) -> None:
    aggregates = summary.get("aggregates") or {}
    n_samples = summary.get("metadata", {}).get("n_samples", len(summary.get("per_sample") or []))
    case = _display_case(summary.get("metadata", {}).get("case", "?"))

    success = [aggregates.get(lb, {}).get("success_rate") for lb in labels]
    hamming = [aggregates.get(lb, {}).get("mean_hamming") for lb in labels]
    iters = [aggregates.get(lb, {}).get("mean_iterations") for lb in labels]
    times = [aggregates.get(lb, {}).get("mean_time_sec") for lb in labels]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    _bar_panel(
        axes[0, 0], labels, success,
        title="A. 成功率", ylabel="可行样本比例", ylim=(0.0, 1.05), as_percent=True,
    )
    _bar_panel(
        axes[0, 1], labels, hamming,
        title="B. 平均 Hamming 距离", ylabel="与真实解平均差异位数",
    )
    _bar_panel(
        axes[1, 0], labels, iters,
        title="C. 平均 FP 迭代次数", ylabel="迭代次数",
    )
    _bar_panel(
        axes[1, 1], labels, times,
        title="D. 平均运行时间", ylabel="秒",
    )

    fig_title = title or f"FP 结果统计（{case}, n={n_samples}）"
    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stats_boxes(summary: dict, labels: List[str], out_path: Path, *, title: Optional[str]) -> None:
    n_samples = summary.get("metadata", {}).get("n_samples", len(summary.get("per_sample") or []))
    case = _display_case(summary.get("metadata", {}).get("case", "?"))

    ham_data = [_per_sample_metric(summary, lb, "hamming_to_true") for lb in labels]
    iter_data = [_per_sample_metric(summary, lb, "total_iterations") for lb in labels]
    time_data = [_per_sample_metric(summary, lb, "wallclock_sec") for lb in labels]
    succ_data = [_per_sample_success(summary, lb) for lb in labels]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    _boxplot_panel(axes[0, 0], succ_data, labels, title="A. 逐样本可行性", ylabel="1=可行, 0=不可行")
    _boxplot_panel(axes[0, 1], ham_data, labels, title="B. Hamming 距离分布", ylabel="与真实解差异位数")
    _boxplot_panel(axes[1, 0], iter_data, labels, title="C. FP 迭代次数分布", ylabel="迭代次数")
    _boxplot_panel(axes[1, 1], time_data, labels, title="D. 运行时间分布", ylabel="秒")

    fig_title = title or f"FP 结果分布（{case}, n={n_samples}）"
    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="bench_fp_4way JSON 路径")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    parser.add_argument(
        "--style",
        choices=("bars", "box", "both"),
        default="bars",
        help="bars=汇总柱状图, box=逐样本箱线图, both=两种都输出",
    )
    parser.add_argument("--title", default=None, help="自定义总标题")
    args = parser.parse_args()
    _setup_chinese_font()

    in_path = _resolve_path(args.input)
    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(in_path.read_text(encoding="utf-8"))
    labels = _pick_compare_labels(summary)
    if not labels:
        raise ValueError("No comparable strategies found in input")

    stem = out_path.stem
    suffix = out_path.suffix or ".png"
    parent = out_path.parent

    if args.style in ("bars", "both"):
        bars_path = out_path if args.style == "bars" else parent / f"{stem}_bars{suffix}"
        _plot_stats_bars(summary, labels, bars_path, title=args.title)
        print(f"[done] wrote {bars_path}")
    if args.style in ("box", "both"):
        box_path = out_path if args.style == "box" else parent / f"{stem}_box{suffix}"
        _plot_stats_boxes(summary, labels, box_path, title=args.title)
        print(f"[done] wrote {box_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
