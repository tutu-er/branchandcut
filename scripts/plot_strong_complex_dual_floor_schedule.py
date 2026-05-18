#!/usr/bin/env python3
"""Plot the time-dependent heuristics used by strong-complex-dual-floor training.

The script intentionally keeps only strategies that are naturally expressed as
training schedules: warm starts, delayed activation, curriculum ramps, floor
switches, caps, and direct fitting phases. Captions under the figure are empty by default (use ``bcd``/``surrogate``
``caption`` in ``TEXT`` or ``--text-json`` if needed).

Default styling is publication-oriented: concise in-bar tags (short math/text),
no color legend (meanings follow row/bar labels), no extra percent labels under
vertical guides unless enabled, and (a)/(b) panel titles.
Toggle verbosity via ``show_bar_text``, ``bar_label_min_width``, ``show_guide_line_labels``,
``show_figure_subtitle``, or ``--text-json`` overrides.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("result/paper_figures/strong_complex_dual_floor_schedule")


COLORS = {
    # 两图同一语义同色：constraint=代理约束主段（渐进加入等）；constraint_light=代理约束末段（全开）；
    # dual/dual_light/flip=对偶；direct=NN-direct；cost+inactive=h(θ) 前段/后段；pretrain/warm_light=机组。
    "pretrain": "#5B677A",
    "warm": "#4C78A8",
    "warm_light": "#A9C6E8",
    "constraint": "#2F6FBB",
    "constraint_light": "#7AC0D8",
    "dual": "#D97924",
    "dual_light": "#F0A54A",
    # 与 dual 同系、略偏深的琥珀色，用于符号翻转条，避免与粉紫混杂。
    "flip": "#B86820",
    "direct": "#2E8B57",
    "direct_light": "#82BC8A",
    "cost": "#697386",
    "inactive": "#CBD2D9",
}


TEXT: dict[str, Any] = {
    # 中文图注需在系统/环境中存在相应字体；亦可改用 --text-json 覆写 font_family。
    "font_family": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans", "sans-serif"],
    "show_bar_text": True,
    "bar_label_min_width": 4.0,
    "show_guide_line_labels": False,
    "show_figure_subtitle": False,
    "title_fontsize": 11.0,
    "title_fontweight": "normal",
    "title_loc": "center",
    "bar_fontsize": 9.5,
    "pre_bar_fontsize": 9.0,
    "ytick_fontsize": 9.5,
    "xtick_fontsize": 10.0,
    "xlabel_fontsize": 11.0,
    "x_label": "训练进度（归一化）",
    "pre_loop_label": "",
    "axis_ticks": ["0", "0.25", "0.5", "0.75", "1"],
    # Kept for --text-json overrides; not drawn when show_guide_line_labels is False.
    "guide_labels": {
        5: "5%",
        10: "10%",
        25: "25%",
        40: "40%",
        45: "45%",
        50: "50%",
        55: "55%",
        85: "85%",
    },
    "bcd": {
        "title": "(a) 主问题训练日程（与启发式对应）",
        "subtitle": "",
        "caption": "",
        "rows": {
            "unit_warm_start": "机组预测热启动",
            "theta_activation": "代理约束渐进加入",
            "theta_stage": "代理约束：结构/数量分阶段",
            "dual_floor": r"对偶变量 $\lambda$：独立下界",
            "dual_flip": r"对偶变量 $\lambda$：符号翻转",
            "single_mu_cap": "对偶变量限制",
            "nn_direct": "NN-direct",
        },
        "bars": {
            "unit_pretrain": "0/1预测器预训",
            "unit_bootstrap": "作热启动",
            "unit_finetune": "随代理微调",
            "theta_delay": "等待",
            "theta_ramp": "渐进加入",
            "theta_full": "全开",
            "stage_delay": "等待",
            "stage_6": r"$K{\leq}6$",
            "stage_12": r"$K{\leq}12$",
            "stage_20": r"$K{\leq}20$",
            "independent_floor": "独立下界",
            "group_floor": r"$\lambda$ 总体下界",
            "floor_release": "解除",
            "sign_flip": "条件允许翻转",
            "single_cap_ramp": "渐进",
            "single_cap_full": "惩罚",
            "pseudo_label": "KKT-Loss",
            "direct_fit": "direct",
        },
    },
    "surrogate": {
        "title": "(b) 子问题训练日程（与启发式对应）",
        "subtitle": "",
        "caption": "",
        "rows": {
            "dual_predictor": r"对偶变量 $\lambda$：拉格朗日松弛预训",
            "unit_predictor": "机组预测热启动",
            "sign4_activation": "代理约束渐进加入",
            "sign4_floor": r"对偶变量 $\lambda$：下界（独立$\rightarrow$打包）",
            "single_mu_cap": "对偶变量限制",
            "main_direct": "NN-direct",
            "c_pg": r"$h(\theta)$ 分阶段加入",
        },
        "bars": {
            "dual_pretrain": r"$\lambda$-LR 预训",
            "unit_pretrain": "0/1预测器预训",
            "unit_finetune": "随代理微调",
            "single_only": "等待",
            "sign4_active": "全开",
            "sign4_individual": r"$\lambda$ 独立下界",
            "sign4_group": r"$\lambda$ 总体下界",
            "floor_release": "解除",
            "sign4_scale": "渐进加入",
            "single_cap_ramp": "渐进",
            "single_cap_full": "惩罚",
            "main_direct": "",
            "cost_wait": r"$h(\theta)$ 未参与",
            "cost_refine": r"纳入 $h(\theta)$",
        },
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_text_config(path: Path | None) -> dict[str, Any]:
    text = deepcopy(TEXT)
    if path is None:
        return text
    with path.open("r", encoding="utf-8") as f:
        override = json.load(f)
    if not isinstance(override, dict):
        raise ValueError("--text-json must contain a JSON object")
    return _deep_update(text, override)


def _configure_fonts(text: dict[str, Any]) -> None:
    plt.rcParams["font.sans-serif"] = list(text.get("font_family", []))
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _add_bar(
    ax: plt.Axes,
    y: float,
    start: float,
    end: float,
    color: str,
    label: str,
    *,
    height: float = 0.58,
    alpha: float = 1.0,
    text_color: str = "#1F2933",
    fontsize: float = 9.5,
    draw_label: bool = False,
    label_min_width: float = 5.0,
) -> None:
    width = max(float(end) - float(start), 0.0)
    ax.barh(
        y,
        width,
        left=start,
        height=height,
        color=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    if draw_label and label and width >= label_min_width:
        ax.text(
            start + width / 2.0,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            zorder=4,
        )


def _add_pre_bar(
    ax: plt.Axes,
    y: float,
    color: str,
    label: str,
    *,
    draw_label: bool,
    label_min_width: float = 5.0,
    fontsize: float = 9.0,
) -> None:
    _add_bar(
        ax,
        y,
        -18,
        -2,
        color,
        label,
        text_color="white",
        fontsize=fontsize,
        draw_label=draw_label,
        label_min_width=label_min_width,
    )


MAJOR_TICK_X = frozenset({0, 25, 50, 75, 100})


def _format_axis(ax: plt.Axes, row_labels: list[str], text: dict[str, Any], guide_lines: list[int]) -> None:
    y_positions = list(reversed(range(len(row_labels))))
    ax.set_xlim(-20, 100)
    ax.set_ylim(-0.9, len(row_labels) - 0.12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=float(text.get("ytick_fontsize", 9.5)))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(text["axis_ticks"], fontsize=float(text.get("xtick_fontsize", 10.0)))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.minorticks_off()
    ax.set_xlabel(text["x_label"], fontsize=float(text.get("xlabel_fontsize", 11.0)))
    ax.grid(axis="x", color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#5B6770", linewidth=1.0, zorder=2)
    show_guide_lbl = bool(text.get("show_guide_line_labels", False))
    for x in guide_lines:
        ax.axvline(x, color="#A0A8B0", linestyle=(0, (3, 3)), linewidth=0.85, zorder=1)
        if not show_guide_lbl:
            continue
        if x in MAJOR_TICK_X:
            continue
        label = text["guide_labels"].get(x, f"{x}%")
        ax.text(x, -1.08, label, ha="center", va="top", fontsize=8.0, color="#5B6770")
    pre_loop = (text.get("pre_loop_label") or "").strip()
    if pre_loop:
        ax.text(-10, -1.08, pre_loop, ha="center", va="top", fontsize=8.0, color="#5B6770")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD2D9")


def _plot_bcd(output_dir: Path, stem: str, text: dict[str, Any], dpi: int) -> list[Path]:
    cfg = text["bcd"]
    draw_bar = bool(text.get("show_bar_text", False))
    lmw = float(text.get("bar_label_min_width", 5.0))
    bf = float(text.get("bar_fontsize", 9.5))
    pbf = float(text.get("pre_bar_fontsize", 9.0))
    row_keys = [
        "unit_warm_start",
        "theta_activation",
        "theta_stage",
        "dual_floor",
        "dual_flip",
        "single_mu_cap",
        "nn_direct",
    ]
    row_labels = [cfg["rows"][key] for key in row_keys]
    y = dict(zip(row_keys, reversed(range(len(row_keys)))))

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _add_pre_bar(
        ax,
        y["unit_warm_start"],
        COLORS["pretrain"],
        cfg["bars"]["unit_pretrain"],
        draw_label=draw_bar,
        label_min_width=lmw,
        fontsize=pbf,
    )
    _add_bar(
        ax,
        y["unit_warm_start"],
        0,
        100,
        COLORS["warm_light"],
        cfg["bars"]["unit_finetune"],
        draw_label=draw_bar,
        label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["theta_activation"],
        0,
        10,
        COLORS["inactive"],
        cfg["bars"]["theta_delay"],
        draw_label=draw_bar,
        label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["theta_activation"],
        10,
        45,
        COLORS["constraint"],
        cfg["bars"]["theta_ramp"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["theta_activation"],
        45,
        100,
        COLORS["constraint_light"],
        cfg["bars"]["theta_full"],
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["theta_stage"],
        0,
        10,
        COLORS["inactive"],
        cfg["bars"]["stage_delay"],
        draw_label=draw_bar,
        label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["theta_stage"],
        10,
        25,
        COLORS["constraint"],
        cfg["bars"]["stage_6"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["theta_stage"],
        25,
        50,
        COLORS["constraint"],
        cfg["bars"]["stage_12"],
        text_color="white",
        alpha=0.86,
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["theta_stage"],
        50,
        100,
        COLORS["constraint_light"],
        cfg["bars"]["stage_20"],
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["dual_floor"],
        0,
        85,
        COLORS["dual"],
        cfg["bars"]["independent_floor"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["dual_floor"],
        85,
        100,
        COLORS["inactive"],
        cfg["bars"]["floor_release"],
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["dual_flip"],
        15,
        55,
        COLORS["flip"],
        cfg["bars"]["sign_flip"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["single_mu_cap"],
        25,
        55,
        COLORS["dual_light"],
        cfg["bars"]["single_cap_ramp"],
        text_color="#1F2933",
        alpha=0.72,
        draw_label=draw_bar,
        label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["single_mu_cap"],
        55,
        100,
        COLORS["dual"],
        cfg["bars"]["single_cap_full"],
        text_color="white",
        alpha=1.0,
        draw_label=draw_bar,
        label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["nn_direct"],
        0,
        100,
        COLORS["direct"],
        cfg["bars"]["pseudo_label"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["nn_direct"] + 0.25,
        0,
        100,
        COLORS["direct_light"],
        cfg["bars"]["direct_fit"],
        height=0.22,
        fontsize=bf,
        alpha=0.82,
        draw_label=draw_bar,
        label_min_width=lmw,
    )

    _format_axis(ax, row_labels, text, [10, 25, 40, 45, 50, 55, 85])
    tit_fs = float(text.get("title_fontsize", 11.0))
    tit_fw = text.get("title_fontweight", "normal")
    tit_loc = str(text.get("title_loc", "center"))
    ax.set_title(cfg["title"], loc=tit_loc, fontsize=tit_fs, fontweight=tit_fw, pad=14)
    sub = (cfg.get("subtitle") or "").strip()
    if bool(text.get("show_figure_subtitle", False)) and sub:
        ax.text(-20, len(row_labels) - 0.35, sub, ha="left", va="top", fontsize=9.0, color="#52616B")
    cap = cfg.get("caption") or ""
    if cap:
        fig.text(0.075, 0.04, cap, ha="left", va="bottom", fontsize=8.0, color="#4A5560")
    fig.subplots_adjust(left=0.21, right=0.93, top=0.92, bottom=0.13)

    paths = [
        output_dir / f"{stem}_bcd.png",
        output_dir / f"{stem}_bcd.pdf",
        output_dir / f"{stem}_bcd.svg",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.2)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return paths


def _plot_surrogate(output_dir: Path, stem: str, text: dict[str, Any], dpi: int) -> list[Path]:
    cfg = text["surrogate"]
    draw_bar = bool(text.get("show_bar_text", False))
    lmw = float(text.get("bar_label_min_width", 5.0))
    bf = float(text.get("bar_fontsize", 9.5))
    pbf = float(text.get("pre_bar_fontsize", 9.0))
    row_keys = [
        "dual_predictor",
        "unit_predictor",
        "sign4_activation",
        "sign4_floor",
        "single_mu_cap",
        "main_direct",
        "c_pg",
    ]
    row_labels = [cfg["rows"][key] for key in row_keys]
    y = dict(zip(row_keys, reversed(range(len(row_keys)))))

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _add_pre_bar(
        ax,
        y["dual_predictor"],
        COLORS["dual"],
        cfg["bars"]["dual_pretrain"],
        draw_label=draw_bar,
        label_min_width=lmw,
        fontsize=pbf,
    )
    _add_pre_bar(
        ax,
        y["unit_predictor"],
        COLORS["pretrain"],
        cfg["bars"]["unit_pretrain"],
        draw_label=draw_bar,
        label_min_width=lmw,
        fontsize=pbf,
    )
    _add_bar(
        ax, y["unit_predictor"], 0, 100, COLORS["warm_light"], cfg["bars"]["unit_finetune"], draw_label=draw_bar, label_min_width=lmw
    )

    _add_bar(
        ax,
        y["sign4_activation"],
        0,
        5,
        COLORS["inactive"],
        cfg["bars"]["single_only"],
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["sign4_activation"],
        5,
        45,
        COLORS["constraint"],
        cfg["bars"]["sign4_scale"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["sign4_activation"],
        45,
        100,
        COLORS["constraint_light"],
        cfg["bars"]["sign4_active"],
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["sign4_floor"],
        0,
        50,
        COLORS["dual"],
        cfg["bars"]["sign4_individual"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(ax, y["sign4_floor"], 50, 85, COLORS["dual_light"], cfg["bars"]["sign4_group"], draw_label=draw_bar, label_min_width=lmw)
    _add_bar(
        ax,
        y["sign4_floor"],
        85,
        100,
        COLORS["inactive"],
        cfg["bars"]["floor_release"],
        draw_label=draw_bar, label_min_width=lmw,
    )

    _add_bar(
        ax,
        y["single_mu_cap"],
        25,
        55,
        COLORS["dual_light"],
        cfg["bars"]["single_cap_ramp"],
        text_color="#1F2933",
        alpha=0.72,
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["single_mu_cap"],
        55,
        100,
        COLORS["dual"],
        cfg["bars"]["single_cap_full"],
        text_color="white",
        alpha=1.0,
        draw_label=draw_bar, label_min_width=lmw,
    )
    # Same stacking as main-problem nn_direct: base vs. thin direct-regression lane.
    bcd_bars = text["bcd"]["bars"]
    _add_bar(
        ax,
        y["main_direct"],
        0,
        100,
        COLORS["direct"],
        bcd_bars["pseudo_label"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["main_direct"] + 0.25,
        0,
        100,
        COLORS["direct_light"],
        bcd_bars["direct_fit"],
        height=0.22,
        fontsize=bf,
        alpha=0.82,
        draw_label=draw_bar, label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["c_pg"],
        0,
        25,
        COLORS["inactive"],
        cfg["bars"]["cost_wait"],
        draw_label=draw_bar,
        label_min_width=lmw,
    )
    _add_bar(
        ax,
        y["c_pg"],
        25,
        100,
        COLORS["cost"],
        cfg["bars"]["cost_refine"],
        text_color="white",
        draw_label=draw_bar, label_min_width=lmw,
    )

    _format_axis(ax, row_labels, text, [5, 25, 45, 50, 55, 85])
    tit_fs = float(text.get("title_fontsize", 11.0))
    tit_fw = text.get("title_fontweight", "normal")
    tit_loc = str(text.get("title_loc", "center"))
    ax.set_title(cfg["title"], loc=tit_loc, fontsize=tit_fs, fontweight=tit_fw, pad=14)
    sub = (cfg.get("subtitle") or "").strip()
    if bool(text.get("show_figure_subtitle", False)) and sub:
        ax.text(-20, len(row_labels) - 0.35, sub, ha="left", va="top", fontsize=9.0, color="#52616B")
    cap = cfg.get("caption") or ""
    if cap:
        fig.text(0.075, 0.04, cap, ha="left", va="bottom", fontsize=8.0, color="#4A5560")
    fig.subplots_adjust(left=0.21, right=0.93, top=0.92, bottom=0.13)

    paths = [
        output_dir / f"{stem}_surrogate.png",
        output_dir / f"{stem}_surrogate.pdf",
        output_dir / f"{stem}_surrogate.svg",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.2)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return paths


def plot_schedules(output_dir: Path, stem: str, text: dict[str, Any], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_fonts(text)
    paths: list[Path] = []
    paths.extend(_plot_bcd(output_dir, stem, text, dpi))
    paths.extend(_plot_surrogate(output_dir, stem, text, dpi))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="strong_complex_dual_floor_schedule")
    parser.add_argument("--text-json", type=Path, default=None, help="JSON file overriding visible text labels.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    text = load_text_config(args.text_json)
    paths = plot_schedules(args.output_dir, args.stem, text, args.dpi)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
