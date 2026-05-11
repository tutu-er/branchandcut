#!/usr/bin/env python3
"""Plot the time-dependent heuristics used by strong-complex-dual-floor training.

The script intentionally keeps only strategies that are naturally expressed as
training schedules: warm starts, delayed activation, curriculum ramps, floor
switches, caps, and direct fitting phases. Static modeling choices are better
described in the caption/table rather than drawn as Gantt bars.
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
from matplotlib.patches import Patch


DEFAULT_OUTPUT_DIR = Path("result/paper_figures/strong_complex_dual_floor_schedule")


COLORS = {
    "pretrain": "#5B677A",
    "warm": "#4C78A8",
    "warm_light": "#A9C6E8",
    "constraint": "#2F6FBB",
    "constraint_light": "#7AC0D8",
    "dual": "#D97924",
    "dual_light": "#F0A54A",
    "flip": "#C06C84",
    "direct": "#2E8B57",
    "direct_light": "#82BC8A",
    "cap": "#8A5BB8",
    "cost": "#697386",
    "inactive": "#CBD2D9",
}


TEXT: dict[str, Any] = {
    "font_family": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "x_label": "Normalized training progress",
    "pre_loop_label": "Pretrain / init",
    "axis_ticks": ["0%", "25%", "50%", "75%", "100%"],
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
        "title": "BCD Training Heuristic Schedule",
        "subtitle": "Only time-dependent strategies are shown; static choices are summarized in the caption.",
        "caption": (
            "Static but important choices not drawn as bars: simplified BCD objective, smooth-|.| loss, "
            "deadband, inter-iteration regularization, and heuristic coefficient initialization."
        ),
        "rows": {
            "unit_warm_start": "Unit predictor warm start",
            "theta_activation": "h(theta) activation",
            "theta_stage": "h(theta) constraint budget",
            "dual_floor": "Dual lower bound",
            "dual_flip": "Dual sign relaxation",
            "nn_direct": "NN-direct local fit",
        },
        "bars": {
            "unit_pretrain": "pretrain predictor",
            "unit_bootstrap": "bootstrap x",
            "unit_finetune": "co-train / adapt",
            "theta_delay": "delayed",
            "theta_ramp": "curriculum ramp",
            "theta_full": "active",
            "stage_delay": "wait",
            "stage_6": "max 6",
            "stage_12": "max 12",
            "stage_20": "max 20",
            "independent_floor": "independent floor",
            "group_floor": "grouped floor",
            "floor_release": "release",
            "sign_flip": "conditional sign flip",
            "pseudo_label": "local pseudo-labels",
            "direct_fit": "direct MSE warm-up",
        },
    },
    "surrogate": {
        "title": "Subproblem Surrogate Training Heuristic Schedule",
        "subtitle": "The figure focuses on staged surrogate learning and strong-complex-dual-floor controls.",
        "caption": (
            "Static but important choices not drawn as bars: all-templates sign4 plus single-time constraint family, "
            "smooth-|.| loss, deadband, delta reference lift, and regularization."
        ),
        "rows": {
            "dual_predictor": "Dual predictor",
            "unit_predictor": "Unit predictor",
            "sign4_activation": "Sign4 activation",
            "sign4_floor": "Sign4 dual floor",
            "sign4_curriculum": "Sign4 curriculum",
            "single_mu_cap": "Single-time mu cap",
            "main_direct": "NN-main direct fit",
            "c_pg": "c_pg cost learning",
        },
        "bars": {
            "dual_pretrain": "pretrain lambda/dual",
            "unit_pretrain": "pretrain x predictor",
            "unit_finetune": "co-train / adapt",
            "single_only": "single-time only",
            "sign4_active": "sign4 enabled",
            "sign4_individual": "individual floor",
            "sign4_group": "grouped floor",
            "floor_release": "release",
            "sign4_scale": "scale 0.1 -> 2.0",
            "single_cap_ramp": "cap ramp-up",
            "single_cap_full": "full cap penalty",
            "main_direct": "full-batch direct targets",
            "cost_wait": "wait",
            "cost_refine": "dispatch-cost surrogate",
        },
    },
    "legend": {
        "pretrain": "Pretrain / warm start",
        "constraint": "Constraint activation / curriculum",
        "dual": "Dual floor / sign control",
        "direct": "Direct neural fitting",
        "cap": "Cap / penalty window",
        "cost": "Cost surrogate phase",
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
    fontsize: float = 8.4,
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
    if label and width >= 8:
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


def _add_pre_bar(ax: plt.Axes, y: float, color: str, label: str) -> None:
    _add_bar(ax, y, -18, -2, color, label, text_color="white", fontsize=7.6)


def _format_axis(ax: plt.Axes, row_labels: list[str], text: dict[str, Any], guide_lines: list[int]) -> None:
    y_positions = list(reversed(range(len(row_labels))))
    ax.set_xlim(-20, 100)
    ax.set_ylim(-0.9, len(row_labels) - 0.12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(text["axis_ticks"], fontsize=9)
    ax.set_xlabel(text["x_label"], fontsize=10.5)
    ax.grid(axis="x", color="#E5E9EF", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#5B6770", linewidth=1.0, zorder=2)
    for x in guide_lines:
        ax.axvline(x, color="#A0A8B0", linestyle=(0, (3, 3)), linewidth=0.85, zorder=1)
        label = text["guide_labels"].get(x, f"{x}%")
        ax.text(x, -1.08, label, ha="center", va="top", fontsize=8.0, color="#5B6770")
    ax.text(-10, -1.08, text["pre_loop_label"], ha="center", va="top", fontsize=8.0, color="#5B6770")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD2D9")


def _legend_handles(text: dict[str, Any]) -> list[Patch]:
    return [
        Patch(facecolor=COLORS["pretrain"], label=text["legend"]["pretrain"]),
        Patch(facecolor=COLORS["constraint"], label=text["legend"]["constraint"]),
        Patch(facecolor=COLORS["dual"], label=text["legend"]["dual"]),
        Patch(facecolor=COLORS["direct"], label=text["legend"]["direct"]),
        Patch(facecolor=COLORS["cap"], label=text["legend"]["cap"]),
        Patch(facecolor=COLORS["cost"], label=text["legend"]["cost"]),
    ]


def _plot_bcd(output_dir: Path, stem: str, text: dict[str, Any], dpi: int) -> list[Path]:
    cfg = text["bcd"]
    row_keys = [
        "unit_warm_start",
        "theta_activation",
        "theta_stage",
        "dual_floor",
        "dual_flip",
        "nn_direct",
    ]
    row_labels = [cfg["rows"][key] for key in row_keys]
    y = dict(zip(row_keys, reversed(range(len(row_keys)))))

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _add_pre_bar(ax, y["unit_warm_start"], COLORS["pretrain"], cfg["bars"]["unit_pretrain"])
    _add_bar(ax, y["unit_warm_start"], 0, 12, COLORS["warm"], cfg["bars"]["unit_bootstrap"], text_color="white")
    _add_bar(ax, y["unit_warm_start"], 12, 100, COLORS["warm_light"], cfg["bars"]["unit_finetune"])

    _add_bar(ax, y["theta_activation"], 0, 10, COLORS["inactive"], cfg["bars"]["theta_delay"], fontsize=7.6)
    _add_bar(ax, y["theta_activation"], 10, 45, COLORS["constraint"], cfg["bars"]["theta_ramp"], text_color="white")
    _add_bar(ax, y["theta_activation"], 45, 100, COLORS["constraint_light"], cfg["bars"]["theta_full"])

    _add_bar(ax, y["theta_stage"], 0, 10, COLORS["inactive"], cfg["bars"]["stage_delay"], fontsize=7.6)
    _add_bar(ax, y["theta_stage"], 10, 25, COLORS["constraint"], cfg["bars"]["stage_6"], text_color="white")
    _add_bar(ax, y["theta_stage"], 25, 50, COLORS["constraint"], cfg["bars"]["stage_12"], text_color="white", alpha=0.86)
    _add_bar(ax, y["theta_stage"], 50, 100, COLORS["constraint_light"], cfg["bars"]["stage_20"])

    _add_bar(ax, y["dual_floor"], 0, 40, COLORS["dual"], cfg["bars"]["independent_floor"], text_color="white")
    _add_bar(ax, y["dual_floor"], 40, 85, COLORS["dual_light"], cfg["bars"]["group_floor"])
    _add_bar(ax, y["dual_floor"], 85, 100, COLORS["inactive"], cfg["bars"]["floor_release"], fontsize=7.6)

    _add_bar(ax, y["dual_flip"], 15, 55, COLORS["flip"], cfg["bars"]["sign_flip"], text_color="white")

    _add_bar(ax, y["nn_direct"], 0, 100, COLORS["direct"], cfg["bars"]["pseudo_label"], text_color="white")
    _add_bar(
        ax,
        y["nn_direct"] + 0.25,
        0,
        100,
        COLORS["direct_light"],
        cfg["bars"]["direct_fit"],
        height=0.20,
        fontsize=7.4,
        alpha=0.82,
    )

    _format_axis(ax, row_labels, text, [10, 25, 40, 45, 50, 55, 85])
    ax.set_title(cfg["title"], loc="left", fontsize=14, fontweight="bold", pad=18)
    ax.text(-20, len(row_labels) - 0.35, cfg["subtitle"], ha="left", va="top", fontsize=9.5, color="#52616B")
    ax.legend(
        handles=_legend_handles(text),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(0.075, 0.045, cfg["caption"], ha="left", va="bottom", fontsize=8.3, color="#4A5560")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.86, bottom=0.27)

    paths = [
        output_dir / f"{stem}_bcd.png",
        output_dir / f"{stem}_bcd.pdf",
        output_dir / f"{stem}_bcd.svg",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_surrogate(output_dir: Path, stem: str, text: dict[str, Any], dpi: int) -> list[Path]:
    cfg = text["surrogate"]
    row_keys = [
        "dual_predictor",
        "unit_predictor",
        "sign4_activation",
        "sign4_floor",
        "sign4_curriculum",
        "single_mu_cap",
        "main_direct",
        "c_pg",
    ]
    row_labels = [cfg["rows"][key] for key in row_keys]
    y = dict(zip(row_keys, reversed(range(len(row_keys)))))

    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _add_pre_bar(ax, y["dual_predictor"], COLORS["pretrain"], cfg["bars"]["dual_pretrain"])
    _add_pre_bar(ax, y["unit_predictor"], COLORS["warm"], cfg["bars"]["unit_pretrain"])
    _add_bar(ax, y["unit_predictor"], 0, 100, COLORS["warm_light"], cfg["bars"]["unit_finetune"])

    _add_bar(ax, y["sign4_activation"], 0, 5, COLORS["inactive"], cfg["bars"]["single_only"], fontsize=7.3)
    _add_bar(ax, y["sign4_activation"], 5, 100, COLORS["constraint"], cfg["bars"]["sign4_active"], text_color="white")

    _add_bar(ax, y["sign4_floor"], 0, 50, COLORS["dual"], cfg["bars"]["sign4_individual"], text_color="white")
    _add_bar(ax, y["sign4_floor"], 50, 85, COLORS["dual_light"], cfg["bars"]["sign4_group"])
    _add_bar(ax, y["sign4_floor"], 85, 100, COLORS["inactive"], cfg["bars"]["floor_release"], fontsize=7.6)

    _add_bar(ax, y["sign4_curriculum"], 5, 45, COLORS["dual_light"], cfg["bars"]["sign4_scale"])
    _add_bar(ax, y["single_mu_cap"], 25, 55, COLORS["cap"], cfg["bars"]["single_cap_ramp"], text_color="white")
    _add_bar(ax, y["single_mu_cap"], 55, 100, COLORS["cap"], cfg["bars"]["single_cap_full"], text_color="white", alpha=0.72)
    _add_bar(ax, y["main_direct"], 0, 100, COLORS["direct"], cfg["bars"]["main_direct"], text_color="white")
    _add_bar(ax, y["c_pg"], 0, 25, COLORS["inactive"], cfg["bars"]["cost_wait"], fontsize=7.6)
    _add_bar(ax, y["c_pg"], 25, 100, COLORS["cost"], cfg["bars"]["cost_refine"], text_color="white")

    _format_axis(ax, row_labels, text, [5, 25, 45, 50, 55, 85])
    ax.set_title(cfg["title"], loc="left", fontsize=14, fontweight="bold", pad=18)
    ax.text(-20, len(row_labels) - 0.35, cfg["subtitle"], ha="left", va="top", fontsize=9.5, color="#52616B")
    ax.legend(
        handles=_legend_handles(text),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(0.075, 0.045, cfg["caption"], ha="left", va="bottom", fontsize=8.3, color="#4A5560")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.24)

    paths = [
        output_dir / f"{stem}_surrogate.png",
        output_dir / f"{stem}_surrogate.pdf",
        output_dir / f"{stem}_surrogate.svg",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
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
