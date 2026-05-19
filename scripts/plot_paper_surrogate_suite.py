#!/usr/bin/env python3
"""Build paper figures for surrogate-model tests.

Input is a directory produced by ``scripts/run_paper_evaluation_suite.py``.
The script focuses on the paper subset:

* LP relaxation baseline
* BCD theta only
* Subproblem sign4 only
* All surrogate rows

It writes tidy CSV files and publication-style PNG/PDF figures for:

* solve time and solution quality
* test-time activity
* constraint count versus model effect
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "result" / "paper_eval" / "20260512_183843"

METHOD_ORDER = ("lp", "bcd_theta", "surrogate_sign4", "full_joint")
MODEL_METHODS = ("bcd_theta", "surrogate_sign4", "full_joint")
METHOD_LABEL = {
    "lp": "LP",
    "bcd_theta": "仅主代理约束",
    "surrogate_sign4": "仅子代理约束",
    "full_joint": "全部代理约束",
}
METHOD_COLOR = {
    "lp": "#4D4D4D",
    "bcd_theta": "#2166AC",
    "surrogate_sign4": "#D6604D",
    "full_joint": "#1B7837",
}
METHOD_TO_RUN_ID = {
    "bcd_theta": "S01_bcd_theta",
    "surrogate_sign4": "S03_surrogate_sign4",
    "full_joint": "S05_full_joint",
}
CASE_LABEL = {
    "case14": "case14(c14)",
    "case30lite": "case30(c30)",
    "case30": "case30(c30)",
    "case3lite": "case3(c3)",
    "case3": "case3(c3)",
}
CASE_SHORT_LABEL = {
    "case14": "c14",
    "case30lite": "c30",
    "case30": "c30",
    "case3lite": "c3",
    "case3": "c3",
}
CASE_MARKER = {
    "case14": "o",
    "case30lite": "s",
    "case30": "s",
    "case3lite": "^",
    "case3": "^",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], preferred: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred)
    extra = sorted({k for row in rows for k in row} - set(fields))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sf(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _si(value) -> int | None:
    val = _sf(value)
    return None if val is None else int(val)


def _case_sample_key(row: dict) -> tuple[str, int] | None:
    case = str(row.get("case", "")).strip()
    sample = _si(row.get("sample_index"))
    if not case or sample is None:
        return None
    return case, sample


def _case_label(case: str) -> str:
    return CASE_LABEL.get(case, case)


def _case_short_label(case: str) -> str:
    return CASE_SHORT_LABEL.get(case, case)


def _case_marker(case: str) -> str:
    return CASE_MARKER.get(case, "o")


def _norm_hamming(row: dict, hamming_key: str, fallback_key: str = "normalized_hamming") -> float | None:
    fallback = _sf(row.get(fallback_key))
    if fallback is not None and hamming_key == "hamming_to_true":
        return fallback
    hamming = _sf(row.get(hamming_key))
    n_units = _si(row.get("n_units"))
    horizon = _si(row.get("T"))
    if hamming is None or not n_units or not horizon:
        return None
    return hamming / float(n_units * horizon)


def _lp_runtime_lookup(rows: list[dict]) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for row in rows:
        case = str(row.get("case", "")).strip()
        sample = _si(row.get("sample_index"))
        if case and sample is not None:
            out[case, sample] = row
    return out


def build_effect_rows(sample_rows: list[dict], lp_runtime_rows: list[dict] | None = None) -> list[dict]:
    """Create one tidy row per case/sample/method for plotting."""
    effect_rows: list[dict] = []
    lp_lookup = _lp_runtime_lookup(lp_runtime_rows or [])

    # One LP row per case/sample, using the first model row carrying baseline fields.
    seen_lp: set[tuple[str, int]] = set()
    for row in sample_rows:
        key = _case_sample_key(row)
        if key is None or key in seen_lp:
            continue
        if _sf(row.get("global_base_hamming_to_true")) is None:
            continue
        case, sample = key
        lp_stats = lp_lookup.get(key, {})
        seen_lp.add(key)
        effect_rows.append(
            {
                "case": case,
                "sample_index": sample,
                "method": "lp",
                "method_label": METHOD_LABEL["lp"],
                "runtime_sec": lp_stats.get("runtime_sec", row.get("global_base_runtime_sec", "")),
                "integrality_gap": row.get("global_base_integrality_gap", ""),
                "l1_to_true": row.get("global_base_l1_to_true", ""),
                "hamming_to_true": row.get("global_base_hamming_to_true", ""),
                "normalized_hamming": _norm_hamming(row, "global_base_hamming_to_true"),
                "num_constraints": lp_stats.get("num_constraints", row.get("global_base_num_constraints", "")),
                "num_vars": lp_stats.get("num_vars", row.get("global_base_num_vars", "")),
                "num_nonzeros": lp_stats.get("num_nonzeros", row.get("global_base_num_nonzeros", "")),
                "num_proxy_constraints": 0,
            }
        )

    for row in sample_rows:
        method = str(row.get("method", "")).strip()
        if method not in MODEL_METHODS:
            continue
        key = _case_sample_key(row)
        if key is None:
            continue
        case, sample = key
        proxy = _proxy_constraint_count(row, method)
        effect_rows.append(
            {
                "case": case,
                "sample_index": sample,
                "method": method,
                "method_label": METHOD_LABEL[method],
                "runtime_sec": row.get("runtime_sec", ""),
                "integrality_gap": row.get("integrality_gap", ""),
                "l1_to_true": row.get("l1_to_true", ""),
                "hamming_to_true": row.get("hamming_to_true", ""),
                "normalized_hamming": row.get("normalized_hamming", ""),
                "num_constraints": row.get("num_constraints", ""),
                "num_vars": row.get("num_vars", ""),
                "num_nonzeros": row.get("num_nonzeros", ""),
                "num_proxy_constraints": proxy if proxy is not None else "",
                "num_base_constraints": row.get("num_base_constraints", ""),
                "num_subproblem_surrogate_constraints": row.get("num_subproblem_surrogate_constraints", ""),
                "num_bcd_theta_constraints": row.get("num_bcd_theta_constraints", ""),
                "num_bcd_zeta_constraints": row.get("num_bcd_zeta_constraints", ""),
            }
        )
    _fill_proxy_count_fallbacks(effect_rows)
    return effect_rows


def _proxy_constraint_count(row: dict, method: str) -> int | None:
    direct = _si(row.get("num_proxy_constraints"))
    if direct is not None:
        return direct
    if method == "bcd_theta":
        return _si(row.get("num_bcd_theta_constraints")) or _si(row.get("num_bcd_theta_slacks"))
    if method == "surrogate_sign4":
        return _si(row.get("num_subproblem_sign4_slacks")) or _si(row.get("num_subproblem_surrogate_constraints"))
    if method == "full_joint":
        parts = [
            _si(row.get("num_subproblem_surrogate_constraints")),
            _si(row.get("num_bcd_theta_constraints")),
            _si(row.get("num_bcd_zeta_constraints")),
        ]
        if any(v is not None for v in parts):
            return int(sum(v or 0 for v in parts))
        parts = [
            _si(row.get("num_subproblem_slacks")),
            _si(row.get("num_bcd_theta_slacks")),
            _si(row.get("num_bcd_zeta_slacks")),
        ]
        if any(v is not None for v in parts):
            return int(sum(v or 0 for v in parts))
    return None


def _fill_proxy_count_fallbacks(effect_rows: list[dict]) -> None:
    """Backfill proxy counts for older result files that dropped hard-row counts."""
    known_base_by_case: dict[str, float] = {}
    for case in sorted({r["case"] for r in effect_rows}):
        candidates = []
        for row in effect_rows:
            if row["case"] != case or row["method"] == "lp":
                continue
            total = _sf(row.get("num_constraints"))
            proxy = _sf(row.get("num_proxy_constraints"))
            if total is not None and proxy is not None:
                candidates.append(total - proxy)
        if candidates:
            known_base_by_case[case] = min(candidates)

    for row in effect_rows:
        if row["method"] == "lp":
            base = known_base_by_case.get(row["case"])
            if _sf(row.get("num_constraints")) is None and base is not None:
                row["num_constraints"] = int(round(base))
            continue
        total = _sf(row.get("num_constraints"))
        current = _sf(row.get("num_proxy_constraints"))
        base = known_base_by_case.get(row["case"])
        if total is None or base is None:
            continue
        fallback = max(0.0, total - base)
        if current is None or current <= 0:
            row["num_proxy_constraints"] = int(round(fallback))


def _boxplot(ax: plt.Axes, values_by_method: list[list[float]], labels: list[str]) -> None:
    kwargs = dict(showfliers=False, patch_artist=True, widths=0.58)
    try:
        box = ax.boxplot(values_by_method, tick_labels=labels, **kwargs)
    except TypeError:
        box = ax.boxplot(values_by_method, labels=labels, **kwargs)
    for patch, label in zip(box["boxes"], labels):
        method = next((m for m, text in METHOD_LABEL.items() if text == label), None)
        patch.set_facecolor(METHOD_COLOR.get(method or "", "#999999"))
        patch.set_alpha(0.22)
        patch.set_edgecolor("#333333")
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)


def plot_effects(effect_rows: list[dict], out_dir: Path) -> None:
    cases = sorted({r["case"] for r in effect_rows})
    metrics = (
        ("runtime_sec", "运行时间", "A"),
        ("l1_to_true", "与真实解距离", "B"),
        ("normalized_hamming", "取整后距离", "C"),
    )
    fig, axes = plt.subplots(len(cases), len(metrics), figsize=(11.5, max(3.0, 2.75 * len(cases))), squeeze=False)
    for row_idx, case in enumerate(cases):
        for col_idx, (metric, ylabel, panel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            plot_data: list[list[float]] = []
            labels: list[str] = []
            for method in METHOD_ORDER:
                vals = [
                    _sf(r.get(metric))
                    for r in effect_rows
                    if r["case"] == case and r["method"] == method
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    plot_data.append(vals)
                    labels.append(METHOD_LABEL[method])
            if plot_data:
                _boxplot(ax, plot_data, labels)
                for idx, vals in enumerate(plot_data, start=1):
                    jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                    ax.scatter(
                        np.full(len(vals), idx) + jitter,
                        vals,
                        s=8,
                        color="#333333",
                        alpha=0.35,
                        linewidths=0,
                    )
            elif metric == "runtime_sec":
                ax.text(0.5, 0.5, "LP runtime not recorded", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{panel}. {_case_label(case)}", loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    _save_figure(fig, out_dir / "solve_effect_runtime_quality")


def _weighted_mean(rows: list[dict], value_key: str, weight_key: str = "n_rows") -> float | None:
    pairs = []
    for row in rows:
        value = _sf(row.get(value_key))
        if value is None:
            continue
        weight = _sf(row.get(weight_key)) or 1.0
        pairs.append((value, max(weight, 0.0)))
    if not pairs:
        return None
    denom = sum(w for _v, w in pairs)
    if denom <= 0:
        return mean(v for v, _w in pairs)
    return sum(v * w for v, w in pairs) / denom


def _find_activity_roots(run_dir: Path, case: str) -> tuple[Path | None, Path | None, Path | None]:
    direct = run_dir / "activity"
    if direct.is_dir():
        return direct, direct, None
    copied = run_dir / "raw" / "figures" / f"{case}_global_surrogate_solve_stats"
    activity = copied / "activity"
    main_activity = copied / "main_activity"
    return (
        activity if activity.is_dir() else None,
        activity if activity.is_dir() else None,
        main_activity if main_activity.is_dir() else None,
    )


def build_activity_rows(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    run_root = eval_dir / "raw" / "runs"
    for case_dir in sorted(run_root.iterdir() if run_root.is_dir() else []):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        for method in MODEL_METHODS:
            run_id = METHOD_TO_RUN_ID[method]
            run_dir = case_dir / run_id
            if not run_dir.is_dir():
                continue
            activity_root, main_root, sample_main_root = _find_activity_roots(run_dir, case)

            sub_rows = _read_csv((activity_root or Path()) / "aggregate_constraint_activity.csv")
            main_rows = _read_csv((main_root or Path()) / "main_model_activity_summary.csv")
            if not main_rows and sample_main_root is not None:
                main_rows = _read_csv(sample_main_root / "main_constraint_activity_summary.csv")

            theta_rows = [r for r in main_rows if str(r.get("kind", "")).startswith("bcd_theta")]
            zeta_rows = [r for r in main_rows if str(r.get("kind", "")).startswith("bcd_zeta")]
            has_subproblem = method in {"surrogate_sign4", "full_joint"}
            has_theta = method in {"bcd_theta", "full_joint"}
            has_zeta = method == "full_joint"
            sub_active = (
                _weighted_mean(sub_rows, "test_active_rate_after_relax", "n_test_rows")
                if has_subproblem
                else None
            )
            sub_relaxed = (
                _weighted_mean(sub_rows, "test_rhs_relaxed_rate", "n_test_rows")
                if has_subproblem
                else None
            )
            theta_active = (
                _weighted_mean(theta_rows, "row_active_rate", "n_rows")
                if has_theta
                else None
            )
            theta_dual = (
                _weighted_mean(theta_rows, "dual_active_rate", "n_rows")
                if has_theta
                else None
            )
            zeta_active = (
                _weighted_mean(zeta_rows, "row_active_rate", "n_rows")
                if has_zeta
                else None
            )

            rows.append(
                {
                    "case": case,
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "n_subproblem_constraints": len(sub_rows) if has_subproblem else 0,
                    "n_theta_rows": len(theta_rows) if has_theta else 0,
                    "n_zeta_rows": len(zeta_rows) if has_zeta else 0,
                    "subproblem_active_rate": sub_active if sub_active is not None else "",
                    "subproblem_rhs_relaxed_rate": sub_relaxed if sub_relaxed is not None else "",
                    "theta_row_active_rate": theta_active if theta_active is not None else "",
                    "theta_dual_active_rate": theta_dual if theta_dual is not None else "",
                    "zeta_row_active_rate": zeta_active if zeta_active is not None else "",
                }
            )
    return rows


def plot_activity(activity_rows: list[dict], out_dir: Path) -> None:
    if not activity_rows:
        return
    cases = sorted({r["case"] for r in activity_rows})
    metrics = (
        ("theta_row_active_rate", "主代理约束活跃率"),
        ("subproblem_active_rate", "子代理约束活跃率"),
        ("subproblem_rhs_relaxed_rate", "子代理约束RHS放松率"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 3.8), squeeze=False)
    x = np.arange(len(cases))
    width = 0.23
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for idx, method in enumerate(MODEL_METHODS):
            vals = []
            for case in cases:
                row = next((r for r in activity_rows if r["case"] == case and r["method"] == method), None)
                vals.append(_sf(row.get(metric)) if row else None)
            y = [np.nan if v is None else v for v in vals]
            ax.bar(
                x + (idx - 1) * width,
                y,
                width=width,
                label=METHOD_LABEL[method],
                color=METHOD_COLOR[method],
                alpha=0.82,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_case_label(case) for case in cases])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.flat[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, out_dir / "activity_rates")


def build_tradeoff_rows(effect_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in effect_rows:
        grouped[(row["case"], row["method"])].append(row)

    out: list[dict] = []
    for (case, method), rows in sorted(grouped.items()):
        summary = {"case": case, "method": method, "method_label": METHOD_LABEL[method]}
        for key in (
            "runtime_sec",
            "integrality_gap",
            "l1_to_true",
            "hamming_to_true",
            "normalized_hamming",
            "num_constraints",
            "num_proxy_constraints",
        ):
            vals = [_sf(r.get(key)) for r in rows]
            vals = [v for v in vals if v is not None]
            summary[f"mean_{key}"] = mean(vals) if vals else ""
            summary[f"median_{key}"] = float(np.median(vals)) if vals else ""
        out.append(summary)
    return out


def plot_tradeoff(tradeoff_rows: list[dict], out_dir: Path) -> None:
    cases = sorted({r["case"] for r in tradeoff_rows})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    y_metrics = (
        ("mean_runtime_sec", "运行时间"),
        ("mean_l1_to_true", "与真实解距离"),
        ("mean_normalized_hamming", "取整后距离"),
    )
    for ax, (metric, ylabel) in zip(axes, y_metrics):
        for method in METHOD_ORDER:
            for case in cases:
                row = next((r for r in tradeoff_rows if r["case"] == case and r["method"] == method), None)
                if row is None:
                    continue
                xval = _sf(row.get("mean_num_constraints"))
                if xval is None:
                    xval = _sf(row.get("mean_num_proxy_constraints"))
                yval = _sf(row.get(metric))
                if xval is None or yval is None:
                    continue
                ax.scatter(
                    [xval],
                    [yval],
                    s=58,
                    color=METHOD_COLOR[method],
                    marker=_case_marker(case),
                    edgecolor="#222222",
                    linewidth=0.45,
                    alpha=0.9,
                )
        ax.set_xlabel("平均约束条数")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor="#222222",
            markersize=7,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    case_handles = [
        Line2D(
            [0],
            [0],
            marker=_case_marker(case),
            color="#555555",
            markerfacecolor="white",
            markeredgecolor="#222222",
            linestyle="none",
            markersize=7,
            label=_case_label(case),
        )
        for case in cases
    ]
    axes[-1].legend(handles=method_handles + case_handles, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    _save_figure(fig, out_dir / "constraint_count_tradeoff")


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.paper_eval_dir.resolve()
    out_dir = (args.output_dir or (eval_dir / "figures_paper")).resolve()
    data_dir = eval_dir / "data"
    sample_rows = _read_csv(data_dir / "sample_metrics.csv")
    if not sample_rows:
        raise SystemExit(f"No sample metrics found: {data_dir / 'sample_metrics.csv'}")
    lp_runtime_rows = _read_csv(data_dir / "lp_runtime_baseline.csv")

    effect_rows = build_effect_rows(sample_rows, lp_runtime_rows=lp_runtime_rows)
    activity_rows = build_activity_rows(eval_dir)
    tradeoff_rows = build_tradeoff_rows(effect_rows)

    _write_csv(
        data_dir / "paper_effect_long.csv",
        effect_rows,
        (
            "case",
            "sample_index",
            "method",
            "method_label",
            "runtime_sec",
            "integrality_gap",
            "l1_to_true",
            "hamming_to_true",
            "normalized_hamming",
            "num_constraints",
            "num_proxy_constraints",
        ),
    )
    _write_csv(
        data_dir / "paper_activity_summary.csv",
        activity_rows,
        (
            "case",
            "method",
            "method_label",
            "n_subproblem_constraints",
            "n_theta_rows",
            "n_zeta_rows",
            "subproblem_active_rate",
            "subproblem_rhs_relaxed_rate",
            "theta_row_active_rate",
            "theta_dual_active_rate",
            "zeta_row_active_rate",
        ),
    )
    _write_csv(
        data_dir / "paper_constraint_tradeoff.csv",
        tradeoff_rows,
        (
            "case",
            "method",
            "method_label",
            "mean_num_proxy_constraints",
            "mean_num_constraints",
            "mean_runtime_sec",
            "mean_integrality_gap",
            "mean_l1_to_true",
            "mean_hamming_to_true",
            "mean_normalized_hamming",
        ),
    )

    plot_effects(effect_rows, out_dir)
    plot_activity(activity_rows, out_dir)
    plot_tradeoff(tradeoff_rows, out_dir)

    print(f"wrote data: {data_dir}")
    print(f"wrote figures: {out_dir}")


if __name__ == "__main__":
    main()
