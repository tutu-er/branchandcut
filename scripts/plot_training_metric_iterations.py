#!/usr/bin/env python3
"""从 result/training_metric 下的 training_metrics_*.json 绘制 BCD/surrogate 迭代曲线.

同名算例若存在多份 JSON（如无 metrics-tag 的「本文」与 --metrics-tag control 的「对照组」），
会自动合并为对比图：training_metrics_<case>_compare_surrogate_aggregate 等；
仍会为未纳入对比的文件生成单文件图。使用 --no-combine 可关闭合并。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.font_manager import FontProperties, fontManager

plt.rcParams["axes.unicode_minus"] = False


def configure_matplotlib_cjk_font(
    explicit_font_path: Path | str | None = None,
) -> str:
    """注册能显示中文的字形（避免方块乱码）；返回选用的字体名便于排查。"""
    plt.rcParams["axes.unicode_minus"] = False

    paths_to_try: list[Path] = []
    if explicit_font_path:
        paths_to_try.append(Path(explicit_font_path))
    import os

    env = os.environ.get("MPL_CJK_FONT_PATH") or os.environ.get("MATPLOTLIB_CJK_FONT")
    if env:
        paths_to_try.append(Path(env))
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for rel in (
        r"Fonts\msyh.ttc",
        r"Fonts\msyhbd.ttc",
        r"Fonts\msyhl.ttc",
        r"Fonts\simhei.ttf",
        r"Fonts\simsun.ttc",
        r"Fonts\msjh.ttc",
        r"Fonts\NotoSansCJKsc-Regular.otf",
        r"Fonts\Deng.ttf",
        r"Fonts\msjhbd.ttc",
    ):
        paths_to_try.append(windir / rel)

    for p in paths_to_try:
        try:
            if not p.is_file():
                continue
            fontManager.addfont(str(p))
            fam = FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans", "sans-serif"]
            plt.rcParams["pdf.fonttype"] = 42
            return fam
        except Exception:
            continue

    preferred = (
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "YaHei UI",
        "SimHei",
        "SimSun",
        "NSimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "HarmonyOS Sans SC",
        "DejaVu Sans",
    )
    available = {f.name for f in fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["pdf.fonttype"] = 42
            return name

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return "DejaVu Sans"


configure_matplotlib_cjk_font(None)

import numpy as np

# JSON 字段 -> 图例显示名
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "obj_primal": "r_prim",
    "obj_dual": "r_stat",
    "obj_opt": "r_comp",
}
# 三台 obj 残差按机组逐点相加后聚合
SUM_RESIDUAL_LEGEND_VERBOSE_UNWEIGHTED = "残差之和(r_prim+r_stat+r_comp)"
SUM_RESIDUAL_LEGEND_VERBOSE_WEIGHTED = (
    "ρ加权残差之和(ρ_prim·r_prim+ρ_stat·r_stat+ρ_comp·r_comp)"
)
SUM_RESIDUAL_LEGEND_SHORT_UNWEIGHTED = r"$\sum r$"
SUM_RESIDUAL_LEGEND_SHORT_WEIGHTED = r"$\sum \rho r$"
SUM_RESIDUAL_LEGEND_VERBOSE = SUM_RESIDUAL_LEGEND_VERBOSE_WEIGHTED
SUM_RESIDUAL_LEGEND = SUM_RESIDUAL_LEGEND_VERBOSE
SUM_RESIDUAL_LEGEND_SHORT = SUM_RESIDUAL_LEGEND_SHORT_WEIGHTED

OBJ_TRIPLE = ("obj_primal", "obj_dual", "obj_opt")
# surrogate JSON 中与 obj_* 同行的罚因子（与 TrainingLogger.log_surrogate_iter 一致）
OBJ_METRIC_TO_RHO: dict[str, str] = {
    "obj_primal": "rho_primal",
    "obj_dual": "rho_dual",
    "obj_opt": "rho_opt",
}
# 聚合 / per-unit surrogate 图仅绘制 r_stat 与「残差之和」；总和仍按三项相加
SURROGATE_PLOT_LINE_KEYS = ("obj_dual",)

# ρ 加权后的图例（与 METRIC_DISPLAY_NAMES 对应）
METRIC_DISPLAY_NAMES_RHO_WEIGHTED: dict[str, str] = {
    "obj_primal": r"$\rho\cdot r_{\mathrm{prim}}$",
    "obj_dual": r"$\rho\cdot r_{\mathrm{stat}}$",
    "obj_opt": r"$\rho\cdot r_{\mathrm{comp}}$",
}

# 默认 False：图内少字；`--verbose-figure-captions` 恢复长标题/脚注。
FIGURE_CAPTIONS_VERBOSE: bool = False


def _sum_residual_legend(*, rho_weighted: bool = True) -> str:
    if rho_weighted:
        return (
            SUM_RESIDUAL_LEGEND_VERBOSE_WEIGHTED
            if FIGURE_CAPTIONS_VERBOSE
            else SUM_RESIDUAL_LEGEND_SHORT_WEIGHTED
        )
    return (
        SUM_RESIDUAL_LEGEND_VERBOSE_UNWEIGHTED
        if FIGURE_CAPTIONS_VERBOSE
        else SUM_RESIDUAL_LEGEND_SHORT_UNWEIGHTED
    )


def _metric_legend_label(metric_key: str, *, rho_weighted: bool = True) -> str:
    if rho_weighted:
        return METRIC_DISPLAY_NAMES_RHO_WEIGHTED.get(
            metric_key, METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
        )
    return METRIC_DISPLAY_NAMES.get(metric_key, metric_key)


def _surrogate_row_weighted_dual_residual(row: dict[str, Any]) -> float | None:
    """与子问题 Loss 中对偶_stationarity 项一致：ρ_pg·r_pg + ρ_x·r_x + ρ_coc·r_coc。"""
    keys_obj = ("obj_dual_pg", "obj_dual_x", "obj_dual_coc")
    keys_rho = ("rho_dual_pg", "rho_dual_x", "rho_dual_coc")
    if any(row.get(k) is None for k in keys_obj + keys_rho):
        return None
    return sum(
        float(row[rk]) * float(row[ok]) for ok, rk in zip(keys_obj, keys_rho)
    )


def _surrogate_row_obj_value(
    row: dict[str, Any], obj_key: str, *, rho_weighted: bool
) -> float | None:
    """读取 surrogate 行 obj_*；加权时与同迭代记录中的罚因子相乘。"""
    v = row.get(obj_key)
    if v is None:
        return None
    if rho_weighted and obj_key == "obj_dual":
        w_dual = _surrogate_row_weighted_dual_residual(row)
        if w_dual is not None:
            return w_dual
    out = float(v)
    if rho_weighted:
        rk = OBJ_METRIC_TO_RHO.get(obj_key)
        if rk is not None:
            rho = row.get(rk)
            if rho is not None:
                out *= float(rho)
    return out


def _training_regions_brief(
    formal_start_iter: int,
    warm_omit_progress_fraction: float,
    *,
    mention_compare_linestyles: bool,
    extra_suffix: str = "",
) -> str:
    """默认（少字）模式：一行说明背景分区；对比图可附带线型↔组别说明。"""
    s = (
        f"浅色：热启动(逐步加约束)；绿色：正式训练(iter≥{formal_start_iter})；竖线为界"
    )
    if warm_omit_progress_fraction > 0:
        s += f"；斜线区：进度<{warm_omit_progress_fraction:.0%}不绘点"
    if mention_compare_linestyles:
        s += "；线型区分实验组与对照组"
    return s + extra_suffix


def _surrogate_aggregate_footnote(
    formal_start_iter: int,
    warm_omit_progress_fraction: float,
    *,
    compare_extra: bool,
) -> str:
    if not FIGURE_CAPTIONS_VERBOSE:
        return _training_regions_brief(
            formal_start_iter,
            warm_omit_progress_fraction,
            mention_compare_linestyles=compare_extra,
            extra_suffix="",
        )
    agg_sub = (
        f"浅色区：早期训练，逐步加入代理约束 (iter<{formal_start_iter})    |"
        f"    分界后：正式训练 (iter≥{formal_start_iter})"
    )
    if warm_omit_progress_fraction > 0:
        agg_sub += (
            f"    |    斜线区：热启动（进度<{warm_omit_progress_fraction:.0%} Loss无意义）"
        )
    if compare_extra:
        agg_sub += (
            "    |    线型区分运行（本文/对照等）；"
            "仅 r_stat（对偶_stationarity）与残差之和"
        )
    return agg_sub


def _default_surrogate_axes_title(
    *, dashed_sum_note: bool
) -> str | None:
    if not FIGURE_CAPTIONS_VERBOSE:
        return None
    base = (
        "r_stat：机组间均值；残差之和："
        "每台机组(r_prim+r_stat+r_comp)再取机组均值"
    )
    return base + "（黑色虚线）" if dashed_sum_note else base


def _default_main_surrogate_aggregate(
    title_core: str, title_suffix: str, *, compare: bool
) -> str:
    ts = title_suffix or ""
    if FIGURE_CAPTIONS_VERBOSE:
        if compare:
            return f"子模型典型残差迭代（对比）: {title_core}{ts}"
        return f"子模型典型残差迭代: {title_core}{ts}"
    if compare:
        return f"Surrogate residuals (compare): {title_core}{ts}"
    return f"Surrogate residuals: {title_core}{ts}"


def _flat_default_subtitle(
    formal_start_iter: int,
    warm_omit_progress_fraction: float,
    *,
    compare: bool,
) -> str:
    if not FIGURE_CAPTIONS_VERBOSE:
        return _training_regions_brief(
            formal_start_iter,
            warm_omit_progress_fraction,
            mention_compare_linestyles=compare,
        )
    if compare:
        return (
            f"浅色：早期训练，逐步加入代理约束 (iter<{formal_start_iter})  |  "
            f"正式训练 (iter≥{formal_start_iter})  |  线型区分运行"
        )
    return (
        f"浅色：早期训练，逐步加入代理约束 (iter<{formal_start_iter})  |  "
        f"正式训练 (iter≥{formal_start_iter})"
    )


def _default_main_flat(title_core: str, title_suffix: str, *, compare: bool) -> str:
    ts = title_suffix or ""
    if FIGURE_CAPTIONS_VERBOSE:
        if compare:
            return f"Training metrics（对比） — flat: {title_core}{ts}"
        return f"Training metrics (flat sections): {title_core}{ts}"
    if compare:
        return f"Flat metrics (compare): {title_core}{ts}"
    return f"Flat metrics: {title_core}{ts}"


def _per_unit_default_subtitle(
    formal_start_iter: int,
    warm_omit_progress_fraction: float,
    *,
    compare: bool,
) -> str:
    if not FIGURE_CAPTIONS_VERBOSE:
        return _training_regions_brief(
            formal_start_iter,
            warm_omit_progress_fraction,
            mention_compare_linestyles=compare,
        )
    if compare:
        pu_sub = (
            f"热启动 iter<{formal_start_iter}  |  正式训练 iter≥{formal_start_iter}  |  线型区分方案"
        )
    else:
        pu_sub = (
            f"热启动 iter<{formal_start_iter}  |  正式训练 iter≥{formal_start_iter}"
        )
    if warm_omit_progress_fraction > 0:
        pu_sub += (
            f"  |  斜线区：热启动迭代（进度<{warm_omit_progress_fraction:.0%} 不绘点）"
        )
    return pu_sub


def _default_main_per_unit(
    title_core: str, title_suffix: str, *, compare: bool
) -> str:
    ts = title_suffix or ""
    if FIGURE_CAPTIONS_VERBOSE:
        if compare:
            return f"Surrogate per-unit 残差（对比）: {title_core}{ts}"
        return f"Surrogate per-unit 残差: {title_core}{ts}"
    if compare:
        return f"Per-unit surrogate (compare): {title_core}{ts}"
    return f"Per-unit surrogate: {title_core}{ts}"


# 对比图：r_stat 固定色，不同运行（本文/对照…）用线型区分
COMPARE_METRIC_COLORS: dict[str, str] = {
    "obj_primal": "#1f77b4",
    "obj_dual": "#ff7f0e",
    "obj_opt": "#2ca02c",
}
COMPARE_RUN_LINESTYLES = ("-", "--", "-.", ":")
COMPARE_SUM_STYLE: list[tuple[str, str]] = [
    ("#1a1a1a", "--"),
    ("#5d6d7e", ":"),
    ("#34495e", "-."),
    ("#7d6608", "-"),
]


def _metrics_case_title_label(json_stem: str) -> str:
    """
    从 training_metrics_case14_... 或 ..._case30lite_... 等文件名 stem 中取出 case14 / case30lite，
    用于图总标题；匹配不到则退回整个 stem。
    """
    m = re.search(r"(case\d+(?:lite)?)", json_stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return json_stem


def _parse_training_metrics_filename(
    stem: str,
) -> tuple[str, str | None, str]:
    """
    解析 result/training_metric 下 JSON 的 stem。
    约定：training_metrics_<case>_<YYYYMMDD>_<HHMMSS>.json
         或 training_metrics_<case>_<tag>_<YYYYMMDD>_<HHMMSS>.json
    返回：(算例键如 case14, metrics 标签或 None, 时间戳排序串)。
    """
    if not stem.startswith("training_metrics_"):
        return _metrics_case_title_label(stem), None, stem
    rest = stem[len("training_metrics_") :]
    m_case = re.match(r"^(case\d+(?:lite)?)_(.+)$", rest, re.IGNORECASE)
    if not m_case:
        return _metrics_case_title_label(stem), None, stem
    case_key = m_case.group(1).lower()
    tail = m_case.group(2)
    parts = tail.split("_")
    if (
        len(parts) >= 2
        and re.fullmatch(r"\d{8}", parts[-2])
        and re.fullmatch(r"\d{6}", parts[-1])
    ):
        ts = f"{parts[-2]}_{parts[-1]}"
        tag_parts = parts[:-2]
        tag = "_".join(tag_parts) if tag_parts else None
        return case_key, tag, ts
    return case_key, None, tail


def _metrics_tag_sort_key(tag: str | None) -> tuple[int, str]:
    """对比图例顺序：无标签(本文) → control → 其它字母序。"""
    if tag is None:
        return (0, "")
    tl = str(tag).strip().lower()
    if tl == "control":
        return (1, "control")
    return (2, tl)


def _metrics_run_display_name(tag: str | None) -> str:
    """图例分组名：无 tag→实验组/本文；control→对照组；其它 tag 保留或截断。"""
    if tag is None:
        return "本文" if FIGURE_CAPTIONS_VERBOSE else "实验组"
    tl = str(tag).strip().lower()
    if tl == "control":
        return "对照组"
    if FIGURE_CAPTIONS_VERBOSE:
        return str(tag)
    t = str(tag).strip()
    return t[:8] if len(t) > 8 else t


def _dedupe_latest_per_tag(
    items: list[tuple[Path, str | None, str]],
) -> list[tuple[Path, str | None]]:
    """同一 metrics 标签只保留时间戳最新的一份（时间串可字典序比较）。"""
    best: dict[str, tuple[Path, str | None, str]] = {}
    for p, tag, ts in items:
        k = tag if tag is not None else ""
        if k not in best or ts > best[k][2]:
            best[k] = (p, tag, ts)
    ordered = sorted(best.values(), key=lambda t: _metrics_tag_sort_key(t[1]))
    return [(p, tag) for p, tag, _ in ordered]


def _group_training_metric_files_by_case(
    paths: list[Path],
) -> dict[str, list[tuple[Path, str | None, str]]]:
    g: dict[str, list[tuple[Path, str | None, str]]] = defaultdict(list)
    for p in paths:
        ck, tag, ts = _parse_training_metrics_filename(p.stem)
        g[ck].append((p, tag, ts))
    return dict(g)


def sign4_curriculum_strength(
    p_norm: Any,
    *,
    p_start: float = 0.05,
    p_ramp_end: float = 0.45,
    s_lo: float = 0.1,
    s_hi: float = 2.0,
) -> np.ndarray:
    """
    与论文甘特图中 Sign4 curriculum 一致：进度 p∈[0,1] 上强度 s。
    - p < p_start: s = s_lo
    - p_start ≤ p ≤ p_ramp_end: s 线性 s_lo → s_hi
    - p > p_ramp_end: s = s_hi
    """
    p = np.asarray(p_norm, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    s_out = np.full(p.shape, s_hi, dtype=float)
    lo = p < p_start
    mid = (p >= p_start) & (p <= p_ramp_end)
    s_out[lo] = s_lo
    denom = max(float(p_ramp_end - p_start), 1e-15)
    t = (p[mid] - p_start) / denom
    s_out[mid] = s_lo + t * (s_hi - s_lo)
    return s_out


def sign4_equivalence_factors_for_iters(
    iters: np.ndarray | list[int],
    iter_denominator: float,
    **kwargs: Any,
) -> np.ndarray:
    """等效换算到满强度 s_hi：乘子 factor = s_hi / s(p)，p = iter / iter_denominator。"""
    denom = max(float(iter_denominator), 1.0)
    p = np.asarray(iters, dtype=float) / denom
    s = sign4_curriculum_strength(p, **kwargs)
    s_hi = float(kwargs.get("s_hi", 2.0))
    return s_hi / np.maximum(s, 1e-12)


def _max_surrogate_iter(sections: dict[str, list[dict[str, Any]]]) -> int:
    m = 0
    for rows in sections.values():
        for r in rows:
            m = max(m, int(r["iter"]))
    return m


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _surrogate_sections(metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    surrogate = metrics.get("surrogate")
    if not isinstance(surrogate, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, rows in surrogate.items():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            out[str(gid)] = rows
    return out


def _aggregate_numeric(
    sections: dict[str, list[dict[str, Any]]],
    key: str,
    *,
    rho_weight_surrogate: bool = False,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    by_iter: dict[int, list[float]] = defaultdict(list)
    for rows in sections.values():
        for r in rows:
            val = _surrogate_row_obj_value(r, key, rho_weighted=rho_weight_surrogate)
            if val is None:
                continue
            by_iter[int(r["iter"])].append(val)
    if not by_iter:
        return [], np.array([]), np.array([])
    iters_sorted = sorted(by_iter)
    means = np.array([float(np.mean(by_iter[i])) for i in iters_sorted])
    stds = np.array(
        [
            float(np.std(by_iter[i], ddof=0)) if len(by_iter[i]) > 1 else 0.0
            for i in iters_sorted
        ]
    )
    return iters_sorted, means, stds


def _aggregate_rowwise_sum(
    sections: dict[str, list[dict[str, Any]]],
    keys: list[str],
    *,
    rho_weight_surrogate: bool = False,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """同一机组、同一 iter 上对各 key 逐项求和（可选 ρ 加权），再对全体机组求均值。"""
    by_iter: dict[int, list[float]] = defaultdict(list)
    for rows in sections.values():
        for r in rows:
            s = 0.0
            ok = True
            for k in keys:
                val = _surrogate_row_obj_value(r, k, rho_weighted=rho_weight_surrogate)
                if val is None:
                    ok = False
                    break
                s += val
            if ok:
                by_iter[int(r["iter"])].append(s)
    if not by_iter:
        return [], np.array([]), np.array([])
    iters_sorted = sorted(by_iter)
    means = np.array([float(np.mean(by_iter[i])) for i in iters_sorted])
    stds = np.array(
        [
            float(np.std(by_iter[i], ddof=0)) if len(by_iter[i]) > 1 else 0.0
            for i in iters_sorted
        ]
    )
    return iters_sorted, means, stds


def _sections_flat(metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """顶层 list[dict] 段（bcd、joint 等），与 surrogate 分开。"""
    sections: dict[str, list[dict[str, Any]]] = {}
    for name, value in metrics.items():
        if name == "surrogate":
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            sections[name] = value
    return sections


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _apply_logarithmic_y(ax: plt.Axes, y_arrays: list[np.ndarray]) -> None:
    """y 轴为对数；若存在非正或对零交叉则改用 symlog，以便与 shading 共存。"""
    parts: list[np.ndarray] = []
    for arr in y_arrays:
        if arr is None or len(arr) == 0:
            continue
        parts.append(np.asarray(arr, dtype=float).ravel())
    if not parts:
        return
    stack = np.concatenate(parts)
    stack = stack[np.isfinite(stack)]
    if stack.size == 0:
        return
    pos = stack[stack > 0]
    neg_or_zero = stack[stack <= 0]
    if neg_or_zero.size == 0 and pos.size > 0:
        ax.set_yscale("log")
        return
    nz = stack[np.abs(stack) > 1e-30]
    if nz.size == 0:
        linthresh = 1e-3
    else:
        linthresh = float(
            np.clip(np.percentile(np.abs(nz), 25), 1e-8, 1e6)
        )
        if not np.isfinite(linthresh) or linthresh <= 0:
            linthresh = 1e-3
    ax.set_yscale("symlog", linthresh=linthresh)


def _finalize_y_limits_from_series(
    ax: plt.Axes,
    y_arrays: list[np.ndarray],
    *,
    pad_linear: float = 0.06,
    pad_log_decades: float = 0.18,
) -> None:
    """按已绘制曲线的实际取值收紧 y 轴，减少上下空白；对 log / symlog 分别加边距。"""
    parts: list[np.ndarray] = []
    for arr in y_arrays:
        if arr is None or len(arr) == 0:
            continue
        parts.append(np.asarray(arr, dtype=float).ravel())
    if not parts:
        return
    y = np.concatenate(parts)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return

    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin == ymax:
        eps = max(abs(ymax) * 1e-6, 1e-9)
        ymin -= eps
        ymax += eps

    scale = ax.get_yscale()

    if scale == "log":
        pos = y[y > 0]
        if pos.size == 0:
            return
        lo, hi = float(pos.min()), float(pos.max())
        if lo == hi:
            lo *= 0.5
            hi *= 2.0
        span = hi / lo if hi > lo else 10.0
        decades = float(np.clip(np.log10(span), 0.08, 14.0))
        d = pad_log_decades * (0.5 + decades / max(decades, 1.0))
        bottom = lo * (10.0 ** (-d))
        top = hi * (10.0 ** (d))
        ax.set_ylim(bottom=max(bottom, lo * 1e-30), top=top)
        return

    # linear / symlog
    span = ymax - ymin
    pad = max(span * pad_linear, max(abs(ymin), abs(ymax), 1.0) * 1e-4)
    ax.set_ylim(ymin - pad, ymax + pad)


def _max_iter_across_keys(
    sections: dict[str, list[dict[str, Any]]], keys: list[str]
) -> int:
    m = 0
    for key in keys:
        iters, _, _ = _aggregate_numeric(sections, key)
        if iters:
            m = max(m, max(iters))
    return m


def _add_training_phase_background(
    ax: plt.Axes,
    xmin: float,
    xmax: float,
    formal_start_iter: int,
) -> None:
    """热启动：iter < formal_start_iter；正式训练：iter ≥ formal_start_iter。离散迭代分界取 formal_start_iter−0.5。"""
    ax.set_axisbelow(True)
    split = float(formal_start_iter) - 0.5
    lo = xmin - 0.5
    hi = xmax + 0.5
    warm_c, formal_c = "#fff3cd", "#d4edda"
    if hi <= split:
        ax.axvspan(lo, hi, facecolor=warm_c, alpha=0.45, zorder=0, linewidth=0)
        return
    ax.axvspan(lo, split, facecolor=warm_c, alpha=0.45, zorder=0, linewidth=0)
    ax.axvspan(split, hi, facecolor=formal_c, alpha=0.32, zorder=0, linewidth=0)
    ax.axvline(split, color="#495057", linestyle="--", linewidth=1.1, alpha=0.85, zorder=1)


def _iter_meets_progress_floor(
    iters: Any, progress_denom: float, omit_first_fraction: float
) -> np.ndarray:
    """p = iter/N >= omit_first_fraction 时保留用于绘图。"""
    it = np.asarray(iters)
    if omit_first_fraction <= 0 or progress_denom <= 0:
        return np.ones(it.shape[0], dtype=bool)
    p = np.asarray(it, dtype=float) / float(progress_denom)
    return p >= omit_first_fraction - 1e-15


def _first_iter_after_warm_cut(progress_denom: float, omit_first_fraction: float) -> int | None:
    """与 _iter_meets_progress_floor 一致的首个计入迭代的整数下界（用于竖带右边界）。"""
    if omit_first_fraction <= 0 or progress_denom <= 0:
        return None
    thresh = omit_first_fraction * float(progress_denom)
    fi = int(np.ceil(thresh - 1e-12))
    return max(fi, 0)


def _shade_early_warm_start_zone(
    ax: plt.Axes,
    xmin: float,
    xmax_axes: float,
    progress_denom: float,
    omit_first_fraction: float,
    label: str = "热启动迭代",
) -> None:
    """在未绘曲线的「前 omit 进度」段画斜线区并标注标签。"""
    fi = _first_iter_after_warm_cut(progress_denom, omit_first_fraction)
    if fi is None or fi <= 0:
        return
    x_right = fi - 0.5
    x_left = xmin - 0.5
    if x_right <= x_left or x_right > xmax_axes + 0.6:
        return
    ax.axvspan(
        x_left,
        x_right,
        facecolor="#dde1ea",
        edgecolor="#8b95a8",
        hatch="///",
        linewidth=0.0,
        alpha=0.55,
        zorder=0.12,
    )
    cx = 0.5 * (x_left + x_right)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if FIGURE_CAPTIONS_VERBOSE:
        ax.text(
            cx,
            0.93,
            label + "\n(曲线略)",
            transform=trans,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#444444",
            zorder=5,
        )


def plot_surrogate_aggregate_compare(
    runs: list[tuple[str, dict[str, Any], str | None]],
    output_base: Path,
    title_suffix: str = "",
    *,
    case_title: str,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
    axes_title: str | None = None,
    sign4_rescale: bool = True,
    sign4_iter_denominator: float | None = None,
    sign4_p_start: float = 0.05,
    sign4_p_ramp_end: float = 0.45,
    sign4_s_lo: float = 0.1,
    sign4_s_hi: float = 2.0,
    warm_omit_progress_fraction: float = 0.05,
    rho_weight_surrogate: bool = True,
) -> bool:
    """
    同一算例多份 metrics（如本文 vs 对照组）在一张 surrogate 聚合图上对比。
    runs: (图例名, data, 原始 tag 或 None)，至少 2 条且均需含 surrogate 段。
    """
    obj_keys_sum = list(OBJ_TRIPLE)
    plot_keys = list(SURROGATE_PLOT_LINE_KEYS)
    all_sections: list[tuple[str, dict[str, list[dict[str, Any]]], int]] = []
    xmax = 0.0
    run_idx = 0
    for legend_name, data, _tag in runs:
        sec = _surrogate_sections(data.get("metrics") or {})
        if not sec:
            continue
        xm = _max_iter_across_keys(sec, obj_keys_sum)
        xmax = max(xmax, float(xm))
        all_sections.append((legend_name, sec, run_idx))
        run_idx += 1
    if len(all_sections) < 2:
        return False

    xmin = 0.0
    sign4_denom = float(sign4_iter_denominator) if sign4_iter_denominator is not None else float(max(xmax, 1))
    sign4_kw = {
        "p_start": sign4_p_start,
        "p_ramp_end": sign4_p_ramp_end,
        "s_lo": sign4_s_lo,
        "s_hi": sign4_s_hi,
    }

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.3), squeeze=True)

    agg_main = (
        main_title
        if main_title is not None
        else _default_main_surrogate_aggregate(case_title, title_suffix, compare=True)
    )
    if subtitle is None:
        agg_sub = _surrogate_aggregate_footnote(
            formal_start_iter,
            warm_omit_progress_fraction,
            compare_extra=True,
        )
    else:
        agg_sub = subtitle
    if axes_title is None:
        agg_axes = _default_surrogate_axes_title(dashed_sum_note=False)
    else:
        agg_axes = axes_title

    fig.suptitle(
        agg_main,
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    subtitle_top = 0.91
    plot_top = 0.84
    if agg_sub:
        fig.text(
            0.5,
            subtitle_top,
            agg_sub,
            ha="center",
            fontsize=8.5,
            color="#333333",
        )
    else:
        plot_top = 0.92

    _add_training_phase_background(ax, xmin, xmax, formal_start_iter)
    if warm_omit_progress_fraction > 0:
        _shade_early_warm_start_zone(
            ax, xmin, xmax, sign4_denom, warm_omit_progress_fraction
        )

    y_for_scale: list[np.ndarray] = []
    plotted = False
    for legend_name, sections, run_idx in all_sections:
        mls = COMPARE_RUN_LINESTYLES[run_idx % len(COMPARE_RUN_LINESTYLES)]
        sum_col, sum_ls = COMPARE_SUM_STYLE[run_idx % len(COMPARE_SUM_STYLE)]
        for key in plot_keys:
            iters, mean, _ = _aggregate_numeric(
                sections, key, rho_weight_surrogate=rho_weight_surrogate
            )
            if not iters:
                continue
            mean = np.asarray(mean, dtype=float)
            msk = _iter_meets_progress_floor(iters, sign4_denom, warm_omit_progress_fraction)
            mean = mean[msk]
            it_a = np.asarray(iters, dtype=int)[msk]
            if it_a.size == 0:
                continue
            if sign4_rescale:
                fac = sign4_equivalence_factors_for_iters(
                    it_a, sign4_denom, **sign4_kw
                )
                mean = mean * fac
            col = COMPARE_METRIC_COLORS.get(key, "#333333")
            lbl = (
                f"{_metric_legend_label(key, rho_weighted=rho_weight_surrogate)} "
                f"({legend_name})"
            )
            ax.plot(
                it_a,
                mean,
                linewidth=1.65,
                color=col,
                linestyle=mls,
                label=lbl,
                zorder=3,
            )
            y_for_scale.append(mean)
            plotted = True

        it_sum, mean_sum, _ = _aggregate_rowwise_sum(
            sections, obj_keys_sum, rho_weight_surrogate=rho_weight_surrogate
        )
        if it_sum:
            mean_sum = np.asarray(mean_sum, dtype=float)
            msk = _iter_meets_progress_floor(it_sum, sign4_denom, warm_omit_progress_fraction)
            mean_sum = mean_sum[msk]
            it_sum_a = np.asarray(it_sum, dtype=int)[msk]
            if it_sum_a.size > 0:
                if sign4_rescale:
                    fac = sign4_equivalence_factors_for_iters(
                        it_sum_a, sign4_denom, **sign4_kw
                    )
                    mean_sum = mean_sum * fac
                ax.plot(
                    it_sum_a,
                    mean_sum,
                    linewidth=2.1,
                    color=sum_col,
                    linestyle=sum_ls,
                    label=(
                        f"{_sum_residual_legend(rho_weighted=rho_weight_surrogate)} "
                        f"({legend_name})"
                    ),
                    zorder=4,
                )
                y_for_scale.append(mean_sum)
                plotted = True

    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    if agg_axes:
        ax.set_title(
            agg_axes,
            loc="left",
            fontsize=9,
            color="#333333",
        )
    ax.grid(True, which="both", alpha=0.28, linestyle="--", zorder=1)
    if plotted:
        _apply_logarithmic_y(ax, y_for_scale)
        _finalize_y_limits_from_series(ax, y_for_scale)
        ax.legend(fontsize=6.6, ncol=2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888888")

    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Iteration", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=plot_top, bottom=0.11)

    _save(fig, output_base)
    return True


def plot_flat_metrics_compare(
    runs: list[tuple[str, dict[str, Any], str | None]],
    output_base: Path,
    title_suffix: str = "",
    *,
    case_title: str,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
) -> bool:
    """bcd/joint 等指标多运行对比（线型区分）。"""
    all_flat: list[tuple[str, dict[str, list[dict[str, Any]]], int]] = []
    xmax = 0.0
    run_idx = 0
    for legend_name, data, _tag in runs:
        sections = _sections_flat(data.get("metrics") or {})
        if not sections:
            continue
        xm = float(_max_iter_flat_sections(sections))
        xmax = max(xmax, xm)
        all_flat.append((legend_name, sections, run_idx))
        run_idx += 1
    if len(all_flat) < 2:
        return False

    xmin = 0.0
    metric_groups = [
        ("Objectives", ["obj_primal", "obj_dual", "obj_opt"]),
        ("Dual objective terms", ["obj_dual_pg", "obj_dual_x"]),
        ("Losses", ["nn_loss", "surr_nn_loss"]),
        ("Integrality", ["integrality"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    fm_main = (
        main_title
        if main_title is not None
        else _default_main_flat(case_title, title_suffix, compare=True)
    )
    if subtitle is None:
        fm_sub = _flat_default_subtitle(
            formal_start_iter, 0.0, compare=True
        )
    else:
        fm_sub = subtitle

    fig.suptitle(
        fm_main,
        fontsize=14,
        fontweight="bold",
    )
    layout_top = 0.91
    if fm_sub:
        fig.text(
            0.5,
            0.94,
            fm_sub,
            ha="center",
            fontsize=9,
            color="#333333",
        )
    else:
        layout_top = 0.96

    for ax, (title, keys) in zip(axes.flat, metric_groups):
        _add_training_phase_background(ax, xmin, xmax, formal_start_iter)
        plotted = False
        for legend_name, sections, run_idx in all_flat:
            mls = COMPARE_RUN_LINESTYLES[run_idx % len(COMPARE_RUN_LINESTYLES)]
            for section_name, records in sections.items():
                x_values = [record.get("iter", idx) for idx, record in enumerate(records)]
                for key in keys:
                    y_values = [record.get(key) for record in records]
                    if all(value is None for value in y_values):
                        continue
                    disp = _metric_legend_label(key)
                    ax.plot(
                        x_values,
                        y_values,
                        marker="o",
                        linewidth=1.5,
                        markersize=2.8,
                        linestyle=mls,
                        label=f"{section_name}:{disp} ({legend_name})",
                        zorder=3,
                    )
                    plotted = True
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        if plotted:
            ax.legend(fontsize=6.2)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888888")

    fig.tight_layout(rect=[0, 0, 1, layout_top])
    _save(fig, output_base)
    return True


def plot_per_unit_surrogate_compare(
    runs: list[tuple[str, dict[str, Any], str | None]],
    output_base: Path,
    title_suffix: str = "",
    *,
    case_title: str,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
    sign4_rescale: bool = True,
    sign4_iter_denominator: float | None = None,
    sign4_p_start: float = 0.05,
    sign4_p_ramp_end: float = 0.45,
    sign4_s_lo: float = 0.1,
    sign4_s_hi: float = 2.0,
    warm_omit_progress_fraction: float = 0.05,
    rho_weight_surrogate: bool = True,
) -> bool:
    """按机组分面：多运行 surrogate 对比（仅 r_stat 与残差之和）。"""
    obj_keys_sum = list(OBJ_TRIPLE)
    plot_keys = list(SURROGATE_PLOT_LINE_KEYS)
    prepared: list[tuple[str, dict[str, list[dict[str, Any]]], int]] = []
    n_global = 0
    run_idx = 0
    for legend_name, data, _tag in runs:
        sec = _surrogate_sections(data.get("metrics") or {})
        if not sec:
            continue
        ng = _max_surrogate_iter(sec)
        n_global = max(n_global, ng)
        prepared.append((legend_name, sec, run_idx))
        run_idx += 1
    if len(prepared) < 2:
        return False

    sign4_denom = (
        float(sign4_iter_denominator)
        if sign4_iter_denominator is not None
        else float(max(n_global, 1))
    )
    sign4_kw = {
        "p_start": sign4_p_start,
        "p_ramp_end": sign4_p_ramp_end,
        "s_lo": sign4_s_lo,
        "s_hi": sign4_s_hi,
    }

    gid_sets: list[set[str]] = []
    for _ln, sec, _ri in prepared:
        gid_sets.append(set(sec.keys()))
    union_gids: set[str] = set()
    for s in gid_sets:
        union_gids |= s
    gids = sorted(
        union_gids,
        key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else x),
    )
    if not gids:
        return False

    n = len(gids)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.95 * nrows), squeeze=False)

    pu_main = (
        main_title
        if main_title is not None
        else _default_main_per_unit(case_title, title_suffix, compare=True)
    )
    if subtitle is None:
        pu_sub = _per_unit_default_subtitle(
            formal_start_iter,
            warm_omit_progress_fraction,
            compare=True,
        )
    else:
        pu_sub = subtitle

    fig.suptitle(
        pu_main,
        fontsize=13,
        fontweight="bold",
    )
    pu_layout_top = 0.93
    if pu_sub:
        fig.text(
            0.5,
            0.97,
            pu_sub,
            ha="center",
            fontsize=8.5,
            color="#333333",
        )
    else:
        pu_layout_top = 0.97

    for idx, gid in enumerate(gids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        xmax = 0.0
        for _legend_name, sections, _run_idx in prepared:
            rows = sections.get(gid)
            if not rows:
                continue
            x = [int(rw["iter"]) for rw in rows]
            if x:
                xmax = max(xmax, float(max(x)))
        if xmax <= 0:
            ax.axis("off")
            continue

        _add_training_phase_background(ax, 0.0, xmax, formal_start_iter)
        if warm_omit_progress_fraction > 0:
            _shade_early_warm_start_zone(
                ax, 0.0, xmax, sign4_denom, warm_omit_progress_fraction
            )

        ys_all: list[float] = []
        for legend_name, sections, run_idx in prepared:
            rows = sections.get(gid)
            if not rows:
                continue
            mls = COMPARE_RUN_LINESTYLES[run_idx % len(COMPARE_RUN_LINESTYLES)]
            sum_col, sum_ls = COMPARE_SUM_STYLE[run_idx % len(COMPARE_SUM_STYLE)]

            x = [int(rw["iter"]) for rw in rows]
            for k in plot_keys:
                y = [
                    _surrogate_row_obj_value(
                        rw, k, rho_weighted=rho_weight_surrogate
                    )
                    for rw in rows
                ]
                if all(v is None for v in y):
                    continue
                if any(v is None for v in y):
                    continue
                ya = np.asarray([float(v) for v in y], dtype=float)
                msk = _iter_meets_progress_floor(x, sign4_denom, warm_omit_progress_fraction)
                ya = ya[msk]
                x_a = np.asarray(x, dtype=int)[msk]
                if x_a.size == 0:
                    continue
                if sign4_rescale:
                    fac = sign4_equivalence_factors_for_iters(x_a, sign4_denom, **sign4_kw)
                    ya = ya * fac
                col = COMPARE_METRIC_COLORS.get(k, "#333333")
                ax.plot(
                    x_a,
                    ya,
                    linewidth=1.35,
                    color=col,
                    linestyle=mls,
                    label=(
                        f"{_metric_legend_label(k, rho_weighted=rho_weight_surrogate)} "
                        f"({legend_name})"
                    ),
                    zorder=3,
                )
                ys_all.extend(ya.tolist())

            x_sum: list[int] = []
            y_sum_row: list[float] = []
            for rw in rows:
                terms = [
                    _surrogate_row_obj_value(
                        rw, k_term, rho_weighted=rho_weight_surrogate,
                    )
                    for k_term in obj_keys_sum
                ]
                if any(t is None for t in terms):
                    continue
                x_sum.append(int(rw["iter"]))
                y_sum_row.append(sum(float(t) for t in terms))
            if x_sum:
                ysr = np.asarray(y_sum_row, dtype=float)
                msum = _iter_meets_progress_floor(x_sum, sign4_denom, warm_omit_progress_fraction)
                ysr = ysr[msum]
                xa = np.asarray(x_sum, dtype=int)[msum]
                if xa.size > 0:
                    if sign4_rescale:
                        fac = sign4_equivalence_factors_for_iters(xa, sign4_denom, **sign4_kw)
                        ysr = ysr * fac
                    ax.plot(
                        xa,
                        ysr,
                        linewidth=1.85,
                        color=sum_col,
                        linestyle=sum_ls,
                        label=(
                            f"{_sum_residual_legend(rho_weighted=rho_weight_surrogate)} "
                            f"({legend_name})"
                        ),
                        zorder=4,
                    )
                    ys_all.extend(ysr.tolist())

        ax.set_title(f"Unit {gid}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_xlim(-0.5, xmax + 0.5)
        if ys_all:
            ylo, yhi = min(ys_all), max(ys_all)
            if ylo == yhi:
                eps = max(abs(yhi) * 1e-6, 1e-9)
                ylo -= eps
                yhi += eps
            ypad = max((yhi - ylo) * 0.06, max(abs(ylo), abs(yhi), 1.0) * 1e-4)
            ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        ax.legend(fontsize=5.3, loc="best")

    for j in range(len(gids), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, pu_layout_top])
    _save(fig, output_base)
    return True


def plot_surrogate_aggregate(
    data: dict[str, Any],
    output_base: Path,
    title_suffix: str = "",
    *,
    metrics_json_stem: str | None = None,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
    axes_title: str | None = None,
    sign4_rescale: bool = True,
    sign4_iter_denominator: float | None = None,
    sign4_p_start: float = 0.05,
    sign4_p_ramp_end: float = 0.45,
    sign4_s_lo: float = 0.1,
    sign4_s_hi: float = 2.0,
    warm_omit_progress_fraction: float = 0.05,
    rho_weight_surrogate: bool = True,
) -> bool:
    """子问题 surrogate：r_stat 与残差之和（三项按机组相加后再取机组均值）；对数 y。"""
    metrics = data.get("metrics") or {}
    sections = _surrogate_sections(metrics)
    if not sections:
        return False

    obj_keys_sum = list(OBJ_TRIPLE)
    plot_keys = list(SURROGATE_PLOT_LINE_KEYS)
    all_keys = list(obj_keys_sum)
    xmax = _max_iter_across_keys(sections, all_keys)
    xmin = 0.0
    sign4_denom = float(sign4_iter_denominator) if sign4_iter_denominator is not None else float(max(xmax, 1))
    sign4_kw = {
        "p_start": sign4_p_start,
        "p_ramp_end": sign4_p_ramp_end,
        "s_lo": sign4_s_lo,
        "s_hi": sign4_s_hi,
    }

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.0), squeeze=True)
    title_core = (
        _metrics_case_title_label(metrics_json_stem)
        if metrics_json_stem
        else output_base.stem
    )

    agg_main = (
        main_title
        if main_title is not None
        else _default_main_surrogate_aggregate(title_core, title_suffix, compare=False)
    )
    if subtitle is None:
        agg_sub = _surrogate_aggregate_footnote(
            formal_start_iter,
            warm_omit_progress_fraction,
            compare_extra=False,
        )
    else:
        agg_sub = subtitle
    if axes_title is None:
        agg_axes = _default_surrogate_axes_title(dashed_sum_note=True)
    else:
        agg_axes = axes_title

    fig.suptitle(
        agg_main,
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plot_top_single = 0.86
    if agg_sub:
        fig.text(
            0.5,
            0.92,
            agg_sub,
            ha="center",
            fontsize=9,
            color="#333333",
        )
    else:
        plot_top_single = 0.94

    _add_training_phase_background(ax, xmin, xmax, formal_start_iter)
    if warm_omit_progress_fraction > 0:
        _shade_early_warm_start_zone(
            ax, xmin, xmax, sign4_denom, warm_omit_progress_fraction
        )

    y_for_scale: list[np.ndarray] = []
    plotted = False
    for key in plot_keys:
        iters, mean, _ = _aggregate_numeric(
            sections, key, rho_weight_surrogate=rho_weight_surrogate
        )
        if not iters:
            continue
        mean = np.asarray(mean, dtype=float)
        msk = _iter_meets_progress_floor(iters, sign4_denom, warm_omit_progress_fraction)
        mean = mean[msk]
        it_a = np.asarray(iters, dtype=int)[msk]
        if it_a.size == 0:
            continue
        if sign4_rescale:
            fac = sign4_equivalence_factors_for_iters(
                it_a, sign4_denom, **sign4_kw
            )
            mean = mean * fac
        lbl = _metric_legend_label(key, rho_weighted=rho_weight_surrogate)
        ax.plot(it_a, mean, linewidth=1.65, label=lbl, zorder=3)
        y_for_scale.append(mean)
        plotted = True

    it_sum, mean_sum, _ = _aggregate_rowwise_sum(
        sections, obj_keys_sum, rho_weight_surrogate=rho_weight_surrogate
    )
    if it_sum:
        mean_sum = np.asarray(mean_sum, dtype=float)
        msk = _iter_meets_progress_floor(it_sum, sign4_denom, warm_omit_progress_fraction)
        mean_sum = mean_sum[msk]
        it_sum_a = np.asarray(it_sum, dtype=int)[msk]
        if it_sum_a.size == 0:
            pass
        else:
            if sign4_rescale:
                fac = sign4_equivalence_factors_for_iters(
                    it_sum_a, sign4_denom, **sign4_kw
                )
                mean_sum = mean_sum * fac
            ax.plot(
                it_sum_a,
                mean_sum,
                linewidth=2.2,
                color="#1f1f1f",
                linestyle="--",
                label=_sum_residual_legend(rho_weighted=rho_weight_surrogate),
                zorder=4,
            )
            y_for_scale.append(mean_sum)
            plotted = True

    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    if agg_axes:
        ax.set_title(
            agg_axes,
            loc="left",
            fontsize=9,
            color="#333333",
        )
    ax.grid(True, which="both", alpha=0.28, linestyle="--", zorder=1)
    if plotted:
        _apply_logarithmic_y(ax, y_for_scale)
        _finalize_y_limits_from_series(ax, y_for_scale)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888888")

    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Iteration", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=plot_top_single, bottom=0.11)

    _save(fig, output_base)
    return True


def _max_iter_flat_sections(sections: dict[str, list[dict[str, Any]]]) -> int:
    m = 0
    for records in sections.values():
        for idx, rec in enumerate(records):
            m = max(m, int(rec.get("iter", idx)))
    return m


def plot_flat_metrics(
    data: dict[str, Any],
    output_base: Path,
    title_suffix: str = "",
    *,
    metrics_json_stem: str | None = None,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
) -> bool:
    """绘制 metrics 顶层 list 段（如 bcd、joint），每个段一张 2×2 图（与 plot_training_metrics 风格一致）。"""
    metrics = data.get("metrics") or {}
    sections = _sections_flat(metrics)
    if not sections:
        return False

    stem = output_base.stem
    title_core = (
        _metrics_case_title_label(metrics_json_stem)
        if metrics_json_stem
        else stem
    )

    metric_groups = [
        ("Objectives", ["obj_primal", "obj_dual", "obj_opt"]),
        ("Dual objective terms", ["obj_dual_pg", "obj_dual_x"]),
        ("Losses", ["nn_loss", "surr_nn_loss"]),
        ("Integrality", ["integrality"]),
    ]
    xmax = float(_max_iter_flat_sections(sections))
    xmin = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    fm_main = (
        main_title
        if main_title is not None
        else _default_main_flat(title_core, title_suffix, compare=False)
    )
    if subtitle is None:
        fm_sub = _flat_default_subtitle(
            formal_start_iter, 0.0, compare=False
        )
    else:
        fm_sub = subtitle

    fig.suptitle(
        fm_main,
        fontsize=14,
        fontweight="bold",
    )
    flat_layout_top = 0.92
    if fm_sub:
        fig.text(
            0.5,
            0.94,
            fm_sub,
            ha="center",
            fontsize=9,
            color="#333333",
        )
    else:
        flat_layout_top = 0.97

    for ax, (title, keys) in zip(axes.flat, metric_groups):
        _add_training_phase_background(ax, xmin, xmax, formal_start_iter)
        plotted = False
        for section_name, records in sections.items():
            x_values = [record.get("iter", idx) for idx, record in enumerate(records)]
            for key in keys:
                y_values = [record.get(key) for record in records]
                if all(value is None for value in y_values):
                    continue
                disp = _metric_legend_label(key)
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.6,
                    markersize=3,
                    label=f"{section_name}:{disp}",
                    zorder=3,
                )
                plotted = True
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888888")

    fig.tight_layout(rect=[0, 0, 1, flat_layout_top])
    _save(fig, output_base)
    return True


def plot_per_unit_surrogate(
    data: dict[str, Any],
    output_base: Path,
    title_suffix: str = "",
    *,
    metrics_json_stem: str | None = None,
    formal_start_iter: int = 20,
    main_title: str | None = None,
    subtitle: str | None = None,
    sign4_rescale: bool = True,
    sign4_iter_denominator: float | None = None,
    sign4_p_start: float = 0.05,
    sign4_p_ramp_end: float = 0.45,
    sign4_s_lo: float = 0.1,
    sign4_s_hi: float = 2.0,
    warm_omit_progress_fraction: float = 0.05,
    rho_weight_surrogate: bool = True,
) -> bool:
    """每个机组：r_stat 与同迭代下三项残差之和。"""
    metrics = data.get("metrics") or {}
    sections = _surrogate_sections(metrics)
    if not sections:
        return False

    obj_keys_sum = list(OBJ_TRIPLE)
    plot_keys = list(SURROGATE_PLOT_LINE_KEYS)

    gids = sorted(sections.keys(), key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else x))
    n_global = _max_surrogate_iter(sections)
    sign4_denom = (
        float(sign4_iter_denominator)
        if sign4_iter_denominator is not None
        else float(max(n_global, 1))
    )
    sign4_kw = {
        "p_start": sign4_p_start,
        "p_ramp_end": sign4_p_ramp_end,
        "s_lo": sign4_s_lo,
        "s_hi": sign4_s_hi,
    }

    n = len(gids)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    title_core = (
        _metrics_case_title_label(metrics_json_stem)
        if metrics_json_stem
        else output_base.stem
    )

    pu_main = (
        main_title
        if main_title is not None
        else _default_main_per_unit(title_core, title_suffix, compare=False)
    )
    if subtitle is None:
        pu_sub = _per_unit_default_subtitle(
            formal_start_iter,
            warm_omit_progress_fraction,
            compare=False,
        )
    else:
        pu_sub = subtitle

    fig.suptitle(
        pu_main,
        fontsize=13,
        fontweight="bold",
    )
    pu_single_layout_top = 0.94
    if pu_sub:
        fig.text(
            0.5,
            0.97,
            pu_sub,
            ha="center",
            fontsize=8.5,
            color="#333333",
        )
    else:
        pu_single_layout_top = 0.98

    for idx, gid in enumerate(gids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        rows = sections[gid]
        x = [int(rw["iter"]) for rw in rows]
        xmax = float(max(x)) if x else 0.0
        _add_training_phase_background(ax, 0.0, xmax, formal_start_iter)
        if warm_omit_progress_fraction > 0:
            _shade_early_warm_start_zone(
                ax, 0.0, xmax, sign4_denom, warm_omit_progress_fraction
            )
        for k in plot_keys:
            y = [
                _surrogate_row_obj_value(
                    rw, k, rho_weighted=rho_weight_surrogate
                )
                for rw in rows
            ]
            if all(v is None for v in y):
                continue
            if any(v is None for v in y):
                continue
            ya = np.asarray([float(v) for v in y], dtype=float)
            msk = _iter_meets_progress_floor(x, sign4_denom, warm_omit_progress_fraction)
            ya = ya[msk]
            x_a = np.asarray(x, dtype=int)[msk]
            if x_a.size == 0:
                continue
            if sign4_rescale:
                fac = sign4_equivalence_factors_for_iters(x_a, sign4_denom, **sign4_kw)
                ya = ya * fac
            ax.plot(
                x_a,
                ya,
                linewidth=1.45,
                label=_metric_legend_label(k, rho_weighted=rho_weight_surrogate),
                zorder=3,
            )

        x_sum: list[int] = []
        y_sum_row: list[float] = []
        for rw in rows:
            terms = [
                _surrogate_row_obj_value(
                    rw, k_term, rho_weighted=rho_weight_surrogate,
                )
                for k_term in obj_keys_sum
            ]
            if any(t is None for t in terms):
                continue
            x_sum.append(int(rw["iter"]))
            y_sum_row.append(sum(float(t) for t in terms))
        if x_sum:
            ysr = np.asarray(y_sum_row, dtype=float)
            msum = _iter_meets_progress_floor(x_sum, sign4_denom, warm_omit_progress_fraction)
            ysr = ysr[msum]
            xa = np.asarray(x_sum, dtype=int)[msum]
            if xa.size > 0:
                if sign4_rescale:
                    fac = sign4_equivalence_factors_for_iters(xa, sign4_denom, **sign4_kw)
                    ysr = ysr * fac
                ax.plot(
                    xa,
                    ysr,
                    linewidth=2.0,
                    linestyle="--",
                    color="#1f1f1f",
                    label=_sum_residual_legend(
                        rho_weighted=rho_weight_surrogate
                    ),
                    zorder=4,
                )
        ax.set_title(f"Unit {gid}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_xlim(-0.5, xmax + 0.5)
        ys_all: list[float] = []
        m_row = _iter_meets_progress_floor(x, sign4_denom, warm_omit_progress_fraction)
        for i, rw in enumerate(rows):
            if i >= len(m_row) or not bool(m_row[i]):
                continue
            for kk in plot_keys:
                v = rw.get(kk)
                if v is None:
                    continue
                fv = float(v)
                if sign4_rescale:
                    it = int(rw["iter"])
                    fv *= float(
                        sign4_equivalence_factors_for_iters(
                            [it], sign4_denom, **sign4_kw
                        )[0]
                    )
                ys_all.append(fv)
        if x_sum:
            msum = _iter_meets_progress_floor(
                x_sum, sign4_denom, warm_omit_progress_fraction
            )
            facm = (
                sign4_equivalence_factors_for_iters(
                    np.asarray(x_sum, dtype=int), sign4_denom, **sign4_kw
                )
                if sign4_rescale
                else None
            )
            for i, t in enumerate(y_sum_row):
                if i >= len(msum) or not bool(msum[i]):
                    continue
                val = float(t)
                if facm is not None:
                    val *= float(facm[i])
                ys_all.append(val)
        if ys_all:
            ylo, yhi = min(ys_all), max(ys_all)
            if ylo == yhi:
                eps = max(abs(yhi) * 1e-6, 1e-9)
                ylo -= eps
                yhi += eps
            ypad = max((yhi - ylo) * 0.06, max(abs(ylo), abs(yhi), 1.0) * 1e-4)
            ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        ax.legend(fontsize=5.8, loc="best")

    for j in range(len(gids), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, pu_single_layout_top])
    _save(fig, output_base)
    return True


def _iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".json" else []
    return sorted(path.glob("training_metrics_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot iteration curves from training_metrics JSON (surrogate nested or flat bcd/joint)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("result/training_metric"),
        help="JSON 文件或目录（默认: result/training_metric）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/figures/training_metric"),
        help="输出 PNG/PDF 目录",
    )
    parser.add_argument(
        "--per-unit",
        action="store_true",
        help="额外生成按机组分面的 surrogate 曲线图（对比模式下亦生成 per-unit 对比页）",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="关闭按算例合并（本文 vs 对照等）：始终为每个 JSON 单独出图。",
    )
    parser.add_argument(
        "--formal-start-iter",
        type=int,
        default=20,
        help="正式训练起始迭代（默认 20）：iter 小于此值为热启动，大于等于为正式训练。",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="中文字体文件 (.ttf/.ttc/.otf)，优先注册；也可用环境变量 MPL_CJK_FONT_PATH。",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        metavar="STR",
        help=(
            "接在默认总标题中「算例简称」后的额外文字。"
            "默认算例简称从 JSON 文件名解析（如 case14、case30lite）；"
            "使用 --aggregate-main-title 等覆盖总标题时此项仍被忽略。"
        ),
    )
    parser.add_argument(
        "--aggregate-main-title",
        type=str,
        default=None,
        metavar="TITLE",
        help="聚合图 fig.suptitle（不传则：子模型典型残差迭代: <case标签> + --title-suffix）。",
    )
    parser.add_argument(
        "--aggregate-subtitle",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "聚合图标题下一行说明；不传为热启动/正式训练文案。"
            '传空字符串表示不绘制此行： PowerShell 用 --aggregate-subtitle "" '
        ),
    )
    parser.add_argument(
        "--aggregate-axes-title",
        type=str,
        default=None,
        metavar="TEXT",
        help="聚合图 axes 顶部说明；不传为分项/总和默认文案。传空字符串则不绘制 axes 标题。",
    )
    parser.add_argument(
        "--flat-main-title",
        type=str,
        default=None,
        metavar="TITLE",
        help="flat_sections 总标题覆盖。",
    )
    parser.add_argument(
        "--per-unit-main-title",
        type=str,
        default=None,
        metavar="TITLE",
        help="per-unit 总标题覆盖。",
    )
    parser.add_argument(
        "--no-sign4-rescale-obj",
        action="store_true",
        help=(
            "关闭 Sign4 等效缩放（默认开启：y'=y*(s_hi/s(p))，与甘特图中 s=0.1→2.0 约定一致）。"
        ),
    )
    parser.add_argument(
        "--sign4-iter-denom",
        type=float,
        default=None,
        metavar="N",
        help="归一化进度分母（默认 surrogate 中出现的最大 iter）。",
    )
    parser.add_argument("--sign4-p-start", type=float, default=0.05)
    parser.add_argument("--sign4-p-ramp-end", type=float, default=0.45)
    parser.add_argument("--sign4-s-lo", type=float, default=0.1)
    parser.add_argument("--sign4-s-hi", type=float, default=2.0)
    parser.add_argument(
        "--verbose-figure-captions",
        action="store_true",
        help="恢复长标题、脚注与轴说明（默认少字、偏论文排版）。",
    )
    parser.add_argument(
        "--warm-omit-fract",
        type=float,
        default=0.05,
        metavar="F",
        help=(
            "不绘制 p=iter/N < F 的残差点（N=max iter 或与 --sign4-iter-denom 一致）；"
            "左侧斜线区标注「热启动迭代」。设为 0 关闭。"
        ),
    )
    args = parser.parse_args()

    global FIGURE_CAPTIONS_VERBOSE
    FIGURE_CAPTIONS_VERBOSE = bool(args.verbose_figure_captions)

    if args.font_path is not None:
        chosen = configure_matplotlib_cjk_font(args.font_path)
        print(f"[font] {chosen} (explicit)")
    else:
        print(f"[font] {plt.rcParams['font.sans-serif'][0]}")

    agg_sub_kw: str | None = None
    if args.aggregate_subtitle is not None:
        agg_sub_kw = args.aggregate_subtitle
    agg_axes_kw: str | None = None
    if args.aggregate_axes_title is not None:
        agg_axes_kw = args.aggregate_axes_title

    root = args.input_path.resolve()
    files = _iter_json_files(root)
    if not files:
        raise SystemExit(f"No training_metrics_*.json under: {root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_case = _group_training_metric_files_by_case(files)
    used_surrogate: set[Path] = set()
    used_flat: set[Path] = set()
    used_per_unit: set[Path] = set()

    if not args.no_combine:
        for case_key, items in sorted(by_case.items()):
            picked = _dedupe_latest_per_tag(items)
            if len(picked) < 2:
                continue
            loaded = [(p, tag, _load_json(p)) for p, tag in picked]

            sur_loaded = [
                (p, t, d)
                for p, t, d in loaded
                if _surrogate_sections(d.get("metrics") or {})
            ]
            flat_loaded = [
                (p, t, d)
                for p, t, d in loaded
                if _sections_flat(d.get("metrics") or {})
            ]

            if len(sur_loaded) >= 2:
                if not args.no_sign4_rescale_obj:
                    xm_all = 0
                    for _p, _t, d in sur_loaded:
                        xm_all = max(
                            xm_all,
                            _max_surrogate_iter(
                                _surrogate_sections(d.get("metrics") or {})
                            ),
                        )
                    d_eff = (
                        float(args.sign4_iter_denom)
                        if args.sign4_iter_denom is not None
                        else float(max(xm_all, 1))
                    )
                    print(
                        f"[sign4] rescale: denom={d_eff}, "
                        f"p ramp [{args.sign4_p_start:.2f},{args.sign4_p_ramp_end:.2f}], "
                        f"s {args.sign4_s_lo}->{args.sign4_s_hi} (compare)"
                    )
                sur_runs = [
                    (_metrics_run_display_name(t), d, t) for p, t, d in sur_loaded
                ]
                out_cmp = (
                    args.output_dir
                    / f"training_metrics_{case_key}_compare_surrogate_aggregate"
                )
                wcmp = plot_surrogate_aggregate_compare(
                    sur_runs,
                    out_cmp,
                    title_suffix=args.title_suffix,
                    case_title=case_key,
                    formal_start_iter=args.formal_start_iter,
                    main_title=args.aggregate_main_title,
                    subtitle=agg_sub_kw,
                    axes_title=agg_axes_kw,
                    sign4_rescale=not args.no_sign4_rescale_obj,
                    sign4_iter_denominator=args.sign4_iter_denom,
                    sign4_p_start=args.sign4_p_start,
                    sign4_p_ramp_end=args.sign4_p_ramp_end,
                    sign4_s_lo=args.sign4_s_lo,
                    sign4_s_hi=args.sign4_s_hi,
                    warm_omit_progress_fraction=args.warm_omit_fract,
                )
                for p, _t, _d in sur_loaded:
                    used_surrogate.add(p)
                print(
                    f"[compare] case={case_key} surrogate_aggregate: "
                    f"{'OK' if wcmp else 'skip'} -> {out_cmp.with_suffix('.png')}"
                )

            if len(flat_loaded) >= 2:
                flat_runs = [
                    (_metrics_run_display_name(t), d, t) for p, t, d in flat_loaded
                ]
                out_f = (
                    args.output_dir / f"training_metrics_{case_key}_compare_flat_sections"
                )
                wf = plot_flat_metrics_compare(
                    flat_runs,
                    out_f,
                    title_suffix=args.title_suffix,
                    case_title=case_key,
                    formal_start_iter=args.formal_start_iter,
                    main_title=args.flat_main_title,
                )
                for p, _t, _d in flat_loaded:
                    used_flat.add(p)
                print(
                    f"[compare] case={case_key} flat_sections: "
                    f"{'OK' if wf else 'skip'} -> {out_f.with_suffix('.png')}"
                )

            if args.per_unit and len(sur_loaded) >= 2:
                pu_runs = [
                    (_metrics_run_display_name(t), d, t) for p, t, d in sur_loaded
                ]
                out_u = (
                    args.output_dir
                    / f"training_metrics_{case_key}_compare_surrogate_per_unit"
                )
                wu = plot_per_unit_surrogate_compare(
                    pu_runs,
                    out_u,
                    title_suffix=args.title_suffix,
                    case_title=case_key,
                    formal_start_iter=args.formal_start_iter,
                    main_title=args.per_unit_main_title,
                    sign4_rescale=not args.no_sign4_rescale_obj,
                    sign4_iter_denominator=args.sign4_iter_denom,
                    sign4_p_start=args.sign4_p_start,
                    sign4_p_ramp_end=args.sign4_p_ramp_end,
                    sign4_s_lo=args.sign4_s_lo,
                    sign4_s_hi=args.sign4_s_hi,
                    warm_omit_progress_fraction=args.warm_omit_fract,
                )
                for p, _t, _d in sur_loaded:
                    used_per_unit.add(p)
                print(
                    f"[compare] case={case_key} per_unit: "
                    f"{'OK' if wu else 'skip'} -> {out_u.with_suffix('.png')}"
                )

    for jp in files:
        data = _load_json(jp)
        stem = jp.stem
        out_agg = args.output_dir / f"{stem}_surrogate_aggregate"
        out_flat = args.output_dir / f"{stem}_flat_sections"
        out_unit = args.output_dir / f"{stem}_surrogate_per_unit"

        sign4_den = args.sign4_iter_denom
        use_sign4 = not args.no_sign4_rescale_obj
        if use_sign4 and jp not in used_surrogate:
            xm = _max_surrogate_iter(_surrogate_sections(data.get("metrics") or {}))
            d_eff = float(sign4_den) if sign4_den is not None else float(max(xm, 1))
            print(
                f"[sign4] rescale: denom={d_eff}, "
                f"p ramp [{args.sign4_p_start:.2f},{args.sign4_p_ramp_end:.2f}], "
                f"s {args.sign4_s_lo}->{args.sign4_s_hi}"
            )

        if jp not in used_surrogate:
            w1 = plot_surrogate_aggregate(
                data,
                out_agg,
                title_suffix=args.title_suffix,
                metrics_json_stem=stem,
                formal_start_iter=args.formal_start_iter,
                main_title=args.aggregate_main_title,
                subtitle=agg_sub_kw,
                axes_title=agg_axes_kw,
                sign4_rescale=use_sign4,
                sign4_iter_denominator=sign4_den,
                sign4_p_start=args.sign4_p_start,
                sign4_p_ramp_end=args.sign4_p_ramp_end,
                sign4_s_lo=args.sign4_s_lo,
                sign4_s_hi=args.sign4_s_hi,
                warm_omit_progress_fraction=args.warm_omit_fract,
            )
        else:
            w1 = False

        if jp not in used_flat:
            w2 = plot_flat_metrics(
                data,
                out_flat,
                title_suffix=args.title_suffix,
                metrics_json_stem=stem,
                formal_start_iter=args.formal_start_iter,
                main_title=args.flat_main_title,
            )
        else:
            w2 = False

        if args.per_unit:
            if jp not in used_per_unit:
                w3 = plot_per_unit_surrogate(
                    data,
                    out_unit,
                    title_suffix=args.title_suffix,
                    metrics_json_stem=stem,
                    formal_start_iter=args.formal_start_iter,
                    main_title=args.per_unit_main_title,
                    sign4_rescale=use_sign4,
                    sign4_iter_denominator=sign4_den,
                    sign4_p_start=args.sign4_p_start,
                    sign4_p_ramp_end=args.sign4_p_ramp_end,
                    sign4_s_lo=args.sign4_s_lo,
                    sign4_s_hi=args.sign4_s_hi,
                    warm_omit_progress_fraction=args.warm_omit_fract,
                )
            else:
                w3 = False
        else:
            w3 = False

        print(f"json: {jp}")
        sk1 = " (skip→已对比)" if jp in used_surrogate else ""
        sk2 = " (skip→已对比)" if jp in used_flat else ""
        print(
            f"  surrogate_aggregate: {'OK' if w1 else '(skip/no surrogate)'}{sk1} -> "
            f"{out_agg.with_suffix('.png')}"
        )
        print(
            f"  flat_sections:       {'OK' if w2 else '(skip/empty bcdjoint)'}{sk2} -> "
            f"{out_flat.with_suffix('.png')}"
        )
        if args.per_unit:
            sk3 = " (skip→已对比)" if jp in used_per_unit else ""
            print(
                f"  per_unit:             {'OK' if w3 else '(skip)'}{sk3} -> "
                f"{out_unit.with_suffix('.png')}"
            )


if __name__ == "__main__":
    main()
