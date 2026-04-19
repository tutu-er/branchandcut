#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练可视化模块，生成 publication-quality 科研图。

风格参考 IEEE / Nature 系列论文：
- serif 字体（Times New Roman 回退 DejaVu Serif）
- 内向刻度线、薄脊线
- 3-4 色精选配色
- 无多余装饰
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from training_logger import TrainingLogger

# ====================================================================== #
#  全局科研绘图风格
# ====================================================================== #
_STYLE_APPLIED = False


def _apply_science_style() -> None:
    """一次性设置全局 rcParams，模拟 SciencePlots 风格。"""
    global _STYLE_APPLIED
    if _STYLE_APPLIED or not MATPLOTLIB_AVAILABLE:
        return
    _STYLE_APPLIED = True

    # 字体：优先 Times New Roman（Windows 自带），回退 DejaVu Serif
    _serif = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams.update({
        # --- 字体 ---
        "font.family": "serif",
        "font.serif": _serif,
        "mathtext.fontset": "stix",  # 数学公式用 STIX（接近 Times）
        # --- 字号 ---
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        # --- 线条 ---
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        # --- 刻度 ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.top": True,
        "ytick.right": True,
        # --- 脊线 ---
        "axes.linewidth": 0.6,
        # --- 网格 ---
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        # --- 图例 ---
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        # --- 输出 ---
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 150,
        # --- 负号 ---
        "axes.unicode_minus": False,
    })

    if SEABORN_AVAILABLE:
        sns.set_style("ticks", {
            "axes.grid": True,
            "grid.color": "#d0d0d0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.4,
        })


# ====================================================================== #
#  配色方案
# ====================================================================== #
# 精选 4 色：来自 Tableau 10 / Okabe-Ito 色盲友好配色
_C = {
    "blue":   "#2166ac",
    "red":    "#d6604d",
    "green":  "#4daf4a",
    "orange": "#e08214",
    "purple": "#7b3294",
    "gray":   "#636363",
}
_PALETTE_3 = [_C["blue"], _C["red"], _C["green"]]
_PALETTE_4 = [_C["blue"], _C["red"], _C["green"], _C["orange"]]

# 热力图用连续色板（白→深蓝）
_HEATMAP_CMAP = "YlGnBu"
# 违反量热力图（白→红）
_VIOL_CMAP = "OrRd"


def _safe_log(arr: np.ndarray) -> np.ndarray:
    """安全 log10，避免 log(0)。"""
    return np.log10(np.maximum(arr, 1e-15))


# ====================================================================== #
#  TrainingVisualizer
# ====================================================================== #
class TrainingVisualizer:
    """基于 TrainingLogger 的科研绘图模块。

    Args:
        logger: 已收集指标的 TrainingLogger 实例。
    """

    DPI = 300
    FIGSIZE_SINGLE = (5.5, 3.8)      # 单栏宽度
    FIGSIZE_WIDE = (7.2, 3.8)        # 双栏宽度
    FIGSIZE_DOUBLE = (5.5, 6.5)      # 单栏双行

    def __init__(self, logger: TrainingLogger) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib not available")
        _apply_science_style()
        self.logger = logger

    def _savefig(self, fig: plt.Figure, save_dir: Path, name: str) -> None:
        for ext in ("pdf", "png"):
            fig.savefig(save_dir / f"{name}.{ext}", dpi=self.DPI, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _despine(ax: plt.Axes) -> None:
        """移除右侧和顶部脊线（保留刻度）。"""
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(top=False, right=False)

    # ------------------------------------------------------------------ #
    #  1. 收敛曲线（违反量 + rho）
    # ------------------------------------------------------------------ #
    def plot_convergence(self, stage: str, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        """2x1: 违反量收敛 (log scale) + penalty rho 演化。"""
        data = self._get_stage_data(stage)
        if not data:
            return None

        iters = np.array([d["iter"] for d in data]) + 1  # 1-indexed

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.FIGSIZE_DOUBLE, sharex=True,
            gridspec_kw={"hspace": 0.12})

        labels_v = ["Primal", "Dual", "Optimality"]
        keys_v = ["obj_primal", "obj_dual", "obj_opt"]
        for key, label, color in zip(keys_v, labels_v, _PALETTE_3):
            vals = np.array([d[key] for d in data])
            ax1.plot(iters, _safe_log(vals), color=color, label=label)
        ax1.set_ylabel(r"$\log_{10}$(violation)")
        ax1.legend(loc="upper right")
        self._despine(ax1)

        labels_r = [r"$\rho_{\mathrm{prim}}$", r"$\rho_{\mathrm{dual}}$", r"$\rho_{\mathrm{opt}}$"]
        keys_r = ["rho_primal", "rho_dual", "rho_opt"]
        for key, label, color in zip(keys_r, labels_r, _PALETTE_3):
            vals = [d[key] for d in data]
            ax2.plot(iters, vals, color=color, label=label)
        ax2.set_ylabel("Penalty parameter")
        ax2.set_xlabel("Iteration")
        ax2.legend(loc="upper left")
        self._despine(ax2)

        stage_title = {"bcd": "BCD", "joint": "Joint"}.get(stage, stage.upper())
        fig.suptitle(f"{stage_title} Training Convergence", fontsize=11, y=0.98)

        if save_dir:
            self._savefig(fig, save_dir, f"convergence_{stage}")
        return fig

    # ------------------------------------------------------------------ #
    #  2. 整数性
    # ------------------------------------------------------------------ #
    def plot_integrality(self, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        data = self.logger.metrics["joint"]
        if not data or "integrality" not in data[0]:
            return None

        iters = np.array([d["iter"] for d in data]) + 1
        vals = [d["integrality"] for d in data]

        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)
        ax.plot(iters, vals, "o-", color=_C["red"], markersize=3.5)
        ax.fill_between(iters, 0, vals, color=_C["red"], alpha=0.08)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\mathrm{avg}\;x_i(1-x_i)$")
        ax.set_ylim(bottom=0)
        ax.set_title("Integrality Gap")
        self._despine(ax)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, "integrality")
        return fig

    # ------------------------------------------------------------------ #
    #  3. x 热力图
    # ------------------------------------------------------------------ #
    def plot_x_heatmap(self, stage: str, iter: int, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        snap = self._find_snapshot("x", stage, iter)
        if snap is None:
            return None
        data = snap["data"]
        if data.ndim != 2:
            return None

        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)
        im = ax.imshow(data, aspect="auto", cmap=_HEATMAP_CMAP, vmin=0, vmax=1,
                        interpolation="nearest")
        ax.set_xlabel("Time period $t$")
        ax.set_ylabel("Generator $g$")
        ax.set_title(f"Commitment $x$ ({stage.upper()}, iter {iter + 1})")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.outline.set_linewidth(0.4)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, f"x_heatmap_{stage}_iter{iter}")
        return fig

    # ------------------------------------------------------------------ #
    #  4. x 初始 vs 最终对比
    # ------------------------------------------------------------------ #
    def plot_x_comparison(self, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        x_snaps = self.logger.snapshots["x"]
        if len(x_snaps) < 2:
            return None
        first, last = x_snaps[0], x_snaps[-1]
        if first["data"].ndim != 2 or last["data"].ndim != 2:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                        gridspec_kw={"wspace": 0.35})
        for ax, snap, lbl in [(ax1, first, "(a) Initial"), (ax2, last, "(b) Final")]:
            im = ax.imshow(snap["data"], aspect="auto", cmap=_HEATMAP_CMAP,
                           vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlabel("Time period $t$")
            ax.set_ylabel("Generator $g$")
            ax.set_title(f"{lbl} (iter {snap['iter'] + 1})", fontsize=10)
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cb.outline.set_linewidth(0.4)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, "x_comparison")
        return fig

    # ------------------------------------------------------------------ #
    #  5. pg 堆叠面积图
    # ------------------------------------------------------------------ #
    def plot_pg_dispatch(self, stage: str, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        pg_snaps = [s for s in self.logger.snapshots["pg"] if s["stage"] == stage]
        if not pg_snaps:
            return None
        snap = pg_snaps[-1]
        data = snap["data"]
        if data.ndim != 2:
            return None

        ng, T = data.shape
        t_axis = np.arange(1, T + 1)

        # 生成配色（从 tab20c 取冷暖交替色）
        cmap = plt.cm.tab20c
        colors = [cmap(i / max(ng - 1, 1) * 0.85) for i in range(ng)]

        fig, ax = plt.subplots(figsize=self.FIGSIZE_WIDE)
        ax.stackplot(t_axis, data, colors=colors,
                     labels=[f"$G_{{{g+1}}}$" for g in range(ng)], alpha=0.85)
        ax.set_xlabel("Time period $t$")
        ax.set_ylabel("Power output (MW)")
        ax.set_xlim(1, T)
        ax.set_title(f"Generation Dispatch ({stage.upper()}, iter {snap['iter'] + 1})")
        ax.legend(loc="upper right", fontsize=7, ncol=min(ng, 5),
                  handlelength=1.2, columnspacing=0.8)
        self._despine(ax)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, f"pg_dispatch_{stage}")
        return fig

    # ------------------------------------------------------------------ #
    #  6. lambda 演化
    # ------------------------------------------------------------------ #
    def plot_lambda_evolution(self, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        lam_snaps = self.logger.snapshots["lambda"]
        if not lam_snaps:
            return None

        n = len(lam_snaps)
        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)

        cmap = plt.cm.coolwarm
        for idx, snap in enumerate(lam_snaps):
            data = snap["data"]
            t_axis = np.arange(1, len(data) + 1)
            frac = idx / max(n - 1, 1)
            ax.plot(t_axis, data, color=cmap(frac), linewidth=0.9, alpha=0.85)

        ax.set_xlabel("Time period $t$")
        ax.set_ylabel(r"$\lambda$ (dual variable)")
        ax.set_title("Dual Variable Evolution")

        # colorbar 表示迭代进度
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=lam_snaps[0]["iter"] + 1, vmax=lam_snaps[-1]["iter"] + 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Iteration", fontsize=8)
        cb.outline.set_linewidth(0.4)
        self._despine(ax)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, "lambda_evolution")
        return fig

    # ------------------------------------------------------------------ #
    #  7. mu 分布
    # ------------------------------------------------------------------ #
    def plot_mu_distribution(self, unit_id: int, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        key = str(unit_id)
        data = self.logger.metrics["surrogate"].get(key, [])
        mu_vals = [d["mu_mean"] for d in data if "mu_mean" in d]
        if not mu_vals:
            return None

        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)
        # 用 mu 随迭代的折线 + 散点代替 boxplot（数据是逐迭代的标量）
        iters = np.array([d["iter"] for d in data if "mu_mean" in d]) + 1
        ax.plot(iters, mu_vals, "o-", color=_C["purple"], markersize=3)
        ax.fill_between(iters, 0, mu_vals, color=_C["purple"], alpha=0.08)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Mean $\mu$")
        ax.set_title(f"Unit {unit_id} — Surrogate Dual $\\mu$")
        ax.set_ylim(bottom=0)
        self._despine(ax)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, f"mu_dist_unit{unit_id}")
        return fig

    # ------------------------------------------------------------------ #
    #  8. 代理约束系数
    # ------------------------------------------------------------------ #
    def plot_surrogate_coefficients(self, unit_id: int, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        key = str(unit_id)
        data = self.logger.metrics["surrogate"].get(key, [])
        if not data or "alpha_mean" not in data[0]:
            return None

        iters = np.array([d["iter"] for d in data]) + 1
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0),
                                  gridspec_kw={"hspace": 0.38, "wspace": 0.35})

        coeff_keys = ["alpha_mean", "beta_mean", "gamma_mean", "delta_mean"]
        labels = [r"$\bar{\alpha}$", r"$\bar{\beta}$", r"$\bar{\gamma}$", r"$\bar{\delta}$"]
        colors = [_C["blue"], _C["red"], _C["green"], _C["orange"]]

        for ax, ckey, lbl, clr in zip(axes.flat, coeff_keys, labels, colors):
            vals = [d.get(ckey, 0) for d in data]
            ax.plot(iters, vals, "o-", color=clr, markersize=2.5)
            ax.set_ylabel(lbl)
            ax.set_xlabel("Iteration")
            self._despine(ax)

        fig.suptitle(f"Unit {unit_id} — Surrogate Coefficients", fontsize=11, y=1.0)
        if save_dir:
            self._savefig(fig, save_dir, f"surrogate_coeff_unit{unit_id}")
        return fig

    # ------------------------------------------------------------------ #
    #  9. 代理约束违反热力图
    # ------------------------------------------------------------------ #
    def plot_surrogate_violation(self, trainers: Optional[Dict] = None,
                                 sample: int = 0,
                                 save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        # None 或空 dict（例如仅训练 dual_predictor、无子问题代理）都不画
        if not trainers:
            return None

        unit_ids = sorted(trainers.keys())
        viol_rows = []
        max_nc = 0
        for g in unit_ids:
            trainer = trainers[g]
            nc = trainer.num_coupling_constraints
            max_nc = max(max_nc, nc)
            x0 = trainer.x[sample]
            viols = []
            for k in range(nc):
                t = trainer.sensitive_timesteps[sample][k] if hasattr(trainer, "sensitive_timesteps") else k
                alpha_k, beta_k, gamma_k, delta_k = trainer._apply_surrogate_direction_to_params(
                    np.array([trainer.alpha_values[sample, k]], dtype=float),
                    np.array([trainer.beta_values[sample, k]], dtype=float),
                    np.array([trainer.gamma_values[sample, k]], dtype=float),
                    np.array([trainer.delta_values[sample, k]], dtype=float),
                )
                lhs = (alpha_k[0] * x0[t]
                       + beta_k[0] * x0[min(t + 1, len(x0) - 1)]
                       + gamma_k[0] * x0[min(t + 2, len(x0) - 1)])
                viols.append(max(0.0, lhs - delta_k[0]))
            viol_rows.append(viols)

        mat = np.zeros((len(unit_ids), max_nc))
        for i, row in enumerate(viol_rows):
            mat[i, :len(row)] = row

        if mat.size == 0 or mat.shape[0] < 1 or mat.shape[1] < 1:
            return None

        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)
        im = ax.imshow(mat, aspect="auto", cmap=_VIOL_CMAP, interpolation="nearest")
        ax.set_xlabel("Constraint index $k$")
        ax.set_ylabel("Generator $g$")
        ax.set_yticks(range(len(unit_ids)))
        ax.set_yticklabels([f"$G_{{{g+1}}}$" for g in unit_ids])
        ax.set_title(f"Surrogate Constraint Violation (sample {sample})")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Violation", fontsize=8)
        cb.outline.set_linewidth(0.4)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, f"surrogate_violation_sample{sample}")
        return fig

    # ------------------------------------------------------------------ #
    #  10. NN Loss
    # ------------------------------------------------------------------ #
    def plot_nn_loss(self, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=self.FIGSIZE_SINGLE)
        has_data = False

        plot_specs = []

        bcd_data = self.logger.metrics["bcd"]
        nn_vals = [(d["iter"] + 1, d["nn_loss"]) for d in bcd_data if "nn_loss" in d]
        if nn_vals:
            plot_specs.append((nn_vals, r"BCD $\theta/\zeta$", _C["blue"], "o"))

        joint_data = self.logger.metrics["joint"]
        nn_vals = [(d["iter"] + 1, d["nn_loss"]) for d in joint_data if "nn_loss" in d]
        if nn_vals:
            plot_specs.append((nn_vals, r"Joint $\theta/\zeta$", _C["red"], "s"))

        surr_vals = [(d["iter"] + 1, d["surr_nn_loss"]) for d in joint_data if "surr_nn_loss" in d]
        if surr_vals:
            plot_specs.append((surr_vals, "Joint surrogate", _C["green"], "^"))

        surr_metrics = self.logger.metrics["surrogate"]
        if surr_metrics:
            all_losses: Dict[int, List[float]] = {}
            for uid, records in surr_metrics.items():
                for d in records:
                    if "nn_loss" in d:
                        all_losses.setdefault(d["iter"] + 1, []).append(d["nn_loss"])
            if all_losses:
                iters_sorted = sorted(all_losses.keys())
                avg_vals = [np.mean(all_losses[i]) for i in iters_sorted]
                plot_specs.append((list(zip(iters_sorted, avg_vals)),
                                   "Surrogate (avg)", _C["orange"], "d"))

        for vals, label, color, marker in plot_specs:
            its, losses = zip(*vals)
            ax.plot(its, losses, marker=marker, linestyle="-", color=color,
                    label=label, markersize=3)
            has_data = True

        if not has_data:
            plt.close(fig)
            return None

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Neural Network Training Loss")
        ax.legend(loc="upper right")
        self._despine(ax)

        fig.tight_layout()
        if save_dir:
            self._savefig(fig, save_dir, "nn_loss")
        return fig

    # ------------------------------------------------------------------ #
    #  11. Dashboard
    # ------------------------------------------------------------------ #
    def plot_dashboard(self, save_dir: Optional[Path] = None) -> Optional[plt.Figure]:
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.5),
                                  gridspec_kw={"hspace": 0.42, "wspace": 0.38})

        # (0,0) BCD convergence
        self._subplot_convergence(axes[0, 0], "bcd", "BCD Violations")

        # (0,1) Surrogate convergence (first unit)
        surr_keys = list(self.logger.metrics["surrogate"].keys())
        if surr_keys:
            self._subplot_convergence_unit(axes[0, 1], surr_keys[0],
                                           f"Surrogate (Unit {surr_keys[0]})")
        else:
            axes[0, 1].set_visible(False)

        # (1,0) Joint convergence
        self._subplot_convergence(axes[1, 0], "joint", "Joint Violations")

        # (1,1) Integrality
        joint = self.logger.metrics["joint"]
        int_vals = [(d["iter"] + 1, d["integrality"]) for d in joint if "integrality" in d]
        if int_vals:
            its, vals = zip(*int_vals)
            axes[1, 1].plot(its, vals, "o-", color=_C["red"], markersize=2.5)
            axes[1, 1].set_title("Integrality Gap", fontsize=9)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel(r"$\mathrm{avg}\;x_i(1-x_i)$")
            self._despine(axes[1, 1])
        else:
            axes[1, 1].set_visible(False)

        # (2,0) NN Loss
        self._subplot_nn_loss(axes[2, 0])

        # (2,1) rho evolution (joint)
        if joint:
            its = np.array([d["iter"] for d in joint]) + 1
            labels_r = [r"$\rho_{\mathrm{p}}$", r"$\rho_{\mathrm{d}}$", r"$\rho_{\mathrm{o}}$"]
            for key, label, color in zip(
                ["rho_primal", "rho_dual", "rho_opt"], labels_r, _PALETTE_3
            ):
                axes[2, 1].plot(its, [d[key] for d in joint], color=color, label=label)
            axes[2, 1].set_title("Joint Penalty Evolution", fontsize=9)
            axes[2, 1].set_xlabel("Iteration")
            axes[2, 1].set_ylabel("Penalty")
            axes[2, 1].legend(fontsize=7)
            self._despine(axes[2, 1])
        else:
            axes[2, 1].set_visible(False)

        fig.suptitle("Training Dashboard", fontsize=12, y=0.99)
        if save_dir:
            self._savefig(fig, save_dir, "dashboard")
        return fig

    # ------------------------------------------------------------------ #
    #  plot_all
    # ------------------------------------------------------------------ #
    def plot_all(self, save_dir: Union[str, Path], trainers: Optional[Dict] = None) -> None:
        """生成所有图表并保存到目录。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for stage in ("bcd", "joint"):
            self.plot_convergence(stage, save_dir)

        for uid in self.logger.metrics["surrogate"]:
            data = self.logger.metrics["surrogate"][uid]
            if data:
                self._plot_surrogate_convergence(int(uid), save_dir)

        self.plot_integrality(save_dir)

        x_snaps = self.logger.snapshots["x"]
        if x_snaps:
            self.plot_x_heatmap(x_snaps[0]["stage"], x_snaps[0]["iter"], save_dir)
            if len(x_snaps) > 1:
                self.plot_x_heatmap(x_snaps[-1]["stage"], x_snaps[-1]["iter"], save_dir)
        self.plot_x_comparison(save_dir)

        for stage in ("bcd", "joint"):
            self.plot_pg_dispatch(stage, save_dir)

        self.plot_lambda_evolution(save_dir)

        for uid in self.logger.metrics["surrogate"]:
            self.plot_mu_distribution(int(uid), save_dir)
            self.plot_surrogate_coefficients(int(uid), save_dir)

        self.plot_surrogate_violation(trainers, save_dir=save_dir)
        self.plot_nn_loss(save_dir)
        self.plot_dashboard(save_dir)

        print(f"[Visualizer] Figures saved to: {save_dir}", flush=True)

    # ------------------------------------------------------------------ #
    #  内部辅助
    # ------------------------------------------------------------------ #
    def _get_stage_data(self, stage: str) -> list:
        if stage in ("bcd", "joint"):
            return self.logger.metrics[stage]
        return []

    def _find_snapshot(self, var: str, stage: str, iter: int) -> Optional[dict]:
        for snap in self.logger.snapshots[var]:
            if snap["stage"] == stage and snap["iter"] == iter:
                return snap
        return None

    def _subplot_convergence(self, ax: plt.Axes, stage: str, title: str) -> None:
        data = self._get_stage_data(stage)
        if not data:
            ax.set_visible(False)
            return
        iters = np.array([d["iter"] for d in data]) + 1
        keys = ["obj_primal", "obj_dual", "obj_opt"]
        labels = ["Primal", "Dual", "Opt"]
        for key, label, color in zip(keys, labels, _PALETTE_3):
            vals = np.array([d[key] for d in data])
            ax.plot(iters, _safe_log(vals), color=color, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\log_{10}$(viol)")
        ax.legend(fontsize=6.5)
        self._despine(ax)

    def _subplot_convergence_unit(self, ax: plt.Axes, uid: str, title: str) -> None:
        data = self.logger.metrics["surrogate"].get(uid, [])
        if not data:
            ax.set_visible(False)
            return
        iters = np.array([d["iter"] for d in data]) + 1
        for key, label, color in zip(
            ["obj_primal", "obj_dual", "obj_opt"],
            ["Primal", "Dual", "Opt"],
            _PALETTE_3,
        ):
            vals = np.array([d[key] for d in data])
            ax.plot(iters, _safe_log(vals), color=color, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\log_{10}$(viol)")
        ax.legend(fontsize=6.5)
        self._despine(ax)

    def _subplot_nn_loss(self, ax: plt.Axes) -> None:
        has_data = False
        specs = [("bcd", "nn_loss", _C["blue"], "BCD"), ("joint", "nn_loss", _C["red"], "Joint")]
        for stage, key_name, color, lbl in specs:
            data = self._get_stage_data(stage)
            vals = [(d["iter"] + 1, d[key_name]) for d in data if key_name in d]
            if vals:
                its, losses = zip(*vals)
                ax.plot(its, losses, "o-", color=color, label=lbl, markersize=2.5)
                has_data = True
        if not has_data:
            ax.set_visible(False)
            return
        ax.set_title("NN Loss", fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6.5)
        self._despine(ax)

    def _plot_surrogate_convergence(self, unit_id: int, save_dir: Path) -> Optional[plt.Figure]:
        key = str(unit_id)
        data = self.logger.metrics["surrogate"].get(key, [])
        if not data:
            return None

        iters = np.array([d["iter"] for d in data]) + 1
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.FIGSIZE_DOUBLE, sharex=True,
            gridspec_kw={"hspace": 0.12})

        for k, label, color in zip(
            ["obj_primal", "obj_dual", "obj_opt"],
            ["Primal", "Dual", "Optimality"],
            _PALETTE_3,
        ):
            vals = np.array([d[k] for d in data])
            ax1.plot(iters, _safe_log(vals), color=color, label=label)
        ax1.set_ylabel(r"$\log_{10}$(violation)")
        ax1.legend(loc="upper right")
        self._despine(ax1)

        for k, label, color in zip(
            ["rho_primal", "rho_dual", "rho_opt"],
            [r"$\rho_{\mathrm{prim}}$", r"$\rho_{\mathrm{dual}}$", r"$\rho_{\mathrm{opt}}$"],
            _PALETTE_3,
        ):
            ax2.plot(iters, [d[k] for d in data], color=color, label=label)
        ax2.set_ylabel("Penalty parameter")
        ax2.set_xlabel("Iteration")
        ax2.legend(loc="upper left")
        self._despine(ax2)

        fig.suptitle(f"Surrogate Training — Unit {unit_id}", fontsize=11, y=0.98)
        self._savefig(fig, save_dir, f"convergence_surrogate_unit{unit_id}")
        return fig
