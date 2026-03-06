#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本（多模式）
- surrogate: 加载已训练的 V3 代理约束模型，输出参数摘要，可选运行可行性泵
- bcd:       加载已训练的 BCD 神经网络模型，报告参数统计
- both:      联合加载 BCD + surrogate 模型，以全体代理约束评估解质量，可选 FP

修改顶部的 MODE / MODEL_DIR / BCD_MODEL_PATH 等变量切换执行模式。
"""

import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# ──────────────────────── 依赖检查 ────────────────────────


def check_and_install_dependencies():
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gurobipy': 'gurobipy',
        'pypower': 'PYPOWER',
    }
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"[OK] {import_name}")
        except ImportError:
            missing.append(package_name)
            print(f"[MISS] {import_name} 未安装")

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        response = input("是否自动安装? (y/n): ")
        if response.strip().lower() == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("[OK] 安装完成")
            return True
        else:
            print("请手动安装后重试")
            return False
    return True


if not check_and_install_dependencies():
    sys.exit(1)

# ──────────────────────── 模式配置 ────────────────────────
#
#   'surrogate' - 加载 V3 代理约束模型并测试
#   'bcd'       - 加载 BCD 神经网络模型并报告参数统计
#   'both'      - 联合加载 BCD + surrogate，以全体代理约束评估（需同时配置下面两个路径）
#
MODE      = 'both'
RUN_FP    = False       # surrogate / both 模式：是否运行可行性泵测试
CASE_NAME = 'case30'   # 'case14' / 'case30' / 'case39'

# surrogate / both 模式：已训练 surrogate 模型目录（训练时输出的带时间戳路径）
MODEL_DIR = 'result/subproblem_models_case30_20260306_171140'

# bcd / both 模式：已训练 BCD 模型 .pth 文件路径
BCD_MODEL_PATH = 'result/bcd_model_case30_20260306_171140.pth'

TEST_SAMPLES = 3   # 测试/评估样本数

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# matplotlib 为可选依赖，绘图功能在不可用时自动跳过
try:
    import matplotlib
    matplotlib.use('Agg')   # 非交互后端，兼容无显示环境
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    from uc_NN_subproblem_v3 import (
        load_trained_models,
        ActiveSetReader,
    )
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

if MODE in ('bcd', 'both'):
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)

if MODE in ('surrogate', 'both'):
    try:
        from feasibility_pump import recover_integer_solution, solve_global_LP_relaxation
    except ImportError as e:
        print(f"feasibility_pump 模块导入失败: {e}")
        sys.exit(1)

# ──────────────────────── 工具函数 ────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json_data(data_file: Path) -> list:
    """加载 JSON 数据文件并规范化为 v3 所需格式。"""
    log(f"加载数据文件: {data_file.name}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_samples = data.get('all_samples', [])
    if not all_samples:
        raise ValueError("JSON 文件中没有样本数据 (all_samples 为空)")

    log(f"  原始样本数: {len(all_samples)}")

    for sample in all_samples:
        if isinstance(sample.get('pd_data'), list):
            sample['pd_data'] = np.array(sample['pd_data'], dtype=float)

    return all_samples


def pick_data_file(result_dir: Path, case_name: str) -> Path:
    """按优先级查找最合适的数据文件。"""
    specific = sorted(result_dir.glob(f'active_sets_{case_name}_*.json'))
    if specific:
        return specific[-1]
    any_files = sorted(result_dir.glob('active_sets_*.json'))
    if any_files:
        log(f"未找到 {case_name} 专属文件，使用: {any_files[-1].name}")
        return any_files[-1]
    return None


# ──────────────────────── 绘图工具 ────────────────────────

# IEEE/学术风格全局设置
_MPL_STYLE = {
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.85,
    'figure.dpi':         120,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'axes.prop_cycle':    matplotlib.cycler(
        color=['#2166AC', '#D6604D', '#4DAC26', '#8073AC',
               '#F4A582', '#92C5DE', '#A1D76A', '#E9A3C9']
    ) if MPL_AVAILABLE else None,
}


def _apply_style() -> None:
    """应用学术绘图风格（matplotlib 可用时）。"""
    if not MPL_AVAILABLE:
        return
    params = {k: v for k, v in _MPL_STYLE.items() if v is not None}
    plt.rcParams.update(params)


def _save_fig(fig: 'plt.Figure', path: Path, label: str) -> None:
    """保存图像（PNG + PDF），并在终端打印路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix('.png')))
    fig.savefig(str(path.with_suffix('.pdf')))
    plt.close(fig)
    log(f"  图像已保存: {path.with_suffix('.png')} / .pdf  [{label}]")


def plot_surrogate_analysis(trainers: dict, all_samples: list,
                            fig_dir: Path, case_name: str) -> None:
    """绘制 surrogate 模型三张分析图：系数分布、约束违反热图、整数性得分。

    Args:
        trainers:    {unit_id: SubproblemSurrogateTrainer} 字典。
        all_samples: v3 格式样本列表。
        fig_dir:     图像输出目录。
        case_name:   算例名，用于图标题和文件名。
    """
    if not MPL_AVAILABLE:
        log("matplotlib 不可用，跳过绘图")
        return

    _apply_style()
    unit_ids = sorted(trainers.keys())
    n_units = len(unit_ids)

    # ── 图 1：代理约束系数分布（2×2 violin） ─────────────────
    log("绘制图1：代理约束系数分布...")
    fig1, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig1.suptitle(
        f'Surrogate Constraint Coefficient Distributions  [{case_name}]',
        fontsize=12, fontweight='bold', y=1.01
    )

    coef_info = [
        ('alpha_values', r'$\alpha$ (Coefficient of $x_t$)',     axes[0, 0]),
        ('beta_values',  r'$\beta$ (Coefficient of $x_{t+1}$)',  axes[0, 1]),
        ('gamma_values', r'$\gamma$ (Coefficient of $x_{t+2}$)', axes[1, 0]),
        ('delta_values', r'$\delta$ (RHS / Slack)',               axes[1, 1]),
    ]

    colors = ['#2166AC', '#D6604D', '#4DAC26', '#8073AC']

    for (attr, ylabel, ax), color in zip(coef_info, colors):
        data_per_unit = []
        labels = []
        for uid in unit_ids:
            arr = getattr(trainers[uid], attr)   # (n_samples, nc)
            data_per_unit.append(arr.ravel())
            labels.append(f'G{uid}')

        parts = ax.violinplot(
            data_per_unit,
            positions=range(n_units),
            showmedians=True,
            showextrema=True,
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(1.2)

        ax.set_xticks(range(n_units))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Generator Unit')

    fig1.tight_layout()
    _save_fig(fig1, fig_dir / f'{case_name}_surrogate_coefficients', '系数分布')

    # ── 图 2：约束违反热图（units × time） ────────────────────
    log("绘制图2：约束违反热图...")
    # 取所有单元中最小的 nc 作为公共时间轴
    nc_list = [trainers[uid].num_coupling_constraints for uid in unit_ids]
    nc_common = min(nc_list)
    n_samples_plot = len(all_samples)

    viol_matrix = np.zeros((n_units, nc_common))   # (ng, nc)
    for gi, uid in enumerate(unit_ids):
        tr = trainers[uid]
        ns = min(n_samples_plot, tr.alpha_values.shape[0])
        for s in range(ns):
            x_s = np.asarray(tr.x[s], dtype=float)   # (T,)
            for t in range(nc_common):
                if t + 2 >= tr.T:
                    break
                lhs = (tr.alpha_values[s, t] * x_s[t]
                       + tr.beta_values[s, t] * x_s[t + 1]
                       + tr.gamma_values[s, t] * x_s[t + 2])
                viol_matrix[gi, t] += max(0.0, lhs - tr.delta_values[s, t])
        viol_matrix[gi] /= ns

    fig2, ax2 = plt.subplots(figsize=(min(14, nc_common * 0.55 + 2), max(3, n_units * 0.5 + 1.5)))
    im = ax2.imshow(
        viol_matrix, aspect='auto', cmap='RdYlGn_r',
        interpolation='nearest',
        norm=Normalize(vmin=0, vmax=max(viol_matrix.max(), 1e-6)),
    )
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('Mean Constraint Violation', fontsize=9)

    ax2.set_yticks(range(n_units))
    ax2.set_yticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax2.set_xlabel('Coupling Constraint Index (Time Period $t$)')
    ax2.set_ylabel('Generator Unit')
    ax2.set_title(
        f'Mean Surrogate Constraint Violation  [{case_name}]  '
        f'(avg over {n_samples_plot} samples)',
        fontweight='bold'
    )
    # 在格子上标注数值（仅当格子足够大时）
    if nc_common <= 30 and n_units <= 15:
        for gi in range(n_units):
            for t in range(nc_common):
                val = viol_matrix[gi, t]
                color_txt = 'white' if val > viol_matrix.max() * 0.6 else '#333333'
                ax2.text(t, gi, f'{val:.2f}', ha='center', va='center',
                         fontsize=6, color=color_txt)

    fig2.tight_layout()
    _save_fig(fig2, fig_dir / f'{case_name}_surrogate_violation_heatmap', '违反热图')

    # ── 图 3：整数性得分（每机组，跨样本均值 ± 标准差） ─────
    log("绘制图3：整数性得分...")
    integ_means, integ_stds = [], []
    for uid in unit_ids:
        tr = trainers[uid]
        ns = tr.alpha_values.shape[0]
        scores = []
        for s in range(ns):
            x_s = np.asarray(tr.x[s], dtype=float)
            scores.append(float(np.sum(x_s * (1.0 - x_s))))
        integ_means.append(np.mean(scores))
        integ_stds.append(np.std(scores))

    fig3, ax3 = plt.subplots(figsize=(max(5, n_units * 0.65 + 1.5), 4))
    xpos = np.arange(n_units)
    bars = ax3.bar(
        xpos, integ_means, yerr=integ_stds,
        capsize=4, width=0.6,
        color='#2166AC', alpha=0.75, edgecolor='#144E7A', linewidth=0.8,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
    )
    ax3.axhline(0, color='#333333', linewidth=0.8, linestyle='-')
    ax3.set_xticks(xpos)
    ax3.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=9)
    ax3.set_xlabel('Generator Unit')
    ax3.set_ylabel(r'Integrality Score  $\sum x_i(1-x_i)$')
    ax3.set_title(
        f'Per-Unit Integrality Score  [{case_name}]  '
        f'(mean ± std over {trainers[unit_ids[0]].alpha_values.shape[0]} samples)',
        fontweight='bold'
    )
    # 在每根柱子顶部标注均值
    for bar, mean in zip(bars, integ_means):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(integ_stds) * 0.05,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=7.5, color='#222222'
        )

    fig3.tight_layout()
    _save_fig(fig3, fig_dir / f'{case_name}_surrogate_integrality', '整数性得分')


def plot_fp_results(fp_results: list, fig_dir: Path, case_name: str) -> None:
    """绘制可行性泵测试汇总图：成功率饼图 + 每样本结果条形图。

    Args:
        fp_results: run_fp_test 返回的 [(idx, success, x_result), ...] 列表。
        fig_dir:    图像输出目录。
        case_name:  算例名。
    """
    if not MPL_AVAILABLE or not fp_results:
        return

    _apply_style()
    n_total = len(fp_results)
    n_success = sum(1 for _, s, _ in fp_results if s)
    n_fail = n_total - n_success

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f'Feasibility Pump Test Results  [{case_name}]',
        fontsize=12, fontweight='bold'
    )

    # 饼图
    wedge_colors = ['#4DAC26', '#D6604D']
    wedges, texts, autotexts = ax_pie.pie(
        [n_success, n_fail],
        labels=['Success', 'Failure'],
        autopct='%1.1f%%',
        colors=wedge_colors,
        startangle=90,
        wedgeprops=dict(linewidth=1.2, edgecolor='white'),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax_pie.set_title(f'Overall ({n_success}/{n_total} succeeded)', fontsize=10)
    ax_pie.grid(False)

    # 每样本条形图
    sample_ids = [r[0] for r in fp_results]
    colors_bar = ['#4DAC26' if r[1] else '#D6604D' for r in fp_results]
    # 用 1/0 高度表示成功/失败；高度 = 成功1, 失败0.3（视觉区分）
    heights = [1.0 if r[1] else 0.3 for r in fp_results]
    ax_bar.bar(
        range(n_total), heights,
        color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.8,
    )
    ax_bar.set_xticks(range(n_total))
    ax_bar.set_xticklabels([f'#{i}' for i in sample_ids], fontsize=8)
    ax_bar.set_yticks([0.3, 1.0])
    ax_bar.set_yticklabels(['Failure', 'Success'], fontsize=9)
    ax_bar.set_xlabel('Sample Index')
    ax_bar.set_title('Per-Sample Outcome', fontsize=10)
    ax_bar.set_ylim(0, 1.35)
    ax_bar.grid(axis='x', alpha=0)

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[Patch(facecolor='#4DAC26', label='Success'),
                 Patch(facecolor='#D6604D', label='Failure')],
        loc='upper right', framealpha=0.85,
    )

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_fp_results', 'FP 测试结果')


def plot_bcd_analysis(agent, fig_dir: Path, case_name: str) -> None:
    """绘制 BCD 模型两张分析图：θ/ζ 参数直方图 + 网络权重分布。

    Args:
        agent:     Agent_NN_BCD 实例（已加载模型参数）。
        fig_dir:   图像输出目录。
        case_name: 算例名。
    """
    if not MPL_AVAILABLE:
        log("matplotlib 不可用，跳过绘图")
        return

    _apply_style()

    # ── 图 1：θ / ζ 参数直方图 ────────────────────────────
    has_theta = hasattr(agent, 'theta_values') and bool(agent.theta_values)
    has_zeta  = hasattr(agent, 'zeta_values')  and bool(agent.zeta_values)

    if has_theta or has_zeta:
        log("绘制图1：θ / ζ 参数直方图...")
        ncols = int(has_theta) + int(has_zeta)
        fig1, axes1 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        if ncols == 1:
            axes1 = [axes1]
        fig1.suptitle(
            f'BCD Surrogate Parameter Distributions  [{case_name}]',
            fontsize=12, fontweight='bold'
        )

        ax_idx = 0
        for flag, attr, label, color in [
            (has_theta, 'theta_values', r'$\theta$ Values', '#2166AC'),
            (has_zeta,  'zeta_values',  r'$\zeta$ Values',  '#D6604D'),
        ]:
            if not flag:
                continue
            vals = np.array(list(getattr(agent, attr).values()), dtype=float)
            ax = axes1[ax_idx]
            n_bins = min(50, max(10, len(vals) // 5))
            ax.hist(vals, bins=n_bins, color=color, alpha=0.72,
                    edgecolor='white', linewidth=0.6, density=True)
            # KDE 曲线
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method='scott')
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color='#333333', linewidth=1.5,
                        linestyle='--', label='KDE')
                ax.legend()
            except Exception:
                pass

            ax.axvline(vals.mean(), color='#333333', linewidth=1.2,
                       linestyle=':', label=f'Mean={vals.mean():.3f}')
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(
                f'{label}  (n={len(vals)}, '
                f'mean={vals.mean():.3f}, std={vals.std():.3f})'
            )
            ax_idx += 1

        fig1.tight_layout()
        _save_fig(fig1, fig_dir / f'{case_name}_bcd_param_hist', 'θ/ζ 直方图')

    # ── 图 2：神经网络权重层分布（violin） ────────────────────
    import torch

    net_specs = []
    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        net_specs.append((agent.theta_net, r'$\theta$-Net', '#2166AC'))
    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        net_specs.append((agent.zeta_net, r'$\zeta$-Net', '#D6604D'))

    if not net_specs:
        return

    log("绘制图2：神经网络权重层分布...")
    n_nets = len(net_specs)
    fig2, axes2 = plt.subplots(1, n_nets, figsize=(6 * n_nets, 5))
    if n_nets == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f'Neural Network Weight Distributions  [{case_name}]',
        fontsize=12, fontweight='bold'
    )

    for ax, (net, net_label, color) in zip(axes2, net_specs):
        layer_data, layer_labels = [], []
        for name, param in net.named_parameters():
            if param.requires_grad:
                w = param.detach().cpu().numpy().ravel()
                if len(w) > 0:
                    layer_data.append(w)
                    # 缩短名字：weight → W, bias → b
                    short = name.replace('weight', 'W').replace('bias', 'b')
                    layer_labels.append(short)

        if not layer_data:
            ax.text(0.5, 0.5, 'No parameters', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            continue

        parts = ax.violinplot(
            layer_data,
            positions=range(len(layer_data)),
            showmedians=True, showextrema=True,
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(1.1)

        ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'{net_label} Layer Weights')

    fig2.tight_layout()
    _save_fig(fig2, fig_dir / f'{case_name}_bcd_weight_dist', '权重分布')


def plot_both_analysis(agent, trainers: dict, fig_dir: Path, case_name: str) -> None:
    """绘制 BCD-Surrogate 联合特征图：4 面板综合约束刻画。

    面板布局 (2×2)：
      (a) 各机组代理约束 RHS (δ) 均值 — 约束紧度
      (b) 各机组代理系数耦合强度 √(ᾱ²+β̄²+γ̄²) — 时序耦合程度
      (c) BCD θ 参数分布（直方图+KDE）
      (d) BCD ζ 参数分布（直方图+KDE）

    Args:
        agent:     Agent_NN_BCD 实例（已加载模型）。
        trainers:  {unit_id: SubproblemSurrogateTrainer} 字典。
        fig_dir:   图像输出目录。
        case_name: 算例名。
    """
    if not MPL_AVAILABLE:
        return

    _apply_style()
    unit_ids = sorted(trainers.keys())
    n_units  = len(unit_ids)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f'BCD–Surrogate Joint Constraint Characterization  [{case_name}]',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── (a) 各机组代理约束 RHS δ 均值（violin） ──────────────
    ax_a = axes[0, 0]
    delta_per_unit = [trainers[uid].delta_values.ravel() for uid in unit_ids]
    parts_a = ax_a.violinplot(
        delta_per_unit, positions=range(n_units),
        showmedians=True, showextrema=True,
    )
    for pc in parts_a['bodies']:
        pc.set_facecolor('#8073AC')
        pc.set_alpha(0.6)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        if key in parts_a:
            parts_a[key].set_color('#333333')
            parts_a[key].set_linewidth(1.1)
    ax_a.set_xticks(range(n_units))
    ax_a.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax_a.set_xlabel('Generator Unit')
    ax_a.set_ylabel(r'$\delta$ (RHS Bound)')
    ax_a.set_title(r'(a) Surrogate Constraint Tightness $\delta$', loc='left', fontsize=10)

    # ── (b) 各机组时序耦合强度 ─────────────────────────────
    ax_b = axes[0, 1]
    coupling_strength = []
    for uid in unit_ids:
        tr = trainers[uid]
        a_mean = np.abs(tr.alpha_values).mean(axis=1)   # (n_samples,)
        b_mean = np.abs(tr.beta_values).mean(axis=1)
        g_mean = np.abs(tr.gamma_values).mean(axis=1)
        strength = np.sqrt(a_mean**2 + b_mean**2 + g_mean**2)  # (n_samples,)
        coupling_strength.append(strength)

    means = [s.mean() for s in coupling_strength]
    stds  = [s.std()  for s in coupling_strength]
    xpos  = np.arange(n_units)
    bars  = ax_b.bar(
        xpos, means, yerr=stds, capsize=4, width=0.6,
        color='#4DAC26', alpha=0.72, edgecolor='#2A7A15', linewidth=0.8,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
    )
    for bar, m in zip(bars, means):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.08,
            f'{m:.3f}', ha='center', va='bottom', fontsize=7.5
        )
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax_b.set_xlabel('Generator Unit')
    ax_b.set_ylabel(r'$\sqrt{\bar\alpha^2+\bar\beta^2+\bar\gamma^2}$')
    ax_b.set_title(r'(b) Temporal Coupling Strength', loc='left', fontsize=10)

    # ── (c) BCD θ 分布 ─────────────────────────────────────
    ax_c = axes[1, 0]
    has_theta = hasattr(agent, 'theta_values') and bool(agent.theta_values)
    if has_theta:
        theta_vals = np.array(list(agent.theta_values.values()), dtype=float)
        n_bins = min(50, max(10, len(theta_vals) // 5))
        ax_c.hist(theta_vals, bins=n_bins, color='#2166AC', alpha=0.65,
                  edgecolor='white', linewidth=0.5, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(theta_vals, bw_method='scott')
            xs = np.linspace(theta_vals.min(), theta_vals.max(), 300)
            ax_c.plot(xs, kde(xs), color='#0A3F70', linewidth=1.8,
                      linestyle='--', label='KDE')
        except Exception:
            pass
        ax_c.axvline(theta_vals.mean(), color='#D6604D', linewidth=1.4,
                     linestyle=':', label=f'mean={theta_vals.mean():.3f}')
        ax_c.legend(fontsize=8)
        ax_c.set_xlabel(r'$\theta$ value')
    else:
        ax_c.text(0.5, 0.5, 'theta_values not available',
                  transform=ax_c.transAxes, ha='center', va='center')
    ax_c.set_ylabel('Density')
    ax_c.set_title(r'(c) BCD $\theta$ Parameter Distribution', loc='left', fontsize=10)

    # ── (d) BCD ζ 分布 ─────────────────────────────────────
    ax_d = axes[1, 1]
    has_zeta = hasattr(agent, 'zeta_values') and bool(agent.zeta_values)
    if has_zeta:
        zeta_vals = np.array(list(agent.zeta_values.values()), dtype=float)
        n_bins = min(50, max(10, len(zeta_vals) // 5))
        ax_d.hist(zeta_vals, bins=n_bins, color='#D6604D', alpha=0.65,
                  edgecolor='white', linewidth=0.5, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(zeta_vals, bw_method='scott')
            xs = np.linspace(zeta_vals.min(), zeta_vals.max(), 300)
            ax_d.plot(xs, kde(xs), color='#7B1A0A', linewidth=1.8,
                      linestyle='--', label='KDE')
        except Exception:
            pass
        ax_d.axvline(zeta_vals.mean(), color='#2166AC', linewidth=1.4,
                     linestyle=':', label=f'mean={zeta_vals.mean():.3f}')
        ax_d.legend(fontsize=8)
        ax_d.set_xlabel(r'$\zeta$ value')
    else:
        ax_d.text(0.5, 0.5, 'zeta_values not available',
                  transform=ax_d.transAxes, ha='center', va='center')
    ax_d.set_ylabel('Density')
    ax_d.set_title(r'(d) BCD $\zeta$ Parameter Distribution', loc='left', fontsize=10)

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_both_joint_characterization', 'BCD-Surrogate 联合图')


# ──────────────────────── 模式实现 ────────────────────────


def print_surrogate_results(trainers: dict, all_samples: list) -> None:
    """打印代理训练结果摘要。"""
    n_samples = len(all_samples)
    print("\n" + "=" * 70)
    log("模型参数摘要")
    print("=" * 70)

    for unit_id, trainer in trainers.items():
        T = trainer.T
        nc = trainer.num_coupling_constraints
        print(f"\n机组 {unit_id}:")
        print(f"  alpha_values shape: {trainer.alpha_values.shape}  "
              f"(期望: ({n_samples}, {nc}))")
        print(f"  beta_values  shape: {trainer.beta_values.shape}")
        print(f"  gamma_values shape: {trainer.gamma_values.shape}")
        print(f"  delta_values shape: {trainer.delta_values.shape}  (RHS，非负)")

        x0 = trainer.x[0]
        print(f"  样本0 时序约束示例（最多5条）:")
        for t in range(min(5, nc)):
            if t + 2 >= T:
                break
            a = trainer.alpha_values[0, t]
            b = trainer.beta_values[0, t]
            g = trainer.gamma_values[0, t]
            d = trainer.delta_values[0, t]
            lhs = a * x0[t] + b * x0[t + 1] + g * x0[t + 2]
            viol = max(0.0, lhs - d)
            print(f"    t={t}: {a:.3f}*x[{t}] + {b:.3f}*x[{t+1}] "
                  f"+ {g:.3f}*x[{t+2}] <= {d:.3f}  "
                  f"(lhs={lhs:.3f}, viol={viol:.4f})")

        integrality = float(np.sum(x0 * (1 - x0)))
        print(f"  整数性指标(样本0): {integrality:.6f}  (0=完全整数)")


def _extract_true_solution(sample: dict, shape: tuple) -> np.ndarray:
    """从样本字典中还原真实 UC 解矩阵 (ng, T)。"""
    ng, T = shape
    x_true = np.zeros((ng, T), dtype=float)
    if 'unit_commitment_matrix' in sample:
        uc = np.array(sample['unit_commitment_matrix'], dtype=float)
        r = min(uc.shape[0], ng)
        c = min(uc.shape[1] if uc.ndim > 1 else T, T)
        x_true[:r, :c] = uc[:r, :c]
    elif 'active_set' in sample:
        for item in sample['active_set']:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                if g < ng and t < T:
                    x_true[g, t] = float(item[1])
    return x_true


def plot_lp_vs_true(x_LP_list: list, x_true_list: list,
                    fig_dir: Path, case_name: str) -> None:
    """绘制全局 LP 松弛解与真实解的对比图（5 面板）。

    面板布局 (2 行)：
      上行 (3 格): (a) 平均 LP 松弛热图  (b) 平均真实解热图  (c) 均值绝对差热图
      下行 (2 格): (d) 逐样本 Hamming 距离柱状图  (e) 逐样本整数性间隙柱状图

    Args:
        x_LP_list:   [n_test] 每个样本的 LP 松弛解 (ng, T)。
        x_true_list: [n_test] 每个样本的真实二值解 (ng, T)。
        fig_dir:     图像输出目录。
        case_name:   算例名。
    """
    if not MPL_AVAILABLE or not x_LP_list:
        return

    _apply_style()
    n = len(x_LP_list)

    x_LP_arr   = np.stack(x_LP_list,   axis=0)   # (n, ng, T)
    x_true_arr = np.stack(x_true_list, axis=0)   # (n, ng, T)

    x_LP_mean   = x_LP_arr.mean(axis=0)                     # (ng, T)
    x_true_mean = x_true_arr.mean(axis=0)                   # (ng, T)
    x_diff_mean = np.abs(x_LP_mean - x_true_mean)           # (ng, T)

    x_LP_rounded = (x_LP_arr >= 0.5).astype(int)
    hamming_dists = [int(np.sum(x_LP_rounded[i] != x_true_arr[i].astype(int)))
                     for i in range(n)]
    integ_gaps = [float(np.mean(np.minimum(x_LP_arr[i], 1.0 - x_LP_arr[i])))
                  for i in range(n)]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f'Global LP Relaxation vs. True UC Solution  [{case_name}]',
        fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_lp   = fig.add_subplot(gs[0, 0])
    ax_true = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 2])
    ax_ham  = fig.add_subplot(gs[1, 0:2])
    ax_gap  = fig.add_subplot(gs[1, 2])

    _cbar_kw = dict(fraction=0.045, pad=0.03)
    ng, T_ = x_LP_mean.shape
    yticks = range(ng)
    ylabels = [f'G{g}' for g in yticks]

    # (a) 平均 LP 松弛热图
    im_lp = ax_lp.imshow(x_LP_mean, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                          interpolation='nearest')
    fig.colorbar(im_lp, ax=ax_lp, **_cbar_kw)
    ax_lp.set_yticks(yticks); ax_lp.set_yticklabels(ylabels, fontsize=7)
    ax_lp.set_xlabel('Time Period $t$'); ax_lp.set_ylabel('Generator')
    ax_lp.set_title(r'(a) Mean LP Relaxation $\bar{x}^{LP}$', loc='left', fontsize=10)

    # (b) 平均真实解热图
    im_true = ax_true.imshow(x_true_mean, aspect='auto', cmap='Oranges', vmin=0, vmax=1,
                              interpolation='nearest')
    fig.colorbar(im_true, ax=ax_true, **_cbar_kw)
    ax_true.set_yticks(yticks); ax_true.set_yticklabels(ylabels, fontsize=7)
    ax_true.set_xlabel('Time Period $t$'); ax_true.set_ylabel('Generator')
    ax_true.set_title(r'(b) Mean True Solution $\bar{x}^*$', loc='left', fontsize=10)

    # (c) 均值绝对差热图
    im_diff = ax_diff.imshow(x_diff_mean, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1,
                              interpolation='nearest')
    fig.colorbar(im_diff, ax=ax_diff, **_cbar_kw)
    ax_diff.set_yticks(yticks); ax_diff.set_yticklabels(ylabels, fontsize=7)
    ax_diff.set_xlabel('Time Period $t$'); ax_diff.set_ylabel('Generator')
    ax_diff.set_title(r'(c) Mean $|\bar{x}^{LP} - \bar{x}^*|$', loc='left', fontsize=10)

    # (d) 逐样本 Hamming 距离
    xpos = np.arange(n)
    bars_h = ax_ham.bar(xpos, hamming_dists, color='#2166AC', alpha=0.75,
                        edgecolor='#144E7A', linewidth=0.8, width=0.6)
    for bar, h in zip(bars_h, hamming_dists):
        ax_ham.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(hamming_dists) * 0.02,
                    str(h), ha='center', va='bottom', fontsize=8)
    ax_ham.set_xticks(xpos)
    ax_ham.set_xticklabels([f'Sample #{i}' for i in range(n)], fontsize=9)
    ax_ham.set_ylabel('Hamming Distance (bits)')
    ax_ham.set_title(
        r'(d) Rounded LP vs. True: Hamming Distance  '
        fr'(mean={np.mean(hamming_dists):.1f})',
        loc='left', fontsize=10
    )

    # (e) 逐样本整数性间隙
    bars_g = ax_gap.bar(xpos, integ_gaps, color='#D6604D', alpha=0.75,
                        edgecolor='#7B1A0A', linewidth=0.8, width=0.6)
    for bar, g_val in zip(bars_g, integ_gaps):
        ax_gap.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(integ_gaps) * 0.02,
                    f'{g_val:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax_gap.set_xticks(xpos)
    ax_gap.set_xticklabels([f'#{i}' for i in range(n)], fontsize=9)
    ax_gap.set_ylabel(r'Mean $\min(x^{LP},\,1-x^{LP})$')
    ax_gap.set_title('(e) LP Integrality Gap', loc='left', fontsize=10)

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_lp_vs_true', 'LP 与真实解对比')


def run_lp_compare_test(ppc, all_samples: list, dual_predictor, trainers: dict,
                        T_DELTA: float, n_test: int,
                        fig_dir: Path) -> list:
    """求解全局 LP 松弛，与真实解对比并绘图，返回 (x_LP, x_true) 列表。

    此函数在可行性泵之前运行，评估 LP 松弛解的质量。

    Args:
        ppc:            PyPower 案例字典。
        all_samples:    v3 格式样本列表。
        dual_predictor: 对偶变量预测器（.predict(pd_data) -> lambda）。
        trainers:       {unit_id: SubproblemSurrogateTrainer}。
        T_DELTA:        时间间隔。
        n_test:         测试样本数。
        fig_dir:        图像输出目录。

    Returns:
        [(x_LP, x_true), ...] 列表，每项对应一个样本。
    """
    test_n = min(n_test, len(all_samples))
    print("\n" + "=" * 70)
    log(f"LP 松弛解质量评估: {test_n} 个样本")
    print("=" * 70)

    x_LP_list, x_true_list = [], []

    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        log(f"  样本 {i + 1}/{test_n}，pd_data shape={pd_data.shape}")

        try:
            lambda_val = dual_predictor.predict(pd_data)
            x_LP = solve_global_LP_relaxation(ppc, pd_data, T_DELTA, trainers, lambda_val)
        except Exception as e:
            log(f"    LP 求解失败: {e}")
            continue

        x_true = _extract_true_solution(sample, x_LP.shape)

        x_LP_rounded = (x_LP >= 0.5).astype(int)
        hamming  = int(np.sum(x_LP_rounded != x_true.astype(int)))
        integ    = float(np.mean(np.minimum(x_LP, 1.0 - x_LP)))
        accuracy = float(np.mean(x_LP_rounded == x_true.astype(int))) * 100
        log(f"    整数性间隙={integ:.4f}  Hamming={hamming}  四舍五入精度={accuracy:.1f}%")

        x_LP_list.append(x_LP)
        x_true_list.append(x_true)

    print("\n" + "=" * 70)
    if x_LP_list:
        mean_hamming = np.mean([int(np.sum((x_LP_list[i] >= 0.5).astype(int)
                                           != x_true_list[i].astype(int)))
                                for i in range(len(x_LP_list))])
        log(f"LP 评估完成: 平均 Hamming 距离 = {mean_hamming:.1f} bits")
        print("=" * 70)
        plot_lp_vs_true(x_LP_list, x_true_list, fig_dir, CASE_NAME)

    return list(zip(x_LP_list, x_true_list))


def run_fp_test(ppc, all_samples: list, dual_predictor, trainers: dict,
                T_DELTA: float, n_test: int) -> list:
    """对多个样本运行可行性泵并汇总结果。"""
    test_n = min(n_test, len(all_samples))
    print("\n" + "=" * 70)
    log(f"可行性泵测试: {test_n} 个样本")
    print("=" * 70)

    results = []
    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        log(f"  样本 {i + 1}/{test_n}，pd_data shape={pd_data.shape}")
        try:
            x_result, success = recover_integer_solution(
                pd_data, trainers, dual_predictor, ppc, T_DELTA,
                verbose=True,
            )
        except Exception as e:
            log(f"    异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((i, False, None))
            continue

        status = "成功" if success else "失败"
        log(f"    可行性泵结果: {status}")
        results.append((i, success, x_result))

    n_success = sum(1 for _, s, _ in results if s)
    print("\n" + "=" * 70)
    log(f"可行性泵完成: {n_success}/{test_n} 样本找到可行解")
    print("=" * 70)
    return results


def test_surrogate(ppc, all_samples: list, T_DELTA: float,
                   model_dir: str, unit_ids, fig_dir: Path) -> None:
    """加载 surrogate 模型，打印参数摘要，绘图，可选运行 FP。"""
    print("\n" + "=" * 70)
    log(f"加载 surrogate 模型: {model_dir}")
    print("=" * 70)

    if not Path(model_dir).exists():
        log(f"错误: 模型目录不存在: {model_dir}")
        log("请先运行 run_training.py 生成模型，或修改 MODEL_DIR 配置")
        sys.exit(1)

    dual_predictor, trainers = load_trained_models(
        ppc, all_samples, T_DELTA,
        load_dir=model_dir,
        unit_ids=unit_ids,
    )

    log(f"已加载 {len(trainers)} 个机组的代理约束模型")
    print_surrogate_results(trainers, all_samples[:TEST_SAMPLES])

    # 绘制 surrogate 分析图
    print("\n" + "=" * 70)
    log("生成 surrogate 分析图表...")
    print("=" * 70)
    plot_surrogate_analysis(trainers, all_samples, fig_dir, CASE_NAME)

    # LP 松弛解质量评估（在可行性泵之前）
    print("\n" + "=" * 70)
    log("LP 松弛解质量评估（FP 前置分析）...")
    print("=" * 70)
    run_lp_compare_test(ppc, all_samples, dual_predictor, trainers,
                        T_DELTA, TEST_SAMPLES, fig_dir)

    if RUN_FP:
        fp_results = run_fp_test(
            ppc, all_samples, dual_predictor, trainers, T_DELTA, TEST_SAMPLES
        )
        plot_fp_results(fp_results, fig_dir, CASE_NAME)


def _print_bcd_stats(agent) -> None:
    """打印 BCD agent 模型参数统计摘要（辅助函数）。"""
    print("\n" + "=" * 70)
    log("BCD 模型参数统计")
    print("=" * 70)

    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        total_params = sum(p.numel() for p in agent.theta_net.parameters())
        log(f"  theta_net 参数量: {total_params:,}")

    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        total_params = sum(p.numel() for p in agent.zeta_net.parameters())
        log(f"  zeta_net  参数量: {total_params:,}")

    if hasattr(agent, 'theta_values') and agent.theta_values:
        vals = np.array(list(agent.theta_values.values()), dtype=float)
        log(f"  theta_values 数量: {len(vals)}，"
            f"均值={vals.mean():.4f}，标准差={vals.std():.4f}，"
            f"范围=[{vals.min():.4f}, {vals.max():.4f}]")

    if hasattr(agent, 'zeta_values') and agent.zeta_values:
        vals = np.array(list(agent.zeta_values.values()), dtype=float)
        log(f"  zeta_values  数量: {len(vals)}，"
            f"均值={vals.mean():.4f}，标准差={vals.std():.4f}，"
            f"范围=[{vals.min():.4f}, {vals.max():.4f}]")


def _load_bcd_agent(ppc, data_file: Path, bcd_model_path: str,
                    MAX_SAMPLES, T_DELTA: float):
    """加载 BCD 数据并恢复 Agent_NN_BCD 模型参数，返回 agent。"""
    model_path = Path(bcd_model_path)
    if not model_path.exists():
        log(f"错误: BCD 模型文件不存在: {bcd_model_path}")
        log("请先运行 run_training.py MODE='bcd'/'both' 生成模型，或修改 BCD_MODEL_PATH 配置")
        sys.exit(1)

    log(f"通过 load_active_set_from_json 加载数据: {data_file.name}")
    all_samples_bcd = load_active_set_from_json(str(data_file))
    if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
        all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
    log(f"  使用 {len(all_samples_bcd)} 个 BCD 样本")

    agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)
    agent.load_model_parameters(str(model_path))
    log("BCD 模型加载成功")
    return agent


def test_bcd(ppc, data_file: Path, bcd_model_path: str,
             MAX_SAMPLES, T_DELTA: float, fig_dir: Path) -> None:
    """加载 BCD 模型，初始化 agent，报告参数统计，绘图。"""
    print("\n" + "=" * 70)
    log(f"加载 BCD 模型: {bcd_model_path}")
    print("=" * 70)

    agent = _load_bcd_agent(ppc, data_file, bcd_model_path, MAX_SAMPLES, T_DELTA)
    _print_bcd_stats(agent)

    print("\n" + "=" * 70)
    log("生成 BCD 分析图表...")
    print("=" * 70)
    plot_bcd_analysis(agent, fig_dir, CASE_NAME)

    log("BCD 测试完成（如需评估解质量，请使用 MODE='both' + RUN_FP=True）")


def test_both(ppc, data_file: Path, all_samples: list, T_DELTA: float,
              model_dir: str, bcd_model_path: str,
              MAX_SAMPLES, unit_ids, fig_dir: Path) -> None:
    """联合加载 BCD + surrogate 模型，以全体代理约束评估解质量，可选运行 FP。

    流程：
      1. 加载 BCD 模型 → 打印参数统计
      2. 加载 surrogate 全体代理约束模型 → 打印约束摘要
      3. 生成各自分析图 + 联合表征图
      4. （可选）使用全体代理约束运行可行性泵

    Args:
        ppc:            PyPower 案例字典。
        data_file:      JSON 数据文件路径（BCD 格式，含 unit_commitment_matrix）。
        all_samples:    v3 格式样本列表（已预处理）。
        T_DELTA:        时间间隔。
        model_dir:      surrogate 模型目录（绝对路径）。
        bcd_model_path: BCD .pth 文件路径（绝对路径）。
        MAX_SAMPLES:    BCD 数据最多使用样本数。
        unit_ids:       机组 ID 列表（None = 全部）。
        fig_dir:        图像输出目录。
    """
    print("\n" + "=" * 70)
    log("模式: both — 联合评估 BCD 神经网络 + 全体 V3 代理约束")
    print("=" * 70)

    # ── Step 1: 加载 BCD 模型 ──────────────────────────────
    log("── Step 1/4  加载 BCD 模型")
    agent = _load_bcd_agent(ppc, data_file, bcd_model_path, MAX_SAMPLES, T_DELTA)
    _print_bcd_stats(agent)

    # ── Step 2: 加载 surrogate 全体代理约束模型 ───────────
    log("── Step 2/4  加载全体代理约束模型")
    if not Path(model_dir).exists():
        log(f"错误: surrogate 模型目录不存在: {model_dir}")
        log("请先运行 run_training.py 生成模型，或修改 MODEL_DIR 配置")
        sys.exit(1)

    dual_predictor, trainers = load_trained_models(
        ppc, all_samples, T_DELTA,
        load_dir=model_dir,
        unit_ids=unit_ids,
    )
    log(f"已加载 {len(trainers)} 个机组的代理约束模型（全体约束）")
    print_surrogate_results(trainers, all_samples[:TEST_SAMPLES])

    # ── Step 3: 绘图 ───────────────────────────────────────
    log("── Step 3/4  生成分析图表")
    print("\n" + "=" * 70)
    log("生成 surrogate 分析图...")
    print("=" * 70)
    plot_surrogate_analysis(trainers, all_samples, fig_dir, CASE_NAME)

    print("\n" + "=" * 70)
    log("生成 BCD 分析图...")
    print("=" * 70)
    plot_bcd_analysis(agent, fig_dir, CASE_NAME)

    print("\n" + "=" * 70)
    log("生成 BCD-Surrogate 联合约束表征图...")
    print("=" * 70)
    plot_both_analysis(agent, trainers, fig_dir, CASE_NAME)

    # ── Step 4: LP 评估 + 可行性泵（全体代理约束） ────────
    log("── Step 4/4  LP 松弛解质量评估（FP 前置分析）")
    run_lp_compare_test(ppc, all_samples, dual_predictor, trainers,
                        T_DELTA, TEST_SAMPLES, fig_dir)

    if RUN_FP:
        log("── Step 4/4  以全体代理约束运行可行性泵")
        fp_results = run_fp_test(
            ppc, all_samples, dual_predictor, trainers, T_DELTA, TEST_SAMPLES
        )
        plot_fp_results(fp_results, fig_dir, CASE_NAME)
    else:
        log("── Step 4/4  跳过可行性泵（RUN_FP=False）")


# ──────────────────────── 主函数 ────────────────────────


def main():
    start_time = time.time()

    print("=" * 70)
    print(f"测试脚本  模式: {MODE}  算例: {CASE_NAME}")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────
    MAX_SAMPLES = None   # 最多使用多少个样本（None=全部）
    T_DELTA     = 1.0
    UNIT_IDS    = None   # None = 所有机组；或如 [0, 1, 2]

    result_dir = Path(__file__).parent / 'result'
    fig_dir    = result_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    log(f"图像输出目录: {fig_dir}")

    # ── 加载 PyPower 案例 ────────────────────────────────
    log(f"加载 PyPower 案例: {CASE_NAME}")
    ppc_map = {
        'case14': pypower.case14.case14,
        'case30': pypower.case30.case30,
        'case39': pypower.case39.case39,
    }
    if CASE_NAME not in ppc_map:
        print(f"未知案例: {CASE_NAME}，可选: {list(ppc_map)}")
        sys.exit(1)
    ppc = ppc_map[CASE_NAME]()
    n_units = ppc['gen'].shape[0]
    n_buses = ppc['bus'].shape[0]
    log(f"  {n_units} 机组，{n_buses} 节点")

    # ── 查找数据文件 ─────────────────────────────────────
    data_file = pick_data_file(result_dir, CASE_NAME)
    if data_file is None:
        log(f"错误: 在 {result_dir} 中未找到 {CASE_NAME} 的 JSON 数据文件。")
        log("请先运行 ActiveSetLearner 生成数据，或在 result/ 目录下放置")
        log(f"命名为 active_sets_{CASE_NAME}_*.json 的数据文件后重试。")
        sys.exit(1)

    # ── 执行模式分支 ─────────────────────────────────────
    try:
        if MODE == 'surrogate':
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            # 将相对路径解析为绝对路径
            model_dir = str((Path(__file__).parent / MODEL_DIR).resolve())
            test_surrogate(ppc, all_samples, T_DELTA, model_dir, UNIT_IDS, fig_dir)

        elif MODE == 'bcd':
            bcd_path = str((Path(__file__).parent / BCD_MODEL_PATH).resolve())
            test_bcd(ppc, data_file, bcd_path, MAX_SAMPLES, T_DELTA, fig_dir)

        elif MODE == 'both':
            # both 模式需要同时加载 v3 格式样本（供 surrogate 用）
            all_samples = load_json_data(data_file)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"  截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(f"  样本 T={T_from_data}，使用 {len(all_samples)} 个样本")

            model_dir  = str((Path(__file__).parent / MODEL_DIR).resolve())
            bcd_path   = str((Path(__file__).parent / BCD_MODEL_PATH).resolve())
            test_both(ppc, data_file, all_samples, T_DELTA,
                      model_dir, bcd_path, MAX_SAMPLES, UNIT_IDS, fig_dir)

        else:
            log(f"未知模式: '{MODE}'，可选: 'surrogate' | 'bcd' | 'both'")
            sys.exit(1)

    except Exception as e:
        log(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 汇总 ─────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    log(f"完成！模式={MODE}，耗时 {total_time / 60:.1f} 分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
