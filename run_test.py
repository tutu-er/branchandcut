#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本（多模式）
- surrogate: 加载已训练的 V3 代理约束模型，输出参数摘要，可选运行可行性泵
- bcd:       加载已训练的 BCD 神经网络模型，报告参数统计

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
#
MODE      = 'surrogate'
RUN_FP    = True        # surrogate 模式：是否运行可行性泵测试
CASE_NAME = 'case30'   # 'case14' / 'case30' / 'case39'

# surrogate 模式：指定已训练模型目录（训练时输出的带时间戳路径）
MODEL_DIR = 'result/subproblem_models_case30_20240101_120000'

# bcd 模式：指定已训练 BCD 模型 .pth 文件路径
BCD_MODEL_PATH = 'result/bcd_model_case30_20240101_120000.pth'

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

if MODE == 'bcd':
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)

if RUN_FP and MODE == 'surrogate':
    try:
        from feasibility_pump import recover_integer_solution
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
    has_theta = hasattr(agent, 'theta_values') and agent.theta_values
    has_zeta  = hasattr(agent, 'zeta_values')  and agent.zeta_values

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

    if RUN_FP:
        fp_results = run_fp_test(
            ppc, all_samples, dual_predictor, trainers, T_DELTA, TEST_SAMPLES
        )
        plot_fp_results(fp_results, fig_dir, CASE_NAME)


def test_bcd(ppc, data_file: Path, bcd_model_path: str,
             MAX_SAMPLES, T_DELTA: float, fig_dir: Path) -> None:
    """加载 BCD 模型，初始化 agent，报告参数统计，绘图。"""
    print("\n" + "=" * 70)
    log(f"加载 BCD 模型: {bcd_model_path}")
    print("=" * 70)

    model_path = Path(bcd_model_path)
    if not model_path.exists():
        log(f"错误: BCD 模型文件不存在: {bcd_model_path}")
        log("请先运行 run_training.py MODE='bcd' 生成模型，或修改 BCD_MODEL_PATH 配置")
        sys.exit(1)

    log(f"通过 load_active_set_from_json 加载数据: {data_file.name}")
    all_samples_bcd = load_active_set_from_json(str(data_file))
    if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
        all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
    log(f"  使用 {len(all_samples_bcd)} 个样本")

    agent = Agent_NN_BCD(ppc, all_samples_bcd, T_DELTA)
    agent.load_model_parameters(str(model_path))
    log("BCD 模型加载成功")

    # 报告 theta / zeta 参数统计
    print("\n" + "=" * 70)
    log("BCD 模型参数统计")
    print("=" * 70)

    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        import torch
        total_params = sum(p.numel() for p in agent.theta_net.parameters())
        log(f"  theta_net 参数量: {total_params:,}")

    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        import torch
        total_params = sum(p.numel() for p in agent.zeta_net.parameters())
        log(f"  zeta_net  参数量: {total_params:,}")

    if hasattr(agent, 'theta_values') and agent.theta_values:
        n_theta = len(agent.theta_values)
        vals = list(agent.theta_values.values())
        log(f"  theta_values 数量: {n_theta}，"
            f"均值={np.mean(vals):.4f}，标准差={np.std(vals):.4f}")

    if hasattr(agent, 'zeta_values') and agent.zeta_values:
        n_zeta = len(agent.zeta_values)
        vals = list(agent.zeta_values.values())
        log(f"  zeta_values  数量: {n_zeta}，"
            f"均值={np.mean(vals):.4f}，标准差={np.std(vals):.4f}")

    # 绘制 BCD 分析图
    print("\n" + "=" * 70)
    log("生成 BCD 分析图表...")
    print("=" * 70)
    plot_bcd_analysis(agent, fig_dir, CASE_NAME)

    log("BCD 测试完成（如需评估解质量，请使用 run_training.py MODE='both' + RUN_FP=True）")


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

        else:
            log(f"未知模式: '{MODE}'，可选: 'surrogate' | 'bcd'")
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
