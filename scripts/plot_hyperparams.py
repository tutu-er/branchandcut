"""可视化 ActiveSetLearner 超参数与窗口大小 W 及停止条件的关系。

展示 alpha, delta, epsilon 如何通过 Beta 分布分位数决定所需采样窗口 W，
以及停止条件 R_M(W) < alpha - epsilon 的含义。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from pathlib import Path

# ---------- 中文字体设置 ----------
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"


def compute_W(alpha: float, delta: float, epsilon: float) -> int:
    """复现 ActiveSetLearner._cal_W 的逻辑，返回窗口大小 W。"""
    for n in range(1, 50000):
        k = max(int((alpha - epsilon) * n), 1)
        upper_bound = beta.ppf(1 - delta, k + 1, n - k)
        if upper_bound < alpha:
            return n
    return 50000  # fallback


# ======================== 图 1: W vs alpha（固定 delta, epsilon） ========================
def plot_W_vs_alpha(ax: plt.Axes) -> None:
    delta, epsilon = 0.01, 0.04
    alphas = np.arange(0.05, 0.85, 0.02)
    Ws = [compute_W(a, delta, epsilon) for a in alphas]

    ax.plot(alphas, Ws, "o-", markersize=3, color="tab:blue")
    ax.set_xlabel(r"$\alpha$（未见活动集概率上界）")
    ax.set_ylabel("窗口大小 $W$")
    ax.set_title(rf"$W$ vs $\alpha$  ($\delta={delta},\;\varepsilon={epsilon}$)")
    ax.grid(True, alpha=0.3)


# ======================== 图 2: W vs epsilon（固定 alpha, delta） ========================
def plot_W_vs_epsilon(ax: plt.Axes) -> None:
    alpha, delta = 0.10, 0.01
    epsilons = np.arange(0.005, alpha - 0.005, 0.005)
    Ws = [compute_W(alpha, delta, e) for e in epsilons]

    ax.plot(epsilons, Ws, "s-", markersize=3, color="tab:orange")
    ax.set_xlabel(r"$\varepsilon$（容差）")
    ax.set_ylabel("窗口大小 $W$")
    ax.set_title(rf"$W$ vs $\varepsilon$  ($\alpha={alpha},\;\delta={delta}$)")
    ax.grid(True, alpha=0.3)


# ======================== 图 3: W vs delta（固定 alpha, epsilon） ========================
def plot_W_vs_delta(ax: plt.Axes) -> None:
    alpha, epsilon = 0.10, 0.04
    deltas = np.logspace(-4, -0.3, 40)
    Ws = [compute_W(alpha, d, epsilon) for d in deltas]

    ax.semilogx(deltas, Ws, "^-", markersize=3, color="tab:green")
    ax.set_xlabel(r"$\delta$（置信度参数，越小越严格）")
    ax.set_ylabel("窗口大小 $W$")
    ax.set_title(rf"$W$ vs $\delta$  ($\alpha={alpha},\;\varepsilon={epsilon}$)")
    ax.grid(True, alpha=0.3)


# ======================== 图 4: Beta 分布 & 停止条件可视化 ========================
def plot_stopping_condition(ax: plt.Axes) -> None:
    """展示在给定 W 下，Beta 分布上界如何与 alpha 比较，形成停止条件。"""
    alpha, delta, epsilon = 0.10, 0.01, 0.04
    W = compute_W(alpha, delta, epsilon)

    # 模拟发现率从高到低的过程
    discovery_rates = np.linspace(0.0, 0.30, 200)
    upper_bounds = []
    for r in discovery_rates:
        k = max(int(r * W), 1)
        k = min(k, W - 1)
        ub = beta.ppf(1 - delta, k + 1, W - k)
        upper_bounds.append(ub)

    ax.plot(discovery_rates, upper_bounds, "-", color="tab:purple", linewidth=2,
            label=rf"Beta 上界 ($W={W}$)")
    ax.axhline(y=alpha, color="tab:red", linestyle="--", linewidth=1.5,
               label=rf"$\alpha = {alpha}$")
    ax.axvline(x=alpha - epsilon, color="tab:blue", linestyle=":", linewidth=1.5,
               label=rf"停止阈值 $\alpha - \varepsilon = {alpha - epsilon}$")

    # 标注停止区域
    ax.fill_betweenx([0, 1], 0, alpha - epsilon, alpha=0.08, color="tab:blue",
                     label="停止区域 $R_M(W) < \\alpha - \\varepsilon$")

    ax.set_xlabel(r"发现率 $R_M(W) = \mathrm{新活动集数} / W$")
    ax.set_ylabel("Beta 分布上界")
    ax.set_title("停止条件：发现率 vs Beta 上界")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 0.40)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


# ======================== 图 5: W 热力图 (alpha × epsilon) ========================
def plot_W_heatmap(ax: plt.Axes) -> None:
    delta = 0.01
    alphas = np.arange(0.05, 0.55, 0.05)
    epsilons = np.arange(0.01, 0.10, 0.01)
    W_grid = np.zeros((len(epsilons), len(alphas)))

    for i, eps in enumerate(epsilons):
        for j, alp in enumerate(alphas):
            if eps < alp:
                W_grid[i, j] = compute_W(alp, delta, eps)
            else:
                W_grid[i, j] = np.nan

    im = ax.imshow(W_grid, aspect="auto", origin="lower",
                   extent=[alphas[0], alphas[-1], epsilons[0], epsilons[-1]],
                   cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="$W$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(rf"窗口大小 $W$ 热力图 ($\delta={delta}$)")


# ======================== 图 6: 模拟 DiscoverMass 迭代过程 ========================
def plot_simulated_iterations(ax: plt.Axes) -> None:
    """模拟发现率随迭代次数单调递减的过程。"""
    alpha, epsilon = 0.10, 0.04
    threshold = alpha - epsilon

    # 模拟：发现率指数衰减
    iterations = np.arange(1, 21)
    np.random.seed(42)
    base_rate = 0.25 * np.exp(-0.18 * iterations) + np.random.normal(0, 0.005, len(iterations))
    base_rate = np.clip(base_rate, 0, 0.3)

    # 找到首次低于阈值的位置
    stop_idx = np.argmax(base_rate < threshold)
    if stop_idx == 0 and base_rate[0] >= threshold:
        stop_idx = len(iterations) - 1

    ax.bar(iterations, base_rate, color="tab:cyan", alpha=0.7, label=r"$R_M(W)$")
    ax.axhline(y=threshold, color="tab:red", linestyle="--", linewidth=1.5,
               label=rf"$\alpha - \varepsilon = {threshold}$")
    if stop_idx < len(iterations):
        ax.axvline(x=iterations[stop_idx], color="tab:orange", linestyle="-.",
                   linewidth=1.5, label=f"停止于第 {iterations[stop_idx]} 轮")

    ax.set_xlabel("迭代轮次 $M$")
    ax.set_ylabel(r"发现率 $R_M(W)$")
    ax.set_title("DiscoverMass 算法迭代模拟")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


# ======================== 主函数 ========================
def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("ActiveSetLearner 超参数与窗口大小 $W$ / 停止条件的关系", fontsize=14, y=0.98)

    plot_W_vs_alpha(axes[0, 0])
    plot_W_vs_epsilon(axes[0, 1])
    plot_W_vs_delta(axes[0, 2])
    plot_stopping_condition(axes[1, 0])
    plot_W_heatmap(axes[1, 1])
    plot_simulated_iterations(axes[1, 2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = Path("result") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hyperparams_vs_W.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图片已保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
