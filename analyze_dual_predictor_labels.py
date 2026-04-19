#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
分析 DualVariablePredictor 的监督标签 ``lambda_pg_electricity_price``
（与 ``build_lambda_pg_electricity_price_targets`` / 训练器真值构造一致）。

用法（仓库根目录，直接运行，无需任何参数）::

    python analyze_dual_predictor_labels.py

也可在命令行覆盖默认值::

    python analyze_dual_predictor_labels.py --max-samples 50 --no-ridge

---

**训练侧可跟进**（改动 ``uc_NN_subproblem`` 中 ``_dual_predictor_trainer_*`` 时须同步 ``predict`` / ``save`` / ``load``）：

- 若少数维度方差极大或重尾：试 ``SmoothL1Loss``（Huber），或按维加权 MSE（权重与 1/(sigma_j^2+eps) 成正比）。
- 若全局尺度大、近似高斯：对输出做标准化 ``(Y-mu)/sigma``，推理反变换；checkpoint 存 ``mu``/``sigma``。
- 若 Ridge 线性基线 R^2 已很高：略增网络容量或调 LR / dropout。
- 若近零维占比高：对小幅度维度降权或分块损失，避免大值维主导 MSE。
- 若样本级离群极少但极值极大：诊断上可做 winsorize 报告；是否在训练裁剪需业务确认。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from case_registry import get_case_ppc  # noqa: E402
from dataset_json_utils import load_v3_active_set_json  # noqa: E402
from uc_NN_subproblem import (  # noqa: E402
    build_lambda_pg_electricity_price_targets,
    get_feature_vector_from_sample,
)

# ──────────────────── 内置默认配置（修改此处即可，无需改命令行） ────────────────────
DEFAULT_JSON = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025.json"
)
DEFAULT_CASE        = "case118"
DEFAULT_T_DELTA     = 1.0
DEFAULT_MAX_SAMPLES = None          # None = 全部样本；改为整数（如 50）做快速探测
DEFAULT_OUT         = "result/dual_label_analysis"
DEFAULT_NEAR_ZERO_EPS = 1e-6
DEFAULT_NO_PLOTS    = False         # True → 跳过 matplotlib 图
DEFAULT_NO_RIDGE    = False         # True → 跳过 sklearn RidgeCV


# ─────────────────────────── 统计辅助函数 ────────────────────────────────────

def _safe_skew_kurtosis(y_flat: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"skew": None, "kurtosis": None}
    try:
        from scipy import stats  # type: ignore

        out["skew"] = float(stats.skew(y_flat, bias=False))
        out["kurtosis"] = float(stats.kurtosis(y_flat, bias=False))
    except Exception:
        pass
    return out


def _ridge_baseline(
    X: np.ndarray,
    Y: np.ndarray,
) -> Optional[Dict[str, Any]]:
    try:
        from sklearn.linear_model import RidgeCV  # type: ignore
        from sklearn.metrics import r2_score  # type: ignore
    except Exception:
        return None

    alphas = np.logspace(-3, 4, 12)
    ridge = RidgeCV(alphas=alphas, fit_intercept=True)
    ridge.fit(X, Y)
    y_hat = ridge.predict(X)
    per_dim = np.array(
        [r2_score(Y[:, j], y_hat[:, j]) for j in range(Y.shape[1])],
        dtype=float,
    )
    return {
        "alpha": float(ridge.alpha_),
        "r2_mean": float(np.mean(per_dim)),
        "r2_median": float(np.median(per_dim)),
        "r2_min": float(np.min(per_dim)),
        "r2_max": float(np.max(per_dim)),
        "r2_p10": float(np.percentile(per_dim, 10)),
        "r2_p90": float(np.percentile(per_dim, 90)),
    }


def _write_per_dim_csv(
    path: Path,
    Y: np.ndarray,
    ng: int,
    T: int,
) -> None:
    """Y shape (N, ng*T); flat index k = g * T + t (C-order ravel of (ng, T))."""
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    vmin = Y.min(axis=0)
    vmax = Y.max(axis=0)
    p01 = np.percentile(Y, 1, axis=0)
    p50 = np.percentile(Y, 50, axis=0)
    p99 = np.percentile(Y, 99, axis=0)
    lines = [
        "flat_idx,gen,time,mean,std,min,max,p01,p50,p99",
    ]
    for k in range(Y.shape[1]):
        g = k // T
        t = k % T
        lines.append(
            f"{k},{g},{t},"
            f"{mean[k]:.8g},{std[k]:.8g},{vmin[k]:.8g},{vmax[k]:.8g},"
            f"{p01[k]:.8g},{p50[k]:.8g},{p99[k]:.8g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_std_heatmap(Y: np.ndarray, ng: int, T: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    std = Y.std(axis=0).reshape(ng, T)
    log_std = np.log10(std + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(log_std, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("time t")
    ax.set_ylabel("generator index g")
    ax.set_title("log10(std per (g,t)) of lambda_pg_electricity_price")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sample_norms(
    per_sample_norm: np.ndarray,
    per_sample_max_abs: np.ndarray,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(per_sample_norm, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title(r"Per-sample $\|y\|_2$")
    axes[0].set_xlabel("norm")
    axes[1].hist(per_sample_max_abs, bins=40, color="coral", edgecolor="white")
    axes[1].set_title(r"Per-sample $\max|y_i|$")
    axes[1].set_xlabel("max abs")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_magnitude_histograms.png", dpi=150)
    plt.close(fig)


# ──────────────────────────── 主函数 ─────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze dual predictor label statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", type=Path, default=DEFAULT_JSON,
                   help="Path to active-set JSON (all_samples).")
    p.add_argument("--case", type=str, default=DEFAULT_CASE,
                   help="Case name for get_case_ppc.")
    p.add_argument("--T-delta", type=float, default=DEFAULT_T_DELTA, dest="T_delta",
                   help="Horizon step for ED.")
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                   help="Use only first N samples (None = all).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory for summary.json, CSV, figures.")
    p.add_argument("--near-zero-eps", type=float, default=DEFAULT_NEAR_ZERO_EPS,
                   help="Threshold for |y| < eps share.")
    p.add_argument("--no-plots", action="store_true", default=DEFAULT_NO_PLOTS,
                   help="Skip matplotlib figures.")
    p.add_argument("--no-ridge", action="store_true", default=DEFAULT_NO_RIDGE,
                   help="Skip sklearn RidgeCV baseline.")
    args = p.parse_args()

    data_file = args.json
    if not isinstance(data_file, Path):
        data_file = Path(data_file)
    if not data_file.is_absolute():
        data_file = ROOT / data_file
    if not data_file.exists():
        print(f"Error: JSON not found: {data_file}", flush=True)
        sys.exit(1)

    out_dir = args.out
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {data_file}", flush=True)
    samples = load_v3_active_set_json(data_file, announce=print)
    if args.max_samples is not None:
        samples = samples[: max(1, int(args.max_samples))]
        print(f"  Truncated to max_samples={len(samples)}", flush=True)

    print(f"Loading ppc: {args.case}", flush=True)
    ppc = get_case_ppc(args.case)

    print("Building Y (cache or ED solve; may take a while)...", flush=True)
    Y, meta = build_lambda_pg_electricity_price_targets(ppc, samples, args.T_delta)
    N, D = Y.shape
    ng = int(meta["ng"])
    T = int(meta["T"])
    assert D == ng * T, (D, ng, T)

    y_flat = Y.reshape(-1)
    near_zero_share = float(np.mean(np.abs(y_flat) < float(args.near_zero_eps)))

    pct_vals = {
        "p01":  float(np.quantile(y_flat, 0.01)),
        "p50":  float(np.quantile(y_flat, 0.50)),
        "p99":  float(np.quantile(y_flat, 0.99)),
        "p999": float(np.quantile(y_flat, 0.999)),
    }

    per_sample_norm    = np.linalg.norm(Y, axis=1)
    per_sample_max_abs = np.max(np.abs(Y), axis=1)
    thr = float(np.quantile(per_sample_max_abs, 0.999))
    outlier_ids = [int(i) for i in np.where(per_sample_max_abs > thr)[0]]

    sk = _safe_skew_kurtosis(y_flat)

    summary: Dict[str, Any] = {
        "json_path": str(data_file),
        "case": args.case,
        "T_delta": args.T_delta,
        "near_zero_eps": args.near_zero_eps,
        "label_meta": meta,
        "Y_shape": [int(N), int(D)],
        "global": {
            "min": float(np.min(y_flat)),
            "max": float(np.max(y_flat)),
            "mean": float(np.mean(y_flat)),
            "mean_abs": float(np.mean(np.abs(y_flat))),
            "std": float(np.std(y_flat)),
            "near_zero_share": near_zero_share,
            **pct_vals,
            **sk,
        },
        "per_sample": {
            "l2_norm_mean": float(np.mean(per_sample_norm)),
            "l2_norm_p99": float(np.quantile(per_sample_norm, 0.99)),
            "max_abs_p999_threshold": thr,
            "outlier_sample_ids_p999_max_abs": outlier_ids[:50],
            "n_outliers_listed_cap50": min(len(outlier_ids), 50),
        },
    }

    X_rows: List[List[float]] = []
    for i in range(len(samples)):
        X_rows.append(list(get_feature_vector_from_sample(dict(samples[i]))))
    X = np.asarray(X_rows, dtype=float)
    summary["X_shape"] = [int(X.shape[0]), int(X.shape[1])]

    if not args.no_ridge:
        ridge = _ridge_baseline(X, Y)
        summary["ridge_cv_train_r2"] = ridge
        if ridge is not None:
            print(
                f"RidgeCV in-sample R2: mean={ridge['r2_mean']:.4f}, "
                f"median={ridge['r2_median']:.4f}, alpha={ridge['alpha']:.4g}",
                flush=True,
            )
        else:
            print("RidgeCV skipped (sklearn not available).", flush=True)
    else:
        summary["ridge_cv_train_r2"] = None

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_per_dim_csv(out_dir / "per_dim_stats.csv", Y, ng, T)
    print(f"Wrote {out_dir / 'summary.json'} and per_dim_stats.csv", flush=True)

    if not args.no_plots:
        _plot_std_heatmap(Y, ng, T, out_dir / "std_heatmap_log10.png")
        _plot_sample_norms(per_sample_norm, per_sample_max_abs, out_dir)
        print(f"Figures under {out_dir}", flush=True)

    print(
        "Global (match DualPredictor log): "
        f"shape={Y.shape}, min={summary['global']['min']:.4g}, "
        f"max={summary['global']['max']:.4g}, "
        f"mean_abs={summary['global']['mean_abs']:.4g}, "
        f"std={summary['global']['std']:.4g}",
        flush=True,
    )
    print(
        f"Label source: n_from_cache={meta['n_from_cache']}, n_from_ed={meta['n_from_ed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
