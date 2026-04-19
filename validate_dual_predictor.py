#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在固定数据集上评估已训练的 ``dual_predictor.pth``（与训练时相同的真值构造）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from case_registry import get_case_ppc  # noqa: E402
from dataset_json_utils import load_v3_active_set_json  # noqa: E402
from uc_NN_subproblem import DualVariablePredictorTrainer  # noqa: E402

DEFAULT_CASE = "case118"
DEFAULT_JSON = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025"
    "_price_only_clipped.json"
)
DEFAULT_T_DELTA = 1.0


def _latest_case118_model_dir(surrogate_root: Path) -> Optional[Path]:
    candidates = sorted(
        p
        for p in surrogate_root.glob("subproblem_models_case118_*")
        if p.is_dir() and (p / "dual_predictor.pth").is_file()
    )
    return candidates[-1] if candidates else None


def _collect_predictions(
    trainer: DualVariablePredictorTrainer,
    samples: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    for sid, sample in enumerate(samples):
        p = np.asarray(trainer.predict(sample), dtype=np.float64).reshape(-1)
        t = np.asarray(trainer.lambda_true[sid], dtype=np.float64).reshape(-1)
        preds.append(p)
        truths.append(t)
    return np.stack(preds, axis=0), np.stack(truths, axis=0)


def _metrics_block(y_hat: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    diff = y_hat - y
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    rel_mae = float(mae / (np.mean(np.abs(y)) + 1e-12))
    # 分样本 MSE（与 evaluate_trained_models 口径一致）
    per_sample_mse = np.mean(diff ** 2, axis=1)
    return {
        "n": int(y.shape[0]),
        "dim": int(y.shape[1]),
        "mse_scalar_mean": mse,
        "mae_scalar_mean": mae,
        "rmse": rmse,
        "rel_mae_to_mean_abs_y": rel_mae,
        "per_sample_mse_mean": float(np.mean(per_sample_mse)),
        "per_sample_mse_median": float(np.median(per_sample_mse)),
        "per_sample_mse_p90": float(np.percentile(per_sample_mse, 90)),
        "per_sample_mse_max": float(np.max(per_sample_mse)),
    }


def _r2_per_dim(y: np.ndarray, y_hat: np.ndarray) -> Optional[Dict[str, float]]:
    try:
        from sklearn.metrics import r2_score  # type: ignore
    except Exception:
        return None
    d = y.shape[1]
    scores = np.array([r2_score(y[:, j], y_hat[:, j]) for j in range(d)], dtype=float)
    return {
        "r2_mean": float(np.mean(scores)),
        "r2_median": float(np.median(scores)),
        "r2_min": float(np.min(scores)),
        "r2_p10": float(np.percentile(scores, 10)),
        "r2_p90": float(np.percentile(scores, 90)),
    }


def _mean_baseline_metrics(y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    mu = y_train.mean(axis=0, keepdims=True)
    hat = np.repeat(mu, y_val.shape[0], axis=0)
    diff = hat - y_val
    return {
        "val_mse_mean_column": float(np.mean(diff ** 2)),
        "val_mae_mean_column": float(np.mean(np.abs(diff))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="评估 dual_predictor.pth")
    ap.add_argument("--case", default=DEFAULT_CASE)
    ap.add_argument(
        "--active-set-json",
        default=DEFAULT_JSON,
        help="与训练相同的 active-set JSON",
    )
    ap.add_argument(
        "--model-dir",
        default="",
        help="含 dual_predictor.pth 的目录；留空则自动选最新 case118 子目录",
    )
    ap.add_argument("--t-delta", type=float, default=DEFAULT_T_DELTA)
    ap.add_argument("--val-fraction", type=float, default=0.2, help="留出验证比例；0 表示不划分")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help=">0 时只使用前 N 条（快速试跑）")
    ap.add_argument("--device", default="", help="cuda / cpu；默认自动")
    ap.add_argument("--json-out", default="", help="可选：将指标写入该路径")
    args = ap.parse_args()

    json_path = (ROOT / args.active_set_json).resolve()
    if not json_path.is_file():
        print(f"[错误] 找不到 active-set 文件: {json_path}", flush=True)
        return 1

    if args.model_dir:
        model_dir = (ROOT / args.model_dir).resolve()
    else:
        surrogate_root = ROOT / "result" / "surrogate_models"
        picked = _latest_case118_model_dir(surrogate_root)
        if picked is None:
            print(f"[错误] 在 {surrogate_root} 下未找到 subproblem_models_case118_*/dual_predictor.pth", flush=True)
            return 1
        model_dir = picked

    ckpt = model_dir / "dual_predictor.pth"
    if not ckpt.is_file():
        print(f"[错误] 找不到权重: {ckpt}", flush=True)
        return 1

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    print(f"[验证] case={args.case}", flush=True)
    print(f"[验证] active_set_json={json_path}", flush=True)
    print(f"[验证] model_dir={model_dir}", flush=True)
    print(f"[验证] checkpoint={ckpt}", flush=True)

    ppc = get_case_ppc(args.case)
    samples = load_v3_active_set_json(json_path, announce=print)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: int(args.max_samples)]
        print(f"[验证] 截断为 max_samples={len(samples)}", flush=True)

    trainer = DualVariablePredictorTrainer(
        ppc,
        samples,
        args.t_delta,
        device=device,
    )
    trainer.load(str(ckpt))

    y_hat, y_true = _collect_predictions(trainer, samples)
    full = _metrics_block(y_hat, y_true)
    full["r2_per_dim"] = _r2_per_dim(y_true, y_hat)

    print("\n" + "=" * 60, flush=True)
    print("对偶预测器数值验证（全数据）", flush=True)
    print("=" * 60, flush=True)
    for k in ("n", "dim", "mse_scalar_mean", "rmse", "mae_scalar_mean", "rel_mae_to_mean_abs_y"):
        print(f"  {k}: {full[k]}", flush=True)
    print(
        f"  per_sample_mse: mean={full['per_sample_mse_mean']:.6f}, "
        f"median={full['per_sample_mse_median']:.6f}, "
        f"p90={full['per_sample_mse_p90']:.6f}, max={full['per_sample_mse_max']:.6f}",
        flush=True,
    )
    if full["r2_per_dim"]:
        r2 = full["r2_per_dim"]
        print(
            f"  R2(按输出维): mean={r2['r2_mean']:.4f}, median={r2['r2_median']:.4f}, "
            f"min={r2['r2_min']:.4f}",
            flush=True,
        )

    out_payload: Dict[str, Any] = {"full_data": full, "model_dir": str(model_dir), "json": str(json_path)}

    vf = float(args.val_fraction)
    if vf > 0 and len(samples) >= 5:
        rng = np.random.default_rng(int(args.seed))
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(samples) * vf)))
        val_idx = np.sort(idx[:n_val])
        train_idx = np.sort(idx[n_val:])
        y_tr, y_va = y_true[train_idx], y_true[val_idx]
        p_tr, p_va = y_hat[train_idx], y_hat[val_idx]
        train_m = _metrics_block(p_tr, y_tr)
        val_m = _metrics_block(p_va, y_va)
        naive = _mean_baseline_metrics(y_tr, y_va)
        print("\n" + "=" * 60, flush=True)
        print(f"留出验证 val_fraction={vf} seed={args.seed}", flush=True)
        print("=" * 60, flush=True)
        print(f"  train n={train_m['n']}  MSE={train_m['mse_scalar_mean']:.6f}  MAE={train_m['mae_scalar_mean']:.6f}", flush=True)
        print(f"  val   n={val_m['n']}    MSE={val_m['mse_scalar_mean']:.6f}  MAE={val_m['mae_scalar_mean']:.6f}", flush=True)
        print(
            f"  朴素基线(验证集上用训练集列均值): MSE={naive['val_mse_mean_column']:.6f}",
            flush=True,
        )
        out_payload["split"] = {
            "val_fraction": vf,
            "seed": int(args.seed),
            "train": train_m,
            "val": val_m,
            "mean_column_baseline_val_mse": naive["val_mse_mean_column"],
        }

    if args.json_out:
        out_path = (ROOT / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
        print(f"\n[验证] 已写入 {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
