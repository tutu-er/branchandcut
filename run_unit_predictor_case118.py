#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 单独训练/评估 UnitPredictor（单机组 0/1 变量预测器）。

用途：
- 独立于 subproblem BCD 训练流程，快速迭代 UnitPredictor 的结构/超参；
- 支持截断样本数、指定机组列表、配置训练超参，并把模型存到 result/ 下的时间戳目录。

默认设定面向「训练集距离（MSE）压到极低」：full-batch、增大 TCN 容量、末阶段纯 MSE 精修；
脚本结束会对训练集逐机组打印 mse/mae/bce 与分类指标（可选 F1 阈值扫优）。

架构提示：``net_variant="tcn_shared_film"`` 时全程只有**共享骨干 +  joint 训练**
（``train_all_shared_joint``），日志均为 ``[UnitPredictor-shared]``，**不会**出现逐机组
``train_unit``；阶段里「选中机组」仅通过 ``unit_loss_weights`` 加权损失。若需要独立机组
网络与逐机组训练日志，请改用 ``mlp`` / ``tcn`` / ``tconv`` 等。

说明：
    本脚本不使用命令行参数；请在 `main()` 顶部显式修改配置变量以进行实验对比。
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import run_training as rt
import run_training_case118 as case118_cfg

from src.case_registry import get_case_ppc
from src.uc_NN_subproblem import SingleUnitBinaryPredictorTrainer
from src.scenario_utils import get_feature_vector_from_sample


def _parse_hidden_dims(text: str) -> list[int] | None:
    raw = [p.strip() for p in str(text).split(",") if p.strip()]
    if not raw:
        return None
    return [int(x) for x in raw]


def _parse_unit_ids(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


def _resolve_checkpoint_path(path_value: str | None) -> str | None:
    """Resolve a checkpoint file, run directory, or LATEST.txt pointer."""
    if path_value is None or str(path_value).strip() == "":
        return None
    p = Path(str(path_value).strip())
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if p.is_dir():
        latest = p / "LATEST.txt"
        if latest.is_file():
            text = latest.read_text(encoding="utf-8").strip()
            if text:
                q = Path(text)
                return str(q if q.is_absolute() else (Path(__file__).resolve().parent / q))
        return str(p / "unit_predictor.pth")
    if p.name.upper() == "LATEST.TXT" and p.is_file():
        text = p.read_text(encoding="utf-8").strip()
        if text:
            q = Path(text)
            return str(q if q.is_absolute() else (Path(__file__).resolve().parent / q))
    return str(p)


def _train_val_split(items: list, *, val_ratio: float, seed: int) -> tuple[list, list]:
    n = len(items)
    if n <= 1 or float(val_ratio) <= 0.0:
        return list(items), []
    vr = float(min(max(float(val_ratio), 0.0), 0.9))
    n_val = max(1, int(round(n * vr)))
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = set(idx[:n_val].tolist())
    train = [items[i] for i in range(n) if i not in val_idx]
    val = [items[i] for i in range(n) if i in val_idx]
    return train, val


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _bce_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """标量 BCE（按元素平均）。"""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)
    # stable: log(1+exp(z)) - y*z
    loss = np.logaddexp(0.0, z) - y * z
    return float(loss.mean())


def _prepare_eval_xy(predictor, samples: list):
    """把 samples 堆叠成 (N,D) 特征张量与 (N,ng,T) 标签数组，避免 per-sample forward 的慢路径。"""
    import torch

    if not samples:
        return (
            torch.empty((0, 0), dtype=torch.float32, device=predictor.device),
            np.zeros((0, 0, 0), dtype=float),
        )

    X = np.asarray(
        [get_feature_vector_from_sample(dict(s)) for s in samples],
        dtype=np.float32,
    )
    X_tensor = torch.tensor(X, dtype=torch.float32, device=predictor.device)

    # y：优先从 unit_commitment_matrix 取（形状一般是 (ng, T)）
    if "unit_commitment_matrix" in samples[0]:
        y_mat = np.stack(
            [np.asarray(s["unit_commitment_matrix"], dtype=float) for s in samples],
            axis=0,
        )  # (N, ng, T)
    else:
        ng = int(getattr(predictor, "ng", len(getattr(predictor, "unit_ids", [])) or 0))
        T = int(getattr(predictor, "T", 0))
        y_mat = np.zeros((len(samples), ng, T), dtype=float)
        for i, s in enumerate(samples):
            if "active_set" in s and s["active_set"] is not None:
                for item in s["active_set"]:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                        g, t = item[0]
                        if 0 <= int(g) < ng and 0 <= int(t) < T:
                            y_mat[i, int(g), int(t)] = float(item[1])

    return X_tensor, y_mat


def _eval_unit_predictor(predictor, samples: list, unit_id: int, threshold: float = 0.5) -> dict:
    if not samples:
        return {"n": 0}
    import torch

    X_tensor, y_mat = _prepare_eval_xy(predictor, samples)
    y_arr = np.asarray(y_mat[:, int(unit_id), :], dtype=float)  # (N, T)

    net = predictor.get_network(unit_id)
    with torch.no_grad():
        logits_t = net(X_tensor)  # (N, T)
        probs_t = torch.sigmoid(logits_t)
    logits_arr = logits_t.detach().cpu().numpy()
    probs = probs_t.detach().cpu().numpy()
    pred = (probs >= float(threshold)).astype(float)

    tp = float(np.sum((pred == 1.0) & (y_arr == 1.0)))
    fp = float(np.sum((pred == 1.0) & (y_arr == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y_arr == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y_arr == 0.0)))
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)

    return {
        "n": int(y_arr.size),
        "bce": _bce_from_logits(logits_arr, y_arr),
        "mse": float(np.mean((probs - y_arr) ** 2)),
        "mae": float(np.mean(np.abs(probs - y_arr))),
        "tv": float(np.mean(np.abs(probs[:, 1:] - probs[:, :-1]))) if probs.shape[1] > 1 else 0.0,
        "y_tv": float(np.mean(np.abs(y_arr[:, 1:] - y_arr[:, :-1]))) if y_arr.shape[1] > 1 else 0.0,
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pos_rate": float(y_arr.mean()),
    }


def _best_threshold_for_f1(predictor, samples: list, unit_id: int) -> tuple[float, dict]:
    """在验证集上网格搜索阈值，返回 (best_thr, metrics_at_best_thr)。"""
    best_thr = 0.5
    best_f1 = -1.0
    best_m = {}
    for thr in np.linspace(0.05, 0.95, 19):
        m = _eval_unit_predictor(predictor, samples, unit_id=unit_id, threshold=float(thr))
        f1 = float(m.get("f1", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_m = m
    return best_thr, best_m


def _pick_low_tv_units(
    predictor,
    samples: list,
    unit_ids: list[int],
    *,
    tv_threshold: float = 0.02,
    tv_floor_scale: float = 0.8,
) -> list[int]:
    picked: list[int] = []
    for g in unit_ids:
        m = _eval_unit_predictor(predictor, samples, unit_id=int(g), threshold=0.5)
        y_tv = float(m.get("y_tv", 0.0))
        p_tv = float(m.get("tv", 0.0))
        # 仅在真值确实存在启停变化时才要求预测也有变化，否则允许常数（比如永远开/关的机组）
        if y_tv >= float(tv_threshold) and p_tv < float(tv_floor_scale) * y_tv:
            picked.append(int(g))
    return picked


def _pick_high_mse_units(predictor, samples: list, unit_ids: list[int], *, mse_threshold: float = 0.05) -> list[int]:
    picked: list[int] = []
    for g in unit_ids:
        m = _eval_unit_predictor(predictor, samples, unit_id=int(g), threshold=0.5)
        if float(m.get("mse", 1e9)) > float(mse_threshold):
            picked.append(int(g))
    return picked


def _make_unit_loss_weights_from_val_mse(predictor, val_samples: list, unit_ids: list[int], *, power: float = 1.0):
    """根据 val MSE 生成 (ng,) 的权重向量：越差的机组权重越大。mean=1 归一化。"""
    ng = int(getattr(predictor, "ng", len(unit_ids)))
    w = np.ones(int(ng), dtype=float)
    if not val_samples:
        return w.tolist()
    mses: dict[int, float] = {}
    for g in unit_ids:
        m = _eval_unit_predictor(predictor, val_samples, unit_id=int(g), threshold=0.5)
        mses[int(g)] = float(m.get("mse", 0.0))
    if not mses:
        return w.tolist()
    base = float(np.median(list(mses.values()))) + 1e-8
    for g, mse in mses.items():
        w[int(g)] = float(max(mse / base, 0.25)) ** float(power)
    w = w / max(float(w.mean()), 1e-12)
    return w.tolist()


def _make_unit_loss_weights_from_selected(predictor, selected_units: list[int], *, boost: float = 3.0):
    """给 selected_units 更大权重，用于 shared joint 训练时等效“补足 epoch”。"""
    ng = int(getattr(predictor, "ng", max(selected_units) + 1 if selected_units else 0))
    w = np.ones(int(ng), dtype=float)
    for g in selected_units:
        if 0 <= int(g) < ng:
            w[int(g)] = float(boost)
    w = w / max(float(w.mean()), 1e-12)
    return w.tolist()


def _reset_unit_network(predictor, unit_id: int) -> None:
    """删除指定机组的网络/优化器，让 trainer 在下次训练时重建（用于跳出局部最优）。"""
    try:
        g = int(unit_id)
    except Exception:
        return
    if hasattr(predictor, "networks") and isinstance(getattr(predictor, "networks"), dict):
        predictor.networks.pop(g, None)
    if hasattr(predictor, "optimizers") and isinstance(getattr(predictor, "optimizers"), dict):
        predictor.optimizers.pop(g, None)
    if hasattr(predictor, "_ensure_network"):
        predictor._ensure_network(g)


def _run_train_set_per_unit_report(
    predictor,
    train_samples: list,
    *,
    unit_ids: list[int],
    do_threshold_sweep: bool = True,
) -> None:
    """在训练集上对每个机组输出距离指标与（可选）F1 最优阈值。"""
    if not train_samples or not unit_ids:
        print("[train_eval] empty train_samples or unit_ids", flush=True)
        return

    print("=" * 72, flush=True)
    suffix = "；含 F1 阈值扫优" if do_threshold_sweep else ""
    print(f"训练集效果汇总（各机组）| mse/mae/bce 与 thr=0.5 分类指标{suffix}", flush=True)
    print("=" * 72, flush=True)

    rows = []
    for g in unit_ids:
        g_int = int(g)
        m05 = _eval_unit_predictor(predictor, train_samples, unit_id=g_int, threshold=0.5)
        row = {
            "unit": g_int,
            "mse": float(m05.get("mse", 0.0)),
            "mae": float(m05.get("mae", 0.0)),
            "bce": float(m05.get("bce", 0.0)),
            "tv": float(m05.get("tv", 0.0)),
            "acc_05": float(m05.get("acc", 0.0)),
            "f1_05": float(m05.get("f1", 0.0)),
            "thr_best": 0.5,
            "f1_best": float(m05.get("f1", 0.0)),
        }
        if do_threshold_sweep:
            thr_b, m_b = _best_threshold_for_f1(predictor, train_samples, unit_id=g_int)
            row["thr_best"] = float(thr_b)
            row["f1_best"] = float(m_b.get("f1", 0.0))
        rows.append(row)

    mse_list = [r["mse"] for r in rows]
    print(
        f"  训练集样本数={len(train_samples)} | "
        f"机组数={len(rows)} | "
        f"MSE mean={float(np.mean(mse_list)):.6f} median={float(np.median(mse_list)):.6f} "
        f"max={float(np.max(mse_list)):.6f}",
        flush=True,
    )
    print("-" * 72, flush=True)
    hdr = "unit |     mse |     mae |     bce |      tv |  acc@.5 |   f1@.5"
    if do_threshold_sweep:
        hdr += " | thr_best |  f1_best"
    print(hdr, flush=True)
    print("-" * 72, flush=True)
    for r in rows:
        line = (
            f"{r['unit']:4d} | {r['mse']:7.5f} | {r['mae']:7.5f} | {r['bce']:7.5f} | "
            f"{r['tv']:7.5f} | {r['acc_05']:7.4f} | {r['f1_05']:7.4f}"
        )
        if do_threshold_sweep:
            line += f" | {r['thr_best']:8.3f} | {r['f1_best']:7.4f}"
        print(line, flush=True)
    print("=" * 72, flush=True)


def main() -> None:
    # ── 实验配置（请在这里显式修改；不使用命令行参数）────────────────────
    # 目标：全量训练数据上把训练集 MSE 压到极低；full-batch + 末段纯 MSE（Stage F）长跑精修。
    # 数据截断：None=不截断；正整数=只取前 N 个样本
    max_samples: int | None = None
    # 是否使用“全部样本作为训练集”（不切分验证集）
    use_all_samples_as_train = True
    # 训练哪些机组：None=训练全部机组（推荐，便于统一评估）
    unit_ids: list[int] | None = None
    # 继续训练：指向已有 checkpoint；None=从头训练
    load_path: str | None = None

    # 训练超参（用于 staged training 的每阶段默认 lr 等）
    lr = 1e-3
    weight_decay = 8e-5
    batch_strategy = "full-batch"
    batch_size = 64  # mini-batch 时生效；full-batch 下有效 batch=n_samples
    shuffle = True

    # LR scheduler：末段纯 MSE 需持续减小 lr
    enable_scheduler = True
    scheduler_patience = 15
    scheduler_factor = 0.5
    min_lr = 1e-7

    # 网络结构：略增容量、略降 dropout，利于训练集拟合
    # tcn_shared_film：全体机组共享一套参数，train_all 内部只跑 shared joint（无逐机组 train_unit）
    net_variant = "tcn_shared_film"  # "mlp" | "resmlp" | "tconv" | "tcn" | "tcn_shared_film"
    hidden_dims = _parse_hidden_dims("256,128")  # 仅对 mlp 生效
    resmlp_width = 512
    resmlp_depth = 4
    tconv_channels = 64
    tconv_depth = 4
    tcn_channels = 160
    tcn_depth = 9
    dropout = 0.02

    # staged training：
    # A–E：课程学习 + 难点机组加权；E 在无 val 时用当前训练误差生成 unit 权重（见主循环）。
    # F：几乎纯 MSE（极小 transition），去掉 floor/binarize，压低训练集距离损失。
    stages = [
        {
            "name": "A_align_mse",
            "epochs": 420,
            "per_unit": None,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.10,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.32,
            "loss_weight_binarize": 0.025,
            "loss_weight_std_floor": 0.20,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.26,
            "tv_floor_scale": 0.9,
        },
        {
            "name": "B_tv_floor_auto",
            "epochs": 160,
            "per_unit": "auto_low_tv",
            "enable_pos_weight": False,
            "loss_weight_bce": 0.0,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.40,
            "loss_weight_binarize": 0.04,
            "loss_weight_std_floor": 0.26,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.36,
            "tv_floor_scale": 0.95,
            "selected_unit_boost": 5.0,
        },
        {
            "name": "C_high_mse_finetune",
            "epochs": 180,
            "per_unit": "auto_high_mse",
            "learning_rate": 2e-3,
            "enable_scheduler": False,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.18,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.50,
            "loss_weight_binarize": 0.04,
            "loss_weight_std_floor": 0.30,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.42,
            "tv_floor_scale": 0.95,
            "high_mse_threshold": 0.05,
            "selected_unit_boost": 5.0,
        },
        {
            "name": "D_reset_and_retrain_high_mse",
            "epochs": 180,
            "per_unit": "auto_high_mse",
            "reset_network": True,
            "learning_rate": 3e-3,
            "enable_scheduler": False,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.22,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.50,
            "loss_weight_binarize": 0.04,
            "loss_weight_std_floor": 0.26,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.42,
            "tv_floor_scale": 0.95,
            "high_mse_threshold": 0.05,
            "selected_unit_boost": 4.0,
        },
        {
            "name": "E_weighted_joint",
            "epochs": 320,
            "per_unit": None,
            "learning_rate": 1.8e-3,
            "enable_scheduler": True,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.18,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.52,
            "loss_weight_binarize": 0.04,
            "loss_weight_std_floor": 0.28,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.42,
            "tv_floor_scale": 0.95,
            "use_unit_loss_weights": True,
            "unit_loss_weight_power": 1.45,
        },
        {
            "name": "F_pure_mse_polish",
            "epochs": 720,
            "per_unit": None,
            "learning_rate": 4e-4,
            "enable_scheduler": True,
            "scheduler_patience": 25,
            "enable_pos_weight": False,
            "loss_weight_bce": 0.0,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.015,
            "loss_weight_binarize": 0.0,
            "loss_weight_std_floor": 0.0,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.0,
            "tv_floor_scale": 0.95,
            "shuffle": False,
        },
    ]
    pos_weight_clip = 20.0

    # 评估：train/val 切分
    seed = 42
    val_ratio = 0.2  # 0=不做验证
    print_all_validation = False
    validation_top_k = 12
    target_mse = 0.05
    train_eval_threshold_sweep = True
    out_dir_override: str | None = None

    # CLI overrides. Leaving an option unset preserves the script-top config
    # above, so this file remains usable in the old "edit main()" style.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-sets", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unit-ids", type=str, default=None, help="comma-separated unit ids; default keeps script config")
    parser.add_argument("--load-path", type=str, default=None, help="unit_predictor.pth, a run dir, or a LATEST.txt file")
    parser.add_argument("--eval-only", action="store_true", help="load and evaluate only; skip all training stages and saving")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-all-validation", action="store_true")
    parser.add_argument("--validation-top-k", type=int, default=None)
    parser.add_argument("--target-mse", type=float, default=None)
    parser.add_argument("--no-threshold-sweep", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--net", choices=("mlp", "resmlp", "tconv", "tcn", "tcn_shared_film"), default=None)
    parser.add_argument("--hidden-dims", type=str, default=None)
    parser.add_argument("--tcn-channels", type=int, default=None)
    parser.add_argument("--tcn-depth", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-all-samples-as-train", dest="use_all_samples_as_train", action="store_true", default=None)
    parser.add_argument("--train-val-split", dest="use_all_samples_as_train", action="store_false")
    args = parser.parse_args()

    if args.max_samples is not None:
        max_samples = args.max_samples
    if args.unit_ids is not None:
        unit_ids = _parse_unit_ids(args.unit_ids)
    if args.load_path is not None:
        load_path = _resolve_checkpoint_path(args.load_path)
    if args.eval_only:
        stages = []
    if args.val_ratio is not None:
        val_ratio = args.val_ratio
    if args.seed is not None:
        seed = args.seed
    if args.print_all_validation:
        print_all_validation = True
    if args.validation_top_k is not None:
        validation_top_k = args.validation_top_k
    if args.target_mse is not None:
        target_mse = args.target_mse
    if args.no_threshold_sweep:
        train_eval_threshold_sweep = False
    if args.out_dir is not None:
        out_dir_override = args.out_dir
    if args.net is not None:
        net_variant = args.net
    if args.hidden_dims is not None:
        hidden_dims = _parse_hidden_dims(args.hidden_dims)
    if args.tcn_channels is not None:
        tcn_channels = args.tcn_channels
    if args.tcn_depth is not None:
        tcn_depth = args.tcn_depth
    if args.dropout is not None:
        dropout = args.dropout
    if args.lr is not None:
        lr = args.lr
    if args.weight_decay is not None:
        weight_decay = args.weight_decay
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.use_all_samples_as_train is not None:
        use_all_samples_as_train = args.use_all_samples_as_train

    # 输出目录：None=使用默认时间戳目录
    out_dir_override: str | None = None

    # 使用 case118 的 active set 配置（与主入口一致）
    rt.CASE_NAME = "case118"
    rt.ACTIVE_SETS_FILE = case118_cfg.CASE118_ACTIVE_SET_JSON
    if args.active_sets is not None:
        rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = None

    rt.ensure_bcd_modules_imported()
    data_file = Path(rt.ACTIVE_SETS_FILE)
    if not data_file.is_absolute():
        data_file = Path(__file__).resolve().parent / data_file
    all_samples = rt.load_active_set_from_json(str(data_file))

    if max_samples is not None and int(max_samples) > 0 and len(all_samples) > int(max_samples):
        all_samples = all_samples[: int(max_samples)]

    if bool(use_all_samples_as_train):
        train_samples, val_samples = list(all_samples), []
        val_ratio = 0.0
    else:
        train_samples, val_samples = _train_val_split(all_samples, val_ratio=val_ratio, seed=seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(out_dir_override)
        if out_dir_override is not None
        else Path("result") / "surrogate_models" / f"unit_predictor_case118_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "unit_predictor.pth"

    if unit_ids is None:
        # 不指定则训练所有机组
        unit_ids = None

    ppc = get_case_ppc("case118")
    predictor = SingleUnitBinaryPredictorTrainer(
        ppc,
        train_samples,
        1.0,
        unit_ids=unit_ids,
        hidden_dims=hidden_dims,
        net_variant=net_variant,
        resmlp_width=resmlp_width,
        resmlp_depth=resmlp_depth,
        tconv_channels=tconv_channels,
        tconv_depth=tconv_depth,
        tcn_channels=tcn_channels,
        tcn_depth=tcn_depth,
        dropout=dropout,
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_strategy=batch_strategy,
        batch_size=batch_size,
        shuffle=shuffle,
        device=None,
    )
    if load_path is not None:
        load_path = _resolve_checkpoint_path(load_path)
        predictor.load(load_path)

    _nv = str(net_variant).strip().lower()
    if _nv == "tcn_shared_film":
        print("-" * 60, flush=True)
        print(
            "[架构] tcn_shared_film：单一共享模型，forward 输出 shape (B, ng, T)；"
            "训练仅调用 train_all → train_all_shared_joint，故日志只有 [UnitPredictor-shared]。",
            flush=True,
        )
        print(
            "      阶段 B/C/D 选中部分机组时：仍是一次 joint 更新，仅通过 unit_loss_weights "
            "提高选中机组在损失里的权重（不是 train_unit 单独优化子网）。",
            flush=True,
        )
        print(
            "      需要逐机组 train_unit 与独立日志时：将 net_variant 改为 mlp / tcn / tconv。",
            flush=True,
        )
        print("-" * 60, flush=True)

    if not stages:
        print("=" * 60, flush=True)
        print("[UnitPredictor] no training stages configured; evaluation only", flush=True)
        print("=" * 60, flush=True)

    for stage in stages:
        stage_shuffle = bool(stage.get("shuffle", shuffle))
        stage_sched_patience = int(stage.get("scheduler_patience", scheduler_patience))
        print("=" * 60, flush=True)
        print(f"Stage: {stage['name']} (epochs={stage['epochs']})", flush=True)
        print(
            "  loss="
            f"bce*{stage['loss_weight_bce']} + mse*{stage['loss_weight_mse']} "
            f"+ l1*{stage['loss_weight_l1']} + tv*{stage['loss_weight_tv']} "
            f"(pos_weight={stage['enable_pos_weight']})",
            flush=True,
        )
        target_units = stage.get("per_unit")
        if target_units == "auto_low_tv":
            eval_units = (unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", [])))
            tv_threshold = float(stage.get("tv_threshold", 0.02))
            tv_floor_scale = float(stage.get("tv_floor_scale", 0.8))
            target_units = _pick_low_tv_units(
                predictor,
                train_samples,
                eval_units,
                tv_threshold=tv_threshold,
                tv_floor_scale=tv_floor_scale,
            )
            print(f"  auto_low_tv_units={target_units}", flush=True)
        elif target_units == "auto_high_mse":
            eval_units = (unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", [])))
            thr = float(stage.get("high_mse_threshold", 0.05))
            ref_samples = val_samples if val_samples else train_samples
            target_units = _pick_high_mse_units(predictor, ref_samples, eval_units, mse_threshold=thr)
            print(f"  auto_high_mse_units(thr={thr})={target_units}", flush=True)
        if target_units is None:
            extra_kwargs = {}
            ref_for_weights = val_samples if val_samples else train_samples
            if bool(stage.get("use_unit_loss_weights", False)) and ref_for_weights:
                eval_units = (unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", [])))
                power = float(stage.get("unit_loss_weight_power", 1.0))
                extra_kwargs["unit_loss_weights"] = _make_unit_loss_weights_from_val_mse(
                    predictor, ref_for_weights, eval_units, power=power
                )
            predictor.train_all(
                num_epochs=int(stage["epochs"]),
                batch_size=batch_size,
                batch_strategy=batch_strategy,
                shuffle=stage_shuffle,
                learning_rate=float(stage.get("learning_rate", lr)),
                enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                scheduler_patience=stage_sched_patience,
                scheduler_factor=scheduler_factor,
                scheduler_min_lr=min_lr,
                enable_pos_weight=bool(stage["enable_pos_weight"]),
                pos_weight_clip=pos_weight_clip,
                loss_weight_bce=float(stage["loss_weight_bce"]),
                loss_weight_mse=float(stage["loss_weight_mse"]),
                loss_weight_l1=float(stage["loss_weight_l1"]),
                loss_weight_tv=float(stage["loss_weight_tv"]),
                loss_weight_transition=float(stage.get("loss_weight_transition", 0.0)),
                loss_weight_binarize=float(stage.get("loss_weight_binarize", 0.0)),
                loss_weight_std_floor=float(stage.get("loss_weight_std_floor", 0.0)),
                std_floor_scale=float(stage.get("std_floor_scale", 0.5)),
                loss_weight_tv_floor=float(stage.get("loss_weight_tv_floor", 0.0)),
                tv_floor_scale=float(stage.get("tv_floor_scale", 0.8)),
                **extra_kwargs,
            )
        else:
            # 共享模型不能逐机组训练，否则会反复覆盖共享骨干导致变差
            if str(net_variant).strip().lower() == "tcn_shared_film":
                extra_kwargs = {}
                boost = float(stage.get("selected_unit_boost", 3.0))
                extra_kwargs["unit_loss_weights"] = _make_unit_loss_weights_from_selected(
                    predictor, [int(x) for x in target_units], boost=boost
                )
                predictor.train_all(
                    num_epochs=int(stage["epochs"]),
                    batch_size=batch_size,
                    batch_strategy=batch_strategy,
                    shuffle=stage_shuffle,
                    learning_rate=float(stage.get("learning_rate", lr)),
                    enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                    scheduler_patience=stage_sched_patience,
                    scheduler_factor=scheduler_factor,
                    scheduler_min_lr=min_lr,
                    enable_pos_weight=bool(stage["enable_pos_weight"]),
                    pos_weight_clip=pos_weight_clip,
                    loss_weight_bce=float(stage["loss_weight_bce"]),
                    loss_weight_mse=float(stage["loss_weight_mse"]),
                    loss_weight_l1=float(stage["loss_weight_l1"]),
                    loss_weight_tv=float(stage["loss_weight_tv"]),
                    loss_weight_transition=float(stage.get("loss_weight_transition", 0.0)),
                    loss_weight_binarize=float(stage.get("loss_weight_binarize", 0.0)),
                    loss_weight_std_floor=float(stage.get("loss_weight_std_floor", 0.0)),
                    std_floor_scale=float(stage.get("std_floor_scale", 0.5)),
                    loss_weight_tv_floor=float(stage.get("loss_weight_tv_floor", 0.0)),
                    tv_floor_scale=float(stage.get("tv_floor_scale", 0.8)),
                    **extra_kwargs,
                )
            else:
                for g in target_units:
                    if bool(stage.get("reset_network", False)):
                        _reset_unit_network(predictor, int(g))
                    predictor.train_unit(
                        unit_id=int(g),
                        num_epochs=int(stage["epochs"]),
                        batch_size=batch_size,
                        batch_strategy=batch_strategy,
                        shuffle=stage_shuffle,
                        learning_rate=float(stage.get("learning_rate", lr)),
                        enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                        scheduler_patience=stage_sched_patience,
                        scheduler_factor=scheduler_factor,
                        scheduler_min_lr=min_lr,
                        enable_pos_weight=bool(stage["enable_pos_weight"]),
                        pos_weight_clip=pos_weight_clip,
                        loss_weight_bce=float(stage["loss_weight_bce"]),
                        loss_weight_mse=float(stage["loss_weight_mse"]),
                        loss_weight_l1=float(stage["loss_weight_l1"]),
                        loss_weight_tv=float(stage["loss_weight_tv"]),
                        loss_weight_transition=float(stage.get("loss_weight_transition", 0.0)),
                        loss_weight_binarize=float(stage.get("loss_weight_binarize", 0.0)),
                        loss_weight_std_floor=float(stage.get("loss_weight_std_floor", 0.0)),
                        std_floor_scale=float(stage.get("std_floor_scale", 0.5)),
                        loss_weight_tv_floor=float(stage.get("loss_weight_tv_floor", 0.0)),
                        tv_floor_scale=float(stage.get("tv_floor_scale", 0.8)),
                    )

        if val_samples:
            print("-" * 60, flush=True)
            print("Distance metrics (train/val, threshold-independent):", flush=True)
            eval_units = (unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", [])))
            for g in eval_units:
                g_int = int(g)
                m_tr = _eval_unit_predictor(predictor, train_samples, unit_id=g_int, threshold=0.5)
                m_va = _eval_unit_predictor(predictor, val_samples, unit_id=g_int, threshold=0.5)
                print(
                    f"  unit={g_int}: train(mse={m_tr['mse']:.4f}, mae={m_tr['mae']:.4f}, tv={m_tr['tv']:.4f}) "
                    f"val(mse={m_va['mse']:.4f}, mae={m_va['mae']:.4f}, tv={m_va['tv']:.4f})",
                    flush=True,
                )
            print("-" * 60, flush=True)

    if stages:
        predictor.save(str(save_path))

        # 额外写一个“最新”指针文件，方便其它脚本引用
        latest_path = out_dir / "LATEST.txt"
        latest_path.write_text(str(save_path).replace("\\", "/"), encoding="utf-8")

        print(f"✓ UnitPredictor 已保存: {save_path}", flush=True)
    else:
        print("UnitPredictor evaluation-only run; checkpoint was not re-saved.", flush=True)
    if predictor is not None:
        print(f"  out_dir={out_dir}", flush=True)
        print(f"  unit_ids={unit_ids!r}", flush=True)
        print(
            f"  train/val={len(train_samples)}/{len(val_samples)} seed={seed} "
            f"net={net_variant} hidden_dims={hidden_dims} resmlp=({resmlp_width},{resmlp_depth}) "
            f"tconv=({tconv_channels},{tconv_depth}) tcn=({tcn_channels},{tcn_depth}) "
            f"dropout={dropout}",
            flush=True,
        )

        if val_samples:
            print("-" * 60, flush=True)
            print("Validation metrics (thr=0.50 only; distance-first):", flush=True)
            eval_units = list(getattr(predictor, "unit_ids", []))
            rows = []
            for g in eval_units:
                g_int = int(g)
                m05 = _eval_unit_predictor(predictor, val_samples, unit_id=g_int, threshold=0.5)
                rows.append((g_int, m05))

            rows_sorted = sorted(rows, key=lambda r: float(r[1].get("mse", 1e9)), reverse=True)
            head_rows = rows_sorted if print_all_validation else rows_sorted[: int(max(1, validation_top_k))]
            if not print_all_validation:
                print(f"  (showing worst-{len(head_rows)} by val MSE; set print_all_validation=True to show all)", flush=True)

            bad_cnt = sum(1 for _, m in rows_sorted if float(m.get("mse", 0.0)) > float(target_mse))
            print(f"  summary: {bad_cnt}/{len(rows_sorted)} units have val_mse>{target_mse:.3f}", flush=True)

            for g_int, m05 in head_rows:
                print(
                    f"  unit={g_int} @thr=0.50: mse={m05['mse']:.4f} mae={m05['mae']:.4f} tv={m05['tv']:.4f} "
                    f"bce={m05['bce']:.4f} f1={m05['f1']:.4f} "
                    f"prec={m05['precision']:.4f} rec={m05['recall']:.4f} "
                    f"acc={m05['acc']:.4f} pos={m05['pos_rate']:.3f}",
                    flush=True,
                )
            print("-" * 60, flush=True)

        eval_units_final = unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", []))
        _run_train_set_per_unit_report(
            predictor,
            train_samples,
            unit_ids=eval_units_final,
            do_threshold_sweep=train_eval_threshold_sweep,
        )


if __name__ == "__main__":
    main()
