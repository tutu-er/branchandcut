#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 单独训练/评估 UnitPredictor（单机组 0/1 变量预测器）。

用途：
- 独立于 subproblem BCD 训练流程，快速迭代 UnitPredictor 的结构/超参；
- 支持截断样本数、指定机组列表、配置训练超参，并把模型存到 result/ 下的时间戳目录。

说明：
    本脚本不使用命令行参数；请在 `main()` 顶部显式修改配置变量以进行实验对比。
"""

from __future__ import annotations
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


def main() -> None:
    # ── 实验配置（请在这里显式修改；不使用命令行参数）────────────────────
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
    weight_decay = 1e-4
    batch_strategy = "mini-batch"
    batch_size = 32  # 或 16
    shuffle = True

    # LR scheduler（少样本/不稳定时可把 min_lr 调高或直接关掉）
    enable_scheduler = True
    scheduler_patience = 10
    scheduler_factor = 0.5
    min_lr = 1e-5

    # 网络结构
    net_variant = "tcn_shared_film"  # "mlp" | "resmlp" | "tconv" | "tcn" | "tcn_shared_film"
    hidden_dims = _parse_hidden_dims("256,128")  # 仅对 mlp 生效
    resmlp_width = 512
    resmlp_depth = 4
    tconv_channels = 64
    tconv_depth = 4
    # 共享骨干下容量要更足，否则容易“平均化”
    tcn_channels = 128
    tcn_depth = 8
    dropout = 0.05

    # staged training（默认面向“距离最小化 + 捕捉启停变化”）：
    # - Stage A：对所有 unit 训练（MSE + transition + binarize + std_floor + tv_floor），避免“均值塌陷/过度平滑”
    # - Stage B：对“变化不足”的机组做短微调（tv_floor 约束，自动挑选）
    # - Stage C：对 val MSE 仍偏高的机组做强化微调（自动挑选，帮助跳出局部最优）
    # - Stage D：对仍然顽固的高 MSE 机组做“重置再训”（带一点 BCE+pos_weight 纠偏）
    stages = [
        {
            "name": "A_align_mse",
            # joint 预训练：别只用 MSE（容易学成“平均概率”），加少量 BCE+pos_weight 稳住方向
            "epochs": 360,
            "per_unit": None,  # None = all units
            "enable_pos_weight": True,
            "loss_weight_bce": 0.12,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.35,
            "loss_weight_binarize": 0.03,
            "loss_weight_std_floor": 0.25,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.30,
            "tv_floor_scale": 0.9,
        },
        {
            "name": "B_tv_floor_auto",
            "epochs": 180,
            "per_unit": "auto_low_tv",  # 自动挑选变化不足的机组（基于 y_tv vs pred_tv）
            "enable_pos_weight": False,
            "loss_weight_bce": 0.0,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.45,
            "loss_weight_binarize": 0.05,
            "loss_weight_std_floor": 0.30,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.40,
            "tv_floor_scale": 0.95,
            "selected_unit_boost": 5.0,
        },
        {
            "name": "C_high_mse_finetune",
            "epochs": 200,
            "per_unit": "auto_high_mse",  # 自动挑选 MSE 偏大的机组（优先用 val 集）
            # 微调阶段：提高学习率、关闭 scheduler，尽量跳出局部最优
            "learning_rate": 2e-3,
            "enable_scheduler": False,
            # 加少量 BCE + pos_weight，避免像 unit0 这种“全预测 0”导致 BCE 爆炸、概率无法校正
            "enable_pos_weight": True,
            "loss_weight_bce": 0.20,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.55,
            "loss_weight_binarize": 0.05,
            "loss_weight_std_floor": 0.35,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.45,
            "tv_floor_scale": 0.95,
            "high_mse_threshold": 0.05,
            "selected_unit_boost": 5.0,
        },
        {
            "name": "D_reset_and_retrain_high_mse",
            "epochs": 200,
            "per_unit": "auto_high_mse",
            "reset_network": True,
            "learning_rate": 3e-3,
            "enable_scheduler": False,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.25,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.55,
            "loss_weight_binarize": 0.05,
            "loss_weight_std_floor": 0.3,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.45,
            "tv_floor_scale": 0.95,
            "high_mse_threshold": 0.05,
            "selected_unit_boost": 4.0,
        },
        {
            "name": "E_weighted_joint_more_epochs",
            "epochs": 360,
            "per_unit": None,  # joint
            "learning_rate": 2e-3,
            "enable_scheduler": False,
            "enable_pos_weight": True,
            "loss_weight_bce": 0.22,
            "loss_weight_mse": 1.0,
            "loss_weight_l1": 0.0,
            "loss_weight_tv": 0.0,
            "loss_weight_transition": 0.6,
            "loss_weight_binarize": 0.05,
            "loss_weight_std_floor": 0.35,
            "std_floor_scale": 0.5,
            "loss_weight_tv_floor": 0.5,
            "tv_floor_scale": 0.95,
            # 关键：按 val_mse 生成 unit 权重，等效给“epoch不足”的机组更多训练力度
            "use_unit_loss_weights": True,
            "unit_loss_weight_power": 1.5,
        },
    ]
    pos_weight_clip = 20.0

    # 评估：train/val 切分
    seed = 42
    val_ratio = 0.2  # 0=不做验证
    # 验证输出控制：避免刷屏，默认只输出最差 Top-K
    print_all_validation = False
    validation_top_k = 12
    # 目标：希望 val MSE <= target_mse（用于汇总提示）
    target_mse = 0.05

    # 输出目录：None=使用默认时间戳目录
    out_dir_override: str | None = None

    # 使用 case118 的 active set 配置（与主入口一致）
    rt.CASE_NAME = "case118"
    rt.ACTIVE_SETS_FILE = case118_cfg.CASE118_ACTIVE_SET_JSON
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
        predictor.load(load_path)

    for stage in stages:
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
            if bool(stage.get("use_unit_loss_weights", False)) and val_samples:
                eval_units = (unit_ids if unit_ids is not None else list(getattr(predictor, "unit_ids", [])))
                power = float(stage.get("unit_loss_weight_power", 1.0))
                extra_kwargs["unit_loss_weights"] = _make_unit_loss_weights_from_val_mse(
                    predictor, val_samples, eval_units, power=power
                )
            predictor.train_all(
                num_epochs=int(stage["epochs"]),
                batch_size=batch_size,
                batch_strategy=batch_strategy,
                shuffle=shuffle,
                learning_rate=float(stage.get("learning_rate", lr)),
                enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                scheduler_patience=scheduler_patience,
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
                    shuffle=shuffle,
                    learning_rate=float(stage.get("learning_rate", lr)),
                    enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                    scheduler_patience=scheduler_patience,
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
                        shuffle=shuffle,
                        learning_rate=float(stage.get("learning_rate", lr)),
                        enable_scheduler=bool(stage.get("enable_scheduler", enable_scheduler)),
                        scheduler_patience=scheduler_patience,
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

    predictor.save(str(save_path))

    # 额外写一个“最新”指针文件，方便其它脚本引用
    latest_path = out_dir / "LATEST.txt"
    latest_path.write_text(str(save_path).replace("\\", "/"), encoding="utf-8")

    print(f"✓ UnitPredictor 已保存: {save_path}", flush=True)
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


if __name__ == "__main__":
    main()

