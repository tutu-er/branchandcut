#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case3lite 单独训练/评估 UnitPredictor（单机组 0/1 变量预测器）。

与 ``run_unit_predictor_case118.py`` 结构相同，仅算例与数据路径针对 **case3lite**（通常 3 台机组）。

数据：默认使用 ``CASE3LITE_ACTIVE_SET_JSON``；若该文件不存在且 ``AUTO_PICK_LATEST_ACTIVE_SET=True``，
则在 ``result/active_set/`` 下选取 **修改时间最新** 的 ``active_sets_case3lite_*.json``。

输出：``result/surrogate_models/unit_predictor_case3lite_<timestamp>/unit_predictor.pth`` 及 ``LATEST.txt``。

说明：本脚本不使用命令行参数；请在 ``main()`` 顶部显式修改配置。
"""

from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import run_training as rt

from src.case_registry import get_case_ppc
from src.uc_NN_subproblem import SingleUnitBinaryPredictorTrainer
from src.scenario_utils import get_feature_vector_from_sample


ROOT = Path(__file__).resolve().parent

# 与 ``run_test.py`` / 仓库惯例一致的占位路径；不存在则依赖 AUTO_PICK_LATEST_ACTIVE_SET
CASE3LITE_ACTIVE_SET_JSON = "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"

AUTO_PICK_LATEST_ACTIVE_SET = False


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


def _bce_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)
    loss = np.logaddexp(0.0, z) - y * z
    return float(loss.mean())


def _prepare_eval_xy(predictor, samples: list):
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

    if "unit_commitment_matrix" in samples[0]:
        y_mat = np.stack(
            [np.asarray(s["unit_commitment_matrix"], dtype=float) for s in samples],
            axis=0,
        )
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
    y_arr = np.asarray(y_mat[:, int(unit_id), :], dtype=float)

    net = predictor.get_network(unit_id)
    with torch.no_grad():
        logits_t = net(X_tensor)
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
    ng = int(getattr(predictor, "ng", max(selected_units) + 1 if selected_units else 0))
    w = np.ones(int(ng), dtype=float)
    for g in selected_units:
        if 0 <= int(g) < ng:
            w[int(g)] = float(boost)
    w = w / max(float(w.mean()), 1e-12)
    return w.tolist()


def _reset_unit_network(predictor, unit_id: int) -> None:
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


def _resolve_case3lite_data_file() -> Path:
    """返回 active set JSON 的绝对路径。"""
    if CASE3LITE_ACTIVE_SET_JSON:
        p = Path(CASE3LITE_ACTIVE_SET_JSON)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return p.resolve()
    if AUTO_PICK_LATEST_ACTIVE_SET:
        d = ROOT / "result" / "active_set"
        if d.is_dir():
            cands = sorted(
                d.glob("active_sets_case3lite_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if cands:
                print(f"[run_unit_predictor_case3lite] 使用最新 active set: {cands[0]}", flush=True)
                return cands[0].resolve()
    raise FileNotFoundError(
        "未找到 case3lite active set JSON。请在 main 顶部设置 CASE3LITE_ACTIVE_SET_JSON，"
        "或将 ``active_sets_case3lite_*.json`` 放入 result/active_set/。"
    )


def _run_train_set_per_unit_report(
    predictor,
    train_samples: list,
    *,
    unit_ids: list[int],
    do_threshold_sweep: bool = True,
) -> None:
    if not train_samples or not unit_ids:
        print("[train_eval] empty train_samples or unit_ids", flush=True)
        return

    print("=" * 72, flush=True)
    suffix = "；含 F1 阈值扫优" if do_threshold_sweep else ""
    print(f"训练集效果汇总（case3lite 各机组）| mse/mae/bce 与 thr=0.5 分类指标{suffix}", flush=True)
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
    max_samples: int | None = 250
    use_all_samples_as_train = 200
    unit_ids: list[int] | None = None
    load_path: str | None = None

    lr = 1e-3
    weight_decay = 8e-5
    batch_strategy = "full-batch"
    batch_size = 64
    shuffle = True

    enable_scheduler = True
    scheduler_patience = 15
    scheduler_factor = 0.5
    min_lr = 1e-7

    net_variant = "tcn_shared_film"
    hidden_dims = _parse_hidden_dims("256,128")
    resmlp_width = 512
    resmlp_depth = 4
    tconv_channels = 64
    tconv_depth = 4
    # case3lite 仅 3 台机，可适当减小容量以加快实验；需与下游 run_training 中 UNIT_PREDICTOR_* 对齐
    tcn_channels = 128
    tcn_depth = 8
    dropout = 0.02

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

    seed = 42
    val_ratio = 0.2
    print_all_validation = False
    validation_top_k = 4
    target_mse = 0.05
    train_eval_threshold_sweep = True

    out_dir_override: str | None = None

    try:
        data_file = _resolve_case3lite_data_file()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    rt.CASE_NAME = "case3lite"
    try:
        rt.ACTIVE_SETS_FILE = str(data_file.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rt.ACTIVE_SETS_FILE = str(data_file)
    rt.MAX_SAMPLES = None

    rt.ensure_bcd_modules_imported()
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
        else Path("result") / "surrogate_models" / f"unit_predictor_case3lite_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "unit_predictor.pth"

    ppc = get_case_ppc("case3lite")
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

    _nv = str(net_variant).strip().lower()
    if _nv == "tcn_shared_film":
        print("-" * 60, flush=True)
        print(
            "[架构] tcn_shared_film：日志为 [UnitPredictor-shared]，无逐机组 train_unit。",
            flush=True,
        )
        print("-" * 60, flush=True)

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

    predictor.save(str(save_path))

    latest_path = out_dir / "LATEST.txt"
    latest_path.write_text(str(save_path).replace("\\", "/"), encoding="utf-8")

    print(f"✓ UnitPredictor (case3lite) 已保存: {save_path}", flush=True)
    if predictor is not None:
        print(f"  data_file={data_file}", flush=True)
        print(f"  out_dir={out_dir}", flush=True)
        print(f"  unit_ids={unit_ids!r}", flush=True)
        print(
            f"  train/val={len(train_samples)}/{len(val_samples)} seed={seed} "
            f"net={net_variant} tcn=({tcn_channels},{tcn_depth}) dropout={dropout}",
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
                print(f"  (showing worst-{len(head_rows)} by val MSE)", flush=True)

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
