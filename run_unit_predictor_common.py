#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared standalone UnitPredictor training runner for UC cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from src.case_registry import get_case_ppc
from src.dataset_json_utils import load_v3_active_set_json
from src.scenario_utils import get_feature_vector_from_sample
from src.uc_NN_subproblem import train_unit_predictor_from_data


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UnitPredictorConfig:
    case_name: str
    max_samples: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dims: tuple[int, ...]
    net_variant: str = "mlp"
    tcn_channels: int = 64
    tcn_depth: int = 6
    tconv_channels: int = 64
    tconv_depth: int = 4
    dropout: float = 0.05
    val_ratio: float = 0.0
    seed: int = 42
    enable_pos_weight: bool = True
    loss_weight_bce: float = 1.0
    loss_weight_mse: float = 0.15
    loss_weight_transition: float = 0.05
    loss_weight_binarize: float = 0.01
    loss_weight_tv_floor: float = 0.05
    tv_floor_scale: float = 0.75
    scheduler_patience: int = 12


def parse_hidden_dims(text: str) -> list[int] | None:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return [int(p) for p in parts] if parts else None


def pick_active_set_file(case_name: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"active-set file does not exist: {p}")
        return p.resolve()

    data_dir = ROOT / "result" / "active_set"
    candidates = sorted(
        data_dir.glob(f"active_sets_{case_name}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no active-set file found for {case_name} under {data_dir}"
        )
    return candidates[0].resolve()


def split_train_val(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    ratio = min(max(float(val_ratio), 0.0), 0.8)
    if ratio <= 0.0 or len(samples) <= 1:
        return list(samples), []
    n_val = max(1, int(round(len(samples) * ratio)))
    rng = np.random.RandomState(int(seed))
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_indices]
    val = [s for i, s in enumerate(samples) if i in val_indices]
    return train, val


def bce_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)
    return float((np.logaddexp(0.0, z) - y * z).mean())


def labels_from_samples(samples: Sequence[dict], ng: int, T: int) -> np.ndarray:
    labels = np.zeros((len(samples), int(ng), int(T)), dtype=float)
    for i, sample in enumerate(samples):
        if "unit_commitment_matrix" in sample:
            mat = np.asarray(sample["unit_commitment_matrix"], dtype=float)
            labels[i, : min(mat.shape[0], ng), : min(mat.shape[1], T)] = mat[:ng, :T]
            continue
        for item in sample.get("active_set", []) or []:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = int(item[0][0]), int(item[0][1])
                if 0 <= g < ng and 0 <= t < T:
                    labels[i, g, t] = float(item[1])
    return labels


def eval_unit_predictor(predictor, samples: list[dict], unit_id: int) -> dict[str, float]:
    if not samples:
        return {"n": 0.0}

    import torch

    X = np.asarray([get_feature_vector_from_sample(dict(s)) for s in samples], dtype=np.float32)
    y = labels_from_samples(samples, predictor.ng, predictor.T)[:, int(unit_id), :]
    X_tensor = torch.tensor(X, dtype=torch.float32, device=predictor.device)
    net = predictor.get_network(int(unit_id))
    net.eval()
    with torch.no_grad():
        logits_t = net(X_tensor)
        probs_t = torch.sigmoid(logits_t)
    logits = logits_t.detach().cpu().numpy()
    probs = probs_t.detach().cpu().numpy()
    pred = (probs >= 0.5).astype(float)
    tp = float(np.sum((pred == 1.0) & (y == 1.0)))
    fp = float(np.sum((pred == 1.0) & (y == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y == 0.0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": float(y.size),
        "pos_rate": float(y.mean()),
        "mse": float(np.mean((probs - y) ** 2)),
        "mae": float(np.mean(np.abs(probs - y))),
        "bce": bce_from_logits(logits, y),
        "acc": float((tp + tn) / max(tp + tn + fp + fn, 1.0)),
        "f1": float(f1),
        "tv": float(np.mean(np.abs(probs[:, 1:] - probs[:, :-1]))) if probs.shape[1] > 1 else 0.0,
        "y_tv": float(np.mean(np.abs(y[:, 1:] - y[:, :-1]))) if y.shape[1] > 1 else 0.0,
    }


def print_metric_report(predictor, train_samples: list[dict], val_samples: list[dict]) -> None:
    unit_ids = [int(g) for g in getattr(predictor, "unit_ids", [])]
    for split_name, samples in (("train", train_samples), ("val", val_samples)):
        if not samples:
            continue
        rows = [(g, eval_unit_predictor(predictor, samples, g)) for g in unit_ids]
        mse_values = [m["mse"] for _, m in rows]
        print("-" * 78, flush=True)
        print(
            f"{split_name}: n_samples={len(samples)} "
            f"mse_mean={np.mean(mse_values):.5f} mse_max={np.max(mse_values):.5f}",
            flush=True,
        )
        print("unit | pos_rate | y_tv   | pred_tv | mse    | mae    | bce    | acc    | f1", flush=True)
        print("-" * 78, flush=True)
        for g, m in rows:
            print(
                f"{g:4d} | {m['pos_rate']:8.3f} | {m['y_tv']:6.3f} | {m['tv']:7.3f} | "
                f"{m['mse']:6.4f} | {m['mae']:6.4f} | {m['bce']:6.4f} | "
                f"{m['acc']:6.3f} | {m['f1']:6.3f}",
                flush=True,
            )


def add_common_args(parser: argparse.ArgumentParser, cfg: UnitPredictorConfig) -> None:
    parser.add_argument("--active-sets", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=cfg.max_samples)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--hidden-dims", type=str, default=",".join(str(x) for x in cfg.hidden_dims))
    parser.add_argument("--net", choices=("mlp", "resmlp", "tconv", "tcn", "tcn_shared_film"), default=cfg.net_variant)
    parser.add_argument("--val-ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--unit-ids", type=str, default=None, help="comma-separated unit ids; default trains all units")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)


def parse_unit_ids(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


def run_unit_predictor_training(cfg: UnitPredictorConfig, args: argparse.Namespace) -> Path:
    data_file = pick_active_set_file(cfg.case_name, args.active_sets)
    samples = load_v3_active_set_json(data_file, announce=lambda msg: print(msg, flush=True))
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples = samples[: int(args.max_samples)]
    train_samples, val_samples = split_train_val(samples, float(args.val_ratio), int(args.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "result" / "surrogate_models" / f"unit_predictor_{cfg.case_name}_{timestamp}"
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "unit_predictor.pth"

    print("=" * 78, flush=True)
    print(
        f"UnitPredictor training | case={cfg.case_name} | samples={len(train_samples)}/{len(val_samples)} "
        f"train/val | data={data_file.name}",
        flush=True,
    )
    print(
        f"net={args.net} hidden={args.hidden_dims} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} out={out_dir}",
        flush=True,
    )
    print("=" * 78, flush=True)

    predictor = train_unit_predictor_from_data(
        get_case_ppc(cfg.case_name),
        train_samples,
        T_delta=1.0,
        unit_ids=parse_unit_ids(args.unit_ids),
        num_epochs=max(0, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        batch_strategy="full-batch",
        shuffle=True,
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        net_variant=str(args.net),
        tcn_channels=cfg.tcn_channels,
        tcn_depth=cfg.tcn_depth,
        tconv_channels=cfg.tconv_channels,
        tconv_depth=cfg.tconv_depth,
        dropout=cfg.dropout,
        save_path=str(save_path),
        load_path=args.load_path,
        enable_scheduler=True,
        scheduler_patience=cfg.scheduler_patience,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-7,
        enable_pos_weight=cfg.enable_pos_weight,
        pos_weight_clip=20.0,
        loss_weight_bce=cfg.loss_weight_bce,
        loss_weight_mse=cfg.loss_weight_mse,
        loss_weight_l1=0.0,
        loss_weight_tv=0.0,
        loss_weight_transition=cfg.loss_weight_transition,
        loss_weight_binarize=cfg.loss_weight_binarize,
        loss_weight_std_floor=0.0,
        std_floor_scale=0.5,
        loss_weight_tv_floor=cfg.loss_weight_tv_floor,
        tv_floor_scale=cfg.tv_floor_scale,
    )

    (out_dir / "LATEST.txt").write_text(str(save_path).replace("\\", "/"), encoding="utf-8")
    print_metric_report(predictor, train_samples, val_samples)
    print(f"Saved UnitPredictor checkpoint: {save_path}", flush=True)
    return save_path
