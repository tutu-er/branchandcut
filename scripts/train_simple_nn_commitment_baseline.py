"""Train a simple NN commitment predictor baseline.

The baseline maps scenario features directly to the flattened UC commitment
matrix. It is meant as a lightweight control group for surrogate-model tests:
no Gurobi solve is used during training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case_registry import get_case_ppc
from src.dataset_json_utils import load_v3_active_set_json
from src.scenario_utils import get_feature_vector_from_sample, get_sample_net_load, normalize_sample_arrays


def _extract_true_solution(sample: dict, shape: tuple[int, int]) -> np.ndarray:
    ng, T = shape
    x_true = np.full((ng, T), np.nan, dtype=float)
    if "unit_commitment_matrix" in sample:
        raw = np.asarray(sample["unit_commitment_matrix"], dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        r = min(raw.shape[0], ng)
        c = min(raw.shape[1], T)
        x_true[:r, :c] = raw[:r, :c]
        return x_true

    if "active_set" in sample:
        x_true[:] = 0.0
        for item in sample["active_set"]:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                if 0 <= int(g) < ng and 0 <= int(t) < T:
                    x_true[int(g), int(t)] = float(item[1])
    return x_true


def _distance_metrics(x_candidate: np.ndarray, x_true: np.ndarray) -> dict[str, Any]:
    x_arr = np.asarray(x_candidate, dtype=float)
    x_ref = np.asarray(x_true, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(x_ref)
    total = int(valid.size)
    covered = int(np.sum(valid))
    if covered == 0:
        return {"available": False, "covered": 0, "total": total, "coverage_ratio": 0.0}

    x_v = x_arr[valid]
    y_v = x_ref[valid]
    rounded = np.rint(np.clip(x_v, 0.0, 1.0)).astype(int)
    y_int = np.rint(np.clip(y_v, 0.0, 1.0)).astype(int)
    false_on = int(np.sum((rounded == 1) & (y_int == 0)))
    false_off = int(np.sum((rounded == 0) & (y_int == 1)))
    hamming = int(false_on + false_off)
    l1 = float(np.sum(np.abs(x_v - y_v)))
    return {
        "available": True,
        "covered": covered,
        "total": total,
        "coverage_ratio": float(covered / total) if total else 0.0,
        "l1": l1,
        "mean_abs": float(l1 / covered),
        "hamming": hamming,
        "normalized_hamming": float(hamming / covered),
        "false_on": false_on,
        "false_off": false_off,
        "integrality_gap": float(np.mean(np.minimum(x_v, 1.0 - x_v))),
    }


def _parse_indices(text: str, n_samples: int) -> list[int]:
    text = str(text).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        lo = int(left) if left else 0
        hi = int(right) if right else n_samples
        return list(range(max(0, lo), min(n_samples, hi)))
    if text.lower() in {"all", "*"}:
        return list(range(n_samples))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _aggregate(records: list[dict[str, Any]], metric_key: str = "metrics") -> dict[str, float]:
    out: dict[str, float] = {}
    keys = ["l1", "mean_abs", "hamming", "normalized_hamming", "false_on", "false_off", "integrality_gap"]
    for key in keys:
        values = [
            float(record[metric_key][key])
            for record in records
            if record.get(metric_key, {}).get("available") and key in record[metric_key]
        ]
        if values:
            out[f"mean_{key}"] = float(np.mean(values))
            out[f"median_{key}"] = float(np.median(values))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case14")
    parser.add_argument(
        "--active-sets",
        default="result/active_set/active_sets_case14_T24_n600_20260503_222929.json",
    )
    parser.add_argument("--train-range", default="50:")
    parser.add_argument("--eval-range", default="0:50")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: result/simple_nn_baselines/<case>_<timestamp>",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    active_path = Path(args.active_sets)
    if not active_path.is_absolute():
        active_path = ROOT / active_path
    ppc = get_case_ppc(args.case)
    samples = load_v3_active_set_json(active_path, announce=lambda msg: print(f"[data] {msg}", flush=True))
    if not samples:
        raise ValueError("No samples loaded")

    first = normalize_sample_arrays(dict(samples[0]))
    ng = int(np.asarray(ppc["gen"]).shape[0])
    T = int(get_sample_net_load(first).shape[1])
    shape = (ng, T)

    features = []
    labels = []
    sample_ids = []
    for idx, sample in enumerate(samples):
        sample = normalize_sample_arrays(dict(sample))
        features.append(np.asarray(get_feature_vector_from_sample(sample), dtype=np.float32))
        labels.append(_extract_true_solution(sample, shape).reshape(-1).astype(np.float32))
        sample_ids.append(sample.get("sample_id", idx))

    X = np.stack(features, axis=0)
    Y = np.stack(labels, axis=0)
    valid_rows = np.isfinite(Y).all(axis=1)
    X = X[valid_rows]
    Y = Y[valid_rows]
    sample_ids = [sid for sid, keep in zip(sample_ids, valid_rows) if keep]

    train_indices = [i for i in _parse_indices(args.train_range, len(samples)) if i < len(valid_rows) and valid_rows[i]]
    eval_indices = [i for i in _parse_indices(args.eval_range, len(samples)) if i < len(valid_rows) and valid_rows[i]]
    old_to_new = {}
    pos = 0
    for old_idx, keep in enumerate(valid_rows):
        if keep:
            old_to_new[old_idx] = pos
            pos += 1
    train_pos = [old_to_new[i] for i in train_indices if i in old_to_new]
    eval_pos = [old_to_new[i] for i in eval_indices if i in old_to_new]
    if not train_pos or not eval_pos:
        raise ValueError(f"Empty split: train={len(train_pos)}, eval={len(eval_pos)}")

    x_mean = X[train_pos].mean(axis=0, keepdims=True)
    x_std = X[train_pos].std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    Xn = (X - x_mean) / x_std

    train_ds = TensorDataset(
        torch.tensor(Xn[train_pos], dtype=torch.float32),
        torch.tensor(Y[train_pos], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )

    model = nn.Sequential(
        nn.Linear(X.shape[1], int(args.hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_dim), Y.shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 25 == 0 or epoch == int(args.epochs):
            model.eval()
            with torch.no_grad():
                train_logits = model(torch.tensor(Xn[train_pos], dtype=torch.float32))
                eval_logits = model(torch.tensor(Xn[eval_pos], dtype=torch.float32))
                train_loss = float(loss_fn(train_logits, torch.tensor(Y[train_pos], dtype=torch.float32)).cpu())
                eval_loss = float(loss_fn(eval_logits, torch.tensor(Y[eval_pos], dtype=torch.float32)).cpu())
            history.append(
                {
                    "epoch": epoch,
                    "batch_loss": float(np.mean(losses)),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                }
            )
            print(
                f"[epoch {epoch:04d}] train_loss={train_loss:.6f} eval_loss={eval_loss:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(Xn[eval_pos], dtype=torch.float32))).cpu().numpy()

    records = []
    for local_pos, (sample_pos, pred_flat) in enumerate(zip(eval_pos, probs)):
        pred = pred_flat.reshape(shape)
        truth = Y[sample_pos].reshape(shape)
        records.append(
            {
                "eval_pos": local_pos,
                "sample_id": sample_ids[sample_pos],
                "sample_index": int(eval_indices[local_pos]) if local_pos < len(eval_indices) else None,
                "metrics": _distance_metrics(pred, truth),
                "rounded_metrics": _distance_metrics(np.rint(pred), truth),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "result" / "simple_nn_baselines" / f"{args.case}_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "simple_commitment_nn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "output_dim": int(Y.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "x_mean": x_mean,
            "x_std": x_std,
            "shape": shape,
            "args": vars(args),
        },
        checkpoint_path,
    )
    result = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": args.case,
            "active_sets": str(active_path),
            "train_range": args.train_range,
            "eval_range": args.eval_range,
            "train_samples": len(train_pos),
            "eval_samples": len(eval_pos),
            "input_dim": int(X.shape[1]),
            "output_dim": int(Y.shape[1]),
            "shape": list(shape),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "checkpoint": str(checkpoint_path),
        },
        "history": history,
        "summary": {
            "probability": _aggregate(records, "metrics"),
            "rounded": _aggregate(records, "rounded_metrics"),
        },
        "records": records,
    }
    report_path = output_dir / "simple_nn_commitment_baseline_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {report_path}", flush=True)
    print(f"[done] wrote {checkpoint_path}", flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
