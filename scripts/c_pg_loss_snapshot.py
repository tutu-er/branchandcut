#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 / 复现子问题 ``c_pg`` 可微损失的原始数据，用于针对性测试。

**export**（多样本训练某次迭代后，用已保存的 checkpoint + 同一份 active set 数据）::

    python scripts/c_pg_loss_snapshot.py export \\
        --case case118 \\
        --active-json result/active_set/active_sets_case118_xxx.json \\
        --model-dir result/surrogate_models/subproblem_models_case118_YYYYMMDD_HHMMSS \\
        --unit 0 \\
        --out result/c_pg_snapshots/unit0_iter42.json \\
        --max-samples 12

**test**（注入 snapshot 中的 ``lambda_vals`` / ``lambda_inherent_c_pg`` / ``prev_pg_cost_values``，仅跑 ``iter_with_c_pg_nn``）::

    python scripts/c_pg_loss_snapshot.py test \\
        --snapshot result/c_pg_snapshots/unit0_iter42.json \\
        --case case118 \\
        --active-json result/active_set/active_sets_case118_xxx.json \\
        --model-dir result/surrogate_models/subproblem_models_case118_YYYYMMDD_HHMMSS \\
        --unit 0 \\
        --epochs 80 \\
        --pg-cost-surr-lr 4e-4

说明：

- ``export`` 在训练结束后用 checkpoint 时，``lambda_inherent`` 以 pth 内为准；与**某一轮迭代中**
  内存状态完全一致时，宜在训练代码里于该轮末尾直接调用
  ``trainer.collect_c_pg_loss_snapshot()`` 并自行 ``json.dump``（或随后再 ``export`` 同目录最新 pth）。
- ``test`` 会从 snapshot 覆盖 ``lambda_vals`` / 对偶中与 c_pg 相关的量，再只跑 ``iter_with_c_pg_nn``。
- case118 子问题若训练时 ``ignore_startup_shutdown_costs=True``，命令行请加 ``--ignore-startup-shutdown``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json_list(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array in {path}")
    return data


def _apply_snapshot_to_trainer(trainer, snap: dict, strict_hparams: bool) -> None:
    """用 snapshot 覆盖 c_pg 损失所依赖的缓存量（在 load(pth) 之后调用）。"""
    import numpy as np

    if int(snap.get("schema_version", 0)) != 1:
        print(
            f"警告: 未知 schema_version={snap.get('schema_version')!r}，仍尝试应用",
            flush=True,
        )
    n = len(snap["samples"])
    if n != int(trainer.n_samples):
        raise ValueError(
            f"snapshot n_samples={n} 与当前 trainer.n_samples={trainer.n_samples} 不一致"
        )
    t_expect = int(snap["T"])
    if t_expect != int(trainer.T):
        raise ValueError(f"snapshot T={t_expect} 与 trainer.T={trainer.T} 不一致")

    lam_rows = [s["lambda_vals"] for s in snap["samples"]]
    trainer.lambda_vals = np.asarray(lam_rows, dtype=np.float64).reshape(
        n, t_expect
    )
    for i, s in enumerate(snap["samples"]):
        if s.get("lambda_inherent_is_none"):
            continue
        cpg = s.get("lambda_inherent_c_pg")
        if not cpg:
            continue
        li = trainer.lambda_inherent[i]
        if li is None:
            print(
                f"警告: sample {i} snapshot 有 lambda_inherent_c_pg 但 trainer.lambda_inherent[i] 为 None，跳过",
                flush=True,
            )
            continue
        for k, v in cpg.items():
            if v is not None and k in li:
                li[k] = np.asarray(v, dtype=np.float64).reshape(-1)

    prev = snap.get("prev_pg_cost_values")
    if prev is not None:
        trainer._prev_pg_cost_values = np.asarray(prev, dtype=np.float64)
    else:
        trainer._prev_pg_cost_values = None

    trainer.iter_number = int(snap.get("iter_number", 0))

    if strict_hparams:
        trainer.rho_dual_pg = float(snap["rho_dual_pg"])
        trainer.loss_ratio_dual_pg = float(snap["loss_ratio_dual_pg"])
        trainer.loss_ratio_reg = float(snap["loss_ratio_reg"])
        trainer._c_pg_reg_loss_scale = float(snap["_c_pg_reg_loss_scale"])
        trainer.reg_weight = float(snap["reg_weight"])
        trainer.pg_cost_reg_deadband = float(snap["pg_cost_reg_deadband"])
        trainer.iter_delta_reg_weight = float(snap["iter_delta_reg_weight"])
        trainer.iter_delta_reg_deadband = float(snap["iter_delta_reg_deadband"])
        trainer.pg_cost_softbound_weight = float(snap["pg_cost_softbound_weight"])
        trainer.pg_cost_smooth_abs_eps = float(snap["pg_cost_smooth_abs_eps"])
        trainer.pg_cost_scale = float(snap["pg_cost_scale"])


def cmd_export(args: argparse.Namespace) -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    case = args.case
    ppc = get_case_ppc(case)
    active_path = Path(args.active_json)
    if not active_path.is_file():
        raise FileNotFoundError(active_path)
    all_samples = _load_json_list(active_path)
    max_s = args.max_samples
    if max_s is not None and len(all_samples) > int(max_s):
        all_samples = all_samples[: int(max_s)]
    t_delta = float(args.t_delta)
    unit = int(args.unit)
    model_dir = str(args.model_dir)

    print(
        f"[export] case={case} n_samples={len(all_samples)} unit={unit} model_dir={model_dir}",
        flush=True,
    )
    _dual, trainers = load_trained_models(
        ppc,
        all_samples,
        t_delta,
        model_dir,
        unit_ids=[unit],
        lp_backend=str(args.lp_backend).strip().lower(),
        constraint_generation_strategy=args.constraint_strategy,
        ignore_startup_shutdown_costs=bool(args.ignore_startup_shutdown),
    )
    if unit not in trainers:
        raise RuntimeError(f"未能加载机组 {unit} 的 surrogate")
    tr = trainers[unit]
    if int(args.iter_number) >= 0:
        tr.iter_number = int(args.iter_number)
    snap = tr.collect_c_pg_loss_snapshot()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"[export] 已写入 {out.resolve()}", flush=True)
    s0 = snap["samples"][0]
    print(
        f"  示例 sample0: loss_total={s0['loss_total']:.6f} "
        f"obj_dual_pg={s0['obj_dual_pg']:.6f} reg={s0['reg_term']:.6f}",
        flush=True,
    )


def cmd_test(args: argparse.Namespace) -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    snap_path = Path(args.snapshot)
    with snap_path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    case = args.case
    ppc = get_case_ppc(case)
    active_path = Path(args.active_json)
    all_samples = _load_json_list(active_path)
    n_need = int(snap["n_samples"])
    if len(all_samples) < n_need:
        raise ValueError(
            f"active set 只有 {len(all_samples)} 条，snapshot 需要 {n_need} 条"
        )
    all_samples = all_samples[:n_need]
    t_delta = float(args.t_delta)
    unit = int(args.unit)
    if int(snap["unit_id"]) != unit:
        print(
            f"警告: snapshot unit_id={snap['unit_id']} 与 --unit {unit} 不一致，以 --unit 为准",
            flush=True,
        )
    _dual, trainers = load_trained_models(
        ppc,
        all_samples,
        t_delta,
        str(args.model_dir),
        unit_ids=[unit],
        lp_backend=str(args.lp_backend).strip().lower(),
        constraint_generation_strategy=args.constraint_strategy,
        ignore_startup_shutdown_costs=bool(args.ignore_startup_shutdown),
    )
    tr = trainers[unit]
    _apply_snapshot_to_trainer(tr, snap, strict_hparams=bool(args.strict_hparams))
    m0 = tr.cal_nn_logging_components()
    print(
        f"[test] 注入后 [NN-metric] obj_dual_pg={m0['obj_dual_pg']:.6f} reg_pg={m0['reg_pg']:.6f}",
        flush=True,
    )
    tr.iter_with_c_pg_nn(
        num_epochs=int(args.epochs),
        learning_rate=float(args.pg_cost_surr_lr)
        if args.pg_cost_surr_lr is not None
        else None,
    )
    m1 = tr.cal_nn_logging_components()
    print(
        f"[test] 训练后 [NN-metric] obj_dual_pg={m1['obj_dual_pg']:.6f} reg_pg={m1['reg_pg']:.6f}",
        flush=True,
    )
    if args.last_pth:
        p = Path(args.last_pth)
        p.parent.mkdir(parents=True, exist_ok=True)
        tr.save(str(p))
        print(f"[test] 已保存 c_pg 后 checkpoint: {p.resolve()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(c: argparse.ArgumentParser) -> None:
        c.add_argument("--case", type=str, default="case118")
        c.add_argument(
            "--active-json",
            type=str,
            required=True,
            help="与训练时相同的 active set JSON 路径",
        )
        c.add_argument("--t-delta", type=float, default=1.0)
        c.add_argument(
            "--lp-backend",
            type=str,
            default="gurobi",
            help="须与训练 surrogate 时一致",
        )
        c.add_argument(
            "--constraint-strategy",
            type=str,
            default=None,
            help="None 表示从 checkpoint 读；若指定须与 pth 一致",
        )
        c.add_argument(
            "--ignore-startup-shutdown",
            action="store_true",
            help="与 SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN 对齐（case118 常开）",
        )

    e = sub.add_parser("export", help="从 checkpoint 导出 c_pg loss 快照")
    add_common(e)
    e.add_argument("--model-dir", type=str, required=True)
    e.add_argument("--unit", type=int, required=True)
    e.add_argument("--out", type=str, required=True, help="输出 JSON 路径")
    e.add_argument("--max-samples", type=int, default=None, help="截取前 N 条，与训练一致")
    e.add_argument(
        "--iter-number",
        type=int,
        default=-1,
        help="写入 trainer.iter_number 仅用于元数据；默认 -1 不改",
    )

    t = sub.add_parser("test", help="注入快照并只训练 c_pg")
    add_common(t)
    t.add_argument("--snapshot", type=str, required=True, help="export 产生的 JSON")
    t.add_argument("--model-dir", type=str, required=True)
    t.add_argument("--unit", type=int, required=True)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument(
        "--pg-cost-surr-lr",
        type=float,
        default=None,
        help="覆盖 c_pg 步学习率；默认 None 用 trainer.pg_cost_surr_lr",
    )
    t.add_argument(
        "--strict-hparams",
        action="store_true",
        help="从 snapshot 写回 rho/正则等，使 loss 与导出时同尺度",
    )
    t.add_argument(
        "--last-pth",
        type=str,
        default="",
        help="若设路径，在 test 结束 save 该 surrogate pth",
    )

    args = p.parse_args()
    if args.cmd == "export":
        cmd_export(args)
    else:
        cmd_test(args)


if __name__ == "__main__":
    main()
