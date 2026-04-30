#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 子问题 BCD **中等规模** 入口（适合服务器试跑）。

由 ``run_training_case118_subproblem_bcd_light.py`` 复制而来，逻辑相同，仅 **默认规模** 不同：
默认使用 **server** 预设（``CASE118_SUBPROBLEM_MAX_ITER_SERVER=200``、HiGHS 子问题 LP、较高并行度）、
**64 个样本**、机组/样本并行度偏大。子问题超参仍全部由 ``run_training_case118.py`` 的 ``CASE118_SUBPROBLEM_*`` 决定。

Unit predictor：与 light 入口相同，由 ``run_training_case118`` 解析 ``unit_predictor.pth``（见该模块文档）。

可在命令行用与 light 相同的参数覆盖（如 ``--max-samples``、``--preset desktop``）。
默认 **Sign4 延期** 为当前 ``max_iter`` 的 10%；未传 ``--max-iter`` 时按 preset 默认轮次计算；
``--sign4-delay-rounds`` 可显式覆盖。

示例::

    # 服务器默认（64 样本 / server / 并行 4×16）
    python run_training_case118_subproblem_bcd_medium.py

    # 更大样本或指定机组
    python run_training_case118_subproblem_bcd_medium.py --max-samples 128 --units 2
    python run_training_case118_subproblem_bcd_medium.py --max-samples 32 --units 0,1,2

    # 显式指定外循环轮次（会按比例重算 warmup / mu-floor）
    python run_training_case118_subproblem_bcd_medium.py --max-iter 200 --warmup-rounds 20
"""

from __future__ import annotations

import argparse

import run_training_case118 as case118_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-samples",
        type=int,
        default=64,
        metavar="N",
        help="截取前 N 个样本（传给 run_training.MAX_SAMPLES），默认 64（中等规模）",
    )
    p.add_argument(
        "--n-workers-unit",
        type=int,
        default=4,
        metavar="K",
        help="机组级并行进程数（run_training.N_WORKERS_UNIT），默认 4",
    )
    p.add_argument(
        "--n-workers-sample",
        type=int,
        default=16,
        metavar="K",
        help="样本级并行线程数（N_WORKERS_SAMPLE / N_WORKERS_SUBPROBLEM），默认 16",
    )
    p.add_argument(
        "--preset",
        choices=("server", "desktop"),
        default="server",
        help="与 run_training_case118.SUBPROBLEM_SOLVE_PRESET 相同，默认 server",
    )
    p.add_argument(
        "--units",
        type=str,
        default=None,
        metavar="IDS",
        help="仅训练所列机组 ID（逗号分隔，如 0,1,2）；未指定则用 run_training_case118 内 CASE118_SUBPROBLEM_UNIT_IDS",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help=(
            "覆盖 SUBPROBLEM_MAX_ITER（未指定时由 preset 取 CASE118_SUBPROBLEM_MAX_ITER_SERVER/DESKTOP）；"
            "同时按比例重算 warmup / mu-floor 轮次"
        ),
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=None,
        metavar="W",
        help=(
            "覆盖 SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS（默认为 max_iter 的 10%%）；"
            "设为 0 可禁用 predictor 热身（Plan B 实现后生效）"
        ),
    )
    p.add_argument(
        "--sign4-delay-rounds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "覆盖 SUBPROBLEM_SIGN4_DELAY_ROUNDS；默认 max_iter 的 10%%（未传 --max-iter 时用 preset 默认）"
        ),
    )
    p.add_argument(
        "--delta-reference-lift",
        choices=("auto", "on", "off"),
        default="auto",
        help="surrogate delta reference lift: auto enables it for sign4 strategies",
    )
    p.add_argument(
        "--delta-reference-scope",
        choices=("sign4_only", "all_coupling"),
        default="sign4_only",
        help="scope for surrogate delta reference lift",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    case118_cfg.TRAIN_TARGET = "subproblem_bcd"
    case118_cfg.SUBPROBLEM_SOLVE_PRESET = args.preset
    case118_cfg.SUBPROBLEM_LIGHT_MAX_SAMPLES = max(1, int(args.max_samples))
    case118_cfg.SUBPROBLEM_LIGHT_N_WORKERS_UNIT = max(1, int(args.n_workers_unit))
    case118_cfg.SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE = max(1, int(args.n_workers_sample))
    if args.units is not None:
        parts = [p.strip() for p in str(args.units).split(",") if p.strip()]
        case118_cfg.CASE118_SUBPROBLEM_UNIT_IDS = [int(x) for x in parts]
    if args.max_iter is not None:
        case118_cfg.SUBPROBLEM_LIGHT_MAX_ITER = max(1, int(args.max_iter))
    if args.warmup_rounds is not None:
        case118_cfg.SUBPROBLEM_LIGHT_PREDICTOR_WARMUP_ROUNDS = max(0, int(args.warmup_rounds))
    delay_resolved = case118_cfg.default_subproblem_entry_sign4_delay_rounds(
        preset=args.preset,
        max_iter_cli=args.max_iter,
        sign4_cli=args.sign4_delay_rounds,
    )
    if delay_resolved is not None:
        case118_cfg.SUBPROBLEM_LIGHT_SIGN4_DELAY_ROUNDS = int(delay_resolved)
    if args.delta_reference_lift == "auto":
        case118_cfg.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = None
    else:
        case118_cfg.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = (
            args.delta_reference_lift == "on"
        )
    case118_cfg.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE = args.delta_reference_scope
    case118_cfg.main()


if __name__ == "__main__":
    main()
