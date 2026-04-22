#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 子问题 BCD 轻量并行入口。

在标准 ``subproblem_bcd`` 配置基础上，减少训练样本数与并行度，便于联调、冒烟测试或资源紧张环境。
子问题 ``c_pg`` 等超参由 ``run_training_case118.py`` 内 ``CASE118_SUBPROBLEM_*`` 常量统一设定。

外循环轮次默认由 preset 决定（server=160，desktop=120）；predictor warmup 默认为 MaxIter 的 10%。
可通过 ``--max-iter`` / ``--warmup-rounds`` 在命令行覆盖。

示例::

    python run_training_case118_subproblem_bcd_light.py
    python run_training_case118_subproblem_bcd_light.py --max-samples 64 --n-workers-unit 2 --n-workers-sample 2
    python run_training_case118_subproblem_bcd_light.py --preset desktop --max-samples 32
    python run_training_case118_subproblem_bcd_light.py --units 0,1,2
    python run_training_case118_subproblem_bcd_light.py --max-iter 60 --warmup-rounds 6
"""

from __future__ import annotations

import argparse

import run_training_case118 as case118_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-samples",
        type=int,
        default=1,
        metavar="N",
        help="截取前 N 个样本（传给 run_training.MAX_SAMPLES），默认 1",
    )
    p.add_argument(
        "--n-workers-unit",
        type=int,
        default=2,
        metavar="K",
        help="机组级并行进程数（run_training.N_WORKERS_UNIT），默认 2",
    )
    p.add_argument(
        "--n-workers-sample",
        type=int,
        default=2,
        metavar="K",
        help="样本级并行线程数（N_WORKERS_SAMPLE / N_WORKERS_SUBPROBLEM），默认 2",
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
        help="仅训练所列机组 ID（逗号分隔，如 0,1,2）；写入 CASE118_SUBPROBLEM_UNIT_IDS",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help=(
            "覆盖 SUBPROBLEM_MAX_ITER（默认 server=160 / desktop=120）；"
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
    case118_cfg.main()


if __name__ == "__main__":
    main()
