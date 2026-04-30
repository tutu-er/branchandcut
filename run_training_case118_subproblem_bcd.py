#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 子问题 BCD 入口（完整训练预设）。

与 ``run_training_case118_subproblem_bcd_light.py`` 的区别：不截断样本数、默认更高并行度。
仅训练部分机组：``--units 0,1,2`` 或在本仓库 ``run_training_case118.py`` 中设置
``CASE118_SUBPROBLEM_UNIT_IDS = [0, 1, 2]``。

Unit predictor：由 ``run_training_case118`` 注入 standalone ``unit_predictor.pth``（默认最新 ``unit_predictor_case118_*``；
环境变量 ``CASE118_UNIT_PREDICTOR_LOAD_PATH`` 可覆盖）。

外循环轮次默认由 preset 决定（server=160，desktop=120）；predictor warmup 默认为 MaxIter 的 10%。
可通过 ``--max-iter`` / ``--warmup-rounds`` 在命令行覆盖。

默认 **Sign4 延期** 为当前 ``max_iter`` 的 10%（与 ``run_training_case118._CASE118_PCT_SUBPROBLEM_SIGN4_DELAY`` 一致）；
未传 ``--max-iter`` 时按 preset 默认轮次计算。``--sign4-delay-rounds K`` 可显式覆盖。

在 ``all_templates_sign4_plus_single`` 策略下，``--sign4-delay-rounds K`` 表示前 ``K`` 个外循环轮次
sign4 段权重强制为 0（仅训练 single-time 段）；仅靠将 ``sign4_initial_scale`` 设为 0 时，从第 1 轮起
仍会按课程爬升，不能得到「前 K 轮完全无 sign4」的 plateau。

示例::

    python run_training_case118_subproblem_bcd.py
    python run_training_case118_subproblem_bcd.py --units 0,1,2
    python run_training_case118_subproblem_bcd.py --preset desktop --max-iter 80
    python run_training_case118_subproblem_bcd.py --max-iter 200 --warmup-rounds 20
    python run_training_case118_subproblem_bcd.py --sign4-delay-rounds 20
"""

from __future__ import annotations

import argparse

import run_training_case118 as case118_cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
        help="仅训练所列机组 ID（逗号分隔）；写入 CASE118_SUBPROBLEM_UNIT_IDS",
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
    p.add_argument(
        "--sign4-delay-rounds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "覆盖 SUBPROBLEM_SIGN4_DELAY_ROUNDS：sign4+single 策略下前 K 轮 sign4 不参与（权重为 0）；"
            "默认按当前 max_iter 的 10%%（未传 --max-iter 时用 preset 默认 max_iter）"
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
