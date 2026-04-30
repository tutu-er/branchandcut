#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 子问题 BCD 轻量并行入口。

在标准 ``subproblem_bcd`` 配置基础上，减少训练样本数与并行度，便于联调、冒烟测试或资源紧张环境。
**默认：1 样本、desktop 预设、机组/样本并行各 1**（本机最小冒烟）。服务器上想试更大规模请用
``run_training_case118_subproblem_bcd_medium.py``。

子问题 BCD 内超参（含 ``c_pg``、surrogate NN）由 ``run_training_case118.py`` 的 ``CASE118_SUBPROBLEM_*`` 设定。
每轮 BCD 中的 **direct-NN-main** / **direct-c_pg** 预训练使用 ``run_training.SUBPROBLEM_*_DIRECT_*``，
在 ``_configure_subproblem_bcd`` 内已与 Case118 的 full-batch、学习率等对齐（见 ``CASE118_SUBPROBLEM_MAIN_DIRECT_EPOCHS``、
``CASE118_SUBPROBLEM_C_PG_DIRECT_EPOCHS`` 及紧随其后的 ``rt.SUBPROBLEM_*_DIRECT_*`` 赋值）；本轻量脚本不单独关闭该路径。

Unit predictor（独立脚本 ``run_unit_predictor_case118.py`` 产出的 ``unit_predictor.pth``）由 ``run_training_case118``
``CASE118_UNIT_PREDICTOR_*`` / ``_resolve_case118_unit_predictor_load_path`` 注入 ``run_training``（默认选用最新的
``result/surrogate_models/unit_predictor_case118_*``）；可用环境变量 ``CASE118_UNIT_PREDICTOR_LOAD_PATH`` 覆盖。

外循环轮次默认由 preset 决定（与 ``run_training_case118.CASE118_SUBPROBLEM_MAX_ITER_*`` 一致）；
``--preset`` 默认为 ``desktop``（与 ``run_training_case118.py`` 顶部一致）；predictor warmup 默认为 MaxIter 的 10%。
可通过 ``--max-iter`` / ``--warmup-rounds`` 在命令行覆盖。
默认 **Sign4 延期** 为当前 ``max_iter`` 的 10%（四舍五入）；未传 ``--max-iter`` 时按 preset 默认轮次计算。
可用 ``--sign4-delay-rounds K`` 显式覆盖（详见 ``run_training_case118_subproblem_bcd.py`` 说明）。

示例::

    python run_training_case118_subproblem_bcd_light.py
    python run_training_case118_subproblem_bcd_light.py --max-samples 64 --n-workers-unit 2 --n-workers-sample 2
    python run_training_case118_subproblem_bcd_light.py --preset server --max-samples 32
    python run_training_case118_subproblem_bcd_light.py --units 0,1,2
    python run_training_case118_subproblem_bcd_light.py --max-iter 60 --warmup-rounds 6
    python run_training_case118_subproblem_bcd_light.py --sign4-delay-rounds 5

中等规模（服务器默认 64 样本 / server）::

    python run_training_case118_subproblem_bcd_medium.py
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
        default=1,
        metavar="K",
        help="机组级并行进程数（run_training.N_WORKERS_UNIT），默认 1（轻量）",
    )
    p.add_argument(
        "--n-workers-sample",
        type=int,
        default=1,
        metavar="K",
        help="样本级并行线程数（N_WORKERS_SAMPLE / N_WORKERS_SUBPROBLEM），默认 1（轻量）",
    )
    p.add_argument(
        "--preset",
        choices=("server", "desktop"),
        default="desktop",
        help="与 run_training_case118.SUBPROBLEM_SOLVE_PRESET 相同，默认 desktop",
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
            "覆盖 SUBPROBLEM_SIGN4_DELAY_ROUNDS（sign4+single：前 K 轮 sign4 权重为 0）；"
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
