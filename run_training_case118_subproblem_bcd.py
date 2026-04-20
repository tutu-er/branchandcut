#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 子问题 BCD 入口（完整训练预设）。

与 ``run_training_case118_subproblem_bcd_light.py`` 的区别：不截断样本数、默认更高并行度。
仅训练部分机组：``--units 0,1,2`` 或在本仓库 ``run_training_case118.py`` 中设置
``CASE118_SUBPROBLEM_UNIT_IDS = [0, 1, 2]``。

示例::

    python run_training_case118_subproblem_bcd.py
    python run_training_case118_subproblem_bcd.py --units 0,1,2
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    case118_cfg.TRAIN_TARGET = "subproblem_bcd"
    case118_cfg.SUBPROBLEM_SOLVE_PRESET = args.preset
    if args.units is not None:
        parts = [p.strip() for p in str(args.units).split(",") if p.strip()]
        case118_cfg.CASE118_SUBPROBLEM_UNIT_IDS = [int(x) for x in parts]
    case118_cfg.main()


if __name__ == "__main__":
    main()
