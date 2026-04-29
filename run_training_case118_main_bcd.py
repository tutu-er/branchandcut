#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 主问题 BCD 入口。

子问题 UnitPredictor（``run_unit_predictor_case118.py`` 产出的 ckpt）仅在 ``run_training_case118`` 选择
``subproblem_bcd`` 时通过 ``CASE118_UNIT_PREDICTOR_*`` 注入 ``run_training``；本入口不使用 UnitPredictor。
"""

from __future__ import annotations

import run_training_case118 as case118_cfg


case118_cfg.TRAIN_TARGET = "main_bcd"


if __name__ == "__main__":
    case118_cfg.main()
