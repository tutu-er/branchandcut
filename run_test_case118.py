#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 测试入口：在 ``run_test.py`` 默认配置基础上对齐子问题训练设定。

与 ``run_training_case118.py`` 使用相同的 ``CASE118_ACTIVE_SET_JSON``，并默认采用与
``run_training.SURROGATE_CONSTRAINT_STRATEGY`` 一致的 ``all_templates_sign4_plus_single``，
以及 ``SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS=True``（与 case118 子问题 BCD 训练一致）。

在项目根目录运行::

    python run_test_case118.py
    python run_test_case118.py --model-dir result/surrogate_models/subproblem_models_case118_YYYYMMDD_HHMMSS
    python run_test_case118.py --mode surrogate --no-fp --max-samples 8
    python run_test_case118.py --mode both --model-dir ... --bcd-model result/bcd_models/bcd_model_case118_xxx.pth

环境变量（与 ``run_test.py`` 相同）::

    RUN_TEST_SURROGATE_MODEL_DIR=...   # 覆盖 surrogate 目录
    RUN_TEST_BCD_MODEL_PATH=...       # 覆盖 BCD 权重
    RUN_TEST_DISABLE_PLOTS=1          # 跳过绘图（CI / 无显示环境）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

def main() -> None:
    # -----------------------
    # 显式参数设置（不使用命令行）
    # -----------------------
    mode: str = "surrogate"  # "surrogate" | "bcd" | "both"
    model_dir: str | None = "result/surrogate_models/subproblem_models_case118_20260420_175002"  # None=自动选 result/surrogate_models/subproblem_models_case118_* 最新
    bcd_model: str | None = None  # None=自动发现 result/bcd_models/bcd_model_case118_*.pth
    strategy: str = "all_templates_sign4_plus_single"  # 或 "auto" 让 run_test 从 checkpoint 读取
    max_samples: int | None = None
    sample_range: str = "0:8"
    test_samples: int = 8
    run_fp: bool = True
    units: list[int] | None = [0]  # None=全部；或例如 [1,2,3]

    root = Path(__file__).resolve().parent
    # 与训练入口共用 active set 路径，避免数据/λ 分布不一致
    import run_training_case118 as case118_cfg

    active = case118_cfg.CASE118_ACTIVE_SET_JSON
    if not (root / active).exists():
        print(
            f"[run_test_case118] 错误: 找不到 active set 文件（与训练共用）:\n  {root / active}",
            file=sys.stderr,
        )
        sys.exit(1)

    import run_test as rt

    rt.MODE = mode
    rt.CASE_NAME = "case118"
    rt.ACTIVE_SETS_FILE = active
    rt.SURROGATE_CONSTRAINT_STRATEGY = strategy
    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
    rt.UNIT_IDS = units

    rt.MODEL_DIR = model_dir
    rt.BCD_MODEL_PATH = bcd_model
    rt.MAX_SAMPLES = max_samples
    rt.SAMPLE_RANGE = sample_range
    rt.TEST_SAMPLES = max(1, int(test_samples))
    rt.RUN_FP = bool(run_fp)

    print("=" * 72, flush=True)
    print("run_test_case118.py", flush=True)
    print(f"  mode={rt.MODE} | active_set={active}", flush=True)
    print(
        f"  strategy={rt.SURROGATE_CONSTRAINT_STRATEGY} | "
        f"ignore_startup_shutdown_costs={rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS}",
        flush=True,
    )
    print(
        f"  sample_range={rt.SAMPLE_RANGE} | max_samples={rt.MAX_SAMPLES} | "
        f"test_samples={rt.TEST_SAMPLES} | run_fp={rt.RUN_FP}",
        flush=True,
    )
    print(f"  unit_ids={rt.UNIT_IDS!r}", flush=True)
    print(f"  model_dir={rt.MODEL_DIR!r} | bcd_model={rt.BCD_MODEL_PATH!r}", flush=True)
    if os.environ.get("RUN_TEST_SURROGATE_MODEL_DIR", "").strip():
        print(
            f"  env RUN_TEST_SURROGATE_MODEL_DIR={os.environ['RUN_TEST_SURROGATE_MODEL_DIR']!r}",
            flush=True,
        )
    if os.environ.get("RUN_TEST_BCD_MODEL_PATH", "").strip():
        print(
            f"  env RUN_TEST_BCD_MODEL_PATH={os.environ['RUN_TEST_BCD_MODEL_PATH']!r}",
            flush=True,
        )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
