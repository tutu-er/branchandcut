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

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("surrogate", "bcd", "both"),
        default="surrogate",
        help="测试模式，与 run_test.MODE 一致，默认 surrogate（加载子问题代理+V3）",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="surrogate 输出目录（含 dual_predictor.pth 与 surrogate_unit_*.pth）；"
        "默认 None=自动选 result/surrogate_models/subproblem_models_case118_* 最新",
    )
    p.add_argument(
        "--bcd-model",
        type=str,
        default=None,
        metavar="PATH",
        help="bcd / both 模式下的 BCD .pth；默认自动发现 result/bcd_models/bcd_model_case118_*.pth",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="all_templates_sign4_plus_single",
        metavar="NAME",
        help="代理约束生成策略，须与训练时一致；也可设 auto 让 run_test 从 checkpoint 读取",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="最多使用前 N 条样本（run_test.MAX_SAMPLES）",
    )
    p.add_argument(
        "--sample-range",
        type=str,
        default="0:8",
        metavar="START:END",
        help="左闭右开区间，默认 0:8 做快速抽查",
    )
    p.add_argument(
        "--test-samples",
        type=int,
        default=8,
        metavar="K",
        help="若干分析/FP 子流程使用的样本数上界（run_test.TEST_SAMPLES）",
    )
    p.add_argument(
        "--no-fp",
        action="store_true",
        help="关闭可行性泵（run_test.RUN_FP=False），显著加快",
    )
    p.add_argument(
        "--units",
        type=str,
        default=None,
        metavar="IDS",
        help="仅评估所列机组代理（逗号分隔）；默认 None=全部 checkpoint 中的机组",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

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

    rt.MODE = args.mode
    rt.CASE_NAME = "case118"
    rt.ACTIVE_SETS_FILE = active
    rt.SURROGATE_CONSTRAINT_STRATEGY = args.strategy
    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
    rt.UNIT_IDS = None
    if args.units:
        rt.UNIT_IDS = [int(x.strip()) for x in str(args.units).split(",") if x.strip()]

    rt.MODEL_DIR = args.model_dir
    rt.BCD_MODEL_PATH = args.bcd_model
    rt.MAX_SAMPLES = args.max_samples
    rt.SAMPLE_RANGE = args.sample_range
    rt.TEST_SAMPLES = max(1, int(args.test_samples))
    rt.RUN_FP = not args.no_fp

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
