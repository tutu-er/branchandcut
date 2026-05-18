#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case118 wrapper for run_test.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_unit_ids(value: str) -> list[int] | None:
    text = (value or "").strip().lower()
    if text in ("", "all", "none"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("surrogate", "bcd", "both"), default="surrogate")
    parser.add_argument("--model-dir", type=str, default="result/surrogate_models/subproblem_models_case118_20260420_175002")
    parser.add_argument("--bcd-model", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="all_templates_sign4_plus_single")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-range", type=str, default="0:8")
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--unit-ids", type=str, default="0", help="'all' or comma-separated ids, e.g. 0,1,2")
    parser.add_argument(
        "--surrogate-constraint-scope",
        choices=("all", "sign4", "none"),
        default="all",
        help="Which surrogate subproblem constraints to apply in LP and FP tests.",
    )
    parser.add_argument(
        "--surrogate-sign4-only",
        action="store_true",
        help="Shortcut for --surrogate-constraint-scope sign4.",
    )
    parser.add_argument(
        "--no-subproblem-surrogate",
        action="store_true",
        help="Shortcut for --surrogate-constraint-scope none; keep BCD theta/zeta constraints available.",
    )
    parser.add_argument("--no-fp", action="store_true", help="Skip feasibility-pump testing.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.surrogate_sign4_only:
        args.surrogate_constraint_scope = "sign4"
    if args.no_subproblem_surrogate:
        args.surrogate_constraint_scope = "none"

    root = Path(__file__).resolve().parent

    import run_training_case118 as case118_cfg
    import run_test as rt

    active = case118_cfg.CASE118_ACTIVE_SET_JSON
    if not (root / active).exists():
        print(
            f"[run_test_case118] missing active-set JSON:\n  {root / active}",
            file=sys.stderr,
        )
        sys.exit(1)

    rt.MODE = args.mode
    rt.CASE_NAME = "case118"
    rt.ACTIVE_SETS_FILE = active
    rt.SURROGATE_CONSTRAINT_STRATEGY = args.strategy
    rt.SURROGATE_CONSTRAINT_SCOPE = args.surrogate_constraint_scope
    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = True
    rt.UNIT_IDS = _parse_unit_ids(args.unit_ids)

    rt.MODEL_DIR = args.model_dir
    rt.BCD_MODEL_PATH = args.bcd_model
    rt.MAX_SAMPLES = args.max_samples
    rt.SAMPLE_RANGE = args.sample_range
    rt.TEST_SAMPLES = max(1, int(args.test_samples))
    rt.TEST_SAMPLES_DEFAULT = rt.TEST_SAMPLES
    rt.RUN_FP = not bool(args.no_fp)

    print("=" * 72, flush=True)
    print("run_test_case118.py", flush=True)
    print(f"  mode={rt.MODE} | active_set={active}", flush=True)
    print(
        f"  strategy={rt.SURROGATE_CONSTRAINT_STRATEGY} | "
        f"surrogate_constraint_scope={rt.SURROGATE_CONSTRAINT_SCOPE} | "
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
