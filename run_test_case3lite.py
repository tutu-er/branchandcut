#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the case3lite surrogate-model test preset.

Examples
--------
    python run_test_case3lite.py
    python run_test_case3lite.py --samples 8
    python run_test_case3lite.py --fp
    python run_test_case3lite.py --model-dir result/surrogate_models/subproblem_models_case3lite_YYYYMMDD_HHMMSS
    python run_test_case3lite.py --activity-only --refresh-train-dual
    python run_test_case3lite.py --activity-only --main-activity-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CASE_NAME = "case3lite"
MODE = "surrogate"  # use "both" to also evaluate a BCD checkpoint
ACTIVE_SETS_FILE = "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"
MODEL_DIR: str | None = "result/surrogate_models/subproblem_models_case3lite_20260508_223657"
BCD_MODEL_PATH: str | None = None
TEST_SAMPLES = 10
SAMPLE_RANGE = "0:100"
SURROGATE_CONSTRAINT_STRATEGY = "all_templates_sign4_plus_single"
rt = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("surrogate", "both"), default=MODE)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--model-dir", type=str, default=MODEL_DIR)
    p.add_argument("--bcd-model", type=str, default=BCD_MODEL_PATH)
    p.add_argument("--samples", type=int, default=TEST_SAMPLES)
    p.add_argument("--sample-range", type=str, default=SAMPLE_RANGE)
    p.add_argument("--unit-ids", type=str, default="all", help="'all' or comma-separated ids, e.g. 0,1,2")
    p.add_argument("--strategy", type=str, default=SURROGATE_CONSTRAINT_STRATEGY)
    p.add_argument("--fp", action="store_true", help="Run feasibility-pump testing.")
    p.add_argument("--no-custom-fp", action="store_true", help="Disable the case3lite custom FP path when --fp is used.")
    p.add_argument("--disable-plots", action="store_true", help="Disable plot generation.")
    p.add_argument("--skip-activity", action="store_true", help="Skip surrogate dual/activity diagnostics.")
    p.add_argument("--skip-main-activity", action="store_true", help="Do not test main-model theta/zeta activity.")
    p.add_argument("--activity-only", action="store_true", help="Run only surrogate dual/activity diagnostics.")
    p.add_argument("--main-activity-only", action="store_true", help="Run only main-model theta/zeta activity diagnostics.")
    p.add_argument("--activity-train-start", type=int, default=0)
    p.add_argument("--activity-train-samples", type=int, default=32)
    p.add_argument("--activity-test-start", type=int, default=0)
    p.add_argument("--activity-test-samples", type=int, default=16)
    p.add_argument("--activity-output-dir", type=str, default=None)
    p.add_argument("--activity-lp-backend", type=str, default=None)
    p.add_argument("--activity-mu-active-tol", type=float, default=1e-7)
    p.add_argument("--activity-active-tol", type=float, default=1e-5)
    p.add_argument("--activity-violation-tol", type=float, default=1e-7)
    p.add_argument("--refresh-train-dual", action="store_true", help="Refresh training duals before activity diagnostics.")
    p.add_argument("--refresh-mode", choices=("dual-only", "full"), default="full")
    p.add_argument("--refresh-nn-epochs", type=int, default=None)
    p.add_argument("--refresh-pg-cost-nn-epochs", type=int, default=None)
    p.add_argument("--refresh-total-max-iter", type=int, default=None)
    return p.parse_args()


def _parse_unit_ids(value: str) -> list[int] | None:
    text = (value or "").strip().lower()
    if text in ("", "all", "none"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _ensure_bcd_imports() -> None:
    global rt
    if rt.MODE != "both":
        return
    if getattr(rt, "Agent_NN_BCD", None) is not None:
        return
    from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json

    rt.Agent_NN_BCD = Agent_NN_BCD
    rt.load_active_set_from_json = load_active_set_from_json


def _append_optional(argv: list[str], flag: str, value) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _run_activity_check(args: argparse.Namespace) -> None:
    from scripts import test_surrogate_dual_activity as activity

    output_dir = (
        Path(args.activity_output_dir)
        if args.activity_output_dir
        else (
            Path(__file__).resolve().parent
            / "result"
            / "figures"
            / "case3lite_global_surrogate_solve_stats"
            / "activity"
        )
    )
    argv = [
        "test_surrogate_dual_activity.py",
        "--case",
        CASE_NAME,
        "--active-set-json",
        args.active_sets,
        "--model-dir",
        args.model_dir,
        "--train-start",
        str(max(0, int(args.activity_train_start))),
        "--train-samples",
        str(max(1, int(args.activity_train_samples))),
        "--test-start",
        str(max(0, int(args.activity_test_start))),
        "--test-samples",
        str(max(1, int(args.activity_test_samples))),
        "--mu-active-tol",
        str(float(args.activity_mu_active_tol)),
        "--active-tol",
        str(float(args.activity_active_tol)),
        "--violation-tol",
        str(float(args.activity_violation_tol)),
        "--output-dir",
        str(output_dir),
    ]
    if str(args.strategy).strip().lower() != "auto":
        argv.extend(["--strategy", args.strategy])
    if args.unit_ids.strip().lower() not in ("", "all", "none"):
        argv.extend(["--units", args.unit_ids])
    if args.activity_lp_backend:
        argv.extend(["--lp-backend", args.activity_lp_backend])
    if not args.skip_main_activity:
        argv.append("--main-activity")
        if args.bcd_model:
            argv.extend(["--bcd-model", args.bcd_model])
    if args.main_activity_only:
        argv.append("--main-only")
    if args.disable_plots:
        argv.append("--no-plots")
    if args.refresh_train_dual:
        argv.append("--refresh-train-dual")
        argv.extend(["--refresh-mode", args.refresh_mode])
        _append_optional(argv, "--refresh-nn-epochs", args.refresh_nn_epochs)
        _append_optional(argv, "--refresh-pg-cost-nn-epochs", args.refresh_pg_cost_nn_epochs)
        _append_optional(argv, "--refresh-total-max-iter", args.refresh_total_max_iter)

    print("=" * 72, flush=True)
    print(
        "case3lite surrogate activity check | "
        f"train={args.activity_train_start}:{args.activity_train_samples} | "
        f"test={args.activity_test_start}:{args.activity_test_samples} | "
        f"main_activity={not args.skip_main_activity} | "
        f"output_dir={output_dir}",
        flush=True,
    )
    print("=" * 72, flush=True)

    old_argv = sys.argv
    try:
        sys.argv = argv
        activity.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    global rt
    args = _parse_args()

    if not args.activity_only:
        import run_test as rt_module

        rt = rt_module
        rt.CASE_NAME = CASE_NAME
        rt.MODE = args.mode
        rt.ACTIVE_SETS_FILE = args.active_sets
        rt.MODEL_DIR = args.model_dir
        rt.BCD_MODEL_PATH = args.bcd_model
        rt.SURROGATE_CONSTRAINT_STRATEGY = args.strategy
        rt.UNIT_IDS = _parse_unit_ids(args.unit_ids)
        rt.MAX_SAMPLES = None
        rt.SAMPLE_RANGE = args.sample_range
        rt.TEST_SAMPLES = max(1, int(args.samples))
        rt.TEST_SAMPLES_DEFAULT = rt.TEST_SAMPLES
        rt.RUN_FP = bool(args.fp)
        rt.RUN_SUBPROBLEM_MILP_TEST = False
        rt.RUN_TEST_DISABLE_PLOTS = bool(args.disable_plots)
        rt.USE_CASE3LITE_CUSTOM_FP = bool(args.fp) and not bool(args.no_custom_fp)
        rt.USE_CASE118_CUSTOM_FP = False

        print("=" * 72, flush=True)
        print(
            "case3lite test | "
            f"mode={rt.MODE} | model_dir={rt.MODEL_DIR} | "
            f"active_sets={rt.ACTIVE_SETS_FILE} | "
            f"unit_ids={rt.UNIT_IDS if rt.UNIT_IDS is not None else 'all'} | "
            f"samples={rt.TEST_SAMPLES} | sample_range={rt.SAMPLE_RANGE} | "
            f"strategy={rt.SURROGATE_CONSTRAINT_STRATEGY} | "
            f"fp={rt.RUN_FP} | custom_fp={rt.USE_CASE3LITE_CUSTOM_FP} | "
            f"plots={not rt.RUN_TEST_DISABLE_PLOTS}",
            flush=True,
        )
        print("=" * 72, flush=True)

        _ensure_bcd_imports()
        rt.main()

    if not args.skip_activity:
        _run_activity_check(args)


if __name__ == "__main__":
    main()
