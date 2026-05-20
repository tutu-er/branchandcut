#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the case3lite-specialized feasibility-pump evaluation preset."""

from __future__ import annotations

import argparse

import run_test as rt


CASE_NAME = "case3lite"
MODE = "both"  # use "surrogate" to evaluate only subproblem models
ACTIVE_SETS_FILE: str | None = (
    "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"
)
MODEL_DIR: str | None = "result/surrogate_models/subproblem_models_case3lite_20260510_merge"
BCD_MODEL_PATH: str | None = "result/bcd_models/bcd_model_case3lite_20260519_235955.pth"
TEST_SAMPLES = 10
SAMPLE_RANGE = "0:100"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("surrogate", "both"), default=MODE)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--model-dir", type=str, default=MODEL_DIR)
    p.add_argument("--bcd-model", type=str, default=BCD_MODEL_PATH)
    p.add_argument("--samples", type=int, default=TEST_SAMPLES)
    p.add_argument("--sample-range", type=str, default=SAMPLE_RANGE)
    p.add_argument("--no-custom-fp", action="store_true")
    p.add_argument(
        "--theta-flip",
        action="store_true",
        help="Use the pinned theta_flip_case3lite benchmark flow "
             "(see run_case3lite_theta_flip_fp.py).",
    )
    p.add_argument(
        "--error-bit-map",
        type=str,
        default="result/fp_diagnostics/history_error_bit_map_case3lite_n50_20260519_235955_merge.json",
    )
    p.add_argument("--flip-top-k", type=int, default=10)
    return p.parse_args()


def _ensure_bcd_imports() -> None:
    if rt.MODE not in ("bcd", "both"):
        return
    if getattr(rt, "Agent_NN_BCD", None) is not None:
        return
    from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json

    rt.Agent_NN_BCD = Agent_NN_BCD
    rt.load_active_set_from_json = load_active_set_from_json


def main() -> None:
    args = _parse_args()
    if args.theta_flip:
        from run_case3lite_theta_flip_fp import cmd_bench

        bench_args = argparse.Namespace(
            active_sets=args.active_sets or ACTIVE_SETS_FILE,
            model_dir=args.model_dir or MODEL_DIR,
            bcd_model=args.bcd_model or BCD_MODEL_PATH,
            start=0,
            log=None,
            error_bit_map=args.error_bit_map,
            flip_top_k=int(args.flip_top_k),
            strategies="theta_flip_case3lite,vanilla",
            samples=max(1, int(args.samples)),
            max_fp_iter=25,
            output=(
                f"result/fp_diagnostics/bench_fp_theta_case3lite_n{max(1, int(args.samples))}.json"
            ),
        )
        raise SystemExit(cmd_bench(bench_args))

    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.mode
    rt.RUN_FP = True
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MODEL_DIR = args.model_dir
    rt.BCD_MODEL_PATH = args.bcd_model
    rt.UNIT_IDS = None
    rt.MAX_SAMPLES = None
    rt.SAMPLE_RANGE = args.sample_range
    rt.TEST_SAMPLES = max(1, int(args.samples))
    rt.RUN_TEST_DISABLE_PLOTS = True
    rt.USE_CASE3LITE_CUSTOM_FP = not bool(args.no_custom_fp)
    rt.USE_CASE118_CUSTOM_FP = False
    rt.FP_CONF_THRESHOLD = 0.15
    rt.FP_MAX_ITER = 50
    rt.FP_MAX_PERTURBATION_HOT_STARTS = 6
    rt.FP_MAX_UNIT_OPTIONS_PER_GENERATOR = 4
    rt.FP_MAX_UNIT_COMBINATION_CANDIDATES = 12
    rt.FP_MAX_NEARBY_COMMITMENT_HOT_STARTS = 4
    rt.FP_NEARBY_COMMITMENT_POOL_SIZE = 12
    rt.FP_PARALLEL_STARTS = 2
    rt.FP_SURROGATE_SCREEN_MODE = "robust"
    rt.FP_SURROGATE_SCREEN_MAX_CONSTRAINTS_PER_UNIT = 3
    rt.FP_SURROGATE_SCREEN_MIN_SUPPORT_RATIO = 0.85
    rt.FP_SURROGATE_SCREEN_MAX_NORMALIZED_VIOLATION = 0.05
    rt.FP_SURROGATE_SCREEN_MIN_MEAN_MARGIN = 0.02
    rt.FP_SURROGATE_SCREEN_CANDIDATE_VIOLATION_TOL = 0.02
    rt.FP_SURROGATE_SCREEN_SOFT_PENALTY = 25.0
    rt.FP_PROJECTION_OBJECTIVE_TAU = "adaptive"
    rt.CASE3LITE_CUSTOM_FP_MAX_GLOBAL_COMBINATIONS = 24

    print("=" * 72, flush=True)
    print(
        f"case3lite FP | mode={rt.MODE} | samples={rt.TEST_SAMPLES} | "
        f"custom_fp={rt.USE_CASE3LITE_CUSTOM_FP} | "
        f"model_dir={rt.MODEL_DIR or 'auto-latest'} | "
        f"bcd={rt.BCD_MODEL_PATH or 'auto-latest'}",
        flush=True,
    )
    print("=" * 72, flush=True)
    _ensure_bcd_imports()
    rt.main()


if __name__ == "__main__":
    main()
