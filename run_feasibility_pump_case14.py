#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the case14-specialized feasibility-pump evaluation preset."""

from __future__ import annotations

import argparse

import run_test as rt


CASE_NAME = "case14"
MODE = "both"  # use "surrogate" to evaluate only subproblem models
ACTIVE_SETS_FILE: str | None = None
MODEL_DIR: str | None = None
BCD_MODEL_PATH: str | None = None
TEST_SAMPLES = 12
SAMPLE_RANGE = "0:120"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("surrogate", "both"), default=MODE)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--model-dir", type=str, default=MODEL_DIR)
    p.add_argument("--bcd-model", type=str, default=BCD_MODEL_PATH)
    p.add_argument("--samples", type=int, default=TEST_SAMPLES)
    p.add_argument("--sample-range", type=str, default=SAMPLE_RANGE)
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
    rt.USE_CASE3LITE_CUSTOM_FP = False
    rt.USE_CASE118_CUSTOM_FP = False
    rt.FP_CONF_THRESHOLD = 0.18
    rt.FP_MAX_ITER = 70
    rt.FP_MAX_PERTURBATION_HOT_STARTS = 8
    rt.FP_MAX_UNIT_OPTIONS_PER_GENERATOR = 5
    rt.FP_MAX_UNIT_COMBINATION_CANDIDATES = 18
    rt.FP_MAX_NEARBY_COMMITMENT_HOT_STARTS = 5
    rt.FP_NEARBY_COMMITMENT_POOL_SIZE = 16
    rt.FP_PARALLEL_STARTS = 2
    rt.FP_SURROGATE_SCREEN_MODE = "robust"
    rt.FP_SURROGATE_SCREEN_MAX_CONSTRAINTS_PER_UNIT = 4
    rt.FP_SURROGATE_SCREEN_MIN_SUPPORT_RATIO = 0.80
    rt.FP_SURROGATE_SCREEN_MAX_NORMALIZED_VIOLATION = 0.07
    rt.FP_SURROGATE_SCREEN_MIN_MEAN_MARGIN = 0.015
    rt.FP_PROJECTION_OBJECTIVE_TAU = "adaptive"

    print("=" * 72, flush=True)
    print(
        f"case14 FP | mode={rt.MODE} | samples={rt.TEST_SAMPLES} | "
        f"model_dir={rt.MODEL_DIR or 'auto-latest'} | bcd={rt.BCD_MODEL_PATH or 'auto-latest'}",
        flush=True,
    )
    print("=" * 72, flush=True)
    _ensure_bcd_imports()
    rt.main()


if __name__ == "__main__":
    main()
