#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train main BCD and subproblem surrogate models for lightweight case30."""

from __future__ import annotations

import argparse

import run_training as rt


CASE_NAME = "case30lite"
TRAIN_TARGET = "both"  # "both" | "surrogate" | "bcd"
ACTIVE_SETS_FILE: str | None = None
RESUME_SURROGATE_DIR: str | None = None
MAX_SAMPLES = 100
BCD_MAX_ITER = 110
SUBPROBLEM_MAX_ITER = 60
DUAL_EPOCHS = 160
UNIT_PREDICTOR_EPOCHS = 500
NN_EPOCHS = 4
N_WORKERS_SAMPLE = 2
N_WORKERS_UNIT = 1
N_WORKERS_BCD = 1
SUBPROBLEM_LP_BACKEND = "cvxpy_highs"
BCD_LP_BACKEND = "gurobi"
SURROGATE_CONSTRAINT_STRATEGY = "all_templates_sign4_plus_single"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("both", "surrogate", "bcd"), default=TRAIN_TARGET)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--resume-dir", type=str, default=RESUME_SURROGATE_DIR)
    p.add_argument("--retrain-existing", action="store_true")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--bcd-iter", type=int, default=BCD_MAX_ITER)
    p.add_argument("--sub-iter", type=int, default=SUBPROBLEM_MAX_ITER)
    p.add_argument("--sample-workers", type=int, default=N_WORKERS_SAMPLE)
    p.add_argument("--unit-ids", type=str, default=None,
                   help="Comma-separated unit ids to train, e.g. '1' or '0,2'.")
    p.add_argument(
        "--no-unit-predictor",
        action="store_true",
        help="Disable unit commitment predictor (no warm-start heuristic in subproblem/BCD).",
    )
    p.add_argument(
        "--vanilla-subproblem",
        action="store_true",
        help=(
            "Disable common subproblem-side heuristics: μ dual floor schedule, predictor "
            "warmup rounds, surrogate δ reference lift, delayed c_pg branch start, "
            "sign4 delay/curriculum; also clear BCD-side predictor warmup / theta delay; "
            "sets c_pg NN epochs to 4 unless --pg-cost-nn-epochs is given."
        ),
    )
    p.add_argument(
        "--nn-no-direct",
        action="store_true",
        help=(
            "Set subproblem main + c_pg direct-target pretrain epochs to 0 "
            "(no direct-NN warm-start; BCD inner NN epochs unchanged)."
        ),
    )
    p.add_argument(
        "--metrics-tag",
        type=str,
        default=None,
        help="Optional tag: result/training_metrics_<case>_<tag>_<timestamp>.json (e.g. control).",
    )
    p.add_argument(
        "--pg-cost-nn-epochs",
        type=int,
        default=None,
        help=(
            "Subproblem c_pg branch NN epochs per BCD iteration "
            "(default: 4 when --vanilla-subproblem, else run_training default e.g. 64)."
        ),
    )
    return p.parse_args()


def _parse_unit_ids(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    unit_ids: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        unit_ids.append(int(item))
    return unit_ids or None


def _configure_iterations(bcd_iter: int, sub_iter: int) -> None:
    rt.MAX_ITER = max(int(bcd_iter), int(sub_iter))
    rt.BCD_MAX_ITER = max(1, int(bcd_iter))
    rt.SUBPROBLEM_MAX_ITER = max(1, int(sub_iter))
    rt.BCD_UNIT_PREDICTOR_WARMUP_ROUNDS = max(0, round(rt.BCD_MAX_ITER * 0.10))
    rt.BCD_THETA_CONSTRAINT_DELAY_ROUNDS = max(0, round(rt.BCD_MAX_ITER * 0.10))
    rt.DUAL_DECAY_ROUND = max(1, round(rt.BCD_MAX_ITER / 8))
    rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.10))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.25))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(0, round(rt.SUBPROBLEM_MAX_ITER * 0.50))
    rt.SUBPROBLEM_PG_COST_START_ROUND = max(0, rt.SUBPROBLEM_MAX_ITER // 4)


def main() -> None:
    args = _parse_args()
    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.target
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.SURROGATE_MODEL_DIR = args.resume_dir
    rt.SURROGATE_CONTINUE_TRAINING = bool(args.resume_dir)
    rt.SURROGATE_SKIP_EXISTING_UNITS = bool(args.resume_dir) and not bool(args.retrain_existing)
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = _parse_unit_ids(args.unit_ids)
    rt.DUAL_EPOCHS = DUAL_EPOCHS
    rt.UNIT_PREDICTOR_EPOCHS = UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_SIZE = 64
    rt.UNIT_PREDICTOR_LR = 1.5e-3
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = 0.0
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = [512, 256, 128]
    rt.UNIT_PREDICTOR_DROPOUT = 0.0
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = True
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.22
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.07
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.02
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.035
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.70
    rt.NN_EPOCHS = NN_EPOCHS
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = N_WORKERS_UNIT
    rt.N_WORKERS_BCD = N_WORKERS_BCD
    rt.SUBPROBLEM_LP_BACKEND = SUBPROBLEM_LP_BACKEND
    rt.BCD_LP_BACKEND = BCD_LP_BACKEND
    rt.SURROGATE_CONSTRAINT_STRATEGY = SURROGATE_CONSTRAINT_STRATEGY
    rt.USE_UNIT_PREDICTOR = True
    rt.BCD_USE_UNIT_PREDICTOR = True
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = True
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_MODEL_FILE = None

    rt.UNIT_PREDICTOR_LOAD_PATH = "result/surrogate_models/unit_predictor_case30lite_20260504_132021/unit_predictor.pth"

    _configure_iterations(args.bcd_iter, args.sub_iter)

    if args.no_unit_predictor:
        rt.USE_UNIT_PREDICTOR = False
        rt.BCD_USE_UNIT_PREDICTOR = False
    if args.vanilla_subproblem:
        rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = 0
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT = 0.0
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = 0
        rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = 0
        rt.SUBPROBLEM_PG_COST_START_ROUND = 0
        rt.SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = False
        rt.SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS = 0
        rt.SUBPROBLEM_SIGN4_DELAY_ROUNDS = 0
        rt.SUBPROBLEM_SIGN4_INITIAL_SCALE = 1.0
        rt.SUBPROBLEM_SIGN4_FINAL_SCALE = 1.0
        rt.BCD_THETA_CONSTRAINT_DELAY_ROUNDS = 0
        rt.BCD_UNIT_PREDICTOR_WARMUP_ROUNDS = 0
        rt.SUBPROBLEM_PG_COST_NN_EPOCHS = 4

    if args.pg_cost_nn_epochs is not None:
        rt.SUBPROBLEM_PG_COST_NN_EPOCHS = max(0, int(args.pg_cost_nn_epochs))

    if args.nn_no_direct:
        rt.SUBPROBLEM_MAIN_DIRECT_EPOCHS = 0
        rt.SUBPROBLEM_C_PG_DIRECT_EPOCHS = 0

    rt.METRICS_NAME_TAG = (args.metrics_tag or "").strip()

    print("=" * 72, flush=True)
    print(
        f"case30lite training | target={rt.MODE} | max_samples={rt.MAX_SAMPLES} | "
        f"bcd_iter={rt.BCD_MAX_ITER} | sub_iter={rt.SUBPROBLEM_MAX_ITER} | "
        f"units={rt.UNIT_IDS or 'all'} | "
        f"no_unit_predictor={bool(args.no_unit_predictor)} | "
        f"vanilla_subproblem={bool(args.vanilla_subproblem)} | "
        f"nn_no_direct={bool(args.nn_no_direct)} | "
        f"pg_cost_nn_epochs={rt.SUBPROBLEM_PG_COST_NN_EPOCHS} | "
        f"metrics_tag={rt.METRICS_NAME_TAG or '(none)'} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'} | "
        f"resume_dir={rt.SURROGATE_MODEL_DIR or '(none)'} | "
        f"skip_existing={rt.SURROGATE_SKIP_EXISTING_UNITS}",
        flush=True,
    )
    print("=" * 72, flush=True)
    rt.main()


if __name__ == "__main__":
    main()
