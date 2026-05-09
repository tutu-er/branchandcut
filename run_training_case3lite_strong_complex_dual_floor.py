#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train case3lite with stronger dual floors for complex surrogate constraints.

This preset is a sibling of ``run_training_case3lite.py``.  It keeps the same
case3lite data/model defaults, but strengthens the dual lower bounds used by
the main theta constraints and subproblem sign4 constraints.
"""

from __future__ import annotations

import argparse

import run_training as rt
import run_training_case3lite as base


CASE_NAME = base.CASE_NAME
TRAIN_TARGET = base.TRAIN_TARGET
ACTIVE_SETS_FILE = base.ACTIVE_SETS_FILE
MAX_SAMPLES = base.MAX_SAMPLES
BCD_MAX_ITER = base.BCD_MAX_ITER
SUBPROBLEM_MAX_ITER = base.SUBPROBLEM_MAX_ITER
N_WORKERS_SAMPLE = base.N_WORKERS_SAMPLE
UNIT_PREDICTOR_LOAD_PATH = base.UNIT_PREDICTOR_LOAD_PATH
N_WORKERS_BCD = base.N_WORKERS_BCD
BCD_LP_BACKEND = base.BCD_LP_BACKEND
BCD_GUROBI_THREADS = 1

# Stronger-than-default complex-constraint dual floors.
# - theta_dual_floor is passed to BCD_MU_DUAL_FLOOR_INIT, i.e. the main-model
#   transmission/theta dual family.
# - sign4_dual_floor is passed to SUBPROBLEM_MU_DUAL_FLOOR_INIT, i.e. the
#   subproblem surrogate/sign4 constraint dual family.
THETA_DUAL_FLOOR = 5.00
ZETA_DUAL_FLOOR = 0.00
SIGN4_DUAL_FLOOR = 5.00
THETA_FLOOR_FRACTION = 0.40
SIGN4_INDIVIDUAL_FRACTION = 0.50
SIGN4_GROUP_FRACTION = 0.85
SIGN4_DELAY_ROUNDS = 3
SIGN4_CURRICULUM_ROUNDS = 24
SIGN4_INITIAL_SCALE = 0.10
SIGN4_FINAL_SCALE = 2.00
THETA_STAGE_LIMITS = "6,12,20"
THETA_STAGE_FRACTIONS = "0.25,0.25,0.50"
THETA_CURRICULUM_DELAY_ROUNDS = 3
THETA_CURRICULUM_ROUNDS = 24
THETA_INITIAL_SCALE = 0.10
THETA_FINAL_SCALE = 2.00
ZETA_ITA_CAP_WEIGHT = 2.00
ZETA_ITA_CAP_INITIAL_WEIGHT = 0.00
ZETA_ITA_CAP_INITIAL = THETA_DUAL_FLOOR
ZETA_ITA_CAP_FINAL = 0.00
ZETA_ITA_CAP_START_FRACTION = 0.25
ZETA_ITA_CAP_END_FRACTION = 0.55
SINGLE_MU_CAP_WEIGHT = 2.00
SINGLE_MU_CAP_INITIAL_WEIGHT = 0.00
SINGLE_MU_CAP_INITIAL = SIGN4_DUAL_FLOOR
SINGLE_MU_CAP_FINAL = 0.00
SINGLE_MU_CAP_START_FRACTION = 0.25
SINGLE_MU_CAP_END_FRACTION = 0.55

# Subproblem [direct-NN-main]: full-batch over all samples; fewer epochs per BCD round.
SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY = "full-batch"
SUBPROBLEM_MAIN_DIRECT_EPOCHS = 60

# Subproblem [c_pg]: start from the first BCD round, but keep the inner
# training lighter than the generic defaults for faster case3lite iteration.
SUBPROBLEM_PG_COST_START_ROUND = 0
SUBPROBLEM_PG_COST_NN_EPOCHS = 24
SUBPROBLEM_C_PG_DIRECT_EPOCHS = 120
SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY = "full-batch"
SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE = 100


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("both", "surrogate", "bcd"), default=TRAIN_TARGET)
    p.add_argument("--active-sets", type=str, default=ACTIVE_SETS_FILE)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--bcd-iter", type=int, default=BCD_MAX_ITER)
    p.add_argument("--sub-iter", type=int, default=SUBPROBLEM_MAX_ITER)
    p.add_argument("--sample-workers", type=int, default=N_WORKERS_SAMPLE)
    p.add_argument("--bcd-workers", type=int, default=N_WORKERS_BCD)
    p.add_argument("--bcd-backend", choices=("gurobi", "cvxpy_highs"), default=BCD_LP_BACKEND)
    p.add_argument("--gurobi-threads", type=int, default=BCD_GUROBI_THREADS)
    p.add_argument("--unit-predictor", type=str, default=UNIT_PREDICTOR_LOAD_PATH)
    p.add_argument("--no-unit-predictor", action="store_true")
    p.add_argument("--no-auto-latest-unit-predictor", action="store_true")
    p.add_argument(
        "--surrogate-model-dir",
        type=str,
        default=None,
        help="Directory containing trained dual_predictor/unit_predictor/surrogate_unit_*.pth checkpoints.",
    )
    p.add_argument(
        "--skip-existing-units",
        action="store_true",
        help="When resuming from --surrogate-model-dir, copy existing surrogate_unit_*.pth and train only missing units.",
    )
    p.add_argument("--theta-dual-floor", type=float, default=THETA_DUAL_FLOOR)
    p.add_argument("--zeta-dual-floor", type=float, default=ZETA_DUAL_FLOOR)
    p.add_argument("--sign4-dual-floor", type=float, default=SIGN4_DUAL_FLOOR)
    p.add_argument("--theta-floor-frac", type=float, default=THETA_FLOOR_FRACTION)
    p.add_argument("--sign4-individual-frac", type=float, default=SIGN4_INDIVIDUAL_FRACTION)
    p.add_argument("--sign4-group-frac", type=float, default=SIGN4_GROUP_FRACTION)
    p.add_argument("--sign4-delay-rounds", type=int, default=SIGN4_DELAY_ROUNDS)
    p.add_argument("--sign4-curriculum-rounds", type=int, default=SIGN4_CURRICULUM_ROUNDS)
    p.add_argument("--sign4-initial-scale", type=float, default=SIGN4_INITIAL_SCALE)
    p.add_argument("--sign4-final-scale", type=float, default=SIGN4_FINAL_SCALE)
    p.add_argument("--theta-stage-limits", type=str, default=THETA_STAGE_LIMITS)
    p.add_argument("--theta-stage-fracs", type=str, default=THETA_STAGE_FRACTIONS)
    p.add_argument("--theta-curriculum-delay-rounds", type=int, default=THETA_CURRICULUM_DELAY_ROUNDS)
    p.add_argument("--theta-curriculum-rounds", type=int, default=THETA_CURRICULUM_ROUNDS)
    p.add_argument("--theta-initial-scale", type=float, default=THETA_INITIAL_SCALE)
    p.add_argument("--theta-final-scale", type=float, default=THETA_FINAL_SCALE)
    p.add_argument("--zeta-ita-cap-weight", type=float, default=ZETA_ITA_CAP_WEIGHT)
    p.add_argument("--zeta-ita-cap-initial-weight", type=float, default=ZETA_ITA_CAP_INITIAL_WEIGHT)
    p.add_argument("--zeta-ita-cap-final-weight", type=float, default=None)
    p.add_argument("--zeta-ita-cap-initial", type=float, default=ZETA_ITA_CAP_INITIAL)
    p.add_argument("--zeta-ita-cap-final", type=float, default=ZETA_ITA_CAP_FINAL)
    p.add_argument("--zeta-ita-cap-start-frac", type=float, default=ZETA_ITA_CAP_START_FRACTION)
    p.add_argument("--zeta-ita-cap-end-frac", type=float, default=ZETA_ITA_CAP_END_FRACTION)
    p.add_argument("--single-mu-cap-weight", type=float, default=SINGLE_MU_CAP_WEIGHT)
    p.add_argument("--single-mu-cap-initial-weight", type=float, default=SINGLE_MU_CAP_INITIAL_WEIGHT)
    p.add_argument("--single-mu-cap-final-weight", type=float, default=None)
    p.add_argument("--single-mu-cap-initial", type=float, default=SINGLE_MU_CAP_INITIAL)
    p.add_argument("--single-mu-cap-final", type=float, default=SINGLE_MU_CAP_FINAL)
    p.add_argument("--single-mu-cap-start-frac", type=float, default=SINGLE_MU_CAP_START_FRACTION)
    p.add_argument("--single-mu-cap-end-frac", type=float, default=SINGLE_MU_CAP_END_FRACTION)
    p.add_argument("--c-pg-start-round", type=int, default=SUBPROBLEM_PG_COST_START_ROUND)
    p.add_argument("--c-pg-nn-epochs", type=int, default=SUBPROBLEM_PG_COST_NN_EPOCHS)
    p.add_argument("--c-pg-direct-epochs", type=int, default=SUBPROBLEM_C_PG_DIRECT_EPOCHS)
    p.add_argument(
        "--c-pg-direct-batch-strategy",
        choices=("full-batch", "mini-batch"),
        default=SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY,
    )
    p.add_argument("--c-pg-direct-batch-size", type=int, default=SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE)
    p.add_argument("--unit-ids", type=str, default=None,
                   help="Comma-separated unit ids to train, e.g. '0' or '0,2'.")
    return p.parse_args()


def _clip_fraction(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _parse_unit_ids(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    unit_ids: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        unit_ids.append(int(part))
    return unit_ids or None


def _parse_csv_numbers(text: str | None, cast=float) -> list:
    if text is None or str(text).strip() == "":
        return []
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def _build_theta_training_stages(bcd_iter: int, limits_text: str, fracs_text: str) -> list[dict] | None:
    limits = [max(1, int(v)) for v in _parse_csv_numbers(limits_text, int)]
    if not limits:
        return None
    fracs = [_clip_fraction(v) for v in _parse_csv_numbers(fracs_text, float)]
    if len(fracs) != len(limits) or sum(fracs) <= 0:
        fracs = [1.0 / len(limits)] * len(limits)
    total = sum(fracs)
    fracs = [v / total for v in fracs]
    durations = [max(1, round(bcd_iter * frac)) for frac in fracs]
    durations[-1] = max(1, bcd_iter - sum(durations[:-1]))
    return [
        {"max_constraints_per_time_slot": limit, "iterations": duration}
        for limit, duration in zip(limits, durations)
    ]


def _configure_strong_complex_dual_floors(args: argparse.Namespace) -> None:
    bcd_iter = max(1, int(args.bcd_iter))
    sub_iter = max(1, int(args.sub_iter))
    theta_floor_frac = _clip_fraction(args.theta_floor_frac)
    sign4_individual_frac = _clip_fraction(args.sign4_individual_frac)
    sign4_group_frac = max(sign4_individual_frac, _clip_fraction(args.sign4_group_frac))
    single_cap_start_frac = _clip_fraction(args.single_mu_cap_start_frac)
    single_cap_end_frac = max(single_cap_start_frac, _clip_fraction(args.single_mu_cap_end_frac))
    zeta_cap_start_frac = _clip_fraction(
        getattr(args, "zeta_ita_cap_start_frac", ZETA_ITA_CAP_START_FRACTION)
    )
    zeta_cap_end_frac = max(
        zeta_cap_start_frac,
        _clip_fraction(getattr(args, "zeta_ita_cap_end_frac", ZETA_ITA_CAP_END_FRACTION)),
    )

    rt.BCD_MU_DUAL_FLOOR_INIT = max(0.0, float(args.theta_dual_floor))
    rt.BCD_ITA_DUAL_FLOOR_INIT = max(0.0, float(args.zeta_dual_floor))
    rt.DUAL_DECAY_ROUND = max(1, round(bcd_iter * theta_floor_frac))
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = 4
    rt.BCD_THETA_TRAINING_STAGES = _build_theta_training_stages(
        bcd_iter,
        getattr(args, "theta_stage_limits", THETA_STAGE_LIMITS),
        getattr(args, "theta_stage_fracs", THETA_STAGE_FRACTIONS),
    )
    if rt.BCD_THETA_TRAINING_STAGES:
        rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = max(
            int(stage["max_constraints_per_time_slot"])
            for stage in rt.BCD_THETA_TRAINING_STAGES
        )
    rt.BCD_THETA_CURRICULUM_DELAY_ROUNDS = max(
        0,
        int(getattr(args, "theta_curriculum_delay_rounds", THETA_CURRICULUM_DELAY_ROUNDS)),
    )
    rt.BCD_THETA_CURRICULUM_ROUNDS = max(
        0,
        int(getattr(args, "theta_curriculum_rounds", THETA_CURRICULUM_ROUNDS)),
    )
    rt.BCD_THETA_INITIAL_SCALE = max(
        0.0,
        float(getattr(args, "theta_initial_scale", THETA_INITIAL_SCALE)),
    )
    rt.BCD_THETA_FINAL_SCALE = max(
        0.0,
        float(getattr(args, "theta_final_scale", THETA_FINAL_SCALE)),
    )
    zeta_cap_final_weight = (
        getattr(args, "zeta_ita_cap_weight", ZETA_ITA_CAP_WEIGHT)
        if getattr(args, "zeta_ita_cap_final_weight", None) is None
        else getattr(args, "zeta_ita_cap_final_weight")
    )
    rt.BCD_ZETA_ITA_CAP_PENALTY_WEIGHT = max(0.0, float(zeta_cap_final_weight))
    rt.BCD_ZETA_ITA_CAP_INITIAL_WEIGHT = max(
        0.0,
        float(getattr(args, "zeta_ita_cap_initial_weight", ZETA_ITA_CAP_INITIAL_WEIGHT)),
    )
    rt.BCD_ZETA_ITA_CAP_FINAL_WEIGHT = max(0.0, float(zeta_cap_final_weight))
    rt.BCD_ZETA_ITA_CAP_INITIAL = max(
        0.0,
        float(getattr(args, "zeta_ita_cap_initial", ZETA_ITA_CAP_INITIAL)),
    )
    rt.BCD_ZETA_ITA_CAP_FINAL = max(
        0.0,
        float(getattr(args, "zeta_ita_cap_final", ZETA_ITA_CAP_FINAL)),
    )
    rt.BCD_ZETA_ITA_CAP_START_ROUND = max(0, round(bcd_iter * zeta_cap_start_frac))
    rt.BCD_ZETA_ITA_CAP_END_ROUND = max(
        rt.BCD_ZETA_ITA_CAP_START_ROUND,
        round(bcd_iter * zeta_cap_end_frac),
    )

    rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT = max(0.0, float(args.sign4_dual_floor))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, round(sub_iter * sign4_individual_frac))
    rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND,
        round(sub_iter * sign4_group_frac),
    )
    rt.SUBPROBLEM_SIGN4_DELAY_ROUNDS = max(0, int(args.sign4_delay_rounds))
    rt.SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS = max(0, int(args.sign4_curriculum_rounds))
    rt.SUBPROBLEM_SIGN4_INITIAL_SCALE = max(0.0, float(args.sign4_initial_scale))
    rt.SUBPROBLEM_SIGN4_FINAL_SCALE = max(0.0, float(args.sign4_final_scale))
    rt.SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = True
    rt.SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE = "sign4_only"
    single_cap_final_weight = (
        args.single_mu_cap_weight
        if args.single_mu_cap_final_weight is None
        else args.single_mu_cap_final_weight
    )
    rt.SUBPROBLEM_SINGLE_MU_CAP_PENALTY_WEIGHT = max(0.0, float(single_cap_final_weight))
    rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT = max(0.0, float(args.single_mu_cap_initial_weight))
    rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT = max(0.0, float(single_cap_final_weight))
    rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL = max(0.0, float(args.single_mu_cap_initial))
    rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL = max(0.0, float(args.single_mu_cap_final))
    rt.SUBPROBLEM_SINGLE_MU_CAP_START_ROUND = max(0, round(sub_iter * single_cap_start_frac))
    rt.SUBPROBLEM_SINGLE_MU_CAP_END_ROUND = max(
        rt.SUBPROBLEM_SINGLE_MU_CAP_START_ROUND,
        round(sub_iter * single_cap_end_frac),
    )


def main() -> None:
    args = _parse_args()

    rt.CASE_NAME = CASE_NAME
    rt.MODE = args.target
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = args.active_sets
    rt.MAX_SAMPLES = max(1, int(args.max_samples))
    rt.UNIT_IDS = _parse_unit_ids(args.unit_ids)

    rt.DUAL_EPOCHS = base.DUAL_EPOCHS
    rt.UNIT_PREDICTOR_EPOCHS = base.UNIT_PREDICTOR_EPOCHS
    rt.UNIT_PREDICTOR_BATCH_SIZE = 64
    rt.UNIT_PREDICTOR_LR = 1.5e-3
    rt.UNIT_PREDICTOR_WEIGHT_DECAY = 0.0
    rt.UNIT_PREDICTOR_HIDDEN_DIMS = [512, 256, 128]
    rt.UNIT_PREDICTOR_NET_VARIANT = "tcn_shared_film"
    rt.UNIT_PREDICTOR_TCN_CHANNELS = 128
    rt.UNIT_PREDICTOR_TCN_DEPTH = 8
    rt.UNIT_PREDICTOR_DROPOUT = 0.0
    rt.UNIT_PREDICTOR_ENABLE_POS_WEIGHT = True
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_MSE = 0.30
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION = 0.12
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE = 0.02
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR = 0.12
    rt.UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR = 0.08
    rt.UNIT_PREDICTOR_TV_FLOOR_SCALE = 0.80

    rt.NN_EPOCHS = base.NN_EPOCHS
    rt.N_WORKERS_SAMPLE = max(1, int(args.sample_workers))
    rt.N_WORKERS_SUBPROBLEM = rt.N_WORKERS_SAMPLE
    rt.N_WORKERS_UNIT = base.N_WORKERS_UNIT
    rt.N_WORKERS_BCD = max(1, int(args.bcd_workers))
    rt.SUBPROBLEM_LP_BACKEND = base.SUBPROBLEM_LP_BACKEND
    rt.BCD_LP_BACKEND = args.bcd_backend
    rt.BCD_GUROBI_THREADS = None if int(args.gurobi_threads) <= 0 else int(args.gurobi_threads)
    rt.SURROGATE_CONSTRAINT_STRATEGY = base.SURROGATE_CONSTRAINT_STRATEGY
    rt.USE_UNIT_PREDICTOR = not bool(args.no_unit_predictor)
    rt.BCD_USE_UNIT_PREDICTOR = rt.USE_UNIT_PREDICTOR
    rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = not bool(args.no_auto_latest_unit_predictor)
    rt.UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.BCD_MODEL_FILE = None
    rt.SURROGATE_MODEL_DIR = args.surrogate_model_dir
    rt.SURROGATE_CONTINUE_TRAINING = bool(args.surrogate_model_dir)
    rt.SURROGATE_SKIP_EXISTING_UNITS = bool(args.skip_existing_units)
    rt.BCD_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.BCD_DUAL_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT = 0.0
    rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT = 0.0

    base._configure_iterations(args.bcd_iter, args.sub_iter)
    _configure_strong_complex_dual_floors(args)

    rt.SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY = SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY
    rt.SUBPROBLEM_MAIN_DIRECT_EPOCHS = max(1, int(SUBPROBLEM_MAIN_DIRECT_EPOCHS))
    rt.SUBPROBLEM_PG_COST_START_ROUND = max(0, int(args.c_pg_start_round))
    rt.SUBPROBLEM_PG_COST_NN_EPOCHS = max(1, int(args.c_pg_nn_epochs))
    rt.SUBPROBLEM_C_PG_DIRECT_EPOCHS = max(1, int(args.c_pg_direct_epochs))
    rt.SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY = args.c_pg_direct_batch_strategy
    rt.SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE = max(1, int(args.c_pg_direct_batch_size))

    print("=" * 72, flush=True)
    print(
        f"case3lite strong-complex-dual-floor training | target={rt.MODE} | "
        f"max_samples={rt.MAX_SAMPLES} | bcd_iter={rt.BCD_MAX_ITER} | "
        f"sub_iter={rt.SUBPROBLEM_MAX_ITER} | units={rt.UNIT_IDS or 'all'} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'}",
        flush=True,
    )
    print(
        f"bcd_backend={rt.BCD_LP_BACKEND} | bcd_workers={rt.N_WORKERS_BCD} | "
        f"gurobi_threads={rt.BCD_GUROBI_THREADS if rt.BCD_GUROBI_THREADS is not None else 'auto'}",
        flush=True,
    )
    print(
        f"quadratic_prox=off | bcd_pg_prox={rt.BCD_PG_BLOCK_PROX_WEIGHT} | "
        f"bcd_dual_prox={rt.BCD_DUAL_BLOCK_PROX_WEIGHT} | "
        f"sub_pg_prox={rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT} | "
        f"sub_dual_prox={rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT}",
        flush=True,
    )
    print(
        f"theta_floor={rt.BCD_MU_DUAL_FLOOR_INIT} until round {rt.DUAL_DECAY_ROUND} | "
        f"zeta_floor={rt.BCD_ITA_DUAL_FLOOR_INIT} | "
        f"sign4_floor={rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT} "
        f"individual_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND}, "
        f"group_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND}",
        flush=True,
    )
    print(
        f"theta_stages={rt.BCD_THETA_TRAINING_STAGES} | "
        f"theta_curriculum=delay {rt.BCD_THETA_CURRICULUM_DELAY_ROUNDS}, "
        f"scale {rt.BCD_THETA_INITIAL_SCALE}->{rt.BCD_THETA_FINAL_SCALE} over "
        f"{rt.BCD_THETA_CURRICULUM_ROUNDS} rounds",
        flush=True,
    )
    print(
        f"zeta_ita_cap_weight={rt.BCD_ZETA_ITA_CAP_INITIAL_WEIGHT}"
        f"->{rt.BCD_ZETA_ITA_CAP_FINAL_WEIGHT} | "
        f"cap={rt.BCD_ZETA_ITA_CAP_INITIAL}->{rt.BCD_ZETA_ITA_CAP_FINAL} | "
        f"rounds={rt.BCD_ZETA_ITA_CAP_START_ROUND}"
        f"..{rt.BCD_ZETA_ITA_CAP_END_ROUND}",
        flush=True,
    )
    print(
        f"sign4_curriculum=delay {rt.SUBPROBLEM_SIGN4_DELAY_ROUNDS} rounds, "
        f"scale {rt.SUBPROBLEM_SIGN4_INITIAL_SCALE}"
        f"->{rt.SUBPROBLEM_SIGN4_FINAL_SCALE} over "
        f"{rt.SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS} rounds",
        flush=True,
    )
    print(
        f"single_mu_cap_weight={rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL_WEIGHT}"
        f"->{rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL_WEIGHT} | "
        f"cap={rt.SUBPROBLEM_SINGLE_MU_CAP_INITIAL}->{rt.SUBPROBLEM_SINGLE_MU_CAP_FINAL} | "
        f"rounds={rt.SUBPROBLEM_SINGLE_MU_CAP_START_ROUND}"
        f"..{rt.SUBPROBLEM_SINGLE_MU_CAP_END_ROUND}",
        flush=True,
    )
    print(
        f"unit_predictor={rt.USE_UNIT_PREDICTOR} | "
        f"load_path={rt.UNIT_PREDICTOR_LOAD_PATH or 'auto-latest'} | "
        f"auto_latest={rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE}",
        flush=True,
    )
    print(
        f"surrogate_resume={rt.SURROGATE_CONTINUE_TRAINING} | "
        f"model_dir={rt.SURROGATE_MODEL_DIR or 'none'} | "
        f"skip_existing_units={rt.SURROGATE_SKIP_EXISTING_UNITS}",
        flush=True,
    )
    print(
        f"subproblem direct-NN-main: batch_strategy={rt.SUBPROBLEM_MAIN_DIRECT_BATCH_STRATEGY}, "
        f"epochs={rt.SUBPROBLEM_MAIN_DIRECT_EPOCHS}",
        flush=True,
    )
    print(
        f"subproblem c_pg: start_round={rt.SUBPROBLEM_PG_COST_START_ROUND}, "
        f"nn_epochs={rt.SUBPROBLEM_PG_COST_NN_EPOCHS}, "
        f"direct_epochs={rt.SUBPROBLEM_C_PG_DIRECT_EPOCHS}, "
        f"direct_batch={rt.SUBPROBLEM_C_PG_DIRECT_BATCH_STRATEGY}/"
        f"{rt.SUBPROBLEM_C_PG_DIRECT_BATCH_SIZE}",
        flush=True,
    )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
