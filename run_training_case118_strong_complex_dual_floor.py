#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run case118 presets with stronger dual floors for complex constraints."""

from __future__ import annotations

import argparse

import run_training as rt
import run_training_case118 as base
import run_training_case3lite_strong_complex_dual_floor as strong


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        choices=("main_bcd", "subproblem_bcd", "dual_predictor"),
        default=base.TRAIN_TARGET,
    )
    p.add_argument("--active-sets", type=str, default=base.CASE118_ACTIVE_SET_JSON)
    p.add_argument(
        "--solve-preset",
        choices=("desktop", "server"),
        default=base.SUBPROBLEM_SOLVE_PRESET,
        help="Subproblem solve preset used by the original case118 entrypoint.",
    )
    p.add_argument(
        "--main-bcd-preset",
        choices=("gurobi", "cvxpy_highs"),
        default=base.MAIN_BCD_SOLVE_PRESET,
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sub-iter", type=int, default=None)
    p.add_argument("--unit-ids", type=str, default=None)
    p.add_argument("--theta-dual-floor", type=float, default=strong.THETA_DUAL_FLOOR)
    p.add_argument("--zeta-dual-floor", type=float, default=strong.ZETA_DUAL_FLOOR)
    p.add_argument("--sign4-dual-floor", type=float, default=strong.SIGN4_DUAL_FLOOR)
    p.add_argument("--theta-floor-frac", type=float, default=strong.THETA_FLOOR_FRACTION)
    p.add_argument("--sign4-individual-frac", type=float, default=strong.SIGN4_INDIVIDUAL_FRACTION)
    p.add_argument("--sign4-group-frac", type=float, default=strong.SIGN4_GROUP_FRACTION)
    p.add_argument("--sign4-delay-rounds", type=int, default=strong.SIGN4_DELAY_ROUNDS)
    p.add_argument("--sign4-curriculum-rounds", type=int, default=strong.SIGN4_CURRICULUM_ROUNDS)
    p.add_argument("--sign4-initial-scale", type=float, default=strong.SIGN4_INITIAL_SCALE)
    p.add_argument("--sign4-final-scale", type=float, default=strong.SIGN4_FINAL_SCALE)
    p.add_argument("--single-mu-cap-weight", type=float, default=strong.SINGLE_MU_CAP_WEIGHT)
    p.add_argument("--single-mu-cap-initial-weight", type=float,
                   default=strong.SINGLE_MU_CAP_INITIAL_WEIGHT)
    p.add_argument("--single-mu-cap-final-weight", type=float, default=None)
    p.add_argument("--single-mu-cap-initial", type=float, default=strong.SINGLE_MU_CAP_INITIAL)
    p.add_argument("--single-mu-cap-final", type=float, default=strong.SINGLE_MU_CAP_FINAL)
    p.add_argument("--single-mu-cap-start-frac", type=float,
                   default=strong.SINGLE_MU_CAP_START_FRACTION)
    p.add_argument("--single-mu-cap-end-frac", type=float,
                   default=strong.SINGLE_MU_CAP_END_FRACTION)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    base.TRAIN_TARGET = args.target
    base.CASE118_ACTIVE_SET_JSON = args.active_sets
    base.SUBPROBLEM_SOLVE_PRESET = args.solve_preset
    base.MAIN_BCD_SOLVE_PRESET = args.main_bcd_preset
    if args.max_samples is not None:
        base.SUBPROBLEM_LIGHT_MAX_SAMPLES = max(1, int(args.max_samples))
    if args.sub_iter is not None:
        base.SUBPROBLEM_LIGHT_MAX_ITER = max(1, int(args.sub_iter))
    parsed_unit_ids = strong._parse_unit_ids(args.unit_ids)
    if parsed_unit_ids is not None:
        base.CASE118_SUBPROBLEM_UNIT_IDS = parsed_unit_ids

    base._validate_inputs()
    base._configure_common()

    light_overrides = False
    if args.target == "main_bcd":
        base._configure_main_bcd()
    elif args.target == "subproblem_bcd":
        base._configure_subproblem_bcd()
        light_overrides = base._apply_subproblem_light_runtime_overrides()
    elif args.target == "dual_predictor":
        base._configure_dual_predictor()
        light_overrides = base._apply_subproblem_light_runtime_overrides()
    else:
        raise ValueError(f"Unsupported target={args.target!r}")

    args.bcd_iter = int(getattr(rt, "BCD_MAX_ITER", getattr(rt, "MAX_ITER", 1)) or 1)
    args.sub_iter = int(getattr(rt, "SUBPROBLEM_MAX_ITER", getattr(rt, "MAX_ITER", 1)) or 1)
    strong._configure_strong_complex_dual_floors(args)

    print("=" * 72, flush=True)
    print(f"run_training_case118_strong_complex_dual_floor.py -> target={args.target}", flush=True)
    print(f"active_set_json={base.CASE118_ACTIVE_SET_JSON}", flush=True)
    print(
        f"theta_floor={rt.BCD_MU_DUAL_FLOOR_INIT} until round {rt.DUAL_DECAY_ROUND} | "
        f"zeta_floor={rt.BCD_ITA_DUAL_FLOOR_INIT} | "
        f"sign4_floor={rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT} "
        f"individual_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND}, "
        f"group_until={rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND}",
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
    if args.target == "subproblem_bcd":
        print(
            f"subproblem_preset={base.SUBPROBLEM_SOLVE_PRESET} | "
            f"backend={rt.SUBPROBLEM_LP_BACKEND} | "
            f"max_samples={rt.MAX_SAMPLES} | max_iter={rt.SUBPROBLEM_MAX_ITER} | "
            f"unit_ids={rt.UNIT_IDS!r} | light_overrides={light_overrides}",
            flush=True,
        )
    elif args.target == "main_bcd":
        print(
            f"main_bcd_preset={base.MAIN_BCD_SOLVE_PRESET} | "
            f"backend={rt.BCD_LP_BACKEND} | bcd_max_iter={rt.BCD_MAX_ITER} | "
            f"n_workers_bcd={rt.N_WORKERS_BCD}",
            flush=True,
        )
    else:
        print(
            f"dual_predictor_only={rt.SURROGATE_DUAL_PREDICTOR_ONLY} | "
            f"epochs={rt.DUAL_EPOCHS} | max_samples={rt.MAX_SAMPLES} | "
            f"light_overrides={light_overrides}",
            flush=True,
        )
    print("=" * 72, flush=True)
    rt.main()


if __name__ == "__main__":
    main()
