#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case3lite subproblem BCD training entrypoint.

This mirrors the case118 subproblem entrypoint but writes configuration directly
into ``run_training.py`` because case3lite already uses that module as its main
training file.

Unit predictor:
- by default, ``run_training`` auto-picks the latest
  ``result/surrogate_models/unit_predictor_case3lite_*/unit_predictor.pth``;
- ``CASE3LITE_UNIT_PREDICTOR_LOAD_PATH`` or ``--unit-predictor`` overrides it;
- checkpoint metadata is used to align net_variant / TCN dimensions before load.
"""

from __future__ import annotations

import argparse

import run_training as rt


# File-level defaults.  Edit these directly when you do not want to pass CLI
# arguments.  CLI options below only override values when explicitly supplied.
TRAIN_TARGET = "subproblem_bcd"  # "subproblem_bcd" | "both"
CASE3LITE_UNIT_IDS: list[int] | None = None
CASE3LITE_MAX_SAMPLES: int | None = rt.MAX_SAMPLES
CASE3LITE_SUBPROBLEM_MAX_ITER: int | None = None
CASE3LITE_PREDICTOR_WARMUP_ROUNDS: int | None = None
CASE3LITE_SUBPROBLEM_LP_BACKEND: str | None = None  # "gurobi" | "cvxpy_highs" | None
CASE3LITE_N_WORKERS_SAMPLE: int | None = None
CASE3LITE_N_WORKERS_UNIT: int | None = None

# UnitPredictor defaults.  Leave path as None to auto-pick latest
# result/surrogate_models/unit_predictor_case3lite_*/unit_predictor.pth.
CASE3LITE_USE_UNIT_PREDICTOR = True
CASE3LITE_UNIT_PREDICTOR_LOAD_PATH: str | None = None
CASE3LITE_UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = True
CASE3LITE_UNIT_PREDICTOR_LOAD_METADATA_CONFIG = True


def _parse_units(text: str | None) -> list[int] | None:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None
    return [int(x) for x in parts]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        choices=("subproblem_bcd", "both"),
        default=None,
        help="Training target. Default uses TRAIN_TARGET in this file.",
    )
    p.add_argument(
        "--units",
        type=str,
        default=None,
        metavar="IDS",
        help="Only train listed unit IDs, comma separated. Default: all units.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override MAX_SAMPLES. Default keeps run_training.py value.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="Override SUBPROBLEM_MAX_ITER and MAX_ITER.",
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=None,
        metavar="W",
        help="Override SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS.",
    )
    p.add_argument(
        "--unit-predictor",
        type=str,
        default=None,
        metavar="PATH",
        help="unit_predictor.pth, LATEST.txt, or directory containing one.",
    )
    p.add_argument(
        "--no-unit-predictor",
        action="store_true",
        help="Disable 0/1 UnitPredictor initialization.",
    )
    p.add_argument(
        "--no-auto-latest",
        action="store_true",
        help="Disable auto-pick of latest standalone case3lite UnitPredictor.",
    )
    p.add_argument(
        "--backend",
        choices=("gurobi", "cvxpy_highs"),
        default=None,
        help="Override SUBPROBLEM_LP_BACKEND.",
    )
    p.add_argument(
        "--sample-workers",
        type=int,
        default=None,
        metavar="N",
        help="Override N_WORKERS_SAMPLE / N_WORKERS_SUBPROBLEM.",
    )
    p.add_argument(
        "--unit-workers",
        type=int,
        default=None,
        metavar="N",
        help="Override N_WORKERS_UNIT.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    target = args.target or TRAIN_TARGET
    if target not in ("subproblem_bcd", "both"):
        raise ValueError(f"Unsupported TRAIN_TARGET={target!r}")

    rt.CASE_NAME = "case3lite"
    rt.MODE = "both" if target == "both" else "surrogate"
    rt.RUN_FP = False
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False
    rt.SURROGATE_CONSTRAINT_STRATEGY = "all_templates_sign4_plus_single"
    rt.USE_UNIT_PREDICTOR = bool(CASE3LITE_USE_UNIT_PREDICTOR)
    rt.UNIT_PREDICTOR_LOAD_PATH = CASE3LITE_UNIT_PREDICTOR_LOAD_PATH
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = bool(
        CASE3LITE_UNIT_PREDICTOR_AUTO_LATEST_STANDALONE
    )
    rt.UNIT_PREDICTOR_LOAD_METADATA_CONFIG = bool(
        CASE3LITE_UNIT_PREDICTOR_LOAD_METADATA_CONFIG
    )

    if CASE3LITE_UNIT_IDS is not None:
        rt.UNIT_IDS = list(CASE3LITE_UNIT_IDS)
    if CASE3LITE_MAX_SAMPLES is not None:
        rt.MAX_SAMPLES = max(1, int(CASE3LITE_MAX_SAMPLES))
    if CASE3LITE_SUBPROBLEM_MAX_ITER is not None:
        n = max(1, int(CASE3LITE_SUBPROBLEM_MAX_ITER))
        rt.MAX_ITER = n
        rt.SUBPROBLEM_MAX_ITER = n
        rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, int(round(n * 0.10)))
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, int(round(n * 0.25)))
        rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(0, int(round(n * 0.50)))
    if CASE3LITE_PREDICTOR_WARMUP_ROUNDS is not None:
        rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(
            0, int(CASE3LITE_PREDICTOR_WARMUP_ROUNDS)
        )
    if CASE3LITE_SUBPROBLEM_LP_BACKEND is not None:
        rt.SUBPROBLEM_LP_BACKEND = str(CASE3LITE_SUBPROBLEM_LP_BACKEND)
    if CASE3LITE_N_WORKERS_SAMPLE is not None:
        w = max(1, int(CASE3LITE_N_WORKERS_SAMPLE))
        rt.N_WORKERS_SAMPLE = w
        rt.N_WORKERS_SUBPROBLEM = w
    if CASE3LITE_N_WORKERS_UNIT is not None:
        rt.N_WORKERS_UNIT = max(1, int(CASE3LITE_N_WORKERS_UNIT))

    if args.no_unit_predictor:
        rt.USE_UNIT_PREDICTOR = False
    if args.no_auto_latest:
        rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = False

    if args.units is not None:
        rt.UNIT_IDS = _parse_units(args.units)
    if args.max_samples is not None:
        rt.MAX_SAMPLES = max(1, int(args.max_samples))
    if args.max_iter is not None:
        n = max(1, int(args.max_iter))
        rt.MAX_ITER = n
        rt.SUBPROBLEM_MAX_ITER = n
        rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, int(round(n * 0.10)))
        rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND = max(0, int(round(n * 0.25)))
        rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND = max(0, int(round(n * 0.50)))
    if args.warmup_rounds is not None:
        rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS = max(0, int(args.warmup_rounds))
    if args.unit_predictor is not None:
        rt.UNIT_PREDICTOR_LOAD_PATH = args.unit_predictor
    if args.backend is not None:
        rt.SUBPROBLEM_LP_BACKEND = args.backend
    if args.sample_workers is not None:
        w = max(1, int(args.sample_workers))
        rt.N_WORKERS_SAMPLE = w
        rt.N_WORKERS_SUBPROBLEM = w
    if args.unit_workers is not None:
        rt.N_WORKERS_UNIT = max(1, int(args.unit_workers))

    print("=" * 72, flush=True)
    print(f"run_training_case3lite_subproblem_bcd.py -> target={target}", flush=True)
    print(
        f"mode={rt.MODE}, "
        f"backend={rt.SUBPROBLEM_LP_BACKEND}, "
        f"unit_ids={rt.UNIT_IDS!r}, "
        f"max_samples={rt.MAX_SAMPLES}, "
        f"max_iter={rt.SUBPROBLEM_MAX_ITER}, "
        f"predictor_warmup_rounds={rt.SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS}, "
        f"use_unit_predictor={rt.USE_UNIT_PREDICTOR}, "
        f"unit_predictor_load_path={rt.UNIT_PREDICTOR_LOAD_PATH!r}, "
        f"auto_latest={rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE}",
        flush=True,
    )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
