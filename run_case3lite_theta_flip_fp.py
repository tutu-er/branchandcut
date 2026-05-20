#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pinned entrypoint for the case3lite BCD-theta + suspicious-bit FP workflow.

Pipeline
========
1. ``build-map``  – offline theta-only LP error-bit map (suspicious on/off bits)
2. ``bench``      – compare ``theta_flip_case3lite`` vs ``vanilla`` via bench_fp_4way
3. ``plot``      – FP result statistics from bench JSON (success / Hamming / iters / time)
4. ``plot-iter`` – optional per-sample FP iteration traces
5. ``all``          – run the three steps in order

Defaults match the validated case3lite setup:
  * surrogate: ``subproblem_models_case3lite_20260510_merge`` (G0/G1/G2)
  * BCD model: ``bcd_model_case3lite_20260519_235955.pth`` (theta hot-start)
  * active set: ``active_sets_case3lite_T24_n1000_20260403_180137.json``

Examples
--------
    python run_case3lite_theta_flip_fp.py build-map
    python run_case3lite_theta_flip_fp.py bench --samples 30
    python run_case3lite_theta_flip_fp.py plot
    python run_case3lite_theta_flip_fp.py all --samples 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

CASE_NAME = "case3lite"
ACTIVE_SETS = "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"
MODEL_DIR = "result/surrogate_models/subproblem_models_case3lite_20260510_merge"
BCD_MODEL = "result/bcd_models/bcd_model_case3lite_20260519_235955.pth"
ERROR_BIT_MAP = (
    "result/fp_diagnostics/history_error_bit_map_case3lite_n50_20260519_235955_merge.json"
)
BENCH_OUTPUT = "result/fp_diagnostics/bench_fp_theta_case3lite_n30.json"
PLOT_OUTPUT_DIR = "result/fp_diagnostics/plots_case3lite/bench_fp_theta_case3lite_n30"
PLOT_STATS_OUTPUT = f"{PLOT_OUTPUT_DIR}/summary_30samples_fp_stats.png"

DEFAULT_STRATEGIES = "theta_flip_case3lite,vanilla"
DEFAULT_FLIP_TOP_K = 10
DEFAULT_MAP_SAMPLES = 50
DEFAULT_BENCH_SAMPLES = 30
DEFAULT_MAX_FP_ITER = 25


def _run(cmd: list[str], *, log_path: Path | None = None) -> int:
    print("[run]", " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.call(cmd, cwd=str(ROOT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"[run] log -> {log_path}", flush=True)
    return int(proc.returncode)


def cmd_build_map(args: argparse.Namespace) -> int:
    output = args.output or ERROR_BIT_MAP
    return _run(
        [
            PYTHON,
            "-u",
            str(ROOT / "scripts" / "build_history_error_bit_map.py"),
            "--case",
            CASE_NAME,
            "--active-sets",
            args.active_sets,
            "--model-dir",
            args.model_dir,
            "--bcd-model",
            args.bcd_model,
            "--start",
            str(int(args.start)),
            "--max-samples",
            str(int(args.map_samples)),
            "--top-k",
            str(int(args.top_k)),
            "--output",
            output,
        ],
        log_path=Path(args.log) if args.log else None,
    )


def cmd_bench(args: argparse.Namespace) -> int:
    output = args.output or BENCH_OUTPUT
    log_path = Path(args.log) if args.log else Path(str(output).replace(".json", ".log"))
    return _run(
        [
            PYTHON,
            "-u",
            str(ROOT / "scripts" / "bench_fp_4way.py"),
            "--case",
            CASE_NAME,
            "--active-sets",
            args.active_sets,
            "--model-dir",
            args.model_dir,
            "--bcd-model",
            args.bcd_model,
            "--error-bit-map",
            args.error_bit_map,
            "--flip-top-k",
            str(int(args.flip_top_k)),
            "--strategies",
            args.strategies,
            "--start",
            str(int(args.start)),
            "--samples",
            str(int(args.samples)),
            "--max-fp-iter",
            str(int(args.max_fp_iter)),
            "--output",
            output,
        ],
        log_path=log_path,
    )


def cmd_plot(args: argparse.Namespace) -> int:
    bench_json = args.input or BENCH_OUTPUT
    out_path = args.stats_output or PLOT_STATS_OUTPUT
    return _run(
        [
            PYTHON,
            str(ROOT / "scripts" / "plot_bench_fp_4way.py"),
            "--input",
            bench_json,
            "--output",
            out_path,
            "--style",
            str(args.stats_style),
            "--title",
            str(args.stats_title),
        ],
    )


def cmd_plot_iter(args: argparse.Namespace) -> int:
    bench_json = args.input or BENCH_OUTPUT
    out_dir = args.output_dir or PLOT_OUTPUT_DIR
    return _run(
        [
            PYTHON,
            str(ROOT / "scripts" / "plot_fp_iterations.py"),
            "--input",
            bench_json,
            "--output-dir",
            out_dir,
            "--max-samples",
            str(int(args.max_plot_samples)),
        ],
    )


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_build_map(args)
    if rc != 0:
        return rc
    rc = cmd_bench(args)
    if rc != 0:
        return rc
    return cmd_plot(args)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--active-sets", default=ACTIVE_SETS)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--bcd-model", default=BCD_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--log", default=None, help="Optional log file path.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_map = sub.add_parser("build-map", help="Build historical error-bit map.")
    _add_shared_args(p_map)
    p_map.add_argument("--map-samples", type=int, default=DEFAULT_MAP_SAMPLES)
    p_map.add_argument("--top-k", type=int, default=30)
    p_map.add_argument("--output", default=ERROR_BIT_MAP)
    p_map.set_defaults(func=cmd_build_map)

    p_bench = sub.add_parser("bench", help="Run theta_flip_case3lite vs vanilla benchmark.")
    _add_shared_args(p_bench)
    p_bench.add_argument("--error-bit-map", default=ERROR_BIT_MAP)
    p_bench.add_argument("--flip-top-k", type=int, default=DEFAULT_FLIP_TOP_K)
    p_bench.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    p_bench.add_argument("--samples", type=int, default=DEFAULT_BENCH_SAMPLES)
    p_bench.add_argument("--max-fp-iter", type=int, default=DEFAULT_MAX_FP_ITER)
    p_bench.add_argument("--output", default=BENCH_OUTPUT)
    p_bench.set_defaults(func=cmd_bench)

    p_plot = sub.add_parser("plot", help="Plot FP result statistics from bench JSON.")
    p_plot.add_argument("--input", default=BENCH_OUTPUT)
    p_plot.add_argument("--stats-output", default=PLOT_STATS_OUTPUT)
    p_plot.add_argument("--stats-style", choices=("bars", "box", "both"), default="both")
    p_plot.add_argument("--stats-title", default="30样本FP结果统计")
    p_plot.set_defaults(func=cmd_plot)

    p_plot_iter = sub.add_parser("plot-iter", help="Plot FP iteration traces from bench JSON.")
    p_plot_iter.add_argument("--input", default=BENCH_OUTPUT)
    p_plot_iter.add_argument("--output-dir", default=PLOT_OUTPUT_DIR)
    p_plot_iter.add_argument("--max-plot-samples", type=int, default=8)
    p_plot_iter.set_defaults(func=cmd_plot_iter)

    p_all = sub.add_parser("all", help="build-map -> bench -> plot")
    _add_shared_args(p_all)
    p_all.add_argument("--error-bit-map", default=ERROR_BIT_MAP)
    p_all.add_argument("--flip-top-k", type=int, default=DEFAULT_FLIP_TOP_K)
    p_all.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    p_all.add_argument("--map-samples", type=int, default=DEFAULT_MAP_SAMPLES)
    p_all.add_argument("--top-k", type=int, default=30)
    p_all.add_argument("--samples", type=int, default=DEFAULT_BENCH_SAMPLES)
    p_all.add_argument("--max-fp-iter", type=int, default=DEFAULT_MAX_FP_ITER)
    p_all.add_argument("--map-output", default=ERROR_BIT_MAP)
    p_all.add_argument("--bench-output", default=BENCH_OUTPUT)
    p_all.add_argument("--plot-output-dir", default=PLOT_OUTPUT_DIR)
    p_all.add_argument("--max-plot-samples", type=int, default=8)

    def _cmd_all(args: argparse.Namespace) -> int:
        map_args = argparse.Namespace(
            active_sets=args.active_sets,
            model_dir=args.model_dir,
            bcd_model=args.bcd_model,
            start=args.start,
            log=args.log,
            map_samples=args.map_samples,
            top_k=args.top_k,
            output=args.map_output,
        )
        rc = cmd_build_map(map_args)
        if rc != 0:
            return rc
        bench_args = argparse.Namespace(
            active_sets=args.active_sets,
            model_dir=args.model_dir,
            bcd_model=args.bcd_model,
            start=args.start,
            log=args.log,
            error_bit_map=args.error_bit_map,
            flip_top_k=args.flip_top_k,
            strategies=args.strategies,
            samples=args.samples,
            max_fp_iter=args.max_fp_iter,
            output=args.bench_output,
        )
        rc = cmd_bench(bench_args)
        if rc != 0:
            return rc
        plot_args = argparse.Namespace(
            input=args.bench_output,
            stats_output=f"{args.plot_output_dir}/summary_30samples_fp_stats.png",
            stats_style="both",
            stats_title="30样本FP结果统计",
        )
        return cmd_plot(plot_args)

    p_all.set_defaults(func=_cmd_all)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
