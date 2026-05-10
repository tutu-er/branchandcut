#!/usr/bin/env python3
"""Run top-10 case118 strong-dual-floor unit submodels in parallel.

This launcher starts one training process per unit, writes one log per unit,
and caps the number of concurrent jobs. It avoids GNU parallel dependency and
is intended for server runs.

Examples
--------
    python scripts/run_case118_strong_dual_floor_top10_parallel.py
    python scripts/run_case118_strong_dual_floor_top10_parallel.py --max-jobs 4
    python scripts/run_case118_strong_dual_floor_top10_parallel.py --units 2,19 --dry-run
    python scripts/run_case118_strong_dual_floor_top10_parallel.py -- --sub-iter 120
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITS = [2, 19, 10, 17, 1, 30, 0, 11, 18, 12]
DEFAULT_ACTIVE_SETS = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025_active_set_like_refined_"
    "20260418_032025_price_only_clipped.json"
)


def _parse_units(text: str) -> list[int]:
    units = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not units:
        raise argparse.ArgumentTypeError("unit list cannot be empty")
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "poweropt"))
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("MAX_JOBS", "4")))
    parser.add_argument("--max-samples", type=int, default=366)
    parser.add_argument("--units", type=_parse_units, default=DEFAULT_UNITS)
    parser.add_argument("--active-sets", default=DEFAULT_ACTIVE_SETS)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT
        / "result"
        / "logs"
        / f"case118_strong_dual_floor_top10_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed after '--' to run_training_case118_strong_complex_dual_floor.py.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, unit: int) -> list[str]:
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        "-u",
        "run_training_case118_strong_complex_dual_floor.py",
        "--target",
        "subproblem_bcd",
        "--solve-preset",
        "server",
        "--active-sets",
        str(args.active_sets),
        "--max-samples",
        str(args.max_samples),
        "--unit-ids",
        str(unit),
        *extra_args,
    ]


def main() -> int:
    args = parse_args()
    if args.max_jobs < 1:
        raise ValueError("--max-jobs must be >= 1")

    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")

    print(f"repo_root={ROOT}", flush=True)
    print(f"conda_env={args.conda_env}", flush=True)
    print(f"active_sets={args.active_sets}", flush=True)
    print(f"units={args.units}", flush=True)
    print(f"max_jobs={args.max_jobs}", flush=True)
    print(f"max_samples={args.max_samples}", flush=True)
    print(f"log_dir={log_dir}", flush=True)
    if args.extra_args:
        print(f"extra_args={args.extra_args}", flush=True)

    running: list[tuple[int, subprocess.Popen, object]] = []
    failures: list[tuple[int, int]] = []

    def reap(block: bool) -> None:
        nonlocal running
        while True:
            for idx, (unit, proc, log_file) in enumerate(running):
                rc = proc.poll()
                if rc is None:
                    continue
                log_file.close()
                print(f"[unit {unit}] finished rc={rc}", flush=True)
                if rc != 0:
                    failures.append((unit, int(rc)))
                running.pop(idx)
                break
            else:
                if block and running:
                    time.sleep(5.0)
                    continue
                return

    for unit in args.units:
        while len(running) >= args.max_jobs:
            reap(block=True)

        cmd = build_command(args, unit)
        log_path = log_dir / f"unit_{unit}.log"
        print(f"[unit {unit}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            log_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
            continue

        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(" ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        running.append((unit, proc, log_file))

    while running:
        reap(block=True)

    if failures:
        print(f"failures={failures}", flush=True)
        return 1
    print("all jobs finished successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
