#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate more diverse small-case active sets and retrain dependent models.

The pipeline intentionally samples total load in deterministic strata before
adding bus-level and temporal noise.  This avoids the narrow commitment
distribution produced by small i.i.d. perturbations around the base profile.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ActiveSetLearner import ActiveSetLearner
from src.active_set_learner_parallel import ParallelActiveSetLearner
from src.case_registry import (
    build_case14_base_load,
    build_case30lite_base_load,
    build_case3lite_base_load,
)
from src.scenario_utils import normalize_sample_arrays


@dataclass(frozen=True)
class CasePlan:
    case_name: str
    build_load: Callable[[int], tuple[dict, np.ndarray]]
    samples: int
    active_set_workers: int
    unit_predictor_script: str
    training_script: str
    unit_predictor_epochs: int
    surrogate_max_samples: int
    bcd_iter: int
    sub_iter: int
    system_scale_low: float
    system_scale_high: float
    bus_scale_low: float = 0.90
    bus_scale_high: float = 1.10
    temporal_wave: float = 0.04
    wave_cycles_low: float = 0.50
    wave_cycles_high: float = 2.25


CASE_PLANS: dict[str, CasePlan] = {
    "case3lite": CasePlan(
        case_name="case3lite",
        build_load=build_case3lite_base_load,
        samples=900,
        active_set_workers=1,
        unit_predictor_script="run_unit_predictor_case3lite_rebalanced.py",
        training_script="run_training_case3lite.py",
        unit_predictor_epochs=700,
        surrogate_max_samples=180,
        bcd_iter=120,
        sub_iter=60,
        system_scale_low=0.60,
        system_scale_high=1.24,
    ),
    "case14": CasePlan(
        case_name="case14",
        build_load=build_case14_base_load,
        samples=720,
        active_set_workers=1,
        unit_predictor_script="run_unit_predictor_case14.py",
        training_script="run_training_case14.py",
        unit_predictor_epochs=720,
        surrogate_max_samples=180,
        bcd_iter=120,
        sub_iter=60,
        system_scale_low=0.55,
        system_scale_high=1.25,
    ),
    "case30lite": CasePlan(
        case_name="case30lite",
        build_load=build_case30lite_base_load,
        samples=720,
        active_set_workers=1,
        unit_predictor_script="run_unit_predictor_case30lite.py",
        training_script="run_training_case30lite.py",
        unit_predictor_epochs=680,
        surrogate_max_samples=180,
        bcd_iter=110,
        sub_iter=60,
        system_scale_low=0.60,
        system_scale_high=1.22,
    ),
}


def _parse_case_list(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in CASE_PLANS]
    if unknown:
        raise ValueError(f"unsupported case(s): {', '.join(unknown)}")
    return names


def _build_stratified_scenarios(
    base_load: np.ndarray,
    *,
    n_samples: int,
    system_scale_low: float,
    system_scale_high: float,
    bus_scale_low: float,
    bus_scale_high: float,
    temporal_wave: float,
    wave_cycles_low: float,
    wave_cycles_high: float,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    n_bins = max(6, min(24, int(round(math.sqrt(max(n_samples, 1))))))
    bin_centers = np.linspace(float(system_scale_low), float(system_scale_high), n_bins)
    scenarios: list[dict] = []

    for i in range(int(n_samples)):
        center = float(bin_centers[i % n_bins])
        bin_width = (float(system_scale_high) - float(system_scale_low)) / max(n_bins - 1, 1)
        jitter = rng.uniform(-0.35, 0.35) * bin_width
        system_scale = float(np.clip(center + jitter, system_scale_low, system_scale_high))
        bus_scale = rng.uniform(float(bus_scale_low), float(bus_scale_high), size=base_load.shape)
        load = np.asarray(base_load, dtype=float) * system_scale * bus_scale

        if float(temporal_wave) > 0.0 and load.shape[1] > 1:
            T = load.shape[1]
            cycles = rng.uniform(float(wave_cycles_low), float(wave_cycles_high))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            t = np.arange(T, dtype=float)
            wave = 1.0 + float(temporal_wave) * np.sin(2.0 * np.pi * cycles * t / T + phase)
            wave = np.maximum(wave, 0.05)
            wave = wave / max(float(np.mean(wave)), 1e-9)
            load = load * wave[None, :]

        scenarios.append(
            {
                "sample_id": i,
                "load_data": load,
                "pd_data": load,
                "renewable_data": np.zeros_like(load),
                "sampling": {
                    "system_scale": system_scale,
                    "stratum": int(i % n_bins),
                },
            }
        )

    return scenarios


def _active_set_to_x(active_set, ng: int, horizon: int) -> np.ndarray:
    x = np.zeros((ng, horizon), dtype=int)
    for item in active_set or []:
        pos, value = item
        g, t = pos
        if 0 <= int(g) < ng and 0 <= int(t) < horizon:
            x[int(g), int(t)] = int(value)
    return x


def _commitment_report(samples: list[dict], ng: int, horizon: int) -> dict:
    matrices = []
    for sample in samples:
        active_set = sample.get("active_set", [])
        matrices.append(_active_set_to_x(active_set, ng, horizon))
    stack = np.stack(matrices, axis=0) if matrices else np.zeros((0, ng, horizon), dtype=int)
    pattern_counts = Counter(tuple(x.reshape(-1).tolist()) for x in stack)
    on_rate = stack.mean(axis=(0, 2)) if len(stack) else np.zeros(ng)
    transition_rate = (
        np.abs(stack[:, :, 1:] - stack[:, :, :-1]).mean(axis=(0, 2))
        if len(stack) and horizon > 1
        else np.zeros(ng)
    )
    top_counts = [int(v) for _, v in pattern_counts.most_common(10)]
    return {
        "n_samples": int(len(samples)),
        "unique_commitments": int(len(pattern_counts)),
        "top_pattern_counts": top_counts,
        "on_rate": [float(x) for x in on_rate],
        "transition_rate": [float(x) for x in transition_rate],
    }


def _print_commitment_report(case_name: str, report: dict) -> None:
    print("-" * 78, flush=True)
    print(
        f"{case_name} commitment diversity: "
        f"unique={report['unique_commitments']} / samples={report['n_samples']} | "
        f"top={report['top_pattern_counts'][:6]}",
        flush=True,
    )
    print("unit | on_rate | transition_rate", flush=True)
    for g, (on, tv) in enumerate(zip(report["on_rate"], report["transition_rate"])):
        print(f"{g:4d} | {on:7.3f} | {tv:15.3f}", flush=True)
    print("-" * 78, flush=True)


def generate_active_set(plan: CasePlan, args: argparse.Namespace) -> Path:
    ppc, base_load = plan.build_load(int(args.horizon))
    n_samples = int(args.samples or plan.samples)
    workers = int(args.active_set_workers or plan.active_set_workers)
    system_scale_low = (
        float(args.system_scale_low)
        if args.system_scale_low is not None
        else float(plan.system_scale_low)
    )
    system_scale_high = (
        float(args.system_scale_high)
        if args.system_scale_high is not None
        else float(plan.system_scale_high)
    )
    scenarios = _build_stratified_scenarios(
        base_load,
        n_samples=n_samples,
        system_scale_low=system_scale_low,
        system_scale_high=system_scale_high,
        bus_scale_low=float(args.bus_scale_low),
        bus_scale_high=float(args.bus_scale_high),
        temporal_wave=float(args.temporal_wave),
        wave_cycles_low=float(args.wave_cycles_low),
        wave_cycles_high=float(args.wave_cycles_high),
        seed=int(args.seed),
    )

    learner_cls = ParallelActiveSetLearner if workers > 1 else ActiveSetLearner
    learner_kwargs = dict(
        alpha=float(args.alpha),
        delta=float(args.delta),
        epsilon=float(args.epsilon),
        ppc=ppc,
        T_delta=1.0,
        Pd=base_load,
        case_name=plan.case_name,
        verbose_solver=bool(args.verbose_solver),
    )
    if workers > 1:
        learner_kwargs.update(n_workers=workers, gurobi_threads=max(1, int(args.gurobi_threads)))
    learner = learner_cls(**learner_kwargs)

    print(
        f"Generating active set | case={plan.case_name} | samples={n_samples} | "
        f"system_scale=[{system_scale_low}, {system_scale_high}] | workers={workers}",
        flush=True,
    )
    learner.run_on_precomputed_scenarios(scenarios, max_samples=n_samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "result" / "active_set" / (
        f"active_sets_{plan.case_name}_rebalanced_T{base_load.shape[1]}_"
        f"n{len(learner.samples)}_{timestamp}.json"
    )
    saved = Path(learner.save_active_sets_json(filename=str(out_path))).resolve()

    normalized = [normalize_sample_arrays(dict(sample)) for sample in learner.samples]
    report = _commitment_report(normalized, ppc["gen"].shape[0], base_load.shape[1])
    _print_commitment_report(plan.case_name, report)
    report_path = saved.with_suffix(".commitment_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"commitment report: {report_path}", flush=True)
    return saved


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_unit_predictor(plan: CasePlan, active_sets: Path, args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "result" / "surrogate_models" / (
        f"unit_predictor_{plan.case_name}_rebalanced_{timestamp}"
    )
    cmd = [
        sys.executable,
        str(ROOT / plan.unit_predictor_script),
        "--active-sets",
        str(active_sets),
        "--max-samples",
        str(int(args.unit_predictor_max_samples or 0)),
        "--epochs",
        str(int(args.unit_predictor_epochs or plan.unit_predictor_epochs)),
        "--out-dir",
        str(out_dir),
    ]
    if args.unit_predictor_val_ratio is not None:
        cmd.extend(["--val-ratio", str(float(args.unit_predictor_val_ratio))])
    _run(cmd, dry_run=bool(args.dry_run))
    ckpt = out_dir / "unit_predictor.pth"
    if not args.dry_run and not ckpt.is_file():
        raise FileNotFoundError(f"UnitPredictor checkpoint missing: {ckpt}")
    return ckpt


def train_surrogate(plan: CasePlan, active_sets: Path, unit_predictor: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / plan.training_script),
        "--target",
        "surrogate",
        "--active-sets",
        str(active_sets),
        "--unit-predictor",
        str(unit_predictor),
        "--max-samples",
        str(int(args.surrogate_max_samples or plan.surrogate_max_samples)),
        "--bcd-iter",
        str(int(args.bcd_iter or plan.bcd_iter)),
        "--sub-iter",
        str(int(args.sub_iter or plan.sub_iter)),
        "--sample-workers",
        str(int(args.sample_workers)),
        "--metrics-tag",
        "rebalanced",
    ]
    _run(cmd, dry_run=bool(args.dry_run))


def planned_active_set_path(plan: CasePlan, horizon: int, n_samples: int) -> Path:
    return ROOT / "result" / "active_set" / (
        f"active_sets_{plan.case_name}_rebalanced_T{int(horizon)}_n{int(n_samples)}_<timestamp>.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="case3lite,case14,case30lite")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--active-set-workers", type=int, default=None)
    parser.add_argument("--gurobi-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--system-scale-low", type=float, default=None)
    parser.add_argument("--system-scale-high", type=float, default=None)
    parser.add_argument("--bus-scale-low", type=float, default=0.90)
    parser.add_argument("--bus-scale-high", type=float, default=1.10)
    parser.add_argument("--temporal-wave", type=float, default=0.04)
    parser.add_argument("--wave-cycles-low", type=float, default=0.50)
    parser.add_argument("--wave-cycles-high", type=float, default=2.25)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--unit-predictor-max-samples", type=int, default=0)
    parser.add_argument("--unit-predictor-epochs", type=int, default=None)
    parser.add_argument("--unit-predictor-val-ratio", type=float, default=0.10)
    parser.add_argument("--surrogate-max-samples", type=int, default=None)
    parser.add_argument("--bcd-iter", type=int, default=None)
    parser.add_argument("--sub-iter", type=int, default=None)
    parser.add_argument("--sample-workers", type=int, default=2)
    parser.add_argument("--skip-unit-predictor", action="store_true")
    parser.add_argument("--skip-surrogate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for case_name in _parse_case_list(args.cases):
        plan = CASE_PLANS[case_name]
        if args.dry_run:
            n_samples = int(args.samples or plan.samples)
            active_sets = planned_active_set_path(plan, args.horizon, n_samples)
            print(
                f"[dry-run] would generate active set | case={plan.case_name} | "
                f"samples={n_samples} | output={active_sets}",
                flush=True,
            )
        else:
            active_sets = generate_active_set(plan, args)
        if args.skip_unit_predictor:
            continue
        unit_predictor = train_unit_predictor(plan, active_sets, args)
        if args.skip_surrogate:
            continue
        train_surrogate(plan, active_sets, unit_predictor, args)


if __name__ == "__main__":
    main()
