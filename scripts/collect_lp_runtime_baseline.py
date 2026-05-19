#!/usr/bin/env python3
"""Collect plain LP relaxation runtimes for paper-eval samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case_registry import get_case_ppc
from src.dataset_json_utils import load_v3_active_set_json
from src.feasibility_pump import solve_global_LP_relaxation_without_surrogate
from src.scenario_utils import normalize_sample_arrays


ACTIVE_SET_FILES = {
    "case14": ROOT / "result" / "active_set" / "active_sets_case14_T24_n600_20260503_222929.json",
    "case30lite": ROOT / "result" / "active_set" / "active_sets_case30lite_T24_n500_20260503_233729.json",
    "case3lite": ROOT / "result" / "active_set" / "active_sets_case3lite_T24_n200_20260328_102856.json",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "sample_index",
        "runtime_sec",
        "status_name",
        "objective",
        "num_vars",
        "num_constraints",
        "num_nonzeros",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sample_indices_from_eval(eval_dir: Path, cases: set[str] | None) -> dict[str, list[int]]:
    rows = _read_csv(eval_dir / "data" / "sample_metrics.csv")
    by_case: dict[str, set[int]] = {}
    for row in rows:
        case = str(row.get("case", "")).strip()
        if not case or (cases and case not in cases):
            continue
        try:
            sample = int(float(row.get("sample_index", "")))
        except Exception:
            continue
        by_case.setdefault(case, set()).add(sample)
    return {case: sorted(samples) for case, samples in by_case.items()}


def _pd_matrix(sample: dict) -> np.ndarray:
    norm = normalize_sample_arrays(dict(sample))
    if "pd_data" in norm:
        return np.asarray(norm["pd_data"], dtype=float)
    if "load_data" in norm:
        return np.asarray(norm["load_data"], dtype=float)
    raise KeyError("sample has no pd_data/load_data")


def collect(eval_dir: Path, output_csv: Path, cases: set[str] | None) -> list[dict]:
    case_indices = _sample_indices_from_eval(eval_dir, cases)
    if not case_indices:
        raise RuntimeError(f"No sample indices found under {eval_dir / 'data' / 'sample_metrics.csv'}")

    out: list[dict] = []
    for case, indices in case_indices.items():
        active_path = ACTIVE_SET_FILES.get(case)
        if active_path is None or not active_path.is_file():
            raise FileNotFoundError(f"No active-set file configured for {case}: {active_path}")
        ppc = get_case_ppc(case)
        samples = load_v3_active_set_json(active_path, announce=lambda _msg: None)
        print(f"{case}: {len(indices)} LP baseline samples", flush=True)
        runtimes: list[float] = []
        for pos, sample_index in enumerate(indices, start=1):
            if sample_index >= len(samples):
                raise IndexError(f"{case} sample_index={sample_index} out of {len(samples)}")
            x_lp, stats = solve_global_LP_relaxation_without_surrogate(
                ppc,
                _pd_matrix(samples[sample_index]),
                1.0,
                return_stats=True,
            )
            runtime = float(stats.get("runtime_sec", float("nan")))
            if np.isfinite(runtime):
                runtimes.append(runtime)
            out.append(
                {
                    "case": case,
                    "sample_index": sample_index,
                    "runtime_sec": runtime,
                    "status_name": stats.get("status_name", ""),
                    "objective": stats.get("objective", ""),
                    "num_vars": stats.get("num_vars", ""),
                    "num_constraints": stats.get("num_constraints", ""),
                    "num_nonzeros": stats.get("num_nonzeros", ""),
                }
            )
            if pos == 1 or pos == len(indices) or pos % 10 == 0:
                avg = mean(runtimes) if runtimes else float("nan")
                print(f"  {pos}/{len(indices)} sample={sample_index} runtime={runtime:.4g}s mean={avg:.4g}s", flush=True)
    _write_csv(output_csv, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-eval-dir", type=Path, default=ROOT / "result" / "paper_eval" / "20260512_183843")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--cases", default=None, help="Comma-separated case filter.")
    args = parser.parse_args()

    eval_dir = args.paper_eval_dir.resolve()
    output_csv = args.output_csv or (eval_dir / "data" / "lp_runtime_baseline.csv")
    cases = {part.strip() for part in args.cases.split(",") if part.strip()} if args.cases else None
    rows = collect(eval_dir, output_csv.resolve(), cases)
    print(f"wrote: {output_csv.resolve()}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
