"""Summarize a saved commitment-pattern-library JSON result."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional


def _numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not cleaned:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    cleaned.sort()
    n = len(cleaned)
    mid = n // 2
    if n % 2 == 1:
        median = cleaned[mid]
    else:
        median = 0.5 * (cleaned[mid - 1] + cleaned[mid])

    p95_idx = int(round(0.95 * (n - 1)))
    return {
        "mean": sum(cleaned) / n,
        "median": median,
        "min": cleaned[0],
        "max": cleaned[-1],
        "p95": cleaned[p95_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    args = parser.parse_args()

    path = Path(args.json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    scenarios = data.get("scenario_results") or data.get("scenarios") or []
    summary = data.get("summary") or {}
    status_counts = Counter(s.get("solver_status", "unknown") for s in scenarios)
    feasible = [s for s in scenarios if s.get("success")]

    gap_stats = _numeric_summary(s.get("cost_increase_pct") for s in feasible)
    matched_stats = _numeric_summary(s.get("matched_unit_patterns") for s in feasible)
    solve_time_stats = _numeric_summary(s.get("solve_time_s") for s in feasible)
    top_gap = sorted(
        [
            (
                s.get("sample_id"),
                s.get("cost_increase_pct"),
                s.get("solver_status"),
                s.get("changed_unit_patterns"),
            )
            for s in feasible
            if isinstance(s.get("cost_increase_pct"), (int, float))
            and math.isfinite(float(s["cost_increase_pct"]))
        ],
        key=lambda row: row[1],
        reverse=True,
    )[:10]

    print(f"file: {path}")
    print(f"case: {data.get('metadata', {}).get('case_name')}")
    print(f"samples: {data.get('metadata', {}).get('n_samples')}")
    print(f"status_counts: {dict(status_counts)}")
    print(f"feasible: {len(feasible)}/{len(scenarios)}")
    print(f"avg_optimal_cost: {summary.get('avg_optimal_cost')}")
    print(f"avg_restricted_cost: {summary.get('avg_restricted_cost')}")
    print(f"gap_mean_pct: {gap_stats['mean']}")
    print(f"gap_median_pct: {gap_stats['median']}")
    print(f"gap_p95_pct: {gap_stats['p95']}")
    print(f"gap_max_pct: {gap_stats['max']}")
    print(f"matched_mean: {matched_stats['mean']}")
    print(f"matched_min: {matched_stats['min']}")
    print(f"matched_max: {matched_stats['max']}")
    print(f"solve_time_mean_s: {solve_time_stats['mean']}")
    print(f"solve_time_p95_s: {solve_time_stats['p95']}")
    print(f"solve_time_max_s: {solve_time_stats['max']}")
    print(f"expansion_count: {len(data.get('expansion_log') or [])}")
    print(f"optimality_repair_count: {len(data.get('optimality_repair_log') or [])}")
    print("top_gap_samples:")
    for sample_id, gap, status, changed in top_gap:
        print(
            f"  sample_id={sample_id}, gap_pct={gap}, "
            f"status={status}, changed_unit_patterns={changed}"
        )


if __name__ == "__main__":
    main()
