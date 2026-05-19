# -*- coding: utf-8 -*-
"""Summarize model/FP diagnostics JSON for quick ablation checks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def _metric(records: List[dict], key: str, field: str) -> Optional[float]:
    vals = []
    for record in records:
        value = ((record.get("metrics") or {}).get(key) or {}).get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    return mean(vals) if vals else None


def _sum_summary(records: List[dict], key: str) -> int:
    total = 0
    for record in records:
        summary = ((record.get("fp_details") or {}).get("heuristic_summary") or {})
        total += int(summary.get(key) or 0)
    return total


def _counter(records: Iterable[dict], path: List[str]) -> Counter:
    counter: Counter = Counter()
    for record in records:
        cur: Any = record
        for part in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        counter[cur] += 1
    return counter


def summarize(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") or []
    n = len(records)
    plain_h = _metric(records, "plain_lp_to_true", "hamming")
    proxy_h = _metric(records, "proxy_lp_to_true", "hamming")
    fp_h = _metric(records, "fp_to_true", "hamming")
    fp_success = sum(1 for r in records if r.get("fp_success") is True)
    fp_errors = [r.get("fp_error") for r in records if r.get("fp_error")]
    rows = [row for r in records for row in (r.get("fp_iteration_plot_rows") or [])]
    iter_rows = [row for row in rows if row.get("event") == "fp_iteration"]
    return {
        "path": str(path),
        "case": (data.get("metadata") or {}).get("case"),
        "n_records": n,
        "scenario_bank_mode": (data.get("metadata") or {}).get("scenario_bank_mode"),
        "surrogate_constraint_scope": (data.get("metadata") or {}).get("surrogate_constraint_scope"),
        "bcd_proxy_scope": (data.get("metadata") or {}).get("bcd_proxy_scope"),
        "mean_hamming": {
            "plain_lp": plain_h,
            "proxy_lp": proxy_h,
            "fp": fp_h,
            "proxy_improvement_vs_plain": (
                None if plain_h is None or proxy_h is None else plain_h - proxy_h
            ),
        },
        "proxy_lp_stages": dict(_counter(records, ["solve_stats", "proxy_lp", "stage_name"])),
        "fp_success": fp_success,
        "fp_total": n,
        "fp_errors": fp_errors[:5],
        "selected_hot_starts": dict(_counter(records, ["fp_details", "selected_hot_start"])),
        "heuristics": {
            "hot_start_already_feasible": _sum_summary(records, "hot_start_already_feasible"),
            "fp_hot_starts_entered": _sum_summary(records, "fp_hot_starts_entered"),
            "fp_iterations": _sum_summary(records, "fp_iterations"),
            "candidate_convex_hull_iterations": _sum_summary(records, "candidate_convex_hull_iterations"),
            "cycle_hits": _sum_summary(records, "cycle_hits"),
            "perturbations": _sum_summary(records, "perturbations"),
            "pool_restarts": _sum_summary(records, "pool_restarts"),
            "flipped_bits": _sum_summary(records, "flipped_bits"),
        },
        "plot_rows": {
            "total": len(rows),
            "iterations": len(iter_rows),
            "events": dict(Counter(row.get("event") for row in rows)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    args = parser.parse_args()
    path = Path(args.diagnostics)
    if not path.is_absolute():
        path = Path.cwd() / path
    print(json.dumps(summarize(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
