# -*- coding: utf-8 -*-
"""Export feasibility-pump iteration rows from diagnostics JSON to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _flatten_rows(diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in diagnostics.get("records") or []:
        sample_id = record.get("sample_id", record.get("sample_index"))
        direct_rows = record.get("fp_iteration_plot_rows") or []
        if direct_rows:
            rows.extend(dict(row) for row in direct_rows)
            continue

        # Backward-compatible fallback for diagnostics that include histories
        # but were generated before fp_iteration_plot_rows existed.
        histories = (record.get("fp_details") or {}).get("fp_histories") or []
        for hist in histories:
            base = {
                "sample_id": sample_id,
                "hot_start_index": hist.get("hot_start_index"),
                "hot_start_name": hist.get("hot_start_name"),
                "parallel": hist.get("parallel"),
                "entered_fp_iterations": hist.get("entered_fp_iterations"),
                "termination": hist.get("termination"),
                "final_reason": hist.get("final_reason"),
            }
            history_rows = hist.get("history") or []
            if not history_rows:
                rows.append({
                    **base,
                    "event": hist.get("termination") or "hot_start_precheck",
                    "iteration": None,
                    "precheck_feasible": hist.get("precheck_feasible"),
                    "precheck_reason": hist.get("precheck_reason"),
                })
                continue
            for step in history_rows:
                rows.append({
                    **base,
                    "event": "fp_iteration",
                    "iteration": step.get("iteration"),
                    "precheck_feasible": hist.get("precheck_feasible"),
                    "precheck_reason": hist.get("precheck_reason"),
                    **step,
                })
    return rows


def _fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = [
        "sample_id",
        "hot_start_index",
        "hot_start_name",
        "event",
        "iteration",
        "entered_fp_iterations",
        "termination",
        "projection_status_name",
        "phi_project",
        "phi_hat",
        "l1_projection",
        "soft_penalty",
        "tau",
        "tau_cost",
        "primal_objective",
        "changed_bits",
        "changed_bits_after_heuristic",
        "trusted_bits",
        "free_bits",
        "candidate_convex_hull_active",
        "candidate_pool_size",
        "surrogate_screen_active",
        "surrogate_screen_constraints",
        "cycle_hit",
        "perturbation_applied",
        "perturbation_mode",
        "pool_restart_applied",
        "flipped_bits",
        "post_feasible",
        "post_reason",
        "precheck_feasible",
        "precheck_reason",
        "final_reason",
    ]
    seen = set()
    extras: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in preferred and key not in seen:
                extras.append(key)
                seen.add(key)
    return preferred + sorted(extras)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    diagnostics_path = Path(args.diagnostics)
    if not diagnostics_path.is_absolute():
        diagnostics_path = Path.cwd() / diagnostics_path
    with diagnostics_path.open("r", encoding="utf-8") as f:
        diagnostics = json.load(f)

    rows = _flatten_rows(diagnostics)
    output_path = Path(args.output) if args.output else diagnostics_path.with_suffix(".fp_trace.csv")
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _fieldnames(rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] rows={len(rows)} wrote {output_path}")


if __name__ == "__main__":
    main()
