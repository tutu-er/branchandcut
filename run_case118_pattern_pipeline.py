"""Strict end-to-end pipeline for case118 pattern-library active-set generation.

Usage
-----
Run all five phases in sequence::

    python run_case118_pattern_pipeline.py

Skip Phase 1 (active-set already exists) and supply the JSON path::

    python run_case118_pattern_pipeline.py --active-set-json result/active_set/active_sets_case118_T0_n366_20260322_063917.json

Skip Phases 1-2 and supply both input paths::

    python run_case118_pattern_pipeline.py \\
        --active-set-json   result/active_set/active_sets_case118_T0_n366_20260322_063917.json \\
        --pattern-lib-json  result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json

Skip Phases 1-3::

    python run_case118_pattern_pipeline.py \\
        --active-set-json      result/active_set/active_sets_case118_T0_n366_20260322_063917.json \\
        --pattern-lib-json     result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json \\
        --active-set-like-json result/commitment_clustering/pattern_library_case118_K10_20260408_132932_active_set_like.json

Skip Phases 1-4::

    python run_case118_pattern_pipeline.py \\
        --active-set-json      result/active_set/active_sets_case118_T0_n366_20260322_063917.json \\
        --pattern-lib-json     result/commitment_clustering/pattern_library_case118_K10_20260408_132932.json \\
        --active-set-like-json result/commitment_clustering/...active_set_like.json \\
        --refinement-json      result/commitment_clustering/sample_refinement_case118_batch.json

Notes
-----
* The pipeline is **strict**: any solver failure, infeasible ED, or missing dual
  raises immediately with a message containing the failing sample_id.
* Phase 1 uses the *serial* ActiveSetLearner so that individual UC failures
  propagate as RuntimeError.  If you need parallelism, fix
  ParallelActiveSetLearner to raise on failed workers first.
* Phase 3 (convert) runs one Gurobi ED per successfully converted sample; the
  first model may print WLS license lines, then there can be long gaps while
  ``getConstrByName`` extracts branch duals (progress prints every 10 samples).
  Set ``BRANCHANDCUT_ED_TIME_LIMIT`` (seconds) to cap each ED if needed.
* --refine-sample-ids accepts a comma-separated list, e.g. ``143,132,257``.
  If omitted, samples whose cost gap exceeds --refine-gap-threshold are chosen
  automatically from the pattern-library summary (same logic as the standalone
  refine script).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

from src.mti118_data_loader import build_case118_daily_samples, load_case118_ppc_with_mti_limits
from src.case118_pattern_pipeline import (
    run_phase_active_set,
    run_phase_pattern_library,
    run_phase_convert,
    run_phase_refine,
    run_phase_build_refined,
    validate_final_commitments_ed_feasible,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict case118 pattern-library pipeline (all 5 phases).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- skip-phase shortcuts ----
    p.add_argument(
        "--active-set-json", default=None,
        help="Skip Phase 1: path to an existing active_set JSON for case118.",
    )
    p.add_argument(
        "--pattern-lib-json", default=None,
        help="Skip Phase 2: path to an existing pattern_library JSON. "
             "Requires --active-set-json.",
    )
    p.add_argument(
        "--active-set-like-json", default=None,
        help="Skip Phase 3: path to an existing active_set_like JSON. "
             "Requires --active-set-json and --pattern-lib-json.",
    )
    p.add_argument(
        "--refinement-json", default=None,
        help="Skip Phase 4: path to an existing refinement batch JSON. "
             "Requires --active-set-json, --pattern-lib-json, and --active-set-like-json.",
    )

    # ---- output ----
    p.add_argument(
        "--output-dir", default=None,
        help="Base output directory. Defaults to result/commitment_clustering "
             "for phases 2-5 and result/active_set for phase 1.",
    )

    # ---- Phase 1 options ----
    p.add_argument("--max-days", type=int, default=None,
                   help="Limit number of daily scenarios loaded (Phase 1).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap the number of scenarios solved in Phase 1.")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="ActiveSetLearner alpha (Phase 1).")
    p.add_argument("--delta", type=float, default=0.15,
                   help="ActiveSetLearner delta (Phase 1).")
    p.add_argument("--epsilon", type=float, default=0.15,
                   help="ActiveSetLearner epsilon (Phase 1).")

    # ---- Phase 2 options ----
    p.add_argument("--initial-patterns-k", type=int, default=10,
                   help="Initial patterns per generator (Phase 2).")
    p.add_argument("--max-patterns-per-unit", type=int, default=None,
                   help="Hard cap on pattern count per generator (Phase 2).")
    p.add_argument("--pattern-time-limit", type=float, default=6000.0,
                   help="Gurobi time limit per scenario in Phase 2 (seconds).")

    # ---- Shared solver options ----
    p.add_argument("--mip-gap", type=float, default=1e-4,
                   help="Gurobi MIP optimality gap.")
    p.add_argument("--t-delta", type=float, default=1.0,
                   help="Time step duration in hours.")
    p.add_argument("--aggregate-thermal-by-bus", action="store_true", default=True,
                   help="Aggregate thermal generators by bus (case118 default).")
    p.add_argument("--verbose-solver", action="store_true", default=False,
                   help="Show Gurobi output in all phases.")

    # ---- Phase 4 options ----
    p.add_argument(
        "--refine-sample-ids", default=None,
        help="Comma-separated sample_ids to refine (Phase 4). "
             "If omitted, samples with cost gap >= --refine-gap-threshold are selected.",
    )
    p.add_argument("--refine-gap-threshold", type=float, default=1.5,
                   help="Cost-gap threshold (%%) for auto-selecting refine targets.")
    p.add_argument("--refine-top-k", type=int, default=20,
                   help="Consider top-k gap samples when selecting refine targets.")
    p.add_argument("--refine-base-time-limit", type=float, default=6000.0,
                   help="Gurobi time limit for baseline solve in Phase 4 (seconds).")
    p.add_argument("--refine-greedy-time-limit", type=float, default=180.0,
                   help="Gurobi time limit per greedy trial in Phase 4 (seconds).")
    p.add_argument("--refine-final-time-limit", type=float, default=6000.0,
                   help="Gurobi time limit for full_repair solve in Phase 4 (seconds).")
    p.add_argument("--refine-max-greedy-steps", type=int, default=8,
                   help="Max greedy augmentation steps per sample in Phase 4.")

    return p.parse_args()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    ts = _ts()
    ppc = load_case118_ppc_with_mti_limits(
        aggregate_thermal_by_bus=args.aggregate_thermal_by_bus,
    )
    print(
        f"[Pipeline] case118 ppc loaded: {ppc['gen'].shape[0]} generators, "
        f"{ppc['bus'].shape[0]} buses, {ppc['branch'].shape[0]} branches",
        flush=True,
    )

    base_out = Path(args.output_dir) if args.output_dir else ROOT / "result"
    active_set_dir = base_out / "active_set"
    clustering_dir = base_out / "commitment_clustering"
    active_set_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(parents=True, exist_ok=True)

    refine_sample_ids = None
    if args.refine_sample_ids:
        refine_sample_ids = [int(x.strip()) for x in args.refine_sample_ids.split(",")]

    # ------------------------------------------------------------------
    # Phase 1 – active set
    # ------------------------------------------------------------------
    active_set_path = args.active_set_json
    if active_set_path is None:
        print("\n" + "=" * 72, flush=True)
        print("  [Phase 1] Solving case118 scenarios -> active set JSON", flush=True)
        print("=" * 72, flush=True)
        scenarios = build_case118_daily_samples(max_days=args.max_days)
        print(f"  Loaded {len(scenarios)} daily scenarios.", flush=True)
        active_set_path = run_phase_active_set(
            ppc=ppc,
            scenarios=scenarios,
            alpha=args.alpha,
            delta=args.delta,
            epsilon=args.epsilon,
            t_delta=args.t_delta,
            max_samples=args.max_samples,
            output_path=str(active_set_dir / f"active_sets_case118_pipeline_{ts}.json"),
            verbose_solver=args.verbose_solver,
        )
    else:
        print(f"\n[Phase 1] Skipped – using: {active_set_path}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2 – pattern library
    # ------------------------------------------------------------------
    pattern_lib_path = args.pattern_lib_json
    if pattern_lib_path is None:
        print("\n" + "=" * 72, flush=True)
        print("  [Phase 2] Building pattern library", flush=True)
        print("=" * 72, flush=True)
        pattern_lib_path = run_phase_pattern_library(
            ppc=ppc,
            active_set_path=active_set_path,
            t_delta=args.t_delta,
            initial_patterns_per_unit=args.initial_patterns_k,
            max_patterns_per_unit=args.max_patterns_per_unit,
            gurobi_time_limit=args.pattern_time_limit,
            mip_gap=args.mip_gap,
            verbose_solver=args.verbose_solver,
            output_path=str(
                clustering_dir / f"pattern_library_case118_K{args.initial_patterns_k}_{ts}.json"
            ),
        )
    else:
        print(f"\n[Phase 2] Skipped – using: {pattern_lib_path}", flush=True)

    # ------------------------------------------------------------------
    # Phase 3 – convert to active_set_like
    # ------------------------------------------------------------------
    active_set_like_path = args.active_set_like_json
    if active_set_like_path is None:
        print("\n" + "=" * 72, flush=True)
        print("  [Phase 3] Converting pattern library -> active_set_like", flush=True)
        print("=" * 72, flush=True)
        lib_stem = Path(pattern_lib_path).stem
        active_set_like_path = str(
            clustering_dir / f"{lib_stem}_active_set_like_{ts}.json"
        )
        active_set_like_path = run_phase_convert(
            active_set_path=active_set_path,
            pattern_library_path=pattern_lib_path,
            output_path=active_set_like_path,
        )
    else:
        print(f"\n[Phase 3] Skipped – using: {active_set_like_path}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4 – refine
    # ------------------------------------------------------------------
    refinement_path = args.refinement_json
    if refinement_path is None:
        print("\n" + "=" * 72, flush=True)
        print("  [Phase 4] Running batch refinement", flush=True)
        print("=" * 72, flush=True)
        refinement_path = str(
            clustering_dir / f"sample_refinement_case118_pipeline_{ts}.json"
        )
        refinement_path = run_phase_refine(
            ppc=ppc,
            active_set_path=active_set_path,
            pattern_library_path=pattern_lib_path,
            active_set_like_path=active_set_like_path,
            output_path=refinement_path,
            sample_ids=refine_sample_ids,
            high_gap_top_k=args.refine_top_k,
            high_gap_threshold_pct=args.refine_gap_threshold,
            t_delta=args.t_delta,
            base_time_limit=args.refine_base_time_limit,
            greedy_trial_time_limit=args.refine_greedy_time_limit,
            final_time_limit=args.refine_final_time_limit,
            mip_gap=args.mip_gap,
            max_greedy_steps=args.refine_max_greedy_steps,
            verbose_solver=args.verbose_solver,
        )
    else:
        print(f"\n[Phase 4] Skipped – using: {refinement_path}", flush=True)

    # ------------------------------------------------------------------
    # Phase 5 – build refined active_set_like
    # ------------------------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("  [Phase 5] Building refined active_set_like JSON", flush=True)
    print("=" * 72, flush=True)
    lib_stem = Path(pattern_lib_path).stem
    refined_path = str(
        clustering_dir / f"{lib_stem}_active_set_like_refined_{ts}.json"
    )
    refined_path = run_phase_build_refined(
        ppc=ppc,
        active_set_like_path=active_set_like_path,
        pattern_library_path=pattern_lib_path,
        refinement_path=refinement_path,
        output_path=refined_path,
        t_delta=args.t_delta,
    )

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("  [Validation] Verifying all final commitments are ED-feasible", flush=True)
    print("=" * 72, flush=True)
    import json
    data = json.loads(Path(refined_path).read_text(encoding="utf-8"))
    validate_final_commitments_ed_feasible(ppc, data["all_samples"], args.t_delta)
    print("  All samples passed ED feasibility check.", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("  Pipeline complete.", flush=True)
    print(f"  Output: {refined_path}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
