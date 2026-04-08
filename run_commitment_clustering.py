"""Run per-generator commitment-pattern library optimization for case118."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

from src.commitment_clustering import CommitmentPatternLibrary
from src.mti118_data_loader import load_case118_ppc_with_mti_limits

# ======================================================================
# Configuration
# ======================================================================

CASE_NAME = "case118"

# Path to active_set JSON containing 366 daily optimal solutions.
# Set to None to auto-detect the latest file in result/active_set/.
ACTIVE_SETS_FILE: str | None = None

# Initial shared pattern count per generator.
# If a scenario is infeasible, the script will add the scenario-optimal pattern
# for the missing generators and continue.
INITIAL_PATTERNS_PER_UNIT = 10

# Optional hard cap. Set to None to allow adaptive expansion beyond the initial
# count so that all scenarios can be made feasible.
MAX_PATTERNS_PER_UNIT: int | None = None

# Optional scenario truncation for debugging; None runs all scenarios.
MAX_SAMPLES: int | None = None

# Gurobi solver time limit per cluster (seconds)
GUROBI_TIME_LIMIT = 600.0

# MIP gap for Gurobi
MIP_GAP = 1e-4

# case118 ppc loading
AGGREGATE_THERMAL_BY_BUS = True

# Time resolution
T_DELTA = 1.0

# Output path; None for auto-generated
OUTPUT_PATH: str | None = None

VERBOSE_SOLVER = False


# ======================================================================
# Helpers
# ======================================================================

def _find_latest_active_set_json(case_name: str) -> str:
    """Find the most recent active_set JSON for the given case."""
    result_dir = ROOT / "result" / "active_set"
    if not result_dir.exists():
        raise FileNotFoundError(f"Directory not found: {result_dir}")

    candidates = sorted(
        result_dir.glob(f"active_sets_{case_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No active_set JSON found for {case_name} in {result_dir}"
        )
    return str(candidates[0])


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    json_path = ACTIVE_SETS_FILE
    if json_path is None:
        json_path = _find_latest_active_set_json(CASE_NAME)
    print(f"Active-set JSON: {json_path}", flush=True)

    ppc = load_case118_ppc_with_mti_limits(
        aggregate_thermal_by_bus=AGGREGATE_THERMAL_BY_BUS,
    )
    print(f"Loaded ppc: {ppc['gen'].shape[0]} generators, "
          f"{ppc['bus'].shape[0]} buses, "
          f"{ppc['branch'].shape[0]} branches", flush=True)

    library = CommitmentPatternLibrary(
        ppc=ppc,
        T_delta=T_DELTA,
        case_name=CASE_NAME,
        initial_patterns_per_unit=INITIAL_PATTERNS_PER_UNIT,
        max_patterns_per_unit=MAX_PATTERNS_PER_UNIT,
        gurobi_time_limit=GUROBI_TIME_LIMIT,
        mip_gap=MIP_GAP,
        max_samples=MAX_SAMPLES,
        verbose=VERBOSE_SOLVER,
    )

    library.run(json_path)
    output = library.save_results(OUTPUT_PATH)
    print(f"\nDone. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
