"""Run commitment clustering for case118.

Cluster 366 daily scenarios by load/renewable/commitment features, then
solve a multi-scenario shared-commitment UC per cluster to produce
representative commitment schedules with fewer start/stop transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

from src.commitment_clustering import CommitmentClusterer
from src.mti118_data_loader import load_case118_ppc_with_mti_limits

# ======================================================================
# Configuration
# ======================================================================

CASE_NAME = "case118"

# Path to active_set JSON containing 366 daily optimal solutions.
# Set to None to auto-detect the latest file in result/active_set/.
ACTIVE_SETS_FILE: str | None = None

# Number of clusters M
N_CLUSTERS = 10

# Transition penalty weight (lambda): higher -> fewer start/stops
TRANSITION_PENALTY = 1.0

# LP relaxation proximity weight (mu): 0 to disable
LP_PROXIMITY_WEIGHT = 0.0

# Max allowed cost increase vs. optimal (e.g. 0.05 = 5%); None to disable
MAX_COST_INCREASE_RATIO: float | None = None

# Subsample scenarios per cluster to control computation
MAX_SCENARIOS_PER_CLUSTER: int | None = 20

# Gurobi solver time limit per cluster (seconds)
GUROBI_TIME_LIMIT = 600.0

# MIP gap for Gurobi
MIP_GAP = 1e-4

# Feature mode: "summary" (load+ren+online_count per hour) or "full" (flatten x)
FEATURE_MODE = "summary"

# PCA components for dimensionality reduction; None to skip
PCA_COMPONENTS: int | None = None

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

    clusterer = CommitmentClusterer(
        ppc=ppc,
        T_delta=T_DELTA,
        case_name=CASE_NAME,
        n_clusters=N_CLUSTERS,
        transition_penalty=TRANSITION_PENALTY,
        lp_proximity_weight=LP_PROXIMITY_WEIGHT,
        max_cost_increase_ratio=MAX_COST_INCREASE_RATIO,
        max_scenarios_per_cluster=MAX_SCENARIOS_PER_CLUSTER,
        gurobi_time_limit=GUROBI_TIME_LIMIT,
        feature_mode=FEATURE_MODE,
        pca_components=PCA_COMPONENTS,
        verbose=VERBOSE_SOLVER,
    )

    clusterer.run(json_path)
    output = clusterer.save_results(OUTPUT_PATH)
    print(f"\nDone. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
