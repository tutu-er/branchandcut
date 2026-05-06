"""Commitment clustering and pattern-library-restricted UC utilities.

This module currently supports two distinct workflows:
- cluster scenarios and solve one shared-commitment UC per cluster
- build a per-generator commitment-pattern library shared across scenarios
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import gurobipy as gp
from gurobipy import GRB
from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A, BR_STATUS

from src.uc_NN_subproblem import ActiveSetReader
from src.ed_gurobipy import EconomicDispatchGurobi
from src.uc_time_utils import (
    get_custom_generator_array,
    get_min_up_down_steps_from_ppc,
    get_ramp_limits_from_ppc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_custom_generator_array(ppc_raw: dict, key: str, ng: int) -> Optional[np.ndarray]:
    return get_custom_generator_array(ppc_raw, ng, key)


def _get_ramp_limits(ppc_raw: dict, gen: np.ndarray, T_delta: float):
    return get_ramp_limits_from_ppc(ppc_raw, gen, T_delta)


def _get_min_up_down_steps(ppc_raw: dict, gen: np.ndarray, T_delta: float):
    min_up, min_down, _, _ = get_min_up_down_steps_from_ppc(
        ppc_raw, int(gen.shape[0]), 10**9, T_delta
    )
    return min_up, min_down


def _build_ptdf_data(ppc_int: dict):
    gen = ppc_int["gen"]
    branch = ppc_int["branch"]
    ng = gen.shape[0]
    nb = ppc_int["bus"].shape[0]
    nl = branch.shape[0]

    PTDF = makePTDF(ppc_int["baseMVA"], ppc_int["bus"], branch)

    G_bus = np.zeros((nb, ng))
    for g in range(ng):
        G_bus[int(gen[g, GEN_BUS]), g] = 1.0

    ptdf_g = PTDF @ G_bus
    branch_limit = branch[:, RATE_A]
    active_lines = [
        l for l in range(nl)
        if branch_limit[l] > 1e-6 and branch[l, BR_STATUS] > 0
    ]
    return PTDF, ptdf_g, branch_limit, active_lines


def _count_transitions(x: np.ndarray) -> int:
    """Count total startup + shutdown transitions in a commitment matrix."""
    return int(np.sum(np.abs(np.diff(x.astype(float), axis=1)) > 0.5))


def _pattern_to_key(pattern: np.ndarray) -> Tuple[int, ...]:
    arr = np.asarray(pattern, dtype=int).reshape(-1)
    return tuple(int(v) for v in arr)


def _status_is_proven_optimal(status: Optional[str]) -> bool:
    return status == "optimal"


def _status_has_solution(status: Optional[str]) -> bool:
    return status in {"optimal", "time_limit_with_solution", "suboptimal"}


def _status_rank(status: Optional[str]) -> int:
    if status == "optimal":
        return 3
    if status == "time_limit_with_solution":
        return 2
    if status == "suboptimal":
        return 1
    return 0


def _numeric_summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _load_commitment_samples_from_json(json_path: str) -> List[Dict]:
    """Load active-set samples into a normalized scenario list."""
    reader = ActiveSetReader(json_path)
    raw_samples = reader.load_all_samples()
    raw_json = reader.data.get("all_samples", [])

    samples: List[Dict] = []
    for idx, s in enumerate(raw_samples):
        if "error" in s:
            print(f"  Skipping sample {idx}: {s['error']}", flush=True)
            continue

        x_mat = s.get("unit_commitment_matrix")
        if x_mat is None or (isinstance(x_mat, np.ndarray) and x_mat.size == 0):
            print(f"  Skipping sample {idx}: no commitment matrix", flush=True)
            continue

        load = s.get("load_data")
        if load is None or (isinstance(load, np.ndarray) and load.size == 0):
            load = s.get("pd_data")
        if load is None or (isinstance(load, np.ndarray) and load.size == 0):
            print(f"  Skipping sample {idx}: no load data", flush=True)
            continue

        load = np.asarray(load, dtype=float)
        x_mat = np.asarray(x_mat, dtype=int)
        renewable = s.get("renewable_data")
        if renewable is not None:
            renewable = np.asarray(renewable, dtype=float)

        cost = None
        raw_s = raw_json[idx] if idx < len(raw_json) else {}
        if "total_cost" in raw_s:
            cost = float(raw_s["total_cost"])

        samples.append({
            "sample_id": s.get("sample_id", idx),
            "load_data": load,
            "renewable_data": renewable,
            "unit_commitment_matrix": x_mat,
            "optimal_cost": cost,
        })

    print(f"  Loaded {len(samples)} valid samples from {json_path}", flush=True)
    return samples


# ---------------------------------------------------------------------------
# Multi-scenario shared-commitment UC solver
# ---------------------------------------------------------------------------

def solve_shared_commitment_uc(
    ppc: dict,
    scenarios: List[Dict],
    T_delta: float,
    transition_penalty: float = 1.0,
    x_lp_avg: Optional[np.ndarray] = None,
    lp_proximity_weight: float = 0.0,
    cost_opt_avg: Optional[float] = None,
    max_cost_increase_ratio: Optional[float] = None,
    time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], float, str]:
    """Solve a multi-scenario UC with shared binary commitment x.

    Returns (x_sol, obj_val, status_str).
    """
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    branch = ppc_int["branch"]
    ng = gen.shape[0]

    sample0_load = np.asarray(scenarios[0]["load_data"], dtype=float)
    T = sample0_load.shape[1]
    nb = sample0_load.shape[0]
    N = len(scenarios)

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_steps(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    # Renewable bus indices (union across all scenarios)
    all_renewable = np.zeros((nb, T), dtype=float)
    for s in scenarios:
        ren = s.get("renewable_data")
        if ren is not None:
            all_renewable += np.abs(np.asarray(ren, dtype=float))
    renewable_bus_ids = np.where(np.any(all_renewable > 1e-9, axis=1))[0]
    nr = len(renewable_bus_ids)

    # Build PTDF
    try:
        PTDF_full, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
    except Exception as e:
        print(f"  Warning: PTDF build failed ({e}); DC flow constraints skipped", flush=True)
        active_lines = []
        ptdf_g = None
        branch_limit = None
        PTDF_full = None

    model = gp.Model("SharedCommitmentUC")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    # --- Shared commitment variables ---
    x = model.addVars(ng, T, vtype=GRB.BINARY, name="x")
    coc = model.addVars(ng, T - 1, lb=0, name="coc")
    z_up = model.addVars(ng, T - 1, lb=0, name="z_up")
    z_dn = model.addVars(ng, T - 1, lb=0, name="z_dn")

    # --- Per-scenario dispatch variables ---
    pg = {}
    p_ren = {}
    cpower = {}
    for d in range(N):
        for g in range(ng):
            for t in range(T):
                pg[g, t, d] = model.addVar(lb=0, name=f"pg_{g}_{t}_{d}")
                cpower[g, t, d] = model.addVar(lb=0, name=f"cp_{g}_{t}_{d}")
        if nr > 0:
            for r in range(nr):
                for t in range(T):
                    p_ren[r, t, d] = model.addVar(lb=0, name=f"pr_{r}_{t}_{d}")

    model.update()

    # ======= Shared x constraints =======

    # Min up-time
    for g in range(ng):
        for tau in range(1, int(min_up_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau])

    # Min down-time
    for g in range(ng):
        for tau in range(1, int(min_down_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau])

    # Commitment change cost
    for t in range(1, T):
        for g in range(ng):
            model.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]))

    # Transition indicators
    for t in range(1, T):
        for g in range(ng):
            model.addConstr(z_up[g, t - 1] >= x[g, t] - x[g, t - 1])
            model.addConstr(z_dn[g, t - 1] >= x[g, t - 1] - x[g, t])

    # ======= Per-scenario constraints =======

    for d in range(N):
        load_d = np.asarray(scenarios[d]["load_data"], dtype=float)
        ren_d = scenarios[d].get("renewable_data")
        if ren_d is not None:
            ren_d = np.asarray(ren_d, dtype=float)

        # Power balance
        for t in range(T):
            ren_supply = (
                gp.quicksum(p_ren[r, t, d] for r in range(nr))
                if nr > 0 else 0
            )
            model.addConstr(
                gp.quicksum(pg[g, t, d] for g in range(ng)) + ren_supply
                == float(np.sum(load_d[:, t])),
                name=f"pb_{d}_{t}",
            )

        # Generation limits
        for g in range(ng):
            for t in range(T):
                model.addConstr(pg[g, t, d] >= gen[g, PMIN] * x[g, t])
                model.addConstr(pg[g, t, d] <= gen[g, PMAX] * x[g, t])

        # Renewable upper bounds
        if nr > 0 and ren_d is not None:
            for r, bus_idx in enumerate(renewable_bus_ids):
                for t in range(T):
                    model.addConstr(p_ren[r, t, d] <= float(ren_d[bus_idx, t]))

        # Ramp constraints
        for g in range(ng):
            for t in range(1, T):
                model.addConstr(
                    pg[g, t, d] - pg[g, t - 1, d]
                    <= Ru[g] * x[g, t - 1] + Ru_co[g] * (1 - x[g, t - 1])
                )
                model.addConstr(
                    pg[g, t - 1, d] - pg[g, t, d]
                    <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
                )

        # Generation cost
        for g in range(ng):
            for t in range(T):
                model.addConstr(
                    cpower[g, t, d]
                    >= gencost[g, -2] / T_delta * pg[g, t, d]
                    + gencost[g, -1] / T_delta * x[g, t]
                )

        # DC power flow
        if active_lines and ptdf_g is not None:
            ptdf_Pd_d = PTDF_full @ load_d
            if nr > 0 and ren_d is not None:
                R_bus = np.zeros((nb, nr))
                for r, bus_idx in enumerate(renewable_bus_ids):
                    R_bus[bus_idx, r] = 1.0
                ptdf_R = PTDF_full @ R_bus
            else:
                ptdf_R = None

            for l_idx in active_lines:
                limit = float(branch_limit[l_idx])
                for t in range(T):
                    flow_expr = gp.quicksum(
                        float(ptdf_g[l_idx, g]) * pg[g, t, d] for g in range(ng)
                    ) - float(ptdf_Pd_d[l_idx, t])
                    if ptdf_R is not None and nr > 0:
                        flow_expr += gp.quicksum(
                            float(ptdf_R[l_idx, r]) * p_ren[r, t, d] for r in range(nr)
                        )
                    model.addConstr(flow_expr <= limit)
                    model.addConstr(flow_expr >= -limit)

    # ======= Cost-increase bound (optional) =======

    if max_cost_increase_ratio is not None and cost_opt_avg is not None:
        total_cost_expr = (
            gp.quicksum(cpower[g, t, d] for g in range(ng) for t in range(T) for d in range(N))
            / N
            + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1))
        )
        model.addConstr(
            total_cost_expr <= (1 + max_cost_increase_ratio) * cost_opt_avg,
            name="cost_cap",
        )

    # ======= Objective =======

    obj = (
        gp.quicksum(cpower[g, t, d] for g in range(ng) for t in range(T) for d in range(N))
        / N
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1))
        + transition_penalty * gp.quicksum(
            z_up[g, t] + z_dn[g, t] for g in range(ng) for t in range(T - 1)
        )
    )

    # LP proximity: x[g,t]*(1 - 2*x_LP_avg[g,t]) is linear for binary x
    if lp_proximity_weight > 0 and x_lp_avg is not None:
        for g in range(ng):
            for t in range(T):
                coeff = float(1.0 - 2.0 * x_lp_avg[g, t])
                obj += lp_proximity_weight * coeff * x[g, t]

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    status_str = "optimal"
    if model.status == GRB.OPTIMAL:
        pass
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        status_str = "time_limit_with_solution"
    elif model.status == GRB.SUBOPTIMAL:
        status_str = "suboptimal"
    else:
        return None, float("inf"), f"infeasible_or_error(status={model.status})"

    x_sol = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
    x_sol = np.round(x_sol).astype(int)
    return x_sol, model.objVal, status_str


# ---------------------------------------------------------------------------
# LP relaxation for centroid scenario
# ---------------------------------------------------------------------------

def solve_lp_relaxation_for_scenario(
    ppc: dict,
    load_data: np.ndarray,
    T_delta: float,
) -> np.ndarray:
    """Solve a standard UC LP relaxation (x in [0,1]) and return x_LP."""
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng = gen.shape[0]
    T = load_data.shape[1]
    Pd_sum = np.sum(load_data, axis=0)

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_steps(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    model = gp.Model("LP_relaxation")
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name="pg")
    x = model.addVars(ng, T, lb=0, ub=1, name="x")
    cpower = model.addVars(ng, T, lb=0, name="cpower")
    coc = model.addVars(ng, T - 1, lb=0, name="coc")

    for t in range(T):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]))

    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t - 1] <= Ru[g] * x[g, t - 1] + Ru_co[g] * (1 - x[g, t - 1])
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
            )
        for tau in range(1, int(min_up_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau])
        for tau in range(1, int(min_down_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau])
        for t in range(1, T):
            model.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]))
        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                + gencost[g, -1] / T_delta * x[g, t]
            )

    try:
        PTDF_full, ptdf_g, branch_limit, active_lines_list = _build_ptdf_data(ppc_int)
        ptdf_Pd = PTDF_full @ load_data
        for l in active_lines_list:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng)) - float(ptdf_Pd[l, t])
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
    except Exception:
        pass

    obj = (
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1))
    )
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
    return np.full((ng, T), 0.5)


# ---------------------------------------------------------------------------
# ED evaluation
# ---------------------------------------------------------------------------

def evaluate_commitment_cost(
    ppc: dict,
    load_data: np.ndarray,
    x_commitment: np.ndarray,
    T_delta: float,
    renewable_data: Optional[np.ndarray] = None,
) -> Dict:
    """Fix commitment x and solve ED; return cost breakdown."""
    x_arr = np.asarray(np.round(x_commitment), dtype=float)
    try:
        ed = EconomicDispatchGurobi(
            ppc, load_data, T_delta, x_arr,
            renewable_data=renewable_data, verbose=False,
        )
        pg_sol, dispatch_cost = ed.solve()
        if pg_sol is None:
            return {"success": False, "reason": "ED_infeasible"}

        gencost = ext2int(ppc)["gencost"]
        start_cost_vec = gencost[:, 1]
        shut_cost_vec = gencost[:, 2]
        ng, T = x_arr.shape
        commitment_cost = 0.0
        for g in range(ng):
            for t in range(1, T):
                diff = x_arr[g, t] - x_arr[g, t - 1]
                if diff > 0.5:
                    commitment_cost += start_cost_vec[g]
                elif diff < -0.5:
                    commitment_cost += shut_cost_vec[g]

        total_cost = dispatch_cost + commitment_cost
        return {
            "success": True,
            "total_cost": total_cost,
            "dispatch_cost": dispatch_cost,
            "commitment_cost": commitment_cost,
            "transitions": _count_transitions(x_arr),
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# CommitmentClusterer
# ---------------------------------------------------------------------------

class CommitmentClusterer:
    """Cluster daily UC scenarios and produce representative commitment schedules.

    By default ``require_all_cluster_days_in_uc=True``: the shared MILP includes
    every day in the cluster, so the returned ``x_rep`` satisfies power balance,
    limits, ramps, and DC flow jointly for all those days. Set
    ``require_all_cluster_days_in_uc=False`` to subsample scenarios (faster but
    ``x_rep`` is only guaranteed feasible on the sampled days).
    """

    def __init__(
        self,
        ppc: dict,
        T_delta: float = 1.0,
        case_name: str = "case118",
        n_clusters: int = 10,
        transition_penalty: float = 1.0,
        lp_proximity_weight: float = 0.0,
        max_cost_increase_ratio: Optional[float] = None,
        max_scenarios_per_cluster: Optional[int] = None,
        require_all_cluster_days_in_uc: bool = True,
        gurobi_time_limit: float = 600.0,
        feature_mode: str = "summary",
        pca_components: Optional[int] = None,
        verbose: bool = False,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        self.ppc = ppc
        self.T_delta = T_delta
        self.case_name = case_name
        self.n_clusters = n_clusters
        self.transition_penalty = transition_penalty
        self.lp_proximity_weight = lp_proximity_weight
        self.max_cost_increase_ratio = max_cost_increase_ratio
        self.max_scenarios_per_cluster = max_scenarios_per_cluster
        self.require_all_cluster_days_in_uc = require_all_cluster_days_in_uc
        self.gurobi_time_limit = gurobi_time_limit
        self.feature_mode = feature_mode
        self.pca_components = pca_components
        self.verbose = verbose

        self.samples: List[Dict] = []
        self.labels: Optional[np.ndarray] = None
        self.cluster_results: List[Dict] = []

    # ----- Data loading -----

    def load_samples_from_json(self, json_path: str) -> List[Dict]:
        """Load samples from an active_set JSON produced by ActiveSetLearner."""
        self.samples = _load_commitment_samples_from_json(json_path)
        return self.samples

    # ----- Feature extraction -----

    def extract_features(self, samples: Optional[List[Dict]] = None) -> np.ndarray:
        """Build feature matrix (n_samples, n_features) for clustering."""
        if samples is None:
            samples = self.samples

        features_list = []
        for s in samples:
            load = s["load_data"]
            x_mat = s["unit_commitment_matrix"]

            total_load = np.sum(load, axis=0)  # (T,)
            renewable = s.get("renewable_data")
            total_ren = np.sum(renewable, axis=0) if renewable is not None else np.zeros_like(total_load)

            if self.feature_mode == "full":
                # Flatten everything: load + renewable + full x
                feat = np.concatenate([total_load, total_ren, x_mat.ravel().astype(float)])
            else:
                # Summary: load curve + renewable curve + hourly online count
                online_count = np.sum(x_mat, axis=0).astype(float)  # (T,)
                feat = np.concatenate([total_load, total_ren, online_count])

            features_list.append(feat)

        features = np.vstack(features_list)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        if self.pca_components is not None and self.pca_components < features.shape[1]:
            pca = PCA(n_components=self.pca_components)
            features = pca.fit_transform(features)
            explained = np.sum(pca.explained_variance_ratio_)
            print(f"  PCA: {self.pca_components} components, {explained:.1%} variance explained", flush=True)

        return features

    # ----- Clustering -----

    def cluster(self, features: np.ndarray) -> np.ndarray:
        """Run KMeans clustering and return labels."""
        actual_k = min(self.n_clusters, features.shape[0])
        kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        self.labels = labels

        for k in range(actual_k):
            count = int(np.sum(labels == k))
            print(f"  Cluster {k}: {count} days", flush=True)

        return labels

    # ----- Representative commitment per cluster -----

    def _select_cluster_scenarios(self, cluster_indices: List[int]) -> List[Dict]:
        """Return scenarios fed into shared UC.

        When ``require_all_cluster_days_in_uc`` is True (default), every day in
        the cluster is included so joint constraints hold for all of them.
        Subsampling is only applied when that flag is False (compute shortcut;
        then x_rep is not guaranteed feasible on omitted days).
        """
        cluster_samples = [self.samples[i] for i in cluster_indices]
        n = len(cluster_samples)

        if self.require_all_cluster_days_in_uc:
            if (
                self.max_scenarios_per_cluster is not None
                and n > self.max_scenarios_per_cluster
            ):
                print(
                    f"  Note: require_all_cluster_days_in_uc=True — shared UC uses "
                    f"all {n} cluster days (ignoring max_scenarios_per_cluster="
                    f"{self.max_scenarios_per_cluster}).",
                    flush=True,
                )
            return cluster_samples

        if (
            self.max_scenarios_per_cluster is not None
            and n > self.max_scenarios_per_cluster
        ):
            print(
                "  WARNING: require_all_cluster_days_in_uc=False — subsampling UC "
                f"to {self.max_scenarios_per_cluster} of {n} days. Representative "
                "x is only guaranteed feasible on sampled days.",
                flush=True,
            )
            rng = np.random.default_rng(42)
            chosen = rng.choice(n, self.max_scenarios_per_cluster, replace=False)
            cluster_samples = [cluster_samples[int(i)] for i in sorted(chosen)]
        return cluster_samples

    def _compute_lp_avg(self, cluster_samples: List[Dict]) -> np.ndarray:
        """Compute average LP relaxation x across cluster scenarios."""
        lp_solutions = []
        for s in cluster_samples:
            load = s["load_data"]
            x_lp = solve_lp_relaxation_for_scenario(self.ppc, load, self.T_delta)
            lp_solutions.append(x_lp)
        return np.mean(lp_solutions, axis=0)

    def _compute_optimal_cost_via_ed(self, sample: Dict) -> float:
        """If optimal_cost is missing, estimate it via ED on optimal x."""
        if sample.get("optimal_cost") is not None:
            return sample["optimal_cost"]
        result = evaluate_commitment_cost(
            self.ppc,
            sample["load_data"],
            sample["unit_commitment_matrix"],
            self.T_delta,
            renewable_data=sample.get("renewable_data"),
        )
        if result["success"]:
            sample["optimal_cost"] = result["total_cost"]
            return result["total_cost"]
        return float("inf")

    def solve_cluster(self, cluster_id: int, cluster_indices: List[int]) -> Dict:
        """Solve representative commitment for one cluster."""
        print(f"\n{'='*60}", flush=True)
        print(f"  Solving cluster {cluster_id} ({len(cluster_indices)} days)", flush=True)
        print(f"{'='*60}", flush=True)

        cluster_samples = self._select_cluster_scenarios(cluster_indices)
        n_used = len(cluster_samples)
        print(
            f"  Shared UC scenario count: {n_used} "
            f"(require_all_cluster_days_in_uc={self.require_all_cluster_days_in_uc})",
            flush=True,
        )

        # Compute optimal costs
        opt_costs = []
        for s in cluster_samples:
            c = self._compute_optimal_cost_via_ed(s)
            opt_costs.append(c)
        avg_opt_cost = float(np.mean([c for c in opt_costs if np.isfinite(c)])) if opt_costs else None

        # LP average (if proximity weight > 0)
        x_lp_avg = None
        if self.lp_proximity_weight > 0:
            print("  Computing LP relaxation average...", flush=True)
            x_lp_avg = self._compute_lp_avg(cluster_samples)

        # Solve multi-scenario shared commitment UC
        t_start = time.time()
        x_rep, obj_val, status = solve_shared_commitment_uc(
            ppc=self.ppc,
            scenarios=cluster_samples,
            T_delta=self.T_delta,
            transition_penalty=self.transition_penalty,
            x_lp_avg=x_lp_avg,
            lp_proximity_weight=self.lp_proximity_weight,
            cost_opt_avg=avg_opt_cost,
            max_cost_increase_ratio=self.max_cost_increase_ratio,
            time_limit=self.gurobi_time_limit,
            verbose=self.verbose,
        )
        solve_time = time.time() - t_start
        print(f"  Solve time: {solve_time:.1f}s, status: {status}", flush=True)

        if x_rep is None:
            print(f"  Cluster {cluster_id}: solver failed ({status})", flush=True)
            return {
                "cluster_id": cluster_id,
                "n_days": len(cluster_indices),
                "n_scenarios_used": n_used,
                "day_indices": cluster_indices,
                "status": status,
                "success": False,
            }

        transitions = _count_transitions(x_rep)
        print(f"  Representative x transitions: {transitions}", flush=True)

        # Evaluate on ALL days in cluster (not just the subsampled ones)
        all_cluster_samples = [self.samples[i] for i in cluster_indices]
        per_day = []
        rep_costs = []
        feasible_count = 0

        for s in all_cluster_samples:
            result = evaluate_commitment_cost(
                self.ppc,
                s["load_data"],
                x_rep,
                self.T_delta,
                renewable_data=s.get("renewable_data"),
            )
            opt_cost_s = self._compute_optimal_cost_via_ed(s)
            opt_transitions_s = _count_transitions(s["unit_commitment_matrix"])

            day_info = {
                "sample_id": s["sample_id"],
                "feasible": result["success"],
                "optimal_cost": opt_cost_s if np.isfinite(opt_cost_s) else None,
                "optimal_transitions": opt_transitions_s,
            }

            if result["success"]:
                feasible_count += 1
                rep_cost = result["total_cost"]
                rep_costs.append(rep_cost)
                day_info["representative_cost"] = rep_cost
                day_info["representative_transitions"] = transitions
                if np.isfinite(opt_cost_s) and opt_cost_s > 0:
                    day_info["cost_increase_pct"] = (rep_cost - opt_cost_s) / opt_cost_s * 100
                else:
                    day_info["cost_increase_pct"] = None
            else:
                day_info["representative_cost"] = None
                day_info["cost_increase_pct"] = None

            per_day.append(day_info)

        n_total = len(cluster_indices)
        feasibility_rate = feasible_count / n_total if n_total > 0 else 0.0
        avg_rep_cost = float(np.mean(rep_costs)) if rep_costs else None

        # Recompute avg_opt_cost over ALL days
        all_opt_costs = [
            self._compute_optimal_cost_via_ed(self.samples[i]) for i in cluster_indices
        ]
        avg_opt_all = float(np.mean([c for c in all_opt_costs if np.isfinite(c)])) if all_opt_costs else None

        cost_increase_pct = None
        if avg_rep_cost is not None and avg_opt_all is not None and avg_opt_all > 0:
            cost_increase_pct = (avg_rep_cost - avg_opt_all) / avg_opt_all * 100

        # Avg optimal transitions
        avg_opt_transitions = float(np.mean([
            _count_transitions(self.samples[i]["unit_commitment_matrix"])
            for i in cluster_indices
        ]))

        # LP distance
        lp_distance = None
        if x_lp_avg is not None:
            lp_distance = float(np.mean((x_rep.astype(float) - x_lp_avg) ** 2))

        result_dict = {
            "cluster_id": cluster_id,
            "n_days": n_total,
            "n_scenarios_used": n_used,
            "day_indices": cluster_indices,
            "representative_x": x_rep.tolist(),
            "total_transitions": transitions,
            "avg_optimal_transitions": avg_opt_transitions,
            "avg_optimal_cost": avg_opt_all,
            "avg_representative_cost": avg_rep_cost,
            "cost_increase_pct": cost_increase_pct,
            "feasibility_rate": feasibility_rate,
            "lp_distance": lp_distance,
            "solve_time_s": solve_time,
            "solver_status": status,
            "success": True,
            "per_day_results": per_day,
        }

        print(f"  Feasibility: {feasibility_rate:.1%} ({feasible_count}/{n_total})", flush=True)
        print(f"  Avg optimal cost: {avg_opt_all}", flush=True)
        print(f"  Avg representative cost: {avg_rep_cost}", flush=True)
        if cost_increase_pct is not None:
            print(f"  Cost increase: {cost_increase_pct:.2f}%", flush=True)
        print(f"  Transition reduction: {avg_opt_transitions:.1f} -> {transitions}", flush=True)

        return result_dict

    # ----- Full pipeline -----

    def run(self, json_path: str) -> List[Dict]:
        """Run the full pipeline: load -> features -> cluster -> solve -> evaluate."""
        print("=" * 60, flush=True)
        print("  Commitment Clustering Pipeline", flush=True)
        print("=" * 60, flush=True)

        # 1. Load data
        print("\n[1/4] Loading samples...", flush=True)
        self.load_samples_from_json(json_path)
        if len(self.samples) < self.n_clusters:
            print(f"  Warning: only {len(self.samples)} samples, reducing clusters to match", flush=True)
            self.n_clusters = max(1, len(self.samples))

        # 2. Extract features & cluster
        print("\n[2/4] Extracting features and clustering...", flush=True)
        features = self.extract_features()
        labels = self.cluster(features)

        # 3. Solve per cluster
        print("\n[3/4] Solving representative commitments...", flush=True)
        self.cluster_results = []
        unique_labels = sorted(set(labels))
        for k in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == k]
            result = self.solve_cluster(k, indices)
            self.cluster_results.append(result)

        # 4. Summary
        print("\n[4/4] Summary", flush=True)
        self._print_summary()

        return self.cluster_results

    def _print_summary(self) -> None:
        print("\n" + "=" * 70, flush=True)
        print(f"{'Cluster':>8} {'Days':>5} {'Feasib%':>8} {'OptCost':>12} "
              f"{'RepCost':>12} {'CostInc%':>9} {'OptTrans':>9} {'RepTrans':>9}", flush=True)
        print("-" * 70, flush=True)
        for r in self.cluster_results:
            if not r.get("success"):
                print(f"{r['cluster_id']:>8} {r['n_days']:>5}   FAILED", flush=True)
                continue
            print(
                f"{r['cluster_id']:>8} {r['n_days']:>5} "
                f"{r['feasibility_rate']:>7.1%} "
                f"{r.get('avg_optimal_cost', 0) or 0:>12.1f} "
                f"{r.get('avg_representative_cost', 0) or 0:>12.1f} "
                f"{r.get('cost_increase_pct', 0) or 0:>8.2f}% "
                f"{r.get('avg_optimal_transitions', 0):>9.1f} "
                f"{r.get('total_transitions', 0):>9}",
                flush=True,
            )
        print("=" * 70, flush=True)

    # ----- Save results -----

    def save_results(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            out_dir = Path(__file__).resolve().parent.parent / "result" / "commitment_clustering"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                out_dir / f"clustering_{self.case_name}_M{self.n_clusters}_{timestamp}.json"
            )

        output = {
            "metadata": {
                "case_name": self.case_name,
                "n_clusters": self.n_clusters,
                "n_samples": len(self.samples),
                "timestamp": datetime.now().isoformat(),
            },
            "parameters": {
                "transition_penalty": self.transition_penalty,
                "lp_proximity_weight": self.lp_proximity_weight,
                "max_cost_increase_ratio": self.max_cost_increase_ratio,
                "max_scenarios_per_cluster": self.max_scenarios_per_cluster,
                "require_all_cluster_days_in_uc": self.require_all_cluster_days_in_uc,
                "gurobi_time_limit": self.gurobi_time_limit,
                "feature_mode": self.feature_mode,
                "T_delta": self.T_delta,
            },
            "clusters": self.cluster_results,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f"\nResults saved to: {output_path}", flush=True)
        return output_path


# ---------------------------------------------------------------------------
# Per-generator commitment-pattern library
# ---------------------------------------------------------------------------

def solve_pattern_restricted_uc(
    ppc: dict,
    scenario: Dict,
    allowed_patterns: List[List[np.ndarray]],
    T_delta: float,
    time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], float, str, Optional[List[int]]]:
    """Solve UC for one scenario with per-generator pattern-library restrictions."""
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng = gen.shape[0]

    if len(allowed_patterns) != ng:
        raise ValueError(
            f"allowed_patterns length {len(allowed_patterns)} does not match ng={ng}"
        )

    load_data = np.asarray(scenario["load_data"], dtype=float)
    renewable_data = scenario.get("renewable_data")
    if renewable_data is not None:
        renewable_data = np.asarray(renewable_data, dtype=float)

    T = load_data.shape[1]
    nb = load_data.shape[0]

    for g, patterns_g in enumerate(allowed_patterns):
        if not patterns_g:
            return None, float("inf"), f"generator_{g}_has_no_patterns", None
        for pattern in patterns_g:
            if np.asarray(pattern).shape != (T,):
                raise ValueError(
                    f"generator {g} pattern shape {np.asarray(pattern).shape} does not match T={T}"
                )

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_steps(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    if renewable_data is not None:
        renewable_bus_ids = np.where(np.any(np.abs(renewable_data) > 1e-9, axis=1))[0]
    else:
        renewable_bus_ids = np.array([], dtype=int)
    nr = len(renewable_bus_ids)

    try:
        PTDF_full, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
    except Exception as e:
        print(f"  Warning: PTDF build failed ({e}); DC flow constraints skipped", flush=True)
        active_lines = []
        ptdf_g = None
        branch_limit = None
        PTDF_full = None

    model = gp.Model("PatternRestrictedUC")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    y = {}
    for g, patterns_g in enumerate(allowed_patterns):
        for k in range(len(patterns_g)):
            y[g, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}_{k}")

    x = model.addVars(ng, T, lb=0.0, ub=1.0, name="x")
    coc = model.addVars(ng, T - 1, lb=0.0, name="coc")
    pg = model.addVars(ng, T, lb=0.0, name="pg")
    cpower = model.addVars(ng, T, lb=0.0, name="cp")
    p_ren = model.addVars(nr, T, lb=0.0, name="pr") if nr > 0 else None

    model.update()

    for g, patterns_g in enumerate(allowed_patterns):
        model.addConstr(
            gp.quicksum(y[g, k] for k in range(len(patterns_g))) == 1,
            name=f"select_pattern_{g}",
        )
        for t in range(T):
            model.addConstr(
                x[g, t] == gp.quicksum(
                    float(patterns_g[k][t]) * y[g, k]
                    for k in range(len(patterns_g))
                ),
                name=f"pattern_link_{g}_{t}",
            )

    for g in range(ng):
        for tau in range(1, int(min_up_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau])

    for g in range(ng):
        for tau in range(1, int(min_down_steps[g]) + 1):
            for t1 in range(T - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau])

    for t in range(1, T):
        for g in range(ng):
            model.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]))

    for t in range(T):
        ren_supply = (
            gp.quicksum(p_ren[r, t] for r in range(nr))
            if nr > 0 and p_ren is not None else 0
        )
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + ren_supply
            == float(np.sum(load_data[:, t])),
            name=f"pb_{t}",
        )
        for g in range(ng):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])
            model.addConstr(
                cpower[g, t]
                >= gencost[g, -2] / T_delta * pg[g, t]
                + gencost[g, -1] / T_delta * x[g, t]
            )

    if nr > 0 and renewable_data is not None and p_ren is not None:
        for r, bus_idx in enumerate(renewable_bus_ids):
            for t in range(T):
                model.addConstr(p_ren[r, t] <= float(renewable_data[bus_idx, t]))

    for g in range(ng):
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t - 1]
                <= Ru[g] * x[g, t - 1] + Ru_co[g] * (1 - x[g, t - 1])
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t]
                <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t])
            )

    if active_lines and ptdf_g is not None:
        ptdf_Pd = PTDF_full @ load_data
        if nr > 0 and renewable_data is not None:
            R_bus = np.zeros((nb, nr))
            for r, bus_idx in enumerate(renewable_bus_ids):
                R_bus[bus_idx, r] = 1.0
            ptdf_R = PTDF_full @ R_bus
        else:
            ptdf_R = None

        for l_idx in active_lines:
            limit = float(branch_limit[l_idx])
            for t in range(T):
                flow_expr = gp.quicksum(
                    float(ptdf_g[l_idx, g]) * pg[g, t] for g in range(ng)
                ) - float(ptdf_Pd[l_idx, t])
                if ptdf_R is not None and nr > 0 and p_ren is not None:
                    flow_expr += gp.quicksum(
                        float(ptdf_R[l_idx, r]) * p_ren[r, t] for r in range(nr)
                    )
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)

    obj = (
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(T - 1))
    )
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        status_str = "optimal"
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        status_str = "time_limit_with_solution"
    elif model.status == GRB.SUBOPTIMAL:
        status_str = "suboptimal"
    else:
        return None, float("inf"), f"infeasible_or_error(status={model.status})", None

    x_sol = np.array([[x[g, t].X for t in range(T)] for g in range(ng)])
    x_sol = np.round(x_sol).astype(int)
    selected_indices = []
    for g, patterns_g in enumerate(allowed_patterns):
        chosen_k = int(np.argmax([y[g, k].X for k in range(len(patterns_g))]))
        selected_indices.append(chosen_k)

    return x_sol, float(model.objVal), status_str, selected_indices


class CommitmentPatternLibrary:
    """Restrict each generator to a shared library of day-ahead commitment patterns."""

    def __init__(
        self,
        ppc: dict,
        T_delta: float = 1.0,
        case_name: str = "case118",
        initial_patterns_per_unit: int = 10,
        max_patterns_per_unit: Optional[int] = None,
        gurobi_time_limit: float = 600.0,
        mip_gap: float = 1e-4,
        max_samples: Optional[int] = None,
        repair_non_optimal: bool = True,
        non_optimal_time_limit_factor: float = 2.0,
        verbose: bool = False,
    ):
        self.ppc = ppc
        self.T_delta = T_delta
        self.case_name = case_name
        self.initial_patterns_per_unit = max(1, int(initial_patterns_per_unit))
        self.max_patterns_per_unit = (
            None if max_patterns_per_unit is None else max(1, int(max_patterns_per_unit))
        )
        self.gurobi_time_limit = gurobi_time_limit
        self.mip_gap = mip_gap
        self.max_samples = max_samples
        self.repair_non_optimal = repair_non_optimal
        self.non_optimal_time_limit_factor = max(float(non_optimal_time_limit_factor), 1.0)
        self.verbose = verbose

        self.samples: List[Dict] = []
        self.pattern_counters: List[Counter] = []
        self.pattern_library: List[List[np.ndarray]] = []
        self.pattern_library_keys: List[List[Tuple[int, ...]]] = []
        self.pattern_library_key_sets: List[set] = []
        self.initial_pattern_counts: List[int] = []
        self.expansion_log: List[Dict] = []
        self.optimality_repair_log: List[Dict] = []
        self.scenario_results: List[Dict] = []
        self.summary: Dict = {}

    def load_samples_from_json(self, json_path: str) -> List[Dict]:
        self.samples = _load_commitment_samples_from_json(json_path)
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            print(f"  Truncated samples to first {len(self.samples)} scenarios", flush=True)
        return self.samples

    def _compute_optimal_cost_via_ed(self, sample: Dict) -> float:
        if sample.get("optimal_cost") is not None:
            return float(sample["optimal_cost"])
        result = evaluate_commitment_cost(
            self.ppc,
            sample["load_data"],
            sample["unit_commitment_matrix"],
            self.T_delta,
            renewable_data=sample.get("renewable_data"),
        )
        if result["success"]:
            sample["optimal_cost"] = float(result["total_cost"])
            return float(result["total_cost"])
        return float("inf")

    def _build_pattern_counters(self) -> List[Counter]:
        if not self.samples:
            raise ValueError("samples are not loaded")
        ng = self.samples[0]["unit_commitment_matrix"].shape[0]
        counters = [Counter() for _ in range(ng)]
        for sample in self.samples:
            x_mat = np.asarray(sample["unit_commitment_matrix"], dtype=int)
            for g in range(ng):
                counters[g][_pattern_to_key(x_mat[g, :])] += 1
        self.pattern_counters = counters
        return counters

    def _initialize_pattern_library(self) -> None:
        if not self.pattern_counters:
            self._build_pattern_counters()

        self.pattern_library = []
        self.pattern_library_keys = []
        self.pattern_library_key_sets = []
        self.initial_pattern_counts = []

        for g, counter in enumerate(self.pattern_counters):
            chosen = counter.most_common(self.initial_patterns_per_unit)
            keys = [key for key, _ in chosen]
            patterns = [np.asarray(key, dtype=int) for key in keys]
            self.pattern_library.append(patterns)
            self.pattern_library_keys.append(keys[:])
            self.pattern_library_key_sets.append(set(keys))
            self.initial_pattern_counts.append(len(patterns))
            print(
                f"  Generator {g:02d}: init patterns={len(patterns)}, "
                f"observed unique={len(counter)}",
                flush=True,
            )

    def _add_pattern_for_generator(
        self,
        generator_id: int,
        pattern_key: Tuple[int, ...],
    ) -> bool:
        if pattern_key in self.pattern_library_key_sets[generator_id]:
            return False
        if (
            self.max_patterns_per_unit is not None
            and len(self.pattern_library[generator_id]) >= self.max_patterns_per_unit
        ):
            return False
        self.pattern_library[generator_id].append(np.asarray(pattern_key, dtype=int))
        self.pattern_library_keys[generator_id].append(pattern_key)
        self.pattern_library_key_sets[generator_id].add(pattern_key)
        return True

    def _expand_library_for_scenario(self, sample: Dict) -> List[int]:
        x_opt = np.asarray(sample["unit_commitment_matrix"], dtype=int)
        added_generators = []
        for g in range(x_opt.shape[0]):
            pattern_key = _pattern_to_key(x_opt[g, :])
            added = self._add_pattern_for_generator(g, pattern_key)
            if added:
                added_generators.append(g)
        return added_generators

    @staticmethod
    def _select_better_solve_info(base_info: Dict, retry_info: Dict) -> Dict:
        base_rank = _status_rank(base_info.get("status"))
        retry_rank = _status_rank(retry_info.get("status"))
        if retry_rank > base_rank:
            return retry_info
        if retry_rank < base_rank:
            return base_info

        if retry_info.get("success") and not base_info.get("success"):
            return retry_info
        if base_info.get("success") and not retry_info.get("success"):
            return base_info

        base_obj = base_info.get("objective_value", float("inf"))
        retry_obj = retry_info.get("objective_value", float("inf"))
        if retry_obj + 1e-6 < base_obj:
            return retry_info
        return base_info

    def _repair_non_optimal_scenario(
        self,
        sample: Dict,
        solve_info: Dict,
        scenario_index: int,
        phase: str,
    ) -> Dict:
        status = solve_info.get("status")
        if not self.repair_non_optimal or not _status_has_solution(status):
            return solve_info
        if _status_is_proven_optimal(status):
            return solve_info

        sample_id = sample["sample_id"]
        added_generators = self._expand_library_for_scenario(sample)
        retry_time_limit = max(
            self.gurobi_time_limit,
            self.gurobi_time_limit * self.non_optimal_time_limit_factor,
        )
        retry_info = self._solve_scenario(sample, time_limit=retry_time_limit)
        final_info = self._select_better_solve_info(solve_info, retry_info)

        self.optimality_repair_log.append({
            "sample_id": sample_id,
            "scenario_index": scenario_index,
            "phase": phase,
            "initial_status": solve_info.get("status"),
            "retry_status": retry_info.get("status"),
            "final_status": final_info.get("status"),
            "initial_objective_value": solve_info.get("objective_value"),
            "retry_objective_value": retry_info.get("objective_value"),
            "final_objective_value": final_info.get("objective_value"),
            "initial_solve_time_s": solve_info.get("solve_time_s"),
            "retry_solve_time_s": retry_info.get("solve_time_s"),
            "added_generators": added_generators,
            "n_added": len(added_generators),
            "retry_time_limit": retry_time_limit,
            "improved": final_info is not solve_info,
        })

        print(
            f"  Scenario {scenario_index + 1}/{len(self.samples)} sample_id={sample_id}: "
            f"non-optimal status {status} -> {final_info.get('status')} after repair "
            f"(added_generators={added_generators}, retry_time_limit={retry_time_limit:.1f}s)",
            flush=True,
        )
        return final_info

    def _solve_scenario(self, sample: Dict, time_limit: Optional[float] = None) -> Dict:
        t_start = time.time()
        x_sol, obj_val, status, selected_indices = solve_pattern_restricted_uc(
            ppc=self.ppc,
            scenario=sample,
            allowed_patterns=self.pattern_library,
            T_delta=self.T_delta,
            time_limit=self.gurobi_time_limit if time_limit is None else float(time_limit),
            mip_gap=self.mip_gap,
            verbose=self.verbose,
        )
        solve_time = time.time() - t_start
        return {
            "x_sol": x_sol,
            "objective_value": obj_val,
            "status": status,
            "selected_pattern_indices": selected_indices,
            "solve_time_s": solve_time,
            "success": x_sol is not None,
        }

    def _ensure_library_feasibility(self) -> None:
        print("\n[2/4] Traversing scenarios and expanding pattern library when needed...", flush=True)
        self.expansion_log = []
        self.optimality_repair_log = []

        for idx, sample in enumerate(self.samples):
            sample_id = sample["sample_id"]
            solve_info = self._solve_scenario(sample)
            if solve_info["success"]:
                if not _status_is_proven_optimal(solve_info["status"]):
                    solve_info = self._repair_non_optimal_scenario(
                        sample,
                        solve_info,
                        scenario_index=idx,
                        phase="feasibility_pass",
                    )
                print(
                    f"  Scenario {idx + 1}/{len(self.samples)} sample_id={sample_id}: "
                    f"feasible with current library (status={solve_info['status']})",
                    flush=True,
                )
                continue

            added_generators = self._expand_library_for_scenario(sample)
            print(
                f"  Scenario {idx + 1}/{len(self.samples)} sample_id={sample_id}: "
                f"infeasible, added patterns for generators {added_generators}",
                flush=True,
            )

            retry_info = self._solve_scenario(sample)
            if not retry_info["success"]:
                opt_feas = evaluate_commitment_cost(
                    self.ppc,
                    sample["load_data"],
                    sample["unit_commitment_matrix"],
                    self.T_delta,
                    renewable_data=sample.get("renewable_data"),
                )
                raise RuntimeError(
                    "Pattern-library-restricted UC remained infeasible after adding "
                    f"scenario-optimal patterns for sample_id={sample_id}. "
                    f"retry_status={retry_info['status']}, "
                    f"optimal_x_feasible={opt_feas.get('success', False)}"
                )

            self.expansion_log.append({
                "sample_id": sample_id,
                "scenario_index": idx,
                "added_generators": added_generators,
                "n_added": len(added_generators),
                "retry_status": retry_info["status"],
            })

    def _evaluate_all_scenarios(self) -> List[Dict]:
        print("\n[3/4] Evaluating final pattern library on all scenarios...", flush=True)
        scenario_results: List[Dict] = []

        for idx, sample in enumerate(self.samples):
            sample_id = sample["sample_id"]
            solve_info = self._solve_scenario(sample)
            if solve_info["success"] and not _status_is_proven_optimal(solve_info["status"]):
                solve_info = self._repair_non_optimal_scenario(
                    sample,
                    solve_info,
                    scenario_index=idx,
                    phase="evaluation_pass",
                )
            x_sol = solve_info["x_sol"]
            obj_val = solve_info["objective_value"]
            status = solve_info["status"]
            selected_indices = solve_info["selected_pattern_indices"]
            solve_time = solve_info["solve_time_s"]

            if x_sol is None:
                scenario_results.append({
                    "sample_id": sample_id,
                    "scenario_index": idx,
                    "success": False,
                    "solver_status": status,
                    "restricted_cost": None,
                    "optimal_cost": None,
                    "cost_increase_pct": None,
                })
                continue

            x_opt = np.asarray(sample["unit_commitment_matrix"], dtype=int)
            optimal_cost = self._compute_optimal_cost_via_ed(sample)
            transitions_opt = _count_transitions(x_opt)
            transitions_restricted = _count_transitions(x_sol)

            selected_keys = [
                self.pattern_library_keys[g][selected_indices[g]]
                for g in range(len(selected_indices))
            ]
            optimal_keys = [_pattern_to_key(x_opt[g, :]) for g in range(x_opt.shape[0])]
            matched_units = int(sum(
                1 for g in range(x_opt.shape[0]) if selected_keys[g] == optimal_keys[g]
            ))

            cost_increase_pct = None
            if np.isfinite(optimal_cost) and optimal_cost > 0:
                cost_increase_pct = (obj_val - optimal_cost) / optimal_cost * 100.0

            scenario_results.append({
                "sample_id": sample_id,
                "scenario_index": idx,
                "success": True,
                "solver_status": status,
                "restricted_cost": obj_val,
                "optimal_cost": optimal_cost if np.isfinite(optimal_cost) else None,
                "cost_increase_pct": cost_increase_pct,
                "optimal_transitions": transitions_opt,
                "restricted_transitions": transitions_restricted,
                "matched_unit_patterns": matched_units,
                "changed_unit_patterns": int(x_opt.shape[0] - matched_units),
                "selected_pattern_indices": selected_indices,
                "solve_time_s": solve_time,
            })

        self.scenario_results = scenario_results
        return scenario_results

    def _build_summary(self) -> Dict:
        status_counts = Counter(r.get("solver_status", "unknown") for r in self.scenario_results)
        feasible_results = [r for r in self.scenario_results if r.get("success")]
        feasible_cost_increases = [
            r["cost_increase_pct"]
            for r in feasible_results
            if r.get("cost_increase_pct") is not None
        ]
        optimal_costs = [
            r["optimal_cost"] for r in feasible_results if r.get("optimal_cost") is not None
        ]
        restricted_costs = [
            r["restricted_cost"] for r in feasible_results if r.get("restricted_cost") is not None
        ]
        final_counts = [len(patterns) for patterns in self.pattern_library]
        added_counts = [
            len(self.pattern_library[g]) - self.initial_pattern_counts[g]
            for g in range(len(self.pattern_library))
        ]
        matched_units = [
            float(r["matched_unit_patterns"])
            for r in feasible_results
            if r.get("matched_unit_patterns") is not None
        ]
        changed_units = [
            float(r["changed_unit_patterns"])
            for r in feasible_results
            if r.get("changed_unit_patterns") is not None
        ]
        solve_times = [
            float(r["solve_time_s"])
            for r in feasible_results
            if r.get("solve_time_s") is not None
        ]
        cost_gap_stats = _numeric_summary(feasible_cost_increases)
        solve_time_stats = _numeric_summary(solve_times)
        matched_unit_stats = _numeric_summary(matched_units)
        changed_unit_stats = _numeric_summary(changed_units)
        n_proven_optimal = int(status_counts.get("optimal", 0))

        summary = {
            "n_scenarios": len(self.samples),
            "n_feasible": len(feasible_results),
            "feasibility_rate": (
                len(feasible_results) / len(self.samples) if self.samples else 0.0
            ),
            "solver_status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "n_proven_optimal": n_proven_optimal,
            "n_feasible_non_optimal": int(len(feasible_results) - n_proven_optimal),
            "avg_optimal_cost": float(np.mean(optimal_costs)) if optimal_costs else None,
            "avg_restricted_cost": float(np.mean(restricted_costs)) if restricted_costs else None,
            "avg_cost_increase_pct": cost_gap_stats["mean"],
            "median_cost_increase_pct": cost_gap_stats["median"],
            "min_cost_increase_pct": cost_gap_stats["min"],
            "max_cost_increase_pct": cost_gap_stats["max"],
            "p95_cost_increase_pct": cost_gap_stats["p95"],
            "matched_unit_patterns_mean": matched_unit_stats["mean"],
            "matched_unit_patterns_min": matched_unit_stats["min"],
            "matched_unit_patterns_max": matched_unit_stats["max"],
            "changed_unit_patterns_mean": changed_unit_stats["mean"],
            "changed_unit_patterns_max": changed_unit_stats["max"],
            "solve_time_s_mean": solve_time_stats["mean"],
            "solve_time_s_median": solve_time_stats["median"],
            "solve_time_s_max": solve_time_stats["max"],
            "solve_time_s_p95": solve_time_stats["p95"],
            "mean_final_patterns_per_unit": (
                float(np.mean(final_counts)) if final_counts else None
            ),
            "max_final_patterns_per_unit": max(final_counts) if final_counts else None,
            "total_added_patterns": int(np.sum(added_counts)) if added_counts else 0,
            "n_scenarios_triggering_expansion": len(self.expansion_log),
            "n_non_optimal_repairs": len(self.optimality_repair_log),
            "n_non_optimal_repaired_to_optimal": int(sum(
                1
                for log in self.optimality_repair_log
                if log.get("initial_status") != "optimal"
                and log.get("final_status") == "optimal"
            )),
        }
        self.summary = summary
        return summary

    def _print_summary(self) -> None:
        summary = self.summary or self._build_summary()
        print("\n[4/4] Summary", flush=True)
        print("=" * 72, flush=True)
        print(
            f"  Feasibility: {summary['feasibility_rate']:.1%} "
            f"({summary['n_feasible']}/{summary['n_scenarios']})",
            flush=True,
        )
        print(f"  Avg optimal cost:    {summary['avg_optimal_cost']}", flush=True)
        print(f"  Avg restricted cost: {summary['avg_restricted_cost']}", flush=True)
        print(f"  Avg cost increase:   {summary['avg_cost_increase_pct']}%", flush=True)
        print(f"  Median cost increase:{summary['median_cost_increase_pct']}%", flush=True)
        print(f"  Max cost increase:   {summary['max_cost_increase_pct']}%", flush=True)
        print(f"  P95 cost increase:   {summary['p95_cost_increase_pct']}%", flush=True)
        print(f"  Solver statuses:     {summary['solver_status_counts']}", flush=True)
        print(
            f"  Matched unit patterns: mean={summary['matched_unit_patterns_mean']}, "
            f"min={summary['matched_unit_patterns_min']}, "
            f"max={summary['matched_unit_patterns_max']}",
            flush=True,
        )
        print(
            f"  Solve time (s):      mean={summary['solve_time_s_mean']}, "
            f"median={summary['solve_time_s_median']}, "
            f"p95={summary['solve_time_s_p95']}, "
            f"max={summary['solve_time_s_max']}",
            flush=True,
        )
        print(
            f"  Pattern expansions:  {summary['n_scenarios_triggering_expansion']} scenarios, "
            f"{summary['total_added_patterns']} added unit-patterns",
            flush=True,
        )
        print(
            f"  Non-optimal repairs: {summary['n_non_optimal_repairs']} attempts, "
            f"{summary['n_non_optimal_repaired_to_optimal']} repaired to optimal",
            flush=True,
        )
        print(
            f"  Final patterns/unit: mean={summary['mean_final_patterns_per_unit']}, "
            f"max={summary['max_final_patterns_per_unit']}",
            flush=True,
        )
        print("=" * 72, flush=True)

    def run(self, json_path: str) -> List[Dict]:
        print("=" * 72, flush=True)
        print("  Commitment Pattern Library Pipeline", flush=True)
        print("=" * 72, flush=True)

        print("\n[1/4] Loading samples and initializing library...", flush=True)
        self.load_samples_from_json(json_path)
        self._build_pattern_counters()
        self._initialize_pattern_library()

        self._ensure_library_feasibility()
        self._evaluate_all_scenarios()
        self._build_summary()
        self._print_summary()
        return self.scenario_results

    def save_results(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            out_dir = Path(__file__).resolve().parent.parent / "result" / "commitment_clustering"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                out_dir
                / f"pattern_library_{self.case_name}_K{self.initial_patterns_per_unit}_{timestamp}.json"
            )

        pattern_library_output = []
        for g, counter in enumerate(self.pattern_counters):
            total_obs = int(sum(counter.values()))
            patterns_out = []
            for k, key in enumerate(self.pattern_library_keys[g]):
                freq = int(counter.get(key, 0))
                share_pct = (freq / total_obs * 100.0) if total_obs > 0 else None
                patterns_out.append({
                    "pattern_index": k,
                    "frequency": freq,
                    "share_pct": share_pct,
                    "pattern": "".join(str(v) for v in key),
                })
            pattern_library_output.append({
                "generator_id": g,
                "initial_pattern_count": self.initial_pattern_counts[g],
                "final_pattern_count": len(self.pattern_library[g]),
                "unique_observed_patterns": len(counter),
                "patterns": patterns_out,
            })

        output = {
            "metadata": {
                "case_name": self.case_name,
                "n_samples": len(self.samples),
                "timestamp": datetime.now().isoformat(),
            },
            "parameters": {
                "initial_patterns_per_unit": self.initial_patterns_per_unit,
                "max_patterns_per_unit": self.max_patterns_per_unit,
                "gurobi_time_limit": self.gurobi_time_limit,
                "mip_gap": self.mip_gap,
                "max_samples": self.max_samples,
                "repair_non_optimal": self.repair_non_optimal,
                "non_optimal_time_limit_factor": self.non_optimal_time_limit_factor,
                "T_delta": self.T_delta,
            },
            "summary": self.summary,
            "pattern_library": pattern_library_output,
            "expansion_log": self.expansion_log,
            "optimality_repair_log": self.optimality_repair_log,
            "scenario_results": self.scenario_results,
            "scenarios": self.scenario_results,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f"\nResults saved to: {output_path}", flush=True)
        return output_path
