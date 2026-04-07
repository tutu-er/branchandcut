"""Commitment Clustering: cluster daily scenarios and generate representative
commitment schedules with fewer unit start/stop transitions.

Multi-scenario shared-commitment UC formulation per cluster:
- Shared binary x[g,t] across all days in the cluster
- Independent dispatch pg[g,t,d] per day
- Transition penalty to reduce startups/shutdowns
- Optional LP-relaxation proximity term
- Optional cost-increase bound
"""

from __future__ import annotations

import json
import time
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_custom_generator_array(ppc_raw: dict, key: str, ng: int) -> Optional[np.ndarray]:
    values = ppc_raw.get(key)
    if values is None:
        return None
    values = np.asarray(values)
    if values.shape[0] != ng:
        return None
    raw_gen = np.asarray(ppc_raw.get("gen"))
    if raw_gen.shape[0] != ng:
        return values
    order = np.argsort(raw_gen[:, GEN_BUS], kind="stable")
    return values[order]


def _get_ramp_limits(ppc_raw: dict, gen: np.ndarray, T_delta: float):
    ng = gen.shape[0]
    default_up = 0.4 * gen[:, PMAX] / T_delta
    default_down = 0.4 * gen[:, PMAX] / T_delta
    default_up_co = 0.3 * gen[:, PMAX]
    default_down_co = 0.3 * gen[:, PMAX]

    ramp_up_h = _get_custom_generator_array(ppc_raw, "uc_ramp_up_mw_per_h", ng)
    ramp_down_h = _get_custom_generator_array(ppc_raw, "uc_ramp_down_mw_per_h", ng)
    if ramp_up_h is None or ramp_down_h is None:
        return default_up, default_down, default_up_co, default_down_co

    Ru = np.maximum(np.asarray(ramp_up_h, dtype=float) * T_delta, default_up)
    Rd = np.maximum(np.asarray(ramp_down_h, dtype=float) * T_delta, default_down)
    Ru_co = np.maximum(Ru, gen[:, PMIN])
    Rd_co = np.maximum(Rd, gen[:, PMIN])
    return Ru, Rd, Ru_co, Rd_co


def _get_min_up_down_steps(ppc_raw: dict, gen: np.ndarray, T_delta: float):
    ng = gen.shape[0]
    min_up_h = _get_custom_generator_array(ppc_raw, "uc_min_up_time_h", ng)
    min_down_h = _get_custom_generator_array(ppc_raw, "uc_min_down_time_h", ng)
    if min_up_h is None or min_down_h is None:
        default_steps = max(int(4 * T_delta), 1)
        return (
            np.full(ng, default_steps, dtype=int),
            np.full(ng, default_steps, dtype=int),
        )
    min_up = np.maximum(np.ceil(np.asarray(min_up_h, dtype=float) / T_delta).astype(int), 1)
    min_down = np.maximum(np.ceil(np.asarray(min_down_h, dtype=float) / T_delta).astype(int), 1)
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

            # Try to recover optimal cost from raw JSON lambda/cost info
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

        self.samples = samples
        print(f"  Loaded {len(samples)} valid samples from {json_path}", flush=True)
        return samples

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
