#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.case_registry import get_case_ppc
from src.scenario_utils import normalize_sample_arrays
from src.uc_NN_subproblem import (
    ActiveSetReader,
    _build_generator_injection_sensitivity,
    _get_custom_generator_array_from_ppc,
    _get_ramp_limits_from_ppc,
)
from src.uc_gurobipy import UnitCommitmentModel


CASE_NAME = "case3lite"
ACTIVE_SET_DIR = ROOT / "result" / "active_set"
ACTIVE_SET_GLOB = "active_sets_case3lite_*.json"
ACTIVE_SET_JSON = None
SAMPLE_ID = 0
T_DELTA = 1.0
SOLVE_GLOBAL_UC = False
VERBOSE_SUBPROBLEM = False
PRICE_MODES = ("manual_ed_pg_dual",)


def _resolve_latest_active_set_json(case_name: str, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    search_dir = ACTIVE_SET_DIR
    pattern = f"active_sets_{case_name}_*.json"
    candidates = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No active set JSON found for case '{case_name}' under {search_dir}"
        )
    return candidates[0]


def _recover_x_true_from_sample(sample: dict, ng: int, T: int) -> np.ndarray:
    x_true = np.zeros((ng, T), dtype=float)
    if "unit_commitment_matrix" in sample:
        uc = np.asarray(sample["unit_commitment_matrix"], dtype=float)
        rows = min(ng, uc.shape[0])
        cols = min(T, uc.shape[1])
        x_true[:rows, :cols] = uc[:rows, :cols]
        return x_true

    active_set = sample.get("active_set", [])
    for item in active_set:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
            g, t = item[0]
            value = item[1]
            if 0 <= g < ng and 0 <= t < T:
                x_true[g, t] = float(value)
    return x_true


def _get_min_up_down_time_steps(ppc_raw, gen: np.ndarray, T_delta: float) -> tuple[np.ndarray, np.ndarray]:
    ng = gen.shape[0]
    min_up_h = _get_custom_generator_array_from_ppc(ppc_raw, ng, "uc_min_up_time_h")
    min_down_h = _get_custom_generator_array_from_ppc(ppc_raw, ng, "uc_min_down_time_h")
    if min_up_h is None or min_down_h is None:
        default_steps = max(int(4 * T_delta), 1)
        return (
            np.full(ng, default_steps, dtype=int),
            np.full(ng, default_steps, dtype=int),
        )

    min_up = np.maximum(np.ceil(np.asarray(min_up_h, dtype=float) / T_delta).astype(int), 1)
    min_down = np.maximum(np.ceil(np.asarray(min_down_h, dtype=float) / T_delta).astype(int), 1)
    return min_up, min_down


def _solve_single_unit_milp(
    ppc,
    unit_id: int,
    lambda_eff: np.ndarray,
    T_delta: float,
    verbose: bool = False,
) -> dict:
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]

    g = int(unit_id)
    T = int(lambda_eff.shape[0])
    Pmin = float(gen[g, PMIN])
    Pmax = float(gen[g, PMAX])
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, gen, T_delta)
    start_cost = float(gencost[g, 1])
    shut_cost = float(gencost[g, 2])
    a = float(gencost[g, -2] / T_delta)
    b = float(gencost[g, -1] / T_delta)

    model = gp.Model(f"subproblem_unit_{g}")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0

    pg = model.addVars(T, lb=0.0, name="pg")
    x = model.addVars(T, vtype=GRB.BINARY, name="x")
    coc = model.addVars(max(T - 1, 0), lb=0.0, name="coc")
    cpower = model.addVars(T, lb=0.0, name="cpower")

    for t in range(T):
        model.addConstr(pg[t] >= Pmin * x[t], name=f"pg_lower_{t}")
        model.addConstr(pg[t] <= Pmax * x[t], name=f"pg_upper_{t}")
        model.addConstr(cpower[t] >= a * pg[t] + b * x[t], name=f"cpower_{t}")

    for t in range(1, T):
        model.addConstr(
            pg[t] - pg[t - 1] <= Ru[g] * x[t - 1] + Ru_co[g] * (1 - x[t - 1]),
            name=f"ramp_up_{t}",
        )
        model.addConstr(
            pg[t - 1] - pg[t] <= Rd[g] * x[t] + Rd_co[g] * (1 - x[t]),
            name=f"ramp_down_{t}",
        )
        model.addConstr(coc[t - 1] >= start_cost * (x[t] - x[t - 1]), name=f"start_cost_{t}")
        model.addConstr(coc[t - 1] >= shut_cost * (x[t - 1] - x[t]), name=f"shut_cost_{t}")

    for tau in range(1, min_up_steps[g] + 1):
        for t1 in range(T - tau):
            model.addConstr(x[t1 + 1] - x[t1] <= x[t1 + tau], name=f"min_on_{tau}_{t1}")

    for tau in range(1, min_down_steps[g] + 1):
        for t1 in range(T - tau):
            model.addConstr(-x[t1 + 1] + x[t1] <= 1 - x[t1 + tau], name=f"min_off_{tau}_{t1}")

    obj = gp.quicksum(cpower[t] for t in range(T))
    obj += gp.quicksum(coc[t] for t in range(max(T - 1, 0)))
    obj -= gp.quicksum(float(lambda_eff[t]) * pg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"subproblem MILP failed for unit {g}, status={model.status}")

    return {
        "x": np.array([x[t].X for t in range(T)], dtype=float),
        "pg": np.array([pg[t].X for t in range(T)], dtype=float),
        "objective": float(model.ObjVal),
    }


def _evaluate_single_unit_fixed_commitment(
    ppc,
    unit_id: int,
    lambda_eff: np.ndarray,
    x_fixed: np.ndarray,
    T_delta: float,
    verbose: bool = False,
) -> dict:
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]

    g = int(unit_id)
    T = int(lambda_eff.shape[0])
    x_fixed = np.asarray(x_fixed, dtype=float).reshape(T)
    Pmin = float(gen[g, PMIN])
    Pmax = float(gen[g, PMAX])
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, gen, T_delta)
    start_cost = float(gencost[g, 1])
    shut_cost = float(gencost[g, 2])
    a = float(gencost[g, -2] / T_delta)
    b = float(gencost[g, -1] / T_delta)

    model = gp.Model(f"subproblem_unit_{g}_fixed_x")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0

    pg = model.addVars(T, lb=0.0, name="pg")
    coc = model.addVars(max(T - 1, 0), lb=0.0, name="coc")
    cpower = model.addVars(T, lb=0.0, name="cpower")

    for t in range(T):
        model.addConstr(pg[t] >= Pmin * x_fixed[t], name=f"pg_lower_{t}")
        model.addConstr(pg[t] <= Pmax * x_fixed[t], name=f"pg_upper_{t}")
        model.addConstr(cpower[t] >= a * pg[t] + b * x_fixed[t], name=f"cpower_{t}")

    for t in range(1, T):
        model.addConstr(
            pg[t] - pg[t - 1] <= Ru[g] * x_fixed[t - 1] + Ru_co[g] * (1 - x_fixed[t - 1]),
            name=f"ramp_up_{t}",
        )
        model.addConstr(
            pg[t - 1] - pg[t] <= Rd[g] * x_fixed[t] + Rd_co[g] * (1 - x_fixed[t]),
            name=f"ramp_down_{t}",
        )
        model.addConstr(
            coc[t - 1] >= start_cost * (x_fixed[t] - x_fixed[t - 1]),
            name=f"start_cost_{t}",
        )
        model.addConstr(
            coc[t - 1] >= shut_cost * (x_fixed[t - 1] - x_fixed[t]),
            name=f"shut_cost_{t}",
        )

    for tau in range(1, min_up_steps[g] + 1):
        for t1 in range(T - tau):
            if x_fixed[t1 + 1] - x_fixed[t1] > x_fixed[t1 + tau] + 1e-9:
                return {
                    "objective": float("inf"),
                    "pg": None,
                    "feasible": False,
                }

    for tau in range(1, min_down_steps[g] + 1):
        for t1 in range(T - tau):
            if -x_fixed[t1 + 1] + x_fixed[t1] > 1 - x_fixed[t1 + tau] + 1e-9:
                return {
                    "objective": float("inf"),
                    "pg": None,
                    "feasible": False,
                }

    obj = gp.quicksum(cpower[t] for t in range(T))
    obj += gp.quicksum(coc[t] for t in range(max(T - 1, 0)))
    obj -= gp.quicksum(float(lambda_eff[t]) * pg[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return {
            "objective": float("inf"),
            "pg": None,
            "feasible": False,
        }

    return {
        "objective": float(model.ObjVal),
        "pg": np.array([pg[t].X for t in range(T)], dtype=float),
        "feasible": True,
    }


def _load_sample(active_set_json: str | Path, sample_id: int) -> dict:
    reader = ActiveSetReader(str(active_set_json))
    sample = reader.get_sample_data(sample_id)
    if sample is None:
        raise IndexError(f"sample_id={sample_id} out of range for {active_set_json}")
    return normalize_sample_arrays(dict(sample))


def _resolve_x_true(ppc, sample: dict, T_delta: float, solve_global_uc: bool) -> tuple[np.ndarray, str]:
    ppc_int = ext2int(ppc)
    ng = ppc_int["gen"].shape[0]
    T = sample["pd_data"].shape[1]

    if not solve_global_uc:
        return _recover_x_true_from_sample(sample, ng, T), "sample"

    uc = UnitCommitmentModel(
        ppc,
        sample["pd_data"],
        T_delta,
        renewable_data=sample.get("renewable_data"),
        verbose=False,
    )
    _, x_sol, _ = uc.solve()
    if x_sol is None:
        raise RuntimeError("global UC solve failed")
    return np.asarray(x_sol, dtype=float), "global_uc"


def _solve_ed_pg_effective_lambda(
    ppc,
    sample: dict,
    x_true: np.ndarray,
    T_delta: float,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    bus = ppc_int["bus"]
    branch = ppc_int["branch"]
    gencost = ppc_int["gencost"]

    ng = gen.shape[0]
    T = sample["pd_data"].shape[1]
    nb = sample["pd_data"].shape[0]
    nl = branch.shape[0]
    load_data = np.asarray(sample["pd_data"], dtype=float)
    renewable_data = sample.get("renewable_data")
    renewable_data = None if renewable_data is None else np.asarray(renewable_data, dtype=float)

    if renewable_data is not None and renewable_data.shape != load_data.shape:
        raise ValueError(
            f"renewable_data shape {renewable_data.shape} does not match load shape {load_data.shape}"
        )

    renewable_bus_ids = (
        np.where(np.any(renewable_data > 1e-9, axis=1))[0]
        if renewable_data is not None else np.array([], dtype=int)
    )
    nr = len(renewable_bus_ids)
    R = np.zeros((nb, nr), dtype=float)
    for r, bus_idx in enumerate(renewable_bus_ids):
        R[bus_idx, r] = 1.0

    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0

    PTDF = makePTDF(ppc_int["baseMVA"], bus, branch)
    PTDF_G = PTDF @ G
    branch_limit = branch[:, RATE_A]
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)

    model = gp.Model("manual_ed_pg_dual")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0

    pg = model.addVars(ng, T, lb=0.0, name="pg")
    cpower = model.addVars(ng, T, lb=0.0, name="cpower")
    p_ren = model.addVars(nr, T, lb=0.0, name="p_ren") if nr > 0 else None

    for t in range(T):
        renewable_supply = gp.quicksum(p_ren[r, t] for r in range(nr)) if nr > 0 else 0.0
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + renewable_supply == float(np.sum(load_data[:, t])),
            name=f"power_balance_{t}",
        )
        for g in range(ng):
            model.addConstr(pg[g, t] >= float(gen[g, PMIN] * x_true[g, t]), name=f"pg_lower_{g}_{t}")
            model.addConstr(pg[g, t] <= float(gen[g, PMAX] * x_true[g, t]), name=f"pg_upper_{g}_{t}")
            model.addConstr(
                cpower[g, t] >= float(gencost[g, -2] / T_delta) * pg[g, t] + float(gencost[g, -1] / T_delta * x_true[g, t]),
                name=f"cpower_{g}_{t}",
            )
        for r, bus_idx in enumerate(renewable_bus_ids):
            model.addConstr(p_ren[r, t] <= float(renewable_data[bus_idx, t]), name=f"ren_upper_{r}_{t}")

    for t in range(1, T):
        for g in range(ng):
            model.addConstr(
                pg[g, t] - pg[g, t - 1] <= float(Ru[g] * x_true[g, t - 1] + Ru_co[g] * (1 - x_true[g, t - 1])),
                name=f"ramp_up_{g}_{t}",
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t] <= float(Rd[g] * x_true[g, t] + Rd_co[g] * (1 - x_true[g, t])),
                name=f"ramp_down_{g}_{t}",
            )

    for t in range(T):
        thermal_injection = G @ np.array([pg[g, t] for g in range(ng)], dtype=object)
        renewable_injection = R @ np.array([p_ren[r, t] for r in range(nr)], dtype=object) if nr > 0 else 0.0
        flow = PTDF @ (thermal_injection + renewable_injection - load_data[:, t])
        for l in range(nl):
            model.addConstr(flow[l] <= float(branch_limit[l]), name=f"flow_upper_{l}_{t}")
            model.addConstr(flow[l] >= -float(branch_limit[l]), name=f"flow_lower_{l}_{t}")

    obj = gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"manual ED solve failed with status={model.status}")

    lambda_power_balance = np.zeros(T, dtype=float)
    lambda_pg_lower = np.zeros((ng, T), dtype=float)
    lambda_pg_upper = np.zeros((ng, T), dtype=float)
    lambda_ramp_up = np.zeros((ng, T - 1), dtype=float)
    lambda_ramp_down = np.zeros((ng, T - 1), dtype=float)
    lambda_flow_upper = np.zeros((nl, T), dtype=float)
    lambda_flow_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        lambda_power_balance[t] = float(model.getConstrByName(f"power_balance_{t}").Pi)
        for g in range(ng):
            lambda_pg_lower[g, t] = float(model.getConstrByName(f"pg_lower_{g}_{t}").Pi)
            lambda_pg_upper[g, t] = float(model.getConstrByName(f"pg_upper_{g}_{t}").Pi)
        for l in range(nl):
            lambda_flow_upper[l, t] = float(model.getConstrByName(f"flow_upper_{l}_{t}").Pi)
            lambda_flow_lower[l, t] = float(model.getConstrByName(f"flow_lower_{l}_{t}").Pi)

    for t in range(1, T):
        for g in range(ng):
            lambda_ramp_up[g, t - 1] = float(model.getConstrByName(f"ramp_up_{g}_{t}").Pi)
            lambda_ramp_down[g, t - 1] = float(model.getConstrByName(f"ramp_down_{g}_{t}").Pi)

    effective = np.zeros((ng, T), dtype=float)
    components = {
        "lambda_power_balance": np.tile(lambda_power_balance, (ng, 1)),
        "lambda_pg_lower": lambda_pg_lower.copy(),
        "lambda_pg_upper": lambda_pg_upper.copy(),
        "lambda_ramp_up": np.zeros((ng, T), dtype=float),
        "lambda_ramp_down": np.zeros((ng, T), dtype=float),
        "lambda_flow": np.zeros((ng, T), dtype=float),
    }

    for g in range(ng):
        for t in range(T):
            price = lambda_power_balance[t]
            price += lambda_pg_lower[g, t]
            price += lambda_pg_upper[g, t]

            ramp_contrib = 0.0
            if t > 0:
                ramp_contrib += lambda_ramp_up[g, t - 1]
                ramp_contrib -= lambda_ramp_down[g, t - 1]
            if t < T - 1:
                ramp_contrib -= lambda_ramp_up[g, t]
                ramp_contrib += lambda_ramp_down[g, t]
            price += ramp_contrib

            flow_contrib = float(
                np.dot(PTDF_G[:, g], lambda_flow_upper[:, t] + lambda_flow_lower[:, t])
            )
            price += flow_contrib

            effective[g, t] = price
            components["lambda_ramp_up"][g, t] = (
                (lambda_ramp_up[g, t - 1] if t > 0 else 0.0)
                - (lambda_ramp_up[g, t] if t < T - 1 else 0.0)
            )
            components["lambda_ramp_down"][g, t] = (
                -(lambda_ramp_down[g, t - 1] if t > 0 else 0.0)
                + (lambda_ramp_down[g, t] if t < T - 1 else 0.0)
            )
            components["lambda_flow"][g, t] = flow_contrib

    payload = {
        "effective_lambda": effective,
        "components": components,
        "lambda_flow_upper": lambda_flow_upper,
        "lambda_flow_lower": lambda_flow_lower,
        "ptdf_g": PTDF_G,
        "pg_solution": np.array([[pg[g, t].X for t in range(T)] for g in range(ng)], dtype=float),
        "objective": float(model.ObjVal),
    }
    return effective, payload


def _normalize_dcpf_shape(arr: np.ndarray, nl: int, T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (nl, T):
        return arr
    if arr.shape == (T, nl):
        return arr.T
    raise ValueError(f"Unexpected DCPF dual shape {arr.shape}, expected {(nl, T)} or {(T, nl)}")


def _resolve_effective_lambda_from_global_uc_lp(
    ppc,
    sample: dict,
    T_delta: float,
) -> tuple[np.ndarray, str]:
    uc = UnitCommitmentModel(
        ppc,
        sample["pd_data"],
        T_delta,
        renewable_data=sample.get("renewable_data"),
        verbose=False,
    )
    _, _, _, lambda_sol = uc.solve_with_dual(type=0)
    if lambda_sol is None:
        raise RuntimeError("global UC LP dual solve failed")

    ppc_int = ext2int(ppc)
    ng = ppc_int["gen"].shape[0]
    nl = ppc_int["branch"].shape[0]
    T = sample["pd_data"].shape[1]
    sensitivity = _build_generator_injection_sensitivity(ppc)

    lambda_pb = np.asarray(lambda_sol["lambda_power_balance"], dtype=float).reshape(T)
    lambda_du = _normalize_dcpf_shape(lambda_sol["lambda_dcpf_upper"], nl, T)
    lambda_dl = _normalize_dcpf_shape(lambda_sol["lambda_dcpf_lower"], nl, T)

    congestion = lambda_du - lambda_dl
    effective = np.zeros((ng, T), dtype=float)
    for t in range(T):
        effective[:, t] = lambda_pb[t] - sensitivity.T @ congestion[:, t]

    return effective, "global_uc_lp_dual"


def _resolve_effective_lambda_from_manual_ed(
    ppc,
    sample: dict,
    x_true: np.ndarray,
    T_delta: float,
) -> tuple[np.ndarray, str, dict]:
    effective, payload = _solve_ed_pg_effective_lambda(
        ppc,
        sample,
        x_true,
        T_delta,
        verbose=False,
    )
    return effective, "manual_ed_pg_dual", payload


def _run_single_price_mode(
    *,
    ppc,
    x_true: np.ndarray,
    lambda_eff: np.ndarray,
    case_name: str,
    active_set_json: Path,
    sample_id: int,
    x_source: str,
    lambda_source: str,
    t_delta: float,
    verbose_subproblem: bool,
    lambda_payload: dict | None = None,
) -> int:
    ng, T = x_true.shape
    if lambda_eff.shape != (ng, T):
        raise ValueError(f"effective lambda shape mismatch: got {lambda_eff.shape}, expected {(ng, T)}")

    x_sub = np.zeros((ng, T), dtype=float)
    unit_hamming = []
    total_subproblem_obj = 0.0

    print("=" * 72)
    print(f"Case: {case_name}")
    print(f"Active set JSON: {active_set_json}")
    print(f"Sample: {sample_id}")
    print(f"x_true source: {x_source}")
    print(f"lambda source: {lambda_source}")
    print("=" * 72)
    print("Effective lambda by unit:")
    for g in range(ng):
        lam = np.asarray(lambda_eff[g], dtype=float)
        print(
            f"  Unit {g:02d}: min={lam.min():.6f}  max={lam.max():.6f}  "
            f"mean={lam.mean():.6f}"
        )
        print(f"    {np.round(lam, 6).tolist()}")
        if lambda_payload is not None and "components" in lambda_payload:
            comps = lambda_payload["components"]
            print(
                "    pb/ramp/flow/pgL/pgU mean="
                f"{np.mean(comps['lambda_power_balance'][g]):.6f}/"
                f"{np.mean(comps['lambda_ramp_up'][g] + comps['lambda_ramp_down'][g]):.6f}/"
                f"{np.mean(comps['lambda_flow'][g]):.6f}/"
                f"{np.mean(comps['lambda_pg_lower'][g]):.6f}/"
                f"{np.mean(comps['lambda_pg_upper'][g]):.6f}"
            )
    print("-" * 72)

    for g in range(ng):
        result = _solve_single_unit_milp(
            ppc,
            g,
            lambda_eff[g],
            t_delta,
            verbose=verbose_subproblem,
        )
        x_unit = np.rint(result["x"]).astype(int)
        fixed_true = _evaluate_single_unit_fixed_commitment(
            ppc,
            g,
            lambda_eff[g],
            np.rint(x_true[g]).astype(int),
            t_delta,
            verbose=False,
        )
        x_sub[g] = x_unit
        total_subproblem_obj += result["objective"]
        mismatch_slots = np.flatnonzero(x_unit != np.rint(x_true[g]).astype(int))
        unit_hamming.append(int(mismatch_slots.size))

        mismatch_text = ",".join(str(int(t)) for t in mismatch_slots[:12])
        if mismatch_slots.size > 12:
            mismatch_text += ",..."
        if mismatch_slots.size == 0:
            mismatch_text = "-"

        print(
            f"Unit {g:02d}: hamming={mismatch_slots.size:2d}  "
            f"sub_obj={result['objective']:12.6f}  "
            f"x_true_obj={fixed_true['objective']:12.6f}  "
            f"mismatch_t={mismatch_text}"
        )

    x_true_bin = np.rint(x_true).astype(int)
    x_sub_bin = np.rint(x_sub).astype(int)
    total_hamming = int(np.sum(x_true_bin != x_sub_bin))
    matched_units = int(sum(val == 0 for val in unit_hamming))

    print("-" * 72)
    print(f"Matched units: {matched_units}/{ng}")
    print(f"Total Hamming distance: {total_hamming}")
    print(f"Sum of unit subproblem objectives: {total_subproblem_obj:.6f}")

    if total_hamming == 0:
        print("Result: exact commitment match")
        return 0

    mismatches = np.argwhere(x_true_bin != x_sub_bin)
    preview = ", ".join(f"(g={int(g)},t={int(t)})" for g, t in mismatches[:20])
    if mismatches.shape[0] > 20:
        preview += ", ..."
    print(f"Result: mismatch detected at {preview}")
    return 1


def run_check(
    case_name: str = CASE_NAME,
    active_set_json: str | Path | None = ACTIVE_SET_JSON,
    sample_id: int = SAMPLE_ID,
    t_delta: float = T_DELTA,
    solve_global_uc: bool = SOLVE_GLOBAL_UC,
    verbose_subproblem: bool = VERBOSE_SUBPROBLEM,
) -> int:
    resolved_active_set_json = _resolve_latest_active_set_json(case_name, active_set_json)

    ppc = get_case_ppc(case_name)
    sample = _load_sample(resolved_active_set_json, sample_id)
    ppc_int = ext2int(ppc)
    ng = ppc_int["gen"].shape[0]
    T = sample["pd_data"].shape[1]

    x_true, x_source = _resolve_x_true(ppc, sample, t_delta, solve_global_uc)
    if x_true.shape != (ng, T):
        raise ValueError(f"x_true shape mismatch: got {x_true.shape}, expected {(ng, T)}")

    rc = 0

    if "sample_lambda" in PRICE_MODES:
        lambda_eff, lambda_source = _resolve_effective_lambda(ppc, sample, x_true, t_delta)
        rc = max(
            rc,
            _run_single_price_mode(
                ppc=ppc,
                x_true=x_true,
                lambda_eff=lambda_eff,
                case_name=case_name,
                active_set_json=resolved_active_set_json,
                sample_id=sample_id,
                x_source=x_source,
                lambda_source=lambda_source,
                t_delta=t_delta,
                verbose_subproblem=verbose_subproblem,
            ),
        )

    if "manual_ed_pg_dual" in PRICE_MODES:
        lambda_eff, lambda_source, lambda_payload = _resolve_effective_lambda_from_manual_ed(
            ppc,
            sample,
            x_true,
            t_delta,
        )
        rc = max(
            rc,
            _run_single_price_mode(
                ppc=ppc,
                x_true=x_true,
                lambda_eff=lambda_eff,
                case_name=case_name,
                active_set_json=resolved_active_set_json,
                sample_id=sample_id,
                x_source=x_source,
                lambda_source=lambda_source,
                t_delta=t_delta,
                verbose_subproblem=verbose_subproblem,
                lambda_payload=lambda_payload,
            ),
        )

    if "global_uc_lp_dual" in PRICE_MODES:
        lambda_eff, lambda_source = _resolve_effective_lambda_from_global_uc_lp(
            ppc,
            sample,
            t_delta,
        )
        rc = max(
            rc,
            _run_single_price_mode(
                ppc=ppc,
                x_true=x_true,
                lambda_eff=lambda_eff,
                case_name=case_name,
                active_set_json=resolved_active_set_json,
                sample_id=sample_id,
                x_source=x_source,
                lambda_source=lambda_source,
                t_delta=t_delta,
                verbose_subproblem=verbose_subproblem,
                lambda_payload=None,
            ),
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(run_check())
