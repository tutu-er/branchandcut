#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze whether x_opt is optimal for the pure-BCD penalized LP used in run_test.

For each sample, the script solves the same penalized LP as
Agent_NN_BCD.solve_LP_with_theta_constraints(sample_id), then resolves the same
problem with x fixed to x_opt. It reports:

- economic cost
- proxy penalty
- total objective = cost + penalty_factor * proxy_penalty
- L1 / Hamming distance between the free solve and x_opt

This answers whether x_opt is optimal for the *current BCD test problem*,
rather than for a harder "all proxy constraints must be exact" problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pypower.case14
import pypower.case30
import pypower.case39
import pypower.case118
from pypower.idx_gen import PMIN, PMAX, GEN_BUS
from pypower.idx_brch import RATE_A
from pypower.makePTDF import makePTDF

try:
    from src.uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json
except ModuleNotFoundError:
    from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json


DEFAULT_CASE = "case30"
DEFAULT_DATA = "result/active_set/active_sets_case30_T24_n340_20260317_152540.json"
DEFAULT_BCD_MODEL = "result/bcd_models/bcd_model_case30_20260322_150043.pth"
DEFAULT_T_DELTA = 1.0
DEFAULT_TOL = 1e-6
DEFAULT_NUM_SAMPLES = 20


def log(msg: str) -> None:
    print(msg, flush=True)


def load_case(case_name: str):
    if case_name == "case14":
        return pypower.case14.case14()
    if case_name == "case30":
        return pypower.case30.case30()
    if case_name == "case39":
        return pypower.case39.case39()
    if case_name == "case118":
        return pypower.case118.case118()
    raise ValueError(f"unsupported case: {case_name}")


def infer_theta_zeta_for_sample(agent: Agent_NN_BCD, sample_id: int) -> tuple[dict, dict]:
    import torch

    features = agent._extract_features(sample_id)
    features_tensor = torch.tensor(np.asarray(features), dtype=torch.float32).unsqueeze(0)
    if getattr(agent, "device", None) is not None:
        features_tensor = features_tensor.to(agent.device)

    theta_was_training = agent.theta_net.training if agent.theta_net is not None else False
    zeta_was_training = agent.zeta_net.training if agent.zeta_net is not None else False
    if agent.theta_net is not None:
        agent.theta_net.eval()
    if agent.zeta_net is not None:
        agent.zeta_net.eval()

    with torch.no_grad():
        theta_out = agent.theta_net(features_tensor)
        zeta_out = agent.zeta_net(features_tensor)

    if agent.theta_net is not None and theta_was_training:
        agent.theta_net.train()
    if agent.zeta_net is not None and zeta_was_training:
        agent.zeta_net.train()

    theta = {name: float(val) for name, val in zip(agent.theta_var_names, theta_out.detach().cpu().numpy().flatten())}
    zeta = {name: float(val) for name, val in zip(agent.zeta_var_names, zeta_out.detach().cpu().numpy().flatten())}
    return theta, zeta


def build_bcd_penalized_model(
    agent: Agent_NN_BCD,
    sample_id: int,
    x_fixed: np.ndarray | None = None,
):
    Pd = np.asarray(agent.active_set_data[sample_id]["pd_data"], dtype=float)
    union_analysis = agent._current_union_analysis
    theta, zeta = infer_theta_zeta_for_sample(agent, sample_id)

    model = gp.Model(f"bcd_penalized_{sample_id}")
    model.Params.OutputFlag = 0
    model.Params.DualReductions = 0
    model.Params.MIPGap = 1e-10

    pg = model.addVars(agent.ng, agent.T, lb=0, name="pg")
    x = model.addVars(agent.ng, agent.T, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    coc = model.addVars(agent.ng, agent.T - 1, lb=0, name="coc")
    cpower = model.addVars(agent.ng, agent.T, lb=0, name="cpower")

    for t in range(agent.T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(agent.ng)) == float(np.sum(Pd[:, t])),
            name=f"power_balance_{t}",
        )
        for g in range(agent.ng):
            model.addConstr(agent.gen[g, PMIN] * x[g, t] - pg[g, t] <= 0, name=f"pg_lower_{g}_{t}")
            model.addConstr(pg[g, t] - agent.gen[g, PMAX] * x[g, t] <= 0, name=f"pg_upper_{g}_{t}")

    Ru = 0.4 * agent.gen[:, PMAX] / agent.T_delta
    Rd = 0.4 * agent.gen[:, PMAX] / agent.T_delta
    Ru_co = 0.3 * agent.gen[:, PMAX]
    Rd_co = 0.3 * agent.gen[:, PMAX]
    for t in range(1, agent.T):
        for g in range(agent.ng):
            model.addConstr(
                pg[g, t] - pg[g, t - 1] - Ru[g] * x[g, t - 1] - Ru_co[g] * (1 - x[g, t - 1]) <= 0,
                name=f"ramp_up_{g}_{t}",
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t]) <= 0,
                name=f"ramp_down_{g}_{t}",
            )

    Ton = min(4, agent.T)
    Toff = min(4, agent.T)
    for g in range(agent.ng):
        for tau in range(1, Ton + 1):
            for t1 in range(agent.T - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] - x[g, t1 + tau] <= 0, name=f"min_on_{g}_{tau}_{t1}")
        for tau in range(1, Toff + 1):
            for t1 in range(agent.T - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] - (1 - x[g, t1 + tau]) <= 0, name=f"min_off_{g}_{tau}_{t1}")

    obj_cost = gp.LinExpr()
    start_cost = agent.gencost[:, 1]
    shut_cost = agent.gencost[:, 2]
    for t in range(1, agent.T):
        for g in range(agent.ng):
            model.addConstr(start_cost[g] * (x[g, t] - x[g, t - 1]) <= coc[g, t - 1], name=f"start_cost_{g}_{t}")
            model.addConstr(shut_cost[g] * (x[g, t - 1] - x[g, t]) <= coc[g, t - 1], name=f"shut_cost_{g}_{t}")
            obj_cost += coc[g, t - 1]

    for t in range(agent.T):
        for g in range(agent.ng):
            model.addConstr(
                cpower[g, t] == agent.gencost[g, -2] / agent.T_delta * pg[g, t] + agent.gencost[g, -1] / agent.T_delta * x[g, t],
                name=f"cpower_{g}_{t}",
            )
            obj_cost += cpower[g, t]

    nb = agent.bus.shape[0]
    G = np.zeros((nb, agent.ng))
    for g in range(agent.ng):
        bus_idx = int(agent.gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1
    PTDF = makePTDF(agent.baseMVA, agent.bus, agent.branch)
    branch_limit = agent.branch[:, RATE_A]
    for t in range(agent.T):
        flow = PTDF @ (G @ np.array([pg[g, t] for g in range(agent.ng)]) - Pd[:, t])
        for l in range(agent.branch.shape[0]):
            model.addConstr(flow[l] - branch_limit[l] <= 0, name=f"dcpf_upper_{l}_{t}")
            model.addConstr(-flow[l] - branch_limit[l] <= 0, name=f"dcpf_lower_{l}_{t}")

    if x_fixed is not None:
        for g in range(agent.ng):
            for t in range(agent.T):
                model.addConstr(x[g, t] == float(x_fixed[g, t]), name=f"x_fix_{g}_{t}")

    model, theta_penalty, _ = agent._add_parametric_penalties_pg_block(
        model, x, sample_id, theta, union_analysis, PTDF=PTDF, branch_limit=branch_limit
    )
    model, zeta_penalty, _ = agent._add_parametric_balance_power_penalties_pg_block(
        model, x, sample_id, zeta, union_analysis
    )
    obj_penalty = theta_penalty + zeta_penalty
    obj_total = obj_cost + agent.penalty_factor * obj_penalty
    model.setObjective(obj_total, GRB.MINIMIZE)

    return model, x, obj_cost, obj_penalty, obj_total


def solve_bcd_penalized_problem(
    agent: Agent_NN_BCD,
    sample_id: int,
    x_fixed: np.ndarray | None = None,
):
    model, x, obj_cost, obj_penalty, _ = build_bcd_penalized_model(agent, sample_id, x_fixed=x_fixed)
    model.optimize()

    report = {"gurobi_status": int(model.status)}
    if model.status != GRB.OPTIMAL:
        return report

    x_sol = np.array([[x[g, t].X for t in range(agent.T)] for g in range(agent.ng)])
    cost_val = float(obj_cost.getValue())
    penalty_raw = float(obj_penalty.getValue())
    penalty_scaled = float(agent.penalty_factor * penalty_raw)
    total_val = float(model.ObjVal)
    report.update(
        {
            "x": x_sol,
            "cost": cost_val,
            "penalty_raw": penalty_raw,
            "penalty_scaled": penalty_scaled,
            "total": total_val,
        }
    )
    return report


def analyze_sample(agent: Agent_NN_BCD, sample_id: int, tol: float) -> dict:
    x_opt = np.asarray(agent.x_opt[sample_id], dtype=float)
    free_report = solve_bcd_penalized_problem(agent, sample_id)
    if free_report["gurobi_status"] != int(GRB.OPTIMAL):
        return {
            "status": "free_problem_not_optimal",
            "gurobi_status": free_report["gurobi_status"],
        }

    fixed_report = solve_bcd_penalized_problem(agent, sample_id, x_fixed=x_opt)
    if fixed_report["gurobi_status"] != int(GRB.OPTIMAL):
        return {
            "status": "x_opt_infeasible_for_bcd_penalized_problem",
            "gurobi_status_fix": fixed_report["gurobi_status"],
            "free": free_report,
        }

    l1 = float(np.sum(np.abs(free_report["x"] - x_opt)))
    hamming = int(np.sum(np.round(free_report["x"]).astype(int) != x_opt.astype(int)))
    total_gap = fixed_report["total"] - free_report["total"]
    scale = max(1.0, abs(free_report["total"]), abs(fixed_report["total"]))
    x_opt_is_optimal = abs(total_gap) <= tol * scale

    return {
        "status": "ok",
        "x_opt_is_optimal": bool(x_opt_is_optimal),
        "total_gap": float(total_gap),
        "cost_gap": float(fixed_report["cost"] - free_report["cost"]),
        "penalty_raw_gap": float(fixed_report["penalty_raw"] - free_report["penalty_raw"]),
        "penalty_scaled_gap": float(fixed_report["penalty_scaled"] - free_report["penalty_scaled"]),
        "l1_free_vs_xopt": l1,
        "hamming_free_vs_xopt": hamming,
        "free": free_report,
        "fixed_xopt": fixed_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze x_opt under the pure-BCD penalized LP used in run_test.")
    parser.add_argument("--case", default=DEFAULT_CASE, choices=["case14", "case30", "case39", "case118"])
    parser.add_argument("--data", default=DEFAULT_DATA, help="BCD active_set JSON path")
    parser.add_argument("--bcd-model", default=DEFAULT_BCD_MODEL, help="BCD model .pth path")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--t-delta", type=float, default=DEFAULT_T_DELTA)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_path = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    bcd_path = (ROOT / args.bcd_model).resolve() if not Path(args.bcd_model).is_absolute() else Path(args.bcd_model)

    ppc = load_case(args.case)
    active_set_data = load_active_set_from_json(str(data_path))
    n_check = min(args.num_samples, len(active_set_data))
    if n_check <= 0:
        raise ValueError("num-samples must be positive")
    active_set_data = active_set_data[:n_check]

    log(f"loading BCD model from: {bcd_path}")
    agent = Agent_NN_BCD(ppc, active_set_data, args.t_delta)
    agent.load_model_parameters(str(bcd_path))

    summary = {
        "ok": 0,
        "x_opt_is_optimal": 0,
        "x_opt_not_optimal": 0,
        "x_opt_infeasible": 0,
        "other_status": 0,
        "mean_l1": [],
        "mean_hamming": [],
        "mean_penalty_raw": [],
    }

    print("\n" + "=" * 72)
    print(f"analyzing first {n_check} samples for pure BCD penalized LP")
    print("=" * 72)

    for sample_id in range(n_check):
        report = analyze_sample(agent, sample_id, args.tol)
        print("\n" + "-" * 72)
        print(f"sample_id: {sample_id}")
        print(f"status: {report['status']}")

        if report["status"] != "ok":
            if report["status"] == "x_opt_infeasible_for_bcd_penalized_problem":
                summary["x_opt_infeasible"] += 1
            else:
                summary["other_status"] += 1
            for key, value in report.items():
                if key != "status":
                    print(f"{key}: {value}")
            continue

        summary["ok"] += 1
        if report["x_opt_is_optimal"]:
            summary["x_opt_is_optimal"] += 1
        else:
            summary["x_opt_not_optimal"] += 1
        summary["mean_l1"].append(report["l1_free_vs_xopt"])
        summary["mean_hamming"].append(report["hamming_free_vs_xopt"])
        summary["mean_penalty_raw"].append(report["free"]["penalty_raw"])

        free = report["free"]
        fixed = report["fixed_xopt"]
        print(f"x_opt_is_optimal: {report['x_opt_is_optimal']}")
        print(f"L1(free, x_opt): {report['l1_free_vs_xopt']:.10f}")
        print(f"Hamming(free, x_opt): {report['hamming_free_vs_xopt']}")
        print(f"free_cost        : {free['cost']:.10f}")
        print(f"free_penalty_raw : {free['penalty_raw']:.10f}")
        print(f"free_penalty     : {free['penalty_scaled']:.10f}")
        print(f"free_total       : {free['total']:.10f}")
        print(f"xopt_cost        : {fixed['cost']:.10f}")
        print(f"xopt_penalty_raw : {fixed['penalty_raw']:.10f}")
        print(f"xopt_penalty     : {fixed['penalty_scaled']:.10f}")
        print(f"xopt_total       : {fixed['total']:.10f}")
        print(f"cost_gap         : {report['cost_gap']:.10f}")
        print(f"penalty_raw_gap  : {report['penalty_raw_gap']:.10f}")
        print(f"penalty_gap      : {report['penalty_scaled_gap']:.10f}")
        print(f"total_gap        : {report['total_gap']:.10f}")

    print("\n" + "=" * 72)
    print("summary")
    print(f"checked_samples: {n_check}")
    print(f"ok: {summary['ok']}")
    print(f"x_opt_is_optimal: {summary['x_opt_is_optimal']}")
    print(f"x_opt_not_optimal: {summary['x_opt_not_optimal']}")
    print(f"x_opt_infeasible: {summary['x_opt_infeasible']}")
    print(f"other_status: {summary['other_status']}")
    if summary["mean_l1"]:
        print(f"mean_L1(free, x_opt): {float(np.mean(summary['mean_l1'])):.10f}")
        print(f"mean_Hamming(free, x_opt): {float(np.mean(summary['mean_hamming'])):.4f}")
        print(f"mean_free_penalty_raw: {float(np.mean(summary['mean_penalty_raw'])):.10f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
