#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
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
    from src.mti118_data_loader import load_case118_ppc_with_mti_limits
except ModuleNotFoundError:
    from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json
    from mti118_data_loader import load_case118_ppc_with_mti_limits


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
        return load_case118_ppc_with_mti_limits()
    raise ValueError(f"unsupported case: {case_name}")


def load_theta_cache(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_theta_zeta(agent: Agent_NN_BCD, sample_id: int) -> tuple[dict, dict]:
    import torch

    features = agent._extract_features(sample_id)
    x = torch.tensor(np.asarray(features), dtype=torch.float32).unsqueeze(0)
    if getattr(agent, "device", None) is not None:
        x = x.to(agent.device)

    theta_was_training = agent.theta_net.training if agent.theta_net is not None else False
    zeta_was_training = agent.zeta_net.training if agent.zeta_net is not None else False

    if agent.theta_net is not None:
        agent.theta_net.eval()
    if agent.zeta_net is not None:
        agent.zeta_net.eval()

    with torch.no_grad():
        theta_out = agent.theta_net(x)
        zeta_out = agent.zeta_net(x)

    if agent.theta_net is not None and theta_was_training:
        agent.theta_net.train()
    if agent.zeta_net is not None and zeta_was_training:
        agent.zeta_net.train()

    theta = {name: float(val) for name, val in zip(agent.theta_var_names, theta_out.detach().cpu().numpy().flatten())}
    zeta = {name: float(val) for name, val in zip(agent.zeta_var_names, zeta_out.detach().cpu().numpy().flatten())}
    return theta, zeta


def dict_diff_stats(a: dict, b: dict) -> dict:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return {"max_abs": 0.0, "mean_abs": 0.0, "l2": 0.0, "num_keys": 0}
    diffs = np.array([abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys], dtype=float)
    return {
        "max_abs": float(np.max(diffs)),
        "mean_abs": float(np.mean(diffs)),
        "l2": float(np.linalg.norm(diffs)),
        "num_keys": int(len(keys)),
    }


def compute_proxy_penalty_for_fixed_x(
    agent: Agent_NN_BCD,
    sample_id: int,
    x_fixed: np.ndarray,
    theta_values: dict,
    zeta_values: dict,
) -> dict:
    Pd = np.asarray(agent.active_set_data[sample_id]["pd_data"], dtype=float)
    union_analysis = agent._current_union_analysis

    model = gp.Model(f"penalty_eval_{sample_id}")
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
            model.addConstr(agent.gen[g, PMIN] * x[g, t] - pg[g, t] <= 0)
            model.addConstr(pg[g, t] - agent.gen[g, PMAX] * x[g, t] <= 0)

    Ru = 0.4 * agent.gen[:, PMAX] / agent.T_delta
    Rd = 0.4 * agent.gen[:, PMAX] / agent.T_delta
    Ru_co = 0.3 * agent.gen[:, PMAX]
    Rd_co = 0.3 * agent.gen[:, PMAX]
    for t in range(1, agent.T):
        for g in range(agent.ng):
            model.addConstr(pg[g, t] - pg[g, t - 1] - Ru[g] * x[g, t - 1] - Ru_co[g] * (1 - x[g, t - 1]) <= 0)
            model.addConstr(pg[g, t - 1] - pg[g, t] - Rd[g] * x[g, t] - Rd_co[g] * (1 - x[g, t]) <= 0)

    Ton = min(4, agent.T)
    Toff = min(4, agent.T)
    for g in range(agent.ng):
        for tau in range(1, Ton + 1):
            for t1 in range(agent.T - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] - x[g, t1 + tau] <= 0)
        for tau in range(1, Toff + 1):
            for t1 in range(agent.T - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] - (1 - x[g, t1 + tau]) <= 0)

    start_cost = agent.gencost[:, 1]
    shut_cost = agent.gencost[:, 2]
    obj_cost = gp.LinExpr()
    for t in range(1, agent.T):
        for g in range(agent.ng):
            model.addConstr(start_cost[g] * (x[g, t] - x[g, t - 1]) <= coc[g, t - 1])
            model.addConstr(shut_cost[g] * (x[g, t - 1] - x[g, t]) <= coc[g, t - 1])
            obj_cost += coc[g, t - 1]

    for t in range(agent.T):
        for g in range(agent.ng):
            model.addConstr(
                cpower[g, t] == agent.gencost[g, -2] / agent.T_delta * pg[g, t] + agent.gencost[g, -1] / agent.T_delta * x[g, t]
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
            model.addConstr(flow[l] - branch_limit[l] <= 0)
            model.addConstr(-flow[l] - branch_limit[l] <= 0)

    for g in range(agent.ng):
        for t in range(agent.T):
            model.addConstr(x[g, t] == float(x_fixed[g, t]), name=f"x_fix_{g}_{t}")

    model, theta_penalty, _ = agent._add_parametric_penalties_pg_block(
        model, x, sample_id, theta_values, union_analysis, PTDF=PTDF, branch_limit=branch_limit
    )
    model, zeta_penalty, _ = agent._add_parametric_balance_power_penalties_pg_block(
        model, x, sample_id, zeta_values, union_analysis
    )

    total_penalty = theta_penalty + zeta_penalty
    model.setObjective(obj_cost + agent.penalty_factor * total_penalty, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return {"status": int(model.status)}

    theta_val = float(theta_penalty.getValue())
    zeta_val = float(zeta_penalty.getValue())
    total_val = float(total_penalty.getValue())
    return {
        "status": int(model.status),
        "theta_penalty_raw": theta_val,
        "zeta_penalty_raw": zeta_val,
        "total_penalty_raw": total_val,
        "scaled_total_penalty": float(agent.penalty_factor * total_val),
        "total_obj": float(model.ObjVal),
    }


def main() -> None:
    CASE_NAME = "case30"
    DATA_PATH = ROOT / "result/active_set/active_sets_case30_T24_n340_20260317_152540.json"
    BCD_MODEL_PATH = ROOT / "result/bcd_models/bcd_model_case30_20260322_150043.pth"
    THETA_JSON_PATH = ROOT / "result/theta_zeta/theta_zeta_values_list_case30_20260322_150043.json"
    T_DELTA = 1.0
    NUM_SAMPLES = 20

    ppc = load_case(CASE_NAME)
    all_samples = load_active_set_from_json(str(DATA_PATH))
    n_check = min(NUM_SAMPLES, len(all_samples))
    all_samples = all_samples[:n_check]

    cache = load_theta_cache(THETA_JSON_PATH)
    theta_values_list_cache = cache["theta_values_list"][:n_check]
    zeta_values_list_cache = cache["zeta_values_list"][:n_check]

    log(f"loading BCD model from: {BCD_MODEL_PATH}")
    agent = Agent_NN_BCD(ppc, all_samples, T_DELTA)
    agent.load_model_parameters(str(BCD_MODEL_PATH))

    theta_max_list = []
    zeta_max_list = []
    cache_penalty_list = []
    forward_penalty_list = []

    print("\n" + "=" * 72)
    print(f"compare cache vs forward on first {n_check} samples")
    print("=" * 72)

    for sample_id in range(n_check):
        x_opt = np.asarray(agent.x_opt[sample_id], dtype=float)

        theta_cache = theta_values_list_cache[sample_id]
        zeta_cache = zeta_values_list_cache[sample_id]
        theta_forward, zeta_forward = infer_theta_zeta(agent, sample_id)

        theta_diff = dict_diff_stats(theta_cache, theta_forward)
        zeta_diff = dict_diff_stats(zeta_cache, zeta_forward)

        cache_pen = compute_proxy_penalty_for_fixed_x(
            agent, sample_id, x_opt, theta_cache, zeta_cache
        )
        forward_pen = compute_proxy_penalty_for_fixed_x(
            agent, sample_id, x_opt, theta_forward, zeta_forward
        )

        theta_max_list.append(theta_diff["max_abs"])
        zeta_max_list.append(zeta_diff["max_abs"])
        if cache_pen["status"] == int(GRB.OPTIMAL):
            cache_penalty_list.append(cache_pen["total_penalty_raw"])
        if forward_pen["status"] == int(GRB.OPTIMAL):
            forward_penalty_list.append(forward_pen["total_penalty_raw"])

        print("\n" + "-" * 72)
        print(f"sample_id: {sample_id}")
        print(f"theta_diff_max_abs : {theta_diff['max_abs']:.10f}")
        print(f"theta_diff_mean_abs: {theta_diff['mean_abs']:.10f}")
        print(f"zeta_diff_max_abs  : {zeta_diff['max_abs']:.10f}")
        print(f"zeta_diff_mean_abs : {zeta_diff['mean_abs']:.10f}")

        print(f"cache_penalty_status    : {cache_pen['status']}")
        if cache_pen["status"] == int(GRB.OPTIMAL):
            print(f"cache_theta_penalty_raw: {cache_pen['theta_penalty_raw']:.10f}")
            print(f"cache_zeta_penalty_raw : {cache_pen['zeta_penalty_raw']:.10f}")
            print(f"cache_total_penalty_raw: {cache_pen['total_penalty_raw']:.10f}")

        print(f"forward_penalty_status    : {forward_pen['status']}")
        if forward_pen["status"] == int(GRB.OPTIMAL):
            print(f"forward_theta_penalty_raw: {forward_pen['theta_penalty_raw']:.10f}")
            print(f"forward_zeta_penalty_raw : {forward_pen['zeta_penalty_raw']:.10f}")
            print(f"forward_total_penalty_raw: {forward_pen['total_penalty_raw']:.10f}")

    print("\n" + "=" * 72)
    print("summary")
    print(f"checked_samples: {n_check}")
    if theta_max_list:
        print(f"mean_theta_diff_max_abs: {float(np.mean(theta_max_list)):.10f}")
        print(f"max_theta_diff_max_abs : {float(np.max(theta_max_list)):.10f}")
    if zeta_max_list:
        print(f"mean_zeta_diff_max_abs : {float(np.mean(zeta_max_list)):.10f}")
        print(f"max_zeta_diff_max_abs  : {float(np.max(zeta_max_list)):.10f}")
    if cache_penalty_list:
        print(f"mean_cache_total_penalty_raw  : {float(np.mean(cache_penalty_list)):.10f}")
    if forward_penalty_list:
        print(f"mean_forward_total_penalty_raw: {float(np.mean(forward_penalty_list)):.10f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
