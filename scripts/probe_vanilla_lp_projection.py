#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal reproducer for the *vanilla FP* LP projection infeasibility.

  We deliberately skip every learned component (BCD agent, surrogate trainers,
  nearby-commitment library) and instead:

    1.  Build the plain UC LP relaxation (the SAME constraint set the FP
        projection uses) for a single sample.
    2.  Round its LP solution to get ``x_init``.
    3.  Run ``run_feasibility_pump`` directly with ``x_curr = x_init``,
        ``trusted_mask`` all-False, ``x_pool = None``, no surrogates,
        ``stall_perturbation_mode='flip'``, max_iter=5,
        ``dump_iis_on_projection_failure=True``.

  When the LP projection reports ``model.status != OPTIMAL`` the IIS is
  printed so we know exactly which constraints are in conflict.  If the IIS
  contains DC line constraints together with ramp / min-up-down, the LP
  relaxation truly has an empty feasible set for that sample (a structural
  property of the case + load profile).  In that case the "traditional FP
  shouldn't infeasible" statement is correct only when DC limits are absent.

Usage::

    python scripts/probe_vanilla_lp_projection.py \\
        --case case14 \\
        --active-sets result/active_set/active_sets_case14_T24_n600_20260503_222929.json \\
        --sample-ids 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402
from pypower.ext2int import ext2int  # noqa: E402
from pypower.idx_gen import PMAX, PMIN  # noqa: E402

from scripts.collect_model_fp_diagnostics import (  # noqa: E402
    _ensure_runtime_imports,
    _latest_active_set,
    _resolve_path,
)
from src.feasibility_pump import (  # noqa: E402
    _build_ptdf_data,
    _get_min_up_down_time_steps,
    _get_ramp_limits_from_ppc,
    check_uc_feasibility,
    run_feasibility_pump,
)


def _build_lp_relaxation(ppc: dict, pd_data: np.ndarray, T_delta: float):
    """Plain UC LP relaxation (no surrogates) — same constraint set as FP."""
    ppc_int = ext2int(ppc)
    gen = ppc_int['gen']
    gencost = ppc_int['gencost']
    ng, T = gen.shape[0], pd_data.shape[1]
    Pd_sum = np.sum(pd_data, axis=0)

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, T_delta, T)

    model = gp.Model('probe_lp_relax')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    x = model.addVars(ng, T, lb=0, ub=1, name='x')
    cpower = model.addVars(ng, T, lb=0, name='cpower')
    coc = model.addVars(ng, max(T - 1, 0), lb=0, name='coc')

    for t in range(T):
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]),
            name=f'pb_{t}',
        )
    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= float(gen[g, PMIN]) * x[g, t], name=f'pmin_{g}_{t}')
            model.addConstr(pg[g, t] <= float(gen[g, PMAX]) * x[g, t], name=f'pmax_{g}_{t}')
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                              + gencost[g, -1] / T_delta * x[g, t],
                name=f'fuelc_{g}_{t}',
            )
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1] <= Ru[g] * x[g, t-1] + Ru_co[g] * (1 - x[g, t-1]),
                name=f'rampup_{g}_{t}',
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t] <= Rd[g] * x[g, t] + Rd_co[g] * (1 - x[g, t]),
                name=f'rampdn_{g}_{t}',
            )
            model.addConstr(
                coc[g, t-1] >= float(start_cost[g]) * (x[g, t] - x[g, t-1]),
                name=f'startc_{g}_{t}',
            )
            model.addConstr(
                coc[g, t-1] >= float(shut_cost[g]) * (x[g, t-1] - x[g, t]),
                name=f'shutc_{g}_{t}',
            )

        Ton_g = int(min_up_steps[g])
        Toff_g = int(min_down_steps[g])
        for tau in range(1, Ton_g + 1):
            for t1 in range(T - tau):
                model.addConstr(
                    x[g, t1+1] - x[g, t1] <= x[g, t1+tau],
                    name=f'minup_{g}_{t1}_{tau}',
                )
        for tau in range(1, Toff_g + 1):
            for t1 in range(T - tau):
                model.addConstr(
                    -x[g, t1+1] + x[g, t1] <= 1 - x[g, t1+tau],
                    name=f'mindn_{g}_{t1}_{tau}',
                )

    dc_added = False
    try:
        PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = PTDF @ pd_data
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit, name=f'dc_p_{l}_{t}')
                model.addConstr(flow_expr >= -limit, name=f'dc_m_{l}_{t}')
        dc_added = True
    except Exception as exc:
        print(f"  [probe] DC line constraints skipped: {exc}")

    obj = (
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(max(T - 1, 0)))
    )
    model.setObjective(obj, GRB.MINIMIZE)
    return model, x, pg, dc_added, ng, T


def _round_to_int(x_lp: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x_lp), 0, 1).astype(int)


def _parse_sample_ids(text: str) -> List[int]:
    parts = [p.strip() for p in str(text).split(',') if p.strip()]
    return [int(p) for p in parts]


def _probe_one_sample(sample: dict, ppc: dict, T_delta: float,
                      max_iter: int) -> None:
    from scripts.collect_model_fp_diagnostics import _extract_true_solution
    from scripts import collect_model_fp_diagnostics as cdiag
    pd_data = cdiag.get_sample_net_load(sample)
    ng = int(np.asarray(ppc['gen']).shape[0])
    T = int(pd_data.shape[1])

    sid = sample.get('sample_id', sample.get('source_sample_id'))
    print(f"\n========== probing sample_id={sid} (ng x T = {ng} x {T}) ==========")

    print("  [step 1] solve plain UC LP relaxation ...")
    model_lp, x_var, _pg_var, dc_added, _, _ = _build_lp_relaxation(ppc, pd_data, T_delta)
    model_lp.optimize()
    print(f"    LP relax status = {model_lp.status} (DC={'on' if dc_added else 'off'})")
    if model_lp.status != GRB.OPTIMAL:
        print(f"    ** LP relaxation itself is infeasible -- constraint set is structurally empty **")
        try:
            model_lp.computeIIS()
            iis_cons = [c.ConstrName for c in model_lp.getConstrs() if c.IISConstr][:30]
            print(f"    IIS sample (up to 30): {iis_cons}")
        except gp.GurobiError as exc:
            print(f"    IIS computation failed: {exc}")
        return

    x_lp = np.array([[x_var[g, t].X for t in range(T)] for g in range(ng)])
    x_init = _round_to_int(x_lp)
    pre_feas, pre_reason = check_uc_feasibility(x_init, ppc, pd_data, T_delta)
    print(f"  [step 2] x_init = round(x_LP); check_uc_feasibility = {pre_feas}  reason={pre_reason!r}")

    print(f"  [step 3] run_feasibility_pump with vanilla settings, max_iter={max_iter}, IIS dump on ...")
    trusted = np.zeros((ng, T), dtype=bool)
    rng = np.random.default_rng(42)
    x_out, ok, details = run_feasibility_pump(
        x_init, trusted, ppc, pd_data, T_delta,
        x_pool=None,
        surrogate_screen_constraints=None,
        surrogate_screen_soft_penalty=0.0,
        projection_objective_tau='none',
        max_iter=max_iter,
        stall_perturbation_mode='flip',
        stall_flip_fraction=0.10,
        rng=rng,
        verbose=True,
        return_history=True,
        rounding_strategy='x_round',
        dump_iis_on_projection_failure=True,
    )
    print(f"  [step 4] run_feasibility_pump done. success={ok}")
    history = details.get('history') or []
    print(f"    history size = {len(history)}")
    for i, h in enumerate(history):
        status = h.get('projection_status_name', '?')
        l1 = h.get('delta_k', h.get('l1_projection', '?'))
        print(f"     iter {h.get('iteration', i)}: status={status}, delta_k={l1}, term={h.get('termination', '-')}")
        if h.get('iis_constraints'):
            print(f"       -- LP IIS: {len(h['iis_constraints'])} constraints --")
            for ic in h['iis_constraints'][:15]:
                print(f"          {ic['name']}  ({ic['sense']} {ic['rhs']:.4f})")
            for iv in (h.get('iis_var_bounds') or [])[:5]:
                tag = ('LB' if iv['lb_in_iis'] else '') + ('UB' if iv['ub_in_iis'] else '')
                print(f"          VAR {iv['name']} [{tag}] bounds=[{iv['lb']:.4g},{iv['ub']:.4g}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--active-sets", default=None)
    parser.add_argument("--sample-ids", default="0,1,2,3,4")
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--max-fp-iter", type=int, default=5)
    args = parser.parse_args()

    _ensure_runtime_imports()
    from scripts import collect_model_fp_diagnostics as cdiag

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    ppc = cdiag.get_case_ppc(args.case)
    samples = cdiag.load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )
    by_id = {int(s.get('sample_id', i)): s for i, s in enumerate(samples)}
    target_ids = _parse_sample_ids(args.sample_ids)

    for sid in target_ids:
        if sid not in by_id:
            print(f"[warn] sample_id={sid} not in active-set; skipping")
            continue
        _probe_one_sample(by_id[sid], ppc, float(args.t_delta), int(args.max_fp_iter))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
