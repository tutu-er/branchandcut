#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose two FP behaviours on a single sample.

  Part A.  Rebuild the *vanilla* feasibility pump's LP projection model
           (UC LP relaxation + L1 objective) standalone and check whether
           it is actually infeasible. If yes, dump the IIS so we can see
           which constraints conflict.

  Part B.  Inspect which hot-start makes the tailored FP "feasible at
           iter 0". Print the hot-start's name, hamming distance vs the
           true commitment, and UC objective gap to the true solution.
           A small gap means the precheck was justified; a large gap
           means we accepted a feasible-but-suboptimal hot-start and
           should have continued iterating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402

from pypower.ext2int import ext2int  # noqa: E402

from scripts.collect_model_fp_diagnostics import (  # noqa: E402
    _ensure_runtime_imports,
    _extract_true_solution,
    _latest_active_set,
    _latest_model_dir,
    _load_bcd_agent,
    _resolve_path,
)
from src.feasibility_pump import (  # noqa: E402
    _build_ptdf_data,
    _get_min_up_down_time_steps,
    _get_ramp_limits_from_ppc,
    check_uc_feasibility,
    recover_integer_solution,
)


def _build_uc_lp_projection_model(
    x_curr: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
    *,
    fix_trusted: bool = False,
    add_dc: bool = True,
):
    """Rebuild the vanilla FP LP-projection model in isolation (no surrogates)."""

    from pypower.idx_gen import PMAX, PMIN

    gen = ppc['gen']
    gencost = ppc['gencost']
    ng, T = x_curr.shape
    Pd_sum = np.sum(pd_data, axis=0)

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, T_delta, T)

    model = gp.Model('debug_fp_projection')
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, T, lb=0, name='pg')
    x = model.addVars(ng, T, lb=0, ub=1, name='x')
    d = model.addVars(ng, T, lb=0, name='d')
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

        for t in range(T):
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                              + gencost[g, -1] / T_delta * x[g, t],
                name=f'fuelc_{g}_{t}',
            )
            x_ref = float(x_curr[g, t])
            if fix_trusted:
                model.addConstr(x[g, t] == x_ref, name=f'fix_{g}_{t}')
            else:
                model.addConstr(d[g, t] >= x[g, t] - x_ref, name=f'L1pos_{g}_{t}')
                model.addConstr(d[g, t] >= x_ref - x[g, t], name=f'L1neg_{g}_{t}')

    if add_dc:
        try:
            ppc_int = ext2int(ppc)
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
            print(f"  [debug] DC line constraints added: {len(active_lines)} lines")
        except Exception as exc:
            print(f"  [debug] DC constraints skipped: {exc}")

    obj = gp.quicksum(d[g, t] for g in range(ng) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    return model


def _compute_uc_dispatch_objective(
    x_int: np.ndarray,
    ppc: dict,
    pd_data: np.ndarray,
    T_delta: float,
) -> Tuple[Optional[float], str]:
    """Return UC cost given an integer commitment (dispatch LP)."""
    from pypower.idx_gen import PMAX, PMIN

    gen = ppc['gen']
    gencost = ppc['gencost']
    ng, T = x_int.shape
    Pd_sum = np.sum(pd_data, axis=0)
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    model = gp.Model('uc_dispatch')
    model.Params.OutputFlag = 0
    pg = model.addVars(ng, T, lb=0, name='pg')
    cpower = model.addVars(ng, T, lb=0, name='cpower')
    coc = model.addVars(ng, max(T - 1, 0), lb=0, name='coc')

    for t in range(T):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(Pd_sum[t]))
    for g in range(ng):
        for t in range(T):
            model.addConstr(pg[g, t] >= float(gen[g, PMIN]) * float(x_int[g, t]))
            model.addConstr(pg[g, t] <= float(gen[g, PMAX]) * float(x_int[g, t]))
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / T_delta * pg[g, t]
                              + gencost[g, -1] / T_delta * float(x_int[g, t])
            )
        for t in range(1, T):
            model.addConstr(
                pg[g, t] - pg[g, t-1]
                <= Ru[g] * float(x_int[g, t-1]) + Ru_co[g] * (1 - float(x_int[g, t-1]))
            )
            model.addConstr(
                pg[g, t-1] - pg[g, t]
                <= Rd[g] * float(x_int[g, t]) + Rd_co[g] * (1 - float(x_int[g, t]))
            )
            model.addConstr(
                coc[g, t-1] >= float(start_cost[g])
                              * (float(x_int[g, t]) - float(x_int[g, t-1]))
            )
            model.addConstr(
                coc[g, t-1] >= float(shut_cost[g])
                              * (float(x_int[g, t-1]) - float(x_int[g, t]))
            )

    try:
        ppc_int = ext2int(ppc)
        PTDF, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_Pd = PTDF @ pd_data
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(T):
                flow_expr = (
                    gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng))
                    - float(ptdf_Pd[l, t])
                )
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
    except Exception:
        pass

    obj = (gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T))
           + gp.quicksum(coc[g, t] for g in range(ng) for t in range(max(T-1, 0))))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        return None, f"dispatch_status={model.status}"
    return float(model.ObjVal), "ok"


def _hamming(a, b):
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        return None
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if not np.any(valid):
        return None
    return int(np.sum(np.round(arr_a[valid]) != np.round(arr_b[valid])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--active-sets", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--bcd-model", default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--t-delta", type=float, default=1.0)
    args = parser.parse_args()

    _ensure_runtime_imports()
    from scripts import collect_model_fp_diagnostics as cdiag

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))
    bcd_model_path = (
        _resolve_path(args.bcd_model, None) if args.bcd_model else None
    )

    print(f"[debug] case={args.case}")
    print(f"[debug] active_sets={active_set_path}")
    print(f"[debug] model_dir={model_dir}")
    if bcd_model_path:
        print(f"[debug] bcd_model={bcd_model_path}")

    ppc = cdiag.get_case_ppc(args.case)
    all_samples = cdiag.load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )

    target = None
    for s in all_samples:
        if int(s.get('sample_id', s.get('source_sample_id', -1))) == int(args.sample_id):
            target = s
            break
    if target is None and 0 <= args.sample_id < len(all_samples):
        target = all_samples[args.sample_id]
        target.setdefault('sample_id', args.sample_id)
    if target is None:
        print(f"[error] sample_id={args.sample_id} not found")
        return 2

    pd_data = cdiag.get_sample_net_load(target)
    T_delta = float(args.t_delta)
    ng = int(np.asarray(ppc['gen']).shape[0])
    T = int(pd_data.shape[1])
    print(f"[debug] sample {args.sample_id}: ng x T = {ng} x {T}")

    shape = (ng, T)
    x_true_arr = _extract_true_solution(target, shape)
    x_true = (
        np.round(np.where(np.isfinite(x_true_arr), x_true_arr, 0.0)).astype(int)
        if x_true_arr is not None else None
    )
    if x_true is not None:
        print(f"[truth] x_true sum={int(x_true.sum())}/{ng*T}")
        obj_true, status_true = _compute_uc_dispatch_objective(x_true, ppc, pd_data, T_delta)
        print(f"[truth] UC objective (dispatch w/ x_true) = {obj_true}  status={status_true}")
    else:
        obj_true = None

    print("\n=== Part A: standalone vanilla FP LP projection ===")
    x_curr_dbg = (
        x_true.copy() if x_true is not None
        else np.ones((ng, T), dtype=int)
    )
    print(f"  anchor x_curr = {'x_true' if x_true is not None else 'all-ones'}")
    model = _build_uc_lp_projection_model(
        x_curr_dbg, ppc, pd_data, T_delta, fix_trusted=False, add_dc=True,
    )
    model.optimize()
    status_name = {
        GRB.OPTIMAL: 'OPTIMAL', GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.UNBOUNDED: 'UNBOUNDED', GRB.INF_OR_UNBD: 'INF_OR_UNBD',
    }.get(model.status, f'STATUS_{model.status}')
    print(f"  status = {status_name}")
    if model.status == GRB.OPTIMAL:
        print(f"  L1 objective = {model.ObjVal:.6f}")
    else:
        print("  computing IIS ...")
        try:
            model.computeIIS()
            iis_cons = [c for c in model.getConstrs() if c.IISConstr]
            print(f"  IIS size = {len(iis_cons)} constraints; first 40 names:")
            for c in iis_cons[:40]:
                print(f"    - {c.ConstrName}")
            iis_vars = [v for v in model.getVars() if v.IISLB or v.IISUB]
            print(f"  IIS variable bounds in conflict: {len(iis_vars)}")
            for v in iis_vars[:15]:
                tag = ('LB' if v.IISLB else '') + ('UB' if v.IISUB else '')
                print(f"    - {v.VarName} [{tag}]")
        except gp.GurobiError as exc:
            print(f"  IIS computation failed: {exc}")

    print("\n  retrying WITHOUT DC line constraints ...")
    model_no_dc = _build_uc_lp_projection_model(
        x_curr_dbg, ppc, pd_data, T_delta, fix_trusted=False, add_dc=False,
    )
    model_no_dc.optimize()
    print(f"  status (no DC) = {model_no_dc.status}")
    if model_no_dc.status == GRB.OPTIMAL:
        print(f"  L1 objective (no DC) = {model_no_dc.ObjVal:.6f}")

    print("\n=== Part A2: feed actual rounded LP solution into projection ===")
    if model_dir is not None:
        dual_predictor_dbg, trainers_dbg = cdiag.load_trained_models(
            ppc,
            all_samples,
            T_delta,
            str(model_dir),
            unit_ids=None,
            case_name=args.case,
            skip_initial_solve=True,
        )
        agent_dbg = (
            _load_bcd_agent(ppc, active_set_path, bcd_model_path, T_delta)
            if bcd_model_path else None
        )
        lam_dbg = dual_predictor_dbg.predict(target)
        cached_dbg = type("Cd", (), {"predict": lambda self, _s, _v=lam_dbg: _v})()
        x_rec_v, ok_v, dets_v = recover_integer_solution(
            target,
            trainers_dbg,
            cached_dbg,
            ppc,
            T_delta,
            agent=agent_dbg,
            fp_strategy='vanilla',
            max_fp_iter=5,
            verbose=True,
            return_details=True,
        )
        print(f"  vanilla success={ok_v}")
        print(f"  vanilla selected_hot_start={dets_v.get('selected_hot_start')!r}")
        v_pre = dets_v.get('hot_start_prechecks') or []
        print(f"  vanilla precheck count = {len(v_pre)}")
        for p in v_pre[:6]:
            print(f"    -> name={p.get('hot_start_name')} feas={p.get('precheck_feasible')} "
                  f"reason={p.get('precheck_reason')}")
        v_hist = dets_v.get('fp_histories') or []
        print(f"  vanilla fp_histories count = {len(v_hist)}")
        for h in v_hist[:3]:
            print(f"    hot_start_name={h.get('hot_start_name')!r} "
                  f"entered_fp={h.get('entered_fp_iterations')} "
                  f"termination={h.get('termination')!r} iters={len(h.get('history') or [])}")
            for step in (h.get('history') or [])[:3]:
                print(f"       step={step.get('iteration')} "
                      f"projection_status={step.get('projection_status')} "
                      f"projection_status_name={step.get('projection_status_name')} "
                      f"termination={step.get('termination')} "
                      f"delta_k={step.get('delta_k')}")

    print("\n=== Part B: tailored hot-start inspection ===")
    if model_dir is None:
        print("  [skip] model_dir required for tailored run")
        return 0
    dual_predictor, trainers = cdiag.load_trained_models(
        ppc,
        all_samples,
        T_delta,
        str(model_dir),
        unit_ids=None,
        case_name=args.case,
        skip_initial_solve=True,
    )
    agent = None
    if bcd_model_path is not None:
        agent = _load_bcd_agent(ppc, active_set_path, bcd_model_path, T_delta)

    lam_val = dual_predictor.predict(target)
    cached = type("Cached", (), {"predict": lambda self, _s, _v=lam_val: _v})()

    print("\n  [nearby source check] inspecting historical scenario bank ...")
    from src.feasibility_pump import _get_scenario_bank, _extract_commitment_from_sample
    from src.scenario_utils import get_feature_vector_from_sample
    try:
        scenario_bank = _get_scenario_bank(trainers)
        bank_size = len(scenario_bank) if scenario_bank else 0
        print(f"    scenario_bank size = {bank_size}")
        if bank_size > 0 and x_true is not None:
            target_feat = get_feature_vector_from_sample(dict(target))
            scored = []
            for s in scenario_bank:
                try:
                    fv = get_feature_vector_from_sample(dict(s))
                    if fv.shape != target_feat.shape:
                        continue
                    d = float(np.linalg.norm(fv - target_feat))
                    scored.append((d, s))
                except Exception:
                    continue
            scored.sort(key=lambda kv: kv[0])
            print(f"    {'src_sample_id':>14s}  {'distance':>12s}  {'ham_vs_truth':>14s}  uc_obj")
            for d, s in scored[:5]:
                commit = _extract_commitment_from_sample(s, ng, T)
                if commit is None:
                    continue
                ham_n = _hamming(commit, x_true)
                obj_n, _ = _compute_uc_dispatch_objective(
                    np.asarray(commit, dtype=int), ppc, pd_data, T_delta,
                )
                sid = s.get('sample_id', s.get('source_sample_id'))
                self_sid = target.get('sample_id', target.get('source_sample_id'))
                tag = ' (TARGET)' if sid == self_sid or d < 1e-12 else ''
                obj_str = '-' if obj_n is None else f"{obj_n:10.4f}"
                print(f"    {str(sid):>14s}  {d:12.4e}  {str(ham_n):>14s}  {obj_str}{tag}")
    except Exception as exc:
        print(f"    [warn] could not inspect scenario bank: {exc}")

    x_rec, success, details = recover_integer_solution(
        target,
        trainers,
        cached,
        ppc,
        T_delta,
        agent=agent,
        fp_strategy='tailored',
        rounding_strategy='chi_argmax',
        max_fp_iter=30,
        verbose=False,
        return_details=True,
    )
    print(f"  tailored success = {success}")
    print(f"  selected_hot_start = {details.get('selected_hot_start')!r}")
    hot_start_prechecks = details.get('hot_start_prechecks') or []
    print(f"  prechecks count = {len(hot_start_prechecks)}")
    print("  per-hot-start summary:")
    print(f"    {'name':32s} {'feasible':>9s} {'ham':>5s} {'uc_obj':>14s} {'gap%':>9s}  reason")
    for pre in hot_start_prechecks:
        name = str(pre.get('hot_start_name'))
        feas = bool(pre.get('precheck_feasible'))
        reason = pre.get('precheck_reason')
        x_arr = pre.get('x_start')
        ham_pre, obj_pre, gap_str = None, None, '-'
        if x_arr is not None:
            x_int_pre = np.round(np.asarray(x_arr, dtype=float)).astype(int)
            if x_true is not None:
                ham_pre = _hamming(x_int_pre, x_true)
            if feas:
                obj_pre, _ = _compute_uc_dispatch_objective(
                    x_int_pre, ppc, pd_data, T_delta,
                )
                if obj_pre is not None and obj_true is not None and obj_true > 1e-9:
                    gap_str = f"{(obj_pre - obj_true) / abs(obj_true) * 100:+8.3f}"
        ham_str = '-' if ham_pre is None else f"{ham_pre:5d}"
        obj_str = '-' if obj_pre is None else f"{obj_pre:14.4f}"
        print(f"    {name:32s} {str(feas):>9s} {ham_str:>5s} {obj_str:>14s} {gap_str:>9s}  {reason}")

    if x_rec is not None and x_true is not None:
        x_int_rec = np.round(np.asarray(x_rec, dtype=float)).astype(int)
        ham_rec = _hamming(x_int_rec, x_true)
        obj_rec, _ = _compute_uc_dispatch_objective(x_int_rec, ppc, pd_data, T_delta)
        print("\n  -- final tailored result --")
        print(f"    hamming_to_true = {ham_rec}")
        print(f"    UC obj (tailored) = {obj_rec}")
        print(f"    UC obj (truth)    = {obj_true}")
        if obj_rec is not None and obj_true is not None and obj_true > 1e-9:
            print(f"    gap = {(obj_rec - obj_true) / abs(obj_true) * 100:+.4f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
