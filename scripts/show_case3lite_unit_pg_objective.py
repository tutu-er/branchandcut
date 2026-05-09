#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show c_pg, pg objective coefficients, and surrogate optima for case3lite.

Default use:
    python scripts/show_case3lite_unit_pg_objective.py

The surrogate unit subproblem objective contains
    cpower[t] + c_pg[t] * pg[t] - lambda[t] * pg[t]
with cpower[t] >= a * pg[t] + b * x[t].  This script reports both c_pg and
the effective marginal pg coefficient a + c_pg - lambda.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cvxpy as cp
from pypower.idx_gen import PMAX, PMIN
from pypower.ext2int import ext2int
from pypower.idx_brch import BR_STATUS, RATE_A
from pypower.idx_gen import GEN_BUS
from pypower.makePTDF import makePTDF

if not hasattr(np, "in1d"):
    np.in1d = np.isin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.case_registry import get_case_ppc
from src.subproblem_lp_solver import (
    build_surrogate_constraint_expression,
)
from src.scenario_utils import normalize_sample_arrays
from src.scenario_utils import get_sample_load_data
from src.uc_NN_subproblem import load_trained_models


def _latest_path(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"no match for {parent / pattern}")
    return matches[0]


def _load_samples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("all_samples") if isinstance(data, dict) else data
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"no samples found in {path}")
    for idx, sample in enumerate(samples):
        normalize_sample_arrays(sample)
        sample.setdefault("sample_id", idx)
        sample.setdefault("source_sample_id", idx)
    return samples


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _resolve_lambda_matrix(dual_predictor, trainer, sample: dict, sample_idx: int, source: str) -> np.ndarray:
    if source == "cached":
        return np.asarray(trainer.lambda_vals[sample_idx], dtype=float)
    pred = np.asarray(dual_predictor.predict(sample), dtype=float)
    if pred.ndim == 2 and pred.shape[0] == trainer.T:
        pred = pred.T
    return pred


def _extract_unit_lambda(lambda_val: np.ndarray, T: int, unit_id: int) -> np.ndarray:
    arr = np.asarray(lambda_val, dtype=float)
    if arr.ndim == 0:
        return np.full(T, float(arr), dtype=float)
    if arr.ndim == 1:
        if arr.size == T:
            return arr.astype(float, copy=True)
        if arr.size % T == 0:
            arr = arr.reshape(-1, T)
        else:
            raise ValueError(f"lambda has incompatible shape {arr.shape}, expected (*, {T})")
    if arr.ndim >= 2:
        arr = arr.reshape(-1, T) if arr.shape[-1] == T else arr.reshape(-1, T)
        return arr[int(unit_id)].astype(float, copy=True)
    raise ValueError(f"unsupported lambda shape: {arr.shape}")


def _resolve_params(trainer, sample: dict, lambda_unit: np.ndarray, sample_idx: int, source: str):
    if source == "cached":
        return (
            np.asarray(trainer.alpha_values[sample_idx], dtype=float).copy(),
            np.asarray(trainer.beta_values[sample_idx], dtype=float).copy(),
            np.asarray(trainer.gamma_values[sample_idx], dtype=float).copy(),
            np.asarray(trainer.delta_values[sample_idx], dtype=float).copy(),
            np.asarray(trainer.cost_values[sample_idx], dtype=float).copy(),
            np.asarray(trainer.pg_cost_values[sample_idx], dtype=float).copy(),
        )
    return trainer.get_surrogate_params(
        sample,
        lambda_unit,
        renewable_data=sample.get("renewable_data"),
    )


def _extract_global_milp_x_from_sample(sample: dict, ng: int, T: int) -> np.ndarray:
    if "unit_commitment_matrix" in sample:
        x = np.asarray(sample["unit_commitment_matrix"], dtype=float)
        if x.shape[0] < ng or x.shape[1] < T:
            raise ValueError(f"unit_commitment_matrix shape {x.shape} is smaller than {(ng, T)}")
        return x[:ng, :T].copy()

    x = np.zeros((ng, T), dtype=float)
    active_set = sample.get("active_set")
    if not isinstance(active_set, list):
        raise KeyError("sample has neither unit_commitment_matrix nor active_set")
    for item in active_set:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], list)
            and len(item[0]) == 2
        ):
            g, t = int(item[0][0]), int(item[0][1])
            if 0 <= g < ng and 0 <= t < T:
                x[g, t] = float(item[1])
    return x


def _solve_fixed_commitment_global_ed(ppc: dict, sample: dict, x_fixed: np.ndarray, t_delta: float):
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    branch = ppc_int["branch"]
    gencost = ppc_int["gencost"]
    bus = ppc_int["bus"]
    ng = int(gen.shape[0])
    nb = int(bus.shape[0])
    T = int(x_fixed.shape[1])
    load = np.asarray(get_sample_load_data(sample), dtype=float)
    if load.shape[0] != nb:
        raise ValueError(f"load rows {load.shape[0]} do not match ppc buses {nb}")
    if load.shape[1] != T:
        raise ValueError(f"load horizon {load.shape[1]} does not match x horizon {T}")

    pg = cp.Variable((ng, T), nonneg=True)
    cpower = cp.Variable((ng, T), nonneg=True)
    constraints = []

    for t in range(T):
        constraints.append(cp.sum(pg[:, t]) == float(np.sum(load[:, t])))
        for g in range(ng):
            constraints.append(pg[g, t] >= float(gen[g, PMIN]) * float(x_fixed[g, t]))
            constraints.append(pg[g, t] <= float(gen[g, PMAX]) * float(x_fixed[g, t]))
            constraints.append(
                cpower[g, t]
                >= float(gencost[g, -2] / t_delta) * pg[g, t]
                + float(gencost[g, -1] / t_delta) * float(x_fixed[g, t])
            )

    # Ramp limits follow the same custom-data helper used by the UC wrapper.
    from src.uc_time_utils import get_ramp_limits_from_ppc

    ru, rd, ru_co, rd_co = get_ramp_limits_from_ppc(ppc, gen, t_delta)
    for t in range(1, T):
        for g in range(ng):
            constraints.append(
                pg[g, t] - pg[g, t - 1]
                <= float(ru[g]) * float(x_fixed[g, t - 1])
                + float(ru_co[g]) * (1.0 - float(x_fixed[g, t - 1]))
            )
            constraints.append(
                pg[g, t - 1] - pg[g, t]
                <= float(rd[g]) * float(x_fixed[g, t])
                + float(rd_co[g]) * (1.0 - float(x_fixed[g, t]))
            )

    g_bus = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        g_bus[int(gen[g, GEN_BUS]), g] = 1.0
    ptdf = makePTDF(ppc_int["baseMVA"], bus, branch)
    ptdf_g = ptdf @ g_bus
    branch_limit = branch[:, RATE_A]
    active_lines = [
        l for l in range(branch.shape[0])
        if float(branch_limit[l]) > 1e-6 and float(branch[l, BR_STATUS]) > 0.0
    ]
    for t in range(T):
        flow = ptdf_g @ pg[:, t] - ptdf @ load[:, t]
        for l in active_lines:
            constraints.append(flow[l] <= float(branch_limit[l]))
            constraints.append(flow[l] >= -float(branch_limit[l]))

    problem = cp.Problem(cp.Minimize(cp.sum(cpower)), constraints)
    problem.solve(solver=cp.HIGHS, threads=1, parallel="off")
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"fixed-commitment global ED failed: status={problem.status!r}")
    return np.asarray(pg.value, dtype=float), {
        "status_name": f"global_milp_active_set+fixed_ed:{problem.status}",
        "objective_value": float(problem.value),
        "used_soft_surrogate": False,
    }


def _solve_unit_lp_highs(
    trainer,
    sample_idx: int,
    lambda_unit: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: np.ndarray | None,
    pg_costs: np.ndarray | None,
    soft_penalty: float = 1e8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    g = int(trainer.unit_id)
    T = int(trainer.T)
    Pmin = float(trainer.gen[g, PMIN])
    Pmax = float(trainer.gen[g, PMAX])
    Ru = float(trainer.Ru_all[g])
    Rd = float(trainer.Rd_all[g])
    Ru_co = float(trainer.Ru_co_all[g])
    Rd_co = float(trainer.Rd_co_all[g])
    a = float(trainer.gencost[g, -2] / trainer.T_delta)
    b = float(trainer.subproblem_generation_no_load_coeff(g))
    Ton = int(trainer.subproblem_Ton)
    Toff = int(trainer.subproblem_Toff)

    def _solve_once(use_soft: bool):
        pg = cp.Variable(T, nonneg=True)
        x = cp.Variable(T)
        cpower = cp.Variable(T, nonneg=True)
        constraints = [x >= 0, x <= 1]

        for t in range(T):
            constraints.append(pg[t] >= Pmin * x[t])
            constraints.append(pg[t] <= Pmax * x[t])
            constraints.append(cpower[t] >= a * pg[t] + b * x[t])

        for t in range(1, T):
            constraints.append(pg[t] - pg[t - 1] <= Ru * x[t - 1] + Ru_co * (1 - x[t - 1]))
            constraints.append(pg[t - 1] - pg[t] <= Rd * x[t] + Rd_co * (1 - x[t]))

        for tau in range(1, Ton + 1):
            for t1 in range(T - tau):
                constraints.append(x[t1 + 1] - x[t1] <= x[t1 + tau])
        for tau in range(1, Toff + 1):
            for t1 in range(T - tau):
                constraints.append(-x[t1 + 1] + x[t1] <= 1 - x[t1 + tau])

        timesteps = list(trainer.sensitive_timesteps[sample_idx])[: len(alphas)]
        offsets = list(trainer._constraint_offsets_for_sample(sample_idx))[: len(alphas)]
        slacks = []
        for k, t_k in enumerate(timesteps):
            lhs = build_surrogate_constraint_expression(
                x,
                int(t_k),
                offsets[k],
                float(alphas[k]),
                float(betas[k]),
                float(gammas[k]),
                T,
            )
            if use_soft:
                slack = cp.Variable(nonneg=True)
                constraints.append(lhs - float(deltas[k]) <= slack)
                slacks.append(slack)
            else:
                constraints.append(lhs <= float(deltas[k]))

        obj = cp.sum(cpower) - np.asarray(lambda_unit, dtype=float) @ pg
        if costs is not None:
            obj += np.asarray(costs[:T], dtype=float) @ x
        if pg_costs is not None:
            obj += np.asarray(pg_costs[:T], dtype=float) @ pg
        if slacks:
            obj += float(soft_penalty) * cp.sum(cp.hstack(slacks))

        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.HIGHS, threads=1, parallel="off")
        return problem, pg, x, bool(use_soft)

    problem, pg, x, used_soft = _solve_once(use_soft=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        problem, pg, x, used_soft = _solve_once(use_soft=True)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        nan = np.full(T, np.nan, dtype=float)
        return nan, nan, {
            "status_name": str(problem.status),
            "objective_value": float("nan"),
            "used_soft_surrogate": used_soft,
        }

    return (
        np.asarray(x.value, dtype=float),
        np.asarray(pg.value, dtype=float),
        {
            "status_name": str(problem.status),
            "objective_value": float(problem.value),
            "used_soft_surrogate": used_soft,
        },
    )


def _print_vector(label: str, values: np.ndarray, precision: int = 4) -> None:
    text = np.array2string(
        np.asarray(values, dtype=float),
        precision=precision,
        suppress_small=True,
        max_line_width=220,
    )
    print(f"{label}: {text}", flush=True)


def _print_sample_report(
    *,
    sample_idx: int,
    trainer,
    lambda_unit: np.ndarray,
    pg_costs: np.ndarray,
    x_sol: np.ndarray,
    pg_sol: np.ndarray,
    details: dict,
    active_set_x: np.ndarray,
) -> None:
    g = int(trainer.unit_id)
    a_pg = float(trainer.gencost[g, -2] / trainer.T_delta)
    lambda_unit = np.asarray(lambda_unit, dtype=float)
    pg_costs = np.asarray(pg_costs, dtype=float)
    x_sol = np.asarray(x_sol, dtype=float)
    active_set_x = np.asarray(active_set_x, dtype=float)
    eff_pg_coef = a_pg + pg_costs - lambda_unit
    active_mismatch = int(np.sum(np.rint(x_sol).astype(int) != np.rint(active_set_x).astype(int)))
    x_binary_ok = bool(np.all((np.isclose(x_sol, 0.0, atol=1e-8)) | (np.isclose(x_sol, 1.0, atol=1e-8))))

    print("\n" + "=" * 96, flush=True)
    print(
        f"sample_index={sample_idx} | unit={g} | status={details.get('status_name')} | "
        f"obj={details.get('objective_value', float('nan')):.6f}",
        flush=True,
    )
    print(
        f"pg objective: cpower[t] + c_pg[t] * pg[t] - lambda[t] * pg[t], "
        f"cpower lower slope a={a_pg:.6f}",
        flush=True,
    )
    print(
        f"active-set check: x_binary_ok={x_binary_ok}, "
        f"active_set_x_mismatch={active_mismatch}",
        flush=True,
    )
    print("-" * 96, flush=True)
    print(" t |   lambda |     c_pg | c_pg-lambda | a+c_pg-lambda |      pg* |      x*", flush=True)
    print("-" * 96, flush=True)
    for t in range(trainer.T):
        print(
            f"{t:2d} | {lambda_unit[t]:8.4f} | {pg_costs[t]:8.4f} | "
            f"{(pg_costs[t] - lambda_unit[t]):11.4f} | {eff_pg_coef[t]:14.4f} | "
            f"{pg_sol[t]:8.4f} | {x_sol[t]:7.4f}",
            flush=True,
        )
    print("-" * 96, flush=True)
    _print_vector("lambda", lambda_unit)
    _print_vector("c_pg", pg_costs)
    _print_vector("eff_pg_coef = a + c_pg - lambda", eff_pg_coef)
    _print_vector("pg_opt", pg_sol)
    _print_vector("x_opt", x_sol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case3lite")
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--samples", default="0,1,2,3,4", help="comma-separated sample indices")
    parser.add_argument("--active-set", default=None, help="active-set JSON; default latest case3lite file")
    parser.add_argument("--model-dir", default=None, help="surrogate model dir; default latest case3lite model")
    parser.add_argument(
        "--source",
        choices=("cached", "forward"),
        default="cached",
        help="cached = saved training cache; forward = current NN forward + dual predictor",
    )
    parser.add_argument("--lp-backend", default=None, help="override model lp_backend if needed")
    parser.add_argument("--solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument(
        "--solution-source",
        choices=("global-milp", "subproblem-lp"),
        default="global-milp",
        help="global-milp reads the MILP commitment from active_set and recovers pg via fixed-x global ED.",
    )
    args = parser.parse_args()

    active_set = (
        Path(args.active_set)
        if args.active_set
        else _latest_path(ROOT / "result" / "active_set", f"active_sets_{args.case}_*.json")
    )
    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else _latest_path(ROOT / "result" / "surrogate_models", f"subproblem_models_{args.case}_*")
    )
    if not active_set.is_absolute():
        active_set = (ROOT / active_set).resolve()
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()

    sample_indices = _parse_int_list(args.samples)
    samples = _load_samples(active_set)
    ppc = get_case_ppc(args.case)
    dual_predictor, trainers = load_trained_models(
        ppc,
        samples,
        T_delta=1.0,
        load_dir=str(model_dir),
        unit_ids=[int(args.unit)],
        lp_backend=args.lp_backend,
        case_name=args.case,
    )
    if args.unit not in trainers:
        raise KeyError(f"unit {args.unit} not loaded from {model_dir}")
    trainer = trainers[args.unit]

    print(f"case={args.case}", flush=True)
    print(f"active_set={active_set}", flush=True)
    print(f"model_dir={model_dir}", flush=True)
    print(f"unit={args.unit}, samples={sample_indices}, source={args.source}", flush=True)

    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= len(samples):
            raise IndexError(f"sample index {sample_idx} out of range [0, {len(samples)})")
        sample = dict(samples[sample_idx])
        sample["sample_id"] = sample_idx
        sample["source_sample_id"] = sample_idx

        lambda_matrix = _resolve_lambda_matrix(dual_predictor, trainer, sample, sample_idx, args.source)
        lambda_unit = _extract_unit_lambda(lambda_matrix, trainer.T, unit_id=args.unit)
        alphas, betas, gammas, deltas, costs, pg_costs = _resolve_params(
            trainer,
            sample,
            lambda_unit,
            sample_idx,
            args.source,
        )
        if args.solution_source == "global-milp":
            x_global = _extract_global_milp_x_from_sample(sample, ppc["gen"].shape[0], trainer.T)
            pg_global, details = _solve_fixed_commitment_global_ed(
                ppc,
                sample,
                x_global,
                t_delta=1.0,
            )
            x_sol = x_global[int(args.unit)]
            pg_sol = pg_global[int(args.unit)]
        elif args.solver == "highs":
            x_sol, pg_sol, details = _solve_unit_lp_highs(
                trainer,
                sample_idx,
                lambda_unit,
                alphas,
                betas,
                gammas,
                deltas,
                costs,
                pg_costs,
            )
        else:
            from src.feasibility_pump import _solve_unit_LP_with_surrogate

            x_sol, status, details = _solve_unit_LP_with_surrogate(
                trainer,
                lambda_unit,
                alphas,
                betas,
                gammas,
                deltas,
                costs=costs,
                pg_costs=pg_costs,
                scenario_sample=sample,
            )
            details.setdefault("status_name", str(status))
            pg_sol = np.asarray(details.get("pg_solution", np.full(trainer.T, np.nan)), dtype=float)
        active_set_x = _extract_global_milp_x_from_sample(sample, ppc["gen"].shape[0], trainer.T)[int(args.unit)]
        _print_sample_report(
            sample_idx=sample_idx,
            trainer=trainer,
            lambda_unit=lambda_unit,
            pg_costs=pg_costs,
            x_sol=np.asarray(x_sol, dtype=float),
            pg_sol=pg_sol,
            details=details,
            active_set_x=active_set_x,
        )


if __name__ == "__main__":
    main()
