#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import gurobipy as gp
from gurobipy import GRB
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX

from src.case30_uc_data import get_case30_uc_ppc
from src.feasibility_pump import _build_ptdf_data
from src.scenario_utils import normalize_sample_arrays
from src.uc_gurobipy import UnitCommitmentModel


DEFAULT_JSON = ROOT / "result" / "active_set" / "active_sets_case30_T24_n53_20260322_172141.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate direct LP-relaxation gap against optimal case30 samples."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="active set json path")
    parser.add_argument("--t-delta", type=float, default=1.0, help="time step in hours")
    parser.add_argument("--limit", type=int, default=None, help="max number of samples")
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="skip resolving MILP objective, only compare LP x against stored optimal x",
    )
    return parser.parse_args()


def load_samples(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("all_samples", [])
    if not samples:
        raise ValueError(f"no samples found in {json_path}")
    return [normalize_sample_arrays(dict(sample)) for sample in samples]


def extract_x_opt(sample: dict, ng: int, horizon: int) -> np.ndarray:
    if "unit_commitment_matrix" in sample and sample["unit_commitment_matrix"] is not None:
        return np.asarray(sample["unit_commitment_matrix"], dtype=float).reshape(ng, horizon)

    x_opt = np.zeros((ng, horizon), dtype=float)
    for item in sample.get("active_set", []):
        (g, t), value = item
        x_opt[int(g), int(t)] = float(value)
    return x_opt


def solve_lp_relaxation_with_obj(ppc: dict, pd_data: np.ndarray, t_delta: float) -> tuple[np.ndarray, float]:
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng = gen.shape[0]
    horizon = pd_data.shape[1]
    pd_sum = np.sum(pd_data, axis=0)

    model = gp.Model("lp_relaxation_eval")
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, horizon, lb=0, name="pg")
    x = model.addVars(ng, horizon, lb=0, ub=1, name="x")
    cpower = model.addVars(ng, horizon, lb=0, name="cpower")
    coc = model.addVars(ng, horizon - 1, lb=0, name="coc")

    for t in range(horizon):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(pd_sum[t]), name=f"pb_{t}")

    ru = 0.4 * gen[:, PMAX] / t_delta
    rd = 0.4 * gen[:, PMAX] / t_delta
    ru_co = 0.3 * gen[:, PMAX]
    rd_co = 0.3 * gen[:, PMAX]
    ton = min(int(4 * t_delta), horizon - 1)
    toff = min(int(4 * t_delta), horizon - 1)
    start_cost = gencost[:, 1]
    shut_cost = gencost[:, 2]

    for g in range(ng):
        for t in range(horizon):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])
            model.addConstr(
                cpower[g, t] >= gencost[g, -2] / t_delta * pg[g, t] + gencost[g, -1] / t_delta * x[g, t]
            )

        for t in range(1, horizon):
            model.addConstr(pg[g, t] - pg[g, t - 1] <= ru[g] * x[g, t - 1] + ru_co[g] * (1 - x[g, t - 1]))
            model.addConstr(pg[g, t - 1] - pg[g, t] <= rd[g] * x[g, t] + rd_co[g] * (1 - x[g, t]))
            model.addConstr(coc[g, t - 1] >= start_cost[g] * (x[g, t] - x[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= shut_cost[g] * (x[g, t - 1] - x[g, t]))

        for tau in range(1, ton + 1):
            for t1 in range(horizon - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau])
        for tau in range(1, toff + 1):
            for t1 in range(horizon - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau])

    try:
        ptdf, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
        ptdf_pd = ptdf @ pd_data
        for l in active_lines:
            limit = float(branch_limit[l])
            for t in range(horizon):
                flow_expr = gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng)) - float(ptdf_pd[l, t])
                model.addConstr(flow_expr <= limit)
                model.addConstr(flow_expr >= -limit)
    except Exception as exc:
        print(f"warning: failed to build DC flow constraints, continue without them: {exc}", flush=True)

    model.setObjective(
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(horizon))
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(horizon - 1)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"LP relaxation solve failed, status={model.status}")

    x_lp = np.array([[x[g, t].X for t in range(horizon)] for g in range(ng)], dtype=float)
    return x_lp, float(model.ObjVal)


def main() -> None:
    args = parse_args()
    json_path = args.json if args.json.is_absolute() else (ROOT / args.json)
    samples = load_samples(json_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    ppc = get_case30_uc_ppc()
    ng = int(ppc["gen"].shape[0])
    horizon = int(np.asarray(samples[0]["pd_data"], dtype=float).shape[1])

    print(f"json: {json_path}", flush=True)
    print(f"samples: {len(samples)}, ng={ng}, T={horizon}, t_delta={args.t_delta}", flush=True)

    rows: list[dict] = []
    for idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", idx)
        pd_data = np.asarray(sample["pd_data"], dtype=float)
        x_opt = extract_x_opt(sample, ng, horizon)

        x_lp, lp_obj = solve_lp_relaxation_with_obj(ppc, pd_data, args.t_delta)

        opt_obj = None
        if not args.skip_milp:
            uc = UnitCommitmentModel(ppc, pd_data, args.t_delta, verbose=False)
            _, x_milp, opt_obj = uc.solve()
            x_opt = np.asarray(x_milp, dtype=float)

        l1 = float(np.sum(np.abs(x_lp - x_opt)))
        hamming = int(np.sum(np.round(x_lp).astype(int) != x_opt.astype(int)))
        frac_mean = float(np.mean(np.minimum(x_lp, 1.0 - x_lp)))

        row = {
            "sample_id": sample_id,
            "lp_obj": float(lp_obj),
            "opt_obj": None if opt_obj is None else float(opt_obj),
            "rel_gap_pct": None if opt_obj is None else float((opt_obj - lp_obj) / max(abs(opt_obj), 1e-9) * 100.0),
            "l1_x_gap": l1,
            "hamming_gap": hamming,
            "frac_mean": frac_mean,
        }
        rows.append(row)

        gap_text = "n/a" if row["rel_gap_pct"] is None else f"{row['rel_gap_pct']:.3f}%"
        print(
            f"[{idx:02d}] sample_id={sample_id} "
            f"lp_obj={lp_obj:.4f} opt_obj={row['opt_obj'] if row['opt_obj'] is not None else 'n/a'} "
            f"gap={gap_text} l1={l1:.2f} hamming={hamming} frac_mean={frac_mean:.4f}",
            flush=True,
        )

    def _mean(key: str) -> float:
        vals = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nSummary", flush=True)
    print(f"  avg l1_x_gap     = {_mean('l1_x_gap'):.4f}", flush=True)
    print(f"  avg hamming_gap  = {_mean('hamming_gap'):.4f}", flush=True)
    print(f"  avg frac_mean    = {_mean('frac_mean'):.4f}", flush=True)
    if not args.skip_milp:
        print(f"  avg rel_gap_pct  = {_mean('rel_gap_pct'):.4f}%", flush=True)
        print(f"  avg lp_obj       = {_mean('lp_obj'):.4f}", flush=True)
        print(f"  avg opt_obj      = {_mean('opt_obj'):.4f}", flush=True)


if __name__ == "__main__":
    main()
