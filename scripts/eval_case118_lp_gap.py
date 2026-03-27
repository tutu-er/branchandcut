#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.numpy_compat import ensure_numpy_compat_for_pypower

ensure_numpy_compat_for_pypower()

import gurobipy as gp
from gurobipy import GRB
from pypower.ext2int import ext2int
from pypower.idx_gen import PMIN, PMAX

from src.feasibility_pump import _build_ptdf_data
from src.mti118_data_loader import load_case118_ppc_with_mti_limits
from src.scenario_utils import normalize_sample_arrays


DEFAULT_JSON = ROOT / "result" / "active_set" / "active_sets_case118_T0_n366_20260322_063917.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare case118 LP-relaxation x against stored optimal unit commitment."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="active set json path")
    parser.add_argument("--t-delta", type=float, default=1.0, help="time step in hours")
    parser.add_argument("--limit", type=int, default=10, help="max number of samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result" / "figures",
        help="directory for LP-vs-true heatmaps",
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
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], list)
            and len(item[0]) == 2
        ):
            g, t = item[0]
            value = item[1]
        elif isinstance(item, dict):
            g = item.get("unit_id", item.get("generator_id", item.get("g")))
            t = item.get("time_slot", item.get("time", item.get("t")))
            value = item.get("value", item.get("x"))
        else:
            continue
        if g is None or t is None or value is None:
            continue
        x_opt[int(g), int(t)] = float(value)
    return x_opt


def solve_lp_relaxation(ppc: dict, pd_data: np.ndarray, t_delta: float) -> np.ndarray:
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng = gen.shape[0]
    horizon = pd_data.shape[1]
    pd_sum = np.sum(pd_data, axis=0)

    model = gp.Model("case118_lp_gap_eval")
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

    ptdf, ptdf_g, branch_limit, active_lines = _build_ptdf_data(ppc_int)
    ptdf_pd = ptdf @ pd_data
    for l in active_lines:
        limit = float(branch_limit[l])
        for t in range(horizon):
            flow_expr = gp.quicksum(float(ptdf_g[l, g]) * pg[g, t] for g in range(ng)) - float(ptdf_pd[l, t])
            model.addConstr(flow_expr <= limit)
            model.addConstr(flow_expr >= -limit)

    model.setObjective(
        gp.quicksum(cpower[g, t] for g in range(ng) for t in range(horizon))
        + gp.quicksum(coc[g, t] for g in range(ng) for t in range(horizon - 1)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"LP relaxation solve failed, status={model.status}")

    return np.array([[x[g, t].X for t in range(horizon)] for g in range(ng)], dtype=float)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lp_vs_true_heatmap(
    sample_id: int,
    x_lp: np.ndarray,
    x_true: np.ndarray,
    output_dir: Path,
) -> Path:
    diff = np.abs(x_lp - x_true)
    ng, _horizon = x_lp.shape

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    cbar_kw = dict(fraction=0.045, pad=0.03)
    yticks = range(ng)
    ylabels = [f"G{g}" for g in yticks]

    im0 = axes[0].imshow(x_true, aspect="auto", cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], **cbar_kw)
    axes[0].set_title("True Unit Commitment", loc="left", fontsize=10)
    axes[0].set_xlabel("Time Period")
    axes[0].set_ylabel("Generator")
    axes[0].set_yticks(list(yticks))
    axes[0].set_yticklabels(ylabels, fontsize=7)

    im1 = axes[1].imshow(x_lp, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], **cbar_kw)
    axes[1].set_title("LP Relaxation", loc="left", fontsize=10)
    axes[1].set_xlabel("Time Period")
    axes[1].set_ylabel("Generator")
    axes[1].set_yticks(list(yticks))
    axes[1].set_yticklabels(ylabels, fontsize=7)

    im2 = axes[2].imshow(diff, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im2, ax=axes[2], **cbar_kw)
    axes[2].set_title(r"Absolute Difference $|x^{LP}-x^*|$", loc="left", fontsize=10)
    axes[2].set_xlabel("Time Period")
    axes[2].set_ylabel("Generator")
    axes[2].set_yticks(list(yticks))
    axes[2].set_yticklabels(ylabels, fontsize=7)

    fig.suptitle(f"Case118 LP vs True Heatmap [sample {sample_id}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_base = output_dir / f"case118_lp_vs_true_sample{sample_id}"
    save_figure(fig, output_base)
    return output_base.with_suffix(".png")


def main() -> None:
    args = parse_args()
    json_path = args.json if args.json.is_absolute() else (ROOT / args.json)
    samples = load_samples(json_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    ppc = load_case118_ppc_with_mti_limits()
    ng = int(ppc["gen"].shape[0])
    horizon = int(np.asarray(samples[0]["pd_data"], dtype=float).shape[1])

    print(f"json: {json_path}", flush=True)
    print(f"samples: {len(samples)}, ng={ng}, T={horizon}, t_delta={args.t_delta}", flush=True)
    print(f"heatmap_output_dir: {args.output_dir}", flush=True)

    rows: list[dict] = []
    for idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", idx)
        pd_data = np.asarray(sample["pd_data"], dtype=float)
        x_opt = extract_x_opt(sample, ng, horizon)
        x_lp = solve_lp_relaxation(ppc, pd_data, args.t_delta)

        l1 = float(np.sum(np.abs(x_lp - x_opt)))
        hamming = int(np.sum(np.round(x_lp).astype(int) != x_opt.astype(int)))
        frac_mean = float(np.mean(np.minimum(x_lp, 1.0 - x_lp)))

        row = {
            "sample_id": sample_id,
            "l1_x_gap": l1,
            "hamming_gap": hamming,
            "frac_mean": frac_mean,
        }
        rows.append(row)
        heatmap_path = plot_lp_vs_true_heatmap(sample_id, x_lp, x_opt, args.output_dir)

        print(
            f"[{idx:02d}] sample_id={sample_id} "
            f"l1={l1:.2f} hamming={hamming} frac_mean={frac_mean:.4f} "
            f"heatmap={heatmap_path}",
            flush=True,
        )

    def _mean(key: str) -> float:
        vals = [row[key] for row in rows]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nSummary", flush=True)
    print(f"  avg l1_x_gap     = {_mean('l1_x_gap'):.4f}", flush=True)
    print(f"  avg hamming_gap  = {_mean('hamming_gap'):.4f}", flush=True)
    print(f"  avg frac_mean    = {_mean('frac_mean'):.4f}", flush=True)


if __name__ == "__main__":
    main()
