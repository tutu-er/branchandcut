"""Run Algorithm-1 style theta constraint screening and compare with activity.

The screening logic follows the user-provided pseudo-code:
1. Solve the UC LP relaxation with DCPF line constraints.
2. Candidate unit-time pairs are those with bad rounding or poor integrality.
3. For each time, sort line-flow duals by absolute value and keep at most K_MAX
   constraints whose line dual exceeds epsilon.
4. A selected row keeps only generators whose PTDF is nonzero and whose
   (unit, time) is in the candidate set.

The selected (line, time) rows are compared against saved final-stage theta
activity from the paper-evaluation run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pypower.ext2int import ext2int  # noqa: E402
from pypower.idx_brch import RATE_A  # noqa: E402
from pypower.idx_gen import GEN_BUS, PMAX, PMIN  # noqa: E402
from pypower.makePTDF import makePTDF  # noqa: E402

from src.case3_uc_data import get_case3lite_uc_ppc  # noqa: E402
from src.uc_NN_BCD import load_active_set_from_json  # noqa: E402
from src.feasibility_pump import _get_min_up_down_time_steps, _get_ramp_limits_from_ppc  # noqa: E402


DEFAULT_ACTIVE_SET = ROOT / "result/active_set/active_sets_case3lite_T24_n200_20260328_102856.json"
DEFAULT_ACTIVITY = (
    ROOT
    / "result/paper_eval/20260512_183843/raw/runs/case3lite/S01_bcd_theta/raw/figures"
    / "case3lite_global_surrogate_solve_stats/main_activity/main_constraint_activity_by_sample_row.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "result/model_tests/case3lite_main_theta_improvement"


def solve_lp_with_line_duals(ppc: dict, pd_data: np.ndarray, t_delta: float) -> tuple[np.ndarray, np.ndarray]:
    ppc_int = ext2int(ppc.copy())
    gen = np.asarray(ppc_int["gen"], dtype=float)
    gencost = np.asarray(ppc_int["gencost"], dtype=float)
    bus = np.asarray(ppc_int["bus"], dtype=float)
    branch = np.asarray(ppc_int["branch"], dtype=float)
    base_mva = float(ppc_int["baseMVA"])
    ng = gen.shape[0]
    nl = branch.shape[0]
    T = pd_data.shape[1]

    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, t_delta)
    min_up_steps, min_down_steps = _get_min_up_down_time_steps(ppc, ng, t_delta, T)

    pg = cp.Variable((ng, T))
    u = cp.Variable((ng, T))
    cpower = cp.Variable((ng, T))
    coc = cp.Variable((ng, max(T - 1, 0)))

    constraints = [pg >= 0, u >= 0, u <= 1, cpower >= 0]
    if T > 1:
        constraints.append(coc >= 0)

    for t in range(T):
        constraints.append(cp.sum(pg[:, t]) == float(np.sum(pd_data[:, t])))

    for g in range(ng):
        for t in range(T):
            constraints.append(pg[g, t] >= float(gen[g, PMIN]) * u[g, t])
            constraints.append(pg[g, t] <= float(gen[g, PMAX]) * u[g, t])

        for t in range(1, T):
            constraints.append(pg[g, t] - pg[g, t - 1] <= Ru[g] * u[g, t - 1] + Ru_co[g] * (1 - u[g, t - 1]))
            constraints.append(pg[g, t - 1] - pg[g, t] <= Rd[g] * u[g, t] + Rd_co[g] * (1 - u[g, t]))

        for tau in range(1, int(min_up_steps[g]) + 1):
            for t1 in range(T - tau):
                constraints.append(u[g, t1 + 1] - u[g, t1] <= u[g, t1 + tau])
        for tau in range(1, int(min_down_steps[g]) + 1):
            for t1 in range(T - tau):
                constraints.append(-u[g, t1 + 1] + u[g, t1] <= 1 - u[g, t1 + tau])

        start_cost = float(gencost[g, 1])
        shut_cost = float(gencost[g, 2])
        for t in range(1, T):
            constraints.append(coc[g, t - 1] >= start_cost * (u[g, t] - u[g, t - 1]))
            constraints.append(coc[g, t - 1] >= shut_cost * (u[g, t - 1] - u[g, t]))

        for t in range(T):
            constraints.append(cpower[g, t] >= float(gencost[g, -2]) / t_delta * pg[g, t] + float(gencost[g, -1]) / t_delta * u[g, t])

    nb = bus.shape[0]
    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        G[int(gen[g, GEN_BUS]), g] = 1.0
    PTDF = np.asarray(makePTDF(base_mva, bus, branch), dtype=float)
    ptdf_g = PTDF @ G
    ptdf_pd = PTDF @ np.asarray(pd_data, dtype=float)
    branch_limit = np.asarray(branch[:, RATE_A], dtype=float)
    line_upper_cons: dict[tuple[int, int], object] = {}
    line_lower_cons: dict[tuple[int, int], object] = {}
    for line in range(nl):
        limit = float(branch_limit[line])
        if not np.isfinite(limit) or limit <= 0:
            continue
        for t in range(T):
            flow = ptdf_g[line, :] @ pg[:, t] - float(ptdf_pd[line, t])
            cons_u = flow <= limit
            cons_l = flow >= -limit
            constraints.append(cons_u)
            constraints.append(cons_l)
            line_upper_cons[line, t] = cons_u
            line_lower_cons[line, t] = cons_l

    objective = cp.sum(cpower) + (cp.sum(coc) if T > 1 else 0.0)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.HIGHS, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"LP failed: {problem.status}")

    dual_abs = np.zeros((nl, T), dtype=float)
    for (line, t), cons in line_upper_cons.items():
        val = cons.dual_value
        dual_abs[line, t] += abs(float(np.asarray(val).reshape(-1)[0])) if val is not None else 0.0
    for (line, t), cons in line_lower_cons.items():
        val = cons.dual_value
        dual_abs[line, t] += abs(float(np.asarray(val).reshape(-1)[0])) if val is not None else 0.0
    return np.asarray(u.value, dtype=float), dual_abs


def true_commitment(sample: dict) -> np.ndarray:
    return np.asarray(sample["unit_commitment_matrix"], dtype=float)


def load_activity(activity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(activity_path)
    theta = df[df["kind"].astype(str).eq("bcd_theta")].copy()
    theta["branch_id"] = theta["branch_id"].astype(float).astype(int)
    theta["time_slot"] = theta["time_slot"].astype(float).astype(int)
    for col in ("is_row_active_1e_6", "is_dual_active_1e_7"):
        if theta[col].dtype == object:
            theta[col] = theta[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return theta.groupby(["branch_id", "time_slot"], as_index=False).agg(
        row_active_rate=("is_row_active_1e_6", "mean"),
        dual_active_rate=("is_dual_active_1e_7", "mean"),
        abs_dual_mean=("abs_dual", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-set", type=Path, default=DEFAULT_ACTIVE_SET)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--dual-eps", type=float, default=1e-7)
    parser.add_argument("--ptdf-eps", type=float, default=1e-3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ppc = get_case3lite_uc_ppc()
    ppc_int = ext2int(ppc.copy())
    gen = np.asarray(ppc_int["gen"], dtype=float)
    PTDF = np.asarray(makePTDF(float(ppc_int["baseMVA"]), ppc_int["bus"], ppc_int["branch"]), dtype=float)
    ng = gen.shape[0]
    nl = PTDF.shape[0]
    samples = load_active_set_from_json(str(args.active_set))[: args.num_samples]
    T = np.asarray(samples[0]["pd_data"]).shape[1]

    selected_records = []
    candidate_records = []
    lp_summary = []

    for sidx, sample in enumerate(samples):
        x_lp, line_dual = solve_lp_with_line_duals(ppc, np.asarray(sample["pd_data"], dtype=float), 1.0)
        x_true = true_commitment(sample)
        rounded_bad = (np.rint(x_lp).astype(int) != x_true.astype(int))
        poor_integrality = (x_lp >= 0.3) & (x_lp <= 0.7)
        candidate = rounded_bad | poor_integrality
        lp_summary.append(
            {
                "sample_index": sidx,
                "candidate_unit_time_count": int(candidate.sum()),
                "rounding_bad_count": int(rounded_bad.sum()),
                "poor_integrality_count": int(poor_integrality.sum()),
                "lp_l1_to_true": float(np.abs(x_lp - x_true).sum()),
                "lp_hamming_to_true": int(rounded_bad.sum()),
            }
        )
        for g in range(ng):
            for t in range(T):
                if candidate[g, t]:
                    candidate_records.append(
                        {
                            "sample_index": sidx,
                            "unit_id": g,
                            "time_slot": t,
                            "x_lp": float(x_lp[g, t]),
                            "x_true": float(x_true[g, t]),
                            "rounding_bad": bool(rounded_bad[g, t]),
                            "poor_integrality": bool(poor_integrality[g, t]),
                        }
                    )

        for t in range(T):
            k_t = 0
            line_order = np.argsort(-line_dual[:, t])
            for line in line_order:
                if line_dual[line, t] <= args.dual_eps or k_t >= args.k_max:
                    break
                units = []
                for g in range(ng):
                    bus_idx = int(gen[g, GEN_BUS])
                    if abs(float(PTDF[line, bus_idx])) > args.ptdf_eps and candidate[g, t]:
                        units.append(g)
                if units:
                    selected_records.append(
                        {
                            "sample_index": sidx,
                            "branch_id": int(line),
                            "time_slot": int(t),
                            "selected_units": ",".join(str(u) for u in units),
                            "n_selected_units": int(len(units)),
                            "line_dual_abs": float(line_dual[line, t]),
                            "rank_within_time": int(k_t + 1),
                        }
                    )
                    k_t += 1

    selected_df = pd.DataFrame(selected_records)
    candidate_df = pd.DataFrame(candidate_records)
    lp_df = pd.DataFrame(lp_summary)
    activity_df = load_activity(args.activity)

    if selected_df.empty:
        selected_df = pd.DataFrame(
            columns=[
                "sample_index",
                "branch_id",
                "time_slot",
                "selected_units",
                "n_selected_units",
                "line_dual_abs",
                "rank_within_time",
            ]
        )

    if selected_df.empty:
        selected_summary = pd.DataFrame(
            columns=[
                "branch_id",
                "time_slot",
                "selected_count",
                "selected_rate",
                "mean_line_dual_abs",
                "mean_n_selected_units",
            ]
        )
    else:
        selected_summary = selected_df.groupby(["branch_id", "time_slot"], as_index=False).agg(
            selected_count=("sample_index", "size"),
            selected_rate=("sample_index", lambda s: len(s) / len(samples)),
            mean_line_dual_abs=("line_dual_abs", "mean"),
            mean_n_selected_units=("n_selected_units", "mean"),
        )
    merged = selected_summary.merge(activity_df, on=["branch_id", "time_slot"], how="outer").fillna(0.0)
    merged["selected_any"] = merged["selected_count"] > 0
    merged["active_any"] = merged["dual_active_rate"] > 0

    candidate_summary = candidate_df.groupby(["unit_id", "time_slot"], as_index=False).agg(
        candidate_count=("sample_index", "size"),
        candidate_rate=("sample_index", lambda s: len(s) / len(samples)),
        mean_x_lp=("x_lp", "mean"),
        true_on_rate=("x_true", "mean"),
    )

    prefix = args.output_dir / f"case3lite_algorithm1_k{args.k_max}"
    selected_df.to_csv(prefix.with_name(prefix.name + "_selected_by_sample.csv"), index=False)
    candidate_df.to_csv(prefix.with_name(prefix.name + "_candidate_unit_times.csv"), index=False)
    candidate_summary.to_csv(prefix.with_name(prefix.name + "_candidate_summary.csv"), index=False)
    merged.sort_values(["branch_id", "time_slot"]).to_csv(prefix.with_name(prefix.name + "_selection_vs_activity.csv"), index=False)
    lp_df.to_csv(prefix.with_name(prefix.name + "_lp_summary.csv"), index=False)

    branches = sorted(set(range(nl)) | set(merged["branch_id"].astype(int).tolist()))
    times = list(range(T))

    def mat(value_col: str) -> np.ndarray:
        out = np.zeros((len(branches), len(times)), dtype=float)
        bpos = {b: i for i, b in enumerate(branches)}
        for row in merged.itertuples(index=False):
            out[bpos[int(row.branch_id)], int(row.time_slot)] = float(getattr(row, value_col))
        return out

    selected_mat = mat("selected_rate")
    active_mat = mat("dual_active_rate")
    overlap_mat = ((selected_mat > 0) & (active_mat > 0)).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    im0 = ax0.imshow(selected_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax0.set_title("Algorithm-1 selected rate")
    ax0.set_ylabel("Line / constraint id")
    ax0.set_xlabel("Time")
    ax0.set_yticks(range(len(branches)), labels=[str(b) for b in branches])
    ax0.set_xticks(range(0, T, 2), labels=[str(t) for t in range(0, T, 2)])
    fig.colorbar(im0, ax=ax0, shrink=0.85)

    im1 = ax1.imshow(active_mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax1.set_title("Actual dual-active rate")
    ax1.set_xlabel("Time")
    ax1.set_yticks(range(len(branches)), labels=[str(b) for b in branches])
    ax1.set_xticks(range(0, T, 2), labels=[str(t) for t in range(0, T, 2)])
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    im2 = ax2.imshow(overlap_mat, aspect="auto", cmap="Purples", vmin=0, vmax=1)
    ax2.set_title("Overlap: selected and active")
    ax2.set_ylabel("Line / constraint id")
    ax2.set_xlabel("Time")
    ax2.set_yticks(range(len(branches)), labels=[str(b) for b in branches])
    ax2.set_xticks(range(0, T, 2), labels=[str(t) for t in range(0, T, 2)])
    fig.colorbar(im2, ax=ax2, shrink=0.85)

    if selected_df.empty:
        selected_count_by_time = pd.Series(0, index=times)
    else:
        selected_count_by_time = selected_df.groupby("time_slot").size().reindex(times, fill_value=0)
    active_count_by_time = merged[merged["active_any"]].groupby("time_slot").size().reindex(times, fill_value=0)
    xs = np.arange(T)
    width = 0.38
    ax3.bar(xs - width / 2, selected_count_by_time.values / len(samples), width, label="selected rows/sample", color="#4c78a8")
    ax3.bar(xs + width / 2, active_count_by_time.values, width, label="active rows", color="#e45756")
    ax3.set_title("Selected vs active by time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Count")
    ax3.set_xticks(range(0, T, 2), labels=[str(t) for t in range(0, T, 2)])
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(frameon=False)

    fig_path = prefix.with_name(prefix.name + "_selection_activity_distribution.png")
    fig.savefig(fig_path, dpi=220)
    fig.savefig(prefix.with_name(prefix.name + "_selection_activity_distribution.pdf"))
    plt.close(fig)

    summary = {
        "num_samples": int(len(samples)),
        "k_max": int(args.k_max),
        "dual_eps": float(args.dual_eps),
        "ptdf_eps": float(args.ptdf_eps),
        "mean_candidate_unit_times": float(lp_df["candidate_unit_time_count"].mean()),
        "mean_lp_l1_to_true": float(lp_df["lp_l1_to_true"].mean()),
        "selected_unique_rows": int(merged["selected_any"].sum()),
        "actual_active_unique_rows": int(merged["active_any"].sum()),
        "overlap_unique_rows": int(((merged["selected_any"]) & (merged["active_any"])).sum()),
        "selected_rows_total": int(len(selected_df)),
        "figure_png": str(fig_path),
    }
    with open(prefix.with_name(prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
