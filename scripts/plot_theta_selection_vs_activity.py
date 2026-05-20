"""Plot selected theta proxy constraints versus actually active constraints.

For case3lite, the selected theta constraints are recovered from the BCD
checkpoint's theta variable names. Actual activity is read from the saved
final-stage LP activity rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BCD_MODEL = ROOT / "result/bcd_models/bcd_model_case3lite_20260511_021417.pth"
DEFAULT_ACTIVITY = (
    ROOT
    / "result/paper_eval/20260512_183843/raw/runs/case3lite/S01_bcd_theta/raw/figures"
    / "case3lite_global_surrogate_solve_stats/main_activity/main_constraint_activity_by_sample_row.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "result/model_tests/case3lite_main_theta_improvement"


def load_selected_constraints(checkpoint_path: Path) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pattern = re.compile(r"^theta_rhs_branch_(\d+)_time_(\d+)$")
    rows = []
    for name in checkpoint.get("theta_var_names", []):
        match = pattern.match(str(name))
        if match:
            rows.append(
                {
                    "branch_id": int(match.group(1)),
                    "time_slot": int(match.group(2)),
                    "selected": 1,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["branch_id", "time_slot"])


def load_activity(activity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(activity_path)
    theta = df[df["kind"].astype(str).eq("bcd_theta")].copy()
    if theta.empty:
        raise ValueError(f"No bcd_theta rows found in {activity_path}")
    for col in ("branch_id", "time_slot"):
        theta[col] = theta[col].astype(float).astype(int)
    for col in ("is_row_active_1e_6", "is_dual_active_1e_7"):
        if theta[col].dtype == object:
            theta[col] = theta[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        theta.groupby(["branch_id", "time_slot"], as_index=False)
        .agg(
            n_samples=("sample_index", "nunique"),
            row_active_rate=("is_row_active_1e_6", "mean"),
            dual_active_rate=("is_dual_active_1e_7", "mean"),
            abs_dual_mean=("abs_dual", "mean"),
            abs_slack_mean=("abs_row_slack", "mean"),
        )
    )


def load_selected_constraints_from_activity(activity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(activity_path)
    theta = df[df["kind"].astype(str).eq("bcd_theta")].copy()
    if theta.empty:
        raise ValueError(f"No bcd_theta rows found in {activity_path}")
    for col in ("branch_id", "time_slot"):
        theta[col] = theta[col].astype(float).astype(int)
    out = theta[["branch_id", "time_slot"]].drop_duplicates().copy()
    out["selected"] = 1
    return out


def matrix_from_rows(df: pd.DataFrame, value_col: str, branches: list[int], times: list[int]) -> np.ndarray:
    mat = np.full((len(branches), len(times)), np.nan, dtype=float)
    b_index = {b: i for i, b in enumerate(branches)}
    t_index = {t: i for i, t in enumerate(times)}
    for row in df.itertuples(index=False):
        b = int(getattr(row, "branch_id"))
        t = int(getattr(row, "time_slot"))
        if b in b_index and t in t_index:
            mat[b_index[b], t_index[t]] = float(getattr(row, value_col))
    return mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcd-model", type=Path, default=DEFAULT_BCD_MODEL)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-name", default="case3lite")
    parser.add_argument(
        "--selected-source",
        choices=("checkpoint", "activity"),
        default="checkpoint",
        help="Use checkpoint theta names or activity rows as the selected constraint set.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.selected_source == "activity":
        selected = load_selected_constraints_from_activity(args.activity)
    else:
        selected = load_selected_constraints(args.bcd_model)
    activity = load_activity(args.activity)
    merged = selected.merge(activity, on=["branch_id", "time_slot"], how="left")
    merged[["row_active_rate", "dual_active_rate", "abs_dual_mean", "abs_slack_mean"]] = merged[
        ["row_active_rate", "dual_active_rate", "abs_dual_mean", "abs_slack_mean"]
    ].fillna(0.0)
    merged["active_any"] = merged["dual_active_rate"] > 0

    branches = sorted(merged["branch_id"].unique().astype(int).tolist())
    times = list(range(int(merged["time_slot"].max()) + 1))
    selected_mat = matrix_from_rows(merged, "selected", branches, times)
    active_mat = matrix_from_rows(merged, "dual_active_rate", branches, times)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [3.0, 1.25], "width_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    im0 = ax0.imshow(selected_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax0.set_title("Algorithm-selected theta constraints")
    ax0.set_ylabel("Line / constraint id")
    ax0.set_xlabel("Time")
    ax0.set_yticks(range(len(branches)), labels=[str(b) for b in branches])
    ax0.set_xticks(range(0, len(times), 2), labels=[str(t) for t in times[::2]])
    for row in merged[merged["active_any"]].itertuples(index=False):
        ax0.scatter(
            int(row.time_slot),
            branches.index(int(row.branch_id)),
            s=220 * max(float(row.dual_active_rate), 0.05),
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.2,
        )
    ax0.text(
        0.01,
        1.04,
        "red circle = active in final LP",
        transform=ax0.transAxes,
        fontsize=9,
        color="#8c1d18",
    )

    im1 = ax1.imshow(active_mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax1.set_title("Actual dual-active rate")
    ax1.set_xlabel("Time")
    ax1.set_yticks(range(len(branches)), labels=[str(b) for b in branches])
    ax1.set_xticks(range(0, len(times), 2), labels=[str(t) for t in times[::2]])
    cbar = fig.colorbar(im1, ax=ax1, shrink=0.9)
    cbar.set_label("dual active rate")

    selected_count = selected.groupby("time_slot").size().reindex(times, fill_value=0)
    active_count = merged[merged["active_any"]].groupby("time_slot").size().reindex(times, fill_value=0)
    width = 0.38
    xs = np.arange(len(times))
    ax2.bar(xs - width / 2, selected_count.values, width=width, label="selected", color="#4c78a8")
    ax2.bar(xs + width / 2, active_count.values, width=width, label="active", color="#e45756")
    ax2.set_title("Counts by time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Rows")
    ax2.set_xticks(range(0, len(times), 2), labels=[str(t) for t in times[::2]])
    ax2.legend(frameon=False, ncols=2)
    ax2.grid(True, axis="y", alpha=0.25)

    active_sorted = merged.sort_values(["dual_active_rate", "abs_dual_mean"], ascending=False)
    active_sorted = active_sorted[active_sorted["active_any"]]
    labels = [f"L{int(r.branch_id)},t{int(r.time_slot)}" for r in active_sorted.itertuples(index=False)]
    rates = active_sorted["dual_active_rate"].to_numpy(dtype=float)
    ax3.barh(np.arange(len(labels)), rates, color="#e45756")
    ax3.set_yticks(np.arange(len(labels)), labels=labels)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1.05)
    ax3.set_title("Active subset")
    ax3.set_xlabel("Dual-active rate")
    ax3.grid(True, axis="x", alpha=0.25)

    prefix = f"{args.case_name}_theta_selection_vs_activity"
    out_png = args.output_dir / f"{prefix}.png"
    out_pdf = args.output_dir / f"{prefix}.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    summary_csv = args.output_dir / f"{prefix}.csv"
    merged.sort_values(["branch_id", "time_slot"]).to_csv(summary_csv, index=False)
    summary = {
        "selected_constraints": int(len(merged)),
        "active_constraints": int(merged["active_any"].sum()),
        "inactive_selected_constraints": int((~merged["active_any"]).sum()),
        "selected_active_fraction": float(merged["active_any"].mean()),
        "active_constraints_list": [
            {"branch_id": int(r.branch_id), "time_slot": int(r.time_slot), "dual_active_rate": float(r.dual_active_rate)}
            for r in active_sorted.itertuples(index=False)
        ],
        "figure_png": str(out_png),
        "summary_csv": str(summary_csv),
    }
    with open(args.output_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
