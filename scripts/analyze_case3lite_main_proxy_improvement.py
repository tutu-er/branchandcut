"""Analyze case3lite LP improvement after adding main theta proxy constraints.

The script compares the plain global LP relaxation with the same LP augmented
only by BCD theta constraints. It writes variable-level improvement statistics
and the corresponding theta constraint coefficient summaries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pypower.ext2int import ext2int  # noqa: E402

from src.case3_uc_data import get_case3lite_uc_ppc  # noqa: E402
from src.feasibility_pump import (  # noqa: E402
    get_feature_vector_from_sample,
    get_sample_net_load,
    normalize_sample_arrays,
    solve_global_LP_relaxation,
    solve_global_LP_relaxation_without_surrogate,
)
from src.uc_NN_BCD import load_active_set_from_json  # noqa: E402


DEFAULT_ACTIVE_SET = ROOT / "result/active_set/active_sets_case3lite_T24_n200_20260328_102856.json"
DEFAULT_BCD_MODEL = ROOT / "result/bcd_models/bcd_model_case3lite_20260511_021417.pth"
DEFAULT_OUTPUT_DIR = ROOT / "result/model_tests/case3lite_main_theta_improvement"


def rebuild_sequential_from_state_dict(state_dict: dict) -> nn.Sequential:
    layers = []
    weight_keys = sorted(
        [k for k in state_dict if k.endswith(".weight")],
        key=lambda k: int(k.split(".")[0]),
    )
    for i, key in enumerate(weight_keys):
        idx = key.split(".")[0]
        out_dim, in_dim = state_dict[key].shape
        layers.append(nn.Linear(int(in_dim), int(out_dim)))
        if i != len(weight_keys) - 1:
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
    net = nn.Sequential(*layers)
    net.load_state_dict(state_dict)
    net.eval()
    return net


class MainThetaAgent:
    """Small agent-like object consumed by solve_global_LP_relaxation."""

    def __init__(self, ppc: dict, samples: list[dict], checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ppc_int = ext2int(ppc.copy())
        self.ng = int(ppc_int["gen"].shape[0])
        self.nl = int(ppc_int["branch"].shape[0])
        self.T = int(np.asarray(samples[0]["pd_data"]).shape[1])
        self.device = torch.device("cpu")
        self.theta_var_names = list(ckpt["theta_var_names"])
        self.zeta_var_names = list(ckpt.get("zeta_var_names", []))
        self.theta_net = rebuild_sequential_from_state_dict(ckpt["theta_net_state_dict"])
        self.zeta_net = rebuild_sequential_from_state_dict(ckpt["zeta_net_state_dict"])
        self.theta_constraint_direction_signs = np.asarray(
            ckpt.get("theta_constraint_direction_signs", np.ones((self.nl, self.T))),
            dtype=float,
        )
        self.theta_values = {name: 0.0 for name in self.theta_var_names}
        self.zeta_values = {name: 0.0 for name in self.zeta_var_names}
        self._current_union_analysis = {
            "union_constraints": self._build_union_constraints_from_theta_names(),
            "union_zeta_constraints": [],
        }

    def _build_union_constraints_from_theta_names(self) -> list[dict]:
        member_pat = re.compile(r"^theta_branch_(\d+)_unit_(\d+)_time_(\d+)$")
        rhs_pat = re.compile(r"^theta_rhs_branch_(\d+)_time_(\d+)$")
        members: dict[tuple[int, int], list[dict]] = {}
        anchors: set[tuple[int, int]] = set()
        for name in self.theta_var_names:
            m = member_pat.match(name)
            if m:
                bid, uid, mt = map(int, m.groups())
                members.setdefault((bid, mt), []).append({"unit_id": uid, "time_index": mt})
                continue
            m = rhs_pat.match(name)
            if m:
                anchors.add(tuple(map(int, m.groups())))
        constraints = []
        for bid, ts in sorted(anchors):
            constraints.append(
                {
                    "branch_id": bid,
                    "time_slot": ts,
                    "constraint_type": "checkpoint_theta",
                    "nonzero_pg_coefficients": members.get((bid, ts), []),
                }
            )
        return constraints

    def _get_expected_feature_dim(self) -> int:
        return int(self.theta_net[0].in_features)

    def _tensor_to_theta_dict(self, theta_tensor) -> dict[str, float]:
        values = theta_tensor.detach().cpu().numpy().reshape(-1)
        return {name: float(val) for name, val in zip(self.theta_var_names, values)}

    def _tensor_to_zeta_dict(self, zeta_tensor) -> dict[str, float]:
        values = zeta_tensor.detach().cpu().numpy().reshape(-1)
        return {name: float(val) for name, val in zip(self.zeta_var_names, values)}

    def _theta_member_time_index(self, constraint_info: dict, coeff_info: dict) -> int:
        return int(coeff_info.get("time_index", constraint_info.get("time_slot", 0)))

    def _theta_var_name(self, branch_id: int, unit_id: int, member_time: int) -> str:
        return f"theta_branch_{branch_id}_unit_{unit_id}_time_{member_time}"

    def _theta_rhs_name(self, branch_id: int, anchor_time: int) -> str:
        return f"theta_rhs_branch_{branch_id}_time_{anchor_time}"

    def _get_theta_constraint_direction(self, branch_id: int, time_slot: int) -> float:
        signs = np.asarray(self.theta_constraint_direction_signs, dtype=float)
        if signs.shape == (self.nl, self.T) and 0 <= branch_id < self.nl and 0 <= time_slot < self.T:
            return float(signs[branch_id, time_slot])
        return 1.0

    def infer_theta_values(self, sample: dict) -> dict[str, float]:
        sample_norm = normalize_sample_arrays(dict(sample))
        features = np.asarray(get_feature_vector_from_sample(sample_norm), dtype=np.float32)
        expected_dim = self._get_expected_feature_dim()
        if features.size != expected_dim:
            features = np.asarray(get_sample_net_load(sample_norm), dtype=np.float32).reshape(-1)
        if features.size != expected_dim:
            raise ValueError(f"feature dim mismatch: got {features.size}, expected {expected_dim}")
        with torch.no_grad():
            out = self.theta_net(torch.tensor(features, dtype=torch.float32).unsqueeze(0))[0]
        return self._tensor_to_theta_dict(out)


def commitment_from_sample(sample: dict, shape: tuple[int, int]) -> np.ndarray:
    for key in ("unit_commitment_matrix", "x_opt", "x", "commitment"):
        if key in sample:
            arr = np.asarray(sample[key], dtype=float)
            if arr.shape == shape:
                return arr
    active_vars = sample.get("active_variables")
    if isinstance(active_vars, list):
        x = np.zeros(shape, dtype=float)
        for item in active_vars:
            if not isinstance(item, dict):
                continue
            g = item.get("unit_id", item.get("g"))
            t = item.get("time_slot", item.get("t"))
            val = item.get("value", item.get("x", 1.0))
            if g is not None and t is not None and 0 <= int(g) < shape[0] and 0 <= int(t) < shape[1]:
                x[int(g), int(t)] = float(val)
        return x
    raise KeyError("cannot extract true commitment matrix from sample")


def describe_theta_form(agent: MainThetaAgent, samples: list[dict], unit_id: int, time_slot: int) -> list[dict]:
    rows = []
    relevant = []
    for ci in agent._current_union_analysis["union_constraints"]:
        bid = int(ci["branch_id"])
        ts = int(ci["time_slot"])
        if ts != time_slot:
            continue
        units = [int(c["unit_id"]) for c in ci.get("nonzero_pg_coefficients", [])]
        if unit_id in units:
            relevant.append((bid, ts, units))

    theta_by_sample = [agent.infer_theta_values(sample) for sample in samples]
    for bid, ts, units in relevant:
        direction = agent._get_theta_constraint_direction(bid, ts)
        coeff_stats = {}
        for uid in units:
            name = agent._theta_var_name(bid, uid, ts)
            vals = np.array([theta.get(name, 0.0) for theta in theta_by_sample], dtype=float)
            coeff_stats[f"x{uid}_mean"] = float(vals.mean())
            coeff_stats[f"x{uid}_std"] = float(vals.std(ddof=0))
            coeff_stats[f"x{uid}_min"] = float(vals.min())
            coeff_stats[f"x{uid}_max"] = float(vals.max())
        rhs_name = agent._theta_rhs_name(bid, ts)
        rhs_vals = np.array([theta.get(rhs_name, 0.0) for theta in theta_by_sample], dtype=float)
        rows.append(
            {
                "branch_id": bid,
                "time_slot": ts,
                "direction": float(direction),
                "rhs_mean": float(rhs_vals.mean()),
                "rhs_std": float(rhs_vals.std(ddof=0)),
                "rhs_min": float(rhs_vals.min()),
                "rhs_max": float(rhs_vals.max()),
                "raw_form": f"direction * (sum(theta_u*x_u) - rhs) <= slack, direction={direction:g}",
                **coeff_stats,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-set", type=Path, default=DEFAULT_ACTIVE_SET)
    parser.add_argument("--bcd-model", type=Path, default=DEFAULT_BCD_MODEL)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ppc = get_case3lite_uc_ppc()
    samples_all = load_active_set_from_json(str(args.active_set))
    samples = samples_all[: max(1, int(args.num_samples))]
    agent = MainThetaAgent(ppc, samples, args.bcd_model)
    lambda_zero = np.zeros((agent.ng, agent.T), dtype=float)

    per_sample = []
    per_variable = []
    activity_rows = []
    x_plain_list = []
    x_theta_list = []
    x_true_list = []

    for sample_index, sample in enumerate(samples):
        x_plain, plain_stats = solve_global_LP_relaxation_without_surrogate(
            ppc, np.asarray(sample["pd_data"], dtype=float), 1.0, return_stats=True
        )
        x_theta, theta_stats = solve_global_LP_relaxation(
            ppc,
            sample,
            1.0,
            {},
            lambda_zero,
            agent=agent,
            surrogate_constraint_scope="none",
            bcd_proxy_scope="theta",
            return_stats=True,
        )
        x_true = commitment_from_sample(sample, x_plain.shape)
        x_plain_list.append(x_plain)
        x_theta_list.append(x_theta)
        x_true_list.append(x_true)

        plain_err = np.abs(x_plain - x_true)
        theta_err = np.abs(x_theta - x_true)
        per_sample.append(
            {
                "sample_index": sample_index,
                "plain_status": plain_stats.get("status_name"),
                "theta_status": theta_stats.get("status_name"),
                "plain_l1": float(plain_err.sum()),
                "theta_l1": float(theta_err.sum()),
                "l1_improvement": float(plain_err.sum() - theta_err.sum()),
                "plain_integrality_gap": float(np.minimum(x_plain, 1.0 - x_plain).sum()),
                "theta_integrality_gap": float(np.minimum(x_theta, 1.0 - x_theta).sum()),
                "plain_rounded_hamming": int(np.sum((x_plain >= 0.5).astype(int) != x_true.astype(int))),
                "theta_rounded_hamming": int(np.sum((x_theta >= 0.5).astype(int) != x_true.astype(int))),
                "theta_stage_name": theta_stats.get("stage_name"),
                "theta_constraints": theta_stats.get("num_bcd_theta_constraints"),
                "theta_runtime_sec": theta_stats.get("runtime_sec"),
                "plain_runtime_sec": plain_stats.get("runtime_sec"),
            }
        )
        for row in theta_stats.get("main_constraint_activity_rows", []) or []:
            out = dict(row)
            out["sample_index"] = sample_index
            activity_rows.append(out)
        for g in range(x_plain.shape[0]):
            for t in range(x_plain.shape[1]):
                per_variable.append(
                    {
                        "sample_index": sample_index,
                        "unit_id": g,
                        "time_slot": t,
                        "true_x": float(x_true[g, t]),
                        "plain_x": float(x_plain[g, t]),
                        "theta_x": float(x_theta[g, t]),
                        "plain_abs_error": float(plain_err[g, t]),
                        "theta_abs_error": float(theta_err[g, t]),
                        "abs_error_improvement": float(plain_err[g, t] - theta_err[g, t]),
                        "plain_round_error": int((x_plain[g, t] >= 0.5) != bool(x_true[g, t] >= 0.5)),
                        "theta_round_error": int((x_theta[g, t] >= 0.5) != bool(x_true[g, t] >= 0.5)),
                    }
                )

    sample_df = pd.DataFrame(per_sample)
    var_long_df = pd.DataFrame(per_variable)
    summary_df = (
        var_long_df.groupby(["unit_id", "time_slot"], as_index=False)
        .agg(
            mean_abs_error_improvement=("abs_error_improvement", "mean"),
            sum_abs_error_improvement=("abs_error_improvement", "sum"),
            mean_plain_abs_error=("plain_abs_error", "mean"),
            mean_theta_abs_error=("theta_abs_error", "mean"),
            mean_plain_x=("plain_x", "mean"),
            mean_theta_x=("theta_x", "mean"),
            true_on_rate=("true_x", "mean"),
            plain_round_error_count=("plain_round_error", "sum"),
            theta_round_error_count=("theta_round_error", "sum"),
        )
        .sort_values(["mean_abs_error_improvement", "sum_abs_error_improvement"], ascending=False)
    )

    top_vars = summary_df.head(12)
    theta_form_rows = []
    for row in top_vars.itertuples(index=False):
        for form in describe_theta_form(agent, samples, int(row.unit_id), int(row.time_slot)):
            form.update({"unit_id": int(row.unit_id), "target_time_slot": int(row.time_slot)})
            theta_form_rows.append(form)
    theta_form_df = pd.DataFrame(theta_form_rows)

    sample_df.to_csv(args.output_dir / "case3lite_main_theta_sample_improvement.csv", index=False)
    var_long_df.to_csv(args.output_dir / "case3lite_main_theta_variable_long.csv", index=False)
    summary_df.to_csv(args.output_dir / "case3lite_main_theta_variable_improvement_summary.csv", index=False)
    theta_form_df.to_csv(args.output_dir / "case3lite_main_theta_top_variable_constraint_forms.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(args.output_dir / "case3lite_main_theta_activity_rows.csv", index=False)

    report = {
        "num_samples": int(len(samples)),
        "active_set": str(args.active_set),
        "bcd_model": str(args.bcd_model),
        "mean_plain_l1": float(sample_df["plain_l1"].mean()),
        "mean_theta_l1": float(sample_df["theta_l1"].mean()),
        "mean_l1_improvement": float(sample_df["l1_improvement"].mean()),
        "mean_plain_integrality_gap": float(sample_df["plain_integrality_gap"].mean()),
        "mean_theta_integrality_gap": float(sample_df["theta_integrality_gap"].mean()),
        "mean_plain_hamming": float(sample_df["plain_rounded_hamming"].mean()),
        "mean_theta_hamming": float(sample_df["theta_rounded_hamming"].mean()),
        "top_variables": top_vars.to_dict(orient="records"),
    }
    with open(args.output_dir / "case3lite_main_theta_improvement_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
