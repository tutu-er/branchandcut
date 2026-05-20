"""Inspect case14 active theta-only main proxy constraint forms.

This utility reads the trained BCD checkpoint directly. It avoids constructing
``Agent_NN_BCD`` so it does not trigger Gurobi-backed initialization solves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scenario_utils import get_feature_vector_from_sample
from src.uc_NN_BCD import build_mlp_with_dropout


ACTIVE_SET = ROOT / "result/active_set/active_sets_case14_T24_n600_20260503_222929.json"
BCD_MODEL = ROOT / "result/bcd_models/bcd_model_case14_20260504_222135.pth"
OUTPUT = (
    ROOT
    / "result/model_tests/case14_theta_only_main_activity/"
    / "case14_active2_theta_constraint_forms_from_checkpoint.json"
)
TARGETS = [(3, 18), (7, 18)]
UNITS = [1, 2, 3, 4]


def main() -> None:
    checkpoint = torch.load(BCD_MODEL, map_location="cpu", weights_only=False)
    theta_names = list(checkpoint["theta_var_names"])
    name_to_idx = {name: idx for idx, name in enumerate(theta_names)}
    hidden_dims = checkpoint.get("nn_hidden_dims") or [24, 48]

    net = build_mlp_with_dropout(336, hidden_dims, len(theta_names), dropout_p=0.1)
    net.load_state_dict(checkpoint["theta_net_state_dict"])
    net.eval()

    data = json.loads(ACTIVE_SET.read_text(encoding="utf-8"))
    samples = data["all_samples"][:20]

    rows = []
    with torch.no_grad():
        for sample_pos, sample in enumerate(samples):
            features = np.asarray(get_feature_vector_from_sample(sample), dtype=np.float32)
            theta = net(torch.tensor(features).unsqueeze(0))[0].numpy()
            for branch_id, time_slot in TARGETS:
                terms = []
                for unit_id in UNITS:
                    name = f"theta_branch_{branch_id}_unit_{unit_id}_time_{time_slot}"
                    terms.append(
                        {
                            "unit_id": unit_id,
                            "time_index": time_slot,
                            "name": name,
                            "coeff": float(theta[name_to_idx[name]]),
                        }
                    )
                rhs_name = f"theta_rhs_branch_{branch_id}_time_{time_slot}"
                rows.append(
                    {
                        "sample_pos": sample_pos,
                        "sample_id": sample.get("sample_id", sample_pos),
                        "branch_id": branch_id,
                        "time_slot": time_slot,
                        "direction": 1.0,
                        "sense": "<=",
                        "terms": terms,
                        "rhs_name": rhs_name,
                        "rhs": float(theta[name_to_idx[rhs_name]]),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")

    for branch_id, time_slot in TARGETS:
        selected = [
            row for row in rows if row["branch_id"] == branch_id and row["time_slot"] == time_slot
        ]
        print(f"\ntheta_surr_{branch_id}_{time_slot}: sum coeff*x <= rhs")
        for unit_id in UNITS:
            values = np.array(
                [
                    next(term["coeff"] for term in row["terms"] if term["unit_id"] == unit_id)
                    for row in selected
                ],
                dtype=float,
            )
            print(
                f"  x[{unit_id},{time_slot}] "
                f"mean={values.mean():.9f}, min={values.min():.9f}, "
                f"max={values.max():.9f}, sample0={values[0]:.9f}"
            )
        rhs = np.array([row["rhs"] for row in selected], dtype=float)
        print(
            f"  rhs mean={rhs.mean():.9f}, min={rhs.min():.9f}, "
            f"max={rhs.max():.9f}, sample0={rhs[0]:.9f}"
        )


if __name__ == "__main__":
    main()
