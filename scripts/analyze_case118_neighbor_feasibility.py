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

from src.feasibility_pump import _feature_distance, check_uc_feasibility
from src.mti118_data_loader import load_case118_ppc_with_mti_limits
from src.scenario_utils import get_feature_vector_from_sample, normalize_sample_arrays


DEFAULT_JSON = ROOT / "result" / "active_set" / "active_sets_case118_T0_n366_20260322_063917.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check feasibility of one case118 sample's optimal commitment on nearby samples."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="active set json path")
    parser.add_argument("--sample-id", type=int, required=True, help="reference sample id")
    parser.add_argument("--neighbors", type=int, default=10, help="number of nearby samples to analyze")
    parser.add_argument("--pool-size", type=int, default=30, help="candidate neighbor pool size before truncation")
    parser.add_argument("--t-delta", type=float, default=1.0, help="time step in hours")
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


def find_nearest_neighbors(samples: list[dict], sample_id: int, pool_size: int) -> list[tuple[int, float]]:
    target = samples[sample_id]
    target_features = get_feature_vector_from_sample(dict(target))

    scored: list[tuple[int, float]] = []
    for idx, sample in enumerate(samples):
        if idx == sample_id:
            continue
        feature_vec = get_feature_vector_from_sample(dict(sample))
        if feature_vec.shape != target_features.shape:
            continue
        distance = _feature_distance(target_features, feature_vec)
        scored.append((idx, distance))

    scored.sort(key=lambda item: item[1])
    return scored[:max(pool_size, 0)]


def main() -> None:
    args = parse_args()
    json_path = args.json if args.json.is_absolute() else (ROOT / args.json)
    samples = load_samples(json_path)

    if not (0 <= args.sample_id < len(samples)):
        raise ValueError(f"sample-id {args.sample_id} out of range [0, {len(samples) - 1}]")

    ppc = load_case118_ppc_with_mti_limits()
    horizon = int(np.asarray(samples[args.sample_id]["pd_data"], dtype=float).shape[1])
    ng = int(ppc["gen"].shape[0])

    x_ref = extract_x_opt(samples[args.sample_id], ng, horizon)
    nearest = find_nearest_neighbors(samples, args.sample_id, args.pool_size)
    nearest = nearest[: args.neighbors]

    print(f"json: {json_path}", flush=True)
    print(f"reference_sample_id: {args.sample_id}", flush=True)
    print(f"neighbors_checked: {len(nearest)}", flush=True)
    print(f"ng={ng}, T={horizon}, t_delta={args.t_delta}", flush=True)

    feasible_count = 0
    rows: list[dict] = []
    for rank, (neighbor_id, distance) in enumerate(nearest, start=1):
        neighbor = samples[neighbor_id]
        neighbor_pd = np.asarray(neighbor["pd_data"], dtype=float)
        neighbor_x = extract_x_opt(neighbor, ng, horizon)

        feasible, reason = check_uc_feasibility(x_ref, ppc, neighbor_pd, args.t_delta)
        hamming = int(np.sum(x_ref.astype(int) != neighbor_x.astype(int)))
        if feasible:
            feasible_count += 1

        row = {
            "rank": rank,
            "neighbor_id": neighbor_id,
            "distance": float(distance),
            "feasible": bool(feasible),
            "reason": reason,
            "hamming_vs_neighbor_opt": hamming,
        }
        rows.append(row)

        print(
            f"[{rank:02d}] neighbor_id={neighbor_id} dist={distance:.6f} "
            f"feasible={feasible} hamming={hamming} reason={reason}",
            flush=True,
        )

    feasibility_rate = 100.0 * feasible_count / max(len(rows), 1)
    mean_distance = float(np.mean([row["distance"] for row in rows])) if rows else float("nan")
    mean_hamming = float(np.mean([row["hamming_vs_neighbor_opt"] for row in rows])) if rows else float("nan")

    print("\nSummary", flush=True)
    print(f"  feasible_neighbors = {feasible_count}/{len(rows)} ({feasibility_rate:.1f}%)", flush=True)
    print(f"  mean_distance      = {mean_distance:.6f}", flush=True)
    print(f"  mean_hamming       = {mean_hamming:.2f}", flush=True)


if __name__ == "__main__":
    main()
