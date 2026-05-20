"""Offline analysis: per-(g, t) error rate of theta-only LP rounding vs x_true.

Pipeline
========
For every training sample with a recorded ``x_true``::

    1. predict lambda via the trained dual predictor
    2. solve the *theta-only* global LP relaxation
       (``bcd_proxy_scope='theta'``, ``surrogate_constraint_scope='none'``)
    3. round + min-up/down repair the LP solution
    4. compare against x_true bit-by-bit

The aggregated error rate per (g, t) is dumped as JSON so that the online
recovery scheme can target the historically error-prone bits for explicit
flipping (see ``recover_via_theta_flip``).

Usage
-----
.. code-block:: powershell

    python scripts/build_history_error_bit_map.py `
        --case case14 `
        --model-dir result/surrogate_models/subproblem_models_case14_20260510_ideal `
        --bcd-model result/bcd_models/bcd_model_case14_20260504_222135.pth `
        --max-samples 100 `
        --output result/fp_diagnostics/history_error_bit_map_case14.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_model_fp_diagnostics import (  # noqa: E402
    _ensure_runtime_imports,
    _latest_active_set,
    _latest_model_dir,
    _load_bcd_agent,
    _resolve_path,
)
from scripts.compare_tailored_vs_vanilla_fp import _extract_x_true  # noqa: E402


def _parse_sample_slice(samples: List[dict], start: int, max_n: int) -> List[dict]:
    if not samples:
        return []
    start = max(0, int(start))
    end = min(len(samples), start + max(1, int(max_n)))
    return samples[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--active-sets", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--bcd-model", required=True,
                        help="Trained BCD agent .pth path (required for theta proxy rows).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K most error-prone bits to retain in output.")
    parser.add_argument("--output", default=None,
                        help="JSON output path; default → "
                             "result/fp_diagnostics/history_error_bit_map_<case>_<ts>.json")
    args = parser.parse_args()

    _ensure_runtime_imports()
    from scripts import collect_model_fp_diagnostics as cdiag
    from src.feasibility_pump import (
        _repair_min_up_down_heuristic,
        check_commitment_logic_feasibility,
        round_to_integer,
        solve_global_LP_relaxation,
    )

    active_set_path = _resolve_path(args.active_sets, _latest_active_set(args.case))
    model_dir = _resolve_path(args.model_dir, _latest_model_dir(args.case))
    bcd_model_path = _resolve_path(args.bcd_model, None)

    print(f"[error-map] case={args.case}")
    print(f"[error-map] active_sets={active_set_path}")
    print(f"[error-map] model_dir={model_dir}")
    print(f"[error-map] bcd_model={bcd_model_path}")

    ppc = cdiag.get_case_ppc(args.case)
    all_samples = cdiag.load_v3_active_set_json(
        active_set_path, announce=lambda msg: print(f"[data] {msg}", flush=True)
    )
    selected = _parse_sample_slice(all_samples, args.start, args.max_samples)
    if not selected:
        raise ValueError("No samples selected")

    T_delta = float(args.t_delta)
    dual_predictor, trainers = cdiag.load_trained_models(
        ppc, all_samples, T_delta, str(model_dir),
        unit_ids=None, case_name=args.case, skip_initial_solve=True,
    )
    agent = _load_bcd_agent(ppc, active_set_path, bcd_model_path, T_delta)

    ng = int(np.asarray(ppc["gen"]).shape[0])

    err_count: Optional[np.ndarray] = None
    sample_count = 0
    per_sample_log: List[Dict[str, Any]] = []

    for local_idx, sample in enumerate(selected, start=1):
        sid = sample.get("sample_id", sample.get("source_sample_id",
                                                  args.start + local_idx - 1))
        pd_data = cdiag.get_sample_net_load(sample)
        T = int(pd_data.shape[1])
        if err_count is None:
            err_count = np.zeros((ng, T), dtype=np.int64)

        x_true = _extract_x_true(sample, ng, T)
        if x_true is None:
            print(f"  [skip] sample {sid}: no x_true", flush=True)
            continue
        try:
            lam_val = dual_predictor.predict(sample)
        except Exception as exc:
            print(f"  [skip] sample {sid}: predictor failed: {exc}", flush=True)
            continue

        try:
            x_LP = solve_global_LP_relaxation(
                ppc, sample, T_delta, trainers, lam_val,
                agent=agent,
                surrogate_constraint_scope="none",
                bcd_proxy_scope="theta",
            )
        except Exception as exc:
            print(f"  [skip] sample {sid}: theta-only LP failed: {exc}", flush=True)
            continue

        x_round = round_to_integer(np.asarray(x_LP, dtype=float))
        x_repaired = _repair_min_up_down_heuristic(
            np.asarray(x_round, dtype=int), T_delta, ppc=ppc, unit_ids=None,
        )
        ok_logic, _ = check_commitment_logic_feasibility(x_repaired, ppc, T_delta)
        mismatch = (np.asarray(x_repaired, dtype=int)
                    != np.asarray(x_true, dtype=int)).astype(np.int64)
        err_count = err_count + mismatch
        sample_count += 1
        ham = int(mismatch.sum())
        print(f"  [sample {local_idx}/{len(selected)}] sid={sid}  "
              f"hamming={ham}  logic_ok={ok_logic}", flush=True)
        per_sample_log.append({
            "sample_id": int(sid) if isinstance(sid, (int, np.integer)) else sid,
            "hamming": ham,
            "logic_ok": bool(ok_logic),
        })

    if sample_count == 0 or err_count is None:
        raise RuntimeError("No samples processed successfully; cannot build map")

    err_rate = err_count.astype(float) / float(sample_count)

    flat: List[Dict[str, float]] = []
    for g in range(err_rate.shape[0]):
        for t in range(err_rate.shape[1]):
            flat.append({"g": int(g), "t": int(t),
                         "error_rate": float(err_rate[g, t]),
                         "n_wrong": int(err_count[g, t])})
    flat.sort(key=lambda d: d["error_rate"], reverse=True)
    top_k = flat[: max(1, int(args.top_k))]

    out = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "case": args.case,
            "active_set_path": str(active_set_path),
            "model_dir": str(model_dir),
            "bcd_model": str(bcd_model_path),
            "T_delta": T_delta,
            "n_samples_processed": int(sample_count),
            "n_samples_requested": int(len(selected)),
            "method": "theta_only_LP_round_repair_vs_x_true",
        },
        "shape": [int(err_rate.shape[0]), int(err_rate.shape[1])],
        "error_rate": err_rate.tolist(),
        "n_wrong": err_count.astype(int).tolist(),
        "top_k_bits": top_k,
        "per_sample": per_sample_log,
    }

    if args.output:
        out_path = _resolve_path(args.output, None)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "result" / "fp_diagnostics" / (
            f"history_error_bit_map_{args.case}_{ts}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[done] error-rate map ({err_rate.shape[0]}x{err_rate.shape[1]}) "
          f"from {sample_count} samples -> {out_path}")
    print(f"[done] top-3 error bits:")
    for r in top_k[:3]:
        print(f"  (g={r['g']}, t={r['t']})  rate={r['error_rate']:.3f}  "
              f"(wrong in {r['n_wrong']}/{sample_count} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
