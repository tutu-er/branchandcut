#!/usr/bin/env python3
"""Benchmark full UC MILP solve time with Gurobi and train a lightweight runtime predictor.

This uses the **same active-set scenarios** as ``run_test_case*.py``: load ``pd_data``
from ``all_samples``, build ``UnitCommitmentModel`` (``src/uc_gurobipy.py``), call
``.solve()``, and record ``model.Runtime``.

Outputs CSV files ready for ``plot_paper_eval_three_way_runtime.py``:

* ``--write-milp-per-sample``: ``milp_per_sample.csv`` (case, sample_index, runtime_sec, ...)
* ``--write-milp-summary``: ``milp_summary.csv`` (case, median_runtime_sec, mean_runtime_sec)

A small NN (``sklearn.neural_network.MLPRegressor`` behind a ``StandardScaler``) learns
features from flattened ``pd_data`` (optional extra: per-period total demand). Evaluate
mean absolute percentage error / R^2 on a held-out subset.

Example::

  conda run -n poweropt python scripts/benchmark_uc_milp_and_nn_predictor.py ^
    --cases case14 ^
    --sample-range 0:5 ^
    --output-dir result/milp_benchmark_smoke ^
    --write-milp-per-sample ^
    --write-milp-summary

Requires: gurobipy, pypower, numpy, sklearn (PyTorch optional; script uses sklearn MLP.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case_registry import get_case_ppc
from src.dataset_json_utils import load_v3_active_set_json
from src.uc_gurobipy import UnitCommitmentModel

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit(f"gurobipy required: {e}") from e

try:
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


DEFAULT_ACTIVE_SET_JSON: dict[str, str] = {
    "case14": "result/active_set/active_sets_case14_T24_n600_20260503_222929.json",
    "case30lite": "result/active_set/active_sets_case30lite_T24_n500_20260503_233729.json",
    "case3lite": "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json",
}

DEFAULT_SAMPLE_RANGES = {
    "case14": (0, 50),
    "case30lite": (0, 50),
    "case3lite": (0, 100),
}


def parse_sample_range(text: str) -> tuple[int, int]:
    parts = text.split(":", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Expected start:end sample range, got {text!r}")
    start, end = int(parts[0].strip()), int(parts[1].strip())
    if start < 0 or end <= start:
        raise ValueError(f"Invalid range {start}:{end}")
    return start, end


def _apply_range(samples: list[dict], rng: tuple[int, int]) -> list[tuple[int, dict]]:
    start, end = rng
    out: list[tuple[int, dict]] = []
    for i in range(start, min(end, len(samples))):
        out.append((i, samples[i]))
    return out


def _renewable_maybe(sample: dict) -> np.ndarray | None:
    if "renewable_data" not in sample:
        return None
    arr = np.asarray(sample["renewable_data"], dtype=float)
    if arr.size == 0 or not np.any(np.abs(arr) > 1e-12):
        return None
    return arr


def _feature_vector(pd_data: np.ndarray) -> np.ndarray:
    pd_data = np.asarray(pd_data, dtype=np.float64)
    flat = pd_data.reshape(-1)
    col_sum = pd_data.sum(axis=0).astype(np.float64)
    return np.concatenate([flat, col_sum])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _median_mean(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.median(arr)), float(np.mean(arr))


def solve_one_uc_milp(
    ppc: dict,
    pd_data: np.ndarray,
    *,
    t_delta: float,
    renewable_data: np.ndarray | None,
    time_limit_sec: float | None,
    mip_gap: float,
    threads: int | None,
    quiet: bool,
) -> tuple[str, float | None, float | None]:
    uc = UnitCommitmentModel(ppc, pd_data, t_delta, renewable_data=renewable_data, verbose=False)
    m = uc.model
    if quiet:
        m.Params.OutputFlag = 0
    m.Params.MIPGap = float(mip_gap)
    if time_limit_sec is not None:
        m.Params.TimeLimit = float(time_limit_sec)
    if threads is not None:
        m.Params.Threads = int(threads)

    uc.solve()

    runtime = float(getattr(m, "Runtime", float("nan")))
    scode = int(getattr(m, "Status", -1))
    _status_map = {
        int(GRB.LOADED): "LOADED",
        int(GRB.OPTIMAL): "OPTIMAL",
        int(GRB.INFEASIBLE): "INFEASIBLE",
        int(GRB.INF_OR_UNBD): "INF_OR_UNBD",
        int(GRB.UNBOUNDED): "UNBOUNDED",
        int(GRB.CUTOFF): "CUTOFF",
        int(GRB.ITERATION_LIMIT): "ITERATION_LIMIT",
        int(GRB.NODE_LIMIT): "NODE_LIMIT",
        int(GRB.TIME_LIMIT): "TIME_LIMIT",
        int(GRB.SOLUTION_LIMIT): "SOLUTION_LIMIT",
        int(GRB.INTERRUPTED): "INTERRUPTED",
        int(GRB.NUMERIC): "NUMERIC",
        int(GRB.SUBOPTIMAL): "SUBOPTIMAL",
        int(GRB.INPROGRESS): "INPROGRESS",
        int(GRB.USER_OBJ_LIMIT): "USER_OBJ_LIMIT",
        int(GRB.WORK_LIMIT): "WORK_LIMIT",
        int(GRB.MEM_LIMIT): "MEM_LIMIT",
    }
    status_label = _status_map.get(scode, f"status_{scode}")

    obj: float | None
    try:
        if m.Status == GRB.OPTIMAL:
            obj = float(m.objVal)
        elif m.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED) and hasattr(m, "SolCount") and m.SolCount > 0:
            obj = float(m.objVal)
        else:
            obj = None
    except Exception:
        obj = None

    rt = float(runtime) if np.isfinite(runtime) else None
    return status_label, rt, obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="case14",
        help="Comma-separated: case14,case30lite,case3lite",
    )
    parser.add_argument(
        "--active-set-json",
        default=None,
        help="Override JSON path for a single-case run; ignored if multiple cases.",
    )
    parser.add_argument(
        "--sample-range",
        default=None,
        help="Applied to each case slice of all_samples, e.g. 0:50 (default varies by case).",
    )
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--gurobi-time-limit", type=float, default=None, help="Seconds per MILP.")
    parser.add_argument("--mip-gap", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "milp_benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nn-train-ratio",
        type=float,
        default=0.7,
        help="Fraction of pooled (case, sample) rows for training the runtime NN.",
    )
    parser.add_argument("--disable-nn", action="store_true")
    parser.add_argument("--write-milp-per-sample", action="store_true")
    parser.add_argument("--write-milp-summary", action="store_true")
    parser.add_argument("--quiet-gurobi-log", action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    nn_feats: list[np.ndarray] = []
    nn_targets: list[float] = []
    nn_meta: list[tuple[str, int]] = []

    for case_name in cases:
        if len(cases) == 1 and args.active_set_json:
            json_path = Path(args.active_set_json)
            if not json_path.is_file():
                json_path = ROOT / args.active_set_json
        else:
            rel = DEFAULT_ACTIVE_SET_JSON.get(case_name)
            if not rel:
                raise SystemExit(f"No default active-set JSON for {case_name}. Pass --active-set-json.")
            json_path = ROOT / rel
        samples = load_v3_active_set_json(json_path, announce=lambda m: None)
        if args.sample_range is not None:
            sr = parse_sample_range(args.sample_range)
        else:
            sr = DEFAULT_SAMPLE_RANGES.get(case_name)
            if sr is None:
                sr = (0, min(50, len(samples)))
            print(f"{case_name}: using default slice {sr[0]}:{sr[1]} (total scenarios {len(samples)})")

        selected = _apply_range(samples, sr)

        ppc = get_case_ppc(case_name)

        print("=" * 72)
        print(f"{case_name} | MILP benchmarks | slices={sr[0]}:{sr[0] + len(selected)} ({len(selected)} solves)")
        print("=" * 72)

        for local_rank, (sample_index, sample) in enumerate(selected):
            pd_raw = sample.get("pd_data") or sample.get("load_data")
            pd_data = np.asarray(pd_raw, dtype=np.float64)
            ren = _renewable_maybe(sample)

            tag = f"[{case_name} idx={sample_index} #{local_rank + 1}/{len(selected)}]"
            print(f"{tag} shape={pd_data.shape}", flush=True)

            try:
                status_label, rt_sec, obj = solve_one_uc_milp(
                    ppc,
                    pd_data,
                    t_delta=float(args.t_delta),
                    renewable_data=ren,
                    time_limit_sec=args.gurobi_time_limit,
                    mip_gap=float(args.mip_gap),
                    threads=args.threads,
                    quiet=bool(args.quiet_gurobi_log),
                )
            except Exception as ex:
                print(f"{tag} EXCEPTION: {ex}", flush=True)
                row = {
                    "case": case_name,
                    "sample_index": sample_index,
                    "runtime_sec": "",
                    "objective_uc": "",
                    "status_name": "EXCEPTION",
                    "error": repr(ex),
                }
                all_rows.append(row)
                continue

            row = {
                "case": case_name,
                "sample_index": sample_index,
                "runtime_sec": "" if rt_sec is None else f"{rt_sec:.12g}",
                "objective_uc": "" if obj is None else f"{obj:.12g}",
                "status_name": status_label,
                "runtime_sec_float": rt_sec,
            }
            all_rows.append(row)
            if rt_sec is not None and np.isfinite(rt_sec):
                nn_feats.append(_feature_vector(pd_data))
                nn_targets.append(float(rt_sec))
                nn_meta.append((case_name, int(sample_index)))
            speed = f"{rt_sec:.4f}s" if rt_sec is not None else "n/a"
            print(f"{tag} status={status_label} runtime={speed} obj={obj}", flush=True)

    # Persist raw results JSON
    dump_path = output_dir / "milp_benchmark_raw.json"
    serializable = []
    for r in all_rows:
        r2 = {k: v for k, v in r.items() if k != "runtime_sec_float"}
        serializable.append(r2)
    dump_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    per_sample_rows = [
        {
            "case": r["case"],
            "sample_index": r["sample_index"],
            "runtime_sec": r.get("runtime_sec") or "",
            "status_name": r.get("status_name", ""),
            "objective_uc": r.get("objective_uc", ""),
        }
        for r in all_rows
        if str(r.get("status_name")) not in {"EXCEPTION"}
        and _safe_float_like(r.get("runtime_sec_float")) is not None
    ]

    if args.write_milp_per_sample:
        pst = output_dir / "milp_per_sample.csv"
        _write_csv(
            pst,
            per_sample_rows,
            ["case", "sample_index", "runtime_sec", "status_name", "objective_uc"],
        )
        print(f"Wrote {pst}")

    if args.write_milp_summary:
        by_case: dict[str, list[float]] = {}
        for r in all_rows:
            rt = _safe_float_like(r.get("runtime_sec_float"))
            if rt is not None:
                by_case.setdefault(str(r["case"]), []).append(float(rt))
        summ = []
        for case_name in sorted(by_case.keys()):
            med, mu = _median_mean(by_case[case_name])
            summ.append(
                {
                    "case": case_name,
                    "median_runtime_sec": f"{med:.12g}",
                    "mean_runtime_sec": f"{mu:.12g}",
                    "n_success": len(by_case[case_name]),
                }
            )
        sp = output_dir / "milp_summary.csv"
        _write_csv(sp, summ, ["case", "median_runtime_sec", "mean_runtime_sec", "n_success"])
        print(f"Wrote {sp}")

    # --- NN (train on load features vs measured wall time) ---
    feats = nn_feats
    targets = nn_targets
    nn_rows_ready = [{"case": a, "sample_index": b} for (a, b) in nn_meta]

    if (
        len(feats) >= 6
        and not args.disable_nn
        and HAS_SKLEARN
        and float(args.nn_train_ratio) > 0.0
        and float(args.nn_train_ratio) < 1.0
    ):
        X = np.stack(feats, axis=0)
        y = np.asarray(targets, dtype=np.float64)
        idx_train, idx_test = train_test_split(
            np.arange(len(y)),
            test_size=1.0 - float(args.nn_train_ratio),
            random_state=int(args.seed),
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 48),
                        max_iter=800,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=int(args.seed),
                    ),
                ),
            ]
        )
        pipe.fit(X[idx_train], y[idx_train])
        y_hat_tr = pipe.predict(X[idx_train])
        y_hat_te = pipe.predict(X[idx_test])
        metrics = {
            "train_r2": float(r2_score(y[idx_train], y_hat_tr)),
            "test_r2": float(r2_score(y[idx_test], y_hat_te)),
            "train_mape": float(_mape(y[idx_train], y_hat_tr)),
            "test_mape": float(_mape(y[idx_test], y_hat_te)),
            "feature_dim": int(X.shape[1]),
            "train_n": int(len(idx_train)),
            "test_n": int(len(idx_test)),
        }

        preds_path = output_dir / "nn_runtime_predictor_eval.csv"
        print(f"[NN] {metrics}")

        try:
            import joblib
        except ImportError:
            from sklearn.externals import joblib  # type: ignore[attr-defined]

        joblib.dump(pipe, output_dir / "nn_runtime_predictor_pipeline.joblib")
        (output_dir / "nn_runtime_predictor_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pred_rows_fixed = []
        for jpos, idx_i in enumerate(idx_test):
            i = int(idx_i)
            pred_rows_fixed.append(
                {
                    "case": nn_rows_ready[i]["case"],
                    "sample_index": nn_rows_ready[i]["sample_index"],
                    "runtime_actual_sec": y[i],
                    "runtime_predicted_sec": float(y_hat_te[jpos]),
                    "absolute_error_sec": abs(y[i] - float(y_hat_te[jpos])),
                }
            )
        _write_csv(
            preds_path,
            pred_rows_fixed,
            ["case", "sample_index", "runtime_actual_sec", "runtime_predicted_sec", "absolute_error_sec"],
        )
        print(f"Saved NN pipeline → {output_dir / 'nn_runtime_predictor_pipeline.joblib'}")
        print(f"Wrote NN eval → {preds_path}")
    else:
        msg = []
        if not HAS_SKLEARN:
            msg.append("install scikit-learn to enable NN (--disable-nn to silence)")
        if len(feats) < 6:
            msg.append(f"need >=6 MILP timings for NN (have {len(feats)})")
        print("NN skipped:", "; ".join(msg) if msg else "disabled")


def _safe_float_like(val: Any) -> float | None:
    if val is None:
        return None
    try:
        x = float(val)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(a - p) / np.maximum(np.abs(a), eps)))


if __name__ == "__main__":
    main()
