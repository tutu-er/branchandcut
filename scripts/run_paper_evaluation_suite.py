#!/usr/bin/env python3
"""Run and collect the paper evaluation suite.

This script orchestrates the recommended ablation tests for case14,
case30lite, and case3lite:

  1. BCD theta constraints only
  2. BCD theta + zeta constraints
  3. Surrogate sign4 constraints only
  4. Surrogate sign4 + single constraints
  5. Full joint surrogate model

For each run it copies the overwritten run_test outputs into a stable
experiment directory, then writes consolidated data CSV files separately from
paper-ready distribution plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "result" / "paper_eval"


@dataclass(frozen=True)
class CaseConfig:
    name: str
    runner: str
    model_dir: str
    bcd_model: str
    samples: int
    sample_range: str


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    case: CaseConfig
    method: str
    label: str
    mode: str
    bcd_proxy_scope: str | None = None
    surrogate_constraint_scope: str = "all"
    strategy: str = "auto"
    run_fp: bool = False


CASES: tuple[CaseConfig, ...] = (
    CaseConfig(
        name="case14",
        runner="run_test_case14.py",
        model_dir="result/surrogate_models/subproblem_models_case14_20260510_ideal",
        bcd_model="result/bcd_models/bcd_model_case14_20260504_222135.pth",
        samples=50,
        sample_range="0:50",
    ),
    CaseConfig(
        name="case30lite",
        runner="run_test_case30lite.py",
        model_dir="result/surrogate_models/subproblem_models_case30lite_20260510_ideal",
        bcd_model="result/bcd_models/bcd_model_case30lite_20260504_222118.pth",
        samples=50,
        sample_range="0:50",
    ),
    CaseConfig(
        name="case3lite",
        runner="run_test_case3lite.py",
        model_dir="result/surrogate_models/subproblem_models_case3lite_20260510_merge",
        bcd_model="result/bcd_models/bcd_model_case3lite_20260511_021417.pth",
        samples=100,
        sample_range="0:100",
    ),
)


METHOD_LABELS = {
    "bcd_theta": "BCD θ",
    "bcd_theta_zeta": "BCD θ+ζ",
    "surrogate_sign4": "Surr. sign4",
    "surrogate_sign4_single": "Surr. sign4+single",
    "full_joint": "Full",
    "full_joint_fp": "Full+FP",
}

METHOD_ORDER = list(METHOD_LABELS)


def _test_keys(spec: RunSpec) -> set[str]:
    prefix = spec.run_id.split("_", 1)[0].lower()
    number = prefix[1:] if prefix.startswith("s") else prefix
    bare_number = number.lstrip("0") or number
    return {
        prefix,
        number,
        bare_number,
        f"s{bare_number}",
        spec.run_id.lower(),
        spec.method.lower(),
    }


def _run_specs(
    selected_cases: set[str] | None,
    selected_tests: set[str] | None,
    *,
    with_fp: bool,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for case in CASES:
        if selected_cases and case.name not in selected_cases:
            continue
        candidates = [
            RunSpec(
                "S01_bcd_theta",
                case,
                "bcd_theta",
                METHOD_LABELS["bcd_theta"],
                "both",
                bcd_proxy_scope="theta",
            ),
            RunSpec(
                "S02_bcd_theta_zeta",
                case,
                "bcd_theta_zeta",
                METHOD_LABELS["bcd_theta_zeta"],
                "both",
                bcd_proxy_scope="both",
            ),
            RunSpec(
                "S03_surrogate_sign4",
                case,
                "surrogate_sign4",
                METHOD_LABELS["surrogate_sign4"],
                "surrogate",
                surrogate_constraint_scope="sign4",
            ),
            RunSpec(
                "S04_surrogate_sign4_single",
                case,
                "surrogate_sign4_single",
                METHOD_LABELS["surrogate_sign4_single"],
                "surrogate",
                surrogate_constraint_scope="sign4",
                strategy="all_templates_sign4_plus_single",
            ),
            RunSpec(
                "S05_full_joint",
                case,
                "full_joint",
                METHOD_LABELS["full_joint"],
                "both",
                bcd_proxy_scope="both",
            ),
        ]
        if with_fp:
            candidates.append(
                RunSpec(
                    "S05_full_joint_fp",
                    case,
                    "full_joint_fp",
                    METHOD_LABELS["full_joint_fp"],
                    "both",
                    bcd_proxy_scope="both",
                    run_fp=True,
                )
            )
        for spec in candidates:
            if selected_tests and not (_test_keys(spec) & selected_tests):
                continue
            specs.append(spec)
    return specs


def _build_command(spec: RunSpec, with_activity: bool) -> list[str]:
    cmd = [
        sys.executable,
        spec.case.runner,
        "--mode",
        spec.mode,
        "--model-dir",
        spec.case.model_dir,
        "--samples",
        str(spec.case.samples),
        "--sample-range",
        spec.case.sample_range,
        "--surrogate-constraint-scope",
        spec.surrogate_constraint_scope,
        "--strategy",
        spec.strategy,
    ]
    if spec.mode == "both":
        cmd.extend(["--bcd-model", spec.case.bcd_model])
        if spec.bcd_proxy_scope:
            cmd.extend(["--bcd-proxy-scope", spec.bcd_proxy_scope])
    if spec.run_fp:
        cmd.append("--fp")
    if not with_activity:
        cmd.append("--skip-activity")
    return cmd


def _latest_report_for_case(case_name: str, newer_than: float) -> Path | None:
    report_dir = ROOT / "result" / "model_tests" / "run_test_reports"
    candidates = sorted(
        report_dir.glob(f"run_test_{case_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= newer_than - 1e-6:
            return path
    return candidates[0] if candidates else None


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dst = dst_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)


def _copy_case_figures(case_name: str, dst_dir: Path) -> None:
    fig_root = ROOT / "result" / "figures"
    if not fig_root.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        f"{case_name}_*.png",
        f"{case_name}_*.pdf",
        f"lp_*_{case_name}.png",
        f"lp_*_{case_name}.pdf",
        f"fp_*_{case_name}.png",
        f"fp_*_{case_name}.pdf",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for src in fig_root.glob(pattern):
            if src in seen:
                continue
            seen.add(src)
            shutil.copy2(src, dst_dir / src.name)
    for src_dir in fig_root.glob(f"{case_name}_*"):
        if src_dir.is_dir():
            _copy_tree_contents(src_dir, dst_dir / src_dir.name)


def _parse_report(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_fp_success_from_log(log_text: str) -> tuple[int | None, int | None]:
    patterns = [
        r"success\s*=\s*(\d+)\s*/\s*(\d+)",
        r"success\s*[:：]\s*(\d+)\s*/\s*(\d+)",
    ]
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        for m in re.finditer(pattern, log_text, flags=re.IGNORECASE):
            matches.append((int(m.group(1)), int(m.group(2))))
    return matches[-1] if matches else (None, None)


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], preferred_fields: Iterable[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        preferred = list(preferred_fields)
        fields = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
    else:
        fields = list(preferred_fields)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def _safe_int(value) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _lookup_lp_metric_block(lp_summary: dict, *keys: str) -> dict:
    """Resolve lp_compare_summary block; supports legacy *_proxy_lp keys."""
    for key in keys:
        block = lp_summary.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _add_lp_summary_rows(report: dict, spec: RunSpec, rows: list[dict]) -> None:
    lp_summary = report.get("lp_compare_summary") or {}
    n_units = _safe_int(report.get("n_units"))
    horizon = _safe_int(report.get("T"))
    denom = (n_units or 0) * (horizon or 0)
    for method_keys, method_name in (
        (("global_base_lp",), "lp_relaxation"),
        (
            ("global_subproblem_surrogate_lp", "global_subproblem_proxy_lp"),
            "surrogate_lp_summary",
        ),
        (
            ("global_all_surrogate_lp", "global_all_proxy_lp"),
            "all_surrogate_lp_summary",
        ),
        (("unit_subproblem_lp",), "unit_subproblem_lp_summary"),
    ):
        item = _lookup_lp_metric_block(lp_summary, *method_keys)
        hamming = _safe_float(item.get("mean_hamming"))
        rows.append(
            {
                "test_id": spec.run_id,
                "case": spec.case.name,
                "experiment_method": spec.method,
                "experiment_label": spec.label,
                "method": method_name,
                "level": "aggregate",
                "mode": spec.mode,
                "bcd_proxy_scope": spec.bcd_proxy_scope or "",
                "surrogate_constraint_scope": spec.surrogate_constraint_scope,
                "strategy": spec.strategy,
                "n_samples": lp_summary.get("n_samples"),
                "sample_range": report.get("sample_range"),
                "mean_l1": item.get("mean_l1"),
                "mean_hamming": item.get("mean_hamming"),
                "mean_normalized_hamming": (
                    hamming / float(denom) if hamming is not None and denom > 0 else ""
                ),
                "mean_integrality_gap": item.get("mean_integrality_gap"),
                "runtime_sec": "",
                "fp_success": "",
                "fp_total": "",
                "report_elapsed_sec": report.get("elapsed_sec"),
            }
        )


def _collect_run_outputs(
    spec: RunSpec,
    run_dir: Path,
    started_at: float,
    stdout_text: str,
) -> tuple[list[dict], dict, Path | None]:
    raw_dir = run_dir / "raw"
    report_path = _latest_report_for_case(spec.case.name, started_at)
    report = _parse_report(report_path)
    if report_path is not None:
        _copy_if_exists(report_path, raw_dir / "reports" / report_path.name)
        md_path = report_path.with_suffix(".md")
        _copy_if_exists(md_path, raw_dir / "reports" / md_path.name)

    stats_src = ROOT / "result" / "solve_stats" / f"{spec.case.name}_global_surrogate_solve_stats.csv"
    stats_json_src = stats_src.with_suffix(".json")
    stats_dst = raw_dir / "solve_stats" / stats_src.name
    _copy_if_exists(stats_src, stats_dst)
    _copy_if_exists(stats_json_src, raw_dir / "solve_stats" / stats_json_src.name)

    _copy_case_figures(spec.case.name, raw_dir / "figures")

    rows = _read_csv_rows(stats_dst)
    n_units = _safe_int(report.get("n_units"))
    horizon = _safe_int(report.get("T"))
    denom = (n_units or 0) * (horizon or 0)
    enriched: list[dict] = []
    for row in rows:
        hamming = _safe_float(row.get("hamming_to_true"))
        enriched.append(
            {
                **row,
                "test_id": spec.run_id,
                "case": spec.case.name,
                "method": spec.method,
                "method_label": spec.label,
                "level": "sample",
                "run_fp": str(bool(spec.run_fp)),
                "mode": spec.mode,
                "model_dir": spec.case.model_dir,
                "bcd_model": spec.case.bcd_model if spec.mode == "both" else "",
                "bcd_proxy_scope": spec.bcd_proxy_scope or "",
                "surrogate_constraint_scope": spec.surrogate_constraint_scope,
                "strategy": spec.strategy,
                "sample_range": report.get("sample_range", spec.case.sample_range),
                "n_units": n_units if n_units is not None else "",
                "T": horizon if horizon is not None else "",
                "normalized_hamming": (
                    hamming / float(denom) if hamming is not None and denom > 0 else ""
                ),
            }
        )

    fp_success, fp_total = _parse_fp_success_from_log(stdout_text)
    summary = {
        "test_id": spec.run_id,
        "case": spec.case.name,
        "method": spec.method,
        "method_label": spec.label,
        "mode": spec.mode,
        "bcd_proxy_scope": spec.bcd_proxy_scope or "",
        "surrogate_constraint_scope": spec.surrogate_constraint_scope,
        "strategy": spec.strategy,
        "run_fp": str(bool(spec.run_fp)),
        "status": report.get("status", ""),
        "sample_range": report.get("sample_range", spec.case.sample_range),
        "test_samples": report.get("test_samples", spec.case.samples),
        "eval_samples": report.get("eval_samples", ""),
        "elapsed_sec": report.get("elapsed_sec", ""),
        "report_path": str(report_path) if report_path else "",
        "stats_rows": len(enriched),
        "fp_success": fp_success if fp_success is not None else "",
        "fp_total": fp_total if fp_total is not None else "",
    }
    if enriched:
        for key in ("runtime_sec", "integrality_gap", "l1_to_true", "hamming_to_true", "normalized_hamming"):
            values = [_safe_float(row.get(key)) for row in enriched]
            arr = np.asarray([v for v in values if v is not None], dtype=float)
            if arr.size:
                summary[f"mean_{key}"] = float(np.mean(arr))
                summary[f"median_{key}"] = float(np.median(arr))
                summary[f"std_{key}"] = float(np.std(arr))

    aggregate_rows: list[dict] = []
    _add_lp_summary_rows(report, spec, aggregate_rows)
    _write_csv(raw_dir / "lp_compare_summary.csv", aggregate_rows)

    return enriched, summary, report_path


def _boxplot_compatible(ax, data: list, tick_labels_: list[str]) -> None:
    """Matplotlib 3.9+ uses tick_labels=; older versions use labels=."""
    kwargs = dict(showfliers=False, patch_artist=True)
    try:
        ax.boxplot(data, tick_labels=tick_labels_, **kwargs)
    except TypeError:
        ax.boxplot(data, labels=tick_labels_, **kwargs)


def _plot_metric_boxplots(sample_rows: list[dict], output_dir: Path) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if any(r.get("method") == method for r in sample_rows)]
    cases = [case.name for case in CASES if any(r.get("case") == case.name for r in sample_rows)]

    for metric, ylabel, filename in (
        ("normalized_hamming", "Normalized Hamming distance", "normalized_hamming_boxplot"),
        ("integrality_gap", "Mean integrality gap", "integrality_gap_boxplot"),
        ("runtime_sec", "Runtime (s)", "runtime_boxplot"),
    ):
        if not cases:
            continue
        fig, axes = plt.subplots(1, len(cases), figsize=(4.0 * len(cases), 3.6), squeeze=False)
        for ax, case_name in zip(axes.flat, cases):
            data = []
            labels = []
            for method in methods:
                vals = [
                    _safe_float(r.get(metric))
                    for r in sample_rows
                    if r.get("case") == case_name and r.get("method") == method
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    data.append(vals)
                    labels.append(METHOD_LABELS[method])
            if data:
                _boxplot_compatible(ax, data, labels)
                for idx, vals in enumerate(data, start=1):
                    x = np.full(len(vals), idx, dtype=float)
                    jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                    ax.scatter(x + jitter, vals, s=10, alpha=0.45, color="#2F5D7C", zorder=3)
            ax.set_title(case_name, loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(plot_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)


def run_suite(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    if args.timestamped:
        output_dir = output_dir / timestamp
    raw_dir = output_dir / "raw"
    data_dir = output_dir / "data"
    log_dir = raw_dir / "logs"
    run_root = raw_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    selected_cases = {item.strip() for item in args.cases.split(",") if item.strip()} if args.cases else None
    selected_tests = {item.strip().lower() for item in args.tests.split(",") if item.strip()} if args.tests else None
    with_fp = bool(args.with_fp)
    specs = _run_specs(selected_cases, selected_tests, with_fp=with_fp)
    if not specs:
        raise SystemExit("No run specs selected.")

    all_sample_rows: list[dict] = []
    run_summaries: list[dict] = []
    aggregate_rows: list[dict] = []

    for spec in specs:
        run_dir = run_root / spec.case.name / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_command(spec, with_activity=bool(args.with_activity))
        log_path = log_dir / f"{spec.case.name}_{spec.run_id}.log"
        print("=" * 80)
        print(f"{spec.case.name} | {spec.run_id}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        started_at = datetime.now().timestamp()
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env["RUN_TEST_SKIP_ZETA_PLOTS"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=child_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
        shutil.copy2(log_path, run_dir / "run.log")
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            raise SystemExit(f"Run failed ({proc.returncode}): {' '.join(cmd)}")

        sample_rows, summary, _report_path = _collect_run_outputs(
            spec,
            run_dir,
            started_at,
            proc.stdout,
        )
        all_sample_rows.extend(sample_rows)
        run_summaries.append(summary)
        lp_rows = _read_csv_rows(run_dir / "raw" / "lp_compare_summary.csv")
        aggregate_rows.extend(lp_rows)

    if args.dry_run:
        return

    sample_metric_fields = [
        "test_id",
        "case",
        "method",
        "method_label",
        "level",
        "mode",
        "bcd_proxy_scope",
        "surrogate_constraint_scope",
        "strategy",
        "sample_index",
        "status_name",
        "runtime_sec",
        "objective",
        "objective_uc_cost",
        "integrality_gap",
        "l1_to_true",
        "hamming_to_true",
        "normalized_hamming",
        "num_vars",
        "num_constraints",
        "num_nonzeros",
        "subproblem_slack_sum",
        "bcd_slack_sum",
    ]
    _write_csv(data_dir / "sample_metrics.csv", all_sample_rows, preferred_fields=sample_metric_fields)
    _write_csv(raw_dir / "sample_metrics.csv", all_sample_rows, preferred_fields=sample_metric_fields)
    _write_csv(data_dir / "run_summary.csv", run_summaries)
    _write_csv(raw_dir / "run_summary.csv", run_summaries)
    _write_csv(data_dir / "lp_compare_summary.csv", aggregate_rows)
    _write_csv(raw_dir / "lp_compare_summary.csv", aggregate_rows)
    _plot_metric_boxplots(all_sample_rows, output_dir)

    manifest = {
        "created_at": timestamp,
        "output_dir": str(output_dir),
        "with_activity": bool(args.with_activity),
        "runs": [summary for summary in run_summaries],
        "with_fp": bool(with_fp),
        "skip_zeta_plots": True,
        "data_dir": str(data_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 80)
    print(f"sample_metrics: {data_dir / 'sample_metrics.csv'}")
    print(f"run_summary:    {data_dir / 'run_summary.csv'}")
    print(f"figures:        {output_dir / 'figures'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamped", action="store_true", help="Write into output-dir/YYYYMMDD_HHMMSS.")
    parser.add_argument("--cases", default=None, help="Comma-separated case filter, e.g. case14,case30lite.")
    parser.add_argument(
        "--tests",
        default=None,
        help=(
            "Comma-separated test filter: 1,2,3,4,5 or "
            "s01,s02,s03,s04,s05 or method names such as bcd_theta."
        ),
    )
    parser.add_argument("--with-activity", action="store_true", help="Do not pass --skip-activity to case runners.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--with-fp",
        action="store_true",
        help="Append an optional full-joint feasibility-pump run.",
    )
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
