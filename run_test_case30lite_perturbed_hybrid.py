#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the hybrid case30lite_perturbed surrogate models."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _latest_path(base: Path, pattern: str, want_dir: bool) -> Path | None:
    if not base.exists():
        return None
    candidates = [p for p in base.glob(pattern) if p.is_dir() == want_dir]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _rel_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-sets", type=str, default=None)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--test-samples", type=int, default=10)
    p.add_argument("--sample-range", type=str, default="0:100")
    p.add_argument("--unit-ids", type=str, default="all", help="'all' or comma-separated ids, e.g. 0,1,2")
    p.add_argument("--no-fp", action="store_true", help="Skip feasibility-pump testing.")
    p.add_argument("--disable-plots", action="store_true")
    return p.parse_args()


def _parse_unit_ids(value: str) -> list[int] | None:
    text = (value or "").strip().lower()
    if text in ("", "all", "none"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = _parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else _latest_path(
        ROOT / "result" / "surrogate_models",
        "subproblem_models_case30lite_perturbed_hybrid_*",
        want_dir=True,
    )
    active_sets = Path(args.active_sets) if args.active_sets else _latest_path(
        ROOT / "result" / "active_set",
        "active_sets_case30lite_perturbed_*.json",
        want_dir=False,
    )

    import run_test as rt

    rt.MODE = "surrogate"
    rt.CASE_NAME = "case30lite_perturbed"
    rt.MODEL_DIR = _rel_or_none(model_dir)
    rt.ACTIVE_SETS_FILE = _rel_or_none(active_sets)
    rt.UNIT_IDS = _parse_unit_ids(args.unit_ids)
    rt.TEST_SAMPLES = max(1, int(args.test_samples))
    rt.TEST_SAMPLES_DEFAULT = rt.TEST_SAMPLES
    rt.SAMPLE_RANGE = args.sample_range
    rt.RUN_FP = not args.no_fp
    rt.RUN_TEST_DISABLE_PLOTS = bool(args.disable_plots)

    print("=" * 72, flush=True)
    print(
        "case30lite_perturbed hybrid test | "
        f"model_dir={rt.MODEL_DIR or 'auto-latest'} | "
        f"active_sets={rt.ACTIVE_SETS_FILE or 'auto-latest'} | "
        f"unit_ids={rt.UNIT_IDS if rt.UNIT_IDS is not None else 'all'} | "
        f"test_samples={rt.TEST_SAMPLES} | fp={rt.RUN_FP}",
        flush=True,
    )
    print("=" * 72, flush=True)

    rt.main()


if __name__ == "__main__":
    main()
