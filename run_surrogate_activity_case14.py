#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the trained-surrogate dual/activity diagnostic for case14."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import test_surrogate_dual_activity as activity


ROOT = Path(__file__).resolve().parent


def main() -> None:
    output_dir = ROOT / "result" / "figures" / "case14_global_surrogate_solve_stats" / "activity"
    argv = [
        "test_surrogate_dual_activity.py",
        "--case",
        "case14",
        "--active-set-json",
        "result/active_set/active_sets_case14_T24_n600_20260503_222929.json",
        "--model-dir",
        "result/surrogate_models/subproblem_models_case14_20260506_001828",
        "--train-samples",
        "32",
        "--test-samples",
        "16",
        "--output-dir",
        str(output_dir),
    ]
    argv.extend(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = argv
        activity.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
