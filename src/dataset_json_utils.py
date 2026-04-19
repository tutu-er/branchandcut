# -*- coding: utf-8 -*-
"""Shared helpers for loading v3 active-set JSON (all_samples) used by training scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

try:
    from scenario_utils import has_meaningful_renewable_data, normalize_sample_arrays
except ImportError:
    from src.scenario_utils import has_meaningful_renewable_data, normalize_sample_arrays


def load_v3_active_set_json(
    data_file: Path,
    announce: Callable[[str], None] | None = None,
) -> List[dict]:
    """Load JSON and normalize samples for uc_NN_subproblem v3 (same as legacy load_json_data)."""
    ann = announce or (lambda _msg: None)
    ann(f"Loading active-set file: {data_file.name}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_samples = data.get("all_samples", [])
    if not all_samples:
        raise ValueError("JSON has no samples (all_samples is empty)")

    ann(f"  Raw sample count: {len(all_samples)}")

    has_dataset_renewable = any(
        has_meaningful_renewable_data(sample) for sample in all_samples
    )

    for sample in all_samples:
        if not has_dataset_renewable:
            sample.pop("renewable_data", None)
        normalize_sample_arrays(sample)

    return all_samples
