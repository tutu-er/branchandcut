#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble a hybrid case30lite_perturbed surrogate model directory.

The hybrid directory uses the newly trained perturbed unit model for one
generator and reuses old case30lite unit models for all remaining generators.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.case30_uc_data import CASE30LITE_PERTURBED_UNIT_ID  # noqa: E402
from src.case_registry import get_case_ppc  # noqa: E402


def _latest_dir(base: Path, pattern: str, exclude_substrings: tuple[str, ...] = ()) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        p
        for p in base.glob(pattern)
        if p.is_dir() and not any(token in p.name for token in exclude_substrings)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_dir(
    value: str | None,
    pattern: str,
    label: str,
    exclude_substrings: tuple[str, ...] = (),
) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
    else:
        path = _latest_dir(ROOT / "result" / "surrogate_models", pattern, exclude_substrings)
        if path is None:
            raise FileNotFoundError(f"Cannot find latest {label} directory matching {pattern}")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required model file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--old-dir",
        type=str,
        default=None,
        help="Old case30lite model directory. Defaults to latest subproblem_models_case30lite_*.",
    )
    p.add_argument(
        "--new-dir",
        type=str,
        default=None,
        help="New perturbed single-unit model directory. Defaults to latest subproblem_models_case30lite_perturbed_*.",
    )
    p.add_argument("--perturbed-unit", type=int, default=CASE30LITE_PERTURBED_UNIT_ID)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--force", action="store_true", help="Overwrite output files if the directory exists.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    old_dir = _resolve_dir(
        args.old_dir,
        "subproblem_models_case30lite_*",
        "old case30lite",
        exclude_substrings=("case30lite_perturbed",),
    )
    new_dir = _resolve_dir(
        args.new_dir,
        "subproblem_models_case30lite_perturbed_*",
        "new case30lite_perturbed",
        exclude_substrings=("_hybrid_",),
    )

    ppc = get_case_ppc("case30lite_perturbed")
    n_units = int(ppc["gen"].shape[0])
    perturbed_unit = int(args.perturbed_unit)
    if perturbed_unit < 0 or perturbed_unit >= n_units:
        raise ValueError(f"perturbed unit {perturbed_unit} is out of range 0..{n_units - 1}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "result" / "surrogate_models" / f"subproblem_models_case30lite_perturbed_hybrid_{ts}"

    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_sources: dict[str, str] = {}

    _copy_required(new_dir / "dual_predictor.pth", output_dir / "dual_predictor.pth")
    file_sources["dual_predictor.pth"] = str(new_dir)

    for unit_id in range(n_units):
        file_name = f"surrogate_unit_{unit_id}.pth"
        src_dir = new_dir if unit_id == perturbed_unit else old_dir
        _copy_required(src_dir / file_name, output_dir / file_name)
        file_sources[file_name] = str(src_dir)

    if _copy_optional(new_dir / "unit_predictor.pth", output_dir / "unit_predictor.pth"):
        file_sources["unit_predictor.pth"] = str(new_dir)
    elif _copy_optional(old_dir / "unit_predictor.pth", output_dir / "unit_predictor.pth"):
        file_sources["unit_predictor.pth"] = str(old_dir)

    for optional_name in ("training_metrics.json", "metrics.json", "config.json"):
        if _copy_optional(new_dir / optional_name, output_dir / optional_name):
            file_sources[optional_name] = str(new_dir)

    metadata = {
        "case_name": "case30lite_perturbed",
        "purpose": "hybrid_generalization_test",
        "created_at": datetime.now().isoformat(),
        "old_case30lite_model_dir": str(old_dir),
        "new_perturbed_model_dir": str(new_dir),
        "perturbed_unit": perturbed_unit,
        "n_units": n_units,
        "file_sources": file_sources,
    }
    (output_dir / "HYBRID_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"Hybrid model directory: {output_dir}")
    print(f"Old model source:        {old_dir}")
    print(f"New unit source:         {new_dir}")
    print(f"Perturbed unit:          {perturbed_unit}")
    print("=" * 72)


if __name__ == "__main__":
    main()
