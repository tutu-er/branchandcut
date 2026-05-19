#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small import helper for the tailored FP config generated from diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "result/fp_diagnostics/tailored_fp_config_case14_20260519_174044.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_recover_kwargs(overrides: dict | None = None) -> dict:
    config = load_config()
    kwargs = dict(config.get("recover_integer_solution_kwargs") or {})
    if overrides:
        kwargs.update(overrides)
    return kwargs


if __name__ == "__main__":
    cfg = load_config()
    print("selected_profile=", cfg.get("selected_profile"))
    print(json.dumps(cfg.get("recover_integer_solution_kwargs", {}), indent=2, ensure_ascii=False))
