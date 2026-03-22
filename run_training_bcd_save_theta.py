#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BCD only, then save:
- BCD neural-network checkpoint (.pth)
- single theta/zeta alias JSON
- per-sample theta_values_list / zeta_values_list JSON

This script is intentionally minimal and is meant for post-training analysis of
"training-time cached theta/zeta" versus "test-time network forward theta/zeta".
It also supports resuming from an existing BCD checkpoint by setting
`BCD_MODEL_FILE` below.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pypower.case14
import pypower.case30
import pypower.case39
import pypower.case118

from uc_NN_BCD import Agent_NN_BCD, load_active_set_from_json
from case30_uc_data import get_case30_uc_ppc
from mti118_data_loader import load_case118_ppc_with_mti_limits


CASE_NAME = "case30"
T_DELTA = 1.0
MAX_SAMPLES = None
MAX_ITER = 50
NN_EPOCHS = 10
DUAL_DECAY_ROUND = 10
RHO_PRIMAL = None
RHO_DUAL = None
RHO_OPT = None

# Set to an existing .pth path to resume training from a checkpoint.
# Relative paths are resolved from the repo root.
BCD_MODEL_FILE = 'result/bcd_models/bcd_model_case30_20260318_000506.pth'

# Keep this script serial so the saved theta/zeta caches are directly available
# on the returned Agent_NN_BCD instance.
USE_PARALLEL = False

# None -> auto-pick latest active set file for the case.
ACTIVE_SETS_FILE = "result/active_set/active_sets_case30_T24_n340_20260317_152540.json"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pick_data_file(result_dir: Path, case_name: str) -> Path | None:
    candidates = sorted(result_dir.glob(f"active_sets_{case_name}_*.json"))
    if candidates:
        return candidates[-1]
    fallback = sorted(result_dir.glob("active_sets_*.json"))
    if fallback:
        log(f"未找到 {case_name} 专属数据，回退使用: {fallback[-1].name}")
        return fallback[-1]
    return None


def load_case(case_name: str):
    case_map = {
        "case14": pypower.case14.case14,
        "case30": get_case30_uc_ppc,
        "case39": pypower.case39.case39,
        "case118": load_case118_ppc_with_mti_limits,
    }
    if case_name not in case_map:
        raise ValueError(f"unsupported case: {case_name}")
    return case_map[case_name]()


def resolve_data_file() -> Path:
    data_dir = ROOT / "result" / "active_set"
    if ACTIVE_SETS_FILE is not None:
        path = Path(ACTIVE_SETS_FILE)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"active set file not found: {path}")
        return path

    path = pick_data_file(data_dir, CASE_NAME)
    if path is None:
        raise FileNotFoundError(f"no active set json found under {data_dir}")
    return path


def resolve_existing_model_file() -> Path | None:
    if BCD_MODEL_FILE is None:
        return None
    path = Path(BCD_MODEL_FILE)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"BCD model file not found: {path}")
    return path


def build_output_paths(case_name: str, timestamp: str) -> tuple[Path, Path, Path]:
    model_dir = ROOT / "result" / "bcd_models"
    theta_dir = ROOT / "result" / "theta_zeta"
    model_dir.mkdir(parents=True, exist_ok=True)
    theta_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"bcd_model_{case_name}_{timestamp}.pth"
    alias_path = theta_dir / f"theta_zeta_alias_{case_name}_{timestamp}.json"
    list_path = theta_dir / f"theta_zeta_values_list_{case_name}_{timestamp}.json"
    return model_path, alias_path, list_path


def main() -> None:
    if USE_PARALLEL:
        raise RuntimeError("run_training_bcd_save_theta.py currently supports only serial Agent_NN_BCD.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = resolve_data_file()
    existing_model_path = resolve_existing_model_file()
    ppc = load_case(CASE_NAME)

    print("\n" + "=" * 72)
    log("BCD only training with theta/zeta cache export")
    print("=" * 72)
    log(f"case: {CASE_NAME}")
    log(f"data: {data_file}")
    log(f"T_DELTA={T_DELTA}, MAX_ITER={MAX_ITER}, NN_EPOCHS={NN_EPOCHS}, DUAL_DECAY_ROUND={DUAL_DECAY_ROUND}")
    log(f"rho_init: primal={RHO_PRIMAL}, dual={RHO_DUAL}, opt={RHO_OPT}")
    if existing_model_path is not None:
        log(f"resume_from: {existing_model_path}")

    log(f"通过 ActiveSetReader 加载数据: {data_file.name}")
    all_samples = load_active_set_from_json(str(data_file))
    if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
        log(f"截取前 {MAX_SAMPLES} 个样本（共 {len(all_samples)}）")
        all_samples = all_samples[:MAX_SAMPLES]
    log(f"实际用于训练的样本数: {len(all_samples)}")

    model_path, alias_path, list_path = build_output_paths(CASE_NAME, timestamp)

    agent = Agent_NN_BCD(ppc, all_samples, T_DELTA)
    if existing_model_path is not None:
        print("\n" + "=" * 72)
        log("加载已有 BCD 模型并继续训练")
        print("=" * 72)
        agent.load_model_parameters(str(existing_model_path))

    if RHO_PRIMAL is not None:
        agent.rho_primal = float(RHO_PRIMAL)
    if RHO_DUAL is not None:
        agent.rho_dual = float(RHO_DUAL)
    if RHO_OPT is not None:
        agent.rho_opt = float(RHO_OPT)

    print("\n" + "=" * 72)
    log("开始 BCD 训练")
    print("=" * 72)
    agent.iter(max_iter=MAX_ITER, dual_decay_round=DUAL_DECAY_ROUND, nn_epochs=NN_EPOCHS)

    print("\n" + "=" * 72)
    log("保存训练产物")
    print("=" * 72)
    agent.save_model_parameters(str(model_path))
    agent.save_theta_values(str(alias_path))
    agent.save_theta_zeta_values_list(str(list_path))

    meta = {
        "case_name": CASE_NAME,
        "data_file": str(data_file),
        "timestamp": timestamp,
        "t_delta": T_DELTA,
        "max_samples": MAX_SAMPLES,
        "num_samples_used": len(all_samples),
        "max_iter": MAX_ITER,
        "nn_epochs": NN_EPOCHS,
        "dual_decay_round": DUAL_DECAY_ROUND,
        "rho_primal": None if RHO_PRIMAL is None else float(RHO_PRIMAL),
        "rho_dual": None if RHO_DUAL is None else float(RHO_DUAL),
        "rho_opt": None if RHO_OPT is None else float(RHO_OPT),
        "resume_from": str(existing_model_path) if existing_model_path is not None else None,
        "model_path": str(model_path),
        "theta_zeta_alias_path": str(alias_path),
        "theta_zeta_values_list_path": str(list_path),
    }
    meta_path = list_path.with_name(list_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    log("训练完成")
    log(f"model_path: {model_path}")
    log(f"theta_zeta_alias_path: {alias_path}")
    log(f"theta_zeta_values_list_path: {list_path}")
    log(f"meta_path: {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
