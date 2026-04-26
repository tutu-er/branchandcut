#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subproblem loss snapshot and tuning helper.

Modes are configured below, no CLI args required.

- main_tune: pre-run to a BCD round, stop just before NN-main, save the fixed
  state for c_pg testing, and run several NN-main trials from the same state.
- main_test: load the synchronized main_tune bundle and tune NN-main directly.
- test: load the synchronized main_tune bundle + c_pg snapshot and only train c_pg.
- light_bake: optional full light bake from scratch, writing a c_pg snapshot/bundle.

The old scripts/c_pg_loss_snapshot.py is a compatibility wrapper that calls this file.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# User configuration
# =============================================================================

MODE: str = "main_test"  # "main_tune" | "main_test" | "test" | "light_bake"

CASE: str = "case118"
ACTIVE_SET_JSON: str = "result/active_set/active_sets_case118_T0_n366_20260322_063917.json"
UNIT_ID: int = 0
T_DELTA: float = 1.0
LP_BACKEND: str = "gurobi"
CONSTRAINT_STRATEGY: str | None = None
IGNORE_STARTUP_SHUTDOWN: bool = True

# Shared light/runtime configuration.
LIGHT_PRESET: str = "desktop"  # "desktop" | "server"
MAX_SAMPLES: int = 10
N_WORKERS_UNIT: int = 1
N_WORKERS_SAMPLE: int = 1
DUAL_EPOCHS: int | None = 32

# main_tune: collect NN-main and c_pg data from the same frozen state.
# Current strategy: run 40 complete BCD rounds first, then stop before NN-main
# in round 41.  If you instead want "before NN-main in round 40", set this to 40.
MAIN_TUNE_TARGET_ITER: int = 41  # 1-indexed BCD round; pre_full_iters = target_iter - 1.
MAIN_TUNE_TRIALS: list[dict] = [
    {"name": "e1_lr1e-4", "epochs": 1, "lr": 1e-4, "cost_lr": 5e-6},
    {"name": "e2_lr1e-4", "epochs": 2, "lr": 1e-4, "cost_lr": 5e-6},
    {"name": "e5_lr5e-5", "epochs": 5, "lr": 5e-5, "cost_lr": 5e-6},
]
MAIN_TUNE_BATCH_STRATEGY: str | None = None
MAIN_TUNE_BATCH_SIZE: int | None = None
MAIN_TUNE_SHUFFLE: bool | None = None
MAIN_TUNE_OUT_JSON: str = "result/subproblem_loss_snapshots/main_tune_unit0_n10_pre40_iter41.json"
MAIN_TUNE_C_PG_SNAPSHOT_JSON: str = "result/subproblem_loss_snapshots/c_pg_from_main_tune_unit0_n10_pre40_iter41.json"
MAIN_TUNE_BUNDLE_DIR: str = "result/subproblem_loss_snapshots/main_tune_bundle_unit0_n10_pre40_iter41"

# test: after running main_tune, tune c_pg separately from the synchronized state.
TEST_SNAPSHOT_JSON: str = MAIN_TUNE_C_PG_SNAPSHOT_JSON
TEST_BUNDLE_DIR: str = MAIN_TUNE_BUNDLE_DIR
TEST_C_PG_EPOCHS: int = 0
TEST_PG_COST_SURR_LR: float | None = 1e-4
TEST_C_PG_BATCH_STRATEGY: str | None = "mini-batch"
TEST_C_PG_BATCH_SIZE: int | None = 16
TEST_C_PG_SHUFFLE: bool | None = True
TEST_C_PG_LOG_INTERVAL: int | None = 20
TEST_C_PG_LOG_METRICS: bool = True
TEST_STRICT_HPARAMS: bool = True
TEST_FULL_BASELINE_METRICS: bool = False
TEST_FULL_FINAL_METRICS: bool = False
TEST_LAST_PTH: str = "result/subproblem_loss_snapshots/c_pg_test_after_main_tune_unit0_n10_pre40_iter41.pth"
TEST_RESULT_JSON: str = "result/subproblem_loss_snapshots/c_pg_test_trials_unit0_n10_pre40_iter41.json"

# main_test: tune alpha/beta/gamma/delta/cost from the synchronized state.
TEST_MAIN_LAST_PTH: str = "result/subproblem_loss_snapshots/main_test_after_main_tune_unit0_n10_pre40_iter41.pth"
TEST_MAIN_RESULT_JSON: str = "result/subproblem_loss_snapshots/main_test_trials_unit0_n10_pre40_iter41.json"
TEST_MAIN_FULL_BASELINE_METRICS: bool = False
TEST_MAIN_FULL_FINAL_METRICS: bool = False
# NN-main main_test: balance speed vs quality. Heuristics (see logs):
# - direct MSE fit plateaus in ~80–120 epochs; 360+ is mostly margin.
# - loss_function_differentiable (proxy KKT) is the main per-step cost;
#   keep a small non-zero weight for quality, not 0.02 on every sample every epoch.
# - direct_mae_target is diagnostic only; direct training always runs all configured epochs.
# - For final polish, duplicate this trial with direct_epochs=200, direct_mae_target=0.075, proxy=0.01.
TEST_MAIN_TRIALS: list[dict] = [
    {
        "name": "direct_kkt_proxy_fast_ok",
        "direct_epochs": 120,
        "direct_lr": 2e-3,
        "direct_cost_lr": 8e-4,
        "direct_eta_min_ratio": 0.20,
        "direct_batch_strategy": "mini-batch",
        "direct_batch_size": 16,
        "direct_shuffle": True,
        "direct_loss": "mse",
        "direct_grad_clip": 3.0,
        "direct_log_interval": 10,
        "direct_target_check_interval": 5,
        "direct_mae_target": 0.076,
        "feature_noise_std": 0.005,
        "adam_weight_decay": 1e-5,
        "target_blend": 0.75,
        "dual_eq_weight": 1.0,
        "active_opt_weight": 0.8,
        "inactive_margin_weight": 0.08,
        "inactive_margin": 0.15,
        "anchor_weight": 0.18,
        "coeff_anchor_weight": 0.22,
        "delta_anchor_weight": 0.12,
        "cost_anchor_weight": 0.10,
        "mu_active_threshold": 1e-7,
        "coeff_loss_weight": 1.0,
        "delta_loss_weight": 0.65,
        "cost_loss_weight": 0.85,
        "proxy_kkt_loss_weight": 0.008,
        "fine_epochs": 0,
        "fine_lr": 5e-5,
        "fine_cost_lr": 5e-6,
        "fine_batch_strategy": "mini-batch",
        "fine_batch_size": 16,
        "fine_shuffle": True,
    },
]

# Test-only c_pg tuning.  Existing bundles restore the saved architecture, so
# network-width changes should be tested by re-running main_tune/light_bake.
TEST_C_PG_MODEL_SIZE_FOR_NEW_BUNDLE: str | None = None  # None | "small" | "medium" | "large"
TEST_C_PG_TRIALS: list[dict] = [
    {
        "name": "balanced_mse_target45",
        "reg_deadband": 12.0,
        "reg_scale": 0.02,
        "softbound_weight": 0.02,
        "iter_delta_reg_weight": 0.0,
        "iter_delta_reg_deadband": 12.0,
        "adam_weight_decay": 1e-5,
        "direct_epochs": 420,
        "direct_lr": 6e-3,
        "direct_eta_min_ratio": 0.25,
        "direct_batch_strategy": "mini-batch",
        "direct_batch_size": 16,
        "direct_shuffle": True,
        "direct_loss": "mse",
        "direct_obj_dual_pg_target": 45.0,
        "direct_beta": 1.0,
        "direct_grad_clip": 5.0,
        "direct_log_interval": 25,
        "direct_target_check_interval": 25,
        "feature_noise_std": 0.005,
        "polish_epochs": 0,
        "polish_lr": 1e-3,
        "polish_eta_min_ratio": 0.10,
        "polish_grad_clip": 3.0,
        "polish_log_interval": 25,
        "fine_epochs": TEST_C_PG_EPOCHS,
        "fine_lr": TEST_PG_COST_SURR_LR,
        "fine_batch_strategy": TEST_C_PG_BATCH_STRATEGY,
        "fine_batch_size": TEST_C_PG_BATCH_SIZE,
        "fine_shuffle": TEST_C_PG_SHUFFLE,
    },
]

# optional full bake output, kept for compatibility.
LIGHT_BAKE_MAX_ITER: int | None = 40
LIGHT_BAKE_OUT_JSON: str = "result/c_pg_snapshots/light_bake_unit0_n10.json"
LIGHT_BAKE_BUNDLE_DIR: str = "result/c_pg_snapshots/bake_bundle_unit0_n10"


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_active_set_samples(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("all_samples"), list):
        from src.uc_NN_subproblem import load_active_set_from_json
        return load_active_set_from_json(str(path))
    raise ValueError(f"expected JSON array or object with all_samples in {path}")


def _resolve_active_set_path() -> Path:
    p = _abs(ACTIVE_SET_JSON)
    if p.is_file():
        return p
    raise FileNotFoundError(f"active set json not found: {p}")


def _bootstrap_case118_light() -> None:
    import run_training_case118 as c118

    active_path = _resolve_active_set_path()
    try:
        c118.CASE118_ACTIVE_SET_JSON = str(active_path.relative_to(ROOT))
    except ValueError:
        c118.CASE118_ACTIVE_SET_JSON = str(active_path)
    c118.TRAIN_TARGET = "subproblem_bcd"
    c118.SUBPROBLEM_SOLVE_PRESET = str(LIGHT_PRESET).strip().lower()
    c118.CASE118_SUBPROBLEM_UNIT_IDS = [int(UNIT_ID)]
    c118.SUBPROBLEM_LIGHT_MAX_SAMPLES = max(1, int(MAX_SAMPLES))
    c118.SUBPROBLEM_LIGHT_N_WORKERS_UNIT = max(1, int(N_WORKERS_UNIT))
    c118.SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE = max(1, int(N_WORKERS_SAMPLE))
    if LIGHT_BAKE_MAX_ITER is not None:
        c118.SUBPROBLEM_LIGHT_MAX_ITER = max(1, int(LIGHT_BAKE_MAX_ITER))
    c118.CASE118_USE_UNIT_PREDICTOR = False
    c118.CASE118_UNIT_PREDICTOR_LOAD_PATH = None

    c118._configure_common()
    c118._configure_subproblem_bcd()
    c118._apply_subproblem_light_runtime_overrides()


def _apply_script_overrides() -> None:
    import run_training as rt
    from src.subproblem_lp_solver import normalize_lp_backend

    rt.T_DELTA = float(T_DELTA)
    rt.SUBPROBLEM_LP_BACKEND = normalize_lp_backend(str(LP_BACKEND).strip().lower())
    if CONSTRAINT_STRATEGY is not None:
        rt.SURROGATE_CONSTRAINT_STRATEGY = str(CONSTRAINT_STRATEGY)
    if DUAL_EPOCHS is not None:
        rt.DUAL_EPOCHS = max(1, int(DUAL_EPOCHS))
    if TEST_C_PG_MODEL_SIZE_FOR_NEW_BUNDLE is not None:
        rt.SUBPROBLEM_C_PG_NN_SIZE = str(TEST_C_PG_MODEL_SIZE_FOR_NEW_BUNDLE)


def _prepare_data_and_dual():
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import train_dual_predictor_from_data
    import run_training as rt

    active_path = _resolve_active_set_path()
    all_samples = _load_active_set_samples(active_path)
    n_use = min(len(all_samples), max(1, int(MAX_SAMPLES)))
    all_samples = all_samples[:n_use]
    ppc = get_case_ppc(CASE)
    print(
        f"[{MODE}] case={CASE} unit={UNIT_ID} n_samples={n_use} "
        f"backend={rt.SUBPROBLEM_LP_BACKEND!r}",
        flush=True,
    )
    dual = train_dual_predictor_from_data(
        ppc,
        all_samples,
        T_delta=float(rt.T_DELTA),
        num_epochs=int(rt.DUAL_EPOCHS),
        batch_size=int(rt.DUAL_BATCH_SIZE),
        batch_strategy=str(rt.DUAL_BATCH_STRATEGY),
        shuffle=bool(rt.DUAL_SHUFFLE),
        learning_rate=float(rt.DUAL_LR),
        save_path=None,
        dual_net_variant=str(rt.DUAL_PREDICTOR_NET_VARIANT),
        dual_normalize_targets=bool(rt.DUAL_PREDICTOR_NORMALIZE_TARGETS),
        dual_cosine_loss_weight=float(rt.DUAL_PREDICTOR_COSINE_LOSS_WEIGHT),
        dual_smooth_l1_beta=float(rt.DUAL_PREDICTOR_SMOOTH_L1_BETA),
    )
    return ppc, all_samples, dual, active_path


def _make_trainer(ppc, all_samples: list, dual_predictor):
    import run_training as rt
    from run_training import create_subproblem_trainer, resolve_nn_hidden_dims, SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS
    from src.subproblem_lp_solver import normalize_lp_backend
    from src.uc_NN_subproblem import normalize_constraint_generation_strategy

    _, nn_h = resolve_nn_hidden_dims(rt.SUBPROBLEM_NN_SIZE, SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS, "SUBPROBLEM_NN_SIZE")
    _, cpg_h = resolve_nn_hidden_dims(rt.SUBPROBLEM_C_PG_NN_SIZE, SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS, "SUBPROBLEM_C_PG_NN_SIZE")
    pg_wd = getattr(rt, "SUBPROBLEM_PG_COST_C_PG_ADAM_WD", None)
    return create_subproblem_trainer(
        ppc,
        all_samples,
        float(rt.T_DELTA),
        int(UNIT_ID),
        n_workers=1,
        lambda_predictor=dual_predictor,
        lp_backend=normalize_lp_backend(str(rt.SUBPROBLEM_LP_BACKEND).strip().lower()),
        constraint_generation_strategy=normalize_constraint_generation_strategy(str(rt.SURROGATE_CONSTRAINT_STRATEGY)),
        rho_primal_init=rt.SUBPROBLEM_RHO_PRIMAL_INIT,
        rho_dual_init=rt.SUBPROBLEM_RHO_DUAL_INIT,
        rho_dual_pg_init=rt.SUBPROBLEM_RHO_DUAL_PG_INIT,
        rho_dual_x_init=rt.SUBPROBLEM_RHO_DUAL_X_INIT,
        rho_dual_coc_init=rt.SUBPROBLEM_RHO_DUAL_COC_INIT,
        rho_binary_init=rt.SUBPROBLEM_RHO_BINARY_INIT,
        rho_opt_init=rt.SUBPROBLEM_RHO_OPT_INIT,
        loss_ratio_primal=rt.SUBPROBLEM_LOSS_RATIO_PRIMAL,
        loss_ratio_dual_pg=rt.SUBPROBLEM_LOSS_RATIO_DUAL_PG,
        loss_ratio_dual_x=rt.SUBPROBLEM_LOSS_RATIO_DUAL_X,
        nn_dual_term_interval=rt.SUBPROBLEM_NN_DUAL_TERM_INTERVAL,
        loss_ratio_opt=rt.SUBPROBLEM_LOSS_RATIO_OPT,
        loss_ratio_reg=rt.SUBPROBLEM_LOSS_RATIO_REG,
        subproblem_gamma_base=rt.SUBPROBLEM_GAMMA_BASE,
        mu_lower_bound_init=rt.SUBPROBLEM_MU_DUAL_FLOOR_INIT,
        mu_individual_lower_bound_round=rt.SUBPROBLEM_MU_DUAL_FLOOR_INDIVIDUAL_ROUND,
        mu_group_lower_bound_round=rt.SUBPROBLEM_MU_DUAL_FLOOR_DECAY_ROUND,
        mu_signed_round_interval=rt.SUBPROBLEM_MU_SIGNED_ROUND_INTERVAL,
        mu_sign_hysteresis_rounds=rt.SUBPROBLEM_MU_SIGN_HYSTERESIS_ROUNDS,
        mu_sign_flip_min_share=rt.SUBPROBLEM_MU_SIGN_FLIP_MIN_SHARE,
        x_bound_dual_zero_rounds=rt.SUBPROBLEM_X_BOUND_DUAL_ZERO_ROUNDS,
        subproblem_nn_hidden_dims=nn_h,
        subproblem_c_pg_nn_hidden_dims=cpg_h,
        subproblem_nn_batch_strategy=rt.SUBPROBLEM_NN_BATCH_STRATEGY,
        subproblem_nn_batch_size=rt.SUBPROBLEM_NN_BATCH_SIZE,
        subproblem_nn_shuffle=rt.SUBPROBLEM_NN_SHUFFLE,
        subproblem_nn_learning_rate=rt.SUBPROBLEM_NN_LR,
        subproblem_cost_learning_rate=rt.SUBPROBLEM_X_COST_NN_LR,
        ignore_startup_shutdown_costs=bool(IGNORE_STARTUP_SHUTDOWN),
        subproblem_nn_smooth_abs_eps=rt.SUBPROBLEM_NN_SMOOTH_ABS_EPS,
        pg_cost_nn_epochs=rt.SUBPROBLEM_PG_COST_NN_EPOCHS,
        pg_cost_start_round=rt.SUBPROBLEM_PG_COST_START_ROUND,
        pg_cost_scale_multiplier=rt.SUBPROBLEM_PG_COST_SCALE_MULTIPLIER,
        pg_cost_lr=rt.SUBPROBLEM_PG_COST_LR,
        pg_cost_surr_lr=rt.SUBPROBLEM_PG_COST_SURR_LR,
        pg_cost_reg_deadband=rt.SUBPROBLEM_PG_COST_REG_DEADBAND,
        pg_cost_softbound_weight=rt.SUBPROBLEM_PG_COST_SOFTBOUND_WEIGHT,
        pg_cost_smooth_abs_eps=rt.SUBPROBLEM_PG_COST_SMOOTH_ABS_EPS,
        pg_cost_batch_strategy=rt.SUBPROBLEM_PG_COST_BATCH_STRATEGY,
        pg_cost_batch_size=rt.SUBPROBLEM_PG_COST_BATCH_SIZE,
        pg_cost_shuffle=rt.SUBPROBLEM_PG_COST_SHUFFLE,
        pg_cost_use_sample_weights=rt.SUBPROBLEM_PG_COST_USE_SAMPLE_WEIGHTS,
        pg_cost_sample_weight_power=rt.SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_POWER,
        pg_cost_sample_weight_clip=rt.SUBPROBLEM_PG_COST_SAMPLE_WEIGHT_CLIP,
        pg_cost_single_sample_reg_scale=rt.SUBPROBLEM_PG_COST_SINGLE_SAMPLE_REG_SCALE,
        pg_cost_c_pg_adam_weight_decay=pg_wd,
        pg_block_prox_weight=rt.SUBPROBLEM_PG_BLOCK_PROX_WEIGHT,
        dual_block_prox_weight=rt.SUBPROBLEM_DUAL_BLOCK_PROX_WEIGHT,
        iter_delta_reg_weight=rt.SUBPROBLEM_ITER_DELTA_REG_WEIGHT,
        iter_delta_reg_deadband=rt.SUBPROBLEM_ITER_DELTA_REG_DEADBAND,
        unit_predictor=None,
        use_unit_predictor=False,
        unit_predictor_finetune_lr=rt.UNIT_PREDICTOR_FINETUNE_LR,
        unit_predictor_weight_decay=rt.UNIT_PREDICTOR_WEIGHT_DECAY,
    )


def _clone_state_dict(module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _capture_state(trainer) -> dict:
    def arr(x):
        return None if x is None else np.asarray(x).copy()
    return {
        "surrogate_net": _clone_state_dict(trainer.surrogate_net),
        "iter_number": int(trainer.iter_number),
        "alpha_values": arr(trainer.alpha_values),
        "beta_values": arr(trainer.beta_values),
        "gamma_values": arr(trainer.gamma_values),
        "delta_values": arr(trainer.delta_values),
        "cost_values": arr(trainer.cost_values),
        "pg_cost_values": arr(trainer.pg_cost_values),
        "x": arr(trainer.x),
        "pg": arr(trainer.pg),
        "coc": arr(trainer.coc),
        "cpower": arr(trainer.cpower),
        "mu": arr(trainer.mu),
        "lambda_inherent": copy.deepcopy(trainer.lambda_inherent),
        "_prev_alpha_values": arr(getattr(trainer, "_prev_alpha_values", None)),
        "_prev_beta_values": arr(getattr(trainer, "_prev_beta_values", None)),
        "_prev_gamma_values": arr(getattr(trainer, "_prev_gamma_values", None)),
        "_prev_delta_values": arr(getattr(trainer, "_prev_delta_values", None)),
        "_prev_cost_values": arr(getattr(trainer, "_prev_cost_values", None)),
        "_prev_pg_cost_values": arr(getattr(trainer, "_prev_pg_cost_values", None)),
        "surrogate_direction_signs": arr(trainer._get_surrogate_direction_signs()),
    }


def _restore_state(trainer, state: dict) -> None:
    trainer.surrogate_net.load_state_dict({k: v.to(trainer.device) for k, v in state["surrogate_net"].items()})
    trainer.iter_number = int(state["iter_number"])
    for name in ("alpha_values", "beta_values", "gamma_values", "delta_values", "cost_values", "pg_cost_values", "x", "pg", "coc", "cpower", "mu"):
        setattr(trainer, name, copy.deepcopy(state[name]))
    trainer.lambda_inherent = copy.deepcopy(state["lambda_inherent"])
    for name in ("_prev_alpha_values", "_prev_beta_values", "_prev_gamma_values", "_prev_delta_values", "_prev_cost_values", "_prev_pg_cost_values"):
        setattr(trainer, name, copy.deepcopy(state.get(name)))
    if state.get("surrogate_direction_signs") is not None:
        trainer.surrogate_direction_signs = copy.deepcopy(state["surrogate_direction_signs"])
    trainer._surr_optimizer = None
    trainer._surr_cost_optimizer = None
    trainer._surr_scheduler = None
    trainer._surr_pg_cost_optimizer = None
    trainer._surr_pg_cost_scheduler = None


def _capture_cpg_state(trainer) -> dict:
    """Small test-mode checkpoint: only c_pg head/caches that c_pg trials mutate."""
    def arr(x):
        return None if x is None else np.asarray(x).copy()

    return {
        "pg_cost_net": {k: v.detach().clone() for k, v in trainer.surrogate_net.pg_cost_net.state_dict().items()},
        "pg_input_proj": {k: v.detach().clone() for k, v in trainer.surrogate_net.pg_input_proj.state_dict().items()},
        "pg_res_blocks": {k: v.detach().clone() for k, v in trainer.surrogate_net.pg_res_blocks.state_dict().items()},
        "pg_cost_values": arr(trainer.pg_cost_values),
        "_prev_pg_cost_values": arr(getattr(trainer, "_prev_pg_cost_values", None)),
        "iter_number": int(trainer.iter_number),
    }


def _restore_cpg_state(trainer, state: dict) -> None:
    trainer.surrogate_net.pg_cost_net.load_state_dict(state["pg_cost_net"])
    trainer.surrogate_net.pg_input_proj.load_state_dict(state["pg_input_proj"])
    trainer.surrogate_net.pg_res_blocks.load_state_dict(state["pg_res_blocks"])
    trainer.pg_cost_values = copy.deepcopy(state["pg_cost_values"])
    trainer._prev_pg_cost_values = copy.deepcopy(state.get("_prev_pg_cost_values"))
    trainer.iter_number = int(state["iter_number"])
    trainer._surr_pg_cost_optimizer = None
    trainer._surr_pg_cost_scheduler = None


def _run_primal_dual_only_round(trainer, iter_idx: int) -> None:
    trainer.iter_number = int(iter_idx)
    trainer._sync_surrogate_direction_strategy_state()
    eps = 1e-10
    for sample_id in range(trainer.n_samples):
        pg_sol, x_sol, coc_sol, cpower_sol = trainer.iter_with_primal_block(
            sample_id,
            trainer.alpha_values[sample_id],
            trainer.beta_values[sample_id],
            trainer.gamma_values[sample_id],
            trainer.delta_values[sample_id],
            trainer.cost_values[sample_id],
            trainer.pg_cost_values[sample_id],
        )
        if pg_sol is not None:
            trainer.pg[sample_id] = np.where(np.abs(pg_sol) < eps, 0, pg_sol)
            trainer.x[sample_id] = np.where(np.abs(x_sol) < eps, 0, x_sol)
            trainer.x[sample_id] = np.where(np.abs(trainer.x[sample_id] - 1) < eps, 1, trainer.x[sample_id])
            trainer.coc[sample_id] = np.where(np.abs(coc_sol) < eps, 0, coc_sol)
            trainer.cpower[sample_id] = np.where(np.abs(cpower_sol) < eps, 0, cpower_sol)

    lb_mu = trainer._current_mu_lower_bound_value()
    sign_relax_round = trainer._is_mu_sign_relaxation_round()
    raw_mu = np.zeros((trainer.n_samples, trainer.num_coupling_constraints), dtype=float)
    solved = np.zeros(trainer.n_samples, dtype=bool)
    for sample_id in range(trainer.n_samples):
        li, mu_sol = trainer.iter_with_dual_block(
            sample_id,
            trainer.alpha_values[sample_id],
            trainer.beta_values[sample_id],
            trainer.gamma_values[sample_id],
            trainer.delta_values[sample_id],
            trainer.cost_values[sample_id],
            trainer.pg_cost_values[sample_id],
        )
        if li is not None:
            trainer.lambda_inherent[sample_id] = li
            raw_mu[sample_id] = mu_sol
            solved[sample_id] = True
    if sign_relax_round and np.any(solved):
        trainer.surrogate_direction_signs = trainer._resolve_surrogate_direction_signs_from_mu(raw_mu[solved])
    signs = trainer._get_surrogate_direction_signs()
    for sample_id in range(trainer.n_samples):
        if solved[sample_id]:
            trainer.mu[sample_id] = trainer._finalize_mu_values(raw_mu[sample_id], lb_mu, direction_signs=signs)

    trainer._prev_alpha_values = trainer.alpha_values.copy()
    trainer._prev_beta_values = trainer.beta_values.copy()
    trainer._prev_gamma_values = trainer.gamma_values.copy()
    trainer._prev_delta_values = trainer.delta_values.copy()
    trainer._prev_cost_values = trainer.cost_values.copy()
    trainer._prev_pg_cost_values = trainer.pg_cost_values.copy()


def _metrics(trainer) -> dict:
    m = trainer.cal_nn_logging_components()
    return {k: float(m[k]) for k in ("obj_primal", "obj_dual_x", "obj_opt", "reg_main", "obj_dual_pg", "reg_pg")}


def _apply_cpg_trial_hparams(trainer, trial: dict) -> dict:
    original = {
        "pg_cost_reg_deadband": float(trainer.pg_cost_reg_deadband),
        "_c_pg_reg_loss_scale": float(trainer._c_pg_reg_loss_scale),
        "pg_cost_softbound_weight": float(trainer.pg_cost_softbound_weight),
        "iter_delta_reg_weight": float(trainer.iter_delta_reg_weight),
        "iter_delta_reg_deadband": float(trainer.iter_delta_reg_deadband),
        "pg_cost_c_pg_adam_weight_decay": trainer.pg_cost_c_pg_adam_weight_decay,
    }
    if "reg_deadband" in trial:
        trainer.pg_cost_reg_deadband = max(float(trial["reg_deadband"]), 0.0)
    if "reg_scale" in trial:
        trainer._c_pg_reg_loss_scale = max(float(trial["reg_scale"]), 0.0)
    if "softbound_weight" in trial:
        trainer.pg_cost_softbound_weight = max(float(trial["softbound_weight"]), 0.0)
    if "iter_delta_reg_weight" in trial:
        trainer.iter_delta_reg_weight = max(float(trial["iter_delta_reg_weight"]), 0.0)
    if "iter_delta_reg_deadband" in trial:
        trainer.iter_delta_reg_deadband = max(float(trial["iter_delta_reg_deadband"]), 0.0)
    if "adam_weight_decay" in trial:
        trainer.pg_cost_c_pg_adam_weight_decay = trial["adam_weight_decay"]
    trainer._surr_pg_cost_optimizer = None
    trainer._surr_pg_cost_scheduler = None
    return original


def _cpg_target_for_sample(trainer, sample_id: int) -> np.ndarray:
    """Analytic target for the c_pg head: c_pg[t] = -pg stationarity constant."""
    g = int(trainer.unit_id)
    lam_inh = trainer.lambda_inherent[sample_id]
    if lam_inh is None:
        return np.zeros(int(trainer.T), dtype=np.float32)
    lambda_val = np.asarray(trainer.lambda_vals[sample_id], dtype=np.float64).reshape(-1)
    lam_ru = np.asarray(lam_inh["lambda_ramp_up"], dtype=np.float64).reshape(-1)
    lam_rd = np.asarray(lam_inh["lambda_ramp_down"], dtype=np.float64).reshape(-1)
    target = np.zeros(int(trainer.T), dtype=np.float32)
    a_val = float(trainer.gencost[g, -2] / trainer.T_delta)
    for t in range(int(trainer.T)):
        pg_const = a_val - float(lambda_val[t])
        pg_const -= float(np.asarray(lam_inh["lambda_pg_lower"], dtype=np.float64).reshape(-1)[t])
        pg_const += float(np.asarray(lam_inh["lambda_pg_upper"], dtype=np.float64).reshape(-1)[t])
        if t > 0:
            pg_const += float(lam_ru[t - 1])
            pg_const -= float(lam_rd[t - 1])
        if t < int(trainer.T) - 1:
            pg_const -= float(lam_ru[t])
            pg_const += float(lam_rd[t])
        target[t] = -float(pg_const)
    return target


def _train_cpg_direct_targets(trainer, trial: dict) -> float | None:
    epochs = int(trial.get("direct_epochs", 0) or 0)
    if epochs <= 0:
        return None
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        print(f"[test][{trial.get('name', 'trial')}] skip direct c_pg prefit: {exc!r}", flush=True)
        return None

    _, batch_size, shuffle = trainer._resolve_nn_batch_config(
        batch_size=trial.get("direct_batch_size"),
        batch_strategy=trial.get("direct_batch_strategy", "full-batch"),
        shuffle=trial.get("direct_shuffle", False),
    )
    lr = float(trial.get("direct_lr", 2e-3))
    beta = float(trial.get("direct_beta", 1.0))
    loss_kind = str(trial.get("direct_loss", "huber")).strip().lower()
    grad_clip = float(trial.get("direct_grad_clip", 2.0))
    feature_noise_std = max(float(trial.get("feature_noise_std", 0.0) or 0.0), 0.0)
    obj_dual_pg_target = trial.get("direct_obj_dual_pg_target")
    obj_dual_pg_target = None if obj_dual_pg_target is None else float(obj_dual_pg_target)
    log_interval = int(trial.get("direct_log_interval", 50) or 0)
    target_check_interval = int(trial.get("direct_target_check_interval", log_interval) or 0)
    wd = trial.get("adam_weight_decay", trainer.pg_cost_c_pg_adam_weight_decay)
    wd = 0.0 if wd is None else float(wd)

    sample_features = [
        torch.tensor(
            trainer._extract_features(i),
            dtype=torch.float32,
            device=trainer.device,
        )
        for i in range(trainer.n_samples)
    ]
    targets = [
        torch.tensor(_cpg_target_for_sample(trainer, i), dtype=torch.float32, device=trainer.device)
        for i in range(trainer.n_samples)
    ]
    all_targets = torch.stack(targets, dim=0)
    target_scale = trial.get("direct_target_scale")
    if target_scale is None:
        target_scale_tensor = torch.clamp(torch.mean(torch.abs(all_targets)), min=1.0)
    else:
        target_scale_tensor = torch.as_tensor(
            max(float(target_scale), 1e-6),
            dtype=torch.float32,
            device=trainer.device,
        )
    print(
        f"[test][{trial.get('name', 'trial')}] direct_loss={loss_kind}, "
        f"target_scale={float(target_scale_tensor.detach().cpu().item()):.6f}",
        flush=True,
    )
    full_features_tensor = torch.stack(sample_features, dim=0)
    full_target_tensor = all_targets

    def direct_target_loss() -> torch.Tensor:
        out = trainer._gate_pg_cost_tensor(
            trainer.surrogate_net.forward_pg_cost(full_features_tensor)[:, : trainer.T]
        )
        if loss_kind in ("mse", "l2"):
            err = (out - full_target_tensor) / target_scale_tensor
            loss_val = torch.sum(err * err)
        else:
            loss_val = F.smooth_l1_loss(out, full_target_tensor, beta=beta, reduction="sum")
        if trainer.pg_cost_softbound_weight > 0 and trainer.pg_cost_scale > 0:
            scale = torch.as_tensor(float(trainer.pg_cost_scale), dtype=out.dtype, device=out.device)
            excess = trainer._smooth_relu(torch.abs(out) - scale, eps=trainer.pg_cost_smooth_abs_eps)
            loss_val = loss_val + float(trainer.pg_cost_softbound_weight) * torch.sum(excess * excess)
        return loss_val

    def direct_target_obj_dual_pg() -> float:
        with torch.no_grad():
            out = trainer._gate_pg_cost_tensor(
                trainer.surrogate_net.forward_pg_cost(full_features_tensor)[:, : trainer.T]
            )
            residual = out - full_target_tensor
            return float(torch.sum(torch.abs(residual)).detach().cpu().item())

    def run_adam_phase(
        phase_name: str,
        phase_epochs: int,
        phase_lr: float,
        phase_eta_min_ratio: float,
        phase_grad_clip: float,
        phase_log_interval: int,
    ) -> float | None:
        if phase_epochs <= 0:
            return None
        optimizer = torch.optim.AdamW(trainer._pg_network_parameters(), lr=phase_lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(phase_epochs, 1),
            eta_min=phase_lr * max(float(phase_eta_min_ratio), 0.0),
        )
        phase_last_avg = None
        for epoch in range(phase_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            sample_indices = np.arange(trainer.n_samples, dtype=int)
            if shuffle and trainer.n_samples > 1:
                np.random.shuffle(sample_indices)
            for sample_pos, sample_id in enumerate(sample_indices):
                batch_start = (sample_pos // batch_size) * batch_size
                is_batch_end = (
                    ((sample_pos + 1) % batch_size == 0)
                    or sample_pos == trainer.n_samples - 1
                )
                if not is_batch_end:
                    continue
                batch_ids = sample_indices[batch_start : sample_pos + 1]
                features_tensor = torch.stack([sample_features[int(i)] for i in batch_ids], dim=0)
                if feature_noise_std > 0:
                    features_tensor = features_tensor + torch.randn_like(features_tensor) * feature_noise_std
                target = torch.stack([targets[int(i)] for i in batch_ids], dim=0)
                out = trainer._gate_pg_cost_tensor(
                    trainer.surrogate_net.forward_pg_cost(features_tensor)[:, : trainer.T]
                )
                if loss_kind in ("mse", "l2"):
                    err = (out - target) / target_scale_tensor
                    loss = torch.sum(err * err)
                else:
                    loss = F.smooth_l1_loss(out, target, beta=beta, reduction="sum")
                if trainer.pg_cost_softbound_weight > 0 and trainer.pg_cost_scale > 0:
                    scale = torch.as_tensor(float(trainer.pg_cost_scale), dtype=out.dtype, device=out.device)
                    excess = trainer._smooth_relu(torch.abs(out) - scale, eps=trainer.pg_cost_smooth_abs_eps)
                    loss = loss + float(trainer.pg_cost_softbound_weight) * torch.sum(excess * excess)
                actual_batch_size = max(1, len(batch_ids))
                (loss / actual_batch_size).backward()
                epoch_loss += float(loss.detach().cpu().item())
                torch.nn.utils.clip_grad_norm_(trainer._pg_network_parameters(), max_norm=phase_grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            phase_last_avg = epoch_loss / max(trainer.n_samples, 1)
            do_log = (
                epoch == 0
                or epoch == phase_epochs - 1
                or (phase_log_interval > 0 and (epoch + 1) % phase_log_interval == 0)
            )
            do_target_check = (
                obj_dual_pg_target is not None
                and (
                    do_log
                    or epoch == phase_epochs - 1
                    or (target_check_interval > 0 and (epoch + 1) % target_check_interval == 0)
                )
            )
            metric_obj_dual_pg = None
            if do_log or do_target_check:
                metric_obj_dual_pg = direct_target_obj_dual_pg()
            if do_log:
                print(
                    f"  [Unit-{trainer.unit_id}][direct-c_pg:{phase_name}] epoch {epoch+1:>4}/{phase_epochs}, "
                    f"avg_target_loss={phase_last_avg:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
                    f"fast_obj_dual_pg={metric_obj_dual_pg:.6f}",
                    flush=True,
                )
        return phase_last_avg

    trainer.surrogate_net.train()
    trainer._set_c_pg_training_mode(True)
    last_avg = None
    try:
        last_avg = run_adam_phase(
            "fit",
            epochs,
            lr,
            float(trial.get("direct_eta_min_ratio", 0.05)),
            grad_clip,
            log_interval,
        )
        polish_avg = run_adam_phase(
            "polish",
            int(trial.get("polish_epochs", 0) or 0),
            float(trial.get("polish_lr", lr * 0.25)),
            float(trial.get("polish_eta_min_ratio", 0.10)),
            float(trial.get("polish_grad_clip", grad_clip)),
            int(trial.get("polish_log_interval", log_interval) or 0),
        )
        if polish_avg is not None:
            last_avg = polish_avg
    finally:
        trainer._set_c_pg_training_mode(False)
    trainer._refresh_cached_surrogate_outputs()
    return {
        "avg_target_loss": None if last_avg is None else float(last_avg),
        "fast_obj_dual_pg": direct_target_obj_dual_pg(),
    }


def _x_stationarity_inherent_terms(trainer, sample_id: int) -> np.ndarray:
    """Constant part of the x-stationarity equation, before surrogate terms."""
    from pypower.idx_gen import PMAX, PMIN

    g = int(trainer.unit_id)
    lam_inh = trainer.lambda_inherent[sample_id]
    out = np.zeros(int(trainer.T), dtype=np.float64)
    b_val = float(trainer.gencost[g, -1] / trainer.T_delta)
    pmin_v = float(trainer.gen[g, PMIN])
    pmax_v = float(trainer.gen[g, PMAX])
    ru_v = float(trainer.Ru_all[g])
    rd_v = float(trainer.Rd_all[g])
    ru_co_v = float(trainer.Ru_co_all[g])
    rd_co_v = float(trainer.Rd_co_all[g])
    start_c = 0.0 if trainer.ignore_startup_shutdown_costs else float(trainer.gencost[g, 1])
    shut_c = 0.0 if trainer.ignore_startup_shutdown_costs else float(trainer.gencost[g, 2])
    ton_l = min(4, int(trainer.T))
    toff_l = min(4, int(trainer.T))
    for t in range(int(trainer.T)):
        val = b_val
        if lam_inh is not None:
            val += pmin_v * float(lam_inh["lambda_pg_lower"][t])
            val -= pmax_v * float(lam_inh["lambda_pg_upper"][t])
            lam_ru = lam_inh["lambda_ramp_up"]
            lam_rd = lam_inh["lambda_ramp_down"]
            if t < int(trainer.T) - 1:
                val += (ru_co_v - ru_v) * float(lam_ru[t])
            if t > 0:
                val += (rd_co_v - rd_v) * float(lam_rd[t - 1])

            for tau in range(1, ton_l + 1):
                tau_row = lam_inh["lambda_min_on"][tau - 1]
                for t1 in range(int(trainer.T) - tau):
                    k = float(tau_row[t1])
                    if t == t1 + 1:
                        val += k
                    if t == t1:
                        val -= k
                    if t == t1 + tau:
                        val -= k
            for tau in range(1, toff_l + 1):
                tau_row = lam_inh["lambda_min_off"][tau - 1]
                for t1 in range(int(trainer.T) - tau):
                    k = float(tau_row[t1])
                    if t == t1 + 1:
                        val -= k
                    if t == t1:
                        val += k
                    if t == t1 + tau:
                        val += k

            lam_sc = lam_inh["lambda_start_cost"]
            lam_shc = lam_inh["lambda_shut_cost"]
            if t > 0:
                val += start_c * float(lam_sc[t - 1])
                val -= shut_c * float(lam_shc[t - 1])
            if t < int(trainer.T) - 1:
                val -= start_c * float(lam_sc[t])
                val += shut_c * float(lam_shc[t])
            val += float(lam_inh["lambda_x_upper"][t])
            val -= float(lam_inh["lambda_x_lower"][t])
        out[t] = val
    return out


def _main_target_bounds(trainer, nc: int) -> tuple[np.ndarray, np.ndarray]:
    coeff_scale = float(getattr(trainer.surrogate_net, "coupling_coeff_scale", 2.0))
    cost_scale = float(getattr(trainer.surrogate_net, "x_cost_scale", getattr(trainer, "x_cost_scale", 2.0)))
    delta_base = float(getattr(trainer.surrogate_net, "delta_base", 3.0))
    delta_scale = float(getattr(trainer.surrogate_net, "delta_scale", 2.0))
    if trainer._uses_template_rhs_bases():
        base = np.asarray(trainer.template_rhs_base_vector[:nc], dtype=np.float64)
    else:
        base = np.zeros(nc, dtype=np.float64)
    lower = np.concatenate([
        np.full(3 * nc, -coeff_scale, dtype=np.float64),
        base + delta_base - delta_scale,
        np.full(int(trainer.T), -cost_scale, dtype=np.float64),
    ])
    upper = np.concatenate([
        np.full(3 * nc, coeff_scale, dtype=np.float64),
        base + delta_base + delta_scale,
        np.full(int(trainer.T), cost_scale, dtype=np.float64),
    ])
    return lower, upper


def _build_main_direct_targets(trainer, trial: dict) -> dict[str, np.ndarray | float]:
    """Build ridge-smoothed KKT proxy labels for alpha/beta/gamma/delta/cost."""
    from src.uc_NN_subproblem import iterate_surrogate_constraint_terms

    n = int(trainer.n_samples)
    t_horizon = int(trainer.T)
    nc = int(trainer.num_coupling_constraints)
    n_vars = 4 * nc + t_horizon
    target_blend = float(trial.get("target_blend", 0.75))
    target_blend = min(max(target_blend, 0.0), 1.0)
    mu_threshold = float(trial.get("mu_active_threshold", 1e-7))
    dual_w = float(trial.get("dual_eq_weight", 1.0))
    active_w = float(trial.get("active_opt_weight", 0.8))
    inactive_w = float(trial.get("inactive_margin_weight", 0.0))
    inactive_margin = float(trial.get("inactive_margin", 0.0))
    anchor = float(trial.get("anchor_weight", 0.15))
    coeff_anchor = float(trial.get("coeff_anchor_weight", anchor))
    delta_anchor = float(trial.get("delta_anchor_weight", anchor))
    cost_anchor = float(trial.get("cost_anchor_weight", anchor))

    lower, upper = _main_target_bounds(trainer, nc)
    alphas = np.zeros((n, nc), dtype=np.float32)
    betas = np.zeros((n, nc), dtype=np.float32)
    gammas = np.zeros((n, nc), dtype=np.float32)
    deltas = np.zeros((n, nc), dtype=np.float32)
    costs = np.zeros((n, t_horizon), dtype=np.float32)
    solved_residuals: list[float] = []

    for sample_id in range(n):
        a0 = np.asarray(trainer.alpha_values[sample_id][:nc], dtype=np.float64)
        b0 = np.asarray(trainer.beta_values[sample_id][:nc], dtype=np.float64)
        g0 = np.asarray(trainer.gamma_values[sample_id][:nc], dtype=np.float64)
        d0 = np.asarray(trainer.delta_values[sample_id][:nc], dtype=np.float64)
        c0 = np.asarray(trainer.cost_values[sample_id][:t_horizon], dtype=np.float64)
        y0 = np.concatenate([a0, b0, g0, d0, c0])
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        weights: list[float] = []
        x_val = np.asarray(trainer.x[sample_id], dtype=np.float64).reshape(-1)
        mu_val = np.asarray(trainer.mu[sample_id], dtype=np.float64).reshape(-1)[:nc]
        inherent = _x_stationarity_inherent_terms(trainer, sample_id)
        sensitive_t = trainer.sensitive_timesteps[sample_id]
        offsets_by_k = trainer._constraint_offsets_for_sample(sample_id)
        signs = trainer._get_surrogate_direction_signs(nc)

        for t in range(t_horizon):
            row = np.zeros(n_vars, dtype=np.float64)
            row[4 * nc + t] = 1.0
            for k, ts in enumerate(sensitive_t[:nc]):
                for time_idx, coeff_placeholder in iterate_surrogate_constraint_terms(
                    int(ts), offsets_by_k[k], 1.0, 1.0, 1.0, t_horizon,
                ):
                    if int(time_idx) != t:
                        continue
                    if coeff_placeholder == 1.0:
                        # The iterator order maps to alpha, beta, gamma for offsets 0, 1, 2.
                        if time_idx == int(ts):
                            row[k] += float(mu_val[k])
                        elif time_idx == int(ts) + 1:
                            row[nc + k] += float(mu_val[k])
                        else:
                            row[2 * nc + k] += float(mu_val[k])
            rows.append(row)
            rhs.append(-float(inherent[t]))
            weights.append(max(dual_w, 0.0))

        for k, ts in enumerate(sensitive_t[:nc]):
            row = np.zeros(n_vars, dtype=np.float64)
            for time_idx, _coeff_placeholder in iterate_surrogate_constraint_terms(
                int(ts), offsets_by_k[k], 1.0, 1.0, 1.0, t_horizon,
            ):
                x_coeff = float(signs[k]) * float(x_val[int(time_idx)])
                if int(time_idx) == int(ts):
                    row[k] += x_coeff
                elif int(time_idx) == int(ts) + 1:
                    row[nc + k] += x_coeff
                else:
                    row[2 * nc + k] += x_coeff
            row[3 * nc + k] = -float(signs[k])
            is_active = abs(float(mu_val[k])) > mu_threshold
            if is_active and active_w > 0:
                rows.append(row)
                rhs.append(0.0)
                weights.append(active_w)
            elif inactive_w > 0:
                rows.append(row)
                rhs.append(-inactive_margin)
                weights.append(inactive_w)

        if rows:
            mat = np.vstack(rows)
            vec = np.asarray(rhs, dtype=np.float64)
            w = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 0.0))
            mat_w = mat * w[:, None]
            vec_w = vec * w
            anchor_diag = np.concatenate([
                np.full(3 * nc, max(coeff_anchor, 1e-8), dtype=np.float64),
                np.full(nc, max(delta_anchor, 1e-8), dtype=np.float64),
                np.full(t_horizon, max(cost_anchor, 1e-8), dtype=np.float64),
            ])
            lhs = mat_w.T @ mat_w + np.diag(anchor_diag)
            rhs_vec = mat_w.T @ vec_w + anchor_diag * y0
            try:
                y = np.linalg.solve(lhs, rhs_vec)
            except np.linalg.LinAlgError:
                y = np.linalg.lstsq(lhs, rhs_vec, rcond=None)[0]
            residual = mat @ y - vec
            solved_residuals.append(float(np.mean(np.abs(residual))) if residual.size else 0.0)
        else:
            y = y0.copy()
            solved_residuals.append(0.0)
        y = np.clip(y, lower, upper)
        y = (1.0 - target_blend) * y0 + target_blend * y
        alphas[sample_id] = y[:nc]
        betas[sample_id] = y[nc:2 * nc]
        gammas[sample_id] = y[2 * nc:3 * nc]
        deltas[sample_id] = y[3 * nc:4 * nc]
        costs[sample_id] = y[4 * nc:4 * nc + t_horizon]

    return {
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "deltas": deltas,
        "costs": costs,
        "proxy_residual_mean": float(np.mean(solved_residuals)) if solved_residuals else 0.0,
    }


def _set_main_training_mode(trainer, enabled: bool) -> None:
    """Freeze c_pg while tuning NN-main heads and the x-cost branch."""
    for param in trainer.surrogate_net.parameters():
        param.requires_grad_(not enabled)
    if enabled:
        for param in trainer._main_network_parameters() + trainer._cost_network_parameters():
            param.requires_grad_(True)


def _main_fast_metrics(trainer, sample_features, targets, target_scales) -> dict:
    import torch

    nc = int(trainer.num_coupling_constraints)
    t_horizon = int(trainer.T)
    totals = {"direct_mae": 0.0, "obj_primal": 0.0, "obj_dual_x": 0.0, "obj_opt": 0.0}
    with torch.no_grad():
        for sample_id, features in enumerate(sample_features):
            out = trainer.surrogate_net.forward_main(features.unsqueeze(0))
            a = out[0].squeeze(0)[:nc]
            b = out[1].squeeze(0)[:nc]
            g = out[2].squeeze(0)[:nc]
            d = trainer._postprocess_delta_tensor(out[3].squeeze(0)[:nc])
            c = out[4].squeeze(0)[:t_horizon]
            direct_err = (
                torch.mean(torch.abs((a - targets["alphas"][sample_id]) / target_scales["coeff"]))
                + torch.mean(torch.abs((b - targets["betas"][sample_id]) / target_scales["coeff"]))
                + torch.mean(torch.abs((g - targets["gammas"][sample_id]) / target_scales["coeff"]))
                + torch.mean(torch.abs((d - targets["deltas"][sample_id]) / target_scales["delta"]))
                + torch.mean(torch.abs((c - targets["costs"][sample_id]) / target_scales["cost"]))
            ) / 5.0
            totals["direct_mae"] += float(direct_err.detach().cpu().item())
            _, comps = trainer.loss_function_differentiable(
                sample_id, a, b, g, d, c, trainer.device, return_components=True,
            )
            totals["obj_primal"] += float(comps["obj_primal"].detach().cpu().item())
            totals["obj_dual_x"] += float(comps["obj_dual_x"].detach().cpu().item())
            totals["obj_opt"] += float(comps["obj_opt"].detach().cpu().item())
    denom = max(int(trainer.n_samples), 1)
    totals["direct_mae"] /= denom
    return totals


def _train_main_direct_targets(trainer, trial: dict) -> dict | None:
    epochs = int(trial.get("direct_epochs", 0) or 0)
    if epochs <= 0:
        return None
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        print(f"[main_test][{trial.get('name', 'trial')}] skip direct NN-main prefit: {exc!r}", flush=True)
        return None

    _, batch_size, shuffle = trainer._resolve_nn_batch_config(
        batch_size=trial.get("direct_batch_size"),
        batch_strategy=trial.get("direct_batch_strategy", "full-batch"),
        shuffle=trial.get("direct_shuffle", False),
    )
    nc = int(trainer.num_coupling_constraints)
    t_horizon = int(trainer.T)
    lr = float(trial.get("direct_lr", 2e-3))
    cost_lr = float(trial.get("direct_cost_lr", lr * 0.25))
    eta_min_ratio = float(trial.get("direct_eta_min_ratio", 0.1))
    grad_clip = float(trial.get("direct_grad_clip", 3.0))
    wd = float(trial.get("adam_weight_decay", 0.0) or 0.0)
    loss_kind = str(trial.get("direct_loss", "mse")).strip().lower()
    beta = float(trial.get("direct_beta", 1.0))
    log_interval = int(trial.get("direct_log_interval", 50) or 0)
    target_check_interval = int(trial.get("direct_target_check_interval", log_interval) or 0)
    direct_mae_target = trial.get("direct_mae_target")
    direct_mae_target = None if direct_mae_target is None else float(direct_mae_target)
    feature_noise_std = max(float(trial.get("feature_noise_std", 0.0) or 0.0), 0.0)
    coeff_w = float(trial.get("coeff_loss_weight", 1.0))
    delta_w = float(trial.get("delta_loss_weight", 0.6))
    cost_w = float(trial.get("cost_loss_weight", 0.8))
    proxy_kkt_w = float(trial.get("proxy_kkt_loss_weight", 0.0))

    np_targets = _build_main_direct_targets(trainer, trial)
    sample_features = [
        torch.tensor(trainer._extract_features(i), dtype=torch.float32, device=trainer.device)
        for i in range(trainer.n_samples)
    ]
    targets = {
        key: torch.tensor(np_targets[key], dtype=torch.float32, device=trainer.device)
        for key in ("alphas", "betas", "gammas", "deltas", "costs")
    }
    target_scales = {
        "coeff": torch.clamp(
            torch.mean(torch.abs(torch.cat([targets["alphas"], targets["betas"], targets["gammas"]], dim=1))),
            min=0.25,
        ),
        "delta": torch.clamp(torch.mean(torch.abs(targets["deltas"])), min=1.0),
        "cost": torch.clamp(torch.mean(torch.abs(targets["costs"])), min=0.25),
    }
    print(
        f"[main_test][{trial.get('name', 'trial')}] direct_loss={loss_kind}, "
        f"proxy_residual_mean={np_targets['proxy_residual_mean']:.6f}, "
        f"scales=(coeff={float(target_scales['coeff'].cpu().item()):.4f}, "
        f"delta={float(target_scales['delta'].cpu().item()):.4f}, "
        f"cost={float(target_scales['cost'].cpu().item()):.4f})",
        flush=True,
    )

    def group_loss(pred, target, scale, weight):
        err = (pred - target) / scale
        if loss_kind in ("mse", "l2"):
            return weight * torch.sum(err * err)
        return weight * F.smooth_l1_loss(err, torch.zeros_like(err), beta=beta, reduction="sum")

    trainer.surrogate_net.train()
    _set_main_training_mode(trainer, True)
    optimizer_main = torch.optim.AdamW(trainer._main_network_parameters(), lr=lr, weight_decay=wd)
    optimizer_cost = torch.optim.AdamW(trainer._cost_network_parameters(), lr=cost_lr, weight_decay=wd)
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=max(epochs, 1), eta_min=lr * max(eta_min_ratio, 0.0),
    )
    scheduler_cost = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_cost, T_max=max(epochs, 1), eta_min=cost_lr * max(eta_min_ratio, 0.0),
    )
    last_avg = None
    last_fast = None
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            optimizer_main.zero_grad()
            optimizer_cost.zero_grad()
            sample_indices = np.arange(trainer.n_samples, dtype=int)
            if shuffle and trainer.n_samples > 1:
                np.random.shuffle(sample_indices)
            for sample_pos, sample_id in enumerate(sample_indices):
                batch_start = (sample_pos // batch_size) * batch_size
                is_batch_end = (
                    ((sample_pos + 1) % batch_size == 0)
                    or sample_pos == trainer.n_samples - 1
                )
                if not is_batch_end:
                    continue
                batch_ids = sample_indices[batch_start: sample_pos + 1]
                features_tensor = torch.stack([sample_features[int(i)] for i in batch_ids], dim=0)
                if feature_noise_std > 0:
                    features_tensor = features_tensor + torch.randn_like(features_tensor) * feature_noise_std
                out = trainer.surrogate_net.forward_main(features_tensor)
                pred_a = out[0][:, :nc]
                pred_b = out[1][:, :nc]
                pred_g = out[2][:, :nc]
                pred_d = trainer._postprocess_delta_tensor(out[3][:, :nc])
                pred_c = out[4][:, :t_horizon]
                batch_index = torch.as_tensor(batch_ids, dtype=torch.long, device=trainer.device)
                loss = (
                    group_loss(pred_a, targets["alphas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                    + group_loss(pred_b, targets["betas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                    + group_loss(pred_g, targets["gammas"].index_select(0, batch_index), target_scales["coeff"], coeff_w)
                    + group_loss(pred_d, targets["deltas"].index_select(0, batch_index), target_scales["delta"], delta_w)
                    + group_loss(pred_c, targets["costs"].index_select(0, batch_index), target_scales["cost"], cost_w)
                )
                if proxy_kkt_w > 0:
                    for local_pos, sid in enumerate(batch_ids):
                        kkt_loss = trainer.loss_function_differentiable(
                            int(sid),
                            pred_a[local_pos],
                            pred_b[local_pos],
                            pred_g[local_pos],
                            pred_d[local_pos],
                            pred_c[local_pos],
                            trainer.device,
                        )
                        loss = loss + proxy_kkt_w * kkt_loss
                actual_batch_size = max(1, len(batch_ids))
                (loss / actual_batch_size).backward()
                epoch_loss += float(loss.detach().cpu().item())
                params = trainer._main_network_parameters() + trainer._cost_network_parameters()
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                optimizer_main.step()
                optimizer_cost.step()
                optimizer_main.zero_grad()
                optimizer_cost.zero_grad()
            scheduler_main.step()
            scheduler_cost.step()
            last_avg = epoch_loss / max(trainer.n_samples, 1)
            do_log = (
                epoch == 0
                or epoch == epochs - 1
                or (log_interval > 0 and (epoch + 1) % log_interval == 0)
            )
            do_target_check = (
                direct_mae_target is not None
                and (
                    do_log
                    or epoch == epochs - 1
                    or (target_check_interval > 0 and (epoch + 1) % target_check_interval == 0)
                )
            )
            if do_log or do_target_check:
                last_fast = _main_fast_metrics(trainer, sample_features, targets, target_scales)
            if do_log:
                print(
                    f"  [Unit-{trainer.unit_id}][direct-NN-main] epoch {epoch+1:>4}/{epochs}, "
                    f"avg_target_loss={last_avg:.6f}, lr={optimizer_main.param_groups[0]['lr']:.2e}, "
                    f"cost_lr={optimizer_cost.param_groups[0]['lr']:.2e}, "
                    f"fast_direct_mae={last_fast['direct_mae']:.6f}, "
                    f"fast_obj_dual_x={last_fast['obj_dual_x']:.6f}",
                    flush=True,
                )
    finally:
        _set_main_training_mode(trainer, False)
    trainer._refresh_cached_surrogate_outputs()
    if last_fast is None:
        last_fast = _main_fast_metrics(trainer, sample_features, targets, target_scales)
    return {
        "avg_target_loss": None if last_avg is None else float(last_avg),
        "fast_metrics": {k: float(v) for k, v in last_fast.items()},
        "proxy_residual_mean": float(np_targets["proxy_residual_mean"]),
    }


def _apply_cpg_snapshot(trainer, snap: dict, strict: bool) -> None:
    n = len(snap["samples"])
    t = int(snap["T"])
    trainer.lambda_vals = np.asarray([s["lambda_vals"] for s in snap["samples"]], dtype=np.float64).reshape(n, t)
    for i, s in enumerate(snap["samples"]):
        cpg = s.get("lambda_inherent_c_pg")
        if not cpg or trainer.lambda_inherent[i] is None:
            continue
        for k, v in cpg.items():
            if v is not None and k in trainer.lambda_inherent[i]:
                trainer.lambda_inherent[i][k] = np.asarray(v, dtype=np.float64).reshape(-1)
    prev = snap.get("prev_pg_cost_values")
    trainer._prev_pg_cost_values = None if prev is None else np.asarray(prev, dtype=np.float64)
    trainer.iter_number = int(snap.get("iter_number", 0))
    if strict:
        trainer.rho_dual_pg = float(snap["rho_dual_pg"])
        trainer.loss_ratio_dual_pg = float(snap["loss_ratio_dual_pg"])
        trainer.loss_ratio_reg = float(snap["loss_ratio_reg"])
        trainer._c_pg_reg_loss_scale = float(snap["_c_pg_reg_loss_scale"])
        trainer.reg_weight = float(snap["reg_weight"])
        trainer.pg_cost_reg_deadband = float(snap["pg_cost_reg_deadband"])
        trainer.iter_delta_reg_weight = float(snap["iter_delta_reg_weight"])
        trainer.iter_delta_reg_deadband = float(snap["iter_delta_reg_deadband"])
        trainer.pg_cost_softbound_weight = float(snap["pg_cost_softbound_weight"])
        trainer.pg_cost_smooth_abs_eps = float(snap["pg_cost_smooth_abs_eps"])
        trainer.pg_cost_scale = float(snap["pg_cost_scale"])


def _apply_main_trial_hparams(trainer, trial: dict) -> dict:
    original = {
        "reg_weight": float(trainer.reg_weight),
        "loss_ratio_primal": float(trainer.loss_ratio_primal),
        "loss_ratio_dual_x": float(trainer.loss_ratio_dual_x),
        "loss_ratio_opt": float(trainer.loss_ratio_opt),
        "loss_ratio_reg": float(trainer.loss_ratio_reg),
        "iter_delta_reg_weight": float(trainer.iter_delta_reg_weight),
        "iter_delta_reg_deadband": float(trainer.iter_delta_reg_deadband),
        "coeff_reg_deadband": float(trainer.coeff_reg_deadband),
        "aux_cost_reg_deadband": float(trainer.aux_cost_reg_deadband),
    }
    for key in original:
        if key in trial:
            setattr(trainer, key, float(trial[key]))
    trainer._surr_optimizer = None
    trainer._surr_cost_optimizer = None
    trainer._surr_scheduler = None
    return original


def run_main_test() -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    _bootstrap_case118_light()
    _apply_script_overrides()
    snap_path = _abs(TEST_SNAPSHOT_JSON)
    with snap_path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    n = int(snap["n_samples"])
    active_path = _resolve_active_set_path()
    samples = _load_active_set_samples(active_path)[:n]
    ppc = get_case_ppc(CASE)
    bundle = _abs(TEST_BUNDLE_DIR)
    _dual, trainers = load_trained_models(
        ppc,
        samples,
        float(T_DELTA),
        str(bundle),
        unit_ids=[int(UNIT_ID)],
        lp_backend=str(LP_BACKEND).strip().lower(),
        constraint_generation_strategy=CONSTRAINT_STRATEGY,
        ignore_startup_shutdown_costs=bool(IGNORE_STARTUP_SHUTDOWN),
    )
    trainer = trainers[int(UNIT_ID)]
    if TEST_FULL_BASELINE_METRICS or TEST_MAIN_FULL_BASELINE_METRICS:
        print("[main_test] calculating baseline metrics...", flush=True)
        base_metrics = _metrics(trainer)
    else:
        base_metrics = {"obj_primal": None, "obj_dual_x": None, "obj_opt": None, "note": "baseline full metrics skipped"}
    base_state = _capture_state(trainer)
    print(f"[main_test] before NN-main metrics: {base_metrics}", flush=True)

    trials = TEST_MAIN_TRIALS or [{
        "name": "legacy_nn_main_only",
        "fine_epochs": 1,
        "fine_lr": 1e-4,
        "fine_cost_lr": 5e-6,
    }]
    results = {
        "schema_version": 1,
        "mode": "main_test",
        "case": CASE,
        "unit_id": int(UNIT_ID),
        "n_samples": n,
        "snapshot_json": str(snap_path),
        "bundle_dir": str(bundle),
        "base_metrics": base_metrics,
        "trials": [],
    }
    best_state = None
    best_result = None
    best_score = float("inf")
    for idx, trial in enumerate(trials):
        _restore_state(trainer, base_state)
        trial = dict(trial)
        name = str(trial.get("name", f"trial{idx}"))
        original_hparams = _apply_main_trial_hparams(trainer, trial)
        before = _metrics(trainer) if TEST_MAIN_FULL_BASELINE_METRICS else dict(base_metrics)
        print(
            f"[main_test][{name}] direct_epochs={int(trial.get('direct_epochs', 0) or 0)} "
            f"fine_epochs={int(trial.get('fine_epochs', 0) or 0)} "
            f"lr={float(trial.get('direct_lr', 0.0) or 0.0):.3g} "
            f"cost_lr={float(trial.get('direct_cost_lr', 0.0) or 0.0):.3g}",
            flush=True,
        )
        direct_result = _train_main_direct_targets(trainer, trial)
        fine_epochs = int(trial.get("fine_epochs", 0) or 0)
        if fine_epochs > 0:
            trainer.iter_with_surrogate_nn(
                num_epochs=fine_epochs,
                batch_size=trial.get("fine_batch_size"),
                batch_strategy=trial.get("fine_batch_strategy"),
                shuffle=trial.get("fine_shuffle"),
                learning_rate=(
                    None
                    if trial.get("fine_lr") is None
                    else float(trial.get("fine_lr"))
                ),
                cost_learning_rate=(
                    None
                    if trial.get("fine_cost_lr") is None
                    else float(trial.get("fine_cost_lr"))
                ),
            )
            trainer._refresh_cached_surrogate_outputs()
        if TEST_MAIN_FULL_FINAL_METRICS:
            after = _metrics(trainer)
            score = (
                float(after["obj_primal"])
                + float(after["obj_dual_x"])
                + float(after["obj_opt"])
            )
        else:
            fast = direct_result.get("fast_metrics") if isinstance(direct_result, dict) else None
            after = {
                "obj_primal": float("inf") if fast is None else float(fast["obj_primal"]),
                "obj_dual_x": float("inf") if fast is None else float(fast["obj_dual_x"]),
                "obj_opt": float("inf") if fast is None else float(fast["obj_opt"]),
                "direct_mae": float("inf") if fast is None else float(fast["direct_mae"]),
                "fast_metric": True,
                "note": "direct target + differentiable NN-main components; full metrics skipped",
            }
            score = float(after["direct_mae"])
        print(f"[main_test][{name}] before={before}", flush=True)
        print(f"[main_test][{name}] after ={after}", flush=True)
        result = {
            "name": name,
            "trial": trial,
            "original_hparams": original_hparams,
            "before": before,
            "after": after,
            "direct_result": direct_result,
        }
        results["trials"].append(result)
        if score < best_score:
            best_score = score
            best_result = result
            best_state = _capture_state(trainer)

    if best_state is not None:
        _restore_state(trainer, best_state)
    results["best_trial"] = None if best_result is None else best_result["name"]
    out_json = _abs(TEST_MAIN_RESULT_JSON)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[main_test] wrote trial results: {out_json}", flush=True)
    if TEST_MAIN_FULL_FINAL_METRICS:
        print(f"[main_test] best NN-main metrics: {_metrics(trainer)}", flush=True)
    elif best_result is not None:
        print(f"[main_test] best NN-main metrics: {best_result['after']}", flush=True)
    if str(TEST_MAIN_LAST_PTH).strip():
        out = _abs(TEST_MAIN_LAST_PTH)
        out.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(out))
        print(f"[main_test] saved checkpoint: {out}", flush=True)


def run_main_tune() -> None:
    import run_training as rt

    _bootstrap_case118_light()
    _apply_script_overrides()
    ppc, all_samples, dual, active_path = _prepare_data_and_dual()
    trainer = _make_trainer(ppc, all_samples, dual)

    target_iter = max(1, int(MAIN_TUNE_TARGET_ITER))
    pre_iters = max(target_iter - 1, 0)
    if pre_iters > 0:
        print(f"[main_tune] running {pre_iters} full BCD rounds first", flush=True)
        trainer.iter(
            max_iter=pre_iters,
            nn_epochs=int(rt.NN_EPOCHS),
            pg_cost_nn_epochs=int(rt.SUBPROBLEM_PG_COST_NN_EPOCHS),
            nn_batch_strategy=rt.SUBPROBLEM_NN_BATCH_STRATEGY,
            nn_batch_size=rt.SUBPROBLEM_NN_BATCH_SIZE,
            nn_shuffle=rt.SUBPROBLEM_NN_SHUFFLE,
            nn_learning_rate=rt.SUBPROBLEM_NN_LR,
            cost_learning_rate=rt.SUBPROBLEM_X_COST_NN_LR,
            pg_cost_surr_learning_rate=rt.SUBPROBLEM_PG_COST_SURR_LR,
        )

    print(f"[main_tune] collecting fixed state before NN-main at BCD round {target_iter}", flush=True)
    _run_primal_dual_only_round(trainer, target_iter - 1)
    base_state = _capture_state(trainer)
    base_metrics = _metrics(trainer)

    bundle = _abs(MAIN_TUNE_BUNDLE_DIR)
    bundle.mkdir(parents=True, exist_ok=True)
    dual.save(str(bundle / "dual_predictor.pth"))
    trainer.save(str(bundle / f"surrogate_unit_{UNIT_ID}.pth"))
    cpg_snap = trainer.collect_c_pg_loss_snapshot()
    cpg_out = _abs(MAIN_TUNE_C_PG_SNAPSHOT_JSON)
    cpg_out.parent.mkdir(parents=True, exist_ok=True)
    with cpg_out.open("w", encoding="utf-8") as f:
        json.dump(cpg_snap, f, indent=2, ensure_ascii=False)
    print(f"[main_tune] wrote synchronized c_pg snapshot: {cpg_out}", flush=True)
    print(f"[main_tune] wrote synchronized bundle: {bundle}", flush=True)

    results = {
        "schema_version": 1,
        "mode": "main_tune",
        "case": CASE,
        "unit_id": int(UNIT_ID),
        "n_samples": len(all_samples),
        "target_iter": target_iter,
        "active_set_json": str(active_path),
        "sync_c_pg_snapshot_json": str(cpg_out),
        "sync_bundle_dir": str(bundle),
        "base_metrics": base_metrics,
        "trials": [],
    }
    for idx, trial in enumerate(MAIN_TUNE_TRIALS):
        _restore_state(trainer, base_state)
        name = str(trial.get("name", f"trial{idx}"))
        epochs = int(trial.get("epochs", rt.NN_EPOCHS))
        lr = float(trial.get("lr", rt.SUBPROBLEM_NN_LR))
        cost_lr = float(trial.get("cost_lr", rt.SUBPROBLEM_X_COST_NN_LR))
        bs = trial.get("batch_strategy", MAIN_TUNE_BATCH_STRATEGY)
        bz = trial.get("batch_size", MAIN_TUNE_BATCH_SIZE)
        sh = trial.get("shuffle", MAIN_TUNE_SHUFFLE)
        before = _metrics(trainer)
        print(f"[main_tune][{name}] epochs={epochs} lr={lr} cost_lr={cost_lr}", flush=True)
        trainer.iter_with_surrogate_nn(
            num_epochs=epochs,
            batch_size=bz,
            batch_strategy=bs,
            shuffle=sh,
            learning_rate=lr,
            cost_learning_rate=cost_lr,
        )
        trainer._refresh_cached_surrogate_outputs()
        after = _metrics(trainer)
        print(f"[main_tune][{name}] before={before}", flush=True)
        print(f"[main_tune][{name}] after ={after}", flush=True)
        results["trials"].append({
            "name": name,
            "epochs": epochs,
            "lr": lr,
            "cost_lr": cost_lr,
            "batch_strategy": bs,
            "batch_size": bz,
            "shuffle": sh,
            "before": before,
            "after": after,
        })

    out = _abs(MAIN_TUNE_OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[main_tune] wrote trial results: {out}", flush=True)


def run_test() -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    _bootstrap_case118_light()
    _apply_script_overrides()
    snap_path = _abs(TEST_SNAPSHOT_JSON)
    with snap_path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    n = int(snap["n_samples"])
    active_path = _resolve_active_set_path()
    samples = _load_active_set_samples(active_path)[:n]
    ppc = get_case_ppc(CASE)
    bundle = _abs(TEST_BUNDLE_DIR)
    _dual, trainers = load_trained_models(
        ppc,
        samples,
        float(T_DELTA),
        str(bundle),
        unit_ids=[int(UNIT_ID)],
        lp_backend=str(LP_BACKEND).strip().lower(),
        constraint_generation_strategy=CONSTRAINT_STRATEGY,
        ignore_startup_shutdown_costs=bool(IGNORE_STARTUP_SHUTDOWN),
    )
    trainer = trainers[int(UNIT_ID)]
    _apply_cpg_snapshot(trainer, snap, strict=bool(TEST_STRICT_HPARAMS))
    if TEST_FULL_BASELINE_METRICS:
        print("[test] calculating baseline metrics...", flush=True)
        base_metrics = _metrics(trainer)
    else:
        base_metrics = {"obj_dual_pg": None, "note": "baseline full metrics skipped"}
    base_state = _capture_cpg_state(trainer)
    print(f"[test] before c_pg metrics: {base_metrics}", flush=True)

    trials = TEST_C_PG_TRIALS or [{
        "name": "legacy_kkt_only",
        "fine_epochs": TEST_C_PG_EPOCHS,
        "fine_lr": TEST_PG_COST_SURR_LR,
        "fine_batch_strategy": TEST_C_PG_BATCH_STRATEGY,
        "fine_batch_size": TEST_C_PG_BATCH_SIZE,
        "fine_shuffle": TEST_C_PG_SHUFFLE,
    }]
    results = {
        "schema_version": 1,
        "mode": "test",
        "case": CASE,
        "unit_id": int(UNIT_ID),
        "n_samples": n,
        "snapshot_json": str(snap_path),
        "bundle_dir": str(bundle),
        "base_metrics": base_metrics,
        "trials": [],
    }
    best_state = None
    best_result = None
    best_obj_dual_pg = float("inf")
    for idx, trial in enumerate(trials):
        _restore_cpg_state(trainer, base_state)
        _apply_cpg_snapshot(trainer, snap, strict=bool(TEST_STRICT_HPARAMS))
        trial = dict(trial)
        name = str(trial.get("name", f"trial{idx}"))
        original_hparams = _apply_cpg_trial_hparams(trainer, trial)
        before = _metrics(trainer) if TEST_FULL_BASELINE_METRICS else dict(base_metrics)
        print(
            f"[test][{name}] direct_epochs={int(trial.get('direct_epochs', 0) or 0)} "
            f"fine_epochs={int(trial.get('fine_epochs', TEST_C_PG_EPOCHS) or 0)} "
            f"reg_deadband={trainer.pg_cost_reg_deadband:.3g} "
            f"reg_scale={trainer._c_pg_reg_loss_scale:.3g} "
            f"softbound={trainer.pg_cost_softbound_weight:.3g}",
            flush=True,
        )
        direct_result = _train_cpg_direct_targets(trainer, trial)
        fine_epochs = int(trial.get("fine_epochs", TEST_C_PG_EPOCHS) or 0)
        if fine_epochs > 0:
            trainer.iter_with_c_pg_nn(
                num_epochs=fine_epochs,
                batch_size=trial.get("fine_batch_size", TEST_C_PG_BATCH_SIZE),
                batch_strategy=trial.get("fine_batch_strategy", TEST_C_PG_BATCH_STRATEGY),
                shuffle=trial.get("fine_shuffle", TEST_C_PG_SHUFFLE),
                learning_rate=(
                    None
                    if trial.get("fine_lr", TEST_PG_COST_SURR_LR) is None
                    else float(trial.get("fine_lr", TEST_PG_COST_SURR_LR))
                ),
                log_interval=TEST_C_PG_LOG_INTERVAL,
                log_metrics=bool(TEST_C_PG_LOG_METRICS),
            )
        if TEST_FULL_FINAL_METRICS:
            after = _metrics(trainer)
        else:
            fast_obj = None
            if isinstance(direct_result, dict):
                fast_obj = direct_result.get("fast_obj_dual_pg")
            after = {
                "obj_dual_pg": float("inf") if fast_obj is None else float(fast_obj),
                "fast_metric": True,
                "note": "sum(abs(c_pg - analytic_target)); full metrics skipped",
            }
        print(f"[test][{name}] before={before}", flush=True)
        print(f"[test][{name}] after ={after}", flush=True)
        result = {
            "name": name,
            "trial": trial,
            "original_hparams": original_hparams,
            "before": before,
            "after": after,
            "direct_result": direct_result,
        }
        results["trials"].append(result)
        if after["obj_dual_pg"] < best_obj_dual_pg:
            best_obj_dual_pg = after["obj_dual_pg"]
            best_result = result
            best_state = _capture_cpg_state(trainer)

    if best_state is not None:
        _restore_cpg_state(trainer, best_state)
    results["best_trial"] = None if best_result is None else best_result["name"]
    out_json = _abs(TEST_RESULT_JSON)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[test] wrote trial results: {out_json}", flush=True)
    if TEST_FULL_FINAL_METRICS:
        print(f"[test] best c_pg metrics: {_metrics(trainer)}", flush=True)
    elif best_result is not None:
        print(f"[test] best c_pg metrics: {best_result['after']}", flush=True)
    if str(TEST_LAST_PTH).strip():
        out = _abs(TEST_LAST_PTH)
        out.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(out))
        print(f"[test] saved checkpoint: {out}", flush=True)


def run_light_bake() -> None:
    import run_training as rt

    _bootstrap_case118_light()
    _apply_script_overrides()
    ppc, samples, dual, _active_path = _prepare_data_and_dual()
    trainer = _make_trainer(ppc, samples, dual)
    trainer.iter(
        max_iter=int(rt.SUBPROBLEM_MAX_ITER),
        nn_epochs=int(rt.NN_EPOCHS),
        pg_cost_nn_epochs=int(rt.SUBPROBLEM_PG_COST_NN_EPOCHS),
        nn_batch_strategy=rt.SUBPROBLEM_NN_BATCH_STRATEGY,
        nn_batch_size=rt.SUBPROBLEM_NN_BATCH_SIZE,
        nn_shuffle=rt.SUBPROBLEM_NN_SHUFFLE,
        nn_learning_rate=rt.SUBPROBLEM_NN_LR,
        cost_learning_rate=rt.SUBPROBLEM_X_COST_NN_LR,
        pg_cost_surr_learning_rate=rt.SUBPROBLEM_PG_COST_SURR_LR,
    )
    snap = trainer.collect_c_pg_loss_snapshot()
    out = _abs(LIGHT_BAKE_OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    bundle = _abs(LIGHT_BAKE_BUNDLE_DIR)
    bundle.mkdir(parents=True, exist_ok=True)
    dual.save(str(bundle / "dual_predictor.pth"))
    trainer.save(str(bundle / f"surrogate_unit_{UNIT_ID}.pth"))
    print(f"[light_bake] wrote snapshot: {out}", flush=True)
    print(f"[light_bake] wrote bundle: {bundle}", flush=True)


def main() -> None:
    print(f"[subproblem_loss_snapshot] MODE={MODE!r}", flush=True)
    mode = str(MODE).strip().lower()
    if mode == "main_tune":
        run_main_tune()
    elif mode in ("main_test", "nn_main_test"):
        run_main_test()
    elif mode == "test":
        run_test()
    elif mode in ("light_bake", "bake"):
        run_light_bake()
    else:
        raise ValueError("MODE must be 'main_tune', 'main_test', 'test', or 'light_bake'")


if __name__ == "__main__":
    main()
