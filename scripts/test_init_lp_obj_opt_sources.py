#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 **SubproblemSurrogateTrainer 构造完成（含 init LP + λ/μ 初始化）** 后，
用 **x_true、init 输出的 pg/coc、lambda_inherent、mu、alpha..delta** 按 ``cal_viol_components``
中 **obj_opt / obj_primal** 的定义做 **分项统计**（含占 obj_opt 总量的比例）。

**默认配置**与 ``run_training_case118_subproblem_bcd_light.py`` 一致（无命令行即可跑）：
``TRAIN_TARGET=subproblem_bcd``、``SUBPROBLEM_SOLVE_PRESET=desktop``、
``SUBPROBLEM_LIGHT_MAX_SAMPLES=1``、``SUBPROBLEM_LIGHT_N_WORKERS_UNIT=2``、
``SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE=2``；机组列表不覆盖时沿用 ``run_training_case118`` 模块内
``CASE118_SUBPROBLEM_UNIT_IDS``（与 light 未传 ``--units`` 时相同）。

运行（项目根目录）::

    python scripts/test_init_lp_obj_opt_sources.py
    python scripts/test_init_lp_obj_opt_sources.py --unit 2 --dual-epochs 8 --max-samples 4

``--dual-epochs 0``：不训练对偶预测器，依赖样本 JSON 中已有 ``lambda`` 或由 trainer 内部 ED 补全（可能较慢）。

依赖与 ``run_training_case118`` 子问题训练相同（torch / cvxpy+HiGHS 或 gurobi 等）。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

# 与 run_training_case118_subproblem_bcd_light.py 默认参数一致（命令行省略时使用）
TRAIN_TARGET = "subproblem_bcd"
SUBPROBLEM_SOLVE_PRESET = "desktop"
SUBPROBLEM_LIGHT_MAX_SAMPLES = 1
SUBPROBLEM_LIGHT_N_WORKERS_UNIT = 2
SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE = 2
DEFAULT_UNIT_ID = 2
DEFAULT_SAMPLE_ID = 0
DEFAULT_DUAL_EPOCHS = 0

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_analyze_module():
    p = ROOT / "scripts" / "analyze_init_lp_objectives.py"
    spec = importlib.util.spec_from_file_location("analyze_init_lp_objectives", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unit", type=int, default=None, help=f"机组 unit_id（默认 {DEFAULT_UNIT_ID}）")
    p.add_argument("--sample", type=int, default=None, help=f"样本下标（默认 {DEFAULT_SAMPLE_ID}）")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=f"截取前 N 个样本（默认内置 {SUBPROBLEM_LIGHT_MAX_SAMPLES}，与 light 一致）",
    )
    p.add_argument(
        "--n-workers-unit",
        type=int,
        default=None,
        metavar="K",
        help=f"SUBPROBLEM_LIGHT_N_WORKERS_UNIT（默认 {SUBPROBLEM_LIGHT_N_WORKERS_UNIT}）",
    )
    p.add_argument(
        "--n-workers-sample",
        type=int,
        default=None,
        metavar="K",
        help=f"SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE（默认 {SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE}）",
    )
    p.add_argument(
        "--preset",
        choices=("server", "desktop"),
        default=None,
        help=f"SUBPROBLEM_SOLVE_PRESET（默认 {SUBPROBLEM_SOLVE_PRESET!r}）",
    )
    p.add_argument(
        "--units",
        type=str,
        default=None,
        metavar="IDS",
        help="仅训练所列机组 ID（逗号分隔）；写入 CASE118_SUBPROBLEM_UNIT_IDS，与 light --units 相同",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="覆盖 SUBPROBLEM_LIGHT_MAX_ITER（与 light --max-iter 相同）",
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=None,
        metavar="W",
        help="覆盖 SUBPROBLEM_LIGHT_PREDICTOR_WARMUP_ROUNDS（与 light --warmup-rounds 相同）",
    )
    p.add_argument(
        "--json",
        type=str,
        default=None,
        help="覆盖 active set JSON 路径（默认用 run_training_case118.CASE118_ACTIVE_SET_JSON）",
    )
    p.add_argument(
        "--dual-epochs",
        type=int,
        default=None,
        help=f"对偶预测器训练轮数；默认 {DEFAULT_DUAL_EPOCHS} 表示不训练",
    )
    p.add_argument(
        "--delta-reference-lift",
        choices=("auto", "on", "off"),
        default="auto",
        help="surrogate delta reference lift: auto enables it for sign4 strategies",
    )
    p.add_argument(
        "--unit-predictor-epochs",
        type=int,
        default=None,
        help="epochs for the single-time unit predictor; default follows run_training_case118",
    )
    p.add_argument(
        "--unit-predictor-load-path",
        type=str,
        default=None,
        help="optional unit_predictor checkpoint, directory, or LATEST.txt",
    )
    p.add_argument(
        "--no-unit-predictor",
        action="store_true",
        help="disable unit predictor so single constraints use surrogate NN initialization",
    )
    p.add_argument(
        "--delta-reference-scope",
        choices=("sign4_only", "all_coupling"),
        default="sign4_only",
        help="scope for surrogate delta reference lift",
    )
    return p.parse_args()


def _make_trainer(
    rt,
    ppc,
    all_samples: list,
    unit_id: int,
    dual_predictor,
    unit_predictor=None,
):
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
        int(unit_id),
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
        rho_binary_max=rt.SUBPROBLEM_RHO_BINARY_MAX,
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
        ignore_startup_shutdown_costs=bool(rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS),
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
        unit_predictor=unit_predictor,
        use_unit_predictor=(unit_predictor is not None),
        predictor_warmup_rounds=getattr(rt, "SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS", 0),
        sign4_curriculum_rounds=getattr(rt, "SUBPROBLEM_SIGN4_CURRICULUM_ROUNDS", 0),
        sign4_initial_scale=getattr(rt, "SUBPROBLEM_SIGN4_INITIAL_SCALE", 1.0),
        sign4_final_scale=getattr(rt, "SUBPROBLEM_SIGN4_FINAL_SCALE", 1.0),
        enable_surrogate_delta_reference_lift=getattr(
            rt, "SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT", None
        ),
        surrogate_delta_reference_eps=getattr(
            rt, "SUBPROBLEM_SURROGATE_DELTA_REFERENCE_EPS", 1e-6
        ),
        surrogate_delta_reference_scope=getattr(
            rt, "SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE", "sign4_only"
        ),
        surrogate_delta_reference_min_abs_factor=getattr(
            rt, "SUBPROBLEM_SURROGATE_DELTA_REFERENCE_MIN_ABS_FACTOR", 1e-9
        ),
        unit_predictor_finetune_lr=rt.UNIT_PREDICTOR_FINETUNE_LR,
        unit_predictor_weight_decay=rt.UNIT_PREDICTOR_WEIGHT_DECAY,
        case_name=str(rt.CASE_NAME),
    )


def _pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "n/a"
    return f"{100.0 * part / whole:.2f}%"


def _resolve_unit_predictor_load_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        latest = p / "LATEST.txt"
        if latest.is_file():
            target = Path(latest.read_text(encoding="utf-8").strip())
            if not target.is_absolute():
                target = p / target
            return str(target) if target.exists() else None
        candidate = p / "unit_predictor.pth"
        return str(candidate) if candidate.exists() else None
    return str(p) if p.exists() else None


def _make_unit_predictor(rt, ppc, all_samples: list, unit_id: int, args):
    if bool(args.no_unit_predictor) or not bool(getattr(rt, "USE_UNIT_PREDICTOR", False)):
        return None

    strategy = str(getattr(rt, "SURROGATE_CONSTRAINT_STRATEGY", "") or "")
    if strategy not in ("all_single_time", "all_templates_sign4_plus_single"):
        return None

    from src.uc_NN_subproblem import train_unit_predictor_from_data

    load_path = _resolve_unit_predictor_load_path(
        args.unit_predictor_load_path or getattr(rt, "UNIT_PREDICTOR_LOAD_PATH", None)
    )
    epochs = (
        int(args.unit_predictor_epochs)
        if args.unit_predictor_epochs is not None
        else int(getattr(rt, "UNIT_PREDICTOR_EPOCHS", 0))
    )
    if load_path:
        epochs = 0 if args.unit_predictor_epochs is None else epochs

    print(
        "[test] unit_predictor enabled: "
        f"unit={unit_id}, epochs={epochs}, load_path={load_path}",
        flush=True,
    )
    return train_unit_predictor_from_data(
        ppc,
        all_samples,
        T_delta=float(rt.T_DELTA),
        unit_ids=[int(unit_id)],
        hidden_dims=getattr(rt, "UNIT_PREDICTOR_HIDDEN_DIMS", None),
        num_epochs=epochs,
        batch_size=int(getattr(rt, "UNIT_PREDICTOR_BATCH_SIZE", 32)),
        batch_strategy=str(getattr(rt, "UNIT_PREDICTOR_BATCH_STRATEGY", "full-batch")),
        shuffle=bool(getattr(rt, "UNIT_PREDICTOR_SHUFFLE", True)),
        learning_rate=float(getattr(rt, "UNIT_PREDICTOR_LR", 1e-3)),
        weight_decay=float(getattr(rt, "UNIT_PREDICTOR_WEIGHT_DECAY", 1e-4)),
        net_variant=str(getattr(rt, "UNIT_PREDICTOR_NET_VARIANT", "mlp")),
        tcn_channels=int(getattr(rt, "UNIT_PREDICTOR_TCN_CHANNELS", 64)),
        tcn_depth=int(getattr(rt, "UNIT_PREDICTOR_TCN_DEPTH", 6)),
        tconv_channels=int(getattr(rt, "UNIT_PREDICTOR_TCONV_CHANNELS", 64)),
        tconv_depth=int(getattr(rt, "UNIT_PREDICTOR_TCONV_DEPTH", 4)),
        dropout=float(getattr(rt, "UNIT_PREDICTOR_DROPOUT", 0.1)),
        load_path=load_path,
        enable_pos_weight=bool(getattr(rt, "UNIT_PREDICTOR_ENABLE_POS_WEIGHT", False)),
        pos_weight_clip=float(getattr(rt, "UNIT_PREDICTOR_POS_WEIGHT_CLIP", 20.0)),
        loss_weight_bce=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_BCE", 1.0)),
        loss_weight_mse=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_MSE", 0.0)),
        loss_weight_l1=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_L1", 0.0)),
        loss_weight_tv=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_TV", 0.0)),
        loss_weight_transition=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_TRANSITION", 0.0)),
        loss_weight_binarize=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_BINARIZE", 0.0)),
        loss_weight_std_floor=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_STD_FLOOR", 0.0)),
        std_floor_scale=float(getattr(rt, "UNIT_PREDICTOR_STD_FLOOR_SCALE", 0.5)),
        loss_weight_tv_floor=float(getattr(rt, "UNIT_PREDICTOR_LOSS_WEIGHT_TV_FLOOR", 0.0)),
        tv_floor_scale=float(getattr(rt, "UNIT_PREDICTOR_TV_FLOOR_SCALE", 0.8)),
    )


def _print_single_rhs_summary(trainer, sample_id: int) -> None:
    if not hasattr(trainer, "_single_time_coupling_slice"):
        return
    st, en = trainer._single_time_coupling_slice()
    if int(en) <= int(st):
        print("\n--- single-time summary: no single constraints ---", flush=True)
        return
    alpha = np.asarray(trainer.alpha_values[sample_id, st:en], dtype=float).reshape(-1)
    delta = np.asarray(trainer.delta_values[sample_id, st:en], dtype=float).reshape(-1)
    x_ref = trainer.active_set_data[sample_id].get("x_true")
    if x_ref is None:
        x_ref = trainer.x[sample_id]
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    n = min(alpha.size, delta.size, x_ref.size)
    if n <= 0:
        return
    alpha = alpha[:n]
    delta = delta[:n]
    x_ref = x_ref[:n]
    valid = np.abs(alpha) > 1e-9
    x_hat = np.full(n, np.nan, dtype=float)
    x_hat[valid] = delta[valid] / alpha[valid]
    le_mask = alpha > 1e-9
    ge_mask = alpha < -1e-9
    bad_range = valid & ((x_hat < -1e-6) | (x_hat > 1.0 + 1e-6))
    print("\n--- single-time decoded RHS ---", flush=True)
    print(
        f"  rows={n}, x<=x_hat={int(np.sum(le_mask))}, x>=x_hat={int(np.sum(ge_mask))}, "
        f"x_hat_range=[{np.nanmin(x_hat):.6g}, {np.nanmax(x_hat):.6g}], "
        f"out_of_[0,1]={int(np.sum(bad_range))}",
        flush=True,
    )
    for t in range(min(8, n)):
        side = "<=" if alpha[t] > 0 else ">=" if alpha[t] < 0 else "?"
        print(
            f"  t={t:02d}: alpha={alpha[t]: .6g}, delta={delta[t]: .6g}, "
            f"decoded: x {side} {x_hat[t]:.6g}, x_true={x_ref[t]:.0f}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    aio = _load_analyze_module()

    import run_training as rt
    import run_training_case118 as c118

    preset = args.preset if args.preset is not None else SUBPROBLEM_SOLVE_PRESET
    max_samples = (
        max(1, int(args.max_samples))
        if args.max_samples is not None
        else SUBPROBLEM_LIGHT_MAX_SAMPLES
    )
    n_workers_unit = (
        max(1, int(args.n_workers_unit))
        if args.n_workers_unit is not None
        else SUBPROBLEM_LIGHT_N_WORKERS_UNIT
    )
    n_workers_sample = (
        max(1, int(args.n_workers_sample))
        if args.n_workers_sample is not None
        else SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE
    )
    unit_id = int(args.unit) if args.unit is not None else DEFAULT_UNIT_ID
    sample_id = int(args.sample) if args.sample is not None else DEFAULT_SAMPLE_ID
    dual_epochs = int(args.dual_epochs) if args.dual_epochs is not None else DEFAULT_DUAL_EPOCHS

    # 与 run_training_case118_subproblem_bcd_light.main() 同序：先写 TRAIN_TARGET / preset / LIGHT_*，再 configure
    c118.TRAIN_TARGET = TRAIN_TARGET
    c118.SUBPROBLEM_SOLVE_PRESET = preset
    c118.SUBPROBLEM_LIGHT_MAX_SAMPLES = max_samples
    c118.SUBPROBLEM_LIGHT_N_WORKERS_UNIT = n_workers_unit
    c118.SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE = n_workers_sample
    if args.units is not None:
        parts = [p.strip() for p in str(args.units).split(",") if p.strip()]
        c118.CASE118_SUBPROBLEM_UNIT_IDS = [int(x) for x in parts]
    if args.max_iter is not None:
        c118.SUBPROBLEM_LIGHT_MAX_ITER = max(1, int(args.max_iter))
    if args.warmup_rounds is not None:
        c118.SUBPROBLEM_LIGHT_PREDICTOR_WARMUP_ROUNDS = max(0, int(args.warmup_rounds))
    if args.delta_reference_lift == "auto":
        c118.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = None
    else:
        c118.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_LIFT = (
            args.delta_reference_lift == "on"
        )
    c118.CASE118_SUBPROBLEM_SURROGATE_DELTA_REFERENCE_SCOPE = args.delta_reference_scope

    c118._validate_inputs()
    c118._configure_common()
    c118._configure_subproblem_bcd()
    c118._apply_subproblem_light_runtime_overrides()

    if args.json:
        data_path = Path(args.json)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
    else:
        data_path = Path(c118.__file__).resolve().parent / c118.CASE118_ACTIVE_SET_JSON
    if not data_path.is_file():
        raise FileNotFoundError(f"active set json not found: {data_path}")

    all_samples = rt.load_json_data(data_path)
    n = max(1, min(max_samples, len(all_samples)))
    all_samples = all_samples[:n]

    from src.case_registry import get_case_ppc

    ppc = get_case_ppc("case118")

    dual_predictor = None
    if dual_epochs > 0:
        from src.uc_NN_subproblem import train_dual_predictor_from_data

        dual_predictor = train_dual_predictor_from_data(
            ppc,
            all_samples,
            T_delta=float(rt.T_DELTA),
            num_epochs=dual_epochs,
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

    unit_predictor = _make_unit_predictor(rt, ppc, all_samples, unit_id, args)
    trainer = _make_trainer(rt, ppc, all_samples, unit_id, dual_predictor, unit_predictor)
    sid = sample_id
    if not (0 <= sid < trainer.n_samples):
        raise ValueError(f"sample_id={sid} out of range n_samples={trainer.n_samples}")

    out = aio.analyze_init_lp_state(trainer, sid, at_x_true=True, verbose=True)
    _print_single_rhs_summary(trainer, sid)

    oo = out["obj_opt"]
    sk = out.get("surrogate_by_kind")
    if sk is not None:
        sur_tot = float(oo["surrogate"])
        print("\n--- obj_opt 代理项按 sign4 / single（|lhs−δ|·μ，与上面 surrogate 行一致）---", flush=True)
        for name in ("sign4", "single", "other"):
            b = sk[name]
            if int(b["n_rows"]) <= 0:
                continue
            w = float(b["opt_mu_weighted_sum"])
            print(f"  {name:8s}  {w:14.6g}  ({_pct(w, sur_tot)})", flush=True)

    otot = float(oo["total"])
    print("\n--- obj_opt 分项占比（相对 obj_opt.total）---", flush=True)
    for key in (
        "surrogate",
        "x_lower",
        "x_upper",
        "pg_lower",
        "pg_upper",
        "ramp_up",
        "ramp_down",
        "min_on",
        "min_off",
        "start_cost",
        "shut_cost",
        "coc_nonneg",
    ):
        v = float(oo[key])
        print(f"  {key:12s}  {v:14.6g}  ({_pct(v, otot)})", flush=True)

    lam = trainer.lambda_inherent[sid]
    if lam is not None:
        lx_lo = np.abs(np.asarray(lam["lambda_x_lower"], dtype=float))
        lx_up = np.abs(np.asarray(lam["lambda_x_upper"], dtype=float))
        print(
            f"\nlambda_x: max|lower|={float(np.max(lx_lo)):.6g}  max|upper|={float(np.max(lx_up)):.6g}  "
            f"sum|lower|={float(np.sum(lx_lo)):.6g}  sum|upper|={float(np.sum(lx_up)):.6g}",
            flush=True,
        )

    mu = np.asarray(trainer.mu[sid], dtype=float)
    print(
        f"mu: min={float(np.min(mu)):.6g} max={float(np.max(mu)):.6g} mean={float(np.mean(mu)):.6g}",
        flush=True,
    )

    lv = np.asarray(trainer.lambda_vals[sid], dtype=float)
    print(
        f"lambda_vals (电价): L2={float(np.linalg.norm(lv)):.6g} max|.|={float(np.max(np.abs(lv))):.6g}",
        flush=True,
    )

    if trainer.n_samples == 1:
        op, _, _, _, _, oopt = trainer.cal_viol_components()
        print(
            f"\n(cal_viol_components 校验) obj_primal={op:.6g} obj_opt={oopt:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
