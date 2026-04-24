#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 / 复现子问题 ``c_pg`` 可微损失的原始数据，用于针对性测试。

在**本文件顶部的用户配置区**设置 ``MODE`` 等；``ACTIVE_SET_JSON`` 默认同
``run_training_case118.CASE118_ACTIVE_SET_JSON``，支持 array 或含 ``all_samples`` 的
对象；若文件不存在则自动选用 ``result/commitment_clustering`` 下最新的
``pattern_library_case118*.json``。``MODEL_DIR`` 默认可选
``result/surrogate_models`` 下**最近**的 ``subproblem_models_case118_*`` 目录
（无则留空，需手写）。然后直接运行::

    python scripts/c_pg_loss_snapshot.py

- **export**：从 ``MODEL_DIR`` 加载 surrogate，写出 ``EXPORT_OUT_JSON`` 快照
- **test**：读 ``TEST_SNAPSHOT_JSON``，注入对偶量后仅跑 ``iter_with_c_pg_nn``。
  当 ``TEST_TRAINER_SOURCE = \"light_bake\"`` 时，从 ``TEST_BAKE_BUNDLE_DIR`` 加载
  本脚本 ``light_bake`` 产出的 ``dual_predictor.pth`` 与 ``surrogate_unit_*.pth``，
  不依赖历史 ``subproblem_models_case118_*`` 目录。
- **light_bake**：不复用子问题 checkpoint；按 case118 子问题 + light 风格（单机组、
  少样本、低并行、可覆写外循环轮次）**从头训练**对偶预测器 + 子问题 BCD 至指定轮次，
  再 ``collect_c_pg_loss_snapshot`` 写入 ``LIGHT_BAKE_OUT_JSON``；若设置
  ``LIGHT_BAKE_BUNDLE_DIR`` 则同目录落盘对偶/机组 pth 供 test 复用。

说明：训练结束后用 checkpoint 做 export 时，``lambda_inherent`` 以 pth 内为准。若需与某次 BCD
迭代内存完全一致，在训练循环里于该轮末尾调用 ``trainer.collect_c_pg_loss_snapshot()`` 并
``json.dump`` 即可；case118 子问题请保持 ``IGNORE_STARTUP_SHUTDOWN = True`` 与训练一致。

``light_bake`` / ``test``(light_bake) 时，**请保持**本文件中的 ``ACTIVE_SET_JSON``、
``UNIT_ID``、``T_DELTA``、``LP_BACKEND`` 等与 bake 时一致，否则维度或约束策略可能不匹配。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_active_set_json() -> str:
    """与 ``run_training_case118.CASE118_ACTIVE_SET_JSON`` 一致（case118 子问题训练默认数据）。"""
    try:
        from run_training_case118 import CASE118_ACTIVE_SET_JSON

        return str(CASE118_ACTIVE_SET_JSON)
    except Exception:
        return (
            "result/commitment_clustering/"
            "pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025"
            "_price_only_clipped.json"
        )


def _default_model_dir() -> str:
    """若存在 ``result/surrogate_models/subproblem_models_case118_*``，取最近修改的一个目录；否则空串。"""
    d = ROOT / "result" / "surrogate_models"
    if not d.is_dir():
        return ""
    cands = sorted(
        d.glob("subproblem_models_case118_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return ""
    try:
        return str(cands[0].relative_to(ROOT).as_posix())
    except ValueError:
        return str(cands[0])


# ═══════════════════════════════════════════════════════════════════════════
# 用户配置：在此修改，勿依赖命令行参数
# ═══════════════════════════════════════════════════════════════════════════

# "export"     = 从 checkpoint 导出 c_pg 损失快照 JSON
# "test"       = 读入快照，注入 lambda / 对偶，仅执行 c_pg 网络训练
# "light_bake" = 不加载子问题 pth；light 风格从头训对偶 + 单机组 BCD，再导出快照（可选 bundle）
# 重新跑全量 bake 时改回 "light_bake" 并先确认 LIGHT_BAKE_OUT_JSON / BUNDLE 路径。
MODE: str = "test"

# 与训练一致
CASE: str = "case118"
# 相对项目根，或写绝对路径。默认与 case118 训练入口相同。支持顶层数组或
# ``{ "all_samples": [ ... ] }``。若该路径不存在，会选用 result/commitment_clustering
# 下最新修改的 pattern_library_case118*.json。
ACTIVE_SET_JSON = "result/active_set/active_sets_case118_T0_n366_20260322_063917.json"
# 默认取 result/surrogate_models 下最新的 subproblem_models_case118_*；若无则空串，请在下方显式填写。
MODEL_DIR: str = _default_model_dir()
UNIT_ID: int = 0
T_DELTA: float = 1.0
# 须与训练 surrogate 时一致（如 case118 服务器用 "cvxpy_highs"）
LP_BACKEND: str = "gurobi"
# None 表示从 checkpoint 元数据解析；显式填则须与 pth 内一致
CONSTRAINT_STRATEGY: str | None = None
# case118 子问题训练常为 True
IGNORE_STARTUP_SHUTDOWN: bool = True

# --- 仅 export ---
EXPORT_OUT_JSON: str = "result/c_pg_snapshots/unit0.json"
# 与训练时 ``MAX_SAMPLES`` 对齐时截取前 N 条；None 表示用满 JSON 中全部样本
EXPORT_MAX_SAMPLES: int | None = 12
# 仅写入快照元数据 ``iter_number``；<0 表示不修改当前 trainer 的 iter_number
EXPORT_ITER_NUMBER_FOR_META: int = -1

# --- 仅 test ---
# 与 light_bake 时 LIGHT_BAKE_OUT_JSON 一致（或你复制后的路径）
TEST_SNAPSHOT_JSON: str = "result/c_pg_snapshots/light_bake_unit0.json"
# c_pg 只训步数/学习率：扫参时优先改这两处
TEST_C_PG_EPOCHS: int = 80
TEST_PG_COST_SURR_LR: float | None = 4e-4
# True：注入快照中的 rho/正则/软界等，与 collect 时尺度一致
TEST_STRICT_HPARAMS: bool = True
# 试训结果存盘；留空不保存。勿覆盖 bake_bundle 里的 surrogate
TEST_LAST_PTH: str = "result/c_pg_snapshots/c_pg_test_after_bake_unit0.pth"

# --- test：训练器来源 ---
# "checkpoint"   使用 MODEL_DIR
# "light_bake"   使用下面目录中的 dual_predictor.pth + surrogate_unit_*.pth（与 bake 同目录）
TEST_TRAINER_SOURCE: str = "light_bake"
# 须与 light_bake 时 LIGHT_BAKE_BUNDLE_DIR 一致
TEST_BAKE_BUNDLE_DIR: str = "result/c_pg_snapshots/bake_bundle_unit0"

# --- 仅 light_bake（对齐 run_training_case118_subproblem_bcd_light 思想：少样本、低并行、可覆写轮次）---
# 与 case118 的 SUBPROBLEM_SOLVE_PRESET 一致："desktop" | "server"
LIGHT_BAKE_SUBPROBLEM_PRESET: str = "desktop"
LIGHT_BAKE_MAX_SAMPLES: int = 10
LIGHT_BAKE_MAX_ITER: int | None = 40  # BCD 外循环轮次；None 表示不覆写 c118 的默认 MaxIter
LIGHT_BAKE_N_WORKERS_UNIT: int = 1
LIGHT_BAKE_N_WORKERS_SAMPLE: int = 1
# 若子问题外循环被 LIGHT_BAKE_MAX_ITER 覆写，case118 会重算 warmup；此处在其后再次覆写
# RT 的 SUBPROBLEM_PREDICTOR_WARMUP_ROUNDS；为 None 则保留上一步重算结果
LIGHT_BAKE_PREDICTOR_WARMUP_ROUNDS: int | None = None
# False 时不预训练 unit_predictor、不加载外部 unit 权重，子问题结构最简单
LIGHT_BAKE_TRAIN_UNIT_PREDICTOR: bool = False
# None 表示使用 case118 配置后的 rt.DUAL_EPOCHS；可改小以加快试验
LIGHT_BAKE_DUAL_EPOCHS: int | None = 32
LIGHT_BAKE_OUT_JSON: str = "result/c_pg_snapshots/light_bake_unit0.json"
# 非空则写入 dual_predictor.pth 与 surrogate_unit_{UNIT_ID}.pth，供 TEST_TRAINER_SOURCE=light_bake
LIGHT_BAKE_BUNDLE_DIR: str = "result/c_pg_snapshots/bake_bundle_unit0"


def _load_active_set_samples(path: Path) -> list:
    """与训练时一致：``{"all_samples": [ ... ]}`` 时走 ``load_active_set_from_json``，将
    ``pd_data`` 等恢复为与 ``load_trained_models`` 兼容的结构；若顶层为样本列表则原样返回。
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("all_samples"), list):
        from src.uc_NN_subproblem import load_active_set_from_json

        return load_active_set_from_json(str(path))
    raise ValueError(
        f"expected JSON array or object with 'all_samples' list in {path}"
    )


def _resolve_active_set_path() -> Path:
    """``ACTIVE_SET_JSON`` 若不存在，则在 ``result/commitment_clustering`` 下按修改时间选最新的
    ``pattern_library_case118*.json``。
    """
    raw = str(ACTIVE_SET_JSON).strip()
    if raw:
        p = _abs(ROOT, raw)
        if p.is_file():
            return p
    cdir = ROOT / "result" / "commitment_clustering"
    if cdir.is_dir():
        cands = sorted(
            cdir.glob("pattern_library_case118*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if cands:
            rel = cands[0]
            try:
                shown = rel.relative_to(ROOT).as_posix()
            except ValueError:
                shown = str(rel)
            print(
                f"[c_pg_loss_snapshot] 未找到 ACTIVE_SET_JSON={raw!r}，"
                f"已自动使用: {shown}",
                flush=True,
            )
            return rel
    raise FileNotFoundError(
        f"找不到 active set 文件。请设置 ACTIVE_SET_JSON，或放入 "
        f"{(ROOT / 'result' / 'commitment_clustering').as_posix()}/pattern_library_case118*.json"
    )


def _abs(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_model_dir() -> str:
    """与 run_export / run_test 共用的子问题模型目录；非法时抛出带说明的 FileNotFoundError。"""
    if not str(MODEL_DIR).strip():
        raise FileNotFoundError(
            "MODEL_DIR 为空，且未在 result/surrogate_models 下发现 subproblem_models_case118_*。\n"
            "请在「用户配置」里将 MODEL_DIR 设为含 surrogate_unit_*.pth 与 dual_predictor.pth 的目录"
            "（相对项目根或绝对路径）。"
        ) from None
    path = str(_abs(ROOT, MODEL_DIR))
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"找不到子问题模型目录: {path}\n"
            "请检查 MODEL_DIR 是否为某次 run_training 生成的 subproblem_models_case118_* 路径。"
        ) from None
    return path


def _bootstrap_case118_light() -> None:
    """与 case118 子问题 + light 覆写一致，将 ``run_training`` 全局配好（不跑 ``rt.main``）。"""
    import run_training_case118 as c118

    active_path = _resolve_active_set_path()
    try:
        c118.CASE118_ACTIVE_SET_JSON = str(active_path.relative_to(ROOT))
    except ValueError:
        c118.CASE118_ACTIVE_SET_JSON = str(active_path)
    c118.CASE118_SUBPROBLEM_UNIT_IDS = [int(UNIT_ID)]
    c118.SUBPROBLEM_LIGHT_MAX_SAMPLES = max(1, int(LIGHT_BAKE_MAX_SAMPLES))
    c118.SUBPROBLEM_LIGHT_N_WORKERS_UNIT = max(1, int(LIGHT_BAKE_N_WORKERS_UNIT))
    c118.SUBPROBLEM_LIGHT_N_WORKERS_SAMPLE = max(1, int(LIGHT_BAKE_N_WORKERS_SAMPLE))
    c118.SUBPROBLEM_SOLVE_PRESET = str(LIGHT_BAKE_SUBPROBLEM_PRESET).strip().lower()
    if LIGHT_BAKE_MAX_ITER is not None:
        c118.SUBPROBLEM_LIGHT_MAX_ITER = max(1, int(LIGHT_BAKE_MAX_ITER))
    if LIGHT_BAKE_PREDICTOR_WARMUP_ROUNDS is not None:
        c118.SUBPROBLEM_LIGHT_PREDICTOR_WARMUP_ROUNDS = max(
            0, int(LIGHT_BAKE_PREDICTOR_WARMUP_ROUNDS)
        )
    c118.CASE118_USE_UNIT_PREDICTOR = bool(LIGHT_BAKE_TRAIN_UNIT_PREDICTOR)
    if not c118.CASE118_USE_UNIT_PREDICTOR:
        c118.CASE118_UNIT_PREDICTOR_LOAD_PATH = None

    c118._configure_common()
    c118._configure_subproblem_bcd()
    c118._apply_subproblem_light_runtime_overrides()


def _apply_run_training_script_overrides() -> None:
    """在 case118 配置后覆写与命令行/本脚本用户区一致的项。"""
    from src.subproblem_lp_solver import normalize_lp_backend
    import run_training as rt

    rt.T_DELTA = float(T_DELTA)
    if str(LP_BACKEND).strip():
        rt.SUBPROBLEM_LP_BACKEND = normalize_lp_backend(
            str(LP_BACKEND).strip().lower()
        )
    if CONSTRAINT_STRATEGY is not None:
        rt.SURROGATE_CONSTRAINT_STRATEGY = str(CONSTRAINT_STRATEGY)
    if LIGHT_BAKE_DUAL_EPOCHS is not None:
        rt.DUAL_EPOCHS = max(1, int(LIGHT_BAKE_DUAL_EPOCHS))


def _make_fresh_subproblem_trainer(
    ppc,
    all_samples: list,
    unit_id: int,
    dual_predictor,
    *,
    use_unit_predictor: bool = False,
    unit_predictor=None,
):
    from run_training import (
        SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS,
        create_subproblem_trainer,
        resolve_nn_hidden_dims,
    )
    from src.subproblem_lp_solver import normalize_lp_backend
    from src.uc_NN_subproblem import normalize_constraint_generation_strategy
    import run_training as rt

    cstr = normalize_constraint_generation_strategy(
        str(rt.SURROGATE_CONSTRAINT_STRATEGY)
    )
    _, nn_h = resolve_nn_hidden_dims(
        rt.SUBPROBLEM_NN_SIZE, SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS, "SUBPROBLEM_NN_SIZE"
    )
    _, cpg_h = resolve_nn_hidden_dims(
        rt.SUBPROBLEM_C_PG_NN_SIZE,
        SUBPROBLEM_NN_HIDDEN_DIM_OPTIONS,
        "SUBPROBLEM_C_PG_NN_SIZE",
    )
    pg_wd = getattr(
        rt,
        "SUBPROBLEM_PG_COST_C_PG_ADAM_WD",
        None,
    )
    return create_subproblem_trainer(
        ppc,
        all_samples,
        float(rt.T_DELTA),
        int(unit_id),
        n_workers=1,
        lambda_predictor=dual_predictor,
        lp_backend=normalize_lp_backend(str(rt.SUBPROBLEM_LP_BACKEND).strip().lower()),
        constraint_generation_strategy=cstr,
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
        unit_predictor=unit_predictor,
        use_unit_predictor=use_unit_predictor
        and unit_predictor is not None,
        unit_predictor_finetune_lr=rt.UNIT_PREDICTOR_FINETUNE_LR,
        unit_predictor_weight_decay=rt.UNIT_PREDICTOR_WEIGHT_DECAY,
    )


def _apply_snapshot_to_trainer(trainer, snap: dict, strict_hparams: bool) -> None:
    import numpy as np

    if int(snap.get("schema_version", 0)) != 1:
        print(
            f"警告: 未知 schema_version={snap.get('schema_version')!r}，仍尝试应用",
            flush=True,
        )
    n = len(snap["samples"])
    if n != int(trainer.n_samples):
        raise ValueError(
            f"snapshot n_samples={n} 与当前 trainer.n_samples={trainer.n_samples} 不一致"
        )
    t_expect = int(snap["T"])
    if t_expect != int(trainer.T):
        raise ValueError(f"snapshot T={t_expect} 与 trainer.T={trainer.T} 不一致")

    lam_rows = [s["lambda_vals"] for s in snap["samples"]]
    trainer.lambda_vals = np.asarray(lam_rows, dtype=np.float64).reshape(
        n, t_expect
    )
    for i, s in enumerate(snap["samples"]):
        if s.get("lambda_inherent_is_none"):
            continue
        cpg = s.get("lambda_inherent_c_pg")
        if not cpg:
            continue
        li = trainer.lambda_inherent[i]
        if li is None:
            print(
                f"警告: sample {i} snapshot 有 lambda_inherent_c_pg 但 "
                f"trainer.lambda_inherent[i] 为 None，跳过",
                flush=True,
            )
            continue
        for k, v in cpg.items():
            if v is not None and k in li:
                li[k] = np.asarray(v, dtype=np.float64).reshape(-1)

    prev = snap.get("prev_pg_cost_values")
    if prev is not None:
        trainer._prev_pg_cost_values = np.asarray(prev, dtype=np.float64)
    else:
        trainer._prev_pg_cost_values = None

    trainer.iter_number = int(snap.get("iter_number", 0))

    if strict_hparams:
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


def run_light_bake() -> None:
    """从头训练对偶 + 单机组子问题 BCD，不加载已有 subproblem checkpoint。"""
    if LIGHT_BAKE_TRAIN_UNIT_PREDICTOR:
        raise ValueError(
            "c_pg_loss_snapshot 的 light_bake 未接入 unit_predictor 预训练；"
            "请将 LIGHT_BAKE_TRAIN_UNIT_PREDICTOR 设为 False"
        )
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import train_dual_predictor_from_data
    import run_training as rt

    _bootstrap_case118_light()
    _apply_run_training_script_overrides()

    active_path = _resolve_active_set_path()
    all_samples = _load_active_set_samples(active_path)
    n_cap = int(rt.MAX_SAMPLES) if rt.MAX_SAMPLES is not None else len(all_samples)
    n_use = min(len(all_samples), max(1, n_cap))
    all_samples = all_samples[:n_use]
    t_delta = float(rt.T_DELTA)
    unit = int(UNIT_ID)
    ppc = get_case_ppc(CASE)

    print(
        f"[light_bake] case={CASE} n_samples={n_use} unit={unit} "
        f"max_iter={rt.SUBPROBLEM_MAX_ITER} nn_epochs={rt.NN_EPOCHS} "
        f"preset={LIGHT_BAKE_SUBPROBLEM_PRESET!r} backend={rt.SUBPROBLEM_LP_BACKEND!r}",
        flush=True,
    )

    bundle = str(LIGHT_BAKE_BUNDLE_DIR).strip()
    dual_save = None
    if bundle:
        bdir = _abs(ROOT, bundle)
        bdir.mkdir(parents=True, exist_ok=True)
        dual_save = str(bdir / "dual_predictor.pth")

    dual = train_dual_predictor_from_data(
        ppc,
        all_samples,
        T_delta=t_delta,
        num_epochs=int(rt.DUAL_EPOCHS),
        batch_size=int(rt.DUAL_BATCH_SIZE),
        batch_strategy=str(rt.DUAL_BATCH_STRATEGY),
        shuffle=bool(rt.DUAL_SHUFFLE),
        learning_rate=float(rt.DUAL_LR),
        save_path=dual_save,
        dual_net_variant=str(rt.DUAL_PREDICTOR_NET_VARIANT),
        dual_normalize_targets=bool(rt.DUAL_PREDICTOR_NORMALIZE_TARGETS),
        dual_cosine_loss_weight=float(rt.DUAL_PREDICTOR_COSINE_LOSS_WEIGHT),
        dual_smooth_l1_beta=float(rt.DUAL_PREDICTOR_SMOOTH_L1_BETA),
    )

    tr = _make_fresh_subproblem_trainer(
        ppc,
        all_samples,
        unit,
        dual,
        use_unit_predictor=False,
        unit_predictor=None,
    )
    tr.iter(
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
    snap = tr.collect_c_pg_loss_snapshot()
    out = _abs(ROOT, LIGHT_BAKE_OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"[light_bake] 已写入快照 {out}", flush=True)
    s0 = snap["samples"][0]
    print(
        f"  示例 sample0: loss_total={s0['loss_total']:.6f} "
        f"obj_dual_pg={s0['obj_dual_pg']:.6f} reg={s0['reg_term']:.6f}",
        flush=True,
    )
    if bundle:
        bdir = _abs(ROOT, bundle)
        unit_p = bdir / f"surrogate_unit_{unit}.pth"
        tr.save(str(unit_p))
        print(
            f"[light_bake] 已保存 bundle: {dual_save!r} , {str(unit_p)!r}",
            flush=True,
        )


def run_export() -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    case = CASE
    ppc = get_case_ppc(case)
    active_path = _resolve_active_set_path()
    all_samples = _load_active_set_samples(active_path)
    if EXPORT_MAX_SAMPLES is not None and len(all_samples) > int(EXPORT_MAX_SAMPLES):
        all_samples = all_samples[: int(EXPORT_MAX_SAMPLES)]
    t_delta = float(T_DELTA)
    unit = int(UNIT_ID)
    model_dir = _resolve_model_dir()

    print(
        f"[export] case={case} n_samples={len(all_samples)} unit={unit} model_dir={model_dir}",
        flush=True,
    )
    _dual, trainers = load_trained_models(
        ppc,
        all_samples,
        t_delta,
        model_dir,
        unit_ids=[unit],
        lp_backend=str(LP_BACKEND).strip().lower(),
        constraint_generation_strategy=CONSTRAINT_STRATEGY,
        ignore_startup_shutdown_costs=bool(IGNORE_STARTUP_SHUTDOWN),
    )
    if unit not in trainers:
        raise RuntimeError(f"未能加载机组 {unit} 的 surrogate")
    tr = trainers[unit]
    if int(EXPORT_ITER_NUMBER_FOR_META) >= 0:
        tr.iter_number = int(EXPORT_ITER_NUMBER_FOR_META)
    snap = tr.collect_c_pg_loss_snapshot()
    out = _abs(ROOT, EXPORT_OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"[export] 已写入 {out}", flush=True)
    s0 = snap["samples"][0]
    print(
        f"  示例 sample0: loss_total={s0['loss_total']:.6f} "
        f"obj_dual_pg={s0['obj_dual_pg']:.6f} reg={s0['reg_term']:.6f}",
        flush=True,
    )


def run_test() -> None:
    from src.case_registry import get_case_ppc
    from src.uc_NN_subproblem import load_trained_models

    snap_path = _abs(ROOT, TEST_SNAPSHOT_JSON)
    with snap_path.open("r", encoding="utf-8") as f:
        snap = json.load(f)
    case = CASE
    ppc = get_case_ppc(case)
    active_path = _resolve_active_set_path()
    all_samples = _load_active_set_samples(active_path)
    n_need = int(snap["n_samples"])
    if len(all_samples) < n_need:
        raise ValueError(
            f"active set 只有 {len(all_samples)} 条，snapshot 需要 {n_need} 条"
        )
    all_samples = all_samples[:n_need]
    t_delta = float(T_DELTA)
    unit = int(UNIT_ID)
    if int(snap["unit_id"]) != unit:
        print(
            f"警告: snapshot unit_id={snap['unit_id']} 与 UNIT_ID={unit} 不一致，以 UNIT_ID 为准",
            flush=True,
        )
    tr_src = str(TEST_TRAINER_SOURCE).strip().lower()
    if tr_src == "light_bake":
        if not str(TEST_BAKE_BUNDLE_DIR).strip():
            raise FileNotFoundError(
                "TEST_TRAINER_SOURCE=light_bake 时需要设置有效的 TEST_BAKE_BUNDLE_DIR\n"
                "（应先运行 MODE=light_bake 并令 LIGHT_BAKE_BUNDLE_DIR 与之一致，"
                "且目录内含 dual_predictor.pth 与 surrogate_unit_{UNIT_ID}.pth）"
            )
        # 与 bake 时同一套 run_training 超参，避免约束名 / 维度不一致
        _bootstrap_case118_light()
        _apply_run_training_script_overrides()
        bundle = str(_abs(ROOT, str(TEST_BAKE_BUNDLE_DIR).strip()))
        if not Path(bundle).is_dir():
            raise FileNotFoundError(
                f"TEST_BAKE_BUNDLE_DIR 不是目录: {bundle}"
            )
        model_dir = bundle
    else:
        model_dir = _resolve_model_dir()
    _dual, trainers = load_trained_models(
        ppc,
        all_samples,
        t_delta,
        model_dir,
        unit_ids=[unit],
        lp_backend=str(LP_BACKEND).strip().lower(),
        constraint_generation_strategy=CONSTRAINT_STRATEGY,
        ignore_startup_shutdown_costs=bool(IGNORE_STARTUP_SHUTDOWN),
    )
    tr = trainers[unit]
    _apply_snapshot_to_trainer(tr, snap, strict_hparams=bool(TEST_STRICT_HPARAMS))
    m0 = tr.cal_nn_logging_components()
    print(
        f"[test] 注入后 [NN-metric] obj_dual_pg={m0['obj_dual_pg']:.6f} reg_pg={m0['reg_pg']:.6f}",
        flush=True,
    )
    tr.iter_with_c_pg_nn(
        num_epochs=int(TEST_C_PG_EPOCHS),
        learning_rate=float(TEST_PG_COST_SURR_LR)
        if TEST_PG_COST_SURR_LR is not None
        else None,
    )
    m1 = tr.cal_nn_logging_components()
    print(
        f"[test] 训练后 [NN-metric] obj_dual_pg={m1['obj_dual_pg']:.6f} reg_pg={m1['reg_pg']:.6f}",
        flush=True,
    )
    if str(TEST_LAST_PTH).strip():
        p = _abs(ROOT, TEST_LAST_PTH)
        p.parent.mkdir(parents=True, exist_ok=True)
        tr.save(str(p))
        print(f"[test] 已保存 c_pg 后 checkpoint: {p}", flush=True)


def main() -> None:
    # 若仍出现 "arguments are required: cmd"，说明正在执行旧版含 argparse 的缓存/副本；请保存本文件后重试。
    print(
        f"[c_pg_loss_snapshot] 内置配置模式（无子命令/无 argparse） MODE={MODE!r}",
        flush=True,
    )
    m = str(MODE).strip().lower()
    if m == "export":
        run_export()
    elif m == "test":
        run_test()
    elif m in ("light_bake", "lightbake", "bake"):
        if m in ("lightbake", "bake"):
            print(
                f"提示: MODE={MODE!r} 将按 light_bake 执行。",
                flush=True,
            )
        run_light_bake()
    else:
        print(
            f"错误: MODE={MODE!r} 须为 'export'、'test' 或 'light_bake'。"
            f"请编辑本文件顶部的用户配置。",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
