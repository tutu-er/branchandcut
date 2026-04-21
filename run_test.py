#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本（多模式�?
- surrogate: 加载已训练的 V3 代理约束模型，输出参数摘要，可选运行可行性泵
- bcd:       加载已训练的 BCD 神经网络模型，报告参数统�?
- both:      联合加载 BCD + surrogate 模型，以全体代理约束评估解质量，可�?FP

修改顶部�?MODE / MODEL_DIR / BCD_MODEL_PATH 等变量切换执行模式�?
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

from gurobipy import GRB

# ──────────────────────── 依赖检�?────────────────────────


def check_and_install_dependencies():
    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gurobipy': 'gurobipy',
        'pypower': 'PYPOWER',
    }
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"[OK] {import_name}")
        except ImportError:
            missing.append(package_name)
            print(f"[MISS] {import_name} 未安装")

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        response = input("是否自动安装? (y/n): ")
        if response.strip().lower() == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("[OK] 安装完成")
            return True
        else:
            print("请手动安装后重试")
            return False
    return True


def check_and_install_dependencies_safe():
    try:
        return check_and_install_dependencies()
    except EOFError:
        print("Non-interactive session detected; please install the missing dependencies manually.")
        return False


if not check_and_install_dependencies_safe():
    sys.exit(1)

# ──────────────────────── 模式配置 ────────────────────────
#
#   'surrogate' - 加载 V3 代理约束模型并测�?
#   'bcd'       - 加载 BCD 神经网络模型并报告参数统�?
#   'both'      - 联合加载 BCD + surrogate，以全体代理约束评估（需同时配置下面两个路径�?
#
MODE      = 'surrogate'
RUN_FP    = True       # surrogate / both 模式：是否运行可行性泵测试
CASE_NAME = 'case3lite'   # 'case3' / 'case3lite' / 'case14' / 'case30' / 'case39' / 'case118'
SURROGATE_CONSTRAINT_STRATEGY = 'all_single_time'  # 'auto' / 'sensitive' / 'all' / 'all_templates_sign4' / 'all_single_time'
SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = False
BCD_LAMBDA_INIT_STRATEGY = 'lp_relaxation'   # 'lp_relaxation' / 'ed_on_x_opt'
THETA_HOT_START_STRATEGY = 'dcpf_relative'   # 'dcpf_relative' / 'gaussian'
ZETA_HOT_START_STRATEGY = 'zero'             # 'zero' / 'gaussian'
THETA_GAUSSIAN_STD = 0.01
ZETA_GAUSSIAN_STD = 0.01
BCD_RHO_PRIMAL_INIT = 1e-2
BCD_RHO_DUAL_INIT = 1e-2
BCD_RHO_DUAL_PG_INIT = 1e-2
BCD_RHO_DUAL_X_INIT = 1e-2
BCD_RHO_DUAL_COC_INIT = 1e-2
BCD_RHO_OPT_INIT = 1e-2
BCD_RESTORE_RHO_FROM_CHECKPOINT = False
BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = 10
BCD_GAMMA_BASE = 1e-2

# surrogate / both 模式：已训练 surrogate 模型目录（训练时输出的带时间戳路径）
# 设为 None 则自动查找 result/surrogate_models/ 下最新的匹配目录
# 环境变量 RUN_TEST_SURROGATE_MODEL_DIR 可覆盖本常量（agentic_fp_optimizer 会注入）
MODEL_DIR = None
# 可选：测试时注入外部 unit_predictor（当 MODEL_DIR 下缺少 unit_predictor.pth 时自动复制一份）
# 环境变量 RUN_TEST_UNIT_PREDICTOR_DIR 可覆盖本常量
UNIT_PREDICTOR_DIR = None

# bcd / both 模式：已训练 BCD 模型 .pth 文件路径
# 设为 None 则自动查找 result/bcd_models/ 下最新的匹配文件
# 环境变量 RUN_TEST_BCD_MODEL_PATH 可覆盖本常量
BCD_MODEL_PATH = None

# 绘图开关：在自动化评估（如 agentic_fp_optimizer）中建议禁用绘图以避免空数据导致的 matplotlib 崩溃
# - 设环境变量 RUN_TEST_DISABLE_PLOTS=1 可跳过所有 plot_* 调用
RUN_TEST_DISABLE_PLOTS = (os.environ.get("RUN_TEST_DISABLE_PLOTS", "").strip() not in ("", "0", "false", "False"))


def _auto_discover_model_path(directory, glob_pattern, label):
    """在 directory 下按 glob_pattern 查找最新文件/目录。"""
    d = Path(__file__).parent / directory
    if not d.exists():
        return None
    candidates = sorted(d.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        rel = str(candidates[0].relative_to(Path(__file__).parent))
        print(f"[自动发现] {label}: {candidates[0].name}")
        return rel
    return None

# 顶部集中配置区：测试相关参数统一在这里调整
MAX_SAMPLES = None         # None = 使用全部样本
SAMPLE_RANGE = "0:1"       # 左闭右开；例如 "210:220"
T_DELTA = 1.0
UNIT_IDS = [1]            # None = 所有机组；或如 [0, 1, 2]
TEST_SAMPLES_DEFAULT = 3
TEST_SAMPLES = 10
RUN_SUBPROBLEM_MILP_TEST = True
FP_MAX_ITER = 50
FP_CONF_THRESHOLD = 0.15
FP_MAX_PERTURBATION_HOT_STARTS = 6
FP_MAX_UNIT_OPTIONS_PER_GENERATOR = 4
FP_MAX_UNIT_COMBINATION_CANDIDATES = 12
FP_MAX_NEARBY_COMMITMENT_HOT_STARTS = 4
FP_NEARBY_COMMITMENT_POOL_SIZE = 12
FP_PARALLEL_STARTS = 2
FP_SURROGATE_SCREEN_MODE = 'robust'   # 'none' / 'robust'
FP_SURROGATE_SCREEN_MAX_CONSTRAINTS_PER_UNIT = 3
FP_SURROGATE_SCREEN_MIN_SUPPORT_RATIO = 0.85
FP_SURROGATE_SCREEN_MAX_NORMALIZED_VIOLATION = 0.05
FP_SURROGATE_SCREEN_MIN_MEAN_MARGIN = 0.02
FP_SURROGATE_SCREEN_CANDIDATE_VIOLATION_TOL = 0.02
FP_SURROGATE_SCREEN_SOFT_PENALTY = 25.0
FP_PROJECTION_OBJECTIVE_TAU = 'adaptive'
USE_CASE3LITE_CUSTOM_FP = True
CASE3LITE_CUSTOM_FP_MAX_GLOBAL_COMBINATIONS = 24
CASE3LITE_CUSTOM_FP_PLOT_DIR = 'result/figures_case3lite_custom_fp'
ACTIVE_SETS_FILE = 'result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json'  # 指定 active_sets JSON 文件路径（None=自动查找最新）

# ──────────────────────── 导入 ────────────────────────

import numpy as np

# NumPy 2.x 移除了顶�?np.in1d，一些旧依赖（例如部分电力系统相关包）仍会导入它�?
# 这里补一个兼容别名，避免在导入第三方模块时直接失败�?
if not hasattr(np, "in1d"):
    def _compat_in1d(ar1, ar2, assume_unique=False, invert=False):
        return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)

    np.in1d = _compat_in1d

# matplotlib 为可选依赖，绘图功能在不可用时自动跳�?
try:
    import matplotlib
    matplotlib.use('Agg')   # 非交互后端，兼容无显示环�?
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

try:
    import pypower.case14
    import pypower.case30
    import pypower.case39
    import pypower.case118
    from ed_gurobipy import EconomicDispatchGurobi
    from uc_NN_subproblem import (
        load_trained_models,
        ActiveSetReader,
        build_surrogate_constraint_expression,
        normalize_constraint_generation_strategy,
        resolve_constraint_offsets_from_trainer,
        _load_surrogate_model_metadata,
    )
    from case_registry import get_case_ppc
    from mti118_data_loader import load_case118_ppc_with_mti_limits
    from scenario_utils import normalize_sample_arrays
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本，且 src/ 目录存在")
    sys.exit(1)

if MODE in ('bcd', 'both'):
    try:
        from uc_NN_BCD import load_active_set_from_json, Agent_NN_BCD
    except ImportError as e:
        print(f"BCD 模块导入失败: {e}")
        sys.exit(1)

if MODE in ('surrogate', 'both'):
    try:
        from feasibility_pump import (
            recover_integer_solution,
            solve_global_LP_relaxation,
            solve_global_LP_relaxation_without_surrogate,
            _solve_unit_LP_with_surrogate,
            _solve_unit_MILP_with_surrogate,
            _extract_unit_lambda,
            _resolve_surrogate_constraint_timesteps,
        )
        from feasibility_pump_case3lite import recover_integer_solution_case3lite
    except ImportError as e:
        print(f"feasibility_pump 模块导入失败: {e}")
        sys.exit(1)

# ──────────────────────── 工具函数 ────────────────────────


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


SECTION_WIDTH = 70


def log_section(title: str, char: str = "=", width: int = SECTION_WIDTH) -> None:
    print("\n" + char * width)
    log(title)
    print(char * width)


def log_rule(char: str = "-", width: int = SECTION_WIDTH) -> None:
    print("\n" + char * width)


def _fmt_sample_tag(index: int, total: int, pd_shape) -> str:
    return f"样本 {index + 1}/{total} | pd_shape={tuple(pd_shape)}"


def _fmt_masked_commitment_metrics(label: str, metrics: dict) -> str:
    hamming = metrics.get('hamming')
    hamming_text = "n/a" if hamming is None else str(hamming)
    return (
        f"{label}: coverage={metrics['covered']}/{metrics['total']} | "
        f"integ={metrics['integrality_gap']:.4f} | "
        f"Hamming={hamming_text}"
    )


def _fmt_distance_metrics(label: str, l1: float, hamming: int | None) -> str:
    hamming_text = "n/a" if hamming is None else str(hamming)
    return f"{label}: L1={l1:.4f} | Hamming={hamming_text}"


def _log_solution_similarity_summary(
    lhs_name: str,
    lhs_list: list[np.ndarray],
    rhs_name: str,
    rhs_list: list[np.ndarray],
    tol: float = 1e-8,
) -> None:
    """Summarize whether two groups of commitment matrices are genuinely different."""
    n = min(len(lhs_list), len(rhs_list))
    if n == 0:
        return

    exact_same = 0
    rounded_same = 0
    l1_diffs = []
    max_diffs = []
    for lhs, rhs in zip(lhs_list[:n], rhs_list[:n]):
        lhs_arr = np.asarray(lhs, dtype=float)
        rhs_arr = np.asarray(rhs, dtype=float)
        diff = np.abs(lhs_arr - rhs_arr)
        l1_diffs.append(float(np.sum(diff)))
        max_diffs.append(float(np.max(diff)))
        if np.all(diff <= tol):
            exact_same += 1
        if np.array_equal((lhs_arr >= 0.5).astype(int), (rhs_arr >= 0.5).astype(int)):
            rounded_same += 1

    log(
        f"{lhs_name} vs {rhs_name}: exact_same={exact_same}/{n}, "
        f"rounded_same={rounded_same}/{n}, "
        f"mean_l1={float(np.mean(l1_diffs)):.4f}, max_abs={float(np.max(max_diffs)):.4e}"
    )
    if exact_same == n:
        log(
            f"警告: {lhs_name} 与 {rhs_name} 在当前样本上完全相同。"
            f"这通常表示附加约束/惩罚没有实际改变 LP 解。"
        )


def _resolve_surrogate_display_layout(trainer, sample: dict | None, n_constraints: int):
    timestep_map = _resolve_surrogate_constraint_timesteps(
        trainer,
        sample,
        trainer.T,
        n_constraints,
    )
    sample_id = None
    if isinstance(sample, dict):
        candidate = sample.get('sample_id', sample.get('source_sample_id'))
        if candidate is not None:
            try:
                sample_id = int(candidate)
            except (TypeError, ValueError):
                sample_id = None
    offset_map = resolve_constraint_offsets_from_trainer(trainer, sample_id, len(timestep_map))
    return timestep_map, offset_map


def _format_surrogate_constraint(a: float, b: float, c: float, timestep: int, offsets, rhs: float) -> str:
    terms = []
    if 0 in offsets:
        terms.append(f"{a:.3f}*x[{timestep}]")
    if 1 in offsets:
        terms.append(f"{b:.3f}*x[{timestep + 1}]")
    if 2 in offsets:
        terms.append(f"{c:.3f}*x[{timestep + 2}]")
    lhs_text = " + ".join(terms) if terms else "0"
    return f"{lhs_text} <= {rhs:.3f}"


def _validate_lambda_prediction_shape(lambda_val, ng: int, T: int) -> tuple[str, tuple]:
    """Validate lambda predictor outputs used by run_test and report their mode."""
    if isinstance(lambda_val, dict):
        if 'lambda_pg_electricity_price' in lambda_val:
            arr = np.asarray(lambda_val['lambda_pg_electricity_price'], dtype=float)
            if arr.shape == (T, ng):
                arr = arr.T
            if arr.shape != (ng, T):
                raise ValueError(
                    f"lambda_pg_electricity_price shape mismatch: got {arr.shape}, expected {(ng, T)}"
                )
            return 'dict.lambda_pg_electricity_price', tuple(arr.shape)
        if 'lambda_pg_effective' in lambda_val:
            arr = np.asarray(lambda_val['lambda_pg_effective'], dtype=float)
            if arr.shape == (T, ng):
                arr = arr.T
            if arr.shape != (ng, T):
                raise ValueError(
                    f"lambda_pg_effective shape mismatch: got {arr.shape}, expected {(ng, T)}"
                )
            return 'dict.lambda_pg_effective', tuple(arr.shape)
        if 'lambda_power_balance' in lambda_val:
            arr = np.asarray(lambda_val['lambda_power_balance'], dtype=float).reshape(-1)
            if arr.shape != (T,):
                raise ValueError(
                    f"lambda_power_balance shape mismatch: got {arr.shape}, expected {(T,)}"
                )
            return 'dict.lambda_power_balance', tuple(arr.shape)
        raise ValueError(f"Unsupported lambda predictor dict keys: {sorted(lambda_val.keys())}")

    arr = np.asarray(lambda_val, dtype=float)
    if arr.shape == (T,):
        return 'array.power_balance_only', tuple(arr.shape)
    if arr.shape == (ng, T):
        return 'array.per_unit_effective', tuple(arr.shape)
    if arr.shape == (T, ng):
        return 'array.per_unit_effective_transposed', tuple(arr.shape)
    raise ValueError(
        f"Unsupported lambda predictor output shape {arr.shape}; expected {(T,)}, {(ng, T)}, or dict payload"
    )


def _log_lambda_prediction_summary(
    dual_predictor,
    sample: dict,
    ng: int,
    T: int,
    prefix: str,
) -> None:
    """Print one-line diagnostics for the current lambda predictor output."""
    lambda_val = dual_predictor.predict(sample)
    mode, shape = _validate_lambda_prediction_shape(lambda_val, ng, T)
    predictor_mode = getattr(dual_predictor, '_legacy_mode', None)
    arr = np.asarray(lambda_val if not isinstance(lambda_val, dict) else (
        lambda_val.get('lambda_pg_electricity_price')
        if 'lambda_pg_electricity_price' in lambda_val
        else lambda_val.get('lambda_pg_effective', lambda_val.get('lambda_power_balance'))
    ), dtype=float)
    log(
        f"{prefix} lambda predictor: mode={mode}, predictor_legacy_mode={predictor_mode}, "
        f"shape={shape}, mean={float(np.mean(arr)):.4f}, std={float(np.std(arr)):.4f}, "
        f"maxabs={float(np.max(np.abs(arr))):.4f}"
    )


def _resolve_requested_surrogate_strategy(
    requested_strategy: str | None,
) -> str | None:
    if requested_strategy is None:
        return None
    strategy_text = str(requested_strategy).strip().lower()
    if strategy_text in ("", "auto", "saved", "checkpoint"):
        return None
    return normalize_constraint_generation_strategy(strategy_text)


def _load_saved_surrogate_strategy(model_dir: str, unit_ids) -> str | None:
    if unit_ids is None:
        unit_candidates = list(range(256))
    else:
        unit_candidates = list(unit_ids)

    discovered = []
    for g in unit_candidates:
        surrogate_path = Path(model_dir) / f"surrogate_unit_{g}.pth"
        if not surrogate_path.exists():
            continue
        metadata = _load_surrogate_model_metadata(str(surrogate_path))
        discovered.append(
            normalize_constraint_generation_strategy(
                metadata.get("constraint_generation_strategy", "sensitive")
            )
        )

    if not discovered:
        return None
    if len(set(discovered)) > 1:
        raise ValueError(
            f"Inconsistent surrogate strategies found in {model_dir}: {sorted(set(discovered))}"
        )
    return discovered[0]


def _load_saved_subproblem_ignore_startup_shutdown_costs(model_dir: str, unit_ids) -> bool | None:
    if unit_ids is None:
        unit_candidates = list(range(256))
    else:
        unit_candidates = list(unit_ids)

    discovered = []
    for g in unit_candidates:
        surrogate_path = Path(model_dir) / f"surrogate_unit_{g}.pth"
        if not surrogate_path.exists():
            continue
        metadata = _load_surrogate_model_metadata(str(surrogate_path))
        discovered.append(bool(metadata.get("ignore_startup_shutdown_costs", False)))

    if not discovered:
        return None
    if len(set(discovered)) > 1:
        raise ValueError(
            f"Inconsistent surrogate startup/shutdown-cost settings found in {model_dir}: "
            f"{sorted(set(discovered))}"
        )
    return discovered[0]


def load_trained_models_for_test(ppc, all_samples: list, T_DELTA: float,
                                 model_dir: str, unit_ids,
                                 requested_strategy: str | None,
                                 requested_ignore_startup_shutdown_costs: bool | None = None):
    resolved_strategy = _resolve_requested_surrogate_strategy(requested_strategy)
    resolved_ignore_startup_shutdown_costs = (
        None
        if requested_ignore_startup_shutdown_costs is None
        else bool(requested_ignore_startup_shutdown_costs)
    )
    if resolved_strategy is None:
        saved_strategy = _load_saved_surrogate_strategy(model_dir, unit_ids)
        if saved_strategy is not None:
            log(f"约束生成策略: 使用模型保存值 {saved_strategy}")
        saved_ignore = _load_saved_subproblem_ignore_startup_shutdown_costs(model_dir, unit_ids)
        if saved_ignore is not None:
            log(f"subproblem 忽略启停成本: 使用模型保存值 {saved_ignore}")
        try:
            return load_trained_models(
                ppc, all_samples, T_DELTA,
                load_dir=model_dir,
                unit_ids=unit_ids,
                constraint_generation_strategy=None,
                ignore_startup_shutdown_costs=resolved_ignore_startup_shutdown_costs,
            )
        except ValueError as exc:
            if "Surrogate model startup/shutdown-cost setting mismatch" not in str(exc):
                raise
            log(
                "subproblem 启停成本配置与模型不一致，自动回退到保存值: "
                f"requested={resolved_ignore_startup_shutdown_costs}, saved={saved_ignore}"
            )
            return load_trained_models(
                ppc, all_samples, T_DELTA,
                load_dir=model_dir,
                unit_ids=unit_ids,
                constraint_generation_strategy=None,
                ignore_startup_shutdown_costs=None,
            )

    try:
        log(f"约束生成策略: {resolved_strategy}")
        if resolved_ignore_startup_shutdown_costs is not None:
            log(f"subproblem 忽略启停成本: {resolved_ignore_startup_shutdown_costs}")
        return load_trained_models(
            ppc, all_samples, T_DELTA,
            load_dir=model_dir,
            unit_ids=unit_ids,
            constraint_generation_strategy=resolved_strategy,
            ignore_startup_shutdown_costs=resolved_ignore_startup_shutdown_costs,
        )
    except ValueError as exc:
        if (
            "Surrogate model strategy mismatch" not in str(exc)
            and "Surrogate model startup/shutdown-cost setting mismatch" not in str(exc)
        ):
            raise
        saved_strategy = _load_saved_surrogate_strategy(model_dir, unit_ids)
        saved_ignore = _load_saved_subproblem_ignore_startup_shutdown_costs(model_dir, unit_ids)
        if saved_strategy is None and saved_ignore is None:
            raise
        log(
            "surrogate 配置与模型不一致，自动回退到保存值: "
            f"requested_strategy={resolved_strategy}, saved_strategy={saved_strategy}, "
            f"requested_ignore_startup_shutdown_costs={resolved_ignore_startup_shutdown_costs}, "
            f"saved_ignore_startup_shutdown_costs={saved_ignore}"
        )
        return load_trained_models(
            ppc, all_samples, T_DELTA,
            load_dir=model_dir,
            unit_ids=unit_ids,
            constraint_generation_strategy=None,
            ignore_startup_shutdown_costs=None,
        )


def load_json_data(data_file: Path) -> list:
    """加载 JSON 数据文件并规范化为 v3 所需格式。"""
    log(f"加载数据文件: {data_file.name}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_samples = data.get('all_samples', [])
    if not all_samples:
        raise ValueError("JSON 文件中没有样本数�?(all_samples 为空)")

    log(f"  原始样本数: {len(all_samples)}")

    for idx, sample in enumerate(all_samples):
        normalize_sample_arrays(sample)
        sample.setdefault('sample_id', idx)

    return all_samples


def parse_sample_range(sample_range: str | tuple[int, int] | None) -> tuple[int, int] | None:
    """Parse a half-open sample range like '210:220' or (210, 220)."""
    if sample_range is None:
        return None

    if isinstance(sample_range, tuple):
        if len(sample_range) != 2:
            raise ValueError("SAMPLE_RANGE 元组必须恰好包含两个整数")
        start, end = int(sample_range[0]), int(sample_range[1])
        if start < 0 or end <= start:
            raise ValueError("SAMPLE_RANGE 要求 0 <= start < end")
        return start, end

    parts = sample_range.split(':', maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("--sample-range 格式必须�?start:end，例�?210:220")

    start = int(parts[0])
    end = int(parts[1])
    if start < 0 or end <= start:
        raise ValueError("--sample-range 要求 0 <= start < end")
    return start, end


def apply_sample_range(all_samples: list, sample_range: tuple[int, int] | None) -> list:
    """Apply a half-open slice to the loaded samples."""
    if sample_range is None:
        return all_samples

    start, end = sample_range
    total = len(all_samples)
    if start >= total:
        raise ValueError(f"--sample-range 起点 {start} 超出样本总数 {total}")

    actual_end = min(end, total)
    selected = all_samples[start:actual_end]
    log(f"  使用样本范围 [{start}:{actual_end})，共 {len(selected)} 个样本")
    return selected
def pick_data_file(result_dir: Path, case_name: str) -> Path:
    """按优先级查找最合适的数据文件。"""
    specific = sorted(result_dir.glob(f'active_sets_{case_name}_*.json'))
    if specific:
        return specific[-1]
    any_files = sorted(result_dir.glob('active_sets_*.json'))
    if any_files:
        log(f"未找到 {case_name} 专属文件，使用 {any_files[-1].name}")
        return any_files[-1]
    return None


# ──────────────────────── 绘图工具 ────────────────────────

# IEEE/学术风格全局设置
_MPL_STYLE = {
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.85,
    'figure.dpi':         120,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'axes.prop_cycle':    matplotlib.cycler(
        color=['#2166AC', '#D6604D', '#4DAC26', '#8073AC',
               '#F4A582', '#92C5DE', '#A1D76A', '#E9A3C9']
    ) if MPL_AVAILABLE else None,
}


def _apply_style() -> None:
    """应用学术绘图风格（matplotlib 可用时）。"""
    if not MPL_AVAILABLE:
        return
    params = {k: v for k, v in _MPL_STYLE.items() if v is not None}
    plt.rcParams.update(params)


def _save_fig(fig: 'plt.Figure', path: Path, label: str) -> None:
    """保存图像（PNG + PDF），并在终端打印路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix('.png')))
    fig.savefig(str(path.with_suffix('.pdf')))
    plt.close(fig)
    log(f"  图像已保存: {path.with_suffix('.png')} / .pdf  [{label}]")


def plot_surrogate_analysis(trainers: dict, all_samples: list,
                            fig_dir: Path, case_name: str) -> None:
    """绘制 surrogate 模型三张分析图：系数分布、约束违反热图、整数性得分�?

    Args:
        trainers:    {unit_id: SubproblemSurrogateTrainer} 字典�?
        all_samples: v3 格式样本列表�?
        fig_dir:     图像输出目录�?
        case_name:   算例名，用于图标题和文件名�?
    """
    if RUN_TEST_DISABLE_PLOTS:
        log("RUN_TEST_DISABLE_PLOTS=1，跳过绘图")
        return
    if not MPL_AVAILABLE:
        log("matplotlib 不可用，跳过绘图")
        return

    _apply_style()
    unit_ids = sorted(trainers.keys())
    n_units = len(unit_ids)
    if n_units == 0:
        log("未加载任何 surrogate trainers，跳过绘图")
        return

    # ── �?1：代理约束系数分布（2×2 violin�?─────────────────
    log("绘制图1：代理约束系数分布...")
    fig1, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig1.suptitle(
        f'Surrogate Constraint Coefficient Distributions  [{case_name}]',
        fontsize=12, fontweight='bold', y=1.01
    )

    coef_info = [
        ('alpha_values', r'$\alpha$ (Coefficient of $x_t$)',     axes[0, 0]),
        ('beta_values',  r'$\beta$ (Coefficient of $x_{t+1}$)',  axes[0, 1]),
        ('gamma_values', r'$\gamma$ (Offset-2 Coefficient)', axes[1, 0]),
        ('delta_values', r'$\delta$ (RHS / Slack)',               axes[1, 1]),
    ]

    colors = ['#2166AC', '#D6604D', '#4DAC26', '#8073AC']

    for (attr, ylabel, ax), color in zip(coef_info, colors):
        data_per_unit = []
        labels = []
        for uid in unit_ids:
            arr = getattr(trainers[uid], attr)   # (n_samples, nc)
            flat = np.asarray(arr).ravel()
            if flat.size == 0:
                continue
            data_per_unit.append(flat)
            labels.append(f'G{uid}')

        if not data_per_unit:
            ax.set_title(f"{attr}: no data", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        parts = ax.violinplot(
            data_per_unit,
            positions=range(len(data_per_unit)),
            showmedians=True,
            showextrema=True,
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(1.2)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Generator Unit')

    fig1.tight_layout()
    _save_fig(fig1, fig_dir / f'{case_name}_surrogate_coefficients', '系数分布')

    # ── �?2：约束违反热图（units × time�?────────────────────
    log("绘制图2：约束违反热图...")
    # 取所有单元中最小的 nc 作为公共时间�?
    nc_list = [trainers[uid].num_coupling_constraints for uid in unit_ids]
    nc_common = min(nc_list)
    n_samples_plot = min(
        len(all_samples),
        *[
            min(trainers[uid].alpha_values.shape[0], len(trainers[uid].x))
            for uid in unit_ids
        ],
    )

    viol_matrix = np.zeros((n_units, nc_common))   # (ng, nc)
    for gi, uid in enumerate(unit_ids):
        tr = trainers[uid]
        ns = min(n_samples_plot, tr.alpha_values.shape[0], len(tr.x))
        if ns == 0:
            continue
        for s in range(ns):
            x_s = np.asarray(tr.x[s], dtype=float)   # (T,)
            sample_s = all_samples[s] if s < len(all_samples) else {'sample_id': s}
            timestep_map, offset_map = _resolve_surrogate_display_layout(tr, sample_s, nc_common)
            for t in range(nc_common):
                ts = timestep_map[t]
                a_t, b_t, g_t, d_t = tr._apply_surrogate_direction_to_params(
                    np.array([tr.alpha_values[s, t]], dtype=float),
                    np.array([tr.beta_values[s, t]], dtype=float),
                    np.array([tr.gamma_values[s, t]], dtype=float),
                    np.array([tr.delta_values[s, t]], dtype=float),
                )
                lhs = build_surrogate_constraint_expression(
                    x_s,
                    ts,
                    offset_map[t],
                    a_t[0],
                    b_t[0],
                    g_t[0],
                    tr.T,
                )
                viol_matrix[gi, t] += max(0.0, lhs - d_t[0])
        viol_matrix[gi] /= ns

    fig2, ax2 = plt.subplots(figsize=(min(14, nc_common * 0.55 + 2), max(3, n_units * 0.5 + 1.5)))
    im = ax2.imshow(
        viol_matrix, aspect='auto', cmap='RdYlGn_r',
        interpolation='nearest',
        norm=Normalize(vmin=0, vmax=max(viol_matrix.max(), 1e-6)),
    )
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('Mean Constraint Violation', fontsize=9)

    ax2.set_yticks(range(n_units))
    ax2.set_yticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax2.set_xlabel('Coupling Constraint Index (Time Period $t$)')
    ax2.set_ylabel('Generator Unit')
    ax2.set_title(
        f'Mean Surrogate Constraint Violation  [{case_name}]  '
        f'(avg over {n_samples_plot} samples)',
        fontweight='bold'
    )
    # 在格子上标注数值（仅当格子足够大时�?
    if nc_common <= 30 and n_units <= 15:
        for gi in range(n_units):
            for t in range(nc_common):
                val = viol_matrix[gi, t]
                color_txt = 'white' if val > viol_matrix.max() * 0.6 else '#333333'
                ax2.text(t, gi, f'{val:.2f}', ha='center', va='center',
                         fontsize=6, color=color_txt)

    fig2.tight_layout()
    _save_fig(fig2, fig_dir / f'{case_name}_surrogate_violation_heatmap', '违反热图')

    # ── �?3：整数性得分（每机组，跨样本均�?± 标准差） ─────
    log("绘制图3：整数性得分...")
    integ_means, integ_stds = [], []
    for uid in unit_ids:
        tr = trainers[uid]
        ns = min(tr.alpha_values.shape[0], len(tr.x))
        scores = []
        for s in range(ns):
            x_s = np.asarray(tr.x[s], dtype=float)
            scores.append(float(np.sum(x_s * (1.0 - x_s))))
        if scores:
            integ_means.append(np.mean(scores))
            integ_stds.append(np.std(scores))
        else:
            integ_means.append(0.0)
            integ_stds.append(0.0)

    fig3, ax3 = plt.subplots(figsize=(max(5, n_units * 0.65 + 1.5), 4))
    xpos = np.arange(n_units)
    bars = ax3.bar(
        xpos, integ_means, yerr=integ_stds,
        capsize=4, width=0.6,
        color='#2166AC', alpha=0.75, edgecolor='#144E7A', linewidth=0.8,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
    )
    ax3.axhline(0, color='#333333', linewidth=0.8, linestyle='-')
    ax3.set_xticks(xpos)
    ax3.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=9)
    ax3.set_xlabel('Generator Unit')
    ax3.set_ylabel(r'Integrality Score  $\sum x_i(1-x_i)$')
    ax3.set_title(
        f'Per-Unit Integrality Score  [{case_name}]  '
        f'(mean ± std over {n_samples_plot} samples)',
        fontweight='bold'
    )
    # 在每根柱子顶部标注均�?
    for bar, mean in zip(bars, integ_means):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(integ_stds) * 0.05,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=7.5, color='#222222'
        )

    fig3.tight_layout()
    _save_fig(fig3, fig_dir / f'{case_name}_surrogate_integrality', '整数性得分')


def plot_fp_results(fp_results: list, fig_dir: Path, case_name: str) -> None:
    """绘制可行性泵测试汇总图：成功率饼图 + 每样本结果条形图�?

    Args:
        fp_results: run_fp_test 返回�?[(idx, success, x_result), ...] 列表�?
        fig_dir:    图像输出目录�?
        case_name:  算例名�?
    """
    if not MPL_AVAILABLE or not fp_results:
        return

    _apply_style()
    n_total = len(fp_results)
    n_success = sum(1 for _, s, _ in fp_results if s)
    n_fail = n_total - n_success

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f'Feasibility Pump Test Results  [{case_name}]',
        fontsize=12, fontweight='bold'
    )

    # 饼图
    wedge_colors = ['#4DAC26', '#D6604D']
    wedges, texts, autotexts = ax_pie.pie(
        [n_success, n_fail],
        labels=['Success', 'Failure'],
        autopct='%1.1f%%',
        colors=wedge_colors,
        startangle=90,
        wedgeprops=dict(linewidth=1.2, edgecolor='white'),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax_pie.set_title(f'Overall ({n_success}/{n_total} succeeded)', fontsize=10)
    ax_pie.grid(False)

    # 每样本条形图
    sample_ids = [r[0] for r in fp_results]
    colors_bar = ['#4DAC26' if r[1] else '#D6604D' for r in fp_results]
    # �?1/0 高度表示成功/失败；高�?= 成功1, 失败0.3（视觉区分）
    heights = [1.0 if r[1] else 0.3 for r in fp_results]
    ax_bar.bar(
        range(n_total), heights,
        color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.8,
    )
    ax_bar.set_xticks(range(n_total))
    ax_bar.set_xticklabels([f'#{i}' for i in sample_ids], fontsize=8)
    ax_bar.set_yticks([0.3, 1.0])
    ax_bar.set_yticklabels(['Failure', 'Success'], fontsize=9)
    ax_bar.set_xlabel('Sample Index')
    ax_bar.set_title('Per-Sample Outcome', fontsize=10)
    ax_bar.set_ylim(0, 1.35)
    ax_bar.grid(axis='x', alpha=0)

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[Patch(facecolor='#4DAC26', label='Success'),
                 Patch(facecolor='#D6604D', label='Failure')],
        loc='upper right', framealpha=0.85,
    )

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_fp_results', 'FP 测试结果')


def plot_fp_screening_comparison(screening_records: list, fig_dir: Path, case_name: str) -> None:
    """Compare baseline FP candidate counts with robust-surrogate-screen counts."""
    if not MPL_AVAILABLE or not screening_records:
        return

    _apply_style()
    sample_ids = [record['sample_index'] for record in screening_records]
    hot_before = [record['hot_starts_before'] for record in screening_records]
    hot_after = [record['hot_starts_after'] for record in screening_records]
    pool_before = [record['x_pool_before'] for record in screening_records]
    pool_after = [record['x_pool_after'] for record in screening_records]
    n_constraints = [record['n_constraints'] for record in screening_records]

    x = np.arange(len(screening_records))
    width = 0.34

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f'FP Commitment Screening Comparison  [{case_name}]',
        fontsize=12, fontweight='bold'
    )

    axes[0].bar(x - width / 2, hot_before, width=width, color='#9ECAE1', label='Baseline')
    axes[0].bar(x + width / 2, hot_after, width=width, color='#3182BD', label='Robust surrogate screen')
    axes[0].set_ylabel('Hot starts')
    axes[0].legend(loc='upper right', framealpha=0.85)

    axes[1].bar(x - width / 2, pool_before, width=width, color='#FDD0A2', label='Baseline')
    axes[1].bar(x + width / 2, pool_after, width=width, color='#E6550D', label='Robust surrogate screen')
    axes[1].set_ylabel('Pool size')
    axes[1].legend(loc='upper right', framealpha=0.85)

    axes[2].plot(x, n_constraints, marker='o', color='#31A354', linewidth=1.8)
    axes[2].set_ylabel('Selected rows')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'#{idx}' for idx in sample_ids], fontsize=8)

    for ax in axes:
        ax.grid(axis='x', alpha=0.0)

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_fp_screening_comparison', 'FP 筛选对比')


def plot_bcd_analysis(agent, fig_dir: Path, case_name: str) -> None:
    """绘制 BCD 模型两张分析图：θ/ζ 参数直方�?+ 网络权重分布�?

    Args:
        agent:     Agent_NN_BCD 实例（已加载模型参数）�?
        fig_dir:   图像输出目录�?
        case_name: 算例名�?
    """
    if not MPL_AVAILABLE:
        log("matplotlib 不可用，跳过绘图")
        return

    _apply_style()

    # ── �?1：�?/ ζ 参数直方�?────────────────────────────
    theta_vals = _collect_bcd_param_values(agent, "theta")
    zeta_vals = _collect_bcd_param_values(agent, "zeta")
    has_theta = theta_vals.size > 0
    has_zeta  = zeta_vals.size > 0

    if has_theta or has_zeta:
        log("绘制图1：theta / zeta 参数直方图...")
        ncols = int(has_theta) + int(has_zeta)
        fig1, axes1 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        if ncols == 1:
            axes1 = [axes1]
        fig1.suptitle(
            f'BCD Surrogate Parameter Distributions  [{case_name}]',
            fontsize=12, fontweight='bold'
        )

        ax_idx = 0
        for flag, attr, label, color in [
            (has_theta, theta_vals, r'$\theta$ Values', '#2166AC'),
            (has_zeta,  zeta_vals,  r'$\zeta$ Values',  '#D6604D'),
        ]:
            if not flag:
                continue
            vals = np.asarray(attr, dtype=float)
            ax = axes1[ax_idx]
            n_bins = min(50, max(10, len(vals) // 5))
            ax.hist(vals, bins=n_bins, color=color, alpha=0.72,
                    edgecolor='white', linewidth=0.6, density=True)
            # KDE 曲线
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method='scott')
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, kde(xs), color='#333333', linewidth=1.5,
                        linestyle='--', label='KDE')
                ax.legend()
            except Exception:
                pass

            ax.axvline(vals.mean(), color='#333333', linewidth=1.2,
                       linestyle=':', label=f'Mean={vals.mean():.3f}')
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(
                f'{label}  (n={len(vals)}, '
                f'mean={vals.mean():.3f}, std={vals.std():.3f})'
            )
            ax_idx += 1

        fig1.tight_layout()
        _save_fig(fig1, fig_dir / f'{case_name}_bcd_param_hist', 'θ/ζ 直方图')

    # ── �?2：神经网络权重层分布（violin�?────────────────────
    import torch

    net_specs = []
    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        net_specs.append((agent.theta_net, r'$\theta$-Net', '#2166AC'))
    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        net_specs.append((agent.zeta_net, r'$\zeta$-Net', '#D6604D'))

    if not net_specs:
        return

    log("绘制图2：神经网络权重层分布...")
    n_nets = len(net_specs)
    fig2, axes2 = plt.subplots(1, n_nets, figsize=(6 * n_nets, 5))
    if n_nets == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f'Neural Network Weight Distributions  [{case_name}]',
        fontsize=12, fontweight='bold'
    )

    for ax, (net, net_label, color) in zip(axes2, net_specs):
        layer_data, layer_labels = [], []
        for name, param in net.named_parameters():
            if param.requires_grad:
                w = param.detach().cpu().numpy().ravel()
                if len(w) > 0:
                    layer_data.append(w)
                    # 缩短名字：weight �?W, bias �?b
                    short = name.replace('weight', 'W').replace('bias', 'b')
                    layer_labels.append(short)

        if not layer_data:
            ax.text(0.5, 0.5, 'No parameters', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            continue

        parts = ax.violinplot(
            layer_data,
            positions=range(len(layer_data)),
            showmedians=True, showextrema=True,
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(1.1)

        ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'{net_label} Layer Weights')

    fig2.tight_layout()
    _save_fig(fig2, fig_dir / f'{case_name}_bcd_weight_dist', '权重分布')


def plot_both_analysis(agent, trainers: dict, fig_dir: Path, case_name: str) -> None:
    """绘制 BCD-Surrogate 联合特征图：4 面板综合约束刻画�?

    面板布局 (2×2)�?
      (a) 各机组代理约�?RHS (δ) 均�?�?约束紧度
      (b) 各机组代理系数耦合强度 �?ᾱ�?β̄²+γ̄²) �?时序耦合程度
      (c) BCD θ 参数分布（直方图+KDE�?
      (d) BCD ζ 参数分布（直方图+KDE�?

    Args:
        agent:     Agent_NN_BCD 实例（已加载模型）�?
        trainers:  {unit_id: SubproblemSurrogateTrainer} 字典�?
        fig_dir:   图像输出目录�?
        case_name: 算例名�?
    """
    if not MPL_AVAILABLE:
        return

    _apply_style()
    unit_ids = sorted(trainers.keys())
    n_units  = len(unit_ids)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f'BCD–Surrogate Joint Constraint Characterization  [{case_name}]',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── (a) 各机组代理约�?RHS δ 均值（violin�?──────────────
    ax_a = axes[0, 0]
    delta_per_unit = [trainers[uid].delta_values.ravel() for uid in unit_ids]
    parts_a = ax_a.violinplot(
        delta_per_unit, positions=range(n_units),
        showmedians=True, showextrema=True,
    )
    for pc in parts_a['bodies']:
        pc.set_facecolor('#8073AC')
        pc.set_alpha(0.6)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        if key in parts_a:
            parts_a[key].set_color('#333333')
            parts_a[key].set_linewidth(1.1)
    ax_a.set_xticks(range(n_units))
    ax_a.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax_a.set_xlabel('Generator Unit')
    ax_a.set_ylabel(r'$\delta$ (RHS Bound)')
    ax_a.set_title(r'(a) Surrogate Constraint Tightness $\delta$', loc='left', fontsize=10)

    # ── (b) 各机组时序耦合强度 ─────────────────────────────
    ax_b = axes[0, 1]
    coupling_strength = []
    for uid in unit_ids:
        tr = trainers[uid]
        a_mean = np.abs(tr.alpha_values).mean(axis=1)   # (n_samples,)
        b_mean = np.abs(tr.beta_values).mean(axis=1)
        g_mean = np.abs(tr.gamma_values).mean(axis=1)
        strength = np.sqrt(a_mean**2 + b_mean**2 + g_mean**2)  # (n_samples,)
        coupling_strength.append(strength)

    means = [s.mean() for s in coupling_strength]
    stds  = [s.std()  for s in coupling_strength]
    xpos  = np.arange(n_units)
    bars  = ax_b.bar(
        xpos, means, yerr=stds, capsize=4, width=0.6,
        color='#4DAC26', alpha=0.72, edgecolor='#2A7A15', linewidth=0.8,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
    )
    for bar, m in zip(bars, means):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.08,
            f'{m:.3f}', ha='center', va='bottom', fontsize=7.5
        )
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([f'G{uid}' for uid in unit_ids], fontsize=8)
    ax_b.set_xlabel('Generator Unit')
    ax_b.set_ylabel(r'$\sqrt{\bar\alpha^2+\bar\beta^2+\bar\gamma^2}$')
    ax_b.set_title(r'(b) Temporal Coupling Strength', loc='left', fontsize=10)

    # ── (c) BCD θ 分布 ─────────────────────────────────────
    ax_c = axes[1, 0]
    theta_vals = _collect_bcd_param_values(agent, "theta")
    has_theta = theta_vals.size > 0
    if has_theta:
        n_bins = min(50, max(10, len(theta_vals) // 5))
        ax_c.hist(theta_vals, bins=n_bins, color='#2166AC', alpha=0.65,
                  edgecolor='white', linewidth=0.5, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(theta_vals, bw_method='scott')
            xs = np.linspace(theta_vals.min(), theta_vals.max(), 300)
            ax_c.plot(xs, kde(xs), color='#0A3F70', linewidth=1.8,
                      linestyle='--', label='KDE')
        except Exception:
            pass
        ax_c.axvline(theta_vals.mean(), color='#D6604D', linewidth=1.4,
                     linestyle=':', label=f'mean={theta_vals.mean():.3f}')
        ax_c.legend(fontsize=8)
        ax_c.set_xlabel(r'$\theta$ value')
    else:
        ax_c.text(0.5, 0.5, 'theta_values not available',
                  transform=ax_c.transAxes, ha='center', va='center')
    ax_c.set_ylabel('Density')
    ax_c.set_title(r'(c) BCD $\theta$ Parameter Distribution', loc='left', fontsize=10)

    # ── (d) BCD ζ 分布 ─────────────────────────────────────
    ax_d = axes[1, 1]
    zeta_vals = _collect_bcd_param_values(agent, "zeta")
    has_zeta = zeta_vals.size > 0
    if has_zeta:
        n_bins = min(50, max(10, len(zeta_vals) // 5))
        ax_d.hist(zeta_vals, bins=n_bins, color='#D6604D', alpha=0.65,
                  edgecolor='white', linewidth=0.5, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(zeta_vals, bw_method='scott')
            xs = np.linspace(zeta_vals.min(), zeta_vals.max(), 300)
            ax_d.plot(xs, kde(xs), color='#7B1A0A', linewidth=1.8,
                      linestyle='--', label='KDE')
        except Exception:
            pass
        ax_d.axvline(zeta_vals.mean(), color='#2166AC', linewidth=1.4,
                     linestyle=':', label=f'mean={zeta_vals.mean():.3f}')
        ax_d.legend(fontsize=8)
        ax_d.set_xlabel(r'$\zeta$ value')
    else:
        ax_d.text(0.5, 0.5, 'zeta_values not available',
                  transform=ax_d.transAxes, ha='center', va='center')
    ax_d.set_ylabel('Density')
    ax_d.set_title(r'(d) BCD $\zeta$ Parameter Distribution', loc='left', fontsize=10)

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_both_joint_characterization', 'BCD-Surrogate 联合表征')


# ──────────────────────── 模式实现 ────────────────────────


def print_surrogate_results(trainers: dict, all_samples: list) -> None:
    """打印代理训练结果摘要。"""
    n_samples = len(all_samples)
    print("=" * 70)

    for unit_id, trainer in trainers.items():
        T = trainer.T
        nc = trainer.num_coupling_constraints
        print(f"\n机组 {unit_id}:")
        print(f"  alpha_values shape: {trainer.alpha_values.shape}  "
              f"(期望: ({n_samples}, {nc}))")
        print(f"  beta_values  shape: {trainer.beta_values.shape}")
        print(f"  gamma_values shape: {trainer.gamma_values.shape}")
        print(f"  delta_values shape: {trainer.delta_values.shape}  (RHS，非负)")
        print(f"  c_x values    shape: {trainer.cost_values.shape}")
        print(f"  c_pg values   shape: {trainer.pg_cost_values.shape}")
        print(
            f"  c_x stats: mean={float(np.mean(trainer.cost_values)):.4f}, "
            f"std={float(np.std(trainer.cost_values)):.4f}, "
            f"maxabs={float(np.max(np.abs(trainer.cost_values))):.4f}"
        )
        print(
            f"  c_pg stats: mean={float(np.mean(trainer.pg_cost_values)):.4f}, "
            f"std={float(np.std(trainer.pg_cost_values)):.4f}, "
            f"maxabs={float(np.max(np.abs(trainer.pg_cost_values))):.4f}"
        )

        x0 = trainer.x[0]
        sample0 = all_samples[0] if all_samples else {'sample_id': 0}
        timestep_map, offset_map = _resolve_surrogate_display_layout(trainer, sample0, nc)
        print(f"  样本0 代理约束示例（最多10条）:")
        for k in range(min(10, nc)):
            ts = timestep_map[k]
            a_arr, b_arr, g_arr, d_arr = trainer._apply_surrogate_direction_to_params(
                np.array([trainer.alpha_values[0, k]], dtype=float),
                np.array([trainer.beta_values[0, k]], dtype=float),
                np.array([trainer.gamma_values[0, k]], dtype=float),
                np.array([trainer.delta_values[0, k]], dtype=float),
            )
            a = a_arr[0]
            b = b_arr[0]
            g = g_arr[0]
            d = d_arr[0]
            lhs = build_surrogate_constraint_expression(
                x0,
                ts,
                offset_map[k],
                a,
                b,
                g,
                T,
            )
            viol = max(0.0, lhs - d)
            expr_text = _format_surrogate_constraint(a, b, g, ts, offset_map[k], d)
            print(f"    k={k}, t={ts}, offsets={tuple(offset_map[k])}: {expr_text}  "
                  f"(lhs={lhs:.3f}, viol={viol:.4f})")
        if T > 0:
            cx0 = np.asarray(trainer.cost_values[0], dtype=float)
            cpg0 = np.asarray(trainer.pg_cost_values[0], dtype=float)
            preview_len = min(6, T)
            print(f"  样本0 c_x 前{preview_len}项: {np.array2string(cx0[:preview_len], precision=3)}")
            print(f"  样本0 c_pg前{preview_len}项: {np.array2string(cpg0[:preview_len], precision=3)}")
            if all_samples:
                renewable0 = sample0.get('renewable_data') if isinstance(sample0, dict) else None
                try:
                    inf_alpha, inf_beta, inf_gamma, inf_delta, inf_cx, inf_cpg = trainer.get_surrogate_params(
                        sample0,
                        np.asarray(trainer.lambda_vals[0], dtype=float),
                        renewable_data=renewable0,
                    )
                    print(
                        f"  推理c_x 前{preview_len}项: "
                        f"{np.array2string(np.asarray(inf_cx[:preview_len], dtype=float), precision=3)}"
                    )
                    print(
                        f"  推理c_pg前{preview_len}项: "
                        f"{np.array2string(np.asarray(inf_cpg[:preview_len], dtype=float), precision=3)}"
                    )
                    diff_cx = float(np.max(np.abs(np.asarray(inf_cx, dtype=float) - cx0[:len(inf_cx)])))
                    diff_cpg = float(np.max(np.abs(np.asarray(inf_cpg, dtype=float) - cpg0[:len(inf_cpg)])))
                    print(f"  缓存/推理 c_x 最大差值: {diff_cx:.4f}")
                    print(f"  缓存/推理 c_pg最大差值: {diff_cpg:.4f}")
                except Exception as exc:
                    print(f"  推理 c_x/c_pg 预览失败: {exc}")

        integrality = float(np.sum(x0 * (1 - x0)))
        print(f"  整数性指标(样本0): {integrality:.6f}  (0=完全整数)")


def _extract_true_solution(sample: dict, shape: tuple) -> np.ndarray:
    """从样本字典中还原真实 UC 解矩阵 `(ng, T)`。"""
    ng, T = shape
    x_true = np.zeros((ng, T), dtype=float)
    if 'unit_commitment_matrix' in sample:
        uc = np.array(sample['unit_commitment_matrix'], dtype=float)
        r = min(uc.shape[0], ng)
        c = min(uc.shape[1] if uc.ndim > 1 else T, T)
        x_true[:r, :c] = uc[:r, :c]
    elif 'active_set' in sample:
        for item in sample['active_set']:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g, t = item[0]
                if g < ng and t < T:
                    x_true[g, t] = float(item[1])
    return x_true


def _compute_commitment_distance_metrics(
    x_candidate: np.ndarray,
    x_true: np.ndarray,
) -> tuple[float, int]:
    """Return L1 distance and rounded Hamming distance to the reference commitment."""
    x_arr = np.asarray(x_candidate, dtype=float)
    x_ref = np.asarray(x_true, dtype=float)
    if x_arr.shape != x_ref.shape:
        raise ValueError(
            f"Commitment shape mismatch: candidate={x_arr.shape}, reference={x_ref.shape}"
        )
    l1_distance = float(np.sum(np.abs(x_arr - x_ref)))
    hamming_distance = int(np.sum(np.round(x_arr).astype(int) != x_ref.astype(int)))
    return l1_distance, hamming_distance


def _can_run_global_fp(ppc, trainers: dict) -> tuple[bool, str]:
    """Return whether the loaded trainers cover all generators for global FP."""
    expected_ng = int(ppc['gen'].shape[0])
    loaded_unit_ids = sorted(
        int(unit_id)
        for unit_id in trainers.keys()
        if 0 <= int(unit_id) < expected_ng
    )
    expected_unit_ids = list(range(expected_ng))
    if loaded_unit_ids != expected_unit_ids:
        missing_unit_ids = [unit_id for unit_id in expected_unit_ids if unit_id not in loaded_unit_ids]
        extra_unit_ids = [unit_id for unit_id in loaded_unit_ids if unit_id not in expected_unit_ids]
        return (
            False,
            "skip global FP because surrogate trainers do not cover all generators: "
            f"loaded_unit_ids={loaded_unit_ids}, expected_unit_ids={expected_unit_ids}, "
            f"missing_unit_ids={missing_unit_ids}, extra_unit_ids={extra_unit_ids}",
        )
    return True, "global FP enabled"


def _describe_loaded_surrogate_units(ppc, trainers: dict) -> str:
    """Return a compact summary of loaded surrogate trainer unit ids."""
    expected_ng = int(ppc['gen'].shape[0])
    loaded_unit_ids = sorted(
        int(unit_id)
        for unit_id in trainers.keys()
        if 0 <= int(unit_id) < expected_ng
    )
    return f"loaded_unit_ids={loaded_unit_ids}, expected_ng={expected_ng}"


def _compute_commitment_distance_metrics_with_mask(
    x_candidate: np.ndarray,
    x_true: np.ndarray,
) -> dict:
    """Return masked commitment distance metrics and coverage."""
    x_arr = np.asarray(x_candidate, dtype=float)
    x_ref = np.asarray(x_true, dtype=float)
    if x_arr.shape != x_ref.shape:
        raise ValueError(
            f"Commitment shape mismatch: candidate={x_arr.shape}, reference={x_ref.shape}"
        )

    valid_mask = np.isfinite(x_arr) & np.isfinite(x_ref)
    covered = int(np.sum(valid_mask))
    total = int(valid_mask.size)
    if covered == 0:
        return {
            'covered': 0,
            'total': total,
            'coverage_ratio': 0.0,
            'l1': float('nan'),
            'hamming': None,
            'integrality_gap': float('nan'),
        }

    x_valid = x_arr[valid_mask]
    x_ref_valid = x_ref[valid_mask]
    return {
        'covered': covered,
        'total': total,
        'coverage_ratio': float(covered / max(total, 1)),
        'l1': float(np.sum(np.abs(x_valid - x_ref_valid))),
        'hamming': int(np.sum(np.round(x_valid).astype(int) != x_ref_valid.astype(int))),
        'integrality_gap': float(np.mean(np.minimum(x_valid, 1.0 - x_valid))),
    }


def _evaluate_commitment_economic_cost(
    ppc,
    sample: dict,
    x_commitment: np.ndarray,
    T_delta: float,
) -> dict:
    """Evaluate a fixed commitment with ED and startup/shutdown costs."""
    pd_data = np.asarray(sample['pd_data'], dtype=float)
    renewable_data = sample.get('renewable_data') if isinstance(sample, dict) else None
    x_arr = np.asarray(np.round(x_commitment), dtype=float)

    try:
        ed = EconomicDispatchGurobi(
            ppc,
            pd_data,
            T_delta,
            x_arr,
            renewable_data=renewable_data,
            verbose=False,
        )
        ed.model.optimize()
    except Exception as exc:
        return {
            'success': False,
            'reason': f'ED exception: {exc}',
        }

    if ed.model.status != GRB.OPTIMAL:
        return {
            'success': False,
            'reason': f'ED status={ed.model.status}',
        }

    dispatch_cost = float(ed.model.objVal)
    startup_cost = 0.0
    shutdown_cost = 0.0
    if x_arr.shape[1] >= 2:
        x_prev = np.clip(x_arr[:, :-1], 0.0, 1.0)
        x_next = np.clip(x_arr[:, 1:], 0.0, 1.0)
        startup_cost = float(
            np.sum(ed.gencost[:, 1][:, None] * np.maximum(x_next - x_prev, 0.0))
        )
        shutdown_cost = float(
            np.sum(ed.gencost[:, 2][:, None] * np.maximum(x_prev - x_next, 0.0))
        )

    total_cost = dispatch_cost + startup_cost + shutdown_cost
    return {
        'success': True,
        'dispatch_cost': dispatch_cost,
        'startup_cost': startup_cost,
        'shutdown_cost': shutdown_cost,
        'total_cost': float(total_cost),
    }


def _build_subproblem_commitment_matrix(
    sample: dict,
    lambda_val: np.ndarray,
    trainers: dict,
    shape: tuple[int, int],
    solve_milp: bool = False,
) -> np.ndarray:
    """逐机组求解 surrogate 子问题，并拼成 `(ng, T)` 启停矩阵。"""
    ng, T = shape
    x_sub = np.full((ng, T), np.nan, dtype=float)
    renewable_data = sample.get('renewable_data') if isinstance(sample, dict) else None
    solve_unit = _solve_unit_MILP_with_surrogate if solve_milp else _solve_unit_LP_with_surrogate

    for unit_id, trainer in sorted(trainers.items()):
        unit_idx = int(unit_id)
        if not (0 <= unit_idx < ng):
            continue
        try:
            lambda_unit = _extract_unit_lambda(
                lambda_val,
                trainer.T,
                unit_id=unit_idx,
                trainer=trainer,
            )
            alphas, betas, gammas, deltas, costs, pg_costs = trainer.get_surrogate_params(
                sample, lambda_unit, renewable_data=renewable_data,
            )
            x_unit, status_unit, details_unit = solve_unit(
                trainer,
                lambda_val,
                alphas,
                betas,
                gammas,
                deltas,
                costs=costs,
                pg_costs=pg_costs,
                scenario_sample=sample,
            )
            if status_unit == GRB.OPTIMAL:
                x_sub[unit_idx, :min(T, x_unit.shape[0])] = x_unit[:T]
            else:
                log(
                    f"  机组 {unit_idx}: surrogate 子问题"
                    f"{' MILP' if solve_milp else ' LP'} 状态="
                    f"{details_unit.get('status_name', status_unit)}"
                )
        except Exception as e:
            log(
                f"  机组 {unit_idx}: surrogate 子问题"
                f"{' MILP' if solve_milp else ' LP'} 求解失败，已跳过 ({e})"
            )

    return x_sub


def plot_lp_vs_true(x_LP_list: list, x_true_list: list,
                    fig_dir: Path, case_name: str) -> None:
    """绘制全局 LP 松弛解与真实解的对比图（5 面板）�?

    面板布局 (2 �?�?
      上行 (3 �?: (a) 平均 LP 松弛热图  (b) 平均真实解热�? (c) 均值绝对差热图
      下行 (2 �?: (d) 逐样�?Hamming 距离柱状�? (e) 逐样本整数性间隙柱状图

    Args:
        x_LP_list:   [n_test] 每个样本�?LP 松弛�?(ng, T)�?
        x_true_list: [n_test] 每个样本的真实二值解 (ng, T)�?
        fig_dir:     图像输出目录�?
        case_name:   算例名�?
    """
    if not MPL_AVAILABLE or not x_LP_list:
        return

    _apply_style()
    n = len(x_LP_list)

    x_LP_arr   = np.stack(x_LP_list,   axis=0)   # (n, ng, T)
    x_true_arr = np.stack(x_true_list, axis=0)   # (n, ng, T)

    x_LP_mean   = x_LP_arr.mean(axis=0)                     # (ng, T)
    x_true_mean = x_true_arr.mean(axis=0)                   # (ng, T)
    x_diff_mean = np.abs(x_LP_mean - x_true_mean)           # (ng, T)

    x_LP_rounded = (x_LP_arr >= 0.5).astype(int)
    hamming_dists = [int(np.sum(x_LP_rounded[i] != x_true_arr[i].astype(int)))
                     for i in range(n)]
    integ_gaps = [float(np.mean(np.minimum(x_LP_arr[i], 1.0 - x_LP_arr[i])))
                  for i in range(n)]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f'Global LP Relaxation vs. True UC Solution  [{case_name}]',
        fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_lp   = fig.add_subplot(gs[0, 0])
    ax_true = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 2])
    ax_ham  = fig.add_subplot(gs[1, 0:2])
    ax_gap  = fig.add_subplot(gs[1, 2])

    _cbar_kw = dict(fraction=0.045, pad=0.03)
    ng, T_ = x_LP_mean.shape
    yticks = range(ng)
    ylabels = [f'G{g}' for g in yticks]

    # (a) 平均 LP 松弛热图
    im_lp = ax_lp.imshow(x_LP_mean, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                          interpolation='nearest')
    fig.colorbar(im_lp, ax=ax_lp, **_cbar_kw)
    ax_lp.set_yticks(yticks); ax_lp.set_yticklabels(ylabels, fontsize=7)
    ax_lp.set_xlabel('Time Period $t$'); ax_lp.set_ylabel('Generator')
    ax_lp.set_title(r'(a) Mean LP Relaxation $\bar{x}^{LP}$', loc='left', fontsize=10)

    # (b) 平均真实解热�?
    im_true = ax_true.imshow(x_true_mean, aspect='auto', cmap='Oranges', vmin=0, vmax=1,
                              interpolation='nearest')
    fig.colorbar(im_true, ax=ax_true, **_cbar_kw)
    ax_true.set_yticks(yticks); ax_true.set_yticklabels(ylabels, fontsize=7)
    ax_true.set_xlabel('Time Period $t$'); ax_true.set_ylabel('Generator')
    ax_true.set_title(r'(b) Mean True Solution $\bar{x}^*$', loc='left', fontsize=10)

    # (c) 均值绝对差热图
    im_diff = ax_diff.imshow(x_diff_mean, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1,
                              interpolation='nearest')
    fig.colorbar(im_diff, ax=ax_diff, **_cbar_kw)
    ax_diff.set_yticks(yticks); ax_diff.set_yticklabels(ylabels, fontsize=7)
    ax_diff.set_xlabel('Time Period $t$'); ax_diff.set_ylabel('Generator')
    ax_diff.set_title(r'(c) Mean $|\bar{x}^{LP} - \bar{x}^*|$', loc='left', fontsize=10)

    # (d) 逐样�?Hamming 距离
    xpos = np.arange(n)
    bars_h = ax_ham.bar(xpos, hamming_dists, color='#2166AC', alpha=0.75,
                        edgecolor='#144E7A', linewidth=0.8, width=0.6)
    for bar, h in zip(bars_h, hamming_dists):
        ax_ham.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(hamming_dists) * 0.02,
                    str(h), ha='center', va='bottom', fontsize=8)
    ax_ham.set_xticks(xpos)
    ax_ham.set_xticklabels([f'Sample #{i}' for i in range(n)], fontsize=9)
    ax_ham.set_ylabel('Hamming Distance (bits)')
    ax_ham.set_title(
        r'(d) Rounded LP vs. True: Hamming Distance  '
        fr'(mean={np.mean(hamming_dists):.1f})',
        loc='left', fontsize=10
    )

    # (e) 逐样本整数性间�?
    bars_g = ax_gap.bar(xpos, integ_gaps, color='#D6604D', alpha=0.75,
                        edgecolor='#7B1A0A', linewidth=0.8, width=0.6)
    for bar, g_val in zip(bars_g, integ_gaps):
        ax_gap.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(integ_gaps) * 0.02,
                    f'{g_val:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax_gap.set_xticks(xpos)
    ax_gap.set_xticklabels([f'#{i}' for i in range(n)], fontsize=9)
    ax_gap.set_ylabel(r'Mean $\min(x^{LP},\,1-x^{LP})$')
    ax_gap.set_title('(e) LP Integrality Gap', loc='left', fontsize=10)

    fig.tight_layout()
    _save_fig(fig, fig_dir / f'{case_name}_lp_vs_true', 'LP 与真实解对比')


def run_lp_compare_test(ppc, all_samples: list, dual_predictor, trainers: dict,
                        T_DELTA: float, n_test: int,
                        fig_dir: Path, agent=None) -> list:
    """求解全局 LP 松弛，与真实解对比并绘图，返�?(x_LP, x_true) 列表�?

    此函数在可行性泵之前运行，评�?LP 松弛解的质量�?

    Args:
        ppc:            PyPower 案例字典�?
        all_samples:    v3 格式样本列表�?
        dual_predictor: 对偶变量预测器（.predict(pd_data) -> lambda）�?
        trainers:       {unit_id: SubproblemSurrogateTrainer}�?
        T_DELTA:        时间间隔�?
        n_test:         测试样本数�?
        fig_dir:        图像输出目录�?
        agent:          （可选）已训练的 Agent_NN_BCD 实例，用于加�?theta/zeta 约束�?

    Returns:
        [(x_LP, x_true), ...] 列表，每项对应一个样本�?
    """
    test_n = min(n_test, len(all_samples))
    log_section(f"LP 松弛解质量评估 | samples={test_n}")

    x_LP_plain_list, x_LP_surr_list, x_true_list = [], [], []
    first_plot_payload = None

    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        log(_fmt_sample_tag(i, test_n, pd_data.shape))
        if i == 0:
            _log_lambda_prediction_summary(dual_predictor, sample, ppc['gen'].shape[0], pd_data.shape[1], "LP compare")

        try:
            lambda_val = dual_predictor.predict(sample)
            x_lp_plain = solve_global_LP_relaxation_without_surrogate(ppc, pd_data, T_DELTA)
            x_lp_surr = solve_global_LP_relaxation(ppc, sample, T_DELTA, trainers, lambda_val,
                                                   agent=agent)
        except Exception as e:
            log(f"  LP 求解失败: {e}")
            continue

        x_true = _extract_true_solution(sample, x_lp_surr.shape)
        x_sub = _build_subproblem_commitment_matrix(sample, lambda_val, trainers, x_true.shape)

        x_lp_plain_rounded = (x_lp_plain >= 0.5).astype(int)
        x_lp_surr_rounded = (x_lp_surr >= 0.5).astype(int)
        hamming_plain = int(np.sum(x_lp_plain_rounded != x_true.astype(int)))
        hamming_surr = int(np.sum(x_lp_surr_rounded != x_true.astype(int)))
        integ_plain = float(np.mean(np.minimum(x_lp_plain, 1.0 - x_lp_plain)))
        integ_surr = float(np.mean(np.minimum(x_lp_surr, 1.0 - x_lp_surr)))
        log(
            f"  LP: integ={integ_plain:.4f} | Hamming={hamming_plain} | "
            f"Surrogate LP: integ={integ_surr:.4f} | Hamming={hamming_surr}"
        )

        x_LP_plain_list.append(x_lp_plain)
        x_LP_surr_list.append(x_lp_surr)
        x_true_list.append(x_true)
        if first_plot_payload is None:
            first_plot_payload = (i, x_true, x_lp_plain, x_lp_surr, x_sub)

    log_section("LP 松弛解质量评估汇总")
    if x_LP_plain_list:
        mean_hamming = np.mean([int(np.sum((x_LP_plain_list[i] >= 0.5).astype(int)
                                           != x_true_list[i].astype(int)))
                                for i in range(len(x_LP_plain_list))])
        log(f"LP 汇总: mean_hamming={mean_hamming:.1f} bits")
        _log_solution_similarity_summary(
            "Standard LP",
            x_LP_plain_list,
            "Surrogate LP",
            x_LP_surr_list,
        )
        plot_lp_vs_true(x_LP_plain_list, x_true_list, fig_dir, CASE_NAME)
        if first_plot_payload is not None:
            sample_id, x_true, x_lp_plain, x_lp_surr, x_sub = first_plot_payload
            plot_lp_surrogate_comparison(
                x_true, x_lp_plain, x_lp_surr, sample_id, fig_dir, CASE_NAME,
            )
            plot_surrogate_commitment_comparison(
                x_true, x_lp_plain, x_lp_surr, x_sub, sample_id, fig_dir, CASE_NAME,
            )

    return list(zip(x_LP_plain_list, x_LP_surr_list, x_true_list))


def run_subproblem_milp_test(
    ppc,
    all_samples: list,
    dual_predictor,
    trainers: dict,
    T_DELTA: float,
    n_test: int,
) -> list[dict]:
    """Compare surrogate unit subproblems solved as LP vs MILP under the same model."""
    test_n = min(n_test, len(all_samples))
    active_units = sorted(
        int(unit_id)
        for unit_id in trainers
        if 0 <= int(unit_id) < int(ppc['gen'].shape[0])
    )
    log_section(
        f"Subproblem surrogate LP/MILP 对比 | samples={test_n} | active_units={active_units}"
    )

    if not active_units:
        log("未加载任何机组 surrogate 模型，跳过 subproblem MILP 测试")
        return []

    results = []
    lp_solutions = []
    milp_solutions = []

    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        log(_fmt_sample_tag(i, test_n, pd_data.shape))
        try:
            lambda_val = dual_predictor.predict(sample)
        except Exception as exc:
            log(f"  dual predictor 失败，跳过: {exc}")
            continue

        x_true_full = _extract_true_solution(sample, (ppc['gen'].shape[0], pd_data.shape[1]))
        x_true = np.asarray(x_true_full[active_units, :], dtype=float)

        x_sub_lp_full = _build_subproblem_commitment_matrix(
            sample,
            lambda_val,
            trainers,
            x_true_full.shape,
            solve_milp=False,
        )
        x_sub_milp_full = _build_subproblem_commitment_matrix(
            sample,
            lambda_val,
            trainers,
            x_true_full.shape,
            solve_milp=True,
        )

        x_sub_lp = np.asarray(x_sub_lp_full[active_units, :], dtype=float)
        x_sub_milp = np.asarray(x_sub_milp_full[active_units, :], dtype=float)

        lp_metrics = _compute_commitment_distance_metrics_with_mask(x_sub_lp, x_true)
        milp_metrics = _compute_commitment_distance_metrics_with_mask(x_sub_milp, x_true)
        lp_vs_milp = _compute_commitment_distance_metrics_with_mask(x_sub_lp, x_sub_milp)

        results.append(
            {
                'sample_index': i,
                'lp': lp_metrics,
                'milp': milp_metrics,
                'lp_vs_milp': lp_vs_milp,
            }
        )
        lp_solutions.append(x_sub_lp)
        milp_solutions.append(x_sub_milp)

        log(
            "  "
            + " | ".join(
                [
                    _fmt_masked_commitment_metrics("LP", lp_metrics),
                    _fmt_masked_commitment_metrics("MILP", milp_metrics),
                    _fmt_distance_metrics("LP vs MILP", lp_vs_milp['l1'], lp_vs_milp['hamming']),
                ]
            )
        )

    if results:
        log_section("Subproblem surrogate LP/MILP 汇总")
        valid_lp = [r['lp'] for r in results if r['lp']['hamming'] is not None]
        valid_milp = [r['milp'] for r in results if r['milp']['hamming'] is not None]
        valid_diff = [r['lp_vs_milp'] for r in results if r['lp_vs_milp']['hamming'] is not None]
        if valid_lp:
            log(
                f"Subproblem LP 汇总: mean_hamming={np.mean([m['hamming'] for m in valid_lp]):.2f}, "
                f"mean_integrality={np.mean([m['integrality_gap'] for m in valid_lp]):.4f}"
            )
        if valid_milp:
            log(
                f"Subproblem MILP 汇总: mean_hamming={np.mean([m['hamming'] for m in valid_milp]):.2f}, "
                f"mean_integrality={np.mean([m['integrality_gap'] for m in valid_milp]):.4f}"
            )
        if valid_diff:
            log(
                f"LP vs MILP 汇总: mean_L1={np.mean([m['l1'] for m in valid_diff]):.4f}, "
                f"mean_hamming={np.mean([m['hamming'] for m in valid_diff]):.2f}"
            )
        _log_solution_similarity_summary(
            "Subproblem LP",
            lp_solutions,
            "Subproblem MILP",
            milp_solutions,
        )

    return results


def run_fp_test(ppc, all_samples: list, dual_predictor, trainers: dict,
                T_DELTA: float, n_test: int, agent=None,
                scenario_bank: list | None = None,
                fig_dir: Path | None = None) -> list:
    """对多个样本运行可行性泵并汇总结果。"""
    test_n = min(n_test, len(all_samples))
    log_section(f"可行性泵测试 | samples={test_n}")

    results = []
    screening_records = []
    if scenario_bank is None:
        scenario_bank = all_samples
    custom_fp_plot_dir = None
    if CASE_NAME == 'case3lite' and USE_CASE3LITE_CUSTOM_FP:
        custom_fp_plot_dir = (Path(__file__).parent / CASE3LITE_CUSTOM_FP_PLOT_DIR).resolve()
        custom_fp_plot_dir.mkdir(parents=True, exist_ok=True)
        log(f"case3lite custom FP plots | dir={custom_fp_plot_dir}")
    for i in range(test_n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        log(_fmt_sample_tag(i, test_n, pd_data.shape))
        if i == 0:
            _log_lambda_prediction_summary(dual_predictor, sample, ppc['gen'].shape[0], pd_data.shape[1], "FP")
        try:
            if CASE_NAME == 'case3lite' and USE_CASE3LITE_CUSTOM_FP:
                x_result, success, _details = recover_integer_solution_case3lite(
                    sample, trainers, dual_predictor, ppc, T_DELTA,
                    agent=agent,
                    max_fp_iter=FP_MAX_ITER,
                    conf_threshold=FP_CONF_THRESHOLD,
                    n_perturbations=FP_MAX_PERTURBATION_HOT_STARTS,
                    scenario_bank=scenario_bank,
                    plot_dir=custom_fp_plot_dir,
                    sample_tag=f'sample_{i:03d}',
                    max_global_combinations=CASE3LITE_CUSTOM_FP_MAX_GLOBAL_COMBINATIONS,
                    verbose=True,
                )
            else:
                x_result, success, fp_details = recover_integer_solution(
                    sample, trainers, dual_predictor, ppc, T_DELTA,
                    agent=agent,
                    max_fp_iter=FP_MAX_ITER,
                    conf_threshold=FP_CONF_THRESHOLD,
                    max_perturbation_hot_starts=FP_MAX_PERTURBATION_HOT_STARTS,
                    max_unit_options_per_generator=FP_MAX_UNIT_OPTIONS_PER_GENERATOR,
                    max_unit_combination_candidates=FP_MAX_UNIT_COMBINATION_CANDIDATES,
                    max_nearby_commitment_hot_starts=FP_MAX_NEARBY_COMMITMENT_HOT_STARTS,
                    nearby_commitment_pool_size=FP_NEARBY_COMMITMENT_POOL_SIZE,
                    parallel_fp_starts=FP_PARALLEL_STARTS,
                    scenario_bank=scenario_bank,
                    surrogate_screen_mode=FP_SURROGATE_SCREEN_MODE,
                    surrogate_screen_max_constraints_per_unit=FP_SURROGATE_SCREEN_MAX_CONSTRAINTS_PER_UNIT,
                    surrogate_screen_min_support_ratio=FP_SURROGATE_SCREEN_MIN_SUPPORT_RATIO,
                    surrogate_screen_max_normalized_violation=FP_SURROGATE_SCREEN_MAX_NORMALIZED_VIOLATION,
                    surrogate_screen_min_mean_margin=FP_SURROGATE_SCREEN_MIN_MEAN_MARGIN,
                    surrogate_screen_candidate_violation_tol=FP_SURROGATE_SCREEN_CANDIDATE_VIOLATION_TOL,
                    surrogate_screen_soft_penalty=FP_SURROGATE_SCREEN_SOFT_PENALTY,
                    projection_objective_tau=FP_PROJECTION_OBJECTIVE_TAU,
                    return_details=True,
                    verbose=True,
                )
                screen_summary = fp_details.get('surrogate_screen_summary', {})
                if screen_summary:
                    screening_records.append(
                        {
                            'sample_index': i,
                            'hot_starts_before': int(screen_summary.get('hot_starts_before', 0)),
                            'hot_starts_after': int(screen_summary.get('hot_starts_after', 0)),
                            'x_pool_before': int(screen_summary.get('x_pool_before', 0)),
                            'x_pool_after': int(screen_summary.get('x_pool_after', 0)),
                            'n_constraints': int(screen_summary.get('n_constraints', 0)),
                        }
                    )
                    log(
                        "  FP筛选: "
                        f"hot_starts {screen_summary.get('hot_starts_before', 0)} -> {screen_summary.get('hot_starts_after', 0)} | "
                        f"x_pool {screen_summary.get('x_pool_before', 0)} -> {screen_summary.get('x_pool_after', 0)} | "
                        f"stable_rows={screen_summary.get('n_constraints', 0)} | "
                        f"soft_penalty={fp_details.get('surrogate_screen_soft_penalty', 0.0):.2f} | "
                        f"tau={fp_details.get('projection_objective_tau', 'adaptive')}"
                    )
        except Exception as e:
            log(f"  异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((i, False, None))
            continue

        status = "成功" if success else "失败"
        log(f"  可行性泵结果: {status}")
        results.append((i, success, x_result))

    n_success = sum(1 for _, s, _ in results if s)
    log_section(f"可行性泵汇总 | success={n_success}/{test_n}")
    if fig_dir is not None and screening_records:
        plot_fp_screening_comparison(screening_records, fig_dir, CASE_NAME)
    return results


def test_surrogate(ppc, all_samples: list, T_DELTA: float,
                   model_dir: str, unit_ids, fig_dir: Path,
                   scenario_bank: list | None = None,
                   constraint_generation_strategy: str | None = None) -> None:
    """加载 surrogate 模型，打印参数摘要，绘图，并可选运行 FP。"""
    log_section(f"加载 surrogate 模型 | dir={model_dir}")
    log("说明: surrogate 模式分析 `Standard LP` 与 `Surrogate LP`，并可选运行 FP 恢复整数解。")

    if not Path(model_dir).exists():
        log(f"错误: 模型目录不存在: {model_dir}")
        log("请先运行 run_training.py 生成模型，或修改 MODEL_DIR 配置")
        sys.exit(1)

    dual_predictor, trainers = load_trained_models_for_test(
        ppc, all_samples, T_DELTA,
        model_dir=model_dir,
        unit_ids=unit_ids,
        requested_strategy=constraint_generation_strategy,
        requested_ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS,
    )

    log(
        f"dual predictor loaded: legacy_mode={getattr(dual_predictor, '_legacy_mode', None)}, "
        f"output_dim={getattr(dual_predictor, 'output_dim', 'n/a')}"
    )
    log(f"已加载 {len(trainers)} 个机组的代理约束模型")
    log(f"surrogate units: {_describe_loaded_surrogate_units(ppc, trainers)}")
    can_run_fp, fp_gate_reason = _can_run_global_fp(ppc, trainers)
    if RUN_FP and can_run_fp:
        log(f"FP gate: enabled ({fp_gate_reason})")
    elif RUN_FP:
        log(f"FP gate: disabled ({fp_gate_reason})")
    print_surrogate_results(trainers, all_samples[:TEST_SAMPLES])

    log_section("生成 surrogate 分析图表")
    plot_surrogate_analysis(trainers, all_samples, fig_dir, CASE_NAME)

    if RUN_FP and can_run_fp:
        log_section("LP 松弛解质量评估 | mode=FP 前置分析")
    elif RUN_FP:
        log_section(f"LP 松弛解质量评估 | mode=仅分析 | reason={fp_gate_reason}")
    else:
        log_section("LP 松弛解质量评估 | mode=RUN_FP=False")
    run_lp_compare_test(ppc, all_samples, dual_predictor, trainers,
                        T_DELTA, TEST_SAMPLES, fig_dir)
    if RUN_SUBPROBLEM_MILP_TEST:
        run_subproblem_milp_test(
            ppc,
            all_samples,
            dual_predictor,
            trainers,
            T_DELTA,
            TEST_SAMPLES,
        )

    fp_results = None
    if RUN_FP and can_run_fp:
        fp_results = run_fp_test(
            ppc, all_samples, dual_predictor, trainers, T_DELTA, TEST_SAMPLES,
            scenario_bank=scenario_bank, fig_dir=fig_dir,
        )
        plot_fp_results(fp_results, fig_dir, CASE_NAME)
        summarize_fp_economicity(
            ppc,
            all_samples,
            fp_results,
            T_DELTA,
            fig_dir,
            CASE_NAME,
        )
    elif RUN_FP:
        log(f"跳过可行性泵（RUN_FP=True, but {fp_gate_reason}）")

    summarize_lp_surrogate_fp_totals(
        ppc,
        all_samples,
        dual_predictor,
        trainers,
        T_DELTA,
        TEST_SAMPLES,
        fig_dir,
        CASE_NAME,
        fp_results=fp_results,
    )


def _print_bcd_stats(agent) -> None:
    """打印 BCD agent 模型参数统计摘要（辅助函数）。"""
    log_section("BCD 模型参数统计")

    if hasattr(agent, 'theta_net') and agent.theta_net is not None:
        total_params = sum(p.numel() for p in agent.theta_net.parameters())
        log(f"  theta_net 参数量: {total_params:,}")

    if hasattr(agent, 'zeta_net') and agent.zeta_net is not None:
        total_params = sum(p.numel() for p in agent.zeta_net.parameters())
        log(f"  zeta_net  参数量: {total_params:,}")

    theta_vals = _collect_bcd_param_values(agent, "theta")
    if theta_vals.size > 0:
        vals = theta_vals
        log(f"  theta_values 数量: {len(vals)}，"
            f"均值={vals.mean():.4f}，标准差={vals.std():.4f}，"
            f"范围=[{vals.min():.4f}, {vals.max():.4f}]")

    zeta_vals = _collect_bcd_param_values(agent, "zeta")
    if zeta_vals.size > 0:
        vals = zeta_vals
        log(f"  zeta_values  数量: {len(vals)}，"
            f"均值={vals.mean():.4f}，标准差={vals.std():.4f}，"
            f"范围=[{vals.min():.4f}, {vals.max():.4f}]")


def _collect_bcd_param_values(agent, prefix: str) -> np.ndarray:
    """Collect sample-specific theta/zeta values; fall back to the legacy single-sample dict."""
    values_list_attr = f"{prefix}_values_list"
    single_attr = f"{prefix}_values"

    if hasattr(agent, values_list_attr):
        values_list = getattr(agent, values_list_attr)
        flat_vals = []
        for values in values_list or []:
            if values:
                flat_vals.extend(float(v) for v in values.values())
        if flat_vals:
            return np.asarray(flat_vals, dtype=float)

    if hasattr(agent, single_attr):
        values = getattr(agent, single_attr)
        if values:
            return np.asarray(list(values.values()), dtype=float)

    return np.asarray([], dtype=float)


def _load_bcd_agent(ppc, data_file: Path, bcd_model_path: str,
                    MAX_SAMPLES, T_DELTA: float, sample_range: tuple[int, int] | None = None):
    """加载 BCD 数据并恢复 Agent_NN_BCD 模型参数，返回 agent。"""
    model_path = Path(bcd_model_path)
    if not model_path.exists():
        log(f"错误: BCD 模型文件不存在: {bcd_model_path}")
        log("请先运行 run_training.py MODE='bcd'/'both' 生成模型，或修改 BCD_MODEL_PATH 配置")
        sys.exit(1)

    log(f"通过 load_active_set_from_json 加载数据: {data_file.name}")
    all_samples_bcd = load_active_set_from_json(str(data_file))
    all_samples_bcd = apply_sample_range(all_samples_bcd, sample_range)
    if MAX_SAMPLES and len(all_samples_bcd) > MAX_SAMPLES:
        all_samples_bcd = all_samples_bcd[:MAX_SAMPLES]
    log(f"  使用 {len(all_samples_bcd)} 个 BCD 样本")
    log(
        "  BCD 配置: "
        f"lambda_init={BCD_LAMBDA_INIT_STRATEGY}, "
        f"max_theta_per_t={BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT}, "
        f"rho=({BCD_RHO_PRIMAL_INIT}, {BCD_RHO_DUAL_INIT}, {BCD_RHO_OPT_INIT}), "
        f"rho_dual_components=({BCD_RHO_DUAL_PG_INIT}, {BCD_RHO_DUAL_X_INIT}, {BCD_RHO_DUAL_COC_INIT}), "
        f"restore_rho={BCD_RESTORE_RHO_FROM_CHECKPOINT}, "
        f"gamma_base={BCD_GAMMA_BASE}"
    )

    agent = Agent_NN_BCD(
        ppc,
        all_samples_bcd,
        T_DELTA,
        lambda_init_strategy=BCD_LAMBDA_INIT_STRATEGY,
        max_theta_constraints_per_time_slot=BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT,
        theta_hot_start_strategy=THETA_HOT_START_STRATEGY,
        zeta_hot_start_strategy=ZETA_HOT_START_STRATEGY,
        theta_gaussian_std=THETA_GAUSSIAN_STD,
        zeta_gaussian_std=ZETA_GAUSSIAN_STD,
        rho_primal_init=BCD_RHO_PRIMAL_INIT,
        rho_dual_init=BCD_RHO_DUAL_INIT,
        rho_dual_pg_init=BCD_RHO_DUAL_PG_INIT,
        rho_dual_x_init=BCD_RHO_DUAL_X_INIT,
        rho_dual_coc_init=BCD_RHO_DUAL_COC_INIT,
        rho_opt_init=BCD_RHO_OPT_INIT,
        gamma_base=BCD_GAMMA_BASE,
    )
    agent.load_model_parameters(
        str(model_path),
        restore_rho_state=BCD_RESTORE_RHO_FROM_CHECKPOINT,
    )
    log("BCD 模型加载成功")
    return agent


def plot_lp_surrogate_comparison(
    x_opt: np.ndarray,
    x_lp: np.ndarray,
    x_surr: np.ndarray,
    sample_id: int,
    fig_dir: Path,
    case_name: str,
    comparison_name: str = "Surrogate LP",
    file_tag: str = "surrogate",
) -> None:
    """绘制单样�?3×2 热图：x* / x_LP / x_LP_surrogate 的状态与误差�?

    左列：连续值热�?(0~1, Blues)；右列：|x - x*| 误差热图 (Reds)�?

    Args:
        x_opt:     最优整数解 (ng, T)�?
        x_lp:      标准 LP 松弛�?(ng, T)�?
        x_surr:    含代理约�?LP �?(ng, T)�?
        sample_id: 样本编号（仅用于标题/文件名）�?
        fig_dir:   图像输出目录�?
        case_name: 算例名�?
    """
    if not MPL_AVAILABLE:
        return

    _apply_style()
    ng, T = x_opt.shape
    diff_lp = np.abs(x_lp - x_opt)
    diff_surr = np.abs(x_surr - x_opt)

    l1_lp = float(np.sum(diff_lp))
    l1_surr = float(np.sum(diff_surr))

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 9),
        gridspec_kw={'width_ratios': [1, 1], 'hspace': 0.45, 'wspace': 0.15},
    )
    fig.suptitle(
        f'LP Relaxation vs {comparison_name}  [Sample {sample_id}, {case_name}]',
        fontsize=13, fontweight='bold', y=1.01,
    )

    data_left = [x_opt, x_lp, x_surr]
    data_right = [np.zeros_like(x_opt), diff_lp, diff_surr]
    row_titles = [
        r'A. $x^*$ (True Optimum)',
        f'B. $x_{{LP}}$  (L1 = {l1_lp:.2f})',
        f'C. {comparison_name}  (L1 = {l1_surr:.2f})',
    ]
    right_titles = [
        '(reference)',
        f'|$x_{{LP}} - x^*$|',
        f'|{comparison_name} - x^*|',
    ]

    yticks = list(range(ng))
    ylabels = [f'G{g}' for g in yticks]

    for i in range(3):
        # 左列：状态热�?
        ax_l = axes[i, 0]
        im_l = ax_l.imshow(data_left[i], aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax_l.set_title(row_titles[i], loc='left', fontsize=11, fontweight='bold')
        ax_l.set_yticks(yticks)
        ax_l.set_yticklabels(ylabels)
        ax_l.set_ylabel('Unit')
        if i == 2:
            ax_l.set_xlabel('Time Period')

        # 右列：误差热�?
        ax_r = axes[i, 1]
        if i == 0:
            ax_r.axis('off')
            ax_r.text(0.5, 0.5, 'No error\n(reference)', ha='center', va='center',
                      fontsize=12, color='#888888', transform=ax_r.transAxes)
        else:
            im_r = ax_r.imshow(data_right[i], aspect='auto', cmap='Reds', vmin=0, vmax=1)
            ax_r.set_title(right_titles[i], loc='left', fontsize=11)
            ax_r.set_yticks(yticks)
            ax_r.set_yticklabels(ylabels)
            if i == 2:
                ax_r.set_xlabel('Time Period')
            fig.colorbar(im_r, ax=ax_r, fraction=0.045, pad=0.03)

    # 给左列第一行加 colorbar
    fig.colorbar(im_l, ax=axes[0, 0], fraction=0.045, pad=0.03)

    path = fig_dir / f'lp_{file_tag}_heatmap_sample{sample_id}_{case_name}'
    _save_fig(fig, path, f'LP vs {comparison_name} heatmap sample {sample_id}')


def plot_surrogate_commitment_comparison(
    x_opt: np.ndarray,
    x_lp: np.ndarray,
    x_surr: np.ndarray,
    x_sub: np.ndarray,
    sample_id: int,
    fig_dir: Path,
    case_name: str,
) -> None:
    """绘制单样本 True / LP / Surrogate LP / 子模型 surrogate 的对比热图。"""
    if not MPL_AVAILABLE:
        return

    _apply_style()
    ng, _ = x_opt.shape
    valid_sub = np.isfinite(x_sub)
    diff_lp = np.abs(x_lp - x_opt)
    diff_surr = np.abs(x_surr - x_opt)
    diff_sub = np.where(valid_sub, np.abs(x_sub - x_opt), np.nan)

    l1_lp = float(np.sum(diff_lp))
    l1_surr = float(np.sum(diff_surr))
    l1_sub = float(np.nansum(diff_sub))

    state_cmap = plt.get_cmap('Blues').copy()
    state_cmap.set_bad('#E6E6E6')
    diff_cmap = plt.get_cmap('Reds').copy()
    diff_cmap.set_bad('#E6E6E6')

    fig, axes = plt.subplots(
        4, 2, figsize=(12, 12),
        gridspec_kw={'width_ratios': [1, 1], 'hspace': 0.42, 'wspace': 0.16},
    )
    fig.suptitle(
        f'Surrogate Commitment Comparison  [Sample {sample_id}, {case_name}]',
        fontsize=13, fontweight='bold', y=1.01,
    )

    left_data = [x_opt, x_lp, x_surr, np.ma.masked_invalid(x_sub)]
    right_data = [None, diff_lp, diff_surr, np.ma.masked_invalid(diff_sub)]
    left_titles = [
        r'A. $x^*$ (True Optimum)',
        f'B. $x_{{LP}}$  (L1 = {l1_lp:.2f})',
        f'C. $x_{{LP,surr}}$  (L1 = {l1_surr:.2f})',
        f'D. $x_{{sub,surr}}$  (L1 = {l1_sub:.2f})',
    ]
    right_titles = [
        '(reference)',
        r'|$x_{LP} - x^*$|',
        r'|$x_{LP,surr} - x^*$|',
        r'|$x_{sub,surr} - x^*$|',
    ]

    yticks = list(range(ng))
    ylabels = [f'G{g}' for g in yticks]
    state_im = None
    for i in range(4):
        ax_l = axes[i, 0]
        state_im = ax_l.imshow(left_data[i], aspect='auto', cmap=state_cmap, vmin=0, vmax=1)
        ax_l.set_title(left_titles[i], loc='left', fontsize=11, fontweight='bold')
        ax_l.set_yticks(yticks)
        ax_l.set_yticklabels(ylabels, fontsize=7)
        ax_l.set_ylabel('Unit')
        if i == 3:
            ax_l.set_xlabel('Time Period')

        ax_r = axes[i, 1]
        if right_data[i] is None:
            ax_r.axis('off')
            ax_r.text(
                0.5, 0.5, 'No error\n(reference)',
                ha='center', va='center',
                fontsize=12, color='#888888',
                transform=ax_r.transAxes,
            )
            continue

        diff_im = ax_r.imshow(right_data[i], aspect='auto', cmap=diff_cmap, vmin=0, vmax=1)
        ax_r.set_title(right_titles[i], loc='left', fontsize=11)
        ax_r.set_yticks(yticks)
        ax_r.set_yticklabels(ylabels, fontsize=7)
        if i == 3:
            ax_r.set_xlabel('Time Period')
        fig.colorbar(diff_im, ax=ax_r, fraction=0.045, pad=0.03)

    if state_im is not None:
        fig.colorbar(state_im, ax=axes[:, 0], fraction=0.025, pad=0.02)

    path = fig_dir / f'surrogate_commitment_comparison_sample{sample_id}_{case_name}'
    _save_fig(fig, path, f'Surrogate commitment comparison sample {sample_id}')


def plot_lp_surrogate_bar(
    dist_lp: np.ndarray,
    dist_surr: np.ndarray,
    hamming_lp: np.ndarray,
    hamming_surr: np.ndarray,
    fig_dir: Path,
    case_name: str,
    comparison_name: str = "Surrogate LP",
    file_tag: str = "surrogate",
) -> None:
    """绘制逐样�?L1 距离�?Hamming 距离对比柱状图（LP vs Surrogate LP）�?

    Args:
        dist_lp:      各样�?|x_LP - x*| �?L1 距离�?
        dist_surr:    各样�?|x_LP_surr - x*| �?L1 距离�?
        hamming_lp:   各样本舍入后 Hamming 距离�?
        hamming_surr: 各样本舍入后 Hamming 距离�?
        fig_dir:      图像输出目录�?
        case_name:    算例名�?
    """
    if not MPL_AVAILABLE:
        return

    _apply_style()
    n = len(dist_lp)
    x_pos = np.arange(n)
    bar_w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'LP Distance Comparison  [{case_name}]',
        fontsize=13, fontweight='bold', y=1.02,
    )

    # 左：L1 距离
    ax1.bar(x_pos - bar_w / 2, dist_lp, bar_w, label='Standard LP', color='#2166AC')
    ax1.bar(x_pos + bar_w / 2, dist_surr, bar_w, label=comparison_name, color='#D6604D')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel(r'$\| x - x^* \|_1$')
    ax1.set_title('L1 Distance to Optimum')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'S{i}' for i in range(n)])
    ax1.legend()

    # 右：Hamming 距离
    ax2.bar(x_pos - bar_w / 2, hamming_lp, bar_w, label='Standard LP', color='#2166AC')
    ax2.bar(x_pos + bar_w / 2, hamming_surr, bar_w, label=comparison_name, color='#D6604D')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Hamming Distance')
    ax2.set_title('Hamming Distance (after rounding)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'S{i}' for i in range(n)])
    ax2.legend()

    fig.tight_layout()
    path = fig_dir / f'lp_{file_tag}_distance_bar_{case_name}'
    _save_fig(fig, path, f'LP vs {comparison_name} distance bar')


def plot_lp_surrogate_fp_total_bar(
    total_l1: dict[str, float],
    total_hamming: dict[str, int],
    fig_dir: Path,
    case_name: str,
) -> None:
    """Plot total L1 and Hamming distances for LP / surrogate / surrogate+FP."""
    if not MPL_AVAILABLE:
        return

    _apply_style()
    method_names = list(total_l1.keys())
    l1_values = [float(total_l1[name]) for name in method_names]
    hamming_values = [float(total_hamming[name]) for name in method_names]
    colors = ['#2166AC', '#D6604D', '#4DAC26'][:len(method_names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f'Total Commitment Distance Summary  [{case_name}]',
        fontsize=12, fontweight='bold', y=1.02,
    )

    x_pos = np.arange(len(method_names))
    bars1 = ax1.bar(x_pos, l1_values, color=colors, alpha=0.82, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names)
    ax1.set_ylabel(r'Total $\|x - x^*\|_1$')
    ax1.set_title('Total L1 Distance')
    for bar, val in zip(bars1, l1_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(l1_values) * 0.02 if l1_values else 0.0,
            f'{val:.1f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    bars2 = ax2.bar(x_pos, hamming_values, color=colors, alpha=0.82, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names)
    ax2.set_ylabel('Total Hamming Distance')
    ax2.set_title('Total Hamming Distance')
    for bar, val in zip(bars2, hamming_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(hamming_values) * 0.02 if hamming_values else 0.0,
            f'{val:.0f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    fig.tight_layout()
    path = fig_dir / f'lp_surrogate_fp_total_distance_{case_name}'
    _save_fig(fig, path, 'LP / surrogate / FP total distance summary')


def plot_fp_economicity_bar(
    sample_ids: list[int],
    optimal_costs: np.ndarray,
    fp_costs: np.ndarray,
    rel_gaps_pct: np.ndarray,
    fig_dir: Path,
    case_name: str,
) -> None:
    """Plot per-sample FP economicity against the optimal commitment."""
    if not MPL_AVAILABLE or len(sample_ids) == 0:
        return

    _apply_style()
    x_pos = np.arange(len(sample_ids))
    bar_w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        f'FP Economicity vs Optimal Commitment  [{case_name}]',
        fontsize=12, fontweight='bold', y=1.02,
    )

    ax1.bar(x_pos - bar_w / 2, optimal_costs, bar_w, label='Optimal', color='#2166AC')
    ax1.bar(x_pos + bar_w / 2, fp_costs, bar_w, label='FP', color='#4DAC26')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'S{i}' for i in sample_ids])
    ax1.set_ylabel('Total Cost')
    ax1.set_title('Cost Comparison')
    ax1.legend()

    bars = ax2.bar(x_pos, rel_gaps_pct, color='#D6604D', alpha=0.82, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'S{i}' for i in sample_ids])
    ax2.set_ylabel('Relative Gap (%)')
    ax2.set_title('FP Cost Gap vs Optimal')
    for bar, val in zip(bars, rel_gaps_pct):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rel_gaps_pct) * 0.02 if len(rel_gaps_pct) else 0.0,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    fig.tight_layout()
    path = fig_dir / f'fp_economicity_{case_name}'
    _save_fig(fig, path, 'FP economicity comparison')


def summarize_lp_surrogate_fp_totals(
    ppc,
    all_samples: list,
    dual_predictor,
    trainers: dict,
    T_DELTA: float,
    test_n: int,
    fig_dir: Path | None,
    case_name: str,
    agent=None,
    fp_results: list | None = None,
) -> None:
    """Summarize total L1 / Hamming distances for LP, surrogate LP, and surrogate+FP."""
    n = min(test_n, len(all_samples))
    if n <= 0:
        return

    log_section(f"LP / Surrogate / FP 总距离汇总 | samples={n}")

    fp_result_map = {}
    if fp_results:
        for sample_idx, success, x_result in fp_results:
            fp_result_map[int(sample_idx)] = {
                'success': bool(success),
                'x': None if x_result is None else np.asarray(x_result, dtype=float),
            }

    totals_l1 = {
        'LP': 0.0,
        'Surrogate': 0.0,
    }
    totals_hamming = {
        'LP': 0,
        'Surrogate': 0,
    }
    fp_totals_l1 = 0.0
    fp_totals_hamming = 0
    fp_available = 0
    fp_success = 0
    valid_samples = 0

    for i in range(n):
        sample = all_samples[i]
        pd_data = sample['pd_data']
        try:
            lambda_val = dual_predictor.predict(sample)
            x_lp = solve_global_LP_relaxation_without_surrogate(ppc, pd_data, T_DELTA)
            x_surr = solve_global_LP_relaxation(
                ppc,
                sample,
                T_DELTA,
                trainers,
                lambda_val,
                agent=agent,
            )
        except Exception as exc:
            log(f"样本 {i + 1}/{n} | LP / Surrogate 求解失败，跳过: {exc}")
            continue

        x_true = _extract_true_solution(sample, x_surr.shape)
        l1_lp, hamming_lp = _compute_commitment_distance_metrics(x_lp, x_true)
        l1_surr, hamming_surr = _compute_commitment_distance_metrics(x_surr, x_true)
        totals_l1['LP'] += l1_lp
        totals_l1['Surrogate'] += l1_surr
        totals_hamming['LP'] += hamming_lp
        totals_hamming['Surrogate'] += hamming_surr
        valid_samples += 1

        fp_msg = "FP=unavailable"
        fp_entry = fp_result_map.get(i)
        if fp_entry is not None and fp_entry['x'] is not None:
            x_fp = fp_entry['x']
            if x_fp.shape == x_true.shape:
                l1_fp, hamming_fp = _compute_commitment_distance_metrics(x_fp, x_true)
                fp_totals_l1 += l1_fp
                fp_totals_hamming += hamming_fp
                fp_available += 1
                fp_success += int(fp_entry['success'])
                fp_msg = (
                    f"FP: L1={l1_fp:.2f}, Hamming={hamming_fp}, "
                    f"success={fp_entry['success']}"
                )
            else:
                fp_msg = f"FP shape mismatch={x_fp.shape}"

        log(
            f"样本 {i + 1}/{n} | "
            f"LP(L1={l1_lp:.2f}, Hamming={hamming_lp}) | "
            f"Surrogate(L1={l1_surr:.2f}, Hamming={hamming_surr}) | "
            f"{fp_msg}"
        )

    if valid_samples == 0:
        log("没有可用于总距离汇总的有效样本")
        return

    display_l1 = dict(totals_l1)
    display_hamming = dict(totals_hamming)
    if fp_available > 0:
        display_l1['Surrogate+FP'] = fp_totals_l1
        display_hamming['Surrogate+FP'] = fp_totals_hamming

    log_rule("-")
    log(f"总量汇总: valid_samples={valid_samples}/{n}")
    log(f"  LP:            L1_sum={totals_l1['LP']:.2f}, Hamming_sum={totals_hamming['LP']}")
    log(
        f"  Surrogate:     L1_sum={totals_l1['Surrogate']:.2f}, "
        f"Hamming_sum={totals_hamming['Surrogate']}"
    )
    log(
        f"  Surrogate 相对 LP 改进: "
        f"L1_sum减少 {totals_l1['LP'] - totals_l1['Surrogate']:.2f}, "
        f"Hamming_sum减少 {totals_hamming['LP'] - totals_hamming['Surrogate']}"
    )

    if fp_results is not None:
        if fp_available > 0:
            log(
                f"  Surrogate+FP:  L1_sum={fp_totals_l1:.2f}, Hamming_sum={fp_totals_hamming} "
                f"(available={fp_available}/{valid_samples}, success={fp_success}/{fp_available})"
            )
            log(
                f"  Surrogate+FP 相对 LP 改进: "
                f"L1_sum减少 {totals_l1['LP'] - fp_totals_l1:.2f}, "
                f"Hamming_sum减少 {totals_hamming['LP'] - fp_totals_hamming}"
            )
            log(
                f"  Surrogate+FP 相对 Surrogate 改进: "
                f"L1_sum减少 {totals_l1['Surrogate'] - fp_totals_l1:.2f}, "
                f"Hamming_sum减少 {totals_hamming['Surrogate'] - fp_totals_hamming}"
            )
        else:
            log("  Surrogate+FP: 没有可用输出，无法统计总距离")
    print("-" * SECTION_WIDTH)

    if fig_dir is not None:
        plot_lp_surrogate_fp_total_bar(display_l1, display_hamming, fig_dir, case_name)


def summarize_fp_economicity(
    ppc,
    all_samples: list,
    fp_results: list | None,
    T_DELTA: float,
    fig_dir: Path | None,
    case_name: str,
) -> None:
    """Compare FP commitments with the sample optimal commitments in economic cost."""
    if not fp_results:
        return

    log_section("FP 经济性评估")

    sample_ids: list[int] = []
    optimal_costs: list[float] = []
    fp_costs: list[float] = []
    abs_gaps: list[float] = []
    rel_gaps_pct: list[float] = []
    exact_match_count = 0

    for sample_idx, success, x_result in fp_results:
        if x_result is None:
            log(f"  样本 {sample_idx}: FP 无输出，跳过经济性评估")
            continue
        if not (0 <= int(sample_idx) < len(all_samples)):
            continue

        sample = all_samples[int(sample_idx)]
        x_true = _extract_true_solution(sample, np.asarray(x_result, dtype=float).shape)
        optimal_eval = _evaluate_commitment_economic_cost(ppc, sample, x_true, T_DELTA)
        fp_eval = _evaluate_commitment_economic_cost(ppc, sample, x_result, T_DELTA)

        if not optimal_eval.get('success', False):
            log(f"  样本 {sample_idx}: 最优解经济性评估失败 ({optimal_eval.get('reason', 'unknown')})")
            continue
        if not fp_eval.get('success', False):
            log(f"  样本 {sample_idx}: FP 解经济性评估失败 ({fp_eval.get('reason', 'unknown')})")
            continue

        optimal_cost = float(optimal_eval['total_cost'])
        fp_cost = float(fp_eval['total_cost'])
        abs_gap = fp_cost - optimal_cost
        rel_gap_pct = 100.0 * abs_gap / max(abs(optimal_cost), 1e-9)

        sample_ids.append(int(sample_idx))
        optimal_costs.append(optimal_cost)
        fp_costs.append(fp_cost)
        abs_gaps.append(abs_gap)
        rel_gaps_pct.append(rel_gap_pct)
        if abs(abs_gap) <= 1e-6:
            exact_match_count += 1

        log(
            f"  样本 {sample_idx}: "
            f"optimal_cost={optimal_cost:.2f}, fp_cost={fp_cost:.2f}, "
            f"abs_gap={abs_gap:.2f}, rel_gap={rel_gap_pct:.2f}%, success={success}"
        )

    if not sample_ids:
        log("没有可用于 FP 经济性评估的有效样本")
        return

    optimal_arr = np.asarray(optimal_costs, dtype=float)
    fp_arr = np.asarray(fp_costs, dtype=float)
    abs_gap_arr = np.asarray(abs_gaps, dtype=float)
    rel_gap_arr = np.asarray(rel_gaps_pct, dtype=float)

    log_rule("-")
    log(f"经济性汇总: evaluated={len(sample_ids)}/{len(fp_results)}")
    log(f"  Optimal cost mean={float(np.mean(optimal_arr)):.2f}, sum={float(np.sum(optimal_arr)):.2f}")
    log(f"  FP cost mean={float(np.mean(fp_arr)):.2f}, sum={float(np.sum(fp_arr)):.2f}")
    log(
        f"  FP abs gap: mean={float(np.mean(abs_gap_arr)):.2f}, "
        f"median={float(np.median(abs_gap_arr)):.2f}, max={float(np.max(abs_gap_arr)):.2f}"
    )
    log(
        f"  FP rel gap: mean={float(np.mean(rel_gap_arr)):.2f}%, "
        f"median={float(np.median(rel_gap_arr)):.2f}%, max={float(np.max(rel_gap_arr)):.2f}%"
    )
    log(f"  FP exact optimal count={exact_match_count}/{len(sample_ids)}")
    print("-" * SECTION_WIDTH)

    if fig_dir is not None:
        plot_fp_economicity_bar(
            sample_ids,
            optimal_arr,
            fp_arr,
            rel_gap_arr,
            fig_dir,
            case_name,
        )


def analyse_lp_distance(agent, test_n: int, fig_dir: Path,
                        case_name: str, ppc=None, all_samples=None,
                        dual_predictor=None, trainers=None,
                        T_DELTA: float | None = None) -> None:
    """计算并对比标�?LP 松弛 vs 含代理约�?LP 与最优解的距离�?

    对每个样本求解两�?LP，计�?L1 距离和舍�?Hamming 距离�?
    打印汇总表格并绘制热图 + 柱状图�?

    Args:
        agent:     已加载参数的 Agent_NN_BCD 实例�?
        test_n:    测试样本数�?
        fig_dir:   图像输出目录�?
        case_name: 算例名�?
    """
    n = min(test_n, agent.n_samples)
    print("\n" + "=" * 70)
    log(f"LP 距离分析: {n} 个样本")
    print("=" * 70)

    dist_lp_arr = np.zeros(n)
    dist_surr_arr = np.zeros(n)
    hamming_lp_arr = np.zeros(n, dtype=int)
    hamming_surr_arr = np.zeros(n, dtype=int)

    x_lp_list: list[np.ndarray] = []
    x_surr_list: list[np.ndarray] = []
    x_opt_list: list[np.ndarray] = []
    valid_ids: list[int] = []

    use_joint_surrogate = (
        ppc is not None and all_samples is not None and dual_predictor is not None
        and trainers is not None and T_DELTA is not None
    )
    comparison_name = "Surrogate LP" if use_joint_surrogate else "BCD LP"
    comparison_short = "surr" if use_joint_surrogate else "bcd"
    comparison_file_tag = "surrogate" if use_joint_surrogate else "bcd"

    for i in range(n):
        sol_lp = agent.solve_LP_without_theta_constraints(i)
        if sol_lp is None:
            log(f"  样本 {i}: LP 求解失败，跳过")
            continue

        x_lp = sol_lp[1]       # (ng, T)
        if use_joint_surrogate:
            sample = all_samples[i]
            pd_data = sample['pd_data']
            lambda_val = dual_predictor.predict(sample)
            x_surr = solve_global_LP_relaxation(
                ppc, sample, T_DELTA, trainers, lambda_val, agent=agent
            )
        else:
            sol_surr = agent.solve_LP_with_theta_constraints(i)
            if sol_surr is None:
                log(f"  样本 {i}: {comparison_name} 求解失败，跳过")
                continue
            x_surr = sol_surr[1]   # (ng, T)
        x_opt = agent.x_opt[i] # (ng, T)

        # L1 距离
        d_lp = float(np.sum(np.abs(x_lp - x_opt)))
        d_surr = float(np.sum(np.abs(x_surr - x_opt)))

        # 舍入�?Hamming 距离
        h_lp = int(np.sum(np.round(x_lp).astype(int) != x_opt.astype(int)))
        h_surr = int(np.sum(np.round(x_surr).astype(int) != x_opt.astype(int)))

        dist_lp_arr[i] = d_lp
        dist_surr_arr[i] = d_surr
        hamming_lp_arr[i] = h_lp
        hamming_surr_arr[i] = h_surr

        valid_ids.append(i)
        x_lp_list.append(x_lp)
        x_surr_list.append(x_surr)
        x_opt_list.append(x_opt)

        log(f"  样本 {i}: L1(LP)={d_lp:.2f}  L1({comparison_short})={d_surr:.2f}  "
            f"Hamming(LP)={h_lp}  Hamming({comparison_short})={h_surr}")

    if not valid_ids:
        log("没有有效样本，跳过分析")
        return

    nv = len(valid_ids)
    print("\n" + "-" * 60)
    log(f"汇总 ({nv} 个有效样本):")
    log(f"  平均 L1 距离:      LP={dist_lp_arr[:nv].mean():.2f}  "
        f"{comparison_name}={dist_surr_arr[:nv].mean():.2f}")
    log(f"  平均 Hamming 距离:  LP={hamming_lp_arr[:nv].mean():.1f}  "
        f"{comparison_name}={hamming_surr_arr[:nv].mean():.1f}")
    reduction = (1 - dist_surr_arr[:nv].mean() / max(dist_lp_arr[:nv].mean(), 1e-9)) * 100
    log(f"  {comparison_name} 相对 LP 的 L1 缩减: {reduction:.1f}%")
    _log_solution_similarity_summary(
        "Standard LP",
        x_lp_list,
        comparison_name,
        x_surr_list,
    )
    print("-" * 60)

    plot_lp_surrogate_comparison(
        x_opt_list[0], x_lp_list[0], x_surr_list[0],
        valid_ids[0], fig_dir, case_name,
        comparison_name=comparison_name,
        file_tag=comparison_file_tag,
    )

    plot_lp_surrogate_bar(
        dist_lp_arr[:nv], dist_surr_arr[:nv],
        hamming_lp_arr[:nv], hamming_surr_arr[:nv],
        fig_dir, case_name,
        comparison_name=comparison_name,
        file_tag=comparison_file_tag,
    )


def test_bcd(ppc, data_file: Path, bcd_model_path: str,
             MAX_SAMPLES, T_DELTA: float, fig_dir: Path,
             sample_range: tuple[int, int] | None = None,
             test_samples: int = TEST_SAMPLES_DEFAULT) -> None:
    """加载 BCD 模型，初始化 agent，报告参数统计，绘图，并分析 LP 距离。"""
    log_section(f"加载 BCD 模型 | path={bcd_model_path}")
    log("说明: bcd 模式只分析 `Standard LP` 与 `BCD LP`，不加载 surrogate 模型，也不运行 FP 恢复整数解。")

    agent = _load_bcd_agent(
        ppc,
        data_file,
        bcd_model_path,
        MAX_SAMPLES,
        T_DELTA,
        sample_range=sample_range,
    )
    _print_bcd_stats(agent)

    log_section("生成 BCD 分析图表")
    plot_bcd_analysis(agent, fig_dir, CASE_NAME)

    analyse_lp_distance(agent, test_samples, fig_dir, CASE_NAME)

    log("BCD 测试完成")


def test_both(ppc, data_file: Path, all_samples: list, T_DELTA: float,
              model_dir: str, bcd_model_path: str,
              MAX_SAMPLES, unit_ids, fig_dir: Path,
              sample_range: tuple[int, int] | None = None,
              test_samples: int = TEST_SAMPLES_DEFAULT,
              scenario_bank: list | None = None,
              constraint_generation_strategy: str | None = None) -> None:
    """联合加载 BCD + surrogate 模型，以全体代理约束评估解质量，可选运�?FP�?

    流程�?
      1. 加载 BCD 模型 �?打印参数统计
      2. 加载 surrogate 全体代理约束模型 �?打印约束摘要
      3. 生成各自分析�?+ 联合表征�?
      4. （可选）使用全体代理约束运行可行性泵

    Args:
        ppc:            PyPower 案例字典�?
        data_file:      JSON 数据文件路径（BCD 格式，含 unit_commitment_matrix）�?
        all_samples:    v3 格式样本列表（已预处理）�?
        T_DELTA:        时间间隔�?
        model_dir:      surrogate 模型目录（绝对路径）�?
        bcd_model_path: BCD .pth 文件路径（绝对路径）�?
        MAX_SAMPLES:    BCD 数据最多使用样本数�?
        unit_ids:       机组 ID 列表（None = 全部）�?
        fig_dir:        图像输出目录�?
    """
    log_section("模式=both | 联合评估 BCD 神经网络 + 全体 V3 代理约束")

    # ── Step 1: 加载 BCD 模型 ──────────────────────────────
    log("── Step 1/4  加载 BCD 模型")
    agent = _load_bcd_agent(
        ppc,
        data_file,
        bcd_model_path,
        MAX_SAMPLES,
        T_DELTA,
        sample_range=sample_range,
    )
    _print_bcd_stats(agent)

    # ── Step 2: 加载 surrogate 全体代理约束模型 ───────────
    log("── Step 2/4  加载全体代理约束模型")
    if not Path(model_dir).exists():
        log(f"错误: surrogate 模型目录不存在: {model_dir}")
        log("请先运行 run_training.py 生成模型，或修改 MODEL_DIR 配置")
        sys.exit(1)

    dual_predictor, trainers = load_trained_models_for_test(
        ppc, all_samples, T_DELTA,
        model_dir=model_dir,
        unit_ids=unit_ids,
        requested_strategy=constraint_generation_strategy,
        requested_ignore_startup_shutdown_costs=SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS,
    )
    log(
        f"dual predictor loaded: legacy_mode={getattr(dual_predictor, '_legacy_mode', None)}, "
        f"output_dim={getattr(dual_predictor, 'output_dim', 'n/a')}"
    )
    log(f"已加载 {len(trainers)} 个机组的代理约束模型（全体约束）")
    log(f"surrogate units: {_describe_loaded_surrogate_units(ppc, trainers)}")
    can_run_fp, fp_gate_reason = _can_run_global_fp(ppc, trainers)
    if RUN_FP and can_run_fp:
        log(f"FP gate: enabled ({fp_gate_reason})")
    elif RUN_FP:
        log(f"FP gate: disabled ({fp_gate_reason})")
    print_surrogate_results(trainers, all_samples[:test_samples])

    # ── Step 3: 绘图 ───────────────────────────────────────
    log("── Step 3/4  生成分析图表")
    log_section("生成 surrogate 分析图表")
    plot_surrogate_analysis(trainers, all_samples, fig_dir, CASE_NAME)

    log_section("生成 BCD 分析图表")
    plot_bcd_analysis(agent, fig_dir, CASE_NAME)

    log_section("生成 BCD-Surrogate 联合约束表征图")
    plot_both_analysis(agent, trainers, fig_dir, CASE_NAME)

    # ── Step 4: LP 评估 + 可行性泵（全体代理约束） ────────
    if RUN_FP and can_run_fp:
        log("── Step 4/4  LP 松弛解质量评估（FP 前置分析）")
    elif RUN_FP:
        log(f"── Step 4/4  LP 松弛解质量评估（仅分析，不运行 FP；{fp_gate_reason}）")
    else:
        log("── Step 4/4  LP 松弛解质量评估（RUN_FP=False）")
    run_lp_compare_test(ppc, all_samples, dual_predictor, trainers,
                        T_DELTA, test_samples, fig_dir, agent=agent)
    if RUN_SUBPROBLEM_MILP_TEST:
        run_subproblem_milp_test(
            ppc,
            all_samples,
            dual_predictor,
            trainers,
            T_DELTA,
            test_samples,
        )
    analyse_lp_distance(
        agent, test_samples, fig_dir, CASE_NAME,
        ppc=ppc, all_samples=all_samples, dual_predictor=dual_predictor,
        trainers=trainers, T_DELTA=T_DELTA,
    )

    if RUN_FP and can_run_fp:
        log("── Step 4/4  以全体代理约束运行可行性泵")
        fp_results = run_fp_test(
            ppc, all_samples, dual_predictor, trainers, T_DELTA, test_samples,
            agent=agent, scenario_bank=scenario_bank, fig_dir=fig_dir,
        )
        plot_fp_results(fp_results, fig_dir, CASE_NAME)
        summarize_fp_economicity(
            ppc,
            all_samples,
            fp_results,
            T_DELTA,
            fig_dir,
            CASE_NAME,
        )
    elif RUN_FP:
        fp_results = None
        log(f"── Step 4/4  跳过可行性泵（RUN_FP=True, but {fp_gate_reason}）")
    else:
        fp_results = None
        log("── Step 4/4  跳过可行性泵（RUN_FP=False）")

    summarize_lp_surrogate_fp_totals(
        ppc,
        all_samples,
        dual_predictor,
        trainers,
        T_DELTA,
        test_samples,
        fig_dir,
        CASE_NAME,
        agent=agent,
        fp_results=fp_results,
    )


# ──────────────────────── 主函数 ────────────────────────


def main():
    start_time = time.time()

    print("=" * SECTION_WIDTH)
    print(f"run_test.py | mode={MODE} | case={CASE_NAME}")
    print("=" * SECTION_WIDTH)

    test_samples = TEST_SAMPLES
    sample_range = parse_sample_range(SAMPLE_RANGE)

    result_dir = Path(__file__).parent / 'result' / 'active_set'
    fig_dir    = Path(__file__).parent / 'result' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    log(f"图像输出目录: {fig_dir}")

    # ── 加载 PyPower 案例 ────────────────────────────────
    log(f"加载 PyPower 案例: {CASE_NAME}")
    supported_cases = ['case3', 'case3lite', 'case14', 'case30', 'case39', 'case118']
    if CASE_NAME not in supported_cases:
        print(f"未知案例: {CASE_NAME}，可选 {supported_cases}")
        sys.exit(1)
    ppc = get_case_ppc(CASE_NAME)
    n_units = ppc['gen'].shape[0]
    n_buses = ppc['bus'].shape[0]
    log(f"case summary | units={n_units} | buses={n_buses}")

    # ── 查找数据文件 ─────────────────────────────────────
    if ACTIVE_SETS_FILE is not None:
        data_file = Path(ACTIVE_SETS_FILE)
        if not data_file.is_absolute():
            data_file = Path(__file__).parent / data_file
        if not data_file.exists():
            log(f"错误: 指定的文件不存在: {data_file}")
            sys.exit(1)
        log(f"使用指定文件: {data_file}")
    else:
        data_file = pick_data_file(result_dir, CASE_NAME)
    if data_file is None:
        log(f"错误: 在 {result_dir} 中未找到 {CASE_NAME} 的 JSON 数据文件")
        log("请先运行 ActiveSetLearner 生成数据，或在 result/ 目录下放置数据文件")
        log(f"命名为 active_sets_{CASE_NAME}_*.json 的数据文件后重试")
        sys.exit(1)

    # ── 自动发现模型路径（当顶部配置为 None 时） ────────────
    resolved_model_dir = MODEL_DIR
    resolved_bcd_path = BCD_MODEL_PATH
    # 外部脚本（如 agentic_fp_optimizer）可通过环境变量固定某次训练输出目录，无需改本文件顶部常量
    _env_surr = os.environ.get("RUN_TEST_SURROGATE_MODEL_DIR")
    if _env_surr and _env_surr.strip():
        resolved_model_dir = _env_surr.strip()
    _env_up = os.environ.get("RUN_TEST_UNIT_PREDICTOR_DIR")
    resolved_unit_predictor_dir = UNIT_PREDICTOR_DIR
    if _env_up and _env_up.strip():
        resolved_unit_predictor_dir = _env_up.strip()
    _env_bcd = os.environ.get("RUN_TEST_BCD_MODEL_PATH")
    if _env_bcd and _env_bcd.strip():
        resolved_bcd_path = _env_bcd.strip()
    if resolved_model_dir is None:
        resolved_model_dir = _auto_discover_model_path(
            'result/surrogate_models', f'subproblem_models_{CASE_NAME}_*', 'Surrogate 模型目录')
    if resolved_bcd_path is None:
        resolved_bcd_path = _auto_discover_model_path(
            'result/bcd_models', f'bcd_model_{CASE_NAME}_*.pth', 'BCD 模型文件')

    if MODE in ('surrogate', 'both') and resolved_model_dir is None:
        log(f"错误: 未找到 {CASE_NAME} 的 surrogate 模型目录")
        log("请先运行 run_training.py 生成模型，或在顶部手动设置 MODEL_DIR")
        sys.exit(1)

    # ── 确保 unit_predictor.pth 可用（若外部提供则复制到 model_dir 里） ─────────
    try:
        if resolved_unit_predictor_dir and resolved_model_dir:
            model_dir_abs = str((Path(__file__).parent / resolved_model_dir).resolve())
            target = Path(model_dir_abs) / "unit_predictor.pth"
            if not target.exists():
                src = Path((Path(__file__).parent / str(resolved_unit_predictor_dir)).resolve())
                if src.is_dir():
                    latest = src / "LATEST.txt"
                    if latest.exists():
                        try:
                            p = latest.read_text(encoding="utf-8").strip()
                            if p:
                                src_file = Path(p)
                            else:
                                src_file = src / "unit_predictor.pth"
                        except Exception:
                            src_file = src / "unit_predictor.pth"
                    else:
                        src_file = src / "unit_predictor.pth"
                else:
                    src_file = src
                if src_file.exists():
                    import shutil
                    shutil.copy2(str(src_file), str(target))
                    log(f"unit_predictor injected for test: {src_file} -> {target}")
    except Exception as exc:
        log(f"warning: failed to inject unit_predictor for test: {exc}")
    if MODE in ('bcd', 'both') and resolved_bcd_path is None:
        log(f"错误: 未找到 {CASE_NAME} 的 BCD 模型文件")
        log("请先运行 run_training.py MODE='bcd'/'both' 生成模型，或在顶部手动设置 BCD_MODEL_PATH")
        sys.exit(1)

    # ── 执行模式分支 ─────────────────────────────────────
    try:
        if MODE == 'surrogate':
            full_samples = load_json_data(data_file)
            all_samples = apply_sample_range(full_samples, sample_range)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"sample cap | kept={MAX_SAMPLES} | original={len(all_samples)}")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(
                f"sample summary | T={T_from_data} | eval_samples={len(all_samples)} | "
                f"scenario_bank={len(full_samples)}"
            )

            model_dir = str((Path(__file__).parent / resolved_model_dir).resolve())
            test_surrogate(
                ppc, all_samples, T_DELTA, model_dir, UNIT_IDS, fig_dir,
                scenario_bank=full_samples,
                constraint_generation_strategy=SURROGATE_CONSTRAINT_STRATEGY,
            )

        elif MODE == 'bcd':
            bcd_path = str((Path(__file__).parent / resolved_bcd_path).resolve())
            test_bcd(
                ppc,
                data_file,
                bcd_path,
                MAX_SAMPLES,
                T_DELTA,
                fig_dir,
                sample_range=sample_range,
                test_samples=test_samples,
            )

        elif MODE == 'both':
            full_samples = load_json_data(data_file)
            all_samples = apply_sample_range(full_samples, sample_range)
            if MAX_SAMPLES and len(all_samples) > MAX_SAMPLES:
                log(f"sample cap | kept={MAX_SAMPLES} | original={len(all_samples)}")
                all_samples = all_samples[:MAX_SAMPLES]
            T_from_data = all_samples[0]['pd_data'].shape[1]
            log(
                f"sample summary | T={T_from_data} | eval_samples={len(all_samples)} | "
                f"scenario_bank={len(full_samples)}"
            )

            model_dir  = str((Path(__file__).parent / resolved_model_dir).resolve())
            bcd_path   = str((Path(__file__).parent / resolved_bcd_path).resolve())
            test_both(ppc, data_file, all_samples, T_DELTA,
                      model_dir, bcd_path, MAX_SAMPLES, UNIT_IDS, fig_dir,
                      sample_range=sample_range, test_samples=test_samples,
                      scenario_bank=full_samples,
                      constraint_generation_strategy=SURROGATE_CONSTRAINT_STRATEGY)

        else:
            log(f"未知模式: '{MODE}'，可选 'surrogate' | 'bcd' | 'both'")
            sys.exit(1)

    except Exception as e:
        log(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 汇总 ─────────────────────────────────────────────
    total_time = time.time() - start_time
    log_section(f"完成 | mode={MODE} | elapsed_min={total_time / 60:.1f}")


if __name__ == '__main__':
    main()
