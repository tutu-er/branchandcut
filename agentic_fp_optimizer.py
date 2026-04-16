#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
让 Cursor、Claude Code、其他编程 agent 或 OpenAI 兼容 API 持续迭代改进 feasibility pump。

核心闭环：
1. 运行评估命令，提取当前可行率 / 收敛率 / gap / 耗时
2. 把当前指标、失败日志、历史尝试和目标文件上下文喂给 agent
3. agent 直接修改代码，或通过 API 返回结构化编辑指令
4. 再次运行评估，根据分数决定保留还是回滚
5. 循环多轮，最后保留最佳版本

支持四种 agent 调用方式：
  cursor            内置 Cursor CLI headless agent（推荐）
  claude-cli        内置 Claude Code CLI
  command           自定义外部命令模板
  openai-compatible 任意 OpenAI 兼容 API

示例 1：Cursor CLI 模式（推荐）
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind cursor

示例 2：Claude Code CLI 模式
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind claude-cli

示例 3：自定义外部 agent 命令模式
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind command ^
        --agent-command-template "your-agent-cli --workspace {workspace} --prompt-file {prompt_file}"

示例 4：OpenAI 兼容 API 模式
    set OPENAI_API_KEY=...
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind openai-compatible ^
        --api-base-url "https://api.openai.com/v1/chat/completions" ^
        --api-model "gpt-4.1" ^
        --api-key-env OPENAI_API_KEY

说明：
- 若要用某次训练输出目录（如 `result/surrogate_models/subproblem_models_case3lite_20260414_191709`）做评估与 FP 调参，请传
  `--surrogate-model-path`；脚本会向子进程注入 `RUN_TEST_SURROGATE_MODEL_DIR`，`run_test.py` 将固定加载该目录。
  预置的 `configure_joint()` 也可通过环境变量 `AGENTIC_FP_SURROGATE_DIR`（或 `RUN_TEST_SURROGATE_MODEL_DIR`）指定同一目录。
- cursor 模式下，使用 `cursor agent -p --force` headless 执行，无需额外配置。
- claude-cli 模式下，使用 `claude -p --output-format text` 非交互执行。
- command 模式下，agent 命令模板可使用占位符：{workspace} {prompt_file} {iteration}
- openai-compatible 模式下，模型需返回 JSON 编辑指令，脚本会自动应用。
- 默认只会在 `--target-files` 内保留或回滚修改，避免误伤其他文件。
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加 src/ 到模块搜索路径
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / 'src'))

# 可选导入：Surrogate和BCD模型相关
try:
    from uc_NN_subproblem import SubproblemSurrogateTrainer, ActiveSetReader, load_trained_models
    from uc_NN_BCD import Agent_NN_BCD
    from feasibility_pump import recover_integer_solution
    SURROGATE_AVAILABLE = True
except ImportError as e:
    SURROGATE_AVAILABLE = False
    print(f"[警告] 无法导入Surrogate/BCD模型模块: {e}")


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


# ==================== 辅助函数 ====================

_ENV_ERROR_PATTERNS = [
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "cannot import name",
    "DLL load failed",
]


def _is_env_error(eval_result: "EvalResult") -> bool:
    """判断评估失败是否由环境依赖缺失引起（而非代码 bug）。"""
    combined = eval_result.stdout + eval_result.stderr
    return any(p in combined for p in _ENV_ERROR_PATTERNS)


def _is_meaningful_result(result: "EvalResult") -> bool:
    """判断评估结果是否有意义：返回码为 0 且至少解析出一个业务指标。"""
    m = result.metrics
    return result.return_code == 0 and any(
        v is not None for v in [m.feasible_rate, m.fp_success_rate, m.avg_gap_percent]
    )


def truncate_at_function_boundary(text: str, max_chars: int) -> str:
    """在字符限制前按函数/类边界截断，避免切断函数中间。"""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\ndef ", 0, max_chars)
    if cut == -1:
        cut = text.rfind("\nclass ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut] + "\n# ... [后续内容已截断]\n"


# ==================== Dataclass 定义 ====================

@dataclass
class Metrics:
    feasible_count: Optional[int] = None
    sample_count: Optional[int] = None
    feasible_rate: Optional[float] = None
    fp_success_count: Optional[int] = None
    fp_success_rate: Optional[float] = None
    avg_gap_percent: Optional[float] = None
    max_gap_percent: Optional[float] = None
    min_gap_percent: Optional[float] = None
    avg_fp_time_sec: Optional[float] = None
    avg_milp_time_sec: Optional[float] = None
    test1_pass: Optional[bool] = None


@dataclass
class EvalResult:
    command: str
    return_code: int
    elapsed_sec: float
    stdout: str
    stderr: str
    metrics: Metrics
    score: float


@dataclass
class IterationRecord:
    iteration: int
    agent_kind: str
    prompt_path: str
    agent_output_path: Optional[str]
    accepted: bool
    changed_files: List[str] = field(default_factory=list)
    summary: str = ""
    eval_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)



# ==================== Surrogate/BCD模型加载与调用接口 ====================

class SurrogateModelLoader:
    """
    统一加载和调用Surrogate模型(BCD/V3/联合训练)的接口

    参考run_training.py的实现方式，支持:
    - V3: 三时段代理约束 (uc_NN_subproblem)
    - BCD: BCD主代理 (uc_NN_BCD)
    - Joint: 联合训练 (joint_trainer)
    """

    def __init__(self, config: Config):
        self.config = config
        self._dual_predictor = None
        self._trainers = None
        self._bcd_agent = None
        self._loaded = False

    def load(self) -> bool:
        """
        加载指定类型的Surrogate模型

        Returns:
            bool: 是否成功加载
        """
        if not SURROGATE_AVAILABLE:
            log("[错误] 无法加载Surrogate模块，请确保uc_NN_subproblem和uc_NN_BCD可用")
            return False

        model_type = self.config.surrogate_model_type
        model_path = self.config.surrogate_model_path

        log(f"[SurrogateLoader] 开始加载模型: type={model_type}, path={model_path}")

        try:
            if model_type in ('v3', 'surrogate'):
                return self._load_v3_model(model_path)
            elif model_type == 'bcd':
                return self._load_bcd_model(model_path)
            elif model_type == 'joint':
                return self._load_joint_model(model_path)
            else:
                log(f"[错误] 未知的模型类型: {model_type}")
                return False
        except Exception as e:
            log(f"[错误] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_v3_model(self, model_path: Optional[Path]) -> bool:
        """加载V3三时段代理约束模型"""
        from uc_NN_subproblem import load_trained_models

        # 加载数据文件
        data_file = self._get_data_file()
        if not data_file:
            return False

        # 加载样本数据
        all_samples = self._load_json_data(data_file)
        if not all_samples:
            return False

        # 获取PPC
        ppc = self._get_ppc()
        if not ppc:
            return False

        if model_path and model_path.exists():
            # 从已有模型加载
            log(f"[V3Loader] 从已有模型加载: {model_path}")
            dual_predictor, trainers = load_trained_models(
                ppc, all_samples, self.config.t_delta, self.config.unit_ids,
                str(model_path)
            )
        else:
            log("[V3Loader] 未提供模型路径，需要重新训练")
            return False

        self._dual_predictor = dual_predictor
        self._trainers = trainers
        self._loaded = True
        log(f"[V3Loader] 成功加载: {len(trainers)} 个机组的trainer")
        return True

    def _load_bcd_model(self, model_path: Optional[Path]) -> bool:
        """加载BCD主代理模型"""
        from uc_NN_BCD import Agent_NN_BCD

        # 加载数据文件
        data_file = self._get_data_file()
        if not data_file:
            return False

        # 使用ActiveSetReader加载
        from uc_NN_BCD import load_active_set_from_json
        all_samples = load_active_set_from_json(str(data_file))
        if not all_samples:
            return False

        # 截取样本
        if self.config.max_samples and len(all_samples) > self.config.max_samples:
            all_samples = all_samples[:self.config.max_samples]

        # 获取PPC
        ppc = self._get_ppc()
        if not ppc:
            return False

        # 创建Agent
        agent = Agent_NN_BCD(
            ppc, all_samples, self.config.t_delta,
            lambda_init_strategy='lp_relaxation',
        )

        if model_path and model_path.exists():
            log(f"[BCDLoader] 从已有模型加载: {model_path}")
            agent.load_model_parameters(str(model_path))
        else:
            log("[BCDLoader] 未提供模型路径，需要重新训练")
            return False

        self._bcd_agent = agent
        self._loaded = True
        log("[BCDLoader] 成功加载BCD模型")
        return True

    def _load_joint_model(
        self,
        bcd_model_path: Optional[Path] = None,
        surrogate_model_path: Optional[Path] = None
    ) -> bool:
        """加载联合训练模型（V3 + BCD）

        Args:
            bcd_model_path: BCD模型文件路径 (.pth)
            surrogate_model_path: Surrogate模型目录路径
        """
        log("[JointLoader] 联合模型加载: 同时加载V3和BCD组件")

        # 使用传入的路径或配置中的路径
        bcd_path = bcd_model_path or getattr(self.config, 'bcd_model_path', None)
        surr_path = surrogate_model_path or getattr(self.config, 'surrogate_model_path', None)

        if not bcd_path:
            log("[JointLoader] 错误: 未指定BCD模型路径")
            return False
        if not surr_path:
            log("[JointLoader] 错误: 未指定Surrogate模型路径")
            return False

        # 先加载V3 (Surrogate)
        log(f"[JointLoader] 加载V3 (Surrogate) 组件: {surr_path}")
        v3_success = self._load_v3_model(surr_path)
        if not v3_success:
            log("[JointLoader] V3组件加载失败")
            return False

        # 再加载BCD
        log(f"[JointLoader] 加载BCD组件: {bcd_path}")
        bcd_success = self._load_bcd_model(bcd_path)
        if not bcd_success:
            log("[JointLoader] BCD组件加载失败")
            return False

        log("[JointLoader] 联合模型加载成功: V3 + BCD 同时可用")
        return True

    def _get_data_file(self) -> Optional[Path]:
        """获取数据文件路径"""
        if self.config.active_sets_file:
            path = Path(self.config.active_sets_file)
            if path.exists():
                return path
            log(f"[错误] 指定的数据文件不存在: {path}")
            return None

        # 自动查找数据文件
        data_dir = Path(__file__).parent / 'result' / 'active_set'
        if not data_dir.exists():
            log(f"[错误] 数据目录不存在: {data_dir}")
            return None

        # 查找匹配case_name的最新文件
        pattern = f"active_sets_{self.config.case_name}_*.json"
        files = sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if files:
            log(f"[DataLoader] 找到数据文件: {files[0].name}")
            return files[0]

        # 回退到任何active_sets文件
        files = sorted(data_dir.glob("active_sets_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            log(f"[DataLoader] 使用回退数据文件: {files[0].name}")
            return files[0]

        log(f"[错误] 在 {data_dir} 中未找到数据文件")
        return None

    def _load_json_data(self, data_file: Path) -> Optional[List[Dict]]:
        """加载JSON数据文件"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_samples = data.get('all_samples', [])
            if not all_samples:
                log(f"[错误] 数据文件中没有样本: {data_file}")
                return None
            log(f"[DataLoader] 加载了 {len(all_samples)} 个样本")
            return all_samples
        except Exception as e:
            log(f"[错误] 加载数据文件失败: {e}")
            return None

    def _get_ppc(self) -> Optional[Dict]:
        """获取PyPower案例数据"""
        try:
            from case_registry import get_case_ppc
            ppc = get_case_ppc(self.config.case_name)
            log(f"[PPCLoader] 加载案例: {self.config.case_name}, "
                f"{ppc['gen'].shape[0]}机组, {ppc['bus'].shape[0]}节点")
            return ppc
        except Exception as e:
            log(f"[错误] 加载PPC失败: {e}")
            return None

    def run_fp_with_model(
        self,
        sample: Dict,
        ppc: Dict,
        verbose: bool = True
    ) -> Tuple[Optional[Any], bool]:
        """
        使用加载的模型运行Feasibility Pump

        Args:
            sample: 样本数据（包含pd_data等）
            ppc: PyPower案例数据
            verbose: 是否打印详细信息

        Returns:
            (x_result, success): 结果解和是否成功
        """
        if not self._loaded:
            log("[错误] 模型未加载，请先调用load()")
            return None, False

        model_type = self.config.surrogate_model_type

        try:
            if model_type in ('v3', 'surrogate', 'joint'):
                # 使用V3模型的recover_integer_solution
                if self._dual_predictor is None or self._trainers is None:
                    log("[错误] V3模型未正确加载")
                    return None, False

                from feasibility_pump import recover_integer_solution
                x_result, success = recover_integer_solution(
                    sample, self._trainers, self._dual_predictor,
                    ppc, self.config.t_delta,
                    verbose=verbose,
                )
                return x_result, success

            elif model_type == 'bcd':
                # BCD模型需要特殊处理，通过Agent的iter方法获取解
                log("[BCD] BCD模型通过iter_with_pg_block获取解")
                if self._bcd_agent is None:
                    log("[错误] BCD模型未正确加载")
                    return None, False

                # 获取当前样本的解
                # 注意：BCD模型的样本索引需要匹配
                log("[BCD] 需要从agent.x中提取对应样本的解")
                # 这里需要进一步实现样本匹配逻辑
                return None, False

            else:
                log(f"[错误] 未知的模型类型: {model_type}")
                return None, False

        except Exception as e:
            log(f"[错误] 运行FP失败: {e}")
            import traceback
            traceback.print_exc()
            return None, False


@dataclass
class Config:
    workspace: Path
    results_dir: Path
    target_files: List[Path]
    eval_command: str
    agent_kind: str
    agent_command_template: Optional[str]
    api_base_url: Optional[str]
    api_model: Optional[str]
    api_key_env: Optional[str]
    api_temperature: float
    api_max_tokens: int
    eval_timeout_sec: int
    agent_timeout_sec: int
    max_iters: int
    stagnation_limit: int
    min_score_improvement: float
    max_context_chars_per_file: int
    max_eval_output_chars: int
    max_history_items: int
    goal: str
    system_prompt_file: Optional[Path]
    accept_regressions: bool

    # ---- Cursor / Claude CLI / Codex CLI 配置 ----
    cursor_model: Optional[str] = None
    cursor_interactive: bool = False
    claude_cli_model: Optional[str] = None
    claude_cli_allowed_tools: str = "Edit,Write,Read"
    codex_cli_model: Optional[str] = None

    # === 新增：Surrogate模型与多策略FP优化器配置 ===

    # ---- Surrogate模型配置 ----
    # 使用的Surrogate模型类型: 'v3'(三时段代理约束), 'bcd'(BCD主代理), 'joint'(联合训练)
    surrogate_model_type: str = "v3"
    # Surrogate模型路径（训练好的模型目录或文件，用于V3/Joint模式）
    surrogate_model_path: Optional[Path] = None
    # BCD模型路径（训练好的模型文件，用于BCD/Joint模式）
    bcd_model_path: Optional[Path] = None
    # 案例名称: 'case3', 'case3lite', 'case14', 'case30', 'case39', 'case118'
    case_name: str = "case3lite"
    # 时间间隔（小时）
    t_delta: float = 1.0
    # 时段数
    time_periods: int = 24
    # 机组ID列表（None表示全部机组）
    unit_ids: Optional[List[int]] = None
    # 最大样本数（None表示全部）
    max_samples: Optional[int] = None
    # 数据文件路径（JSON格式，包含all_samples）
    active_sets_file: Optional[Path] = None

    # ---- 多策略FP优化器配置 ----
    # 可用的FP策略列表
    fp_strategy_pool: List[str] = field(default_factory=lambda: [
        "legacy",           # 旧版FP (feasibility_pump.py)
        "case3lite",        # case3lite特调FP (feasibility_pump_case3lite.py)
        "guided",           # 增强型定向FP (fp_guided_recovery.py)
    ])
    # 默认使用的FP策略（可按场景动态选择）
    default_fp_strategy: str = "guided"
    # 是否启用策略自适应选择（根据代理模型特征自动选择最优策略）
    enable_strategy_adaptation: bool = True
    # FP策略性能跟踪数据库路径（用于自适应策略选择）
    fp_strategy_perf_db: Optional[Path] = None

    # ---- 代理模型定向适配配置 ----
    # 代理模型特征配置文件路径（JSON），描述代理模型的系统性偏差模式
    surrogate_profile_path: Optional[Path] = None
    # 是否启用代理模型偏差检测与自适应修正
    enable_surrogate_adaptation: bool = False
    # 启停时序偏差修正窗口大小（用于代理模型适配）
    startup_shift_window: int = 2

    # ---- 物理信息增强配置 ----
    # 是否启用物理信息增强（历史解 + 邻近场景）
    enable_physics_informed_recovery: bool = False
    # 邻近场景解库路径（用于物理信息增强）
    neighbor_solutions_path: Optional[Path] = None
    # 历史最优解存档路径
    historical_best_path: Optional[Path] = None

    # ---- 评估模式 ----
    # 传给 run_test.py 的 RUN_TEST_MODE 环境变量（both/surrogate/bcd）
    eval_mode: str = "both"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def snapshot_files(files: List[Path]) -> Dict[str, Optional[str]]:
    snapshot: Dict[str, Optional[str]] = {}
    for path in files:
        key = str(path)
        if path.exists():
            snapshot[key] = read_text(path)
        else:
            snapshot[key] = None
    return snapshot


def restore_snapshot(snapshot: Dict[str, Optional[str]]) -> None:
    for path_str, content in snapshot.items():
        path = Path(path_str)
        if content is None:
            if path.exists():
                path.unlink()
            continue
        write_text(path, content)


def list_changed_files(
    before: Dict[str, Optional[str]],
    after: Dict[str, Optional[str]],
) -> List[str]:
    changed: List[str] = []
    for key in before:
        if before.get(key) != after.get(key):
            changed.append(key)
    return changed


def build_diff_snippet(
    before: Dict[str, Optional[str]],
    after: Dict[str, Optional[str]],
    limit_chars: int = 12000,
) -> str:
    chunks: List[str] = []
    used = 0
    for path_str in before:
        old = before.get(path_str) or ""
        new = after.get(path_str) or ""
        if old == new:
            continue
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"{path_str} (before)",
                tofile=f"{path_str} (after)",
                n=3,
            )
        )
        if not diff:
            continue
        remain = limit_chars - used
        if remain <= 0:
            break
        if len(diff) > remain:
            diff = diff[:remain] + "\n... [diff truncated]\n"
        chunks.append(diff)
        used += len(diff)
    return "\n".join(chunks) if chunks else "(no diff)"


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]\n"


def tail_text(text: str, max_chars: int = 4000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def parse_metrics(output: str) -> Metrics:
    metrics = Metrics()

    m = re.search(r"结果:\s*(\d+)/(\d+)\s*通过", output)
    if m:
        metrics.test1_pass = m.group(1) == m.group(2)

    # ---- 旧格式 ----
    m = re.search(r"可行率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", output)
    if m:
        metrics.feasible_count = int(m.group(1))
        metrics.sample_count = int(m.group(2))
        metrics.feasible_rate = float(m.group(3)) / 100.0

    # ---- run_test.py 输出格式 ----
    # "可行性泵完成: 2/2 样本找到可行解"
    if metrics.feasible_rate is None:
        m = re.search(r"可行性泵完成:\s*(\d+)/(\d+)\s*样本找到可行解", output)
        if m:
            metrics.feasible_count = int(m.group(1))
            metrics.sample_count = int(m.group(2))
            metrics.feasible_rate = metrics.feasible_count / max(metrics.sample_count, 1)
            metrics.fp_success_count = metrics.feasible_count
            metrics.fp_success_rate = metrics.feasible_rate

    m = re.search(r"FP 收敛率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", output)
    if m:
        metrics.fp_success_count = int(m.group(1))
        metrics.fp_success_rate = float(m.group(3)) / 100.0

    # ---- 旧格式 gap ----
    m = re.search(
        r"Gap \(vs MILP\):\s*平均=([+\-]?\d+(?:\.\d+)?)%,\s*最大=([+\-]?\d+(?:\.\d+)?)%,\s*最小=([+\-]?\d+(?:\.\d+)?)%",
        output,
    )
    if m:
        metrics.avg_gap_percent = float(m.group(1))
        metrics.max_gap_percent = float(m.group(2))
        metrics.min_gap_percent = float(m.group(3))

    # ---- run_test.py 格式: "FP rel gap: mean=0.38%, median=0.38%, max=0.76%" ----
    if metrics.avg_gap_percent is None:
        m = re.search(
            r"FP rel gap:\s*mean=([\d.]+)%,\s*median=([\d.]+)%,\s*max=([\d.]+)%",
            output,
        )
        if m:
            metrics.avg_gap_percent = float(m.group(1))
            metrics.max_gap_percent = float(m.group(3))
            metrics.min_gap_percent = 0.0

    # ---- 耗时 ----
    m = re.search(r"FP 平均耗时:\s*([\d.]+)s", output)
    if m:
        metrics.avg_fp_time_sec = float(m.group(1))

    # "完成！模式 both，耗时 4.0 分钟"
    if metrics.avg_fp_time_sec is None:
        m = re.search(r"耗时\s*([\d.]+)\s*分钟", output)
        if m:
            metrics.avg_fp_time_sec = float(m.group(1)) * 60.0

    m = re.search(r"MILP 平均耗时:\s*([\d.]+)s", output)
    if m:
        metrics.avg_milp_time_sec = float(m.group(1))

    return metrics


def score_metrics(metrics: Metrics, return_code: int) -> float:
    score = 0.0

    if return_code != 0:
        score -= 1000.0

    if metrics.test1_pass is False:
        score -= 200.0
    elif metrics.test1_pass is True:
        score += 20.0

    if metrics.feasible_rate is not None:
        score += 600.0 * metrics.feasible_rate

    if metrics.fp_success_rate is not None:
        score += 300.0 * metrics.fp_success_rate

    if metrics.avg_gap_percent is not None:
        score -= 6.0 * max(metrics.avg_gap_percent, 0.0)
        score -= 1.5 * abs(min(metrics.avg_gap_percent, 0.0))

    if metrics.avg_fp_time_sec is not None:
        score -= 0.25 * metrics.avg_fp_time_sec

    return score


def _path_for_run_test_env(workspace: Path, p: Optional[Path]) -> Optional[str]:
    """将绝对路径转为相对 workspace 的路径字符串，供 run_test 环境变量使用。"""
    if p is None:
        return None
    try:
        ws = workspace.resolve()
        rp = p.resolve().relative_to(ws)
        return str(rp).replace("\\", "/")
    except ValueError:
        return str(p.resolve())


def run_command(
    command: str,
    cwd: Path,
    timeout_sec: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> tuple[int, str, str, float]:
    start = time.time()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            shell=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        stderr = (
            stderr
            + f"\n[timeout]\nCommand exceeded timeout of {timeout_sec} seconds.\n"
        )
        return 124, stdout, stderr, elapsed
    elapsed = time.time() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def evaluate(config: Config, iteration_dir: Path, label: str) -> EvalResult:
    log(f"运行评估: {label}")
    extra_env: Dict[str, str] = {}
    # 自动化评估不需要绘图；禁用 run_test.py 的绘图可避免空数据触发 matplotlib 崩溃
    extra_env["RUN_TEST_DISABLE_PLOTS"] = "1"
    if config.surrogate_model_path:
        s = _path_for_run_test_env(config.workspace, config.surrogate_model_path)
        extra_env["RUN_TEST_SURROGATE_MODEL_DIR"] = s
        log(f"评估环境 RUN_TEST_SURROGATE_MODEL_DIR={s}")
    if config.bcd_model_path:
        b = _path_for_run_test_env(config.workspace, config.bcd_model_path)
        extra_env["RUN_TEST_BCD_MODEL_PATH"] = b
        log(f"评估环境 RUN_TEST_BCD_MODEL_PATH={b}")
    if config.eval_mode and config.eval_mode != "both":
        extra_env["RUN_TEST_MODE"] = config.eval_mode
        log(f"评估环境 RUN_TEST_MODE={config.eval_mode}")
    rc, stdout, stderr, elapsed = run_command(
        config.eval_command,
        config.workspace,
        timeout_sec=config.eval_timeout_sec,
        extra_env=extra_env if extra_env else None,
    )
    output = stdout + ("\n[stderr]\n" + stderr if stderr else "")
    metrics = parse_metrics(output)
    score = score_metrics(metrics, rc)

    write_text(iteration_dir / f"{label}_stdout.txt", stdout)
    write_text(iteration_dir / f"{label}_stderr.txt", stderr)
    write_text(
        iteration_dir / f"{label}_metrics.json",
        json.dumps(
            {
                "command": config.eval_command,
                "return_code": rc,
                "elapsed_sec": elapsed,
                "score": score,
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    return EvalResult(
        command=config.eval_command,
        return_code=rc,
        elapsed_sec=elapsed,
        stdout=stdout,
        stderr=stderr,
        metrics=metrics,
        score=score,
    )


def build_context_block(config: Config, max_chars_per_file: Optional[int] = None) -> str:
    chunks: List[str] = []
    limit = config.max_context_chars_per_file if max_chars_per_file is None else max_chars_per_file
    for path in config.target_files:
        rel = path.relative_to(config.workspace)
        content = read_text(path) if path.exists() else ""
        chunks.append(
            f"### FILE: {rel}\n"
            f"```python\n{truncate_at_function_boundary(content, limit)}\n```\n"
        )
    return "\n".join(chunks)


def build_history_block(history: List[IterationRecord], limit: int) -> str:
    if not history:
        return "暂无历史尝试。"

    rows: List[str] = []
    for item in history[-limit:]:
        status = "accepted" if item.accepted else "reverted"
        score = "n/a" if item.eval_score is None else f"{item.eval_score:.3f}"
        rows.append(
            f"- iter={item.iteration}, status={status}, score={score}, "
            f"files={item.changed_files or ['<none>']}, summary={item.summary or '<empty>'}"
        )
    return "\n".join(rows)


def build_surrogate_profile_block(config: Config) -> str:
    """构建代理模型特征配置块。仅在启用适配且提供了有效的 profile 文件时生成内容。"""
    if not config.enable_surrogate_adaptation:
        return ""
    # 无 profile 文件时整块省略，避免输出无意义的空占位符
    if config.surrogate_profile_path is None:
        return ""
    if not config.surrogate_profile_path.exists():
        return f"[警告: 代理模型配置文件不存在: {config.surrogate_profile_path}]"
    try:
        profile = json.loads(read_text(config.surrogate_profile_path))
        patterns = profile.get("bias_patterns", [])
        patterns_str = "\n  - ".join([""] + patterns) if patterns else "未定义"
        surrogate_name = profile.get('name', 'unknown')
        return (
            "=== 代理模型定向适配配置 ===\n"
            f"代理模型名称: {surrogate_name}\n"
            f"已知系统性偏差模式:{patterns_str}\n"
            f"启停时序偏差修正窗口: {config.startup_shift_window} 个时段\n"
            "\n"
            "定向优化要求:\n"
            "1. 在 `run_feasibility_pump()` 中添加对代理模型特定偏差的检测逻辑\n"
            "2. 针对启停时序偏移，实现 `shift_correction_heuristic()` 启发式调整\n"
            "3. 在 `identify_trusted_mask()` 中降低代理模型高偏差区域的可信度阈值\n"
            "4. 实现 `adapt_priority_by_surrogate_bias()` 动态调整组合优先级\n"
            "==="
        )
    except Exception as e:
        return f"[错误: 无法解析代理模型配置文件: {e}]"


def build_physics_informed_block(config: Config) -> str:
    """构建物理信息增强配置块。"""
    if not config.enable_physics_informed_recovery:
        return ""
    neighbor_info = ""
    historical_info = ""
    if config.neighbor_solutions_path and config.neighbor_solutions_path.exists():
        neighbor_info = f"邻近场景解库: {config.neighbor_solutions_path}"
    else:
        neighbor_info = "[警告: 邻近场景解库未配置或不存在]"
    if config.historical_best_path and config.historical_best_path.exists():
        historical_info = f"历史最优解存档: {config.historical_best_path}"
    else:
        historical_info = "[警告: 历史最优解存档未配置或不存在]"
    return textwrap.dedent(
        f"""
        === 物理信息增强解恢复配置 ===
        {neighbor_info}
        {historical_info}

        物理信息增强要求:
        1. 实现 `physics_informed_projection()` 函数，融合以下信息源:
           - 历史最优解的时序模式特征
           - 邻近场景解的局部连续性约束
           - 机组启停物理约束（最小开停机时间）

        2. 实现 `temporal_coherence_heuristic()` 时序一致性启发式:
           - 利用邻近时段的决策相关性平滑当前解
           - 识别并修复违反物理约束的跳变

        3. 在 `collect_integer_solutions()` 中增强候选解生成:
           - 从历史解库中检索相似场景的成功模式
           - 结合邻近场景解进行交叉组合

        4. 实现 `proximity_guided_rounding()` 邻近引导舍入:
           - 利用邻近场景解的整数模式指导当前舍入方向
           - 优先选择与历史最优解兼容的舍入路径
        ===
        """
    )


def build_multi_strategy_fp_block(config: Config) -> str:
    """构建多策略FP优化器配置块。"""
    if not config.fp_strategy_pool:
        return ""

    strategy_details = []
    for strategy in config.fp_strategy_pool:
        if strategy == "legacy":
            strategy_details.append("  - legacy: 旧版FP (feasibility_pump.py:run_feasibility_pump)")
        elif strategy == "case3lite":
            strategy_details.append("  - case3lite: case3lite特调FP (feasibility_pump_case3lite.py)")
        elif strategy == "guided":
            strategy_details.append("  - guided: 增强型定向FP (fp_guided_recovery.py:GuidedFeasibilityPump)")
        else:
            strategy_details.append(f"  - {strategy}: 自定义策略")

    # 根据Surrogate模型类型推荐最优策略
    recommended_strategy = config.default_fp_strategy
    if config.surrogate_model_type == "v3":
        recommended_strategy = "guided"  # V3模型适合定向FP
    elif config.surrogate_model_type == "bcd":
        recommended_strategy = "legacy"  # BCD模型适合标准FP
    elif config.surrogate_model_type == "joint":
        recommended_strategy = "guided"  # 联合训练模型适合定向FP

    adaptation_info = ""
    if config.enable_strategy_adaptation:
        adaptation_info = f"""
策略自适应选择: 启用
自适应选择逻辑:
1. 基于代理模型特征 (surrogate_profile) 识别偏差模式
2. 查询策略性能数据库 (fp_strategy_perf_db) 获取历史表现
3. 根据当前Surrogate模型类型 '{config.surrogate_model_type}' 推荐策略: {recommended_strategy}
4. 动态选择预期最优策略:
   - V3 Surrogate (三时段代理约束) → 优先选择 guided (支持shift_correction)
   - BCD Surrogate → 优先选择 legacy (标准BCD兼容)
   - 联合训练模型 → 优先选择 guided 或 case3lite
   - 延迟启动/停机偏差为主 → 优先选择 guided (支持shift_correction)
   - 复杂约束场景 → 优先选择 case3lite (特调启发式)
   - 通用场景 → 选择 legacy 或 guided
默认策略: {config.default_fp_strategy}"""
    else:
        adaptation_info = f"""策略自适应选择: 禁用
固定使用策略: {config.default_fp_strategy}
基于Surrogate模型类型 '{config.surrogate_model_type}' 的推荐策略: {recommended_strategy}"""

    return textwrap.dedent(
        f"""
        === 多策略FP优化器配置 ===
        可用策略池:
        {chr(10).join(strategy_details)}

        {adaptation_info}

        Surrogate模型类型与FP策略匹配指南:
        - 'v3' (三时段代理约束): 推荐 'guided' 策略，利用SurrogateDiagnostics进行偏差修正
        - 'bcd' (BCD主代理): 推荐 'legacy' 策略，标准BCD流程兼容
        - 'joint' (联合训练): 推荐 'guided' 或 'case3lite'，利用联合训练的丰富特征

        多策略优化要求:
        1. 实现 `select_fp_strategy()` 策略选择函数:
           - 输入: Surrogate模型类型、代理模型特征、历史性能数据
           - 输出: 选择的策略名称
           - 逻辑: 基于Surrogate类型和场景特征进行智能匹配
        2. 实现各策略的统一调用接口 `run_fp_with_strategy()`:
           - legacy: 调用 feasibility_pump.run_feasibility_pump()
           - case3lite: 调用 feasibility_pump_case3lite.fp_case3lite()
           - guided: 调用 fp_guided_recovery.GuidedFeasibilityPump.run_guided_fp()
           - 统一接口接收相同的参数（场景数据、Surrogate模型、配置）
        3. 实现 `update_strategy_performance()` 性能跟踪:
           - 记录每个策略在各场景下的表现（可行率、gap、耗时）
           - 按Surrogate模型类型分类统计
           - 用于自适应策略选择
        4. 在 fp_guided_recovery.py 中完善定向FP:
           - 实现 SurrogateDiagnostics 偏差分析
           - 实现 PhysicsGuidedRecovery 物理约束修正
           - 实现 SolutionMemoryBank 历史解管理
           - 针对V3 Surrogate模型优化：利用三时段代理约束特征进行偏差诊断
        ===
        """
    )


def build_compact_strategy_block(config: Config) -> str:
    if not config.fp_strategy_pool:
        return ""

    recommended_strategy = config.default_fp_strategy
    if config.surrogate_model_type == "v3":
        recommended_strategy = "guided"
    elif config.surrogate_model_type == "bcd":
        recommended_strategy = "legacy"
    elif config.surrogate_model_type == "joint":
        recommended_strategy = "guided"

    strategy_details = ", ".join(config.fp_strategy_pool)
    adaptation_mode = "enabled" if config.enable_strategy_adaptation else "disabled"
    return (
        "=== FP 多策略指导 ===\n"
        f"可用策略: {strategy_details}\n"
        f"策略自适应: {adaptation_mode}\n"
        f"Surrogate 模型类型: {config.surrogate_model_type}\n"
        f"推荐默认策略: {recommended_strategy}\n"
        "保持轻量：优先调整阈值、添加简单回退逻辑，在现有代码路径内路由。\n"
        "除非目标文件已有对应接入点，否则不要新建多策略框架。\n"
        "==="
    )


def has_recent_agent_timeout(history: List[IterationRecord], limit: int = 1) -> bool:
    if limit <= 0:
        return False
    return any("exit_code=124" in (item.summary or "") for item in history[-limit:])


def build_prompt(
    config: Config,
    baseline: EvalResult,
    latest: EvalResult,
    best: EvalResult,
    history: List[IterationRecord],
) -> str:
    compact_prompt = config.agent_kind == "claude-cli" and has_recent_agent_timeout(history, limit=1)
    system_extra = ""
    if config.system_prompt_file and config.system_prompt_file.exists():
        system_extra = read_text(config.system_prompt_file).strip()

    latest_output = truncate(
        latest.stdout + ("\n[stderr]\n" + latest.stderr if latest.stderr else ""),
        min(config.max_eval_output_chars, 4000) if compact_prompt else config.max_eval_output_chars,
    )

    # 构建新增的功能模块提示
    surrogate_block = build_surrogate_profile_block(config)
    physics_block = build_physics_informed_block(config)
    multi_strategy_block = build_compact_strategy_block(config)
    context_limit = min(config.max_context_chars_per_file, 8000) if compact_prompt else config.max_context_chars_per_file
    history_limit = min(config.max_history_items, 3) if compact_prompt else config.max_history_items
    timeout_hint = ""
    if compact_prompt:
        timeout_hint = (
            "\n补充约束：上一轮 claude-cli 已超时。"
            " 这一轮只允许做一个小改动，先读必要片段，尽快直接编辑，不要长时间规划或运行耗时命令。\n"
        )

    # 构建Surrogate模型配置说明（逐行 join，避免 textwrap.dedent 嵌入时缩进错位）
    surrogate_config = "\n".join([
        "=== Surrogate 模型配置 ===",
        f"模型类型: {config.surrogate_model_type}",
        f"案例名称: {config.case_name}",
        f"时间间隔: {config.t_delta} 小时",
        f"时段数: {config.time_periods}",
        f"机组IDs: {config.unit_ids if config.unit_ids else '全部'}",
        f"最大样本数: {config.max_samples if config.max_samples else '全部'}",
        f"数据文件: {config.active_sets_file if config.active_sets_file else '自动查找'}",
        f"模型路径: {config.surrogate_model_path if config.surrogate_model_path else '训练新模型'}",
        "===",
    ])

    # 构建核心优化目标说明
    optimization_focus = ""
    if config.enable_surrogate_adaptation and config.enable_physics_informed_recovery:
        optimization_focus = "\n【核心优化方向】\n1. 代理模型定向适配: 针对特定代理模型输出特征实现偏差感知恢复\n2. 物理信息增强: 融合历史解和邻近场景信息的约束感知恢复\n3. 多策略FP: 基于场景特征自适应选择最优FP策略 (legacy/case3lite/guided)\n"
    elif config.enable_surrogate_adaptation:
        optimization_focus = "\n【核心优化方向】\n代理模型定向适配: 针对特定代理模型输出特征实现偏差感知恢复\n"
    elif config.enable_physics_informed_recovery:
        optimization_focus = "\n【核心优化方向】\n物理信息增强: 融合历史解和邻近场景信息的约束感知恢复\n"
    else:
        optimization_focus = "\n【核心优化方向】\n多策略FP: 基于场景特征自适应选择最优FP策略 (legacy/case3lite/guided)\n"

    # Fix 5: 评估失败时注入诊断说明
    eval_status_note = ""
    if latest.return_code != 0:
        if _is_env_error(latest):
            eval_status_note = (
                "\n[注意] 评估命令因环境依赖缺失失败（见下方 stderr），"
                "这是运行环境问题，无法通过修改目标文件解决。"
                "如果你确认代码没有引入新的 import，请保持代码不变并输出说明。\n"
            )
        else:
            eval_status_note = (
                "\n[注意] 评估命令以非零状态退出，请检查是否是代码 bug 导致，并尝试修复。\n"
            )

    # Fix 6: 三个指标块相同时去重，只展示一块
    _baseline_metrics_json = json.dumps(asdict(baseline.metrics), ensure_ascii=False, indent=2)
    _latest_metrics_json = json.dumps(asdict(latest.metrics), ensure_ascii=False, indent=2)
    _best_metrics_json = json.dumps(asdict(best.metrics), ensure_ascii=False, indent=2)
    _latest_same_as_baseline = (
        latest.score == baseline.score and _latest_metrics_json == _baseline_metrics_json
    )
    _best_same_as_baseline = (
        best.score == baseline.score and _best_metrics_json == _baseline_metrics_json
    )
    if _latest_same_as_baseline and _best_same_as_baseline:
        metrics_block = (
            f"当前分数（基线/当前/最佳均相同）：\n"
            f"{baseline.score:.3f}\n"
            f"当前指标：\n{_baseline_metrics_json}"
        )
    else:
        metrics_block = (
            f"基线分数：\n{baseline.score:.3f}\n"
            f"基线指标：\n{_baseline_metrics_json}"
        )
        if not _latest_same_as_baseline:
            metrics_block += (
                f"\n\n当前工作树分数：\n{latest.score:.3f}\n"
                f"当前指标：\n{_latest_metrics_json}"
            )
        if not _best_same_as_baseline:
            metrics_block += (
                f"\n\n当前最佳分数：\n{best.score:.3f}\n"
                f"当前最佳指标：\n{_best_metrics_json}"
            )

    return textwrap.dedent(
        f"""
        你是一个专门优化 Unit Commitment feasibility pump 的代码 agent。

        目标：
        {config.goal}

        工作区：
        {config.workspace}

        必须遵守：
        1. 只修改允许的目标文件。
        2. 每轮只做 1-2 个彼此相关的启发式改动，避免大范围重构。
        3. 优先改 `run_feasibility_pump()`、`identify_trusted_mask()`、`collect_integer_solutions()` 等启发式逻辑。
        4. 不要引入新的第三方依赖。
        5. 要兼顾可行率、FP 收敛率、平均 gap、平均耗时。
        6. 如果某个改动风险较高，请采用更保守版本。
        7. 修改后请输出简洁总结：改了什么、预期为什么会更好。
        8. 【重要】如启用了代理模型适配、物理信息增强或多策略FP，必须实现对应的专用函数（见下方模块说明）。

        {optimization_focus}
        {timeout_hint}
        {surrogate_config}
        {surrogate_block}
        {physics_block}
        {multi_strategy_block}

        允许修改的文件：
        {chr(10).join(f"- {p.relative_to(config.workspace)}" for p in config.target_files)}

        {metrics_block}
        {eval_status_note}
        最近评估输出：
        ```text
        {latest_output}
        ```

        历史尝试摘要：
        {build_history_block(history, history_limit)}

        目标文件上下文：
        {build_context_block(config, max_chars_per_file=context_limit)}

        {system_extra}
        """
    ).strip() + "\n"


def call_openai_compatible_api(config: Config, prompt: str) -> str:
    if not config.api_base_url or not config.api_model or not config.api_key_env:
        raise ValueError("openai-compatible 模式要求提供 api_base_url / api_model / api_key_env")

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"环境变量 {config.api_key_env} 未设置")

    payload = {
        "model": config.api_model,
        "temperature": config.api_temperature,
        "max_tokens": config.api_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个代码修改代理。"
                    "你必须返回 JSON，对象格式为："
                    "{"
                    "\"summary\": \"一句话说明\","
                    "\"operations\": ["
                    "  {\"type\": \"replace\", \"path\": \"src/feasibility_pump.py\", "
                    "\"old_text\": \"精确旧文本\", \"new_text\": \"精确新文本\", \"count\": 1},"
                    "  {\"type\": \"write\", \"path\": \"foo.py\", \"content\": \"完整文件内容\"}"
                    "]"
                    "}。"
                    "只能返回 JSON，不能带 markdown。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.api_base_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 请求失败: {exc.code} {body}") from exc


def extract_api_message(raw_json_text: str) -> str:
    payload = json.loads(raw_json_text)
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("API 响应缺少 choices")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                parts.append(item.get("text", ""))
        if parts:
            return "".join(parts)

    raise RuntimeError("API 响应中未找到可解析的文本 content")


def apply_structured_operations(
    config: Config,
    response_text: str,
) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    response = json.loads(cleaned)
    summary = str(response.get("summary", "")).strip()
    operations = response.get("operations", [])
    if not isinstance(operations, list):
        raise RuntimeError("operations 必须是列表")

    allowed = {str(path.resolve()) for path in config.target_files}

    for op in operations:
        if not isinstance(op, dict):
            raise RuntimeError("operation 必须是对象")
        op_type = op.get("type")
        rel_path = op.get("path")
        if not isinstance(rel_path, str):
            raise RuntimeError("operation.path 必须是字符串")
        abs_path = (config.workspace / rel_path).resolve()
        if str(abs_path) not in allowed:
            raise RuntimeError(f"模型尝试修改未授权文件: {rel_path}")

        if op_type == "write":
            content = op.get("content")
            if not isinstance(content, str):
                raise RuntimeError("write 操作缺少 content")
            write_text(abs_path, content)
            continue

        if op_type == "replace":
            old_text = op.get("old_text")
            new_text = op.get("new_text")
            expected_count = int(op.get("count", 1))
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise RuntimeError("replace 操作要求 old_text/new_text 为字符串")
            original = read_text(abs_path)
            count = original.count(old_text)
            if count != expected_count:
                raise RuntimeError(
                    f"{rel_path} 替换计数不匹配，预期 {expected_count}，实际 {count}"
                )
            updated = original.replace(old_text, new_text)
            write_text(abs_path, updated)
            continue

        raise RuntimeError(f"不支持的操作类型: {op_type}")

    return summary


def run_command_agent(
    config: Config,
    prompt_file: Path,
    iteration: int,
) -> tuple[str, int, str, str]:
    if not config.agent_command_template:
        raise ValueError("command 模式要求提供 --agent-command-template")

    command = config.agent_command_template.format(
        workspace=str(config.workspace),
        prompt_file=str(prompt_file),
        iteration=iteration,
    )
    rc, stdout, stderr, _elapsed = run_command(
        command,
        config.workspace,
        timeout_sec=config.agent_timeout_sec,
    )
    return command, rc, stdout, stderr


def _find_cursor_exe() -> Optional[str]:
    """尝试在常见安装路径中找到 cursor 可执行文件（Windows）。"""
    import shutil
    found = shutil.which("cursor")
    if found:
        return found
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        for sub in ("Programs/cursor/resources/app/bin/cursor.cmd",
                     "Programs/Cursor/resources/app/bin/cursor.cmd"):
            p = Path(local) / sub
            if p.exists():
                return str(p)
    return None


def _find_claude_exe() -> Optional[str]:
    """尝试在 PATH 中找到 claude 可执行文件。"""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    if sys.platform == "win32":
        # npm global install 常见路径
        appdata = os.environ.get("APPDATA", "")
        p = Path(appdata) / "npm/claude.cmd"
        if p.exists():
            return str(p)
    return None


def _cursor_supports_agent(cursor_exe: str) -> bool:
    """检测 cursor CLI 是否支持 `agent` 子命令。"""
    try:
        proc = subprocess.run(
            [cursor_exe, "agent", "--help"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
        if proc.returncode == 0 and "agent" in proc.stdout.lower():
            return True
        if "is not in the list of known options" in (proc.stderr or ""):
            return False
        return proc.returncode == 0
    except Exception:
        return False


def run_cursor_agent(
    config: Config,
    prompt_file: Path,
    iteration: int,
) -> tuple[str, int, str, str]:
    """使用 Cursor CLI agent 模式执行代码修改。

    - cursor_interactive=True  → 编辑器轮询模式（在 Cursor UI 中可视化工作）
    - cursor_interactive=False → cursor agent CLI headless 模式（需要 Cursor 0.48+）
    """
    cursor_exe = _find_cursor_exe()
    if not cursor_exe:
        return "cursor agent ...", 1, "", (
            "[错误] 找不到 cursor 命令。\n"
            "请确保 Cursor 已安装且 cursor 在 PATH 中，或手动指定 --agent-command-template。\n"
            "安装说明: https://docs.cursor.com/en/cli/overview"
        )

    prompt_content = read_text(prompt_file)

    if config.cursor_interactive:
        log("[Cursor] 交互模式：使用编辑器轮询方式（在 Cursor 中可视化工作）")
        return _run_cursor_editor_mode(config, cursor_exe, prompt_file, prompt_content)

    has_agent = _cursor_supports_agent(cursor_exe)
    if has_agent:
        return _run_cursor_agent_cli(config, cursor_exe, prompt_file, prompt_content)
    else:
        log("[Cursor] 当前 Cursor CLI 不支持 `cursor agent` 子命令，使用编辑器轮询模式")
        return _run_cursor_editor_mode(config, cursor_exe, prompt_file, prompt_content)


def _run_cursor_agent_cli(
    config: Config,
    cursor_exe: str,
    prompt_file: Path,
    prompt_content: str,
) -> tuple[str, int, str, str]:
    """cursor agent CLI headless 模式（Cursor 0.48+）。通过 stdin 管道传入 prompt。"""
    parts = [cursor_exe, "agent", "-p", "--force", "--trust"]

    if config.cursor_model:
        parts.extend(["--model", config.cursor_model])

    command_display = " ".join(parts) + f" (stdin<{prompt_file})"
    log(f"[Cursor] agent CLI headless 模式 (prompt={len(prompt_content)}字符)")

    start = time.time()
    try:
        proc = subprocess.run(
            parts,
            cwd=str(config.workspace),
            input=prompt_content,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=config.agent_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            command_display, 124,
            getattr(exc, "stdout", None) or "",
            (getattr(exc, "stderr", None) or "") + f"\n[timeout] {config.agent_timeout_sec}s\n",
        )

    elapsed = time.time() - start
    if elapsed < 3 and len(proc.stdout.strip()) < 100:
        log(f"[Cursor] agent CLI 似乎未正确执行（{elapsed:.1f}s 内退出），回退到编辑器轮询模式")
        return _run_cursor_editor_mode(config, cursor_exe, prompt_file, prompt_content)

    return command_display, proc.returncode, proc.stdout, proc.stderr


def _run_cursor_editor_mode(
    config: Config,
    cursor_exe: str,
    prompt_file: Path,
    prompt_content: str,
) -> tuple[str, int, str, str]:
    """兼容模式：用 cursor 打开 prompt 文件和目标文件，然后轮询等待用户/agent 完成修改。

    工作流程:
    1. 用 cursor 打开 prompt 文件（让你可以在 Cursor 中看到指令）
    2. 同时打开所有目标文件
    3. 轮询等待目标文件被修改
    4. 检测到修改后返回
    """
    target_rel = [str(p.relative_to(config.workspace)) for p in config.target_files]

    # 记录当前目标文件的 mtime
    before_mtimes = {}
    for p in config.target_files:
        if p.exists():
            before_mtimes[str(p)] = p.stat().st_mtime

    # 打开 prompt 文件和目标文件
    files_to_open = [str(prompt_file)] + [str(p) for p in config.target_files if p.exists()]
    for f in files_to_open:
        subprocess.Popen(
            [cursor_exe, f],
            cwd=str(config.workspace),
        )

    command_display = f"cursor (编辑器轮询模式: 请在 Cursor 中根据 prompt.txt 修改目标文件)"

    log(f"[Cursor] 编辑器轮询模式: 已在 Cursor 中打开 prompt 和 {len(target_rel)} 个目标文件")
    log(f"[Cursor] 请在 Cursor 中使用 Composer/Agent 功能根据 prompt 修改目标文件")
    log(f"[Cursor] prompt 文件: {prompt_file}")
    log(f"[Cursor] 目标文件: {', '.join(target_rel)}")
    log(f"[Cursor] 等待文件变更（最长 {config.agent_timeout_sec}s）...")

    # 轮询等待文件变更
    poll_interval = 5
    elapsed = 0
    while elapsed < config.agent_timeout_sec:
        time.sleep(poll_interval)
        elapsed += poll_interval

        changed = False
        for p in config.target_files:
            if p.exists():
                current_mtime = p.stat().st_mtime
                old_mtime = before_mtimes.get(str(p), 0)
                if current_mtime > old_mtime:
                    changed = True
                    break

        if changed:
            time.sleep(3)
            log(f"[Cursor] 检测到文件变更（等待 {elapsed}s 后）")
            return command_display, 0, f"文件已被修改 (elapsed={elapsed}s)", ""

        if elapsed % 30 == 0:
            log(f"[Cursor] 仍在等待文件变更... ({elapsed}s/{config.agent_timeout_sec}s)")

    return command_display, 124, "", f"[timeout] 等待 {config.agent_timeout_sec}s 无文件变更\n"


def run_claude_cli_agent(
    config: Config,
    prompt_file: Path,
    iteration: int,
) -> tuple[str, int, str, str]:
    """使用 Claude Code CLI 非交互模式执行代码修改。

    调用方式: claude -p --output-format text [--model MODEL] --allowedTools Edit,Write,Read
    通过 stdin 管道传入 prompt。
    """
    claude_exe = _find_claude_exe()
    if not claude_exe:
        return "claude -p ...", 1, "", (
            "[错误] 找不到 claude 命令。\n"
            "请先安装 Claude Code CLI: npm install -g @anthropic-ai/claude-code\n"
            "然后运行: claude auth login"
        )

    prompt_content = read_text(prompt_file)

    parts = [
        claude_exe, "-p",
        "--output-format", "text",
        "--allowedTools", config.claude_cli_allowed_tools,
    ]
    if config.claude_cli_model:
        parts.extend(["--model", config.claude_cli_model])

    command_display = " ".join(parts) + f" (stdin from {prompt_file})"
    log(f"[Claude CLI] 执行非交互模式 (prompt={len(prompt_content)}字符)")

    start = time.time()
    try:
        proc = subprocess.run(
            parts,
            cwd=str(config.workspace),
            input=prompt_content,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.agent_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        return (
            command_display, 124,
            exc.stdout or "",
            (exc.stderr or "") + f"\n[timeout] Claude CLI 超时 ({config.agent_timeout_sec}s)\n",
        )

    return command_display, proc.returncode, proc.stdout, proc.stderr


def _find_codex_exe() -> Optional[str]:
    """尝试在 PATH 中找到 codex 可执行文件（npm global install）。"""
    import shutil
    found = shutil.which("codex")
    if found:
        return found
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        p = Path(appdata) / "npm/codex.cmd"
        if p.exists():
            return str(p)
    return None


def run_codex_cli_agent(
    config: Config,
    prompt_file: Path,
    iteration: int,
) -> tuple[str, int, str, str]:
    """使用 OpenAI Codex CLI 非交互模式执行代码修改。

    调用方式: codex exec - --full-auto --sandbox danger-full-access
    通过 stdin 管道传入 prompt（codex exec 的 '-' 表示从 stdin 读取完整 prompt）。

    依赖:
      - npm install -g @openai/codex
      - 环境变量 OPENAI_API_KEY 已设置
      - 设置 CODEX_QUIET_MODE=1 可消除交互式 UI 噪音
    """
    codex_exe = _find_codex_exe()
    if not codex_exe:
        return "codex exec ...", 1, "", (
            "[错误] 找不到 codex 命令。\n"
            "请先安装 OpenAI Codex CLI: npm install -g @openai/codex\n"
            "然后运行: codex login（使用你的订阅账号登录）或设置 OPENAI_API_KEY。\n"
            "安装说明: https://github.com/openai/codex"
        )

    prompt_content = read_text(prompt_file)

    parts = [
        codex_exe,
        # 非交互：不请求审批，直接在沙盒内执行（这些是 codex 顶层参数，必须放在子命令 exec 之前）
        "-a", "never",
        "--sandbox", "danger-full-access",
        # 让 Codex 执行命令时继承当前 shell/conda 环境（例如 poweropt）
        "-c", "shell_environment_policy.inherit=all",
        "exec", "-",
    ]
    if config.codex_cli_model:
        parts.extend(["--model", config.codex_cli_model])

    command_display = " ".join(parts) + f" (stdin from {prompt_file})"
    log(f"[Codex CLI] 执行非交互模式 (prompt={len(prompt_content)}字符)")

    # CODEX_QUIET_MODE=1 抑制交互式 UI 噪音，便于捕获纯文本输出
    env = os.environ.copy()
    env["CODEX_QUIET_MODE"] = "1"

    start = time.time()
    try:
        proc = subprocess.run(
            parts,
            cwd=str(config.workspace),
            input=prompt_content,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.agent_timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        return (
            command_display, 124,
            getattr(exc, "stdout", None) or "",
            (getattr(exc, "stderr", None) or "") + f"\n[timeout] Codex CLI 超时 ({config.agent_timeout_sec}s)\n",
        )

    return command_display, proc.returncode, proc.stdout, proc.stderr


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="让外部 agent / 大模型 API 自动迭代优化 feasibility pump"
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="工作区路径，默认当前目录",
    )
    parser.add_argument(
        "--results-dir",
        default="result/agentic_fp_optimizer",
        help="保存日志、prompt、快照和历史记录的目录",
    )
    parser.add_argument(
        "--target-files",
        nargs="+",
        required=True,
        help="允许被 agent 修改的文件，相对 workspace",
    )
    parser.add_argument(
        "--eval-command",
        required=True,
        help="每轮评估命令，例如: python tests/test_feasibility_pump.py",
    )
    parser.add_argument(
        "--goal",
        default=(
            "尽量提高 feasibility pump 的可行率与 FP 收敛率，同时降低平均 gap 和平均耗时；"
            "优先优化 src/feasibility_pump.py 中的启发式逻辑。"
        ),
        help="优化目标，会直接传给 agent",
    )
    parser.add_argument(
        "--agent-kind",
        choices=["cursor", "claude-cli", "codex-cli", "command", "openai-compatible"],
        default="cursor",
        help="agent 调用方式: cursor/claude-cli/codex-cli/command/openai-compatible",
    )
    parser.add_argument(
        "--agent-command-template",
        help="command 模式的命令模板，可用 {workspace} {prompt_file} {iteration}",
    )
    parser.add_argument(
        "--cursor-model",
        default=None,
        help="cursor 模式下使用的模型名称，如 claude-3.5-sonnet（默认使用 Cursor 订阅的默认模型）",
    )
    parser.add_argument(
        "--cursor-interactive",
        action="store_true",
        help="cursor 模式下使用交互模式：在 Cursor UI 中可视化查看 agent 工作过程（默认 headless）",
    )
    parser.add_argument(
        "--claude-cli-model",
        default=None,
        help="claude-cli 模式下使用的模型名称，如 claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--codex-cli-model",
        default=None,
        help="codex-cli 模式下使用的模型名称，如 codex-1（默认使用 OPENAI_API_KEY 对应账号的默认模型）",
    )
    parser.add_argument("--api-base-url", help="OpenAI 兼容 chat completions URL")
    parser.add_argument("--api-model", help="OpenAI 兼容模型名")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="保存 API Key 的环境变量名",
    )
    parser.add_argument("--api-temperature", type=float, default=0.2)
    parser.add_argument("--api-max-tokens", type=int, default=4000)
    parser.add_argument(
        "--eval-timeout-sec",
        type=int,
        default=1800,
        help="单次评估命令超时时间（秒）",
    )
    parser.add_argument(
        "--agent-timeout-sec",
        type=int,
        default=900,
        help="单次 agent 命令超时时间（秒）",
    )
    parser.add_argument("--max-iters", type=int, default=8, help="最多优化轮数")
    parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=3,
        help="连续多少轮无提升后提前停止",
    )
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=1e-6,
        help="分数超过该阈值才视为提升",
    )
    parser.add_argument(
        "--max-context-chars-per-file",
        type=int,
        default=18000,
        help="每个目标文件最多提供给模型的字符数",
    )
    parser.add_argument(
        "--max-eval-output-chars",
        type=int,
        default=12000,
        help="评估日志最多传给模型的字符数",
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=5,
        help="prompt 中最多回顾多少条历史",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="可选的附加系统提示文件",
    )
    parser.add_argument(
        "--accept-regressions",
        action="store_true",
        help="即使退化也保留本轮修改；默认会自动回滚退化版本",
    )

    # === 新增：Surrogate模型与多策略FP优化器参数 ===

    # ---- Surrogate模型配置 ----
    parser.add_argument(
        "--surrogate-model-type",
        type=str,
        default="v3",
        choices=["v3", "bcd", "joint"],
        help="使用的Surrogate模型类型: 'v3'(三时段代理约束), 'bcd'(BCD主代理), 'joint'(联合训练)",
    )
    parser.add_argument(
        "--surrogate-model-path",
        type=str,
        default=None,
        help=(
            "Surrogate 训练输出目录（含 dual_predictor.pth、surrogate_unit_*.pth 等）。"
            "设置后会在运行 eval-command 时注入环境变量 RUN_TEST_SURROGATE_MODEL_DIR，"
            "使 run_test.py 固定加载该目录而非自动选最新。"
        ),
    )
    parser.add_argument(
        "--bcd-model-path",
        type=str,
        default=None,
        help="BCD模型路径（训练好的模型文件，用于BCD/Joint模式）",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="case3lite",
        choices=["case3", "case3lite", "case14", "case30", "case39", "case118"],
        help="案例名称",
    )
    parser.add_argument(
        "--t-delta",
        type=float,
        default=1.0,
        help="时间间隔（小时）",
    )
    parser.add_argument(
        "--time-periods",
        type=int,
        default=24,
        help="时段数",
    )
    parser.add_argument(
        "--unit-ids",
        type=int,
        nargs="+",
        default=None,
        help="机组ID列表（None表示全部机组）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（None表示全部）",
    )
    parser.add_argument(
        "--active-sets-file",
        type=str,
        default=None,
        help="数据文件路径（JSON格式，包含all_samples）",
    )

    # ---- 多策略FP优化器配置 ----
    parser.add_argument(
        "--fp-strategy-pool",
        nargs="+",
        default=["legacy", "case3lite", "guided"],
        help="可用的FP策略列表，可选: legacy, case3lite, guided",
    )
    parser.add_argument(
        "--default-fp-strategy",
        type=str,
        default="guided",
        help="默认使用的FP策略",
    )
    parser.add_argument(
        "--enable-strategy-adaptation",
        action="store_true",
        default=True,
        help="启用策略自适应选择（根据代理模型特征自动选择最优策略）",
    )
    parser.add_argument(
        "--fp-strategy-perf-db",
        type=str,
        default=None,
        help="FP策略性能跟踪数据库路径（用于自适应策略选择）",
    )

    # ---- 代理模型定向适配配置 ----
    parser.add_argument(
        "--surrogate-profile-path",
        type=str,
        default=None,
        help="代理模型特征配置文件路径(JSON)，描述系统性偏差模式",
    )
    parser.add_argument(
        "--enable-surrogate-adaptation",
        action="store_true",
        help="启用代理模型偏差检测与自适应修正",
    )
    parser.add_argument(
        "--startup-shift-window",
        type=int,
        default=2,
        help="启停时序偏差修正窗口大小（用于代理模型适配）",
    )

    # ---- 物理信息增强配置 ----
    parser.add_argument(
        "--enable-physics-informed-recovery",
        action="store_true",
        help="启用物理信息增强（历史解+邻近场景）",
    )
    parser.add_argument(
        "--neighbor-solutions-path",
        type=str,
        default=None,
        help="邻近场景解库路径（用于物理信息增强）",
    )
    parser.add_argument(
        "--historical-best-path",
        type=str,
        default=None,
        help="历史最优解存档路径",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    results_dir = (workspace / args.results_dir).resolve()
    target_files = [(workspace / item).resolve() for item in args.target_files]

    for path in target_files:
        if not path.exists():
            log(f"警告: 目标文件当前不存在，将允许 agent 新建: {path}")

    return Config(
        workspace=workspace,
        results_dir=results_dir,
        target_files=target_files,
        eval_command=args.eval_command,
        agent_kind=args.agent_kind,
        agent_command_template=args.agent_command_template,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
        api_key_env=args.api_key_env,
        api_temperature=args.api_temperature,
        api_max_tokens=args.api_max_tokens,
        eval_timeout_sec=args.eval_timeout_sec,
        agent_timeout_sec=args.agent_timeout_sec,
        max_iters=args.max_iters,
        stagnation_limit=args.stagnation_limit,
        min_score_improvement=args.min_score_improvement,
        max_context_chars_per_file=args.max_context_chars_per_file,
        max_eval_output_chars=args.max_eval_output_chars,
        max_history_items=args.max_history_items,
        goal=args.goal,
        system_prompt_file=Path(args.system_prompt_file).resolve()
        if args.system_prompt_file
        else None,
        accept_regressions=args.accept_regressions,
        # Cursor / Claude CLI / Codex CLI
        cursor_model=args.cursor_model,
        cursor_interactive=args.cursor_interactive,
        claude_cli_model=args.claude_cli_model,
        codex_cli_model=args.codex_cli_model,
        # === 新增：Surrogate模型与多策略FP优化器配置 ===
        # Surrogate模型配置
        surrogate_model_type=args.surrogate_model_type,
        surrogate_model_path=Path(args.surrogate_model_path).resolve() if args.surrogate_model_path else None,
        bcd_model_path=Path(args.bcd_model_path).resolve() if args.bcd_model_path else None,
        case_name=args.case_name,
        t_delta=args.t_delta,
        time_periods=args.time_periods,
        unit_ids=args.unit_ids,
        max_samples=args.max_samples,
        active_sets_file=Path(args.active_sets_file).resolve() if args.active_sets_file else None,
        # 多策略FP优化器配置
        fp_strategy_pool=args.fp_strategy_pool,
        default_fp_strategy=args.default_fp_strategy,
        enable_strategy_adaptation=args.enable_strategy_adaptation,
        fp_strategy_perf_db=Path(args.fp_strategy_perf_db).resolve() if args.fp_strategy_perf_db else None,
        # 代理模型定向适配配置
        surrogate_profile_path=Path(args.surrogate_profile_path).resolve() if args.surrogate_profile_path else None,
        enable_surrogate_adaptation=args.enable_surrogate_adaptation,
        startup_shift_window=args.startup_shift_window,
        # 物理信息增强配置
        enable_physics_informed_recovery=args.enable_physics_informed_recovery,
        neighbor_solutions_path=Path(args.neighbor_solutions_path).resolve() if args.neighbor_solutions_path else None,
        historical_best_path=Path(args.historical_best_path).resolve() if args.historical_best_path else None,
    )


def ensure_workspace(config: Config) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    for path in config.target_files:
        try:
            path.relative_to(config.workspace)
        except ValueError as exc:
            raise ValueError(f"目标文件不在工作区内: {path}") from exc

def _find_latest_file(directory: Path, glob_pattern: str) -> Optional[Path]:
    """在 directory 下按 glob_pattern 查找最新文件/目录（按修改时间倒序）。"""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def configure_joint():
    """预配置joint模式，自动发现最新可用模型路径。"""
    root = Path(__file__).parent.resolve()

    bcd_dir = root / "result" / "bcd_models"
    surr_dir = root / "result" / "surrogate_models"

    bcd_path = _find_latest_file(bcd_dir, "bcd_model_case3lite_*.pth")
    surr_path = None
    env_surr = os.environ.get("AGENTIC_FP_SURROGATE_DIR") or os.environ.get("RUN_TEST_SURROGATE_MODEL_DIR")
    if env_surr and env_surr.strip():
        p = Path(env_surr.strip())
        if not p.is_absolute():
            p = (root / p).resolve()
        surr_path = p if p.is_dir() else None
        if surr_path:
            log(f"[configure_joint] 使用环境变量指定的 surrogate 目录: {surr_path.name}")
        else:
            log(f"[configure_joint] 警告: AGENTIC_FP_SURROGATE_DIR 不是有效目录: {env_surr!r}")
    if surr_path is None:
        surr_path = _find_latest_file(surr_dir, "subproblem_models_case3lite_*")

    if bcd_path:
        log(f"[configure_joint] 自动发现 BCD 模型: {bcd_path.name}")
    else:
        log("[configure_joint] 警告: 未找到 case3lite 的 BCD 模型文件，"
            "请先运行 run_training.py MODE='both' 生成模型")
    if surr_path:
        log(f"[configure_joint] 自动发现 surrogate 模型: {surr_path.name}")
    else:
        log("[configure_joint] 警告: 未找到 case3lite 的 surrogate 模型目录，"
            "请先运行 run_training.py MODE='both' 生成模型")

    eval_mode = "both" if bcd_path and surr_path else "surrogate" if surr_path else "both"
    eval_command = "python run_test.py"
    if eval_mode != "both":
        log(f"[configure_joint] BCD 模型缺失，评估将使用 MODE='{eval_mode}'，已注入环境变量 RUN_TEST_MODE")

    config = Config(
        workspace=root,
        results_dir=root / "result/agentic_fp_optimizer",
        target_files=[
            root / "src/feasibility_pump.py",
            root / "src/fp_guided_recovery.py",
            root / "src/feasibility_pump_case3lite.py",
        ],
        eval_command=eval_command,
        agent_kind="codex-cli",
        cursor_interactive=False,
        agent_command_template=None,
        api_base_url=None,
        api_model=None,
        api_key_env="OPENAI_API_KEY",
        api_temperature=0.2,
        api_max_tokens=4000,
        eval_timeout_sec=1800,
        agent_timeout_sec=1200,
        max_iters=5,
        stagnation_limit=3,
        min_score_improvement=1e-6,
        max_context_chars_per_file=18000,
        max_eval_output_chars=12000,
        max_history_items=5,
        goal=(
            "尽量提高 feasibility pump 的可行率与 FP 收敛率，同时降低平均 gap 和平均耗时；"
            "优先优化 src/feasibility_pump.py 中的启发式逻辑。"
        ),
        system_prompt_file=None,
        accept_regressions=False,
        surrogate_model_type="joint",
        surrogate_model_path=surr_path,
        bcd_model_path=bcd_path,
        case_name="case3lite",
        t_delta=1.0,
        time_periods=24,
        fp_strategy_pool=["legacy", "case3lite", "guided"],
        default_fp_strategy="guided",
        enable_strategy_adaptation=True,
        # 无 surrogate_profile_path 时禁用适配，避免生成空洞的提示块
        enable_surrogate_adaptation=False,
        eval_mode=eval_mode,
    )
    return config

def main_with_config(config: Config) -> int:
    """使用预构建的 Config 对象运行优化循环（供 configure_joint() 等使用）。"""
    ensure_workspace(config)
    return _run_optimization_loop(config)


def main() -> int:
    config = parse_args()
    ensure_workspace(config)
    return _run_optimization_loop(config)


def _run_optimization_loop(config: Config) -> int:
    """核心优化循环实现。"""
    session_dir = config.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        session_dir / "config.json",
        json.dumps(
            {
                **asdict(config),
                "workspace": str(config.workspace),
                "results_dir": str(config.results_dir),
                "target_files": [str(p) for p in config.target_files],
                "system_prompt_file": str(config.system_prompt_file)
                if config.system_prompt_file
                else None,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    log(f"工作区: {config.workspace}")
    log(f"结果目录: {session_dir}")
    log(f"允许修改文件数: {len(config.target_files)}")

    original_snapshot = snapshot_files(config.target_files)

    baseline_dir = session_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_eval = evaluate(config, baseline_dir, "baseline")
    best_eval = copy.deepcopy(baseline_eval)
    latest_eval = copy.deepcopy(baseline_eval)
    best_snapshot = snapshot_files(config.target_files)

    log(
        "基线评估完成: "
        f"return_code={baseline_eval.return_code}, score={baseline_eval.score:.3f}, "
        f"metrics={asdict(baseline_eval.metrics)}"
    )
    if baseline_eval.return_code != 0:
        eval_tail = tail_text(
            baseline_eval.stdout + ("\n" + baseline_eval.stderr if baseline_eval.stderr else ""),
            max_chars=1500,
        )
        log(f"[警告] 基线评估失败 (return_code={baseline_eval.return_code})，评估输出尾部:\n{eval_tail}")
        if _is_env_error(baseline_eval):
            log(
                "[中止] 基线评估因环境依赖缺失失败（ModuleNotFoundError/ImportError），"
                "无法通过修改代码解决，请先检查并修复运行环境后重试。"
            )
            restore_snapshot(original_snapshot)
            return 1

    history: List[IterationRecord] = []
    no_improve_rounds = 0

    for iteration in range(1, config.max_iters + 1):
        if no_improve_rounds >= config.stagnation_limit:
            log(f"连续 {no_improve_rounds} 轮无提升，提前停止")
            break

        iter_dir = session_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(exist_ok=True)

        before_snapshot = snapshot_files(config.target_files)
        prompt = build_prompt(config, baseline_eval, latest_eval, best_eval, history)
        prompt_path = iter_dir / "prompt.txt"
        write_text(prompt_path, prompt)

        log(f"第 {iteration} 轮：调用 agent ({config.agent_kind})")
        agent_output_path: Optional[Path] = None
        summary = ""

        try:
            if config.agent_kind in ("cursor", "claude-cli", "codex-cli", "command"):
                if config.agent_kind == "cursor":
                    command, rc, stdout, stderr = run_cursor_agent(config, prompt_path, iteration)
                elif config.agent_kind == "claude-cli":
                    command, rc, stdout, stderr = run_claude_cli_agent(config, prompt_path, iteration)
                elif config.agent_kind == "codex-cli":
                    command, rc, stdout, stderr = run_codex_cli_agent(config, prompt_path, iteration)
                else:
                    command, rc, stdout, stderr = run_command_agent(config, prompt_path, iteration)
                write_text(iter_dir / "agent_command.txt", command)
                write_text(iter_dir / "agent_stdout.txt", stdout)
                write_text(iter_dir / "agent_stderr.txt", stderr)
                agent_output_path = iter_dir / "agent_stdout.txt"
                if rc != 0:
                    stderr_tail = tail_text(stderr, max_chars=2000)
                    stdout_tail = tail_text(stdout, max_chars=1000)
                    detail_parts = [f"exit_code={rc}"]
                    if stderr_tail:
                        detail_parts.append(f"stderr_tail=\n{stderr_tail}")
                    elif stdout_tail:
                        detail_parts.append(f"stdout_tail=\n{stdout_tail}")
                    raise RuntimeError("agent 命令失败，" + "\n".join(detail_parts))
                summary = stdout.strip().splitlines()[-1] if stdout.strip() else ""
            else:
                raw_api = call_openai_compatible_api(config, prompt)
                write_text(iter_dir / "agent_api_raw.json", raw_api)
                content = extract_api_message(raw_api)
                write_text(iter_dir / "agent_response.txt", content)
                agent_output_path = iter_dir / "agent_response.txt"
                summary = apply_structured_operations(config, content)
        except Exception as exc:
            log(f"第 {iteration} 轮 agent 执行失败: {exc}")
            restore_snapshot(before_snapshot)
            history.append(
                IterationRecord(
                    iteration=iteration,
                    agent_kind=config.agent_kind,
                    prompt_path=str(prompt_path),
                    agent_output_path=str(agent_output_path) if agent_output_path else None,
                    accepted=False,
                    summary=f"agent failure: {exc}",
                )
            )
            write_text(
                session_dir / "history.json",
                json.dumps([asdict(item) for item in history], ensure_ascii=False, indent=2),
            )
            no_improve_rounds += 1
            if (
                config.agent_kind == "claude-cli"
                and "exit_code=124" in str(exc)
                and has_recent_agent_timeout(history, limit=2)
            ):
                log("claude-cli 连续两轮超时，提前停止后续重试")
                break
            continue

        after_snapshot = snapshot_files(config.target_files)
        changed_files = [
            str(Path(p).relative_to(config.workspace))
            for p in list_changed_files(before_snapshot, after_snapshot)
        ]

        diff_snippet = build_diff_snippet(before_snapshot, after_snapshot)
        write_text(iter_dir / "changes.diff", diff_snippet)

        if not changed_files:
            log(f"第 {iteration} 轮未检测到目标文件变更")
            history.append(
                IterationRecord(
                    iteration=iteration,
                    agent_kind=config.agent_kind,
                    prompt_path=str(prompt_path),
                    agent_output_path=str(agent_output_path) if agent_output_path else None,
                    accepted=False,
                    changed_files=[],
                    summary=summary or "no changes",
                )
            )
            write_text(
                session_dir / "history.json",
                json.dumps([asdict(item) for item in history], ensure_ascii=False, indent=2),
            )
            no_improve_rounds += 1
            continue

        latest_eval = evaluate(config, iter_dir, f"iter_{iteration:02d}")
        previous_best_score = best_eval.score
        improved = latest_eval.score > previous_best_score + config.min_score_improvement
        # 当基线本身无意义（eval 崩溃）时，要求本轮至少有有意义的结果才能接受
        # 避免仅因"让 eval 跑起来"而接受无实际改善的修改
        if not _is_meaningful_result(baseline_eval):
            accepted = _is_meaningful_result(latest_eval) or config.accept_regressions
        else:
            accepted = improved or config.accept_regressions

        if improved:
            best_eval = copy.deepcopy(latest_eval)
            best_snapshot = snapshot_files(config.target_files)
            no_improve_rounds = 0
            log(
                f"第 {iteration} 轮提升成功: "
                f"score {latest_eval.score:.3f} > {previous_best_score:.3f}"
            )
        elif accepted and not improved:
            # 基线无意义时首轮有意义结果被接受，但分数未必超过 best（best=baseline=-1000）
            # 此时应更新 best，不回滚
            if _is_meaningful_result(latest_eval) and not _is_meaningful_result(baseline_eval):
                best_eval = copy.deepcopy(latest_eval)
                best_snapshot = snapshot_files(config.target_files)
                no_improve_rounds = 0
                log(
                    f"第 {iteration} 轮：基线无意义，本轮首次获得有意义结果 "
                    f"(score={latest_eval.score:.3f})，接受并更新最佳"
                )
            else:
                no_improve_rounds += 1
                log(f"第 {iteration} 轮未提升，但按配置保留修改")
        else:
            no_improve_rounds += 1
            if not config.accept_regressions:
                restore_snapshot(before_snapshot)
                latest_eval = copy.deepcopy(best_eval)
                log(f"第 {iteration} 轮退化，已回滚")
            else:
                log(f"第 {iteration} 轮未提升，但按配置保留修改")

        record = IterationRecord(
            iteration=iteration,
            agent_kind=config.agent_kind,
            prompt_path=str(prompt_path),
            agent_output_path=str(agent_output_path) if agent_output_path else None,
            accepted=accepted,
            changed_files=changed_files,
            summary=summary,
            eval_score=latest_eval.score,
            metrics=asdict(latest_eval.metrics),
        )
        history.append(record)

        write_text(
            session_dir / "history.json",
            json.dumps([asdict(item) for item in history], ensure_ascii=False, indent=2),
        )

    restore_snapshot(best_snapshot)

    final_report = {
        "baseline_score": baseline_eval.score,
        "best_score": best_eval.score,
        "baseline_metrics": asdict(baseline_eval.metrics),
        "best_metrics": asdict(best_eval.metrics),
        "iterations": [asdict(item) for item in history],
        "final_target_files": [str(p.relative_to(config.workspace)) for p in config.target_files],
    }
    write_text(session_dir / "final_report.json", json.dumps(final_report, ensure_ascii=False, indent=2))

    log("优化循环结束")
    log(f"基线分数: {baseline_eval.score:.3f}")
    log(f"最佳分数: {best_eval.score:.3f}")
    log(f"最佳指标: {asdict(best_eval.metrics)}")

    # 若没有任何改动被接受，则恢复到原始状态；否则保留最佳版本。
    if best_snapshot == original_snapshot:
        restore_snapshot(original_snapshot)

    return 0


if __name__ == "__main__":
    # 方式 A: 命令行参数模式
    #   python agentic_fp_optimizer.py --target-files src/feasibility_pump.py --eval-command "..." --agent-kind cursor
    #   sys.exit(main())
    #
    # 方式 B: 预配置模式（无需传参，直接运行）
    config = configure_joint()
    sys.exit(main_with_config(config))
