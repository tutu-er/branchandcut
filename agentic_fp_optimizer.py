#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
让 Claude Code、其他编程 agent，或 OpenAI 兼容 API 持续迭代改进 feasibility pump。

核心闭环：
1. 运行评估命令，提取当前可行率 / 收敛率 / gap / 耗时
2. 把当前指标、失败日志、历史尝试和目标文件上下文喂给 agent
3. agent 直接修改代码，或通过 API 返回结构化编辑指令
4. 再次运行评估，根据分数决定保留还是回滚
5. 循环多轮，最后保留最佳版本

默认推荐的使用方式是 command 模式：脚本只负责“编排”，具体改代码由外部 agent CLI 完成。
如果你想接入任意 OpenAI 兼容模型 API，也可使用 openai-compatible 模式。

示例 1：外部 agent 命令模式
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind command ^
        --agent-command-template "your-agent-cli --workspace {workspace} --prompt-file {prompt_file}"

示例 2：OpenAI 兼容 API 模式
    set OPENAI_API_KEY=...
    python agentic_fp_optimizer.py ^
        --target-files src/feasibility_pump.py ^
        --eval-command "python tests/test_feasibility_pump.py" ^
        --agent-kind openai-compatible ^
        --api-base-url "https://api.openai.com/v1/chat/completions" ^
        --api-model "gpt-4.1" ^
        --api-key-env OPENAI_API_KEY

说明：
- command 模式下，agent 命令模板可使用以下占位符：
    {workspace}   工作区绝对路径
    {prompt_file} 本轮 prompt 文件路径
    {iteration}   当前轮次
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
from typing import Any, Dict, List, Optional


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


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
    max_iters: int
    stagnation_limit: int
    min_score_improvement: float
    max_context_chars_per_file: int
    max_eval_output_chars: int
    max_history_items: int
    goal: str
    system_prompt_file: Optional[Path]
    accept_regressions: bool


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


def parse_metrics(output: str) -> Metrics:
    metrics = Metrics()

    m = re.search(r"结果:\s*(\d+)/(\d+)\s*通过", output)
    if m:
        metrics.test1_pass = m.group(1) == m.group(2)

    m = re.search(r"可行率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", output)
    if m:
        metrics.feasible_count = int(m.group(1))
        metrics.sample_count = int(m.group(2))
        metrics.feasible_rate = float(m.group(3)) / 100.0

    m = re.search(r"FP 收敛率:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", output)
    if m:
        metrics.fp_success_count = int(m.group(1))
        metrics.fp_success_rate = float(m.group(3)) / 100.0

    m = re.search(
        r"Gap \(vs MILP\):\s*平均=([+\-]?\d+(?:\.\d+)?)%,\s*最大=([+\-]?\d+(?:\.\d+)?)%,\s*最小=([+\-]?\d+(?:\.\d+)?)%",
        output,
    )
    if m:
        metrics.avg_gap_percent = float(m.group(1))
        metrics.max_gap_percent = float(m.group(2))
        metrics.min_gap_percent = float(m.group(3))

    m = re.search(r"FP 平均耗时:\s*([\d.]+)s", output)
    if m:
        metrics.avg_fp_time_sec = float(m.group(1))

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


def run_command(command: str, cwd: Path) -> tuple[int, str, str, float]:
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        shell=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def evaluate(config: Config, iteration_dir: Path, label: str) -> EvalResult:
    log(f"运行评估: {label}")
    rc, stdout, stderr, elapsed = run_command(config.eval_command, config.workspace)
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


def build_context_block(config: Config) -> str:
    chunks: List[str] = []
    for path in config.target_files:
        rel = path.relative_to(config.workspace)
        content = read_text(path) if path.exists() else ""
        chunks.append(
            f"### FILE: {rel}\n"
            f"```python\n{truncate(content, config.max_context_chars_per_file)}\n```\n"
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


def build_prompt(
    config: Config,
    baseline: EvalResult,
    latest: EvalResult,
    best: EvalResult,
    history: List[IterationRecord],
) -> str:
    system_extra = ""
    if config.system_prompt_file and config.system_prompt_file.exists():
        system_extra = read_text(config.system_prompt_file).strip()

    latest_output = truncate(
        latest.stdout + ("\n[stderr]\n" + latest.stderr if latest.stderr else ""),
        config.max_eval_output_chars,
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

        允许修改的文件：
        {chr(10).join(f"- {p.relative_to(config.workspace)}" for p in config.target_files)}

        基线分数：
        {baseline.score:.3f}
        基线指标：
        {json.dumps(asdict(baseline.metrics), ensure_ascii=False, indent=2)}

        当前工作树分数：
        {latest.score:.3f}
        当前指标：
        {json.dumps(asdict(latest.metrics), ensure_ascii=False, indent=2)}

        当前最佳分数：
        {best.score:.3f}
        当前最佳指标：
        {json.dumps(asdict(best.metrics), ensure_ascii=False, indent=2)}

        最近评估输出：
        ```text
        {latest_output}
        ```

        历史尝试摘要：
        {build_history_block(history, config.max_history_items)}

        目标文件上下文：
        {build_context_block(config)}

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
    rc, stdout, stderr, _elapsed = run_command(command, config.workspace)
    return command, rc, stdout, stderr


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
        choices=["command", "openai-compatible"],
        default="command",
        help="外部 agent 调用方式",
    )
    parser.add_argument(
        "--agent-command-template",
        help="command 模式的命令模板，可用 {workspace} {prompt_file} {iteration}",
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
    )


def ensure_workspace(config: Config) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    for path in config.target_files:
        try:
            path.relative_to(config.workspace)
        except ValueError as exc:
            raise ValueError(f"目标文件不在工作区内: {path}") from exc


def main() -> int:
    config = parse_args()
    ensure_workspace(config)

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
            if config.agent_kind == "command":
                command, rc, stdout, stderr = run_command_agent(config, prompt_path, iteration)
                write_text(iter_dir / "agent_command.txt", command)
                write_text(iter_dir / "agent_stdout.txt", stdout)
                write_text(iter_dir / "agent_stderr.txt", stderr)
                agent_output_path = iter_dir / "agent_stdout.txt"
                if rc != 0:
                    raise RuntimeError(f"agent 命令失败，exit_code={rc}")
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
        accepted = improved or config.accept_regressions

        if improved:
            best_eval = copy.deepcopy(latest_eval)
            best_snapshot = snapshot_files(config.target_files)
            no_improve_rounds = 0
            log(
                f"第 {iteration} 轮提升成功: "
                f"score {latest_eval.score:.3f} > {previous_best_score:.3f}"
            )
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
    sys.exit(main())
