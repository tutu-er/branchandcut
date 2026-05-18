#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量测试 ``result/training_metric/models`` 下的子问题 surrogate checkpoint，与本仓库对照组成对对比。

训练产物目录名里若出现 ``case30``（非 case30lite），测试时一律使用 **case30lite**
案例与对应的 active_set（与 lightweight 算例一致），避免误用 IEEE 30 母线与数据维度不一致。

典型布局（示例）::

    result/training_metric/models/subproblem_models_case14_YYYYMMDD_HHMMSS/
    result/training_metric/models/subproblem_models_case14_control_YYYYMMDD_HHMMSS/
    result/training_metric/models/subproblem_models_case30_YYYYMMDD_HHMMSS/   -> 实测 case30lite

用法::

    python run_test_training_metric_models.py
    python run_test_training_metric_models.py --samples 16 --disable-plots
    python run_test_training_metric_models.py --models-root result/training_metric/models
    python run_test_training_metric_models.py --main-dir ... --control-dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent

# active set 路径与 ``run_test_case14`` / ``run_test_case30lite`` 默认值保持一致（此处硬编码，
# 避免 import 链路拉取 ``run_training``/gurobi 导致仅做 --dry-run 也无法启动）。
DEFAULT_ACTIVE_JSON: dict[str, str] = {
    "case14": "result/active_set/active_sets_case14_T24_n600_20260503_222929.json",
    "case30lite": "result/active_set/active_sets_case30lite_T24_n500_20260503_233729.json",
}

Role = Literal["main", "control"]

_SUBMODEL_RE = re.compile(
    # case14 / case30 / case30lite / case30lite_perturbed / case118 等训练输出目录名
    r"^subproblem_models_(case\d+(?:lite(?:_perturbed)?)?)_(.+)$",
    re.IGNORECASE,
)


def _runtime_case_from_token(case_token: str) -> str:
    """checkpoint 命名里的 case_token -> ``run_test`` 使用的 CASE_NAME。"""
    c = str(case_token).strip().lower()
    if c == "case30":
        return "case30lite"
    return c


def _parse_subproblem_dir(model_dir: Path) -> tuple[str, Role] | None:
    """
    目录名形如 subproblem_models_case14_<ts> / subproblem_models_case14_control_<ts>.
    返回 (checkpoint 内的 case token, role)。
    """
    name = model_dir.name
    m = _SUBMODEL_RE.match(name)
    if not m:
        return None
    case_tok = m.group(1).lower()
    rest = str(m.group(2)).strip()
    rl = rest.lower()
    role: Role = "control" if rl.startswith("control_") else "main"
    return case_tok, role


def _discover_candidates(models_root: Path) -> list[Path]:
    if not models_root.is_dir():
        return []
    out: list[Path] = []
    for p in models_root.glob("**/subproblem_models_*"):
        if p.is_dir() and _parse_subproblem_dir(p) is not None:
            out.append(p)
    uniq = {p.resolve(): p for p in out}
    return sorted(uniq.values(), key=lambda x: x.stat().st_mtime, reverse=True)


def _latest_by_runtime_case_role(
    candidates: list[Path],
) -> dict[tuple[str, Role], Path]:
    """每个 (运行时案例名, role) 只保留时间上最新的一份目录。"""
    picked: dict[tuple[str, Role], Path] = {}
    for p in candidates:
        parsed = _parse_subproblem_dir(p)
        if not parsed:
            continue
        case_tok, role = parsed
        run_case = _runtime_case_from_token(case_tok)
        key = (run_case, role)
        if key not in picked:
            picked[key] = p
    return picked


def _rel_repo(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _snapshot_reports(case_name: str) -> frozenset[Path]:
    report_dir = ROOT / "result" / "model_tests" / "run_test_reports"
    if not report_dir.is_dir():
        return frozenset()
    return frozenset(report_dir.glob(f"run_test_{case_name}_*.json"))


def _new_reports_after(case_name: str, before: frozenset[Path]) -> Path | None:
    report_dir = ROOT / "result" / "model_tests" / "run_test_reports"
    if not report_dir.is_dir():
        return None
    after = sorted(
        (p for p in report_dir.glob(f"run_test_{case_name}_*.json") if p not in before),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    return after[0] if after else None


def _run_one_surrogate_eval(
    *,
    run_case: str,
    active_json: Path,
    model_dir: Path,
    samples: int,
    sample_range: str,
    run_fp: bool,
    disable_plots: bool,
    strategy: str,
    constraint_scope: str,
) -> Path | None:
    import run_test as rt

    if run_case not in (
        "case3lite",
        "case14",
        "case30lite",
        "case30lite_perturbed",
        "case118",
        "case39",
    ):
        print(f"[skip] run_test 未在此脚本白名单校验 case={run_case}，请改用 run_test 系列入口。")
        return None

    model_rel = _rel_repo(model_dir)
    if not model_dir.is_dir():
        print(f"[err] model_dir 不存在: {model_dir}", file=sys.stderr)
        return None
    aj = Path(active_json)
    if not aj.is_absolute():
        aj = ROOT / aj
    if not aj.is_file():
        print(f"[err] active_set 不存在: {aj}", file=sys.stderr)
        return None

    before = _snapshot_reports(run_case)

    rt.MODE = "surrogate"
    rt.CASE_NAME = run_case
    rt.ACTIVE_SETS_FILE = _rel_repo(aj)
    rt.MODEL_DIR = model_rel
    rt.RUN_TEST_DISABLE_PLOTS = bool(disable_plots)
    rt.RUN_FP = bool(run_fp)
    rt.RUN_SUBPROBLEM_MILP_TEST = False
    rt.TEST_SAMPLES = max(1, int(samples))
    rt.TEST_SAMPLES_DEFAULT = rt.TEST_SAMPLES
    rt.SAMPLE_RANGE = sample_range
    rt.MAX_SAMPLES = None
    rt.UNIT_IDS = None
    rt.SURROGATE_CONSTRAINT_STRATEGY = strategy
    rt.SURROGATE_CONSTRAINT_SCOPE = constraint_scope
    rt.BCD_PROXY_SCOPE = getattr(rt, "BCD_PROXY_SCOPE", "both")
    rt.USE_CASE3LITE_CUSTOM_FP = False
    rt.USE_CASE118_CUSTOM_FP = False
    rt.SUBPROBLEM_IGNORE_STARTUP_SHUTDOWN_COSTS = False

    os.environ.pop("RUN_TEST_SURROGATE_MODEL_DIR", None)

    print("=" * 72, flush=True)
    print(
        f"run_test surrogate | CASE_NAME={run_case} | model_dir={model_rel}",
        flush=True,
    )
    print(f"  active_set={_rel_repo(aj)} | samples={samples} range={sample_range} fp={run_fp}",
          flush=True)
    print("=" * 72, flush=True)

    rt.main()

    rp = _new_reports_after(run_case, before)
    if rp is None:
        print("[warn] 未找到新增的 run_test 报告 json", flush=True)
    else:
        print(f"[report] {rp.relative_to(ROOT)}", flush=True)
    return rp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models-root",
        type=Path,
        default=ROOT / "result" / "training_metric" / "models",
        help="内含 subproblem_models_* 目录的根路径",
    )
    p.add_argument(
        "--main-dir",
        type=Path,
        default=None,
        help="跳过自动发现：显式指定本文模型目录（绝对路径或相对仓库根）。",
    )
    p.add_argument(
        "--control-dir",
        type=Path,
        default=None,
        help="跳过自动发现：显式指定对照组模型目录。",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=16,
        help="TEST_SAMPLES（默认 16）。",
    )
    p.add_argument(
        "--sample-range",
        type=str,
        default="0:120",
        help="传给 run_test 的 SAMPLE_RANGE。",
    )
    p.add_argument("--fp", action="store_true", help="启用可行性泵（较慢）。")
    p.add_argument("--disable-plots", action="store_true", help="禁止 matplotlib 绘图。")
    p.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="SURROGATE_CONSTRAINT_STRATEGY（默认 auto）。",
    )
    p.add_argument(
        "--surrogate-constraint-scope",
        choices=("all", "sign4"),
        default="all",
    )
    p.add_argument(
        "--active-case14",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"覆盖 case14 active_set（默认: {DEFAULT_ACTIVE_JSON['case14']}）",
    )
    p.add_argument(
        "--active-case30lite",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"覆盖 case30lite active_set（默认: {DEFAULT_ACTIVE_JSON['case30lite']}）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印配对与路径，不执行 run_test。",
    )
    args = p.parse_args()

    models_root = args.models_root.resolve() if args.models_root.is_absolute() else (ROOT / args.models_root).resolve()
    active14 = Path(args.active_case14 or DEFAULT_ACTIVE_JSON["case14"])
    active30 = Path(args.active_case30lite or DEFAULT_ACTIVE_JSON["case30lite"])

    pairs_spec: list[tuple[str, Path, Path, str]]

    if args.main_dir is not None and args.control_dir is not None:
        main_p = (
            args.main_dir.resolve()
            if args.main_dir.is_absolute()
            else (ROOT / args.main_dir).resolve()
        )
        ctrl_p = (
            args.control_dir.resolve()
            if args.control_dir.is_absolute()
            else (ROOT / args.control_dir).resolve()
        )
        pm = _parse_subproblem_dir(main_p)
        pc = _parse_subproblem_dir(ctrl_p)
        if not pm or not pc:
            raise SystemExit("无法从目录名解析 subproblem_models_*，请改用标准命名的目录：subproblem_models_<case>_...")
        ct_m, ct_c = pm[0].lower(), pc[0].lower()
        rt_m = _runtime_case_from_token(ct_m)
        rt_c = _runtime_case_from_token(ct_c)
        if rt_m != rt_c:
            raise SystemExit(f"两组模型解析出的测试案例不一致: {rt_m} vs {rt_c}")
        note = ""
        if "case30" in ct_m and ct_m == "case30":
            note = "checkpoint_case30→runtime_case30lite"
        pairs_spec = [(rt_m, main_p, ctrl_p, note)]
    else:
        cands = _discover_candidates(models_root)
        if not cands:
            raise SystemExit(
                f"在 {models_root} 未发现 subproblem_models_* 目录。\n"
                "请确认模型已解压到该路径，或使用 --main-dir/--control-dir 显式传入。"
            )
        latest_map = _latest_by_runtime_case_role(cands)

        grouped: dict[str, tuple[Path | None, Path | None]] = {}
        for (run_case, role), path in latest_map.items():
            if run_case not in grouped:
                grouped[run_case] = (None, None)
            m, c = grouped[run_case]
            if role == "main":
                m = path
            else:
                c = path
            grouped[run_case] = (m, c)

        pairs_spec = []
        for run_case in sorted(grouped.keys()):
            m_dir, c_dir = grouped[run_case]
            if m_dir is None or c_dir is None:
                print(
                    f"[skip pair] case={run_case} 缺少 main 或 control，"
                    f"main={m_dir} control={c_dir}",
                    flush=True,
                )
                continue
            note = ""
            for d in (m_dir, c_dir):
                pr = _parse_subproblem_dir(d)
                if pr and pr[0].lower() == "case30":
                    note = "checkpoint_case30→runtime_case30lite"
                    break
            pairs_spec.append((run_case, m_dir, c_dir, note))

        if not pairs_spec:
            raise SystemExit(
                "未发现「同一测试案例同时具备 main + control」的配对；请检查命名（对照组应为 "
                "subproblem_models_<case>_control_<ts>）。"
            )

    batch_out_dir = ROOT / "result" / "model_tests" / "training_metric_compare"
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle: dict[str, object] = {
        "batch_timestamp": stamp,
        "models_root": _rel_repo(models_root),
        "pairs": [],
    }

    for run_case, main_p, ctrl_p, remap_note in pairs_spec:
        if run_case == "case14":
            aj = active14.resolve() if active14.is_absolute() else (ROOT / active14).resolve()
        elif run_case in ("case30lite", "case30lite_perturbed"):
            aj = active30.resolve() if active30.is_absolute() else (ROOT / active30).resolve()
        else:
            print(
                f"[warn] active_set 未为该 case 内置单独 JSON，fallback 仍用 "
                f"case30lite 默认：run_case={run_case}",
                flush=True,
            )
            aj = (ROOT / DEFAULT_ACTIVE_JSON["case30lite"]).resolve()

        entry: dict[str, object] = {
            "runtime_case_name": run_case,
            "remap_note": remap_note or None,
            "main_dir": _rel_repo(main_p),
            "control_dir": _rel_repo(ctrl_p),
            "reports": {},
        }
        print("", flush=True)
        print("*" * 72, flush=True)
        print(
            f"Compare pair | CASE_NAME={run_case} | {_rel_repo(main_p)}  vs  {_rel_repo(ctrl_p)}",
            flush=True,
        )
        if remap_note:
            print(f"  note: {remap_note}", flush=True)
        print("*" * 72, flush=True)

        if args.dry_run:
            bundle["pairs"].append(entry)
            continue

        rp_main = _run_one_surrogate_eval(
            run_case=run_case,
            active_json=Path(aj),
            model_dir=main_p,
            samples=args.samples,
            sample_range=args.sample_range,
            run_fp=args.fp,
            disable_plots=args.disable_plots,
            strategy=args.strategy,
            constraint_scope=args.surrogate_constraint_scope,
        )
        entry["reports"]["main"] = str(rp_main.relative_to(ROOT)).replace("\\", "/") if rp_main else None

        rp_ctl = _run_one_surrogate_eval(
            run_case=run_case,
            active_json=Path(aj),
            model_dir=ctrl_p,
            samples=args.samples,
            sample_range=args.sample_range,
            run_fp=args.fp,
            disable_plots=args.disable_plots,
            strategy=args.strategy,
            constraint_scope=args.surrogate_constraint_scope,
        )
        entry["reports"]["control"] = str(rp_ctl.relative_to(ROOT)).replace("\\", "/") if rp_ctl else None

        lp_main = lp_ctl = {}
        try:
            if rp_main:
                lp_main = json.loads(rp_main.read_text(encoding="utf-8")).get(
                    "lp_compare_summary"
                ) or {}
            if rp_ctl:
                lp_ctl = json.loads(rp_ctl.read_text(encoding="utf-8")).get(
                    "lp_compare_summary"
                ) or {}
        except Exception as ex:
            print(f"[warn] 读取 LP summary 失败: {ex}", flush=True)

        entry["lp_compare_summary_keys_diff"] = {
            "keys_main": sorted(lp_main.keys())[:50],
            "keys_control": sorted(lp_ctl.keys())[:50],
        }
        bundle["pairs"].append(entry)

    summary_path = batch_out_dir / f"training_metric_compare_batch_{stamp}.json"
    summary_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("", flush=True)
    print(f"[batch done] summary → {summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
