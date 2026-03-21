#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可直接作为 agentic_fp_optimizer.py 的 --eval-command 使用的评估脚本。

特点：
1. 自动发现最新 surrogate / BCD 模型
2. 优先读取 result/active_set 下的 active_sets_*.json
3. 若缺少 active_set 数据，则回退到合成样本
4. 输出格式兼容 agentic_fp_optimizer.py 的 parse_metrics()

示例：
    python tests/eval_feasibility_pump.py
    python tests/eval_feasibility_pump.py --n-test 3 --n-check 2
    python tests/eval_feasibility_pump.py --case case30 --no-bcd-surrogate
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 某些 torch / numpy / MKL 组合在 Windows 下会触发重复 OpenMP runtime 报错。
# 这里采用保守兼容设置，优先保证评估脚本可被 agent 循环稳定调用。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pypower.case14
import pypower.case30
import pypower.case39
from pypower.ext2int import ext2int
from pypower.idx_gen import PMIN, PMAX

from src.feasibility_pump import check_uc_feasibility, recover_integer_solution
from src.uc_NN_BCD import Agent_NN_BCD
from src.uc_NN_subproblem import (
    ActiveSetReader,
    DualVariablePredictorTrainer,
    SubproblemSurrogateTrainer,
    generate_test_data,
)
from src.uc_unified_surrogate import UnifiedSurrogateManager


def _pick_latest_path(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _build_ppc(case_name: str):
    ppc_map = {
        "case14": pypower.case14.case14,
        "case30": pypower.case30.case30,
        "case39": pypower.case39.case39,
    }
    if case_name not in ppc_map:
        raise ValueError(f"未知案例: {case_name}")
    return ppc_map[case_name]()


def _resolve_surrogate_model_dir(case_name: str) -> Path:
    candidates = []
    for parent in (ROOT / "result" / "subproblem_models", ROOT / "result" / "surrogate_models"):
        if not parent.exists():
            continue
        candidates.extend(parent.glob(f"subproblem_models_{case_name}_*"))
    resolved = _pick_latest_path(candidates)
    if resolved is None:
        raise FileNotFoundError(f"未找到 {case_name} 的 surrogate 模型目录")
    return resolved


def _resolve_bcd_model_path(case_name: str) -> Optional[Path]:
    model_dir = ROOT / "result" / "bcd_models"
    if not model_dir.exists():
        return None
    return _pick_latest_path(list(model_dir.glob(f"bcd_model_{case_name}_*.pth")))


def _parse_sample_range(sample_range: Optional[str]) -> Optional[tuple[int, int]]:
    if sample_range is None:
        return None
    parts = sample_range.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("--sample-range 格式必须为 start:end，例如 10:20")
    start = int(parts[0])
    end = int(parts[1])
    if start < 0 or end < 0 or end <= start:
        raise ValueError("--sample-range 要求 0 <= start < end")
    return start, end


def _load_active_set_data(case_name: str, ppc, t_periods: int, n_samples: int, sample_range: Optional[tuple[int, int]] = None):
    active_set_dir = ROOT / "result" / "active_set"
    data_path = None
    if active_set_dir.exists():
        data_path = _pick_latest_path(list(active_set_dir.glob(f"active_sets_{case_name}_*.json")))

    if data_path is not None:
        print(f"加载 active_set 数据: {data_path}", flush=True)
        reader = ActiveSetReader(str(data_path))
        all_samples = reader.load_all_samples()
        total_available = len(all_samples)
        if sample_range is not None:
            start, end = sample_range
            if start >= total_available:
                raise ValueError(f"--sample-range 起点 {start} 超出样本数 {total_available}")
            actual_end = min(end, total_available)
            all_samples = all_samples[start:actual_end]
            print(f"  使用样本范围 [{start}:{actual_end}) 进行评估", flush=True)
        elif len(all_samples) > n_samples:
            all_samples = all_samples[:n_samples]
            print(f"  截取前 {n_samples} 个样本用于评估", flush=True)
        return all_samples, data_path, False

    print("未找到 active_set JSON，回退到合成样本", flush=True)
    synthetic_samples = generate_test_data(ppc, T=t_periods, n_samples=n_samples, seed=42)
    return synthetic_samples, None, True


def load_models(ppc, active_set_data, t_delta: float, model_dir: Path):
    print("加载 surrogate 模型 ...", flush=True)
    print(f"  surrogate 模型目录: {model_dir}", flush=True)

    lambda_predictor = DualVariablePredictorTrainer(ppc, active_set_data, t_delta)
    pred_path = model_dir / "dual_predictor.pth"
    lambda_predictor.load(str(pred_path))

    ppc_int = ext2int(ppc)
    ng = ppc_int["gen"].shape[0]
    trainers = {}
    for g in range(ng):
        path = model_dir / f"surrogate_unit_{g}.pth"
        if not path.exists():
            continue
        trainer = SubproblemSurrogateTrainer(
            ppc,
            active_set_data,
            t_delta,
            unit_id=g,
            lambda_predictor=lambda_predictor,
        )
        trainer.load(str(path))
        trainers[g] = trainer

    if not trainers:
        raise RuntimeError(f"在 {model_dir} 中未加载到任何 surrogate_unit_*.pth")
    print(f"  已加载 {len(trainers)} 个机组 surrogate 模型", flush=True)
    return lambda_predictor, trainers


def load_bcd_manager(ppc, active_set_data, t_delta: float, bcd_model_path: Optional[Path], trainers):
    if bcd_model_path is None:
        print("未找到 BCD 模型，跳过 UnifiedSurrogateManager", flush=True)
        return None

    print(f"加载 BCD 模型: {bcd_model_path}", flush=True)
    agent = Agent_NN_BCD(ppc, active_set_data, t_delta)
    agent.load_model_parameters(str(bcd_model_path))
    print("  BCD agent: 已加载", flush=True)

    manager = UnifiedSurrogateManager(agent, trainers, active_set_data)
    print("  UnifiedSurrogateManager: 已启用（BCD + subproblem 联合约束）", flush=True)
    return manager


def solve_milp_basic(ppc, pd_data, t_delta: float = 1.0, time_limit: float = 120.0):
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng = gen.shape[0]
    t_horizon = pd_data.shape[1]
    pd_sum = np.sum(pd_data, axis=0)

    model = gp.Model("uc_milp_eval")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = time_limit

    pg = model.addVars(ng, t_horizon, lb=0, name="pg")
    x = model.addVars(ng, t_horizon, vtype=GRB.BINARY, name="x")
    coc = model.addVars(ng, t_horizon - 1, lb=0, name="coc")
    cpower = model.addVars(ng, t_horizon, lb=0, name="cpower")

    ru = 0.4 * gen[:, PMAX] / t_delta
    rd = 0.4 * gen[:, PMAX] / t_delta
    ru_co = 0.3 * gen[:, PMAX]
    rd_co = 0.3 * gen[:, PMAX]
    ton = min(int(4 * t_delta), t_horizon - 1)
    toff = min(int(4 * t_delta), t_horizon - 1)
    sc = gencost[:, 1]
    hc = gencost[:, 2]

    for t in range(t_horizon):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(pd_sum[t]))
        for g in range(ng):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x[g, t])

    for g in range(ng):
        for t in range(1, t_horizon):
            model.addConstr(pg[g, t] - pg[g, t - 1] <= ru[g] * x[g, t - 1] + ru_co[g] * (1 - x[g, t - 1]))
            model.addConstr(pg[g, t - 1] - pg[g, t] <= rd[g] * x[g, t] + rd_co[g] * (1 - x[g, t]))
        for tau in range(1, ton + 1):
            for t1 in range(t_horizon - tau):
                model.addConstr(x[g, t1 + 1] - x[g, t1] <= x[g, t1 + tau])
        for tau in range(1, toff + 1):
            for t1 in range(t_horizon - tau):
                model.addConstr(-x[g, t1 + 1] + x[g, t1] <= 1 - x[g, t1 + tau])
        for t in range(1, t_horizon):
            model.addConstr(coc[g, t - 1] >= sc[g] * (x[g, t] - x[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= hc[g] * (x[g, t - 1] - x[g, t]))
        for t in range(t_horizon):
            model.addConstr(cpower[g, t] >= gencost[g, -2] / t_delta * pg[g, t] + gencost[g, -1] / t_delta * x[g, t])

    obj = gp.quicksum(cpower[g, t] for g in range(ng) for t in range(t_horizon))
    obj += gp.quicksum(coc[g, t] for g in range(ng) for t in range(t_horizon - 1))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        x_sol = np.round(np.array([[x[g, t].X for t in range(t_horizon)] for g in range(ng)])).astype(int)
        return x_sol, float(model.ObjVal)
    return None, None


def compute_obj_fixed_x(x_int, ppc, pd_data, t_delta: float = 1.0):
    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    gencost = ppc_int["gencost"]
    ng, t_horizon = x_int.shape
    pd_sum = np.sum(pd_data, axis=0)

    model = gp.Model("obj_lp_eval")
    model.Params.OutputFlag = 0

    pg = model.addVars(ng, t_horizon, lb=0, name="pg")
    coc = model.addVars(ng, t_horizon - 1, lb=0, name="coc")
    cpower = model.addVars(ng, t_horizon, lb=0, name="cpower")

    ru = 0.4 * gen[:, PMAX] / t_delta
    rd = 0.4 * gen[:, PMAX] / t_delta
    ru_co = 0.3 * gen[:, PMAX]
    rd_co = 0.3 * gen[:, PMAX]
    sc = gencost[:, 1]
    hc = gencost[:, 2]

    for t in range(t_horizon):
        model.addConstr(gp.quicksum(pg[g, t] for g in range(ng)) == float(pd_sum[t]))
    for g in range(ng):
        for t in range(t_horizon):
            model.addConstr(pg[g, t] >= gen[g, PMIN] * x_int[g, t])
            model.addConstr(pg[g, t] <= gen[g, PMAX] * x_int[g, t])
        for t in range(1, t_horizon):
            model.addConstr(pg[g, t] - pg[g, t - 1] <= ru[g] * x_int[g, t - 1] + ru_co[g] * (1 - x_int[g, t - 1]))
            model.addConstr(pg[g, t - 1] - pg[g, t] <= rd[g] * x_int[g, t] + rd_co[g] * (1 - x_int[g, t]))
            model.addConstr(coc[g, t - 1] >= sc[g] * (x_int[g, t] - x_int[g, t - 1]))
            model.addConstr(coc[g, t - 1] >= hc[g] * (x_int[g, t - 1] - x_int[g, t]))
        for t in range(t_horizon):
            model.addConstr(cpower[g, t] >= gencost[g, -2] / t_delta * pg[g, t] + gencost[g, -1] / t_delta * x_int[g, t])

    obj = gp.quicksum(cpower[g, t] for g in range(ng) for t in range(t_horizon))
    obj += gp.quicksum(coc[g, t] for g in range(ng) for t in range(t_horizon - 1))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    return float(model.ObjVal) if model.status == GRB.OPTIMAL else float("inf")


def test_feasibility_check_on_milp(ppc, active_set_data, t_delta: float, n_check: int):
    print("\n" + "=" * 60)
    print("测试 1：check_uc_feasibility 对 MILP 解的正确性")
    print("=" * 60)

    n_pass = 0
    total = min(n_check, len(active_set_data))
    for sid in range(total):
        sample = active_set_data[sid]
        sample_id = sample.get("sample_id", sid)
        pd_data = np.array(sample["pd_data"])
        t0 = time.time()
        x_milp, obj_milp = solve_milp_basic(ppc, pd_data, t_delta)
        t_milp = time.time() - t0
        if x_milp is None:
            print(f"  样本 {sample_id}: MILP 未求得可用解，跳过")
            continue
        feas, reason = check_uc_feasibility(x_milp, ppc, pd_data, t_delta)
        n_pass += int(feas)
        print(f"  样本 {sample_id}: MILP obj={obj_milp:10.2f}, 可行={feas!s:5}, [{'PASS' if feas else 'FAIL'}]  (MILP {t_milp:.1f}s)")
        if not feas:
            print(f"    原因: {reason}")

    print(f"\n  结果: {n_pass}/{total} 通过")
    return n_pass == total


def test_recover_integer_solution(ppc, active_set_data, trainers, lambda_predictor, manager, t_delta: float, n_test: int, fp_iter: int):
    print("\n" + "=" * 60)
    print("测试 2：recover_integer_solution 端到端 pipeline")
    print("=" * 60)

    n_samples = min(n_test, len(active_set_data))
    results = []

    for sid in range(n_samples):
        sample = active_set_data[sid]
        sample_id = sample.get("sample_id", sid)
        pd_data = np.array(sample["pd_data"])
        print(f"\n--- 样本 {sample_id} (Pd_total={np.sum(pd_data[:, 0]):.1f} MW @ t=0) ---")

        t0 = time.time()
        x_milp, obj_milp = solve_milp_basic(ppc, pd_data, t_delta)
        t_milp = time.time() - t0
        print(f"  MILP: obj={obj_milp:.2f}, t={t_milp:.1f}s")

        t0 = time.time()
        x_fp, fp_success = recover_integer_solution(
            pd_data,
            trainers,
            lambda_predictor,
            ppc,
            t_delta,
            manager=manager,
            n_perturbations=5,
            conf_threshold=0.15,
            max_fp_iter=fp_iter,
            verbose=True,
            rng=np.random.default_rng(42 + sid),
        )
        t_fp = time.time() - t0

        if x_fp is None:
            feas_fp, reason_fp = False, "recover_integer_solution 返回 None"
            obj_fp = float("inf")
        else:
            feas_fp, reason_fp = check_uc_feasibility(x_fp, ppc, pd_data, t_delta)
            obj_fp = compute_obj_fixed_x(x_fp, ppc, pd_data, t_delta) if feas_fp else float("inf")

        gap = (
            (obj_fp - obj_milp) / max(abs(obj_milp), 1.0) * 100.0
            if obj_milp is not None and np.isfinite(obj_fp)
            else float("nan")
        )

        print(f"  FP:   obj={obj_fp:.2f}, feas={feas_fp}, gap={gap:+.2f}%, fp_success={fp_success}, t={t_fp:.1f}s [{'OK' if feas_fp else 'INFEAS'}]")
        if not feas_fp:
            print(f"    原因: {reason_fp}")

        results.append(
            {
                "sid": sample_id,
                "success": bool(fp_success),
                "feas": bool(feas_fp),
                "obj_milp": obj_milp,
                "obj_fp": obj_fp,
                "gap": gap,
                "t_milp": t_milp,
                "t_fp": t_fp,
            }
        )

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    n_feas = sum(1 for r in results if r["feas"])
    n_success = sum(1 for r in results if r["success"])
    gaps = [r["gap"] for r in results if r["feas"] and not np.isnan(r["gap"])]
    print(f"  样本数:       {n_samples}")
    print(f"  可行率:       {n_feas}/{n_samples}  ({100 * n_feas / max(n_samples, 1):.1f}%)")
    print(f"  FP 收敛率:    {n_success}/{n_samples}  ({100 * n_success / max(n_samples, 1):.1f}%)")
    if gaps:
        print(f"  Gap (vs MILP): 平均={np.mean(gaps):+.2f}%, 最大={np.max(gaps):+.2f}%, 最小={np.min(gaps):+.2f}%")
    print(f"  FP 平均耗时:  {np.mean([r['t_fp'] for r in results]):.1f}s")
    print(f"  MILP 平均耗时:{np.mean([r['t_milp'] for r in results]):.1f}s")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="评估 feasibility pump，输出格式兼容 agentic_fp_optimizer.py")
    parser.add_argument("--case", default="case30", choices=["case14", "case30", "case39"])
    parser.add_argument("--n-check", type=int, default=1, help="MILP 可行性检查样本数")
    parser.add_argument("--n-test", type=int, default=2, help="端到端评估样本数")
    parser.add_argument("--sample-range", help="指定样本范围，格式 start:end，采用左闭右开切片语义")
    parser.add_argument("--synthetic-samples", type=int, default=4, help="无 active_set 时生成的合成样本数")
    parser.add_argument("--time-periods", type=int, default=24, help="合成样本的时间长度 T")
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--fp-iter", type=int, default=30)
    parser.add_argument("--no-bcd-surrogate", action="store_true", help="禁用 BCD 联合代理，仅使用 subproblem surrogate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overall_t0 = time.time()

    print("=" * 60)
    print("Feasibility Pump 自动评估")
    print("=" * 60)
    print(f"工作目录: {ROOT}")
    print(f"案例: {args.case}")

    ppc = _build_ppc(args.case)
    surrogate_dir = _resolve_surrogate_model_dir(args.case)
    bcd_model_path = None if args.no_bcd_surrogate else _resolve_bcd_model_path(args.case)
    sample_range = _parse_sample_range(args.sample_range)

    sample_budget = max(args.synthetic_samples, args.n_check, args.n_test)
    active_set_data, data_path, used_synthetic = _load_active_set_data(
        args.case,
        ppc,
        args.time_periods,
        sample_budget,
        sample_range=sample_range,
    )
    print(f"样本来源: {'synthetic' if used_synthetic else data_path}")
    print(f"样本数: {len(active_set_data)}")

    lambda_predictor, trainers = load_models(ppc, active_set_data, args.t_delta, surrogate_dir)
    manager = load_bcd_manager(ppc, active_set_data, args.t_delta, bcd_model_path, trainers)

    test1_pass = test_feasibility_check_on_milp(ppc, active_set_data, args.t_delta, args.n_check)
    results = test_recover_integer_solution(
        ppc,
        active_set_data,
        trainers,
        lambda_predictor,
        manager,
        args.t_delta,
        args.n_test,
        args.fp_iter,
    )

    print(f"\n总耗时: {time.time() - overall_t0:.1f}s")
    print("=" * 60)

    n_feas = sum(1 for r in results if r["feas"])
    n_total = len(results)
    if test1_pass and n_feas == n_total:
        print("全部测试通过")
        return 0

    print(f"部分测试未通过 (check={test1_pass}, feas={n_feas}/{n_total})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
