#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁剪 case118 refined active set 中的对偶电价，精简 JSON 结构。

功能
----
1. 从 ppc['gencost'][:, -2] 读取各机组线性边际成本（$/MWh），计算：
       price_cap = 2.0 * max(gencost[:, -2] / T_delta)
       price_min = 0.0
2. 遍历所有样本，将 lambda_pg_electricity_price 裁剪到 [price_min, price_cap]。
3. 从每个样本中移除以下冗余对偶字段（三条训练路径均不依赖）：
       'lambda'               （含 lambda_power_balance / lambda_dcpf_upper/lower 等）
       'lambda_refresh_source'
4. 保留 lambda_pg_electricity_price（裁剪后），其余非-lambda 字段原样保留。
5. 打印裁剪前后统计，写出新 JSON 文件（后缀 _price_only_clipped.json）。

用法（仓库根目录）::

    python clip_dual_prices_case118.py

也可覆盖源文件路径或 T_delta::

    python clip_dual_prices_case118.py --src result/commitment_clustering/xxx.json
    python clip_dual_prices_case118.py --t-delta 1.0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from case_registry import get_case_ppc  # noqa: E402
from dataset_json_utils import load_v3_active_set_json  # noqa: E402

# ─────────────────────── 默认配置（直接运行时使用） ──────────────────────────

DEFAULT_SRC = (
    "result/commitment_clustering/"
    "pattern_library_case118_K10_20260418_032025"
    "_active_set_like_refined_20260418_032025.json"
)
DEFAULT_CASE = "case118"
DEFAULT_T_DELTA = 1.0
# 写出文件后缀（追加在 src stem 之后）
OUT_SUFFIX = "_price_only_clipped"


# ─────────────────────── 辅助函数 ────────────────────────────────────────────

def _compute_price_cap(ppc: dict, T_delta: float) -> float:
    """从 gencost 计算电价上限 = 2 × 最大机组线性边际成本（$/MWh）。

    gencost 格式（每行 7 列）：
        [type, startup, shutdown, n_terms, c2, c1(=linear_cost), c0(=no_load)]
    故 gencost[:, -2] == gencost[:, 5] 为线性成本系数，单位为 $/MWh（T_delta=1）。
    """
    gencost = np.asarray(ppc["gencost"], dtype=float)
    if gencost.ndim != 2 or gencost.shape[1] < 2:
        raise ValueError(f"ppc['gencost'] 形状异常: {gencost.shape}")
    linear_costs = gencost[:, -2]  # gencost[:, 5]，$/MWh
    max_cost = float(np.max(linear_costs)) / float(T_delta)
    price_cap = 2.0 * max_cost
    return price_cap


def _read_price_matrix(sample: Dict[str, Any], ng: int, T: int) -> Optional[np.ndarray]:
    """从样本中读取 lambda_pg_electricity_price，返回 (ng, T) ndarray 或 None。"""
    raw = sample.get("lambda_pg_electricity_price")
    if raw is None:
        # 尝试从 lambda dict 中读取 lambda_pg_effective 或 lambda_pg_electricity_price
        lam_dict = sample.get("lambda")
        if isinstance(lam_dict, dict):
            raw = lam_dict.get("lambda_pg_electricity_price") or lam_dict.get("lambda_pg_effective")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.shape == (ng, T):
        return arr
    if arr.shape == (T, ng):
        return arr.T
    # 尝试 reshape
    if arr.size == ng * T:
        return arr.reshape(ng, T)
    return None


def _clip_samples(
    samples: List[Dict[str, Any]],
    ng: int,
    T: int,
    price_min: float,
    price_cap: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """裁剪所有样本的电价，移除冗余 lambda 字段，返回处理后样本列表与统计信息。"""
    _REMOVE_KEYS = {"lambda", "lambda_refresh_source"}

    n_clipped_total = 0
    n_missing = 0
    before_vals: List[float] = []
    after_vals: List[float] = []

    for idx, sample in enumerate(samples):
        price = _read_price_matrix(sample, ng, T)
        if price is None:
            n_missing += 1
            print(f"  [WARN] 样本 {idx} 缺少 lambda_pg_electricity_price，跳过裁剪", flush=True)
            # 移除冗余字段，但不写入裁剪后的价格
            for k in _REMOVE_KEYS:
                sample.pop(k, None)
            continue

        before_vals.extend(price.ravel().tolist())

        clipped = np.clip(price, price_min, price_cap)
        n_clipped_total += int(np.sum(price != clipped))

        after_vals.extend(clipped.ravel().tolist())

        # 写回裁剪后的价格（保持 list-of-lists 格式，与 JSON 惯例一致）
        sample["lambda_pg_electricity_price"] = clipped.tolist()

        # 移除冗余对偶字段
        for k in _REMOVE_KEYS:
            sample.pop(k, None)

    before_arr = np.asarray(before_vals, dtype=float)
    after_arr = np.asarray(after_vals, dtype=float)

    stats = {
        "n_samples": len(samples),
        "n_missing_price": n_missing,
        "n_clipped_values": n_clipped_total,
        "price_cap": price_cap,
        "price_min": price_min,
        "before": {
            "min": float(np.min(before_arr)) if before_arr.size else float("nan"),
            "max": float(np.max(before_arr)) if before_arr.size else float("nan"),
            "mean": float(np.mean(before_arr)) if before_arr.size else float("nan"),
            "std": float(np.std(before_arr)) if before_arr.size else float("nan"),
        },
        "after": {
            "min": float(np.min(after_arr)) if after_arr.size else float("nan"),
            "max": float(np.max(after_arr)) if after_arr.size else float("nan"),
            "mean": float(np.mean(after_arr)) if after_arr.size else float("nan"),
            "std": float(np.std(after_arr)) if after_arr.size else float("nan"),
        },
    }
    return samples, stats


def _print_stats(stats: Dict[str, Any], ppc: dict, T_delta: float) -> None:
    gencost = np.asarray(ppc["gencost"], dtype=float)
    linear_costs = gencost[:, -2] / float(T_delta)
    print(f"\n机组线性边际成本（$/MWh）: "
          f"min={np.min(linear_costs):.4g}, max={np.max(linear_costs):.4g}, "
          f"mean={np.mean(linear_costs):.4g}", flush=True)
    print(f"price_min={stats['price_min']:.4g}, "
          f"price_cap={stats['price_cap']:.4g}  "
          f"[= 2 × {np.max(linear_costs):.4g}]", flush=True)
    print(f"\n裁剪前统计: min={stats['before']['min']:.4g}, "
          f"max={stats['before']['max']:.4g}, "
          f"mean={stats['before']['mean']:.4g}, "
          f"std={stats['before']['std']:.4g}", flush=True)
    print(f"裁剪后统计: min={stats['after']['min']:.4g}, "
          f"max={stats['after']['max']:.4g}, "
          f"mean={stats['after']['mean']:.4g}, "
          f"std={stats['after']['std']:.4g}", flush=True)
    print(f"\n共裁剪 {stats['n_clipped_values']} 个元素 "
          f"（{stats['n_samples']} 样本 × ng × T = "
          f"{stats['n_samples'] * (stats['n_clipped_values'] // max(stats['n_samples'], 1) + 1)} 预估总元素）",
          flush=True)
    if stats["n_missing_price"] > 0:
        print(f"[WARN] {stats['n_missing_price']} 个样本缺少电价数据", flush=True)


# ─────────────────────── 主函数 ──────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Clip lambda_pg_electricity_price to [0, 2×max_unit_cost] "
                    "and strip dcpf/balance lambda fields from case118 active set JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="Source active-set JSON path (relative to repo root or absolute).")
    p.add_argument("--case", type=str, default=DEFAULT_CASE,
                   help="Case name for get_case_ppc.")
    p.add_argument("--t-delta", type=float, default=DEFAULT_T_DELTA, dest="T_delta",
                   help="Time step duration (hours); affects marginal cost $/MWh scaling.")
    p.add_argument("--out-suffix", type=str, default=OUT_SUFFIX,
                   help="Suffix appended to the source file stem for the output file.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print statistics but do not write output file.")
    args = p.parse_args()

    # 解析源文件路径
    src_path = args.src if isinstance(args.src, Path) else Path(args.src)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    if not src_path.exists():
        print(f"错误：源 JSON 未找到: {src_path}", flush=True)
        sys.exit(1)

    # 确定输出路径
    out_path = src_path.parent / (src_path.stem + args.out_suffix + src_path.suffix)

    print("=" * 72, flush=True)
    print("clip_dual_prices_case118.py", flush=True)
    print(f"  src  : {src_path}", flush=True)
    print(f"  out  : {out_path}", flush=True)
    print(f"  case : {args.case}, T_delta={args.T_delta}", flush=True)
    print("=" * 72, flush=True)

    # 加载 ppc
    print(f"加载 ppc: {args.case} ...", flush=True)
    ppc = get_case_ppc(args.case)

    # 计算电价上限
    price_cap = _compute_price_cap(ppc, args.T_delta)
    price_min = 0.0
    print(f"price_cap = {price_cap:.6g} $/MWh  (= 2 × max_linear_cost / T_delta)", flush=True)

    # 加载 active set
    print(f"加载 active set JSON: {src_path.name} ...", flush=True)
    samples = load_v3_active_set_json(src_path, announce=print)
    n_samples = len(samples)

    # 读取 ng, T
    from pypower.ext2int import ext2int
    ppc_int = ext2int(ppc)
    ng = int(ppc_int["gen"].shape[0])
    T = int(samples[0]["pd_data"].shape[1])
    print(f"ng={ng}, T={T}, n_samples={n_samples}", flush=True)

    # 裁剪
    print("\n裁剪中 ...", flush=True)
    samples, stats = _clip_samples(samples, ng, T, price_min, price_cap)

    # 打印统计
    _print_stats(stats, ppc, args.T_delta)

    if args.dry_run:
        print("\n[dry-run] 跳过文件写出", flush=True)
        return

    # 写出新 JSON（递归把 ndarray 转回 list，json 标准库不支持 ndarray）
    print(f"\n写出 {out_path} ...", flush=True)

    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = {"all_samples": _to_serializable(samples)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[OK] 已写出: {out_path.name}  ({size_mb:.1f} MB)", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
