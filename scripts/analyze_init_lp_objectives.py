#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""init LP 之后（或任意给定 \((x,pg,coc)\)）分解 ``obj_primal`` / ``obj_opt``，与 ``SubproblemSurrogateTrainer.cal_viol_components`` 一致。

理论要点（为何 \(x=x_{\mathrm{true}}\) 时并非「只有代理 obj_opt 且很小」）
------------------------------------------------------------------------

**Primal 块总目标**（Gurobi）近似为::

    rho_primal * obj_primal + rho_opt * obj_opt + rho_binary * obj_binary + rho_prox * prox

1. **若** \(x=x_{\mathrm{true}}\)、\(pg,coc\) 为 **init LP 在固定 \(x_{\mathrm{true}}\) 下的最优解**，
   则 **不含代理** 的 **物理** 线性约束应可行，故 **物理部分的 ``obj_primal``**（relu 违反）应为 **0**。
   **代理** 部分 ``obj_primal`` 仍为 \(\sum_k \max(0,\mathrm{lhs}_k-\delta_k)\)，是否为零取决于 \(\alpha,\beta,\gamma,\delta\) 与 \(x_{\mathrm{true}}\)，
   与 δ 锚定/训练有关，**不是**自动为小。

2. **``obj_opt`` 不是「只有代理」**。当前实现里与 **\(x\) 的界** 相关项为::

       \sum_t  x_t |\lambda^{\mathrm{lower}}_{x,t}|
             + (1-x_t)|\lambda^{\mathrm{upper}}_{x,t}|

   在 **0/1** \(x\) 下，每个时段 **必有一项非零**（除非两侧 λ 都为 0）。init 时 \(\lambda_x\) 来自 **\(x\) 固定约束的影子价格拆分**
   （``_apply_x_fix_dual_init``），通常 **不小**，因此 **即便物理松弛全为 0，``obj_opt`` 仍可很大**。
   这与「标准互补松弛 \(\lambda^\top s=0\)」不是同一公式。

3. **因此**：\(x=x_{\mathrm{true}}\) **既不蕴含** 总目标只剩 ``obj_opt``，**也不蕴含** 只有代理 ``obj_opt`` 可控；
   **``obj_binary``** 可为 0，**``obj_primal``** 可能只剩代理违反，**``obj_opt``** 仍含 **大额的 \(x\) 加权项** 与其它 \(|slack|\cdot|\lambda|\) 项。

4. **原始 \(\delta\) vs 有效系数（代码语义）**：``SubproblemSurrogateTrainer._apply_surrogate_direction_to_params`` 对
   \(\alpha,\beta,\gamma,\delta\) 乘以同一 **逐行因子** \(f_k=\texttt{sign}_k\times\texttt{curriculum}_k\)。
   因而 \(\mathrm{lhs}_{\mathrm{eff}}=f_k\,\mathrm{lhs}_{\mathrm{raw}}\)、\(\delta_{\mathrm{eff}}=f_k\,\delta_{\mathrm{raw}}\)（``build_surrogate_constraint_expression`` 对 \(x\) 线性），
   **\(\mathrm{relu}(\mathrm{lhs}_{\mathrm{eff}}-\delta_{\mathrm{eff}})=\mathrm{relu}(f_k(\mathrm{lhs}_{\mathrm{raw}}-\delta_{\mathrm{raw}}))\)**。
   若在 **未乘 \(f_k\)** 的 raw  slack 上断言「无违背」，会与 ``cal_viol_components`` / primal 块 **不一致**；分解中的 ``raw_eff_primal_relu_consistency`` 用于自检二者是否对齐。

5. **sign4 vs single**：按 ``constraint_offsets[k]`` 区分 — ``(0,1,2)`` 为 sign4 族模板；``(0,)`` 为 single-time。``other`` 为敏感模式等非常规模板。

用法
----

在已成功构造 ``SubproblemSurrogateTrainer``、且尚未被 BCD 改写状态前（或任意调试断点）::

    from scripts.analyze_init_lp_objectives import analyze_init_lp_state
    analyze_init_lp_state(trainer, sample_id=0, at_x_true=True)

``at_x_true=True``：用 ``active_set_data[s]['x_true']`` 与当前 ``trainer.pg[s], trainer.coc[s]`` 评估分解
（适用于 init LP 在钉死 \(x_{\mathrm{true}}\) 下求解的情形；若你故意换了 \(pg\) 而未重解 LP，物理项会爆）。

``at_x_true=False``：用当前 ``trainer.x[s], pg[s], coc[s]``（与 ``cal_viol_components`` 一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypower.idx_gen import PMIN, PMAX  # noqa: E402

from src.uc_NN_subproblem import (  # noqa: E402
    SURROGATE_SINGLE_TIME_OFFSETS,
    SURROGATE_TRIPLE_WINDOW_OFFSETS,
    build_surrogate_constraint_expression,
)


def _empty_surrogate_kind_bucket() -> dict[str, Any]:
    return {
        "primal_relu_sum": 0.0,
        "opt_mu_weighted_sum": 0.0,
        "n_rows": 0,
        "n_rows_positive_viol": 0,
        "max_primal_relu_row": 0.0,
        "argmax_k": None,
    }


def _surrogate_row_kind(constraint_offset_k: tuple[int, ...]) -> str:
    """与布局一致：triple 窗口为 sign4 族；单时段 (0,) 为 single。"""
    if constraint_offset_k == tuple(SURROGATE_SINGLE_TIME_OFFSETS):
        return "single"
    if constraint_offset_k == tuple(SURROGATE_TRIPLE_WINDOW_OFFSETS):
        return "sign4"
    return "other"


def _feed_kind_bucket(bucket: dict[str, Any], viol: float, oterm: float, k: int) -> None:
    bucket["primal_relu_sum"] += viol
    bucket["opt_mu_weighted_sum"] += oterm
    bucket["n_rows"] += 1
    if viol > 1e-15:
        bucket["n_rows_positive_viol"] += 1
    if viol > float(bucket["max_primal_relu_row"]):
        bucket["max_primal_relu_row"] = viol
        bucket["argmax_k"] = int(k)


def _one_sample_breakdown(
    trainer: Any,
    sample_id: int,
    x_val: np.ndarray,
    pg_val: np.ndarray,
    coc_val: np.ndarray,
) -> dict[str, Any]:
    g = int(trainer.unit_id)
    x_val = np.asarray(x_val, dtype=float).reshape(-1)
    pg_val = np.asarray(pg_val, dtype=float).reshape(-1)
    coc_val = np.asarray(coc_val, dtype=float).reshape(-1)

    a_raw = np.asarray(trainer.alpha_values[sample_id], dtype=float)
    b_raw = np.asarray(trainer.beta_values[sample_id], dtype=float)
    g_raw = np.asarray(trainer.gamma_values[sample_id], dtype=float)
    d_raw = np.asarray(trainer.delta_values[sample_id], dtype=float)
    mu_vals = np.abs(np.asarray(trainer.mu[sample_id], dtype=float))
    ae, be, ge, de = trainer._apply_surrogate_direction_to_params(a_raw, b_raw, g_raw, d_raw)

    nc = int(trainer.num_coupling_constraints)
    signs = np.asarray(trainer._get_surrogate_direction_signs(nc), dtype=float)
    factors = np.asarray(trainer._sign4_curriculum_factors(nc), dtype=float)
    f_arr = signs * factors

    lam_inh = trainer.lambda_inherent[sample_id]
    Pmin_v = float(trainer.gen[g, PMIN])
    Pmax_v = float(trainer.gen[g, PMAX])
    Ru_v = float(trainer.Ru_all[g])
    Rd_v = float(trainer.Rd_all[g])
    Ru_co_v = float(trainer.Ru_co_all[g])
    Rd_co_v = float(trainer.Rd_co_all[g])
    sc_v = 0.0 if trainer.ignore_startup_shutdown_costs else float(trainer.gencost[g, 1])
    shc_v = 0.0 if trainer.ignore_startup_shutdown_costs else float(trainer.gencost[g, 2])
    Ton_v = int(trainer.subproblem_Ton)
    Toff_v = int(trainer.subproblem_Toff)
    T = int(trainer.T)

    sensitive_t = trainer.sensitive_timesteps[sample_id]
    constraint_offsets = trainer._constraint_offsets_for_sample(sample_id)
    n_sur = min(
        len(sensitive_t),
        len(constraint_offsets),
        nc,
        ae.shape[0],
        mu_vals.shape[0],
        d_raw.shape[0],
    )

    buckets = {
        "sign4": _empty_surrogate_kind_bucket(),
        "single": _empty_surrogate_kind_bucket(),
        "other": _empty_surrogate_kind_bucket(),
    }
    primal_sur = 0.0
    opt_sur = 0.0
    max_raw_eff_viol_diff = 0.0
    top_rows: list[tuple[float, int, str, float, float, float, float, float, float]] = []

    for k in range(n_sur):
        ts = int(sensitive_t[k])
        off_t = tuple(int(x) for x in constraint_offsets[k])
        kind = _surrogate_row_kind(off_t)
        lhs_e = float(
            build_surrogate_constraint_expression(
                x_val,
                ts,
                constraint_offsets[k],
                float(ae[k]),
                float(be[k]),
                float(ge[k]),
                T,
            )
        )
        lhs_r = float(
            build_surrogate_constraint_expression(
                x_val,
                ts,
                constraint_offsets[k],
                float(a_raw[k]),
                float(b_raw[k]),
                float(g_raw[k]),
                T,
            )
        )
        dk_e = float(de[k])
        dk_r = float(d_raw[k])
        viol = max(0.0, lhs_e - dk_e)
        oterm = abs(lhs_e - dk_e) * float(mu_vals[k])
        primal_sur += viol
        opt_sur += oterm
        _feed_kind_bucket(buckets[kind], viol, oterm, k)

        fk = float(f_arr[k]) if k < f_arr.shape[0] else 1.0
        viol_from_f_scale = max(0.0, fk * lhs_r - fk * dk_r)
        max_raw_eff_viol_diff = max(max_raw_eff_viol_diff, abs(viol - viol_from_f_scale))
        top_rows.append((viol, k, kind, lhs_e, dk_e, fk, lhs_r, dk_r, viol_from_f_scale))

    top_rows.sort(key=lambda z: z[0], reverse=True)
    top_rows_detail = [
        {
            "k": int(row[1]),
            "kind": row[2],
            "primal_relu": float(row[0]),
            "lhs_eff": float(row[3]),
            "delta_eff": float(row[4]),
            "f_k": float(row[5]),
            "lhs_raw": float(row[6]),
            "delta_raw": float(row[7]),
            "relu_from_f_raw": float(row[8]),
        }
        for row in top_rows[:12]
    ]

    surrogate_by_kind = {
        "constraint_generation_strategy": str(trainer.constraint_generation_strategy),
        "sign4": buckets["sign4"],
        "single": buckets["single"],
        "other": buckets["other"],
        "raw_eff_primal_relu_consistency": {
            "max_abs_diff": float(max_raw_eff_viol_diff),
            "ok": bool(max_raw_eff_viol_diff < 1e-7),
            "note": "viol=relu(lhs_eff−δ_eff) 应等于 max(0,f*(lhs_raw−δ_raw))；与 cal_viol_components 一致。",
        },
        "top_rows_by_primal_relu": top_rows_detail,
    }

    primal_pglo = primal_pgup = primal_ru = primal_rd = 0.0
    primal_sc = primal_shc = 0.0
    primal_mon = primal_moff = 0.0

    for t in range(T):
        primal_pglo += max(0.0, Pmin_v * x_val[t] - pg_val[t])
        primal_pgup += max(0.0, pg_val[t] - Pmax_v * x_val[t])

    for t in range(1, T):
        primal_ru += max(
            0.0,
            pg_val[t] - pg_val[t - 1] - Ru_v * x_val[t - 1] - Ru_co_v * (1 - x_val[t - 1]),
        )
        primal_rd += max(
            0.0,
            pg_val[t - 1] - pg_val[t] - Rd_v * x_val[t] - Rd_co_v * (1 - x_val[t]),
        )
        primal_sc += max(0.0, sc_v * (x_val[t] - x_val[t - 1]) - coc_val[t - 1])
        primal_shc += max(0.0, shc_v * (x_val[t - 1] - x_val[t]) - coc_val[t - 1])

    for tau in range(1, Ton_v + 1):
        for t1 in range(T - tau):
            primal_mon += max(0.0, x_val[t1 + 1] - x_val[t1] - x_val[t1 + tau])

    for tau in range(1, Toff_v + 1):
        for t1 in range(T - tau):
            primal_moff += max(
                0.0, -x_val[t1 + 1] + x_val[t1] - (1 - x_val[t1 + tau])
            )

    primal_phys = (
        primal_pglo
        + primal_pgup
        + primal_ru
        + primal_rd
        + primal_sc
        + primal_shc
        + primal_mon
        + primal_moff
    )
    primal_total = primal_sur + primal_phys

    opt = {
        "surrogate": opt_sur,
        "pg_lower": 0.0,
        "pg_upper": 0.0,
        "x_lower": 0.0,
        "x_upper": 0.0,
        "ramp_up": 0.0,
        "ramp_down": 0.0,
        "min_on": 0.0,
        "min_off": 0.0,
        "start_cost": 0.0,
        "shut_cost": 0.0,
        "coc_nonneg": 0.0,
    }

    if lam_inh is not None:
        for t in range(T):
            opt["pg_lower"] += abs(Pmin_v * x_val[t] - pg_val[t]) * abs(
                float(lam_inh["lambda_pg_lower"][t])
            )
            opt["pg_upper"] += abs(pg_val[t] - Pmax_v * x_val[t]) * abs(
                float(lam_inh["lambda_pg_upper"][t])
            )
            opt["x_lower"] += x_val[t] * abs(float(lam_inh["lambda_x_lower"][t]))
            opt["x_upper"] += (1 - x_val[t]) * abs(float(lam_inh["lambda_x_upper"][t]))

        for t in range(1, T):
            ru_expr = pg_val[t] - pg_val[t - 1] - Ru_v * x_val[t - 1] - Ru_co_v * (
                1 - x_val[t - 1]
            )
            rd_expr = pg_val[t - 1] - pg_val[t] - Rd_v * x_val[t] - Rd_co_v * (
                1 - x_val[t]
            )
            start_expr = sc_v * (x_val[t] - x_val[t - 1]) - coc_val[t - 1]
            shut_expr = shc_v * (x_val[t - 1] - x_val[t]) - coc_val[t - 1]
            opt["ramp_up"] += abs(ru_expr) * abs(float(lam_inh["lambda_ramp_up"][t - 1]))
            opt["ramp_down"] += abs(rd_expr) * abs(float(lam_inh["lambda_ramp_down"][t - 1]))
            opt["start_cost"] += abs(start_expr) * abs(
                float(lam_inh["lambda_start_cost"][t - 1])
            )
            opt["shut_cost"] += abs(shut_expr) * abs(
                float(lam_inh["lambda_shut_cost"][t - 1])
            )
            opt["coc_nonneg"] += coc_val[t - 1] * abs(
                float(lam_inh["lambda_coc_nonneg"][t - 1])
            )

        for tau in range(1, Ton_v + 1):
            for t1 in range(T - tau):
                expr = x_val[t1 + 1] - x_val[t1] - x_val[t1 + tau]
                opt["min_on"] += abs(expr) * abs(
                    float(lam_inh["lambda_min_on"][tau - 1][t1])
                )

        for tau in range(1, Toff_v + 1):
            for t1 in range(T - tau):
                expr = -x_val[t1 + 1] + x_val[t1] - (1 - x_val[t1 + tau])
                opt["min_off"] += abs(expr) * abs(
                    float(lam_inh["lambda_min_off"][tau - 1][t1])
                )

    opt_nosur = sum(v for k, v in opt.items() if k != "surrogate")
    opt["total"] = opt["surrogate"] + opt_nosur

    x_true = trainer.active_set_data[sample_id].get("x_true")
    obj_binary = 0.0
    if x_true is not None:
        xt = np.asarray(x_true, dtype=float).reshape(-1)
        obj_binary = float(np.sum(np.abs(x_val - xt)))

    return {
        "x_used_note": "x_true from sample" if x_true is not None else "x only",
        "obj_binary_l1_vs_x_true": obj_binary,
        "primal": {
            "surrogate": primal_sur,
            "physical_total": primal_phys,
            "physical_pg_lower": primal_pglo,
            "physical_pg_upper": primal_pgup,
            "physical_ramp_up": primal_ru,
            "physical_ramp_down": primal_rd,
            "physical_start": primal_sc,
            "physical_shut": primal_shc,
            "physical_min_on": primal_mon,
            "physical_min_off": primal_moff,
            "total": primal_total,
        },
        "obj_opt": opt,
        "rho_scaled": {
            "rho_primal * primal_total": float(trainer.rho_primal) * primal_total,
            "rho_opt * obj_opt_total": float(trainer.rho_opt) * float(opt["total"]),
            "rho_binary * obj_binary": float(trainer.rho_binary) * obj_binary,
        },
        "surrogate_by_kind": surrogate_by_kind,
    }


def analyze_init_lp_state(
    trainer: Any,
    sample_id: int = 0,
    *,
    at_x_true: bool = False,
    verbose: bool = True,
) -> Mapping[str, Any]:
    """打印（可选）并返回单样本分解。"""
    sid = int(sample_id)
    if at_x_true:
        xt = trainer.active_set_data[sid].get("x_true")
        if xt is None:
            raise ValueError(f"sample {sid} has no x_true in active_set_data")
        x_val = np.asarray(xt, dtype=float)
    else:
        x_val = np.asarray(trainer.x[sid], dtype=float)

    pg_val = np.asarray(trainer.pg[sid], dtype=float)
    coc_val = np.asarray(trainer.coc[sid], dtype=float)

    out = _one_sample_breakdown(trainer, sid, x_val, pg_val, coc_val)

    if verbose:
        u = int(trainer.unit_id)
        print(f"=== init/BCD 状态分解 unit={u} sample={sid} at_x_true={at_x_true} ===", flush=True)
        print(f"obj_binary (L1 vs x_true): {out['obj_binary_l1_vs_x_true']:.6g}", flush=True)
        p = out["primal"]
        print(
            f"obj_primal: total={p['total']:.6g}  surrogate={p['surrogate']:.6g}  "
            f"physical={p['physical_total']:.6g}",
            flush=True,
        )
        sk = out.get("surrogate_by_kind")
        if sk is not None:
            cons = sk["raw_eff_primal_relu_consistency"]
            print(
                f"  surrogate split: strategy={sk['constraint_generation_strategy']}  "
                f"raw/eff relu 一致 max|Δ|={cons['max_abs_diff']:.3e} ok={cons['ok']}",
                flush=True,
            )
            for name in ("sign4", "single", "other"):
                b = sk[name]
                if b["n_rows"] <= 0:
                    continue
                print(
                    f"    [{name}] rows={b['n_rows']} viol_rows={b['n_rows_positive_viol']}  "
                    f"sum_relu={b['primal_relu_sum']:.6g} sum_|slack|·μ={b['opt_mu_weighted_sum']:.6g}  "
                    f"max_relu={b['max_primal_relu_row']:.6g} @k={b['argmax_k']}",
                    flush=True,
                )
            tr = sk.get("top_rows_by_primal_relu") or []
            shown = 0
            for d in tr:
                if float(d["primal_relu"]) <= 1e-14:
                    continue
                print(
                    f"      k={d['k']} {d['kind']} relu={d['primal_relu']:.6g}  "
                    f"lhs_eff={d['lhs_eff']:.6g} δ_eff={d['delta_eff']:.6g}  "
                    f"f={d['f_k']:.6g} lhs_raw={d['lhs_raw']:.6g} δ_raw={d['delta_raw']:.6g} "
                    f"relu_f_raw={d['relu_from_f_raw']:.6g}",
                    flush=True,
                )
                shown += 1
                if shown >= 8:
                    break
        if p["physical_total"] > 1e-9:
            print(
                f"  phys detail: pg_lo={p['physical_pg_lower']:.4g} pg_up={p['physical_pg_upper']:.4g} "
                f"ru={p['physical_ramp_up']:.4g} rd={p['physical_ramp_down']:.4g} "
                f"sc={p['physical_start']:.4g} shc={p['physical_shut']:.4g} "
                f"mon={p['physical_min_on']:.4g} moff={p['physical_min_off']:.4g}",
                flush=True,
            )
        oo = out["obj_opt"]
        print(
            f"obj_opt: total={oo['total']:.6g}  surrogate={oo['surrogate']:.6g}  "
            f"non_surrogate={oo['total'] - oo['surrogate']:.6g}",
            flush=True,
        )
        print(
            f"  opt detail: x_lo={oo['x_lower']:.4g} x_up={oo['x_upper']:.4g} "
            f"pg_lo={oo['pg_lower']:.4g} pg_up={oo['pg_upper']:.4g} "
            f"ru={oo['ramp_up']:.4g} rd={oo['ramp_down']:.4g}",
            flush=True,
        )
        rs = out["rho_scaled"]
        print(
            f"rho-weighted (no prox): rho_primal*primal={rs['rho_primal * primal_total']:.6g} "
            f"rho_opt*opt={rs['rho_opt * obj_opt_total']:.6g} "
            f"rho_binary*binary={rs['rho_binary * obj_binary']:.6g}",
            flush=True,
        )

        if trainer.n_samples == 1:
            op, _, _, _, _, oopt = trainer.cal_viol_components()
            print(
                f"(check) cal_viol_components totals: obj_primal={op:.6g} obj_opt={oopt:.6g}",
                flush=True,
            )

    return out


def main() -> None:
    print(__doc__)
    print(
        "未加载真实 trainer。请在训练脚本中 import analyze_init_lp_state(trainer, 0)。",
        flush=True,
    )


if __name__ == "__main__":
    main()
