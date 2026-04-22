import numpy as np
import os

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    cp = None
    CVXPY_AVAILABLE = False

try:
    import highspy  # noqa: F401
    HIGHSPY_AVAILABLE = True
except ImportError:
    HIGHSPY_AVAILABLE = False

from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A


LP_BACKEND_GUROBI = "gurobi"
LP_BACKEND_CVXPY_HIGHS = "cvxpy_highs"
SUPPORTED_LP_BACKENDS = (LP_BACKEND_GUROBI, LP_BACKEND_CVXPY_HIGHS)
SURROGATE_TRIPLE_WINDOW_OFFSETS = (0, 1, 2)
_CVXPY_HIGHS_INSTALLED_CACHE = None
_CVXPY_HIGHS_STATUS_CACHE = None


def normalize_lp_backend(lp_backend: str | None) -> str:
    resolved = LP_BACKEND_GUROBI if lp_backend is None else str(lp_backend).strip().lower()
    if resolved not in SUPPORTED_LP_BACKENDS:
        raise ValueError(
            f"Unsupported lp_backend: {lp_backend!r}. "
            f"Supported backends: {SUPPORTED_LP_BACKENDS}"
        )
    return resolved


def _cvxpy_highs_installed(force_refresh: bool = False) -> bool:
    global _CVXPY_HIGHS_INSTALLED_CACHE
    if _CVXPY_HIGHS_INSTALLED_CACHE is not None and not force_refresh:
        return _CVXPY_HIGHS_INSTALLED_CACHE
    if not CVXPY_AVAILABLE:
        _CVXPY_HIGHS_INSTALLED_CACHE = False
        return False
    try:
        _CVXPY_HIGHS_INSTALLED_CACHE = "HIGHS" in cp.installed_solvers()
    except Exception:
        _CVXPY_HIGHS_INSTALLED_CACHE = False
    return _CVXPY_HIGHS_INSTALLED_CACHE


def get_cvxpy_highs_status(force_refresh: bool = False) -> dict:
    global _CVXPY_HIGHS_STATUS_CACHE
    if _CVXPY_HIGHS_STATUS_CACHE is not None and not force_refresh:
        return dict(_CVXPY_HIGHS_STATUS_CACHE)
    status = {
        "cvxpy_available": CVXPY_AVAILABLE,
        "highspy_available": HIGHSPY_AVAILABLE,
        "highs_solver_available": _cvxpy_highs_installed(force_refresh=force_refresh),
    }
    _CVXPY_HIGHS_STATUS_CACHE = dict(status)
    return status


def is_lp_backend_available(lp_backend: str | None) -> bool:
    backend = normalize_lp_backend(lp_backend)
    if backend == LP_BACKEND_GUROBI:
        return True
    status = get_cvxpy_highs_status()
    return all(status.values())


def assert_lp_backend_available(lp_backend: str | None) -> None:
    backend = normalize_lp_backend(lp_backend)
    if backend == LP_BACKEND_GUROBI:
        return
    status = get_cvxpy_highs_status()
    missing = [name for name, ok in status.items() if not ok]
    if missing:
        raise RuntimeError(
            f"lp_backend={backend!r} is unavailable; missing requirements: {', '.join(missing)}"
        )


def _strict_cvxpy_highs_diagnostics_enabled() -> bool:
    """Enable strong diagnostics and abort on HiGHS failures."""
    flag = os.environ.get("STRICT_CVXPY_HIGHS", "0")
    return str(flag).strip().lower() in ("1", "true", "yes", "on")


def _array_finite_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {"size": 0, "min": None, "max": None, "abs_max": None, "l2": 0.0, "nan": 0, "inf": 0}
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    if finite_vals.size == 0:
        return {
            "size": int(arr.size),
            "min": None,
            "max": None,
            "abs_max": None,
            "l2": None,
            "nan": nan_count,
            "inf": inf_count,
        }
    abs_max = float(np.max(np.abs(finite_vals)))
    l2 = float(np.linalg.norm(finite_vals.reshape(-1), ord=2))
    return {
        "size": int(arr.size),
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "abs_max": abs_max,
        "l2": l2,
        "nan": nan_count,
        "inf": inf_count,
    }


def _require_finite(name: str, arr: np.ndarray, context: str = "") -> None:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return
    if not np.isfinite(arr).all():
        stats = _array_finite_stats(arr)
        ctx = f" ({context})" if context else ""
        raise ValueError(f"Non-finite values detected in {name}{ctx}: stats={stats}")


def _summarize_lambda_inherent(lam_inh: dict | None) -> dict:
    if lam_inh is None:
        return {"present": False}
    summary = {"present": True}
    for key in (
        "lambda_pg_lower",
        "lambda_pg_upper",
        "lambda_ramp_up",
        "lambda_ramp_down",
        "lambda_start_cost",
        "lambda_shut_cost",
        "lambda_coc_nonneg",
        "lambda_x_upper",
        "lambda_x_lower",
    ):
        try:
            summary[key] = _array_finite_stats(np.asarray(lam_inh.get(key)))
        except Exception:
            summary[key] = {"error": "unavailable"}
    return summary


def _solve_with_cvxpy_highs(problem, verbose: bool = False, threads: int = 1,
                             warm_start: bool = False) -> float:
    """Solve a CVXPY problem with HiGHS.

    Args:
        problem: CVXPY problem to solve.
        verbose: Pass verbose=True to HiGHS for debugging.
        threads: Number of threads for HiGHS to use.  Set >1 only when the
            caller guarantees no nested parallelism (e.g. the main-BCD
            single-process thread-pool path).  The default of 1 is safe for
            the subproblem process-pool path where HiGHS and Python multiprocessing
            would otherwise compete for cores.
        warm_start: When True CVXPY reuses cached solver data and passes the
            previous primal/dual solution as a hot-start.  Effective only when
            the SAME problem object is solved repeatedly (persistent problems
            built with cp.Parameter).
    """
    assert_lp_backend_available(LP_BACKEND_CVXPY_HIGHS)
    solver_name = getattr(cp, "HIGHS", "HIGHS")
    highs_threads = max(1, int(threads))
    highs_parallel = "on" if highs_threads > 1 else "off"
    try:
        return problem.solve(
            solver=solver_name,
            verbose=verbose,
            threads=highs_threads,
            parallel=highs_parallel,
            warm_start=warm_start,
        )
    except Exception as exc:
        if _strict_cvxpy_highs_diagnostics_enabled() and CVXPY_AVAILABLE:
            try:
                print("[STRICT_CVXPY_HIGHS] HiGHS solve failed; dumping problem info.", flush=True)
                try:
                    print(f"  status={getattr(problem, 'status', None)}", flush=True)
                except Exception:
                    pass
                try:
                    print(f"  size_metrics={getattr(problem, 'size_metrics', None)}", flush=True)
                except Exception:
                    pass
                print("  retrying once with verbose=True to capture HiGHS logs...", flush=True)
                _ = problem.solve(
                    solver=solver_name,
                    verbose=True,
                    threads=highs_threads,
                    parallel=highs_parallel,
                    warm_start=warm_start,
                )
            except Exception:
                pass
        raise exc


def _problem_is_optimal(problem) -> bool:
    return problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def _normalize_constraint_offsets(offsets) -> tuple[int, ...]:
    if offsets is None:
        return SURROGATE_TRIPLE_WINDOW_OFFSETS
    normalized = []
    for offset in offsets:
        try:
            offset_int = int(offset)
        except (TypeError, ValueError):
            continue
        if 0 <= offset_int <= 2 and offset_int not in normalized:
            normalized.append(offset_int)
    return tuple(normalized) if normalized else SURROGATE_TRIPLE_WINDOW_OFFSETS


def iterate_surrogate_constraint_terms(
    timestep: int,
    offsets,
    alpha_value,
    beta_value,
    gamma_value,
    horizon: int,
):
    active_offsets = _normalize_constraint_offsets(offsets)
    if 0 in active_offsets and 0 <= timestep < horizon:
        yield timestep, alpha_value
    if 1 in active_offsets and 0 <= timestep + 1 < horizon:
        yield timestep + 1, beta_value
    if 2 in active_offsets and 0 <= timestep + 2 < horizon:
        yield timestep + 2, gamma_value


def build_surrogate_constraint_expression(
    x_values,
    timestep: int,
    offsets,
    alpha_value,
    beta_value,
    gamma_value,
    horizon: int,
):
    expr = 0
    for time_idx, coeff in iterate_surrogate_constraint_terms(
        timestep,
        offsets,
        alpha_value,
        beta_value,
        gamma_value,
        horizon,
    ):
        expr += coeff * x_values[time_idx]
    return expr


def _recover_unit_x_from_sample(trainer, sample_id: int) -> np.ndarray:
    """恢复真值启停行向量 x_init（长度 T），供 init LP 的「真值钉死」约束使用。

    与 ``uc_NN_subproblem`` 中 Gurobi 初值路径**必须**语义一致。

    规则（重要）:
    - 若存在 ``unit_commitment_matrix``：先取完整行 ``uc[g, :T]`` 作为基线，避免
      pattern_library 等仅**稀疏** ``active_set`` 时、未在列表中的 ``(g,t)`` 被
      误当作 0 而与矩阵真值矛盾。
    - 若同时存在 ``active_set``：对列出的 ``(g_idx,t)`` 再**覆盖**相应分量（可修正
      与矩阵不一致的单元）。
    - 仅 ``active_set`` 时：从全零起按列表填写（与旧行为相同）。
    """
    g = trainer.unit_id
    T = trainer.T
    sample = trainer.active_set_data[sample_id]

    if "unit_commitment_matrix" in sample:
        uc = sample["unit_commitment_matrix"]
        if isinstance(uc, np.ndarray) and uc.ndim == 2 and g < uc.shape[0]:
            n_t = int(min(uc.shape[1], T))
            x_init = np.zeros(T, dtype=float)
            x_init[:n_t] = np.asarray(uc[g, :n_t], dtype=float)
        else:
            x_init = np.zeros(T, dtype=float)
    else:
        x_init = np.zeros(T, dtype=float)

    if "active_set" in sample:
        for item in sample["active_set"]:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                g_idx, t = item[0]
                value = item[1]
                if g_idx == g and 0 <= int(t) < T:
                    x_init[int(t)] = float(value)
    return x_init


def _cvxpy_pi_to_gurobi_pi(constraint, sense: str) -> float:
    dual_val = constraint.dual_value
    if dual_val is None:
        return 0.0
    raw = float(np.asarray(dual_val, dtype=float).reshape(-1)[0])
    if sense == "le":
        return -raw
    if sense == "ge":
        return raw
    return raw


def _nonnegative_pi(constraint, sense: str) -> float:
    return max(0.0, _cvxpy_pi_to_gurobi_pi(constraint, sense))


def _abs_distance(expr, reference_value: float, scale: float):
    """L1 (absolute) distance used for prox terms under cvxpy_highs.

    Using abs instead of square keeps problems in LP form, which is
    substantially more stable for HiGHS in this project.
    """
    scale = max(float(scale), 1e-6)
    return cp.abs((expr - float(reference_value)) / scale)


def _sum_scalar_terms(terms) -> object:
    """Build a shallow cvxpy sum from many scalar terms."""
    if not terms:
        return cp.Constant(0.0)
    normalized = []
    for term in terms:
        if isinstance(term, (int, float, np.integer, np.floating)):
            normalized.append(cp.Constant(float(term)))
        else:
            normalized.append(term)
    return cp.sum(cp.hstack(normalized))


def _sum_pos(expr) -> object:
    return cp.sum(cp.pos(expr))


def _weighted_abs_sum(expr, weights) -> object:
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)
    if weights_arr.size == 0:
        return cp.Constant(0.0)
    return cp.sum(cp.multiply(weights_arr, cp.abs(expr)))


def _build_primal_block_prox_expr(trainer, sample_id: int, pg, x, coc):
    if trainer.pg_block_prox_weight <= 0:
        return 0.0
    g = trainer.unit_id
    prev_pg = trainer.pg[sample_id]
    prev_x = trainer.x[sample_id]
    prev_coc = trainer.coc[sample_id]
    pg_scale = max(float(trainer.gen[g, PMAX]), 1.0)
    coc_scale = (
        1.0
        if trainer.ignore_startup_shutdown_costs
        else max(float(trainer.gencost[g, 1]), float(trainer.gencost[g, 2]), 1.0)
    )
    expr = cp.sum(cp.abs((pg - prev_pg) / pg_scale))
    expr += cp.sum(cp.abs(x - prev_x))
    if trainer.T > 1:
        expr += cp.sum(cp.abs((coc - prev_coc) / coc_scale))
    return expr


def _empty_lambda_inherent(trainer, Ton: int, Toff: int) -> dict:
    return {
        "lambda_pg_lower": np.zeros(trainer.T, dtype=float),
        "lambda_pg_upper": np.zeros(trainer.T, dtype=float),
        "lambda_ramp_up": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_ramp_down": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_min_on": np.array(
            [np.zeros(trainer.T - tau, dtype=float) for tau in range(1, Ton + 1)],
            dtype=object,
        ),
        "lambda_min_off": np.array(
            [np.zeros(trainer.T - tau, dtype=float) for tau in range(1, Toff + 1)],
            dtype=object,
        ),
        "lambda_start_cost": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_shut_cost": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_coc_nonneg": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_x_upper": np.zeros(trainer.T, dtype=float),
        "lambda_x_lower": np.zeros(trainer.T, dtype=float),
    }


def _build_dual_block_prox_expr(
    trainer,
    sample_id: int,
    lam_pg_lower,
    lam_pg_upper,
    lam_ramp_up,
    lam_ramp_down,
    lam_start_cost,
    lam_shut_cost,
    lam_coc_nonneg,
    lam_x_upper,
    lam_x_lower,
    lam_min_on,
    lam_min_off,
    mu,
    Ton: int,
    Toff: int,
):
    if trainer.dual_block_prox_weight <= 0:
        return 0.0
    prev_lambda = trainer.lambda_inherent[sample_id]
    if prev_lambda is None:
        prev_lambda = _empty_lambda_inherent(trainer, Ton, Toff)
    prev_mu = trainer.mu[sample_id]
    terms = []

    for t in range(trainer.T):
        for var_name, var_container in (
            ("lambda_pg_lower", lam_pg_lower),
            ("lambda_pg_upper", lam_pg_upper),
            ("lambda_x_upper", lam_x_upper),
            ("lambda_x_lower", lam_x_lower),
        ):
            prev_val = float(prev_lambda[var_name][t])
            terms.append(_abs_distance(var_container[t], prev_val, max(1.0, abs(prev_val))))

    for t in range(trainer.T - 1):
        for var_name, var_container in (
            ("lambda_ramp_up", lam_ramp_up),
            ("lambda_ramp_down", lam_ramp_down),
            ("lambda_start_cost", lam_start_cost),
            ("lambda_shut_cost", lam_shut_cost),
            ("lambda_coc_nonneg", lam_coc_nonneg),
        ):
            prev_val = float(prev_lambda[var_name][t])
            terms.append(_abs_distance(var_container[t], prev_val, max(1.0, abs(prev_val))))

    for tau in range(1, Ton + 1):
        for t1 in range(trainer.T - tau):
            prev_val = float(prev_lambda["lambda_min_on"][tau - 1][t1])
            terms.append(_abs_distance(lam_min_on[tau - 1, t1], prev_val, max(1.0, abs(prev_val))))

    for tau in range(1, Toff + 1):
        for t1 in range(trainer.T - tau):
            prev_val = float(prev_lambda["lambda_min_off"][tau - 1][t1])
            terms.append(_abs_distance(lam_min_off[tau - 1, t1], prev_val, max(1.0, abs(prev_val))))

    for k in range(trainer.num_coupling_constraints):
        prev_val = float(prev_mu[k])
        terms.append(_abs_distance(mu[k], prev_val, max(1.0, abs(prev_val))))

    return _sum_scalar_terms(terms)


def solve_ed_electricity_price(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    renewable_data: np.ndarray | None = None,
    verbose: bool = False,
    lp_backend: str = LP_BACKEND_GUROBI,
) -> dict:
    backend = normalize_lp_backend(lp_backend)
    if backend == LP_BACKEND_GUROBI:
        return _solve_ed_electricity_price_gurobi(
            ppc,
            Pd,
            T_delta,
            x_sol,
            renewable_data=renewable_data,
            verbose=verbose,
        )
    return _solve_ed_electricity_price_cvxpy_highs(
        ppc,
        Pd,
        T_delta,
        x_sol,
        renewable_data=renewable_data,
        verbose=verbose,
    )


def _get_ramp_limits_from_ppc(ppc, gen_array: np.ndarray, T_delta: float):
    raw_gen = np.asarray(ppc.get("gen", gen_array), dtype=float)
    rates = {}
    if raw_gen.ndim == 2:
        for row in raw_gen:
            if row.size <= 21:
                continue
            bus_idx = int(round(row[GEN_BUS]))
            rates[bus_idx] = {
                "ru": max(float(row[16]), 0.0),
                "rd": max(float(row[17]), 0.0),
                "startup": max(float(row[20]), 0.0),
                "shutdown": max(float(row[21]), 0.0),
            }

    ng = gen_array.shape[0]
    base_ru = 0.4 * gen_array[:, PMAX] / T_delta
    base_rd = 0.4 * gen_array[:, PMAX] / T_delta
    base_ru_co = 0.3 * gen_array[:, PMAX]
    base_rd_co = 0.3 * gen_array[:, PMAX]

    Ru = np.asarray(base_ru, dtype=float).copy()
    Rd = np.asarray(base_rd, dtype=float).copy()
    Ru_co = np.asarray(base_ru_co, dtype=float).copy()
    Rd_co = np.asarray(base_rd_co, dtype=float).copy()

    for g in range(ng):
        bus_idx = int(round(gen_array[g, GEN_BUS]))
        entry = rates.get(bus_idx)
        if not entry:
            continue
        if entry["ru"] > 0:
            Ru[g] = entry["ru"]
        if entry["rd"] > 0:
            Rd[g] = entry["rd"]
        if entry["startup"] > 0:
            Ru_co[g] = entry["startup"]
        if entry["shutdown"] > 0:
            Rd_co[g] = entry["shutdown"]

    return Ru, Rd, Ru_co, Rd_co


def _solve_ed_electricity_price_gurobi(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    renewable_data: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    import gurobipy as gp
    from gurobipy import GRB

    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    bus = ppc_int["bus"]
    branch = ppc_int["branch"]
    gencost = ppc_int["gencost"]

    ng = gen.shape[0]
    nb = bus.shape[0]
    nl = branch.shape[0]
    T = Pd.shape[1]

    load_data = np.asarray(Pd, dtype=float)
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    if renewable_arr is not None and renewable_arr.shape != load_data.shape:
        raise ValueError(
            f"renewable_data shape {renewable_arr.shape} does not match load shape {load_data.shape}"
        )

    renewable_bus_ids = (
        np.where(np.any(renewable_arr > 1e-9, axis=1))[0]
        if renewable_arr is not None
        else np.array([], dtype=int)
    )
    nr = len(renewable_bus_ids)
    R = np.zeros((nb, nr), dtype=float)
    for r, bus_idx in enumerate(renewable_bus_ids):
        R[bus_idx, r] = 1.0

    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0

    PTDF = makePTDF(ppc_int["baseMVA"], bus, branch)
    PTDF_G = PTDF @ G
    branch_limit = branch[:, RATE_A]
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)

    model = gp.Model("subproblem_ed_price")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LogToConsole = 1 if verbose else 0

    pg = model.addVars(ng, T, lb=0.0, name="pg")
    cpower = model.addVars(ng, T, lb=0.0, name="cpower")
    p_ren = model.addVars(nr, T, lb=0.0, name="p_ren") if nr > 0 else None

    for t in range(T):
        renewable_supply = gp.quicksum(p_ren[r, t] for r in range(nr)) if nr > 0 else 0.0
        model.addConstr(
            gp.quicksum(pg[g, t] for g in range(ng)) + renewable_supply == float(np.sum(load_data[:, t])),
            name=f"power_balance_{t}",
        )
        for g in range(ng):
            model.addConstr(pg[g, t] >= float(gen[g, PMIN] * x_sol[g, t]), name=f"pg_lower_{g}_{t}")
            model.addConstr(pg[g, t] <= float(gen[g, PMAX] * x_sol[g, t]), name=f"pg_upper_{g}_{t}")
            model.addConstr(
                cpower[g, t] >= float(gencost[g, -2] / T_delta) * pg[g, t]
                + float(gencost[g, -1] / T_delta * x_sol[g, t]),
                name=f"cpower_{g}_{t}",
            )
        for r, bus_idx in enumerate(renewable_bus_ids):
            model.addConstr(p_ren[r, t] <= float(renewable_arr[bus_idx, t]), name=f"ren_upper_{r}_{t}")

    for t in range(1, T):
        for g in range(ng):
            model.addConstr(
                pg[g, t] - pg[g, t - 1]
                <= float(Ru[g] * x_sol[g, t - 1] + Ru_co[g] * (1 - x_sol[g, t - 1])),
                name=f"ramp_up_{g}_{t}",
            )
            model.addConstr(
                pg[g, t - 1] - pg[g, t]
                <= float(Rd[g] * x_sol[g, t] + Rd_co[g] * (1 - x_sol[g, t])),
                name=f"ramp_down_{g}_{t}",
            )

    for t in range(T):
        thermal_injection = G @ np.array([pg[g, t] for g in range(ng)], dtype=object)
        renewable_injection = (
            R @ np.array([p_ren[r, t] for r in range(nr)], dtype=object) if nr > 0 else 0.0
        )
        flow = PTDF @ (thermal_injection + renewable_injection - load_data[:, t])
        for l in range(nl):
            model.addConstr(flow[l] <= float(branch_limit[l]), name=f"flow_upper_{l}_{t}")
            model.addConstr(flow[l] >= -float(branch_limit[l]), name=f"flow_lower_{l}_{t}")

    model.setObjective(gp.quicksum(cpower[g, t] for g in range(ng) for t in range(T)), GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"ED solve for electricity price failed with status={model.status}")

    lambda_power_balance = np.zeros(T, dtype=float)
    lambda_pg_lower = np.zeros((ng, T), dtype=float)
    lambda_pg_upper = np.zeros((ng, T), dtype=float)
    lambda_ramp_up = np.zeros((ng, T - 1), dtype=float)
    lambda_ramp_down = np.zeros((ng, T - 1), dtype=float)
    lambda_flow_upper = np.zeros((nl, T), dtype=float)
    lambda_flow_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        lambda_power_balance[t] = float(model.getConstrByName(f"power_balance_{t}").Pi)
        for g in range(ng):
            lambda_pg_lower[g, t] = float(model.getConstrByName(f"pg_lower_{g}_{t}").Pi)
            lambda_pg_upper[g, t] = float(model.getConstrByName(f"pg_upper_{g}_{t}").Pi)
        for l in range(nl):
            lambda_flow_upper[l, t] = float(model.getConstrByName(f"flow_upper_{l}_{t}").Pi)
            lambda_flow_lower[l, t] = float(model.getConstrByName(f"flow_lower_{l}_{t}").Pi)

    for t in range(1, T):
        for g in range(ng):
            lambda_ramp_up[g, t - 1] = float(model.getConstrByName(f"ramp_up_{g}_{t}").Pi)
            lambda_ramp_down[g, t - 1] = float(model.getConstrByName(f"ramp_down_{g}_{t}").Pi)

    effective = np.zeros((ng, T), dtype=float)
    lambda_ramp_contrib = np.zeros((ng, T), dtype=float)
    lambda_flow_contrib = np.zeros((ng, T), dtype=float)
    for g in range(ng):
        for t in range(T):
            ramp_contrib = 0.0
            if t > 0:
                ramp_contrib += lambda_ramp_up[g, t - 1]
                ramp_contrib -= lambda_ramp_down[g, t - 1]
            if t < T - 1:
                ramp_contrib -= lambda_ramp_up[g, t]
                ramp_contrib += lambda_ramp_down[g, t]

            flow_contrib = float(np.dot(PTDF_G[:, g], lambda_flow_upper[:, t] + lambda_flow_lower[:, t]))
            effective[g, t] = (
                lambda_power_balance[t]
                + lambda_pg_lower[g, t]
                + lambda_pg_upper[g, t]
                + ramp_contrib
                + flow_contrib
            )
            lambda_ramp_contrib[g, t] = ramp_contrib
            lambda_flow_contrib[g, t] = flow_contrib

    return {
        "lambda_pg_electricity_price": effective,
        "lambda_power_balance": lambda_power_balance,
        "lambda_pg_lower": lambda_pg_lower,
        "lambda_pg_upper": lambda_pg_upper,
        "lambda_ramp_up": lambda_ramp_up,
        "lambda_ramp_down": lambda_ramp_down,
        "lambda_flow_upper": lambda_flow_upper,
        "lambda_flow_lower": lambda_flow_lower,
        "lambda_ramp_contrib": lambda_ramp_contrib,
        "lambda_flow_contrib": lambda_flow_contrib,
    }


def _solve_ed_electricity_price_cvxpy_highs(
    ppc,
    Pd: np.ndarray,
    T_delta: float,
    x_sol: np.ndarray,
    renewable_data: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    assert_lp_backend_available(LP_BACKEND_CVXPY_HIGHS)

    ppc_int = ext2int(ppc)
    gen = ppc_int["gen"]
    bus = ppc_int["bus"]
    branch = ppc_int["branch"]
    gencost = ppc_int["gencost"]

    ng = gen.shape[0]
    nb = bus.shape[0]
    nl = branch.shape[0]
    T = Pd.shape[1]

    load_data = np.asarray(Pd, dtype=float)
    renewable_arr = None if renewable_data is None else np.asarray(renewable_data, dtype=float)
    if renewable_arr is not None and renewable_arr.shape != load_data.shape:
        raise ValueError(
            f"renewable_data shape {renewable_arr.shape} does not match load shape {load_data.shape}"
        )

    renewable_bus_ids = (
        np.where(np.any(renewable_arr > 1e-9, axis=1))[0]
        if renewable_arr is not None
        else np.array([], dtype=int)
    )
    nr = len(renewable_bus_ids)
    R = np.zeros((nb, nr), dtype=float)
    for r, bus_idx in enumerate(renewable_bus_ids):
        R[bus_idx, r] = 1.0

    G = np.zeros((nb, ng), dtype=float)
    for g in range(ng):
        bus_idx = int(gen[g, GEN_BUS])
        if 0 <= bus_idx < nb:
            G[bus_idx, g] = 1.0

    PTDF = makePTDF(ppc_int["baseMVA"], bus, branch)
    PTDF_G = PTDF @ G
    branch_limit = branch[:, RATE_A]
    Ru, Rd, Ru_co, Rd_co = _get_ramp_limits_from_ppc(ppc, gen, T_delta)

    pg = cp.Variable((ng, T), nonneg=True)
    cpower = cp.Variable((ng, T), nonneg=True)
    p_ren = cp.Variable((nr, T), nonneg=True) if nr > 0 else None

    constraints = []
    power_balance_cons = []
    pg_lower_cons = {}
    pg_upper_cons = {}
    ramp_up_cons = {}
    ramp_down_cons = {}
    flow_upper_cons = {}
    flow_lower_cons = {}

    for t in range(T):
        renewable_supply = cp.sum(p_ren[:, t]) if nr > 0 else 0.0
        cons = cp.sum(pg[:, t]) + renewable_supply == float(np.sum(load_data[:, t]))
        constraints.append(cons)
        power_balance_cons.append(cons)
        for g in range(ng):
            cons = pg[g, t] >= float(gen[g, PMIN] * x_sol[g, t])
            constraints.append(cons)
            pg_lower_cons[g, t] = cons
            cons = pg[g, t] <= float(gen[g, PMAX] * x_sol[g, t])
            constraints.append(cons)
            pg_upper_cons[g, t] = cons
            constraints.append(
                cpower[g, t] >= float(gencost[g, -2] / T_delta) * pg[g, t]
                + float(gencost[g, -1] / T_delta * x_sol[g, t])
            )
        for r, bus_idx in enumerate(renewable_bus_ids):
            constraints.append(p_ren[r, t] <= float(renewable_arr[bus_idx, t]))

    for t in range(1, T):
        for g in range(ng):
            cons = (
                pg[g, t] - pg[g, t - 1]
                <= float(Ru[g] * x_sol[g, t - 1] + Ru_co[g] * (1 - x_sol[g, t - 1]))
            )
            constraints.append(cons)
            ramp_up_cons[g, t - 1] = cons
            cons = (
                pg[g, t - 1] - pg[g, t]
                <= float(Rd[g] * x_sol[g, t] + Rd_co[g] * (1 - x_sol[g, t]))
            )
            constraints.append(cons)
            ramp_down_cons[g, t - 1] = cons

    for t in range(T):
        thermal_injection = G @ pg[:, t]
        renewable_injection = R @ p_ren[:, t] if nr > 0 else 0.0
        flow = PTDF @ (thermal_injection + renewable_injection - load_data[:, t])
        for l in range(nl):
            cons = flow[l] <= float(branch_limit[l])
            constraints.append(cons)
            flow_upper_cons[l, t] = cons
            cons = flow[l] >= -float(branch_limit[l])
            constraints.append(cons)
            flow_lower_cons[l, t] = cons

    problem = cp.Problem(cp.Minimize(cp.sum(cpower)), constraints)
    _solve_with_cvxpy_highs(problem, verbose=verbose)
    if not _problem_is_optimal(problem):
        raise RuntimeError(f"ED solve for electricity price failed with status={problem.status}")

    lambda_power_balance = np.zeros(T, dtype=float)
    lambda_pg_lower = np.zeros((ng, T), dtype=float)
    lambda_pg_upper = np.zeros((ng, T), dtype=float)
    lambda_ramp_up = np.zeros((ng, T - 1), dtype=float)
    lambda_ramp_down = np.zeros((ng, T - 1), dtype=float)
    lambda_flow_upper = np.zeros((nl, T), dtype=float)
    lambda_flow_lower = np.zeros((nl, T), dtype=float)

    for t in range(T):
        lambda_power_balance[t] = _cvxpy_pi_to_gurobi_pi(power_balance_cons[t], "eq")
        for g in range(ng):
            lambda_pg_lower[g, t] = _cvxpy_pi_to_gurobi_pi(pg_lower_cons[g, t], "ge")
            lambda_pg_upper[g, t] = _cvxpy_pi_to_gurobi_pi(pg_upper_cons[g, t], "le")
        for l in range(nl):
            lambda_flow_upper[l, t] = _cvxpy_pi_to_gurobi_pi(flow_upper_cons[l, t], "le")
            lambda_flow_lower[l, t] = _cvxpy_pi_to_gurobi_pi(flow_lower_cons[l, t], "ge")

    for t in range(1, T):
        for g in range(ng):
            lambda_ramp_up[g, t - 1] = _cvxpy_pi_to_gurobi_pi(ramp_up_cons[g, t - 1], "le")
            lambda_ramp_down[g, t - 1] = _cvxpy_pi_to_gurobi_pi(ramp_down_cons[g, t - 1], "le")

    effective = np.zeros((ng, T), dtype=float)
    lambda_ramp_contrib = np.zeros((ng, T), dtype=float)
    lambda_flow_contrib = np.zeros((ng, T), dtype=float)
    for g in range(ng):
        for t in range(T):
            ramp_contrib = 0.0
            if t > 0:
                ramp_contrib += lambda_ramp_up[g, t - 1]
                ramp_contrib -= lambda_ramp_down[g, t - 1]
            if t < T - 1:
                ramp_contrib -= lambda_ramp_up[g, t]
                ramp_contrib += lambda_ramp_down[g, t]

            flow_contrib = float(np.dot(PTDF_G[:, g], lambda_flow_upper[:, t] + lambda_flow_lower[:, t]))
            effective[g, t] = (
                lambda_power_balance[t]
                + lambda_pg_lower[g, t]
                + lambda_pg_upper[g, t]
                + ramp_contrib
                + flow_contrib
            )
            lambda_ramp_contrib[g, t] = ramp_contrib
            lambda_flow_contrib[g, t] = flow_contrib

    return {
        "lambda_pg_electricity_price": effective,
        "lambda_power_balance": lambda_power_balance,
        "lambda_pg_lower": lambda_pg_lower,
        "lambda_pg_upper": lambda_pg_upper,
        "lambda_ramp_up": lambda_ramp_up,
        "lambda_ramp_down": lambda_ramp_down,
        "lambda_flow_upper": lambda_flow_upper,
        "lambda_flow_lower": lambda_flow_lower,
        "lambda_ramp_contrib": lambda_ramp_contrib,
        "lambda_flow_contrib": lambda_flow_contrib,
    }


def solve_init_lp(trainer, sample_id: int):
    """单机组初始化 LP（cvxpy+HiGHS）：在**固定 MILP 真值启停轨迹**下求 pg/coc/cpower 及固有约束对偶。

    **“真值”单机组启停 x 如何来**
    - 函数 ``_recover_unit_x_from_sample`` 按样本 ``active_set_data[sample_id]`` 恢复长度 T 的向量 ``x_init``：
      * 若存在 ``active_set``：遍历 ``[[机组 g, 时段 t], 值]``，只取 **当前机组** ``g == unit_id`` 的条目写入 ``x_init[t]``；
      * 否则若存在 ``unit_commitment_matrix``：取第 ``g`` 行 ``uc[g, :]`` 作为 ``x_init``。
    - **不**在此处做 0/1 舍入；舍入在约束里用阈值 ``x_true_eps=0.5``：
      * ``x_init[t] >= 0.5`` → 加 ``x[t] >= 1.0``（开机）；
      * 否则 → 加 ``x[t] <= 0.0``（停机）。
    即 init LP 中 **x 被钉在 0/1**，与 JSON 标签一致，再在连续变量 **pg, coc, cpower** 上求可行最小成本。返回的
    ``x_true`` 为上述 ``x_init`` 的拷贝，供 ``x_true``/敏感时段等后续使用。

    **init LP 包含的约束（与 Gurobi 初值路径语义对齐）**
    - ``0 <= x <= 1``，随后被每时段的 ``x>=1`` 或 ``x<=0`` **收紧为 MILP 标签对应的 0/1**；
    - 发电上下界：``pg[t] >= Pmin*x[t]``, ``pg[t] <= Pmax*x[t]``；
    - 爬坡：``t>=1`` 的 ramp up / ramp down（含 Ru_co, Rd_co 项）；
    - 最小开/关时间：``Ton=Toff=min(4,T)`` 下的 min_on / min_off 线性化；
    - 启停成本：``coc`` 的 start/shut 线性不等式（``ignore_startup_shutdown_costs`` 时 sc=shc=0）；
    - 电力成本定义：``cpower[t] == a*pg[t] + b*x[t]``（``a,b`` 来自 gencost / T_delta）；
    - 目标：``min sum(cpower)+sum(coc) - lambda_vals·pg``（与节点电价 ``lambda`` 一致）。

    **成功条件**  
    仅当 HiGHS 使 ``problem.status`` 为 ``OPTIMAL`` 或 ``OPTIMAL_INACCURATE`` 时成功；否则抛出 ``RuntimeError``，**不**在调用方用假对偶回退。
    """
    assert_lp_backend_available(LP_BACKEND_CVXPY_HIGHS)

    g = trainer.unit_id
    Pmin = trainer.gen[g, PMIN]
    Pmax = trainer.gen[g, PMAX]
    Ru = float(trainer.Ru_all[g])
    Rd = float(trainer.Rd_all[g])
    Ru_co = float(trainer.Ru_co_all[g])
    Rd_co = float(trainer.Rd_co_all[g])
    a = trainer.gencost[g, -2] / trainer.T_delta
    b = trainer.gencost[g, -1] / trainer.T_delta
    sc = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 1]
    shc = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 2]
    Ton = min(4, trainer.T)
    Toff = min(4, trainer.T)

    lambda_val = np.asarray(trainer.lambda_vals[sample_id], dtype=float)
    x_init = _recover_unit_x_from_sample(trainer, sample_id)

    pg = cp.Variable(trainer.T, nonneg=True)
    x = cp.Variable(trainer.T)
    coc = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    cpower = cp.Variable(trainer.T, nonneg=True)

    constraints = [x >= 0, x <= 1]
    pg_lower_cons = []
    pg_upper_cons = []
    ramp_up_cons = []
    ramp_down_cons = []
    min_on_cons = {}
    min_off_cons = {}
    start_cost_cons = []
    shut_cost_cons = []

    for t in range(trainer.T):
        cons = pg[t] >= Pmin * x[t]
        constraints.append(cons)
        pg_lower_cons.append(cons)
        cons = pg[t] <= Pmax * x[t]
        constraints.append(cons)
        pg_upper_cons.append(cons)

    for t in range(1, trainer.T):
        cons = pg[t] - pg[t - 1] <= Ru * x[t - 1] + Ru_co * (1 - x[t - 1])
        constraints.append(cons)
        ramp_up_cons.append(cons)
        cons = pg[t - 1] - pg[t] <= Rd * x[t] + Rd_co * (1 - x[t])
        constraints.append(cons)
        ramp_down_cons.append(cons)

    for tau in range(1, Ton + 1):
        for t1 in range(trainer.T - tau):
            cons = x[t1 + 1] - x[t1] <= x[t1 + tau]
            constraints.append(cons)
            min_on_cons[tau - 1, t1] = cons
    for tau in range(1, Toff + 1):
        for t1 in range(trainer.T - tau):
            cons = -x[t1 + 1] + x[t1] <= 1 - x[t1 + tau]
            constraints.append(cons)
            min_off_cons[tau - 1, t1] = cons

    for t in range(1, trainer.T):
        cons = coc[t - 1] >= sc * (x[t] - x[t - 1])
        constraints.append(cons)
        start_cost_cons.append(cons)
        cons = coc[t - 1] >= shc * (x[t - 1] - x[t])
        constraints.append(cons)
        shut_cost_cons.append(cons)

    for t in range(trainer.T):
        constraints.append(cpower[t] == a * pg[t] + b * x[t])

    # 真值边界：与 MILP 标签一致，用于提取 μ₂（single-time 耦合对偶）初值的影子价格
    x_true_eps = 0.5
    x_true_fix_meta = []
    for t in range(trainer.T):
        if float(x_init[t]) >= x_true_eps:
            cons = x[t] >= 1.0
            constraints.append(cons)
            x_true_fix_meta.append(("ge", cons))
        else:
            cons = x[t] <= 0.0
            constraints.append(cons)
            x_true_fix_meta.append(("le", cons))

    objective = cp.sum(cpower) + cp.sum(coc) - lambda_val @ pg
    problem = cp.Problem(cp.Minimize(objective), constraints)
    _solve_with_cvxpy_highs(problem, verbose=False)
    if not _problem_is_optimal(problem):
        st = getattr(problem, "status", None)
        val = getattr(problem, "value", None)
        raise RuntimeError(
            f"init LP 未得到可接受最优解: unit_id={g}, sample_id={sample_id}, "
            f"status={st!r}, objective_value={val!r}. "
            "需要 HiGHS 状态为 OPTIMAL 或 OPTIMAL_INACCURATE；"
            "请检查真值 x 与机组参数（爬坡/最小开停等）是否导致不可行或数值问题。"
        )

    lambda_inherent = {
        "lambda_pg_lower": np.array([_nonnegative_pi(cons, "ge") for cons in pg_lower_cons]),
        "lambda_pg_upper": np.array([_nonnegative_pi(cons, "le") for cons in pg_upper_cons]),
        "lambda_ramp_up": np.array([_nonnegative_pi(cons, "le") for cons in ramp_up_cons]),
        "lambda_ramp_down": np.array([_nonnegative_pi(cons, "le") for cons in ramp_down_cons]),
        "lambda_min_on": np.array(
            [
                [_nonnegative_pi(min_on_cons[tau - 1, t1], "le") for t1 in range(trainer.T - tau)]
                for tau in range(1, Ton + 1)
            ],
            dtype=object,
        ),
        "lambda_min_off": np.array(
            [
                [_nonnegative_pi(min_off_cons[tau - 1, t1], "le") for t1 in range(trainer.T - tau)]
                for tau in range(1, Toff + 1)
            ],
            dtype=object,
        ),
        "lambda_start_cost": np.array([_nonnegative_pi(cons, "ge") for cons in start_cost_cons]),
        "lambda_shut_cost": np.array([_nonnegative_pi(cons, "ge") for cons in shut_cost_cons]),
        "lambda_coc_nonneg": np.zeros(max(trainer.T - 1, 0), dtype=float),
        "lambda_x_upper": np.zeros(trainer.T, dtype=float),
        "lambda_x_lower": np.zeros(trainer.T, dtype=float),
    }

    mu2_coupling_dual = np.array(
        [_nonnegative_pi(cons, sense) for sense, cons in x_true_fix_meta],
        dtype=float,
    )

    return {
        "pg_sol": np.asarray(pg.value, dtype=float),
        "x_sol": np.asarray(x.value, dtype=float),
        "coc_sol": np.asarray(coc.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "cpower_sol": np.asarray(cpower.value, dtype=float),
        "x_true": x_init.copy(),
        "lambda_inherent": lambda_inherent,
        "mu2_coupling_dual": mu2_coupling_dual,
    }


def solve_primal_block(
    trainer,
    sample_id: int,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: np.ndarray = None,
    pg_costs: np.ndarray = None,
):
    assert_lp_backend_available(LP_BACKEND_CVXPY_HIGHS)

    g = trainer.unit_id
    mu_vals = np.abs(trainer.mu[sample_id])
    lam_inh = trainer.lambda_inherent[sample_id]

    if _strict_cvxpy_highs_diagnostics_enabled():
        ctx = f"unit_id={g}, sample_id={sample_id}, block=primal"
        _require_finite("mu_vals", mu_vals, context=ctx)
        _require_finite("alphas", alphas, context=ctx)
        _require_finite("betas", betas, context=ctx)
        _require_finite("gammas", gammas, context=ctx)
        _require_finite("deltas", deltas, context=ctx)
        if costs is not None:
            _require_finite("costs", costs, context=ctx)
        if pg_costs is not None:
            _require_finite("pg_costs", pg_costs, context=ctx)
        if lam_inh is None:
            raise ValueError(
                f"lambda_inherent is None ({ctx}); "
                f"lambda_inherent_summary={_summarize_lambda_inherent(lam_inh)}"
            )
        for key in (
            "lambda_pg_lower",
            "lambda_pg_upper",
            "lambda_ramp_up",
            "lambda_ramp_down",
            "lambda_min_on",
            "lambda_min_off",
            "lambda_start_cost",
            "lambda_shut_cost",
            "lambda_coc_nonneg",
            "lambda_x_upper",
            "lambda_x_lower",
        ):
            if key not in lam_inh:
                raise KeyError(f"Missing {key!r} in lambda_inherent ({ctx}).")
        _require_finite("lambda_pg_lower", np.asarray(lam_inh["lambda_pg_lower"]), context=ctx)
        _require_finite("lambda_pg_upper", np.asarray(lam_inh["lambda_pg_upper"]), context=ctx)
        _require_finite("lambda_ramp_up", np.asarray(lam_inh["lambda_ramp_up"]), context=ctx)
        _require_finite("lambda_ramp_down", np.asarray(lam_inh["lambda_ramp_down"]), context=ctx)
        _require_finite("lambda_start_cost", np.asarray(lam_inh["lambda_start_cost"]), context=ctx)
        _require_finite("lambda_shut_cost", np.asarray(lam_inh["lambda_shut_cost"]), context=ctx)
        _require_finite("lambda_coc_nonneg", np.asarray(lam_inh["lambda_coc_nonneg"]), context=ctx)
        _require_finite("lambda_x_upper", np.asarray(lam_inh["lambda_x_upper"]), context=ctx)
        _require_finite("lambda_x_lower", np.asarray(lam_inh["lambda_x_lower"]), context=ctx)

    alphas, betas, gammas, deltas = trainer._apply_surrogate_direction_to_params(
        alphas,
        betas,
        gammas,
        deltas,
    )

    Pmin = trainer.gen[g, PMIN]
    Pmax = trainer.gen[g, PMAX]
    a = trainer.gencost[g, -2] / trainer.T_delta
    b = trainer.gencost[g, -1] / trainer.T_delta
    Ru = float(trainer.Ru_all[g])
    Rd = float(trainer.Rd_all[g])
    Ru_co = float(trainer.Ru_co_all[g])
    Rd_co = float(trainer.Rd_co_all[g])
    sc = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 1]
    shc = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 2]
    Ton = min(4, trainer.T)
    Toff = min(4, trainer.T)

    pg = cp.Variable(trainer.T, nonneg=True)
    x = cp.Variable(trainer.T)
    coc = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    cpower = cp.Variable(trainer.T, nonneg=True)

    constraints = [x >= 0, x <= 1]

    x_true = trainer.active_set_data[sample_id].get("x_true", None)
    if x_true is None:
        x_true = trainer.x[sample_id]

    obj_binary = cp.sum(cp.abs(x - x_true))
    lam_pg_lower = np.abs(np.asarray(lam_inh["lambda_pg_lower"], dtype=float))
    lam_pg_upper = np.abs(np.asarray(lam_inh["lambda_pg_upper"], dtype=float))
    lam_x_lower = np.abs(np.asarray(lam_inh["lambda_x_lower"], dtype=float))
    lam_x_upper = np.abs(np.asarray(lam_inh["lambda_x_upper"], dtype=float))

    obj_primal_terms = []
    obj_opt_terms = []

    pg_lower_expr = Pmin * x - pg
    obj_primal_terms.append(_sum_pos(pg_lower_expr))
    obj_opt_terms.append(_weighted_abs_sum(pg_lower_expr, lam_pg_lower))

    pg_upper_expr = pg - Pmax * x
    obj_primal_terms.append(_sum_pos(pg_upper_expr))
    obj_opt_terms.append(_weighted_abs_sum(pg_upper_expr, lam_pg_upper))

    obj_opt_terms.append(cp.sum(cp.multiply(lam_x_lower, x)))
    obj_opt_terms.append(cp.sum(cp.multiply(lam_x_upper, 1 - x)))

    if trainer.T > 1:
        lam_ramp_up = np.abs(np.asarray(lam_inh["lambda_ramp_up"], dtype=float))
        lam_ramp_down = np.abs(np.asarray(lam_inh["lambda_ramp_down"], dtype=float))
        ramp_up_expr = pg[1:] - pg[:-1] - Ru * x[:-1] - Ru_co * (1 - x[:-1])
        obj_primal_terms.append(_sum_pos(ramp_up_expr))
        obj_opt_terms.append(_weighted_abs_sum(ramp_up_expr, lam_ramp_up))

        ramp_down_expr = pg[:-1] - pg[1:] - Rd * x[1:] - Rd_co * (1 - x[1:])
        obj_primal_terms.append(_sum_pos(ramp_down_expr))
        obj_opt_terms.append(_weighted_abs_sum(ramp_down_expr, lam_ramp_down))

    for tau in range(1, Ton + 1):
        min_on_expr = x[1:trainer.T - tau + 1] - x[:trainer.T - tau] - x[tau:]
        lam_min_on = np.abs(np.asarray(lam_inh["lambda_min_on"][tau - 1], dtype=float))
        obj_primal_terms.append(_sum_pos(min_on_expr))
        obj_opt_terms.append(_weighted_abs_sum(min_on_expr, lam_min_on))

    for tau in range(1, Toff + 1):
        min_off_expr = -x[1:trainer.T - tau + 1] + x[:trainer.T - tau] - (1 - x[tau:])
        lam_min_off = np.abs(np.asarray(lam_inh["lambda_min_off"][tau - 1], dtype=float))
        obj_primal_terms.append(_sum_pos(min_off_expr))
        obj_opt_terms.append(_weighted_abs_sum(min_off_expr, lam_min_off))

    if trainer.T > 1:
        lam_start_cost = np.abs(np.asarray(lam_inh["lambda_start_cost"], dtype=float))
        lam_shut_cost = np.abs(np.asarray(lam_inh["lambda_shut_cost"], dtype=float))
        lam_coc_nonneg = np.abs(np.asarray(lam_inh["lambda_coc_nonneg"], dtype=float))
        start_expr = sc * (x[1:] - x[:-1]) - coc
        obj_primal_terms.append(_sum_pos(start_expr))
        obj_opt_terms.append(_weighted_abs_sum(start_expr, lam_start_cost))

        shut_expr = shc * (x[:-1] - x[1:]) - coc
        obj_primal_terms.append(_sum_pos(shut_expr))
        obj_opt_terms.append(_weighted_abs_sum(shut_expr, lam_shut_cost))
        obj_opt_terms.append(cp.sum(cp.multiply(lam_coc_nonneg, coc)))

    constraints.append(cpower == a * pg + b * x)

    sensitive_t = trainer.sensitive_timesteps[sample_id]
    constraint_offsets = trainer._constraint_offsets_for_sample(sample_id)
    for k, t in enumerate(sensitive_t):
        coupling_lhs = build_surrogate_constraint_expression(
            x,
            t,
            constraint_offsets[k],
            alphas[k],
            betas[k],
            gammas[k],
            trainer.T,
        )
        obj_primal_terms.append(cp.pos(coupling_lhs - deltas[k]))
        obj_opt_terms.append(cp.abs(coupling_lhs - deltas[k]) * mu_vals[k])

    obj_prox = _build_primal_block_prox_expr(trainer, sample_id, pg, x, coc)
    obj_primal = _sum_scalar_terms(obj_primal_terms)
    obj_opt = _sum_scalar_terms(obj_opt_terms)
    objective = (
        trainer.rho_primal * obj_primal
        + trainer.rho_opt * obj_opt
        + trainer.rho_binary * obj_binary
        + trainer.pg_block_prox_weight * obj_prox
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        _solve_with_cvxpy_highs(problem, verbose=False)
    except Exception:
        if _strict_cvxpy_highs_diagnostics_enabled():
            ctx = f"unit_id={g}, sample_id={sample_id}, block=primal"
            print("[STRICT_CVXPY_HIGHS] primal_block failure context:", flush=True)
            print(f"  {ctx}", flush=True)
            try:
                print(f"  mu_vals={_array_finite_stats(mu_vals)}", flush=True)
                print(f"  alphas={_array_finite_stats(alphas)}", flush=True)
                print(f"  betas={_array_finite_stats(betas)}", flush=True)
                print(f"  gammas={_array_finite_stats(gammas)}", flush=True)
                print(f"  deltas={_array_finite_stats(deltas)}", flush=True)
                if costs is not None:
                    print(f"  costs={_array_finite_stats(costs)}", flush=True)
                if pg_costs is not None:
                    print(f"  pg_costs={_array_finite_stats(pg_costs)}", flush=True)
                print(f"  lambda_inherent_summary={_summarize_lambda_inherent(lam_inh)}", flush=True)
            except Exception:
                pass
        raise
    display_sample_id = int(getattr(trainer, "display_sample_id", sample_id))
    defer_log = bool(getattr(trainer, "_defer_lp_block_log", False))
    if defer_log:
        trainer._deferred_lp_block_log = None
    if display_sample_id <= 2:
        try:
            def _maybe_float(val):
                if val is None:
                    return None
                try:
                    return float(np.asarray(val, dtype=float).reshape(-1)[0])
                except Exception:
                    return None

            obj_primal_v = _maybe_float(getattr(obj_primal, "value", None))
            obj_opt_v = _maybe_float(getattr(obj_opt, "value", None))
            obj_binary_v = _maybe_float(getattr(obj_binary, "value", None))
            obj_prox_v = _maybe_float(getattr(obj_prox, "value", None))
            status = getattr(problem, "status", None)
            line = (
                f"[Unit-{g}] primal_block, sample_id: {display_sample_id}, "
                f"status: {status}, "
                f"obj_primal: {obj_primal_v if obj_primal_v is not None else 'None'}, "
                f"obj_opt: {obj_opt_v if obj_opt_v is not None else 'None'}, "
                f"obj_binary: {obj_binary_v if obj_binary_v is not None else 'None'}, "
                f"obj_prox: {obj_prox_v if obj_prox_v is not None else 'None'}"
            )
            if defer_log:
                trainer._deferred_lp_block_log = line
            else:
                print(line, flush=True)
        except Exception:
            pass

    if not _problem_is_optimal(problem):
        return None, None, None, None

    return (
        np.asarray(pg.value, dtype=float),
        np.asarray(x.value, dtype=float),
        np.asarray(coc.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        np.asarray(cpower.value, dtype=float),
    )


def solve_dual_block(
    trainer,
    sample_id: int,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    costs: np.ndarray = None,
    pg_costs: np.ndarray = None,
):
    assert_lp_backend_available(LP_BACKEND_CVXPY_HIGHS)

    g = trainer.unit_id
    pg_val = np.asarray(trainer.pg[sample_id], dtype=float)
    x_val = np.asarray(trainer.x[sample_id], dtype=float)
    coc_val = np.asarray(trainer.coc[sample_id], dtype=float)
    lambda_val = np.asarray(trainer.lambda_vals[sample_id], dtype=float)

    if _strict_cvxpy_highs_diagnostics_enabled():
        ctx = f"unit_id={g}, sample_id={sample_id}, block=dual"
        _require_finite("pg_val", pg_val, context=ctx)
        _require_finite("x_val", x_val, context=ctx)
        _require_finite("coc_val", coc_val, context=ctx)
        _require_finite("lambda_val", lambda_val, context=ctx)
        _require_finite("alphas", alphas, context=ctx)
        _require_finite("betas", betas, context=ctx)
        _require_finite("gammas", gammas, context=ctx)
        _require_finite("deltas", deltas, context=ctx)
        if costs is not None:
            _require_finite("costs", costs, context=ctx)
        if pg_costs is not None:
            _require_finite("pg_costs", pg_costs, context=ctx)

    a = trainer.gencost[g, -2] / trainer.T_delta
    b = trainer.gencost[g, -1] / trainer.T_delta
    Pmin = trainer.gen[g, PMIN]
    Pmax = trainer.gen[g, PMAX]
    Ru = float(trainer.Ru_all[g])
    Rd = float(trainer.Rd_all[g])
    Ru_co = float(trainer.Ru_co_all[g])
    Rd_co = float(trainer.Rd_co_all[g])
    start_cost = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 1]
    shut_cost = 0.0 if trainer.ignore_startup_shutdown_costs else trainer.gencost[g, 2]
    Ton = min(4, trainer.T)
    Toff = min(4, trainer.T)

    phase = trainer._get_mu_lower_bound_phase()
    lb = trainer._current_mu_lower_bound_value()
    sign_relax_round = trainer._is_mu_sign_relaxation_round()
    alphas, betas, gammas, deltas = trainer._apply_surrogate_direction_to_params(
        alphas,
        betas,
        gammas,
        deltas,
    )

    lam_pg_lower = cp.Variable(trainer.T, nonneg=True)
    lam_pg_upper = cp.Variable(trainer.T, nonneg=True)
    lam_ramp_up = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    lam_ramp_down = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    lam_start_cost = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    lam_shut_cost = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    lam_coc_nonneg = cp.Variable(max(trainer.T - 1, 0), nonneg=True)
    lam_x_upper = cp.Variable(trainer.T, nonneg=True)
    lam_x_lower = cp.Variable(trainer.T, nonneg=True)

    constraints = []
    if trainer._force_zero_x_bound_duals():
        constraints.extend([lam_x_upper <= 0.0, lam_x_lower <= 0.0])

    lam_min_on = {}
    lam_min_off = {}
    for tau in range(1, Ton + 1):
        for t1 in range(trainer.T - tau):
            lam_min_on[tau - 1, t1] = cp.Variable(nonneg=True)
    for tau in range(1, Toff + 1):
        for t1 in range(trainer.T - tau):
            lam_min_off[tau - 1, t1] = cp.Variable(nonneg=True)

    mu = (
        cp.Variable(trainer.num_coupling_constraints)
        if (phase != "none" and sign_relax_round)
        else cp.Variable(trainer.num_coupling_constraints, nonneg=True)
    )
    mu_abs = cp.Variable(trainer.num_coupling_constraints, nonneg=True)
    constraints.extend([mu_abs >= mu, mu_abs >= -mu])
    if phase == "individual" and lb > 0:
        if sign_relax_round:
            constraints.append(mu_abs >= lb)
        else:
            constraints.append(mu >= lb)
    if phase == "group" and lb > 0 and trainer._uses_group_mu_lower_bound():
        for group_idx in range(trainer.num_coupling_constraints // trainer.all_mode_group_size):
            group_start = group_idx * trainer.all_mode_group_size
            group_stop = group_start + trainer.all_mode_group_size
            group_expr = cp.sum(mu_abs[group_start:group_stop]) if sign_relax_round else cp.sum(mu[group_start:group_stop])
            constraints.append(group_expr >= lb)

    obj_dual_pg_terms = []
    obj_dual_x_terms = []
    obj_dual_coc_terms = []
    obj_opt_terms = []

    for t in range(trainer.T):
        expr_terms = [
            a + (pg_costs[t] if pg_costs is not None else 0) - lambda_val[t],
            -lam_pg_lower[t],
            lam_pg_upper[t],
        ]
        if t > 0:
            expr_terms.extend([lam_ramp_up[t - 1], -lam_ramp_down[t - 1]])
        if t < trainer.T - 1:
            expr_terms.extend([-lam_ramp_up[t], lam_ramp_down[t]])
        obj_dual_pg_terms.append(cp.abs(_sum_scalar_terms(expr_terms)))

    sensitive_t = trainer.sensitive_timesteps[sample_id]
    constraint_offsets = trainer._constraint_offsets_for_sample(sample_id)
    for t in range(trainer.T):
        expr_terms = [
            b + (costs[t] if costs is not None else 0),
            Pmin * lam_pg_lower[t],
            -Pmax * lam_pg_upper[t],
        ]
        if t < trainer.T - 1:
            expr_terms.append((Ru_co - Ru) * lam_ramp_up[t])
        if t > 0:
            expr_terms.append((Rd_co - Rd) * lam_ramp_down[t - 1])

        for tau in range(1, Ton + 1):
            for t1 in range(trainer.T - tau):
                k = lam_min_on[tau - 1, t1]
                if t == t1 + 1:
                    expr_terms.append(k)
                if t == t1:
                    expr_terms.append(-k)
                if t == t1 + tau:
                    expr_terms.append(-k)

        for tau in range(1, Toff + 1):
            for t1 in range(trainer.T - tau):
                k = lam_min_off[tau - 1, t1]
                if t == t1 + 1:
                    expr_terms.append(-k)
                if t == t1:
                    expr_terms.append(k)
                if t == t1 + tau:
                    expr_terms.append(k)

        if t > 0:
            expr_terms.extend([
                start_cost * lam_start_cost[t - 1],
                -shut_cost * lam_shut_cost[t - 1],
            ])
        if t < trainer.T - 1:
            expr_terms.extend([
                -start_cost * lam_start_cost[t],
                shut_cost * lam_shut_cost[t],
            ])

        for k, ts in enumerate(sensitive_t):
            for time_idx, coeff in iterate_surrogate_constraint_terms(
                ts,
                constraint_offsets[k],
                alphas[k],
                betas[k],
                gammas[k],
                trainer.T,
            ):
                if time_idx == t:
                    expr_terms.append(coeff * mu[k])

        expr_terms.extend([lam_x_upper[t], -lam_x_lower[t]])
        obj_dual_x_terms.append(cp.abs(_sum_scalar_terms(expr_terms)))

    for t in range(trainer.T - 1):
        expr = 1 - lam_start_cost[t] - lam_shut_cost[t] - lam_coc_nonneg[t]
        obj_dual_coc_terms.append(cp.abs(expr))

    for t in range(trainer.T):
        obj_opt_terms.append(abs(pg_val[t] - Pmin * x_val[t]) * lam_pg_lower[t])
        obj_opt_terms.append(abs(Pmax * x_val[t] - pg_val[t]) * lam_pg_upper[t])

    for t in range(1, trainer.T):
        limit = Ru * x_val[t - 1] + Ru_co * (1 - x_val[t - 1])
        obj_opt_terms.append(abs(pg_val[t] - pg_val[t - 1] - limit) * lam_ramp_up[t - 1])
        limit = Rd * x_val[t] + Rd_co * (1 - x_val[t])
        obj_opt_terms.append(abs(pg_val[t - 1] - pg_val[t] - limit) * lam_ramp_down[t - 1])

    for tau in range(1, Ton + 1):
        for t1 in range(trainer.T - tau):
            obj_opt_terms.append(abs(x_val[t1 + 1] - x_val[t1] - x_val[t1 + tau]) * lam_min_on[tau - 1, t1])

    for tau in range(1, Toff + 1):
        for t1 in range(trainer.T - tau):
            obj_opt_terms.append(abs(-x_val[t1 + 1] + x_val[t1] - 1 + x_val[t1 + tau]) * lam_min_off[tau - 1, t1])

    for t in range(trainer.T - 1):
        obj_opt_terms.append(abs(coc_val[t] - start_cost * (x_val[t + 1] - x_val[t])) * lam_start_cost[t])
        obj_opt_terms.append(abs(coc_val[t] - shut_cost * (x_val[t] - x_val[t + 1])) * lam_shut_cost[t])
        obj_opt_terms.append(abs(coc_val[t]) * lam_coc_nonneg[t])

    for t in range(trainer.T):
        obj_opt_terms.append(abs(x_val[t]) * lam_x_lower[t])
        obj_opt_terms.append(abs(x_val[t] - 1) * lam_x_upper[t])

    for k, t in enumerate(sensitive_t):
        lhs = build_surrogate_constraint_expression(
            x_val,
            t,
            constraint_offsets[k],
            alphas[k],
            betas[k],
            gammas[k],
            trainer.T,
        )
        obj_opt_terms.append(abs(lhs - deltas[k]) * mu_abs[k])

    obj_dual_prox = _build_dual_block_prox_expr(
        trainer,
        sample_id,
        lam_pg_lower,
        lam_pg_upper,
        lam_ramp_up,
        lam_ramp_down,
        lam_start_cost,
        lam_shut_cost,
        lam_coc_nonneg,
        lam_x_upper,
        lam_x_lower,
        lam_min_on,
        lam_min_off,
        mu,
        Ton,
        Toff,
    )
    obj_dual_pg = _sum_scalar_terms(obj_dual_pg_terms)
    obj_dual_x = _sum_scalar_terms(obj_dual_x_terms)
    obj_dual_coc = _sum_scalar_terms(obj_dual_coc_terms)
    obj_opt = _sum_scalar_terms(obj_opt_terms)
    objective = (
        trainer.rho_dual_pg * obj_dual_pg
        + trainer.rho_dual_x * obj_dual_x
        + trainer.rho_dual_coc * obj_dual_coc
        + trainer.rho_opt * obj_opt
        + trainer.dual_block_prox_weight * obj_dual_prox
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        _solve_with_cvxpy_highs(problem, verbose=False)
    except Exception:
        if _strict_cvxpy_highs_diagnostics_enabled():
            ctx = f"unit_id={g}, sample_id={sample_id}, block=dual"
            print("[STRICT_CVXPY_HIGHS] dual_block failure context:", flush=True)
            print(f"  {ctx}", flush=True)
            try:
                print(f"  pg_val={_array_finite_stats(pg_val)}", flush=True)
                print(f"  x_val={_array_finite_stats(x_val)}", flush=True)
                print(f"  coc_val={_array_finite_stats(coc_val)}", flush=True)
                print(f"  lambda_val={_array_finite_stats(lambda_val)}", flush=True)
                print(f"  alphas={_array_finite_stats(alphas)}", flush=True)
                print(f"  betas={_array_finite_stats(betas)}", flush=True)
                print(f"  gammas={_array_finite_stats(gammas)}", flush=True)
                print(f"  deltas={_array_finite_stats(deltas)}", flush=True)
                if costs is not None:
                    print(f"  costs={_array_finite_stats(costs)}", flush=True)
                if pg_costs is not None:
                    print(f"  pg_costs={_array_finite_stats(pg_costs)}", flush=True)
            except Exception:
                pass
        raise
    if not _problem_is_optimal(problem):
        return None, None

    display_sample_id = int(getattr(trainer, "display_sample_id", sample_id))
    defer_log = bool(getattr(trainer, "_defer_lp_block_log", False))
    if defer_log:
        trainer._deferred_lp_block_log = None
    # 与 solve_primal_block 一致：每个机组仅打印前 3 个样本（sample_id 0..2）
    # dual_block_prox_weight==0 时 obj_dual_prox 为 Python float（无 .value），须与 primal 的 obj_prox 同理处理
    if display_sample_id <= 2:
        try:
            def _maybe_float(val):
                if val is None:
                    return None
                try:
                    return float(np.asarray(val, dtype=float).reshape(-1)[0])
                except Exception:
                    return None

            def _term_solved_value(term) -> float:
                if term is None:
                    return 0.0
                if isinstance(term, (int, float, np.integer, np.floating)):
                    return float(term)
                v = _maybe_float(getattr(term, "value", None))
                return 0.0 if v is None else v

            obj_dual_pg_v = _term_solved_value(obj_dual_pg)
            obj_dual_x_v = _term_solved_value(obj_dual_x)
            obj_dual_coc_v = _term_solved_value(obj_dual_coc)
            obj_dual_v = obj_dual_pg_v + obj_dual_x_v + obj_dual_coc_v
            obj_opt_v = _term_solved_value(obj_opt)
            obj_dual_prox_v = _term_solved_value(obj_dual_prox)
            status = getattr(problem, "status", None)
            line = (
                f"[Unit-{g}] dual_block, sample_id: {display_sample_id}, "
                f"status: {status}, "
                f"obj_dual_pg: {obj_dual_pg_v:.6f}, "
                f"obj_dual_x: {obj_dual_x_v:.6f}, "
                f"obj_dual_coc: {obj_dual_coc_v:.6f}, "
                f"obj_dual: {obj_dual_v:.6f}, "
                f"obj_opt: {obj_opt_v:.6f}, "
                f"obj_dual_prox: {obj_dual_prox_v:.6f}"
            )
            if defer_log:
                trainer._deferred_lp_block_log = line
            else:
                print(line, flush=True)
        except Exception:
            pass

    lambda_inherent_sol = {
        "lambda_pg_lower": np.asarray(lam_pg_lower.value, dtype=float),
        "lambda_pg_upper": np.asarray(lam_pg_upper.value, dtype=float),
        "lambda_ramp_up": np.asarray(lam_ramp_up.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "lambda_ramp_down": np.asarray(lam_ramp_down.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "lambda_min_on": np.array(
            [
                [float(lam_min_on[tau - 1, t1].value) for t1 in range(trainer.T - tau)]
                for tau in range(1, Ton + 1)
            ],
            dtype=object,
        ),
        "lambda_min_off": np.array(
            [
                [float(lam_min_off[tau - 1, t1].value) for t1 in range(trainer.T - tau)]
                for tau in range(1, Toff + 1)
            ],
            dtype=object,
        ),
        "lambda_start_cost": np.asarray(lam_start_cost.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "lambda_shut_cost": np.asarray(lam_shut_cost.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "lambda_coc_nonneg": np.asarray(lam_coc_nonneg.value, dtype=float) if trainer.T > 1 else np.zeros(0, dtype=float),
        "lambda_x_upper": np.asarray(lam_x_upper.value, dtype=float),
        "lambda_x_lower": np.asarray(lam_x_lower.value, dtype=float),
    }
    mu_sol = np.asarray(mu.value, dtype=float)
    return lambda_inherent_sol, mu_sol
