"""
稀疏参数化代理约束挖掘与注入工具。

设计目标：
1. 离线枚举大量候选约束，并根据真实解/LP 解的区分能力打分；
2. 用轻量 greedy 去冗余策略保留少量高价值约束；
3. 在在线求解阶段将保留约束以软约束形式注入到全局 LP 中；
4. 默认保持完全可选，不影响现有 BCD / feasibility pump 主流程。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConstraintTerm:
    """单个线性项：coefficient * var[unit_id, time_index]"""

    var_kind: str
    unit_id: int
    time_index: int
    coefficient: float


@dataclass
class RhsFeatureSpec:
    """右端项特征描述，均由 Pd 预先计算成数值，保持在线模型仍为 LP。"""

    name: str
    transform: str
    time_indices: List[int]


@dataclass
class CandidateMetrics:
    """候选约束在训练样本上的统计指标。"""

    true_satisfaction: float
    lp_satisfaction: float
    lp_violation_mean: float
    true_violation_mean: float
    margin_gap_mean: float
    violation_gap_mean: float
    support_cost: float
    score: float


@dataclass
class SparseConstraintCandidate:
    """固定模板 + 可拟合右端项的参数化约束。"""

    constraint_id: str
    family: str
    terms: List[ConstraintTerm]
    rhs_features: List[RhsFeatureSpec]
    rhs_weights: List[float] = field(default_factory=list)
    rhs_bias: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[CandidateMetrics] = None
    lp_activation_pattern: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.metrics is not None:
            payload["metrics"] = asdict(self.metrics)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SparseConstraintCandidate":
        metrics = payload.get("metrics")
        return cls(
            constraint_id=payload["constraint_id"],
            family=payload["family"],
            terms=[ConstraintTerm(**item) for item in payload["terms"]],
            rhs_features=[RhsFeatureSpec(**item) for item in payload["rhs_features"]],
            rhs_weights=list(payload.get("rhs_weights", [])),
            rhs_bias=float(payload.get("rhs_bias", 0.0)),
            metadata=dict(payload.get("metadata", {})),
            metrics=CandidateMetrics(**metrics) if metrics is not None else None,
            lp_activation_pattern=list(payload.get("lp_activation_pattern", [])),
        )


@dataclass
class SparseSurrogateLibrary:
    """稀疏代理约束库，可保存/加载并直接用于 LP 软约束注入。"""

    constraints: List[SparseConstraintCandidate]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SparseSurrogateLibrary":
        return cls(
            constraints=[
                SparseConstraintCandidate.from_dict(item)
                for item in payload.get("constraints", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SparseSurrogateLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _normalized_total_load(pd_data: np.ndarray) -> np.ndarray:
    """总负荷归一化，便于构建稳定的右端项特征。"""

    total = np.sum(np.asarray(pd_data, dtype=float), axis=0)
    scale = max(float(np.max(np.abs(total))), 1.0)
    return total / scale


def _build_rhs_feature_specs(time_indices: Sequence[int]) -> List[RhsFeatureSpec]:
    """为一个时间窗口构建简单但有区分度的 Pd 特征族。"""

    time_list = sorted(int(t) for t in time_indices)
    return [
        RhsFeatureSpec(name="const", transform="constant", time_indices=[]),
        RhsFeatureSpec(name="window_mean", transform="mean", time_indices=time_list),
        RhsFeatureSpec(name="window_max", transform="max", time_indices=time_list),
        RhsFeatureSpec(name="window_std", transform="std", time_indices=time_list),
        RhsFeatureSpec(name="window_mean_sq", transform="mean_square", time_indices=time_list),
        RhsFeatureSpec(name="window_mean_sqrt", transform="sqrt_mean", time_indices=time_list),
        RhsFeatureSpec(name="window_ramp_abs", transform="ramp_abs_mean", time_indices=time_list),
    ]


def compute_rhs_feature_value(pd_data: np.ndarray, feature: RhsFeatureSpec) -> float:
    """计算单个右端项特征值。"""

    load_norm = _normalized_total_load(pd_data)
    values = load_norm[np.asarray(feature.time_indices, dtype=int)] if feature.time_indices else load_norm

    if feature.transform == "constant":
        return 1.0
    if feature.transform == "mean":
        return float(np.mean(values))
    if feature.transform == "max":
        return float(np.max(values))
    if feature.transform == "std":
        return float(np.std(values))
    if feature.transform == "mean_square":
        return float(np.mean(values ** 2))
    if feature.transform == "sqrt_mean":
        return float(np.sqrt(max(float(np.mean(values)), 0.0)))
    if feature.transform == "ramp_abs_mean":
        if len(values) <= 1:
            return 0.0
        return float(np.mean(np.abs(np.diff(values))))
    raise ValueError(f"Unsupported feature transform: {feature.transform}")


def compute_rhs_value(candidate: SparseConstraintCandidate, pd_data: np.ndarray) -> float:
    """根据候选约束的右端特征和权重，计算当前样本的 RHS 数值。"""

    if not candidate.rhs_features:
        return float(candidate.rhs_bias)

    weights = candidate.rhs_weights or [0.0] * len(candidate.rhs_features)
    rhs = float(candidate.rhs_bias)
    for weight, feature in zip(weights, candidate.rhs_features):
        rhs += float(weight) * compute_rhs_feature_value(pd_data, feature)
    return rhs


def _lhs_value(
    terms: Sequence[ConstraintTerm],
    x_value: Optional[np.ndarray],
    pg_value: Optional[np.ndarray],
) -> float:
    """计算给定样本上约束左端项数值。"""

    lhs = 0.0
    for term in terms:
        if term.var_kind == "pg":
            if pg_value is None:
                raise ValueError("pg_value is required for pg terms")
            lhs += float(term.coefficient) * float(pg_value[term.unit_id, term.time_index])
        elif term.var_kind == "x":
            if x_value is None:
                raise ValueError("x_value is required for x terms")
            lhs += float(term.coefficient) * float(x_value[term.unit_id, term.time_index])
        else:
            raise ValueError(f"Unsupported var_kind: {term.var_kind}")
    return lhs


def _ridge_fit(features: np.ndarray, targets: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """对右端项做轻量 ridge 拟合，避免奇异矩阵。"""

    eye = np.eye(features.shape[1], dtype=float)
    return np.linalg.solve(features.T @ features + reg * eye, features.T @ targets)


def fit_candidate_rhs(
    candidate: SparseConstraintCandidate,
    samples: Sequence[Dict[str, Any]],
    target_quantile: float = 1.0,
    reg: float = 1e-6,
) -> SparseConstraintCandidate:
    """
    用真实解 lhs 拟合 RHS(Pd)。

    拟合后再用高分位残差上移 bias，以优先保证真实解满足率。
    """

    if not samples:
        raise ValueError("samples must not be empty")

    rhs_features = candidate.rhs_features or [RhsFeatureSpec("const", "constant", [])]
    phi = []
    lhs_true_values = []
    for sample in samples:
        pd_data = np.asarray(sample["pd_data"], dtype=float)
        x_true = np.asarray(sample.get("x_true")) if sample.get("x_true") is not None else None
        pg_true = np.asarray(sample.get("pg_true")) if sample.get("pg_true") is not None else None
        phi.append([compute_rhs_feature_value(pd_data, spec) for spec in rhs_features])
        lhs_true_values.append(_lhs_value(candidate.terms, x_true, pg_true))

    phi_mat = np.asarray(phi, dtype=float)
    lhs_true_arr = np.asarray(lhs_true_values, dtype=float)
    weights = _ridge_fit(phi_mat, lhs_true_arr, reg=reg)
    base_rhs = phi_mat @ weights
    residual = lhs_true_arr - base_rhs
    upward_shift = float(np.quantile(residual, target_quantile))

    fitted = SparseConstraintCandidate.from_dict(candidate.to_dict())
    fitted.rhs_features = rhs_features
    fitted.rhs_weights = [float(item) for item in weights]
    fitted.rhs_bias = upward_shift
    return fitted


def evaluate_candidate(
    candidate: SparseConstraintCandidate,
    samples: Sequence[Dict[str, Any]],
    support_penalty: float = 0.05,
    violation_tolerance: float = 1e-6,
) -> SparseConstraintCandidate:
    """统计候选约束对真实解与 LP 解的区分能力。"""

    if not samples:
        raise ValueError("samples must not be empty")

    true_sat = []
    lp_sat = []
    true_viols = []
    lp_viols = []
    margin_gaps = []
    activation_pattern = []

    for sample in samples:
        pd_data = np.asarray(sample["pd_data"], dtype=float)
        x_lp = np.asarray(sample.get("x_lp")) if sample.get("x_lp") is not None else None
        pg_lp = np.asarray(sample.get("pg_lp")) if sample.get("pg_lp") is not None else None
        x_true = np.asarray(sample.get("x_true")) if sample.get("x_true") is not None else None
        pg_true = np.asarray(sample.get("pg_true")) if sample.get("pg_true") is not None else None

        rhs = compute_rhs_value(candidate, pd_data)
        lhs_true = _lhs_value(candidate.terms, x_true, pg_true)
        lhs_lp = _lhs_value(candidate.terms, x_lp, pg_lp)

        true_viol = max(lhs_true - rhs, 0.0)
        lp_viol = max(lhs_lp - rhs, 0.0)

        true_viols.append(true_viol)
        lp_viols.append(lp_viol)
        true_sat.append(1.0 if true_viol <= violation_tolerance else 0.0)
        lp_sat.append(1.0 if lp_viol <= violation_tolerance else 0.0)
        margin_gaps.append((lhs_lp - rhs) - (lhs_true - rhs))
        activation_pattern.append(1 if lp_viol > violation_tolerance else 0)

    support_cost = float(len(candidate.terms) + 0.5 * len(candidate.rhs_features))
    true_satisfaction = float(np.mean(true_sat))
    lp_satisfaction = float(np.mean(lp_sat))
    lp_violation_mean = float(np.mean(lp_viols))
    true_violation_mean = float(np.mean(true_viols))
    violation_gap_mean = lp_violation_mean - true_violation_mean
    margin_gap_mean = float(np.mean(margin_gaps))

    score = (
        3.0 * true_satisfaction
        + 2.0 * violation_gap_mean
        + 1.0 * margin_gap_mean
        - 0.5 * lp_satisfaction
        - support_penalty * support_cost
    )

    evaluated = SparseConstraintCandidate.from_dict(candidate.to_dict())
    evaluated.metrics = CandidateMetrics(
        true_satisfaction=true_satisfaction,
        lp_satisfaction=lp_satisfaction,
        lp_violation_mean=lp_violation_mean,
        true_violation_mean=true_violation_mean,
        margin_gap_mean=margin_gap_mean,
        violation_gap_mean=violation_gap_mean,
        support_cost=support_cost,
        score=float(score),
    )
    evaluated.lp_activation_pattern = activation_pattern
    return evaluated


def build_pg_window_candidates(
    ng: int,
    T: int,
    window_sizes: Sequence[int] = (1, 2, 3),
) -> List[SparseConstraintCandidate]:
    """枚举局部时间窗上的 pg 约束模板。"""

    candidates: List[SparseConstraintCandidate] = []
    for g in range(int(ng)):
        for window in window_sizes:
            width = int(window)
            if width <= 0 or width > T:
                continue
            for start in range(0, T - width + 1):
                time_indices = list(range(start, start + width))
                terms = [
                    ConstraintTerm("pg", unit_id=g, time_index=t, coefficient=1.0)
                    for t in time_indices
                ]
                candidates.append(
                    SparseConstraintCandidate(
                        constraint_id=f"pg_window_g{g}_t{start}_w{width}",
                        family="pg_window",
                        terms=terms,
                        rhs_features=_build_rhs_feature_specs(time_indices),
                        metadata={"unit_ids": [g], "time_indices": time_indices},
                    )
                )
    return candidates


def build_hybrid_window_candidates(
    unit_caps: Sequence[float],
    T: int,
    window_sizes: Sequence[int] = (1, 2),
    x_weight: float = 0.5,
) -> List[SparseConstraintCandidate]:
    """枚举简单 hybrid x-pg 模板，默认不在主流程自动启用。"""

    candidates: List[SparseConstraintCandidate] = []
    for g, cap in enumerate(unit_caps):
        for window in window_sizes:
            width = int(window)
            if width <= 0 or width > T:
                continue
            for start in range(0, T - width + 1):
                time_indices = list(range(start, start + width))
                terms = []
                for t in time_indices:
                    terms.append(ConstraintTerm("pg", unit_id=g, time_index=t, coefficient=1.0))
                    terms.append(
                        ConstraintTerm(
                            "x",
                            unit_id=g,
                            time_index=t,
                            coefficient=-float(x_weight) * float(cap),
                        )
                    )
                candidates.append(
                    SparseConstraintCandidate(
                        constraint_id=f"hybrid_window_g{g}_t{start}_w{width}",
                        family="hybrid_x_pg",
                        terms=terms,
                        rhs_features=_build_rhs_feature_specs(time_indices),
                        metadata={"unit_ids": [g], "time_indices": time_indices, "cap": float(cap)},
                    )
                )
    return candidates


def build_residual_shaping_candidates(
    samples: Sequence[Dict[str, Any]],
    top_k_per_unit: int = 2,
    window_sizes: Sequence[int] = (1, 2, 3),
) -> List[SparseConstraintCandidate]:
    """
    从 `pg_true - pg_lp` 的窗口差异中筛选高价值模板。

    这些模板的约束形式仍是线性的 `sum(pg window) <= rhs(Pd)`，
    只是用 residual 统计作为离线候选预筛。
    """

    if not samples:
        return []

    pg_lp0 = np.asarray(samples[0]["pg_lp"], dtype=float)
    ng, T = pg_lp0.shape
    scored_windows: Dict[int, List[Tuple[float, int, int]]] = {g: [] for g in range(ng)}

    for g in range(ng):
        for window in window_sizes:
            width = int(window)
            if width <= 0 or width > T:
                continue
            for start in range(0, T - width + 1):
                diffs = []
                for sample in samples:
                    pg_lp = np.asarray(sample["pg_lp"], dtype=float)
                    pg_true = np.asarray(sample["pg_true"], dtype=float)
                    lhs_lp = float(np.sum(pg_lp[g, start:start + width]))
                    lhs_true = float(np.sum(pg_true[g, start:start + width]))
                    diffs.append(lhs_lp - lhs_true)
                score = float(np.mean(diffs))
                scored_windows[g].append((score, start, width))

    candidates: List[SparseConstraintCandidate] = []
    for g, windows in scored_windows.items():
        windows.sort(key=lambda item: item[0], reverse=True)
        for rank, (_score, start, width) in enumerate(windows[:top_k_per_unit]):
            time_indices = list(range(start, start + width))
            candidates.append(
                SparseConstraintCandidate(
                    constraint_id=f"residual_pg_g{g}_t{start}_w{width}_r{rank}",
                    family="residual_shaping",
                    terms=[
                        ConstraintTerm("pg", unit_id=g, time_index=t, coefficient=1.0)
                        for t in time_indices
                    ],
                    rhs_features=_build_rhs_feature_specs(time_indices),
                    metadata={"unit_ids": [g], "time_indices": time_indices},
                )
            )
    return candidates


def jaccard_similarity(lhs: Iterable[int], rhs: Iterable[int]) -> float:
    """比较两个激活模式的冗余度。"""

    lhs_set = {idx for idx, item in enumerate(lhs) if int(item) != 0}
    rhs_set = {idx for idx, item in enumerate(rhs) if int(item) != 0}
    if not lhs_set and not rhs_set:
        return 1.0
    union = lhs_set | rhs_set
    if not union:
        return 0.0
    return len(lhs_set & rhs_set) / len(union)


def greedy_select_candidates(
    candidates: Sequence[SparseConstraintCandidate],
    max_selected: int = 20,
    redundancy_threshold: float = 0.85,
    min_true_satisfaction: float = 0.95,
    min_violation_gap: float = 1e-5,
) -> List[SparseConstraintCandidate]:
    """按分数排序并用激活模式去冗余，保留少量高价值约束。"""

    filtered = []
    for candidate in candidates:
        if candidate.metrics is None:
            raise ValueError("candidate.metrics must be populated before selection")
        if candidate.metrics.true_satisfaction < min_true_satisfaction:
            continue
        if candidate.metrics.violation_gap_mean < min_violation_gap:
            continue
        filtered.append(candidate)

    filtered.sort(key=lambda item: item.metrics.score if item.metrics else -np.inf, reverse=True)

    selected: List[SparseConstraintCandidate] = []
    for candidate in filtered:
        is_redundant = False
        for chosen in selected:
            sim = jaccard_similarity(candidate.lp_activation_pattern, chosen.lp_activation_pattern)
            if sim >= redundancy_threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(candidate)
        if len(selected) >= max_selected:
            break
    return selected


def mine_sparse_surrogate_library(
    samples: Sequence[Dict[str, Any]],
    ng: Optional[int] = None,
    T: Optional[int] = None,
    unit_caps: Optional[Sequence[float]] = None,
    include_hybrid: bool = False,
    window_sizes: Sequence[int] = (1, 2, 3),
    max_selected: int = 20,
    redundancy_threshold: float = 0.85,
) -> SparseSurrogateLibrary:
    """
    端到端离线流程：
    1. 枚举候选模板；
    2. 用真实解拟合 RHS(Pd)；
    3. 统计真实/LP 区分指标；
    4. 排序 + greedy 去冗余筛选。
    """

    if not samples:
        raise ValueError("samples must not be empty")

    sample0 = samples[0]
    pg_lp0 = np.asarray(sample0["pg_lp"], dtype=float)
    ng = int(ng if ng is not None else pg_lp0.shape[0])
    T = int(T if T is not None else pg_lp0.shape[1])

    candidates = build_pg_window_candidates(ng=ng, T=T, window_sizes=window_sizes)
    candidates.extend(build_residual_shaping_candidates(samples, window_sizes=window_sizes))
    if include_hybrid and unit_caps is not None:
        candidates.extend(build_hybrid_window_candidates(unit_caps, T=T, window_sizes=(1, 2)))

    evaluated: List[SparseConstraintCandidate] = []
    for candidate in candidates:
        fitted = fit_candidate_rhs(candidate, samples)
        evaluated.append(evaluate_candidate(fitted, samples))

    selected = greedy_select_candidates(
        evaluated,
        max_selected=max_selected,
        redundancy_threshold=redundancy_threshold,
    )
    metadata = {
        "n_input_samples": len(samples),
        "n_generated_candidates": len(candidates),
        "n_selected_constraints": len(selected),
        "window_sizes": list(window_sizes),
        "include_hybrid": bool(include_hybrid and unit_caps is not None),
    }
    return SparseSurrogateLibrary(constraints=selected, metadata=metadata)


def build_mining_samples_from_agent(
    agent,
    true_pg_keys: Sequence[str] = ("pg_true", "pg_opt", "pg_solution"),
    true_x_keys: Sequence[str] = ("x_true", "unit_commitment_matrix"),
) -> List[Dict[str, Any]]:
    """
    从 `Agent_NN_BCD` 一类对象中抽取稀疏约束挖掘样本。

    该函数只读取 agent 暴露的缓存，不修改其现有训练/推理方法。
    """

    active_set_data = getattr(agent, "active_set_data", None)
    x_lp_all = getattr(agent, "x", None)
    pg_lp_all = getattr(agent, "pg", None)
    x_opt_all = getattr(agent, "x_opt", None)
    if active_set_data is None or x_lp_all is None or pg_lp_all is None or x_opt_all is None:
        raise ValueError("agent must expose active_set_data, x, pg, and x_opt")

    samples: List[Dict[str, Any]] = []
    for sample_id, sample in enumerate(active_set_data):
        pd_data = np.asarray(sample["pd_data"], dtype=float)

        x_true = None
        for key in true_x_keys:
            if sample.get(key) is not None:
                x_true = np.asarray(sample[key], dtype=float)
                break
        if x_true is None:
            x_true = np.asarray(x_opt_all[sample_id], dtype=float)

        pg_true = None
        for key in true_pg_keys:
            if sample.get(key) is not None:
                pg_true = np.asarray(sample[key], dtype=float)
                break

        if pg_true is None:
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "pd_data": pd_data,
                "x_lp": np.asarray(x_lp_all[sample_id], dtype=float),
                "pg_lp": np.asarray(pg_lp_all[sample_id], dtype=float),
                "x_true": x_true,
                "pg_true": pg_true,
            }
        )
    return samples


def mine_sparse_surrogate_library_from_agent(
    agent,
    unit_caps: Optional[Sequence[float]] = None,
    include_hybrid: bool = False,
    window_sizes: Sequence[int] = (1, 2, 3),
    max_selected: int = 20,
    redundancy_threshold: float = 0.85,
) -> SparseSurrogateLibrary:
    """从现有 agent 缓存直接生成稀疏参数化约束库。"""

    samples = build_mining_samples_from_agent(agent)
    if not samples:
        raise ValueError("No samples with pg_true were found on the provided agent")
    return mine_sparse_surrogate_library(
        samples=samples,
        unit_caps=unit_caps,
        include_hybrid=include_hybrid,
        window_sizes=window_sizes,
        max_selected=max_selected,
        redundancy_threshold=redundancy_threshold,
    )


def evaluate_library_on_samples(
    library: SparseSurrogateLibrary,
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """输出离线阶段评估摘要，用于注入前后的快速对比。"""

    if not library.constraints:
        return {
            "n_constraints": 0,
            "avg_true_satisfaction": 0.0,
            "avg_lp_satisfaction": 0.0,
            "avg_violation_gap": 0.0,
            "avg_score": 0.0,
        }

    refreshed = [evaluate_candidate(candidate, samples) for candidate in library.constraints]
    return {
        "n_constraints": len(refreshed),
        "avg_true_satisfaction": float(np.mean([c.metrics.true_satisfaction for c in refreshed if c.metrics])),
        "avg_lp_satisfaction": float(np.mean([c.metrics.lp_satisfaction for c in refreshed if c.metrics])),
        "avg_violation_gap": float(np.mean([c.metrics.violation_gap_mean for c in refreshed if c.metrics])),
        "avg_score": float(np.mean([c.metrics.score for c in refreshed if c.metrics])),
    }


def summarize_library(library: SparseSurrogateLibrary) -> Dict[str, Any]:
    """返回便于日志/调试输出的库摘要。"""

    families: Dict[str, int] = {}
    best_score = None
    for candidate in library.constraints:
        families[candidate.family] = families.get(candidate.family, 0) + 1
        if candidate.metrics is not None:
            best_score = candidate.metrics.score if best_score is None else max(best_score, candidate.metrics.score)
    return {
        "n_constraints": len(library.constraints),
        "families": families,
        "best_score": best_score,
        **library.metadata,
    }


def add_sparse_parameterized_constraints(
    model,
    x_vars,
    pg_vars,
    pd_data: np.ndarray,
    sparse_library: Optional[SparseSurrogateLibrary],
    surr_slacks: Optional[List[Any]] = None,
    name_prefix: str = "kappa",
) -> List[Any]:
    """
    将稀疏参数化约束以软约束形式加入 Gurobi 模型。

    约束形式统一为：
        lhs( x / pg ) - rhs(Pd) <= slack
    """

    if sparse_library is None or not sparse_library.constraints:
        return surr_slacks if surr_slacks is not None else []

    import gurobipy as gp

    if surr_slacks is None:
        surr_slacks = []

    for idx, candidate in enumerate(sparse_library.constraints):
        lhs = gp.LinExpr()
        for term in candidate.terms:
            if term.var_kind == "x":
                lhs += float(term.coefficient) * x_vars[term.unit_id, term.time_index]
            elif term.var_kind == "pg":
                lhs += float(term.coefficient) * pg_vars[term.unit_id, term.time_index]
            else:
                raise ValueError(f"Unsupported var_kind for online injection: {term.var_kind}")

        rhs = compute_rhs_value(candidate, pd_data)
        slack = model.addVar(lb=0, name=f"{name_prefix}_slack_{idx}")
        model.addConstr(lhs - float(rhs) <= slack, name=f"{name_prefix}_surr_{idx}")
        surr_slacks.append(slack)
    return surr_slacks
