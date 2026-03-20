"""
针对 x[g, t] 的稀疏支持集发现。

目标不是直接学习完整约束参数，而是先从大量候选变量中筛出
高价值参与变量，并据此构造少量支持集模板。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SparseVariableRef:
    unit_id: int
    time_index: int


@dataclass
class SparseVariableScore:
    variable: SparseVariableRef
    fractionality_score: float
    disagreement_score: float
    lp_distance_score: float
    pd_mutual_score: float
    stability_score: float
    score: float
    activation_pattern: List[int] = field(default_factory=list)


@dataclass
class SparseSupportGroup:
    group_id: str
    variables: List[SparseVariableRef]
    score: float
    pairwise_score: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SparseSupportDiscoveryResult:
    selected_variables: List[SparseVariableScore]
    support_groups: List[SparseSupportGroup]
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "selected_variables": [asdict(item) for item in self.selected_variables],
            "support_groups": [asdict(item) for item in self.support_groups],
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
    def from_dict(cls, payload: Dict) -> "SparseSupportDiscoveryResult":
        selected_variables = []
        for item in payload.get("selected_variables", []):
            variable = SparseVariableRef(**item["variable"])
            selected_variables.append(
                SparseVariableScore(
                    variable=variable,
                    fractionality_score=float(item["fractionality_score"]),
                    disagreement_score=float(item["disagreement_score"]),
                    lp_distance_score=float(item["lp_distance_score"]),
                    pd_mutual_score=float(item["pd_mutual_score"]),
                    stability_score=float(item["stability_score"]),
                    score=float(item["score"]),
                    activation_pattern=list(item.get("activation_pattern", [])),
                )
            )

        support_groups = []
        for item in payload.get("support_groups", []):
            support_groups.append(
                SparseSupportGroup(
                    group_id=item["group_id"],
                    variables=[SparseVariableRef(**var) for var in item["variables"]],
                    score=float(item["score"]),
                    pairwise_score=float(item["pairwise_score"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )

        return cls(
            selected_variables=selected_variables,
            support_groups=support_groups,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SparseSupportDiscoveryResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _safe_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _normalized_total_load(pd_data: np.ndarray) -> np.ndarray:
    total = np.sum(np.asarray(pd_data, dtype=float), axis=0)
    scale = max(float(np.max(np.abs(total))), 1.0)
    return total / scale


def _build_pd_feature_bank(samples: Sequence[Dict]) -> np.ndarray:
    """
    为每个样本构造按时间展开的 Pd 特征。

    输出 shape: (n_samples, n_features)
    """

    rows = []
    for sample in samples:
        total = _normalized_total_load(sample["pd_data"])
        ramp = np.concatenate([[0.0], np.diff(total)])
        abs_ramp = np.abs(ramp)
        square = total ** 2
        sqrt_total = np.sqrt(np.clip(total, 0.0, None))
        rows.append(np.concatenate([total, square, sqrt_total, ramp, abs_ramp]))
    return np.asarray(rows, dtype=float)


def extract_support_discovery_samples_from_agent(
    agent,
    true_x_keys: Sequence[str] = ("x_true", "unit_commitment_matrix"),
) -> List[Dict]:
    """从 Agent_NN_BCD 一类对象中提取 sparse 发现样本。"""

    active_set_data = getattr(agent, "active_set_data", None)
    x_lp_all = getattr(agent, "x", None)
    x_opt_all = getattr(agent, "x_opt", None)
    if active_set_data is None or x_lp_all is None or x_opt_all is None:
        raise ValueError("agent must expose active_set_data, x, and x_opt")

    samples = []
    for sample_id, sample in enumerate(active_set_data):
        x_true = None
        for key in true_x_keys:
            if sample.get(key) is not None:
                x_true = np.asarray(sample[key], dtype=float)
                break
        if x_true is None:
            x_true = np.asarray(x_opt_all[sample_id], dtype=float)

        samples.append(
            {
                "sample_id": sample_id,
                "pd_data": np.asarray(sample["pd_data"], dtype=float),
                "x_lp": np.asarray(x_lp_all[sample_id], dtype=float),
                "x_true": x_true,
            }
        )
    return samples


def score_sparse_variables(
    samples: Sequence[Dict],
    mismatch_threshold: float = 0.2,
    round_threshold: float = 0.5,
) -> List[SparseVariableScore]:
    """为每个 x[g, t] 变量计算 sparse 价值分数。"""

    if not samples:
        raise ValueError("samples must not be empty")

    x_lp0 = np.asarray(samples[0]["x_lp"], dtype=float)
    ng, T = x_lp0.shape
    pd_feature_bank = _build_pd_feature_bank(samples)

    scores: List[SparseVariableScore] = []
    for g in range(ng):
        for t in range(T):
            x_lp_series = np.asarray([sample["x_lp"][g, t] for sample in samples], dtype=float)
            x_true_series = np.asarray([sample["x_true"][g, t] for sample in samples], dtype=float)

            fractionality = float(np.mean(x_lp_series * (1.0 - x_lp_series)))
            disagreement = float(np.mean(np.abs((x_lp_series >= round_threshold).astype(float) - x_true_series)))
            lp_distance = float(np.mean(np.abs(x_lp_series - x_true_series)))
            activation = (np.abs(x_lp_series - x_true_series) >= mismatch_threshold).astype(int)
            activation_pattern = activation.tolist()

            pd_corrs = [_safe_corr(np.abs(x_lp_series - x_true_series), pd_feature_bank[:, idx]) for idx in range(pd_feature_bank.shape[1])]
            pd_mutual_score = float(max((abs(item) for item in pd_corrs), default=0.0))

            stability = float(max(0.0, 1.0 - np.std(x_true_series)))
            final_score = (
                0.35 * disagreement
                + 0.30 * lp_distance
                + 0.20 * fractionality
                + 0.10 * pd_mutual_score
                + 0.05 * stability
            )

            scores.append(
                SparseVariableScore(
                    variable=SparseVariableRef(unit_id=g, time_index=t),
                    fractionality_score=fractionality,
                    disagreement_score=disagreement,
                    lp_distance_score=lp_distance,
                    pd_mutual_score=pd_mutual_score,
                    stability_score=stability,
                    score=final_score,
                    activation_pattern=activation_pattern,
                )
            )

    scores.sort(key=lambda item: item.score, reverse=True)
    return scores


def select_top_sparse_variables(
    scores: Sequence[SparseVariableScore],
    top_k: int = 20,
    min_score: float = 0.0,
) -> List[SparseVariableScore]:
    """保留分数最高的一小批变量。"""

    selected = [item for item in scores if item.score >= min_score]
    selected.sort(key=lambda item: item.score, reverse=True)
    return selected[:top_k]


def _activation_jaccard(lhs: Iterable[int], rhs: Iterable[int]) -> float:
    lhs_set = {idx for idx, value in enumerate(lhs) if int(value) != 0}
    rhs_set = {idx for idx, value in enumerate(rhs) if int(value) != 0}
    if not lhs_set and not rhs_set:
        return 1.0
    union = lhs_set | rhs_set
    if not union:
        return 0.0
    return len(lhs_set & rhs_set) / len(union)


def build_support_groups(
    selected_variables: Sequence[SparseVariableScore],
    max_groups: int = 5,
    group_size: int = 3,
    min_pairwise_jaccard: float = 0.15,
) -> List[SparseSupportGroup]:
    """
    从高价值单变量出发，按共同激活模式组合成少量支持集。
    """

    if not selected_variables:
        return []

    groups: List[SparseSupportGroup] = []
    seen = set()
    for anchor_idx, anchor in enumerate(selected_variables):
        members = [anchor]
        pairwise_values: List[float] = []
        for partner in selected_variables:
            if partner.variable == anchor.variable:
                continue
            sim = _activation_jaccard(anchor.activation_pattern, partner.activation_pattern)
            if sim < min_pairwise_jaccard:
                continue
            members.append(partner)
            pairwise_values.append(sim)
            if len(members) >= group_size:
                break

        member_refs = tuple(sorted((item.variable.unit_id, item.variable.time_index) for item in members))
        if member_refs in seen:
            continue
        seen.add(member_refs)

        group_score = float(np.mean([item.score for item in members]))
        pairwise_score = float(np.mean(pairwise_values)) if pairwise_values else 0.0
        groups.append(
            SparseSupportGroup(
                group_id=f"sparse_group_{anchor_idx}",
                variables=[item.variable for item in members],
                score=group_score + 0.25 * pairwise_score,
                pairwise_score=pairwise_score,
                metadata={
                    "anchor_score": anchor.score,
                    "member_count": float(len(members)),
                },
            )
        )
        if len(groups) >= max_groups:
            break

    groups.sort(key=lambda item: item.score, reverse=True)
    return groups


def discover_sparse_supports(
    samples: Sequence[Dict],
    top_k_variables: int = 20,
    max_groups: int = 5,
    group_size: int = 3,
    min_score: float = 0.0,
) -> SparseSupportDiscoveryResult:
    """端到端执行变量打分、筛选和支持集构造。"""

    all_scores = score_sparse_variables(samples)
    selected_variables = select_top_sparse_variables(all_scores, top_k=top_k_variables, min_score=min_score)
    support_groups = build_support_groups(
        selected_variables,
        max_groups=max_groups,
        group_size=group_size,
    )
    metadata = {
        "n_samples": float(len(samples)),
        "n_scored_variables": float(len(all_scores)),
        "n_selected_variables": float(len(selected_variables)),
        "n_support_groups": float(len(support_groups)),
    }
    return SparseSupportDiscoveryResult(
        selected_variables=selected_variables,
        support_groups=support_groups,
        metadata=metadata,
    )
