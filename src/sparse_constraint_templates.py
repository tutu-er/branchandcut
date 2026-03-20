"""
将 sparse 支持集发现结果转成可序列化、可注入的约束模板。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sparse_support_discovery import (
        SparseSupportDiscoveryResult,
        SparseSupportGroup,
        SparseVariableRef,
    )
except ImportError:
    from src.sparse_support_discovery import (
        SparseSupportDiscoveryResult,
        SparseSupportGroup,
        SparseVariableRef,
    )


@dataclass
class SparseConstraintTemplate:
    template_id: str
    support_variables: List[SparseVariableRef]
    initial_coefficients: List[float]
    initial_rhs: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SparseConstraintTemplateLibrary:
    templates: List[SparseConstraintTemplate]
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "templates": [asdict(item) for item in self.templates],
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
    def from_dict(cls, payload: Dict) -> "SparseConstraintTemplateLibrary":
        templates = []
        for item in payload.get("templates", []):
            templates.append(
                SparseConstraintTemplate(
                    template_id=item["template_id"],
                    support_variables=[SparseVariableRef(**var) for var in item["support_variables"]],
                    initial_coefficients=[float(val) for val in item["initial_coefficients"]],
                    initial_rhs=float(item["initial_rhs"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return cls(templates=templates, metadata=dict(payload.get("metadata", {})))

    @classmethod
    def from_json(cls, path: str | Path) -> "SparseConstraintTemplateLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _estimate_initial_rhs(group: SparseSupportGroup, samples: Sequence[Dict]) -> float:
    """根据真实解上的支持集和估计一个保守 RHS。"""

    support_sum = []
    for sample in samples:
        x_true = np.asarray(sample["x_true"], dtype=float)
        support_sum.append(
            float(
                np.sum(
                    [
                        x_true[var.unit_id, var.time_index]
                        for var in group.variables
                    ]
                )
            )
        )
    return float(np.quantile(np.asarray(support_sum, dtype=float), 0.95))


def build_sparse_template_library(
    discovery_result: SparseSupportDiscoveryResult,
    samples: Sequence[Dict],
    max_templates: int = 5,
) -> SparseConstraintTemplateLibrary:
    """从 sparse 支持集构造固定模板库。"""

    templates: List[SparseConstraintTemplate] = []
    for idx, group in enumerate(discovery_result.support_groups[:max_templates]):
        rhs = _estimate_initial_rhs(group, samples)
        templates.append(
            SparseConstraintTemplate(
                template_id=f"sparse_template_{idx}",
                support_variables=list(group.variables),
                initial_coefficients=[1.0] * len(group.variables),
                initial_rhs=rhs,
                metadata={
                    "group_score": group.score,
                    "pairwise_score": group.pairwise_score,
                    "member_count": float(len(group.variables)),
                },
            )
        )
    return SparseConstraintTemplateLibrary(
        templates=templates,
        metadata={
            "n_templates": float(len(templates)),
            "n_groups_available": float(len(discovery_result.support_groups)),
        },
    )


def template_library_to_bcd_union_constraints(
    library: Optional[SparseConstraintTemplateLibrary],
    start_branch_id: int = 100000,
) -> List[Dict]:
    """
    将模板库转成 `uc_NN_BCD.py` 可消费的 union_constraints 结构。

    `time_slot` 仅作为锚点时间和 RHS 命名键，真正的成员时间由
    `nonzero_pg_coefficients[*].time_index` 显式给出。
    """

    if library is None:
        return []

    constraints = []
    for idx, template in enumerate(library.templates):
        if not template.support_variables:
            continue
        anchor_time = min(var.time_index for var in template.support_variables)
        branch_id = start_branch_id + idx
        constraints.append(
            {
                "branch_id": branch_id,
                "time_slot": anchor_time,
                "constraint_type": "external_sparse_support",
                "constraint_name": template.template_id,
                "template_id": template.template_id,
                "is_external_sparse_template": True,
                "initial_rhs": template.initial_rhs,
                "nonzero_pg_coefficients": [
                    {
                        "unit_id": var.unit_id,
                        "time_index": var.time_index,
                        "coefficient": float(template.initial_coefficients[pos]),
                    }
                    for pos, var in enumerate(template.support_variables)
                ],
            }
        )
    return constraints


def add_sparse_x_templates_to_model(
    model,
    x_vars,
    template_library: Optional[SparseConstraintTemplateLibrary],
    surr_slacks: Optional[List] = None,
    name_prefix: str = "sparse_x",
) -> List:
    """将固定 sparse x 模板以软约束形式注入 Gurobi 模型。"""

    if template_library is None or not template_library.templates:
        return surr_slacks if surr_slacks is not None else []

    import gurobipy as gp

    if surr_slacks is None:
        surr_slacks = []

    for idx, template in enumerate(template_library.templates):
        lhs = gp.LinExpr()
        for coeff, var in zip(template.initial_coefficients, template.support_variables):
            lhs += float(coeff) * x_vars[var.unit_id, var.time_index]

        slack = model.addVar(lb=0, name=f"{name_prefix}_slack_{idx}")
        model.addConstr(lhs - float(template.initial_rhs) <= slack, name=f"{name_prefix}_{idx}")
        surr_slacks.append(slack)

    return surr_slacks
