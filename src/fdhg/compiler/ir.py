from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PrimitiveFamily(str, Enum):
    BASELINE = "baseline"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class PairwiseHistorySpec:
    table: str
    key: str | None = None
    left_key: str | None = None
    right_key: str | None = None
    related_col: str | None = None
    time_col: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseSpec:
    left_key: str
    right_key: str
    target_right_key: str
    left_history: PairwiseHistorySpec | None = None
    right_history: PairwiseHistorySpec | None = None
    pair_history: PairwiseHistorySpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task: str
    problem_type: str
    label_col: str
    entity_key: str
    target_time_col: str
    child_table: str | None = None
    child_time_col: str | None = None
    numeric_col: str | None = None
    baseline_operations: tuple[str, ...] | None = None
    horizon_days: int | None = None
    feature_budget: int = 32
    primary_metric: str | None = None
    secondary_metric: str | None = None
    metric_direction: str | None = None
    seeds: tuple[int, ...] = (41, 42, 43, 44)
    pairwise: PairwiseSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Primitive:
    primitive_id: str
    family: PrimitiveFamily
    operation: str
    source_table: str | None = None
    group_key: str | None = None
    event_time_col: str | None = None
    numeric_col: str | None = None
    window_days: int | None = None
    dependency_lhs: str | None = None
    dependency_rhs: str | None = None
    temporal_predicate: str | None = None
    temporally_safe: bool = True
    estimated_cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["family"] = self.family.value
        return out


@dataclass
class CompiledTask:
    task_spec: TaskSpec
    candidate_primitives: list[Primitive]
    selected_primitive_ids: list[str] = field(default_factory=list)
    fallback_to_baseline: bool = False
    compiler_version: str = "fdhg-compiler-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler_version": self.compiler_version,
            "task_spec": self.task_spec.to_dict(),
            "candidate_primitives": [
                primitive.to_dict()
                for primitive in self.candidate_primitives
            ],
            "selected_primitive_ids": self.selected_primitive_ids,
            "fallback_to_baseline": self.fallback_to_baseline,
        }
